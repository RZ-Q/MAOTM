import torch.nn as nn
import torch.nn.functional as F
import torch as th
import math

# class EnvModel(nn.Module):
#     def __init__(self, args):
#         super(EnvModel, self).__init__()
#         self.args = args
#         self.transition_net = nn.Linear(args.hidden_dim + args.n_agents, self.args.hidden_dim)

#     def transition_model(self, transition_inputs):
#         return self.transition_net(transition_inputs)

class EnvModel(nn.Module):
    def __init__(self, args):
        super(EnvModel, self).__init__()
        self.args = args
        if args.use_cuda:
            self.transition_net = nn.Linear(args.hidden_dim + args.n_agents, args.hidden_dim).cuda()
            self.infer_net = nn.Linear(args.hidden_dim, args.n_agents * args.n_actions).cuda()
            self.get_visibility_matrix = nn.Linear(self.args.hidden_dim, self.args.n_agents * 2).cuda()
        else:
            self.transition_net = nn.Linear(args.hidden_dim + args.n_agents, args.hidden_dim)
            self.infer_net = nn.Linear(args.hidden_dim, args.n_agents * args.n_actions)
            self.get_visibility_matrix = nn.Linear(self.args.hidden_dim, self.args.n_agents * 2)

    def transition_model(self, transition_inputs):
        return self.transition_net(transition_inputs)

    def infer_model(self, h):
        return self.infer_net(h)
    
    def get_visible_matrix(self, h):
        mask = self.get_visibility_matrix(h).reshape(-1, self.args.n_agents, self.args.n_agents, 2)        ## bs, n, n, 2
        mask = F.softmax(mask, dim=-1)         ## bs, n, n, 2   # 0 看不见 1 看得见
        return mask


class MBOMAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(MBOMAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        
        self.dec_net = nn.Linear(args.hidden_dim, args.n_actions)
        self.q_net_ij = nn.Linear(args.hidden_dim +args.n_agents * args.n_rollout_steps * self.n_actions, args.n_actions)

        # self.q_net_ij = nn.Linear(args.hidden_dim + args.mess_dim, args.n_actions)

        # self.trajectory_dim = args.n_rollout_steps * args.n_actions
        # self.state2query = nn.Linear(self.trajectory_dim, self.args.query_dim)
        # self.state2key = nn.Linear(self.trajectory_dim, self.args.key_dim)
        # self.state2value = nn.Linear(self.trajectory_dim, self.args.mess_dim) 


    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, old_hstate):
        bs = int(inputs.shape[0]/self.n_agents)

        x = F.relu(self.fc1(inputs))    # bs*n, input_dim --> bs * n, h_dim
        old_hstate = old_hstate.reshape(-1, self.args.hidden_dim)    # bs， n, h_dim --> bs * n, h_dim
        h = self.rnn(x, old_hstate).reshape(bs, self.n_agents, -1)      # bs*n, h_dim    -->   bs, n, h_dim

        # trajectory = self.imagination_module(h.detach())      ## bs, n, n_rollout_steps * n_ac
        # traj_msg = self.attention(bs, trajectory)
        # q_ij_inputs = th.cat((h, traj_msg),dim=-1)  # bs, n, h_dim * 2

        trajectory, h_infer= self.imagination_module(h.detach())
        trajectory = trajectory.reshape(bs, self.n_agents, -1)     ## bs, n, n* n_rollout_steps * n_ac
        q_local = self.dec_net(h)

        q_ij_inputs = th.cat((h, trajectory),dim=-1)  # bs, n, n* n_rollout_steps * n_ac + h_dim
        q_ij = self.q_net_ij(q_ij_inputs)       # bs, n, n_ac

        q = q_local + self.args.lamda_q * q_ij

        return q, h, h_infer
    
    # ======================================= infer ac version =======================================================================    
    def imagination_module(self, h):
        # h : bs, n, h_dim
        bs = h.shape[0]
        world_model = EnvModel(self.args)
        h_infer=[]

        for i in range(self.args.n_rollout_steps):
            mask = world_model.get_visible_matrix(h)      ## bs, n, n, 2
            mask = mask.max(-1)[1]          ## bs, n, n

            q_infer = world_model.infer_model(h).reshape(-1, self.n_agents, self.n_agents, self.n_actions)     # bs, n, n * n_ac  -->   bs, n, n, n_ac

            q_infer = q_infer * mask.unsqueeze(-1)
            ac_infer = q_infer.max(-1)[1]   ## argmax q to get ac   bs, n, n

            transition_inputs = th.cat((h, ac_infer), dim=-1)      # bs, n, n + h_dim
            h = world_model.transition_model(transition_inputs)     # bs, n, h_dim
            h_infer.append(h)

            if i == 0:
                trajectory = q_infer    ## bs, n, n, n_ac
            else: 
                trajectory = th.cat((trajectory, q_infer),dim=-1)    ## bs, n, n, n_rollout_steps * n_ac

        return trajectory, th.stack(h_infer,dim=2)  # bs,n,k,h_dim
    
    # def build_inputs(self, obs, last_ac):
    #     # Assumes homogenous agents with flat observations.
    #     # Other MACs might want to e.g. delegate building inputs to each agent
    #     bs = obs.shape[0]
    #     inputs = []
    #     inputs.append(obs)  
    #     if self.args.obs_last_action:
    #         inputs.append(last_ac)
    #     if self.args.obs_agent_id:
    #         inputs.append(th.eye(self.args.n_agents, device=self.args.device).unsqueeze(0).expand(bs, -1, -1))

    #     inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
    #     return inputs
    
    # ======================================= communication version =======================================================================
    # def imagination_module(self, h):
    #     # h : bs, n, h_dim
    #     world_model = EnvModel(self.args)

    #     for i in range(self.args.n_rollout_steps):
    #         q_infer = self.dec_net(h)     # bs, n, n_ac
    #         ac_infer = q_infer.max(-1)[1]   ## argmax q to get ac   bs, n
    #         ac_infer = ac_infer.reshape(-1, 1, self.args.n_agents).repeat(1, self.args.n_agents, 1)
    #         transition_inputs = th.cat((h, ac_infer), dim=-1)      # bs, n, n + h_dim
    #         h = world_model.transition_model(transition_inputs)     # bs, n, h_dim

    #         if i == 0:
    #             trajectory = q_infer    ## bs, n, n_ac
    #         else: 
    #             trajectory = th.cat((trajectory, q_infer),dim=-1)    ## bs, n, n_rollout_steps * n_ac

    #     return trajectory
    # ======================================================================================================================================
    
    # def attention(self, bs, trajectory):
    #     ## bs, n, n_rollout_steps * n_ac
    #     # trajectory = trajectory.reshape(bs, self.args.n_agents, -1)

    #     query = self.state2query(trajectory).view(-1, self.args.n_agents, self.args.query_dim)
    #     key = self.state2key(trajectory).view(-1, self.args.n_agents, self.args.key_dim)
    #     value = self.state2value(trajectory).view(-1, self.args.n_agents, self.args.mess_dim)
    #     score = th.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.args.mess_dim)
    #     attn = F.softmax(score, dim=-1)                     
    #     traj_attn = th.matmul(attn, value)

    #     return traj_attn

    # ==================================================================================================
    
    # def get_ally_visible(self, obs):
    #     # obs: bs, n, input_shape
    #     bs = obs.shape[0]
    #     move_dim = 4
    #     enemy_dim = 5
    #     ally_dim = 5+self.args.n_actions
    #     own_dim = 1
    #     if self.args.shield_bits_enemy:
    #         enemy_dim += 1
    #     if self.args.shield_bits_ally:
    #         ally_dim += 1
    #         own_dim += 1
    #     if self.args.unit_type_bits:
    #         enemy_dim += 1
    #         ally_dim += 1
    #         own_dim += 1
    #     enemy_dim = self.args.n_enemies * enemy_dim
    #     ally_dim = (self.args.n_agents-1) * ally_dim
    #     idx_start = move_dim + enemy_dim
    #     idx_end = move_dim + enemy_dim + ally_dim
    #     ally_features = obs[:,:,idx_start:idx_end]
    #     ally_features = ally_features.reshape(bs, self.args.n_agents, self.n_agents - 1 ,-1)     ## bs,n,n-1, ally_feat_dim
    #     is_visible = ally_features[:,:,:,0]              ## bs,n,n-1
    #     # is_visible = is_visible.reshape(bs, self.args.n_agents, -1)         
    #     mask = th.ones(bs, self.args.n_agents, self.args.n_agents)
    #     for i in range(self.args.n_agents):
    #         mask[:,i,:i] = is_visible[:,i,:i]
    #         mask[:,i,i+1:] = is_visible[:,i,i:]

    #     return mask.long()
