import torch.nn as nn
import torch.nn.functional as F
import torch as th
import math

class EnvModel(nn.Module):
    def __init__(self, args):
        super(EnvModel, self).__init__()
        self.args = args

        # self.obs_net = nn.Linear(args.state_shape, args.n_agents * args.hidden_dim)
        self.transition_net = nn.Linear(args.hidden_dim + args.n_agents, args.hidden_dim)
        # self.infer_net = nn.Linear(args.hidden_dim, args.n_agents * args.n_actions)


    # def obs_model(self, state):
    #     return self.obs_net(state)
    
    def transition_model(self, transition_inputs):
        return self.transition_net(transition_inputs)

    # def infer_model(self, h):
    #     return self.infer_net(h)


class MBOMAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(MBOMAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        self.q_net_ij = nn.Linear(args.hidden_dim + args.mess_dim, args.n_actions)
        self.dec_net = nn.Linear(args.hidden_dim, args.n_actions)

        self.world_model = EnvModel(args)

        self.trajectory_dim = args.n_rollout_steps * args.n_actions

        self.state2query = nn.Linear(self.trajectory_dim, self.args.query_dim)
        self.state2key = nn.Linear(self.trajectory_dim, self.args.key_dim)
        self.state2value = nn.Linear(self.trajectory_dim, self.args.mess_dim) 


    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()
    
    def imagination_module(self, h):
        # h : bs, n, h_dim
        world_model = self.world_model

        for i in range(self.args.n_rollout_steps):
            # q_infer = world_model.infer_model(h)     # bs, n, n * n_ac
            # q_infer = q_infer.reshape(-1, self.n_agents, self.n_agents, self.n_actions)
            # ac_infer = q_infer.max(-1)[1]   ## argmax q to get ac   bs, n, n
            q_infer = self.dec_net(h)     # bs, n, n_ac
            ac_infer = q_infer.max(-1)[1]   ## argmax q to get ac   bs, n
            ac_infer = ac_infer.reshape(-1, 1, self.args.n_agents).repeat(1, self.args.n_agents, 1)
            transition_inputs = th.cat((h, ac_infer), dim=-1)      # bs, n, n + h_dim
            h = world_model.transition_model(transition_inputs)     # bs, n, h_dim

            if i == 0:
                trajectory = q_infer    ## bs, n, n_ac
            else: 
                trajectory = th.cat((trajectory, q_infer),dim=-1)    ## bs, n, n_rollout_steps * n_ac

        return trajectory

        #     if i == 0:
        #         trajectory = ac_infer.unsqueeze(-1)      ## bs, n, n, 1
        #         hidden_model = h
        #     else: 
        #         hidden_model = th.cat((hidden_model, h), dim=-1)                    ## bs, n, h_dim * n_rollout_steps
        #         trajectory = th.cat((trajectory, ac_infer.unsqueeze(-1)),dim=-1)    ## bs, n, n, n_rollout_steps

        # return trajectory, hidden_model
    
    def utility_ij(self, h, others_trajectory, hidden_model):
        # h : bs, n, h_dim
        # others_trajectory : bs, n, n, n_rollout_steps
        # hidden_model: bs, n, h_dim * n_rollout_steps

        # bs = h.shape[0]
        # hj = th.cat((h, hidden_model),dim=-1)   ## bs, n, h_dim * (n_rollout_steps +1)
        # hi = h.repeat(1,1,self.n_agents).reshape(bs, self.n_agents, self.n_agents, -1)
        # hj = hj.reshape(bs, 1, -1).repeat(1,self.n_agents, 1).reshape(bs, self.n_agents, self.n_agents, -1)
        # h_inputs = th.cat((hi,hj),dim=-1)   # bs, n, n, (n_rollout_steps+2)*h_dim
        # q_inputs = th.cat((h_inputs, others_trajectory),dim=-1)  # bs, n, n,  (n_rollout_steps+2)*h_dim + n_rollout_steps * n_ac) 
        # qij = self.q_net_ij(q_inputs)       # bs, n, n, n_ac

        bs = h.shape[0]
        others_trajectory = others_trajectory.reshape(bs, self.n_agents, -1)
        q_inputs = th.cat((h, others_trajectory),dim=-1)  # bs, n, h_dim + n*n_rollout_steps
        qij = self.q_net_ij(q_inputs)       # bs, n, n_ac

        return qij


    def forward(self, inputs, old_hstate):
        bs = int(inputs.shape[0]/self.n_agents)

        ob_feature = F.relu(self.fc1(inputs))    # bs*n, obs_dim --> bs * n, h_dim
        old_hstate = old_hstate.reshape(-1, self.args.hidden_dim)    # bs， n, h_dim --> bs * n, h_dim
        h = self.rnn(ob_feature, old_hstate).reshape(bs, self.n_agents, -1)      # bs*n, h_dim    -->   bs, n, h_dim
        # h_comm = self.comm_obs(h)

        # trajectory, hidden_others = self.imagination_module(h.detach())      # bs, n, n, n_rollout_steps    ## bs, n, h_dim * n_rollout_steps
        trajectory = self.imagination_module(h.detach())      ## bs, n, n_rollout_steps * n_ac

        traj_msg = self.attention(bs, trajectory)
        q_local = self.dec_net(h)

        q_ij_inputs = th.cat((h, traj_msg),dim=-1)  # bs, n, h_dim * 2
        q_ij = self.q_net_ij(q_ij_inputs)       # bs, n, n_ac

        q = q_local + self.args.lamda_q * q_ij

        # q = self.utility_ij(h, trajectory.detach())  # bs, n, n_ac

        return q, h
    
    def attention(self, bs, trajectory):
        ## bs, n, n_rollout_steps * n_ac
        # trajectory = trajectory.reshape(bs, self.args.n_agents, -1)

        query = self.state2query(trajectory).view(-1, self.args.n_agents, self.args.query_dim)
        key = self.state2key(trajectory).view(-1, self.args.n_agents, self.args.key_dim)
        value = self.state2value(trajectory).view(-1, self.args.n_agents, self.args.mess_dim)
        score = th.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.args.mess_dim)
        attn = F.softmax(score, dim=-1)                     
        traj_attn = th.matmul(attn, value)

        return traj_attn
