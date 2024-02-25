import torch.nn as nn
import torch.nn.functional as F
import torch as th

class RBOMAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RBOMAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)

        self.q_local = nn.Linear(args.hidden_dim, args.n_actions)
        # self.q_net_ij = nn.Linear(args.hidden_dim +args.n_agents * args.n_actions, args.n_actions)
        self.q_net_ij = nn.Linear(args.hidden_dim + args.n_actions, args.n_actions)
        self.infer_net = nn.Linear(args.hidden_dim, args.n_agents * args.n_actions)

        self.g_net = nn.Linear(args.hidden_dim + args.n_agents * args.n_actions, args.embed_dim)
        self.f_net = nn.Linear(args.hidden_dim, args.embed_dim * args.n_actions)


    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()
    
    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        h = self.rnn(x, h_in)
        return h
    
    # def decision_module(self, hidden, roles, roles_own, role_action_spaces):
    #     ## hidden: bs*n, h_dim,  others_ac: bs*n, n_ac  roles: bs * n * n   roles_own: bs * n   role_action_spaces: n_role, n_ac
    #     bs = int(hidden.shape[0]/self.args.n_agents)
    #     hidden = hidden.reshape(bs, self.args.n_agents, -1)
    #     roles = roles.reshape(bs, self.args.n_agents, self.args.n_agents)
    #     q_local = self.q_local(hidden)              ## bs, n, n_ac
    #     q_infer = self.infer_net(hidden.detach()).reshape(bs, self.args.n_agents, self.args.n_agents, -1)   ## bs,n, n , n_ac
    #     mask = th.eye(self.args.n_agents).to(self.args.device).unsqueeze(0).unsqueeze(-1).repeat(bs, 1,1, self.args.n_actions)
    #     roles_mask = th.eye(self.args.n_agents).to(self.args.device).unsqueeze(0).repeat(bs, 1,1)
    #     roles = th.where(roles_mask>0, roles_own.reshape(bs, -1).unsqueeze(1).repeat(1, self.args.n_agents, 1), roles).reshape(bs * self.args.n_agents, self.args.n_agents)

    #     q_infer = th.where(mask>0, q_local.unsqueeze(1).repeat(1, self.args.n_agents, 1,1), q_infer)

    #     role_avail_actions = th.gather(role_action_spaces.unsqueeze(0).unsqueeze(0).repeat(bs*self.args.n_agents,self.args.n_agents, 1, 1), dim=2, index=roles.unsqueeze(-1).unsqueeze(-1).repeat(1,1, 1, self.args.n_actions).long()).squeeze()
    #     role_avail_actions = role_avail_actions.int().view(bs, self.args.n_agents, self.args.n_agents, -1)   ## bs, n, n, n_ac

    #     role_avail_actions_own = th.gather(role_action_spaces.unsqueeze(0).repeat(bs * self.args.n_agents, 1, 1), dim=1, index=roles_own.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.args.n_actions).long()).squeeze()
    #     role_avail_actions_own = role_avail_actions_own.int().view(bs, self.args.n_agents, -1)

    #     q_infer = (q_infer * role_avail_actions)## bs,n, n, n_ac
    #     q_ij_inputs = th.cat((hidden.detach().unsqueeze(-2).repeat(1,1,self.args.n_agents, 1), q_infer.detach()),dim=-1)     ## bs, n, n, h_dim + n_ac
    #     q_ij = self.q_net_ij(q_ij_inputs).mean(dim=-2)       # bs, n, n, n_ac --> bs,n,n_ac

    #     q_local = q_local * role_avail_actions_own
    #     q = q_local + self.args.lamda_q * q_ij

    #     return q, q_local, q_infer

    def decision_module(self, hidden, roles, roles_own, role_action_spaces):
        ## hidden: bs*n, h_dim,  others_ac: bs*n, n_ac  roles: bs * n * n   roles_own: bs * n   role_action_spaces: n_role, n_ac
        bs = int(hidden.shape[0]/self.args.n_agents)
        hidden = hidden.reshape(bs, self.args.n_agents, -1)
        roles = roles.reshape(bs, self.args.n_agents, self.args.n_agents)
        # q_local = self.q_local(hidden)              ## bs, n, n_ac
        q_infer = self.infer_net(hidden.detach()).reshape(bs, self.args.n_agents, self.args.n_agents, -1)   ## bs,n, n , n_ac
        mask = th.eye(self.args.n_agents).to(self.args.device).unsqueeze(0).unsqueeze(-1).repeat(bs, 1,1, self.args.n_actions)
        roles_mask = th.eye(self.args.n_agents).to(self.args.device).unsqueeze(0).repeat(bs, 1,1)
        roles = th.where(roles_mask>0, roles_own.reshape(bs, -1).unsqueeze(1).repeat(1, self.args.n_agents, 1), roles).reshape(bs * self.args.n_agents, self.args.n_agents)

        # q_infer = th.where(mask>0, q_local.unsqueeze(1).repeat(1, self.args.n_agents, 1,1), q_infer)

        role_avail_actions = th.gather(role_action_spaces.unsqueeze(0).unsqueeze(0).repeat(bs*self.args.n_agents,self.args.n_agents, 1, 1), dim=2, index=roles.unsqueeze(-1).unsqueeze(-1).repeat(1,1, 1, self.args.n_actions).long()).squeeze()
        role_avail_actions = role_avail_actions.int().view(bs, self.args.n_agents, self.args.n_agents, -1)   ## bs, n, n, n_ac

        role_avail_actions_own = th.gather(role_action_spaces.unsqueeze(0).repeat(bs * self.args.n_agents, 1, 1), dim=1, index=roles_own.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.args.n_actions).long()).squeeze()
        role_avail_actions_own = role_avail_actions_own.int().view(bs, self.args.n_agents, -1)

        q_infer = (q_infer * role_avail_actions)## bs,n, n, n_ac

        # q_ij_inputs = th.cat((hidden.detach().unsqueeze(-2).repeat(1,1,self.args.n_agents, 1), q_infer.detach()),dim=-1)     ## bs, n, n, h_dim + n_ac
        # q_ij = self.q_net_ij(q_ij_inputs).mean(dim=-2)       # bs, n, n, n_ac --> bs,n,n_ac

        # q_local = q_local * role_avail_actions_own
        # q = q_local + self.args.lamda_q * q_ij


        g_inp = th.cat((hidden, q_infer.reshape(bs, self.args.n_agents, -1)), dim=-1)   ## bs, n, h+n*n_ac
        f_inp = hidden  ## bs,n,h_dim

        g = self.g_net(g_inp).reshape(-1, 1, self.args.embed_dim)  ## bs , n, embed_dim --> bs * n, 1 embed_dim
        f = self.f_net(f_inp).reshape(-1, self.args.embed_dim, self.args.n_actions)  ## bs, n, embed_dim * n_ac --> bs * n, embed_dim, n_ac
        q = th.bmm(g,f).reshape(bs, self.args.n_agents, self.args.n_actions)   ## bs, n, n_ac
        q = q * role_avail_actions_own

        # mask = th.tril(th.ones(bs, self.args.n_agents, self.args.n_agents),diagonal=-1).to(self.args.device)   ## bs, n, n
        # mask = mask.unsqueeze(-1).repeat(1,1,1,self.args.n_actions)    ## bs, n, n, n_ac
        # factor = th.arange(self.args.n_agents).to(self.args.device)
        # factor[0] += 1
        # factor = (1/factor).unsqueeze(0).unsqueeze(-1).repeat(bs, 1, self.args.n_actions)
        # qij = (qij * mask).sum(dim=-2)    ## bs, n, n_ac
        
        # q = q_local + self.args.lamda_q * qij * factor

        # qij = qij.sum(dim=-2)       ## bs, n, n_ac
        # q = q_local + self.args.lamda_q * qij

        return q

