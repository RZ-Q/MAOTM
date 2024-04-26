import torch.nn as nn
import torch.nn.functional as F
import torch as th

class EWMAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(EWMAgent, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
      
        self.g_net = nn.Linear(args.rnn_hidden_dim + args.n_agents * args.n_actions, args.embed_dim)
        self.f_net = nn.Linear(args.rnn_hidden_dim, args.embed_dim * args.n_actions)

        self.q_lambda = args.q_lambda

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        return h
    
    def decision_module(self, hidden, actions_, q_i):
        rollout_steps = actions_.shape[1]
        bs = int(hidden.shape[0]/self.args.n_agents)
        hidden = hidden.reshape(bs, self.args.n_agents, -1)
        q_i = q_i.reshape(bs, self.args.n_agents, self.args.n_actions)

        matrix = th.ones(self.args.n_agents, self.args.n_agents).int()
        diagonal_matrix = th.diag(th.zeros(self.args.n_agents)).int()
        masked_matrix = (matrix - diagonal_matrix).unsqueeze(0).repeat(bs, 1, 1).to(self.args.device).unsqueeze(1).repeat(1, rollout_steps, 1, 1).unsqueeze(-1)
        actions_ = th.nn.functional.one_hot(actions_, num_classes=self.args.n_actions)
        actions_ = actions_ * masked_matrix  # mask self predict

        hidden = hidden.unsqueeze(2).repeat(1, 1, rollout_steps, 1)
        g_inp = th.cat((hidden, actions_.reshape(bs, self.args.n_agents, rollout_steps, -1)), dim=-1)   ## bs, n, h+n*n_ac
        f_inp = hidden  ## bs,n,h_dim

        g = self.g_net(g_inp).reshape(-1, 1, self.args.embed_dim)  ## bs , n, embed_dim --> bs * n, 1 embed_dim
        f = self.f_net(f_inp).reshape(-1, self.args.embed_dim, self.args.n_actions)  ## bs, n, embed_dim * n_ac --> bs * n, embed_dim, n_ac
        q_i_j = th.bmm(g,f).reshape(bs, self.args.n_agents, rollout_steps, self.args.n_actions).mean(2)   ## bs, n, n_ac

        q = q_i + self.q_lambda * q_i_j
        return q, q_i_j
