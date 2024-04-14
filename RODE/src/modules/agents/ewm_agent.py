import torch.nn as nn
import torch.nn.functional as F


class EWMAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(EWMAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.q_local = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        self.g_net = nn.Linear(args.rnn_hidden_dim + args.n_agents * args.n_actions, args.embed_dim)
        self.f_net = nn.Linear(args.rnn_hidden_dim, args.embed_dim * args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        h = self.rnn(x, h_in)
        return h
    
    def decision_module(self, hidden, actions_):
        q_i = self.q_local(hidden)
        q_i_j = 0
