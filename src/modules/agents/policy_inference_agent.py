import torch.nn as nn
import torch.nn.functional as F
import torch as th
import datetime

class PIAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(PIAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        self.fc_q = nn.Linear(args.hidden_dim * 2 , args.n_actions)

        self.rnn_o = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        self.fc_pi = nn.Linear(args.hidden_dim, args.n_actions * args.n_agents)


        # self.fc_q_feature = nn.Linear(args.hidden_dim, args.hidden_dim)
        # self.fc_q = nn.Linear(args.hidden_dim * (self.args.n_agents + 1), args.n_actions)

        # self.fc_pi_feature = nn.Linear(args.hidden_dim, args.hidden_dim * args.n_agents)
        # self.fc_pi = nn.Linear(args.hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_(), self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state, hidden_pi):
        bs = int(inputs.shape[0]/self.args.n_agents)
        x = F.relu(self.fc1(inputs))
        h_q = hidden_state.reshape(-1, self.args.hidden_dim)
        h_q = self.rnn(x, h_q)

        h_pi = hidden_pi.reshape(-1, self.args.hidden_dim)
        h_pi = self.rnn_o(x, h_pi)
        pi_o = self.fc_pi(h_pi).reshape(-1, self.args.n_agents, self.args.n_agents, self.args.n_actions)
        pi_infer = F.softmax(pi_o,dim=-1)     ## bs, n, n, n_ac

        h_c = th.cat((h_q, h_pi),dim = -1)      ## bs, n, 2* hidden
        q = self.fc_q(h_c).reshape(bs, self.args.n_agents, -1)

        return q, h_q, h_pi, pi_infer

    def forward_old(self, inputs, hidden_state):
        bs = int(inputs.shape[0]/self.args.n_agents)
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        h = self.rnn(x, h_in)
        
        q_feature = (F.relu(self.fc_q_feature(h))).reshape(bs, self.args.n_agents, self.args.hidden_dim)    ### n, h_dim
        pi_feature = (F.relu(self.fc_pi_feature(h.detach()))).reshape(bs, self.args.n_agents, self.args.n_agents, self.args.hidden_dim)    ### n, n, h_dim -- n, n*h_dim
        q_feature = th.cat((q_feature, pi_feature.reshape(-1,self.args.n_agents,self.args.n_agents * self.args.hidden_dim)),dim = -1)   ### n, (n+1)*h_dim
        q = self.fc_q(q_feature)
        pi_infer = F.softmax(self.fc_pi(pi_feature),dim=-1)

        return q, h, pi_infer

    # def train_forward(self, inputs, hidden_state):
    #     bs = int(inputs.shape[0]/self.args.n_agents)
    #     x = F.relu(self.fc1(inputs))
    #     h_in = hidden_state.reshape(-1, self.args.hidden_dim)
    #     h = self.rnn(x, h_in)
        
    #     q_feature = (F.relu(self.fc_q_feature(h))).reshape(bs, self.args.n_agents, self.args.hidden_dim)    ### n, h_dim
        
    #     pi_feature = (F.relu(self.fc_pi_feature(h.detach()))).reshape(bs, self.args.n_agents, self.args.n_agents, self.args.hidden_dim)    ### n, n, h_dim -- n, n*h_dim
    #     q_feature = th.cat((q_feature, pi_feature.reshape(-1,self.args.n_agents,self.args.n_agents * self.args.hidden_dim).detach()),dim = -1)   ### n, (n+1)*h_dim
    #     q = self.fc_q(q_feature)
    #     pi_infer = F.softmax(self.fc_pi(pi_feature),dim=-1)
    #     # q = self.fc2(h)
    #     return q, h, pi_infer
