import torch.nn as nn
import torch.nn.functional as F

import torch as th
from torch.distributions import Categorical


class DotSelector(nn.Module):
    def __init__(self, input_shape, args):
        super(DotSelector, self).__init__()
        self.args = args
        self.epsilon_start = self.args.epsilon_start
        self.epsilon_finish = self.args.role_epsilon_finish
        self.epsilon_anneal_time = self.args.epsilon_anneal_time
        self.epsilon_anneal_time_exp = self.args.epsilon_anneal_time_exp
        self.delta = (self.epsilon_start - self.epsilon_finish) / self.epsilon_anneal_time
        self.role_action_spaces_update_start = self.args.role_action_spaces_update_start
        self.epsilon_start_t = 0
        self.epsilon_reset = True

        self.fc1 = nn.Linear(args.hidden_dim, 2 * args.hidden_dim)
        self.fc2_all = nn.Linear(2 * args.hidden_dim, args.n_agents * args.action_latent_dim)
        self.fc2 = nn.Linear(2 * args.hidden_dim, args.action_latent_dim)
        self.get_visibility_matrix = nn.Linear(self.args.hidden_dim, self.args.n_agents * 2)

        self.epsilon = 0.05

    def forward(self, inputs, role_latent):
        x = F.relu(self.fc1(inputs))   # bs*n, h_dim --> bs*n, 2*h_dim
        # role_latent: n_role, action_latent_dim
        x_own = self.fc2(x).unsqueeze(-1)  # bs*n, action_latent_dim, 1
        role_latent_reshaped = role_latent.unsqueeze(0).repeat(x.shape[0], 1, 1)         ## n_role, action_latent_dim --> bs*n, n_role, action_latent_dim
        role_q = th.bmm(role_latent_reshaped, x_own).squeeze()  ## bs*n, n_role

        x_all = self.fc2_all(x.detach()).reshape(inputs.shape[0] * self.args.n_agents, -1).unsqueeze(-1)
        # bs*n, 2*h_dim --> bs*n, n* action_latent_dim  --> bs*n*n, action_latent_dim --> bs*n*n, action_latent_dim, 1
        role_latent_reshaped_all = role_latent.unsqueeze(0).repeat(x_all.shape[0], 1, 1)          ## n_role, action_latent_dim --> 1, n_role, action_latent_dim  --> bs*n*n, n_role, action_latent_dim

        role_q_all = th.bmm(role_latent_reshaped_all, x_all).squeeze()   ## bs*n*n, n_role

        ###########################################################################################################
        mask = self.get_visible_matrix(inputs).reshape(-1, 2)   ## bs*n*n, 2    # 0 看不见 1 看得见
        # mask = mask.max(-1)[1]  ## bs*n*n
        # role_q_all = role_q_all * mask.max(-1)[1]
        ###########################################################################################################

        return role_q, role_q_all, mask
    
    def get_visible_matrix(self, h):
        mask = self.get_visibility_matrix(h).reshape(-1, self.args.n_agents, self.args.n_agents, 2)        ## bs, n, n, 2
        mask = F.softmax(mask, dim=-1)         ## bs, n, n, 2   # 0 看不见 1 看得见
        return mask


    def select_role(self, role_qs, test_mode=False, t_env=None):
        # role_qs: bs*n*n, n_role
        self.epsilon = self.epsilon_schedule(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        # mask actions that are excluded from selection
        masked_q_values = role_qs.detach().clone()

        random_numbers = th.rand_like(role_qs[:, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_roles = Categorical(th.ones(role_qs.shape).float().to(self.args.device)).sample().long()

        picked_roles = pick_random * random_roles + (1 - pick_random) * masked_q_values.max(dim=1)[1]
        # [bs*n*n]
        return picked_roles

    def epsilon_schedule(self, t_env):
        if t_env is None:
            return 0.05

        if t_env > self.role_action_spaces_update_start and self.epsilon_reset:
            self.epsilon_reset = False
            self.epsilon_start_t = t_env
            self.epsilon_anneal_time = self.epsilon_anneal_time_exp
            self.delta = (self.epsilon_start - self.epsilon_finish) / self.epsilon_anneal_time

        if t_env - self.epsilon_start_t > self.epsilon_anneal_time:
            epsilon = self.epsilon_finish
        else:
            epsilon = self.epsilon_start - (t_env - self.epsilon_start_t) * self.delta

        return epsilon
