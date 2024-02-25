import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule, DecayThenFlatSchedule_RBOM
from torch.distributions import Normal

REGISTRY = {}


class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False, mask=True):
        
        masked_policies = agent_inputs.clone()
        if mask:
            masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
            
        else:
            # print("===========================================================================")
            # print('agent_inp:',agent_inputs)
            # print('0000000000000000000000000000000000000000000000000')
            # print('avail_actions',avail_actions)
            # print('0000000000000000000000000000000000000000000000000')
            # print('masked_policies:',masked_policies)
            picked_actions = Categorical(masked_policies).sample().long()
            
        # sample like q
            random_numbers = th.rand_like(agent_inputs[:, :, 0])        
            pick_random = (random_numbers < self.epsilon).long()       
            random_actions = Categorical(avail_actions.float()).sample().long() 
            picked_actions = pick_random * random_actions + (1 - pick_random) * picked_actions

        return picked_actions


REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False, mask=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])   # [1,2]
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector

class SoftEpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule_RBOM(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              args.epsilon_anneal_time_exp,
                                              args.role_action_spaces_update_start,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, role_avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        d_avail_actions = avail_actions * role_avail_actions
        masked_q_values[d_avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        ind = th.gather(avail_actions, dim=2, index=picked_actions.unsqueeze(2)) > 0.99
        if not ind.all():
            # print(">>> Action Selection Error")
            ind = ind.squeeze().long()
            picked_actions = picked_actions * ind + (1 - ind) * random_actions
        return picked_actions


REGISTRY["soft_epsilon_greedy"] = SoftEpsilonGreedyActionSelector


class GivenActionSelector():

    def __init__(self, args):
        self.args = args
        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        
        #TODO 
        self.noise = 0

        if test_mode:
            self.noise = 0

# - construct a distribution 
        # gauss distribution 
        # action is mean, std is hyper-parameter related to time
        # probability of action[i] = action_cdf[i] - action_cdf[i-1]
        self.epsilon = self.schedule.eval(t_env)
        mean = agent_inputs.reshape(-1,1).squeeze()
        if test_mode:
            std = th.zeros_like(mean) + 0.1
        else:
            std = th.zeros_like(mean) + self.epsilon
        action_dis = Normal(mean, std)
        action_cdf = th.zeros([mean.shape[0], self.args.n_actions])
        for i in range(self.args.n_actions):
            action_cdf[:, i] = action_dis.cdf(th.tensor([float(i)]))

        for i in range(self.args.n_actions-1, 0, -1):
            action_cdf[:, i] = action_cdf[:, i] - action_cdf[:, i-1]

        action_cdf = action_cdf.unsqueeze(0)
        action_cdf[avail_actions == 0.0] = 0
        # print(mean, test_mode, action_cdf)
        picked_actions = Categorical(action_cdf).sample().long()

# # - clamp directly
#         # clamp action into [0, action_space_shape]
#         clamp_inputs = th.clamp(agent_inputs, 0, self.args.n_actions-0.1).floor().long()
#         # make one hot action
#         ac_one_hot = th.zeros(clamp_inputs.shape[0], self.args.n_agents, self.args.n_actions).scatter_(2,clamp_inputs,1)+ 1e-10
#         # mask actions that are excluded from selection
#         ac_one_hot[avail_actions == 0.0] = 0
#         # get action for each agent in batchs
#         picked_actions = Categorical(ac_one_hot).sample().long()

#         #picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        
        
        return picked_actions


REGISTRY["given_action"] = GivenActionSelector