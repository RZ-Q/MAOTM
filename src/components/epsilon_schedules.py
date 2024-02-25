import numpy as np

class DecayThenFlatSchedule_RBOM():

    def __init__(self,
                 start,
                 finish,
                 time_length,
                 time_length_exp,
                 role_action_spaces_update_start,
                 decay="exp"):

        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay
        self.role_action_spaces_update_start = role_action_spaces_update_start
        self.reset = True
        self.time_length_exp = time_length_exp
        self.start_t = 0

        if self.decay in ["exp"]:
            self.exp_scaling = (-1) * self.time_length / np.log(self.finish) if self.finish > 0 else 1

    def eval(self, T):
        if T > self.role_action_spaces_update_start and self.reset:
            self.reset = False
            self.time_length = self.time_length_exp
            self.delta = (self.start - self.finish) / self.time_length
            self.start_t = T

        if self.decay in ["linear"]:
            return max(self.finish, self.start - self.delta * (T-self.start_t))
        elif self.decay in ["exp"]:
            return min(self.start, max(self.finish, np.exp(- T / self.exp_scaling)))
    pass


class DecayThenFlatSchedule():

    def __init__(self,
                 start,
                 finish,
                 time_length,
                 decay="exp"):

        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay

        if self.decay in ["exp"]:
            self.exp_scaling = (-1) * self.time_length / np.log(self.finish) if self.finish > 0 else 1

    def eval(self, T):
        if self.decay in ["linear"]:
            return max(self.finish, self.start - self.delta * T)
        elif self.decay in ["exp"]:
            return min(self.start, max(self.finish, np.exp(- T / self.exp_scaling)))
    pass

# class OrnsteinUhlenbeckActionNoise():

#     def __init__(self, action_dim=1, mu=0, theta=0.15, sigma=1):
#         self.action_dim = action_dim
#         self.mu = mu
#         self.theta = theta
#         self.sigma = sigma
#         self.x = np.ones(self.action_dim) * mu

#     def reset(self):
#         self.x = = np.ones(self.action_dim) * mu

#     def sample(self):
#         dx = self.theta * (self.mu - self.x)
#         dx += self.sigma * np.random(len(self.x))
#         self.x += dx
#         return self.x