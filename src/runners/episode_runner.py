from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import torch as th


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        if self.args.use_rnn:
            self.mac.init_hidden(batch_size=self.batch_size)
       
        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
            }
            
            if self.args.agent in ["mbom", "rbom"]:
                visible_matrix = self.env.get_visibility_matrix()
                visible_matrix = th.from_numpy(visible_matrix[:,:self.args.n_agents])
                visible_matrix = th.where(visible_matrix == True, th.ones(self.args.n_agents,self.args.n_agents), th.zeros(self.args.n_agents,self.args.n_agents))
                for i in range(self.args.n_agents):
                    visible_matrix[i,i] = 1
                pre_transition_data.update({"visible_matrix": visible_matrix[:,:self.args.n_agents]})

            self.batch.update(pre_transition_data, ts=self.t)
            if self.args.agent in ['policy_infer']:
                actions, pi_infer = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            elif self.args.agent in ['som_new']:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            elif self.args.agent in ['som']:
                actions, log_probs, value, entropy = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            elif self.args.agent in ['rbom']:
                actions, roles_all, roles_own, role_avail_actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            else:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            if self.args.agent == 'policy_infer':
                pi_infer = pi_infer.reshape(-1, 1, self.args.n_agents, self.args.n_agents * self.args.n_actions)
                post_transition_data.update({
                    "pi_infer": pi_infer,})
            if self.args.agent == "rbom":
                post_transition_data.update({
                    "roles_all": roles_all.reshape(-1, 1, self.args.n_agents, self.args.n_agents),
                    "roles_own": roles_own.reshape(-1, 1, self.args.n_agents),})  ## "role_avail_actions": role_avail_actions,  
                # "agent_outs_infer": agent_outs_infer.reshape(-1, 1, self.args.n_agents, self.args.n_actions),
            if self.args.agent == 'som':
                value = value.reshape(-1,1)
                entropy = entropy.reshape(-1,self.args.n_agents)
                log_probs = log_probs.reshape(-1,self.args.n_agents, self.args.n_actions)
                post_transition_data.update({
                    "value": value,
                    "entropy": entropy,
                    "log_probs": log_probs})

            self.batch.update(post_transition_data, ts=self.t)                               
            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)
       
        if self.args.agent in ['policy_infer']:
            actions, _ = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        elif self.args.agent in ['som']:
            actions, _, _, _ = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        elif self.args.agent in ["rbom"]:
            actions, roles_all, roles_own, role_avail_actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        else:
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)     

        if self.args.agent == "rbom":
            self.batch.update({"actions": actions, "roles_all": roles_all.reshape(-1, 1, self.args.n_agents, self.args.n_agents), "roles_own": roles_own.reshape(-1, 1, self.args.n_agents)}, ts=self.t)  ## "role_avail_actions": role_avail_actions
        else:
            self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
