import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop
from modules.agents.mbom_agent import EnvModel
# from modules.agents.mbom_agent_new import EnvModel
import torch.nn.functional as F

import numpy as np

class MBOMLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.mac = mac
        self.logger = logger

        self.env_model = EnvModel(args)

        self.params = list(mac.parameters())
        self.env_model_params = list(self.env_model.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.env_model_optimiser = RMSprop(params=self.env_model_params, lr=args.lr_env, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, show_demo=False, save_data=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]   ## bs, t, 1
        actions = batch["actions"][:, :-1]     ## bs, t, n, 1 
        # actions_onehot = batch["actions_onehot"][:, :-1]      ## bs, t, n, n_ac

        terminated = batch["terminated"][:, :-1].float()           ## bs, t, 1
        mask = batch["filled"][:, :-1].float()        ## bs, t, 1
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]           ## bs, t, n, n_ac
        visibility_matrix = batch["visible_matrix"][:, :-1]   ## bs, t, n,n

        # Calculate estimated Q-Values
        mac_out = []
        mac_hidden = []
        h_infer_k = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, hidden, h_infer = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            mac_hidden.append(hidden)
            h_infer_k.append(h_infer)    # bs,n,k,h_dim
        mac_out = th.stack(mac_out, dim=1)  # Concat over time    bs, t, n, n_ac
        mac_hidden = th.stack(mac_hidden, dim=1)  # Concat over time     bs, t, n, h_dim
        h_infer_k = th.stack(h_infer_k, dim=1)             # bs,t, n, k, h_dim

        env_model_loss, transition_mse_loss, visibility_loss, q_td_loss = self.train_env_model(mac_hidden[:,:-1], mac_out[:,:-1], actions, visibility_matrix)
        if t_env - self.log_stats_t >= 10000:
            th.save(h_infer_k,"h_infer"+str(t_env) +".pt")
            th.save(mac_hidden,"h"+str(t_env)+self.args.env_args.map_name+".pt")
            print("t_env:" )
            print(t_env)
            print("env_model_loss:")
            print(env_model_loss)
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim       bs, t, n

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs, _, _ = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("env_model_loss", env_model_loss.item(), t_env)
            self.logger.log_stat("transition_mse_loss", transition_mse_loss.item(), t_env)
            self.logger.log_stat("visibility_loss", visibility_loss.item(), t_env)
            self.logger.log_stat("q_td_loss", q_td_loss.item(), t_env)
            # self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    
    
    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    # ======================================= ac infer version =======================================================================
    def train_env_model(self, hidden, mac_out, actions, visibility_matrix):
        ## visibility_matrix:   bs, t, n, n
        bs, t = actions.shape[0], actions.shape[1]             ## actions: bs, t, n, 1
        # obs_cat = self._build_inputs(obs, ac_onehot)
        # obs_cat = obs_cat.reshape(bs * t, self.n_agents, -1)
        hidden = hidden.detach().reshape(bs * t * self.n_agents, -1)
        mac_out = mac_out.detach().reshape(bs * t, 1, -1).repeat(1,self.n_agents,1).reshape(bs * t * self.n_agents, -1)

        visibility_matrix_infer = self.env_model.get_visible_matrix(hidden).reshape(-1,2)          # bs* t, n, n, 2
        visibility_loss = F.cross_entropy(visibility_matrix_infer.reshape(-1,2),visibility_matrix.long().detach().reshape(-1))

        q_infer = self.env_model.infer_model(hidden)       ## bs*t*n, n * n_ac
        q_td_loss = ((q_infer - mac_out)**2).sum() / (th.ones(bs, t, self.n_agents).sum())

        actions = actions.detach().reshape(bs * t, self.n_agents).repeat(1, self.n_agents).reshape(bs * t * self.n_agents, -1)
        transition_inputs = th.cat((hidden, actions), dim=-1)    ## bs * t * self.n_agents, h_dim + n

        next_hidden = self.env_model.transition_model(transition_inputs).reshape(bs, t, self.n_agents, -1)
        hidden = hidden.reshape(bs, t, self.n_agents, -1)
        
        transition_mse_loss = ((next_hidden[:,:-1] - hidden[:,1:])**2).sum() / (th.ones(bs, t-1, self.n_agents).sum())
        env_model_loss = transition_mse_loss + visibility_loss + q_td_loss

        # Optimise
        self.env_model_optimiser.zero_grad()
        env_model_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.env_model_params, self.args.grad_norm_clip)
        self.env_model_optimiser.step()

        return env_model_loss, transition_mse_loss, visibility_loss, q_td_loss
    # ======================================================================================================================================
    
    # # ======================================= communication version =======================================================================
    # def train_env_model(self, hidden, mac_out, actions):
    #     bs, t = actions.shape[0], actions.shape[1]             ## actions: bs, t, n, 1
    #     # obs_cat = self._build_inputs(obs, ac_onehot)
    #     # obs_cat = obs_cat.reshape(bs * t, self.n_agents, -1)
    #     hidden = hidden.detach().reshape(bs * t * self.n_agents, -1).detach()
    #     mac_out = mac_out.detach().reshape(bs * t * self.n_agents, -1)     # bs* t*n, n_ac

    #     # actions = actions.detach().reshape(bs * t, self.n_agents).repeat(1, self.n_agents).reshape(bs * t * self.n_agents, -1)
    #     actions = mac_out.max(-1)[1] .reshape(bs * t,1, self.n_agents).repeat(1, self.n_agents,1).reshape(bs * t * self.n_agents, -1)
    #     transition_inputs = th.cat((hidden, actions), dim=-1)    ## bs * t * self.n_agents, h_dim + n

    #     next_hidden = self.env_model.transition_model(transition_inputs).reshape(bs, t, self.n_agents, -1)
    #     hidden = hidden.reshape(bs, t, self.n_agents, -1)
        
    #     transition_mse_loss = ((next_hidden[:,:-1] - hidden[:,1:])**2).sum() / (th.ones(bs, t, self.n_agents).sum())
    #     env_model_loss = transition_mse_loss

    #     # Optimise
    #     self.env_model_optimiser.zero_grad()
    #     env_model_loss.backward()
    #     grad_norm = th.nn.utils.clip_grad_norm_(self.env_model_params, self.args.grad_norm_clip)
    #     self.env_model_optimiser.step()

    #     return env_model_loss
    # # ======================================================================================================================================

    # # ======================================= original version =======================================================================
    # def train_env_model(self, hidden, mac_out, actions):
    #     bs, t = actions.shape[0], actions.shape[1]             ## actions: bs, t, n, 1
    #     # obs_cat = self._build_inputs(obs, ac_onehot)
    #     # obs_cat = obs_cat.reshape(bs * t, self.n_agents, -1)
    #     hidden = hidden.detach().reshape(bs * t * self.n_agents, -1).detach()
    #     mac_out = mac_out.detach().reshape(bs * t, 1, -1).repeat(1,self.n_agents,1).reshape(bs * t * self.n_agents, -1)     # bs, t, 1, n * n_ac

    #     others_q_infers = self.env_model.infer_model(hidden)              ## bs * t * self.n_agents, n * n_ac

    #     infer_mse_loss = ((others_q_infers - mac_out)**2).sum() / (th.ones(bs, t, self.n_agents).sum())

    #     # actions = actions.detach().reshape(bs * t, self.n_agents).repeat(1, self.n_agents).reshape(bs * t * self.n_agents, -1)
    #     actions = mac_out.max(-1)[1] .reshape(bs * t, self.n_agents).repeat(1, self.n_agents).reshape(bs * t * self.n_agents, -1)
    #     transition_inputs = th.cat((hidden, actions), dim=-1)    ## bs * t * self.n_agents, h_dim + n

    #     next_hidden = self.env_model.transition_model(transition_inputs).reshape(bs, t, self.n_agents, -1)
    #     hidden = hidden.reshape(bs, t, self.n_agents, -1)
    #     transition_mse_loss = ((next_hidden[:,:-1] - hidden[:,1:])**2).sum() / (th.ones(bs, t, self.n_agents).sum())

    #     env_model_loss = infer_mse_loss + transition_mse_loss

    #     # Optimise
    #     self.env_model_optimiser.zero_grad()
    #     env_model_loss.backward()
    #     grad_norm = th.nn.utils.clip_grad_norm_(self.env_model_params, self.args.grad_norm_clip)
    #     self.env_model_optimiser.step()

    #     return env_model_loss
    # # ======================================================================================================================================

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.env_model.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
