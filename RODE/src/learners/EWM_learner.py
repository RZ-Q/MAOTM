import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop
from components.world_model import MADTWorldModel
from torch.nn import functional as F

import numpy as np


class EWMLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.n_role_clusters = self.args.n_role_clusters
        self.context_length = args.context_length
        self.wm_rollout_steps = args.rollout_steps
        self.agent_rollout_steps = args.agent_rollout_steps

        self.params = list(mac.parameters())
        self.world_model_params = None
        self.world_model_optimizer = None

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

        self.role_mixer = None
        if args.role_mixer is not None:
            if args.role_mixer == "vdn":
                self.role_mixer = VDNMixer()
            elif args.role_mixer == "qmix":
                self.role_mixer = QMixer(args)
            else:
                raise ValueError("Role Mixer {} not recognised.".format(args.role_mixer))
            self.params += list(self.role_mixer.parameters())
            self.target_role_mixer = copy.deepcopy(self.role_mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.role_interval = args.role_interval
        self.device = self.args.device

        self.role_action_spaces_updated = True

        # action encoder
        self.action_encoder_params = list(self.mac.action_encoder_params())
        self.action_encoder_optimiser = RMSprop(params=self.action_encoder_params, lr=args.lr,
                                                alpha=args.optim_alpha, eps=args.optim_eps)

        self.world_model = MADTWorldModel(args, self.n_role_clusters)
        self.world_model_params = list(self.world_model.parameters())
        self.world_model_optimizer = self.world_model.configure_optimizers(self.args.weight_decay, self.args.betas, lr=self.args.lr)
        self.init_world_model_flag = False
        self.world_model_buffer_size = self.args.world_model_buffer_size
        self.world_model_utd = args.world_model_utd

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, running_log=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        # role_avail_actions = batch["role_avail_actions"]
        roles_shape_o = batch["roles"][:, :-1].shape
        role_at = int(np.ceil(roles_shape_o[1] / self.role_interval))
        role_t = role_at * self.role_interval

        roles_shape = list(roles_shape_o)
        roles_shape[1] = role_t
        roles = th.zeros(roles_shape).to(self.device)
        roles[:, :roles_shape_o[1]] = batch["roles"][:, :-1]
        roles = roles.view(batch.batch_size, role_at, self.role_interval, self.n_agents, -1)[:, :, 0]

        # Calculate estimated Q-Values
        mac_out = []
        role_out = []
        mac_out_ij = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            actions_ = None if self.role_action_spaces_updated else self.build_actions_(batch, t)
            agent_outs, agent_outs_ij, role_outs = self.mac.forward(batch, t=t, actions_=actions_)
            mac_out.append(agent_outs)
            mac_out_ij.append(agent_outs_ij)
            if t % self.role_interval == 0 and t < batch.max_seq_length - 1:
                role_out.append(role_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        role_out = th.stack(role_out, dim=1)  # Concat over time
        mac_out_ij = th.stack(mac_out_ij, dim=1)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        chosen_role_qvals = th.gather(role_out, dim=3, index=roles.long()).squeeze(3)
        chosen_action_qvals_ij = th.gather(mac_out_ij[:, :-1], dim=3, index=actions).squeeze(3)

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        target_role_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            actions_ = None if self.role_action_spaces_updated else self.build_actions_(batch, t)
            target_agent_outs, target_agent_outs_ij, target_role_outs = self.target_mac.forward(batch, t=t, actions_=actions_)
            target_mac_out.append(target_agent_outs)
            if t % self.role_interval == 0 and t < batch.max_seq_length - 1:
                target_role_out.append(target_role_outs)

        target_role_out.append(th.zeros(batch.batch_size, self.n_agents, self.mac.n_roles).to(self.device))
        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
        target_role_out = th.stack(target_role_out[1:], dim=1)

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999
        # target_mac_out[role_avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            # mac_out_detach[role_avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

            role_out_detach = role_out.clone().detach()
            role_out_detach = th.cat([role_out_detach[:, 1:], role_out_detach[:, 0:1]], dim=1)
            cur_max_roles = role_out_detach.max(dim=3, keepdim=True)[1]
            target_role_max_qvals = th.gather(target_role_out, 3, cur_max_roles).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]
            target_role_max_qvals = target_role_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])
        if self.role_mixer is not None:
            state_shape_o = batch["state"][:, :-1].shape
            state_shape = list(state_shape_o)
            state_shape[1] = role_t
            role_states = th.zeros(state_shape).to(self.device)
            role_states[:, :state_shape_o[1]] = batch["state"][:, :-1].detach().clone()
            role_states = role_states.view(batch.batch_size, role_at,
                                           self.role_interval, -1)[:, :, 0]
            chosen_role_qvals = self.role_mixer(chosen_role_qvals, role_states)
            role_states = th.cat([role_states[:, 1:], role_states[:, 0:1]], dim=1)
            target_role_max_qvals = self.target_role_mixer(target_role_max_qvals, role_states)

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        rewards_shape = list(rewards.shape)
        rewards_shape[1] = role_t
        role_rewards = th.zeros(rewards_shape).to(self.device)
        role_rewards[:, :rewards.shape[1]] = rewards.detach().clone()
        role_rewards = role_rewards.view(batch.batch_size, role_at,
                                         self.role_interval).sum(dim=-1, keepdim=True)
        # role_terminated
        terminated_shape_o = terminated.shape
        terminated_shape = list(terminated_shape_o)
        terminated_shape[1] = role_t
        role_terminated = th.zeros(terminated_shape).to(self.device)
        role_terminated[:, :terminated_shape_o[1]] = terminated.detach().clone()
        role_terminated = role_terminated.view(batch.batch_size, role_at, self.role_interval).sum(dim=-1, keepdim=True)
        # role_terminated
        role_targets = role_rewards + self.args.gamma * (1 - role_terminated) * target_role_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        role_td_error = (chosen_role_qvals - role_targets.detach())

        mask = mask.expand_as(td_error)
        mask_shape = list(mask.shape)
        mask_shape[1] = role_t
        role_mask = th.zeros(mask_shape).to(self.device)
        role_mask[:, :mask.shape[1]] = mask.detach().clone()
        role_mask = role_mask.view(batch.batch_size, role_at, self.role_interval, -1)[:, :, 0]

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask
        masked_role_td_error = role_td_error * role_mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()
        role_loss = (masked_role_td_error ** 2).sum() / role_mask.sum()
        loss += role_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        pred_obs_loss = None
        pred_r_loss = None
        pred_grad_norm = None
        # retain, action repr learning
        if self.role_action_spaces_updated:
            # train action encoder
            no_pred = []
            r_pred = []
            for t in range(batch.max_seq_length):
                no_preds, r_preds = self.mac.action_repr_forward(batch, t=t)
                no_pred.append(no_preds)
                r_pred.append(r_preds)
            no_pred = th.stack(no_pred, dim=1)[:, :-1]  # Concat over time
            r_pred = th.stack(r_pred, dim=1)[:, :-1]
            no = batch["obs"][:, 1:].detach().clone()
            repeated_rewards = batch["reward"][:, :-1].detach().clone().unsqueeze(2).repeat(1, 1, self.n_agents, 1)

            pred_obs_loss = th.sqrt(((no_pred - no) ** 2).sum(dim=-1)).mean()
            pred_r_loss = ((r_pred - repeated_rewards) ** 2).mean()

            pred_loss = pred_obs_loss + 10 * pred_r_loss
            self.action_encoder_optimiser.zero_grad()
            pred_loss.backward()
            pred_grad_norm = th.nn.utils.clip_grad_norm_(self.action_encoder_params, self.args.grad_norm_clip)
            self.action_encoder_optimiser.step()

            if t_env > self.args.role_action_spaces_update_start:
                self.mac.update_role_action_spaces()
                if 'noar' in self.args.mac:
                    self.target_mac.role_selector.update_roles(self.mac.n_roles)
                self.role_action_spaces_updated = False
                self._update_targets()
                self.last_target_update_episode = episode_num

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", (loss - role_loss).item(), t_env)
            self.logger.log_stat("role_loss", role_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            running_log.update({
                "loss": (loss - role_loss).item(),
                "role_loss": role_loss.item(),
                "grad_norm": grad_norm,
            })
            if pred_obs_loss is not None:
                self.logger.log_stat("pred_obs_loss", pred_obs_loss.item(), t_env)
                self.logger.log_stat("pred_r_loss", pred_r_loss.item(), t_env)
                self.logger.log_stat("action_encoder_grad_norm", pred_grad_norm, t_env)
                running_log.update({
                "pred_obs_loss": pred_obs_loss.item(),
                "pred_r_loss": pred_r_loss.item(),
                "action_encoder_grad_norm": pred_grad_norm,
                })

            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("qij_taken_mean",
                                 (chosen_action_qvals_ij * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("role_q_taken_mean",
                                 (chosen_role_qvals * role_mask).sum().item() / (role_mask.sum().item() * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env
            running_log.update({
                "td_error_abs": (masked_td_error.abs().sum().item() / mask_elems),
                "q_taken_mean": (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents),
                "qij_taken_mean": (chosen_action_qvals_ij * mask).sum().item() / (mask_elems * self.args.n_agents),
                "role_q_taken_mean": (chosen_role_qvals * role_mask).sum().item() / (role_mask.sum().item() * self.args.n_agents),
                "target_mean": (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
            })

    def train_world_model(self, buffer, t_env: int, running_log=None):
        # mean losses
        o_loss, ro_loss, a_loss, r_loss = [[] for _ in range(4)]
        # update to data ratio
        for _ in range(self.world_model_utd):
            episode_sample = buffer.sample_for_wm(self.args.batch_size, self.world_model_buffer_size)
            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != self.args.device:
                episode_sample.to(self.args.device)

            bs, seq_len, obses, actions, roles, rtgs, rewards, timesteps, indi_mask, \
                target_obses, target_actions, target_roles, target_rewards = self.decode_batch_for_wm(episode_sample)

            if self.mac.n_roles == 4:
                roles = roles % 2
                target_roles = target_roles % 2

            o, role, a, r = self.world_model(obses, actions, roles, rewards, rtgs, timesteps)
            # o, r MSE
            obs_loss = (((o - target_obses) * indi_mask) **2).sum() / indi_mask.sum()
            reward_loss = (((r -target_rewards) * indi_mask) **2).sum() / indi_mask.sum()
            o_loss.append(obs_loss), r_loss.append(reward_loss)
            # role, a -logpi, role and a -> pi
            # TODO: test retain/remove self action and role predict loss
            # matrix = th.ones(self.n_agents, self.n_agents).int().to(self.device)
            # diagonal_matrix = th.diag(th.zeros(self.n_agents)).int().to(self.device)
            # masked_matrix = (matrix - diagonal_matrix).unsqueeze(0).repeat(bs * (seq_len - 1), 1, 1).reshape(-1, self.n_agents)

            if self.role_action_spaces_updated == False:
                a = th.softmax(a.reshape(-1, self.n_agents, self.n_actions), dim=-1)
                target_actions = target_actions.reshape(-1, self.n_agents, 1)
                a = th.gather(a, dim=-1, index=target_actions)
                a[indi_mask.repeat(1, self.n_agents, 1) == 0] = 1.  # pi=1, logpi=0
                action_loss = - a.log().sum() / indi_mask.sum()

                role = role.reshape(-1, self.n_agents, self.n_role_clusters)
                if self.mac.n_roles == 4:
                    role[:, :, -1] = -999999999  # mask last role (the repeat one)
                role = th.softmax(role, dim=-1)
                target_roles = target_roles.reshape(-1, self.n_agents, 1)
                role = th.gather(role, dim=-1, index=target_roles)
                role[indi_mask.repeat(1, self.n_agents, 1) == 0] = 1.
                role_loss = - role.log().sum() / indi_mask.sum()
                world_model_loss = obs_loss + reward_loss + action_loss + role_loss
                a_loss.append(action_loss), ro_loss.append(role_loss)
            else:
                world_model_loss = obs_loss + reward_loss
            
            # Optimize
            self.world_model_optimizer.zero_grad()
            world_model_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.world_model_params, self.args.grad_norm_clip)
            self.world_model_optimizer.step()

        reward_loss = sum(r_loss) / self.world_model_utd
        obs_loss = sum(o_loss) / self.world_model_utd
        self.logger.log_stat("wm_reward_loss", reward_loss.item(), t_env)
        self.logger.log_stat("wm_obs_loss", obs_loss.item(), t_env)
        self.logger.log_stat("wm_grad_norm", grad_norm.item(), t_env)
        running_log.update({
        "wm_reward_loss": reward_loss.item(),
        "wm_obs_loss": obs_loss.item(),
        "wm_grad_norm": grad_norm.item(),
        })
        action_loss = sum(a_loss) / self.world_model_utd
        role_loss = sum(ro_loss) / self.world_model_utd
        self.logger.log_stat("wm_action_loss", action_loss.item(), t_env)
        self.logger.log_stat("wm_role_loss", role_loss.item(), t_env)
        running_log.update({
            "wm_action_loss": action_loss.item(),
            "wm_role_loss": role_loss.item(),
        })
 
    def agent_world_model_rollout(self, obs, rtg, t):
        obs = th.from_numpy(np.array(obs)).squeeze(0).unsqueeze(1).repeat(1, self.context_length, 1).to(self.device)
        rtg = th.tensor(rtg).unsqueeze(0).unsqueeze(0).repeat(self.n_agents, self.context_length, 1).to(self.device)
        role_ = th.zeros_like(rtg).repeat(1, 1, self.n_agents).to(self.device)
        actions_ = th.zeros_like(rtg).repeat(1, 1, self.n_agents).to(self.device)
        r = th.zeros_like(rtg).to(self.device)
        time_step = th.ones_like(rtg).long().to(self.device) * t

        _, role_, _, _ = self.world_model(obs, actions_, role_, r, rtg, time_step)
        role_ = role_.reshape(-1, self.context_length, self.n_agents, self.n_role_clusters).max(-1)[1]
        _, _, actions_, _ = self.world_model(obs, actions_, role_, r, rtg, time_step)
        actions_ = actions_.reshape(-1, self.context_length, self.n_agents, self.n_actions).max(-1)[1]
        _, _, _, r = self.world_model(obs, actions_, role_, r, rtg, time_step)
        return_actions_ = []
        for _ in range(self.agent_rollout_steps):
            obs, role_, actions_, r = self.world_model(obs, actions_, role_, r, rtg, time_step)
            role_ = role_.reshape(-1, self.context_length, self.n_agents, self.n_role_clusters).max(-1)[1]
            actions_ = actions_.reshape(-1, self.context_length, self.n_agents, self.n_actions).max(-1)[1]
            rtg = rtg - r
            time_step += 1
            return_actions_.append(actions_.squeeze(1).unsqueeze(0))

        return th.stack(return_actions_, dim=1)

    def decode_batch_for_wm(self, batch):
        # process batch to agent trajs flat
        rewards = batch["reward"]
        bs, seq_len = rewards.shape[0], rewards.shape[1]
        actions = batch["actions"].repeat(1, 1, self.n_agents, 1)
        obses = batch["obs"]
        terminated = batch["terminated"].float()
        indi_terminated = batch["indi_terminated"].float()
        mask = batch["filled"].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        indi_mask = th.ones_like(batch["actions"]).float()
        indi_mask[:, 1:] = batch["filled"][:, 1:].float().repeat(1, 1, self.n_agents).unsqueeze(-1) * (1 - indi_terminated[:, :-1].unsqueeze(-1))
        roles = batch["roles"].repeat(1, 1, self.n_agents, 1)
        rtgs = th.flip(rewards.cumsum(dim=1), dims=[1]).repeat(1, 1, self.n_agents).unsqueeze(-1)
        rewards = rewards.repeat(1, 1, self.n_agents).unsqueeze(-1)
        timesteps = th.arange(seq_len).unsqueeze(0).repeat(bs, 1).unsqueeze(-1).to(self.args.device).repeat(1, 1, self.n_agents).unsqueeze(-1)

        target_obses = obses[:, 1:].reshape(-1, self.context_length, obses.shape[-1])
        target_actions = actions[:, 1:].reshape(-1, self.context_length, self.n_agents)
        target_roles = roles[:, 1:].reshape(-1, self.context_length, self.n_agents)
        target_rewards = rewards[:, 1:].reshape(-1, self.context_length, 1)

        obses = obses[:, :-1].reshape(-1, self.context_length, obses.shape[-1])
        actions = actions[:, :-1].reshape(-1, self.context_length, self.n_agents)
        roles = roles[:, :-1].reshape(-1, self.context_length, self.n_agents)
        rewards = rewards[:, :-1].reshape(-1, self.context_length, 1)
        rtgs = rtgs[:, :-1].reshape(-1, self.context_length, 1)
        timesteps = timesteps[:, :-1].reshape(-1, self.context_length, 1)
        indi_mask = indi_mask[:, :-1].reshape(-1, self.context_length, 1)

        return bs, seq_len, obses, actions, roles, rtgs, rewards, timesteps, indi_mask, target_obses, target_actions, target_roles, target_rewards

    def build_actions_(self, batch, t):
        # for Q_-i input [actions_-i]_k
        seq_len = batch.max_seq_length
        if t < seq_len - self.wm_rollout_steps:
            actions_ = batch["actions"][:, t+1: t+self.wm_rollout_steps+1]
        else:
            actions_ = th.stack([batch["actions"][:, -1] for _ in range(t + self.wm_rollout_steps - seq_len + 1)], dim=1)
            actions_ = th.cat([batch["actions"][:, t+1:], actions_], dim=1)
        actions_ = actions_.repeat(1, 1, self.n_agents, 1).reshape(batch.batch_size, self.wm_rollout_steps, self.n_agents, self.n_agents)
        return actions_

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        if self.role_mixer is not None:
            self.target_role_mixer.load_state_dict(self.role_mixer.state_dict())
        self.target_mac.role_action_spaces_updated = self.role_action_spaces_updated
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.world_model.to('cuda')
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
        if self.role_mixer is not None:
            self.role_mixer.cuda()
            self.target_role_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        if self.role_mixer is not None:
            th.save(self.role_mixer.state_dict(), "{}/role_mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
        th.save(self.action_encoder_optimiser.state_dict(), "{}/action_repr_opt.th".format(path))
        th.save(self.world_model.state_dict(), "{}/world_model.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        if self.role_mixer is not None:
            self.role_mixer.load_state_dict(
                th.load("{}/role_mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        self.action_encoder_optimiser.load_state_dict(th.load("{}/action_repr_opt.th".format(path),
                                                              map_location=lambda storage, loc: storage))
        self.world_model.load_state_dict(th.load("{}/world_model.th".format(path), map_location=lambda storage, loc: storage))
