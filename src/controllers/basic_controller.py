from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None
        self.goals = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        if self.args.agent in ["policy_infer"]:
            q, pi_infer = self.forward(ep_batch, t_ep, t_env, test_mode=test_mode)
            agent_outputs = self.action_selector.select_action(q, avail_actions, t_env, test_mode=test_mode, mask=True)
            return agent_outputs, pi_infer
        elif self.args.agent in ["som_new"]:
            q = self.forward(ep_batch, t_ep, t_env, test_mode=test_mode)
            agent_outputs = self.action_selector.select_action(q, avail_actions, t_env, test_mode=test_mode, mask=True)
            return agent_outputs
        elif self.args.agent in ["som"]:
            agent_outputs, log_probs, value, entropy = self.forward(ep_batch, t_ep, t_env, test_mode=test_mode)
            chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
            return chosen_actions, log_probs, value, entropy
        else:
            agent_outputs = self.forward(ep_batch, t_ep, t_env, test_mode=test_mode)
            chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode, mask=True)
            return chosen_actions

    def train_forward(self, ep_batch, t):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        actions = ep_batch["actions"][:, t]

        if self.args.agent in ["policy_infer"]:
            q, self.hidden_states, _ = self.agent(agent_inputs, self.hidden_states)
            return q
    
    def forward(self, ep_batch, t, t_env=None, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        if self.args.agent in ["policy_infer"]:
            q, self.hidden_states, self.hidden_pi, pi_infer = self.agent(agent_inputs, self.hidden_states, self.hidden_pi)
            return q, pi_infer
        elif self.args.agent in ["som_new"]:
            q, self.hidden_states = self.agent(agent_inputs, self.hidden_states, avail_actions, t_env=t_env, test_mode=test_mode)
            return q
        elif self.args.agent in ["som"]:
            ac_outs, log_probs, value, entropy, self.hidden_states = self.agent(agent_inputs, self.hidden_states, avail_actions, t_env=t_env, test_mode=test_mode)
            return ac_outs, log_probs, value, entropy
        else:
            agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
            agent_outs = agent_outs.view(ep_batch.batch_size, self.n_agents, -1)
            return agent_outs

    def init_hidden(self, batch_size):
        if self.args.agent in ["policy_infer"]:
            self.hidden_states, self.hidden_pi = self.agent.init_hidden()
            self.hidden_states, self.hidden_pi = self.hidden_states.unsqueeze(0).expand(batch_size, self.n_agents, -1), self.hidden_pi.unsqueeze(0).expand(batch_size, self.n_agents, -1)
        # elif self.args.agent in ["som"]:
        #     self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
        #     self.goals = self.agent.init_goals().unsqueeze(0).expand(batch_size, self.n_agents, self.n_agents, -1)  ## bs, n, n, n_ac
        else:
            self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav
            
    def init_tree(self, batch_size):
        self.tree = th.zeros(batch_size, self.n_agents**2)
        self.tree_hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        if self.args.agent == "bicnet":
            inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)
        else:
            inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape

    def _build_gates_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["gates"][:, t].detach())
        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs
