import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from gpt_model import Block

# MADT world model
# sequential input: rtg, oi, rou_-i, a_-i, ri
# sequential output: ___, oi, rou_-i, a_-i, ri
# train loss MSE

class MADTWorldModel(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, args, head_type='actor'):
        super().__init__()
        # args
        self.args = args
        # config for gpt
        self.model_type = args.model_type
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.vocab_size = args.n_actions
        self.n_roles = args.n_roles
        self.block_size = args.block_size
        self.n_layer= args.n_layer
        self.n_head= args.n_head
        self.n_embd= args.n_embd
        self.max_timestep = args.max_timestep

        # input embedding stem
        self.tok_emb = nn.Embedding(self.vocab_size, self.n_embd)
        # self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.pos_emb = nn.Parameter(torch.zeros(1, self.block_size + 1, self.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, self.max_timestep + 1, self.n_embd))
        self.drop = nn.Dropout(args.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(args) for _ in range(args.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(self.n_embd)
        if head_type == 'actor':
            self.head = nn.Linear(self.n_embd, self.vocab_size, bias=False)
        elif head_type == 'critic':
            self.head = nn.Linear(self.n_embd, 1, bias=False)
        else:
            raise NotImplementedError

        self.block_size = self.block_size
        self.apply(self._init_weights)

        # self.state_encoder = nn.Sequential(nn.Linear(self.state_shape, self.n_embd), nn.Tanh())
        self.obs_encoder = nn.Sequential(nn.Linear(self.obs_shape, self.n_embd), nn.Tanh())
        self.ret_emb = nn.Sequential(nn.Linear(1, self.n_embd), nn.Tanh())
        self.r_emb = nn.Sequential(nn.Linear(1, self.n_embd), nn.Tanh())
        self.action_embeddings = nn.Sequential(nn.Embedding(self.vocab_size, self.n_embd), nn.Tanh())
        self.role_embeddings = nn.Sequential(nn.Embedding(self.n_roles, self.n_embd), nn.Tanh())

        self.mask_emb = nn.Sequential(nn.Linear(1, self.n_embd), nn.Tanh())
        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config, lr):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=train_config.betas)
        return optimizer
        
    # state, action, and return
    def forward(self, obses, actions, roles, rewards, rtgs=None, timesteps=None):
        # obses: (batch, context_length, obs_shape)
        # actions_-i(contains own): (batch, context_length, num_agents)
        # roles_-i(contains own): (batch, context_length, num_agents)
        # rtgs: (batch, context_length, 1)
        # r: (batch, context_length, 1)
        # timesteps: (batch, context_length, 1)

        #TODO:MSE loss: remove self dialog

        obs_embeddings = self.obs_encoder(
            obses.reshape(-1, self.obs_shape).type(torch.float32).contiguous())
        obs_embeddings = obs_embeddings.reshape(obses.shape[0], obses.shape[1],
                                                    self.config.n_embd)  # (batch, block_size, n_embd)

        if self.model_type == 'rtgs_obs_roles_actions_reward':
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            reward_embeddings = self.r_emb(rewards.type(torch.float32))
            action_embeddings = self.action_embeddings(actions.long())  # (batch, block_size, n_embd)
            role_embeddings = self.role_embeddings(roles.long())

            token_embeddings = torch.zeros(
                (obses.shape[0], obses.shape[1] * 5, self.config.n_embd), dtype=torch.float32,
                device=obs_embeddings.device)
            token_embeddings[:, ::5, :] = rtg_embeddings
            token_embeddings[:, 1::5, :] = obs_embeddings
            token_embeddings[:, 2::5, :] = role_embeddings
            token_embeddings[:, 3::5, :] = action_embeddings
            token_embeddings[:, 4::5, :] = reward_embeddings
            num_elements = 5

        elif self.model_type == 'rtgs_state_action':
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))

            action_embeddings = self.action_embeddings(
                actions.type(torch.long).squeeze(-1))  # (batch, block_size, n_embd)

            token_embeddings = torch.zeros(
                (obses.shape[0], obses.shape[1] * 3, self.config.n_embd), dtype=torch.float32,
                device=obs_embeddings.device)
            token_embeddings[:, ::3, :] = rtg_embeddings
            token_embeddings[:, 1::3, :] = obs_embeddings
            token_embeddings[:, 2::3, :] = action_embeddings
            num_elements = 3
        elif self.model_type == 'state_action':
            action_embeddings = self.action_embeddings(
                actions.type(torch.long).squeeze(-1))  # (batch, block_size, n_embd)

            token_embeddings = torch.zeros(
                (obses.shape[0], obses.shape[1] * 2, self.config.n_embd), dtype=torch.float32,
                device=obs_embeddings.device)
            token_embeddings[:, ::2, :] = obs_embeddings
            token_embeddings[:, 1::2, :] = action_embeddings
            num_elements = 2
        elif self.model_type == 'state_only':
            token_embeddings = obs_embeddings
            num_elements = 1
        else:
            raise NotImplementedError()

        batch_size = obses.shape[0]
        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0)
        global_pos_emb = torch.gather(all_global_pos_emb, 1, torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1))
        global_pos_emb = torch.repeat_interleave(global_pos_emb, num_elements, dim=1)
        context_pos_emb = self.pos_emb[:, :token_embeddings.shape[1], :]
        position_embeddings = global_pos_emb + context_pos_emb

        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        # TODO: add pre_action
        if self.model_type == 'rtgs_obs_roles_actions_reward':
            o = logits[:, 0::5, :]
            role = logits[:, 1::5, :]
            a = logits[:, 2::5, :]
            r = logits[:, 3::5, :]
            return o, role, a, r

        elif self.model_type == 'rtgs_state_action':
            # logits = logits[:, 1::3, :]  # only keep predictions from state_embeddings
            logits = logits[:, 2::3, :]  # consider all tokens
        elif self.model_type == 'state_action':
            # logits = logits[:, ::2, :]  # only keep predictions from state_embeddings
            logits = logits[:, 1::2, :]  # consider all tokens
        elif self.model_type == 'state_only':
            logits = logits
        else:
            raise NotImplementedError()

        return logits
