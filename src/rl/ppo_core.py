import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):

    def __init__(self, obs_dim=80, move_dim=9, kick_dim=2):
        super().__init__()
        self.shared = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 512)),
            nn.LayerNorm(512),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.LayerNorm(512),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.LayerNorm(512),
            nn.Tanh()
        )
        self.actor_move = layer_init(nn.Linear(512, move_dim), std=0.01)
        self.actor_kick = layer_init(nn.Linear(512, kick_dim), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def forward(self, obs):
        feat = self.shared(obs)
        return self.actor_move(feat), self.actor_kick(feat), self.critic(feat)

    def get_action_and_value(self, obs, action=None, deterministic=False):
        logits_move, logits_kick, value = self.forward(obs)
        dist_move = Categorical(logits=logits_move)
        dist_kick = Categorical(logits=logits_kick)

        if action is None:
            if deterministic and not deterministic:
                act_m = torch.argmax(logits_move, dim=-1)
                act_k = torch.argmax(logits_kick, dim=-1)
            else:
                act_m = dist_move.sample()
                act_k = dist_kick.sample()
            action = torch.stack([act_m, act_k], dim=1)
        else:
            act_m, act_k = action[:, 0], action[:, 1]

        log_prob = dist_move.log_prob(act_m) + dist_kick.log_prob(act_k)
        entropy = dist_move.entropy() + dist_kick.entropy()

        return action, log_prob, entropy, value.squeeze(-1)