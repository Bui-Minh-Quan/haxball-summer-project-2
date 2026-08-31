import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            layer_init(nn.Linear(dim, dim)),
            nn.LayerNorm(dim),
            nn.GELU(),
            layer_init(nn.Linear(dim, dim)),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class ActorCritic(nn.Module):
    """Decoupled Two-Tower Architecture with Residual Connections."""

    def __init__(self, obs_dim: int = 80, move_dim: int = 9, kick_dim: int = 2):
        super().__init__()

        # --- ACTOR TOWER (Policy) ---
        self.actor_encoder = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.LayerNorm(256),
            nn.GELU(),
            ResidualBlock(256),
            layer_init(nn.Linear(256, 128)),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.actor_move = layer_init(nn.Linear(128, move_dim), std=0.01)
        self.actor_kick = layer_init(nn.Linear(128, kick_dim), std=0.01)

        # --- CRITIC TOWER (Value Function) ---
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.LayerNorm(256),
            nn.GELU(),
            ResidualBlock(256),
            layer_init(nn.Linear(256, 128)),
            nn.LayerNorm(128),
            nn.GELU(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def forward(self, obs: torch.Tensor):
        feat_actor = self.actor_encoder(obs)
        logits_move = self.actor_move(feat_actor)
        logits_kick = self.actor_kick(feat_actor)
        value = self.critic(obs).squeeze(-1)
        return logits_move, logits_kick, value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
        **kwargs,
    ):
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

        return action, log_prob, entropy, value