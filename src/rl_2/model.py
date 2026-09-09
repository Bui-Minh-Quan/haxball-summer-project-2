# src/rl_2/model.py
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from src.rl_2.obs import ACTOR_OBS_DIM, CRITIC_STATE_DIM


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class ActorCritic(nn.Module):
    """Decoupled Two-Tower MAPPO Architecture."""

    def __init__(
        self,
        obs_dim: int = ACTOR_OBS_DIM,       # 64 dims
        state_dim: int = CRITIC_STATE_DIM,   # 58 dims
        move_dim: int = 9,
        kick_dim: int = 2,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.state_dim = state_dim

        # --- ACTOR TOWER (Decentralized Execution) ---
        self.actor_encoder = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, 256)),
            nn.LayerNorm(256),
            nn.GELU(),
            ResidualBlock(256),
            ResidualBlock(256),
            layer_init(nn.Linear(256, 256)),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        self.actor_move = layer_init(nn.Linear(256, move_dim), std=0.01)
        self.actor_kick = layer_init(nn.Linear(256, kick_dim), std=0.01)

        # --- CRITIC TOWER (Centralized Training: 512-Width High-Capacity) ---
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.state_dim, 512)),
            nn.LayerNorm(512),
            nn.GELU(),
            ResidualBlock(512),
            ResidualBlock(512),
            layer_init(nn.Linear(512, 256)),
            nn.LayerNorm(256),
            nn.GELU(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(state).squeeze(-1)

    def forward(self, obs: torch.Tensor, state: torch.Tensor | None = None):
        feat = self.actor_encoder(obs)
        logits_move = self.actor_move(feat)
        logits_kick = self.actor_kick(feat)

        if state is not None:
            value = self.critic(state).squeeze(-1)
        else:
            value = torch.zeros(obs.shape[0], device=obs.device)

        return logits_move, logits_kick, value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        state: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ):
        logits_move, logits_kick, value = self.forward(obs, state)
        dist_move = Categorical(logits=logits_move)
        dist_kick = Categorical(logits=logits_kick)

        if action is None:
            if deterministic:
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

    def load_actor_weights(self, checkpoint_path: str, device: torch.device):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

        actor_dict = {k: v for k, v in state_dict.items() if not k.startswith("critic")}
        self.load_state_dict(actor_dict, strict=False)
        print(f"✅ Bootstrapped Actor weights from: {checkpoint_path}")