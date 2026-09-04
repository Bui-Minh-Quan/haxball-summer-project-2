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
    """Decoupled Two-Tower MAPPO Architecture."""

    def __init__(
        self, 
        obs_dim: int = 80, 
        state_dim: int | None = None, 
        move_dim: int = 9, 
        kick_dim: int = 2
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.state_dim = state_dim if state_dim is not None else obs_dim

        # --- ACTOR TOWER (Decentralized Execution) ---
        self.actor_encoder = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, 256)),
            nn.LayerNorm(256),
            nn.GELU(),
            ResidualBlock(256),
            layer_init(nn.Linear(256, 128)),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.actor_move = layer_init(nn.Linear(128, move_dim), std=0.01)
        self.actor_kick = layer_init(nn.Linear(128, kick_dim), std=0.01)

        # --- CRITIC TOWER (Centralized Training) ---
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.state_dim, 256)),
            nn.LayerNorm(256),
            nn.GELU(),
            ResidualBlock(256),
            layer_init(nn.Linear(256, 128)),
            nn.LayerNorm(128),
            nn.GELU(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Critic evaluates the centralized state."""
        return self.critic(state).squeeze(-1)

    def forward(self, obs: torch.Tensor, state: torch.Tensor | None = None):
        feat_actor = self.actor_encoder(obs)
        logits_move = self.actor_move(feat_actor)
        logits_kick = self.actor_kick(feat_actor)

        # 1. Centralized Training: Critic consumes global state
        if state is not None:
            value = self.critic(state).squeeze(-1)
        # 2. Stages 1-3 Backward Compatibility (IPPO 1v1 where state_dim == 80)
        elif self.state_dim == self.obs_dim:
            value = self.critic(obs).squeeze(-1)
        # 3. Decentralized Execution Fallback (Evaluation / Replay generation)
        else:
            value = torch.zeros(obs.shape[0], device=obs.device)

        return logits_move, logits_kick, value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        state: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
        **kwargs,
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

    def load_actor_weights(self, path: str, device: torch.device):
        """Bootstraps Actor weights from Stage 3 while initializing a fresh Critic."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

        actor_dict = {k: v for k, v in state_dict.items() if not k.startswith("critic")}
        self.load_state_dict(actor_dict, strict=False)
        print(f"✅ Bootstrapped Decentralized Actor weights from: {path}")