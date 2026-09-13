import torch
import torch.nn as nn
import math
from torch.distributions.categorical import Categorical

from src.rl_transformer.entity_obs import (
    BALL_DIM,
    EGO_DIM,
    PLAYER_DIM,
)


def layer_init(layer: nn.Linear, std: float = math.sqrt(2), bias_const: float = 0.0):
  """Orthogonal initialization standard for stable PPO dynamics."""
  nn.init.orthogonal_(layer.weight, std)
  nn.init.constant_(layer.bias, bias_const)
  return layer


class TransformerActor(nn.Module):
  """Permutation-invariant Entity-Transformer Actor.

  Processes local ego perspective and reads out categorical movement and kick actions.
  """

  def __init__(
      self,
      ego_dim: int = EGO_DIM,          # 9
      ball_dim: int = BALL_DIM,        # 10
      player_dim: int = PLAYER_DIM,    # 7
      d_model: int = 64,
      nhead: int = 4,
      num_layers: int = 2,
  ):
    super().__init__()
    self.d_model = d_model

    # 1. Dedicated feature encoders for each entity type
    self.ego_encoder = nn.Sequential(
        layer_init(nn.Linear(ego_dim, d_model)),
        nn.LayerNorm(d_model),
        nn.ReLU(),
    )
    self.ball_encoder = nn.Sequential(
        layer_init(nn.Linear(ball_dim, d_model)),
        nn.LayerNorm(d_model),
        nn.ReLU(),
    )
    self.teammate_encoder = nn.Sequential(
        layer_init(nn.Linear(player_dim, d_model)),
        nn.LayerNorm(d_model),
        nn.ReLU(),
    )
    self.opponent_encoder = nn.Sequential(
        layer_init(nn.Linear(player_dim, d_model)),
        nn.LayerNorm(d_model),
        nn.ReLU(),
    )

    # 2. Learnable categorical type embeddings: 0: Ego, 1: Ball, 2: Teammate, 3: Opponent
    self.type_embed = nn.Embedding(4, d_model)

    # 3. Transformer Encoder Backbone (Zero Dropout for PPO on-policy stability)
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=d_model * 2,
        dropout=0.0,
        activation="relu",
        batch_first=True,
        norm_first=True,
    )
    self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

    # 4. Action Readout Heads (operating on contextualized Ego token at Index 0)
    self.move_head = layer_init(nn.Linear(d_model, 9), std=0.01)
    self.kick_head = layer_init(nn.Linear(d_model, 2), std=0.01)

  def forward(
      self,
      ego: torch.Tensor,
      ball: torch.Tensor,
      teammates: torch.Tensor,
      opponents: torch.Tensor,
      key_padding_mask: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    # Project raw features to d_model
    e_ego = self.ego_encoder(ego).unsqueeze(1) + self.type_embed(
        torch.tensor(0, device=ego.device)
    )
    e_ball = self.ball_encoder(ball).unsqueeze(1) + self.type_embed(
        torch.tensor(1, device=ball.device)
    )
    e_mates = self.teammate_encoder(teammates) + self.type_embed(
        torch.tensor(2, device=teammates.device)
    )
    e_opps = self.opponent_encoder(opponents) + self.type_embed(
        torch.tensor(3, device=opponents.device)
    )

    # Sequence shape: [Batch, Tokens=7, d_model]
    tokens = torch.cat([e_ego, e_ball, e_mates, e_opps], dim=1)

    h = self.transformer(tokens, src_key_padding_mask=key_padding_mask)

    # Readout from contextualized Ego token (Index 0)
    h_ego = h[:, 0, :]
    move_logits = self.move_head(h_ego)
    kick_logits = self.kick_head(h_ego)

    return move_logits, kick_logits


class TransformerCritic(nn.Module):
  """Centralized Entity-Transformer Critic.

  Processes pitch-centric global entities, applies masked mean pooling across all tokens,
  and outputs a scalar state-value estimate V(s).
  """

  def __init__(
      self,
      ball_dim: int = BALL_DIM,        # 10
      player_dim: int = PLAYER_DIM,    # 7
      d_model: int = 64,
      nhead: int = 4,
      num_layers: int = 2,
  ):
    super().__init__()
    self.d_model = d_model

    # 1. Dedicated feature encoders for pitch-centric entities
    self.ball_encoder = nn.Sequential(
        layer_init(nn.Linear(ball_dim, d_model)),
        nn.LayerNorm(d_model),
        nn.ReLU(),
    )
    self.learner_encoder = nn.Sequential(
        layer_init(nn.Linear(player_dim, d_model)),
        nn.LayerNorm(d_model),
        nn.ReLU(),
    )
    self.opponent_encoder = nn.Sequential(
        layer_init(nn.Linear(player_dim, d_model)),
        nn.LayerNorm(d_model),
        nn.ReLU(),
    )

    # 2. Learnable categorical type embeddings: 0: Ball, 1: Learner, 2: Opponent
    self.type_embed = nn.Embedding(3, d_model)

    # 3. Transformer Encoder Backbone
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=d_model * 2,
        dropout=0.0,
        activation="relu",
        batch_first=True,
        norm_first=True,
    )
    self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

    # 4. Value Head projecting pooled pitch representation to V(s)
    self.value_head = nn.Sequential(
        layer_init(nn.Linear(d_model, d_model)),
        nn.ReLU(),
        layer_init(nn.Linear(d_model, 1), std=1.0),
    )

  def forward(
      self,
      ball: torch.Tensor,
      learners: torch.Tensor,
      opponents: torch.Tensor,
      key_padding_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    e_ball = self.ball_encoder(ball).unsqueeze(1) + self.type_embed(
        torch.tensor(0, device=ball.device)
    )
    e_learners = self.learner_encoder(learners) + self.type_embed(
        torch.tensor(1, device=learners.device)
    )
    e_opps = self.opponent_encoder(opponents) + self.type_embed(
        torch.tensor(2, device=opponents.device)
    )

    # Sequence shape: [Batch, Tokens=7, d_model]
    tokens = torch.cat([e_ball, e_learners, e_opps], dim=1)

    h = self.transformer(tokens, src_key_padding_mask=key_padding_mask)

    # Masked mean pooling across all active entities
    if key_padding_mask is not None:
      mask = (~key_padding_mask).unsqueeze(-1).float()  # True=ignore -> invert
      h_pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    else:
      h_pooled = h.mean(dim=1)

    value = self.value_head(h_pooled)
    return value.squeeze(-1)


class TransformerActorCritic(nn.Module):
  """Unified container housing both TransformerActor and TransformerCritic."""

  def __init__(
      self,
      ego_dim: int = EGO_DIM,
      ball_dim: int = BALL_DIM,
      player_dim: int = PLAYER_DIM,
      d_model: int = 64,
      nhead: int = 4,
      num_layers: int = 2,
  ):
    super().__init__()
    self.actor = TransformerActor(
        ego_dim=ego_dim,
        ball_dim=ball_dim,
        player_dim=player_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
    )
    self.critic = TransformerCritic(
        ball_dim=ball_dim,
        player_dim=player_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
    )

  def get_action_and_value(
      self,
      actor_obs: dict[str, torch.Tensor],
      critic_obs: dict[str, torch.Tensor] | None = None,
      action: torch.Tensor | None = None,
      deterministic: bool = False,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    move_logits, kick_logits = self.actor(
        ego=actor_obs["ego"],
        ball=actor_obs["ball"],
        teammates=actor_obs["teammates"],
        opponents=actor_obs["opponents"],
        key_padding_mask=actor_obs.get("key_padding_mask", None),
    )

    dist_move = Categorical(logits=move_logits)
    dist_kick = Categorical(logits=kick_logits)

    if action is None:
      if deterministic:
        act_move = torch.argmax(move_logits, dim=-1)
        act_kick = torch.argmax(kick_logits, dim=-1)
      else:
        act_move = dist_move.sample()
        act_kick = dist_kick.sample()
      action = torch.stack([act_move, act_kick], dim=-1)
    else:
      act_move = action[:, 0]
      act_kick = action[:, 1]

    logprob = dist_move.log_prob(act_move) + dist_kick.log_prob(act_kick)
    entropy = dist_move.entropy() + dist_kick.entropy()

    value = None
    if critic_obs is not None:
      value = self.critic(
          ball=critic_obs["ball"],
          learners=critic_obs["learners"],
          opponents=critic_obs["opponents"],
          key_padding_mask=critic_obs.get("key_padding_mask", None),
      )

    return action, logprob, entropy, value

  def get_value(self, critic_obs: dict[str, torch.Tensor]) -> torch.Tensor:
    return self.critic(
        ball=critic_obs["ball"],
        learners=critic_obs["learners"],
        opponents=critic_obs["opponents"],
        key_padding_mask=critic_obs.get("key_padding_mask", None),
    )