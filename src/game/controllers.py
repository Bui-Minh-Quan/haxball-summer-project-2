from pathlib import Path
from typing import Any
import numpy as np
import onnxruntime as ort
import pygame

from src.engine.controllers import Controller
from src.engine.vector import Vec2
from src.rl.obs import extract_actor_obs

# Global session cache to avoid duplicate model loads across player slots
_ONNX_SESSIONS: dict[str, ort.InferenceSession] = {}


def get_or_create_onnx_session(model_path: str | Path) -> ort.InferenceSession:
  """Retrieves a cached ONNX InferenceSession or instantiates a single-threaded CPU session."""
  resolved_path = str(Path(model_path).resolve())
  if resolved_path not in _ONNX_SESSIONS:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _ONNX_SESSIONS[resolved_path] = ort.InferenceSession(
        resolved_path,
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
  return _ONNX_SESSIONS[resolved_path]


class KeyboardController(Controller):
  """Polls Pygame keyboard state for a human player slot."""

  def __init__(
      self,
      up=pygame.K_w,
      down=pygame.K_s,
      left=pygame.K_a,
      right=pygame.K_d,
      kick=pygame.K_SPACE,
  ):
    self.up = up
    self.down = down
    self.left = left
    self.right = right
    self.kick = kick

  def get_action(self, player_idx: int, sim: Any) -> tuple[Vec2, bool]:
    keys = pygame.key.get_pressed()
    move = Vec2(0.0, 0.0)
    if keys[self.up]:
      move.y -= 1.0
    if keys[self.down]:
      move.y += 1.0
    if keys[self.left]:
      move.x -= 1.0
    if keys[self.right]:
      move.x += 1.0
    is_kicking = keys[self.kick]
    return move, is_kicking


class ONNXBotController(Controller):
  """Ultra-fast ONNX Runtime inference controller for 1v1, 2v2, and 3v3."""

  def __init__(self, model_path: str | Path, team: str):
    self.session = get_or_create_onnx_session(model_path)
    self.team = team.lower()
    self.sign = 1.0 if self.team == "red" else -1.0

    # Ego-centric discrete move mapping: index -> (dx, dy)
    self._ego_dirs = [
        (0.0, 0.0),   # 0: None
        (0.0, -1.0),  # 1: Up
        (0.0, 1.0),   # 2: Down
        (-1.0, 0.0),  # 3: Backward
        (1.0, 0.0),   # 4: Forward
        (-1.0, -1.0), # 5: Backward-Up
        (1.0, -1.0),  # 6: Forward-Up
        (-1.0, 1.0),  # 7: Backward-Down
        (1.0, 1.0),   # 8: Forward-Down
    ]

  def get_action(self, player_idx: int, sim: Any) -> tuple[Vec2, bool]:
    player = sim.all_players[player_idx]
    
    # 1. Extract Gen 2 64-dim ego observation
    obs = extract_actor_obs(sim, player, self.team)

    # 2. Batch dim injection: (64,) -> (1, 64)
    feed = {"obs": obs[np.newaxis, :]}
    logits_move, logits_kick = self.session.run(None, feed)

    # 3. Deterministic Argmax
    m_idx = int(np.argmax(logits_move, axis=-1)[0])
    kick_val = bool(np.argmax(logits_kick, axis=-1)[0])

    # 4. Map ego direction back to pitch world coordinates
    ego_x, ego_y = self._ego_dirs[m_idx]
    world_move = Vec2(ego_x * self.sign, ego_y)

    return world_move, kick_val