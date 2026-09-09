import os
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import HTML, display

from config.match_config import MatchConfig, PlayerSlot, PlayerStats
from src.bots.heuristic_bot import TeamHeuristicCoordinator
from src.engine.controllers import Controller, HeuristicBotController
from src.engine.modes.classic_mode import ClassicMatchMode
from src.engine.simulation import Simulation
from src.engine.vector import Vec2
from src.rl_2.env_adapter import RandomOpponentController
from src.rl_2.model import ActorCritic
from src.rl_2.obs import extract_actor_obs


def render_match_video(
    model_path: str = "models/stage2/final_model.pt",
    opponent_type: str = "random",  # "random" or "heuristic"
    pitch_width: float = 1200.0,
    pitch_height: float = 800.0,
    goal_height: float = 220.0,
    max_steps: int = 900,
    frame_skip: int = 2,  # Renders every 2nd frame (30fps playback)
    save_path: str | None = None,  # e.g., "replay.mp4" or "replay.gif"
):
  device = torch.device("cpu")
  ego_dirs = [
      (0.0, 0.0),
      (0.0, -1.0),
      (0.0, 1.0),
      (-1.0, 0.0),
      (1.0, 0.0),
      (-1.0, -1.0),
      (1.0, -1.0),
      (-1.0, 1.0),
      (1.0, 1.0),
  ]

  # 1. Load Model
  model = ActorCritic().to(device)
  if os.path.exists(model_path):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = (
        ckpt["model_state_dict"]
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt
        else ckpt
    )
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded weights from: {model_path}")
  else:
    print(f"⚠️ Model path {model_path} not found. Running untrained weights.")
  model.eval()

  # 2. Setup Match
  class ActionPlaceholder(Controller):

    def __init__(self):
      self.action = (Vec2(0, 0), False)

    def get_action(self, idx, sim):
      return self.action

  learner_ctrl = ActionPlaceholder()
  if opponent_type == "heuristic":
    opp_ctrl = HeuristicBotController(TeamHeuristicCoordinator(team="blue"))
  else:
    opp_ctrl = RandomOpponentController()

  roster = [
      PlayerSlot(
          "red", PlayerStats("Learner", accel=3200.0), controller=learner_ctrl
      ),
      PlayerSlot(
          "blue", PlayerStats("Opponent", accel=3200.0), controller=opp_ctrl
      ),
  ]

  cfg = MatchConfig(
      mode=ClassicMatchMode(time_limit=max_steps / 60.0, score_limit=99),
      roster=roster,
      goal_height=goal_height,
      pitch_width=pitch_width,
      pitch_height=pitch_height,
  )
  sim = Simulation(match_config=cfg, goal_height=goal_height)

  # 3. Simulate and Record State History
  frames = []
  goal_event = None

  for step_idx in range(max_steps):
    red_player = sim.red_team[0]
    blue_player = sim.blue_team[0]

    # Model policy inference (Deterministic)
    obs = extract_actor_obs(sim, red_player, "red")
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(
        0
    )
    with torch.no_grad():
      act, _, _, _ = model.get_action_and_value(obs_t, deterministic=True)

    m_idx = int(act[0, 0].item())
    kick = bool(act[0, 1].item())
    ex, ey = ego_dirs[m_idx]
    learner_ctrl.action = (Vec2(ex, ey), kick)

    # Record frame snapshot
    if step_idx % frame_skip == 0:
      frames.append({
          "step": step_idx,
          "ball": (sim.ball.pos.x, sim.ball.pos.y),
          "ball_radius": sim.ball.radius,
          "red": (red_player.pos.x, red_player.pos.y),
          "red_radius": red_player.radius,
          "red_kick": kick,
          "blue": (blue_player.pos.x, blue_player.pos.y),
          "blue_radius": blue_player.radius,
          "score": (sim.score_red, sim.score_blue),
      })

    goal = sim.step(1.0 / 60.0)
    if goal is not None:
      goal_event = goal
      # Append final frame
      frames.append({
          "step": step_idx + 1,
          "ball": (sim.ball.pos.x, sim.ball.pos.y),
          "ball_radius": sim.ball.radius,
          "red": (red_player.pos.x, red_player.pos.y),
          "red_radius": red_player.radius,
          "red_kick": False,
          "blue": (blue_player.pos.x, blue_player.pos.y),
          "blue_radius": blue_player.radius,
          "score": (sim.score_red, sim.score_blue),
      })
      break

  print(
      f"Simulation finished: {len(frames)} rendered frames. Result:"
      f" {goal_event or 'Timeout (No Goal)'}"
  )

  # 4. Render Animation using Matplotlib
  fig, ax = plt.subplots(figsize=(10, 6.5), dpi=100)
  ax.set_facecolor("#2e5c38")  # Pitch grass green
  fig.patch.set_facecolor("#1a1a1a")

  p = sim.pitch
  ax.set_xlim(p.left - 60, p.right + 60)
  ax.set_ylim(p.bottom + 40, p.top - 40)  # Inverted Y for 2D screen coords
  ax.set_aspect("equal")
  ax.axis("off")

  # Draw Static Pitch Markings
  pitch_rect = patches.Rectangle(
      (p.left, p.top),
      p.width,
      p.height,
      linewidth=2.5,
      edgecolor="#ffffff",
      facecolor="none",
  )
  ax.add_patch(pitch_rect)

  # Center line & center circle
  ax.plot(
      [p.center.x, p.center.x], [p.top, p.bottom], color="#ffffff", linewidth=2
  )
  center_circle = patches.Circle(
      (p.center.x, p.center.y), 100.0, edgecolor="#ffffff", facecolor="none", linewidth=2
  )
  ax.add_patch(center_circle)

  # Goal boxes
  left_goal = patches.Rectangle(
      (p.left - 40, p.goal_top),
      40,
      p.goal_height,
      linewidth=2,
      edgecolor="#ffffff",
      facecolor="#1e3d25",
      alpha=0.6,
  )
  right_goal = patches.Rectangle(
      (p.right, p.goal_top),
      40,
      p.goal_height,
      linewidth=2,
      edgecolor="#ffffff",
      facecolor="#1e3d25",
      alpha=0.6,
  )
  ax.add_patch(left_goal)
  ax.add_patch(right_goal)

  # Dynamic Entity Artists
  red_artist = patches.Circle(
      (0, 0), frames[0]["red_radius"], facecolor="#e74c3c", edgecolor="#ffffff", linewidth=2
  )
  red_ring = patches.Circle(
      (0, 0),
      frames[0]["red_radius"] + 6,
      fill=False,
      edgecolor="#f1c40f",
      linewidth=2.5,
      visible=False,
  )
  blue_artist = patches.Circle(
      (0, 0),
      frames[0]["blue_radius"],
      facecolor="#3498db",
      edgecolor="#ffffff",
      linewidth=2,
  )
  ball_artist = patches.Circle(
      (0, 0),
      frames[0]["ball_radius"],
      facecolor="#ffffff",
      edgecolor="#000000",
      linewidth=1.5,
  )

  ax.add_patch(red_ring)
  ax.add_patch(red_artist)
  ax.add_patch(blue_artist)
  ax.add_patch(ball_artist)

  title_text = ax.text(
      p.center.x,
      p.top - 18,
      "",
      ha="center",
      va="center",
      color="#ffffff",
      fontsize=13,
      fontweight="bold",
  )

  def update(frame_data):
    # Update positions
    rx, ry = frame_data["red"]
    red_artist.center = (rx, ry)
    red_ring.center = (rx, ry)
    red_ring.set_visible(frame_data["red_kick"])  # Glow ring when agent kicks

    bx, by = frame_data["blue"]
    blue_artist.center = (bx, by)

    ball_x, ball_y = frame_data["ball"]
    ball_artist.center = (ball_x, ball_y)

    sec = frame_data["step"] / 60.0
    r_score, b_score = frame_data["score"]
    title_text.set_text(
        f"Step: {frame_data['step']:03d} ({sec:4.1f}s) | Score: [RED {r_score} -"
        f" {b_score} BLUE]"
    )
    return red_artist, red_ring, blue_artist, ball_artist, title_text

  ani = animation.FuncAnimation(
      fig, update, frames=frames, interval=1000 / (60 / frame_skip), blit=True
  )
  plt.close(fig)

  if save_path:
    if save_path.endswith(".gif"):
      ani.save(save_path, writer="pillow", fps=int(60 / frame_skip))
    else:
      ani.save(save_path, writer="ffmpeg", fps=int(60 / frame_skip))
    print(f"🎬 Video saved to: {save_path}")

  return HTML(ani.to_jshtml())