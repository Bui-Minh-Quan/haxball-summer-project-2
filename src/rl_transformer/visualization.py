import json
import math
import os
import random
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from collections import deque

from config.match_config import MatchConfig, PlayerSlot, PlayerStats
from src.bots.heuristic_bot import TeamHeuristicCoordinator
from src.engine.controllers import Controller, HeuristicBotController
from src.engine.modes.classic_mode import ClassicMatchMode
from src.engine.simulation import Simulation
from src.engine.vector import Vec2
from src.rl_transformer.entity_obs import extract_entity_obs
from src.rl_transformer.transformer_model import TransformerActorCritic

_EGO_DIRS = [
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


class ActionPlaceholder(Controller):
  """Allows external controllers to inject actions per physics frame."""

  def __init__(self):
    self.action = (Vec2(0.0, 0.0), False)

  def get_action(self, player_idx: int, sim: Any) -> tuple[Vec2, bool]:
    return self.action


def _load_agent(agent_spec: Any, team: str, device: torch.device):
  """Resolves agent paths, PyTorch modules, or heuristic bots into callable policies."""
  if isinstance(agent_spec, str):
    if agent_spec.lower() in ("heuristic", "bot"):
      coord = TeamHeuristicCoordinator(team=team)
      return "Heuristic", HeuristicBotController(coord), False
    elif agent_spec.lower() == "random":
      return "Random", None, False
    elif os.path.exists(agent_spec):
      model = TransformerActorCritic().to(device)
      ckpt = torch.load(agent_spec, map_location=device, weights_only=False)
      state_dict = (
          ckpt["model_state_dict"]
          if isinstance(ckpt, dict) and "model_state_dict" in ckpt
          else ckpt
      )
      model.load_state_dict(state_dict, strict=True)
      model.eval()
      label = os.path.splitext(os.path.basename(agent_spec))[0]
      return label, model, True
    else:
      raise FileNotFoundError(f"Agent checkpoint not found: {agent_spec}")
  elif isinstance(agent_spec, nn.Module):
    agent_spec.eval()
    return "Transformer", agent_spec.to(device), True
  else:
    raise ValueError(f"Unsupported agent specification: {agent_spec}")


def _build_html_template(pitch_data: dict, episodes_data: list) -> str:
  episodes_json = json.dumps(episodes_data)
  pitch_json = json.dumps(pitch_data)

  return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Haxball Multi-Agent Match Viewer</title>
<style>
  body {{
    margin: 0;
    padding: 20px;
    background: #0f172a;
    color: #e2e8f0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
  }}
  .container {{
    max-width: 1060px;
    width: 100%;
    background: #1e293b;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.5);
  }}
  .header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #334155;
    padding-bottom: 12px;
    margin-bottom: 16px;
  }}
  .canvas-wrapper {{
    position: relative;
    width: 100%;
    display: flex;
    justify-content: center;
    background: #0b1120;
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 16px;
    border: 1px solid #334155;
  }}
  canvas {{
    display: block;
    background: #14532d;
  }}
  .controls {{
    display: flex;
    flex-direction: column;
    gap: 12px;
  }}
  .control-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
  }}
  .btn-group {{
    display: flex;
    gap: 8px;
  }}
  button, select {{
    background: #334155;
    color: #f8fafc;
    border: 1px solid #475569;
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 14px;
    cursor: pointer;
    font-weight: 500;
  }}
  button:hover, select:hover {{
    background: #475569;
  }}
  button.active {{
    background: #2563eb;
    border-color: #3b82f6;
  }}
  .scrubber-container {{
    display: flex;
    align-items: center;
    gap: 10px;
    width: 100%;
  }}
  input[type="range"] {{
    flex: 1;
    accent-color: #3b82f6;
  }}
  .telemetry {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    background: #0f172a;
    padding: 12px;
    border-radius: 6px;
    font-family: monospace;
    font-size: 13px;
    margin-top: 12px;
  }}
  .telemetry-item span {{
    color: #94a3b8;
    display: block;
    font-size: 11px;
    text-transform: uppercase;
  }}
</style>
</head>
<body>

<div class="container">
  <div class="header">
    <h2 style="margin:0;">⚡ Match Replay Visualizer</h2>
    <div>
      <label for="epSelect">Episode: </label>
      <select id="epSelect"></select>
    </div>
  </div>

  <div class="canvas-wrapper">
    <canvas id="pitchCanvas" width="960" height="560"></canvas>
  </div>

  <div class="controls">
    <div class="scrubber-container">
      <span id="timeDisplay" style="font-family: monospace; min-width: 60px;">0.00s</span>
      <input type="range" id="scrubber" min="0" max="0" value="0">
      <span id="frameDisplay" style="font-family: monospace; min-width: 80px;">0 / 0</span>
    </div>

    <div class="control-row">
      <div class="btn-group">
        <button id="btnPlay">▶ Play</button>
        <button id="btnPrev">|◀ Step</button>
        <button id="btnNext">Step ▶|</button>
        <button id="btnReset">↺ Reset</button>
      </div>

      <div class="btn-group">
        <label style="align-self:center; font-size:14px;">Speed: </label>
        <button class="btnSpeed" data-speed="0.25">0.25x</button>
        <button class="btnSpeed active" data-speed="1.0">1.0x</button>
        <button class="btnSpeed" data-speed="2.0">2.0x</button>
        <button class="btnSpeed" data-speed="4.0">4.0x</button>
      </div>
    </div>
  </div>

  <div class="telemetry">
    <div class="telemetry-item">
      <span>Score</span>
      <strong id="telScore" style="color: #f59e0b; font-size: 16px;">0 - 0</strong>
    </div>
    <div class="telemetry-item">
      <span>Matchup</span>
      <strong id="telMatchup">-</strong>
    </div>
    <div class="telemetry-item">
      <span>Ball Velocity</span>
      <strong id="telBallVel">0.0 px/s</strong>
    </div>
    <div class="telemetry-item">
      <span>Active Kicking</span>
      <strong id="telKickState">None</strong>
    </div>
  </div>
</div>

<script>
const pitch = {pitch_json};
const episodes = {episodes_json};

let currentEpIdx = 0;
let currentFrameIdx = 0;
let isPlaying = false;
let playbackSpeed = 1.0;
let lastAnimTime = 0;
let frameAccumulator = 0;

const canvas = document.getElementById("pitchCanvas");
const ctx = canvas.getContext("2d");

const epSelect = document.getElementById("epSelect");
const scrubber = document.getElementById("scrubber");
const btnPlay = document.getElementById("btnPlay");
const btnPrev = document.getElementById("btnPrev");
const btnNext = document.getElementById("btnNext");
const btnReset = document.getElementById("btnReset");
const timeDisplay = document.getElementById("timeDisplay");
const frameDisplay = document.getElementById("frameDisplay");

const telScore = document.getElementById("telScore");
const telMatchup = document.getElementById("telMatchup");
const telBallVel = document.getElementById("telBallVel");
const telKickState = document.getElementById("telKickState");

episodes.forEach((ep, idx) => {{
  const opt = document.createElement("option");
  opt.value = idx;
  opt.textContent = `Match ${{ep.episode_idx}}: ${{ep.red_agent}} vs ${{ep.blue_agent}} (${{ep.final_score}})`;
  epSelect.appendChild(opt);
}});

epSelect.addEventListener("change", (e) => {{
  loadEpisode(parseInt(e.target.value));
}});

function loadEpisode(idx) {{
  currentEpIdx = idx;
  currentFrameIdx = 0;
  const ep = episodes[currentEpIdx];
  scrubber.max = Math.max(0, ep.frames.length - 1);
  scrubber.value = 0;
  telMatchup.innerHTML = `<span style="color:#ef4444">${{ep.red_agent}}</span> vs <span style="color:#3b82f6">${{ep.blue_agent}}</span>`;
  renderFrame();
}}

scrubber.addEventListener("input", (e) => {{
  currentFrameIdx = parseInt(e.target.value);
  renderFrame();
}});

btnPlay.addEventListener("click", () => {{
  isPlaying = !isPlaying;
  btnPlay.textContent = isPlaying ? "⏸ Pause" : "▶ Play";
  btnPlay.classList.toggle("active", isPlaying);
  if (isPlaying) {{
    lastAnimTime = performance.now();
    requestAnimationFrame(animationLoop);
  }}
}});

btnPrev.addEventListener("click", () => {{
  if (currentFrameIdx > 0) {{
    currentFrameIdx--;
    scrubber.value = currentFrameIdx;
    renderFrame();
  }}
}});

btnNext.addEventListener("click", () => {{
  const ep = episodes[currentEpIdx];
  if (currentFrameIdx < ep.frames.length - 1) {{
    currentFrameIdx++;
    scrubber.value = currentFrameIdx;
    renderFrame();
  }}
}});

btnReset.addEventListener("click", () => {{
  currentFrameIdx = 0;
  scrubber.value = 0;
  renderFrame();
}});

document.querySelectorAll(".btnSpeed").forEach(btn => {{
  btn.addEventListener("click", (e) => {{
    document.querySelectorAll(".btnSpeed").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    playbackSpeed = parseFloat(btn.dataset.speed);
  }});
}});

function animationLoop(timestamp) {{
  if (!isPlaying) return;

  const dt = (timestamp - lastAnimTime) / 1000.0;
  lastAnimTime = timestamp;

  frameAccumulator += dt * 60.0 * playbackSpeed;
  const ep = episodes[currentEpIdx];

  while (frameAccumulator >= 1.0) {{
    if (currentFrameIdx < ep.frames.length - 1) {{
      currentFrameIdx++;
    }} else {{
      isPlaying = false;
      btnPlay.textContent = "▶ Play";
      btnPlay.classList.remove("active");
      break;
    }}
    frameAccumulator -= 1.0;
  }}

  scrubber.value = currentFrameIdx;
  renderFrame();

  if (isPlaying) {{
    requestAnimationFrame(animationLoop);
  }}
}}

function renderFrame() {{
  const ep = episodes[currentEpIdx];
  if (!ep || !ep.frames || ep.frames.length === 0) return;
  const frame = ep.frames[currentFrameIdx];
  if (!frame) return;

  timeDisplay.textContent = frame.time.toFixed(2) + "s";
  frameDisplay.textContent = `${{frame.step}} / ${{ep.frames.length - 1}}`;
  telScore.textContent = `${{frame.score_red}} - ${{frame.score_blue}}`;

  const ballSpeed = Math.hypot(frame.ball.vx, frame.ball.vy);
  telBallVel.textContent = ballSpeed.toFixed(1) + " px/s";

  const kickingPlayers = frame.players.filter(p => p.is_kicking);
  if (kickingPlayers.length > 0) {{
    telKickState.textContent = kickingPlayers.map(p => p.name).join(", ");
    telKickState.style.color = "#fbbf24";
  }} else {{
    telKickState.textContent = "None";
    telKickState.style.color = "#94a3b8";
  }}

  const margin = 40;
  const scaleX = (canvas.width - margin * 2) / (pitch.right - pitch.left);
  const scaleY = (canvas.height - margin * 2) / (pitch.bottom - pitch.top);
  const scale = Math.min(scaleX, scaleY);

  const toScreenX = (x) => margin + (x - pitch.left) * scale;
  const toScreenY = (y) => margin + (y - pitch.top) * scale;

  ctx.fillStyle = "#15803d";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = "rgba(255, 255, 255, 0.7)";
  ctx.lineWidth = 3;

  const left = toScreenX(pitch.left);
  const right = toScreenX(pitch.right);
  const top = toScreenY(pitch.top);
  const bottom = toScreenY(pitch.bottom);
  const centerX = (left + right) / 2;
  const centerY = (top + bottom) / 2;

  ctx.strokeRect(left, top, right - left, bottom - top);

  ctx.beginPath();
  ctx.moveTo(centerX, top);
  ctx.lineTo(centerX, bottom);
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(centerX, centerY, 70 * scale, 0, Math.PI * 2);
  ctx.stroke();

  const goalTop = toScreenY(pitch.goal_top);
  const goalBottom = toScreenY(pitch.goal_bottom);
  const goalWidth = 25 * scale;

  ctx.fillStyle = "rgba(255, 255, 255, 0.15)";
  ctx.fillRect(left - goalWidth, goalTop, goalWidth, goalBottom - goalTop);
  ctx.strokeRect(left - goalWidth, goalTop, goalWidth, goalBottom - goalTop);

  ctx.fillRect(right, goalTop, goalWidth, goalBottom - goalTop);
  ctx.strokeRect(right, goalTop, goalWidth, goalBottom - goalTop);

  // Render Players with Jersey Numbers
  frame.players.forEach((p, idx) => {{
    const px = toScreenX(p.x);
    const py = toScreenY(p.y);
    const pr = p.r * scale;

    if (p.is_kicking) {{
      ctx.beginPath();
      ctx.arc(px, py, pr + 6, 0, Math.PI * 2);
      ctx.strokeStyle = "#fbbf24";
      ctx.lineWidth = 4;
      ctx.stroke();
    }}

    ctx.beginPath();
    ctx.arc(px, py, pr, 0, Math.PI * 2);
    ctx.fillStyle = p.team === "red" ? "#dc2626" : "#2563eb";
    ctx.fill();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Player Number Tag
    ctx.fillStyle = "#ffffff";
    ctx.font = `bold ${{Math.round(11 * scale)}}px sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const num = p.name.split("_")[1] || (idx + 1);
    ctx.fillText(num, px, py);

    // Orientation Heading
    if (Math.hypot(p.vx, p.vy) > 10) {{
      const headingAngle = Math.atan2(p.vy, p.vx);
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(px + Math.cos(headingAngle) * pr * 1.5, py + Math.sin(headingAngle) * pr * 1.5);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.8)";
      ctx.lineWidth = 2;
      ctx.stroke();
    }}
  }});

  // Render Ball
  const bx = toScreenX(frame.ball.x);
  const by = toScreenY(frame.ball.y);
  const br = frame.ball.r * scale;

  ctx.beginPath();
  ctx.arc(bx + 2, by + 3, br, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(0, 0, 0, 0.35)";
  ctx.fill();

  ctx.beginPath();
  ctx.arc(bx, by, br, 0, Math.PI * 2);
  ctx.fillStyle = "#ffffff";
  ctx.fill();
  ctx.strokeStyle = "#000000";
  ctx.lineWidth = 1.5;
  ctx.stroke();

  if (frame.goal_event) {{
    ctx.fillStyle = "rgba(251, 191, 36, 0.4)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.font = "bold 36px sans-serif";
    ctx.fillStyle = "#ffffff";
    ctx.textAlign = "center";
    ctx.fillText("⚽ GOAL!", canvas.width / 2, 70);
  }}
}}

if (episodes && episodes.length > 0) {{
  loadEpisode(0);
}}
</script>
</body>
</html>
"""


def _apply_dynamic_restart(sim: Simulation, ep_idx: int, is_initial: bool = False):
  """Sweeps ball coordinates and player spawn angles to ensure diverse match scenarios."""
  p = sim.pitch
  safe_m = 50.0

  if (ep_idx % 5 == 0) and is_initial:
    bx, by = sim.center.x, sim.center.y
    dist = min(140.0, p.width * 0.16)
    angle = 0.0
  else:
    sweep = ep_idx if is_initial else (ep_idx + int(sim.score_red + sim.score_blue) * 7)
    dist = min(220.0, p.width * 0.25) * (0.70 + (sweep % 4) * 0.08)
    angle = -math.pi / 4 + ((sweep * 31.0) % 90.0) * (math.pi / 180.0)
    bx = sim.center.x + (((sweep % 3) - 1) * (p.width * 0.12))
    by = sim.center.y + ((((sweep // 3) % 3) - 1) * (p.height * 0.14))

  sim.ball.pos = Vec2(bx, by)
  sim.ball.vel = Vec2(0.0, 0.0)

  vx, vy = dist * math.cos(angle), dist * math.sin(angle)
  rx_lead, ry_lead = bx - vx, by - vy
  bx_lead, by_lead = bx + vx, by + vy

  for idx, pl in enumerate(sim.red_team):
    if idx == 0:
      px = max(p.left + safe_m, min(p.right - safe_m, rx_lead))
      py = max(p.top + safe_m, min(p.bottom - safe_m, ry_lead))
    else:
      back_offset = min(240.0, max(100.0, (rx_lead - p.left) * 0.40)) * (1.0 + (idx - 1) * 0.4)
      px = max(p.left + safe_m, rx_lead - back_offset)
      lane_sign = 1.0 if (idx % 2 == 1) else -1.0
      y_offset = lane_sign * (65.0 + (idx // 2) * 55.0)
      py = min(max(p.top + safe_m, sim.center.y + y_offset), p.bottom - safe_m)
    pl.pos = Vec2(px, py)
    pl.vel = Vec2(0.0, 0.0)
    pl.kick_cooldown_timer = 0.0

  for idx, pl in enumerate(sim.blue_team):
    if idx == 0:
      px = max(p.left + safe_m, min(p.right - safe_m, bx_lead))
      py = max(p.top + safe_m, min(p.bottom - safe_m, by_lead))
    else:
      back_offset = min(240.0, max(100.0, (p.right - bx_lead) * 0.40)) * (1.0 + (idx - 1) * 0.4)
      px = min(p.right - safe_m, bx_lead + back_offset)
      lane_sign = -1.0 if (idx % 2 == 1) else 1.0
      y_offset = lane_sign * (65.0 + (idx // 2) * 55.0)
      py = min(max(p.top + safe_m, sim.center.y + y_offset), p.bottom - safe_m)
    pl.pos = Vec2(px, py)
    pl.vel = Vec2(0.0, 0.0)
    pl.kick_cooldown_timer = 0.0

  if hasattr(sim, "mode"):
    sim.mode.state = "PLAYING"

def evaluate_and_generate_html(
    red_agent: str | nn.Module,
    blue_agent: str | nn.Module = "heuristic",
    red_team_size: int = 1,
    blue_team_size: int = 1,
    device: torch.device = torch.device("cpu"),
    output_dir: str = "render/",
    filename: str = "match_replay.html",
    num_episodes: int = 5,
    max_steps: int = 1800,
    action_repeat: int = 10,
    pitch_width: float = 1200.0,
    pitch_height: float = 800.0,
    goal_height: float = 220.0,
    frame_stack: int = 3,
    seed: int = 42,
) -> str:
  """Executes matches, records 60 FPS state telemetry with temporal frame stacking, and produces a standalone HTML visualizer."""
  os.makedirs(output_dir, exist_ok=True)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  red_label, red_model, is_red_model = _load_agent(red_agent, "red", device)
  blue_label, blue_model, is_blue_model = _load_agent(blue_agent, "blue", device)

  episodes_data = []

  for ep in range(num_episodes):
    red_phs = [ActionPlaceholder() for _ in range(red_team_size)]
    blue_phs = [ActionPlaceholder() for _ in range(blue_team_size)]

    roster = []
    for i in range(red_team_size):
      roster.append(
          PlayerSlot("red", PlayerStats(f"R_{i+1}", accel=3200.0), red_phs[i])
      )
    for j in range(blue_team_size):
      roster.append(
          PlayerSlot("blue", PlayerStats(f"B_{j+1}", accel=3200.0), blue_phs[j])
      )

    cfg = MatchConfig(
        mode=ClassicMatchMode(time_limit=max_steps / 60.0, score_limit=99),
        roster=roster,
        pitch_width=pitch_width,
        pitch_height=pitch_height,
        goal_height=goal_height,
    )

    sim = Simulation(
        center_x=pitch_width / 2.0,
        center_y=pitch_height / 2.0,
        match_config=cfg,
        goal_height=goal_height,
    )
    if hasattr(sim, "mode"):
      sim.mode.state = "PLAYING"
      if hasattr(sim.mode, "score_limit"):
        sim.mode.score_limit = 999  # Disable 3-goal cap
    if hasattr(sim, "match_config") and hasattr(sim.match_config, "score_limit"):
      sim.match_config.score_limit = 999


    _apply_dynamic_restart(sim, ep_idx=ep, is_initial=True)

    # Frame stacking history queues per player
    red_histories = [deque(maxlen=frame_stack) for _ in range(red_team_size)]
    blue_histories = [deque(maxlen=frame_stack) for _ in range(blue_team_size)]

    frames = []
    dt = 1.0 / 60.0

    for step in range(max_steps):
      # ── 1. Action Decision Tick ──
      if step % action_repeat == 0:
        # Red Team Decision
        if is_red_model:
          for pl_idx, pl in enumerate(sim.red_team):
            tokens = extract_entity_obs(sim, pl, "red")
            h_dq = red_histories[pl_idx]

            if len(h_dq) == 0:
              for _ in range(frame_stack):
                h_dq.append(tokens)
            else:
              h_dq.append(tokens)

            stacked_ego = np.concatenate([f["ego"] for f in h_dq], axis=-1)
            stacked_ball = np.concatenate([f["ball"] for f in h_dq], axis=-1)
            stacked_mates = np.concatenate([f["teammates"] for f in h_dq], axis=-1)
            stacked_opps = np.concatenate([f["opponents"] for f in h_dq], axis=-1)

            actor_obs = {
                "ego": torch.as_tensor(stacked_ego, dtype=torch.float32, device=device).unsqueeze(0),
                "ball": torch.as_tensor(stacked_ball, dtype=torch.float32, device=device).unsqueeze(0),
                "teammates": torch.as_tensor(stacked_mates, dtype=torch.float32, device=device).unsqueeze(0),
                "opponents": torch.as_tensor(stacked_opps, dtype=torch.float32, device=device).unsqueeze(0),
                "key_padding_mask": torch.as_tensor(tokens["key_padding_mask"], dtype=torch.bool, device=device).unsqueeze(0),
            }
            with torch.inference_mode():
              act, _, _, _ = red_model.get_action_and_value(actor_obs, deterministic=True)
            m_idx = int(act[0, 0].item())
            k_val = bool(act[0, 1].item())
            ex, ey = _EGO_DIRS[m_idx]
            red_phs[pl_idx].action = (Vec2(ex, ey), k_val)
        elif red_model is not None:
          for pl_idx, pl in enumerate(sim.red_team):
            red_phs[pl_idx].action = red_model.get_action(sim.all_players.index(pl), sim)
        else:
          for pl_idx in range(red_team_size):
            dx, dy = random.choice(_EGO_DIRS)
            red_phs[pl_idx].action = (Vec2(dx, dy), random.random() < 0.20)

        # Blue Team Decision
        if is_blue_model:
          for pl_idx, pl in enumerate(sim.blue_team):
            tokens = extract_entity_obs(sim, pl, "blue")
            h_dq = blue_histories[pl_idx]

            if len(h_dq) == 0:
              for _ in range(frame_stack):
                h_dq.append(tokens)
            else:
              h_dq.append(tokens)

            stacked_ego = np.concatenate([f["ego"] for f in h_dq], axis=-1)
            stacked_ball = np.concatenate([f["ball"] for f in h_dq], axis=-1)
            stacked_mates = np.concatenate([f["teammates"] for f in h_dq], axis=-1)
            stacked_opps = np.concatenate([f["opponents"] for f in h_dq], axis=-1)

            actor_obs = {
                "ego": torch.as_tensor(stacked_ego, dtype=torch.float32, device=device).unsqueeze(0),
                "ball": torch.as_tensor(stacked_ball, dtype=torch.float32, device=device).unsqueeze(0),
                "teammates": torch.as_tensor(stacked_mates, dtype=torch.float32, device=device).unsqueeze(0),
                "opponents": torch.as_tensor(stacked_opps, dtype=torch.float32, device=device).unsqueeze(0),
                "key_padding_mask": torch.as_tensor(tokens["key_padding_mask"], dtype=torch.bool, device=device).unsqueeze(0),
            }
            with torch.inference_mode():
              act, _, _, _ = blue_model.get_action_and_value(actor_obs, deterministic=True)
            m_idx = int(act[0, 0].item())
            k_val = bool(act[0, 1].item())
            ex, ey = _EGO_DIRS[m_idx]
            blue_phs[pl_idx].action = (Vec2(-ex, ey), k_val)
        elif blue_model is not None:
          for pl_idx, pl in enumerate(sim.blue_team):
            blue_phs[pl_idx].action = blue_model.get_action(sim.all_players.index(pl), sim)
        else:
          for pl_idx in range(blue_team_size):
            dx, dy = random.choice(_EGO_DIRS)
            blue_phs[pl_idx].action = (Vec2(dx, dy), random.random() < 0.20)

      # ── 2. Telemetry Capture ──
      frame_players = []
      for idx, pl in enumerate(sim.all_players):
        frame_players.append({
            "name": roster[idx].stats.name,
            "team": pl.team,
            "x": round(pl.pos.x, 2),
            "y": round(pl.pos.y, 2),
            "vx": round(pl.vel.x, 2),
            "vy": round(pl.vel.y, 2),
            "r": round(pl.radius, 2),
            "is_kicking": bool(pl.is_kicking),
        })

      # ── 3. Step Physics ──
      goal_event = sim.step(dt)
      if hasattr(sim, "mode"):
        sim.mode.state = "PLAYING"
        if hasattr(sim.mode, "score_limit"):
          sim.mode.score_limit = 999

      frames.append({
          "step": step,
          "time": round(step * dt, 2),
          "score_red": sim.score_red,
          "score_blue": sim.score_blue,
          "ball": {
              "x": round(sim.ball.pos.x, 2),
              "y": round(sim.ball.pos.y, 2),
              "vx": round(sim.ball.vel.x, 2),
              "vy": round(sim.ball.vel.y, 2),
              "r": round(sim.ball.radius, 2),
          },
          "players": frame_players,
          "goal_event": goal_event,
      })

      if goal_event is not None:
        _apply_dynamic_restart(sim, ep_idx=ep, is_initial=False)
        for dq in red_histories:
          dq.clear()
        for dq in blue_histories:
          dq.clear()

    episodes_data.append({
        "episode_idx": ep + 1,
        "red_agent": red_label,
        "blue_agent": blue_label,
        "final_score": f"{sim.score_red} - {sim.score_blue}",
        "frames": frames,
    })

  p = sim.pitch
  pitch_data = {
      "left": p.left,
      "right": p.right,
      "top": p.top,
      "bottom": p.bottom,
      "goal_top": p.goal_top,
      "goal_bottom": p.goal_bottom,
  }

  html_content = _build_html_template(pitch_data, episodes_data)
  out_path = os.path.join(output_dir, filename)
  with open(out_path, "w", encoding="utf-8") as f:
    f.write(html_content)

  print(f"🎬 Replay generated successfully: {os.path.abspath(out_path)}")
  return out_path
