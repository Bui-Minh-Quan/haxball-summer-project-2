import json
import os
import torch
import torch.nn as nn

from config.match_config import MatchConfig, PlayerSlot, PlayerStats
from src.bots.heuristic_bot import TeamHeuristicCoordinator
from src.engine.controllers import HeuristicBotController
from src.engine.modes.classic_mode import ClassicMatchMode
from src.engine.simulation import Simulation
from src.engine.vector import Vec2
from src.rl.benchmarker import RLController
from src.rl.env_wrapper import RandomController
from src.rl.ppo_core import ActorCritic
from src.rl.reset_strategies import RandomReset


def evaluate_and_generate_html(
    model_or_path: str | nn.Module,
    device: torch.device = torch.device("cpu"),
    baseline_type: str = "random",
    output_dir: str = "render/",
    filename: str = "match_replay.html",
    num_episodes: int = 5,
    max_steps: int = 1800,
    base_seed: int = 70000,
) -> str:
    """Runs evaluation matches and exports an interactive HTML5 canvas replay file."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)

    # 1. Load Model
    if isinstance(model_or_path, str):
        model = ActorCritic(obs_dim=80).to(device)
        model.load_state_dict(
            torch.load(model_or_path, map_location=device, weights_only=False)
        )
    else:
        model = model_or_path.to(device)
    model.eval()

    reset_strat = RandomReset()
    episodes_data = []
    pitch_data = None
    dt = 1.0 / 60.0

    # 2. Run Evaluation Episodes
    for ep_idx in range(num_episodes):
        is_red = ep_idx % 2 == 0
        learner_team = "red" if is_red else "blue"
        opp_team = "blue" if is_red else "red"
        seed = base_seed + ep_idx

        learner_ctrl = RLController(
            model, learner_team, device, deterministic=True
        )
        if baseline_type == "random":
            opp_ctrl = RandomController()
        elif baseline_type == "heuristic":
            opp_ctrl = HeuristicBotController(
                TeamHeuristicCoordinator(opp_team)
            )
        else:
            raise ValueError(f"Unknown baseline: {baseline_type}")

        if is_red:
            roster = [
                PlayerSlot(
                    "red",
                    PlayerStats("Learner_Red", accel=3200.0),
                    learner_ctrl,
                ),
                PlayerSlot(
                    "blue",
                    PlayerStats("Opponent_Blue", accel=3200.0),
                    opp_ctrl,
                ),
            ]
        else:
            roster = [
                PlayerSlot(
                    "red",
                    PlayerStats("Opponent_Red", accel=3200.0),
                    opp_ctrl,
                ),
                PlayerSlot(
                    "blue",
                    PlayerStats("Learner_Blue", accel=3200.0),
                    learner_ctrl,
                ),
            ]

        cfg = MatchConfig(
            mode=ClassicMatchMode(
                time_limit=max_steps * dt, score_limit=99
            ),
            roster=roster,
            time_limit=max_steps * dt,
            score_limit=99,
        )

        sim = Simulation(match_config=cfg)
        reset_strat.set_seed(seed)
        reset_strat.reset(sim)

        if pitch_data is None:
            pitch_data = {
                "width": sim.pitch.width,
                "height": sim.pitch.height,
                "left": sim.pitch.left,
                "right": sim.pitch.right,
                "top": sim.pitch.top,
                "bottom": sim.pitch.bottom,
                "goal_depth": getattr(sim.pitch, "goal_depth", 60.0),
                "goal_top": sim.pitch.center.y - 80.0,
                "goal_bottom": sim.pitch.center.y + 80.0,
            }

        frames = []
        score_red = 0
        score_blue = 0

        for step in range(max_steps):
            # Record current state
            players_state = []
            for p_idx, player in enumerate(sim.all_players):
                players_state.append(
                    {
                        "team": player.team,
                        "name": player.stats.name,
                        "x": round(float(player.pos.x), 2),
                        "y": round(float(player.pos.y), 2),
                        "vx": round(float(player.vel.x), 2),
                        "vy": round(float(player.vel.y), 2),
                        "r": float(player.radius),
                        "is_kicking": bool(player.is_kicking),
                    }
                )

            ball_state = {
                "x": round(float(sim.ball.pos.x), 2),
                "y": round(float(sim.ball.pos.y), 2),
                "vx": round(float(sim.ball.vel.x), 2),
                "vy": round(float(sim.ball.vel.y), 2),
                "r": float(sim.ball.radius),
            }

            goal_event = sim.step(dt)

            if goal_event == "red_goal":
                score_red += 1
            elif goal_event == "blue_goal":
                score_blue += 1

            frames.append(
                {
                    "step": step,
                    "time": round(step * dt, 2),
                    "ball": ball_state,
                    "players": players_state,
                    "score_red": score_red,
                    "score_blue": score_blue,
                    "goal_event": goal_event,
                }
            )

            if goal_event is not None:
                reset_strat.reset(sim)

        episodes_data.append(
            {
                "episode_idx": ep_idx + 1,
                "learner_team": learner_team,
                "opponent": baseline_type,
                "seed": seed,
                "final_score": f"{score_red} - {score_blue}",
                "frames": frames,
            }
        )

    # 3. Build HTML Output
    html_content = _build_html_template(pitch_data, episodes_data)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"🎬 Replay generated successfully: {os.path.abspath(out_path)}")
    return out_path


def _build_html_template(pitch_data: dict, episodes_data: list) -> str:
    episodes_json = json.dumps(episodes_data)
    pitch_json = json.dumps(pitch_data)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Haxball RL Replay Viewer</title>
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
    max-width: 1000px;
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
    <canvas id="pitchCanvas" width="900" height="500"></canvas>
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
      <span>Learner Team</span>
      <strong id="telLearner">-</strong>
    </div>
    <div class="telemetry-item">
      <span>Ball Velocity</span>
      <strong id="telBallVel">0.0 px/s</strong>
    </div>
    <div class="telemetry-item">
      <span>Learner Kick State</span>
      <strong id="telKickState">OFF</strong>
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
const telLearner = document.getElementById("telLearner");
const telBallVel = document.getElementById("telBallVel");
const telKickState = document.getElementById("telKickState");

// Setup Episode Options
episodes.forEach((ep, idx) => {{
  const opt = document.createElement("option");
  opt.value = idx;
  opt.textContent = `Ep ${{ep.episode_idx}} (Learner: ${{ep.learner_team.toUpperCase()}} | Result: ${{ep.final_score}})`;
  epSelect.appendChild(opt);
}});

epSelect.addEventListener("change", (e) => {{
  loadEpisode(parseInt(e.target.value));
}});

function loadEpisode(idx) {{
  currentEpIdx = idx;
  currentFrameIdx = 0;
  const ep = episodes[currentEpIdx];
  scrubber.max = ep.frames.length - 1;
  scrubber.value = 0;
  telLearner.textContent = ep.learner_team.toUpperCase();
  telLearner.style.color = ep.learner_team === "red" ? "#ef4444" : "#3b82f6";
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
  const frame = ep.frames[currentFrameIdx];
  if (!frame) return;

  // Update Controls & Telemetry
  timeDisplay.textContent = frame.time.toFixed(2) + "s";
  frameDisplay.textContent = `${{frame.step}} / ${{ep.frames.length - 1}}`;
  telScore.textContent = `${{frame.score_red}} - ${{frame.score_blue}}`;

  const ballSpeed = Math.hypot(frame.ball.vx, frame.ball.vy);
  telBallVel.textContent = ballSpeed.toFixed(1) + " px/s";

  const learnerPlayer = frame.players.find(p => p.team === ep.learner_team);
  if (learnerPlayer) {{
    telKickState.textContent = learnerPlayer.is_kicking ? "KICKING" : "READY";
    telKickState.style.color = learnerPlayer.is_kicking ? "#22c55e" : "#94a3b8";
  }}

  // Coordinate Scale
  const margin = 40;
  const scaleX = (canvas.width - margin * 2) / (pitch.right - pitch.left);
  const scaleY = (canvas.height - margin * 2) / (pitch.bottom - pitch.top);
  const scale = Math.min(scaleX, scaleY);

  const toScreenX = (x) => margin + (x - pitch.left) * scale;
  const toScreenY = (y) => margin + (y - pitch.top) * scale;

  // Clear Canvas
  ctx.fillStyle = "#15803d";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Pitch Border & Markings
  ctx.strokeStyle = "rgba(255, 255, 255, 0.7)";
  ctx.lineWidth = 3;

  const left = toScreenX(pitch.left);
  const right = toScreenX(pitch.right);
  const top = toScreenY(pitch.top);
  const bottom = toScreenY(pitch.bottom);
  const centerX = (left + right) / 2;
  const centerY = (top + bottom) / 2;

  // Boundary
  ctx.strokeRect(left, top, right - left, bottom - top);

  // Halfway Line
  ctx.beginPath();
  ctx.moveTo(centerX, top);
  ctx.lineTo(centerX, bottom);
  ctx.stroke();

  // Center Circle
  ctx.beginPath();
  ctx.arc(centerX, centerY, 70 * scale, 0, Math.PI * 2);
  ctx.stroke();

  // Goals
  const goalTop = toScreenY(pitch.goal_top);
  const goalBottom = toScreenY(pitch.goal_bottom);
  const goalWidth = 25 * scale;

  ctx.fillStyle = "rgba(255, 255, 255, 0.15)";
  // Left Goal (Red Defends)
  ctx.fillRect(left - goalWidth, goalTop, goalWidth, goalBottom - goalTop);
  ctx.strokeRect(left - goalWidth, goalTop, goalWidth, goalBottom - goalTop);

  // Right Goal (Blue Defends)
  ctx.fillRect(right, goalTop, goalWidth, goalBottom - goalTop);
  ctx.strokeRect(right, goalTop, goalWidth, goalBottom - goalTop);

  // Render Players
  frame.players.forEach(p => {{
    const px = toScreenX(p.x);
    const py = toScreenY(p.y);
    const pr = p.r * scale;

    // Kick Highlight Ring
    if (p.is_kicking) {{
      ctx.beginPath();
      ctx.arc(px, py, pr + 6, 0, Math.PI * 2);
      ctx.strokeStyle = "#fbbf24";
      ctx.lineWidth = 4;
      ctx.stroke();
    }}

    // Player Body
    ctx.beginPath();
    ctx.arc(px, py, pr, 0, Math.PI * 2);
    ctx.fillStyle = p.team === "red" ? "#dc2626" : "#2563eb";
    ctx.fill();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Direction Heading Line
    if (Math.hypot(p.vx, p.vy) > 10) {{
      const headingAngle = Math.atan2(p.vy, p.vx);
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(px + Math.cos(headingAngle) * pr * 1.5, py + Math.sin(headingAngle) * pr * 1.5);
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.stroke();
    }}
  }});

  // Render Ball
  const bx = toScreenX(frame.ball.x);
  const by = toScreenY(frame.ball.y);
  const br = frame.ball.r * scale;

  // Ball Shadow
  ctx.beginPath();
  ctx.arc(bx + 2, by + 3, br, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(0, 0, 0, 0.35)";
  ctx.fill();

  // Ball Body
  ctx.beginPath();
  ctx.arc(bx, by, br, 0, Math.PI * 2);
  ctx.fillStyle = "#ffffff";
  ctx.fill();
  ctx.strokeStyle = "#000000";
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Goal Flash Event
  if (frame.goal_event) {{
    ctx.fillStyle = "rgba(251, 191, 36, 0.4)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.font = "bold 36px sans-serif";
    ctx.fillStyle = "#ffffff";
    ctx.textAlign = "center";
    ctx.fillText("⚽ GOAL!", canvas.width / 2, 70);
  }}
}}

// Initialize
loadEpisode(0);
</script>
</body>
</html>
"""