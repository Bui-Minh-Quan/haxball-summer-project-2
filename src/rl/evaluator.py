import os
import json
import torch
import numpy as np
from src.rl.ppo_core import ActorCritic


def evaluate_and_generate_html(
    env,
    model_or_path: str | ActorCritic,
    device: torch.device,
    output_dir: str = "training/renders/stage1",
    filename: str = "eval_replay.html",
    num_episodes: int = 5,
    max_steps: int = 300,
) -> str:
    """
    Runs evaluation episodes and saves an interactive standalone HTML replay.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)

    # 1. Load Model
    if isinstance(model_or_path, str):
        model = ActorCritic(obs_dim=68).to(device)
        ckpt = torch.load(model_or_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    else:
        model = model_or_path.to(device)
    model.eval()

    # 2. Extract Static Pitch Geometry
    pitch = env.sim.pitch
    pitch_data = {
        "width": float(pitch.width),
        "height": float(pitch.height),
        "left": float(pitch.left),
        "right": float(pitch.right),
        "top": float(pitch.top),
        "bottom": float(pitch.bottom),
        "center_x": float(env.sim.center.x),
        "center_y": float(env.sim.center.y),
        "center_radius": float(pitch.cfg.CENTER_CIRCLE_RADIUS),
        "goal_top": float(pitch.goal_top),
        "goal_bottom": float(pitch.goal_bottom),
        "goal_depth": 35.0,
    }

    episodes_data = []

    # 3. Collect Rollouts
    for ep in range(num_episodes):
        obs, _ = env.reset()
        frames = []
        ep_reward = 0.0
        scored = False
        touched = False

        for step in range(max_steps):
            agent = env.sim.red_team[0]
            ball = env.sim.ball

            # Record Frame State
            frames.append({
                "step": step + 1,
                "agent_x": round(float(agent.pos.x), 2),
                "agent_y": round(float(agent.pos.y), 2),
                "agent_vx": round(float(agent.vel.x), 2),
                "agent_vy": round(float(agent.vel.y), 2),
                "agent_radius": float(agent.radius),
                "ball_x": round(float(ball.pos.x), 2),
                "ball_y": round(float(ball.pos.y), 2),
                "ball_vx": round(float(ball.vel.x), 2),
                "ball_vy": round(float(ball.vel.y), 2),
                "ball_radius": float(ball.radius),
                "is_kicking": bool(agent.is_kicking),
                "cooldown": round(float(agent.kick_cooldown_timer), 3),
                "dist_to_ball": round(float(agent.pos.distance_to(ball.pos)), 2),
                "cum_reward": round(float(ep_reward), 2),
            })

            # Step Action
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _ = model.get_action_and_value(obs_tensor, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
            ep_reward += reward
            scored = scored or info.get("is_goal", False)
            touched = touched or info.get("touched", False)

            if terminated or truncated:
                break

        episodes_data.append({
            "episode_id": ep + 1,
            "total_steps": len(frames),
            "total_reward": round(ep_reward, 2),
            "scored": scored,
            "touched": touched,
            "frames": frames,
        })

    # 4. Generate Interactive HTML Page
    html_content = _build_html_template(pitch_data, episodes_data)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"🎬 Interactive replay saved to: {out_path}")
    return out_path


def _build_html_template(pitch_data: dict, episodes_data: list) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Haxball RL Evaluation Replay</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: #14171f; color: #f0f2f5;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            display: flex; flex-direction: column; align-items: center; padding: 24px;
        }}
        h1 {{ font-size: 24px; margin-bottom: 8px; color: #ffffff; }}
        .subtitle {{ font-size: 14px; color: #64a5ff; margin-bottom: 20px; }}
        .tabs {{ display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }}
        .tab-btn {{
            background: #232733; color: #8c97ad; border: 1px solid #363d50;
            padding: 8px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: bold;
        }}
        .tab-btn.active {{ background: #2f6bf0; color: #fff; border-color: #2f6bf0; }}
        .badge {{ padding: 2px 6px; border-radius: 4px; font-size: 11px; margin-left: 6px; }}
        .badge.goal {{ background: #22a06b; color: #fff; }}
        .badge.miss {{ background: #d93d4a; color: #fff; }}
        .player-container {{
            background: #1a1d26; border: 1px solid #2b3040; border-radius: 10px;
            padding: 16px; display: flex; flex-direction: column; align-items: center;
        }}
        canvas {{ background: #285536; border-radius: 6px; box-shadow: 0 4px 12px rgba(0,0,0,0.4); }}
        .hud {{
            display: flex; justify-content: space-between; width: 100%; max-width: 900px;
            margin: 10px 0; font-size: 13px; color: #b4bfd4; font-family: monospace;
        }}
        .controls {{ display: flex; align-items: center; gap: 12px; margin-top: 10px; width: 100%; max-width: 900px; }}
        .btn {{
            background: #2d3345; color: #fff; border: none; padding: 6px 12px;
            border-radius: 5px; cursor: pointer; font-size: 13px;
        }}
        .btn:hover {{ background: #3d465e; }}
        .btn.active {{ background: #2f6bf0; }}
        input[type=range] {{ flex: 1; cursor: pointer; }}
    </style>
</head>
<body>
    <h1>Stage 1 Striker Evaluation Replay</h1>
    <div class="subtitle">Autonomous 60 FPS Rollout Viewer</div>

    <div class="tabs" id="tabs-container"></div>

    <div class="player-container">
        <div class="hud">
            <div>Step: <span id="hud-step" style="color:#fff;">0</span> / <span id="hud-total-steps">0</span></div>
            <div>Distance to Ball: <span id="hud-dist" style="color:#ffd043;">0px</span></div>
            <div>Kicking: <span id="hud-kick" style="color:#ff5e5e;">FALSE</span></div>
            <div>Reward: <span id="hud-reward" style="color:#5eff8b;">0.00</span></div>
        </div>

        <canvas id="pitchCanvas" width="900" height="600"></canvas>

        <div class="controls">
            <button class="btn" id="playBtn" onclick="togglePlay()">❚❚ Pause</button>
            <input type="range" id="scrubber" min="0" max="100" value="0" oninput="onScrub(this.value)">
            <button class="btn speed-btn" onclick="setSpeed(0.5, this)">0.5x</button>
            <button class="btn speed-btn active" onclick="setSpeed(1.0, this)">1.0x</button>
            <button class="btn speed-btn" onclick="setSpeed(2.0, this)">2.0x</button>
        </div>
    </div>

    <script>
        const PITCH = {json.dumps(pitch_data)};
        const EPISODES = {json.dumps(episodes_data)};

        let currentEpIdx = 0;
        let frameIdx = 0;
        let isPlaying = true;
        let playSpeed = 1.0;
        let lastTime = 0;
        const FPS = 60;
        const frameDuration = 1000 / FPS;

        const canvas = document.getElementById("pitchCanvas");
        const ctx = canvas.getContext("2d");
        const scaleX = canvas.width / PITCH.width;
        const scaleY = canvas.height / PITCH.height;

        function initTabs() {{
            const container = document.getElementById("tabs-container");
            container.innerHTML = "";
            EPISODES.forEach((ep, i) => {{
                const btn = document.createElement("button");
                btn.className = "tab-btn" + (i === 0 ? " active" : "");
                btn.innerHTML = `Ep ${{ep.episode_id}} (${{ep.total_steps}}s)` + 
                    `<span class="badge ${{ep.scored ? 'goal' : 'miss'}}">${{ep.scored ? 'GOAL' : 'MISS'}}</span>`;
                btn.onclick = () => selectEpisode(i);
                container.appendChild(btn);
            }});
        }}

        function selectEpisode(idx) {{
            currentEpIdx = idx;
            frameIdx = 0;
            document.querySelectorAll(".tab-btn").forEach((b, i) => b.classList.toggle("active", i === idx));
            document.getElementById("scrubber").max = EPISODES[currentEpIdx].frames.length - 1;
            drawFrame();
        }}

        function togglePlay() {{
            isPlaying = !isPlaying;
            document.getElementById("playBtn").innerText = isPlaying ? "❚❚ Pause" : "▶ Play";
        }}

        function setSpeed(spd, el) {{
            playSpeed = spd;
            document.querySelectorAll(".speed-btn").forEach(b => b.classList.remove("active"));
            el.classList.add("active");
        }}

        function onScrub(val) {{
            frameIdx = parseInt(val);
            drawFrame();
        }}

        function drawPitch() {{
            ctx.fillStyle = "#285536";
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            ctx.strokeStyle = "#ffffff";
            ctx.lineWidth = 3;

            // Outer Boundary Lines
            const pl = PITCH.left * scaleX, pr = PITCH.right * scaleX;
            const pt = PITCH.top * scaleY, pb = PITCH.bottom * scaleY;
            ctx.strokeRect(pl, pt, pr - pl, pb - pt);

            // Midline
            const cx = PITCH.center_x * scaleX;
            ctx.beginPath();
            ctx.moveTo(cx, pt); ctx.lineTo(cx, pb);
            ctx.stroke();

            // Center Circle
            ctx.beginPath();
            ctx.arc(cx, PITCH.center_y * scaleY, PITCH.center_radius * scaleX, 0, 2 * Math.PI);
            ctx.stroke();

            // Right Target Goal Net
            const gt = PITCH.goal_top * scaleY, gb = PITCH.goal_bottom * scaleY;
            const gd = PITCH.goal_depth * scaleX;
            ctx.strokeStyle = "#ffe600";
            ctx.strokeRect(pr, gt, gd, gb - gt);
        }}

        function drawFrame() {{
            const ep = EPISODES[currentEpIdx];
            if (!ep || !ep.frames[frameIdx]) return;
            const f = ep.frames[frameIdx];

            drawPitch();

            // Draw Ball Trajectory Ghost Line
            ctx.beginPath();
            ctx.strokeStyle = "rgba(255, 255, 255, 0.25)";
            ctx.lineWidth = 2;
            for (let i = 0; i <= frameIdx; i++) {{
                const bf = ep.frames[i];
                if (i === 0) ctx.moveTo(bf.ball_x * scaleX, bf.ball_y * scaleY);
                else ctx.lineTo(bf.ball_x * scaleX, bf.ball_y * scaleY);
            }}
            ctx.stroke();

            // Draw Ball
            ctx.fillStyle = "#ffffff";
            ctx.beginPath();
            ctx.arc(f.ball_x * scaleX, f.ball_y * scaleY, f.ball_radius * scaleX, 0, 2 * Math.PI);
            ctx.fill();
            ctx.strokeStyle = "#111";
            ctx.lineWidth = 1;
            ctx.stroke();

            // Draw Agent (Red)
            ctx.fillStyle = f.is_kicking ? "#ffffff" : "#d9383a";
            ctx.beginPath();
            ctx.arc(f.agent_x * scaleX, f.agent_y * scaleY, f.agent_radius * scaleX, 0, 2 * Math.PI);
            ctx.fill();
            ctx.strokeStyle = "#ffffff";
            ctx.lineWidth = f.is_kicking ? 3 : 1.5;
            ctx.stroke();

            // Update HUD
            document.getElementById("hud-step").innerText = f.step;
            document.getElementById("hud-total-steps").innerText = ep.total_steps;
            document.getElementById("hud-dist").innerText = f.dist_to_ball.toFixed(1) + "px";
            document.getElementById("hud-kick").innerText = f.is_kicking ? "TRUE" : "FALSE";
            document.getElementById("hud-kick").style.color = f.is_kicking ? "#5eff8b" : "#8c97ad";
            document.getElementById("hud-reward").innerText = f.cum_reward.toFixed(2);
            document.getElementById("scrubber").value = frameIdx;
        }}

        function loop(timestamp) {{
            if (!lastTime) lastTime = timestamp;
            const delta = timestamp - lastTime;

            if (isPlaying && delta >= (frameDuration / playSpeed)) {{
                const ep = EPISODES[currentEpIdx];
                if (ep && frameIdx < ep.frames.length - 1) {{
                    frameIdx++;
                    drawFrame();
                }} else {{
                    isPlaying = false;
                    document.getElementById("playBtn").innerText = "▶ Play";
                }}
                lastTime = timestamp;
            }}
            requestAnimationFrame(loop);
        }}

        initTabs();
        selectEpisode(0);
        requestAnimationFrame(loop);
    </script>
</body>
</html>"""