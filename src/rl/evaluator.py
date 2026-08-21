import json
import os
import numpy as np
import torch
from src.rl.ppo_core import ActorCritic


def evaluate_and_generate_html(
    env,
    model_or_path: str | ActorCritic,
    device: torch.device,
    output_dir: str = "training/renders/stage2",
    filename: str = "stage2_eval.html",
    num_episodes: int = 5,
    max_steps: int = 600,
) -> str:
    """
    Evaluates policy rollouts and renders a full multi-agent match replay
    including Red team, Blue team, jersey numbers, and halo indicators.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)

    obs_dim = env.observation_space.shape[0]
    if isinstance(model_or_path, str):
        model = ActorCritic(obs_dim=obs_dim).to(device)
        ckpt = torch.load(model_or_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    else:
        model = model_or_path.to(device)
    model.eval()

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

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        frames = []
        ep_reward = 0.0
        scored = False
        conceded = False
        touched = False

        for step in range(max_steps):
            ball = env.sim.ball

            red_players = [
                {
                    "x": round(float(p.pos.x), 2),
                    "y": round(float(p.pos.y), 2),
                    "radius": float(p.radius),
                    "is_kicking": bool(p.is_kicking),
                    "num": i + 1,
                }
                for i, p in enumerate(env.sim.red_team)
            ]

            blue_players = [
                {
                    "x": round(float(p.pos.x), 2),
                    "y": round(float(p.pos.y), 2),
                    "radius": float(p.radius),
                    "is_kicking": bool(p.is_kicking),
                    "num": j + 1,
                }
                for j, p in enumerate(env.sim.blue_team)
            ]

            agent = env.sim.red_team[0]
            frames.append({
                "step": step + 1,
                "red_players": red_players,
                "blue_players": blue_players,
                "ball_x": round(float(ball.pos.x), 2),
                "ball_y": round(float(ball.pos.y), 2),
                "ball_radius": float(ball.radius),
                "dist_to_ball": round(float(agent.pos.distance_to(ball.pos)), 2),
                "cum_reward": round(float(ep_reward), 2),
                "score_red": env.sim.score_red,
                "score_blue": env.sim.score_blue,
            })

            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _ = model.get_action_and_value(obs_tensor, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
            ep_reward += reward
            scored = scored or info.get("is_goal", False)
            conceded = conceded or info.get("conceded", False)
            touched = touched or info.get("touched", False)

            if terminated or truncated:
                break

        episodes_data.append({
            "episode_id": ep + 1,
            "total_steps": len(frames),
            "total_reward": round(ep_reward, 2),
            "scored": scored,
            "conceded": conceded,
            "touched": touched,
            "frames": frames,
        })

    html_content = _build_html_template(pitch_data, episodes_data)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"🎬 Standalone multi-agent replay saved to: {out_path}")
    return out_path


def _build_html_template(pitch_data: dict, episodes_data: list) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Haxball Multi-Agent Replay</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: #14171f; color: #f0f2f5;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            display: flex; flex-direction: column; align-items: center; padding: 20px;
        }}
        h1 {{ font-size: 22px; margin-bottom: 4px; color: #fff; }}
        .subtitle {{ font-size: 13px; color: #64a5ff; margin-bottom: 16px; }}
        .tabs {{ display: flex; gap: 8px; margin-bottom: 14px; flex-wrap: wrap; }}
        .tab-btn {{
            background: #232733; color: #8c97ad; border: 1px solid #363d50;
            padding: 7px 12px; border-radius: 6px; cursor: pointer; font-size: 12px; font-weight: bold;
        }}
        .tab-btn.active {{ background: #2f6bf0; color: #fff; border-color: #2f6bf0; }}
        .badge {{ padding: 2px 6px; border-radius: 4px; font-size: 10px; margin-left: 6px; }}
        .badge.goal {{ background: #22a06b; color: #fff; }}
        .badge.conceded {{ background: #d9383a; color: #fff; }}
        .badge.draw {{ background: #64748b; color: #fff; }}
        .player-container {{
            background: #1a1d26; border: 1px solid #2b3040; border-radius: 10px;
            padding: 14px; display: flex; flex-direction: column; align-items: center;
        }}
        canvas {{ background: #285536; border-radius: 6px; box-shadow: 0 4px 12px rgba(0,0,0,0.4); }}
        .hud {{
            display: flex; justify-content: space-between; width: 100%; max-width: 900px;
            margin: 8px 0; font-size: 12px; color: #b4bfd4; font-family: monospace;
        }}
        .controls {{ display: flex; align-items: center; gap: 10px; margin-top: 8px; width: 100%; max-width: 900px; }}
        .btn {{
            background: #2d3345; color: #fff; border: none; padding: 6px 12px;
            border-radius: 5px; cursor: pointer; font-size: 12px;
        }}
        .btn:hover {{ background: #3d465e; }}
        .btn.active {{ background: #2f6bf0; }}
        input[type=range] {{ flex: 1; cursor: pointer; }}
    </style>
</head>
<body>
    <h1>Stage 2 Sparring Evaluation</h1>
    <div class="subtitle">Multi-Agent Neural Replay Viewer</div>

    <div class="tabs" id="tabs-container"></div>

    <div class="player-container">
        <div class="hud">
            <div>Step: <span id="hud-step" style="color:#fff;">0</span> / <span id="hud-total-steps">0</span></div>
            <div>Score: <span id="hud-score" style="color:#ffd043;">RED 0 - 0 BLUE</span></div>
            <div>Distance to Ball: <span id="hud-dist" style="color:#5eff8b;">0px</span></div>
            <div>Reward: <span id="hud-reward" style="color:#fff;">0.00</span></div>
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
                let tag = ep.scored ? "WON" : (ep.conceded ? "LOST" : "DRAW");
                let badgeClass = ep.scored ? "goal" : (ep.conceded ? "conceded" : "draw");
                btn.innerHTML = `Ep ${{ep.episode_id}} (${{ep.total_steps}}s) <span class="badge ${{badgeClass}}">${{tag}}</span>`;
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

            const pl = PITCH.left * scaleX, pr = PITCH.right * scaleX;
            const pt = PITCH.top * scaleY, pb = PITCH.bottom * scaleY;
            ctx.strokeRect(pl, pt, pr - pl, pb - pt);

            const cx = PITCH.center_x * scaleX;
            ctx.beginPath();
            ctx.moveTo(cx, pt); ctx.lineTo(cx, pb);
            ctx.stroke();

            ctx.beginPath();
            ctx.arc(cx, PITCH.center_y * scaleY, PITCH.center_radius * scaleX, 0, 2 * Math.PI);
            ctx.stroke();

            // Left Net (Red Goal) & Right Net (Blue Goal)
            const gt = PITCH.goal_top * scaleY, gb = PITCH.goal_bottom * scaleY;
            const gd = PITCH.goal_depth * scaleX;
            ctx.strokeStyle = "#ffe600";
            ctx.strokeRect(pl - gd, gt, gd, gb - gt);
            ctx.strokeRect(pr, gt, gd, gb - gb + (gb - gt));
        }}

        function drawFrame() {{
            const ep = EPISODES[currentEpIdx];
            if (!ep || !ep.frames[frameIdx]) return;
            const f = ep.frames[frameIdx];

            drawPitch();

            // Ball Ghost Trajectory
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

            // Draw Red Players (RL Agent + Red Bots)
            f.red_players.forEach((p, idx) => {{
                const px = p.x * scaleX, py = p.y * scaleY, r = p.radius * scaleX;
                
                // Controlled Agent Halo Indicator (Slot 0)
                if (idx === 0) {{
                    ctx.fillStyle = "rgba(255, 255, 255, 0.25)";
                    ctx.beginPath();
                    ctx.arc(px, py, r + 5, 0, 2 * Math.PI);
                    ctx.fill();
                    ctx.strokeStyle = "rgba(255, 255, 255, 0.6)";
                    ctx.lineWidth = 1.5;
                    ctx.stroke();
                }}

                ctx.fillStyle = "#e13737";
                ctx.beginPath();
                ctx.arc(px, py, r, 0, 2 * Math.PI);
                ctx.fill();
                ctx.strokeStyle = p.is_kicking ? "#ffffff" : "#1a1a1a";
                ctx.lineWidth = p.is_kicking ? 3 : 1.5;
                ctx.stroke();

                ctx.fillStyle = "#ffffff";
                ctx.font = "bold 11px sans-serif";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(p.num, px, py);
            }});

            // Draw Blue Players (Heuristic Opponents)
            f.blue_players.forEach((p) => {{
                const px = p.x * scaleX, py = p.y * scaleY, r = p.radius * scaleX;
                ctx.fillStyle = "#326eeb";
                ctx.beginPath();
                ctx.arc(px, py, r, 0, 2 * Math.PI);
                ctx.fill();
                ctx.strokeStyle = p.is_kicking ? "#ffffff" : "#1a1a1a";
                ctx.lineWidth = p.is_kicking ? 3 : 1.5;
                ctx.stroke();

                ctx.fillStyle = "#ffffff";
                ctx.font = "bold 11px sans-serif";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(p.num, px, py);
            }});

            // Update HUD
            document.getElementById("hud-step").innerText = f.step;
            document.getElementById("hud-total-steps").innerText = ep.total_steps;
            document.getElementById("hud-score").innerText = `RED ${{f.score_red}} - ${{f.score_blue}} BLUE`;
            document.getElementById("hud-dist").innerText = f.dist_to_ball.toFixed(1) + "px";
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