import os
import time
import json
from pathlib import Path
import torch
from src.engine.simulation import Simulation
from src.engine.vector import Vec2
from src.engine.controllers import Controller
from config.match_config import MatchConfig, PlayerSlot, PlayerStats
from src.engine.modes.classic_mode import ClassicMatchMode
from src.rl.obs_extractor import extract_universal_obs


class RLController(Controller):
    """Wraps a PyTorch ActorCritic model to act as a standard engine controller."""
    def __init__(self, model, team: str, device="cpu"):
        self.model = model
        self.team = team
        self.device = device
        self.model.eval()

    def get_action(self, player_idx: int, sim: Simulation) -> tuple[Vec2, bool]:
        player = sim.all_players[player_idx]
        obs = extract_universal_obs(sim, player, self.team)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            action, _, _, _ = self.model.get_action_and_value(obs_tensor, deterministic=True)
            
        act = action.squeeze(0).cpu().numpy()
        move_dir = Vec2(float(act[0]), float(act[1]))
        
        # If action space is 2D (movement only), auto-kick is always True
        # If 3D, act[2] determines kick trigger
        kick = float(act[2]) > 0 if len(act) > 2 else True
        
        return move_dir, kick


def run_arena(
    red_roster: list[PlayerSlot],
    blue_roster: list[PlayerSlot],
    num_matches: int = 100,
    time_limit: float = 60.0,
    score_limit: int = 3,
):
    """
    Headless N vs M benchmarker. Runs simulation at maximum CPU speed.
    """
    print(f"🏟️ Running Arena: {len(red_roster)} RED vs {len(blue_roster)} BLUE ({num_matches} Matches)")
    
    cfg = MatchConfig(
        mode=ClassicMatchMode(time_limit=time_limit, score_limit=score_limit),
        roster=red_roster + blue_roster,
        time_limit=time_limit,
        score_limit=score_limit
    )

    stats = {"RED_WINS": 0, "BLUE_WINS": 0, "DRAWS": 0, "RED_GOALS": 0, "BLUE_GOALS": 0}
    start_time = time.time()

    for i in range(num_matches):
        sim = Simulation(match_config=cfg, center_x=cfg.pitch_width/2, center_y=cfg.pitch_height/2)
        
        # Run match until mode triggers game over
        while not sim.mode.is_game_over(sim):
            sim.step(1.0 / 60.0)
            
        r_score, b_score = sim.score_red, sim.score_blue
        stats["RED_GOALS"] += r_score
        stats["BLUE_GOALS"] += b_score
        
        if r_score > b_score:
            stats["RED_WINS"] += 1
        elif b_score > r_score:
            stats["BLUE_WINS"] += 1
        else:
            stats["DRAWS"] += 1

    elapsed = time.time() - start_time
    print(f"✅ Completed in {elapsed:.2f}s")
    print(f"🏆 Results: RED {stats['RED_WINS']} | BLUE {stats['BLUE_WINS']} | DRAWS {stats['DRAWS']}")
    print(f"⚽ Average Goals: RED {stats['RED_GOALS']/num_matches:.2f} | BLUE {stats['BLUE_GOALS']/num_matches:.2f}\n")
    
    return stats


def run_solo_drill(
    agent_roster: list[PlayerSlot],
    num_episodes: int = 10,
    time_limit: float = 60.0,
):
    """
    Diagnostic Drill: Empty net scenario. Instantly resets ball and player upon scoring.
    Measures raw mechanical scoring efficiency.
    """
    print(f"🎯 Running Solo Drill: {time_limit}s per episode ({num_episodes} Episodes)")
    
    cfg = MatchConfig(
        mode=ClassicMatchMode(time_limit=time_limit, score_limit=0), # 0 = infinite goals
        roster=agent_roster,
    )

    total_goals = []
    
    for ep in range(num_episodes):
        sim = Simulation(match_config=cfg, center_x=cfg.pitch_width/2, center_y=cfg.pitch_height/2)
        sim.score_red = 0
        prev_score = 0
        
        # 60 FPS * time_limit
        max_steps = int(time_limit * 60)
        
        for step in range(max_steps):
            sim.step(1.0 / 60.0)
            
            if sim.score_red > prev_score:
                prev_score = sim.score_red
                # Instant diagnostic reset (agent left of center, ball at center)
                sim.ball.pos = Vec2(sim.center.x, sim.center.y)
                sim.ball.vel = Vec2(0, 0)
                sim.red_team[0].pos = Vec2(sim.center.x - 150, sim.center.y)
                sim.red_team[0].vel = Vec2(0, 0)
                
        total_goals.append(sim.score_red)
        print(f"   Ep {ep+1}: {sim.score_red} goals")

    avg_goals = sum(total_goals) / num_episodes
    print(f"📊 Average Scoring Rate: {avg_goals:.2f} goals per {time_limit}s\n")
    return avg_goals


def render_match(
    red_roster: list[PlayerSlot],
    blue_roster: list[PlayerSlot],
    num_matches: int = 1,
    time_limit: float = 180.0,
    score_limit: int = 3,
    save_path: str = "training/renders/matches",
):
    """
    Visualizes an isolated match and outputs a Kaggle-style HTML file.
    """
    print(f"🎬 Generating {num_matches} replays...")
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    cfg = MatchConfig(
        mode=ClassicMatchMode(time_limit=time_limit, score_limit=score_limit),
        roster=red_roster + blue_roster,
        time_limit=time_limit,
        score_limit=score_limit
    )

    for i in range(num_matches):
        sim = Simulation(match_config=cfg, center_x=cfg.pitch_width/2, center_y=cfg.pitch_height/2)
        frames = []
        step = 0
        
        while not sim.mode.is_game_over(sim):
            sim.step(1.0 / 60.0)
            step += 1
            
            # Serialize state for HTML rendering
            frames.append({
                "step": step,
                "score_red": sim.score_red,
                "score_blue": sim.score_blue,
                "ball_x": round(float(sim.ball.pos.x), 2),
                "ball_y": round(float(sim.ball.pos.y), 2),
                "ball_radius": float(sim.ball.radius),
                "red_players": [{"x": round(float(p.pos.x), 2), "y": round(float(p.pos.y), 2), "radius": float(p.radius), "is_kicking": bool(p.is_kicking), "num": idx+1} for idx, p in enumerate(sim.red_team)],
                "blue_players": [{"x": round(float(p.pos.x), 2), "y": round(float(p.pos.y), 2), "radius": float(p.radius), "is_kicking": bool(p.is_kicking), "num": idx+1} for idx, p in enumerate(sim.blue_team)],
                "dist_to_ball": 0, "cum_reward": 0 # Ignored in pure match render
            })

        r_score, b_score = sim.score_red, sim.score_blue
        if r_score > b_score:
            print(f"Game {i+1} Result: RED WINS! 🎉 ({r_score} - {b_score})")
        elif b_score > r_score:
            print(f"Game {i+1} Result: BLUE WINS! 🎉 ({b_score} - {r_score})")
        else:
            print(f"Game {i+1} Result: DRAW. ({r_score} - {b_score})")

        print(f"Total Turns (Steps): {step}")
        
        # Build HTML Data
        pitch = sim.pitch
        pitch_data = {
            "width": float(pitch.width), "height": float(pitch.height), "left": float(pitch.left), "right": float(pitch.right),
            "top": float(pitch.top), "bottom": float(pitch.bottom), "center_x": float(sim.center.x), "center_y": float(sim.center.y),
            "center_radius": float(pitch.cfg.CENTER_CIRCLE_RADIUS), "goal_top": float(pitch.goal_top), "goal_bottom": float(pitch.goal_bottom), "goal_depth": 35.0,
        }
        episodes_data = [{"episode_id": 1, "total_steps": step, "scored": r_score > b_score, "conceded": b_score > r_score, "frames": frames}]
        
        # Uses the HTML builder function from evaluator (import it)
        from src.rl.evaluator import _build_html_template
        html_output = _build_html_template(pitch_data, episodes_data)

        current_time = time.strftime('%Y-%m-%d_%H-%M-%S')
        file_path = Path(save_path) / f"{current_time}_match_{i+1}.html"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_output)
            
        print(f"Replay saved to: {file_path}\n")