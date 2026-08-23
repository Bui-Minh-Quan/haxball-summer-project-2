import json
import os
import random
import time
from pathlib import Path
from typing import Any
import torch

from config.match_config import MatchConfig, PlayerSlot, PlayerStats
from src.engine.controllers import Controller
from src.engine.modes.base_mode import GameMode
from src.engine.modes.classic_mode import ClassicMatchMode
from src.engine.modes.drill_mode import SoloDrillMode
from src.engine.simulation import Simulation
from src.engine.vector import Vec2
from src.rl.env_wrapper import extract_role_obs
from src.rl.obs_extractor import extract_universal_obs


class RLController(Controller):
    """Wraps an ActorCritic model to act as a standard engine controller.

    Automatically handles both 80-dim base models and 84-dim role models.
    """

    def __init__(self, model, team: str, device="cpu", role: str = "ST"):
        self.model = model
        self.team = team
        self.device = device
        self.role = role
        self.model.eval()

        # Automatically detect expected input feature size
        if hasattr(model, "shared") and len(model.shared) > 0 and hasattr(model.shared[0], "in_features"):
            self.in_features = model.shared[0].in_features
        elif hasattr(model, "actor_net") and len(model.actor_net) > 0 and hasattr(model.actor_net[0], "in_features"):
            self.in_features = model.actor_net[0].in_features
        else:
            self.in_features = 80

    def get_action(self, player_idx: int, sim: Simulation) -> tuple[Vec2, bool]:
        player = sim.all_players[player_idx]
        
        # Read slot role if available in match config
        role = self.role
        if hasattr(sim, "match_config") and sim.match_config and hasattr(sim.match_config, "roster"):
            if player_idx < len(sim.match_config.roster):
                role = getattr(sim.match_config.roster[player_idx], "role", self.role)

        # Extract 84-dim or 80-dim obs based on model architecture
        if self.in_features == 84:
            obs = extract_role_obs(sim, player, self.team, role)
        else:
            obs = extract_universal_obs(sim, player, self.team)

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action, _, _, _ = self.model.get_action_and_value(obs_tensor, deterministic=True)

        act = action.squeeze(0).cpu().numpy()
        action_map = {
            0: Vec2(0, 0), 1: Vec2(0, -1), 2: Vec2(0, 1),
            3: Vec2(-1, 0), 4: Vec2(1, 0), 5: Vec2(-1, -1),
            6: Vec2(1, -1), 7: Vec2(-1, 1), 8: Vec2(1, 1),
        }
        move_dir = action_map.get(int(act[0]), Vec2(0, 0))
        kick = bool(act[1])

        return move_dir, kick


def _randomize_solo_positions(sim: Simulation):
    """Randomizes ball and agent positions ensuring fair spacing."""
    p = sim.pitch
    agent = sim.red_team[0]

    # Ball spawns anywhere in midfield/attacking half, away from target net
    sim.ball.pos.x = random.uniform(p.left + 180.0, p.right - 240.0)
    sim.ball.pos.y = random.uniform(p.top + 80.0, p.bottom - 80.0)
    sim.ball.vel = Vec2(0.0, 0.0)

    # Agent spawns on pitch with at least 60px clearance from the ball
    while True:
        ax = random.uniform(p.left + 100.0, p.right - 180.0)
        ay = random.uniform(p.top + 80.0, p.bottom - 80.0)
        if math_hypot := Vec2(ax, ay).distance_to(sim.ball.pos) >= 60.0:
            agent.pos = Vec2(ax, ay)
            break

    agent.vel = Vec2(0.0, 0.0)
    agent.kick_cooldown_timer = 0.0


def run_solo_drill(
    agent_roster: list[PlayerSlot],
    num_episodes: int = 5,
    time_limit: float = 60.0,
):
    """Diagnostic Drill: Headless fast benchmark measuring genuine scoring rate."""
    print(f"🎯 Running Solo Drill: {time_limit}s per episode ({num_episodes} Episodes)")
    cfg = MatchConfig(
        mode=SoloDrillMode(time_limit=time_limit),
        roster=agent_roster,
    )

    total_goals = []
    for ep in range(num_episodes):
        sim = Simulation(match_config=cfg, center_x=cfg.pitch_width / 2, center_y=cfg.pitch_height / 2)
        _randomize_solo_positions(sim)
        ep_goals = 0

        max_steps = int(time_limit * 60)
        for _ in range(max_steps):
            evt = sim.step(1.0 / 60.0)
            if evt == "red_goal":
                ep_goals += 1
                _randomize_solo_positions(sim)

        total_goals.append(ep_goals)
        print(f"   Episode {ep + 1}: {ep_goals} goals")

    avg_goals = sum(total_goals) / max(1, num_episodes)
    print(f"📊 Average Scoring Rate: {avg_goals:.2f} goals / {time_limit}s\n")
    return avg_goals


def run_arena(
    red_roster: list[PlayerSlot],
    blue_roster: list[PlayerSlot],
    num_matches: int = 100,
    time_limit: float = 60.0,
    score_limit: int = 3,
):
    """Headless N vs M match benchmarker."""
    print(f"🏟️ Running Arena: {len(red_roster)} RED vs {len(blue_roster)} BLUE ({num_matches} Matches)")
    cfg = MatchConfig(
        mode=ClassicMatchMode(time_limit=time_limit, score_limit=score_limit),
        roster=red_roster + blue_roster,
        time_limit=time_limit,
        score_limit=score_limit,
    )

    stats = {"RED_WINS": 0, "BLUE_WINS": 0, "DRAWS": 0, "RED_GOALS": 0, "BLUE_GOALS": 0}
    start_time = time.time()

    for _ in range(num_matches):
        sim = Simulation(match_config=cfg, center_x=cfg.pitch_width / 2, center_y=cfg.pitch_height / 2)
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
    print(f"🏆 Series Outcome (Wins): RED {stats['RED_WINS']} | BLUE {stats['BLUE_WINS']} | DRAWS {stats['DRAWS']}")
    print(f"⚽ Avg Goals / Match:     RED {stats['RED_GOALS']/num_matches:.2f} | BLUE {stats['BLUE_GOALS']/num_matches:.2f}\n")
    return stats


def render_solo_drill(
    agent_slot: PlayerSlot,
    num_episodes: int = 3,
    time_limit: float = 30.0,
    save_path: str = "training/renders/solo_drills",
) -> str:
    """
    Renders an empty-net solo scoring evaluation to a standalone HTML replay.
    """
    print(f"🎬 Generating {num_episodes} Solo Drill Replays ({time_limit}s each)...")
    Path(save_path).mkdir(parents=True, exist_ok=True)

    cfg = MatchConfig(
        mode=SoloDrillMode(time_limit=time_limit),
        roster=[agent_slot],
    )

    episodes_data = []
    total_goals_scored = []

    for ep in range(num_episodes):
        sim = Simulation(match_config=cfg, center_x=cfg.pitch_width / 2, center_y=cfg.pitch_height / 2)
        _randomize_solo_positions(sim)
        sim.score_red = 0

        frames = []
        max_steps = int(time_limit * 60)

        for step in range(max_steps):
            agent = sim.red_team[0]
            ball = sim.ball

            frames.append({
                "step": step + 1,
                "score_red": sim.score_red,
                "score_blue": 0,
                "ball_x": round(float(ball.pos.x), 2),
                "ball_y": round(float(ball.pos.y), 2),
                "ball_radius": float(ball.radius),
                "red_players": [{
                    "x": round(float(agent.pos.x), 2),
                    "y": round(float(agent.pos.y), 2),
                    "radius": float(agent.radius),
                    "is_kicking": bool(agent.is_kicking),
                    "num": 1,
                }],
                "blue_players": [],
                "dist_to_ball": round(float(agent.pos.distance_to(ball.pos)), 2),
                "cum_reward": sim.score_red,
            })

            evt = sim.step(1.0 / 60.0)
            if evt == "red_goal":
                _randomize_solo_positions(sim)

        total_goals_scored.append(sim.score_red)
        print(f"   Episode {ep + 1} Finished: {sim.score_red} Goals Scored")

        episodes_data.append({
            "episode_id": ep + 1,
            "total_steps": len(frames),
            "scored": sim.score_red > 0,
            "conceded": False,
            "frames": frames,
        })

    pitch = sim.pitch
    pitch_data = {
        "width": float(pitch.width),
        "height": float(pitch.height),
        "left": float(pitch.left),
        "right": float(pitch.right),
        "top": float(pitch.top),
        "bottom": float(pitch.bottom),
        "center_x": float(sim.center.x),
        "center_y": float(sim.center.y),
        "center_radius": float(pitch.cfg.CENTER_CIRCLE_RADIUS),
        "goal_top": float(pitch.goal_top),
        "goal_bottom": float(pitch.goal_bottom),
        "goal_depth": 35.0,
    }

    from src.rl.evaluator import _build_html_template

    html_output = _build_html_template(pitch_data, episodes_data)

    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    file_path = Path(save_path) / f"{current_time}_solo_drill.html"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_output)

    avg_goals = sum(total_goals_scored) / max(1, num_episodes)
    print(f"🏆 Overall: {avg_goals:.2f} Avg Goals / {time_limit}s")
    print(f"Replay saved to: {file_path}\n")

    return str(file_path)


def render_match(
    red_roster: list[PlayerSlot],
    blue_roster: list[PlayerSlot],
    num_matches: int = 1,
    time_limit: float = 180.0,
    score_limit: int = 3,
    save_path: str = "training/renders/matches",
):
    """Visualizes an isolated match and outputs a Kaggle-style HTML file."""
    print(f"🎬 Generating {num_matches} replays...")
    Path(save_path).mkdir(parents=True, exist_ok=True)

    cfg = MatchConfig(
        mode=ClassicMatchMode(time_limit=time_limit, score_limit=score_limit),
        roster=red_roster + blue_roster,
        time_limit=time_limit,
        score_limit=score_limit,
    )

    for i in range(num_matches):
        sim = Simulation(match_config=cfg, center_x=cfg.pitch_width / 2, center_y=cfg.pitch_height / 2)
        frames = []
        step = 0

        while not sim.mode.is_game_over(sim):
            sim.step(1.0 / 60.0)
            step += 1

            frames.append({
                "step": step,
                "score_red": sim.score_red,
                "score_blue": sim.score_blue,
                "ball_x": round(float(sim.ball.pos.x), 2),
                "ball_y": round(float(sim.ball.pos.y), 2),
                "ball_radius": float(sim.ball.radius),
                "red_players": [{"x": round(float(p.pos.x), 2), "y": round(float(p.pos.y), 2), "radius": float(p.radius), "is_kicking": bool(p.is_kicking), "num": idx + 1} for idx, p in enumerate(sim.red_team)],
                "blue_players": [{"x": round(float(p.pos.x), 2), "y": round(float(p.pos.y), 2), "radius": float(p.radius), "is_kicking": bool(p.is_kicking), "num": idx + 1} for idx, p in enumerate(sim.blue_team)],
                "dist_to_ball": 0, "cum_reward": 0,
            })

        r_score, b_score = sim.score_red, sim.score_blue
        if r_score > b_score:
            print(f"Game {i+1} Result: RED WINS! 🎉 ({r_score} - {b_score})")
        elif b_score > r_score:
            print(f"Game {i+1} Result: BLUE WINS! 🎉 ({b_score} - {r_score})")
        else:
            print(f"Game {i+1} Result: DRAW. ({r_score} - {b_score})")

        pitch = sim.pitch
        pitch_data = {
            "width": float(pitch.width), "height": float(pitch.height), "left": float(pitch.left), "right": float(pitch.right),
            "top": float(pitch.top), "bottom": float(pitch.bottom), "center_x": float(sim.center.x), "center_y": float(sim.center.y),
            "center_radius": float(pitch.cfg.CENTER_CIRCLE_RADIUS), "goal_top": float(pitch.goal_top), "goal_bottom": float(pitch.goal_bottom), "goal_depth": 35.0,
        }
        episodes_data = [{"episode_id": 1, "total_steps": step, "scored": r_score > b_score, "conceded": b_score > r_score, "frames": frames}]

        from src.rl.evaluator import _build_html_template

        html_output = _build_html_template(pitch_data, episodes_data)
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        file_path = Path(save_path) / f"{current_time}_match_{i+1}.html"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_output)
        print(f"Replay saved to: {file_path}\n")