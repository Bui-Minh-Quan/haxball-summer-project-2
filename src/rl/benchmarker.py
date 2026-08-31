import math
import random
import time
from pathlib import Path
import torch

from config.match_config import MatchConfig, PlayerSlot
from src.engine.controllers import Controller
from src.engine.modes.classic_mode import ClassicMatchMode
from src.engine.modes.drill_mode import SoloDrillMode
from src.engine.simulation import Simulation
from src.engine.vector import Vec2
from src.rl.obs_extractor import extract_obs

class RLController(Controller):
    def __init__(self, model, team: str, device="cpu", deterministic: bool = False):
        self.model = model
        self.team = team
        self.device = device
        self.deterministic = deterministic
        self.model.eval()

    def get_action(self, player_idx: int, sim: Simulation) -> tuple[Vec2, bool]:
        player = sim.all_players[player_idx]
        obs = extract_obs(sim, player, self.team)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action, _, _, _ = self.model.get_action_and_value(
                obs_tensor, deterministic=self.deterministic
            )

        act = action.squeeze(0).cpu().numpy()
        action_map = {
            0: Vec2(0, 0), 1: Vec2(0, -1), 2: Vec2(0, 1),
            3: Vec2(-1, 0), 4: Vec2(1, 0), 5: Vec2(-1, -1),
            6: Vec2(1, -1), 7: Vec2(-1, 1), 8: Vec2(1, 1),
        }
        return action_map.get(int(act[0]), Vec2(0, 0)), bool(act[1])
    
def _randomize_solo_positions(sim: Simulation):
    p = sim.pitch
    agent = sim.red_team[0]
    sim.ball.pos.x = random.uniform(p.left + 180.0, p.right - 240.0)
    sim.ball.pos.y = random.uniform(p.top + 80.0, p.bottom - 80.0)
    sim.ball.vel = Vec2(0.0, 0.0)

    while True:
        ax = random.uniform(p.left + 100.0, p.right - 180.0)
        ay = random.uniform(p.top + 80.0, p.bottom - 80.0)
        if Vec2(ax, ay).distance_to(sim.ball.pos) >= 60.0:
            agent.pos = Vec2(ax, ay)
            break
    agent.vel = Vec2(0.0, 0.0)
    agent.kick_cooldown_timer = 0.0

def run_solo_drill(agent_roster: list[PlayerSlot], num_episodes: int = 5, time_limit: float = 60.0):
    print(f"🎯 Running Solo Drill: {time_limit}s per episode ({num_episodes} Episodes)")
    cfg = MatchConfig(mode=SoloDrillMode(time_limit=time_limit), roster=agent_roster)
    total_goals = []
    
    for ep in range(num_episodes):
        sim = Simulation(match_config=cfg, center_x=cfg.pitch_width / 2, center_y=cfg.pitch_height / 2)
        _randomize_solo_positions(sim)
        ep_goals = 0
        for _ in range(int(time_limit * 60)):
            if sim.step(1.0 / 60.0) == "red_goal":
                ep_goals += 1
                _randomize_solo_positions(sim)
        total_goals.append(ep_goals)
        print(f"   Episode {ep + 1}: {ep_goals} goals")

    avg_goals = sum(total_goals) / max(1, num_episodes)
    print(f"📊 Average Scoring Rate: {avg_goals:.2f} goals / {time_limit}s\n")
    return avg_goals

def run_arena(red_roster, blue_roster, num_matches=100, time_limit=60.0, score_limit=3):
    print(f"🏟️ Running Arena: 1 RED vs 1 BLUE ({num_matches} Matches)")
    cfg = MatchConfig(
        mode=ClassicMatchMode(time_limit=time_limit, score_limit=score_limit),
        roster=red_roster + blue_roster, time_limit=time_limit, score_limit=score_limit
    )
    stats = {"RED_WINS": 0, "BLUE_WINS": 0, "DRAWS": 0, "RED_GOALS": 0, "BLUE_GOALS": 0}
    start_time = time.time()

    for _ in range(num_matches):
        sim = Simulation(match_config=cfg, center_x=cfg.pitch_width / 2, center_y=cfg.pitch_height / 2)
        while not sim.mode.is_game_over(sim):
            sim.step(1.0 / 60.0)
        stats["RED_GOALS"] += sim.score_red
        stats["BLUE_GOALS"] += sim.score_blue
        if sim.score_red > sim.score_blue: stats["RED_WINS"] += 1
        elif sim.score_blue > sim.score_red: stats["BLUE_WINS"] += 1
        else: stats["DRAWS"] += 1

    print(f"✅ Completed in {time.time() - start_time:.2f}s")
    print(f"🏆 Wins: RED {stats['RED_WINS']} | BLUE {stats['BLUE_WINS']} | DRAWS {stats['DRAWS']}")
    return stats