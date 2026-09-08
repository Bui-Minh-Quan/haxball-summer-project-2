# src/rl_2/pool.py
import glob
import os
import random
import torch
import torch.nn as nn

from config.match_config import MatchConfig, PlayerSlot, PlayerStats
from src.bots.heuristic_bot import TeamHeuristicCoordinator
from src.engine.controllers import Controller, HeuristicBotController
from src.engine.modes.classic_mode import ClassicMatchMode
from src.engine.simulation import Simulation
from src.engine.vector import Vec2
from src.rl_2.env_adapter import RandomOpponentController
from src.rl_2.model import ActorCritic
from src.rl_2.obs import extract_actor_obs, extract_global_state


class PoolOpponentController(Controller):
    """Dynamic opponent controller that samples across Random, Heuristic, and Self-Play models."""

    def __init__(
        self,
        pool_dir: str | None = None,
        team: str = "blue",
        device: str = "cpu",
        p_random: float = 0.10,
        p_heuristic: float = 0.20,
    ):
        self.pool_dir = pool_dir
        self.team = team
        self.sign = 1.0 if team == "red" else -1.0
        self.device = torch.device(device)

        self.p_random = p_random
        self.p_heuristic = p_heuristic

        self.random_ctrl = RandomOpponentController()
        self.heuristic_ctrl = HeuristicBotController(TeamHeuristicCoordinator(team=team))
        self.model = ActorCritic().to(self.device)
        self.model.eval()

        self.current_mode = "random"
        self._ego_dirs = [
            (0.0, 0.0),   (0.0, -1.0),  (0.0, 1.0),
            (-1.0, 0.0),  (1.0, 0.0),   (-1.0, -1.0),
            (1.0, -1.0),  (-1.0, 1.0),  (1.0, 1.0),
        ]

    def reset_opponent(self):
        """Called by MatchEnv.reset() to resample opponent persona for the round."""
        roll = random.random()

        # Mode 1: Random Baseline
        if roll < self.p_random or not self.pool_dir or not os.path.exists(self.pool_dir):
            self.current_mode = "random"
            return

        # Mode 2: Heuristic Bot Baseline
        if roll < (self.p_random + self.p_heuristic):
            self.current_mode = "heuristic"
            return

        # Mode 3: Self-Play Model from Pool
        history_files = glob.glob(os.path.join(self.pool_dir, "history_*.pt"))
        latest_file = os.path.join(self.pool_dir, "latest.pt")
        target_file = None

        if history_files and random.random() < 0.50:
            target_file = random.choice(history_files)
        elif os.path.exists(latest_file):
            target_file = latest_file

        if target_file:
            try:
                ckpt = torch.load(target_file, map_location=self.device, weights_only=False)
                state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
                actor_dict = {k: v for k, v in state_dict.items() if not k.startswith("critic")}
                self.model.load_state_dict(actor_dict, strict=False)
                self.current_mode = "model"
                return
            except Exception:
                pass

        self.current_mode = "heuristic"

    def get_action(self, player_idx: int, sim: Simulation) -> tuple[Vec2, bool]:
        if self.current_mode == "random":
            return self.random_ctrl.get_action(player_idx, sim)
        elif self.current_mode == "heuristic":
            return self.heuristic_ctrl.get_action(player_idx, sim)

        player = sim.all_players[player_idx]
        obs = extract_actor_obs(sim, player, self.team)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action, _, _, _ = self.model.get_action_and_value(obs_tensor, deterministic=True)

        m_idx = int(action[0, 0].item())
        kick = bool(action[0, 1].item())
        ego_x, ego_y = self._ego_dirs[m_idx]

        return Vec2(ego_x * self.sign, ego_y), kick


class SelfPlayPool:
    """Manages snapshots, champions, and gauntlet evaluations."""

    def __init__(self, pool_dir: str):
        self.pool_dir = pool_dir
        os.makedirs(self.pool_dir, exist_ok=True)
        self.champion_path = os.path.join(self.pool_dir, "champion.pt")

    def save_latest(self, model: nn.Module):
        torch.save(model.state_dict(), os.path.join(self.pool_dir, "latest.pt"))

    def register_champion(self, model: nn.Module, step: int):
        torch.save(model.state_dict(), self.champion_path)
        history_path = os.path.join(self.pool_dir, f"history_{step}.pt")
        torch.save(model.state_dict(), history_path)
        print(f"🏆 NEW CHAMPION REGISTERED @ step {step:,} -> {history_path}")

    def evaluate_matchup(
        self,
        learner_model: nn.Module,
        opponent_type: str,
        opponent_model: nn.Module | None = None,
        num_episodes: int = 20,
        team_size: int = 1,
        goal_height: float | None = None,
        max_steps: int = 900,
        device: torch.device = torch.device("cpu"),
    ) -> dict:
        """Simulates evaluation matches with mirrored sides (half Red, half Blue)."""
        learner_model.eval()
        if opponent_model:
            opponent_model.eval()

        wins = 0
        losses = 0
        draws = 0
        total_scored = 0
        total_conceded = 0

        _ego_dirs = [
            (0.0, 0.0),   (0.0, -1.0),  (0.0, 1.0),
            (-1.0, 0.0),  (1.0, 0.0),   (-1.0, -1.0),
            (1.0, -1.0),  (-1.0, 1.0),  (1.0, 1.0),
        ]

        class LocalPlaceholder(Controller):
            def __init__(self):
                self.action = (Vec2(0, 0), False)
            def get_action(self, idx, sim):
                return self.action

        for ep in range(num_episodes):
            learner_team = "red" if ep % 2 == 0 else "blue"
            opp_team = "blue" if learner_team == "red" else "red"
            sign = 1.0 if learner_team == "red" else -1.0

            learner_placeholders = [LocalPlaceholder() for _ in range(team_size)]
            opp_placeholders = [LocalPlaceholder() for _ in range(team_size)]

            roster = []
            for i in range(team_size):
                roster.append(PlayerSlot(learner_team, PlayerStats(f"L{i}", accel=3200.0), learner_placeholders[i]))
            for j in range(team_size):
                if opponent_type == "model":
                    ctrl = opp_placeholders[j]
                elif opponent_type == "heuristic":
                    ctrl = HeuristicBotController(TeamHeuristicCoordinator(team=opp_team))
                else:
                    ctrl = RandomOpponentController()
                roster.append(PlayerSlot(opp_team, PlayerStats(f"O{j}", accel=3200.0), ctrl))

            cfg = MatchConfig(
                mode=ClassicMatchMode(time_limit=max_steps / 60.0, score_limit=99),
                roster=roster,
                goal_height=goal_height,
            )
            sim = Simulation(match_config=cfg, goal_height=goal_height)

            for _ in range(max_steps):
                # 1. Learner actions
                l_squad = sim.red_team if learner_team == "red" else sim.blue_team
                for idx, player in enumerate(l_squad):
                    obs = extract_actor_obs(sim, player, learner_team)
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        act, _, _, _ = learner_model.get_action_and_value(obs_t, deterministic=True)
                    m_idx = int(act[0, 0].item())
                    ex, ey = _ego_dirs[m_idx]
                    learner_placeholders[idx].action = (Vec2(ex * sign, ey), bool(act[0, 1].item()))

                # 2. Model opponent actions
                if opponent_type == "model" and opponent_model:
                    o_squad = sim.blue_team if learner_team == "red" else sim.red_team
                    for idx, opp_player in enumerate(o_squad):
                        obs = extract_actor_obs(sim, opp_player, opp_team)
                        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                        with torch.no_grad():
                            act, _, _, _ = opponent_model.get_action_and_value(obs_t, deterministic=True)
                        m_idx = int(act[0, 0].item())
                        ex, ey = _ego_dirs[m_idx]
                        opp_placeholders[idx].action = (Vec2(ex * -sign, ey), bool(act[0, 1].item()))

                goal = sim.step(1.0 / 60.0)
                if goal is not None:
                    break

            # Outcome assessment
            scored = sim.score_red if learner_team == "red" else sim.score_blue
            conceded = sim.score_blue if learner_team == "red" else sim.score_red

            total_scored += scored
            total_conceded += conceded

            if scored > conceded:
                wins += 1
            elif conceded > scored:
                losses += 1
            else:
                draws += 1

        learner_model.train()
        return {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / max(1, num_episodes),
            "scored": total_scored,
            "conceded": total_conceded,
            "net": total_scored - total_conceded,
        }

    def run_gatekeeper_gauntlet(
        self,
        learner_model: nn.Module,
        active_tiers: list[str],  # e.g. ["random"], ["random", "heuristic"], or ["heuristic", "champion"]
        team_size: int = 1,
        goal_height: float | None = None,
        device: torch.device = torch.device("cpu"),
    ) -> tuple[bool, dict]:
        """Runs sequential gatekeeper trials. Must pass tier N to attempt tier N+1."""
        results = {}

        # Tier 1: Random Bot
        if "random" in active_tiers:
            r1 = self.evaluate_matchup(
                learner_model, opponent_type="random", team_size=team_size,
                goal_height=goal_height, device=device, num_episodes=20
            )
            results["random"] = r1
            if r1["win_rate"] < 0.85:
                return False, results

        # Tier 2: Heuristic Bot
        if "heuristic" in active_tiers:
            r2 = self.evaluate_matchup(
                learner_model, opponent_type="heuristic", team_size=team_size,
                goal_height=goal_height, device=device, num_episodes=20
            )
            results["heuristic"] = r2
            if r2["win_rate"] < 0.60 or r2["net"] < 5:
                return False, results

        # Tier 3: Current Champion Snapshot
        if "champion" in active_tiers and os.path.exists(self.champion_path):
            champ = ActorCritic().to(device)
            ckpt = torch.load(self.champion_path, map_location=device, weights_only=False)
            champ.load_state_dict(ckpt)
            champ.eval()

            r3 = self.evaluate_matchup(
                learner_model, opponent_type="model", opponent_model=champ,
                team_size=team_size, goal_height=goal_height, device=device, num_episodes=30
            )
            results["champion"] = r3
            # Must decisively beat previous best
            if r3["win_rate"] < 0.55 or r3["net"] < 3:
                return False, results

        return True, results