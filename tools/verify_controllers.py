from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from config.match_config import MatchConfig, PlayerSlot, PlayerStats
from src.engine.controllers import NumpyRLController
from src.engine.modes.classic_mode import ClassicMatchMode
from src.engine.simulation import Simulation
from src.rl.benchmarker import RLController
from src.rl.ppo_core import ActorCritic


def load_pytorch_controller(checkpoint_path: str, team: str) -> RLController:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = (
        ckpt["model_state_dict"]
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt
        else ckpt
    )
    model = ActorCritic(obs_dim=80, state_dim=80)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return RLController(model, team=team, device="cpu", deterministic=True)


def test_controller_parity(team: str, num_steps: int = 150):
    print(f"\n⚽ Testing Controller Parity for Team: {team.upper()} ({num_steps} steps)")

    pt_ckpt = str(ROOT_DIR / "training/models/stage2/best_model.pt")
    npz_path = str(ROOT_DIR / "assets/models/stage2_actor.npz")

    # Controller instances to compare
    pt_ctrl = load_pytorch_controller(pt_ckpt, team=team)
    np_ctrl = NumpyRLController(npz_path, team=team)

    # Active controllers for the simulation so sim.step() has valid actors
    sim_red_ctrl = pt_ctrl if team == "red" else load_pytorch_controller(pt_ckpt, team="red")
    sim_blue_ctrl = pt_ctrl if team == "blue" else load_pytorch_controller(pt_ckpt, team="blue")

    cfg = MatchConfig(
        mode=ClassicMatchMode(time_limit=180.0, score_limit=3),
        roster=[
            PlayerSlot(team="red", stats=PlayerStats(), controller=sim_red_ctrl),
            PlayerSlot(team="blue", stats=PlayerStats(), controller=sim_blue_ctrl),
        ],
    )
    sim = Simulation(center_x=600, center_y=400, match_config=cfg)

    tested_player_idx = 0 if team == "red" else 1

    for step in range(num_steps):
        # 1. Query PyTorch controller
        pt_move, pt_kick = pt_ctrl.get_action(tested_player_idx, sim)

        # 2. Query NumPy controller
        np_move, np_kick = np_ctrl.get_action(tested_player_idx, sim)

        # 3. Assert exact match in both direction and kick trigger
        assert np.isclose(pt_move.x, np_move.x, atol=1e-4), (
            f"Step {step}: move.x mismatch (PT={pt_move.x}, NP={np_move.x})"
        )
        assert np.isclose(pt_move.y, np_move.y, atol=1e-4), (
            f"Step {step}: move.y mismatch (PT={pt_move.y}, NP={np_move.y})"
        )
        assert pt_kick == np_kick, (
            f"Step {step}: kick mismatch (PT={pt_kick}, NP={np_kick})"
        )

        # Step physics forward to generate organic gameplay trajectory
        sim.step(dt=1.0 / 60.0)

    print(f"   ✓ All {num_steps} simulation steps match.")


def main():
    test_controller_parity("red", num_steps=150)
    test_controller_parity("blue", num_steps=150)
    print("\n✅ Verification passed: NumpyRLController matches PyTorch RLController.")


if __name__ == "__main__":
    main()