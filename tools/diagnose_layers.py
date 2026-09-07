# tools/diagnose_layers.py
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from src.rl.ppo_core import ActorCritic

# 1. Print NPZ Keys & Shapes
npz_path = ROOT_DIR / "assets/models/stage2_actor.npz"
data = np.load(npz_path)
print("=== NumPy Archive Keys & Shapes ===")
for k in sorted(data.files):
    print(f"  {k:35s}: {data[k].shape}")

# 2. Print PyTorch Architecture
ckpt = torch.load(
    ROOT_DIR / "training/models/stage2/best_model.pt",
    map_location="cpu",
    weights_only=False,
)
state_dict = (
    ckpt["model_state_dict"]
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt
    else ckpt
)
pt_model = ActorCritic(obs_dim=80, state_dim=80)
pt_model.load_state_dict(state_dict, strict=False)

print("\n=== PyTorch Actor Model Structure ===")
print("actor_encoder:")
print(pt_model.actor_encoder)
print("\nactor_move:", pt_model.actor_move)
print("actor_kick:", pt_model.actor_kick)