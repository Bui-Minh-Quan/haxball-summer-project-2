import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import onnxruntime as ort
import torch
from src.rl.numpy_actor import NumpyActor
from src.rl.ppo_core import ActorCritic

dummy_obs = np.random.randn(1, 80).astype(np.float32)

# 1. PyTorch
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
pt_model.eval()

with torch.no_grad():
    feat = pt_model.actor_encoder(torch.from_numpy(dummy_obs))
    pt_move = pt_model.actor_move(feat).numpy()
    pt_kick = pt_model.actor_kick(feat).numpy()

# 2. ONNX
session = ort.InferenceSession(str(ROOT_DIR / "assets/models/stage2_actor.onnx"))
onnx_move, onnx_kick = session.run(None, {"obs": dummy_obs})

# 3. Pure NumPy
numpy_actor = NumpyActor(str(ROOT_DIR / "assets/models/stage2_actor.npz"))
np_move, np_kick = numpy_actor.get_logits(dummy_obs)

# Validation
diff_pt_onnx_move = np.max(np.abs(pt_move - onnx_move))
diff_pt_np_move = np.max(np.abs(pt_move - np_move))
diff_pt_np_kick = np.max(np.abs(pt_kick - np_kick))

print(f"PyTorch vs ONNX Move Diff : {diff_pt_onnx_move:.6e}")
print(f"PyTorch vs NumPy Move Diff: {diff_pt_np_move:.6e}")
print(f"PyTorch vs NumPy Kick Diff: {diff_pt_np_kick:.6e}")

assert diff_pt_np_move < 1e-4 and diff_pt_np_kick < 1e-4, "NumPy output mismatch!"
print("✅ Verification passed: NumPy, ONNX, and PyTorch match.")