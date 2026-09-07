import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxscript

# Add project root to path
#ROOT_DIR = Path(__file__).resolve().parent.parent
#sys.path.insert(0, str(ROOT_DIR))

from src.rl.ppo_core import ActorCritic

class ActorInferenceWrapper(nn.Module):
    """Encapsulates strictly the Actor components for clean ONNX graph generation."""

    def __init__(self, actor_critic: ActorCritic):
        super().__init__()
        self.actor_encoder = actor_critic.actor_encoder
        self.actor_move = actor_critic.actor_move
        self.actor_kick = actor_critic.actor_kick

    def forward(self, obs: torch.Tensor):
        feat = self.actor_encoder(obs)
        logits_move = self.actor_move(feat)
        logits_kick = self.actor_kick(feat)
        return logits_move, logits_kick

def export_model(checkpoint_path: str, stage_name: str, output_dir: str):
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ Checkpoint not found: {checkpoint_path}. Skipping.")
        return

    print(f"\n📦 Processing {stage_name} from: {checkpoint_path}")

    # 1. Load trained ActorCritic checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    model = ActorCritic(obs_dim=80, state_dim=80)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 2. Export Pure NumPy (.npz)
    actor_weights = {
        k: v.cpu().numpy()
        for k, v in state_dict.items()
        if not k.startswith("critic")
    }

    npz_path = os.path.join(output_dir, f"{stage_name}_actor.npz")
    np.savez_compressed(npz_path, **actor_weights)
    npz_size_kb = os.path.getsize(npz_path) / 1024
    print(f"   ✓ Saved NumPy Archive: {npz_path} ({npz_size_kb:.1f} KB)")

    # 3. Export ONNX (.onnx)
    actor_only = ActorInferenceWrapper(model)
    actor_only.eval()

    dummy_input = torch.randn(1, 80, dtype=torch.float32)
    onnx_path = os.path.join(output_dir, f"{stage_name}_actor.onnx")

    torch.onnx.export(
        actor_only,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,  # Matches native LayerNorm operator
        do_constant_folding=True,
        dynamo=False,  # Uses stable TorchScript tracer
        input_names=["obs"],
        output_names=["logits_move", "logits_kick"],
        dynamic_axes={
            "obs": {0: "batch_size"},
            "logits_move": {0: "batch_size"},
            "logits_kick": {0: "batch_size"},
        },
    )
    onnx_size_kb = os.path.getsize(onnx_path) / 1024
    print(f"   ✓ Saved ONNX Graph   : {onnx_path} ({onnx_size_kb:.1f} KB)")

def main():
    output_dir = os.path.join(ROOT_DIR, "assets", "models")
    os.makedirs(output_dir, exist_ok=True)

    models_to_export = [
        ("training/models/stage2/best_model.pt", "stage2"),
        ("training/models/stage3/best_model.pt", "stage3"),
    ]

    for rel_path, stage_name in models_to_export:
        full_path = os.path.join(ROOT_DIR, rel_path)
        export_model(full_path, stage_name, output_dir)

    print("\n✅ Export complete. Files generated in `assets/models/`.")


if __name__ == "__main__":
    main()