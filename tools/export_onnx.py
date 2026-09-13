import argparse
import os
from pathlib import Path
import sys
import time
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
  sys.path.insert(0, str(ROOT_DIR))

from src.rl.model import ActorCritic as Gen2ActorCritic
from src.rl.obs import ACTOR_OBS_DIM as GEN2_OBS_DIM
from src.rl_transformer.entity_obs import (
    BALL_DIM,
    EGO_DIM,
    MAX_OPPONENTS,
    MAX_TEAMMATES,
    PLAYER_DIM,
    TOTAL_TOKENS,
)
from src.rl_transformer.transformer_model import (
    TransformerActorCritic as Gen3ActorCritic,
)


class Gen2ActorWrapper(nn.Module):
  """Isolates the Gen 2 MLP Actor forward pass for ONNX tracing."""

  def __init__(self, model: Gen2ActorCritic):
    super().__init__()
    self.actor_encoder = model.actor_encoder
    self.actor_move = model.actor_move
    self.actor_kick = model.actor_kick

  def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    feat = self.actor_encoder(obs)
    logits_move = self.actor_move(feat)
    logits_kick = self.actor_kick(feat)
    return logits_move, logits_kick


class Gen3ActorWrapper(nn.Module):
  """Isolates the Gen 3 Entity-Transformer Actor forward pass for ONNX tracing."""

  def __init__(self, model: Gen3ActorCritic):
    super().__init__()
    self.actor = model.actor

  def forward(
      self,
      ego: torch.Tensor,
      ball: torch.Tensor,
      teammates: torch.Tensor,
      opponents: torch.Tensor,
      key_padding_mask: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    return self.actor(ego, ball, teammates, opponents, key_padding_mask)


def benchmark_onnx_latency(
    session: ort.InferenceSession,
    dummy_feed: dict[str, np.ndarray],
    warmup_runs: int = 200,
    test_runs: int = 10_000,
) -> dict[str, float]:
  """Measures latency with nanosecond precision."""
  for _ in range(warmup_runs):
    session.run(None, dummy_feed)

  latencies_ms = np.empty(test_runs, dtype=np.float64)

  for i in range(test_runs):
    t0 = time.perf_counter_ns()
    session.run(None, dummy_feed)
    t1 = time.perf_counter_ns()
    latencies_ms[i] = (t1 - t0) / 1_000_000.0

  return {
      "mean": float(np.mean(latencies_ms)),
      "p50": float(np.percentile(latencies_ms, 50)),
      "p95": float(np.percentile(latencies_ms, 95)),
      "p99": float(np.percentile(latencies_ms, 99)),
  }


def export_gen2_mlp(checkpoint_path: str, output_path: str):
  print(f"\n📦 [Gen 2 MLP] Loading: {checkpoint_path}")
  ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
  state_dict = (
      ckpt["model_state_dict"]
      if isinstance(ckpt, dict) and "model_state_dict" in ckpt
      else ckpt
  )

  model = Gen2ActorCritic()
  model.load_state_dict(state_dict, strict=False)
  model.eval()

  wrapper = Gen2ActorWrapper(model)
  wrapper.eval()

  dummy_input = torch.randn(1, GEN2_OBS_DIM, dtype=torch.float32)

  with torch.no_grad():
    pt_move, pt_kick = wrapper(dummy_input)

  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  torch.onnx.export(
      wrapper,
      dummy_input,
      output_path,
      export_params=True,
      opset_version=18,
      do_constant_folding=True,
      input_names=["obs"],
      output_names=["logits_move", "logits_kick"],
      dynamic_axes={
          "obs": {0: "batch_size"},
          "logits_move": {0: "batch_size"},
          "logits_kick": {0: "batch_size"},
      },
  )
  print(
      f"   ✓ Exported ONNX: {output_path} ({os.path.getsize(output_path) / 1024:.1f} KB)"
  )

  # Verification & Benchmarking
  session = ort.InferenceSession(
      output_path, providers=["CPUExecutionProvider"]
  )
  feed = {"obs": dummy_input.numpy()}
  onnx_move, onnx_kick = session.run(None, feed)

  diff_move = np.max(np.abs(pt_move.numpy() - onnx_move))
  diff_kick = np.max(np.abs(pt_kick.numpy() - onnx_kick))
  assert (
      diff_move < 1e-4 and diff_kick < 1e-4
  ), f"Parity mismatch! Move: {diff_move}, Kick: {diff_kick}"
  print(
      f"   ✓ Parity Verified (Max Diff: Move={diff_move:.2e}, Kick={diff_kick:.2e})"
  )

  metrics = benchmark_onnx_latency(session, feed)
  print(
      f"   ⚡ Latency (10k runs): Mean={metrics['mean']:.3f}ms | P50={metrics['p50']:.3f}ms | P95={metrics['p95']:.3f}ms | P99={metrics['p99']:.3f}ms"
  )


def export_gen3_transformer(checkpoint_path: str, output_path: str):
  print(f"\n📦 [Gen 3 Transformer] Loading: {checkpoint_path}")
  ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
  state_dict = (
      ckpt["model_state_dict"]
      if isinstance(ckpt, dict) and "model_state_dict" in ckpt
      else ckpt
  )

  model = Gen3ActorCritic()
  model.load_state_dict(state_dict, strict=False)
  model.eval()

  wrapper = Gen3ActorWrapper(model)
  wrapper.eval()

  dummy_feed = {
      "ego": torch.randn(1, EGO_DIM, dtype=torch.float32),
      "ball": torch.randn(1, BALL_DIM, dtype=torch.float32),
      "teammates": torch.randn(
          1, MAX_TEAMMATES, PLAYER_DIM, dtype=torch.float32
      ),
      "opponents": torch.randn(
          1, MAX_OPPONENTS, PLAYER_DIM, dtype=torch.float32
      ),
      "key_padding_mask": torch.zeros(1, TOTAL_TOKENS, dtype=torch.bool),
  }

  with torch.no_grad():
    pt_move, pt_kick = wrapper(**dummy_feed)

  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  torch.onnx.export(
      wrapper,
      tuple(dummy_feed.values()),
      output_path,
      export_params=True,
      opset_version=18,
      do_constant_folding=True,
      input_names=[
          "ego",
          "ball",
          "teammates",
          "opponents",
          "key_padding_mask",
      ],
      output_names=["logits_move", "logits_kick"],
      dynamic_axes={
          "ego": {0: "batch_size"},
          "ball": {0: "batch_size"},
          "teammates": {0: "batch_size"},
          "opponents": {0: "batch_size"},
          "key_padding_mask": {0: "batch_size"},
          "logits_move": {0: "batch_size"},
          "logits_kick": {0: "batch_size"},
      },
  )
  print(
      f"   ✓ Exported ONNX: {output_path} ({os.path.getsize(output_path) / 1024:.1f} KB)"
  )

  # Verification & Benchmarking
  session = ort.InferenceSession(
      output_path, providers=["CPUExecutionProvider"]
  )
  feed_np = {k: v.numpy() for k, v in dummy_feed.items()}
  onnx_move, onnx_kick = session.run(None, feed_np)

  diff_move = np.max(np.abs(pt_move.numpy() - onnx_move))
  diff_kick = np.max(np.abs(pt_kick.numpy() - onnx_kick))
  assert (
      diff_move < 1e-4 and diff_kick < 1e-4
  ), f"Parity mismatch! Move: {diff_move}, Kick: {diff_kick}"
  print(
      f"   ✓ Parity Verified (Max Diff: Move={diff_move:.2e}, Kick={diff_kick:.2e})"
  )

  metrics = benchmark_onnx_latency(session, feed_np)
  print(
      f"   ⚡ Latency (10k runs): Mean={metrics['mean']:.3f}ms | P50={metrics['p50']:.3f}ms | P95={metrics['p95']:.3f}ms | P99={metrics['p99']:.3f}ms"
  )


def main():
  parser = argparse.ArgumentParser(
      description="Export PyTorch RL Checkpoints to Optimized ONNX graphs."
  )
  parser.add_argument("--checkpoint", type=str, help="Path to .pt checkpoint")
  parser.add_argument("--output", type=str, help="Destination .onnx file path")
  parser.add_argument(
      "--arch",
      choices=["mlp", "transformer"],
      default="mlp",
      help="Model architecture",
  )
  args = parser.parse_args()

  if args.checkpoint and args.output:
    if args.arch == "mlp":
      export_gen2_mlp(args.checkpoint, args.output)
    else:
      export_gen3_transformer(args.checkpoint, args.output)
    return

  # Default batch export for tonight's game deployment
  default_deployments = [
      (
          "Notebooks/training_2/models/stage1/phase4/best_model.pt",
          "assets/models/stage1.onnx",
          "mlp",
      ),
      (
          "Notebooks/training_2/models/stage2/phase3/best_model.pt",
          "assets/models/stage2.onnx",
          "mlp",
      ),
      (
          "Notebooks/training_2/models/stage3/phase2/best_model.pt",
          "assets/models/stage3.onnx",
          "mlp",
      ),
  ]

  print("🚀 Exporting active deployment models to assets/models/...")
  for pt_path, onnx_path, arch in default_deployments:
    full_pt = os.path.join(ROOT_DIR, pt_path)
    full_onnx = os.path.join(ROOT_DIR, onnx_path)
    if os.path.exists(full_pt):
      if arch == "mlp":
        export_gen2_mlp(full_pt, full_onnx)
      else:
        export_gen3_transformer(full_pt, full_onnx)
    else:
      print(f"⚠️ Skipping missing checkpoint: {full_pt}")


if __name__ == "__main__":
  main()