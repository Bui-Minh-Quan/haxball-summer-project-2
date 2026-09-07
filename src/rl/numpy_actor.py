import math
import numpy as np


def _layer_norm(
    x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps)) * weight + bias


# Vectorized math.erf with explicit float32 output (pure Python standard library)
_erf_fn = np.vectorize(math.erf, otypes=[np.float32])


def _gelu(x: np.ndarray) -> np.ndarray:
    """Exact GELU activation matching PyTorch nn.GELU(approximate='none')."""
    return (0.5 * x * (1.0 + _erf_fn(x / math.sqrt(2.0)))).astype(np.float32)


class NumpyActor:
    """Lightweight inference-only Actor running on pure NumPy."""

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.weights = {k: data[k] for k in data.files}

    def _linear(self, x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
        return x @ weight.T + bias

    def _forward_encoder(self, x: np.ndarray) -> np.ndarray:
        w = self.weights

        # 0..2: Stem Linear -> LayerNorm -> GELU
        x = self._linear(x, w["actor_encoder.0.weight"], w["actor_encoder.0.bias"])
        x = _layer_norm(x, w["actor_encoder.1.weight"], w["actor_encoder.1.bias"])
        x = _gelu(x)

        # 3: ResidualBlock (block + residual addition + post GELU)
        if "actor_encoder.3.block.0.weight" in w:
            res = x
            x = self._linear(
                x,
                w["actor_encoder.3.block.0.weight"],
                w["actor_encoder.3.block.0.bias"],
            )
            x = _layer_norm(
                x,
                w["actor_encoder.3.block.1.weight"],
                w["actor_encoder.3.block.1.bias"],
            )
            x = _gelu(x)
            x = self._linear(
                x,
                w["actor_encoder.3.block.3.weight"],
                w["actor_encoder.3.block.3.bias"],
            )
            x = _layer_norm(
                x,
                w["actor_encoder.3.block.4.weight"],
                w["actor_encoder.3.block.4.bias"],
            )
            # Apply self.act(res + block_out)
            x = _gelu(res + x)

        # 4..6: Linear -> LayerNorm -> GELU
        x = self._linear(x, w["actor_encoder.4.weight"], w["actor_encoder.4.bias"])
        x = _layer_norm(x, w["actor_encoder.5.weight"], w["actor_encoder.5.bias"])
        x = _gelu(x)

        return x

    def forward(self, obs: np.ndarray) -> tuple[int, int]:
        """Runs single observation through Actor and returns discrete (move_idx, kick_idx)."""
        x = np.asarray(obs, dtype=np.float32)
        if x.ndim == 1:
            x = x[np.newaxis, :]

        feat = self._forward_encoder(x)
        logits_move = self._linear(
            feat, self.weights["actor_move.weight"], self.weights["actor_move.bias"]
        )
        logits_kick = self._linear(
            feat, self.weights["actor_kick.weight"], self.weights["actor_kick.bias"]
        )

        move_action = int(np.argmax(logits_move, axis=-1)[0])
        kick_action = int(np.argmax(logits_kick, axis=-1)[0])
        return move_action, kick_action

    def get_logits(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns raw logits for validation."""
        x = np.asarray(obs, dtype=np.float32)
        if x.ndim == 1:
            x = x[np.newaxis, :]

        feat = self._forward_encoder(x)
        logits_move = self._linear(
            feat, self.weights["actor_move.weight"], self.weights["actor_move.bias"]
        )
        logits_kick = self._linear(
            feat, self.weights["actor_kick.weight"], self.weights["actor_kick.bias"]
        )
        return logits_move, logits_kick