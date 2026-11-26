"""uLipSync-style MFCC centroid/prototype baseline classifier."""

from __future__ import annotations

import numpy as np
from scipy.fft import dct


def log_mel_to_mfcc_50hz(log_mel_10ms: np.ndarray, n_mfcc: int = 16) -> np.ndarray:
    """Convert [T_10ms, 80] Log-Mel frames to 50 Hz [T_50hz, n_mfcc + 1] MFCC+energy features."""
    mel = np.asarray(log_mel_10ms, dtype=np.float32)
    t_even = (mel.shape[0] // 2) * 2
    if t_even == 0:
        return np.zeros((0, n_mfcc + 1), dtype=np.float32)

    # Average consecutive 10 ms frames into 20 ms (50 Hz) frames before DCT-II.
    mel_50 = mel[:t_even].reshape(t_even // 2, 2, mel.shape[1]).mean(axis=1)
    mfcc = dct(mel_50, type=2, axis=-1, norm="ortho")[:, :n_mfcc]
    energy = mel_50.mean(axis=-1, keepdims=True)
    return np.concatenate([mfcc, energy], axis=-1).astype(np.float32)


class MFCCBaseline:
    """uLipSync-style L2/cosine prototype centroid classifier on 50 Hz MFCC profiles."""

    def __init__(self, n_classes: int, n_mfcc: int = 16, temperature: float = 0.25) -> None:
        self.n_classes = n_classes
        self.n_mfcc = n_mfcc
        self.temperature = temperature
        self.centroids_: np.ndarray | None = None
        self.class_present_: np.ndarray | None = None

    @staticmethod
    def _normalize_rows(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / np.maximum(norms, 1e-8)

    def fit(self, features_50hz: np.ndarray, labels_50hz: np.ndarray) -> "MFCCBaseline":
        """Compute normalized per-class MFCC centroids from [N, D] features and [N] labels."""
        feats = np.asarray(features_50hz, dtype=np.float32)
        labels = np.asarray(labels_50hz, dtype=np.int64)
        normed = self._normalize_rows(feats)
        dim = normed.shape[1]

        centroids = np.zeros((self.n_classes, dim), dtype=np.float32)
        present = np.zeros(self.n_classes, dtype=bool)
        for c in range(self.n_classes):
            mask = labels == c
            if np.any(mask):
                mean_vec = normed[mask].mean(axis=0)
                norm = np.linalg.norm(mean_vec)
                if norm > 1e-8:
                    centroids[c] = mean_vec / norm
                    present[c] = True
        self.centroids_ = centroids
        self.class_present_ = present
        return self

    def predict_proba(self, features_50hz: np.ndarray) -> np.ndarray:
        """Return softmax probabilities [N, n_classes] from cosine similarity to centroids."""
        if self.centroids_ is None or self.class_present_ is None:
            raise RuntimeError("MFCCBaseline must be fit before calling predict_proba")
        feats = self._normalize_rows(np.asarray(features_50hz, dtype=np.float32))
        logits = (feats @ self.centroids_.T) / self.temperature
        logits[:, ~self.class_present_] = -1e9
        logits -= np.max(logits, axis=-1, keepdims=True)
        exp = np.exp(logits)
        return (exp / np.sum(exp, axis=-1, keepdims=True)).astype(np.float32)

    def predict(self, features_50hz: np.ndarray) -> np.ndarray:
        """Return hard class predictions [N]."""
        return np.argmax(self.predict_proba(features_50hz), axis=-1)
