"""Learned clip ranker (EXP-016): a tiny linear model over per-scene features that says how reel-worthy a scene looks.

Why linear + a prior: the only training data is a few dozen rated frames (tools/bench_clips_ratings.json), so an unconstrained fit would memorise them.
The fit is ridge regression that shrinks toward PRIOR weights (what we believe before seeing data), and it is validated by holding out whole videos
(tools/fit_ranker.py). The artifact (data/ranker.json) is only used when present, and the `learned` scorer has default weight 0: opt-in.

Features come straight from the VisionResult the pipeline already stores, so nothing new runs at scoring time."""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence
import numpy as np

PATH = Path(__file__).parent / "data" / "ranker.json"
FEATURES = ["aesthetic", "no_text", "devotional", "sharpness", "exposure", "face"]
PRIOR = {"aesthetic": 0.5, "no_text": 0.25, "devotional": 0.5, "sharpness": 0.0, "exposure": 0.0, "face": 0.0}      # what we believed before the data (EXP-012)


def featurize(vr, devotional_weights: Optional[dict] = None) -> Optional[np.ndarray]:
    """Per-scene feature vector from a VisionResult, or None without an image-text model (aesthetic missing)."""
    if vr is None or vr.aesthetic is None or vr.no_text is None:
        return None
    if devotional_weights is None:
        from .scoring import DEVOTIONAL as devotional_weights
    dev = max([devotional_weights.get(k, 0.0) * v for k, v in vr.labels.items()] or [0.0])
    face = min(1.0, sum(f[2] * f[3] for f in vr.faces) * 8)
    return np.array([vr.aesthetic, vr.no_text, dev, vr.sharpness, 1.0 - abs(vr.brightness - 0.5) * 2, face], float)


@dataclass
class Ranker:
    mean: List[float]
    std: List[float]
    w: List[float]
    features: List[str] = field(default_factory=lambda: list(FEATURES))
    meta: dict = field(default_factory=dict)

    def raw(self, x: np.ndarray) -> float:
        return float(((np.asarray(x, float) - self.mean) / np.maximum(self.std, 1e-9)) @ np.asarray(self.w))

    def save(self, path: Optional[Path] = None) -> None:
        Path(path or PATH).write_text(json.dumps({"features": self.features, "mean": self.mean, "std": self.std, "w": self.w, "meta": self.meta}, indent=1))


def load(path: Optional[Path] = None) -> Optional[Ranker]:
    p = Path(path or PATH)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if d.get("features") != FEATURES:                                    # an artifact from another feature set must not be applied silently
        return None
    return Ranker(d["mean"], d["std"], d["w"], d["features"], d.get("meta", {}))


def fit(X: np.ndarray, y: np.ndarray, prior: Optional[Sequence[float]] = None, lam: float = 8.0) -> Ranker:
    """Ridge toward `prior` on standardised features: w = argmin ||Xw - y||^2 + lam ||w - w0||^2.  lam large = trust the prior, small = trust the data."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-9
    Z = (X - mean) / std; yc = (y - y.mean()) / (y.std() + 1e-9)
    w0 = np.array(prior if prior is not None else [PRIOR[f] for f in FEATURES], float)
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ yc + lam * w0)
    return Ranker(mean.tolist(), std.tolist(), w.tolist(), list(FEATURES), {"lam": lam, "n": int(len(y))})
