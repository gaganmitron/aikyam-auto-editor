"""Shot size: HOW MUCH of the world a shot shows (EXP-025): wide / medium / close / detail. Editors cut by it (wide -> medium -> close is the classic ordering cue), and the one
published ablation on shot ordering found shot category to matter far more than genre. Beats say what a shot is FOR; this says how it is framed.

Zero-shot from the image-text model we already run, exactly like `beats.py`: prompt ensembles per class, best phrasing per class, softmax across the classes.
Rated ground truth and the measurement: tools/bench_shotsize.py, tools/bench_shotsize_truth.json.

The prompts below were fixed BEFORE the first measurement (pre-registered in docs/research/experiment_matrix.md EXP-025); do not tune them on the benchmark frames."""
from __future__ import annotations
from typing import Dict, List
import numpy as np

SIZES: Dict[str, List[str]] = {
    "wide": ["a wide establishing shot of a temple or landscape where everything is small and far away",
             "an aerial or long shot showing a whole place from a distance",
             "a wide angle photo of a large scene with tiny distant people"],
    "medium": ["a medium shot of people or an altar at human scale",
               "a group of people seen from the waist up",
               "a mid-distance photo of a shrine or ritual with people in frame"],
    "close": ["a close-up shot of a face or a single subject filling the frame",
              "a close-up portrait of a deity idol or a person",
              "a tight close-up of one subject filling most of the frame"],
    "detail": ["an extreme close-up detail shot of a small object such as a flame, flowers, hands or texture",
               "a macro photo of a lamp, a garland or an ornament",
               "an extreme close-up of hands, fabric or written text"],
}
KEYS = list(SIZES)
ORDINAL = {"wide": 0.0, "medium": 1.0, "close": 2.0, "detail": 3.0}      # how tight the shot is; used by sequencing terms


def text_embeddings(embed_text) -> np.ndarray:
    """(n_prompts, d) unit vectors and the owner index (class) of each prompt; embed_text: list[str] -> (n, d) unit vectors."""
    flat = [(i, p) for i, k in enumerate(KEYS) for p in SIZES[k]]
    return embed_text([p for _, p in flat]), np.array([i for i, _ in flat])


def score(emb: np.ndarray, T: np.ndarray, owner: np.ndarray, scale: float, temperature: float = 1.0) -> Dict[str, float]:
    """Size distribution for one image/shot embedding: per-class best-phrasing similarity, softmaxed across the classes (a shot has ONE size)."""
    sims = T @ np.asarray(emb, float)
    per = np.array([sims[owner == i].max() for i in range(len(KEYS))]) * scale / temperature
    e = np.exp(per - per.max()); p = e / e.sum()
    return {k: float(p[i]) for i, k in enumerate(KEYS)}


def tightness(dist: Dict[str, float]) -> float:
    """Expected ordinal tightness of a distribution (0 = wide ... 3 = detail): one number for transition terms."""
    return float(sum(ORDINAL[k] * v for k, v in dist.items()))
