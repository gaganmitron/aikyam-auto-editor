"""Story beats: what is HAPPENING in a shot, in the terms an editor cuts by (EXP-018). Labels say what is visible (idol, crowd, lamps); a beat says what the shot is FOR
in the story: an establishing view, people approaching, a rite being performed, a procession, prayer, darshan of the deity, a shared meal, music, a detail.

Zero-shot from the image-text model we already run, with action-phrased prompt ensembles per beat: no extra model, no per-frame cost beyond one text-embedding pass."""
from __future__ import annotations
from typing import Dict, List
import numpy as np

BEATS: Dict[str, List[str]] = {
    "establishing": ["a wide view of a temple building from outside, across a pool or at night", "an aerial or skyline view of a temple complex", "the exterior of a temple with its towers and courtyard"],
    "approach": ["pilgrims queueing and walking towards a temple entrance", "a large crowd of devotees gathering outside a temple", "people walking in a line along a railing to reach a shrine"],
    "ritual_action": ["a priest performing a ritual, waving a lit lamp or arranging offerings at a shrine", "attendants in white carrying out a ceremony inside a temple", "a priest waving a burning aarti lamp in front of a deity"],
    "offering": ["hands offering flowers, fruit or a thali of prasad to a deity", "close-up of a devotee placing an offering at an altar", "a priest applying tilak on a devotee's forehead"],
    "procession": ["men carrying a decorated palanquin or canopy in a religious procession", "a festival procession with a decorated chariot or float moving through a crowd", "devotees carrying a sacred object on their heads"],
    "prayer": ["a devotee praying with folded hands, eyes closed", "worshippers bowing and praying in a temple", "close-up faces of people in quiet prayer"],
    "darshan": ["a decorated deity idol inside a shrine seen close up", "the sacred altar or scripture in the inner sanctum of a temple", "a deity statue with garlands in a small home shrine"],
    "community": ["people sitting in rows sharing a free meal in a large hall", "volunteers serving food to devotees", "a community kitchen serving prasadam"],
    "music": ["devotional musicians playing harmonium and tabla and singing", "a group of people singing hymns with instruments", "kirtan singers performing in a temple"],
    "detail": ["a close-up of ornate gold decoration, carved stone or a chandelier", "a close-up of a page of written scripture", "a decorative detail such as lamps, flowers or beads in extreme close-up"],
}
KEYS = list(BEATS)


def text_embeddings(embed_text) -> np.ndarray:
    """(n_prompts, d) unit vectors and the owner index of each prompt; embed_text: list[str] -> (n, d) unit vectors."""
    flat = [(i, p) for i, k in enumerate(KEYS) for p in BEATS[k]]
    return embed_text([p for _, p in flat]), np.array([i for i, _ in flat])


def score(emb: np.ndarray, T: np.ndarray, owner: np.ndarray, scale: float, temperature: float = 1.0) -> Dict[str, float]:
    """Beat distribution for one image/shot embedding: per-beat best-phrasing similarity, softmaxed across beats (the beats compete: a shot is mostly ONE thing)."""
    sims = T @ np.asarray(emb, float)
    per = np.array([sims[owner == i].max() for i in range(len(KEYS))]) * scale / temperature
    e = np.exp(per - per.max()); p = e / e.sum()
    return {k: float(p[i]) for i, k in enumerate(KEYS)}
