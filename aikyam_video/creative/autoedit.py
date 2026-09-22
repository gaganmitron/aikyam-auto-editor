"""Self-editing loop: make several DIFFERENT edits of the same footage, render + QC + measure each, keep the best.

Variants change the decisions that matter most to how a reel reads (opening style, pacing, length), never randomly: the same inputs always give the same table.
Analysis is done once (variant 1) and copied to the others, so extra variants cost plan + render only. Every variant and its score is kept in <out>/variants.json.
The score is a documented weighted sum of measured things (below), not a learned taste model: it ranks edits, the person still judges."""
from __future__ import annotations
import copy, json, os, shutil
from typing import Callable, Dict, List, Optional, Tuple
from ..options import Options

# (name, Options overrides). Order = priority when N < len(VARIANTS).
VARIANTS: List[Tuple[str, dict]] = [("default", {}), ("establish", {"opening": "establish"}), ("hook", {"opening": "hook"}), ("contemplative", {"pacing": "contemplative"}), ("festive", {"pacing": "festive"})]
W = {"hook": 1.0, "beat": 0.5, "novelty": 0.3, "sliced": 0.2, "fail": 1.0, "warn": 0.25, "reedit": 0.5}
KEEP = ("reel-9x16.mp4", "square-1x1.mp4", "landscape-16x9.mp4", "edit-plan.json", "qc.json", "thumbnail.jpg", "mix.wav", "overlay_reel.ass", "overlay_square.ass", "overlay_landscape.ass")
_OUT = ("*.mp4", "edit-plan*.json", "qc.json", "mix.wav", "overlay*.ass", "thumbnail*", "reel*", "variants.json", "var_*")          # copied analysis must not carry another variant's outputs


def reel_score(m: dict, qc: dict) -> float:
    """Higher = better. hook + cuts on the beat + novelty (low redundancy) - sliced edges - QC failures/warnings - automatic re-edits. Missing metrics count as neutral."""
    checks = qc.get("checks", []); fails = sum(c["status"] == "fail" for c in checks); warns = sum(c["status"] == "warn" for c in checks)
    return round(W["hook"] * (m.get("hook_score") or 0.0) + W["beat"] * (m.get("cuts_on_beat_pct") or 0.0) + W["novelty"] * (1.0 - min(1.0, max(0.0, m.get("redundancy_max") or 0.0)))
                 - W["sliced"] * (m.get("edges_sliced_pct") or 0.0) - W["fail"] * fails - W["warn"] * warns - W["reedit"] * (m.get("auto_reedits") or 0), 4)


def best_of(run_fn: Callable[[Options, str], Dict[str, str]], out: str, o: Options, n: int, evaluate: Optional[Callable[[str], dict]] = None) -> Tuple[str, List[dict]]:
    """run_fn(options, dir) does the whole pipeline into dir. Returns (winning variant name, table). The winner's outputs are copied into `out`."""
    from .. import evaluation
    evaluate = evaluate or evaluation.evaluate
    todo = [v for v in VARIANTS if not any(getattr(o, k, None) for k in v[1])][:max(1, n)]                # a variant never overrides what the person chose explicitly
    rows: List[dict] = []; first_dir = None
    for name, over in todo:
        d = os.path.join(out, f"var_{name}"); shutil.rmtree(d, ignore_errors=True)
        if first_dir: shutil.copytree(first_dir, d, ignore=shutil.ignore_patterns(*_OUT))              # analysis is reused
        os.makedirs(d, exist_ok=True); vo = copy.copy(o)
        for k, v in over.items(): setattr(vo, k, v)
        try:
            run_fn(vo, d); m = evaluate(d); qc = json.load(open(os.path.join(d, "qc.json"))) if os.path.exists(os.path.join(d, "qc.json")) else {}
            rows.append({"variant": name, "dir": d, "score": reel_score(m, qc), "qc": qc.get("status"), "metrics": m})
        except Exception as e:                                                                          # noqa: BLE001 -- one bad variant must not lose the others
            rows.append({"variant": name, "dir": d, "score": float("-inf"), "error": str(e)[:300]})
        first_dir = first_dir or (d if rows[-1].get("error") is None else None)
    ok = [r for r in rows if "error" not in r]
    if not ok: raise RuntimeError(f"every variant failed: {[r.get('error') for r in rows]}")
    best = max(ok, key=lambda r: (r["score"], -rows.index(r)))                                        # ties: the earlier (default) variant
    for f in KEEP:
        if os.path.exists(os.path.join(best["dir"], f)): shutil.copy2(os.path.join(best["dir"], f), os.path.join(out, f))
    json.dump({"winner": best["variant"], "weights": W, "variants": [{k: v for k, v in r.items() if k != "metrics"} | {"metrics": r.get("metrics")} for r in rows]}, open(os.path.join(out, "variants.json"), "w"), indent=1, default=float)
    return best["variant"], rows
