#!/usr/bin/env python
"""Evaluate rendered Reels (see aikyam_video/evaluation.py for every metric).

    python tools/eval_reel.py results/my_hook results/my_bar --tag hook_vs_bar

Each folder = one run of `aikyam-video process` (edit-plan.json, qc.json, reel-9x16.mp4). Prints a comparison table and writes results/eval_<tag>.json."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aikyam_video.evaluation import compare, evaluate


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("dirs", nargs="+"); ap.add_argument("--tag", default="eval"); a = ap.parse_args()
    rows = {os.path.basename(d.rstrip("/")): evaluate(d) for d in a.dirs}
    print(compare(rows)); out = os.path.join("results", f"eval_{a.tag}.json"); os.makedirs("results", exist_ok=True); json.dump(rows, open(out, "w"), indent=1, default=float); print("->", out); return 0


if __name__ == "__main__":
    sys.exit(main())
