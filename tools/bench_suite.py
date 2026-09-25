"""Multi-video regression suite: the same videos, the same options, before and after every change, so a change is judged on several videos, not one.
  python tools/bench_suite.py run [names...]        run the pipeline on each suite video -> results/suite/<name>/  (one at a time: RAM), records seconds and peak RSS
  python tools/bench_suite.py score                 results/suite/scorecard.json = evaluation.evaluate() + story metrics per video
  python tools/bench_suite.py compare BASELINE.json diff the scorecard against a saved one and flag regressions (exit 1 if any)
  python tools/bench_suite.py save                  copy the scorecard to docs/research/suite_baseline.json
Options are fixed (--source-audio bed --no-transcript --no-captions): no downloads, no speech model, so a run is repeatable. Videos: five different sources (inputs/, not committed)."""
import json, os, resource, shutil, subprocess, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUITE = {"golden": "inputs/golden_temple_full.webm", "darshan": "inputs/tirumala_darshan_2021.mp4", "drone": "inputs/tirumala_drone_2025.mp4",
         "janakpur": "inputs/bench_ganga_aarti_janakpur.webm", "kolkata": "inputs/bench_temple_aarti_kolkata.webm"}
OPTS = ["--source-audio", "bed", "--no-transcript", "--no-captions"]
OUT = "results/suite"; BASE = "docs/research/suite_baseline.json"


def run(names):
    os.makedirs(OUT, exist_ok=True); tm = json.load(open(f"{OUT}/timing.json")) if os.path.exists(f"{OUT}/timing.json") else {}
    for n in names or SUITE:
        d = f"{OUT}/{n}"; shutil.rmtree(d, ignore_errors=True); before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss; t0 = time.time()
        p = subprocess.run([".venv/bin/aikyam-video", "process", SUITE[n], "-o", d, *OPTS], capture_output=True, text=True)
        tm[n] = {"seconds": round(time.time() - t0), "ok": p.returncode == 0, "peak_rss_mb_children": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024)}
        print(n, tm[n], (p.stderr[-300:] if p.returncode else ""), flush=True); json.dump(tm, open(f"{OUT}/timing.json", "w"), indent=1)


def story_metrics(d):
    """What the plan says about the story, plus the one footage check that needs the source: dead footage inside a chosen clip."""
    from aikyam_video.creative.shots import analyze_window
    plan = json.load(open(f"{d}/edit-plan.json")); segs = plan["segments"]; cr = plan.get("creative", {}); lo, hi = cr.get("targetRange", [0, 1e9])
    beats = [s.get("beat") for s in segs if s.get("beat")]; dead = tot = 0
    for s in segs:
        if s.get("kind", "video") != "video": continue
        sh = np.asarray(analyze_window(plan["source"]["path"], s["start"], s["end"], None).sharpness, float)
        if len(sh) >= 6: dead += int((sh < 0.30 * np.median(sh)).sum()); tot += len(sh)
    return {"in_target_range": bool(lo <= plan["durationSeconds"] <= hi), "target_range": [lo, hi], "distinct_beats": len(set(beats)), "roles": len({s.get("role") for s in segs}),
            "dead_slot_share": round(dead / tot, 3) if tot else 0.0}


def score():
    from aikyam_video.evaluation import evaluate
    card = {}
    for n in SUITE:
        d = f"{OUT}/{n}"
        if not os.path.exists(f"{d}/edit-plan.json"): continue
        card[n] = {**evaluate(d), **story_metrics(d)}
    tm = f"{OUT}/timing.json"
    for n, t in (json.load(open(tm)) if os.path.exists(tm) else {}).items():
        if n in card: card[n].update(seconds=t["seconds"])
    json.dump(card, open(f"{OUT}/scorecard.json", "w"), indent=1, default=float); show(card); return card


def show(card):
    cols = ["clips", "duration_s", "in_target_range", "distinct_beats", "roles", "dead_slot_share", "redundancy_max", "hook_luma", "qc_status", "seconds"]
    print(f"{'video':10s} " + " ".join(f"{c[:15]:>15s}" for c in cols))
    for n, r in card.items(): print(f"{n:10s} " + " ".join(f"{str(round(r.get(c), 2) if isinstance(r.get(c), float) else r.get(c)):>15s}" for c in cols)); print(f"{'':10s} qc: {r.get('qc_warn_or_fail')}")


def compare(base_path):
    """Regression = the run got worse on something a change must not make worse. Improvements and unrelated numbers are not flagged."""
    base, now = json.load(open(base_path)), json.load(open(f"{OUT}/scorecard.json")); bad = []
    for n, r in now.items():
        b = base.get(n)
        if not b: print(n, "(no baseline)"); continue
        if b.get("in_target_range") and not r.get("in_target_range"): bad.append(f"{n}: left the target duration range ({b['duration_s']:.1f}s -> {r['duration_s']:.1f}s, range {r['target_range']})")
        if r["dead_slot_share"] > b["dead_slot_share"] + 0.02: bad.append(f"{n}: more dead footage in clips ({b['dead_slot_share']} -> {r['dead_slot_share']})")
        if r["distinct_beats"] < b["distinct_beats"] - 1: bad.append(f"{n}: less story variety (beats {b['distinct_beats']} -> {r['distinct_beats']})")
        if r["clips"] < b["clips"] - 2: bad.append(f"{n}: far fewer clips ({b['clips']} -> {r['clips']})")
        if None not in (r.get("redundancy_max"), b.get("redundancy_max")) and r["redundancy_max"] > b["redundancy_max"] + 0.05: bad.append(f"{n}: clips look more alike (redundancy {b['redundancy_max']:.2f} -> {r['redundancy_max']:.2f})")
        if r.get("hook_dark") and not b.get("hook_dark"): bad.append(f"{n}: the reel now opens dark")
        new = sorted(set(r.get("qc_warn_or_fail", [])) - set(b.get("qc_warn_or_fail", [])))
        if new: bad.append(f"{n}: new QC findings {new}")
        if r.get("qc_status") == "fail" and b.get("qc_status") != "fail": bad.append(f"{n}: QC status is now fail")
    print("\nREGRESSIONS:" if bad else f"\nno regressions vs {base_path}", *bad, sep="\n  "); return 1 if bad else 0


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "run": run(sys.argv[2:])
    elif c == "score": score()
    elif c == "compare": sys.exit(compare(sys.argv[2]))
    elif c == "save": shutil.copy(f"{OUT}/scorecard.json", BASE); print("saved", BASE)
