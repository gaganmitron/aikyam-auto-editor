"""ONE new reel from all the clips used across the E4 pair reels (user request 2026-09-26): the candidate pool = every shot that appears in any of the pair reels (terms ON or OFF); the auto editor
builds the opening / middle / ending and the order itself. The standard variants (default, establish, hook, contemplative, festive) are each planned, rendered and QC'd, `autoedit.best_of` keeps the one
with the best measured score (hook quality, cuts, redundancy, sliced edges, QC failures/warnings, re-edits).
Own audio only: the recorded sound of the videos is NOT used (`--source-audio off`); the soundtrack is our own library music.
usage: python tools/final_reel.py [--target 55]   -> results/final_reel/reel-9x16.mp4 (+ variants.json). The reel uses only what it needs from the pool: it is NOT every clip of the pairs."""
import copy, dataclasses, json, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import e4_pairs as P
from aikyam_video.creative import autoedit, engine as E
from aikyam_video.options import Options

OUT = "results/final_reel"


def pool_ids(n_pairs=20):
    """The shots of the pairs the E4 run renders (same selection rule as tools/e4_pairs.py): both versions of each pair."""
    screen = json.load(open("results/e4/screen.json")); per, chosen = {}, []
    for c in [c for c in screen if c["duration_ok"]]:
        k = tuple(c["assets"])
        if per.get(k, 0) < 3 and len(chosen) < n_pairs: chosen.append(c); per[k] = per.get(k, 0) + 1
    return {sid for c in chosen for seq in (c["base"], c["treat"]) for sid, _ in seq}, len(chosen)


def main():
    import argparse; ap = argparse.ArgumentParser(); ap.add_argument("--target", type=float, default=55.0, help="reel length in seconds (user asked for 50-60)"); a = ap.parse_args()
    allowed, n = pool_ids(); print(f"pool: {len(allowed)} distinct shots from {n} pairs", flush=True)
    o, srcs = P.setup(); memo = E.build_shots

    def only_pool(*a, **kw):                                                          # ids are prefixed with the asset id inside build_plans: filter on the same name
        return [k for k in memo(*a, **kw) if f"{kw.get('asset_id')}_{k.id}" in allowed]
    E.build_shots = only_pool
    o = dataclasses.replace(o, source_audio="off", music="auto", target_seconds=a.target, qc=True, transcript=False, captions=False, formats=["reel"], allow_silent=True)

    def run_fn(vo, d):
        plan = E.build_plans(P.ctx_for(srcs, vo))[0]
        files, qcd, final = E.render_reel(plan, srcs[0].path, d, vo, P.audio_for(srcs), ["reel"], "reel", None)
        json.dump(final, open(os.path.join(d, "edit-plan.json"), "w"), indent=1, default=str); json.dump(qcd, open(os.path.join(d, "qc.json"), "w"), indent=1, default=str)
        return files
    shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT)
    name, rows = autoedit.best_of(run_fn, OUT, o, 5)
    print("\nvariant scores:", flush=True)
    for r in rows: print(f"  {r['variant']:14s} score {r['score']}  QC {r.get('qc')}  {r.get('error', '')}", flush=True)
    print(f"WINNER: {name} -> {OUT}/reel-9x16.mp4", flush=True)


if __name__ == "__main__":
    main()
