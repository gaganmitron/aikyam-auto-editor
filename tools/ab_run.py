#!/usr/bin/env python
"""OLD vs NEW on the SAME inputs: one analysis, two plan+render runs (timed), then the evaluation table.

    python tools/ab_run.py --name ganga samples/ganga-aarti-haridwar.webm
    python tools/ab_run.py --name four v1.mp4 v2.mp4 v3.mp4 v4.mp4 --music-file song.mp3 --i-own-the-music-rights [--seed-assets results/my_hook/assets]

OLD = --no-edge-snap --no-motion-dedupe (the earlier fixed-grid windows / appearance-only duplicate rule). NEW = defaults. Writes results/ab_<name>_<variant>/ and results/ab_<name>.json."""
import argparse, json, os, shutil, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aikyam_video import evaluation, multi, music as _music, pipeline, stages
from aikyam_video.options import Options


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("inputs", nargs="+"); ap.add_argument("--name", required=True); ap.add_argument("--music-file"); ap.add_argument("--i-own-the-music-rights", action="store_true")
    ap.add_argument("--seed-assets"); ap.add_argument("--format", default="reel"); ap.add_argument("--variants", default="new,old", help="new,old | new"); a = ap.parse_args()
    base = os.path.join("results", f"ab_{a.name}"); rows, times = {}, {}
    for variant, flags in [v for v in (("new", {}), ("old", {"edge_snap": False, "motion_dedupe": False})) if v[0] in a.variants.split(",")]:
        out = f"{base}_{variant}"; shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
        seed = a.seed_assets if variant == "new" else f"{base}_new/assets"
        if seed and os.path.isdir(seed): shutil.copytree(seed, os.path.join(out, "assets"))
        o = Options(formats=[a.format], allow_silent=True, **flags)
        if a.music_file: o.music_track = _music.register_user_track(a.music_file, out, a.i_own_the_music_rights)
        t0 = time.perf_counter(); srcs, _ = multi.ingest(a.inputs, out, o); t_an = time.perf_counter() - t0                    # analysis (cached for the 2nd variant)
        t1 = time.perf_counter(); plan = multi.plan(a.inputs, out, o); t_plan = time.perf_counter() - t1
        t2 = time.perf_counter(); files = stages.render(plan["source"]["path"], out, o); t_render = time.perf_counter() - t2
        m = evaluation.evaluate(out); m["plan_s"], m["render_s"], m["analysis_s"] = t_plan, t_render, t_an; rows[f"{a.name}_{variant}"] = m
        print(f"{variant}: analysis {t_an:.0f}s plan {t_plan:.0f}s render {t_render:.0f}s -> {files.get('reel')}", flush=True)
    print(evaluation.compare(rows)); json.dump(rows, open(f"{base}.json", "w"), indent=1, default=float); print("->", f"{base}.json"); return 0


if __name__ == "__main__":
    sys.exit(main())
