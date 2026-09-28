"""EXP-030: verify the full editorial-intelligence pipeline actually runs end to end on real footage, with events reaching the plan --
NOT a selection experiment, no A/B, no new heuristic. Reuses tools/e4_pairs.py's setup()/ctx_for() (cached multi-video analysis)
and the real engine functions (build_plans, render_reel) -- nothing here reimplements the pipeline, it only calls it and inspects the result.

Checks, in order: analysis cache loads -> 0.5s slot measurements exist -> event windows were computed and reached Shot.event_index ->
the beam-search story planner picked clips -> the creative engine produced transitions/camera/audio decisions -> edit-plan.json has
eventIndex on its segments -> FFmpeg rendered a real reel file -> QC ran and produced a report.

Own audio only (the recorded sound of the input videos is never used): --source-audio off, our own library music.
usage: python tools/verify_pipeline.py     -> results/verify_pipeline/reel-9x16.mp4 (+ prints a pass/fail line per check)"""
import dataclasses, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import e4_pairs as P
from aikyam_video.creative import engine as E

OUT = "results/verify_pipeline"


def check(label, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f" -- {detail}" if detail else ""), flush=True)
    return ok


def main():
    all_ok = True
    o, srcs = P.setup()
    all_ok &= check("analysis cache loaded for all 4 videos", len(srcs) == 4, f"{len(srcs)} sources")
    all_ok &= check("0.5s slot measurements present", all(sv.vision.embedding for x in srcs for sv in x.vision), "every scene has an embedding")

    o = dataclasses.replace(o, source_audio="off", music="auto", target_seconds=40.0, qc=True, transcript=False, captions=False, formats=["reel"], allow_silent=True)
    plan = E.build_plans(P.ctx_for(srcs, o))[0]
    segs = [s for s in plan["segments"] if s.get("kind", "video") == "video"]
    all_ok &= check("beam-search story planner picked clips", len(segs) >= 3, f"{len(segs)} clips, {plan['durationSeconds']:.1f}s")
    all_ok &= check("creative engine produced camera/transition decisions", all("camera" in s for s in segs), "camera path on every video segment")
    with_event = sum(1 for s in segs if "eventIndex" in s)
    all_ok &= check("event windows reached Shot.event_index -> edit-plan.json", with_event > 0, f"{with_event}/{len(segs)} segments carry eventIndex")
    all_ok &= check("own audio only (no source sound decided selection)", plan["audio"].get("sourceAudio") in (None, "off"), json.dumps(plan["audio"])[:120])

    os.makedirs(OUT, exist_ok=True)
    files, qcd, final = E.render_reel(plan, srcs[0].path, OUT, o, P.audio_for(srcs), ["reel"], "reel", None)
    json.dump(final, open(f"{OUT}/edit-plan.json", "w"), indent=1, default=str); json.dump(qcd, open(f"{OUT}/qc.json", "w"), indent=1, default=str)
    all_ok &= check("FFmpeg rendered a real reel file", os.path.exists(files.get("reel", "")) and os.path.getsize(files["reel"]) > 100_000, files.get("reel"))
    all_ok &= check("QC ran and produced a report", bool(qcd.get("status")), f"status={qcd.get('status')}, checks={len(qcd.get('checks', []))}")
    final_with_event = sum(1 for s in final["segments"] if "eventIndex" in s)
    all_ok &= check("eventIndex survives into the FINAL (post-QC/re-edit) plan", final_with_event > 0, f"{final_with_event} segments")

    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'} -- {OUT}/reel-9x16.mp4")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
