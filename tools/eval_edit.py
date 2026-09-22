"""Objective creative-quality report for rendered reels: usage  python tools/eval_edit.py results/<dir> [...]  (each dir = one run of `aikyam-video process`).
Reads edit-plan.json + reel-9x16.mp4. Prints one row per reel and writes results/eval_<tag>.json (tag from --tag)."""
import json, os, subprocess, sys
import cv2, numpy as np
from aikyam_video import measure as M


def edges(plan):
    """Edge times on the OUTPUT timeline (start of the overlap/cut between segment i and i+1) + the transition used."""
    segs, out, t = plan["segments"], [], 0.0
    tr = plan["transitions"]; ov = tr["durationSeconds"] if tr["type"] == "crossfade" else 0.0
    for i, s in enumerate(segs[:-1]):
        t += (s["end"] - s["start"]) - ov
        e = s.get("transitionOut") or {}
        out.append({"t": t + ov / 2, "type": (segs[i + 1].get("transitionIn") or {}).get("type", tr["type"]),
                    "d": (segs[i + 1].get("transitionIn") or {}).get("durationSeconds", ov)})
    return out


def sharp_expo(path, fps=2):
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-an", "-vf", f"fps={fps},scale=320:180,format=gray", "-f", "rawvideo", "-"], capture_output=True)
    fr = np.frombuffer(p.stdout, np.uint8).reshape(-1, 180, 320)
    sharp = [cv2.Laplacian(f, cv2.CV_64F).var() for f in fr]
    return fr, float(np.median(sharp)), float(np.mean(fr) / 255), float(np.mean(fr >= 250)), float(np.mean(fr <= 8))


def ahash(f):
    s = cv2.resize(f, (8, 8), interpolation=cv2.INTER_AREA); return (s > s.mean()).flatten()


def evaluate(d):
    plan = json.load(open(f"{d}/edit-plan.json")); reel = f"{d}/reel-9x16.mp4"
    x = M.decode_audio(reel); L = M.loudness(x); S = M.silence_stats(x)
    lens = [s["end"] - s["start"] for s in plan["segments"]]
    fr = M.frame_series(reel, fps=10); diffs = M.frame_diffs(fr); E = edges(plan)
    jumps = []
    for e in E:
        k = int(e["t"] * 10)
        z = M.robust_z(diffs[min(max(k - 1, 0), len(diffs) - 1)], diffs) if len(diffs) else 0
        jumps.append({"t": round(e["t"], 2), "type": e["type"], "picture_z": round(z, 1), "audio_jump_db": round(M.audio_jump_db(x, e["t"]), 1)})
    abrupt = [j for j in jumps if j["type"] == "cut" and abs(j["audio_jump_db"]) > 10]
    f2, sharp, mean_luma, clipped_hi, crushed_lo = sharp_expo(reel)
    hashes = [ahash(f) for f in f2[::max(1, len(f2) // max(1, len(lens) * 2))]]
    dup = sum(1 for i in range(len(hashes)) for j in range(i + 3, len(hashes)) if (hashes[i] != hashes[j]).sum() <= 6)
    m = plan["audio"].get("music", {})
    return {"reel": os.path.basename(d), "duration": round(plan["durationSeconds"], 1), "segments": len(lens), "shot_len_mean": round(float(np.mean(lens)), 1), "shot_len_min": round(min(lens), 1),
            "transitions": sorted({j["type"] for j in jumps}) or ["-"], "abrupt_cuts": len(abrupt), "edges": jumps,
            "lufs": round(L["lufs"], 1), "lra": round(L["lra"], 1), "true_peak_db": round(L["true_peak_db"], 1), "clipped": L["clipped"],
            "silence_ratio": round(S["ratio"], 2), "silence_longest_s": round(S["longest_s"], 1),
            "black_s": round(sum(b - a for a, b in M.black_intervals(reel)), 2), "freeze_s": round(sum(b - a for a, b in M.freeze_intervals(reel)), 2),
            "sharpness": round(sharp), "mean_luma": round(mean_luma, 2), "blown_highlights": round(clipped_hi, 3), "crushed_blacks": round(crushed_lo, 3),
            "near_duplicate_frames": dup, "music": m.get("trackId") if m.get("enabled") else None}


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]; tag = next((a.split("=")[1] for a in sys.argv if a.startswith("--tag=")), "run")
    rows = [evaluate(d) for d in args]
    json.dump(rows, open(f"results/eval_{tag}.json", "w"), indent=1)
    hdr = "reel dur segs shot(mean/min) transitions abrupt LUFS LRA TP clip sil(%,longest) black frz sharp luma dup music".split()
    print(f"{'reel':34s} {'dur':>5} {'seg':>3} {'shot':>9} {'trans':>10} {'abr':>3} {'LUFS':>6} {'LRA':>4} {'TP':>6} {'clip':>4} {'silence':>10} {'blk':>4} {'frz':>4} {'shp':>5} {'luma':>5} {'dup':>3}  music")
    for r in rows:
        print(f"{r['reel'][:34]:34s} {r['duration']:5.1f} {r['segments']:3d} {r['shot_len_mean']:4.1f}/{r['shot_len_min']:<4.1f} {','.join(r['transitions'])[:10]:>10} {r['abrupt_cuts']:3d} {r['lufs']:6.1f} {r['lra']:4.1f} {r['true_peak_db']:6.1f} {r['clipped']:4d} {r['silence_ratio']*100:4.0f}%,{r['silence_longest_s']:<4.1f} {r['black_s']:4.1f} {r['freeze_s']:4.1f} {r['sharpness']:5d} {r['mean_luma']:5.2f} {r['near_duplicate_frames']:3d}  {r['music'] or '-'}")
