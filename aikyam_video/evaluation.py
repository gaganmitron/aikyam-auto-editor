"""Objective evaluation of rendered Reels: everything measured on the ENCODED file and the plan it came from, nothing self-reported by the editor.

  hook             how the first 3 s rank (sharpness, contrast, subject concentration) against the whole reel + first-clip length + not-dark
  shot lengths     on-screen seconds per clip (mean / median / min / max / std)
  redundancy       colour-histogram similarity between clip mid-frames (max, mean)
  transitions      share of edges that are blends (crossfade / dips) vs hard cuts
  cut-to-beat      distance of every cut (or blend centre) to the nearest beat / bar start of the music actually used
  loudness         integrated LUFS, true peak (recomputed from the file)
  music margin     dB the music sits under the live sound (needs qc.json from a run made after this metric existed)
  subject in frame plan-measured share of tracked samples inside the crop (crop-mode clips) + share of clips shown whole (fit_blur)
  cutoffs          faces sliced by the frame edge (QC bad_crop count). NOT measurable for deities: there is no deity localiser yet
  safe zones       caption / title / logo violations of the platform UI zones (libass measures the real ink)
  qc               final status, first-attempt status, number of automatic re-edits
"""
from __future__ import annotations
import glob, json, os, subprocess
from typing import Dict, List, Optional
import cv2
import numpy as np
from . import ffmpeg as ff, measure as M
from .plan import edge_overlaps, edge_transitions, timeline

HOOK_S = 3.0
BLENDS = {"crossfade", "dip_black", "dip_white"}


def _frames(path: str, fps: float, w: int = 320) -> np.ndarray:
    info = ff.probe(path); h = int(round(w * info.height / info.width / 2)) * 2
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-an", "-vf", f"fps={fps},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True)
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, h, w, 3)


def _rank(v: np.ndarray, x) -> float:
    return float((np.searchsorted(np.sort(v), x, side="left") + np.searchsorted(np.sort(v), x, side="right")) / (2.0 * len(v)))


def hook_quality(video: str, first_clip_s: float) -> Dict[str, float]:
    fr = _frames(video, 4.0); g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in fr]; n_hook = max(1, int(HOOK_S * 4))
    sharp = np.array([cv2.Laplacian(x, cv2.CV_64F).var() for x in g]); con = np.array([(np.percentile(x, 95) - np.percentile(x, 5)) / 255 for x in g])
    def conc(x):
        m = np.abs(cv2.Sobel(cv2.GaussianBlur(x, (5, 5), 0), cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(cv2.GaussianBlur(x, (5, 5), 0), cv2.CV_32F, 0, 1)); h, w = x.shape
        return float(m[int(h * .2):int(h * .8), int(w * .2):int(w * .8)].sum() / (m.sum() + 1e-9))
    cc = np.array([conc(x) for x in g]); luma = float(np.mean(g[:n_hook]) / 255)
    ranks = [float(np.mean([_rank(a, v) for v in a[:n_hook]])) for a in (sharp, con, cc)]
    return {"hook_score": float(np.mean(ranks)), "hook_sharpness_rank": ranks[0], "hook_contrast_rank": ranks[1], "hook_concentration_rank": ranks[2],
            "hook_luma": luma, "hook_dark": bool(luma < 0.12), "first_clip_s": float(first_clip_s), "hook_short": bool(first_clip_s <= 3.5 + 1e-6)}


def redundancy(video: str, tl: List[dict]) -> Dict[str, Optional[float]]:
    if len(tl) < 2:
        return {"redundancy_max": None, "redundancy_mean": None}
    h = []
    for c in tl:
        t = (c["start"] + c["end"]) / 2
        f = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", video, "-frames:v", "1", "-vf", "scale=160:-2", "-f", "image2pipe", "-vcodec", "bmp", "-"], capture_output=True).stdout
        img = cv2.imdecode(np.frombuffer(f, np.uint8), cv2.IMREAD_COLOR)
        hist = cv2.calcHist([cv2.cvtColor(img, cv2.COLOR_BGR2HSV)], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten(); h.append(hist / (hist.sum() + 1e-9))
    sims = [float(np.minimum(a, b).sum()) for i, a in enumerate(h) for b in h[i + 1:]]                      # histogram intersection: 1 = identical colour distribution
    return {"redundancy_max": max(sims), "redundancy_mean": float(np.mean(sims))}


def edge_stats(plan: dict, window: float = 1.5, tol: float = 0.12) -> Dict[str, Optional[float]]:
    """Clip edges vs the REAL cuts of their source (video clips only): on a cut (within `tol`), sliced (a real cut lies within `window` but the edge is not on it), or free (no cut nearby:
    continuous footage, nothing to snap to)."""
    from .creative.edges import detect_cuts
    paths = {a["id"]: a["path"] for a in plan.get("assets", [])} or {None: plan["source"]["path"]}; on = sliced = free = 0
    for s in plan["segments"]:
        if s.get("kind", "video") != "video":
            continue
        p = paths.get(s.get("assetId"));
        if not p or not os.path.exists(p):
            continue
        cuts = detect_cuts(p, max(0.0, s["start"] - window), s["end"] + window)
        for t in (s["start"], s["end"]):
            near = [c for c in cuts if abs(c - t) <= window]
            if not near: free += 1
            elif min(abs(c - t) for c in near) <= tol: on += 1
            else: sliced += 1
    n = on + sliced + free
    return {"edges_on_cut": on, "edges_sliced": sliced, "edges_free": free, "edges_sliced_pct": (sliced / (sliced + on)) if (sliced + on) else None}


def _music_grid(plan: dict, d: str):
    """(beats, downbeats) of the music actually used, relative to the reel's t=0; None when no rhythmic music."""
    m = next((x for x in plan.get("creative", {}).get("decisions", []) if x["type"] == "music"), None)
    if not m or not m.get("track") or m.get("offset") is None or not m.get("bpm"):
        return None
    tid, off = m["track"], float(m["offset"])
    cands = glob.glob(os.path.join(d, "user_music", f"{tid}*.beats.json")) + glob.glob(os.path.join(os.environ.get("MUSIC_LIBRARY_DIR", "music"), f"{tid}*.beats.json"))
    if not cands:
        return None
    b = json.load(open(cands[0])); return np.asarray(b["beats"], float) - off, np.asarray(b["downbeats"], float) - off


def evaluate(d: str, video: Optional[str] = None, grid=None) -> Dict:
    """Metrics of ONE run folder (edit-plan.json, qc.json, reel-9x16.mp4, overlay_reel.ass). `grid` = (beats, downbeats) overrides the music lookup (tests)."""
    plan = json.load(open(os.path.join(d, "edit-plan.json"))); video = video or os.path.join(d, "reel-9x16.mp4")
    qc = json.load(open(os.path.join(d, "qc.json"))) if os.path.exists(os.path.join(d, "qc.json")) else {"status": "n/a", "stats": {}, "checks": [], "attempts": []}
    segs, tr = plan["segments"], plan["transitions"]; tl = timeline(segs, tr); lens = np.array([c["end"] - c["start"] for c in tl]); edges = edge_transitions(segs, tr); ov = edge_overlaps(segs, tr)
    out: Dict = {"reel": os.path.abspath(video), "clips": len(segs), "duration_s": float(plan["durationSeconds"])}
    out.update({"shot_mean_s": float(lens.mean()), "shot_median_s": float(np.median(lens)), "shot_min_s": float(lens.min()), "shot_max_s": float(lens.max()), "shot_std_s": float(lens.std())})
    out["blend_pct"] = float(np.mean([e["type"] in BLENDS and e["durationSeconds"] > 0 for e in edges])) if edges else 0.0
    out["cut_pct"] = 1.0 - out["blend_pct"] if edges else 0.0
    out.update(hook_quality(video, float(lens[0]))); out.update(redundancy(video, tl)); out.update(edge_stats(plan))
    x = M.decode_audio(video); L = M.loudness(x) if len(x) else {}
    out["lufs"] = float(L.get("lufs", float("nan"))); out["true_peak_db"] = float(L.get("true_peak_db", float("nan")))
    grid = grid if grid is not None else _music_grid(plan, d)
    if grid is not None and len(edges):
        beats, downs = grid; cut_t = [tl[i]["end"] - ov[i] / 2 for i in range(len(edges))]
        err = np.array([float(np.min(np.abs(beats - t))) for t in cut_t]); derr = np.array([float(np.min(np.abs(downs - t))) for t in cut_t]) if len(downs) else None
        out.update({"cut_to_beat_mean_ms": float(err.mean() * 1000), "cut_to_beat_max_ms": float(err.max() * 1000), "cuts_on_beat_pct": float(np.mean(err <= 0.04)),
                    "cuts_on_bar_pct": float(np.mean(derr <= 0.04)) if derr is not None else None})
    else:
        out.update({"cut_to_beat_mean_ms": None, "cut_to_beat_max_ms": None, "cuts_on_beat_pct": None, "cuts_on_bar_pct": None})
    st = qc.get("stats", {}); out["music_margin_db_median"] = st.get("music_margin_db_median"); out["music_margin_db_p5"] = st.get("music_margin_db_p5")
    crop = [s["subjectInFrame"] for s in segs if "subjectInFrame" in s and s.get("kind", "video") == "video" and (s.get("layout") or {}).get("mode") != "fit_blur"]
    vids = [s for s in segs if s.get("kind", "video") == "video"]
    out["subject_in_frame"] = float(np.mean(crop)) if crop else None
    out["fit_blur_pct"] = float(np.mean([(s.get("layout") or {}).get("mode") == "fit_blur" for s in vids])) if vids else 0.0
    checks = {c["name"]: c for c in qc.get("checks", [])}
    out["face_cutoffs"] = int(checks["bad_crop"]["value"] or 0) if "bad_crop" in checks and checks["bad_crop"]["status"] != "pass" else 0
    out["deity_cutoffs"] = None                                                                             # no deity localiser: not measurable
    viol = 0
    for k in ("caption_overflow", "caption_overlay_collision", "caption_overlap", "overlay_safe_zone"):
        c = checks.get(k); viol += len(c["where"]) if c and c["status"] != "pass" else 0
    out["safe_zone_violations"] = viol
    att = qc.get("attempts", []); out["qc_status"] = qc.get("status"); out["qc_first_attempt"] = att[0]["status"] if att else None; out["auto_reedits"] = max(0, len(att) - 1)
    out["qc_warn_or_fail"] = [c["name"] for c in qc.get("checks", []) if c["status"] != "pass"]
    return out


def compare(rows: Dict[str, Dict]) -> str:
    """Plain-text table: one column per run."""
    keys = [("duration_s", "duration s", "{:.1f}"), ("clips", "clips", "{}"), ("shot_mean_s", "shot mean s", "{:.1f}"), ("shot_min_s", "shot min s", "{:.1f}"), ("shot_max_s", "shot max s", "{:.1f}"),
            ("hook_score", "hook score 0-1", "{:.2f}"), ("edges_on_cut", "edges on a cut", "{}"), ("edges_sliced", "edges sliced", "{}"), ("edges_free", "edges no cut near", "{}"), ("first_clip_s", "first clip s", "{:.1f}"), ("redundancy_max", "redundancy max", "{:.2f}"), ("blend_pct", "blend share", "{:.0%}"),
            ("cut_to_beat_mean_ms", "cut->beat ms", "{:.0f}"), ("cuts_on_beat_pct", "cuts on beat", "{:.0%}"), ("cuts_on_bar_pct", "cuts on bar", "{:.0%}"),
            ("lufs", "LUFS", "{:.1f}"), ("true_peak_db", "true peak dB", "{:.1f}"), ("music_margin_db_median", "music margin dB", "{:.1f}"),
            ("subject_in_frame", "subject in frame", "{:.0%}"), ("fit_blur_pct", "shown whole", "{:.0%}"), ("face_cutoffs", "face cutoffs", "{}"), ("safe_zone_violations", "safe-zone viol.", "{}"),
            ("qc_status", "QC", "{}"), ("auto_reedits", "re-edits", "{}"), ("plan_s", "plan time s", "{:.1f}"), ("render_s", "render time s", "{:.1f}")]
    names = list(rows); w = max(len(n) for n in names + [k[1] for k in keys]) + 1
    lines = [" " * 17 + "".join(n.ljust(14) for n in names)]
    for k, label, fmt in keys:
        cells = [("n/a" if rows[n].get(k) is None else fmt.format(rows[n][k])) for n in names]; lines.append(label.ljust(17) + "".join(c.ljust(14) for c in cells))
    return "\n".join(lines)
