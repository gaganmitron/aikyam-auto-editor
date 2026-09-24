"""Post-render Quality Control. Measures the ENCODED reel (decoded audio, sampled frames, libass-rendered captions) against the plan.

Limits are standards or adaptive statistics, not taste:
  loudness  target +-1.5 LU (BS.1770 / EBU R128 tolerance),  true peak <= -1.0 dBTP (common delivery ceiling),  no clipped samples
  black/frozen picture  studio black (luma<=16), freezedetect -60 dB;  planned dips/outro are excluded
  captions  ink inside the safe area (platform UI zones), <= 2 lines is a layout rule, reading speed <= 20 chars/s (broadcast norm 17-21)
  abrupt transitions  audio level jump across a hard cut > 9 dB or a waveform click; picture jump INSIDE a blend (robust z vs this video's own frame differences)
  music balance  measured from the stems: while the live sound is devotional, music must stay >= 12 dB under it
Each Check carries where it happened and which clip it concerns, so replan.py can fix exactly that.
"""
from __future__ import annotations
import os, re, subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from .. import ffmpeg as ff, measure as M
from ..safezone import PORTRAIT, glyphs, zones
from ..plan import edge_transitions, seg_source, timeline

TARGET_LUFS, LUFS_TOL = -16.0, 1.5
TP_LIMIT, MAX_CPS = -1.0, 20.0
MIN_MARGIN_DB = 12.0


@dataclass
class Check:
    name: str
    status: str                       # pass | warn | fail
    value: Optional[float] = None
    limit: Optional[float] = None
    where: List[float] = field(default_factory=list)      # times (s) on the output timeline
    clip: Optional[int] = None
    detail: str = ""


@dataclass
class QCReport:
    checks: List[Check]
    stats: Dict[str, float] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return "fail" if any(c.status == "fail" for c in self.checks) else "warn" if any(c.status == "warn" for c in self.checks) else "pass"

    def failed(self) -> List[Check]:
        return [c for c in self.checks if c.status == "fail"]

    def to_dict(self) -> dict:
        return {"status": self.status, "stats": self.stats, "checks": [asdict(c) for c in self.checks]}


def _clip_at(tl: List[dict], t: float) -> int:
    return next((i for i, c in enumerate(tl) if c["start"] - 1e-6 <= t < c["end"] + 1e-6), max(0, len(tl) - 1))


def _ok(name, **kw) -> Check:
    return Check(name, "pass", **kw)


# ---------------------------------------------------------------- captions measured by libass itself (correct Indic shaping)
def caption_ink(ass_path: str, W: int, H: int, dur: float, style_filter: Optional[str] = None, fps: int = 4) -> List[Tuple[float, Optional[Tuple[int, int, int, int]]]]:
    """Render the ASS onto black and return [(t, (x0, y0, x1, y1) of the text ink or None)] every 1/fps s. style_filter keeps only Dialogue lines with that Style."""
    src = Path(ass_path).read_text(encoding="utf-8")
    if style_filter is not None:
        src = "\n".join(l for l in src.splitlines() if not l.startswith("Dialogue:") or f",{style_filter}," in l)
    tmp = ass_path + f".{style_filter or 'all'}.ass"; Path(tmp).write_text(src, encoding="utf-8")
    esc = tmp.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    p = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"color=c=black:s={W}x{H}:r={fps}:d={dur:.3f}", "-vf", f"ass='{esc}',scale=iw/4:ih/4:flags=area,format=gray",
                        "-f", "rawvideo", "-"], capture_output=True)
    w, h = W // 4, H // 4; fr = np.frombuffer(p.stdout, np.uint8).reshape(-1, h, w); out = []
    for i, f in enumerate(fr):
        m = f > 90
        if m.sum() < 6:
            out.append((i / fps, None)); continue
        ys, xs = np.where(m); out.append((i / fps, (int(xs.min() * 4), int(ys.min() * 4), int(xs.max() * 4 + 3), int(ys.max() * 4 + 3))))
    return out


def check_captions(plan: dict, ass_path: str, W: int, H: int) -> List[Check]:
    out: List[Check] = []; cues = plan["captions"].get("cues", []) if plan["captions"]["enabled"] else []
    if not cues:
        return [_ok("captions", detail="no captions in this reel")]
    tl = timeline(plan["segments"], plan["transitions"])
    ov = [(i, c) for i, c in enumerate(cues)]
    bad_overlap = [c["start"] for (i, c), (j, d) in zip(ov, ov[1:]) if d["start"] < c["end"] - 0.02]
    out.append(Check("caption_overlap", "fail", len(bad_overlap), 0, bad_overlap, detail="two captions on screen at once") if bad_overlap else _ok("caption_overlap"))
    bad_speed = [(c["start"], glyphs(c["text"]) / max(c["end"] - c["start"], 1e-3)) for c in cues]
    slow = [t for t, cps in bad_speed if cps > MAX_CPS]
    worst = max(cps for _, cps in bad_speed)
    out.append(Check("caption_reading_speed", "fail" if worst > 1.4 * MAX_CPS else "warn", worst, MAX_CPS, slow, detail="too many characters per second to read") if slow else _ok("caption_reading_speed", value=worst))
    cap = caption_ink(ass_path, W, H, plan["durationSeconds"], "Cap")
    z = zones(W, H)          # platform UI zones on portrait: nothing in the top 12%, bottom 22% or outer 8% of the sides
    over = [(t, b) for t, b in cap if b and (b[0] < z["side"] * W - 2 or b[2] > (1 - z["side"]) * W + 2 or b[3] > (1 - z["bottom"]) * H + 2 or b[1] < z["top"] * H - 2)]
    if over:
        worst_x = max(max(0.0, z["side"] * W - b[0], b[2] - (1 - z["side"]) * W, b[3] - (1 - z["bottom"]) * H, z["top"] * H - b[1]) for _, b in over)
        out.append(Check("caption_overflow", "fail", float(worst_x), 0.0, [t for t, _ in over], _clip_at(tl, over[0][0]), "caption ink outside the safe area"))
    else:
        out.append(_ok("caption_overflow"))
    tit = caption_ink(ass_path, W, H, plan["durationSeconds"], "Ovl") ; tit2 = caption_ink(ass_path, W, H, plan["durationSeconds"], "OvlBox")
    hit = []
    for (t, cb), (_, tb), (_, tb2) in zip(cap, tit, tit2):
        for ob in (tb, tb2):
            if cb and ob and cb[0] < ob[2] and ob[0] < cb[2] and cb[1] < ob[3] and ob[1] < cb[3]: hit.append(t)
    out.append(Check("caption_overlay_collision", "fail", len(hit), 0, hit, detail="caption overlaps template text") if hit else _ok("caption_overlay_collision"))
    if zones(W, H) is PORTRAIT:                                                    # template titles must not sit under the platform's top bar / above the bottom controls either
        bad = [t for (t, a), (_, b) in zip(tit, tit2) for ob in (a, b) if ob and (ob[1] < z["top"] * H - 2 or ob[3] > (1 - z["bottom"]) * H + 2)]
        if bad: out.append(Check("overlay_safe_zone", "fail", len(bad), 0, bad, detail="template title inside the platform UI zone"))
        else: out.append(_ok("overlay_safe_zone"))
    return out


# ---------------------------------------------------------------- picture
def _planned_dark(plan: dict, tl: List[dict]) -> List[Tuple[float, float]]:
    win = []
    for i, e in enumerate(edge_transitions(plan["segments"], plan["transitions"])):
        if e["type"] == "dip_black":
            t = tl[i + 1]["start"] + e["durationSeconds"] / 2; win.append((t - e["durationSeconds"] / 2 - 0.15, t + e["durationSeconds"] / 2 + 0.15))
    f = float(plan.get("creative", {}).get("outro", {}).get("fadeSeconds", 0))
    if f > 0: win.append((plan["durationSeconds"] - f - 0.2, plan["durationSeconds"] + 0.1))
    return win


def check_picture(plan: dict, video: str, tl: List[dict]) -> List[Check]:
    out: List[Check] = []; D = plan["durationSeconds"]; dark = _planned_dark(plan, tl)
    black = [(a, b) for a, b in M.black_intervals(video) if not any(a >= x0 and b <= x1 for x0, x1 in dark)]
    tot = sum(b - a for a, b in black)
    out.append(Check("black_frames", "fail", tot, 0.25, [a for a, _ in black], _clip_at(tl, black[0][0]), "unplanned black picture") if tot >= 0.25 else _ok("black_frames", value=tot))
    fr = M.freeze_intervals(video); ftot = sum(b - a for a, b in fr); longest = max([b - a for a, b in fr] or [0.0])
    if longest >= 2.0 or ftot > 0.3 * D:
        out.append(Check("frozen_frames", "fail", ftot, 0.3 * D, [a for a, _ in fr], _clip_at(tl, fr[0][0]), "picture stuck"))
    elif fr:
        out.append(Check("frozen_frames", "warn", ftot, 0.6, [a for a, _ in fr], _clip_at(tl, fr[0][0]), "static picture (may be an intentionally still shot)"))
    else:
        out.append(_ok("frozen_frames"))
    return out


IN_FRAME_WARN, IN_FRAME_FAIL = 0.8, 0.5


def check_subject_in_frame(plan: dict, tl: List[dict]) -> List[Check]:
    """From the plan's own measurement (`subjectInFrame`: share of tracked samples whose subject lies inside the crop window): a clip that keeps the subject in frame
    less than half the time is a fail (replan shows it whole over blur), under 80% a warning."""
    vals = [(i, s["subjectInFrame"]) for i, s in enumerate(plan["segments"]) if "subjectInFrame" in s and s.get("kind", "video") == "video" and (s.get("layout") or {}).get("mode") != "fit_blur"]
    if not vals:
        return [_ok("subject_in_frame")]
    bad = [(i, v) for i, v in vals if v < IN_FRAME_FAIL]; low = [(i, v) for i, v in vals if IN_FRAME_FAIL <= v < IN_FRAME_WARN]; mean = float(np.mean([v for _, v in vals]))
    if bad:
        return [Check("subject_in_frame", "fail", bad[0][1], IN_FRAME_FAIL, [tl[i]["start"] for i, _ in bad], bad[0][0], f"the subject is inside the crop only {bad[0][1]:.0%} of the time")]
    if low:
        return [Check("subject_in_frame", "warn", low[0][1], IN_FRAME_WARN, [tl[i]["start"] for i, _ in low], low[0][0], f"the subject leaves the crop part of the time (mean {mean:.0%})")]
    return [_ok("subject_in_frame", value=mean)]


def _mean_embedding(emb: dict, vis: list, a: float, b: float) -> Optional[np.ndarray]:
    vecs, wts = [], []
    for sv in vis:
        ov = max(0.0, min(sv["end"], b) - max(sv["start"], a))
        if ov > 0 and sv["sceneId"] in emb:
            vecs.append(np.asarray(emb[sv["sceneId"]], dtype=float)); wts.append(ov)
    if not vecs:
        return None
    w = np.asarray(wts); m = (w[:, None] * np.asarray(vecs)).sum(axis=0) / w.sum()
    n = np.linalg.norm(m); return m / n if n > 1e-9 else None


def _label_anchor(vis: list, emb: dict, wants: float, k: int = 3, floor: float = 0.3) -> Optional[np.ndarray]:
    """The mean (renormalized) embedding of the k scenes most confident in `wants` -- several
    examples, not one, so the anchor isn't at the mercy of a single scene's idiosyncrasies
    (EXP-010 found a single-anchor version flips its verdict on the exact clip that matters
    depending on which one scene got picked; averaging 3 fixed that on real data)."""
    ranked = sorted(vis, key=lambda sv: -wants(sv["vision"]["labels"]))[:k]
    ranked = [sv for sv in ranked if wants(sv["vision"]["labels"]) >= floor]
    if not ranked:
        return None
    vs = np.asarray([emb[sv["sceneId"]] for sv in ranked if sv["sceneId"] in emb], dtype=float)
    if not len(vs):
        return None
    m = vs.mean(axis=0); n = np.linalg.norm(m); return m / n if n > 1e-9 else None


# labels reused from scoring.DEVOTIONAL, minus "aarti"/"abhishekam"/"procession": EXP-004-007 found
# those specific labels are the ones that misfire on fireworks/spark footage, so they make bad
# ANCHOR material even though a clip claiming them is still a valid thing to cross-check (below).
_DEVOTION_ANCHOR_LABELS = ("deity", "idol", "priest", "devotees", "lamps", "flowers")
_DEVOTIONAL_REASONS = {"deity", "idol", "priest", "devotees", "lamps", "flowers", "aarti", "abhishekam", "procession"}


def _vlm_second_opinion(frame_path: str) -> Optional[str]:
    """EXP-011 (docs/research/experiment_matrix.md): a local VLM (Ollama/Moondream, Apache-2.0),
    asked one-word, correctly resolved all 9/9 real cases tested -- including the exact Diwali
    scenes the embedding check above exists to flag. Opt-in (AIKYAM_VLM_VERIFY=1), OFF by
    default: ~195s/frame measured on a modest CPU, and it needs Ollama actually running, neither
    of which any other check in this file requires. Returns 'devotion' / 'fireworks' / None
    (disabled, or ANY failure -- service down, model not pulled, timeout: this is a second
    opinion, never a reason to fail or slow down a render by default)."""
    if os.environ.get("AIKYAM_VLM_VERIFY") != "1":
        return None
    import base64, requests
    try:
        b64 = base64.b64encode(open(frame_path, "rb").read()).decode()
        url = os.environ.get("AIKYAM_OLLAMA_URL", "http://localhost:11434")
        model = os.environ.get("AIKYAM_VLM_MODEL", "moondream")
        timeout = float(os.environ.get("AIKYAM_VLM_TIMEOUT", "300"))
        r = requests.post(f"{url}/api/generate", json={
            "model": model, "images": [b64], "stream": False, "options": {"num_predict": 12, "num_ctx": 512},
            "prompt": "One word: is this 'ritual' (a shrine/deity/puja/aarti) or 'fireworks'?",
        }, timeout=timeout)
        r.raise_for_status()
        ans = r.json().get("response", "").lower()
        return "devotion" if "ritual" in ans else "fireworks" if "fireworks" in ans else None
    except Exception:                                     # noqa: BLE001 -- best-effort second opinion, never fail the render over it
        return None


def check_role_label_consistency(plan: dict, workdir: str, margin: float = 0.02) -> List[Check]:
    """EXP-010 (docs/research/experiment_matrix.md): a clip labeled devotional content should
    look more like this source's OWN clearest devotional footage than its clearest fireworks
    footage, by SigLIP embedding similarity -- and vice versa. Scoped to this one confusion
    (not a fully generic any-label-vs-any-label checker: an earlier, broader version of this
    check false-flagged idol/priest clips that are legitimately part of the same real event
    but happen to carry a different scene-level label -- see the experiment log) because
    that's the one pattern this session's research actually found and validated real footage
    against; a broader claim would be unvalidated. Warn-only, best-effort: silently skips
    anything it can't compute cleanly (no embeddings on disk, no usable anchors, an image
    segment) -- optional evidence, never blocks a render."""
    import json
    cache: Dict[Optional[str], Optional[Tuple[dict, list]]] = {}

    def pool(asset_id):
        if asset_id in cache:
            return cache[asset_id]
        for base in ([os.path.join(workdir, "assets", asset_id)] if asset_id else []) + [workdir]:
            e, v = os.path.join(base, "embeddings.json"), os.path.join(base, "vision.json")
            if os.path.exists(e) and os.path.exists(v):
                try:
                    got = (json.load(open(e)), json.load(open(v))); cache[asset_id] = got; return got
                except Exception:                     # noqa: BLE001 -- malformed cache: skip this asset, never fail the render over it
                    break
        cache[asset_id] = None; return None

    flags = []
    for i, s in enumerate(plan.get("segments", [])):
        reason = (s.get("reason") or "").strip()
        expected = "devotion" if reason in _DEVOTIONAL_REASONS else ("fireworks" if reason == "fireworks" else None)
        if expected is None or s.get("kind", "video") != "video":
            continue                                  # not a label this check knows how to cross-check (includes role-name fallbacks like "reveal"/"closing")
        got = pool(s.get("assetId"))
        if got is None:
            continue
        emb, vis = got
        own = _mean_embedding(emb, vis, s["start"], s["end"])
        dev_anchor = _label_anchor(vis, emb, lambda labs: max(labs.get(k, 0.0) for k in _DEVOTION_ANCHOR_LABELS))
        fw_anchor = _label_anchor(vis, emb, lambda labs: labs.get("fireworks", 0.0))
        if own is None or dev_anchor is None or fw_anchor is None:
            continue
        sim_dev, sim_fw = float(own @ dev_anchor), float(own @ fw_anchor)
        got_class = "devotion" if sim_dev > sim_fw else "fireworks"
        if got_class != expected and abs(sim_dev - sim_fw) >= margin:
            flags.append([i, s["start"], reason, got_class, sim_dev, sim_fw, False])   # last field: VLM-confirmed?

    if flags and os.environ.get("AIKYAM_VLM_VERIFY") == "1":
        import tempfile
        segs = plan.get("segments", [])
        for flag in flags:
            i = flag[0]; seg = segs[i]
            src = seg_source(plan, seg, plan.get("source", {}).get("path"))
            if not src:
                continue
            with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
                try:
                    ff.frame_at(src, (seg["start"] + seg["end"]) / 2, tmp.name)
                    verdict = _vlm_second_opinion(tmp.name)
                except Exception:             # noqa: BLE001 -- frame extraction failure: keep the embedding-only flag, never fail the render
                    verdict = None
            flag_expected = "devotion" if flag[2] in _DEVOTIONAL_REASONS else "fireworks"
            if verdict == flag_expected:
                flag[6] = "drop"               # VLM sides with the original label: the embedding flag was the false positive here
            elif verdict is not None:
                flag[6] = True                 # VLM agrees the label is wrong: upgrade confidence in the flag
        flags = [f for f in flags if f[6] != "drop"]

    if not flags:
        return [_ok("role_label_consistency")]
    i0, t0, r0, got0, sd0, sf0, vlm0 = flags[0]
    vlm_note = " -- VLM confirms" if vlm0 is True else ""
    return [Check("role_label_consistency", "warn", len(flags), 0, [t for _, t, *_ in flags], i0,
                  f"clip {i0} is labeled {r0!r} (devotional) but its embedding looks more like this source's own {got0} footage (sim {sd0:.2f} vs {sf0:.2f}){vlm_note}"
                  if r0 in _DEVOTIONAL_REASONS else
                  f"clip {i0} is labeled fireworks but its embedding looks more like this source's own devotional footage (sim {sd0:.2f} vs {sf0:.2f}){vlm_note}")]


def check_crop(plan: dict, video: str, tl: List[dict]) -> List[Check]:
    """A face sliced by the frame edge is a bad crop. Sampled at 2 fps on the OUTPUT frames."""
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-an", "-vf", "fps=2,scale=320:-2,format=gray", "-f", "rawvideo", "-"], capture_output=True)
    info = ff.probe(video); h = int(round(320 * info.height / info.width / 2)) * 2
    fr = np.frombuffer(p.stdout, np.uint8).reshape(-1, h, 320); det = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    cut = []
    for i, f in enumerate(fr):
        for (x, y, w, hh) in det.detectMultiScale(f, 1.15, 5, minSize=(16, 16)):
            if (x <= 0.012 * 320 or x + w >= 0.988 * 320) and w > 0.08 * 320: cut.append(i / 2)
    if len(cut) >= 2:
        return [Check("bad_crop", "fail", len(cut), 1, cut, _clip_at(tl, cut[0]), "a face is cut by the frame edge")]
    return [_ok("bad_crop")]


def check_duplicates_and_weak(plan: dict, video: str, tl: List[dict]) -> List[Check]:
    """Duplicate = the same picture twice: a fine (16x16) average-hash within 5% of the bits AND agreeing colour histograms. Both must hold, so two similar-looking
    but different shots (another aarti, another crowd) are not flagged."""
    out: List[Check] = []; hashes, hists, sharp = [], [], []
    info = ff.probe(video); h = int(round(160 * info.height / info.width / 2)) * 2
    for c in tl:
        t = (c["start"] + c["end"]) / 2
        p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", video, "-frames:v", "1", "-vf", "scale=160:-2,format=bgr24", "-f", "rawvideo", "-"], capture_output=True)
        f = np.frombuffer(p.stdout, np.uint8)[:160 * h * 3].reshape(h, 160, 3) if p.stdout else np.zeros((h, 160, 3), np.uint8)
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY); s16 = cv2.resize(g, (16, 16), interpolation=cv2.INTER_AREA)
        hashes.append((s16 > s16.mean()).flatten()); sharp.append(cv2.Laplacian(g, cv2.CV_64F).var())
        hv = cv2.calcHist([cv2.cvtColor(f, cv2.COLOR_BGR2HSV)], [0, 1], None, [12, 4], [0, 180, 0, 256]); hists.append(cv2.normalize(hv, hv).flatten())
    dup = [(i, j) for i in range(len(hashes)) for j in range(i + 1, len(hashes))
           if (hashes[i] != hashes[j]).sum() <= 0.05 * 256 and cv2.compareHist(hists[i], hists[j], cv2.HISTCMP_CORREL) >= 0.97]
    if dup and plan.get("creative", {}).get("motionDedupe", True):                          # the same venue looks identical at any moment: two clips only repeat if they also MOVE alike (hypecut)
        from .edges import gray_frames, motion_differs, motion_signature
        sig = {}
        for k in {x for pair in dup for x in pair}:
            c = tl[k]; sig[k] = motion_signature(gray_frames(video, c["start"], min(c["end"] - c["start"], 6.0), fps=8))
        dup = [(i, j) for i, j in dup if not motion_differs(sig[i], sig[j])]
    out.append(Check("duplicate_shots", "fail", len(dup), 0, [tl[j]["start"] for _, j in dup], dup[0][1], f"clips {dup[0][0]} and {dup[0][1]} look identical") if dup else _ok("duplicate_shots"))
    med = float(np.median(sharp)) if sharp else 0.0
    weak = [i for i, s in enumerate(sharp) if len(sharp) >= 3 and s < 0.25 * med]
    out.append(Check("weak_shots", "warn", len(weak), 0, [tl[i]["start"] for i in weak], weak[0], "much softer than the rest of the reel") if weak else _ok("weak_shots"))
    return out


def check_transitions(plan: dict, video: str, x: np.ndarray, tl: List[dict]) -> List[Check]:
    out: List[Check] = []; edges = edge_transitions(plan["segments"], plan["transitions"]); d20 = M.frame_diffs(M.frame_series(video, 20))
    bad, jumps = [], []
    for i, e in enumerate(edges):
        t0 = tl[i + 1]["start"]
        if e["type"] in ("cut", "fade") or e["durationSeconds"] <= 0:
            if len(x):
                j = abs(M.audio_jump_db(x, t0)); k = int(t0 * 48000); w = x[max(0, k - 240): k + 240, 0]
                click = float(np.abs(np.diff(w)).max()) if len(w) > 2 else 0.0
                if j > 9.0 or click > 0.35: bad.append((i + 1, t0, j, click))
        else:
            # a correct blend spreads the picture change EVENLY over its frames (even between very different shots); a jump concentrates it in one step
            a, b = int(t0 * 20), min(len(d20), int((t0 + e["durationSeconds"]) * 20) + 1); seg = d20[a:b]
            if len(seg) >= 4:
                ratio = float(seg.max() / (np.median(seg) + 1e-6)); z = M.robust_z(float(seg.max()), d20)
                if ratio > 4.0 and z > 8.0: bad.append((i + 1, t0, z, 0.0))                # one dominant step inside what should be a smooth ramp
    if bad:
        out.append(Check("abrupt_transition", "fail", len(bad), 0, [b[1] for b in bad], bad[0][0], f"edge {bad[0][0]}: level jump/click or picture jump {bad[0][2]:.1f}"))
    else:
        out.append(_ok("abrupt_transition"))
    return out


# ---------------------------------------------------------------- audio
def mix_margin(mix) -> Dict[str, float]:
    """How far the music sits BELOW the live sound, measured on the stems: median and 5th percentile (dB) over frames where the live sound is present (evaluation harness input)."""
    if mix is None or not mix.report.get("music_used") or mix.live_db is None:
        return {}
    n = min(len(mix.live_db), len(mix.music_db)); live, mus = mix.live_db[:n], mix.music_db[:n]; act = live > -55.0
    if act.sum() < 10:
        return {}
    m = live[act] - mus[act]; return {"music_margin_db_median": float(np.median(m)), "music_margin_db_p5": float(np.percentile(m, 5)), "live_active_share": float(act.mean())}


def check_audio(plan: dict, x: np.ndarray, mix) -> List[Check]:
    out: List[Check] = []
    if not len(x):
        return [Check("audio_present", "fail", 0, 1, detail="no audio stream")]
    L = M.loudness(x); S = M.silence_stats(x)
    out.append(Check("loudness", "fail" if abs(L["lufs"] - TARGET_LUFS) > 2.0 else "warn" if abs(L["lufs"] - TARGET_LUFS) > LUFS_TOL else "pass", L["lufs"], TARGET_LUFS, detail=f"integrated {L['lufs']:.1f} LUFS"))
    if out[-1].status == "warn": out[-1].status = "fail" if abs(L["lufs"] - TARGET_LUFS) > LUFS_TOL else "warn"
    out.append(Check("true_peak", "fail", L["true_peak_db"], TP_LIMIT, detail="over the delivery ceiling") if L["true_peak_db"] > TP_LIMIT else _ok("true_peak", value=L["true_peak_db"]))
    out.append(Check("clipping", "fail", L["clipped"], 0, detail="clipped samples") if L["clipped"] > 0 else _ok("clipping"))
    out.append(Check("loudness_range", "warn", L["lra"], 12.0, detail="wide loudness range") if L["lra"] > 12.0 else _ok("loudness_range", value=L["lra"]))
    if S["ratio"] > 0.5 or S["longest_s"] > 4.0:
        out.append(Check("silence", "fail", S["ratio"], 0.5, detail=f"{S['ratio']:.0%} silent, longest {S['longest_s']:.1f}s"))
    elif S["ratio"] > 0.25 or S["longest_s"] > 2.5:
        out.append(Check("silence", "warn", S["ratio"], 0.25, detail=f"{S['ratio']:.0%} silent, longest {S['longest_s']:.1f}s"))
    else:
        out.append(_ok("silence", value=S["ratio"]))
    if mix is not None and mix.report.get("music_used") and mix.live_db is not None:
        n = min(len(mix.live_db), len(mix.music_db), len(mix.presence)); live, mus, pres = mix.live_db[:n], mix.music_db[:n], mix.presence[:n]
        act = (live > -55.0) & (pres >= 0.5); m = live - mus
        if act.sum() >= 10:
            p5 = float(np.percentile(m[act], 5))
            out.append(Check("music_balance", "fail", p5, MIN_MARGIN_DB, [float(i * 0.05) for i in np.where(act & (m < MIN_MARGIN_DB))[0][:20]], detail=f"music only {p5:.1f} dB under devotional live sound") if p5 < MIN_MARGIN_DB else _ok("music_balance", value=p5))
        allact = live > -55.0
        over = np.where(allact & (mus > live - 3.0))[0]
        out.append(Check("music_overpowers", "fail", len(over), 0, [float(i * 0.05) for i in over[:20]], detail="music within 3 dB of the live sound") if len(over) > 20 else _ok("music_overpowers"))
    return out


def check_duration(plan: dict, video: str) -> List[Check]:
    d = ff.probe(video).duration; want = plan["durationSeconds"]; out = []
    if not (5.0 <= d <= 60.0) or abs(d - want) > 0.4:
        out.append(Check("duration", "fail", d, want, detail=f"rendered {d:.1f}s vs planned {want:.1f}s (limits 5-60s)"))
    else:
        rng = plan.get("creative", {}).get("targetRange")
        if rng and not (rng[0] - 6 <= d <= rng[1] + 4): out.append(Check("duration", "warn", d, rng[0], detail=f"outside the profile's {rng[0]}-{rng[1]}s window"))
        else: out.append(_ok("duration", value=d))
    return out


END_MOTION_RATIO = 1.6      # the last second may not move more than this many times the reel's typical motion
OPEN_MIN_LUMA, OPEN_SHARP_RATIO = 12.0, 0.5


def check_start_end(plan: dict, video: str, tl: List[dict]) -> List[Check]:
    """A reel must read as STARTED and FINISHED. Opening: the first frames are not black and about as sharp as the reel's typical frame. Ending: the last second before the outro fade
    has settled (motion not above END_MOTION_RATIO x the reel's median motion). Warnings only: they say where the edit feels unfinished."""
    fr = M.frame_series(video)
    if len(fr) < 15:
        return []
    d = M.frame_diffs(fr); D = plan["durationSeconds"]; f = float(plan.get("creative", {}).get("outro", {}).get("fadeSeconds", 0)); out: List[Check] = []
    sharp = np.array([cv2.Laplacian(x, cv2.CV_64F).var() for x in fr]); k = min(3, len(fr) - 1)
    luma, sh0 = float(fr[:k + 1].mean()), float(np.median(sharp[:k + 1])); typ = float(np.median(sharp)) + 1e-9
    weak = luma < OPEN_MIN_LUMA or sh0 < OPEN_SHARP_RATIO * typ
    out.append(Check("opening_strong", "warn", sh0 / typ, OPEN_SHARP_RATIO, [0.0], 0, f"first frames: luma {luma:.0f}, sharpness {sh0 / typ:.2f}x typical: the reel opens on a dark or soft frame") if weak else _ok("opening_strong", value=sh0 / typ))
    e1 = min(len(d), max(0, int((D - f) * 10))); e0 = max(0, e1 - 10); med = float(np.median(d)) + 1e-6
    tail = float(d[e0:e1].mean()) if e1 > e0 else 0.0; ratio = tail / med
    out.append(Check("ending_resolved", "warn", ratio, END_MOTION_RATIO, [max(0.0, D - f - 1.0)], len(tl) - 1, f"the last second still moves {ratio:.1f}x the typical amount: the reel ends mid-action") if e1 > e0 and ratio > END_MOTION_RATIO and tail > 1.0 else _ok("ending_resolved", value=ratio))
    return out


def run_qc(plan: dict, video: str, workdir: str, mix=None, fmt: str = "reel") -> QCReport:
    from ..render import FORMATS
    spec = FORMATS[fmt]; W, H = spec["w"], spec["h"]; tl = timeline(plan["segments"], plan["transitions"])
    x = M.decode_audio(video); checks: List[Check] = []
    checks += check_duration(plan, video); checks += check_picture(plan, video, tl); checks += check_crop(plan, video, tl); checks += check_subject_in_frame(plan, tl)
    checks += check_audio(plan, x, mix); checks += check_transitions(plan, video, x, tl); checks += check_duplicates_and_weak(plan, video, tl); checks += check_start_end(plan, video, tl)
    checks += check_role_label_consistency(plan, workdir)
    ass = os.path.join(workdir, f"overlay_{fmt}.ass")
    if os.path.exists(ass): checks += check_captions(plan, ass, W, H)
    L = M.loudness(x) if len(x) else {}
    return QCReport(checks, {"lufs": L.get("lufs", float("nan")), "true_peak_db": L.get("true_peak_db", float("nan")), "duration": ff.probe(video).duration, **mix_margin(mix)})
