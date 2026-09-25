"""Shot analysis: measure a source window in 0.5 s slots, rank slots against the whole footage, find the best in/out inside a window.

Why ranks and not thresholds: a night aarti is legitimately dark and a phone clip legitimately soft. "Sharp" means sharp *for this footage*,
so every quality signal is turned into its percentile among all slots analysed for the video (Population). No fixed pixel thresholds.
"""
from __future__ import annotations
import os, subprocess
from typing import Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np
from ..audio import AudioProfile
from ..models import Moment, SceneVision
from . import edges as _edges, mlx as _mlx
from .model import Shot, SlotFeatures

HOP = 0.5
DEAD_Q = 0.10                   # slot quality below this = unusable footage (lens blocked, whip pan, focus lost)
DEAD_DIP = 0.30                # ...or sharpness under this share of the shot's own median (the blocked lens inside one continuous handheld take)
DEAD_MIN_SLOTS = 6             # a clean stretch must be at least 3 s to be worth keeping
SAMPLE_FPS = 4                   # 2 frames per slot
_FACE = None


def _face():
    global _FACE
    if _FACE is None:
        _FACE = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    return _FACE


def sample_frames(src: str, start: float, end: float, fps: int = SAMPLE_FPS) -> np.ndarray:
    """Gray 320x180 frames sampled at `fps` over [start, end): (N, 180, 320) uint8."""
    p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", src, "-an",
                        "-vf", f"fps={fps},scale=320:180:flags=area,format=gray", "-f", "rawvideo", "-"], capture_output=True)
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, 180, 320)


def _shift(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    (dx, dy), _ = cv2.phaseCorrelate(cv2.resize(a, (160, 90)).astype(np.float32), cv2.resize(b, (160, 90)).astype(np.float32))
    return dx, dy


def analyze_window(src: str, start: float, end: float, audio: Optional[AudioProfile]) -> SlotFeatures:
    fr = sample_frames(src, start, end)
    per = SAMPLE_FPS * HOP                                                     # frames per slot
    n_slots = max(1, int(np.ceil(len(fr) / per)))
    shifts = [(0.0, 0.0)] + [_shift(fr[i - 1], fr[i]) for i in range(1, len(fr))]
    S = {k: [] for k in ("sharpness", "jitter", "motion", "luma", "contrast", "clipped", "concentration", "face", "rms_db", "presence")}
    for s in range(n_slots):
        idx = range(int(s * per), min(len(fr), int((s + 1) * per)))
        if not len(idx):
            for k in S: S[k].append(S[k][-1] if S[k] else 0.0)
            continue
        f = [fr[i] for i in idx]
        S["sharpness"].append(float(np.median([cv2.Laplacian(x, cv2.CV_64F).var() for x in f])))
        S["luma"].append(float(np.mean(f) / 255))
        S["contrast"].append(float(np.mean([(np.percentile(x, 95) - np.percentile(x, 5)) / 255 for x in f])))
        S["clipped"].append(float(np.mean([(x >= 250).mean() for x in f])))
        d = [np.abs(fr[i].astype(np.int16) - fr[i - 1].astype(np.int16)).mean() for i in idx if i > 0]
        S["motion"].append(float(np.mean(d)) if d else 0.0)
        sh = [shifts[i] for i in idx]
        S["jitter"].append(float(np.mean([np.hypot(sh[j][0] - sh[j - 1][0], sh[j][1] - sh[j - 1][1]) for j in range(1, len(sh))])) if len(sh) > 1 else 0.0)
        g = cv2.GaussianBlur(f[len(f) // 2], (5, 5), 0)
        mag = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1))
        c0, c1 = int(180 * 0.2), int(180 * 0.8); x0, x1 = int(320 * 0.2), int(320 * 0.8)
        S["concentration"].append(float(mag[c0:c1, x0:x1].sum() / (mag.sum() + 1e-9)))
        fc = _face().detectMultiScale(f[len(f) // 2], 1.15, 5, minSize=(14, 14))
        S["face"].append(float(max((w * h for _, _, w, h in fc), default=0) / (320 * 180)))
        t0, t1 = start + s * HOP, start + (s + 1) * HOP
        if audio is not None and len(audio.rms_db):
            sl = audio.slice(t0, t1); seg = audio.rms_db[sl]
            S["rms_db"].append(float(seg.mean()) if len(seg) else -80.0)
            S["presence"].append(max([audio.event_score(k, t0, t1) for k in ("chant", "bhajan", "speech", "bell", "conch") if k in audio.events] or [0.0]))
        else:
            S["rms_db"].append(-80.0); S["presence"].append(0.0)
    return SlotFeatures(HOP, **S, motion_sig=_edges.motion_signature(fr))


SPLIT_ABOVE, SPLIT_TO = 22.0, 12.0     # only a window longer than 22 s is cut into equal parts of at most 12 s (a 20 s take stays whole: two halves of one short take are near-duplicates)


def partition_moments(moments: Sequence[Moment], min_len: float = 3.0) -> List[Moment]:
    """Disjoint windows of footage. Overlapping moments (a 20 s and a 18 s moment sharing 10 s) used to exclude each other wholesale, so one continuous take of 28 s could never
    yield more than one or two clips. Highest score first, each later moment keeps only the part nobody claimed; long windows are split into parts <= SPLIT_TO s."""
    claimed: List[tuple] = []; out: List[Moment] = []
    for m in sorted(moments, key=lambda m: (-m.score, m.start)):
        free = [(m.start, m.end)]
        for a, b in claimed:
            free = [x for lo, hi in free for x in ((lo, min(hi, a)), (max(lo, b), hi)) if x[1] - x[0] > 1e-6]
        for lo, hi in [f for f in free if f[1] - f[0] >= min_len]:
            n = int(np.ceil((hi - lo) / SPLIT_TO)) if hi - lo > SPLIT_ABOVE else 1; step = (hi - lo) / n
            for k in range(n):
                out.append(m.model_copy(update={"momentId": m.momentId if (n == 1 and (lo, hi) == (m.start, m.end)) else f"{m.momentId}.{len(out)}", "start": round(lo + k * step, 3), "end": round(lo + (k + 1) * step, 3)}))
        claimed.append((m.start, m.end))
    return sorted(out, key=lambda m: m.start)


def moment_beats(m, vision) -> Dict[str, float]:
    """Story-beat distribution of a moment: time-weighted mean over the scenes it covers ({} when any scene has none, e.g. no image-text model)."""
    sv = [v for v in vision if v.end > m.start and v.start < m.end]
    if not sv or not all(v.vision.beats for v in sv):
        return {}
    wts = [max(min(v.end, m.end) - max(v.start, m.start), 1e-3) for v in sv]; out: Dict[str, float] = {}
    for v, w in zip(sv, wts):
        for k, c in v.vision.beats.items(): out[k] = out.get(k, 0.0) + c * w / sum(wts)
    return out


def diverse_top(moments, beats_of, top_k: int, reserve: int = 4):
    """The best `top_k` moments by score, EXCEPT that up to `reserve` slots go to the best moment of any story beat the score ranking would leave out
    (EXP-018: wide establishing shots score low on devotional relevance, so they never reached the planner's pool; the beat-fit cannot pick what is not there).
    A beat only counts when it clearly dominates its moment (share >= 0.4). No beats known -> identical to the plain top_k."""
    ranked = sorted(moments, key=lambda m: (-m.score, m.start)); head = ranked[:max(0, top_k - reserve)]
    dom = lambda m: (lambda b: max(b, key=b.get) if b else None)(beats_of(m))
    have = {dom(m) for m in head} - {None}; extra = []
    for m in ranked[len(head):]:
        b = beats_of(m); d = dom(m)
        if d is not None and d not in have and b[d] >= 0.4 and len(extra) < reserve:
            extra.append(m); have.add(d)
    rest = [m for m in ranked[len(head):] if m not in extra]
    return (head + extra + rest)[:top_k]


def build_shots(src: str, moments: Sequence[Moment], vision: Sequence[SceneVision], audio: Optional[AudioProfile], top_k: int = 16, asset_id: str = "a1", edge_info: bool = True,
                duration: Optional[float] = None, use_speech: bool = True, diverse: bool = True) -> List[Shot]:
    """One Shot per validated moment (best `top_k` by score): labels, mean embedding, event means and per-slot features."""
    shots = []; ml_cuts = _mlx.cuts(src) if edge_info else None; vad = _mlx.speech(src) if (edge_info and use_speech) else None       # neural evidence, once per file (None -> heuristics); use_speech=False: the recorded voice is not evidence for anything
    parts = partition_moments(moments)
    chosen = diverse_top(parts, lambda m: moment_beats(m, vision), top_k) if diverse else sorted(parts, key=lambda m: (-m.score, m.start))[:top_k]
    for m in sorted(chosen, key=lambda m: (-m.score, m.start)):
        sv = [v for v in vision if v.end > m.start and v.start < m.end]
        labels: Dict[str, float] = {}                                                                   # time-weighted mean over the scenes the window covers: the labels describe the WINDOW, not its luckiest frame
        wts = [max(min(v.end, m.end) - max(v.start, m.start), 1e-3) for v in sv]
        for v, w in zip(sv, wts):
            for k, c in v.vision.labels.items(): labels[k] = labels.get(k, 0.0) + c * w / sum(wts)
        embs = [np.asarray(v.vision.embedding) for v in sv if v.vision.embedding]
        bts: Dict[str, float] = {}
        if sv and all(v.vision.beats for v in sv):                                                      # what the window is FOR in the story, time-weighted like the labels
            for v, w in zip(sv, wts):
                for k, c in v.vision.beats.items(): bts[k] = bts.get(k, 0.0) + c * w / sum(wts)
        emb = None
        if embs:
            e = np.mean(embs, axis=0); emb = (e / (np.linalg.norm(e) + 1e-9)).tolist()
        ev = {k: audio.event_score(k, m.start, m.end) for k in (audio.events if audio is not None else {})}
        sh = Shot(m.momentId, m.start, m.end, m.momentId, m.score, labels, [e.entityId for e in m.entities], emb, analyze_window(src, m.start, m.end, audio), ev, asset_id=asset_id, beats=bts)
        if edge_info:                                                                                   # real cuts and pauses around the window (edge snapping)
            lo, hi = max(0.0, m.start - 1.5), (min(duration, m.end + 1.5) if duration else m.end + 1.5)
            sh.cuts = [c for c in ml_cuts if lo <= c <= hi] if ml_cuts is not None else [c for c in _edges.detect_cuts(src, lo, hi)]      # TransNetV2 when available, else the frame-difference detector
            vad_in = [x for x in (vad or []) if x[1] > lo and x[0] < hi]
            if vad_in and sum(min(x[1], hi) - max(x[0], lo) for x in vad_in) >= 0.3 * (hi - lo): sh.pauses = _mlx.pauses(vad, lo, hi)       # real speech (Silero VAD): pauses are the gaps between phrases
            elif audio is not None and len(audio.rms_db): sh.pauses = [[a, b] for a, b in _edges.pause_runs(audio.rms_db, audio.hop, lo, hi)]
        shots.append(sh)
    return shots


IMAGE_MAX_S = 5.0                # an image may be held up to this long; the story picks 2.5-5 s


def image_shot(path: str, asset_id: str, vp=None) -> Shot:
    """A still as a flexible-length shot. Slot features are constant (no motion, no shake) but sharpness/contrast/clipping/closeness are MEASURED,
    labels/embedding come from the vision provider, and the subject box (largest face, else the edge-energy centroid) drives Ken Burns."""
    fr = cv2.imread(path)
    if fr is None:
        raise ValueError(f"unreadable image {path}")
    h, w = fr.shape[:2]; sc = 320 / w; g = cv2.cvtColor(cv2.resize(fr, (320, max(2, int(h * sc))), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    gh = g.shape[0]; mag = np.abs(cv2.Sobel(cv2.GaussianBlur(g, (5, 5), 0), cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(cv2.GaussianBlur(g, (5, 5), 0), cv2.CV_32F, 0, 1))
    c = mag[int(gh * .2):int(gh * .8), int(320 * .2):int(320 * .8)].sum() / (mag.sum() + 1e-9)
    fc = _face().detectMultiScale(g, 1.15, 5, minSize=(14, 14))
    if len(fc):
        x, y, fw, fh = max(fc, key=lambda f: f[2] * f[3]); focus = [(x + fw / 2) / 320, (y + fh / 2) / gh, fw / 320, fh / gh]
    else:                                                                            # smoothed edge-energy centroid, both axes
        ky = np.convolve(mag.sum(axis=1) + 1e-6, np.ones(9) / 9, "same"); kx = np.convolve(mag.sum(axis=0) + 1e-6, np.ones(9) / 9, "same")
        focus = [float((kx * np.arange(320)).sum() / kx.sum() / 320), float((ky * np.arange(gh)).sum() / ky.sum() / gh), 0.0, 0.0]
    n = int(IMAGE_MAX_S / HOP); one = lambda v: [float(v)] * n
    sl = SlotFeatures(HOP, one(cv2.Laplacian(g, cv2.CV_64F).var()), one(0.0), one(0.0), one(g.mean() / 255), one((np.percentile(g, 95) - np.percentile(g, 5)) / 255),
                      one((g >= 250).mean()), one(c), one(focus[2] * focus[3]), one(-80.0), one(0.0))
    labels, emb = {}, None
    if vp is not None:
        r = vp.analyze(fr); labels = dict(r.labels); emb = r.embedding
    return Shot(asset_id, 0.0, IMAGE_MAX_S, asset_id, 0.5, labels, [], emb, sl, {}, asset_id=asset_id, kind="image", position=0.5, focus=focus, size=[w, h])


def carve(shot: Shot, a: float, b: float, min_len: float = 3.0, suffix: str = "") -> List[Shot]:
    """Sub-shots of `shot`: [a, b] itself (suffix "w") and the footage BEFORE and AFTER it that is at least `min_len` long ("pre" / "post"): per-slot features sliced to the part,
    everything else inherited. Used so that a short hook does not swallow a whole 20 s shot."""
    import dataclasses
    def part(lo, hi, tag):
        i0, i1 = int(round((lo - shot.start) / HOP)), max(int(round((hi - shot.start) / HOP)), int(round((lo - shot.start) / HOP)) + 1)
        sl = shot.slots; f = {k: getattr(sl, k)[i0:i1] or getattr(sl, k)[-1:] for k in ("sharpness", "jitter", "motion", "luma", "contrast", "clipped", "concentration", "face", "rms_db", "presence")}
        return dataclasses.replace(shot, id=f"{shot.id}{tag}", start=lo, end=hi, slots=SlotFeatures(HOP, **f, motion_sig=sl.motion_sig),
                                   cuts=[c for c in shot.cuts if lo <= c <= hi], pauses=[p for p in shot.pauses if p[1] >= lo and p[0] <= hi])
    out = [part(a, b, suffix or "")]
    if a - shot.start >= min_len: out.append(part(shot.start, a, "~pre"))
    if shot.end - b >= min_len: out.append(part(b, shot.end, "~post"))
    return out


class Population:
    """Percentile ranks of every slot signal across ALL analysed shots of this video (adaptive normalisation)."""

    def __init__(self, shots: Sequence[Shot]):
        cat = lambda k: np.concatenate([np.asarray(getattr(s.slots, k), float) for s in shots if s.slots]) if shots else np.zeros(0)
        self.pop = {k: np.sort(cat(k)) for k in ("sharpness", "jitter", "motion", "luma", "contrast", "clipped", "concentration", "face", "rms_db")}

    def rank(self, key: str, v) -> np.ndarray:
        p = self.pop[key]
        if not len(p):
            return np.full(np.shape(v), 0.5)
        return (np.searchsorted(p, v, side="left") + np.searchsorted(p, v, side="right")) / (2.0 * len(p))

    def slot_quality(self, sl: SlotFeatures) -> np.ndarray:
        """Weakest-link quality per slot: geometric mean of (sharpness rank, steadiness rank, detail rank, non-blown rank).
        A slot must be decent on ALL of them: a sharp but shaking or blown-out slot is not a good cut point."""
        r_sharp = self.rank("sharpness", np.asarray(sl.sharpness))
        r_steady = 1.0 - self.rank("jitter", np.asarray(sl.jitter))
        r_detail = self.rank("contrast", np.asarray(sl.contrast))
        r_clip = 1.0 - self.rank("clipped", np.asarray(sl.clipped))
        q = np.stack([r_sharp, r_steady, r_detail, r_clip]); q = np.clip(q, 0.02, 1.0)
        return np.exp(np.log(q).mean(axis=0))

    def energy(self, sl: SlotFeatures) -> np.ndarray:
        """Action/energy per slot: motion rank and live-audio level rank (equal weight); 0 calm .. 1 driving."""
        return 0.5 * self.rank("motion", np.asarray(sl.motion)) + 0.5 * self.rank("rms_db", np.asarray(sl.rms_db))


def dead_slots(shot: Shot, pop: Population) -> np.ndarray:
    """Slots that are unusable footage (lens blocked, whip pan, focus lost): weak against the footage, or a sharpness dip inside this very shot."""
    sl = shot.slots; n = len(sl.sharpness)
    if not n or DEAD_Q <= 0: return np.zeros(n, bool)
    sh = np.asarray(sl.sharpness, float)
    return (pop.slot_quality(sl) < DEAD_Q) | ((sh < DEAD_DIP * np.median(sh)) if n >= 6 else False)


def has_dead(shot: Shot, pop: Population, a: float, b: float) -> bool:
    d = dead_slots(shot, pop)
    return any(d[i] for i in range(len(d)) if shot.start + i * HOP < b and shot.start + (i + 1) * HOP > a)


def best_window(shot: Shot, pop: Population, length: float, snap=None, lit: bool = False) -> Tuple[float, float, float]:
    """Best contiguous [a, a+length) inside the shot maximising mean slot quality; (start, end, mean quality).
    The first and last slot count double: they are where the eye lands, so they must be sharp and steady.
    `snap(t)` may move the OUT point to a quiet moment (never past the shot).
    lit=True (an OPENING): slots are also weighted by brightness (0.10 at luma <= 0.14 rising to 1.0 at >= 0.30, EXP-017): a shot whose average is fine can still hold dark seconds,
    and quality alone (sharpness, steadiness, contrast) happily lands the opening on them."""
    sl = shot.slots; q = pop.slot_quality(sl); n = len(q)
    if lit and len(sl.luma) == n:
        q = q * np.clip((np.asarray(sl.luma, float) - 0.14) / 0.16, 0.10, 1.0)
    k = max(1, int(round(min(length, shot.length) / HOP)))
    ok = np.ones(n, bool)
    dead = dead_slots(shot, pop)
    if dead.any():
        runs, i = [], 0
        while i < n:
            if dead[i]: i += 1; continue
            j = i
            while j < n and not dead[j]: j += 1
            runs.append((i, j)); i = j
        longest = max((j - i for i, j in runs), default=0)
        if longest >= DEAD_MIN_SLOTS:                                       # dead slots never inside a clip, unless no clean stretch is long enough
            ok = ~dead; k = min(k, longest)
    if n <= k and ok.all():
        a, b = shot.start, shot.end
        return a, b, float(q.mean())
    best, bi = -1.0, None
    for i in range(0, n - k + 1):
        if not ok[i:i + k].all(): continue
        w = q[i:i + k]; s = (w.sum() + w[0] + w[-1]) / (k + 2)
        if s > best + 1e-9: best, bi = s, i
    if bi is None:
        bi, best = 0, float(q[:k].mean())
    a = shot.start + bi * HOP; b = min(shot.end, a + k * HOP)
    if snap and b < shot.end - 1e-6:
        b = min(max(snap(b), a + max(HOP * 2, 0.5 * (b - a))), shot.end)
    return round(a, 3), round(b, 3), float(best)
