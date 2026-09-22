"""Render-side check: a second ASR pass over the FINISHED reel vs the words the plan intended (adapted from claude-youtube-editor verify_cut.py, MIT; see NOTICE).
Advisory ('where to listen'): extra words = ghost speech that rode along, missing words = clipped content, drift = audio vs planned timeline.
Opt-in (runs Whisper): tools/verify_reel.py RUN_DIR. Meaningless on footage without speech (returns one pass check saying so)."""
from __future__ import annotations
import re
from difflib import SequenceMatcher
from typing import Dict, List, Union
from ..captions import build_cues
from ..models import Transcript
from ..plan import edge_overlaps
from .qc import Check

DRIFT_TOL_S = 0.25          # ASR word times jitter ~0.1 s; beyond this the audio is not where the plan says
MIN_CONF = 0.70


def _norm(t: str) -> List[str]:
    return re.sub(r"[^\w' ]+", " ", t.lower()).split()


def _words(items) -> List[tuple]:
    """[(normalised word, start, source word)] from objects/dicts with .text/.start"""
    out = []
    for w in items:
        g = (lambda k: w[k]) if isinstance(w, dict) else (lambda k: getattr(w, k))
        out += [(p, g("start"), g("text")) for p in _norm(g("text"))]
    return out


def verify_render(plan: dict, transcripts: Union[Transcript, Dict[str, Transcript]], rendered: Transcript) -> List[Check]:
    segs = plan["segments"]
    cues = build_cues(transcripts, segs, "word", edge_overlaps(segs, plan["transitions"]))
    a = _words(w for c in cues for w in c["words"])
    if not a: return [Check("verify_words", "pass", detail="no speech in the plan: nothing to verify")]
    b = _words(w for s in rendered.segments for w in s.words)
    ops = SequenceMatcher(None, [x[0] for x in a], [x[0] for x in b], autojunk=False).get_opcodes()
    extra = [b[j] for t, _, _, j1, j2 in ops if t in ("insert", "replace") for j in range(j1, j2)]
    missing = [a[i] for t, i1, i2, _, _ in ops if t in ("delete", "replace") for i in range(i1, i2)]
    drift = [b[j][1] - a[i1 + (j - j1)][1] for t, i1, i2, j1, j2 in ops if t == "equal" for j in range(j1, j2)]
    lowc = [w for s in rendered.segments for w in s.words if w.confidence < MIN_CONF]
    med = sorted(drift)[len(drift) // 2] if drift else 0.0
    drifted = [d for d in drift if abs(d - med) > DRIFT_TOL_S]
    n = max(len(a), 1)
    return [
        Check("verify_extra_words", "warn" if len(extra) / n > 0.1 else "pass", len(extra), where=[x[1] for x in extra[:20]], detail="heard in the render but not intended: " + " ".join(x[2] for x in extra[:10])),
        Check("verify_missing_words", "warn" if len(missing) / n > 0.1 else "pass", len(missing), where=[], detail="intended but not heard: " + " ".join(x[2] for x in missing[:10])),
        Check("verify_av_drift", "warn" if drifted else "pass", round(med, 3), DRIFT_TOL_S, detail=f"median audio-vs-plan offset {med:+.2f}s; {len(drifted)}/{len(drift)} words off by > {DRIFT_TOL_S}s"),
        Check("verify_low_confidence", "warn" if len(lowc) / max(len(b), 1) > 0.3 else "pass", len(lowc), where=[w.start for w in lowc[:20]]),
    ]
