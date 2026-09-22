"""Neural boundary evidence, optional. TransNetV2 (shot cuts) and Silero VAD (speech segments) run in the .venv-beat interpreter (see ml_worker.py), once per file, cached next to it
(<file>.ml.json). Any failure -> None and the heuristic detectors (edges.py, RMS pauses) stay in charge. Disable with AIKYAM_NO_ML=1."""
from __future__ import annotations
import hashlib, json, logging, os, subprocess
from pathlib import Path
from typing import List, Optional
from .beats import python_exe

WORKER = Path(__file__).resolve().parent.parent / "ml_worker.py"


def _run(kind: str, path: str, timeout: int = 900) -> Optional[list]:
    exe = python_exe()
    if exe is None or os.environ.get("AIKYAM_NO_ML"):
        return None
    st = os.stat(path); key = hashlib.sha1(f"{st.st_size}:{int(st.st_mtime)}".encode()).hexdigest()[:12]; cache = path + ".ml.json"; d = {}
    try:
        d = json.load(open(cache))
        if d.get("_key") == key and kind in d: return d[kind]
    except (OSError, ValueError): d = {}
    if d.get("_key") != key: d = {}
    try:
        r = subprocess.run([exe, str(WORKER), kind, path], capture_output=True, text=True, timeout=timeout)
        if r.returncode: raise RuntimeError(r.stderr[-300:])
        res = json.loads(r.stdout.strip().splitlines()[-1])[kind]
    except Exception as e:                                                                   # noqa: BLE001 -- heuristics take over
        logging.getLogger("aikyam").warning(f"neural {kind} unavailable ({e}); using heuristics"); return None
    d.update({"_key": key, kind: res})
    try: json.dump(d, open(cache, "w"))
    except OSError: pass
    return res


def cuts(path: str) -> Optional[List[float]]:
    """Shot-boundary times (s) of the whole video (TransNetV2), or None."""
    return _run("cuts", path)


def speech(path: str) -> Optional[List[List[float]]]:
    """Speech segments [[start, end]...] (Silero VAD), or None. Needs an audio stream."""
    from .. import ffmpeg as ff
    return _run("speech", path) if ff.has_audio_stream(path) else None


def pauses(segs: List[List[float]], t0: float, t1: float, min_gap: float = 0.3) -> List[List[float]]:
    """Silences BETWEEN speech segments inside [t0, t1] as [[start, end]...]."""
    return [[a[1], b[0]] for a, b in zip(segs, segs[1:]) if b[0] - a[1] >= min_gap and a[1] >= t0 and b[0] <= t1]
