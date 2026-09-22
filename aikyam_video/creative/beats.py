"""Beat and downbeat tracking with Beat This! (Foscarin et al., ISMIR 2024; MIT) run in a separate interpreter (see beat_this_worker.py).

Optional: point AIKYAM_BEAT_THIS_PYTHON at a Python >= 3.10 with `beat-this` + CPU torch (default: <repo>/.venv-beat/bin/python). Without it, or on any failure,
`track()` returns None and the built-in autocorrelation tracker in musicdna.py is used. Results are cached next to the audio file (<file>.beats.json)."""
from __future__ import annotations
import hashlib, json, logging, os, subprocess
from pathlib import Path
from typing import Optional

WORKER = Path(__file__).resolve().parent.parent / "beat_this_worker.py"


def python_exe() -> Optional[str]:
    p = os.environ.get("AIKYAM_BEAT_THIS_PYTHON") or str(Path(__file__).resolve().parents[2] / ".venv-beat" / "bin" / "python")
    return p if os.path.isfile(p) else None


def track(path: str, use_cache: bool = True, timeout: int = 600) -> Optional[dict]:
    """{'beats': [s...], 'downbeats': [s...]} or None."""
    exe = python_exe()
    if exe is None:
        return None
    st = os.stat(path); key = hashlib.sha1(f"{st.st_size}:{int(st.st_mtime)}".encode()).hexdigest()[:12]; cache = path + ".beats.json"
    if use_cache and os.path.exists(cache):
        try:
            d = json.load(open(cache))
            if d.get("_key") == key: return {"beats": d["beats"], "downbeats": d["downbeats"]}
        except (OSError, ValueError, KeyError): pass
    try:
        r = subprocess.run([exe, str(WORKER), path], capture_output=True, text=True, timeout=timeout)
        if r.returncode:
            raise RuntimeError(r.stderr[-300:])
        d = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:                                                          # noqa: BLE001 -- the built-in tracker takes over
        logging.getLogger("aikyam").warning(f"Beat This! unavailable ({e}); using the built-in tempo tracker"); return None
    if use_cache:
        try: json.dump({**d, "_key": key}, open(cache, "w"))
        except OSError: pass
    return d
