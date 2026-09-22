"""Aikyam music library: licence-gated tracks + automatic matching to a reel's devotional content.

Rules (from the product brief): never add copyrighted music; original audio is always kept; music is optional.
  * A track is usable only if it is listed in `<MUSIC_LIBRARY_DIR>/library.json` with `licenceVerified: true` and an allowed licence
    (AIKYAM_ALLOWED_MUSIC_LICENCES, default: AIKYAM-OWNED, CC0, PUBLIC-DOMAIN, CC-BY-4.0, CC-BY-3.0, PIXABAY-CONTENT-LICENSE).
    CC-BY tracks must carry `attribution`, which is written into the plan and the feed payload. NC / SA licences are refused.
  * `auto` picks a track only when its tags match the reel (ritual / deity / festival ids, visual labels, energy) and the original
    audio is NOT already devotional music/chanting (music on top of a live bhajan clashes). Otherwise: no music, with the reason recorded.
  * A track id is never guessed: unlisted files are refused (AIKYAM_ALLOW_UNLISTED_MUSIC=1 is a dev-only escape hatch).
"""
from __future__ import annotations
import json, logging, os, re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEFAULT_ALLOWED = "AIKYAM-OWNED,CC0,PUBLIC-DOMAIN,CC-BY-4.0,CC-BY-3.0,PIXABAY-CONTENT-LICENSE"
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
EXTS = (".mp3", ".wav", ".m4a", ".ogg", ".flac")
DEVOTIONAL_SOUNDS = ("bell", "conch", "chant", "bhajan")
# how driving the reel's content is: what a track's `energy` should roughly match
LABEL_ENERGY = {"procession": 0.85, "ritual_dance": 0.85, "fireworks": 0.8, "crowd": 0.6, "decorations": 0.5, "aarti": 0.3,
                "abhishekam": 0.2, "deity": 0.2, "idol": 0.2, "priest": 0.25, "lamps": 0.25, "devotees": 0.3}


class LicenceError(ValueError):
    pass


@dataclass
class Track:
    id: str
    title: str
    path: str
    licence: str
    attribution: Optional[str]
    energy: float
    labels: List[str] = field(default_factory=list)
    deities: List[str] = field(default_factory=list)
    rituals: List[str] = field(default_factory=list)
    festivals: List[str] = field(default_factory=list)
    duration: float = 0.0
    kind: str = ""               # rhythmic | drone | melodic ('' = analyse the audio)


def library_dir() -> Path:
    return Path(os.environ.get("MUSIC_LIBRARY_DIR", "./music"))


def _allowed() -> set:
    return {x.strip().upper() for x in os.environ.get("AIKYAM_ALLOWED_MUSIC_LICENCES", DEFAULT_ALLOWED).split(",") if x.strip()}


def load_library(directory: Optional[str] = None) -> Tuple[List[Track], List[Tuple[str, str]]]:
    """Return (usable tracks, [(track id, reason it was refused)])."""
    d = Path(directory) if directory else library_dir()
    mf = d / "library.json"
    if not mf.is_file():
        return [], []
    ok, bad, allowed = [], [], _allowed()
    for t in json.loads(mf.read_text(encoding="utf-8")).get("tracks", []):
        tid = str(t.get("id", "?"))
        lic = str(t.get("licence", "")).upper()
        path = d / str(t.get("file", ""))
        if not ID_RE.match(tid):
            bad.append((tid, "invalid id")); continue
        if lic not in allowed:
            bad.append((tid, f"licence {t.get('licence')!r} not allowed")); continue
        if not t.get("licenceVerified"):
            bad.append((tid, "licenceVerified is not true")); continue
        if lic.startswith("CC-BY") and not t.get("attribution"):
            bad.append((tid, "CC-BY track without attribution text")); continue
        if not path.is_file() or d.resolve() not in path.resolve().parents:
            bad.append((tid, "audio file missing (or outside the library)")); continue
        tags = t.get("tags", {})
        ok.append(Track(tid, t.get("title", tid), str(path), t["licence"], t.get("attribution"), float(t.get("energy", 0.4)),
                        tags.get("labels", []), tags.get("deities", []), tags.get("rituals", []), tags.get("festivals", []),
                        float(t.get("durationSeconds", 0)), str(t.get("kind", ""))))
    for tid, why in bad:
        logging.getLogger("aikyam").warning(f"music track {tid} refused: {why}")
    return ok, bad


def get_track(track_id: str, directory: Optional[str] = None) -> Track:
    """Resolve an explicit track id. Unlisted / unlicensed / missing => error (never silently substitutes)."""
    if not ID_RE.match(track_id or ""):
        raise FileNotFoundError(f"invalid music track id {track_id!r}")
    tracks, bad = load_library(directory)
    for t in tracks:
        if t.id == track_id:
            return t
    for tid, why in bad:
        if tid == track_id:
            raise LicenceError(f"music track {track_id!r} cannot be used: {why}")
    if os.environ.get("AIKYAM_ALLOW_UNLISTED_MUSIC") == "1":        # dev only: bare file, no licence record
        d = Path(directory) if directory else library_dir()
        for ext in EXTS:
            if (d / f"{track_id}{ext}").is_file():
                return Track(track_id, track_id, str(d / f"{track_id}{ext}"), "UNVERIFIED-DEV", None, 0.4)
    raise FileNotFoundError(f"music track {track_id!r} is not in the licensed library {library_dir()}/library.json")


def wanted_energy(reasons: List[str]) -> float:
    vals = [LABEL_ENERGY[r] for r in reasons if r in LABEL_ENERGY]
    return sum(vals) / len(vals) if vals else 0.4


def score(track: Track, plan: dict) -> float:
    src = plan["source"]; segs = plan["segments"]
    reasons = [s["reason"] for s in segs]
    lab = sum(1 for r in reasons if r in track.labels) / max(1, len(reasons))         # share of segments this track suits
    ids = 0.5 * ((src.get("ritualId") in track.rituals) + (src.get("deityId") in track.deities) + (src.get("festivalId") in track.festivals))
    return round(lab + ids - 0.5 * abs(track.energy - wanted_energy(reasons)), 3)


def original_devotional_level(plan: dict, audio) -> float:
    """Duration-weighted mean of the strongest devotional sound (bell/conch/chant/bhajan) in the segments that made the reel."""
    if audio is None or not getattr(audio, "events", None):
        return 0.0
    tot = w = 0.0
    for s in plan["segments"]:
        L = s["end"] - s["start"]
        tot += L * max((audio.event_score(k, s["start"], s["end"]) for k in DEVOTIONAL_SOUNDS if k in audio.events), default=0.0); w += L
    return tot / w if w else 0.0


def choose(plan: dict, audio, tracks: List[Track], skip_threshold: float = 0.6, min_score: float = 0.3) -> Tuple[Optional[Track], str]:
    """(track or None, human-readable reason)."""
    if not tracks:
        return None, "no licensed music library"
    lvl = original_devotional_level(plan, audio)
    if lvl >= skip_threshold:
        return None, f"original audio is already devotional music/chanting ({lvl:.2f} >= {skip_threshold})"
    ranked = sorted(((score(t, plan), t) for t in tracks), key=lambda x: (-x[0], x[1].id))
    s, best = ranked[0]
    if s < min_score:
        return None, f"no track matches this reel (best {best.id} scored {s} < {min_score})"
    return best, f"matched '{best.id}' (score {s}; original devotional-sound level {lvl:.2f})"


def music_block(track: Track, reason: str, volume: float = 0.5) -> dict:
    return {"enabled": True, "trackId": track.id, "title": track.title, "licence": track.licence, "attribution": track.attribution,
            "volume": volume, "reason": reason}


def register_user_track(path: str, out_dir: str, rights_confirmed: bool) -> str:
    """Your own music file: copied into <out_dir>/user_music/ with a library entry, and made the ONLY library for this run (env MUSIC_LIBRARY_DIR).
    Licence USER-ATTESTED requires the explicit rights flag: nothing is verified by us, the person running it takes responsibility."""
    import shutil
    if not rights_confirmed:
        raise LicenceError("user music needs --i-own-the-music-rights (you confirm you own it or hold a licence to use it in published Reels)")
    if not os.path.isfile(path) or not path.lower().endswith(EXTS):
        raise FileNotFoundError(f"music file {path!r} missing or not one of {EXTS}")
    d = Path(out_dir) / "user_music"; d.mkdir(parents=True, exist_ok=True)
    tid = "user_" + re.sub(r"[^A-Za-z0-9_-]", "_", Path(path).stem)[:50]; f = tid + Path(path).suffix.lower(); shutil.copy(path, d / f)
    (d / "library.json").write_text(json.dumps({"tracks": [{"id": tid, "title": Path(path).stem, "file": f, "licence": "USER-ATTESTED", "licenceVerified": True,
                                                               "attribution": None, "energy": 0.5, "tags": {}}]}), encoding="utf-8")
    os.environ["MUSIC_LIBRARY_DIR"] = str(d.resolve())
    os.environ["AIKYAM_ALLOWED_MUSIC_LICENCES"] = os.environ.get("AIKYAM_ALLOWED_MUSIC_LICENCES", DEFAULT_ALLOWED) + ",USER-ATTESTED"
    return tid
