"""Find music for a reel on the internet, automatically, and only if the licence allows it.

  content -> queries : the reel's dominant labels + pacing profile become search phrases (aarti -> "aarti bhajan", contemplative -> "meditation ambient flute", ...)
  search             : Openverse (api.openverse.org, no key). Kept: CC0, public domain, CC BY 3.0/4.0 (attribution is stored and written into the plan). Dropped: NC, ND, SA, other versions.
  download           : a few candidates, size/length capped, cached in <library>/web/
  rank               : CLAP (zero-shot audio-text model, laion/clap-htsat-unfused) similarity of the audio to a description of THIS reel, minus similarity to non-music prompts (speech, noise,
                       sound effect, silence); + a small fit bonus from our own music analysis (drone/melodic for calm reels, rhythmic for festive ones)
  register           : the winner is added to the web library (music/web/library.json, untracked) so the normal gate (music.load_library) applies.
Licence data is Openverse's metadata: not independently verified. Look at the attribution/source recorded in library.json before publishing."""
from __future__ import annotations
import hashlib, json, logging, os, re, urllib.parse, urllib.request
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from . import music as M

API = "https://api.openverse.org/v1/audio/"
UA = {"User-Agent": "aikyam-video/0.1 (music finder)"}
LICENCES = {("cc0", None): "CC0", ("pdm", None): "PUBLIC-DOMAIN", ("by", "4.0"): "CC-BY-4.0", ("by", "3.0"): "CC-BY-3.0"}
MAX_BYTES, MAX_S = 14_000_000, 420.0

# Openverse matches titles/tags, so queries are SINGLE words ("aarti bhajan devotional" finds nothing); relevance is then decided by CLAP, not by the search.
LABEL_WORDS = {"aarti": ["mantra", "indian"], "abhishekam": ["mantra", "raga"], "procession": ["drums", "festival"], "ritual_dance": ["drums", "world"], "fireworks": ["festival"],
               "deity": ["spiritual", "indian"], "idol": ["spiritual", "indian"], "priest": ["mantra"], "devotees": ["spiritual"], "lamps": ["meditation"], "flowers": ["ambient"],
               "temple_architecture": ["temple", "raga"], "decorations": ["festival"], "crowd": ["festival"]}
PROFILE_WORDS = {"contemplative": ["meditation", "flute", "ambient"], "devotional": ["indian", "sitar", "spiritual", "raga"], "festive": ["drums", "festival", "world"]}
PROFILE_DESC = {"contemplative": "slow calm meditative Indian devotional music with flute and tanpura drone", "devotional": "gentle devotional Indian temple music, instrumental bhajan",
                "festive": "energetic festive Indian temple music with drums and percussion"}
NEGATIVE = ["a person talking", "noise and static", "a short sound effect", "silence"]
FIT = {"contemplative": {"drone": 0.06, "melodic": 0.04, "rhythmic": -0.05}, "devotional": {"melodic": 0.04, "drone": 0.02, "rhythmic": 0.0}, "festive": {"rhythmic": 0.06, "melodic": 0.0, "drone": -0.05}}


def queries(labels: Dict[str, float], profile: str, extra: Optional[List[str]] = None, n: int = 6) -> List[str]:
    """Up to n distinct single-word searches: what the reel shows first, then what its pacing wants."""
    top = [k for k, _ in sorted(labels.items(), key=lambda kv: -kv[1]) if k in LABEL_WORDS][:2]
    out: List[str] = []
    for q in [w for k in top for w in LABEL_WORDS[k]] + PROFILE_WORDS.get(profile, PROFILE_WORDS["devotional"]):
        if q not in out: out.append(q)
    ext = [q for i, q in enumerate(extra or []) if q not in (extra or [])[:i]]                    # extras keep their place: the usual list is trimmed to make room, never the other way round
    return ([q for q in out if q not in ext][:max(0, n - len(ext))] + ext)[:n]


def description(labels: Dict[str, float], profile: str) -> str:
    top = [k.replace("_", " ") for k, _ in sorted(labels.items(), key=lambda kv: -kv[1]) if k in LABEL_WORDS][:2]
    return PROFILE_DESC.get(profile, PROFILE_DESC["devotional"]) + (f", for a {' and '.join(top)} scene" if top else "")


def _get(url: str, timeout: int = 30) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r: return r.read()


def search(q: str, min_s: float, n: int = 20, fetch: Callable[[str], bytes] = _get) -> List[dict]:
    """Candidates whose licence we may use, long enough, as dicts: id, title, creator, url, landing, licence, seconds, provider."""
    d = json.loads(fetch(f"{API}?{urllib.parse.urlencode({'q': q, 'license_type': 'commercial', 'category': 'music', 'page_size': n})}"))
    out = []
    for r in d.get("results", []):
        lic = LICENCES.get((str(r.get("license", "")).lower(), None if str(r.get("license", "")).lower() in ("cc0", "pdm") else str(r.get("license_version"))))
        sec = (r.get("duration") or 0) / 1000.0
        if lic is None or not r.get("url") or not (min_s <= sec <= MAX_S): continue
        out.append({"id": r["id"], "title": r.get("title") or "untitled", "creator": r.get("creator") or "unknown", "url": r["url"], "landing": r.get("foreign_landing_url") or r["url"],
                    "licence": lic, "licenceUrl": r.get("license_url"), "seconds": sec, "provider": r.get("provider")})
    return out


def download(c: dict, cache: str, fetch=_get) -> Optional[str]:
    os.makedirs(cache, exist_ok=True); p = os.path.join(cache, "web_" + hashlib.sha1(c["id"].encode()).hexdigest()[:12] + ".mp3")
    if os.path.exists(p): return p
    req = urllib.request.Request(c["url"], headers=UA); n = 0
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(p + ".part", "wb") as f:
            while True:
                b = r.read(1 << 16)
                if not b: break
                n += len(b)
                if n > MAX_BYTES: raise ValueError("file too large")
                f.write(b)
        os.replace(p + ".part", p); return p
    except Exception as e:                                                                    # noqa: BLE001 -- a dead link only removes that candidate
        logging.getLogger("aikyam").warning(f"music download failed ({c['title']}): {e}")
        try: os.remove(p + ".part")
        except OSError: pass
        return None


class Clap:
    """CLAP audio/text embeddings (the same checkpoint audio.py uses for sound tagging)."""
    def __init__(self, model: str = "laion/clap-htsat-unfused"):
        import torch
        from transformers import ClapModel, ClapProcessor
        self.torch = torch; self.m = ClapModel.from_pretrained(model).eval(); self.p = ClapProcessor.from_pretrained(model)

    def text(self, texts: List[str]) -> np.ndarray:
        with self.torch.no_grad(): t = self.m.get_text_features(**self.p(text=texts, return_tensors="pt", padding=True))
        return (t / t.norm(dim=-1, keepdim=True)).numpy()

    def audio(self, path: str) -> np.ndarray:
        from . import ffmpeg as ff
        x = ff.extract_audio_pcm(path, 48000); W = 10 * 48000
        if len(x) < W: x = np.pad(x, (0, W - len(x)))
        ch = [x[int(f * (len(x) - W)):int(f * (len(x) - W)) + W] for f in (0.3, 0.65)]                  # two 10 s excerpts, away from intro/outro
        with self.torch.no_grad(): a = self.m.get_audio_features(**self.p(audios=ch, sampling_rate=48000, return_tensors="pt"))
        a = (a / a.norm(dim=-1, keepdim=True)).numpy().mean(axis=0); return a / (np.linalg.norm(a) + 1e-9)


def rank(paths: Dict[str, str], desc: str, profile: str, embedder=None, kind_of: Optional[Callable[[str], str]] = None, beat_bonus: float = 0.0) -> List[Tuple[float, str, dict]]:
    """[(score, id, detail)] best first. paths: candidate id -> local file."""
    from . import ffmpeg as ff
    from .creative.musicdna import SR, analyze_samples
    emb = embedder or Clap(); T = emb.text([desc] + NEGATIVE); out = []
    for cid, p in paths.items():
        a = emb.audio(p); sims = T @ a
        an = None if kind_of else analyze_samples(ff.extract_audio_pcm(p, SR))                     # fast autocorrelation only: the full beat-grid analysis (Beat This!) is for the winner
        kind = kind_of(p) if kind_of else an.kind
        beat = bool(an is not None and an.bpm)                                                      # a detectable tempo: the cuts can land on it (beat_bonus > 0 when they must follow the music)
        s = float(sims[0] - 0.5 * sims[1:].max() + FIT.get(profile, {}).get(kind, 0.0) + (beat_bonus if beat else 0.0))
        out.append((round(s, 4), cid, {"similarity": round(float(sims[0]), 4), "nonMusic": round(float(sims[1:].max()), 4), "kind": kind, "beat": beat}))
    return sorted(out, key=lambda x: (-x[0], x[1]))


def register(c: dict, path: str, kind: str, labels: List[str], lib: Optional[str] = None) -> M.Track:
    """Add the downloaded track to the licensed library (attribution written for CC BY) and return it."""
    d = (M.library_dir() if lib is None else __import__("pathlib").Path(lib)) / "web"; d.mkdir(parents=True, exist_ok=True); mf = d / "library.json"      # its OWN library: third-party audio never enters the curated one
    data = json.loads(mf.read_text(encoding="utf-8")) if mf.is_file() else {"tracks": []}
    tid = "web_" + re.sub(r"[^A-Za-z0-9]", "", c["id"])[:12]; rel = os.path.relpath(path, d).replace(os.sep, "/")
    attr = f'"{c["title"]}" by {c["creator"]} ({c["landing"]}), {c["licence"]}' if c["licence"].startswith("CC-BY") else None
    data["tracks"] = [t for t in data["tracks"] if t.get("id") != tid] + [{"id": tid, "title": c["title"], "file": rel, "licence": c["licence"], "licenceVerified": True, "attribution": attr, "energy": 0.4, "kind": kind,
        "durationSeconds": round(c["seconds"], 1), "tags": {"labels": labels}, "source": f"openverse:{c['id']}", "sourceUrl": c["landing"], "licenceUrl": c.get("licenceUrl"),
        "note": "licence taken from Openverse metadata, not independently verified"}]
    mf.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return M.get_track(tid)


def find(labels: Dict[str, float], profile: str, reel_seconds: float, n_download: int = 8, fetch=_get, embedder=None, lib: Optional[str] = None, beat_bonus: float = 0.0, extra: Optional[List[str]] = None) -> Tuple[Optional[M.Track], str]:
    """(Track or None, why). Never raises: no network / no match -> (None, reason) and the normal music choice stands."""
    try:
        cands: Dict[str, dict] = {}
        for q in queries(labels, profile, extra, n=6 + len(extra or [])):                    # extras are searched IN ADDITION to the usual six (queries() would cut them off)
            for c in search(q, reel_seconds + 5.0, fetch=fetch):
                cands.setdefault(c["id"], {**c, "query": q, "hits": 0})["hits"] += 1                                            # found by several searches = more likely on topic
        if not cands: return None, "web music: no track with an allowed licence matched the searches"
        cache = str((M.library_dir() if lib is None else __import__("pathlib").Path(lib)) / "web"); paths: Dict[str, str] = {}
        for c in sorted(cands.values(), key=lambda c: (-c["hits"], -c["seconds"], c["id"]))[:n_download * 2]:
            if len(paths) >= n_download: break
            p = download(c, cache)
            if p: paths[c["id"]] = p
        if not paths: return None, "web music: every download failed"
        best = rank(paths, description(labels, profile), profile, embedder, beat_bonus=beat_bonus)[0]; c = cands[best[1]]
        t = register(c, paths[best[1]], best[2]["kind"], sorted(labels, key=lambda k: -labels[k])[:4], lib)
        return t, f"found online: '{c['title']}' by {c['creator']} ({c['licence']}, {c['provider']}), query '{c['query']}', CLAP similarity {best[2]['similarity']}, {len(paths)} candidates compared"
    except Exception as e:                                                                           # noqa: BLE001
        logging.getLogger("aikyam").warning(f"web music unavailable: {e}"); return None, f"web music unavailable: {e}"
