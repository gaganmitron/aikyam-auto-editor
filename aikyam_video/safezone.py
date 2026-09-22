"""Platform UI safe zones and script-aware text length.

Portrait (Reels / Shorts / TikTok) covers the top with the account bar and the bottom with caption, buttons and progress bar. Conservative union of the published guides
(TikTok ~320 px bottom / 108 top / 60 left / 120 right of 1920; Reels ~400-500 px bottom): keep text out of the top 12%, bottom 22% and the outer 8% of the sides.
Square / landscape have no such overlays: small margins."""
from __future__ import annotations
import unicodedata
from typing import Dict

try:
    import regex as _re
except ImportError:                                            # pragma: no cover - regex is a declared dependency
    _re = None

PORTRAIT = {"top": 0.12, "bottom": 0.22, "side": 0.08}
OTHER = {"top": 0.05, "bottom": 0.08, "side": 0.05}


def is_portrait(w: int, h: int) -> bool:
    return h >= 1.5 * w


def zones(w: int, h: int) -> Dict[str, float]:
    return PORTRAIT if is_portrait(w, h) else OTHER


def glyphs(text: str) -> int:
    """User-perceived characters (extended grapheme clusters), spaces excluded. Kannada 'ಕ್ಷೇತ್ರ' is 5 clusters but 8 code points: reading speed and line width follow the clusters."""
    t = "".join(text.split())
    if _re is not None:
        return len(_re.findall(r"\X", t))
    return sum(1 for c in t if not unicodedata.combining(c) and unicodedata.category(c) not in ("Mn", "Mc", "Me", "Cf"))       # fallback: base characters only
