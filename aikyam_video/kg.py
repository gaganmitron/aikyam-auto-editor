"""Aikyam Knowledge Graph integration (Phase 2).

ASSUMED contract (no Aikyam KG API was available to read; change `KGClient.resolve` in ONE place when it is known):
  POST {AIKYAM_KG_URL}/v1/entities/resolve   Authorization: Bearer $AIKYAM_KG_TOKEN
  body  {"texts": ["chamundeshwari", ...], "language": "kn"}
  reply {"matches": [{"text": "chamundeshwari", "entityId": "deity_45", "entityType": "DEITY",
                      "name": "Chamundeshwari", "confidence": 0.96}, ...]}
KGEntityExtractor uses the KG first and falls back to the local gazetteer if the KG is unreachable, so processing never
blocks on it (the fallback is logged)."""
from __future__ import annotations
import logging, os, re
from typing import Dict, List, Optional
import requests
from .entities import GazetteerEntityExtractor
from .models import EntityRef
from .providers import EntityExtractionProvider, register


class KGClient:
    def __init__(self, url: Optional[str] = None, token: Optional[str] = None, timeout: float = 5.0):
        self.url = (url or os.environ["AIKYAM_KG_URL"]).rstrip("/")
        self.token, self.timeout = token or os.environ.get("AIKYAM_KG_TOKEN"), timeout

    def resolve(self, texts: List[str], language: str = "") -> List[dict]:
        h = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        r = requests.post(f"{self.url}/v1/entities/resolve", json={"texts": texts, "language": language}, headers=h, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["matches"]


def _ngrams(text: str, n: int = 3) -> List[str]:
    toks = re.findall(r"\w+", text.casefold())
    out = {" ".join(toks[i:i + k]) for k in range(1, n + 1) for i in range(len(toks) - k + 1)}
    return sorted(t for t in out if len(t) >= 4)[:200]


class KGEntityExtractor(EntityExtractionProvider):
    name = "kg"

    def __init__(self, client: Optional[KGClient] = None):
        self.client = client or KGClient()
        self.fallback = GazetteerEntityExtractor()
        self.by_id: Dict[str, dict] = dict(self.fallback.by_id)
        self.ritual_labels = self.fallback.ritual_labels

    def extract(self, text: str, language: str = "") -> List[EntityRef]:
        try:
            best: Dict[str, EntityRef] = {}
            for m in self.client.resolve(_ngrams(text), language):
                self.by_id.setdefault(m["entityId"], {"id": m["entityId"], "type": m["entityType"], "name": m.get("name", m["text"])})
                r = EntityRef(text=m["text"], name=m.get("name"), entityType=m["entityType"], entityId=m["entityId"], confidence=float(m["confidence"]))
                if r.entityId not in best or best[r.entityId].confidence < r.confidence:
                    best[r.entityId] = r
            return sorted(best.values(), key=lambda r: -r.confidence)
        except (requests.RequestException, KeyError, ValueError) as e:
            logging.getLogger("aikyam").warning(f"KG unavailable ({e}); using local gazetteer")
            return self.fallback.extract(text, language)

    def from_vision(self, labels):
        return self.fallback.from_vision(labels)


register("entity", "kg", KGEntityExtractor)
