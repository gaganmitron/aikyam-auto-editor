"""Raw observation -> candidate entities -> canonical Aikyam IDs (section 8). Independent of rendering."""
from __future__ import annotations
import difflib, json, re, unicodedata
from pathlib import Path
from typing import Dict, List, Optional
from .models import EntityRef, Transcript, SceneVision
from .providers import EntityExtractionProvider, register

SEED = Path(__file__).parent / "data" / "kg_seed.json"


def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", s).casefold().strip()


class GazetteerEntityExtractor(EntityExtractionProvider):
    """Alias-table matcher: exact phrase match (conf .96) then fuzzy token match (conf = similarity*.9).
    Note: in Indic scripts fuzzy matching is on whole tokens, so inflected forms may be missed."""
    name = "gazetteer"

    def __init__(self, path: Optional[str] = None):
        kg = json.loads(Path(path or SEED).read_text(encoding="utf-8"))
        self.by_id: Dict[str, dict] = {e["id"]: e for e in kg["entities"]}
        self.ritual_labels: Dict[str, str] = kg.get("ritual_labels", {})
        self.alias: Dict[str, str] = {}
        for e in kg["entities"]:
            for a in e["aliases"] + [e["name"]]:
                self.alias[_norm(a)] = e["id"]

    def extract(self, text: str, language: str = "") -> List[EntityRef]:
        t = _norm(text)
        found: Dict[str, EntityRef] = {}

        def add(matched, eid, conf):
            e = self.by_id[eid]
            if eid not in found or found[eid].confidence < conf:
                found[eid] = EntityRef(text=matched, name=e["name"], entityType=e["type"], entityId=eid, confidence=round(conf, 2))

        for a, eid in self.alias.items():
            if a in t:
                add(a, eid, 0.96)
        toks = set(re.findall(r"\w+", t))
        for tok in toks:
            if len(tok) < 5:
                continue
            for m in difflib.get_close_matches(tok, [a for a in self.alias if " " not in a], n=1, cutoff=0.82):
                add(tok, self.alias[m], 0.9 * difflib.SequenceMatcher(None, tok, m).ratio())
        return sorted(found.values(), key=lambda r: -r.confidence)

    def from_vision(self, labels: Dict[str, float]) -> List[EntityRef]:
        return [EntityRef(text=k, name=self.by_id[eid]["name"] if eid in self.by_id else k, entityType="RITUAL", entityId=eid, confidence=round(labels[k], 2))
                for k, eid in self.ritual_labels.items() if labels.get(k, 0) >= 0.3]


def resolve_media_entities(transcript: Transcript, vision: List[SceneVision],
                           extractor: GazetteerEntityExtractor, hints: Optional[Dict[str, str]] = None) -> Dict[str, List[EntityRef]]:
    """Whole-video entities. `hints` (templeId, etc. from upload metadata) are authoritative, confidence 1.0."""
    ents: Dict[str, EntityRef] = {}
    for s in transcript.segments:
        for r in extractor.extract(s.text, transcript.language):
            if r.entityId not in ents or ents[r.entityId].confidence < r.confidence:
                ents[r.entityId] = r
    for sv in vision:
        for r in extractor.from_vision(sv.vision.labels):
            ents.setdefault(r.entityId, r)
        for did, conf in sv.vision.deities.items():     # visual deity GUESS: discounted, never overrides speech/upload hints
            if did in extractor.by_id:
                d = extractor.by_id[did]
                ents.setdefault(did, EntityRef(text=d["name"], name=d["name"], entityType="DEITY", entityId=did, confidence=round(0.6 * conf, 2)))
    for typ, eid in (hints or {}).items():
        if eid in extractor.by_id:
            e = extractor.by_id[eid]
            ents[eid] = EntityRef(text=e["name"], name=e["name"], entityType=e["type"], entityId=eid, confidence=1.0)
    out: Dict[str, List[EntityRef]] = {}
    for r in ents.values():
        out.setdefault(r.entityType, []).append(r)
    return out


register("entity", "gazetteer", GazetteerEntityExtractor)
