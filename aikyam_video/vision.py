"""VisionProvider implementations (section 7).

HeuristicVision: OpenCV only, no model download. Reports ONLY what it can measure (faces, flame/lamp-coloured
light, warm decoration colour, brightness, sharpness) with honest confidences. It does NOT claim to recognise deities.
ClipVision: open_clip zero-shot over the devotional label set (extra: pip install .[clip]).
"""
from __future__ import annotations
import os
from typing import Dict, List, Sequence
import cv2
import numpy as np
from .models import VisionResult
from .providers import EmbeddingProvider, VisionProvider, register

# Canonical label -> prompt phrasings (max similarity over phrasings is used). Independent per-label scoring:
# each label competes only against NEGATIVES, not against other labels, so co-occurring labels (priest + aarti + deity) all fire.
LABEL_PROMPTS: Dict[str, List[str]] = {
    "deity": ["a photo of a decorated Hindu deity idol inside a temple shrine", "a garlanded statue of a Hindu god or goddess", "a photo of a Hindu deity idol"],
    "idol": ["a close-up of a stone or metal temple idol", "a sanctum with a decorated murti"],
    "priest": ["a Hindu priest in traditional dress performing a ritual", "a pujari offering worship at a temple", "a priest performing a ritual in front of a deity"],
    "devotees": ["devotees praying with folded hands in a temple", "a group of worshippers sitting and singing bhajans",
                 "a crowd of people sitting on the floor singing devotional songs", "a group of people sitting in a temple hall"],
    "temple_architecture": ["Indian temple architecture, carved gopuram or stone pillars", "the exterior of a Hindu temple"],
    "flowers": ["a plate of marigold flowers and flower garlands offered to a deity", "a photo of marigold flowers", "flower garlands"],
    "lamps": ["burning oil lamps and diyas", "a lit brass lamp flame in a temple", "a photo of a burning lamp", "a flame in a temple"],
    "aarti": ["a priest waving a lit aarti lamp in front of a deity", "aarti ritual with fire being circled before an idol", "fire flame in front of a deity idol",
              "a person holding a large brass aarti lamp with many flames", "Ganga aarti ceremony with fire lamps on the river bank at dusk"],
    "abhishekam": ["abhishekam: milk and water being poured over an idol", "ritual bathing of a deity idol"],
    "procession": ["a religious procession with a decorated chariot in the street", "devotees carrying a palanquin in a festival procession",
                   "a huge decorated temple chariot pulled by a crowd", "a street procession with flags and a decorated tractor or truck"],
    "fireworks": ["fireworks exploding and a burning effigy of Ravana", "a large effigy on fire with sparks and smoke", "fireworks exploding in the sky"],
    "ritual_dance": ["a ritual dance performer in an elaborate traditional costume and headdress", "a Theyyam performer in a temple courtyard",
                     "people dancing in a religious festival"],
    "crowd": ["a large crowd of people gathered", "a packed temple crowd"],
    "decorations": ["temple decorated with festival lights and flower garlands", "colourful festival decorations", "a photo of festival decorations"],
    # shots a temple videographer routinely takes (docs/research: EXP-012 coverage review) that the labels above did not name
    "incense_smoke": ["incense smoke rising in front of a temple deity", "smoke from agarbatti and camphor in a temple"],
    "offerings": ["a puja thali with fruits, coconut and flowers as offerings", "prasad and offerings placed before a deity"],
    "temple_bell": ["large brass temple bells hanging at a temple entrance", "a devotee ringing a temple bell"],
    "ritual_hands": ["close-up of a priest applying tilak on a devotee's forehead", "close-up of hands offering flowers or lighting a lamp in a temple", "hands folded in prayer close-up"],
    "sanctum_view": ["a deity idol seen through a temple doorway in a dim sanctum", "darshan of a deity framed by a temple doorway"],
    "devotees_walking": ["devotees walking around a temple in circumambulation", "a queue of devotees entering a temple for darshan", "a devotee prostrating in prayer"],
    "food_service": ["devotees being served prasadam food in a temple hall", "a community meal with people sitting in rows being served food"],
    "architectural_detail": ["a close-up of carved stone temple pillars and sculptures", "an ornate carved temple doorway"],
    "temple_tank": ["a temple tank pond with steps and the temple reflected in the water"],
    "temple_night": ["a temple lit up with lights at night", "an illuminated temple gopuram at night"],
}
NEGATIVES: List[str] = [
    "a photo of an ordinary room", "a photo of a street with cars", "a photo of a person's face", "a landscape photo",
    "a photo of a shop", "a photo of an office", "a photo of a kitchen", "a blurry dark photo", "a photo of an animal",
    "a photo of a building exterior", "a screenshot with text", "a photo of a wall",
]
KEEP = 0.2   # minimum reported confidence
# CLIP-IQA-style antonym pairs (EXP-012, tools/bench_clips.py): score = logit(good) - logit(bad); within-video rho with reel-worthiness +0.52 / +0.85 / +0.79 on 3 videos
NO_TEXT_PAIR = ("a photo with no text", "a screenshot with subtitles, credits and text overlay")      # EXP-012: rho +0.21 with reel-worthiness; credits/title cards are never reel material
AESTHETIC_PAIRS = [("a high quality, beautiful, well composed photo", "a low quality, ugly, badly composed photo"),
                   ("a striking cinematic photo of a temple ritual", "a boring, cluttered snapshot")]

# Content moderation (zero-shot, conservative): a clip whose scene scores >= MODERATION_BLOCK is rejected by validation.
MODERATION_PROMPTS: Dict[str, List[str]] = {
    "explicit_nudity": ["an explicit nude person", "a sexually explicit photo"],
    "graphic_violence": ["a photo of graphic violence with blood and injuries", "a dead body or gore"],
    "weapons_violence": ["a person attacking another person with a weapon", "a violent riot with fighting"],
}
DEFAULT_MODEL = "hf-hub:timm/ViT-B-16-SigLIP"


def model_name() -> str:
    """The image-text model in use: VISION_MODEL, else the default. Scores that were FITTED on one model (ranker.py) are only valid for that model."""
    return os.environ.get("VISION_MODEL", DEFAULT_MODEL)


MODERATION_BLOCK = float(os.environ.get("MODERATION_BLOCK", "0.6"))
# Visual deity guesses come from the KG (`visual` prompt per deity). Report only a clear winner.
DEITY_MIN, DEITY_MARGIN = 0.5, 0.15


def _sharpness(gray) -> float:
    return float(min(1.0, cv2.Laplacian(gray, cv2.CV_64F).var() / 500.0))


def _saliency_x(gray, faces) -> float:
    if faces:
        f = max(faces, key=lambda f: f[2] * f[3])
        return float(f[0] + f[2] / 2)
    g = cv2.resize(gray, (160, 90))
    mag = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1))
    col = mag.sum(axis=0) + 1e-6
    # smoothed centroid of edge energy: where the detail is, not the frame centre
    col = np.convolve(col, np.ones(9) / 9, mode="same")
    return float((col * np.arange(len(col))).sum() / col.sum() / len(col))


class HeuristicVision(VisionProvider):
    name = "opencv-heuristic"

    def __init__(self):
        self._face = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))

    def analyze(self, frame: np.ndarray) -> VisionResult:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        rects = self._face.detectMultiScale(cv2.resize(gray, (w // 2, h // 2)), 1.15, 5, minSize=(24, 24))
        faces = [[x * 2 / w, y * 2 / h, fw * 2 / w, fh * 2 / h] for x, y, fw, fh in rects]
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        flame = float(((H >= 5) & (H <= 30) & (S > 120) & (V > 200)).mean())        # bright orange/yellow light
        warm = float((((H <= 25) | (H >= 165)) & (S > 130) & (V > 90)).mean())      # marigold / kumkum / silk
        labels: Dict[str, float] = {}
        if flame > 0.002:
            labels["lamps"] = min(1.0, flame * 60)
        if warm > 0.05:
            labels["decorations"] = min(1.0, warm * 3)
        if faces:
            labels["devotees" if len(faces) >= 2 else "priest"] = min(1.0, 0.4 + 0.15 * len(faces))
        if len(faces) >= 5:
            labels["crowd"] = min(1.0, len(faces) / 10)
        return VisionResult(labels=labels, faces=faces, saliency_x=_saliency_x(gray, faces),
                            brightness=float(V.mean() / 255), sharpness=_sharpness(gray), provider=self.name)


class _PromptGroup:
    """Text embeddings for {key: [phrasings]}; score(image_emb) -> {key: best phrasing probability}."""
    def __init__(self, owner_model, prompts: Dict[str, List[str]]):
        self.keys = list(prompts)
        flat = [(k, p) for k, ps in prompts.items() for p in ps]
        self.owner = np.array([self.keys.index(k) for k, _ in flat])
        self.emb = owner_model.embed_text([p for _, p in flat]) if flat else np.zeros((0, 1))

    def score(self, m: "ClipVision", emb: np.ndarray) -> Dict[str, float]:
        if not len(self.keys):
            return {}
        if m.sigmoid:
            p = 1 / (1 + np.exp(-(float(m.model.logit_scale.exp()) * (self.emb @ emb) + float(m.model.logit_bias))))
            return {k: float(p[self.owner == i].max()) for i, k in enumerate(self.keys)}
        pos, neg = 100 * (self.emb @ emb), 100 * (m._neg @ emb)
        mx = max(pos.max(), neg.max()); dn = np.exp(neg - mx).sum()
        return {k: float(np.exp(pos[self.owner == i].max() - mx) / (np.exp(pos[self.owner == i].max() - mx) + dn))
                for i, k in enumerate(self.keys)}


class ClipVision(VisionProvider, EmbeddingProvider):
    """Zero-shot image-text model (default SigLIP B/16). SigLIP: confidence = sigmoid(scale*cos+bias) per label.
    Plain CLIP fallback: softmax of the label vs NEGATIVES. Zero-shot, uncalibrated: a ranking signal; tune KEEP on your footage."""
    name = "open-clip"

    def __init__(self, model=None, pretrained=None, kg_path=None):
        import json, open_clip, torch
        from .entities import SEED
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
        self.torch = torch
        model = model or model_name()
        self.model, _, self.pre = open_clip.create_model_and_transforms(model, pretrained=pretrained)
        self.tok = open_clip.get_tokenizer(model); self.model.eval()
        # SigLIP-style models carry a learned bias: sigmoid(scale*cos+bias) is a per-label probability.
        self.sigmoid = hasattr(self.model, "logit_bias")
        self._neg = self.embed_text(NEGATIVES)
        self._aes = self.embed_text([t for pair in AESTHETIC_PAIRS for t in pair]); self._nt = self.embed_text(list(NO_TEXT_PAIR))
        from . import beats as _beats; self._beat_T, self._beat_owner = _beats.text_embeddings(self.embed_text)
        from . import shotsize as _ss; self._size_T, self._size_owner = _ss.text_embeddings(self.embed_text)
        self.labels = _PromptGroup(self, LABEL_PROMPTS)
        self.moderation = _PromptGroup(self, MODERATION_PROMPTS)
        kg = json.load(open(kg_path or SEED, encoding="utf-8"))
        self.deities = _PromptGroup(self, {e["id"]: [e["visual"]] for e in kg["entities"] if e.get("visual")})
        self._heur = HeuristicVision()

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        with self.torch.no_grad():
            e = self.model.encode_text(self.tok(list(texts)))
            return (e / e.norm(dim=-1, keepdim=True)).numpy()

    def embed_image(self, frame: np.ndarray) -> np.ndarray:
        from PIL import Image
        img = self.pre(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0)
        with self.torch.no_grad():
            e = self.model.encode_image(img)
            return (e / e.norm(dim=-1, keepdim=True)).numpy()[0]

    def scores(self, emb: np.ndarray) -> Dict[str, float]:
        return self.labels.score(self, emb)

    def no_text_score(self, frame: np.ndarray) -> float:
        """Only the title-card / graphic score of one frame (analysis.text_track): no labels, no beats."""
        nt = float(self.model.logit_scale.exp()) * (self._nt @ self.embed_image(frame)); return float(nt[0] - nt[1])

    @staticmethod
    def _gate(s: Dict[str, float]) -> Dict[str, float]:
        """Fireworks / effigy burning outranks aarti (both are 'fire'); a person-with-lamp aarti prompt keeps night river aartis."""
        s = dict(s)
        if s.get("fireworks", 0) > s.get("aarti", 0):
            s["aarti"] *= 0.3
        return s

    def analyze(self, frame: np.ndarray) -> VisionResult:
        base = self._heur.analyze(frame)  # faces/brightness/sharpness/saliency stay measured, not guessed
        emb = self.embed_image(frame)
        s = self._gate(self.labels.score(self, emb))
        # the heuristic's colour-based labels are NOT merged (its "lamps" fires on marigolds, "priest" on any face)
        base.labels = {k: v for k, v in s.items() if v >= KEEP}
        base.moderation = {k: v for k, v in self.moderation.score(self, emb).items() if v >= 0.1}
        d = sorted(self.deities.score(self, emb).items(), key=lambda kv: -kv[1])
        if d and d[0][1] >= DEITY_MIN and (len(d) < 2 or d[0][1] - d[1][1] >= DEITY_MARGIN):
            base.deities = {d[0][0]: d[0][1]}
        lg = float(self.model.logit_scale.exp()) * (self._aes @ emb)                  # bias cancels in a pos-neg difference
        base.aesthetic = float(np.mean(lg[0::2] - lg[1::2]))
        nt = float(self.model.logit_scale.exp()) * (self._nt @ emb); base.no_text = float(nt[0] - nt[1])
        from . import beats as _beats; base.beats = _beats.score(emb, self._beat_T, self._beat_owner, float(self.model.logit_scale.exp()))
        from . import shotsize as _ss; base.shot_size = _ss.score(emb, self._size_T, self._size_owner, float(self.model.logit_scale.exp()))
        base.embedding, base.provider = emb.tolist(), self.name
        return base


def _auto() -> VisionProvider:
    try:
        return ClipVision()
    except ImportError:   # extras not installed: degrade to the measurable-only heuristic, loudly
        import logging
        logging.getLogger("aikyam").warning("open_clip not installed; using HeuristicVision (pip install '.[clip]')")
        return HeuristicVision()


register("vision", "auto", _auto)
register("vision", "heuristic", HeuristicVision)
register("vision", "clip", ClipVision)
register("embedding", "clip", ClipVision)
