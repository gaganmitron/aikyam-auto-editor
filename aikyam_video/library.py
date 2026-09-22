"""Semantic library over analysed media (Phase 2): scene/video embeddings, semantic search, duplicate detection,
"similar" recommendations. Vectors are SigLIP image embeddings (the same space as text queries).
ponytail: brute-force cosine in numpy (fine to ~1e5 scenes); move to pgvector/an ANN index beyond that."""
from __future__ import annotations
import json, os
from typing import Dict, List, Optional
import numpy as np
from sqlalchemy import delete, select
from .jobs import JobStore, scene_embeddings, video_embeddings

DUP_COS = float(os.environ.get("DUPLICATE_COSINE", "0.985"))


def _vec(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


def index_media(store: JobStore, media_id: str, tenant_id: str, art: str) -> dict:
    """Store scene embeddings + a duration-weighted video embedding; flag near-duplicates within the tenant."""
    emb = json.load(open(os.path.join(art, "embeddings.json")))
    scenes = {s["sceneId"]: s for s in json.load(open(os.path.join(art, "scenes.json")))}
    vis = {v["sceneId"]: v for v in json.load(open(os.path.join(art, "vision.json")))}
    if not emb:
        return {"indexed": 0, "duplicateOf": None}
    with store.e.begin() as c:
        c.execute(delete(scene_embeddings).where(scene_embeddings.c.media_id == media_id))
        acc, wsum = 0, 0.0
        for sid, v in emb.items():
            s = scenes[sid]; vec = np.asarray(v, dtype=np.float32)
            c.execute(scene_embeddings.insert().values(
                media_id=media_id, scene_id=sid, tenant_id=tenant_id, start=s["start"], end=s["end"], vec=vec.tobytes(),
                labels=json.dumps({k: round(x, 2) for k, x in vis[sid]["vision"]["labels"].items()})))
            w = s["end"] - s["start"]; acc = acc + w * vec; wsum += w
    v = (acc / wsum).astype(np.float32); v /= np.linalg.norm(v)
    dur = max(s["end"] for s in scenes.values())
    dup = find_duplicate(store, tenant_id, v, dur, exclude=media_id)
    with store.e.begin() as c:
        c.execute(delete(video_embeddings).where(video_embeddings.c.media_id == media_id))
        c.execute(video_embeddings.insert().values(media_id=media_id, tenant_id=tenant_id, duration=dur, vec=v.tobytes(), duplicate_of=dup))
    return {"indexed": len(emb), "duplicateOf": dup}


def find_duplicate(store: JobStore, tenant_id: str, vec: np.ndarray, duration: float, exclude: str = "") -> Optional[str]:
    """Near-duplicate = cosine >= DUP_COS and duration within 10%. (Catches re-uploads / re-encodes; not partial overlaps.)"""
    with store.e.connect() as c:
        for r in c.execute(select(video_embeddings).where(video_embeddings.c.tenant_id == tenant_id)):
            if r.media_id != exclude and abs(r.duration - duration) <= 0.1 * max(duration, r.duration) and float(_vec(r.vec) @ vec) >= DUP_COS:
                return r.media_id
    return None


def search(store: JobStore, embedder, tenant_id: str, query: str, k: int = 10) -> List[dict]:
    """Text -> best-matching SCENES across the tenant's library."""
    q = embedder.embed_text([query])[0]
    with store.e.connect() as c:
        rows = list(c.execute(select(scene_embeddings).where(scene_embeddings.c.tenant_id == tenant_id)))
    if not rows:
        return []
    sims = np.stack([_vec(r.vec) for r in rows]) @ q
    return [{"mediaId": rows[i].media_id, "sceneId": rows[i].scene_id, "start": rows[i].start, "end": rows[i].end,
             "score": round(float(sims[i]), 4), "labels": json.loads(rows[i].labels)} for i in np.argsort(-sims)[:k]]


def similar(store: JobStore, media_id: str, tenant_id: str, k: int = 10) -> List[dict]:
    """Recommendation hook: nearest videos in the tenant (feed rankers can call this or pull vectors via `video_vector`)."""
    with store.e.connect() as c:
        me = c.execute(select(video_embeddings).where(video_embeddings.c.media_id == media_id)).first()
        if me is None:
            return []
        rows = [r for r in c.execute(select(video_embeddings).where(video_embeddings.c.tenant_id == tenant_id)) if r.media_id != media_id]
    v = _vec(me.vec)
    scored = sorted(((float(_vec(r.vec) @ v), r.media_id) for r in rows), reverse=True)[:k]
    return [{"mediaId": m, "similarity": round(s, 4)} for s, m in scored]


def video_vector(store: JobStore, media_id: str) -> Optional[List[float]]:
    with store.e.connect() as c:
        r = c.execute(select(video_embeddings).where(video_embeddings.c.media_id == media_id)).first()
    return None if r is None else _vec(r.vec).tolist()
