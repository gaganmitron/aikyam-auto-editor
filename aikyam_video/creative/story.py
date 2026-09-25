"""Story planner: choose WHICH shots, in WHAT narrative order, for how long. Deterministic beam search; no randomness, ties broken by id.

Objective for a sequence of (role, shot) pairs in arc order:
    sum( affinity(role, shot) * quality(shot) * moment_rank(shot) )        how well each shot serves its role and how good it is
  - redundancy_penalty * sum( max(0, sim(i, j) - s0) )                       near-duplicate shots waste the reel (s0 = pool's 75th percentile similarity)
  - skip_penalty * importance(role)  for every role left out                 a story without its reveal/climax is weaker
  + chronology_bonus * (fraction of consecutive pairs in source order)       temple rituals have a natural order; break it only when it pays
  + coverage * (gain in how well the picked shots stand in for the WHOLE pool)  facility-location coverage; opt-in (0 = off), EXP-014
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from .model import Clip, ROLES, Shot, Timeline
from .pacing import PacingProfile, shot_length, target_duration
from .roles import CFG, affinities, aggregates
from .edges import motion_differs, snap_window
from .shots import HOP, Population, best_window, carve, has_dead

S = CFG["story"]
MIN_REEL_S = 15.0       # a Reel is at least this long whenever the footage can supply it
HOOK_MAX = 3.5          # seconds: longest opening clip when the opening is a hook
REPEAT = ("BUILDUP", "RITUAL", "REVEAL")   # roles a long reel may use twice
DUP_SIM = 0.95          # image-embedding cosine at which two shots are the same footage twice: never used together (soft penalty covers the merely similar)


def _overlap(a, b) -> float:
    """Seconds two source windows share. Windows are (start, end) or (asset, start, end): windows of different assets never overlap, and images never conflict by time."""
    if len(a) == 3 or len(b) == 3:
        if len(a) != 3 or len(b) != 3 or a[0] != b[0]:
            return 0.0
        a, b = a[1:], b[1:]
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _win(s: Shot):
    return (s.asset_id, s.start, s.end)


def _clash(a: Shot, b: Shot) -> bool:
    """Same footage twice: the same window of one video asset, or the same image."""
    if a.asset_id != b.asset_id:
        return False
    return a.kind == "image" or _overlap((a.start, a.end), (b.start, b.end)) > 0.5


def similarity(a: Shot, b: Shot) -> float:
    if a.embedding and b.embedding:
        return float(np.dot(a.embedding, b.embedding))
    ks = sorted(set(a.labels) | set(b.labels))                                   # fallback: label-vector cosine
    if not ks:
        return 0.0
    x, y = np.array([a.labels.get(k, 0.0) for k in ks]), np.array([b.labels.get(k, 0.0) for k in ks])
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))


def is_duplicate(a: Shot, b: Shot, motion_dedupe: bool = True) -> bool:
    """Same footage twice: near-identical appearance (SigLIP >= DUP_SIM) UNLESS the two clearly move differently (hypecut motion signature): a locked camera on one venue looks
    the same in every shot, but a different happening in it is a different moment."""
    if similarity(a, b) < DUP_SIM:
        return False
    return not (motion_dedupe and motion_differs(a.slots.motion_sig if a.slots else None, b.slots.motion_sig if b.slots else None))


def pool_similarity(pool: Sequence[Shot], q: float) -> float:
    v = [similarity(a, b) for i, a in enumerate(pool) for b in pool[i + 1:]]
    return float(np.percentile(v, q)) if v else 1.0


def plan_story(shots: Sequence[Shot], pop: Population, profile: PacingProfile, source_duration: float, target_s: Optional[float] = None,
               exclude: Sequence[tuple] = (), avoid: Sequence[Shot] = (), snap=None, max_clips: Optional[int] = None, director=None, notes: Optional[Dict[str, str]] = None, forced_order: Optional[Sequence[str]] = None, edger=None, motion_dedupe: bool = True, hook_first: bool = False, coverage: Optional[float] = None) -> Timeline:
    pool = sorted([s for s in shots if s.slots and not any(_overlap(_win(s), e) > 0.5 or (s.kind == "image" and e[0] == s.asset_id) for e in exclude)], key=lambda s: s.id)
    dec: List[dict] = []
    if not pool:
        return Timeline([], profile.name, [], decisions=[{"type": "empty", "why": "no usable shots"}])
    agg = {s.id: aggregates(s, pop, s.position if s.position is not None else (s.start + s.end) / 2 / max(source_duration, 1e-9)) for s in pool}
    aff = {s.id: affinities(s, pop, agg[s.id]["position"], profile.opening) for s in pool}
    order = sorted(pool, key=lambda s: (s.score, s.id)); mrank = {s.id: (i + 1) / len(order) for i, s in enumerate(order)}
    val = lambda role, s: aff[s.id][role] * (0.5 + 0.5 * agg[s.id]["quality"]) * (0.6 + 0.4 * mrank[s.id])
    single_asset = len({s.asset_id for s in pool if s.kind == "video"}) <= 1     # EXP-002b: a same-asset chronology bonus big enough to fix single-asset ordering also outweighs cross-asset/still selection in a mixed pool (proven: 0.15+ drops multi-asset from 3 assets+1 still to 2 assets+0 stills) -- so it only gets the strong, symmetric treatment when there's just one asset to order; a multi-asset pool keeps the exact original (weak, forward-only, /n_t) formula, unchanged
    val_any = lambda s: max(val(r, s) for r in ROLES)
    s0 = pool_similarity(pool, 75)
    red = lambda a, b: max(0.0, similarity(a, b) - s0) / max(1.0 - s0, 1e-6)
    min_aff = {r: float(np.percentile([aff[s.id][r] for s in pool], S["min_affinity_percentile"])) for r in ROLES}

    energy = float(np.average([agg[s.id]["energy"] for s in pool], weights=[max(s.score, 1e-3) for s in pool]))
    Lbar = shot_length(profile, energy)
    avail = sum(s.length for s in pool)
    T = target_s or target_duration(profile, avail)
    n_t = int(np.clip(round(T / Lbar), 1 if len(pool) < 3 else 3, min(7, len(pool))))
    n_lo, n_hi = max(1, n_t - 1), min(len(pool), n_t + 1, max_clips or 99)
    max_img = max(1, int(np.ceil(n_hi / 3)))                                             # stills season a reel, they do not carry it
    n_lo = min(n_lo, n_hi)
    dec.append({"type": "plan", "profile": profile.name, "targetSeconds": round(T, 1), "nominalShotSeconds": round(Lbar, 2), "clips": [n_lo, n_hi],
                "poolShots": len(pool), "similarityFloor": round(s0, 3)})

    # ---- beam search over roles in arc order
    beam: List[Tuple[float, List[Tuple[str, Shot]]]] = [(0.0, [])]
    hook_shot = None; pool0 = list(pool)                                                    # the director and forced orders see the ORIGINAL shots, not the carved pieces
    if profile.opening == "hook" and hook_first and not forced_order and len(pool) >= 2:                 # hook-FIRST is a rule, not a preference: the strongest opener starts the reel, the story follows
        hook_shot = max(pool, key=lambda x: (val("OPENING", x), x.id)); hook_val = val("OPENING", hook_shot)
        if hook_shot.kind == "video" and hook_shot.length > HOOK_MAX + 1.0:                # a short hook must not swallow the whole shot: its window becomes the opener, the rest stays usable
            ha, hb, _ = best_window(hook_shot, pop, HOOK_MAX, snap); parts = carve(hook_shot, ha, hb)
            for r in parts:
                if r.id != hook_shot.id:
                    agg[r.id], aff[r.id], mrank[r.id] = agg[hook_shot.id], aff[hook_shot.id], mrank[hook_shot.id]; pool.append(r)
            opener = parts[0]; agg[opener.id], aff[opener.id], mrank[opener.id] = agg[hook_shot.id], aff[hook_shot.id], mrank[hook_shot.id]; pool = [x for x in pool if x is not hook_shot]
            pool.append(opener); hook_shot = opener
        beam = [(hook_val, [("OPENING", hook_shot)])]
        dec.append({"type": "hook", "shot": hook_shot.id, "score": round(val("OPENING", hook_shot), 3), "labels": {k: round(v, 2) for k, v in sorted(hook_shot.labels.items(), key=lambda kv: -kv[1])[:3]}})
    cov_w = S.get("coverage", 0.0) if coverage is None else coverage                       # per-call override (Options.experimental_selection) of the config default
    if cov_w:                                                                                # R[i, j]: how much shot j stands in for shot i, beyond the pool's typical similarity (same transform as `red`)
        ids = {s.id: i for i, s in enumerate(pool)}
        if all(s.embedding for s in pool):
            Emb = np.array([s.embedding for s in pool], float); R = np.maximum(0.0, Emb @ Emb.T - s0) / max(1.0 - s0, 1e-6)
        else:
            R = np.array([[red(a, b) for b in pool] for a in pool])
        Wt = np.array([max(s.length, 1e-3) * max(s.score, 1e-3) for s in pool]); Wt = Wt / Wt.sum()      # long, well-scored footage counts more
    for ri, role in enumerate(ROLES):
        if hook_shot is not None and role == "OPENING":
            continue
        nxt = []
        for score, chosen in beam:
            if cov_w:
                have = [ids[c.id] for _, c in chosen if c.id in ids]
                cur = R[:, have].max(axis=1) if have else np.zeros(len(pool))
            nxt.append((score - S["skip_penalty"] * CFG["importance"][role], chosen))       # any role may be skipped (thin pools): the skip penalty and the clip-count preference do the steering
            if len(chosen) >= n_hi:
                continue
            for s in pool:
                if s.kind == "image" and (sum(c.kind == "image" for _, c in chosen) >= max_img or (chosen and chosen[-1][1].kind == "image")):
                    continue                                                              # cap on stills, and never two stills back to back
                if any(s.id == c.id or _clash(s, c) or is_duplicate(s, c, motion_dedupe) for _, c in chosen) \
                        or any(is_duplicate(s, a, motion_dedupe) for a in avoid) or aff[s.id][role] < min_aff[role]:
                    continue
                inc = val(role, s) - S["redundancy_penalty"] * sum(red(s, c) for _, c in chosen) - S["redundancy_penalty"] * sum(red(s, a) for a in avoid) * 0.5
                if inc <= -S["skip_penalty"] * CFG["importance"][role]:
                    continue                                                                 # EXP-002: too weak to beat skipping on its OWN merit; chronology only ranks candidates that already clear this bar, never rescues one that doesn't
                prev_same = next((c for _, c in reversed(chosen) if c.asset_id == s.asset_id and c.kind == "video"), None)
                if prev_same is not None and s.kind == "video":
                    if single_asset:                                                         # EXP-001/002b: reward forward-in-source, PENALIZE backward equally (was bonus-only + divided by n_t, ~6x too weak to compete) -- single-asset pools only, see note above
                        inc += S["chronology_bonus"] if s.start > prev_same.start else -S["chronology_bonus"]
                    elif s.start > prev_same.start:                                           # multi-asset: byte-identical to the pre-EXP-001 original -- forward-only, divided by n_t, its own pinned constant (not coupled to the single-asset value)
                        inc += S["chronology_bonus_multi_asset"] / max(n_t, 1)
                if chosen and s.beats and chosen[-1][1].beats:                                  # same beat back to back = repetition even when the pictures differ
                    inc -= S.get("beat_repeat_penalty", 0.0) * sum(s.beats.get(b, 0.0) * chosen[-1][1].beats.get(b, 0.0) for b in s.beats)
                if cov_w and s.id in ids:                                                    # after the quality gate, like chronology: ranks candidates that already clear the bar, never rescues one that doesn't
                    inc += cov_w * float((Wt * np.maximum(R[:, ids[s.id]] - cur, 0.0)).sum())
                nxt.append((score + inc, chosen + [(role, s)]))
        nxt.sort(key=lambda x: (-round(x[0], 9), tuple(s.id for _, s in x[1])))
        beam = nxt[:S["beam"]]
    ok = [b for b in beam if n_lo <= len(b[1]) <= n_hi] or [b for b in beam if b[1]]
    if not ok:                                                                            # nothing cleared the affinity floors: never fail while moments exist
        top = max(pool, key=lambda x: (val_any(x), x.id))
        role = max(ROLES, key=lambda r: aff[top.id][r])
        ok = [(val(role, top), [(role, top)])]
        dec.append({"type": "fallback", "why": f"no shot cleared the role affinity floors: using the single strongest shot {top.id} as {role}"})
    best_score, best = ok[0]
    seconds: Optional[List[float]] = None; directed = False
    if forced_order:                                                                      # the person fixed the order: best window of each listed asset, roles spread along the arc
        picked: List[Shot] = []
        for aid in forced_order:
            cands = [s for s in pool if s.asset_id == aid and not any(s.id == c.id or _clash(s, c) for c in picked)]
            if not cands:
                raise ValueError(f"order: nothing usable left in asset {aid!r}")
            picked.append(max(cands, key=lambda x: (val_any(x), x.id)))
        n = len(picked); best = [(ROLES[int(round(i * (len(ROLES) - 1) / (n - 1)))] if n > 1 else "RITUAL", s) for i, s in enumerate(picked)]
        dec.append({"type": "order", "assets": list(forced_order), "shots": [s.id for s in picked]}); director = None
    if director is not None:                                                              # LLM director: proposes WHAT/ORDER/LENGTH; every proposal is validated, any problem -> the beam result above
        try:
            info = [{"shotId": s.id, "kind": s.kind, "source": s.asset_id, "seconds": round(s.length, 1), "labels": {k: round(v, 2) for k, v in sorted(s.labels.items(), key=lambda kv: -kv[1])[:4]},
                     "entities": s.entities[:3], "said": (notes or {}).get(s.id, "")[:160], "quality": round(agg[s.id]["quality"], 2), "energy": round(agg[s.id]["energy"], 2),
                     "bestRoles": [r for r, _ in sorted(aff[s.id].items(), key=lambda kv: -kv[1])[:2]]} for s in pool0]
            prop = director(info, {"name": profile.name, "minShot": profile.min_shot, "maxShot": profile.max_shot, "opening": profile.opening}, T)
            byid = {s.id: s for s in pool0}; picked: List[Tuple[str, Shot]] = []; secs: List[float] = []
            for role, sid, sec, why in prop["clips"]:
                s = byid.get(sid)
                if s is None or role not in ROLES:
                    raise ValueError(f"unknown shot/role {sid!r}/{role!r}")
                if any(s.id == c.id or _clash(s, c) or is_duplicate(s, c, motion_dedupe) for _, c in picked) or any(is_duplicate(s, a, motion_dedupe) for a in avoid):
                    raise ValueError(f"shot {sid} repeats footage")
                if s.kind == "image" and (sum(c.kind == "image" for _, c in picked) >= max_img or (picked and picked[-1][1].kind == "image")):
                    raise ValueError("too many / adjacent stills")
                picked.append((role, s)); secs.append(sec)
            if not (1 <= len(picked) <= min(7, max_clips or 99)):
                raise ValueError(f"{len(picked)} clips proposed")
            best, seconds = picked, secs; directed = True
            dec.append({"type": "director", "model": prop.get("model"), "title": prop.get("title"), "clips": [{"shot": s.id, "role": r, "seconds": round(x, 1)} for (r, s), x in zip(best, secs)],
                        "why": [w for *_, w in prop["clips"]]})
        except Exception as e:                                                            # noqa: BLE001 -- anything wrong with the proposal: keep the deterministic story
            dec.append({"type": "director-fallback", "error": str(e)[:200]})

    # ---- clip lengths: role share of the nominal shot length, scaled to T (+ the overlap the transitions will consume), clamped, redistributed
    est_overlap = (len(best) - 1) * profile.base_blend * (1 - profile.cut_bias)
    want = np.array([CFG["length_share"][r] for r, _ in best]) * Lbar
    want = want * (T + est_overlap) / want.sum()
    lo = np.array([min(profile.min_shot, s.length) for _, s in best]); hi = np.array([min(profile.max_shot, s.length) for _, s in best])
    if profile.opening == "hook" and best and best[0][0] == "OPENING":                    # a hook is short: the first impression lands in ~3 s, then the story moves on
        hi[0] = min(hi[0], HOOK_MAX); lo[0] = min(lo[0], hi[0])
    L = np.clip(np.asarray(seconds, float) if seconds else want, lo, hi)                # a director's lengths are kept (clamped); the deterministic ones are rescaled to T below
    for _ in range(0 if seconds else 4):
        gap = (T + est_overlap) - L.sum(); room = (hi - L) if gap > 0 else (L - lo)
        if abs(gap) < 0.05 or room.sum() < 1e-6:
            break
        L = L + np.sign(gap) * np.minimum(room, abs(gap) * room / room.sum())

    clips: List[Clip] = []
    for ci, ((role, s), length) in enumerate(zip(best, L)):
        a, b, q = best_window(s, pop, float(length), snap, lit=(role == "OPENING"))
        a0, b0, locked = a, b, False
        if edger is not None and s.kind == "video":                                            # real cuts / pauses beat the fixed slot grid
            a1, b1, einfo = edger(s, a, b, max(1.5, 0.75 * (b - a)), min(profile.max_shot, 1.25 * (b - a)), first=ci == 0, last=ci == len(best) - 1)
            if has_dead(s, pop, a1, b1) and not has_dead(s, pop, a, b):        # a snap must not pull an edge back onto dead footage (the blocked-lens cut)
                einfo = {}
            else:
                a, b = a1, b1
            if einfo:
                dec.append({"type": "edge", "shot": s.id, "moved": {k: {"seconds": v[0], "to": v[1]} for k, v in einfo.items()}}); locked = "end" in einfo and einfo["end"][1] in ("cut", "phrase")
        clips.append(Clip(s, a, b, role, locked_end=locked, why=f"{role.lower()}: affinity {aff[s.id][role]:.2f}, quality {q:.2f}, labels {sorted(k for k, v in s.labels.items() if v >= 0.5)[:3]}"))
        dec.append({"type": "select", "role": role, "shot": s.id, "source": [round(a, 2), round(b, 2)], "affinity": round(aff[s.id][role], 3),
                    "quality": round(q, 3), "labels": {k: round(v, 2) for k, v in sorted(s.labels.items(), key=lambda kv: -kv[1])[:3]}})
    def resolve_overlaps():                                           # windows lie inside their own shots, which overlap <= 0.5 s (and edge snapping may move an edge a little): resolve any residue
        for i in range(len(clips)):
            for j in range(i + 1, len(clips)):
                if clips[i].shot.asset_id != clips[j].shot.asset_id or clips[i].shot.kind == "image":
                    continue
                ov = _overlap((clips[i].start, clips[i].end), (clips[j].start, clips[j].end))
                if ov > 0:
                    loser = clips[i] if clips[i].shot.score < clips[j].shot.score else clips[j]
                    if loser.start < (clips[j] if loser is clips[i] else clips[i]).start: loser.end = round(loser.end - ov, 3)
                    else: loser.start = round(loser.start + ov, 3)

    resolve_overlaps()
    clips = [c for c in clips if c.length >= 1.0]
    # ---- a Reel is >= MIN_REEL_S if the footage allows it: the duplicate rules may have left too little (a repetitive ritual in one take). Add the best UNUSED, non-overlapping footage,
    #      repeats of the movement allowed, and say so in the plan.
    est_ov = profile.base_blend * (1 - profile.cut_bias)
    total = lambda cs: sum(c.length for c in cs) - est_ov * max(0, len(cs) - 1)
    floor = max(MIN_REEL_S, T - 2.0)                                                      # defend the REAL target (explicit, or the pacing profile's own pick) -- the six roles alone cannot fill it, so the middle of the arc may repeat
    if forced_order is None and not directed:
        used = {c.shot.id for c in clips}
        free_pool = lambda: [x for x in pool if x.id not in used and x.kind == "video" and not any(_clash(x, c.shot) for c in clips)]
        while total(clips) < floor:
            cands = free_pool()
            if not cands:
                break
            fresh = [x for x in cands if not any(is_duplicate(x, c.shot, motion_dedupe) for c in clips)]     # prefer footage that isn't a near-duplicate of anything already picked
            free = [r for r in ROLES if r not in {c.role for c in clips}] or list(REPEAT)
            if not free or len(clips) >= min(9 if target_s else 7, max_clips or 99):
                break

            def beat_overlap(x):                                                                             # how much x would repeat the beat of the clips it would sit between
                if not x.beats: return 0.0
                ri = ROLES.index(max(free, key=lambda r: aff[x.id][r]))
                before = [c for c in clips if ROLES.index(c.role) <= ri]; after = [c for c in clips if ROLES.index(c.role) > ri]
                nb = ([max(before, key=lambda c: ROLES.index(c.role))] if before else []) + ([min(after, key=lambda c: ROLES.index(c.role))] if after else [])
                return max([sum(x.beats.get(b, 0.0) * c.shot.beats.get(b, 0.0) for b in x.beats) for c in nb if c.shot.beats] or [0.0])
            bp = S.get("beat_repeat_penalty", 0.0)
            cand = max(fresh or cands, key=lambda x: (val_any(x) - bp * beat_overlap(x), x.id))              # only reach for a duplicate (repeat) when nothing fresh is left
            if bp and beat_overlap(cand) > 0.6 and total(clips) >= floor - 3.0:                              # the only footage left repeats the neighbouring beat and the reel is nearly long enough: stop
                break
            role = max(free, key=lambda r: aff[cand.id][r]); L2 = float(min(profile.max_shot, cand.length, max(profile.min_shot, floor - total(clips) + est_ov)))
            a2, b2, q2 = best_window(cand, pop, L2, snap); before = total(clips)
            clips.append(Clip(cand, a2, b2, role, why=f"fill: the reel was {before:.1f}s, under {floor:.0f}s")); used.add(cand.id)
            clips.sort(key=lambda c: ROLES.index(c.role)); dec.append({"type": "fill", "shot": cand.id, "role": role, "seconds": round(b2 - a2, 1), "why": f"reel would be {before:.1f}s, under {floor:.0f}s"})
    resolve_overlaps(); clips = [c for c in clips if c.length >= 1.0]                                          # the top-up can add a clip that touches an edge-snapped neighbour of the same take
    asc = len({c.shot.asset_id for c in clips}) == 1 and all(clips[k].start < clips[k + 1].start for k in range(len(clips) - 1))
    return Timeline(clips, profile.name, [c.role for c in clips], "source" if asc else "narrative", dec, float(best_score), {"sim_floor": s0, "T": T})


def plan_stories(shots: Sequence[Shot], pop: Population, profile: PacingProfile, source_duration: float, k: int = 1, target_s: Optional[float] = None,
                 snap=None, min_clips: int = 2, director=None, notes=None, order=None, edger=None, motion_dedupe: bool = True, hook_first: bool = False, coverage: Optional[float] = None) -> List[Timeline]:
    """Up to k different reels from ONE source: each story excludes the footage already used and avoids look-alike shots."""
    out: List[Timeline] = []; used: List[Tuple[float, float]] = []; avoid: List[Shot] = []
    cap = None if k <= 1 else max(2, len([s for s in shots if s.slots]) // k)      # share the footage fairly: reel 1 must not eat the pool
    for _ in range(k):
        t = plan_story(shots, pop, profile, source_duration, target_s, used, avoid, snap, cap, director, notes, order if not out else None, edger, motion_dedupe, hook_first, coverage)
        if len(t.clips) < min_clips and out:
            break
        if not t.clips:
            break
        out.append(t); used += [(c.shot.asset_id, c.shot.start, c.shot.end) for c in t.clips]; avoid += [c.shot for c in t.clips]
    return out
