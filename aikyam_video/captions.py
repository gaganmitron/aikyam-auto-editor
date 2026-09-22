"""Caption cue building + ASS rendering (section 13). Language behaviour lives in data/languages.json."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional
from .safezone import glyphs, zones
from .models import Transcript

_LANGS = json.loads((Path(__file__).parent / "data" / "languages.json").read_text(encoding="utf-8"))


def lang_cfg(lang: str) -> dict:
    return {**_LANGS["default"], **_LANGS.get(lang, {})}


def build_cues(transcript: Transcript, segments: List[dict], mode: str = "sentence", overlap=0.0) -> List[dict]:
    """Cues on the OUTPUT timeline. Each plan segment is laid end-to-end; source words/sentences are clipped to
    the segment and shifted. With a crossfade, segment i+1 starts `overlap` s before segment i ends, so the timeline advances by
    (length - overlap) and cues of a segment are cut at the next segment's start (no two captions stacked).
    Cue = {start, end, text, words:[{start,end,text}]}."""
    cues, offset = [], 0.0
    for si, seg in enumerate(segments):
        a, b = seg["start"], seg["end"]
        if isinstance(transcript, dict):                          # multi-asset plan: each segment speaks with ITS asset's transcript; images are silent
            tr = None if seg.get("kind") == "image" else transcript.get(seg.get("assetId"))
            if tr is None:
                last = si == len(segments) - 1
                offset += (b - a) - ((overlap[si] if si < len(overlap) else 0.0) if isinstance(overlap, (list, tuple)) else overlap)
                continue
        else:
            tr = transcript
        last = si == len(segments) - 1
        ov = (overlap[si] if si < len(overlap) else 0.0) if isinstance(overlap, (list, tuple)) else overlap   # overlap into the NEXT segment
        seg_cues_from = len(cues)
        for s in tr.segments:
            if s.end <= a or s.start >= b:
                continue
            if mode == "word" and s.words:
                ws = [w for w in s.words if w.start >= a - 0.05 and w.end <= b + 0.05 and w.text]
                n = lang_cfg(tr.language)["wordsPerCue"]
                for i in range(0, len(ws), n):
                    g = ws[i:i + n]
                    cues.append({"start": round(g[0].start - a + offset, 3), "end": round(g[-1].end - a + offset, 3),
                                 "text": " ".join(w.text for w in g),
                                 "words": [{"start": round(w.start - a + offset, 3), "end": round(w.end - a + offset, 3),
                                            "text": w.text} for w in g]})
            elif s.text.strip():
                st, en = max(s.start, a), min(s.end, b)
                if en - st >= 0.3:
                    cues.append({"start": round(st - a + offset, 3), "end": round(en - a + offset, 3),
                                 "text": s.text.strip(), "words": []})
        if not last and ov:                      # clip captions that would run into the transition
            limit = offset + (b - a) - ov
            for q in cues[seg_cues_from:]:
                q["end"] = round(min(q["end"], limit), 3)
                for w in q.get("words", []): w["end"] = round(min(w["end"], limit), 3)
            cues[seg_cues_from:] = [q for q in cues[seg_cues_from:] if q["end"] - q["start"] >= 0.2]
        offset += (b - a) - ov
    return cues


def _t(x: float) -> str:
    cs = int(round(x * 100)); h, r = divmod(cs, 360000); m, r = divmod(r, 6000); s, c = divmod(r, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", "\\N")


def _wrap(text: str, width: int) -> str:
    lines, cur = [], ""
    for w in text.split():
        if cur and glyphs(cur) + 1 + glyphs(w) > width:
            lines.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    return "\\N".join(_esc(l) for l in lines + ([cur] if cur else []))


def ass_header(w: int, h: int, styles: List[str]) -> str:
    return ("[Script Info]\nScriptType: v4.00+\nWrapStyle: 2\nScaledBorderAndShadow: yes\n"
            f"PlayResX: {w}\nPlayResY: {h}\n\n[V4+ Styles]\n"
            "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,"
            "StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
            + "\n".join(styles) + "\n\n[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n")


def caption_style(name: str, w: int, h: int, lang: str, cfg: dict) -> str:
    """cfg: font, fontSize (fraction of height), position bottom|center|top, background bool."""
    font = cfg.get("font") or lang_cfg(lang)["font"]
    size = int(h * cfg.get("fontSize", 0.032))
    align = {"bottom": 2, "center": 5, "top": 8}[cfg.get("position", "bottom")]
    border, back = (3, "&H99000000") if cfg.get("background", True) else (1, "&H00000000")
    z = zones(w, h)                                                # portrait: text stays out of the platform UI (top 12%, bottom 22%, sides 8%)
    mv = int(h * z["bottom"]) if align == 2 else int(h * z["top"]) if align == 8 else 0
    return (f"Style: {name},{font},{size},&H00FFFFFF,&H0000D7FF,&H00000000,{back},0,0,0,0,100,100,0,0,{border},"
            f"{max(2, size // 12)},0,{align},{int(w * z['side'])},{int(w * z['side'])},{mv},1")


def fit_chars(lang: str, w: Optional[int], h: Optional[int], cfg: dict) -> int:
    """Characters per line: the language's readable maximum, cut down so the line fits between the side safe-zone margins at this font size."""
    lc = lang_cfg(lang); n = lc["maxCharsPerLine"]
    if not (w and h):
        return n
    size = int(h * cfg.get("fontSize", 0.032)); usable = w * (1 - 2 * zones(w, h)["side"])
    return max(8, min(n, int(usable / (size * lc.get("em", 0.62)))))


def caption_events(cues: List[dict], style: str, lang: str, cfg: dict, w: Optional[int] = None, h: Optional[int] = None) -> List[str]:
    anim = cfg.get("animation", "fade")
    width = fit_chars(lang, w, h, cfg)
    ev = []
    for c in cues:
        if cfg.get("mode") == "word" and c.get("words"):
            # karaoke sweep: \kf durations in centiseconds, words highlight as they are spoken
            parts, prev_end = [], c["start"]
            for w in c["words"]:
                gap = max(0, w["start"] - prev_end)
                d = max(1, int(round((w["end"] - w["start"]) * 100 + gap * 100)))
                parts.append(f"{{\\kf{d}}}{_esc(w['text'])} "); prev_end = w["end"]
            text = "".join(parts).strip()
        else:
            text = _wrap(c["text"], width)
        fx = "{\\fad(120,120)}" if anim == "fade" else "{\\fad(60,60)\\t(0,120,\\fscx108\\fscy108)\\t(120,240,\\fscx100\\fscy100)}" if anim == "pop" else ""
        ev.append(f"Dialogue: 1,{_t(c['start'])},{_t(c['end'])},{style},,0,0,0,,{fx}{text}")
    return ev


def retime_cues(cues: List[dict], max_cps: float = 17.0, min_dur: float = 1.0, gap: float = 0.05, end: Optional[float] = None) -> List[dict]:
    """Reading speed: a cue stays up for max(min_dur, chars/max_cps) when the next cue (or the reel end) leaves room. Never overlaps the next cue."""
    out = [dict(c) for c in cues]
    for i, q in enumerate(out):
        need = max(min_dur, glyphs(q["text"]) / max_cps)
        room = (out[i + 1]["start"] - gap) if i + 1 < len(out) else (end if end is not None else q["end"] + need)
        q["end"] = round(max(q["end"], min(q["start"] + need, room)), 3)
    return out
