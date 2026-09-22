import React from "react";
import { interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import type { Props } from "./types";

/** Sentence or word-level (karaoke highlight) captions from plan.captions.cues (already on the OUTPUT timeline). */
export const Captions: React.FC<{ p: Props }> = ({ p }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const c = p.plan.captions;
  if (!c.enabled || !c.cues?.length) return null;
  const t = frame / fps;
  const cue = c.cues.find((q) => t >= q.start && t < q.end);
  if (!cue) return null;
  const st = c.style ?? {};
  const pos = st.position ?? "bottom";
  const opacity = interpolate(t, [cue.start, cue.start + 0.12, cue.end - 0.12, cue.end], [0, 1, 1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const box: React.CSSProperties = st.background === false ? {} : { background: "rgba(0,0,0,0.6)", padding: "0.15em 0.5em", borderRadius: 8 };
  const words = c.mode === "word" && cue.words?.length ? cue.words : null;
  return (
    <div style={{ position: "absolute", left: p.width * 0.06, right: p.width * 0.06, textAlign: "center", opacity,
      ...(pos === "bottom" ? { bottom: p.height * 0.12 } : pos === "top" ? { top: p.height * 0.05 } : { top: "45%" }),
      fontFamily: `"${p.captionFont}", "Noto Sans", sans-serif`, fontSize: (st.fontSize ?? 0.032) * p.height, color: "#fff", textShadow: "0 0 6px #000" }}>
      <span style={{ ...box, display: "inline" }}>
        {words ? words.map((w, i) => <span key={i} style={{ color: t >= w.start ? "#ffd700" : "#fff" }}>{w.text} </span>) : cue.text}
      </span>
    </div>
  );
};
