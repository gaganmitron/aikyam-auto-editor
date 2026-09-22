import React from "react";
import { Img, interpolate, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import type { Props } from "./types";

// ASS colours are &HAABBGGRR; the template accent is stored that way so both renderers share it.
const accentCss = (a?: string): string | undefined => {
  const m = a?.match(/&H[0-9A-Fa-f]{2}([0-9A-Fa-f]{2})([0-9A-Fa-f]{2})([0-9A-Fa-f]{2})/);
  return m ? `#${m[3]}${m[2]}${m[1]}` : undefined;
};

/** Template text lines (temple / deity / ritual / festival...), the Aikyam logo, all faded in and out. */
export const Overlay: React.FC<{ p: Props }> = ({ p }) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();
  const fade = Math.round(0.4 * fps);
  const opacity = interpolate(frame, [0, fade, durationInFrames - fade, durationInFrames], [0, 1, 1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const accent = accentCss(p.template.accent);
  const ov = p.plan.overlays as Record<string, string | null | undefined>;
  const logoOn = !!p.logoFile && (!p.template.brandOnlyAtEnd || frame > durationInFrames - 2.5 * fps);
  return (
    <div style={{ position: "absolute", inset: 0, fontFamily: `"Noto Sans", "${p.captionFont}", sans-serif`, opacity }}>
      {p.template.lines.map((ln, i) => ov[ln.field] ? (
        <div key={i} style={{ position: "absolute", left: 0, right: 0, top: ln.y * p.height, textAlign: "center", color: ln.bold && accent ? accent : "#fff",
          fontWeight: ln.bold ? 700 : 400, fontSize: ln.size * p.height, textShadow: "0 0 8px #000, 0 0 3px #000",
          ...(ln.box ? { background: "rgba(0,0,0,0.65)", padding: "0.2em 0.6em", width: "fit-content", margin: "0 auto", borderRadius: 8 } : {}) }}>
          {ov[ln.field]}
        </div>) : null)}
      {logoOn && p.template.brand ? <Img src={staticFile(p.logoFile!)} style={{ position: "absolute", right: p.width * 0.03, top: p.height * 0.03, width: p.width * 0.13 }} /> : null}
    </div>
  );
};
