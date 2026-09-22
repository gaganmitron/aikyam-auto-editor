import React from "react";
import { OffthreadVideo, Sequence, interpolate, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import type { Props, Segment } from "./types";

/** x (0..1) of the tracked subject at local time t seconds: linear interpolation of the plan's subjectPath. */
const subjectAt = (s: Segment, t: number): number => {
  const p = s.subjectPath;
  if (!p || p.length === 0) return s.subjectX ?? 0.5;
  if (p.length === 1) return p[0][1];
  return interpolate(t, p.map((k) => k[0]), p.map((k) => k[1]), { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
};

const Clip: React.FC<{ seg: Segment; p: Props; idx: number; last: boolean; overlapFrames: number }> = ({ seg, p, idx, last, overlapFrames }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  // "cover" crop that keeps the subject centred: object-position X% such that subject lands at the container centre.
  const scaledW = (p.srcWidth / p.srcHeight) * p.height >= p.width ? (p.srcWidth / p.srcHeight) * p.height : p.width;
  const overflow = Math.max(scaledW - p.width, 1e-6);
  const s = subjectAt(seg, frame / fps);
  const x = Math.min(Math.max((s * scaledW - p.width / 2) / overflow, 0), 1) * 100;
  const fade = p.plan.transitions.type === "fade" ? p.plan.transitions.durationSeconds * fps : 0;
  const len = (seg.end - seg.start) * fps;
  let opacity = fade ? Math.min(1, frame / fade, (len - frame) / fade) : 1;
  // crossfade: this clip sits on top of the previous one (overlapping Sequences) and fades IN; the previous stays underneath.
  if (overlapFrames && idx > 0) opacity = Math.min(1, frame / overlapFrames);
  const volume = (f: number) => overlapFrames ? Math.min(idx > 0 ? f / overlapFrames : 1, !last ? (len - f) / overlapFrames : 1) : 1;
  return (
    <OffthreadVideo src={staticFile(p.videoFile)} startFrom={Math.round(seg.start * fps)} volume={volume}
      style={{ width: "100%", height: "100%", objectFit: "cover", objectPosition: `${x}% 50%`, opacity }} />
  );
};

export const Segments: React.FC<{ p: Props }> = ({ p }) => {
  const { fps } = useVideoConfig();
  const n = p.plan.segments.length;
  const ov = p.plan.transitions.type === "crossfade" && n > 1 ? p.plan.transitions.durationSeconds : 0;   // seconds of overlap
  let at = 0;
  return (
    <>
      {p.plan.segments.map((seg, i) => {
        const from = Math.round(at * fps), dur = Math.max(1, Math.round((seg.end - seg.start) * fps));
        at += seg.end - seg.start - ov;      // next segment starts `ov` before this one ends (same rule as the plan/captions)
        return (<Sequence key={i} from={from} durationInFrames={dur}><Clip seg={seg} p={p} idx={i} last={i === n - 1} overlapFrames={Math.round(ov * fps)} /></Sequence>);
      })}
    </>
  );
};
