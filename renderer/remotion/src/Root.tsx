import React from "react";
import { AbsoluteFill, Composition } from "remotion";
import { Captions } from "./Captions";
import { Overlay } from "./Overlay";
import { Segments } from "./Segments";
import type { Props } from "./types";

const Video: React.FC<Props> = (p) => (
  <AbsoluteFill style={{ backgroundColor: "#000" }}>
    <Segments p={p} />
    <Overlay p={p} />
    <Captions p={p} />
  </AbsoluteFill>
);

const defaults: Props = {
  plan: { durationSeconds: 1, segments: [{ start: 0, end: 1, reason: "x", score: 1 }], captions: { enabled: false, language: "en" }, overlays: {}, transitions: { type: "cut", durationSeconds: 0 } },
  template: { lines: [] }, width: 1080, height: 1920, fps: 30, srcWidth: 1920, srcHeight: 1080, videoFile: "source.mp4", captionFont: "Noto Sans",
};

export const Root: React.FC = () => (
  <Composition id="AikyamVideo" component={Video} durationInFrames={30} fps={30} width={1080} height={1920} defaultProps={defaults}
    // Size and length come from the plan, so ONE composition renders every format/aspect ratio.
    calculateMetadata={({ props }) => ({ durationInFrames: Math.max(1, Math.round(props.plan.durationSeconds * props.fps)), fps: props.fps, width: props.width, height: props.height })} />
);
