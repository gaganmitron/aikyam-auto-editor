// Mirrors aikyam_video/data/edit_plan.schema.json (the renderer-independent contract).
export type Segment = { start: number; end: number; reason: string; score: number; subjectX?: number; subjectPath?: [number, number][] };
export type Cue = { start: number; end: number; text: string; words?: { start: number; end: number; text: string }[] };
export type Plan = {
  durationSeconds: number;
  segments: Segment[];
  captions: { enabled: boolean; language: string; mode?: "sentence" | "word"; style?: { fontSize?: number; position?: "bottom" | "center" | "top"; background?: boolean; animation?: string }; cues?: Cue[] };
  overlays: { template?: string; temple?: string | null; deity?: string | null; ritual?: string | null; festival?: string | null; location?: string | null };
  transitions: { type: "cut" | "fade" | "crossfade"; durationSeconds: number };
};
export type Template = { lines: { field: string; y: number; size: number; bold?: boolean; box?: boolean }[]; brand?: boolean; brandOnlyAtEnd?: boolean; accent?: string };
export type Props = {
  plan: Plan; template: Template; width: number; height: number; fps: number;
  srcWidth: number; srcHeight: number; videoFile: string; logoFile?: string; captionFont: string;
};
