/** Types for the parts of the .wreel 1.x manifest the player uses (spec/wreel.md). */

import type { Quat, Vec3 } from "../math";

export type ProvenanceClass = "observed" | "inferred" | "guessed" | "retrieved" | "generated";
export const PROVENANCE_CLASSES: readonly ProvenanceClass[] = [
  "observed",
  "inferred",
  "guessed",
  "retrieved",
  "generated",
];

export interface Lod {
  level: number;
  path: string;
  count: number;
}

export interface GeometryLayer {
  id: string;
  kind: "points" | "gaussians";
  /** points only */
  point_size?: number;
  lods: Lod[];
}

export interface Keyframe {
  t: number;
  position: Vec3;
  rotation: Quat;
  fov_deg: number;
}

export interface Camera {
  id: string;
  name: string;
  kind: "director" | "alternative";
  keyframes: Keyframe[];
}

export interface Actor {
  id: string;
  type: "person" | "animal" | "vehicle" | "object";
  label: string;
  character_id?: string;
  geometry: GeometryLayer;
  eye_height?: number;
  track: { path: string };
}

export type PovMode = "directors-cut" | "first-person" | "third-person" | "overhead" | "free" | "interactive";

export interface PovPreset {
  id: string;
  name: string;
  mode: PovMode;
  camera_id?: string;
  actor_id?: string;
  offset?: Vec3;
  height?: number;
  fov_deg?: number;
  comfort?: boolean;
}

export interface Provenance {
  ai_label: string;
  notice: string;
  tool: string;
  tool_version: string;
  created_at: string;
  source: { kind: string; sha256?: string };
  identity_policy: string;
  rights: { upload_rights: string; license?: string; takedown: { status: string; contact?: string } };
}

export interface Rule {
  id: string;
  trigger: { type: "time" | "proximity" | "view_mode" | "select"; t?: number; actor_id?: string; radius?: number; mode?: string };
  action: { type: "switch_pov" | "set_tension" | "play_event" | "branch"; preset_id?: string; value?: number };
}

export interface FileEntry {
  sha256: string;
  bytes: number;
  media_type: string;
}

export interface Manifest {
  format: "wreel";
  format_version: string;
  title: string;
  duration: number;
  provenance: Provenance;
  scene: {
    bounds: { min: Vec3; max: Vec3 };
    background: Vec3;
    layers: GeometryLayer[];
  };
  actors: Actor[];
  cameras: Camera[];
  pov_presets: PovPreset[];
  regions: { id: string; class: ProvenanceClass; confidence: number }[];
  audio: {
    tracks: AudioTrack[];
    events: AudioEvent[];
    mood: { t: number; tension: number }[];
    mix?: { ducking?: boolean; loudness_target_lufs?: number };
  };
  interaction: { rules: Rule[] };
  files: Record<string, FileEntry>;
}

export interface Track {
  times: number[];
  positions: Vec3[];
  rotations: Quat[];
}

/** Spec §11 audio track: one stem per layer (dialogue, foley, ambience, effects, music, description). */
export interface AudioTrack {
  id: string;
  layer: string;
  path: string;
  media_type: string;
  start: number;
  license: string;
  positional?: boolean;
  actor_id?: string;
}

/** Spec §11 sound event; `attributes.clip` names the generated sound (Phase 5 engine). */
export interface AudioEvent {
  t: number;
  type: string;
  actor_id?: string;
  position?: [number, number, number];
  attributes?: Record<string, string | number | boolean>;
}
