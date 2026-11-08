/** Camera rigs for the viewer menu (spec/wreel.md §9). */

import {
  add,
  clamp,
  lerp3,
  lookRotation,
  quatFromYawPitch,
  rotate,
  slerp,
  smoothing,
  yawOf,
  type Quat,
  type Vec3,
} from "./math";
import { sampleCamera, sampleTrack } from "./wreel/track";
import type { Actor, Manifest, PovPreset, Track } from "./wreel/types";

export interface CameraState {
  eye: Vec3;
  rotation: Quat;
  fov: number;
  /** Actor not drawn in this view (the one whose eyes we are looking through). */
  hiddenActor?: string;
}

export interface FreeInput {
  /** Accumulated pointer-drag deltas in pixels since the last update. */
  dragX: number;
  dragY: number;
  /** Accumulated wheel delta. */
  wheel: number;
  /** Keys currently held (KeyboardEvent.code). */
  keys: Set<string>;
}

const TRANSITION_SECONDS = 0.5;
const FOLLOW_RATE = 4;
const COMFORT_TURN_RATE = 3;
const FREE_SPEED = 5; // m/s
const LOOK_SPEED = 0.004; // rad/pixel

export class CameraDirector {
  preset: PovPreset;
  private state: CameraState;
  private from: CameraState | null = null;
  private blend = 1;
  private freeYaw = 0;
  private freePitch = 0;
  private freeEye: Vec3 = [0, 2, 10];
  private followEye: Vec3 | null = null;
  private fpRotation: Quat | null = null;

  constructor(
    private readonly manifest: Manifest,
    private readonly tracks: Map<string, Track>,
    initial: PovPreset,
  ) {
    this.preset = initial;
    this.state = { eye: [0, 2, 10], rotation: [0, 0, 0, 1], fov: 50 };
  }

  /** Current camera state (for click-to-select actor projection). */
  current(): CameraState | null {
    return this.state;
  }

  setPreset(preset: PovPreset): void {
    if (preset.id === this.preset.id) return;
    this.from = { ...this.state };
    this.blend = 0;
    this.preset = preset;
    this.followEye = null;
    this.fpRotation = null;
    if (preset.mode === "free" || preset.mode === "interactive") {
      // Start the free camera where the viewer already is.
      this.freeEye = [...this.state.eye];
      this.freeYaw = yawOf(this.state.rotation);
      const f = rotate(this.state.rotation, [0, 0, -1]);
      this.freePitch = Math.asin(clamp(f[1], -1, 1));
    }
  }

  update(time: number, dt: number, input: FreeInput): CameraState {
    const target = this.compute(time, dt, input);
    if (this.from && this.blend < 1) {
      this.blend = Math.min(1, this.blend + dt / TRANSITION_SECONDS);
      const s = this.blend * this.blend * (3 - 2 * this.blend);
      this.state = {
        eye: lerp3(this.from.eye, target.eye, s),
        rotation: slerp(this.from.rotation, target.rotation, s),
        fov: this.from.fov + (target.fov - this.from.fov) * s,
        ...(target.hiddenActor ? { hiddenActor: target.hiddenActor } : {}),
      };
    } else {
      this.state = target;
    }
    return this.state;
  }

  private actor(id: string | undefined): { actor: Actor; track: Track } | null {
    const actor = this.manifest.actors.find((a) => a.id === id);
    const track = id ? this.tracks.get(id) : undefined;
    return actor && track ? { actor, track } : null;
  }

  private compute(time: number, dt: number, input: FreeInput): CameraState {
    const p = this.preset;
    switch (p.mode) {
      case "directors-cut": {
        const cam = this.manifest.cameras.find((c) => c.id === p.camera_id) ?? this.manifest.cameras[0];
        if (!cam) return this.free(dt, input, p.fov_deg);
        const s = sampleCamera(cam.keyframes, time);
        return { eye: s.position, rotation: s.rotation, fov: s.fov };
      }
      case "third-person": {
        const a = this.actor(p.actor_id);
        if (!a) return this.free(dt, input, p.fov_deg);
        const pose = sampleTrack(a.track, time);
        const heading = quatFromYawPitch(yawOf(pose.rotation), 0);
        const desired = add(pose.position, rotate(heading, p.offset ?? [0, 2, 4]));
        this.followEye = this.followEye ? lerp3(this.followEye, desired, smoothing(FOLLOW_RATE, dt)) : desired;
        // Aim at chest/head height of this actor. Scene units are arbitrary (monocular scale),
        // so a fixed height (was 1.3) looked far above small-unit people.
        const lookAt = add(pose.position, [0, 0.8 * (a.actor.eye_height ?? 1.6), 0]);
        return { eye: this.followEye, rotation: lookRotation(this.followEye, lookAt), fov: p.fov_deg ?? 55 };
      }
      case "first-person": {
        const a = this.actor(p.actor_id);
        if (!a) return this.free(dt, input, p.fov_deg);
        const pose = sampleTrack(a.track, time);
        const heading = quatFromYawPitch(yawOf(pose.rotation), 0);
        const eye = add(add(pose.position, [0, a.actor.eye_height ?? 1.6, 0]), rotate(heading, [0, 0, -0.15]));
        const comfort = p.comfort ?? true;
        this.fpRotation =
          comfort && this.fpRotation ? slerp(this.fpRotation, heading, smoothing(COMFORT_TURN_RATE, dt)) : heading;
        return { eye, rotation: this.fpRotation, fov: p.fov_deg ?? (comfort ? 60 : 75), hiddenActor: a.actor.id };
      }
      case "overhead": {
        const b = this.manifest.scene.bounds;
        const center: Vec3 = [(b.min[0] + b.max[0]) / 2, b.max[1], (b.min[2] + b.max[2]) / 2];
        return {
          eye: add(center, [0, p.height ?? 20, 0]),
          rotation: quatFromYawPitch(0, -Math.PI / 2),
          fov: p.fov_deg ?? 60,
        };
      }
      case "free":
      case "interactive":
        return this.free(dt, input, p.fov_deg);
    }
  }

  private free(dt: number, input: FreeInput, fov: number | undefined): CameraState {
    this.freeYaw -= input.dragX * LOOK_SPEED;
    this.freePitch = clamp(this.freePitch - input.dragY * LOOK_SPEED, -1.5, 1.5);
    const rotation = quatFromYawPitch(this.freeYaw, this.freePitch);
    const k = input.keys;
    const move: Vec3 = [
      (k.has("KeyD") || k.has("ArrowRight") ? 1 : 0) - (k.has("KeyA") || k.has("ArrowLeft") ? 1 : 0),
      (k.has("KeyE") ? 1 : 0) - (k.has("KeyQ") ? 1 : 0),
      (k.has("KeyS") || k.has("ArrowDown") ? 1 : 0) - (k.has("KeyW") || k.has("ArrowUp") ? 1 : 0),
    ];
    const speed = FREE_SPEED * (k.has("ShiftLeft") || k.has("ShiftRight") ? 3 : 1) * dt;
    let delta = rotate(rotation, [move[0] * speed, 0, move[2] * speed]);
    delta = add(delta, [0, move[1] * speed, 0]);
    delta = add(delta, rotate(rotation, [0, 0, input.wheel * 0.01]));
    this.freeEye = add(this.freeEye, delta);
    return { eye: this.freeEye, rotation, fov: fov ?? 60 };
  }
}
