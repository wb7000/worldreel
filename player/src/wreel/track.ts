/** Sampling of actor tracks and camera paths (linear position, slerp rotation, clamped ends). */

import { lerp3, slerp, type Quat, type Vec3 } from "../math";
import type { Keyframe, Track } from "./types";

export interface Pose {
  position: Vec3;
  rotation: Quat;
}

/** Index i and blend u such that time t lies between times[i] and times[i+1]. */
export function bracket(times: readonly number[], t: number): [number, number] {
  const n = times.length;
  if (n === 0) throw new Error("empty time list");
  if (n === 1 || t <= times[0]!) return [0, 0];
  if (t >= times[n - 1]!) return [n - 1, 0];
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (times[mid]! <= t) lo = mid;
    else hi = mid;
  }
  const t0 = times[lo]!;
  const t1 = times[lo + 1]!;
  return [lo, (t - t0) / (t1 - t0)];
}

export function sampleTrack(track: Track, t: number): Pose {
  const [i, u] = bracket(track.times, t);
  const j = Math.min(i + 1, track.times.length - 1);
  return {
    position: lerp3(track.positions[i]!, track.positions[j]!, u),
    rotation: slerp(track.rotations[i]!, track.rotations[j]!, u),
  };
}

export function sampleCamera(keys: readonly Keyframe[], t: number): Pose & { fov: number } {
  const [i, u] = bracket(
    keys.map((k) => k.t),
    t,
  );
  const j = Math.min(i + 1, keys.length - 1);
  const a = keys[i]!;
  const b = keys[j]!;
  return {
    position: lerp3(a.position, b.position, u),
    rotation: slerp(a.rotation, b.rotation, u),
    fov: a.fov_deg + (b.fov_deg - a.fov_deg) * u,
  };
}
