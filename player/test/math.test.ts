import { describe, expect, it } from "vitest";
import {
  lookRotation,
  perspective,
  poseMatrix,
  quatFromYawPitch,
  rotate,
  slerp,
  transformPoint,
  viewMatrix,
  yawOf,
  type Vec3,
} from "../src/math";
import { bracket, sampleCamera, sampleTrack } from "../src/wreel/track";

const close = (a: readonly number[], b: readonly number[], digits = 5) =>
  a.forEach((v, i) => expect(v).toBeCloseTo(b[i]!, digits));

describe("math", () => {
  it("rotates -Z by yaw like the Python generator", () => {
    // yaw = atan2(-vx, -vz) points -Z along (vx, vz)
    const q = quatFromYawPitch(Math.atan2(-1, 0), 0);
    close(rotate(q, [0, 0, -1]), [1, 0, 0]);
  });

  it("lookRotation points the camera at the target", () => {
    const eye: Vec3 = [3, 2, 5];
    const target: Vec3 = [0, 1, 0];
    const q = lookRotation(eye, target);
    const view = viewMatrix(eye, q);
    const p = transformPoint(view, target);
    expect(Math.abs(p[0])).toBeLessThan(1e-5);
    expect(Math.abs(p[1])).toBeLessThan(1e-5);
    expect(p[2]).toBeLessThan(0); // in front of the camera
  });

  it("view matrix inverts the camera pose", () => {
    const eye: Vec3 = [1, 2, 3];
    const q = quatFromYawPitch(0.7, -0.3);
    const world = transformPoint(poseMatrix(eye, q), [0.5, -0.2, -4]);
    close(transformPoint(viewMatrix(eye, q), [world[0], world[1], world[2]]).slice(0, 3), [0.5, -0.2, -4]);
  });

  it("perspective maps near/far to the right depth range", () => {
    for (const zeroToOne of [true, false]) {
      const m = perspective(60, 1, 0.1, 100, zeroToOne);
      const n = transformPoint(m, [0, 0, -0.1]);
      const f = transformPoint(m, [0, 0, -100]);
      expect(n[2] / n[3]).toBeCloseTo(zeroToOne ? 0 : -1, 5);
      expect(f[2] / f[3]).toBeCloseTo(1, 5);
    }
  });

  it("slerp stays unit length and yawOf recovers yaw", () => {
    const a = quatFromYawPitch(0, 0);
    const b = quatFromYawPitch(2, 0);
    const m = slerp(a, b, 0.5);
    expect(Math.hypot(...m)).toBeCloseTo(1, 6);
    expect(yawOf(m)).toBeCloseTo(1, 5);
  });
});

describe("tracks", () => {
  const track = {
    times: [0, 1, 3],
    positions: [
      [0, 0, 0],
      [1, 0, 0],
      [1, 0, 2],
    ] as Vec3[],
    rotations: [quatFromYawPitch(0, 0), quatFromYawPitch(1, 0), quatFromYawPitch(1, 0)],
  };

  it("brackets times and clamps the ends", () => {
    expect(bracket([0, 1, 3], -1)).toEqual([0, 0]);
    expect(bracket([0, 1, 3], 2)).toEqual([1, 0.5]);
    expect(bracket([0, 1, 3], 9)).toEqual([2, 0]);
    expect(bracket([5], 9)).toEqual([0, 0]);
  });

  it("interpolates position and rotation", () => {
    const s = sampleTrack(track, 0.5);
    close(s.position, [0.5, 0, 0]);
    expect(yawOf(s.rotation)).toBeCloseTo(0.5, 5);
    close(sampleTrack(track, 10).position, [1, 0, 2]);
  });

  it("samples camera fov", () => {
    const keys = [
      { t: 0, position: [0, 0, 0] as Vec3, rotation: quatFromYawPitch(0, 0), fov_deg: 40 },
      { t: 2, position: [2, 0, 0] as Vec3, rotation: quatFromYawPitch(0, 0), fov_deg: 60 },
    ];
    expect(sampleCamera(keys, 1).fov).toBeCloseTo(50);
  });
});
