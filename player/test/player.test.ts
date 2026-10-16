import { describe, expect, it } from "vitest";
import { CameraDirector, type FreeInput } from "../src/cameras";
import { hasWebGPU, isViewMode, VIEW_MODES } from "../src/index";
import { length, rotate, sub, type Vec3 } from "../src/math";
import { createRuleEngine, moodTension } from "../src/rules";
import type { Manifest, PovPreset, Track } from "../src/wreel/types";
import { sampleManifest } from "./fixtures";

const noInput = (): FreeInput => ({ dragX: 0, dragY: 0, wheel: 0, keys: new Set() });

function scene(): { manifest: Manifest; tracks: Map<string, Track> } {
  const manifest = sampleManifest() as unknown as Manifest;
  manifest.actors[0]!.eye_height = 1.6;
  manifest.cameras = [
    {
      id: "director",
      name: "Director",
      kind: "director",
      keyframes: [{ t: 0, position: [0, 2, 10], rotation: [0, 0, 0, 1], fov_deg: 50 }],
    },
  ];
  // Actor walks along +X, facing +X (yaw -90°).
  const s = Math.SQRT1_2;
  const facingX: [number, number, number, number] = [0, -s, 0, s];
  const tracks = new Map<string, Track>([
    ["a1", { times: [0, 10], positions: [[0, 0, 0], [10, 0, 0]], rotations: [facingX, facingX] }],
  ]);
  return { manifest, tracks };
}

/** Run the director long enough for transitions and smoothing to settle. */
function settle(director: CameraDirector, t: number) {
  let cam = director.update(t, 0.016, noInput());
  for (let i = 0; i < 200; i++) cam = director.update(t, 0.05, noInput());
  return cam;
}

describe("viewer menu", () => {
  it("knows the brief's view modes", () => {
    expect(VIEW_MODES).toEqual(expect.arrayContaining(["directors-cut", "free", "first-person", "third-person"]));
    expect(isViewMode("overhead")).toBe(true);
    expect(isViewMode("vr")).toBe(false);
    expect(hasWebGPU({ gpu: {} })).toBe(true);
    expect(hasWebGPU({})).toBe(false);
  });
});

describe("camera rigs", () => {
  const director: PovPreset = { id: "dc", name: "Director", mode: "directors-cut", camera_id: "director" };

  it("plays the director's camera path", () => {
    const { manifest, tracks } = scene();
    const cam = new CameraDirector(manifest, tracks, director).update(0, 0.016, noInput());
    expect(cam.eye).toEqual([0, 2, 10]);
    expect(cam.fov).toBe(50);
  });

  it("third-person follows from behind the actor", () => {
    const { manifest, tracks } = scene();
    const d = new CameraDirector(manifest, tracks, director);
    d.setPreset({ id: "f", name: "Follow", mode: "third-person", actor_id: "a1", offset: [0, 2, 4] });
    const cam = settle(d, 5);
    // Actor at (5,0,0) facing +X, so "behind" is -X.
    expect(cam.eye[0]).toBeCloseTo(1, 1);
    expect(cam.eye[1]).toBeCloseTo(2, 1);
    expect(cam.hiddenActor).toBeUndefined();
    const forward = rotate(cam.rotation, [0, 0, -1]);
    expect(forward[0]).toBeGreaterThan(0.9); // looking towards the actor
  });

  it("first-person looks through the actor's eyes and hides its body", () => {
    const { manifest, tracks } = scene();
    const d = new CameraDirector(manifest, tracks, director);
    d.setPreset({ id: "e", name: "Eyes", mode: "first-person", actor_id: "a1", comfort: true });
    const cam = settle(d, 5);
    expect(cam.hiddenActor).toBe("a1");
    expect(cam.eye[1]).toBeCloseTo(1.6, 3);
    expect(length(sub(cam.eye, [5.15, 1.6, 0] as Vec3))).toBeLessThan(0.01);
    expect(cam.fov).toBe(60); // comfort mode narrows the view
  });

  it("free camera starts from the current view and moves with keys", () => {
    const { manifest, tracks } = scene();
    const d = new CameraDirector(manifest, tracks, director);
    d.update(0, 0.016, noInput());
    d.setPreset({ id: "free", name: "Free", mode: "free" });
    settle(d, 0);
    const before = d.update(0, 0.016, noInput()).eye;
    const after = d.update(0, 1, { ...noInput(), keys: new Set(["KeyW"]) }).eye;
    expect(before[2]).toBeCloseTo(10, 3);
    expect(after[2]).toBeCloseTo(5, 3); // 5 m/s along -Z
  });
});

describe("rules", () => {
  it("fires time rules once when crossed and ignores scrubbing back", () => {
    const engine = createRuleEngine([
      { id: "r1", trigger: { type: "time", t: 2 }, action: { type: "set_tension", value: 7 } },
      { id: "r2", trigger: { type: "proximity", actor_id: "a1", radius: 1 }, action: { type: "set_tension", value: 9 } },
    ]);
    expect(engine.skipped).toEqual(["r2 (proximity → set_tension)"]);
    expect(engine.step(1, 1.5)).toEqual([]);
    expect(engine.step(1.5, 2.1)).toEqual([{ type: "set_tension", value: 7 }]);
    expect(engine.step(2.1, 3)).toEqual([]);
    expect(engine.step(3, 0)).toEqual([]);
  });

  it("reads tension from the mood curve", () => {
    const m = sampleManifest() as unknown as Manifest;
    m.audio.mood = [
      { t: 0, tension: 2 },
      { t: 5, tension: 8 },
    ];
    expect(moodTension(m, 4.9)).toBe(2);
    expect(moodTension(m, 5)).toBe(8);
  });
});
