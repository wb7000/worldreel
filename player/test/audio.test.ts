import { describe, expect, it } from "vitest";
import { distanceGain, eventsInWindow, heard, listenerFrame, stemsToPlay } from "../src/audio";
import type { CameraState } from "../src/cameras";
import { quatFromYawPitch, type Vec3 } from "../src/math";
import type { AudioEvent, AudioTrack } from "../src/wreel/types";

const facingMinusZ = quatFromYawPitch(0, 0);

describe("listener", () => {
  it("faces where the view looks", () => {
    const l = listenerFrame({ eye: [1, 2, 3], rotation: facingMinusZ, fov: 60 });
    expect(l.position).toEqual([1, 2, 3]);
    expect(l.forward[2]).toBeCloseTo(-1);
    expect(l.up[1]).toBeCloseTo(1);
  });

  it("uses the Web Audio inverse distance model", () => {
    expect(distanceGain(0.5)).toBe(1); // inside the reference distance
    expect(distanceGain(1)).toBe(1);
    expect(distanceGain(4)).toBeCloseTo(0.25);
  });

  it("places a source to the right at +90 degrees", () => {
    const l = listenerFrame({ eye: [0, 0, 0], rotation: facingMinusZ, fov: 60 });
    expect(heard(l, [3, 0, 0]).azimuthDeg).toBeCloseTo(90);
    expect(heard(l, [-3, 0, 0]).azimuthDeg).toBeCloseTo(-90);
    expect(heard(l, [0, 0, -3]).azimuthDeg).toBeCloseTo(0);
  });
});

describe("first person vs third person (brief Phase 5 acceptance)", () => {
  // A walker at the origin facing -Z; another person 3 m to the walker's right.
  const own: Vec3 = [0, 0, 0];
  const other: Vec3 = [3, 0, 0];
  const firstPerson: CameraState = { eye: [0, 1.62, -0.15], rotation: facingMinusZ, fov: 60 };
  // Third person: the player's follow offset [0, 1.15 h, 1.0 h] behind, looking at the walker.
  const h = 1.75;
  const thirdPerson: CameraState = { eye: [0, 1.15 * h, 1.0 * h], rotation: facingMinusZ, fov: 55 };

  it("hears the walker's own step louder through their ears than from the camera", () => {
    const fp = heard(listenerFrame(firstPerson), own);
    const tp = heard(listenerFrame(thirdPerson), own);
    expect(fp.gainDb - tp.gainDb).toBeGreaterThan(3);
  });

  it("hears another person from a different direction", () => {
    const fp = heard(listenerFrame(firstPerson), other);
    const tp = heard(listenerFrame(thirdPerson), other);
    expect(Math.abs(fp.azimuthDeg - tp.azimuthDeg)).toBeGreaterThan(10);
  });
});

describe("scheduling", () => {
  const events: AudioEvent[] = [
    { t: 0.5, type: "footstep", position: [0, 0, 0], attributes: { clip: "a.wav" } },
    { t: 1.0, type: "footstep", position: [0, 0, 0], attributes: { clip: "a.wav" } },
    { t: 1.1, type: "door" }, // no position / clip: never scheduled
  ];

  it("takes each event once, in its window", () => {
    expect(eventsInWindow(events, 0, 0.6).map((e) => e.t)).toEqual([0.5]);
    expect(eventsInWindow(events, 0.6, 1.2).map((e) => e.t)).toEqual([1.0]);
  });

  it("drops the pre-mixed foley stem when positional footsteps exist", () => {
    const tracks = ["foley", "music", "ambience"].map(
      (layer): AudioTrack => ({ id: layer, layer, path: `${layer}.wav`, media_type: "audio/wav", start: 0, license: "synthetic" }),
    );
    expect(stemsToPlay(tracks, events).map((t) => t.layer)).toEqual(["music", "ambience"]);
    expect(stemsToPlay(tracks, []).map((t) => t.layer)).toEqual(["foley", "music", "ambience"]);
  });
});
