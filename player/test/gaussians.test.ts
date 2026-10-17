import { describe, expect, it } from "vitest";
import { quatFromYawPitch, viewMatrix, type Vec3 } from "../src/math";
import {
  backToFront,
  GAUSSIAN_STRIDE,
  gaussianChunk,
  reorder,
  SPLAT_FLOATS,
  splatInstances,
} from "../src/wreel/gaussians";
import { openWreel } from "../src/wreel/reader";
import { buildWreel, sampleEntries, sampleManifest } from "./fixtures";

interface G {
  p: Vec3;
  s: Vec3;
  q: [number, number, number, number];
  rgb: [number, number, number];
  opacity: number;
}

function pack(gs: G[]): Uint8Array {
  const out = new Uint8Array(gs.length * GAUSSIAN_STRIDE);
  const v = new DataView(out.buffer);
  gs.forEach((g, i) => {
    const o = i * GAUSSIAN_STRIDE;
    [...g.p, ...g.s, ...g.q].forEach((x, k) => v.setFloat32(o + 4 * k, x, true));
    out.set([...g.rgb, Math.round(g.opacity * 255), 0], o + 40);
    v.setFloat32(o + 48, 0.9, true);
  });
  return out;
}

const s = Math.SQRT1_2;

describe("gaussian instances", () => {
  it("builds Σ = R·diag(s²)·Rᵀ", () => {
    // 90° about z: local x → world y, so the long axis (s_x = 2) ends up along y.
    const chunk = gaussianChunk(pack([{ p: [1, 2, 3], s: [2, 0.5, 1], q: [0, 0, s, s], rgb: [255, 0, 0], opacity: 1 }]), 1);
    const inst = splatInstances(chunk);
    expect(inst.length).toBe(SPLAT_FLOATS);
    expect(Array.from(inst.slice(0, 3))).toEqual([1, 2, 3]);
    const [xx, xy, xz, yy, yz, zz] = Array.from(inst.slice(3, 9));
    expect(xx).toBeCloseTo(0.25, 5);
    expect(yy).toBeCloseTo(4, 5);
    expect(zz).toBeCloseTo(1, 5);
    expect(xy).toBeCloseTo(0, 5);
    expect(xz).toBeCloseTo(0, 5);
    expect(yz).toBeCloseTo(0, 5);
    expect(inst[9]).toBeCloseTo(1);
    expect(inst[12]).toBeCloseTo(1);
  });

  it("rejects a chunk of the wrong size", () => {
    expect(() => gaussianChunk(new Uint8Array(51), 1)).toThrow(/52/);
  });

  it("sorts back to front for any camera", () => {
    const gs: G[] = [-1, -9, -4, -2, -7].map((z) => ({ p: [0, 0, z], s: [1, 1, 1], q: [0, 0, 0, 1], rgb: [0, 0, 0], opacity: 1 }));
    const inst = splatInstances(gaussianChunk(pack(gs), gs.length));
    // Camera at the origin looking down -z: depth = -z, so the farthest is z = -9.
    const order = backToFront(inst, viewMatrix([0, 0, 0], [0, 0, 0, 1]));
    expect(Array.from(order)).toEqual([1, 4, 2, 3, 0]);
    // Turn around (yaw 180°): now z = -1 is farthest... all are behind; order reverses.
    const back = backToFront(inst, viewMatrix([0, 0, -10], quatFromYawPitch(Math.PI, 0)));
    expect(Array.from(back)).toEqual([0, 3, 2, 4, 1]);
    const sorted = reorder(inst, order);
    expect(sorted[2]).toBe(-9);
    expect(sorted[SPLAT_FLOATS * 4 + 2]).toBe(-1);
  });
});

describe("reading .wreel 1.1 with a gaussian layer", () => {
  it("opens and refines gaussian LODs", async () => {
    const manifest = sampleManifest();
    manifest.format_version = "1.1";
    const scene = manifest.scene as { layers: unknown[] };
    scene.layers = [
      {
        id: "static",
        kind: "gaussians",
        lods: [
          { level: 0, path: "scene/static/lod0.gsp", count: 1 },
          { level: 1, path: "scene/static/lod1.gsp", count: 2 },
        ],
      },
    ];
    const g: G = { p: [0, 0, -3], s: [0.1, 0.1, 0.1], q: [0, 0, 0, 1], rgb: [10, 20, 30], opacity: 0.5 };
    const entries = { ...sampleEntries() };
    delete entries["scene/static/lod0.pts"];
    delete entries["scene/static/lod1.pts"];
    entries["scene/static/lod0.gsp"] = pack([g]);
    entries["scene/static/lod1.gsp"] = pack([g, { ...g, p: [1, 0, -3] }]);
    const w = await openWreel(await buildWreel({ manifest, entries }));
    expect(w.warnings).toEqual([]);
    const layer = w.layers.get("static")!;
    expect(layer.layer.kind).toBe("gaussians");
    expect(layer.chunk.count).toBe(1);
    await w.refine();
    expect(layer.chunk.count).toBe(2);
  });
});
