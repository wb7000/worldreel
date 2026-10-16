/** Build small .wreel files in memory for tests (mirrors the Python writer's layout). */

import { zipSync } from "fflate";
import { POINT_STRIDE } from "../src/wreel/points";
import { sha256Hex } from "../src/wreel/reader";

const enc = new TextEncoder();

export function packPoints(points: { p: [number, number, number]; rgb: [number, number, number]; cls: number; conf: number }[]): Uint8Array {
  const out = new Uint8Array(points.length * POINT_STRIDE);
  const view = new DataView(out.buffer);
  points.forEach((pt, i) => {
    const o = i * POINT_STRIDE;
    view.setFloat32(o, pt.p[0], true);
    view.setFloat32(o + 4, pt.p[1], true);
    view.setFloat32(o + 8, pt.p[2], true);
    out.set([...pt.rgb, pt.cls], o + 12);
    view.setFloat32(o + 16, pt.conf, true);
  });
  return out;
}

export function sampleEntries(): Record<string, Uint8Array> {
  const coarse = packPoints([{ p: [0, 0, 0], rgb: [200, 10, 10], cls: 0, conf: 1 }]);
  const fine = packPoints([
    { p: [0, 0, 0], rgb: [200, 10, 10], cls: 0, conf: 1 },
    { p: [1, 0, 0], rgb: [128, 128, 128], cls: 2, conf: 0.2 },
  ]);
  const body = packPoints([{ p: [0, 1, 0], rgb: [10, 10, 200], cls: 0, conf: 0.9 }]);
  const track = { times: [0, 2], positions: [[0, 0, 0], [2, 0, 0]], rotations: [[0, 0, 0, 1], [0, 0, 0, 1]] };
  return {
    "scene/static/lod0.pts": coarse,
    "scene/static/lod1.pts": fine,
    "actors/a1/lod0.pts": body,
    "actors/a1/track.json": enc.encode(JSON.stringify(track)),
  };
}

export function sampleManifest(): Record<string, unknown> {
  return {
    format: "wreel",
    format_version: "1.0",
    title: "Test",
    duration: 2,
    provenance: {
      ai_label: "AI-generated",
      notice: "Test scene. Not a recording.",
      tool: "worldreel",
      tool_version: "test",
      created_at: "2026-10-16T00:00:00+00:00",
      source: { kind: "synthetic" },
      identity_policy: "synthetic-only",
      rights: { upload_rights: "synthetic", takedown: { status: "none" } },
    },
    generation: { seeds: {}, stages: [], models: [] },
    scene: {
      bounds: { min: [-1, 0, -1], max: [1, 1, 1] },
      background: [0, 0, 0],
      lighting: { ambient: { color: [1, 1, 1], intensity: 1 }, lights: [] },
      layers: [
        {
          id: "static",
          kind: "points",
          point_size: 0.1,
          lods: [
            { level: 0, path: "scene/static/lod0.pts", count: 1 },
            { level: 1, path: "scene/static/lod1.pts", count: 2 },
          ],
        },
      ],
    },
    actors: [
      {
        id: "a1",
        type: "object",
        label: "Box",
        geometry: { id: "body", kind: "points", point_size: 0.05, lods: [{ level: 0, path: "actors/a1/lod0.pts", count: 1 }] },
        track: { path: "actors/a1/track.json" },
      },
    ],
    cameras: [],
    pov_presets: [{ id: "free", name: "Free", mode: "free" }],
    regions: [],
    audio: { tracks: [], events: [], mood: [] },
    interaction: { rules: [] },
    files: {},
  };
}

export interface BuildOptions {
  manifest?: Record<string, unknown>;
  entries?: Record<string, Uint8Array>;
  /** Mutate the entries after the files table was computed (simulates tampering). */
  tamper?: (entries: Record<string, Uint8Array>) => void;
  /** Mutate the manifest after signing. */
  tamperManifest?: (manifest: Record<string, unknown>) => void;
  sign?: boolean;
}

export async function buildWreel(opts: BuildOptions = {}): Promise<Uint8Array> {
  const manifest = opts.manifest ?? sampleManifest();
  const entries = opts.entries ?? sampleEntries();
  const files: Record<string, unknown> = {};
  for (const [path, data] of Object.entries(entries).sort()) {
    files[path] = { sha256: await sha256Hex(data), bytes: data.byteLength, media_type: "application/octet-stream" };
  }
  manifest.files = files;
  const manifestBytes = enc.encode(JSON.stringify(manifest, null, 2) + "\n");
  const zipEntries: Record<string, Uint8Array> = { "manifest.json": manifestBytes };
  if (opts.sign !== false) {
    const keys = (await crypto.subtle.generateKey({ name: "Ed25519" }, true, ["sign", "verify"])) as CryptoKeyPair;
    const raw = new Uint8Array(await crypto.subtle.exportKey("raw", keys.publicKey));
    const signature = new Uint8Array(await crypto.subtle.sign({ name: "Ed25519" }, keys.privateKey, manifestBytes));
    const b64 = (b: Uint8Array) => Buffer.from(b).toString("base64");
    zipEntries["signature.json"] = enc.encode(
      JSON.stringify({ alg: "Ed25519", public_key: b64(raw), signature: b64(signature), signed: "manifest.json" }),
    );
  }
  if (opts.tamperManifest) {
    const m = JSON.parse(new TextDecoder().decode(manifestBytes)) as Record<string, unknown>;
    opts.tamperManifest(m);
    zipEntries["manifest.json"] = enc.encode(JSON.stringify(m, null, 2) + "\n");
  }
  const data = { ...entries };
  opts.tamper?.(data);
  return zipSync({ ...zipEntries, ...data });
}
