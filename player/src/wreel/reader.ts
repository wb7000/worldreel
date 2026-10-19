/**
 * Browser/Node reader for .wreel 1.x (spec/wreel.md).
 *
 * Progressive: `openWreel` extracts the manifest, tracks and the coarsest LOD of every layer;
 * `refine()` then extracts the finer LODs. Every extracted entry is hash-checked against the
 * manifest before use. Full schema validation lives in the Python validator
 * (`python -m formats validate`); the player only checks what it needs to play safely.
 */

import { unzipSync } from "fflate";
import { gaussianChunk, type GaussianChunk } from "./gaussians";
import { pointChunk, type PointChunk } from "./points";
import type { GeometryLayer, Manifest, Track } from "./types";

export const SUPPORTED_MAJOR = 1;
export const SUPPORTED_MINOR = 1;

export type SignatureStatus = "valid" | "invalid" | "missing" | "unsupported";

export class WreelError extends Error {}

export interface LayerData {
  layer: GeometryLayer;
  /** Highest LOD loaded so far. */
  level: number;
  /** PointChunk for `points` layers, GaussianChunk for `gaussians` layers (layer.kind). */
  chunk: PointChunk | GaussianChunk;
}

export interface OpenedWreel {
  manifest: Manifest;
  signature: SignatureStatus;
  warnings: string[];
  /** Static layers by id. */
  layers: Map<string, LayerData>;
  /** Actor geometry by actor id. */
  actorGeometry: Map<string, LayerData>;
  tracks: Map<string, Track>;
  /** Load the finest LOD of every layer. Resolves to the number of layers that improved. */
  refine(): Promise<number>;
  /** Audio entries (stems and event clips), hash-checked, by path. */
  loadAudio(): Promise<Map<string, Uint8Array>>;
}

const REQUIRED_KEYS = [
  "format",
  "format_version",
  "duration",
  "provenance",
  "scene",
  "actors",
  "cameras",
  "pov_presets",
  "regions",
  "audio",
  "interaction",
  "files",
] as const;

const decoder = new TextDecoder();

export async function sha256Hex(data: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", data);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

function base64ToBytes(text: string): Uint8Array {
  const bin = atob(text);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

export async function verifySignature(manifestBytes: Uint8Array, signatureBytes: Uint8Array | undefined): Promise<SignatureStatus> {
  if (!signatureBytes) return "missing";
  let sig: { alg?: string; public_key?: string; signature?: string; signed?: string };
  try {
    sig = JSON.parse(decoder.decode(signatureBytes));
  } catch {
    return "invalid";
  }
  if (sig.alg !== "Ed25519" || sig.signed !== "manifest.json" || !sig.public_key || !sig.signature) {
    return "invalid";
  }
  let key: CryptoKey;
  try {
    key = await crypto.subtle.importKey("raw", base64ToBytes(sig.public_key), { name: "Ed25519" }, false, ["verify"]);
  } catch (err) {
    // Older browsers do not implement Ed25519 in WebCrypto; say so instead of guessing.
    if (err instanceof Error && err.name === "NotSupportedError") return "unsupported";
    return "invalid";
  }
  const ok = await crypto.subtle.verify({ name: "Ed25519" }, key, base64ToBytes(sig.signature), manifestBytes);
  return ok ? "valid" : "invalid";
}

function checkManifest(value: unknown, warnings: string[]): Manifest {
  if (typeof value !== "object" || value === null) throw new WreelError("manifest.json is not an object");
  const m = value as Record<string, unknown>;
  for (const key of REQUIRED_KEYS) {
    if (!(key in m)) throw new WreelError(`manifest is missing required "${key}"`);
  }
  if (m.format !== "wreel") throw new WreelError(`not a .wreel manifest (format=${String(m.format)})`);
  const match = /^(\d+)\.(\d+)$/.exec(String(m.format_version));
  if (!match) throw new WreelError(`bad format_version ${String(m.format_version)}`);
  const major = Number(match[1]);
  const minor = Number(match[2]);
  if (major !== SUPPORTED_MAJOR) throw new WreelError(`.wreel ${major}.x is not supported by this player`);
  if (minor > SUPPORTED_MINOR) warnings.push(`file is .wreel ${major}.${minor}; player knows ${SUPPORTED_MAJOR}.${SUPPORTED_MINOR}`);
  const provenance = m.provenance as Record<string, unknown>;
  if (!provenance || typeof provenance.ai_label !== "string" || typeof provenance.notice !== "string") {
    throw new WreelError("provenance label is missing");
  }
  return value as Manifest;
}

/** Extract the named entries, checking each against the files table. */
async function extract(data: Uint8Array, manifest: Manifest, names: Set<string>): Promise<Map<string, Uint8Array>> {
  const files = unzipSync(data, { filter: (f) => names.has(f.name) });
  const out = new Map<string, Uint8Array>();
  for (const name of names) {
    const bytes = files[name];
    const entry = manifest.files[name];
    if (!bytes || !entry) throw new WreelError(`entry ${name} is missing`);
    if (bytes.byteLength !== entry.bytes || (await sha256Hex(bytes)) !== entry.sha256) {
      throw new WreelError(`entry ${name} does not match its SHA-256 in the manifest`);
    }
    out.set(name, bytes);
  }
  return out;
}

function lodAt(layer: GeometryLayer, which: "coarse" | "fine") {
  const sorted = [...layer.lods].sort((a, b) => a.level - b.level);
  const lod = which === "coarse" ? sorted[0] : sorted[sorted.length - 1];
  if (!lod) throw new WreelError(`layer ${layer.id} has no LODs`);
  return lod;
}

function chunkFor(layer: GeometryLayer, bytes: Uint8Array, count: number): PointChunk | GaussianChunk {
  if (layer.kind === "gaussians") return gaussianChunk(bytes, count);
  if (layer.kind === "points") return pointChunk(bytes, count);
  throw new WreelError(`layer ${layer.id} has unsupported kind ${String(layer.kind)}`);
}

export async function openWreel(data: Uint8Array): Promise<OpenedWreel> {
  const warnings: string[] = [];
  const names: string[] = [];
  const head = unzipSync(data, {
    filter: (f) => {
      names.push(f.name);
      return f.name === "manifest.json" || f.name === "signature.json";
    },
  });
  const manifestBytes = head["manifest.json"];
  if (!manifestBytes) throw new WreelError("manifest.json is missing (is this a .wreel file?)");
  const signature = await verifySignature(manifestBytes, head["signature.json"]);
  let parsed: unknown;
  try {
    parsed = JSON.parse(decoder.decode(manifestBytes));
  } catch {
    throw new WreelError("manifest.json is not valid JSON");
  }
  const manifest = checkManifest(parsed, warnings);

  const listed = new Set(Object.keys(manifest.files));
  for (const name of names) {
    if (name !== "manifest.json" && name !== "signature.json" && !listed.has(name)) {
      throw new WreelError(`entry ${name} is not listed in the manifest`);
    }
  }

  const coarse = new Set<string>();
  for (const layer of manifest.scene.layers) coarse.add(lodAt(layer, "coarse").path);
  for (const actor of manifest.actors) {
    coarse.add(lodAt(actor.geometry, "coarse").path);
    coarse.add(actor.track.path);
  }
  const got = await extract(data, manifest, coarse);

  const toLayer = (layer: GeometryLayer): LayerData => {
    const lod = lodAt(layer, "coarse");
    return { layer, level: lod.level, chunk: chunkFor(layer, got.get(lod.path)!, lod.count) };
  };
  const layers = new Map(manifest.scene.layers.map((l) => [l.id, toLayer(l)] as const));
  const actorGeometry = new Map(manifest.actors.map((a) => [a.id, toLayer(a.geometry)] as const));
  const tracks = new Map(
    manifest.actors.map((a) => [a.id, JSON.parse(decoder.decode(got.get(a.track.path)!)) as Track] as const),
  );

  async function refine(): Promise<number> {
    const targets: LayerData[] = [...layers.values(), ...actorGeometry.values()].filter(
      (d) => lodAt(d.layer, "fine").level > d.level,
    );
    if (targets.length === 0) return 0;
    const fine = await extract(data, manifest, new Set(targets.map((d) => lodAt(d.layer, "fine").path)));
    for (const d of targets) {
      const lod = lodAt(d.layer, "fine");
      d.chunk = chunkFor(d.layer, fine.get(lod.path)!, lod.count);
      d.level = lod.level;
    }
    return targets.length;
  }

  async function loadAudio(): Promise<Map<string, Uint8Array>> {
    const paths = new Set<string>(manifest.audio.tracks.map((t) => t.path));
    for (const e of manifest.audio.events) {
      const clip = e.attributes?.["clip"];
      if (typeof clip === "string" && manifest.files[clip]) paths.add(clip);
    }
    return paths.size ? extract(data, manifest, paths) : new Map();
  }

  return { manifest, signature, warnings, layers, actorGeometry, tracks, refine, loadAudio };
}
