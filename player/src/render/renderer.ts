/**
 * Point renderer interface. Points are drawn as camera-facing discs (instanced quads) straight
 * from the .pts records (20-byte stride: xyz f32, rgb u8 + class u8, confidence f32).
 *
 * Confidence rendering (spec/wreel.md §10.3): points that are not `observed`, or whose
 * confidence is below 0.5, are blended towards a soft vertical colour gradient in proportion
 * to (1 - confidence). Guessed/retrieved/generated points are always blended at least 60 %.
 */

import type { Mat4, Vec3 } from "../math";
import type { PointChunk } from "../wreel/points";

export type RendererKind = "webgpu" | "webgl2";

export interface DrawItem {
  id: string;
  model: Mat4;
  /** World-space disc diameter in metres. */
  pointSize: number;
}

export interface FrameParams {
  view: Mat4;
  proj: Mat4;
  background: Vec3;
  fogTop: Vec3;
  fogBottom: Vec3;
  /** 0 = off, 1 = full strength (default). */
  fogStrength: number;
}

export interface Renderer {
  readonly kind: RendererKind;
  /** True when clip-space depth is [0, 1] (WebGPU) rather than [-1, 1] (WebGL). */
  readonly zeroToOneDepth: boolean;
  setPoints(id: string, chunk: PointChunk): void;
  resize(width: number, height: number): void;
  render(frame: FrameParams, items: readonly DrawItem[]): void;
}

/** Minimum on-screen disc diameter in pixels, so distant points do not vanish. */
export const MIN_POINT_PIXELS = 1.5;

/** Uniform block layout shared by both back ends (floats). */
export const UNIFORM_FLOATS = 16 * 3 + 4 * 3;

export function packUniforms(frame: FrameParams, item: DrawItem, viewportHeight: number): Float32Array {
  const u = new Float32Array(UNIFORM_FLOATS);
  u.set(item.model, 0);
  u.set(frame.view, 16);
  u.set(frame.proj, 32);
  u.set([...frame.fogTop, 1], 48);
  u.set([...frame.fogBottom, 1], 52);
  u.set([item.pointSize, frame.fogStrength, viewportHeight, MIN_POINT_PIXELS], 56);
  return u;
}

export async function createRenderer(canvas: HTMLCanvasElement, prefer: RendererKind | "auto" = "auto"): Promise<Renderer> {
  if (prefer !== "webgl2") {
    const { createWebGPURenderer } = await import("./webgpu");
    const gpu = await createWebGPURenderer(canvas).catch((err: unknown) => {
      console.warn("WebGPU unavailable, falling back to WebGL2:", err);
      return null;
    });
    if (gpu) return gpu;
    if (prefer === "webgpu") throw new Error("WebGPU was requested but is not available");
  }
  const { createWebGL2Renderer } = await import("./webgl2");
  return createWebGL2Renderer(canvas);
}
