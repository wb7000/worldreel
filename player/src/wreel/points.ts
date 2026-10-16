/** Point chunk (`.pts`) layout: 20-byte little-endian records (spec/wreel.md §6.2). */

export const POINT_STRIDE = 20;

export interface PointChunk {
  count: number;
  /** The raw records, uploaded to the GPU as-is (interleaved vertex buffer). */
  bytes: Uint8Array;
}

export interface DecodedPoint {
  position: [number, number, number];
  rgb: [number, number, number];
  cls: number;
  confidence: number;
}

export function pointChunk(bytes: Uint8Array, expectedCount: number): PointChunk {
  if (bytes.byteLength !== expectedCount * POINT_STRIDE) {
    throw new Error(`point chunk is ${bytes.byteLength} bytes, expected ${expectedCount} × ${POINT_STRIDE}`);
  }
  return { count: expectedCount, bytes };
}

/** Decode one record (used by tests and inspection tools, not the render path). */
export function decodePoint(chunk: PointChunk, index: number): DecodedPoint {
  const view = new DataView(chunk.bytes.buffer, chunk.bytes.byteOffset, chunk.bytes.byteLength);
  const o = index * POINT_STRIDE;
  return {
    position: [view.getFloat32(o, true), view.getFloat32(o + 4, true), view.getFloat32(o + 8, true)],
    rgb: [view.getUint8(o + 12), view.getUint8(o + 13), view.getUint8(o + 14)],
    cls: view.getUint8(o + 15),
    confidence: view.getFloat32(o + 16, true),
  };
}

/** Count points per provenance class code. */
export function classHistogram(chunk: PointChunk): number[] {
  const counts = [0, 0, 0, 0, 0];
  for (let i = 0; i < chunk.count; i++) {
    const c = chunk.bytes[i * POINT_STRIDE + 15]!;
    if (c < counts.length) counts[c]! += 1;
  }
  return counts;
}
