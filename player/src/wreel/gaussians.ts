/**
 * Gaussian chunks (`.gsp`, spec/wreel.md §6.3): 52-byte little-endian records.
 * The renderer consumes a per-splat instance array (16 floats) and needs it sorted back to front.
 */

export const GAUSSIAN_STRIDE = 52;
/** Floats per instance: centre(3) cov(6: xx xy xz yy yz zz) rgba(4) class conf pad. */
export const SPLAT_FLOATS = 16;

export interface GaussianChunk {
  count: number;
  bytes: Uint8Array;
}

export function gaussianChunk(bytes: Uint8Array, expectedCount: number): GaussianChunk {
  if (bytes.byteLength !== expectedCount * GAUSSIAN_STRIDE) {
    throw new Error(`gaussian chunk is ${bytes.byteLength} bytes, expected ${expectedCount} × ${GAUSSIAN_STRIDE}`);
  }
  return { count: expectedCount, bytes };
}

/**
 * Decode records into renderer instances: Σ = R·diag(s²)·Rᵀ is computed once here, so the
 * vertex shader only projects it.
 */
export function splatInstances(chunk: GaussianChunk): Float32Array {
  const out = new Float32Array(chunk.count * SPLAT_FLOATS);
  const view = new DataView(chunk.bytes.buffer, chunk.bytes.byteOffset, chunk.bytes.byteLength);
  for (let i = 0; i < chunk.count; i++) {
    const o = i * GAUSSIAN_STRIDE;
    const f = (k: number) => view.getFloat32(o + k, true);
    const [px, py, pz] = [f(0), f(4), f(8)];
    const [sx, sy, sz] = [f(12), f(16), f(20)];
    let [x, y, z, w] = [f(24), f(28), f(32), f(36)];
    const n = Math.hypot(x, y, z, w) || 1;
    x /= n;
    y /= n;
    z /= n;
    w /= n;
    // Rotation matrix (row-major r[row][col]).
    const r00 = 1 - 2 * (y * y + z * z), r01 = 2 * (x * y - z * w), r02 = 2 * (x * z + y * w);
    const r10 = 2 * (x * y + z * w), r11 = 1 - 2 * (x * x + z * z), r12 = 2 * (y * z - x * w);
    const r20 = 2 * (x * z - y * w), r21 = 2 * (y * z + x * w), r22 = 1 - 2 * (x * x + y * y);
    const a = sx * sx, b = sy * sy, c = sz * sz;
    const base = i * SPLAT_FLOATS;
    out[base] = px;
    out[base + 1] = py;
    out[base + 2] = pz;
    out[base + 3] = r00 * r00 * a + r01 * r01 * b + r02 * r02 * c; // xx
    out[base + 4] = r00 * r10 * a + r01 * r11 * b + r02 * r12 * c; // xy
    out[base + 5] = r00 * r20 * a + r01 * r21 * b + r02 * r22 * c; // xz
    out[base + 6] = r10 * r10 * a + r11 * r11 * b + r12 * r12 * c; // yy
    out[base + 7] = r10 * r20 * a + r11 * r21 * b + r12 * r22 * c; // yz
    out[base + 8] = r20 * r20 * a + r21 * r21 * b + r22 * r22 * c; // zz
    out[base + 9] = chunk.bytes[o + 40]! / 255;
    out[base + 10] = chunk.bytes[o + 41]! / 255;
    out[base + 11] = chunk.bytes[o + 42]! / 255;
    out[base + 12] = chunk.bytes[o + 43]! / 255;
    out[base + 13] = chunk.bytes[o + 44]!;
    out[base + 14] = f(48);
    out[base + 15] = 0;
  }
  return out;
}

/**
 * Back-to-front order for a column-major view matrix (camera looks along -Z, so view-space depth
 * is -z). 16-bit counting sort: O(n), stable, precise enough for blending.
 */
export function backToFront(instances: Float32Array, view: Float32Array): Uint32Array {
  const n = instances.length / SPLAT_FLOATS;
  const depth = new Float32Array(n);
  let lo = Infinity;
  let hi = -Infinity;
  const v2 = view[2]!, v6 = view[6]!, v10 = view[10]!, v14 = view[14]!;
  for (let i = 0; i < n; i++) {
    const b = i * SPLAT_FLOATS;
    const d = -(v2 * instances[b]! + v6 * instances[b + 1]! + v10 * instances[b + 2]! + v14);
    depth[i] = d;
    if (d < lo) lo = d;
    if (d > hi) hi = d;
  }
  const buckets = 65536;
  const scale = hi > lo ? (buckets - 1) / (hi - lo) : 0;
  const counts = new Uint32Array(buckets);
  const key = new Uint16Array(n);
  for (let i = 0; i < n; i++) {
    // Far first: invert the key so larger depth sorts earlier.
    const k = buckets - 1 - Math.floor((depth[i]! - lo) * scale);
    key[i] = k;
    counts[k]!++;
  }
  for (let k = 1; k < buckets; k++) counts[k]! += counts[k - 1]!;
  const order = new Uint32Array(n);
  for (let i = n - 1; i >= 0; i--) order[--counts[key[i]!]!] = i;
  return order;
}

/** Gather instances into the given order (the renderer uploads the result). */
export function reorder(instances: Float32Array, order: Uint32Array, out?: Float32Array): Float32Array {
  const target = out && out.length === instances.length ? out : new Float32Array(instances.length);
  for (let j = 0; j < order.length; j++) {
    const src = order[j]! * SPLAT_FLOATS;
    target.set(instances.subarray(src, src + SPLAT_FLOATS), j * SPLAT_FLOATS);
  }
  return target;
}
