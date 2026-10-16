/** Minimal column-major 3D math (Y up, right-handed, -Z forward), matching spec/wreel.md §2. */

export type Vec3 = [number, number, number];
export type Quat = [number, number, number, number]; // [x, y, z, w]
export type Mat4 = Float32Array;

export const IDENTITY_QUAT: Quat = [0, 0, 0, 1];

export function add(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

export function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

export function scale(a: Vec3, s: number): Vec3 {
  return [a[0] * s, a[1] * s, a[2] * s];
}

export function lerp3(a: Vec3, b: Vec3, t: number): Vec3 {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

export function length(a: Vec3): number {
  return Math.hypot(a[0], a[1], a[2]);
}

export function normalize(a: Vec3): Vec3 {
  const l = length(a) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
}

export function quatMul(a: Quat, b: Quat): Quat {
  const [ax, ay, az, aw] = a;
  const [bx, by, bz, bw] = b;
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}

export function quatNormalize(q: Quat): Quat {
  const l = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0] / l, q[1] / l, q[2] / l, q[3] / l];
}

export function quatFromAxisAngle(axis: Vec3, angle: number): Quat {
  const s = Math.sin(angle / 2);
  const n = normalize(axis);
  return [n[0] * s, n[1] * s, n[2] * s, Math.cos(angle / 2)];
}

/** Yaw (about +Y) then pitch (about local +X), no roll. */
export function quatFromYawPitch(yaw: number, pitch: number): Quat {
  return quatMul(quatFromAxisAngle([0, 1, 0], yaw), quatFromAxisAngle([1, 0, 0], pitch));
}

export function rotate(q: Quat, v: Vec3): Vec3 {
  const [x, y, z, w] = q;
  // t = 2 * cross(q.xyz, v); v' = v + w * t + cross(q.xyz, t)
  const tx = 2 * (y * v[2] - z * v[1]);
  const ty = 2 * (z * v[0] - x * v[2]);
  const tz = 2 * (x * v[1] - y * v[0]);
  return [
    v[0] + w * tx + (y * tz - z * ty),
    v[1] + w * ty + (z * tx - x * tz),
    v[2] + w * tz + (x * ty - y * tx),
  ];
}

export function slerp(a: Quat, b: Quat, t: number): Quat {
  let cos = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  let bb: Quat = b;
  if (cos < 0) {
    cos = -cos;
    bb = [-b[0], -b[1], -b[2], -b[3]];
  }
  if (cos > 0.9995) {
    return quatNormalize([
      a[0] + (bb[0] - a[0]) * t,
      a[1] + (bb[1] - a[1]) * t,
      a[2] + (bb[2] - a[2]) * t,
      a[3] + (bb[3] - a[3]) * t,
    ]);
  }
  const theta = Math.acos(cos);
  const s = Math.sin(theta);
  const wa = Math.sin((1 - t) * theta) / s;
  const wb = Math.sin(t * theta) / s;
  return [
    a[0] * wa + bb[0] * wb,
    a[1] * wa + bb[1] * wb,
    a[2] * wa + bb[2] * wb,
    a[3] * wa + bb[3] * wb,
  ];
}

/** Rotation that points -Z from `eye` towards `target`, keeping +Y up (no roll). */
export function lookRotation(eye: Vec3, target: Vec3): Quat {
  const f = normalize(sub(target, eye));
  const yaw = Math.atan2(-f[0], -f[2]);
  const pitch = Math.asin(Math.max(-1, Math.min(1, f[1])));
  return quatFromYawPitch(yaw, pitch);
}

/** Yaw angle of a rotation (heading of its -Z axis projected on the ground plane). */
export function yawOf(q: Quat): number {
  const f = rotate(q, [0, 0, -1]);
  return Math.atan2(-f[0], -f[2]);
}

/** Model matrix from translation + rotation (column-major). */
export function poseMatrix(p: Vec3, q: Quat): Mat4 {
  const [x, y, z, w] = q;
  const m = new Float32Array(16);
  m[0] = 1 - 2 * (y * y + z * z);
  m[1] = 2 * (x * y + z * w);
  m[2] = 2 * (x * z - y * w);
  m[4] = 2 * (x * y - z * w);
  m[5] = 1 - 2 * (x * x + z * z);
  m[6] = 2 * (y * z + x * w);
  m[8] = 2 * (x * z + y * w);
  m[9] = 2 * (y * z - x * w);
  m[10] = 1 - 2 * (x * x + y * y);
  m[12] = p[0];
  m[13] = p[1];
  m[14] = p[2];
  m[15] = 1;
  return m;
}

/** View matrix (inverse of the camera pose). */
export function viewMatrix(eye: Vec3, q: Quat): Mat4 {
  const inv: Quat = [-q[0], -q[1], -q[2], q[3]];
  const m = poseMatrix([0, 0, 0], inv);
  const t = rotate(inv, scale(eye, -1));
  m[12] = t[0];
  m[13] = t[1];
  m[14] = t[2];
  return m;
}

/**
 * Perspective projection. `zeroToOne` selects WebGPU clip depth [0, 1];
 * otherwise WebGL clip depth [-1, 1].
 */
export function perspective(fovYDeg: number, aspect: number, near: number, far: number, zeroToOne: boolean): Mat4 {
  const f = 1 / Math.tan((fovYDeg * Math.PI) / 360);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  m[11] = -1;
  if (zeroToOne) {
    m[10] = far / (near - far);
    m[14] = (far * near) / (near - far);
  } else {
    m[10] = (far + near) / (near - far);
    m[14] = (2 * far * near) / (near - far);
  }
  return m;
}

export function transformPoint(m: Mat4, v: Vec3): [number, number, number, number] {
  const x = v[0];
  const y = v[1];
  const z = v[2];
  return [
    m[0]! * x + m[4]! * y + m[8]! * z + m[12]!,
    m[1]! * x + m[5]! * y + m[9]! * z + m[13]!,
    m[2]! * x + m[6]! * y + m[10]! * z + m[14]!,
    m[3]! * x + m[7]! * y + m[11]! * z + m[15]!,
  ];
}

export function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

/** Frame-rate independent exponential smoothing factor. */
export function smoothing(rate: number, dt: number): number {
  return 1 - Math.exp(-rate * dt);
}
