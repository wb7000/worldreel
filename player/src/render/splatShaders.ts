/**
 * Gaussian splat shaders (GLSL ES 3.0 and WGSL), identical maths:
 * project the 3D covariance to screen space with the perspective Jacobian (EWA splatting,
 * Zwicker et al. 2001, as in 3D Gaussian Splatting), add a 0.3 px² low-pass, take its
 * eigen-decomposition and draw a quad spanning ±3σ along both axes. The fragment alpha is
 * opacity · exp(-½ d²) with d in σ units; blending is back to front (instances are pre-sorted).
 *
 * Camera space follows WorldReel: x right, y up, looking along -z, so depth is -z.
 * Instance layout (16 floats): centre(3) cov xx xy xz (3) cov yy yz zz (3) rgba(4) class conf pad.
 */

export const SPLAT_VS = `#version 300 es
precision highp float;
layout(location = 0) in vec3 aCenter;
layout(location = 1) in vec3 aCovA;
layout(location = 2) in vec3 aCovB;
layout(location = 3) in vec4 aColor;
layout(location = 4) in vec2 aMeta;
uniform mat4 uModel, uView, uProj;
uniform vec4 uParams;   // unused, fogStrength, viewportHeight, unused
uniform vec4 uViewport; // width, height
out vec4 vColor;
out vec2 vD;
out float vFog;

const vec2 CORNERS[6] = vec2[6](
  vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
  vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0));

float fogAmount(uint cls, float conf) {
  if (cls == 0u && conf >= 0.5) return 0.0;
  float a = 1.0 - clamp(conf, 0.0, 1.0);
  if (cls >= 2u) a = max(a, 0.6);
  return clamp(a, 0.0, 0.92);
}

void main() {
  mat4 mv = uView * uModel;
  vec4 cam = mv * vec4(aCenter, 1.0);
  float z = -cam.z;
  if (z < 0.02) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
  vec4 clip = uProj * cam;
  float fx = uProj[0][0] * uViewport.x * 0.5;
  float fy = uProj[1][1] * uViewport.y * 0.5;
  mat3 sigma = mat3(aCovA.x, aCovA.y, aCovA.z, aCovA.y, aCovB.x, aCovB.y, aCovA.z, aCovB.y, aCovB.z);
  mat3 w = mat3(mv);
  // Jacobian of (fx·x/z, fy·y/z) with z = -cam.z; columns of a column-major mat3.
  mat3 j = mat3(fx / z, 0.0, 0.0,  0.0, fy / z, 0.0,  fx * cam.x / (z * z), fy * cam.y / (z * z), 0.0);
  mat3 t = j * w;
  mat3 cov = t * sigma * transpose(t);
  float a = cov[0][0] + 0.3;
  float b = cov[0][1];
  float c = cov[1][1] + 0.3;
  float mid = 0.5 * (a + c);
  float rad = length(vec2(0.5 * (a - c), b));
  float l1 = mid + rad;
  float l2 = max(mid - rad, 0.1);
  vec2 v1 = abs(b) > 1e-8 ? normalize(vec2(b, l1 - a)) : (a >= c ? vec2(1.0, 0.0) : vec2(0.0, 1.0));
  vec2 major = 3.0 * sqrt(l1) * v1;
  vec2 minor = 3.0 * sqrt(l2) * vec2(-v1.y, v1.x);
  vec2 corner = CORNERS[gl_VertexID];
  vec2 offsetPx = corner.x * major + corner.y * minor;
  if (max(length(major), length(minor)) > 2048.0) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
  clip.xy += offsetPx / uViewport.xy * 2.0 * clip.w;
  gl_Position = clip;
  vColor = aColor;
  vD = corner * 3.0;
  vFog = fogAmount(uint(aMeta.x + 0.5), aMeta.y) * uParams.y;
}`;

export const SPLAT_FS = `#version 300 es
precision highp float;
in vec4 vColor;
in vec2 vD;
in float vFog;
uniform vec4 uFogTop, uFogBottom, uParams;
out vec4 outColor;
void main() {
  float alpha = vColor.a * exp(-0.5 * dot(vD, vD));
  if (alpha < 1.0 / 255.0) discard;
  float screenY = clamp(1.0 - gl_FragCoord.y / uParams.z, 0.0, 1.0);
  vec3 fogColor = mix(uFogTop.rgb, uFogBottom.rgb, screenY);
  outColor = vec4(mix(vColor.rgb, fogColor, vFog), min(alpha, 0.99));
}`;

export const SPLAT_WGSL = /* wgsl */ `
struct U {
  model: mat4x4f,
  view: mat4x4f,
  proj: mat4x4f,
  fogTop: vec4f,
  fogBottom: vec4f,
  params: vec4f,
  viewport: vec4f,
};
@group(0) @binding(0) var<uniform> u: U;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec4f,
  @location(1) d: vec2f,
  @location(2) fog: f32,
};

fn fogAmount(cls: u32, conf: f32) -> f32 {
  if (cls == 0u && conf >= 0.5) { return 0.0; }
  var a = 1.0 - clamp(conf, 0.0, 1.0);
  if (cls >= 2u) { a = max(a, 0.6); }
  return clamp(a, 0.0, 0.92);
}

@vertex
fn vs(@builtin(vertex_index) vi: u32,
      @location(0) center: vec3f,
      @location(1) covA: vec3f,
      @location(2) covB: vec3f,
      @location(3) color: vec4f) -> VOut {
  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
    vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
  var o: VOut;
  let mv = u.view * u.model;
  let cam = mv * vec4f(center, 1.0);
  let z = -cam.z;
  if (z < 0.02) { o.pos = vec4f(0.0, 0.0, 2.0, 1.0); return o; }
  var clip = u.proj * cam;
  let fx = u.proj[0][0] * u.viewport.x * 0.5;
  let fy = u.proj[1][1] * u.viewport.y * 0.5;
  let sigma = mat3x3f(covA.x, covA.y, covA.z, covA.y, covB.x, covB.y, covA.z, covB.y, covB.z);
  let w = mat3x3f(mv[0].xyz, mv[1].xyz, mv[2].xyz);
  let j = mat3x3f(fx / z, 0.0, 0.0,  0.0, fy / z, 0.0,  fx * cam.x / (z * z), fy * cam.y / (z * z), 0.0);
  let t = j * w;
  let cov = t * sigma * transpose(t);
  let a = cov[0][0] + 0.3;
  let b = cov[0][1];
  let c = cov[1][1] + 0.3;
  let mid = 0.5 * (a + c);
  let rad = length(vec2f(0.5 * (a - c), b));
  let l1 = mid + rad;
  let l2 = max(mid - rad, 0.1);
  var v1 = vec2f(1.0, 0.0);
  if (abs(b) > 1e-8) { v1 = normalize(vec2f(b, l1 - a)); } else if (a < c) { v1 = vec2f(0.0, 1.0); }
  let major = 3.0 * sqrt(l1) * v1;
  let minor = 3.0 * sqrt(l2) * vec2f(-v1.y, v1.x);
  if (max(length(major), length(minor)) > 2048.0) { o.pos = vec4f(0.0, 0.0, 2.0, 1.0); return o; }
  let corner = corners[vi];
  let offsetPx = corner.x * major + corner.y * minor;
  clip = vec4f(clip.xy + offsetPx / u.viewport.xy * 2.0 * clip.w, clip.zw);
  o.pos = clip;
  o.color = color;
  o.d = corner * 3.0;
  o.fog = fogAmount(1u, 1.0) * u.params.y;
  return o;
}

@fragment
fn fs(i: VOut) -> @location(0) vec4f {
  let alpha = i.color.a * exp(-0.5 * dot(i.d, i.d));
  if (alpha < 1.0 / 255.0) { discard; }
  let screenY = clamp(i.pos.y / u.params.z, 0.0, 1.0);
  let fogColor = mix(u.fogTop.rgb, u.fogBottom.rgb, screenY);
  return vec4f(mix(i.color.rgb, fogColor, i.fog), min(alpha, 0.99));
}
`;
