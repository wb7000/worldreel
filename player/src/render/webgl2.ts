import type { PointChunk } from "../wreel/points";
import { POINT_STRIDE } from "../wreel/points";
import { drawOrder, packUniforms, type DrawItem, type FrameParams, type Renderer } from "./renderer";
import { SPLAT_FS, SPLAT_VS } from "./splatShaders";
import { SPLAT_FLOATS } from "../wreel/gaussians";

const VS = `#version 300 es
precision highp float;
layout(location = 0) in vec3 p;
layout(location = 1) in vec4 c;
layout(location = 2) in float conf;
uniform mat4 uModel, uView, uProj;
uniform vec4 uParams; // pointSize, fogStrength, viewportHeight, minPixels
out vec3 vColor;
out vec2 vCorner;
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
  vec2 corner = CORNERS[gl_VertexID];
  vec4 clip = uProj * (uView * (uModel * vec4(p, 1.0)));
  float minRadius = uParams.w / uParams.z * clip.w;
  float ry = max(uProj[1][1] * uParams.x * 0.5, minRadius);
  float rx = ry * uProj[0][0] / uProj[1][1];
  clip.xy += corner * vec2(rx, ry);
  gl_Position = clip;
  vColor = c.rgb;
  vCorner = corner;
  vFog = fogAmount(uint(round(c.a * 255.0)), conf) * uParams.y;
}`;

const FS = `#version 300 es
precision highp float;
in vec3 vColor;
in vec2 vCorner;
in float vFog;
uniform vec4 uFogTop, uFogBottom, uParams;
out vec4 outColor;
void main() {
  if (dot(vCorner, vCorner) > 1.0) discard;
  float screenY = clamp(1.0 - gl_FragCoord.y / uParams.z, 0.0, 1.0);
  vec3 fogColor = mix(uFogTop.rgb, uFogBottom.rgb, screenY);
  outColor = vec4(mix(vColor, fogColor, vFog), 1.0);
}`;

function compile(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(`shader compile failed: ${gl.getShaderInfoLog(shader) ?? ""}`);
  }
  return shader;
}

interface GlDrawable {
  vao: WebGLVertexArrayObject;
  buffer: WebGLBuffer;
  count: number;
}

function link(gl: WebGL2RenderingContext, vs: string, fs: string): WebGLProgram {
  const program = gl.createProgram()!;
  gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(`program link failed: ${gl.getProgramInfoLog(program) ?? ""}`);
  }
  return program;
}

function uniforms(gl: WebGL2RenderingContext, program: WebGLProgram) {
  const loc = (name: string) => gl.getUniformLocation(program, name);
  return {
    model: loc("uModel"),
    view: loc("uView"),
    proj: loc("uProj"),
    fogTop: loc("uFogTop"),
    fogBottom: loc("uFogBottom"),
    params: loc("uParams"),
    viewport: loc("uViewport"),
  };
}

export function createWebGL2Renderer(canvas: HTMLCanvasElement): Renderer {
  const gl = canvas.getContext("webgl2", { antialias: true });
  if (!gl) throw new Error("WebGL2 is not available in this browser");
  const program = link(gl, VS, FS);
  const splatProgram = link(gl, SPLAT_VS, SPLAT_FS);
  const pointU = uniforms(gl, program);
  const splatU = uniforms(gl, splatProgram);

  const drawables = new Map<string, GlDrawable>();
  const splats = new Map<string, GlDrawable>();
  gl.enable(gl.DEPTH_TEST);

  function setPoints(id: string, chunk: PointChunk): void {
    const old = drawables.get(id);
    if (old) {
      gl!.deleteBuffer(old.buffer);
      gl!.deleteVertexArray(old.vao);
    }
    const vao = gl!.createVertexArray()!;
    const buffer = gl!.createBuffer()!;
    gl!.bindVertexArray(vao);
    gl!.bindBuffer(gl!.ARRAY_BUFFER, buffer);
    gl!.bufferData(gl!.ARRAY_BUFFER, chunk.bytes, gl!.STATIC_DRAW);
    gl!.enableVertexAttribArray(0);
    gl!.vertexAttribPointer(0, 3, gl!.FLOAT, false, POINT_STRIDE, 0);
    gl!.vertexAttribDivisor(0, 1);
    gl!.enableVertexAttribArray(1);
    gl!.vertexAttribPointer(1, 4, gl!.UNSIGNED_BYTE, true, POINT_STRIDE, 12);
    gl!.vertexAttribDivisor(1, 1);
    gl!.enableVertexAttribArray(2);
    gl!.vertexAttribPointer(2, 1, gl!.FLOAT, false, POINT_STRIDE, 16);
    gl!.vertexAttribDivisor(2, 1);
    gl!.bindVertexArray(null);
    drawables.set(id, { vao, buffer, count: chunk.count });
  }

  function setSplats(id: string, instances: Float32Array): void {
    const count = instances.length / SPLAT_FLOATS;
    const existing = splats.get(id);
    if (existing && existing.count === count) {
      gl!.bindBuffer(gl!.ARRAY_BUFFER, existing.buffer);
      gl!.bufferSubData(gl!.ARRAY_BUFFER, 0, instances);
      return;
    }
    if (existing) {
      gl!.deleteBuffer(existing.buffer);
      gl!.deleteVertexArray(existing.vao);
    }
    const vao = gl!.createVertexArray()!;
    const buffer = gl!.createBuffer()!;
    gl!.bindVertexArray(vao);
    gl!.bindBuffer(gl!.ARRAY_BUFFER, buffer);
    gl!.bufferData(gl!.ARRAY_BUFFER, instances, gl!.DYNAMIC_DRAW);
    const stride = SPLAT_FLOATS * 4;
    const layout: [number, number, number][] = [
      [0, 3, 0],
      [1, 3, 12],
      [2, 3, 24],
      [3, 4, 36],
      [4, 2, 52],
    ];
    for (const [location, size, offset] of layout) {
      gl!.enableVertexAttribArray(location);
      gl!.vertexAttribPointer(location, size, gl!.FLOAT, false, stride, offset);
      gl!.vertexAttribDivisor(location, 1);
    }
    gl!.bindVertexArray(null);
    splats.set(id, { vao, buffer, count });
  }

  function resize(w: number, h: number): void {
    canvas.width = Math.max(1, w);
    canvas.height = Math.max(1, h);
    gl!.viewport(0, 0, canvas.width, canvas.height);
  }

  function render(frame: FrameParams, items: readonly DrawItem[]): void {
    gl!.clearColor(frame.background[0], frame.background[1], frame.background[2], 1);
    gl!.clear(gl!.COLOR_BUFFER_BIT | gl!.DEPTH_BUFFER_BIT);
    gl!.depthMask(true);
    gl!.disable(gl!.BLEND);
    let current: WebGLProgram | null = null;
    for (const item of drawOrder(items)) {
      const isSplat = item.kind === "splats";
      const d = (isSplat ? splats : drawables).get(item.id);
      if (!d || d.count === 0) continue;
      const prog = isSplat ? splatProgram : program;
      const loc = isSplat ? splatU : pointU;
      if (prog !== current) {
        gl!.useProgram(prog);
        current = prog;
        if (isSplat) {
          // Back-to-front "over" compositing; splats test against points but do not write depth.
          gl!.enable(gl!.BLEND);
          gl!.blendFuncSeparate(gl!.SRC_ALPHA, gl!.ONE_MINUS_SRC_ALPHA, gl!.ONE, gl!.ONE_MINUS_SRC_ALPHA);
          gl!.depthMask(false);
        }
      }
      const u = packUniforms(frame, item, canvas.height, canvas.width);
      gl!.uniformMatrix4fv(loc.model, false, u.subarray(0, 16));
      gl!.uniformMatrix4fv(loc.view, false, u.subarray(16, 32));
      gl!.uniformMatrix4fv(loc.proj, false, u.subarray(32, 48));
      gl!.uniform4fv(loc.fogTop, u.subarray(48, 52));
      gl!.uniform4fv(loc.fogBottom, u.subarray(52, 56));
      gl!.uniform4fv(loc.params, u.subarray(56, 60));
      if (loc.viewport) gl!.uniform4fv(loc.viewport, u.subarray(60, 64));
      gl!.bindVertexArray(d.vao);
      gl!.drawArraysInstanced(gl!.TRIANGLES, 0, 6, d.count);
    }
    gl!.bindVertexArray(null);
    gl!.depthMask(true);
    gl!.disable(gl!.BLEND);
  }

  return { kind: "webgl2", zeroToOneDepth: false, setPoints, setSplats, resize, render };
}
