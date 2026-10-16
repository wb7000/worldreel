import type { PointChunk } from "../wreel/points";
import { POINT_STRIDE } from "../wreel/points";
import { packUniforms, UNIFORM_FLOATS, type DrawItem, type FrameParams, type Renderer } from "./renderer";

const SHADER = /* wgsl */ `
struct U {
  model: mat4x4f,
  view: mat4x4f,
  proj: mat4x4f,
  fogTop: vec4f,
  fogBottom: vec4f,
  params: vec4f, // pointSize, fogStrength, viewportHeight, minPixels
};
@group(0) @binding(0) var<uniform> u: U;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
  @location(1) corner: vec2f,
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
      @location(0) p: vec3f,
      @location(1) c: vec4f,
      @location(2) conf: f32) -> VOut {
  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
    vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
  let corner = corners[vi];
  var clip = u.proj * (u.view * (u.model * vec4f(p, 1.0)));
  let minRadius = u.params.w / u.params.z * clip.w;
  let ry = max(u.proj[1][1] * u.params.x * 0.5, minRadius);
  let rx = ry * u.proj[0][0] / u.proj[1][1];
  clip = vec4f(clip.xy + corner * vec2f(rx, ry), clip.zw);
  var o: VOut;
  o.pos = clip;
  o.color = c.rgb;
  o.corner = corner;
  o.fog = fogAmount(u32(round(c.a * 255.0)), conf) * u.params.y;
  return o;
}

@fragment
fn fs(i: VOut) -> @location(0) vec4f {
  if (dot(i.corner, i.corner) > 1.0) { discard; }
  let screenY = clamp(i.pos.y / u.params.z, 0.0, 1.0);
  let fogColor = mix(u.fogTop.rgb, u.fogBottom.rgb, screenY);
  return vec4f(mix(i.color, fogColor, i.fog), 1.0);
}
`;

interface GpuDrawable {
  buffer: GPUBuffer;
  count: number;
  uniforms: GPUBuffer;
  bindGroup: GPUBindGroup;
}

export async function createWebGPURenderer(canvas: HTMLCanvasElement): Promise<Renderer> {
  if (!("gpu" in navigator) || !navigator.gpu) throw new Error("navigator.gpu is not available");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("no WebGPU adapter");
  const device = await adapter.requestDevice();
  const context = canvas.getContext("webgpu");
  if (!context) throw new Error("could not get a webgpu canvas context");
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "opaque" });

  const module = device.createShaderModule({ code: SHADER });
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }],
  });
  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    vertex: {
      module,
      entryPoint: "vs",
      buffers: [
        {
          arrayStride: POINT_STRIDE,
          stepMode: "instance",
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x3" },
            { shaderLocation: 1, offset: 12, format: "unorm8x4" },
            { shaderLocation: 2, offset: 16, format: "float32" },
          ],
        },
      ],
    },
    fragment: { module, entryPoint: "fs", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
    depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
  });

  const drawables = new Map<string, GpuDrawable>();
  let depth: GPUTexture | null = null;
  let height = canvas.height;

  function resize(w: number, h: number): void {
    canvas.width = Math.max(1, w);
    canvas.height = Math.max(1, h);
    height = canvas.height;
    depth?.destroy();
    depth = device.createTexture({
      size: [canvas.width, canvas.height],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }
  resize(canvas.width, canvas.height);

  function setPoints(id: string, chunk: PointChunk): void {
    const old = drawables.get(id);
    old?.buffer.destroy();
    const buffer = device.createBuffer({
      size: Math.max(POINT_STRIDE, chunk.bytes.byteLength),
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buffer, 0, chunk.bytes);
    const uniforms =
      old?.uniforms ??
      device.createBuffer({ size: UNIFORM_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const bindGroup =
      old?.bindGroup ?? device.createBindGroup({ layout: bindGroupLayout, entries: [{ binding: 0, resource: { buffer: uniforms } }] });
    drawables.set(id, { buffer, count: chunk.count, uniforms, bindGroup });
  }

  function render(frame: FrameParams, items: readonly DrawItem[]): void {
    if (!depth) return;
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context!.getCurrentTexture().createView(),
          clearValue: { r: frame.background[0], g: frame.background[1], b: frame.background[2], a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
      depthStencilAttachment: { view: depth.createView(), depthClearValue: 1, depthLoadOp: "clear", depthStoreOp: "store" },
    });
    pass.setPipeline(pipeline);
    for (const item of items) {
      const d = drawables.get(item.id);
      if (!d || d.count === 0) continue;
      device.queue.writeBuffer(d.uniforms, 0, packUniforms(frame, item, height));
      pass.setBindGroup(0, d.bindGroup);
      pass.setVertexBuffer(0, d.buffer);
      pass.draw(6, d.count);
    }
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  return { kind: "webgpu", zeroToOneDepth: true, setPoints, resize, render };
}
