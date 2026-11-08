/** WorldReel player entry point: load a .wreel, render it, and drive the viewer menu. */

import { AudioEngine } from "./audio";
import { CameraDirector, type FreeInput } from "./cameras";
import { perspective, poseMatrix, viewMatrix, type Vec3 } from "./math";
import { createRenderer, type DrawItem, type Renderer } from "./render/renderer";
import { backToFront, reorder, splatInstances, type GaussianChunk } from "./wreel/gaussians";
import type { PointChunk } from "./wreel/points";
import {
  createRuleEngine,
  moodTension,
  type RuleEngine,
  type RuleEffect,
  branchTarget,
} from "./rules";
import { openWreel, WreelError, type OpenedWreel, type SignatureStatus } from "./wreel/reader";
import { sampleCamera, sampleTrack } from "./wreel/track";
import type { PovPreset } from "./wreel/types";

const FOG_TOP: Vec3 = [0.86, 0.72, 0.98];
const FOG_BOTTOM: Vec3 = [0.56, 0.42, 0.82];
const DEFAULT_SAMPLE = "samples/courtyard.wreel";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = $<HTMLCanvasElement>("view");
const message = $<HTMLDivElement>("message");

const SIGNATURE_TEXT: Record<SignatureStatus, string> = {
  valid: "signature valid",
  invalid: "signature INVALID",
  missing: "unsigned",
  unsupported: "signature not checked (browser lacks Ed25519)",
};

const MODE_HINTS: Record<PovPreset["mode"], string> = {
  "directors-cut": "Playing the original camera path.",
  free: "Drag to look · WASD / arrows to move · Q/E down/up · Shift = faster · wheel = forward.",
  interactive: "Free camera with the scene's interaction rules active.",
  "third-person": "Following the actor from behind.",
  "first-person": "Through the actor's eyes (comfort mode: smoothed turning, narrower view).",
  overhead: "Top-down view of the whole scene.",
};

interface Player {
  wreel: OpenedWreel;
  director: CameraDirector;
  rules: RuleEngine;
  time: number;
  playing: boolean;
  tension: number;
  fogStrength: number;
  branch: string | null; // active story branch, null = main
}

const input: FreeInput = { dragX: 0, dragY: 0, wheel: 0, keys: new Set() };
let renderer: Renderer | null = null;
let player: Player | null = null;
let frameCount = 0;
let fps = 0;
let fpsClock = performance.now();

function showMessage(text: string | null): void {
  message.hidden = text === null;
  message.textContent = text ?? "";
}

function setupInput(): void {
  let dragging = false;
  let lastX = 0;
  let lastY = 0;
  canvas.addEventListener("pointerdown", (e) => {
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    input.dragX += e.clientX - lastX;
    input.dragY += e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
  });
  const end = () => (dragging = false);
  canvas.addEventListener("pointerup", end);
  canvas.addEventListener("pointercancel", end);
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    input.wheel -= e.deltaY;
  }, { passive: false });
  window.addEventListener("keydown", (e) => {
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
    if (e.code === "Space") {
      e.preventDefault();
      togglePlay();
      return;
    }
    input.keys.add(e.code);
  });
  window.addEventListener("keyup", (e) => input.keys.delete(e.code));
  window.addEventListener("blur", () => input.keys.clear());
  canvas.addEventListener("click", (e) => {
    if (!player || !renderer) return;
    // Project each actor's current position to screen and check click proximity.
    const cam = player.director.current();
    if (!cam) return;
    const cx = canvas.clientWidth / 2;
    const cy = canvas.clientHeight / 2;
    const f = (canvas.height / 2) / Math.tan((cam.fov * Math.PI) / 180 / 2);
    for (const actor of player.wreel.manifest.actors) {
      const track = player.wreel.tracks.get(actor.id);
      if (!track) continue;
      const pose = sampleTrack(track, player.time);
      const dx = pose.position[0] - cam.eye[0];
      const dy = pose.position[1] - cam.eye[1];
      const dz = pose.position[2] - cam.eye[2];
      // simple: only actors in front of camera
      if (dz <= 0.1) continue;
      const sx = cx + f * dx / dz;
      const sy = cy + f * dy / dz;
      const dist = Math.hypot(e.clientX - sx, e.clientY - sy);
      if (dist < 60) {
        // select event: fire rules
        for (const eff of player.rules.onEvent("select", actor.id)) {
          handleEffect(eff);
        }
        showMessage(`Selected: ${actor.label}`);
        setTimeout(() => showMessage(null), 1500);
        break;
      }
    }
  });
}

function togglePlay(): void {
  if (!player) return;
  player.playing = !player.playing;
  const btn = $<HTMLButtonElement>("play");
  btn.textContent = player.playing ? "❚❚" : "▶";
  btn.setAttribute("aria-label", player.playing ? "Pause" : "Play");
  if (audio) {
    if (player.playing) audio.play(player.time);
    else audio.stop();
  }
}

/** Sound needs a user gesture before the browser allows it, hence a button. */
let audio: AudioEngine | null = null;
async function toggleSound(): Promise<void> {
  if (!player) return;
  const btn = $<HTMLButtonElement>("sound");
  if (audio) {
    audio.stop();
    audio = null;
    btn.textContent = "🔈 Sound off";
    btn.setAttribute("aria-pressed", "false");
    return;
  }
  btn.textContent = "🔈 Loading…";
  const { manifest } = player.wreel;
  audio = await AudioEngine.create(manifest.audio.tracks, manifest.audio.events, await player.wreel.loadAudio());
  if (player.playing) audio.play(player.time);
  btn.textContent = "🔊 Sound on";
  btn.setAttribute("aria-pressed", "true");
}

function selectPreset(preset: PovPreset): void {
  if (!player) return;
  player.director.setPreset(preset);
  for (const btn of $<HTMLDivElement>("presets").querySelectorAll("button")) {
    btn.setAttribute("aria-pressed", String(btn.dataset.id === preset.id));
  }
  $<HTMLParagraphElement>("hint").textContent = presetHint(preset, player.wreel.manifest.cameras);
}

/** A directors-cut preset may play an alternative path, e.g. a planned follow camera (A-2). */
export function presetHint(preset: PovPreset, cameras: { id: string; kind: string }[]): string {
  if (preset.mode === "directors-cut") {
    const cam = cameras.find((c) => c.id === preset.camera_id);
    if (cam && cam.kind === "alternative") return "Planned camera path that keeps to filmed areas.";
  }
  return MODE_HINTS[preset.mode];
}

function buildMenu(wreel: OpenedWreel): void {
  const box = $<HTMLDivElement>("presets");
  box.replaceChildren();
  for (const preset of wreel.manifest.pov_presets) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = preset.name;
    btn.dataset.id = preset.id;
    btn.addEventListener("click", () => selectPreset(preset));
    box.append(btn);
  }
}

function buildBranches(wreel: OpenedWreel): void {
  const box = $<HTMLDivElement>("branches");
  box.replaceChildren();
  const branchList = wreel.manifest.interaction.branches;
  if (!branchList.length) {
    box.hidden = true;
    return;
  }
  box.hidden = false;
  const label = document.createElement("span");
  label.textContent = "Story branches:";
  label.style.fontWeight = "600";
  label.style.marginRight = "8px";
  box.append(label);
  for (const b of branchList) {
    const target = wreel.manifest.pov_presets.find((p) => p.id === b.switch_to);
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = b.label || `Branch: ${target?.name ?? b.id}`;
    btn.dataset.id = b.id;
    btn.title = b.description ?? "";
    btn.addEventListener("click", () => {
      if (!player) return;
      player.branch = b.id;
      if (target) selectPreset(target);
      showMessage(`Branch: ${target?.name ?? b.id}`);
      setTimeout(() => showMessage(null), 2000);
    });
    box.append(btn);
  }
}

function showProvenance(wreel: OpenedWreel): void {
  const p = wreel.manifest.provenance;
  $("provenance").hidden = false;
  $("ai-label").textContent = p.ai_label;
  $("notice").textContent = p.notice;
  const sig = $("sig");
  sig.textContent = SIGNATURE_TEXT[wreel.signature];
  sig.className = `badge sig ${wreel.signature}`;
  $("scene-title").textContent = wreel.manifest.title;
}

/** Splat layers: decoded instances, a reusable sorted copy, and when/where they were sorted. */
interface SplatState {
  instances: Float32Array;
  sorted: Float32Array;
  lastSortMs: number;
  lastView: Float32Array | null;
}
const splatStates = new Map<string, SplatState>();
const SORT_INTERVAL_MS = 120;

function upload(id: string, kind: string, chunk: PointChunk | GaussianChunk): void {
  if (!renderer) return;
  if (kind === "gaussians") {
    const instances = splatInstances(chunk);
    splatStates.set(id, { instances, sorted: new Float32Array(instances.length), lastSortMs: 0, lastView: null });
  } else {
    splatStates.delete(id);
    renderer.setPoints(id, chunk);
  }
}

function viewChanged(a: Float32Array | null, b: Float32Array): boolean {
  if (!a) return true;
  for (let i = 0; i < 16; i++) if (Math.abs(a[i]! - b[i]!) > 1e-4) return true;
  return false;
}

/** Re-sort splats back to front when the camera moved (at most every SORT_INTERVAL_MS). */
function sortSplats(view: Float32Array, now: number): void {
  if (!renderer) return;
  for (const [id, s] of splatStates) {
    if (s.lastView && (now - s.lastSortMs < SORT_INTERVAL_MS || !viewChanged(s.lastView, view))) continue;
    s.sorted = reorder(s.instances, backToFront(s.instances, view), s.sorted);
    renderer.setSplats(id, s.sorted);
    s.lastSortMs = now;
    s.lastView = view.slice();
  }
}

/** Scene Gaussian layers are merged into one splat set: splats do not write depth, so separately
 * sorted layers would draw on top of each other regardless of distance (the fill layer covered
 * the TV on the walking clip, D-4.3). Point layers and actors upload as before. */
const MERGED_SPLATS = "layer:__splats";

function uploadScene(wreel: OpenedWreel): void {
  const parts: Float32Array[] = [];
  for (const [id, d] of wreel.layers) {
    if (d.layer.kind === "gaussians") parts.push(splatInstances(d.chunk as GaussianChunk));
    else upload(`layer:${id}`, d.layer.kind, d.chunk);
  }
  if (parts.length) {
    const instances = new Float32Array(parts.reduce((n, a) => n + a.length, 0));
    let offset = 0;
    for (const a of parts) {
      instances.set(a, offset);
      offset += a.length;
    }
    splatStates.set(MERGED_SPLATS, { instances, sorted: new Float32Array(instances.length), lastSortMs: 0, lastView: null });
  }
  for (const [id, d] of wreel.actorGeometry) upload(`actor:${id}`, d.layer.kind, d.chunk);
}

async function load(data: Uint8Array): Promise<void> {
  showMessage("Opening…");
  const wreel = await openWreel(data);
  if (!renderer) renderer = await createRenderer(canvas, preferredRenderer());
  splatStates.clear();
  uploadScene(wreel);

  const presets = wreel.manifest.pov_presets;
  const initial =
    presets.find((p) => p.mode === "directors-cut") ?? presets[0] ?? { id: "free", name: "Free camera", mode: "free" as const };
  player = {
    wreel,
    director: new CameraDirector(wreel.manifest, wreel.tracks, initial),
    rules: createRuleEngine(wreel.manifest.interaction.rules),
    time: 0,
    playing: true,
    tension: moodTension(wreel.manifest, 0),
    fogStrength: Number($<HTMLSelectElement>("fog").value),
    branch: null,
  };
  $<HTMLInputElement>("scrub").max = String(wreel.manifest.duration);
  buildMenu(wreel);
  buildBranches(wreel);
  showProvenance(wreel);
  selectPreset(initial);
  for (const w of wreel.warnings) console.warn(w);
  showMessage(null);

  // Coarse LOD is on screen; now stream in the fine LOD (spec §6.1 progressive loading).
  setTimeout(() => {
    void wreel.refine().then((n) => {
      if (!renderer || player?.wreel !== wreel || n === 0) return;
      uploadScene(wreel);
    });
  }, 300);
}

function preferredRenderer(): "auto" | "webgl2" | "webgpu" {
  const r = new URLSearchParams(location.search).get("renderer");
  return r === "webgl2" || r === "webgpu" ? r : "auto";
}

function resize(): void {
  if (!renderer) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.round(canvas.clientWidth * dpr);
  const h = Math.round(canvas.clientHeight * dpr);
  if (w !== canvas.width || h !== canvas.height) renderer.resize(w, h);
}

let last = performance.now();
function frame(now: number): void {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  requestAnimationFrame(frame);
  if (!renderer || !player) return;
  resize();

  const { wreel, director } = player;
  const duration = wreel.manifest.duration;
  const prev = player.time;
  if (player.playing) {
    player.time += dt;
    if (player.time > duration) {
      player.time = 0;
      audio?.play(0); // loop: restart the stems from the top
    }
  }
  if (director.preset.mode === "interactive") {
    for (const effect of player.rules.step(prev, player.time)) {
      const handled = handleEffect(effect);
      if (!handled && effect.type === "set_tension") player.tension = effect.value;
      else if (!handled && effect.type === "switch_pov") {
        const target = wreel.manifest.pov_presets.find((p) => p.id === effect.presetId);
        if (target) selectPreset(target);
      }
    }
  } else {
    player.tension = moodTension(wreel.manifest, player.time);
  }

  const cam = director.update(player.time, dt, input);
  audio?.update(player.time, cam);
  input.dragX = input.dragY = input.wheel = 0;

  const aspect = canvas.width / Math.max(1, canvas.height);
  // Far plane 1000 m: splat scenes keep a background sphere about ten scene sizes away.
  const proj = perspective(cam.fov, aspect, 0.05, 1000, renderer.zeroToOneDepth);
  const view = viewMatrix(cam.eye, cam.rotation);
  sortSplats(view, now);
  const items: DrawItem[] = [];
  const identity = poseMatrix([0, 0, 0], [0, 0, 0, 1]);
  let splatsDrawn = false;
  for (const [id, d] of wreel.layers) {
    if (d.layer.kind === "gaussians") {
      if (!splatsDrawn) items.push({ id: MERGED_SPLATS, model: identity, pointSize: 0.05, kind: "splats" });
      splatsDrawn = true;
      continue;
    }
    items.push({ id: `layer:${id}`, model: identity, pointSize: d.layer.point_size ?? 0.05, kind: "points" });
  }
  for (const actor of wreel.manifest.actors) {
    if (actor.id === cam.hiddenActor) continue;
    const track = wreel.tracks.get(actor.id);
    if (!track) continue;
    const pose = sampleTrack(track, player.time);
    items.push({
      id: `actor:${actor.id}`,
      model: poseMatrix(pose.position, pose.rotation),
      pointSize: actor.geometry.point_size ?? 0.05,
      kind: actor.geometry.kind === "gaussians" ? "splats" : "points",
    });
  }
  renderer.render(
    {
      view,
      proj,
      background: wreel.manifest.scene.background,
      fogTop: FOG_TOP,
      fogBottom: FOG_BOTTOM,
      fogStrength: player.fogStrength,
    },
    items,
  );

  const scrub = $<HTMLInputElement>("scrub");
  if (document.activeElement !== scrub) scrub.value = String(player.time);
  $("clock").textContent = `${player.time.toFixed(1)} / ${duration.toFixed(1)} s`;

  frameCount++;
  if (now - fpsClock > 500) {
    fps = (frameCount * 1000) / (now - fpsClock);
    frameCount = 0;
    fpsClock = now;
    const layers = [...wreel.layers.values(), ...wreel.actorGeometry.values()];
    const points = layers.reduce((n, d) => n + d.chunk.count, 0);
    const level = Math.min(...layers.map((d) => d.level));
    $("stats").textContent = [
      `renderer  ${renderer.kind}`,
      `fps       ${fps.toFixed(0)}`,
      `points    ${points.toLocaleString()}`,
      `LOD       ${level}${level === 0 ? " (coarse)" : ""}`,
      `tension   ${"▮".repeat(Math.round(player.tension))}${"▯".repeat(10 - Math.round(player.tension))}`,
    ].join("\n");
  }
}

async function loadFromUrl(url: string): Promise<void> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new WreelError(
      `Could not load ${url} (${res.status}). Generate the sample with:\n` +
        "python -m tools.make_sample_wreel\n(from the repository root), or open a .wreel file.",
    );
  }
  await load(new Uint8Array(await res.arrayBuffer()));
}

/** Apply a rule effect: set_tension, switch_pov, play_event, branch. Returns true if handled. */
function handleEffect(effect: RuleEffect): boolean {
  if (!player) return false;
  const { wreel } = player;
  switch (effect.type) {
    case "set_tension":
      player.tension = effect.value;
      return true;
    case "switch_pov": {
      const target = wreel.manifest.pov_presets.find((p) => p.id === effect.presetId);
      if (target) selectPreset(target);
      return true;
    }
    case "play_event": {
      // Play a one-shot audio event at the named event's position/time.
      const ev = wreel.manifest.audio.events.find((e) => e.id === effect.eventId);
      if (ev && audio) audio.playEvent(ev);
      return true;
    }
    case "branch": {
      const target = branchTarget(wreel.manifest, effect.branchId);
      if (target) {
        player.branch = effect.branchId;
        selectPreset(target);
        showMessage(`Story branch: ${target.name}`);
        setTimeout(() => showMessage(null), 2000);
      }
      return true;
    }
  }
  return false;
}

function reportError(err: unknown): void {
  console.error(err);
  showMessage(err instanceof Error ? err.message : String(err));
}

function main(): void {
  setupInput();
  window.addEventListener("resize", resize);
  $<HTMLButtonElement>("play").addEventListener("click", togglePlay);
  $<HTMLInputElement>("scrub").addEventListener("input", (e) => {
    if (!player) return;
    player.time = Number((e.target as HTMLInputElement).value);
    if (audio && player.playing) audio.play(player.time);
  });
  $<HTMLButtonElement>("sound").addEventListener("click", () => {
    toggleSound().catch((err: unknown) => {
      console.error(err);
      showMessage(err instanceof Error ? err.message : String(err));
    });
  });
  $<HTMLSelectElement>("fog").addEventListener("change", (e) => {
    if (player) player.fogStrength = Number((e.target as HTMLSelectElement).value);
  });
  $<HTMLInputElement>("file-input").addEventListener("change", (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (file) void file.arrayBuffer().then((b) => load(new Uint8Array(b))).catch(reportError);
  });
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => {
    e.preventDefault();
    const file = e.dataTransfer?.files[0];
    if (file) void file.arrayBuffer().then((b) => load(new Uint8Array(b))).catch(reportError);
  });
  requestAnimationFrame(frame);
  const url = new URLSearchParams(location.search).get("file") ?? DEFAULT_SAMPLE;
  loadFromUrl(url).catch(reportError);
}

main();
