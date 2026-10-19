/**
 * Layered, positional audio (spec §11, brief §7) with the Web Audio API.
 *
 * - Stems (ambience, music, effects, dialogue) play as tracks synchronised to the player clock.
 *   The pre-mixed foley stem is skipped when positional footstep events exist, so steps are not
 *   heard twice.
 * - Sound events with a position and an `attributes.clip` play through an HRTF PannerNode at
 *   their position, scheduled slightly ahead of the clock.
 * - The listener is the current view: `CameraState.eye` is the character's head in first person
 *   and the camera otherwise, so first- and third-person audio differ by construction (brief §7).
 *
 * The geometry (listener frame, distance gain, azimuth) is in pure functions for tests; the
 * gain follows the Web Audio "inverse" distance model so tests predict what the panner does.
 */

import type { CameraState } from "./cameras";
import { length, normalize, rotate, sub, type Vec3 } from "./math";
import type { AudioEvent, AudioTrack } from "./wreel/types";

export const REF_DISTANCE = 1;
export const ROLLOFF = 1;
const LOOKAHEAD_S = 0.25;

export interface ListenerFrame {
  position: Vec3;
  forward: Vec3;
  up: Vec3;
}

/** Where the viewer hears from: the current view's eye, facing where it looks. */
export function listenerFrame(cam: CameraState): ListenerFrame {
  return { position: cam.eye, forward: rotate(cam.rotation, [0, 0, -1]), up: rotate(cam.rotation, [0, 1, 0]) };
}

/** Web Audio "inverse" distance model: ref / (ref + rolloff * (max(d, ref) - ref)). */
export function distanceGain(d: number, ref = REF_DISTANCE, rolloff = ROLLOFF): number {
  return ref / (ref + rolloff * (Math.max(d, ref) - ref));
}

/** Distance, gain (dB) and azimuth (degrees, + = to the right) of a source for a listener. */
export function heard(listener: ListenerFrame, source: Vec3): { distance: number; gainDb: number; azimuthDeg: number } {
  const v = sub(source, listener.position);
  const d = length(v);
  const right = normalize(cross(listener.forward, listener.up));
  const f = dot(v, listener.forward);
  const r = dot(v, right);
  return {
    distance: d,
    gainDb: 20 * Math.log10(distanceGain(d)),
    azimuthDeg: (Math.atan2(r, f) * 180) / Math.PI,
  };
}

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

/** Events that start in (from, to] — the scheduling window. */
export function eventsInWindow(events: AudioEvent[], from: number, to: number): AudioEvent[] {
  return events.filter((e) => e.t > from && e.t <= to && e.position && typeof e.attributes?.["clip"] === "string");
}

/** Stems to play: all tracks except foley when positional events will carry the footsteps. */
export function stemsToPlay(tracks: AudioTrack[], events: AudioEvent[]): AudioTrack[] {
  const positional = events.some((e) => e.position && typeof e.attributes?.["clip"] === "string");
  return tracks.filter((t) => !(positional && t.layer === "foley"));
}

const LAYER_GAIN: Record<string, number> = { dialogue: 1, foley: 0.8, effects: 0.8, ambience: 0.5, music: 0.4 };

export class AudioEngine {
  private ctx: AudioContext;
  private master: GainNode;
  private buffers = new Map<string, AudioBuffer>();
  private running: AudioBufferSourceNode[] = [];
  private scheduledUntil = 0;
  private startedAt = 0; // context time when playback time 0 would have been
  private playing = false;

  private constructor(
    ctx: AudioContext,
    private readonly tracks: AudioTrack[],
    private readonly events: AudioEvent[],
  ) {
    this.ctx = ctx;
    this.master = ctx.createGain();
    this.master.connect(ctx.destination);
  }

  /** Create on a user gesture (browsers block audio before one). */
  static async create(tracks: AudioTrack[], events: AudioEvent[], files: Map<string, Uint8Array>): Promise<AudioEngine> {
    const engine = new AudioEngine(new AudioContext(), tracks, events);
    for (const [path, bytes] of files) {
      const copy = bytes.slice().buffer; // decodeAudioData detaches its input
      engine.buffers.set(path, await engine.ctx.decodeAudioData(copy));
    }
    return engine;
  }

  /** Start (or restart after a seek) at playback time `t`. */
  play(t: number): void {
    this.stop();
    void this.ctx.resume();
    this.playing = true;
    const now = this.ctx.currentTime;
    this.startedAt = now - t;
    for (const track of stemsToPlay(this.tracks, this.events)) {
      const buffer = this.buffers.get(track.path);
      if (!buffer) continue;
      const offset = t - track.start;
      if (offset >= buffer.duration) continue;
      const src = this.ctx.createBufferSource();
      src.buffer = buffer;
      const gain = this.ctx.createGain();
      gain.gain.value = LAYER_GAIN[track.layer] ?? 0.5;
      src.connect(gain).connect(this.master);
      src.start(now + Math.max(0, -offset), Math.max(0, offset));
      this.running.push(src);
    }
    this.scheduledUntil = t;
  }

  stop(): void {
    for (const src of this.running) {
      try {
        src.stop();
      } catch {
        // already stopped
      }
    }
    this.running = [];
    this.playing = false;
  }

  /** Per frame: move the listener with the view and schedule upcoming positional events. */
  update(t: number, cam: CameraState): void {
    const l = listenerFrame(cam);
    const listener = this.ctx.listener;
    if (listener.positionX) {
      listener.positionX.value = l.position[0];
      listener.positionY.value = l.position[1];
      listener.positionZ.value = l.position[2];
      listener.forwardX.value = l.forward[0];
      listener.forwardY.value = l.forward[1];
      listener.forwardZ.value = l.forward[2];
      listener.upX.value = l.up[0];
      listener.upY.value = l.up[1];
      listener.upZ.value = l.up[2];
    }
    if (!this.playing) return;
    const until = t + LOOKAHEAD_S;
    for (const e of eventsInWindow(this.events, this.scheduledUntil, until)) {
      const buffer = this.buffers.get(String(e.attributes!["clip"]));
      if (!buffer || !e.position) continue;
      const src = this.ctx.createBufferSource();
      src.buffer = buffer;
      const panner = new PannerNode(this.ctx, {
        panningModel: "HRTF",
        distanceModel: "inverse",
        refDistance: REF_DISTANCE,
        rolloffFactor: ROLLOFF,
        positionX: e.position[0],
        positionY: e.position[1],
        positionZ: e.position[2],
      });
      src.connect(panner).connect(this.master);
      src.start(Math.max(this.ctx.currentTime, this.startedAt + e.t));
      src.onended = () => {
        this.running = this.running.filter((s) => s !== src);
      };
      this.running.push(src);
    }
    this.scheduledUntil = Math.max(this.scheduledUntil, until);
  }

  setVolume(v: number): void {
    this.master.gain.value = v;
  }
}
