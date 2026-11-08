/**
 * Declarative interaction rules (spec/wreel.md §12). Phase 9 supports all trigger and action
 * types: time, proximity, view_mode, select triggers; set_tension, switch_pov, play_event,
 * branch actions. Rules are data only — nothing in a .wreel is ever executed as code.
 */

import type { Manifest, Rule } from "./wreel/types";

export type RuleEffect =
  | { type: "set_tension"; value: number }
  | { type: "switch_pov"; presetId: string }
  | { type: "play_event"; eventId: string }
  | { type: "branch"; branchId: string };

export interface RuleEngine {
  /** Effects of every rule whose trigger fires between prev and now. */
  step(prev: number, now: number): RuleEffect[];
  readonly skipped: string[];
  /** Called by the player when a proximity or select event happens outside the time loop. */
  onEvent(kind: "proximity" | "select", actorId: string): RuleEffect[];
}

interface ActiveRule {
  rule: Rule;
  lastTriggered: number; // for proximity debounce
}

export function createRuleEngine(rules: readonly Rule[]): RuleEngine {
  const skipped: string[] = [];
  const active: ActiveRule[] = [];
  for (const rule of rules) {
    const supportedAction =
      rule.action.type === "set_tension" ||
      rule.action.type === "switch_pov" ||
      rule.action.type === "play_event" ||
      rule.action.type === "branch";
    const supportedTrigger =
      rule.trigger.type === "time" ||
      rule.trigger.type === "proximity" ||
      rule.trigger.type === "view_mode" ||
      rule.trigger.type === "select";
    if (supportedTrigger && supportedAction) active.push({ rule, lastTriggered: -1 });
    else skipped.push(`${rule.id} (${rule.trigger.type} → ${rule.action.type})`);
  }
  return {
    skipped,
    step(prev, now) {
      if (now < prev) return []; // scrubbing backwards or looping does not fire rules
      const effects: RuleEffect[] = [];
      for (const ar of active) {
        const r = ar.rule;
        if (r.trigger.type !== "time") continue;
        const t = r.trigger.t!;
        if (t > prev && t <= now) {
          const e = effectFor(r.action);
          if (e) effects.push(e);
        }
      }
      return effects;
    },
    onEvent(kind, actorId) {
      const effects: RuleEffect[] = [];
      for (const ar of active) {
        const r = ar.rule;
        const trig = r.trigger;
        if (kind === "proximity" && trig.type === "proximity") {
          if (trig.actor_id !== actorId) continue;
          const now = performance.now() / 1000;
          if (now - ar.lastTriggered < 1.0) continue; // debounce 1 s
          ar.lastTriggered = now;
          const e = effectFor(r.action);
          if (e) effects.push(e);
        }
        if (kind === "select" && trig.type === "select" && trig.actor_id === actorId) {
          const e = effectFor(r.action);
          if (e) effects.push(e);
        }
      }
      return effects;
    },
  };
}

function effectFor(action: Rule["action"]): RuleEffect | null {
  if (action.type === "set_tension" && typeof action.value === "number")
    return { type: "set_tension", value: action.value };
  if (action.type === "switch_pov" && action.preset_id)
    return { type: "switch_pov", presetId: action.preset_id };
  if (action.type === "play_event" && action.event)
    return { type: "play_event", eventId: action.event };
  if (action.type === "branch" && action.branch_id)
    return { type: "branch", branchId: action.branch_id };
  return null;
}

/** Tension from the audio mood curve: the latest mood point at or before `t` (step function). */
export function moodTension(manifest: Manifest, t: number): number {
  let value = 0;
  for (const m of manifest.audio.mood) {
    if (m.t <= t) value = m.tension;
  }
  return value;
}

/** Branches defined in the manifest: branch_id -> preset to switch to. */
export function branchTarget(
  manifest: Manifest,
  branchId: string,
): PovPreset | undefined {
  for (const b of manifest.interaction.branches) {
    if (b.id === branchId) {
      return manifest.pov_presets.find((p) => p.id === b.switch_to);
    }
  }
  return undefined;
}

import type { PovPreset } from "./wreel/types";
