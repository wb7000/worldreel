/**
 * Declarative interaction rules (spec/wreel.md §12). Phase 1 supports `time` triggers with
 * `set_tension` and `switch_pov` actions; other trigger/action types are reported as skipped.
 * Rules are data only — nothing in a .wreel is ever executed as code.
 */

import type { Manifest, Rule } from "./wreel/types";

export type RuleEffect = { type: "set_tension"; value: number } | { type: "switch_pov"; presetId: string };

export interface RuleEngine {
  /** Effects of every rule whose time trigger lies in (prev, now]. */
  step(prev: number, now: number): RuleEffect[];
  readonly skipped: string[];
}

export function createRuleEngine(rules: readonly Rule[]): RuleEngine {
  const skipped: string[] = [];
  const active: Rule[] = [];
  for (const rule of rules) {
    const supportedAction = rule.action.type === "set_tension" || rule.action.type === "switch_pov";
    if (rule.trigger.type === "time" && typeof rule.trigger.t === "number" && supportedAction) active.push(rule);
    else skipped.push(`${rule.id} (${rule.trigger.type} → ${rule.action.type})`);
  }
  return {
    skipped,
    step(prev, now) {
      if (now < prev) return []; // scrubbing backwards or looping does not fire rules
      const effects: RuleEffect[] = [];
      for (const rule of active) {
        const t = rule.trigger.t!;
        if (t > prev && t <= now) {
          if (rule.action.type === "set_tension" && typeof rule.action.value === "number") {
            effects.push({ type: "set_tension", value: rule.action.value });
          } else if (rule.action.type === "switch_pov" && rule.action.preset_id) {
            effects.push({ type: "switch_pov", presetId: rule.action.preset_id });
          }
        }
      }
      return effects;
    },
  };
}

/** Tension from the audio mood curve: the latest mood point at or before `t` (step function). */
export function moodTension(manifest: Manifest, t: number): number {
  let value = 0;
  for (const m of manifest.audio.mood) {
    if (m.t <= t) value = m.tension;
  }
  return value;
}
