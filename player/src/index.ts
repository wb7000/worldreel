/**
 * WorldReel player entry point. Phase 0 skeleton: only the viewer-menu vocabulary exists.
 * Rendering, .wreel loading and POV switching arrive in Phase 1.
 */

export const PLAYER_VERSION = "0.0.1";

/** The viewer menu from the brief ("How would you like to see this?"). */
export const VIEW_MODES = [
  "directors-cut",
  "first-person",
  "third-person",
  "free",
  "overhead",
  "interactive",
] as const;

export type ViewMode = (typeof VIEW_MODES)[number];

export function isViewMode(value: string): value is ViewMode {
  return (VIEW_MODES as readonly string[]).includes(value);
}

/** True when the current environment exposes WebGPU; the player falls back otherwise. */
export function hasWebGPU(nav: unknown = globalThis.navigator): boolean {
  return typeof nav === "object" && nav !== null && "gpu" in nav;
}
