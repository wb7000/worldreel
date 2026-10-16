/** Public surface of the WorldReel player library (the app entry is main.ts). */

export const PLAYER_VERSION = "0.1.0";

export { openWreel, verifySignature, WreelError } from "./wreel/reader";
export type { OpenedWreel, SignatureStatus } from "./wreel/reader";
export type { Manifest, PovMode, PovPreset } from "./wreel/types";

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

/** True when the current environment exposes WebGPU; the player falls back to WebGL2 otherwise. */
export function hasWebGPU(nav: unknown = globalThis.navigator): boolean {
  return typeof nav === "object" && nav !== null && "gpu" in nav;
}
