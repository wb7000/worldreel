import { describe, expect, it } from "vitest";
import { hasWebGPU, isViewMode, VIEW_MODES } from "./index";

describe("viewer menu", () => {
  it("includes the brief's view modes", () => {
    expect(VIEW_MODES).toContain("directors-cut");
    expect(VIEW_MODES).toContain("free");
    expect(VIEW_MODES).toContain("first-person");
  });

  it("validates mode strings", () => {
    expect(isViewMode("overhead")).toBe(true);
    expect(isViewMode("vr")).toBe(false);
  });
});

describe("WebGPU detection", () => {
  it("detects gpu on a navigator-like object", () => {
    expect(hasWebGPU({ gpu: {} })).toBe(true);
    expect(hasWebGPU({})).toBe(false);
    expect(hasWebGPU(undefined)).toBe(false);
  });
});
