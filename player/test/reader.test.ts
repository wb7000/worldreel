import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { classHistogram, decodePoint } from "../src/wreel/points";
import { openWreel, WreelError } from "../src/wreel/reader";
import { buildWreel, sampleManifest } from "./fixtures";

describe("openWreel", () => {
  it("opens a signed file coarse-first, then refines", async () => {
    const w = await openWreel(await buildWreel());
    expect(w.signature).toBe("valid");
    expect(w.manifest.provenance.ai_label).toBe("AI-generated");
    const layer = w.layers.get("static")!;
    expect(layer.level).toBe(0);
    expect(layer.chunk.count).toBe(1);
    expect(await w.refine()).toBe(1);
    expect(layer.level).toBe(1);
    expect(layer.chunk.count).toBe(2);
    expect(decodePoint(layer.chunk, 1)).toEqual({ position: [1, 0, 0], rgb: [128, 128, 128], cls: 2, confidence: expect.closeTo(0.2, 5) });
    expect(classHistogram(layer.chunk)).toEqual([1, 0, 1, 0, 0]);
    expect(await w.refine()).toBe(0);
    expect(w.tracks.get("a1")!.times).toEqual([0, 2]);
  });

  it("reports an unsigned file as unsigned", async () => {
    const w = await openWreel(await buildWreel({ sign: false }));
    expect(w.signature).toBe("missing");
  });

  it("detects a manifest changed after signing", async () => {
    const data = await buildWreel({ tamperManifest: (m) => (m.title = "Changed") });
    expect((await openWreel(data)).signature).toBe("invalid");
  });

  it("refuses an entry that does not match its hash", async () => {
    const data = await buildWreel({ tamper: (e) => (e["actors/a1/track.json"] = new TextEncoder().encode("{}")) });
    await expect(openWreel(data)).rejects.toThrow(/SHA-256/);
  });

  it("refuses unlisted entries", async () => {
    const data = await buildWreel({ tamper: (e) => (e["extra.bin"] = new Uint8Array([1, 2, 3])) });
    await expect(openWreel(data)).rejects.toThrow(/not listed/);
  });

  it("requires the audio slot and the provenance label", async () => {
    const noAudio = sampleManifest();
    delete noAudio.audio;
    await expect(openWreel(await buildWreel({ manifest: noAudio }))).rejects.toThrow(/audio/);
    const noLabel = sampleManifest();
    (noLabel.provenance as Record<string, unknown>).ai_label = undefined;
    await expect(openWreel(await buildWreel({ manifest: noLabel }))).rejects.toThrow(WreelError);
  });

  it("rejects another major version and warns on a newer minor", async () => {
    const v2 = sampleManifest();
    v2.format_version = "2.0";
    await expect(openWreel(await buildWreel({ manifest: v2 }))).rejects.toThrow(/2\.x/);
    const v13 = sampleManifest();
    v13.format_version = "1.3";
    expect((await openWreel(await buildWreel({ manifest: v13 }))).warnings).toHaveLength(1);
  });

  it("rejects something that is not a .wreel", async () => {
    await expect(openWreel(new TextEncoder().encode("hello"))).rejects.toThrow();
  });
});

const SAMPLE = fileURLToPath(new URL("../public/samples/courtyard.wreel", import.meta.url));

describe.skipIf(!existsSync(SAMPLE))("cross-language: sample written by Python", () => {
  it("opens, verifies the Ed25519 signature and refines", async () => {
    const w = await openWreel(new Uint8Array(readFileSync(SAMPLE)));
    expect(w.signature).toBe("valid");
    expect(w.manifest.audio).toBeDefined();
    expect(w.manifest.pov_presets.map((p) => p.mode)).toEqual(
      expect.arrayContaining(["directors-cut", "free", "third-person", "first-person", "overhead"]),
    );
    const before = w.layers.get("static")!.chunk.count;
    expect(await w.refine()).toBeGreaterThan(0);
    const layer = w.layers.get("static")!;
    expect(layer.chunk.count).toBeGreaterThan(before);
    const hist = classHistogram(layer.chunk);
    expect(hist[0]).toBeGreaterThan(0); // observed
    expect(hist[1]).toBeGreaterThan(0); // inferred
    expect(hist[2]).toBeGreaterThan(0); // guessed
  });
});
