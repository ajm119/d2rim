/**
 * @module tests/render.pixelRatio
 *
 * The pixel-ratio cap: the largest single term in a quality tier's budget.
 *
 * `devicePixelRatio` is squared into the fragment count, so a flat cap of 2 —
 * which is what `Engine` used to hold as a constant — makes every full-screen
 * pass on a Retina laptop cost four times what the same window costs at DPR 1.
 * That lands almost entirely on the terrain uber-shader, which is already the
 * frame's dominant cost. The cap therefore belongs to the tier, alongside every
 * other decision about what this machine can afford, and these tests pin both
 * halves of that: the ladder itself, and the wiring that makes a tier change
 * reach the renderer without a reload.
 */

import { describe, expect, it, vi } from 'vitest';

import type { GameContext } from '../src/core/types';
import { EventBus } from '../src/core/EventBus';
import { ServiceLocator } from '../src/core/ServiceLocator';
import {
  RENDER_QUALITIES,
  RENDER_TIERS,
  RenderSettings,
  detectQuality,
  type RenderQuality,
} from '../src/render/RenderSettings';

/** A context whose engine records what the settings module asks of it. */
function makeContext(): { ctx: GameContext; caps: number[] } {
  const caps: number[] = [];
  const ctx = {
    engine: {
      setPixelRatioCap: (cap: number) => {
        caps.push(cap);
      },
    } as unknown as GameContext['engine'],
    scene: {} as GameContext['scene'],
    camera: {} as GameContext['camera'],
    renderer: {} as GameContext['renderer'],
    input: {} as GameContext['input'],
    events: new EventBus(),
    time: { elapsed: 0, delta: 1 / 60, frame: 0, scale: 1 },
    services: new ServiceLocator(),
  } satisfies GameContext;
  return { ctx, caps };
}

describe('the pixel-ratio ladder', () => {
  it('gives every tier a cap', () => {
    for (const quality of RENDER_QUALITIES) {
      expect(RENDER_TIERS[quality].pixelRatioCap).toBeGreaterThan(0);
    }
  });

  it('never decreases as the tier goes up', () => {
    const caps = RENDER_QUALITIES.map((q) => RENDER_TIERS[q].pixelRatioCap);
    for (let i = 1; i < caps.length; i++) {
      expect(caps[i]!).toBeGreaterThanOrEqual(caps[i - 1]!);
    }
  });

  it('keeps the automatic tiers off the 4x Retina cliff', () => {
    // `detectQuality` only ever returns these two, so these two are what an
    // unattended first-time visitor gets. Neither may cost 4x fill.
    for (const quality of ['low', 'medium'] as const) {
      expect(RENDER_TIERS[quality].pixelRatioCap).toBeLessThan(2);
    }
    // ...and neither may be so low that a capable machine renders soft. Below 1
    // the frame is upscaled, which is a different (and worse) decision.
    for (const quality of ['low', 'medium'] as const) {
      expect(RENDER_TIERS[quality].pixelRatioCap).toBeGreaterThanOrEqual(1);
    }
  });

  it('lets an explicit high/ultra opt-in have the full ratio', () => {
    expect(RENDER_TIERS.high.pixelRatioCap).toBe(2);
    expect(RENDER_TIERS.ultra.pixelRatioCap).toBe(2);
  });
});

describe('a Retina laptop', () => {
  /** The MacBook Air M4's default scaled resolution, and its DPR. */
  const AIR = { width: 1512, height: 945, dpr: 2 };

  it('is profiled against the fill it would be asked for, not its panel', () => {
    // The signal used to cap at 2, which measured 5.7 Mpx and tripped the 5 Mpx
    // rule — demoting a 10-core M4 to `low` for a cost it would never pay,
    // because automatic detection never selects a tier that renders at DPR 2.
    const profile = {
      deviceMemoryGb: 8,
      cores: 10,
      backingStorePixels: Math.round(
        AIR.width * AIR.height * RENDER_TIERS.medium.pixelRatioCap ** 2,
      ),
      mobile: false,
    };
    expect(profile.backingStorePixels).toBeLessThan(5_000_000);
    expect(detectQuality(profile)).toBe('medium');
  });

  it('still lands on low when the machine really is small', () => {
    expect(
      detectQuality({ deviceMemoryGb: 4, cores: 10, backingStorePixels: 2_000_000, mobile: false }),
    ).toBe('low');
    expect(
      detectQuality({ deviceMemoryGb: 8, cores: 2, backingStorePixels: 2_000_000, mobile: false }),
    ).toBe('low');
  });

  it('renders 3.22 Mpx at medium instead of the uncapped 5.72', () => {
    const capped = new RenderSettings({ quality: 'medium' }).bufferSize(
      AIR.width,
      AIR.height,
      AIR.dpr,
    );
    const uncapped = AIR.width * AIR.dpr * AIR.height * AIR.dpr;
    expect(capped.megapixels).toBeCloseTo(3.22, 2);
    // A 44% cut in fragments, on the tier a first-time visitor actually gets.
    expect(capped.megapixels / (uncapped / 1e6)).toBeCloseTo(0.5625, 3);
  });
});

describe('RenderSettings.bufferSize', () => {
  const settings = (quality: RenderQuality): RenderSettings => new RenderSettings({ quality });

  it('caps the device ratio rather than replacing it', () => {
    // A DPR-1 desktop is unaffected by any cap at or above 1.
    expect(settings('medium').bufferSize(1920, 1080, 1).ratio).toBe(1);
    // A Retina laptop is held at the tier's cap.
    expect(settings('medium').bufferSize(1512, 945, 2).ratio).toBe(1.5);
    expect(settings('low').bufferSize(1512, 945, 2).ratio).toBe(1);
    expect(settings('high').bufferSize(1512, 945, 2).ratio).toBe(2);
  });

  it('reports the drawing buffer, which is the number that costs money', () => {
    const retina = settings('medium').bufferSize(1512, 945, 2);
    expect(retina.width).toBe(2268);
    expect(retina.height).toBe(1418);
    // 3.22 Mpx against the 5.72 Mpx an uncapped DPR 2 would have asked for.
    expect(retina.megapixels).toBeCloseTo(3.22, 2);
    expect(settings('high').bufferSize(1512, 945, 2).megapixels).toBeCloseTo(5.72, 2);
  });

  it('treats a zero or missing ratio as 1 rather than collapsing the buffer', () => {
    expect(settings('high').bufferSize(1280, 720, 0).ratio).toBe(1);
  });
});

describe('wiring', () => {
  it('pushes the tier cap at init, so the first frame is already the right size', () => {
    const { ctx, caps } = makeContext();
    new RenderSettings({ quality: 'low' }).init(ctx);
    expect(caps).toEqual([1]);
  });

  it('a live quality change reaches the renderer without a reload', () => {
    const { ctx, caps } = makeContext();
    const settings = new RenderSettings({ quality: 'low' });
    settings.init(ctx);
    caps.length = 0;
    settings.setQuality('ultra');
    expect(settings.quality).toBe('ultra');
    expect(caps).toEqual([2]);
  });

  it('does not churn the drawing buffer when the tier has not moved', () => {
    const { ctx, caps } = makeContext();
    const settings = new RenderSettings({ quality: 'high' });
    settings.init(ctx);
    caps.length = 0;
    settings.setQuality('high');
    expect(caps).toEqual([]);
  });

  it('survives being driven before init, rather than throwing into the frame', () => {
    const settings = new RenderSettings({ quality: 'low' });
    expect(() => settings.setQuality('medium')).not.toThrow();
    expect(settings.quality).toBe('medium');
  });

  it('logs the ratio it settled on, because that is how the user checks it', () => {
    const info = vi.spyOn(console, 'info').mockImplementation(() => undefined);
    const { ctx } = makeContext();
    new RenderSettings({ quality: 'medium' }).init(ctx);
    const lines = info.mock.calls.map((call) => String(call[0]));
    expect(lines.some((line) => line.includes('pixel ratio'))).toBe(true);
    expect(lines.some((line) => line.includes('drawing buffer'))).toBe(true);
    info.mockRestore();
  });
});
