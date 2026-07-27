/**
 * The default quality tier a first-time visitor gets.
 *
 * This used to be a hardcoded `high` — 4x2048 shadow cascades, TAA, GTAO and
 * SSR — handed to every machine sight unseen. On the reporter's laptop that was
 * enough, together with ~716 MB of uncompressed texture, to get the tab killed
 * by Chromium's out-of-memory handler before they could reach a settings menu.
 *
 * The asymmetry these tests encode: guessing too low costs a softer picture the
 * player can undo with `?quality=high`. Guessing too high costs the session.
 * Detection is therefore only ever allowed to choose between `low` and
 * `medium`, and the expensive tiers must be explicitly asked for.
 */

import { describe, expect, it } from 'vitest';

import {
  detectQuality,
  RENDER_TIERS,
  type DeviceProfile,
} from '../src/render/RenderSettings';

/** A machine with nothing obviously wrong with it. */
const capable: DeviceProfile = {
  deviceMemoryGb: 8,
  cores: 16,
  backingStorePixels: 1920 * 1080,
  mobile: false,
};

describe('detectQuality', () => {
  it('never returns an expensive tier, whatever the machine claims', () => {
    // The whole point. Even a machine reporting 8 GB and 32 cores gets
    // `medium`, because `deviceMemory` is clamped at 8 by every browser that
    // implements it and neither signal says anything at all about the GPU.
    const generous: DeviceProfile = {
      deviceMemoryGb: 8,
      cores: 32,
      backingStorePixels: 1280 * 720,
      mobile: false,
    };
    expect(detectQuality(generous)).toBe('medium');
    expect(['low', 'medium']).toContain(detectQuality(capable));
  });

  it('picks medium on an unremarkable desktop', () => {
    expect(detectQuality(capable)).toBe('medium');
  });

  it('drops to low on a phone or tablet regardless of reported specs', () => {
    expect(detectQuality({ ...capable, mobile: true })).toBe('low');
  });

  it('drops to low when the device admits to 4 GB or less', () => {
    expect(detectQuality({ ...capable, deviceMemoryGb: 4 })).toBe('low');
    expect(detectQuality({ ...capable, deviceMemoryGb: 2 })).toBe('low');
    expect(detectQuality({ ...capable, deviceMemoryGb: 8 })).toBe('medium');
  });

  it('drops to low on four logical cores or fewer', () => {
    expect(detectQuality({ ...capable, cores: 4 })).toBe('low');
    expect(detectQuality({ ...capable, cores: 8 })).toBe('medium');
  });

  it('drops to low when asked to fill a very large backing store', () => {
    // A 4K panel at DPR 1, or a 1440p panel at DPR 2. Render targets scale with
    // this and nothing else, so it is the one signal that speaks directly to
    // the memory that was blowing up.
    expect(detectQuality({ ...capable, backingStorePixels: 3840 * 2160 })).toBe('low');
  });

  it('treats missing signals as non-evidence rather than as bad news', () => {
    // Safari reports neither `deviceMemory` nor a useful core count. Punishing
    // a whole browser for its privacy posture would be the wrong call.
    expect(
      detectQuality({
        deviceMemoryGb: null,
        cores: null,
        backingStorePixels: 1920 * 1080,
        mobile: false,
      }),
    ).toBe('medium');
  });

  it('only ever chooses a tier that exists', () => {
    const profiles: DeviceProfile[] = [
      capable,
      { ...capable, mobile: true },
      { ...capable, cores: 1 },
      { deviceMemoryGb: null, cores: null, backingStorePixels: 0, mobile: false },
    ];
    for (const profile of profiles) {
      expect(RENDER_TIERS[detectQuality(profile)]).toBeDefined();
    }
  });
});

describe('the tier ladder still deletes work at the bottom', () => {
  it('turns off the passes that allocate full-resolution targets at low', () => {
    // `low` has to be genuinely cheap in *memory*, not just in shading, because
    // it is now the tier a constrained machine is automatically given.
    expect(RENDER_TIERS.low.gtao).toBe('off');
    expect(RENDER_TIERS.low.ssr).toBe('off');
  });

  it('keeps shadow maps small at low', () => {
    // 2 x 1024 R8 + depth is ~12 MB; the 4 x 2048 of the old default was ~67 MB.
    expect(RENDER_TIERS.low.shadowCascades).toBeLessThanOrEqual(2);
    expect(RENDER_TIERS.low.shadowMapSize).toBeLessThanOrEqual(1024);
  });

  it('orders the ladder monotonically in shadow cost', () => {
    const cost = (q: 'low' | 'medium' | 'high' | 'ultra'): number =>
      RENDER_TIERS[q].shadowCascades * RENDER_TIERS[q].shadowMapSize ** 2;
    expect(cost('low')).toBeLessThan(cost('medium'));
    expect(cost('medium')).toBeLessThan(cost('high'));
    expect(cost('high')).toBeLessThan(cost('ultra'));
  });
});
