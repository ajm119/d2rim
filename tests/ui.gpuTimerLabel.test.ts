/**
 * What the overlay prints after `gpu`.
 *
 * ### The bug this file exists to prevent recurring
 *
 * The overlay reported `gpu …` for an entire session on the machine this whole
 * investigation is about. It reads as "still resolving". It meant "this device
 * will never answer", and six reasonable-looking links produced it:
 *
 * 1. `?stats=1` asks three for `trackTimestamp`.
 * 2. `WebGLBackend.initTimestampQuery` returns early without
 *    `EXT_disjoint_timer_query_webgl2`, so no timestamp query pool is created.
 * 3. `Backend.resolveTimestampsAsync` hits `if (!queryPool) return;` and
 *    resolves with `undefined`, never writing `info.render.timestamp`.
 * 4. three initialises that field to `0` and leaves it there.
 * 5. `resolveGpuTime` read the 0, found it finite, and reported it as a
 *    successful measurement of zero — four times a second, forever.
 * 6. `FrameStats.gpuAvailable` only flips on a sample above zero, so it stayed
 *    false and the display fell through to its "pending" branch.
 *
 * Chromium gates `EXT_disjoint_timer_query_webgl2` because a GPU timer is a
 * timing side channel, and hardened builds (Brave with fingerprint protection,
 * Safari) withhold it outright — so "no timer" is the *common* case on exactly
 * the consumer machines this game has to run on, not an exotic one. The label
 * has to say so, and it has to point at the fallback.
 */

import { describe, expect, it } from 'vitest';

import { gpuTimeLabel } from '../src/ui/DebugOverlay';

const base = {
  gpuAvailable: false,
  gpuMs: 0,
  state: undefined,
  statsRequested: true,
  refused: false,
} as const;

describe('gpuTimeLabel', () => {
  it('prints the measurement when there is one', () => {
    expect(gpuTimeLabel({ ...base, gpuAvailable: true, gpuMs: 12.34, state: 'available' })).toBe(
      '12.3 ms',
    );
  });

  it('never says "pending" for a device that cannot answer', () => {
    const label = gpuTimeLabel({ ...base, state: 'unsupported' });
    expect(label).toContain('n/a');
    expect(label).not.toContain('…');
  });

  it('points at the fallback probe when the timer is unavailable', () => {
    // The whole reason `?gpusync=1` exists is that this case is common. A label
    // that only says "n/a" leaves the reader with nothing to do next.
    expect(gpuTimeLabel({ ...base, state: 'unsupported' })).toContain('?gpusync=1');
  });

  it('treats a refused resolve the same as an unsupported device', () => {
    expect(gpuTimeLabel({ ...base, state: 'available', refused: true })).toContain('n/a');
  });

  it('says the readout is off rather than unavailable when nobody asked', () => {
    // Two genuinely different facts: "you did not turn it on" is fixable by the
    // reader in one keystroke; "your driver will not do it" is not.
    expect(gpuTimeLabel({ ...base, statsRequested: false })).toBe('off (?stats=1)');
    expect(gpuTimeLabel({ ...base, state: 'off' })).toBe('off (?stats=1)');
  });

  it('only ever uses the ellipsis for a device that genuinely might answer', () => {
    expect(gpuTimeLabel({ ...base, state: 'available' })).toBe('pending…');
  });

  it('falls back to pending on a renderer that reports no state at all', () => {
    // A test double or a future backend need not implement `gpuTimer`. Leaving
    // the old behaviour there is right: it is the honest answer for "unknown".
    expect(gpuTimeLabel({ ...base, state: undefined })).toBe('pending…');
  });
});
