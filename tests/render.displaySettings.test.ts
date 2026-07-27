/**
 * The player's brightness trim.
 *
 * A grimdark frame puts most of its information in the bottom third of the
 * histogram, which is the first thing a bright room erases. The renderer's key
 * is locked on purpose (`FrameGraph`), so the only correct place for a
 * per-player, per-monitor adjustment is a trim *on top of* that key — and it
 * has to be a trim that a zone transition cannot silently revert.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  EXPOSURE_STOPS_MAX,
  EXPOSURE_STOPS_MIN,
  EXPOSURE_STORAGE_KEY,
  clampExposureStops,
  exposureStopsFromUrl,
  initialExposureStops,
  loadExposureStops,
  saveExposureStops,
} from '../src/render/DisplaySettings';

/** A minimal in-memory `Storage`, so these tests do not need a DOM. */
function installStorage(): Map<string, string> {
  const backing = new Map<string, string>();
  Object.defineProperty(globalThis, 'localStorage', {
    configurable: true,
    value: {
      getItem: (key: string): string | null => backing.get(key) ?? null,
      setItem: (key: string, value: string): void => void backing.set(key, value),
      removeItem: (key: string): void => void backing.delete(key),
    },
  });
  return backing;
}

let storage: Map<string, string>;

beforeEach(() => {
  storage = installStorage();
  vi.spyOn(console, 'info').mockImplementation(() => undefined);
  vi.spyOn(console, 'warn').mockImplementation(() => undefined);
});

afterEach(() => {
  vi.restoreAllMocks();
  Reflect.deleteProperty(globalThis, 'localStorage');
});

describe('clampExposureStops', () => {
  it('bounds the trim so no setting can black out or blow out the frame', () => {
    expect(clampExposureStops(99)).toBe(EXPOSURE_STOPS_MAX);
    expect(clampExposureStops(-99)).toBe(EXPOSURE_STOPS_MIN);
  });

  it('treats a non-finite value as "no trim" rather than propagating a NaN', () => {
    // A NaN reaching the uniform makes the whole frame black, which is exactly
    // the failure this control exists to fix.
    expect(clampExposureStops(Number.NaN)).toBe(0);
    // Infinity is not "as bright as possible", it is a corrupt value, and it is
    // treated the same way as a NaN: fall back to the authored look.
    expect(clampExposureStops(Number.POSITIVE_INFINITY)).toBe(0);
  });
});

describe('exposureStopsFromUrl', () => {
  it('reads stops, positive and negative', () => {
    expect(exposureStopsFromUrl('?exposure=1')).toBe(1);
    expect(exposureStopsFromUrl('?exposure=-0.5')).toBe(-0.5);
  });

  it('clamps rather than trusting the URL', () => {
    expect(exposureStopsFromUrl('?exposure=12')).toBe(EXPOSURE_STOPS_MAX);
  });

  it('returns null when absent, so the stored preference still wins', () => {
    expect(exposureStopsFromUrl('')).toBeNull();
    expect(exposureStopsFromUrl('?quality=low')).toBeNull();
  });

  it('warns on an unparseable value instead of silently doing nothing', () => {
    expect(exposureStopsFromUrl('?exposure=bright')).toBeNull();
    expect(console.warn).toHaveBeenCalled();
  });
});

describe('persistence', () => {
  it('round-trips through storage', () => {
    saveExposureStops(0.75);
    expect(storage.get(EXPOSURE_STORAGE_KEY)).toBe('0.75');
    expect(loadExposureStops()).toBe(0.75);
  });

  it('clamps on the way in as well as on the way out', () => {
    saveExposureStops(50);
    expect(loadExposureStops()).toBe(EXPOSURE_STOPS_MAX);
  });

  it('defaults to the authored look when nothing is stored', () => {
    expect(loadExposureStops()).toBe(0);
  });

  it('survives storage being unavailable', () => {
    Reflect.deleteProperty(globalThis, 'localStorage');
    expect(() => saveExposureStops(1)).not.toThrow();
    expect(loadExposureStops()).toBe(0);
  });

  it('ignores corrupt stored data rather than propagating it into the frame', () => {
    storage.set(EXPOSURE_STORAGE_KEY, 'very bright please');
    expect(loadExposureStops()).toBe(0);
  });
});

describe('initialExposureStops', () => {
  it('prefers the URL over the stored preference', () => {
    saveExposureStops(-1);
    expect(initialExposureStops('?exposure=1.5')).toBe(1.5);
  });

  it('does not persist the URL override', () => {
    // `?exposure=` is a diagnostic. Sending a player a link with it must not
    // permanently rewrite the setting they chose.
    saveExposureStops(-1);
    initialExposureStops('?exposure=1.5');
    expect(loadExposureStops()).toBe(-1);
  });

  it('falls back to the stored preference, then to zero', () => {
    saveExposureStops(0.25);
    expect(initialExposureStops('')).toBe(0.25);
    Reflect.deleteProperty(globalThis, 'localStorage');
    expect(initialExposureStops('')).toBe(0);
  });
});
