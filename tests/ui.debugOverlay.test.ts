/**
 * The overlay's flag parsing.
 *
 * The DOM half is not testable here — this project has no jsdom, deliberately —
 * but the flag is worth pinning because it has two spellings on purpose:
 * `?mem=1` already existed and was documented to players in the memory
 * investigation, and `?stats=1` is the name that describes what the expanded
 * readout now actually shows. Breaking the old spelling would invalidate every
 * instruction already given out.
 */

import { describe, expect, it } from 'vitest';

import { statsRequested } from '../src/ui/DebugOverlay';

describe('statsRequested', () => {
  it('accepts the new spelling', () => {
    expect(statsRequested('?stats=1')).toBe(true);
  });

  it('still accepts the memory-investigation spelling', () => {
    expect(statsRequested('?mem=1')).toBe(true);
  });

  it('is off by default, because the expanded readout walks the scene graph', () => {
    expect(statsRequested('')).toBe(false);
    expect(statsRequested('?quality=low&backend=webgl2')).toBe(false);
  });

  it('does not treat any truthy-looking value as on', () => {
    // `?stats=0` must mean off. A flag that cannot be turned off in a URL is a
    // flag a player cannot be talked out of once they have been talked into it.
    expect(statsRequested('?stats=0')).toBe(false);
    expect(statsRequested('?stats=true')).toBe(false);
  });
});
