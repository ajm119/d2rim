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

import { describeGpuString, statsRequested } from '../src/ui/DebugOverlay';

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

/**
 * The renderer string, and the one thing it exists to say out loud.
 *
 * Every performance report this project gets comes from a machine it cannot
 * touch, and a browser that has fallen back to a software rasteriser does not
 * announce it: the context is healthy, the image is correct, and the frame
 * takes tens of milliseconds with the cost sitting exactly where a GPU-bound
 * frame's cost sits. That is indistinguishable from "the scene is too heavy"
 * from the inside, and it is the hypothesis that survives every round of
 * removing work. So the overlay has to name it rather than print the string and
 * leave the reader to recognise it.
 */
describe('describeGpuString', () => {
  it('passes a real GPU through unchanged', () => {
    expect(describeGpuString('Apple M4')).toBe('Apple M4');
  });

  it('flags SwiftShader as not a GPU', () => {
    const text = describeGpuString(
      'ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero)), SwiftShader driver)',
    );
    expect(text).toContain('SOFTWARE RASTERISER');
  });

  it('flags the other software names too', () => {
    for (const name of ['SwANGLE (Apple)', 'llvmpipe (LLVM 15)', 'Microsoft Basic Render Driver']) {
      expect(describeGpuString(name)).toContain('SOFTWARE RASTERISER');
    }
  });

  it('says the browser withheld it rather than printing nothing', () => {
    expect(describeGpuString(null)).toContain('withheld');
    expect(describeGpuString(undefined)).toContain('withheld');
  });
});
