/**
 * Which compressed format the GPU actually gets.
 *
 * This is the test for a factor-of-two memory bug that survived a whole
 * optimisation pass because nothing ever asserted the *target* of the
 * transcode, only that a transcode happened. The archive is ETC1S; three's
 * loader ranks BPTC/BC7 above BC1 for ETC1S data; BC7 is 1.0 byte per texel and
 * BC1 is 0.5. Everything downstream — the 256 MB cache budget, the "~90 MB
 * preload set" in the AssetManager's own comments — was written against 0.5.
 */

import { describe, expect, it } from 'vitest';

import { narrowKtx2Support, type Ktx2Support } from '../src/assets/AssetManager';

const NONE: Ktx2Support = {
  astcSupported: false,
  etc1Supported: false,
  etc2Supported: false,
  dxtSupported: false,
  bptcSupported: false,
  pvrtcSupported: false,
};

describe('narrowKtx2Support', () => {
  it('refuses BC7 when BC1 is available — the doubling bug', () => {
    // A typical desktop ANGLE configuration: S3TC and BPTC, no ETC2. three
    // would pick BC7 here, at twice the VRAM, for source data that is already
    // ETC1S and cannot benefit.
    const decision = narrowKtx2Support({ ...NONE, dxtSupported: true, bptcSupported: true });

    expect(decision.config.bptcSupported).toBe(false);
    expect(decision.bytesPerPixel).toBe(0.5);
    expect(decision.format).toContain('BC1');
    expect(decision.overridden).toBe(true);
  });

  it('prefers ETC2 where the device has it, at the same half-rate cost', () => {
    // Apple Silicon and most mobile-class GPUs. ETC2 already outranks BPTC in
    // three's table, so nothing needed changing except the reported name.
    const decision = narrowKtx2Support({
      ...NONE,
      etc2Supported: true,
      bptcSupported: true,
      astcSupported: true,
    });

    expect(decision.bytesPerPixel).toBe(0.5);
    expect(decision.format).toContain('ETC2');
  });

  it('keeps BC7 when it is the only compressed format on offer', () => {
    // 1.0 byte/texel is still 32x better than the RGBA8 JPEG fallback. A
    // working image beats a smaller one.
    const decision = narrowKtx2Support({ ...NONE, bptcSupported: true });

    expect(decision.config.bptcSupported).toBe(true);
    expect(decision.bytesPerPixel).toBe(1);
    expect(decision.format).toContain('BC7');
    expect(decision.overridden).toBe(false);
  });

  it('reports the uncompressed fallback honestly', () => {
    const decision = narrowKtx2Support({ ...NONE });
    expect(decision.bytesPerPixel).toBe(4);
    expect(decision.format).toContain('RGBA8');
  });

  it('ignores ASTC, which three never selects for ETC1S data', () => {
    // `priorityETC1S: Infinity` on the ASTC row. Reporting ASTC here would be a
    // lie that happens to sound like good news.
    const decision = narrowKtx2Support({ ...NONE, astcSupported: true, bptcSupported: true });
    expect(decision.format).not.toContain('ASTC');
    expect(decision.bytesPerPixel).toBe(1);
  });

  it('treats missing flags as unsupported rather than throwing', () => {
    expect(() => narrowKtx2Support({})).not.toThrow();
    expect(narrowKtx2Support({}).bytesPerPixel).toBe(4);
  });

  it('halves the preload set: 32 plates at 2048 square', () => {
    // The number that matters. 12 mip levels of a 2048x2048 plate is 4/3 the
    // base area; 32 of those is the preload set.
    const plate = (bytesPerPixel: number): number => (2048 * 2048 * 4) / 3 * bytesPerPixel;
    const bc7 = narrowKtx2Support({ ...NONE, bptcSupported: true }).bytesPerPixel;
    const bc1 = narrowKtx2Support({ ...NONE, bptcSupported: true, dxtSupported: true })
      .bytesPerPixel;

    expect((plate(bc7) * 32) / (1024 * 1024)).toBeCloseTo(170.7, 0);
    expect((plate(bc1) * 32) / (1024 * 1024)).toBeCloseTo(85.3, 0);
  });
});
