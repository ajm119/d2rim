/**
 * Regression tests for the WebGPU capture readback row unpacking.
 *
 * WebGPU's `copyTextureToBuffer` requires `bytesPerRow` to be a multiple of 256,
 * and three.js returns that padded buffer verbatim. Before this was handled, any
 * capture whose width was not a multiple of 64 came back sheared: the image was
 * still colourful and high-contrast, so the blank-frame guard could not be relied
 * on to catch it. Verified empirically at 840x470 on the WebGPU backend — garbage
 * before, pixel-comparable to WebGL2 after.
 */

import { describe, expect, it } from 'vitest';

import { unpackRows } from '../src/render/RendererFactory';

const ALIGNMENT = 256;

/** Build a tightly packed RGBA8 image whose every pixel encodes its own (x, y). */
function makeTightImage(width: number, height: number): Uint8Array {
  const out = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      out[i] = x & 0xff;
      out[i + 1] = y & 0xff;
      out[i + 2] = (x + y) & 0xff;
      out[i + 3] = 255;
    }
  }
  return out;
}

/** Re-pack a tight image the way a WebGPU readback delivers it. */
function padRows(tight: Uint8Array, width: number, height: number): Uint8Array {
  const tightStride = width * 4;
  const paddedStride = Math.ceil(tightStride / ALIGNMENT) * ALIGNMENT;
  const out = new Uint8Array((height - 1) * paddedStride + tightStride);
  // Fill with a poison value so any padding that leaks through is obvious.
  out.fill(0xcd);
  for (let y = 0; y < height; y++) {
    out.set(tight.subarray(y * tightStride, (y + 1) * tightStride), y * paddedStride);
  }
  return out;
}

describe('unpackRows', () => {
  it('passes a tightly packed buffer through untouched', () => {
    const tight = makeTightImage(64, 8); // 64 * 4 = 256, already aligned
    const result = unpackRows(tight, 64, 8);
    expect(result).toBe(tight);
  });

  // 64 and 128 are aligned; the rest are not and would shear without unpacking.
  for (const width of [17, 63, 64, 100, 128, 200, 840]) {
    it(`restores every row at width ${width}`, () => {
      const height = 13;
      const tight = makeTightImage(width, height);
      const padded = padRows(tight, width, height);

      const result = unpackRows(padded, width, height);

      expect(result.length).toBe(width * height * 4);
      expect(Array.from(result)).toEqual(Array.from(tight));
    });
  }

  it('keeps the last row intact even though the padded buffer stops short of a full stride', () => {
    const width = 840;
    const height = 470;
    const tight = makeTightImage(width, height);
    const padded = padRows(tight, width, height);
    const paddedStride = Math.ceil(width * 4 / ALIGNMENT) * ALIGNMENT;

    // This is the shape three.js actually allocates: whole strides for every row
    // but the last, which is only as long as the real pixel data.
    expect(padded.length).toBe((height - 1) * paddedStride + width * 4);

    const result = unpackRows(padded, width, height);
    const lastRow = result.subarray((height - 1) * width * 4);
    expect(lastRow[0]).toBe(0);
    expect(lastRow[1]).toBe((height - 1) & 0xff);
    expect(lastRow[lastRow.length - 1]).toBe(255);
  });

  it('refuses a buffer that is too short for either stride rather than emitting garbage', () => {
    expect(() => unpackRows(new Uint8Array(100), 840, 470)).toThrow(/too short/);
  });
});
