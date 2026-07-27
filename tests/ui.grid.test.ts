/**
 * The drag-and-drop arithmetic.
 *
 * `cellFromPoint` and `grabOffset` are the only part of the inventory screen
 * whose correctness is not visible in a screenshot: an off-by-one puts every
 * dropped item one cell from where the player let go, and the panel still looks
 * perfect. So they are pure functions, and this is where they are pinned.
 *
 * The rect origins are deliberately non-zero and the cell size is not a power
 * of two, because an implementation that forgets to subtract the origin, or one
 * that divides by the wrong constant, agrees with a correct one when the origin
 * is (0, 0) and the cell size is 1.
 */

import { describe, expect, it } from 'vitest';

import { CELL_SIZE, cellFromPoint, fraction, grabOffset, UI, Z } from '../src/ui/theme';
import { displayModeFor } from '../src/ui/UiManager';

const RECT = { left: 137, top: 291 };

describe('cellFromPoint', () => {
  it('maps the top-left pixel of a cell to that cell', () => {
    expect(cellFromPoint({ x: 137, y: 291 }, RECT, 34)).toEqual({ x: 0, y: 0 });
    expect(cellFromPoint({ x: 137 + 34, y: 291 + 34 }, RECT, 34)).toEqual({ x: 1, y: 1 });
  });

  it('maps the last pixel of a cell to that same cell, not the next', () => {
    expect(cellFromPoint({ x: 137 + 33, y: 291 + 33 }, RECT, 34)).toEqual({ x: 0, y: 0 });
    expect(cellFromPoint({ x: 137 + 34 - 0.001, y: 291 }, RECT, 34)).toEqual({ x: 0, y: 0 });
  });

  it('subtracts the grid origin', () => {
    // Without the subtraction this would be cell (4, 8) rather than (0, 0).
    expect(cellFromPoint({ x: 137, y: 291 }, RECT, 34)).toEqual({ x: 0, y: 0 });
    expect(cellFromPoint({ x: 137, y: 291 }, { left: 0, top: 0 }, 34)).not.toEqual({ x: 0, y: 0 });
  });

  it('goes negative outside the grid rather than clamping', () => {
    // Clamping here would silently place an out-of-bounds drop at (0, 0); the
    // grid's own bounds check is what refuses it, and it can only do that if
    // this reports the truth.
    expect(cellFromPoint({ x: 100, y: 291 }, RECT, 34)).toEqual({ x: -2, y: 0 });
  });

  it('shifts the result by the grab offset, so an item lands where it looks', () => {
    // The pointer is in the middle cell of a 1x3 sword grabbed at its second
    // cell: the item's top-left belongs one cell above the pointer.
    const point = { x: 137 + 34 * 3, y: 291 + 34 * 2 };
    expect(cellFromPoint(point, RECT, 34)).toEqual({ x: 3, y: 2 });
    expect(cellFromPoint(point, RECT, 34, { x: 0, y: 1 })).toEqual({ x: 3, y: 1 });
    expect(cellFromPoint(point, RECT, 34, { x: 2, y: 0 })).toEqual({ x: 1, y: 2 });
  });

  it('uses the cell size it is given', () => {
    expect(cellFromPoint({ x: 137 + 60, y: 291 }, RECT, 30)).toEqual({ x: 2, y: 0 });
    expect(cellFromPoint({ x: 137 + 60, y: 291 }, RECT, 34)).toEqual({ x: 1, y: 0 });
  });

  it('degrades rather than dividing by zero', () => {
    expect(Number.isFinite(cellFromPoint({ x: 200, y: 300 }, RECT, 0).x)).toBe(true);
  });
});

describe('grabOffset', () => {
  const itemRect = { left: 137 + 34 * 2, top: 291 + 34 * 1 };

  it('reports which cell of the item the pointer is over', () => {
    const item = { width: 2, height: 3 };
    expect(grabOffset({ x: itemRect.left + 5, y: itemRect.top + 5 }, itemRect, 34, item)).toEqual({
      x: 0,
      y: 0,
    });
    expect(
      grabOffset({ x: itemRect.left + 34 + 5, y: itemRect.top + 68 + 5 }, itemRect, 34, item),
    ).toEqual({ x: 1, y: 2 });
  });

  it('clamps inside the item, so an edge grab cannot shift every later drop', () => {
    const item = { width: 2, height: 3 };
    expect(grabOffset({ x: itemRect.left - 3, y: itemRect.top - 3 }, itemRect, 34, item)).toEqual({
      x: 0,
      y: 0,
    });
    expect(
      grabOffset({ x: itemRect.left + 999, y: itemRect.top + 999 }, itemRect, 34, item),
    ).toEqual({ x: 1, y: 2 });
  });

  it('round-trips: grabbing then dropping at the same point leaves the item put', () => {
    // The property that makes drag-and-drop feel correct — picking an item up
    // and releasing without moving must not shift it.
    const item = { width: 2, height: 3 };
    for (const [x, y] of [
      [0, 0],
      [3, 1],
      [7, 0],
      [5, 1],
    ] as const) {
      const rect = { left: RECT.left + x * 34, top: RECT.top + y * 34 };
      for (let dx = 0; dx < item.width; dx++) {
        for (let dy = 0; dy < item.height; dy++) {
          const point = { x: rect.left + dx * 34 + 12, y: rect.top + dy * 34 + 12 };
          const offset = grabOffset(point, rect, 34, item);
          expect(cellFromPoint(point, RECT, 34, offset)).toEqual({ x, y });
        }
      }
    }
  });
});

describe('theme tokens', () => {
  it('uses a cell size the whole 10x4 grid fits a sensible panel in', () => {
    expect(CELL_SIZE).toBeGreaterThan(24);
    expect(CELL_SIZE * 10).toBeLessThan(420);
  });

  it('layers the HUD under screens under dialogue under the menu', () => {
    expect(Z.hud).toBeLessThan(Z.screen);
    expect(Z.screen).toBeLessThan(Z.dialogue);
    expect(Z.dialogue).toBeLessThan(Z.menu);
    expect(Z.menu).toBeLessThan(Z.cursor);
  });

  it('keeps panels opaque, so text stays legible over a dark scene', () => {
    // A translucent panel reads beautifully over a bright frame and becomes
    // unreadable in a cave. Both panel colours must be solid hex.
    expect(UI.panel).toMatch(/^#[0-9a-f]{6}$/i);
    expect(UI.panelRaised).toMatch(/^#[0-9a-f]{6}$/i);
  });
});

describe('fraction', () => {
  it('clamps into [0, 1] and never returns NaN', () => {
    expect(fraction(5, 10)).toBe(0.5);
    expect(fraction(20, 10)).toBe(1);
    expect(fraction(-5, 10)).toBe(0);
    expect(fraction(5, 0)).toBe(0);
    expect(fraction(Number.NaN, 10)).toBe(0);
  });
});

describe('screen visibility', () => {
  it('never opens a screen into "none" or into the browser default', () => {
    // Both halves of this are regressions. Restoring `''` drops the screen's own
    // `display: flex` and jams every panel into the top-left corner; capturing
    // `'none'` from a screen that hid itself during construction opens every
    // panel into a zero-sized box. Both shipped, both were caught only by
    // reading a screenshot, and both are pinned here.
    expect(displayModeFor('')).toBe('flex');
    expect(displayModeFor('none')).toBe('flex');
  });

  it('honours a display mode the screen really did declare', () => {
    expect(displayModeFor('flex')).toBe('flex');
    expect(displayModeFor('grid')).toBe('grid');
    expect(displayModeFor('block')).toBe('block');
  });
});
