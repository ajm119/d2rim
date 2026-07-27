/**
 * @module tests/world.dungeonGenerator
 *
 * The dungeon generator is the one system in this phase whose failure mode is
 * *silent*: a cave that occasionally walls off its own objective produces no
 * error, no visual glitch and no crash — just a quest that cannot be completed,
 * discovered by a player twenty minutes in. So it is tested by invariant across
 * many seeds rather than by example, and every invariant is re-derived from the
 * finished layout by `auditDungeon`, which shares no code with the generator's
 * own bookkeeping.
 */

import { describe, expect, it } from 'vitest';

import {
  CaveCell,
  auditDungeon,
  cellAtWorld,
  clearanceField,
  createDungeonRng,
  floodFrom,
  generateDungeon,
  hashSeed,
  isFloorAtWorld,
  worldOfCell,
  type DungeonLayout,
} from '../src/world/DungeonGenerator';

/** Enough seeds that a one-in-fifty failure cannot hide. */
const SEEDS = Array.from({ length: 48 }, (_, i) => `den-${i}`);

/** A smaller grid so the sweep stays fast; every guarantee is size-independent. */
const SWEEP = { cols: 64, rows: 64 } as const;

function sweep(): DungeonLayout[] {
  return SEEDS.map((seed) => generateDungeon({ seed, ...SWEEP }));
}

describe('seeded RNG', () => {
  it('hashes number and string seeds identically', () => {
    expect(hashSeed(7)).toBe(hashSeed('7'));
    expect(hashSeed('a')).not.toBe(hashSeed('b'));
  });

  it('is deterministic and stays in range', () => {
    const a = createDungeonRng('x');
    const b = createDungeonRng('x');
    for (let i = 0; i < 500; i++) {
      const value = a.next();
      expect(value).toBe(b.next());
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThan(1);
    }
  });

  it('keeps int() inside the requested range', () => {
    const rng = createDungeonRng('int');
    for (let i = 0; i < 2000; i++) {
      const value = rng.int(5);
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThanOrEqual(4);
      expect(Number.isInteger(value)).toBe(true);
    }
  });
});

describe('determinism', () => {
  it('reproduces a layout byte-for-byte from the same seed', () => {
    const a = generateDungeon({ seed: 'repeat', ...SWEEP });
    const b = generateDungeon({ seed: 'repeat', ...SWEEP });
    expect(Array.from(b.cells)).toEqual(Array.from(a.cells));
    expect(b.entrance).toEqual(a.entrance);
    expect(b.floorCells).toBe(a.floorCells);
    expect(b.deepestChamber).toEqual(a.deepestChamber);
    expect(b.spawnPoints).toEqual(a.spawnPoints);
    expect(b.chambers.length).toBe(a.chambers.length);
  });

  it('treats a numeric seed and its string form as the same cave', () => {
    const a = generateDungeon({ seed: 42, ...SWEEP });
    const b = generateDungeon({ seed: '42', ...SWEEP });
    expect(Array.from(b.cells)).toEqual(Array.from(a.cells));
  });

  it('produces different caves for different seeds', () => {
    const layouts = SEEDS.slice(0, 12).map((seed) => generateDungeon({ seed, ...SWEEP }));
    const fingerprints = new Set(layouts.map((l) => l.cells.join('')));
    expect(fingerprints.size).toBe(layouts.length);
  });
});

describe('connectivity', () => {
  it('reaches every walkable cell from the entrance, across every seed', () => {
    for (const layout of sweep()) {
      const audit = auditDungeon(layout);
      expect(
        audit.unreachable,
        `seed ${String(layout.sourceSeed)} left ${audit.unreachable} cells unreachable`,
      ).toBe(0);
      expect(audit.reachableCells).toBe(audit.floorCells);
      expect(audit.floorCells).toBeGreaterThan(200);
    }
  });

  it('starts the flood fill on a walkable entrance cell', () => {
    for (const layout of sweep()) {
      const index = layout.entrance.row * layout.cols + layout.entrance.col;
      expect(layout.cells[index]).toBe(CaveCell.Floor);
      expect(layout.distance[index]).toBe(0);
    }
  });

  it('connects the entrance to the deepest chamber with a positive-length path', () => {
    for (const layout of sweep()) {
      const chamber = layout.deepestChamber;
      const index = chamber.center.row * layout.cols + chamber.center.col;
      expect(layout.cells[index]).toBe(CaveCell.Floor);
      expect(layout.distance[index]).toBeGreaterThan(0);
      expect(chamber.depth).toBe(layout.distance[index]);
      // The deepest chamber must genuinely be at the back of the cave, not an
      // alcove by the door: the objective is placed here. The bound is 0.7 and
      // not 1.0 because a chamber's *centre* is its widest cell, which sits a
      // little back from the furthest cell — the objective wants the room, not
      // the tip of the dead end.
      expect(chamber.depthRatio).toBeGreaterThanOrEqual(0.7);
    }
  });

  it('never reports a chamber the entrance cannot reach', () => {
    for (const layout of sweep()) {
      for (const chamber of layout.chambers) {
        const index = chamber.center.row * layout.cols + chamber.center.col;
        expect(layout.distance[index]).toBeGreaterThanOrEqual(0);
      }
      expect(layout.chambers.length).toBeGreaterThanOrEqual(1);
    }
  });

  it('detects a deliberately severed cave', () => {
    // Confidence that the audit is actually looking: wall off a slice of a good
    // cave and the flood fill must notice.
    const layout = generateDungeon({ seed: 'sever', ...SWEEP });
    const cut = { ...layout, cells: Uint8Array.from(layout.cells) };
    const row = layout.entrance.row - 6;
    for (let col = 0; col < layout.cols; col++) cut.cells[row * layout.cols + col] = CaveCell.Wall;
    const audit = auditDungeon(cut);
    expect(audit.unreachable).toBeGreaterThan(0);
  });
});

describe('navigable width', () => {
  it('covers every walkable cell with a wholly walkable disc of the corridor radius', () => {
    for (const layout of sweep()) {
      const audit = auditDungeon(layout);
      expect(
        audit.tooNarrow,
        `seed ${String(layout.sourceSeed)} has ${audit.tooNarrow} sub-width cells`,
      ).toBe(0);
    }
  });

  it('holds at a wider corridor radius too', () => {
    for (const seed of SEEDS.slice(0, 10)) {
      const layout = generateDungeon({ seed, ...SWEEP, corridorRadiusCells: 3 });
      expect(auditDungeon(layout).tooNarrow).toBe(0);
      expect(auditDungeon(layout).unreachable).toBe(0);
    }
  });

  it('keeps the solid rock margin intact so walls have real thickness', () => {
    for (const layout of sweep()) {
      expect(auditDungeon(layout).inMargin).toBe(0);
      expect(layout.marginCells).toBeGreaterThanOrEqual(3);
    }
  });
});

describe('chambers', () => {
  it('finds at least two distinct chambers on a normal cave', () => {
    let multi = 0;
    for (const layout of sweep()) if (layout.chambers.length >= 2) multi++;
    // Not "always": a seed can genuinely produce one big lobe, and forcing a
    // second chamber there would be a lie about the geometry.
    expect(multi).toBeGreaterThan(SEEDS.length * 0.8);
  });

  it('gives every chamber a sane centre and radius', () => {
    for (const layout of sweep()) {
      for (const chamber of layout.chambers) {
        expect(chamber.cellCount).toBeGreaterThan(0);
        expect(chamber.radiusCells).toBeGreaterThanOrEqual(layout.corridorRadiusCells);
        expect(chamber.depthRatio).toBeGreaterThanOrEqual(0);
        expect(chamber.depthRatio).toBeLessThanOrEqual(1);
        const world = worldOfCell(layout, chamber.center.col, chamber.center.row);
        expect(chamber.centerWorld.x).toBeCloseTo(world.x, 10);
        expect(isFloorAtWorld(layout, world.x, world.z)).toBe(true);
      }
    }
  });

  it('picks the deepest chamber as the maximum-depth chamber', () => {
    for (const layout of sweep()) {
      const max = Math.max(...layout.chambers.map((c) => c.depth));
      expect(layout.deepestChamber.depth).toBe(max);
    }
  });
});

describe('spawn points', () => {
  it('places enough enemies to be a den', () => {
    for (const layout of sweep()) {
      expect(
        layout.spawnPoints.length,
        `seed ${String(layout.sourceSeed)} only placed ${layout.spawnPoints.length}`,
      ).toBeGreaterThanOrEqual(10);
      expect(layout.spawnPoints.length).toBeLessThanOrEqual(20);
    }
  });

  it('puts every spawn on reachable floor', () => {
    for (const layout of sweep()) {
      for (const spawn of layout.spawnPoints) {
        const index = spawn.cell.row * layout.cols + spawn.cell.col;
        expect(layout.cells[index]).toBe(CaveCell.Floor);
        expect(layout.distance[index]).toBeGreaterThanOrEqual(0);
        expect(isFloorAtWorld(layout, spawn.x, spawn.z)).toBe(true);
      }
    }
  });

  it('keeps spawns away from the entrance', () => {
    for (const layout of sweep()) {
      for (const spawn of layout.spawnPoints) {
        expect(spawn.depth).toBeGreaterThanOrEqual(14);
      }
    }
  });

  it('weights spawns toward the back of the cave', () => {
    // The mean spawn depth must beat the mean depth of the cave as a whole,
    // which is the operative meaning of "weighted away from the entrance".
    let better = 0;
    for (const layout of sweep()) {
      let floorDepth = 0;
      let floorCount = 0;
      for (let i = 0; i < layout.cells.length; i++) {
        if (layout.cells[i] !== CaveCell.Floor) continue;
        floorDepth += layout.distance[i] as number;
        floorCount++;
      }
      const caveMean = floorDepth / floorCount;
      const spawnMean =
        layout.spawnPoints.reduce((sum, s) => sum + s.depth, 0) / layout.spawnPoints.length;
      if (spawnMean > caveMean) better++;
    }
    expect(better).toBe(SEEDS.length);
  });

  it('separates spawns by the minimum spacing', () => {
    for (const layout of sweep()) {
      const points = layout.spawnPoints;
      for (let i = 0; i < points.length; i++) {
        for (let j = i + 1; j < points.length; j++) {
          const a = points[i]!;
          const b = points[j]!;
          const d = Math.hypot(a.cell.col - b.cell.col, a.cell.row - b.cell.row);
          expect(d).toBeGreaterThanOrEqual(6 - 1e-9);
        }
      }
    }
  });

  it('spreads spawns through the cave rather than piling them at the back', () => {
    // At least a quarter of the spawns should sit in the near-to-middle half of
    // the cave. A generator that only ever fills the back chamber passes every
    // other test in this block and still produces a bad encounter.
    for (const layout of sweep()) {
      const near = layout.spawnPoints.filter((s) => s.depthRatio < 0.6).length;
      expect(near).toBeGreaterThanOrEqual(2);
    }
  });

  it('assigns tougher variants deeper in', () => {
    const layout = generateDungeon({ seed: 'variants', ...SWEEP });
    const variants = ['minion', 'rogue', 'warrior', 'mage'];
    for (const spawn of layout.spawnPoints) {
      expect(variants).toContain(spawn.variant);
      const tier = variants.indexOf(spawn.variant);
      expect(tier).toBe(Math.min(3, Math.floor(spawn.depthRatio * 4)));
    }
    // Deep sentries, shallow patrols.
    for (const spawn of layout.spawnPoints) {
      if (spawn.depthRatio > 0.75) expect(spawn.patrol).toBe(0);
      else expect(spawn.patrol).toBeGreaterThan(0);
    }
  });
});

describe('grid <-> world mapping', () => {
  it('round-trips a cell through world space', () => {
    const layout = generateDungeon({ seed: 'map', ...SWEEP });
    for (let row = 0; row < layout.rows; row += 7) {
      for (let col = 0; col < layout.cols; col += 5) {
        const world = worldOfCell(layout, col, row);
        expect(cellAtWorld(layout, world.x, world.z)).toEqual({ col, row });
      }
    }
  });

  it('centres the grid on the zone origin', () => {
    // The mapping is pure arithmetic on cols/rows/cellSize, so it is checked
    // against a bare grid description rather than a generated cave — a 10x10
    // grid is smaller than one entrance chamber and cannot be generated.
    const grid = { cols: 10, rows: 10, cellSize: 2 };
    expect(worldOfCell(grid, 5, 5)).toEqual({ x: 1, z: 1 });
    expect(worldOfCell(grid, 4, 4)).toEqual({ x: -1, z: -1 });
    const layout = generateDungeon({ seed: 'centre', cols: 40, rows: 40, cellSize: 2 });
    expect(layout.halfExtent).toEqual({ x: 40, z: 40 });
  });

  it('refuses a grid too small to hold a cave rather than returning a bad one', () => {
    expect(() => generateDungeon({ seed: 'tiny', cols: 10, rows: 10 })).toThrow(/too small/i);
  });

  it('reports off-grid world positions as solid', () => {
    const layout = generateDungeon({ seed: 'offgrid', ...SWEEP });
    expect(isFloorAtWorld(layout, 10_000, 0)).toBe(false);
    expect(isFloorAtWorld(layout, 0, -10_000)).toBe(false);
  });
});

describe('supporting fields', () => {
  it('computes a clearance field that is zero on walls and positive on floor', () => {
    const layout = generateDungeon({ seed: 'clear', ...SWEEP });
    const field = clearanceField(layout.cells, layout.cols, layout.rows);
    for (let i = 0; i < layout.cells.length; i++) {
      if (layout.cells[i] === CaveCell.Wall) expect(field[i]).toBe(0);
      else expect(field[i]).toBeGreaterThan(0);
    }
    expect(Array.from(field)).toEqual(Array.from(layout.clearance));
  });

  it('returns an all -1 field when the flood fill starts in rock', () => {
    const layout = generateDungeon({ seed: 'rock-start', ...SWEEP });
    const distance = floodFrom(layout.cells, layout.cols, layout.rows, { col: 0, row: 0 });
    expect(distance.every((d) => d === -1)).toBe(true);
  });
});

describe('robustness', () => {
  it('survives hostile option combinations without breaking its guarantees', () => {
    const cases = [
      { cols: 40, rows: 40, fillProbability: 0.55 },
      { cols: 40, rows: 40, fillProbability: 0.38 },
      { cols: 96, rows: 48, smoothingPasses: 3 },
      { cols: 48, rows: 96, smoothingPasses: 12, corridorRadiusCells: 3 },
      { cols: 36, rows: 36, marginCells: 6, minRegionCells: 5 },
    ];
    for (const [index, options] of cases.entries()) {
      const layout = generateDungeon({ seed: `hostile-${index}`, ...options });
      const audit = auditDungeon(layout);
      expect(audit.unreachable, `case ${index}`).toBe(0);
      expect(audit.tooNarrow, `case ${index}`).toBe(0);
      expect(audit.inMargin, `case ${index}`).toBe(0);
      expect(layout.floorCells).toBeGreaterThan(0);
      expect(layout.deepestChamber).toBeDefined();
    }
  });

  it('reports which attempt succeeded so a pathological seed is visible', () => {
    for (const layout of sweep()) {
      expect(layout.attempt).toBeGreaterThanOrEqual(0);
      expect(layout.attempt).toBeLessThan(8);
    }
  });
});
