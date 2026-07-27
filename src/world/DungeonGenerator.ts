/**
 * @module world/DungeonGenerator
 *
 * Seeded procedural cave generation for the Den of Evil.
 *
 * ## Why cellular automata, and not rooms-and-corridors
 *
 * The Den is a *cave*. The classic roguelike generator — partition a rectangle,
 * put a room in each cell, join the rooms with L-shaped corridors — produces
 * geometry whose every surface is axis-aligned, and no amount of rock texture
 * makes an axis-aligned room read as a cave. Metaballs go too far the other
 * way: a sum of radial falloffs is smooth everywhere, so the result is a string
 * of circular pods with no irregular lobes, no dead-end fissures and no pinch
 * points, which reads as a lava lamp.
 *
 * Cellular automata over a boolean grid sit exactly between the two. Seed the
 * grid with noise, then repeatedly replace each cell by the majority of its
 * neighbourhood: noise at the scale of one cell is destroyed, correlated
 * structure at the scale of five or six cells survives and thickens. What comes
 * out is lobed open areas joined by winding, irregular-width passages — the
 * shape a real solution cave has, produced by the same *kind* of process
 * (local rules, no global plan) that produces one.
 *
 * The price is that CA offers no structural guarantees at all: it will happily
 * hand back three disconnected caves and a scattering of one-cell pockets. Every
 * guarantee this module makes is therefore imposed *after* the automaton has
 * run, by the pipeline below.
 *
 * ## Pipeline
 *
 * ```
 *  1. seed        random fill at `fillProbability`, solid border margin
 *  2. smooth      `smoothingPasses` of the 4-5 rule (see #smooth)
 *  3. open        morphological opening by a disc of radius R
 *  4. prune       flood fill; fill in every region below `minRegionCells`
 *  5. connect     carve region-to-region tunnels with a radius-R disc brush
 *  6. entrance    carve the entrance chamber and its approach tunnel
 *  7. verify      flood fill from the entrance; repair, then assert
 *  8. analyse     clearance field, chambers, spawn points
 * ```
 *
 * ### The width guarantee (step 3)
 *
 * Morphological opening — erode by a disc of radius R, then dilate the result
 * by the same disc — has a property that is exactly the invariant wanted here:
 *
 * > every cell of `dilate(erode(F, R), R)` lies inside some disc of radius R
 * > that is itself wholly contained in `dilate(erode(F, R), R)`.
 *
 * Proof: a cell `c` survives the dilation only because some `e` in the eroded
 * set is within R of it. `disc(e, R)` is wholly floor by the definition of
 * erosion, and every cell of `disc(e, R)` is within R of `e`, so every one of
 * them also survives the dilation. `c` is in `disc(e, R)`, and `disc(e, R)` is
 * in the opened set. ∎
 *
 * So "no passage is narrower than 2R cells" is not a statistical property of
 * this generator, it is a theorem about step 3 — and it is preserved by steps 5
 * and 6 because those carve with the *same* disc brush, and a union of discs
 * cannot violate a property that is closed under union. `assertNavigable` in
 * the tests checks it directly rather than trusting the argument.
 *
 * ### The connectivity guarantee (steps 4-7)
 *
 * An unreachable objective is a silently unwinnable quest, so connectivity is
 * enforced three times over: by construction (every surviving region is tunnelled
 * to the main region, and the entrance is tunnelled to the main region), by an
 * explicit repair pass (any floor cell the entrance flood fill fails to reach is
 * tunnelled to the nearest reached cell), and finally by an assertion. If the
 * assertion ever fires the attempt is discarded and the next seeded attempt runs.
 *
 * ## No three.js here
 *
 * Deliberately dependency-free: the layout is plain typed arrays and plain
 * objects. That keeps the unit tests instant (no WASM, no renderer, no WebGL
 * stub), lets the quest system in phase 5 query the layout without touching the
 * scene, and makes the whole thing serialisable. `scene/DenOfEvil` is the only
 * thing that turns it into geometry.
 */

/* -------------------------------------------------------------------------- */
/* Seeded RNG                                                                 */
/* -------------------------------------------------------------------------- */

/** The random source the generator draws from. Deterministic given a seed. */
export interface Rng {
  /** Uniform in `[0, 1)`. */
  next(): number;
  /** Uniform in `[min, max)`. */
  range(min: number, max: number): number;
  /** Uniform integer in `[0, maxExclusive)`. */
  int(maxExclusive: number): number;
}

/**
 * FNV-1a over the decimal form of the seed.
 *
 * String and number seeds go through the same path so that `seed: 7` and
 * `seed: '7'` produce the same cave — a difference there is the kind of thing
 * that makes a "deterministic" generator mysteriously non-reproducible between
 * a test (which passes a number) and a scene (which passes a config string).
 */
export function hashSeed(seed: string | number): number {
  const text = String(seed);
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

/** mulberry32. Fast, 2^32 period, and passes the smallcrush suite. */
export function createDungeonRng(seed: string | number): Rng {
  let state = hashSeed(seed);
  const next = (): number => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return {
    next,
    range: (min, max) => min + next() * (max - min),
    int: (maxExclusive) => Math.min(maxExclusive - 1, Math.floor(next() * maxExclusive)),
  };
}

/* -------------------------------------------------------------------------- */
/* Layout types                                                               */
/* -------------------------------------------------------------------------- */

/** Grid cell states. Stored in a `Uint8Array`, so these are plain numbers. */
export const CaveCell = {
  Wall: 0,
  Floor: 1,
} as const;

/** Integer grid coordinate. Column runs along +x, row along +z. */
export interface GridPoint {
  readonly col: number;
  readonly row: number;
}

/** A point in the zone's local world space, on the cave floor plane. */
export interface WorldPoint {
  readonly x: number;
  readonly z: number;
}

/**
 * An open area large enough to fight in.
 *
 * Chambers are the connected components of "cells whose distance to the nearest
 * wall is at least `chamberClearance`", which is a shape-driven definition
 * rather than an authored one: a chamber is wherever the cave happens to be
 * wide, which is exactly where a player would call it a room.
 */
export interface CaveChamber {
  readonly id: number;
  /** The cell furthest from any wall in this chamber — its natural centre. */
  readonly center: GridPoint;
  readonly centerWorld: WorldPoint;
  readonly cellCount: number;
  /** Largest inscribed radius, in cells. */
  readonly radiusCells: number;
  /** Shortest path from the entrance to `center`, in cells. */
  readonly depth: number;
  /** `depth` normalised against the deepest reachable cell, in `[0, 1]`. */
  readonly depthRatio: number;
}

/** Where the {@link scene/DenOfEvil} asks {@link ai/EnemyDirector} for a skeleton. */
export interface CaveSpawnPoint {
  readonly cell: GridPoint;
  readonly x: number;
  readonly z: number;
  /** Shortest path from the entrance, in cells. */
  readonly depth: number;
  /** `depth` normalised against the deepest reachable cell, in `[0, 1]`. */
  readonly depthRatio: number;
  /** Skeleton variant key, chosen by depth. */
  readonly variant: string;
  /** Patrol radius in metres; 0 makes a sentry. */
  readonly patrol: number;
}

export interface DungeonOptions {
  /** Anything stable. The same seed always yields a byte-identical layout. */
  readonly seed?: string | number;
  readonly cols?: number;
  readonly rows?: number;
  /** Metres per cell. */
  readonly cellSize?: number;
  /** Fraction of cells seeded as wall before smoothing. 0.45-0.48 is cave-like. */
  readonly fillProbability?: number;
  readonly smoothingPasses?: number;
  /**
   * Half the guaranteed minimum passage width, in cells. Every walkable cell is
   * covered by a disc of this radius that is itself entirely walkable.
   */
  readonly corridorRadiusCells?: number;
  /** Solid rock ring around the grid, in cells. Gives the walls real thickness. */
  readonly marginCells?: number;
  /** Regions smaller than this are filled in rather than connected. */
  readonly minRegionCells?: number;
  /** Wall-distance at which an area counts as a chamber, in cells. */
  readonly chamberClearance?: number;
  /** How many skeletons to place. Fewer are placed if the cave cannot hold them. */
  readonly spawnCount?: number;
  /** No spawn closer to the entrance than this, in cells. */
  readonly minSpawnDepthCells?: number;
  /** Minimum separation between two spawns, in cells. */
  readonly minSpawnSpacingCells?: number;
  /** Enemy variants, ordered shallowest to deepest. */
  readonly variants?: readonly string[];
  /** Fraction of the interior that must end up walkable, or the attempt is retried. */
  readonly minFloorFraction?: number;
}

/**
 * The generated cave, as data.
 *
 * Everything the scene builder and the phase-5 quest system need is here, and
 * nothing here knows about three.js. `cells`, `clearance` and `distance` are
 * parallel arrays indexed `row * cols + col`.
 */
export interface DungeonLayout {
  /** The hashed seed actually used, including the retry salt. */
  readonly seed: number;
  /** The seed as passed in, for diagnostics. */
  readonly sourceSeed: string | number;
  /** Which generation attempt produced this layout (0-based). */
  readonly attempt: number;

  readonly cols: number;
  readonly rows: number;
  readonly cellSize: number;
  readonly marginCells: number;
  readonly corridorRadiusCells: number;

  /** `CaveCell.Wall` or `CaveCell.Floor` per cell. */
  readonly cells: Uint8Array;
  /** Approximate distance to the nearest wall, in cells. Zero on walls. */
  readonly clearance: Float32Array;
  /** Cells from the entrance along the floor; `-1` for walls (never for floor). */
  readonly distance: Int32Array;

  readonly entrance: GridPoint;
  readonly entranceWorld: WorldPoint;
  /** Radius of the carved entrance chamber, in cells. */
  readonly entranceRadiusCells: number;

  readonly chambers: readonly CaveChamber[];
  /** The chamber furthest from the entrance. Where the quest objective goes. */
  readonly deepestChamber: CaveChamber;

  readonly spawnPoints: readonly CaveSpawnPoint[];

  readonly floorCells: number;
  /** Longest shortest-path from the entrance, in cells. */
  readonly maxDepth: number;
  /** Half-extent of the whole grid in metres, `{ x, z }`. */
  readonly halfExtent: WorldPoint;
}

const DEFAULTS: Required<DungeonOptions> = {
  seed: 'den-of-evil',
  // 88 x 88 at 1 m is an 88 m cave: about forty seconds of walking end to end,
  // which is the right size for a five-minute clearing objective. Larger grids
  // cost nothing to generate but do cost wall triangles in `DenOfEvil`.
  cols: 88,
  rows: 88,
  cellSize: 1,
  fillProbability: 0.46,
  smoothingPasses: 7,
  // 2 cells => a guaranteed 4 m minimum passage width. The player capsule is
  // 0.35 m and a skeleton is about the same, so 4 m is two abreast plus room to
  // sidestep — the narrowest a melee fight stays readable in.
  corridorRadiusCells: 2,
  marginCells: 4,
  minRegionCells: 40,
  chamberClearance: 4.2,
  spawnCount: 20,
  minSpawnDepthCells: 14,
  minSpawnSpacingCells: 6,
  variants: ['minion', 'rogue', 'warrior', 'mage'],
  minFloorFraction: 0.16,
};

/** How many seeded attempts before the best-so-far is accepted. */
const MAX_ATTEMPTS = 8;

/* -------------------------------------------------------------------------- */
/* Grid helpers                                                               */
/* -------------------------------------------------------------------------- */

/** World position of a cell centre. The grid is centred on the zone origin. */
export function worldOfCell(
  layout: Pick<DungeonLayout, 'cols' | 'rows' | 'cellSize'>,
  col: number,
  row: number,
): WorldPoint {
  return {
    x: (col + 0.5 - layout.cols / 2) * layout.cellSize,
    z: (row + 0.5 - layout.rows / 2) * layout.cellSize,
  };
}

/** The cell containing a world position. May be outside the grid. */
export function cellAtWorld(
  layout: Pick<DungeonLayout, 'cols' | 'rows' | 'cellSize'>,
  x: number,
  z: number,
): GridPoint {
  return {
    col: Math.floor(x / layout.cellSize + layout.cols / 2),
    row: Math.floor(z / layout.cellSize + layout.rows / 2),
  };
}

/** Is the cell containing this world position walkable? */
export function isFloorAtWorld(layout: DungeonLayout, x: number, z: number): boolean {
  const { col, row } = cellAtWorld(layout, x, z);
  if (col < 0 || row < 0 || col >= layout.cols || row >= layout.rows) return false;
  return layout.cells[row * layout.cols + col] === CaveCell.Floor;
}

/**
 * Offsets of every cell within `radius` of the origin, as a flat `[dc, dr, ...]`
 * list. Precomputed once per radius because erosion and dilation walk it for
 * every cell in the grid.
 */
function discOffsets(radius: number): Int16Array {
  const r = Math.ceil(radius);
  const r2 = radius * radius;
  const out: number[] = [];
  for (let dr = -r; dr <= r; dr++) {
    for (let dc = -r; dc <= r; dc++) {
      if (dc * dc + dr * dr <= r2) out.push(dc, dr);
    }
  }
  return Int16Array.from(out);
}

/* -------------------------------------------------------------------------- */
/* Generation                                                                 */
/* -------------------------------------------------------------------------- */

/**
 * Build a cave.
 *
 * Deterministic: identical options produce a byte-identical {@link DungeonLayout}.
 * Attempts that come out too solid (the automaton occasionally chokes itself
 * off) are retried with a salted seed, so a caller never has to handle failure.
 */
export function generateDungeon(options: DungeonOptions = {}): DungeonLayout {
  const config: Required<DungeonOptions> = { ...DEFAULTS, ...options };

  // Fail here, with the numbers, rather than eight attempts later with a
  // generic "no valid cave": a grid this small is a caller bug, not bad luck.
  const span = Math.min(config.cols, config.rows) - config.marginCells * 2;
  const needed = (config.corridorRadiusCells + 2) * 2 + 2;
  if (span < needed) {
    throw new Error(
      `[DungeonGenerator] grid too small: ${config.cols}x${config.rows} with a ` +
        `${config.marginCells}-cell margin leaves ${span} cells, and the entrance ` +
        `chamber alone needs ${needed}`,
    );
  }

  const interior =
    (config.cols - config.marginCells * 2) * (config.rows - config.marginCells * 2);
  const wanted = interior * config.minFloorFraction;

  let best: DungeonLayout | null = null;
  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
    const layout = generateAttempt(config, attempt);
    if (layout === null) continue;
    if (layout.floorCells >= wanted) return layout;
    if (best === null || layout.floorCells > best.floorCells) best = layout;
  }
  if (best !== null) return best;
  // Unreachable in practice: the entrance chamber alone guarantees floor cells,
  // and `generateAttempt` only returns null when its own connectivity assertion
  // fires. Throwing beats returning a cave the player cannot stand in.
  throw new Error(
    `[DungeonGenerator] no valid cave for seed "${String(config.seed)}" in ${MAX_ATTEMPTS} attempts`,
  );
}

function generateAttempt(config: Required<DungeonOptions>, attempt: number): DungeonLayout | null {
  const { cols, rows, marginCells: margin } = config;
  const radius = config.corridorRadiusCells;
  // The attempt index is folded into the seed rather than into the RNG stream,
  // so attempt N is reproducible on its own without replaying attempts 0..N-1.
  const seedText = `${String(config.seed)}#${attempt}`;
  const rng = createDungeonRng(seedText);

  let cells = seedGrid(config, rng);
  for (let pass = 0; pass < config.smoothingPasses; pass++) {
    cells = smooth(cells, cols, rows, margin, pass);
  }

  cells = open(cells, cols, rows, radius);

  /* -- prune and connect ------------------------------------------------- */

  let regions = findRegions(cells, cols, rows);
  if (regions.length === 0) return null;

  regions.sort((a, b) => b.length - a.length);
  const survivors: number[][] = [];
  for (const region of regions) {
    if (region.length >= config.minRegionCells || survivors.length === 0) survivors.push(region);
    else for (const index of region) cells[index] = CaveCell.Wall;
  }

  const main = survivors[0];
  if (main === undefined) return null;
  for (let i = 1; i < survivors.length; i++) {
    const region = survivors[i];
    if (region === undefined) continue;
    connectRegions(cells, cols, rows, margin, radius, region, main, rng);
  }

  /* -- entrance ----------------------------------------------------------- */

  // Fixed at the south edge on the grid's centre line. Authored rather than
  // seeded so the portal down from the Blood Moor always arrives in the same
  // place, and so the deepest chamber is genuinely "the far end" rather than
  // "wherever the entrance happened not to be".
  const entranceRadius = radius + 2;
  const entrance: GridPoint = {
    col: Math.floor(cols / 2),
    row: rows - 1 - margin - entranceRadius,
  };
  carveChamber(cells, cols, rows, margin, radius, entrance.col, entrance.row, entranceRadius);

  const approach = nearestCell(main, cols, entrance);
  if (approach !== null) {
    carveTunnel(cells, cols, rows, margin, radius, entrance, approach, rng);
  }

  /* -- verify and repair --------------------------------------------------- */

  let distance = floodFrom(cells, cols, rows, entrance);
  for (let repair = 0; repair < 4; repair++) {
    const orphan = firstOrphan(cells, distance);
    if (orphan === null) break;
    const target = nearestReached(distance, cols, rows, orphan);
    if (target === null) return null;
    carveTunnel(cells, cols, rows, margin, radius, orphan, target, rng);
    distance = floodFrom(cells, cols, rows, entrance);
  }
  // The hard failure the whole module exists to prevent. Discarding the attempt
  // is always correct here: another seed is free, an unreachable objective is not.
  if (firstOrphan(cells, distance) !== null) return null;

  /* -- analysis ------------------------------------------------------------ */

  const clearance = clearanceField(cells, cols, rows);
  let floorCells = 0;
  let maxDepth = 0;
  for (let i = 0; i < cells.length; i++) {
    if (cells[i] !== CaveCell.Floor) continue;
    floorCells++;
    const d = distance[i] ?? 0;
    if (d > maxDepth) maxDepth = d;
  }

  const partial = {
    cols,
    rows,
    cellSize: config.cellSize,
  };

  const chambers = findChambers(cells, clearance, distance, config, partial, maxDepth);
  const deepestChamber = chambers.reduce((a, b) => (b.depth > a.depth ? b : a));
  const spawnPoints = placeSpawns(cells, clearance, distance, config, partial, maxDepth, rng);

  return {
    seed: hashSeed(seedText),
    sourceSeed: config.seed,
    attempt,
    cols,
    rows,
    cellSize: config.cellSize,
    marginCells: margin,
    corridorRadiusCells: radius,
    cells,
    clearance,
    distance,
    entrance,
    entranceWorld: worldOfCell(partial, entrance.col, entrance.row),
    entranceRadiusCells: entranceRadius,
    chambers,
    deepestChamber,
    spawnPoints,
    floorCells,
    maxDepth,
    halfExtent: { x: (cols * config.cellSize) / 2, z: (rows * config.cellSize) / 2 },
  };
}

/* -- step 1: seed ---------------------------------------------------------- */

function seedGrid(config: Required<DungeonOptions>, rng: Rng): Uint8Array {
  const { cols, rows, marginCells: margin, fillProbability } = config;
  const cells = new Uint8Array(cols * rows);
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const border = col < margin || row < margin || col >= cols - margin || row >= rows - margin;
      // The RNG is advanced even for border cells so that the stream position
      // depends only on the grid size, not on the margin.
      const roll = rng.next();
      cells[row * cols + col] = border || roll < fillProbability ? CaveCell.Wall : CaveCell.Floor;
    }
  }
  return cells;
}

/* -- step 2: smooth -------------------------------------------------------- */

/**
 * One automaton pass.
 *
 * The 4-5 rule: a cell becomes wall when five or more of its nine-cell
 * neighbourhood are wall. On the first passes an extra term also walls in cells
 * whose *two*-ring neighbourhood is almost empty, which stops the automaton
 * settling into one enormous cavern — that second term is the difference
 * between a cave with passages and a car park with rounded corners.
 */
function smooth(
  cells: Uint8Array,
  cols: number,
  rows: number,
  margin: number,
  pass: number,
): Uint8Array {
  const out = new Uint8Array(cells.length);
  const wide = pass < 4;
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const index = row * cols + col;
      if (col < margin || row < margin || col >= cols - margin || row >= rows - margin) {
        out[index] = CaveCell.Wall;
        continue;
      }
      const near = wallsWithin(cells, cols, rows, col, row, 1);
      if (near >= 5) {
        out[index] = CaveCell.Wall;
      } else if (wide && wallsWithin(cells, cols, rows, col, row, 2) <= 2) {
        out[index] = CaveCell.Wall;
      } else {
        out[index] = CaveCell.Floor;
      }
    }
  }
  return out;
}

/** Wall cells in the square neighbourhood of `radius`. Off-grid counts as wall. */
function wallsWithin(
  cells: Uint8Array,
  cols: number,
  rows: number,
  col: number,
  row: number,
  radius: number,
): number {
  let count = 0;
  for (let dr = -radius; dr <= radius; dr++) {
    for (let dc = -radius; dc <= radius; dc++) {
      const c = col + dc;
      const r = row + dr;
      if (c < 0 || r < 0 || c >= cols || r >= rows) count++;
      else if (cells[r * cols + c] === CaveCell.Wall) count++;
    }
  }
  return count;
}

/* -- step 3: opening ------------------------------------------------------- */

/**
 * Morphological opening by a disc of `radius`.
 *
 * This is the width guarantee; see the module header for the proof that every
 * surviving cell sits inside a wholly-walkable disc of this radius.
 */
function open(cells: Uint8Array, cols: number, rows: number, radius: number): Uint8Array {
  const offsets = discOffsets(radius);
  const eroded = new Uint8Array(cells.length);
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const index = row * cols + col;
      if (cells[index] !== CaveCell.Floor) continue;
      let fits = true;
      for (let k = 0; k < offsets.length && fits; k += 2) {
        const c = col + (offsets[k] ?? 0);
        const r = row + (offsets[k + 1] ?? 0);
        if (c < 0 || r < 0 || c >= cols || r >= rows) fits = false;
        else if (cells[r * cols + c] !== CaveCell.Floor) fits = false;
      }
      if (fits) eroded[index] = CaveCell.Floor;
    }
  }

  const opened = new Uint8Array(cells.length);
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      if (eroded[row * cols + col] !== CaveCell.Floor) continue;
      for (let k = 0; k < offsets.length; k += 2) {
        const c = col + (offsets[k] ?? 0);
        const r = row + (offsets[k + 1] ?? 0);
        if (c < 0 || r < 0 || c >= cols || r >= rows) continue;
        opened[r * cols + c] = CaveCell.Floor;
      }
    }
  }
  return opened;
}

/* -- steps 4-6: regions, tunnels, carving ---------------------------------- */

/** Every 4-connected component of floor cells, as arrays of cell indices. */
function findRegions(cells: Uint8Array, cols: number, rows: number): number[][] {
  const seen = new Uint8Array(cells.length);
  const regions: number[][] = [];
  const queue: number[] = [];
  for (let start = 0; start < cells.length; start++) {
    if (cells[start] !== CaveCell.Floor || seen[start] === 1) continue;
    const region: number[] = [];
    queue.length = 0;
    queue.push(start);
    seen[start] = 1;
    while (queue.length > 0) {
      const index = queue.pop() as number;
      region.push(index);
      const col = index % cols;
      const row = (index - col) / cols;
      if (col > 0) pushIfFloor(cells, seen, queue, index - 1);
      if (col < cols - 1) pushIfFloor(cells, seen, queue, index + 1);
      if (row > 0) pushIfFloor(cells, seen, queue, index - cols);
      if (row < rows - 1) pushIfFloor(cells, seen, queue, index + cols);
    }
    regions.push(region);
  }
  return regions;
}

function pushIfFloor(
  cells: Uint8Array,
  seen: Uint8Array,
  queue: number[],
  index: number,
): void {
  if (seen[index] === 1 || cells[index] !== CaveCell.Floor) return;
  seen[index] = 1;
  queue.push(index);
}

/** The cell of `region` closest to `point`, by squared grid distance. */
function nearestCell(region: readonly number[], cols: number, point: GridPoint): GridPoint | null {
  let bestIndex = -1;
  let bestDistance = Infinity;
  for (const index of region) {
    const col = index % cols;
    const row = (index - col) / cols;
    const d = (col - point.col) ** 2 + (row - point.row) ** 2;
    if (d < bestDistance) {
      bestDistance = d;
      bestIndex = index;
    }
  }
  if (bestIndex < 0) return null;
  const col = bestIndex % cols;
  return { col, row: (bestIndex - col) / cols };
}

/**
 * Join two regions with a tunnel between their closest pair of cells.
 *
 * Only boundary cells are considered, which cuts the pair search by an order of
 * magnitude and cannot change the answer: the closest pair of two disjoint sets
 * is always a pair of boundary cells.
 */
function connectRegions(
  cells: Uint8Array,
  cols: number,
  rows: number,
  margin: number,
  radius: number,
  from: readonly number[],
  to: readonly number[],
  rng: Rng,
): void {
  const a = boundaryOf(cells, cols, rows, from);
  const b = boundaryOf(cells, cols, rows, to);
  let bestA: GridPoint | null = null;
  let bestB: GridPoint | null = null;
  let best = Infinity;
  for (const ia of a) {
    const ca = ia % cols;
    const ra = (ia - ca) / cols;
    for (const ib of b) {
      const cb = ib % cols;
      const rb = (ib - cb) / cols;
      const d = (ca - cb) ** 2 + (ra - rb) ** 2;
      if (d < best) {
        best = d;
        bestA = { col: ca, row: ra };
        bestB = { col: cb, row: rb };
      }
    }
  }
  if (bestA === null || bestB === null) return;
  carveTunnel(cells, cols, rows, margin, radius, bestA, bestB, rng);
}

/** Floor cells of `region` that touch a wall (or the grid edge). */
function boundaryOf(
  cells: Uint8Array,
  cols: number,
  rows: number,
  region: readonly number[],
): number[] {
  const out: number[] = [];
  for (const index of region) {
    const col = index % cols;
    const row = (index - col) / cols;
    const exposed =
      col === 0 ||
      row === 0 ||
      col === cols - 1 ||
      row === rows - 1 ||
      cells[index - 1] === CaveCell.Wall ||
      cells[index + 1] === CaveCell.Wall ||
      cells[index - cols] === CaveCell.Wall ||
      cells[index + cols] === CaveCell.Wall;
    if (exposed) out.push(index);
  }
  return out;
}

/**
 * Carve a disc of floor, clamped so no brush ever writes into the solid margin.
 *
 * The clamp is what keeps the width guarantee intact near the edges — a disc
 * clipped by the grid boundary would leave cells that are *not* covered by a
 * whole radius-R disc.
 */
function carveDisc(
  cells: Uint8Array,
  cols: number,
  rows: number,
  margin: number,
  col: number,
  row: number,
  radius: number,
): void {
  const r = Math.ceil(radius);
  const cc = clamp(col, margin + r, cols - 1 - margin - r);
  const cr = clamp(row, margin + r, rows - 1 - margin - r);
  const r2 = radius * radius;
  for (let dr = -r; dr <= r; dr++) {
    for (let dc = -r; dc <= r; dc++) {
      if (dc * dc + dr * dr > r2) continue;
      const c = cc + dc;
      const rr = cr + dr;
      if (c < 0 || rr < 0 || c >= cols || rr >= rows) continue;
      cells[rr * cols + c] = CaveCell.Floor;
    }
  }
}

/**
 * Carve an open area of `radius` as a *union of brush-sized discs*.
 *
 * Not one big disc, and the difference is not cosmetic. The width guarantee is
 * "every walkable cell sits inside a wholly walkable disc of the brush radius",
 * and on a continuous plane a big disc trivially satisfies it. On an integer
 * grid it does not: rasterised discs are not nested at the rim, so the four
 * cells at the extreme of a radius-(R+2) disc have no wholly-walkable radius-R
 * disc containing them, and the audit — correctly — flags them. Sweeping the
 * brush over a smaller disc of centres is morphological dilation, and it makes
 * the invariant true by construction at every radius.
 */
function carveChamber(
  cells: Uint8Array,
  cols: number,
  rows: number,
  margin: number,
  brush: number,
  col: number,
  row: number,
  radius: number,
): void {
  const centres = discOffsets(Math.max(0, radius - brush));
  for (let k = 0; k < centres.length; k += 2) {
    carveDisc(cells, cols, rows, margin, col + (centres[k] ?? 0), row + (centres[k + 1] ?? 0), brush);
  }
}

/**
 * Carve a passage between two cells with a radius-R disc brush.
 *
 * Two segments through a jittered midpoint rather than one straight line: a
 * dead-straight tunnel is the one shape a cave never has, and these are the only
 * authored passages in the layout, so they are the only ones at risk of reading
 * as machined. The jitter is drawn from the same seeded stream as everything
 * else, so it does not break determinism.
 */
function carveTunnel(
  cells: Uint8Array,
  cols: number,
  rows: number,
  margin: number,
  radius: number,
  from: GridPoint,
  to: GridPoint,
  rng: Rng,
): void {
  const span = Math.hypot(to.col - from.col, to.row - from.row);
  const wobble = Math.min(6, span * 0.25);
  const mid: GridPoint = {
    col: Math.round((from.col + to.col) / 2 + rng.range(-wobble, wobble)),
    row: Math.round((from.row + to.row) / 2 + rng.range(-wobble, wobble)),
  };
  carveSegment(cells, cols, rows, margin, radius, from, mid);
  carveSegment(cells, cols, rows, margin, radius, mid, to);
}

function carveSegment(
  cells: Uint8Array,
  cols: number,
  rows: number,
  margin: number,
  radius: number,
  from: GridPoint,
  to: GridPoint,
): void {
  const steps = Math.max(1, Math.ceil(Math.hypot(to.col - from.col, to.row - from.row)));
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    carveDisc(
      cells,
      cols,
      rows,
      margin,
      Math.round(from.col + (to.col - from.col) * t),
      Math.round(from.row + (to.row - from.row) * t),
      radius,
    );
  }
}

/* -- step 7: reachability -------------------------------------------------- */

/**
 * Breadth-first distance from `origin` over floor cells.
 *
 * `-1` means "wall, or floor the entrance cannot reach". A single `-1` on a
 * floor cell is a quest-breaking bug, which is why this returns the whole field
 * rather than a boolean: the repair pass needs to know *which* cells.
 */
export function floodFrom(
  cells: Uint8Array,
  cols: number,
  rows: number,
  origin: GridPoint,
): Int32Array {
  const distance = new Int32Array(cells.length).fill(-1);
  const start = origin.row * cols + origin.col;
  if (start < 0 || start >= cells.length || cells[start] !== CaveCell.Floor) return distance;

  distance[start] = 0;
  // A plain array used as a ring buffer via a head index: BFS pushes every cell
  // exactly once, so `shift()` (O(n)) would make this quadratic on a big grid.
  const queue = new Int32Array(cells.length);
  queue[0] = start;
  let head = 0;
  let tail = 1;
  while (head < tail) {
    const index = queue[head++] as number;
    const next = (distance[index] as number) + 1;
    const col = index % cols;
    const row = (index - col) / cols;
    if (col > 0) tail = visit(cells, distance, queue, tail, index - 1, next);
    if (col < cols - 1) tail = visit(cells, distance, queue, tail, index + 1, next);
    if (row > 0) tail = visit(cells, distance, queue, tail, index - cols, next);
    if (row < rows - 1) tail = visit(cells, distance, queue, tail, index + cols, next);
  }
  return distance;
}

function visit(
  cells: Uint8Array,
  distance: Int32Array,
  queue: Int32Array,
  tail: number,
  index: number,
  value: number,
): number {
  if (cells[index] !== CaveCell.Floor || distance[index] !== -1) return tail;
  distance[index] = value;
  queue[tail] = index;
  return tail + 1;
}

/** The first floor cell the entrance flood fill failed to reach, if any. */
function firstOrphan(cells: Uint8Array, distance: Int32Array): GridPoint | null {
  for (let i = 0; i < cells.length; i++) {
    if (cells[i] === CaveCell.Floor && distance[i] === -1) {
      // Column recovery needs the width, which the caller knows; store the raw
      // index in `col` and let `nearestReached` resolve it. Kept private.
      return { col: i, row: -1 };
    }
  }
  return null;
}

/** Resolve a raw-index orphan against the grid and find its nearest reached cell. */
function nearestReached(
  distance: Int32Array,
  cols: number,
  rows: number,
  orphan: GridPoint,
): GridPoint | null {
  const index = orphan.row === -1 ? orphan.col : orphan.row * cols + orphan.col;
  const oc = index % cols;
  const or = (index - oc) / cols;
  let best = Infinity;
  let found: GridPoint | null = null;
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      if (distance[row * cols + col] === -1) continue;
      const d = (col - oc) ** 2 + (row - or) ** 2;
      if (d < best) {
        best = d;
        found = { col, row };
      }
    }
  }
  return found;
}

/* -- step 8: analysis ------------------------------------------------------ */

/**
 * Distance from every floor cell to the nearest wall, in cells.
 *
 * Two-pass chamfer with (1, √2) weights — a 4% -accurate Euclidean transform in
 * two linear sweeps. Exactness is not needed: this drives chamber detection and
 * spawn roominess, both of which are threshold comparisons with slack in them.
 * The *width guarantee* does not use this field; it is a theorem about the
 * opening in {@link open}, and the tests check it with exact disc arithmetic.
 */
export function clearanceField(cells: Uint8Array, cols: number, rows: number): Float32Array {
  const D1 = 1;
  const D2 = Math.SQRT2;
  const out = new Float32Array(cells.length);
  const big = cols + rows;
  for (let i = 0; i < cells.length; i++) out[i] = cells[i] === CaveCell.Floor ? big : 0;

  const relax = (index: number, from: number, weight: number): void => {
    const candidate = (out[from] as number) + weight;
    if (candidate < (out[index] as number)) out[index] = candidate;
  };

  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const index = row * cols + col;
      if (out[index] === 0) continue;
      if (row > 0) relax(index, index - cols, D1);
      if (col > 0) relax(index, index - 1, D1);
      if (row > 0 && col > 0) relax(index, index - cols - 1, D2);
      if (row > 0 && col < cols - 1) relax(index, index - cols + 1, D2);
    }
  }
  for (let row = rows - 1; row >= 0; row--) {
    for (let col = cols - 1; col >= 0; col--) {
      const index = row * cols + col;
      if (out[index] === 0) continue;
      if (row < rows - 1) relax(index, index + cols, D1);
      if (col < cols - 1) relax(index, index + 1, D1);
      if (row < rows - 1 && col < cols - 1) relax(index, index + cols + 1, D2);
      if (row < rows - 1 && col > 0) relax(index, index + cols - 1, D2);
    }
  }
  return out;
}

/**
 * Chambers: connected components of "wide enough to be a room".
 *
 * The threshold walks down from `chamberClearance` until at least two chambers
 * exist, because a cave that happens to be uniformly narrow would otherwise have
 * no deepest chamber for the objective to sit in. The floor of the walk is the
 * corridor radius, at which point every floor cell qualifies and the whole cave
 * is one chamber — still a valid answer, and the entrance/deepest pair below
 * then falls back to raw distance.
 */
function findChambers(
  cells: Uint8Array,
  clearance: Float32Array,
  distance: Int32Array,
  config: Required<DungeonOptions>,
  grid: Pick<DungeonLayout, 'cols' | 'rows' | 'cellSize'>,
  maxDepth: number,
): CaveChamber[] {
  const { cols, rows } = grid;
  let threshold = config.chamberClearance;
  let groups: number[][] = [];
  for (; threshold >= config.corridorRadiusCells; threshold -= 0.4) {
    const mask = new Uint8Array(cells.length);
    for (let i = 0; i < cells.length; i++) {
      if (cells[i] === CaveCell.Floor && (clearance[i] as number) >= threshold) {
        mask[i] = CaveCell.Floor;
      }
    }
    groups = findRegions(mask, cols, rows).filter((g) => g.length >= 6);
    if (groups.length >= 2) break;
  }

  // The far end is always a chamber, whether or not it is wide.
  //
  // Chambers are detected by width, and the deepest *wide* place is not always
  // the deepest place: a cave can trail off into a long narrow passage, and then
  // `deepestChamber` — where the quest objective goes — lands a third of the way
  // in while the player walks past it to the actual end. So when no detected
  // chamber reaches the back of the cave, the terminal region (everything within
  // 15% of the maximum depth) is admitted as a chamber in its own right.
  const FAR_BAND = 0.85;
  const reachesBack = groups.some((group) =>
    group.some((index) => (distance[index] ?? -1) >= maxDepth * FAR_BAND),
  );
  if (!reachesBack && maxDepth > 0) {
    const far: number[] = [];
    for (let i = 0; i < cells.length; i++) {
      if (cells[i] === CaveCell.Floor && (distance[i] as number) >= maxDepth * FAR_BAND) far.push(i);
    }
    if (far.length > 0) groups.push(far);
  }

  if (groups.length === 0) {
    // Degenerate cave: a single chamber at the far end is still a valid answer.
    groups = [[deepestCell(cells, distance)]];
  }

  const chambers = groups.map((group, id) => {
    let center = group[0] as number;
    let bestClearance = -1;
    let count = 0;
    for (const index of group) {
      count++;
      const c = clearance[index] as number;
      if (c > bestClearance) {
        bestClearance = c;
        center = index;
      }
    }
    const col = center % cols;
    const row = (center - col) / cols;
    const depth = Math.max(0, distance[center] ?? 0);
    return {
      id,
      center: { col, row },
      centerWorld: worldOfCell(grid, col, row),
      cellCount: count,
      radiusCells: bestClearance,
      depth,
      depthRatio: maxDepth > 0 ? depth / maxDepth : 0,
    } satisfies CaveChamber;
  });

  return chambers;
}

function deepestCell(cells: Uint8Array, distance: Int32Array): number {
  let best = -1;
  let index = 0;
  for (let i = 0; i < cells.length; i++) {
    if (cells[i] !== CaveCell.Floor) continue;
    const d = distance[i] as number;
    if (d > best) {
      best = d;
      index = i;
    }
  }
  return index;
}

/**
 * Distribute enemy spawns, weighted away from the entrance.
 *
 * Weighted sampling without replacement rather than "take the N deepest": the
 * latter piles every skeleton into the back chamber and leaves the approach
 * empty, which turns a den into a corridor with a boss room. The weight is
 * `depthRatio²`, so the far half of the cave gets roughly four times the density
 * of the near half while the near half is still populated.
 *
 * A greedy minimum-spacing filter runs on top, because two skeletons spawned in
 * the same square metre push each other through a wall on the first frame.
 */
function placeSpawns(
  cells: Uint8Array,
  clearance: Float32Array,
  distance: Int32Array,
  config: Required<DungeonOptions>,
  grid: Pick<DungeonLayout, 'cols' | 'rows' | 'cellSize'>,
  maxDepth: number,
  rng: Rng,
): CaveSpawnPoint[] {
  const { cols } = grid;
  const roomy = config.corridorRadiusCells + 0.5;

  interface Candidate {
    index: number;
    depth: number;
    weight: number;
  }
  const candidates: Candidate[] = [];
  for (let i = 0; i < cells.length; i++) {
    if (cells[i] !== CaveCell.Floor) continue;
    const depth = distance[i] as number;
    if (depth < config.minSpawnDepthCells) continue;
    if ((clearance[i] as number) < roomy) continue;
    const ratio = maxDepth > 0 ? depth / maxDepth : 0;
    candidates.push({ index: i, depth, weight: ratio * ratio + 0.08 });
  }
  if (candidates.length === 0) return [];

  // Efraimidis-Spirakis: one exponential key per item, sort descending, and the
  // prefix of any length is a weighted sample without replacement. One pass, no
  // rejection loop, and fully determined by the seeded stream.
  const keyed = candidates.map((candidate) => ({
    candidate,
    key: Math.pow(rng.next(), 1 / candidate.weight),
  }));
  keyed.sort((a, b) => b.key - a.key || a.candidate.index - b.candidate.index);

  const spacing2 = config.minSpawnSpacingCells * config.minSpawnSpacingCells;
  const chosen: CaveSpawnPoint[] = [];
  const variants = config.variants;
  for (const { candidate } of keyed) {
    if (chosen.length >= config.spawnCount) break;
    const col = candidate.index % cols;
    const row = (candidate.index - col) / cols;
    let clear = true;
    for (const point of chosen) {
      if ((point.cell.col - col) ** 2 + (point.cell.row - row) ** 2 < spacing2) {
        clear = false;
        break;
      }
    }
    if (!clear) continue;

    const ratio = maxDepth > 0 ? candidate.depth / maxDepth : 0;
    const tier = Math.min(variants.length - 1, Math.floor(ratio * variants.length));
    const world = worldOfCell(grid, col, row);
    chosen.push({
      cell: { col, row },
      x: world.x,
      z: world.z,
      depth: candidate.depth,
      depthRatio: ratio,
      variant: variants[tier] ?? 'minion',
      // Sentries deep in, patrols near the front: a wandering skeleton in the
      // back chamber walks out of the encounter it belongs to.
      patrol: ratio > 0.75 ? 0 : 2 + rng.next() * 2,
    });
  }
  return chosen;
}

function clamp(value: number, min: number, max: number): number {
  return value < min ? min : value > max ? max : value;
}

/* -------------------------------------------------------------------------- */
/* Verification helpers (used by the tests and by the drive harness)          */
/* -------------------------------------------------------------------------- */

/** What {@link auditDungeon} found. All three counts must be zero. */
export interface DungeonAudit {
  /** Floor cells the entrance cannot reach. Must be 0. */
  readonly unreachable: number;
  /** Floor cells not covered by a wholly-walkable disc of R. Must be 0. */
  readonly tooNarrow: number;
  /** Floor cells inside the solid margin. Must be 0. */
  readonly inMargin: number;
  readonly floorCells: number;
  readonly reachableCells: number;
}

/**
 * Re-derive every guarantee from the layout alone.
 *
 * Deliberately independent of the code that built the layout: it re-runs the
 * flood fill and re-checks the disc containment from scratch, so a bug in the
 * generator's own bookkeeping cannot make the audit pass.
 */
export function auditDungeon(layout: DungeonLayout): DungeonAudit {
  const { cells, cols, rows, corridorRadiusCells: radius, marginCells: margin } = layout;
  const distance = floodFrom(cells, cols, rows, layout.entrance);
  const offsets = discOffsets(radius);

  let unreachable = 0;
  let tooNarrow = 0;
  let inMargin = 0;
  let floorCells = 0;
  let reachableCells = 0;

  // A cell is wide enough if *some* disc centre within R of it has a wholly
  // walkable disc. Precompute those centres once.
  const centres = new Uint8Array(cells.length);
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const index = row * cols + col;
      if (cells[index] !== CaveCell.Floor) continue;
      let fits = true;
      for (let k = 0; k < offsets.length && fits; k += 2) {
        const c = col + (offsets[k] ?? 0);
        const r = row + (offsets[k + 1] ?? 0);
        if (c < 0 || r < 0 || c >= cols || r >= rows) fits = false;
        else if (cells[r * cols + c] !== CaveCell.Floor) fits = false;
      }
      if (fits) centres[index] = 1;
    }
  }

  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const index = row * cols + col;
      if (cells[index] !== CaveCell.Floor) continue;
      floorCells++;
      if (distance[index] === -1) unreachable++;
      else reachableCells++;
      if (col < margin || row < margin || col >= cols - margin || row >= rows - margin) inMargin++;

      let covered = false;
      for (let k = 0; k < offsets.length && !covered; k += 2) {
        const c = col + (offsets[k] ?? 0);
        const r = row + (offsets[k + 1] ?? 0);
        if (c < 0 || r < 0 || c >= cols || r >= rows) continue;
        if (centres[r * cols + c] === 1) covered = true;
      }
      if (!covered) tooNarrow++;
    }
  }

  return { unreachable, tooNarrow, inMargin, floorCells, reachableCells };
}
