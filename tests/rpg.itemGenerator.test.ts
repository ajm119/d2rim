/**
 * Item generation: determinism, affix count bounds per quality, item-level
 * gating, and that no affix ever rolls outside its declared range.
 *
 * Every property here is checked over a *sweep* of seeds rather than one, and
 * several are checked in both directions. A "no affix rolls above its range"
 * test passes trivially against a generator that always returns the minimum, so
 * the extremes are asserted to be reachable too; a "seed is deterministic" test
 * passes trivially against a generator that ignores its input, so distinct
 * seeds are asserted to diverge.
 */

import { describe, expect, it } from 'vitest';

import { mulberry32 } from '../src/combat/DamageModel';
import {
  AFFIXES,
  AFFIX_BOUNDS,
  ITEM_BASES,
  ITEM_COLOURS,
  UNIQUES,
  cloneItem,
  composeName,
  describeItem,
  eligibleAffixes,
  eligibleBases,
  findAffix,
  findBase,
  generateItem,
  isBroken,
  itemColour,
  requiredLevelFor,
  rollQuality,
  sumItemModifiers,
  type Item,
  type ItemQuality,
} from '../src/rpg/ItemGenerator';
import { MODIFIER_KEYS } from '../src/rpg/Stats';

/** Generate a sweep of items at one item level. */
function sweep(count: number, itemLevel: number, options: Record<string, unknown> = {}): Item[] {
  const out: Item[] = [];
  for (let seed = 1; seed <= count; seed++) {
    out.push(generateItem({ seed: seed * 2654435761, itemLevel, ...options } as never));
  }
  return out;
}

describe('the tables themselves', () => {
  it('has unique base ids and a positive footprint for every base', () => {
    const ids = new Set(ITEM_BASES.map((entry) => entry.id));
    expect(ids.size).toBe(ITEM_BASES.length);
    for (const base of ITEM_BASES) {
      expect(base.width).toBeGreaterThan(0);
      expect(base.height).toBeGreaterThan(0);
      // Nothing may be larger than the 10x4 bag, or it could never be picked up.
      expect(base.width).toBeLessThanOrEqual(10);
      expect(base.height).toBeLessThanOrEqual(4);
    }
  });

  it('has unique affix ids, sane ranges and only known modifier keys', () => {
    const ids = new Set(AFFIXES.map((entry) => entry.id));
    expect(ids.size).toBe(AFFIXES.length);
    for (const affix of AFFIXES) {
      expect(affix.weight).toBeGreaterThan(0);
      expect(affix.level).toBeGreaterThanOrEqual(1);
      expect(affix.categories.length).toBeGreaterThan(0);
      expect(affix.mods.length).toBeGreaterThan(0);
      for (const mod of affix.mods) {
        expect(MODIFIER_KEYS).toContain(mod.stat);
        expect(mod.max).toBeGreaterThanOrEqual(mod.min);
      }
    }
  });

  it('names every unique against a base that exists', () => {
    for (const unique of UNIQUES) {
      expect(findBase(unique.baseId), `${unique.id} -> ${unique.baseId}`).not.toBeNull();
    }
  });

  it('uses Diablo II item colours', () => {
    expect(ITEM_COLOURS.normal).toBe('#d6cfc0');
    expect(ITEM_COLOURS.magic).toBe('#6f6fff');
    expect(ITEM_COLOURS.rare).toBe('#ffff64');
    expect(ITEM_COLOURS.unique).toBe('#a08554');
  });
});

describe('determinism', () => {
  it('produces an identical item for the same seed', () => {
    for (const seed of [1, 7, 42, 9999, 0x7fffffff]) {
      const a = generateItem({ seed, itemLevel: 12 });
      const b = generateItem({ seed, itemLevel: 12 });
      expect(JSON.stringify(b)).toBe(JSON.stringify(a));
    }
  });

  it('produces different items for different seeds', () => {
    // Without this the determinism test above is satisfied by a generator that
    // returns a constant.
    const names = new Set(sweep(60, 12).map((item) => `${item.name}|${JSON.stringify(item.mods)}`));
    expect(names.size).toBeGreaterThan(20);
  });

  it('is insensitive to the order items are generated in', () => {
    const forwards = [3, 5, 8].map((seed) => generateItem({ seed, itemLevel: 9 }));
    const backwards = [8, 5, 3].map((seed) => generateItem({ seed, itemLevel: 9 })).reverse();
    expect(JSON.stringify(backwards)).toBe(JSON.stringify(forwards));
  });

  it('derives a reproducible uid from the seed', () => {
    const item = generateItem({ seed: 11, itemLevel: 4, baseId: 'short-sword' });
    expect(item.uid).toBe(generateItem({ seed: 11, itemLevel: 4, baseId: 'short-sword' }).uid);
    expect(item.seed).toBe(11);
  });
});

describe('affix count bounds', () => {
  it('gives a normal item no affixes at all', () => {
    for (const item of sweep(40, 20, { quality: 'normal' })) {
      expect(item.affixes).toHaveLength(0);
      expect(item.mods).toEqual({});
      expect(item.name).toBe(findBase(item.baseId)?.name);
    }
  });

  it('gives a magic item one prefix and/or one suffix, never two of a kind', () => {
    const shapes = new Set<string>();
    for (const item of sweep(200, 20, { quality: 'magic' })) {
      const prefixes = item.affixes.filter((entry) => entry.kind === 'prefix');
      const suffixes = item.affixes.filter((entry) => entry.kind === 'suffix');
      expect(prefixes.length).toBeLessThanOrEqual(1);
      expect(suffixes.length).toBeLessThanOrEqual(1);
      expect(item.affixes.length).toBeGreaterThanOrEqual(AFFIX_BOUNDS.magic.min);
      expect(item.affixes.length).toBeLessThanOrEqual(AFFIX_BOUNDS.magic.max);
      shapes.add(`${prefixes.length}${suffixes.length}`);
    }
    // All three legal shapes must actually occur, or the bound test above is
    // satisfied by a generator that only ever rolls one of them.
    expect(shapes).toEqual(new Set(['10', '01', '11']));
  });

  it('gives a rare between two and six affixes', () => {
    const counts = new Set<number>();
    for (const item of sweep(300, 30, { quality: 'rare' })) {
      expect(item.affixes.length).toBeGreaterThanOrEqual(AFFIX_BOUNDS.rare.min);
      expect(item.affixes.length).toBeLessThanOrEqual(AFFIX_BOUNDS.rare.max);
      counts.add(item.affixes.length);
    }
    // The top of the range has to be reachable.
    expect(Math.max(...counts)).toBe(6);
    expect(Math.min(...counts)).toBe(2);
  });

  it('never repeats an affix group on one item', () => {
    for (const item of sweep(400, 30, { quality: 'rare' })) {
      const groups = item.affixes.map((entry) => findAffix(entry.affixId)?.group ?? entry.affixId);
      expect(new Set(groups).size).toBe(groups.length);
    }
  });

  it('gives a rare a two-word name in front of its base', () => {
    for (const item of sweep(30, 30, { quality: 'rare' })) {
      const base = findBase(item.baseId);
      expect(base).not.toBeNull();
      expect(item.name.endsWith(base?.name ?? '')).toBe(true);
      const prefixWords = item.name.slice(0, item.name.length - (base?.name.length ?? 0)).trim();
      expect(prefixWords.split(/\s+/)).toHaveLength(2);
    }
  });
});

describe('item level gating', () => {
  it('never rolls an affix whose level exceeds the item level', () => {
    for (const itemLevel of [1, 2, 3, 5, 8, 12, 20, 30]) {
      for (const item of sweep(120, itemLevel, { quality: 'rare' })) {
        expect(item.itemLevel).toBe(itemLevel);
        for (const rolled of item.affixes) {
          const affix = findAffix(rolled.affixId);
          if (affix === null) continue; // uniques carry their own id
          expect(affix.level, `${affix.id} on ilvl ${itemLevel}`).toBeLessThanOrEqual(itemLevel);
        }
      }
    }
  });

  it('does let high-level affixes through at a high item level', () => {
    // The negative test above passes against a generator that rolls nothing at
    // all; this one fails against it.
    const seen = new Set<string>();
    for (const item of sweep(400, 30, { quality: 'rare' })) {
      for (const rolled of item.affixes) seen.add(rolled.affixId);
    }
    expect(seen.has('vicious') || seen.has('ferocious') || seen.has('of-butchery')).toBe(true);
  });

  it('excludes low-level items from high-level affixes entirely', () => {
    const lowLevelPool = eligibleAffixes('weapon', 2).map((entry) => entry.id);
    expect(lowLevelPool).toContain('sharp');
    expect(lowLevelPool).not.toContain('vicious');
    expect(lowLevelPool).not.toContain('warriors');
  });

  it('respects the category an affix declares', () => {
    for (const affix of eligibleAffixes('ring', 30)) {
      expect(affix.categories).toContain('ring');
    }
    expect(eligibleAffixes('ring', 30).map((a) => a.id)).not.toContain('sharp');
    expect(eligibleAffixes('weapon', 30).map((a) => a.id)).not.toContain('sturdy');
  });

  it('gates base items by their own quality level', () => {
    const lowBases = eligibleBases(2).map((base) => base.id);
    expect(lowBases).toContain('hand-axe');
    expect(lowBases).not.toContain('war-axe');
    expect(eligibleBases(30).map((base) => base.id)).toContain('war-axe');
  });

  it('only generates bases eligible at the item level', () => {
    for (const item of sweep(200, 3)) {
      const base = findBase(item.baseId);
      expect(base?.qualityLevel ?? 99).toBeLessThanOrEqual(3);
    }
  });
});

describe('affix ranges', () => {
  it('never rolls a value outside the declared min and max', () => {
    for (const itemLevel of [1, 5, 12, 30]) {
      for (const item of sweep(250, itemLevel, { quality: 'rare' })) {
        for (const rolled of item.affixes) {
          const affix = findAffix(rolled.affixId);
          if (affix === null) continue;
          expect(rolled.values).toHaveLength(affix.mods.length);
          rolled.values.forEach((value, index) => {
            const mod = affix.mods[index];
            expect(mod).toBeDefined();
            expect(value.stat).toBe(mod?.stat);
            expect(value.value).toBeGreaterThanOrEqual(mod?.min ?? 0);
            expect(value.value).toBeLessThanOrEqual(mod?.max ?? 0);
            expect(Number.isInteger(value.value)).toBe(true);
          });
        }
      }
    }
  });

  it('reaches both ends of a range over enough rolls', () => {
    // Guards against a generator pinned to one end, which would satisfy the
    // bounds test above without rolling anything.
    const values: number[] = [];
    for (let seed = 1; seed <= 4000; seed++) {
      const item = generateItem({
        seed,
        itemLevel: 4,
        baseId: 'short-sword',
        quality: 'magic',
      });
      for (const rolled of item.affixes) {
        if (rolled.affixId !== 'sharp') continue;
        values.push(rolled.values[0]?.value ?? 0);
      }
    }
    expect(values.length).toBeGreaterThan(50);
    expect(Math.min(...values)).toBe(10);
    expect(Math.max(...values)).toBe(20);
  });

  it('aggregates affix values into the modifier bag exactly', () => {
    for (const item of sweep(200, 30, { quality: 'rare' })) {
      const expected = new Map<string, number>();
      for (const rolled of item.affixes) {
        for (const { stat, value } of rolled.values) {
          expected.set(stat, (expected.get(stat) ?? 0) + value);
        }
      }
      for (const [stat, value] of expected) {
        expect(item.mods[stat as never]).toBe(value);
      }
      // And nothing that was not rolled appears in the bag.
      for (const key of Object.keys(item.mods)) expect(expected.has(key)).toBe(true);
    }
  });
});

describe('uniques', () => {
  it('produces the named unique with its own base and mods in range', () => {
    for (const unique of UNIQUES) {
      const item = generateItem({ seed: 1234, itemLevel: 30, uniqueId: unique.id });
      expect(item.quality).toBe('unique');
      expect(item.uniqueId).toBe(unique.id);
      expect(item.baseId).toBe(unique.baseId);
      expect(item.name.startsWith(unique.name)).toBe(true);
      const rolled = item.affixes[0];
      expect(rolled).toBeDefined();
      rolled?.values.forEach((value, index) => {
        const mod = unique.mods[index];
        expect(value.value).toBeGreaterThanOrEqual(mod?.min ?? 0);
        expect(value.value).toBeLessThanOrEqual(mod?.max ?? 0);
      });
    }
  });

  it('downgrades rather than emitting a unique with no unique properties', () => {
    // No unique exists for a Club, so forcing unique on one must not produce an
    // item whose colour promises something it does not have.
    const item = generateItem({ seed: 5, itemLevel: 30, baseId: 'club', quality: 'unique' });
    expect(item.quality).not.toBe('unique');
    if (item.affixes.length > 0) expect(item.mods).not.toEqual({});
  });

  it('gates uniques by item level', () => {
    const item = generateItem({ seed: 5, itemLevel: 1, baseId: 'ring', quality: 'unique' });
    // Nagelring is level 7; at ilvl 1 nothing is eligible.
    expect(item.uniqueId).toBeUndefined();
  });
});

describe('quality rolls', () => {
  it('produces a Diablo-shaped distribution', () => {
    const counts: Record<ItemQuality, number> = { normal: 0, magic: 0, rare: 0, unique: 0 };
    const rng = mulberry32(99);
    for (let i = 0; i < 40_000; i++) counts[rollQuality(5, rng)]++;

    expect(counts.normal).toBeGreaterThan(counts.magic);
    expect(counts.magic).toBeGreaterThan(counts.rare);
    expect(counts.rare).toBeGreaterThan(counts.unique);
    expect(counts.unique).toBeGreaterThan(0);
    expect(counts.normal / 40_000).toBeGreaterThan(0.55);
  });

  it('gets more generous with item level and with the bias', () => {
    const rate = (itemLevel: number, bias: number): number => {
      const rng = mulberry32(7);
      let interesting = 0;
      for (let i = 0; i < 20_000; i++) {
        if (rollQuality(itemLevel, rng, bias) !== 'normal') interesting++;
      }
      return interesting / 20_000;
    };
    expect(rate(30, 1)).toBeGreaterThan(rate(2, 1));
    expect(rate(5, 2)).toBeGreaterThan(rate(5, 1));
  });
});

describe('derived item properties', () => {
  it('sets a required level from the strongest affix', () => {
    const base = findBase('short-sword');
    expect(base).not.toBeNull();
    if (base === null) return;
    expect(requiredLevelFor(base, [], 'normal')).toBe(1);
    const strong = requiredLevelFor(
      base,
      [{ affixId: 'vicious', kind: 'prefix', name: 'Vicious', values: [] }],
      'magic',
    );
    expect(strong).toBe(Math.ceil(20 * 0.75));
  });

  it('prices quality and affixes into the value', () => {
    const normal = generateItem({ seed: 1, itemLevel: 5, baseId: 'short-sword', quality: 'normal' });
    const magic = generateItem({ seed: 1, itemLevel: 5, baseId: 'short-sword', quality: 'magic' });
    const rare = generateItem({ seed: 1, itemLevel: 20, baseId: 'short-sword', quality: 'rare' });
    expect(magic.value).toBeGreaterThan(normal.value);
    expect(rare.value).toBeGreaterThan(magic.value);
  });

  it('composes magic names as prefix + base + suffix', () => {
    const base = findBase('short-sword');
    if (base === null) return;
    const name = composeName(base, 'magic', [
      { affixId: 'sharp', kind: 'prefix', name: 'Sharp', values: [] },
      { affixId: 'of-the-fox', kind: 'suffix', name: 'of the Fox', values: [] },
    ]);
    expect(name).toBe('Sharp Short Sword of the Fox');
  });

  it('reports the quality colour on the item', () => {
    const magic = generateItem({ seed: 3, itemLevel: 5, quality: 'magic', baseId: 'ring' });
    expect(itemColour(magic)).toBe(ITEM_COLOURS.magic);
  });

  it('describes an item as a tooltip would', () => {
    const item = generateItem({ seed: 3, itemLevel: 12, baseId: 'short-sword', quality: 'magic' });
    const lines = describeItem(item);
    expect(lines[0]).toMatch(/^Damage: \d+ to \d+$/);
    expect(lines.some((line) => line.includes('Durability'))).toBe(true);
  });

  it('clones deeply enough that mutating a copy cannot touch the original', () => {
    const item = generateItem({ seed: 3, itemLevel: 12, baseId: 'short-sword', quality: 'rare' });
    const copy = cloneItem(item);
    copy.durability = 0;
    (copy.mods as Record<string, number>)['strength'] = 999;
    expect(item.durability).toBe(item.maxDurability);
    expect(item.mods['strength']).not.toBe(999);
  });
});

describe('broken items', () => {
  it('counts as broken only once durability is exhausted', () => {
    const item = generateItem({ seed: 3, itemLevel: 5, baseId: 'short-sword', quality: 'magic' });
    expect(isBroken(item)).toBe(false);
    item.durability = 0;
    expect(isBroken(item)).toBe(true);

    const ring = generateItem({ seed: 3, itemLevel: 5, baseId: 'ring', quality: 'magic' });
    // Jewellery has no durability and can never break.
    expect(isBroken(ring)).toBe(false);
  });

  it('contributes nothing while broken', () => {
    const item = generateItem({ seed: 21, itemLevel: 20, baseId: 'short-sword', quality: 'rare' });
    const working = sumItemModifiers([item]);
    item.durability = 0;
    const broken = sumItemModifiers([item]);
    expect(Object.values(working).some((value) => value !== 0)).toBe(true);
    expect(Object.values(broken).every((value) => value === 0)).toBe(true);
  });
});
