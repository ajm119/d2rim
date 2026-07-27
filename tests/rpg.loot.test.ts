/**
 * Drop tables and the loot roll.
 *
 * Rates are asserted over tens of thousands of seeded rolls rather than on one
 * lucky seed, and the *ordering* between tables is asserted too — a table set
 * where every enemy drops identically would satisfy a per-table rate test and
 * would make the mage no more worth killing than the minion.
 */

import { describe, expect, it } from 'vitest';

import {
  DEFAULT_DROP_TABLE,
  DROP_TABLES,
  dropTableFor,
  killSeed,
  rollDrops,
  sampleDropRates,
  variantFromLabel,
} from '../src/rpg/Loot';
import { GOLD_COLOUR, ITEM_COLOURS } from '../src/rpg/ItemGenerator';

describe('table lookup', () => {
  it('has a table for every skeleton variant', () => {
    for (const variant of ['warrior', 'minion', 'rogue', 'mage']) {
      expect(DROP_TABLES[variant], variant).toBeDefined();
      expect(dropTableFor(variant).id).toBe(variant);
    }
  });

  it('falls back rather than throwing on an unknown variant', () => {
    expect(dropTableFor('balrog')).toBe(DEFAULT_DROP_TABLE);
  });

  it("parses EnemyBase's label format", () => {
    expect(variantFromLabel('warrior#7')).toBe('warrior');
    expect(variantFromLabel('mage#128')).toBe('mage');
    // A label with no id still resolves, so a renamed enemy degrades to a
    // lookup miss rather than to a crash.
    expect(variantFromLabel('warrior')).toBe('warrior');
  });

  it('scales experience and value with the monster', () => {
    const minion = dropTableFor('minion');
    const mage = dropTableFor('mage');
    expect(mage.experience).toBeGreaterThan(minion.experience);
    expect(mage.monsterLevel).toBeGreaterThan(minion.monsterLevel);
    expect(mage.qualityBias).toBeGreaterThan(minion.qualityBias);
  });
});

describe('kill seeds', () => {
  it('differs between two kills of the same combatant', () => {
    expect(killSeed(4, 0)).not.toBe(killSeed(4, 1));
  });

  it('differs between two combatants killed at the same index', () => {
    expect(killSeed(4, 0)).not.toBe(killSeed(5, 0));
  });

  it('reproduces exactly for the same pair', () => {
    expect(killSeed(11, 3)).toBe(killSeed(11, 3));
  });

  it('stays a 32-bit unsigned integer', () => {
    for (let i = 0; i < 200; i++) {
      const seed = killSeed(i * 37, i);
      expect(Number.isInteger(seed)).toBe(true);
      expect(seed).toBeGreaterThanOrEqual(0);
      expect(seed).toBeLessThanOrEqual(0xffffffff);
    }
  });
});

describe('rollDrops', () => {
  it('is deterministic for a seed', () => {
    const table = dropTableFor('warrior');
    const a = rollDrops(table, { seed: 12345 });
    const b = rollDrops(table, { seed: 12345 });
    expect(JSON.stringify(b)).toBe(JSON.stringify(a));
  });

  it('produces different results for different seeds', () => {
    const table = dropTableFor('warrior');
    const results = new Set<string>();
    for (let seed = 1; seed <= 50; seed++) {
      results.add(JSON.stringify(rollDrops(table, { seed })));
    }
    expect(results.size).toBeGreaterThan(20);
  });

  it('lands within a couple of points of the declared gold chance', () => {
    for (const variant of ['minion', 'warrior', 'mage']) {
      const table = dropTableFor(variant);
      const measured = sampleDropRates(table, 20_000);
      expect(measured.goldRate, variant).toBeGreaterThan(table.goldChance - 0.02);
      expect(measured.goldRate, variant).toBeLessThan(table.goldChance + 0.02);
    }
  });

  it('lands within a couple of points of the declared item chance', () => {
    for (const variant of ['minion', 'warrior', 'mage']) {
      const table = dropTableFor(variant);
      const measured = sampleDropRates(table, 20_000);
      expect(measured.itemRate, variant).toBeGreaterThan(table.itemChance - 0.02);
      expect(measured.itemRate, variant).toBeLessThan(table.itemChance + 0.02);
    }
  });

  it('keeps gold inside the declared range', () => {
    const table = dropTableFor('mage');
    let seenMin = Infinity;
    let seenMax = -Infinity;
    for (let seed = 1; seed <= 20_000; seed++) {
      const gold = rollDrops(table, { seed }).gold;
      if (gold === 0) continue;
      seenMin = Math.min(seenMin, gold);
      seenMax = Math.max(seenMax, gold);
    }
    expect(seenMin).toBeGreaterThanOrEqual(table.goldMin);
    expect(seenMax).toBeLessThanOrEqual(table.goldMax);
    // Both ends of the range must actually be reachable.
    expect(seenMin).toBe(table.goldMin);
    expect(seenMax).toBe(table.goldMax);
  });

  it('rolls items at the monster level unless told otherwise', () => {
    const table = dropTableFor('warrior');
    for (let seed = 1; seed <= 500; seed++) {
      for (const item of rollDrops(table, { seed }).items) {
        expect(item.itemLevel).toBe(table.monsterLevel);
      }
    }
    for (let seed = 1; seed <= 200; seed++) {
      for (const item of rollDrops(table, { seed, itemLevel: 25 }).items) {
        expect(item.itemLevel).toBe(25);
      }
    }
  });

  it('makes the better table drop better items', () => {
    const minion = sampleDropRates(dropTableFor('minion'), 30_000);
    const mage = sampleDropRates(dropTableFor('mage'), 30_000);
    const interesting = (counts: Record<string, number>): number =>
      (counts['magic'] ?? 0) + (counts['rare'] ?? 0) + (counts['unique'] ?? 0);
    expect(interesting(mage.qualities) / mage.itemRate).toBeGreaterThan(
      interesting(minion.qualities) / minion.itemRate,
    );
  });

  it('does not correlate the gold roll with the item roll', () => {
    // If the two shared a stream, "gold dropped" would predict "item dropped".
    const table = dropTableFor('warrior');
    let goldAndItem = 0;
    let gold = 0;
    let items = 0;
    const samples = 30_000;
    for (let seed = 1; seed <= samples; seed++) {
      const roll = rollDrops(table, { seed });
      if (roll.gold > 0) gold++;
      if (roll.items.length > 0) items++;
      if (roll.gold > 0 && roll.items.length > 0) goldAndItem++;
    }
    const joint = goldAndItem / samples;
    const independent = (gold / samples) * (items / samples);
    expect(Math.abs(joint - independent)).toBeLessThan(0.02);
  });

  it('honours an extra quality bias', () => {
    const table = dropTableFor('minion');
    const plain = { normal: 0, interesting: 0 };
    const boosted = { normal: 0, interesting: 0 };
    for (let seed = 1; seed <= 20_000; seed++) {
      for (const item of rollDrops(table, { seed }).items) {
        if (item.quality === 'normal') plain.normal++;
        else plain.interesting++;
      }
      for (const item of rollDrops(table, { seed, qualityBias: 3 }).items) {
        if (item.quality === 'normal') boosted.normal++;
        else boosted.interesting++;
      }
    }
    expect(boosted.interesting).toBeGreaterThan(plain.interesting);
  });
});

describe('label colours', () => {
  it('uses the D2 convention for every quality', () => {
    const seen = new Set<string>();
    for (let seed = 1; seed <= 5000; seed++) {
      for (const item of rollDrops(dropTableFor('mage'), { seed, itemLevel: 30 }).items) {
        seen.add(ITEM_COLOURS[item.quality]);
      }
    }
    expect(seen.has(ITEM_COLOURS.normal)).toBe(true);
    expect(seen.has(ITEM_COLOURS.magic)).toBe(true);
    expect(seen.has(ITEM_COLOURS.rare)).toBe(true);
  });

  it('gives gold its own colour, distinct from every item quality', () => {
    expect(Object.values(ITEM_COLOURS)).not.toContain(GOLD_COLOUR);
  });
});
