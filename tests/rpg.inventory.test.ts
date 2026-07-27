/**
 * The grid inventory and the equipment paper doll.
 *
 * The grid tests check occupancy *cell by cell* rather than by counting items,
 * because an implementation that tracks a free-slot count instead of a real
 * footprint agrees with a correct one on every test that only counts. Item
 * shapes are deliberately non-square (1x3 swords, 2x1 belts, 1x1 rings) for the
 * same reason: a width/height transposition is invisible on a square.
 */

import { describe, expect, it } from 'vitest';

import { Character } from '../src/rpg/Character';
import {
  Equipment,
  EQUIPMENT_SLOTS,
  INVENTORY_HEIGHT,
  INVENTORY_WIDTH,
  InventoryGrid,
  meetsRequirements,
  slotsFor,
  unmetRequirement,
} from '../src/rpg/Inventory';
import { generateItem, type Item } from '../src/rpg/ItemGenerator';

/** A plain item of a known base, so footprints are known exactly. */
function item(baseId: string, seed = 1): Item {
  return generateItem({ seed, itemLevel: 1, baseId, quality: 'normal' });
}

/** Full occupancy map, as a string per row. `.` is empty. */
function occupancy(grid: InventoryGrid): string[] {
  const rows: string[] = [];
  for (let y = 0; y < grid.height; y++) {
    let row = '';
    for (let x = 0; x < grid.width; x++) {
      const held = grid.itemAt(x, y);
      row += held === null ? '.' : held.baseId.charAt(0);
    }
    rows.push(row);
  }
  return rows;
}

describe('InventoryGrid geometry', () => {
  it("is Diablo II's ten by four", () => {
    const grid = new InventoryGrid();
    expect(grid.width).toBe(INVENTORY_WIDTH);
    expect(grid.height).toBe(INVENTORY_HEIGHT);
    expect(grid.freeCells).toBe(40);
  });

  it('marks exactly the cells an item covers, and no others', () => {
    const grid = new InventoryGrid();
    const sword = item('short-sword'); // 1 wide, 3 tall
    expect(sword.width).toBe(1);
    expect(sword.height).toBe(3);
    expect(grid.place(sword, 2, 0)).toBe(true);

    expect(occupancy(grid)).toEqual([
      '..s.......',
      '..s.......',
      '..s.......',
      '..........',
    ]);
    expect(grid.freeCells).toBe(37);
  });

  it('distinguishes width from height', () => {
    const grid = new InventoryGrid();
    const belt = item('sash'); // 2 wide, 1 tall
    expect(belt.width).toBe(2);
    expect(belt.height).toBe(1);
    grid.place(belt, 0, 1);
    expect(occupancy(grid)).toEqual([
      '..........',
      'ss........',
      '..........',
      '..........',
    ]);
  });

  it('refuses placements that would hang off any edge', () => {
    const grid = new InventoryGrid();
    const armour = item('quilted-armor'); // 2 x 3
    expect(grid.canPlace(armour, 9, 0)).toBe(false); // one column short
    expect(grid.canPlace(armour, 0, 2)).toBe(false); // one row short
    expect(grid.canPlace(armour, 8, 1)).toBe(true);
    expect(grid.canPlace(armour, -1, 0)).toBe(false);
    expect(grid.canPlace(armour, 0, -1)).toBe(false);
  });

  it('refuses non-integer coordinates rather than silently flooring them', () => {
    const grid = new InventoryGrid();
    expect(grid.place(item('ring'), 1.5, 0)).toBe(false);
    expect(grid.count).toBe(0);
  });
});

describe('collision', () => {
  it('refuses a placement that overlaps by even one cell', () => {
    const grid = new InventoryGrid();
    const a = item('short-sword', 1);
    const b = item('short-sword', 2);
    grid.place(a, 3, 0);

    expect(grid.canPlace(b, 3, 0)).toBe(false);
    expect(grid.place(b, 3, 0)).toBe(false);
    // Directly beside it is fine.
    expect(grid.place(b, 4, 0)).toBe(true);
  });

  it('reports every distinct item a footprint would touch', () => {
    const grid = new InventoryGrid();
    const left = item('short-sword', 1);
    const right = item('short-sword', 2);
    grid.place(left, 0, 0);
    grid.place(right, 1, 0);

    const belt = item('sash');
    expect(grid.overlapping(belt, 0, 1).sort()).toHaveLength(2);
  });

  it('ignores the item being moved when testing its own destination', () => {
    const grid = new InventoryGrid();
    const armour = item('quilted-armor');
    grid.place(armour, 0, 0);
    // Overlaps its own current footprint, but only its own.
    expect(grid.canPlace(armour, 1, 0, armour)).toBe(true);
    expect(grid.canPlace(armour, 1, 0)).toBe(false);
  });
});

describe('first-fit insertion', () => {
  it('fills in reading order', () => {
    const grid = new InventoryGrid();
    const first = grid.add(item('ring', 1));
    const second = grid.add(item('ring', 2));
    expect(first).toEqual(expect.objectContaining({ x: 0, y: 0 }));
    expect(second).toEqual(expect.objectContaining({ x: 1, y: 0 }));
  });

  it('skips a row that cannot take a tall item', () => {
    const grid = new InventoryGrid();
    // A 1x1 in the last row is irrelevant to a 1x3 sword, which can only start
    // at row 0 or 1.
    grid.place(item('ring'), 0, 3);
    const sword = grid.add(item('short-sword'));
    expect(sword?.y).toBe(0);
    expect(sword?.x).toBe(0);
  });

  it('reports honestly when nothing fits', () => {
    const grid = new InventoryGrid(2, 2);
    const belt = item('sash'); // 2x1
    expect(grid.add(belt)).not.toBeNull();
    expect(grid.add(item('sash', 2))).not.toBeNull();
    const third = item('sash', 3);
    expect(grid.hasRoomFor(third)).toBe(false);
    expect(grid.add(third)).toBeNull();
  });

  it('hasRoomFor agrees with add', () => {
    const grid = new InventoryGrid(3, 1);
    const belt = item('sash');
    expect(grid.hasRoomFor(belt)).toBe(true);
    grid.add(belt);
    const second = item('sash', 2);
    expect(grid.hasRoomFor(second)).toBe(false);
    expect(grid.add(second)).toBeNull();
  });
});

describe('drop, move and swap', () => {
  it('places a held item on empty ground', () => {
    const grid = new InventoryGrid();
    const result = grid.drop(item('ring'), 4, 2);
    expect(result).toEqual({ placed: true, displaced: null, reason: 'ok' });
    expect(grid.itemAt(4, 2)).not.toBeNull();
  });

  it('swaps when the target covers exactly one item', () => {
    const grid = new InventoryGrid();
    const sitting = item('ring', 1);
    grid.place(sitting, 4, 2);

    const held = item('ring', 2);
    const result = grid.drop(held, 4, 2);
    expect(result.placed).toBe(true);
    expect(result.displaced).toBe(sitting);
    expect(grid.itemAt(4, 2)).toBe(held);
    expect(grid.contains(sitting)).toBe(false);
  });

  it('refuses when the target covers two items, leaving the grid untouched', () => {
    const grid = new InventoryGrid();
    grid.place(item('ring', 1), 0, 0);
    grid.place(item('ring', 2), 1, 0);
    const before = occupancy(grid);

    const belt = item('sash');
    const result = grid.drop(belt, 0, 0);
    expect(result.placed).toBe(false);
    expect(result.reason).toBe('blocked');
    expect(occupancy(grid)).toEqual(before);
  });

  it('refuses an out-of-bounds drop', () => {
    const grid = new InventoryGrid();
    expect(grid.drop(item('quilted-armor'), 9, 0).reason).toBe('out-of-bounds');
    expect(grid.count).toBe(0);
  });

  it('moves an item without colliding with itself', () => {
    const grid = new InventoryGrid();
    const armour = item('quilted-armor');
    grid.place(armour, 0, 0);
    const result = grid.move({ x: 0, y: 0 }, { x: 1, y: 0 });
    expect(result.placed).toBe(true);
    expect(grid.placementOf(armour)).toEqual({ item: armour, x: 1, y: 0 });
  });

  it('rehomes a swapped-out item when a move displaces one', () => {
    const grid = new InventoryGrid();
    const ring = item('ring', 1);
    const other = item('ring', 2);
    grid.place(ring, 0, 0);
    grid.place(other, 5, 0);

    const result = grid.move({ x: 5, y: 0 }, { x: 0, y: 0 });
    expect(result.placed).toBe(true);
    expect(grid.itemAt(0, 0)).toBe(other);
    // The displaced ring found a home rather than vanishing.
    expect(grid.contains(ring)).toBe(true);
    expect(result.displaced).toBeNull();
  });

  it('reports "not-found" for a move from an empty cell', () => {
    const grid = new InventoryGrid();
    expect(grid.move({ x: 0, y: 0 }, { x: 1, y: 1 }).reason).toBe('not-found');
  });

  it('leaves the grid consistent after a long sequence of operations', () => {
    const grid = new InventoryGrid();
    const items = [
      item('short-sword', 1),
      item('quilted-armor', 2),
      item('sash', 3),
      item('ring', 4),
      item('cap', 5),
    ];
    for (const entry of items) grid.add(entry);

    grid.move({ x: 0, y: 0 }, { x: 7, y: 1 });
    grid.removeAt(7, 1);
    grid.add(items[0] as Item);

    // Every occupied cell must resolve back to a placed item, and every placed
    // item must occupy exactly its own footprint.
    let counted = 0;
    for (const placement of grid.placements) {
      for (let dy = 0; dy < placement.item.height; dy++) {
        for (let dx = 0; dx < placement.item.width; dx++) {
          expect(grid.itemAt(placement.x + dx, placement.y + dy)).toBe(placement.item);
          counted++;
        }
      }
    }
    expect(grid.freeCells).toBe(grid.width * grid.height - counted);
  });
});

describe('persistence', () => {
  it('round-trips placements exactly', () => {
    const grid = new InventoryGrid();
    grid.place(item('short-sword'), 3, 1);
    grid.place(item('ring'), 9, 3);
    const snapshot = grid.toJSON();

    const restored = new InventoryGrid();
    restored.load(snapshot);
    expect(restored.toJSON()).toEqual(snapshot);
    expect(occupancy(restored)).toEqual(occupancy(grid));
  });

  it('rescues a snapshot whose placements no longer fit', () => {
    const grid = new InventoryGrid();
    grid.place(item('short-sword'), 9, 1);
    const snapshot = grid.toJSON();

    const small = new InventoryGrid(4, 4);
    small.load(snapshot);
    // The item survives even though its recorded column is off the edge.
    expect(small.count).toBe(1);
  });
});

describe('Equipment', () => {
  it('accepts an item only in the slot its type belongs to', () => {
    const equipment = new Equipment();
    const sword = item('short-sword');
    expect(equipment.accepts(sword, 'weapon')).toBe(true);
    expect(equipment.accepts(sword, 'head')).toBe(false);
    expect(equipment.equip(sword, 'head').reason).toBe('wrong-slot');
    expect(equipment.get('head')).toBeNull();
  });

  it('puts the second ring on the free finger', () => {
    const equipment = new Equipment();
    const first = item('ring', 1);
    const second = item('ring', 2);
    expect(equipment.equip(first).slot).toBe('ring1');
    expect(equipment.equip(second).slot).toBe('ring2');
    expect(equipment.get('ring1')).toBe(first);
    expect(equipment.get('ring2')).toBe(second);
    expect(slotsFor('ring')).toEqual(['ring1', 'ring2']);
  });

  it('displaces the occupant when both fingers are full', () => {
    const equipment = new Equipment();
    equipment.equip(item('ring', 1));
    equipment.equip(item('ring', 2));
    const third = item('ring', 3);
    const result = equipment.equip(third);
    expect(result.equipped).toBe(true);
    expect(result.displaced).toHaveLength(1);
    expect(equipment.get('ring1')).toBe(third);
  });

  it('enforces the two-handed rule in both directions', () => {
    const equipment = new Equipment();
    const shield = item('buckler');
    const twoHander = generateItem({
      seed: 1,
      itemLevel: 20,
      baseId: 'war-axe',
      quality: 'normal',
    });
    expect(twoHander.twoHanded).toBe(true);

    equipment.equip(shield);
    const wielded = equipment.equip(twoHander);
    expect(wielded.equipped).toBe(true);
    expect(wielded.displaced).toContain(shield);
    expect(equipment.get('offhand')).toBeNull();

    // ... and back the other way.
    const raised = equipment.equip(shield);
    expect(raised.equipped).toBe(true);
    expect(raised.displaced).toContain(twoHander);
    expect(equipment.get('weapon')).toBeNull();
  });

  it('sums modifiers across every equipped item', () => {
    const equipment = new Equipment();
    const ringA = generateItem({ seed: 5, itemLevel: 20, baseId: 'ring', quality: 'magic' });
    const ringB = generateItem({ seed: 6, itemLevel: 20, baseId: 'ring', quality: 'magic' });
    equipment.equip(ringA);
    equipment.equip(ringB);
    const totals = equipment.modifiers();
    for (const key of Object.keys(ringA.mods)) {
      const expected = (ringA.mods[key as never] ?? 0) + (ringB.mods[key as never] ?? 0);
      expect(totals[key as never]).toBe(expected);
    }
  });

  it('reports the weapon damage, armour defence and shield block it holds', () => {
    const equipment = new Equipment();
    expect(equipment.weaponDamage()).toBeUndefined();
    expect(equipment.armourDefense()).toBe(0);
    expect(equipment.blockChance()).toBe(0);

    const sword = item('short-sword');
    const armour = item('quilted-armor');
    const shield = item('buckler');
    equipment.equip(sword);
    equipment.equip(armour);
    equipment.equip(shield);

    expect(equipment.weaponDamage()).toEqual(sword.damage);
    expect(equipment.armourDefense()).toBe((armour.defense ?? 0) + (shield.defense ?? 0));
    expect(equipment.blockChance()).toBe(shield.block);
  });

  it('drops a broken item out of every derived readout', () => {
    const equipment = new Equipment();
    const sword = item('short-sword');
    equipment.equip(sword);
    expect(equipment.weaponDamage()).toBeDefined();
    sword.durability = 0;
    expect(equipment.weaponDamage()).toBeUndefined();
  });

  it('round-trips and rejects a hand-edited slot mismatch', () => {
    const equipment = new Equipment();
    equipment.equip(item('short-sword'));
    equipment.equip(item('cap'));
    const snapshot = equipment.toJSON();

    const restored = new Equipment();
    restored.load(snapshot);
    expect(restored.toJSON()).toEqual(snapshot);

    const tampered = new Equipment();
    tampered.load({ head: item('short-sword') });
    expect(tampered.get('head')).toBeNull();
  });

  it('has a label for every slot', () => {
    for (const slot of EQUIPMENT_SLOTS) {
      expect(slotsFor(slot === 'ring1' || slot === 'ring2' ? 'ring' : slot).length).toBeGreaterThan(0);
    }
  });
});

describe('requirements', () => {
  const strong = { level: 8, strength: 80, dexterity: 40 };

  it('checks level, strength and dexterity', () => {
    const axe = generateItem({ seed: 1, itemLevel: 20, baseId: 'war-axe', quality: 'normal' });
    expect(meetsRequirements(axe, strong)).toBe(true);
    expect(meetsRequirements(axe, { ...strong, strength: 10 })).toBe(false);
    expect(unmetRequirement(axe, { ...strong, strength: 10 })).toBe('strength');
    expect(unmetRequirement(axe, { ...strong, level: 1 })).toBe('level');
    expect(unmetRequirement(axe, strong)).toBeNull();
  });
});

describe('Character equip integration', () => {
  it('moves an item from the bag to the paper doll', () => {
    const character = new Character();
    const sword = item('short-sword');
    character.acquire(sword);
    expect(character.inventory.contains(sword)).toBe(true);

    expect(character.equip(sword).equipped).toBe(true);
    expect(character.inventory.contains(sword)).toBe(false);
    expect(character.equipment.get('weapon')).toBe(sword);
  });

  it('refuses an item whose requirements are not met, without moving it', () => {
    const character = new Character();
    const axe = generateItem({ seed: 1, itemLevel: 20, baseId: 'war-axe', quality: 'normal' });
    character.acquire(axe);
    expect(character.equip(axe).equipped).toBe(false);
    expect(character.inventory.contains(axe)).toBe(true);
  });

  it('never destroys a displaced item when the bag is full', () => {
    const character = new Character();
    const worn = item('ring', 1);
    const worn2 = item('ring', 2);
    character.equipment.equip(worn);
    character.equipment.equip(worn2);

    // Fill the bag completely with 1x1s except one cell for the new ring.
    for (let i = 0; i < 40; i++) character.acquire(item('ring', 100 + i));
    const incoming = character.inventory.items[0] as Item;

    const before = character.inventory.count;
    const result = character.equip(incoming);
    // Either it swapped cleanly, or it refused — but nothing may be lost.
    const carried = character.inventory.count + character.equipment.items().length;
    expect(carried).toBe(before + 2);
    if (!result.equipped) expect(character.inventory.contains(incoming)).toBe(true);
  });

  it('leaves the derived sheet unchanged after equip then unequip', () => {
    const character = new Character();
    // A level 8 character, so the rare ring's own level requirement is met and
    // the equip genuinely happens — an equip that silently refuses would make
    // this test pass for the wrong reason.
    character.addExperience(32_175);
    const before = JSON.stringify(character.derived);
    const ring = generateItem({ seed: 9, itemLevel: 8, baseId: 'ring', quality: 'rare' });
    character.acquire(ring);
    expect(character.equip(ring).equipped).toBe(true);
    const during = JSON.stringify(character.derived);
    const slot = character.equipment.slotOf(ring);
    expect(slot).not.toBeNull();
    if (slot !== null) character.unequip(slot);
    expect(JSON.stringify(character.derived)).toBe(before);
    expect(during).not.toBe(before);
  });

  it('bumps its version counter on every mutation so a UI can diff it', () => {
    const character = new Character();
    const start = character.version;
    character.addGold(10);
    expect(character.version).toBeGreaterThan(start);
    const afterGold = character.version;
    character.acquire(item('ring'));
    expect(character.version).toBeGreaterThan(afterGold);
  });
});
