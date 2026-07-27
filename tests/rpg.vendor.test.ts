/**
 * Buying, selling and repairing.
 *
 * The transaction tests all check the *whole* ledger — gold, bag and shelf —
 * after every operation, because a vendor bug is almost never "the price was
 * wrong"; it is "the gold left but the item did not arrive", and only a check
 * on both sides of the trade can see that.
 */

import { describe, expect, it } from 'vitest';

import { Character } from '../src/rpg/Character';
import { generateItem, type Item } from '../src/rpg/ItemGenerator';
import { Vendor, VENDORS, VendorRegistry, findVendor } from '../src/rpg/Vendor';

function vendor(id: string): Vendor {
  const definition = findVendor(id);
  if (definition === null) throw new Error(`no vendor ${id}`);
  return new Vendor(definition);
}

function item(baseId: string, seed = 1): Item {
  return generateItem({ seed, itemLevel: 1, baseId, quality: 'normal' });
}

describe('the vendor definitions', () => {
  it('covers the camp merchants and only Charsi repairs', () => {
    const ids = VENDORS.map((entry) => entry.id);
    expect(ids).toContain('charsi');
    expect(ids).toContain('gheed');
    expect(VENDORS.filter((entry) => entry.repairs).map((entry) => entry.id)).toEqual(['charsi']);
  });

  it('keeps a wide spread between buying and selling', () => {
    for (const definition of VENDORS) {
      expect(definition.buyMarkup).toBeGreaterThan(1);
      expect(definition.sellMarkdown).toBeLessThan(0.5);
      // Buying back what you just sold must always be a loss.
      expect(definition.buyMarkup).toBeGreaterThan(definition.sellMarkdown * 2);
    }
  });
});

describe('stock generation', () => {
  it('fills the shelf to the declared size', () => {
    const gheed = vendor('gheed');
    expect(gheed.stock).toHaveLength(0);
    gheed.refresh();
    expect(gheed.stock).toHaveLength(gheed.definition.stockSize);
  });

  it('produces identical stock for an identical seed', () => {
    const a = vendor('gheed');
    const b = vendor('gheed');
    a.refresh(4242);
    b.refresh(4242);
    expect(JSON.stringify(b.stock)).toBe(JSON.stringify(a.stock));
  });

  it('produces different stock on each visit', () => {
    const gheed = vendor('gheed');
    const seen = new Set<string>();
    for (let visit = 0; visit < 8; visit++) {
      gheed.refresh();
      seen.add(JSON.stringify(gheed.stock.map((entry) => entry.name)));
    }
    expect(seen.size).toBeGreaterThan(5);
    expect(gheed.visits).toBe(8);
  });

  it('reproduces a given visit number across sessions', () => {
    const first = vendor('charsi');
    first.refresh();
    first.refresh();
    const second = vendor('charsi');
    second.refresh();
    second.refresh();
    expect(JSON.stringify(second.stock)).toBe(JSON.stringify(first.stock));
  });

  it('only stocks the categories the merchant deals in', () => {
    const charsi = vendor('charsi');
    for (let visit = 0; visit < 20; visit++) {
      charsi.refresh();
      for (const entry of charsi.stock) {
        expect(charsi.definition.categories).toContain(entry.category);
      }
    }
  });
});

describe('buying', () => {
  it('moves gold one way and the item the other', () => {
    const charsi = vendor('charsi');
    charsi.refresh(1);
    const character = new Character({ gold: 10_000 });
    const target = charsi.stock[0] as Item;
    const price = charsi.priceToBuy(target);

    const result = charsi.buy(0, character);
    expect(result.result).toBe('ok');
    expect(result.gold).toBe(price);
    expect(character.gold).toBe(10_000 - price);
    expect(character.inventory.contains(target)).toBe(true);
    expect(charsi.stock).not.toContain(target);
  });

  it('refuses without moving anything when the player cannot afford it', () => {
    const charsi = vendor('charsi');
    charsi.refresh(1);
    const character = new Character({ gold: 0 });
    const shelfBefore = charsi.stock.length;

    expect(charsi.buy(0, character).result).toBe('cannot-afford');
    expect(character.gold).toBe(0);
    expect(character.inventory.count).toBe(0);
    expect(charsi.stock).toHaveLength(shelfBefore);
  });

  it('refuses without taking gold when the bag is full', () => {
    const gheed = vendor('gheed');
    gheed.refresh(1);
    const character = new Character({ gold: 100_000 });
    // Fill the grid completely with 1x1 rings.
    for (let i = 0; i < 40; i++) character.acquire(item('ring', 500 + i));
    expect(character.inventory.freeCells).toBe(0);

    // Find something that genuinely will not fit.
    const index = gheed.stock.findIndex((entry) => entry.width * entry.height > 0);
    const before = character.gold;
    expect(gheed.buy(index, character).result).toBe('inventory-full');
    expect(character.gold).toBe(before);
  });

  it('refuses an index that is not on the shelf', () => {
    const charsi = vendor('charsi');
    charsi.refresh(1);
    expect(charsi.buy(99, new Character({ gold: 1000 })).result).toBe('not-in-stock');
  });
});

describe('selling', () => {
  it('moves the item onto the shelf and pays the player', () => {
    const charsi = vendor('charsi');
    charsi.refresh(1);
    const character = new Character({ gold: 0 });
    const sword = item('short-sword');
    character.acquire(sword);
    const price = charsi.priceToSell(sword);

    const result = charsi.sell(sword, character);
    expect(result.result).toBe('ok');
    expect(result.gold).toBe(-price);
    expect(character.gold).toBe(price);
    expect(character.inventory.contains(sword)).toBe(false);
    expect(charsi.stock).toContain(sword);
  });

  it('refuses an item the player is not carrying', () => {
    const charsi = vendor('charsi');
    const character = new Character();
    expect(charsi.sell(item('short-sword'), character).result).toBe('not-owned');
  });

  it('refuses a category the merchant does not deal in', () => {
    const charsi = vendor('charsi');
    const character = new Character();
    const ring = item('ring');
    character.acquire(ring);
    expect(charsi.accepts(ring)).toBe(false);
    expect(charsi.sell(ring, character).result).toBe('not-accepted');
    expect(character.inventory.contains(ring)).toBe(true);
  });

  it('always loses the player money on a round trip', () => {
    const charsi = vendor('charsi');
    charsi.refresh(1);
    const character = new Character({ gold: 10_000 });
    const target = charsi.stock[0] as Item;
    charsi.buy(0, character);
    const afterBuy = character.gold;
    charsi.sell(target, character);
    expect(character.gold).toBeGreaterThan(afterBuy);
    expect(character.gold).toBeLessThan(10_000);
  });

  it('pays less for a broken item', () => {
    const charsi = vendor('charsi');
    const sword = item('short-sword');
    const full = charsi.priceToSell(sword);
    sword.durability = 0;
    expect(charsi.priceToSell(sword)).toBeLessThan(full);
  });
});

describe('repair', () => {
  it('charges nothing for an item at full durability', () => {
    const charsi = vendor('charsi');
    expect(charsi.repairCost(item('short-sword'))).toBe(0);
    expect(charsi.repairCost(item('ring'))).toBe(0);
  });

  it('charges in proportion to what is missing', () => {
    const charsi = vendor('charsi');
    const sword = item('short-sword');
    sword.durability = sword.maxDurability - 1;
    const small = charsi.repairCost(sword);
    sword.durability = 0;
    const large = charsi.repairCost(sword);
    expect(large).toBeGreaterThan(small);
  });

  it('restores every worn item and takes the gold', () => {
    const charsi = vendor('charsi');
    const character = new Character({ gold: 1000 });
    const worn = item('short-sword');
    const armour = item('quilted-armor');
    worn.durability = 0;
    armour.durability = 2;
    character.acquire(worn);
    character.acquire(armour);
    character.equip(worn);

    const cost = charsi.repairAllCost(character);
    expect(cost).toBeGreaterThan(0);
    const result = charsi.repairAll(character);
    expect(result.result).toBe('ok');
    expect(result.gold).toBe(cost);
    expect(character.gold).toBe(1000 - cost);
    expect(worn.durability).toBe(worn.maxDurability);
    expect(armour.durability).toBe(armour.maxDurability);
  });

  it('makes the character sheet recover with the weapon', () => {
    const charsi = vendor('charsi');
    const character = new Character({ gold: 1000 });
    const sword = item('short-sword');
    character.acquire(sword);
    character.equip(sword);
    const healthy = character.derived.damage.max;

    sword.durability = 0;
    character.touch();
    const brokenDamage = character.derived.damage.max;
    expect(brokenDamage).toBeLessThan(healthy);

    charsi.repairAll(character);
    expect(character.derived.damage.max).toBe(healthy);
  });

  it('refuses a repair the player cannot afford, without repairing anything', () => {
    const charsi = vendor('charsi');
    const character = new Character({ gold: 0 });
    const sword = item('short-sword');
    sword.durability = 0;
    character.acquire(sword);
    expect(charsi.repairAll(character).result).toBe('cannot-afford');
    expect(sword.durability).toBe(0);
  });

  it('says so when there is nothing to repair', () => {
    const charsi = vendor('charsi');
    expect(charsi.repairAll(new Character({ gold: 100 })).result).toBe('nothing-to-repair');
  });

  it('refuses the service from a merchant who does not offer it', () => {
    const gheed = vendor('gheed');
    const character = new Character({ gold: 1000 });
    const sword = item('short-sword');
    sword.durability = 0;
    character.acquire(sword);
    expect(gheed.repairAll(character).result).toBe('no-repair-service');
  });
});

describe('VendorRegistry', () => {
  it('resolves by vendor id and by NPC id', () => {
    const registry = new VendorRegistry();
    expect(registry.get('charsi')?.id).toBe('charsi');
    expect(registry.forNpc('gheed')?.id).toBe('gheed');
    expect(registry.forNpc('kashya')).toBeNull();
  });

  it('round-trips every shelf', () => {
    const registry = new VendorRegistry();
    for (const entry of registry.all) entry.refresh();
    const snapshot = registry.toJSON();

    const restored = new VendorRegistry();
    restored.load(snapshot);
    expect(JSON.stringify(restored.toJSON())).toBe(JSON.stringify(snapshot));
  });

  it('reports the qualities currently on sale', () => {
    const registry = new VendorRegistry();
    for (const entry of registry.all) entry.refresh(7);
    const counts = registry.stockQualities();
    const total = Object.values(counts).reduce((sum, value) => sum + value, 0);
    expect(total).toBe(registry.all.reduce((sum, entry) => sum + entry.stock.length, 0));
  });
});
