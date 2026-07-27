/**
 * @module rpg/Vendor
 *
 * Buying, selling and repairing: Gheed's stock, Charsi's forge, and the gold
 * that moves between them and the player.
 *
 * ### Stock is generated, not authored
 *
 * A vendor's inventory is {@link Vendor.refresh}ed from a seed on every visit,
 * which is what makes checking the merchant a thing worth doing rather than a
 * one-time transaction. The seed is derived from a visit counter, so the stock
 * is reproducible for a given save — reload the game and Gheed is selling the
 * same things he was, which is both correct and testable.
 *
 * ### The spread is the sink
 *
 * Buying costs {@link VendorDefinition.buyMarkup} times an item's value;
 * selling returns {@link VendorDefinition.sellMarkdown} times it. The gap
 * between the two is the only gold sink in the act besides repairs, and it is
 * set wide (2x against 0.25x) on purpose: a narrow spread turns the vendor into
 * an arbitrage machine and gold stops meaning anything by the second visit.
 *
 * ### Repair
 *
 * Charsi's service. An item's repair cost is proportional to the durability
 * missing and to what the item is worth, so patching a rare costs real money
 * and patching a Cap costs nearly nothing. Repairing is what makes durability a
 * mechanic rather than a countdown to an item silently doing nothing — see
 * `sumItemModifiers`, which excludes broken items.
 */

import { mulberry32 } from '../combat/DamageModel';
import type { Character } from './Character';
import {
  generateItem,
  isBroken,
  type Item,
  type ItemCategory,
  type ItemQuality,
} from './ItemGenerator';

/* -------------------------------------------------------------------------- */
/* Definitions                                                                 */
/* -------------------------------------------------------------------------- */

export interface VendorDefinition {
  readonly id: string;
  /** NPC this vendor belongs to. */
  readonly npcId: string;
  readonly displayName: string;
  /** How many items are on the shelf after a refresh. */
  readonly stockSize: number;
  /** Base categories this vendor deals in. */
  readonly categories: readonly ItemCategory[];
  /** Item level the stock is generated at. */
  readonly itemLevel: number;
  /** Multiplier on the magic/rare/unique odds of the stock. */
  readonly qualityBias: number;
  /** Price multiplier when the player buys. */
  readonly buyMarkup: number;
  /** Price multiplier when the player sells. */
  readonly sellMarkdown: number;
  /** Whether this vendor repairs. */
  readonly repairs: boolean;
  /** What this vendor refuses to buy. Charsi does not want your rings. */
  readonly buysCategories?: readonly ItemCategory[];
}

/** The camp's three merchants. */
export const VENDORS: readonly VendorDefinition[] = [
  {
    id: 'charsi',
    npcId: 'charsi',
    displayName: 'Charsi',
    stockSize: 8,
    categories: ['weapon', 'armour', 'helm', 'shield', 'gloves', 'boots'],
    itemLevel: 5,
    qualityBias: 1.4,
    buyMarkup: 2,
    sellMarkdown: 0.25,
    repairs: true,
    buysCategories: ['weapon', 'armour', 'helm', 'shield', 'gloves', 'boots', 'belt'],
  },
  {
    id: 'gheed',
    npcId: 'gheed',
    displayName: 'Gheed',
    stockSize: 10,
    categories: ['ring', 'amulet', 'belt', 'boots', 'gloves', 'weapon'],
    itemLevel: 8,
    qualityBias: 2.4,
    buyMarkup: 2.4,
    sellMarkdown: 0.3,
    repairs: false,
  },
  {
    id: 'akara',
    npcId: 'akara',
    displayName: 'Akara',
    stockSize: 6,
    categories: ['amulet', 'ring', 'helm'],
    itemLevel: 6,
    qualityBias: 1.8,
    buyMarkup: 2.2,
    sellMarkdown: 0.25,
    repairs: false,
  },
];

export function findVendor(id: string): VendorDefinition | null {
  return VENDORS.find((entry) => entry.id === id) ?? null;
}

/* -------------------------------------------------------------------------- */
/* Transactions                                                                */
/* -------------------------------------------------------------------------- */

/** Why a transaction was refused. `'ok'` means it went through. */
export type TransactionResult =
  | 'ok'
  | 'not-in-stock'
  | 'cannot-afford'
  | 'inventory-full'
  | 'not-owned'
  | 'not-accepted'
  | 'nothing-to-repair'
  | 'no-repair-service';

export interface Transaction {
  readonly result: TransactionResult;
  /** Gold that changed hands. Positive means the player paid. */
  readonly gold: number;
  readonly item: Item | null;
}

export interface VendorSnapshot {
  readonly id: string;
  readonly visits: number;
  readonly stock: readonly Item[];
}

/**
 * One merchant's shop.
 *
 * Instances are cheap and hold no engine references, so a test constructs one
 * and trades against a `Character` with nothing else present.
 */
export class Vendor {
  readonly definition: VendorDefinition;

  #stock: Item[] = [];
  #visits = 0;

  constructor(definition: VendorDefinition) {
    this.definition = definition;
  }

  get id(): string {
    return this.definition.id;
  }

  get stock(): readonly Item[] {
    return this.#stock;
  }

  get visits(): number {
    return this.#visits;
  }

  /**
   * Regenerate the shelf.
   *
   * `seed` defaults to the visit counter, so successive visits differ and a
   * reloaded save reproduces the same shop. Pass an explicit seed to pin the
   * stock in a test.
   */
  refresh(seed?: number): readonly Item[] {
    this.#visits++;
    const root = seed ?? this.#visitSeed(this.#visits);
    const rng = mulberry32(root);
    const stock: Item[] = [];
    for (let i = 0; i < this.definition.stockSize; i++) {
      stock.push(
        generateItem({
          seed: Math.floor(rng() * 0xffffffff) >>> 0,
          itemLevel: this.definition.itemLevel,
          categories: this.definition.categories,
          qualityBias: this.definition.qualityBias,
        }),
      );
    }
    this.#stock = stock;
    return this.#stock;
  }

  /** What the player pays for `item`. Always at least 1 gold. */
  priceToBuy(item: Item): number {
    return Math.max(1, Math.round(item.value * this.definition.buyMarkup));
  }

  /**
   * What the player is paid for `item`.
   *
   * A broken item fetches a quarter of its price, because a merchant who pays
   * full value for a ruined sword makes durability meaningless in the other
   * direction.
   */
  priceToSell(item: Item): number {
    const condition = isBroken(item) ? 0.25 : 1;
    return Math.max(1, Math.round(item.value * this.definition.sellMarkdown * condition));
  }

  /** Whether this vendor will take `item` off the player. */
  accepts(item: Item): boolean {
    const allowed = this.definition.buysCategories;
    if (allowed === undefined) return true;
    return allowed.includes(item.category);
  }

  /**
   * Buy stock item `index` for `character`.
   *
   * The order — check affordability, check room, *then* move anything — is what
   * makes a failed purchase a no-op. Taking the gold first and discovering the
   * bag is full afterwards is the classic vendor bug.
   */
  buy(index: number, character: Character): Transaction {
    const item = this.#stock[index];
    if (item === undefined) return { result: 'not-in-stock', gold: 0, item: null };

    const price = this.priceToBuy(item);
    if (character.gold < price) return { result: 'cannot-afford', gold: 0, item };
    if (!character.inventory.hasRoomFor(item)) {
      return { result: 'inventory-full', gold: 0, item };
    }

    character.spendGold(price);
    character.acquire(item);
    this.#stock.splice(index, 1);
    return { result: 'ok', gold: price, item };
  }

  /** Sell an item the character owns. It leaves the bag and joins the shelf. */
  sell(item: Item, character: Character): Transaction {
    if (!this.accepts(item)) return { result: 'not-accepted', gold: 0, item };
    if (!character.inventory.contains(item)) return { result: 'not-owned', gold: 0, item };

    const price = this.priceToSell(item);
    character.inventory.remove(item);
    character.touch();
    character.addGold(price);
    // Sold goods reappear on the shelf, so a player who sells by mistake can
    // buy it back — at the spread, which is the price of the mistake.
    this.#stock.push(item);
    return { result: 'ok', gold: -price, item };
  }

  /* -- repair -------------------------------------------------------------- */

  /** Cost to restore one item to full durability. 0 when it needs nothing. */
  repairCost(item: Item): number {
    if (item.maxDurability <= 0) return 0;
    const missing = Math.max(0, item.maxDurability - item.durability);
    if (missing === 0) return 0;
    const fraction = missing / item.maxDurability;
    return Math.max(1, Math.round(item.value * fraction * 0.5));
  }

  /** Total to repair everything the character is carrying and wearing. */
  repairAllCost(character: Character): number {
    let total = 0;
    for (const item of this.#repairable(character)) total += this.repairCost(item);
    return total;
  }

  /**
   * Charsi's service: repair everything, for one price.
   *
   * All-or-nothing. Partially repairing "as much as the player could afford"
   * leaves them unable to tell what happened, and the amounts here are small
   * enough that a partial service is never the interesting outcome.
   */
  repairAll(character: Character): Transaction {
    if (!this.definition.repairs) return { result: 'no-repair-service', gold: 0, item: null };
    const items = this.#repairable(character);
    if (items.length === 0) return { result: 'nothing-to-repair', gold: 0, item: null };

    const cost = this.repairAllCost(character);
    if (character.gold < cost) return { result: 'cannot-afford', gold: cost, item: null };

    character.spendGold(cost);
    for (const item of items) item.durability = item.maxDurability;
    // Durability is mutated in place, so the derived-stat cache must be told:
    // repairing a broken weapon changes the character's damage.
    character.touch();
    return { result: 'ok', gold: cost, item: null };
  }

  toJSON(): VendorSnapshot {
    return { id: this.definition.id, visits: this.#visits, stock: this.#stock };
  }

  load(snapshot: VendorSnapshot): void {
    this.#visits = Math.max(0, Math.round(snapshot.visits));
    this.#stock = snapshot.stock.map((item) => ({ ...item }));
  }

  #repairable(character: Character): Item[] {
    const all = [...character.equipment.items(), ...character.inventory.items];
    return all.filter((item) => item.maxDurability > 0 && item.durability < item.maxDurability);
  }

  #visitSeed(visit: number): number {
    let hash = 0x811c9dc5;
    for (const ch of this.definition.id) {
      hash = Math.imul(hash ^ ch.charCodeAt(0), 0x01000193) >>> 0;
    }
    return (hash ^ Math.imul(visit, 0x9e3779b1)) >>> 0;
  }
}

/* -------------------------------------------------------------------------- */
/* Registry                                                                    */
/* -------------------------------------------------------------------------- */

/**
 * All three merchants, constructed once.
 *
 * Kept as a class rather than a bare map because a vendor's stock has to
 * survive the player walking away and coming back within the same visit — the
 * refresh is on *entering* the shop, not on every dialogue node.
 */
export class VendorRegistry {
  readonly #vendors = new Map<string, Vendor>();

  constructor(definitions: readonly VendorDefinition[] = VENDORS) {
    for (const definition of definitions) {
      this.#vendors.set(definition.id, new Vendor(definition));
    }
  }

  get(id: string): Vendor | null {
    return this.#vendors.get(id) ?? null;
  }

  /** The vendor an NPC runs, if any. */
  forNpc(npcId: string): Vendor | null {
    for (const vendor of this.#vendors.values()) {
      if (vendor.definition.npcId === npcId) return vendor;
    }
    return null;
  }

  get all(): Vendor[] {
    return Array.from(this.#vendors.values());
  }

  toJSON(): VendorSnapshot[] {
    return this.all.map((vendor) => vendor.toJSON());
  }

  load(snapshots: readonly VendorSnapshot[]): void {
    for (const snapshot of snapshots) this.get(snapshot.id)?.load(snapshot);
  }

  /** Quality tiers currently on every shelf. Diagnostics for balance passes. */
  stockQualities(): Record<ItemQuality, number> {
    const counts: Record<ItemQuality, number> = { normal: 0, magic: 0, rare: 0, unique: 0 };
    for (const vendor of this.all) {
      for (const item of vendor.stock) counts[item.quality]++;
    }
    return counts;
  }
}
