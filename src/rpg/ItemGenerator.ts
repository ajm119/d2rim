/**
 * @module rpg/ItemGenerator
 *
 * Seeded, deterministic item generation in the Diablo II idiom: a base type, a
 * quality tier, and — for magic and rare items — affixes drawn from a weighted
 * table gated by the item's level.
 *
 * ### The rules this implements, and why each one is load-bearing
 *
 * - **A magic item gets one prefix and/or one suffix.** Never two prefixes.
 *   That single constraint is what makes "Sharp Short Sword of the Fox" the
 *   ceiling of a blue item and gives rares a reason to exist.
 * - **A rare gets a two-word random name and up to six affixes**, at least two.
 *   The name comes from a pair of tables and carries no mechanical meaning,
 *   which is exactly right: the name is flavour and the affixes are the item.
 * - **Item level gates the affix pool.** An affix declares the minimum item
 *   level it can spawn at ({@link Affix.level}); an item generated at level 3
 *   cannot roll an affix declared at level 12, no matter how many times it is
 *   rerolled. This is the mechanism that makes deeper monsters worth killing,
 *   and it is asserted per-affix in the tests rather than assumed.
 * - **Affix groups are exclusive.** Two affixes from the same group never
 *   appear on one item, so a rare cannot roll `+15 defence` three times and
 *   present as a jackpot that is really one modifier written thrice.
 * - **Every roll comes from the seed.** `generateItem({ seed: 7, … })` produces
 *   a byte-identical item on every machine and every run. Loot that cannot be
 *   replayed cannot be balanced, and a drop-rate bug that only appears one time
 *   in four hundred is undebuggable without it.
 *
 * ### How an item reaches combat
 *
 * An item's affixes are aggregated into a {@link Modifiers} bag at generation
 * time. `rpg/Inventory` sums the bags of everything equipped; `rpg/Stats`
 * derives an `OffenseStats`/`DefenseStats` pair from that sum; `combat/
 * DamageModel` resolves the swing. Nothing in this file knows what a swing is,
 * and nothing in combat knows what an affix is.
 */

import { mulberry32, type DamageRange, type Rng } from '../combat/DamageModel';
import type { ModifierKey, Modifiers, ModifierTotals } from './Stats';
import { addModifiers, describeModifier, MODIFIER_KEYS, zeroModifiers } from './Stats';

/* -------------------------------------------------------------------------- */
/* Vocabulary                                                                  */
/* -------------------------------------------------------------------------- */

/** Where an item can be worn. `ring` covers both ring slots. */
export type ItemSlot =
  | 'head'
  | 'body'
  | 'weapon'
  | 'offhand'
  | 'gloves'
  | 'boots'
  | 'belt'
  | 'ring'
  | 'amulet';

/**
 * What an item *is*, for affix eligibility.
 *
 * Distinct from {@link ItemSlot} because affixes care about the kind of object
 * ("+2 to Maximum Damage" belongs on a weapon) while equipment cares about
 * where it goes, and a shield is the case that proves they differ: it occupies
 * the offhand slot but takes armour affixes.
 */
export type ItemCategory =
  | 'weapon'
  | 'shield'
  | 'armour'
  | 'helm'
  | 'gloves'
  | 'boots'
  | 'belt'
  | 'ring'
  | 'amulet';

export type ItemQuality = 'normal' | 'magic' | 'rare' | 'unique';

/** Diablo II's item colours, which the loot label and tooltips must honour. */
export const ITEM_COLOURS: Readonly<Record<ItemQuality, string>> = {
  normal: '#d6cfc0',
  magic: '#6f6fff',
  rare: '#ffff64',
  unique: '#a08554',
};

/** Gold, which is not an item but shares the label treatment. */
export const GOLD_COLOUR = '#d4af37';

/** Quality tiers in ascending order of desirability. */
export const ITEM_QUALITIES: readonly ItemQuality[] = ['normal', 'magic', 'rare', 'unique'];

/** Categories that take armour-shaped affixes rather than weapon-shaped ones. */
export const ARMOUR_CATEGORIES: readonly ItemCategory[] = [
  'shield',
  'armour',
  'helm',
  'gloves',
  'boots',
  'belt',
];

/* -------------------------------------------------------------------------- */
/* Base items                                                                  */
/* -------------------------------------------------------------------------- */

export interface ItemBase {
  readonly id: string;
  readonly name: string;
  readonly category: ItemCategory;
  readonly slot: ItemSlot;
  /** Grid footprint, in inventory cells. */
  readonly width: number;
  readonly height: number;
  /** Lowest item level at which this base can drop. */
  readonly qualityLevel: number;
  /** Character level required to equip it before affixes are considered. */
  readonly requiredLevel: number;
  readonly requiredStrength: number;
  readonly requiredDexterity: number;
  /** Physical damage, for weapons. */
  readonly damage?: DamageRange;
  /** Base defence, for armour and shields. */
  readonly defense?: number;
  /** Innate block chance in `[0, 1]`, for shields. */
  readonly block?: number;
  /** Base gold value before quality and affixes. */
  readonly value: number;
  /** Maximum durability. Zero means the item never wears (jewellery). */
  readonly durability: number;
  readonly twoHanded?: boolean;
}

function base(definition: ItemBase): ItemBase {
  return definition;
}

/**
 * The Act I base item table.
 *
 * Deliberately shallow — one or two bases per tier per slot — because the
 * interesting variance in a Diablo item is the affixes, not whether the sword
 * is a Short Sword or a Scimitar. Every base here can drop in the Den of Evil
 * or be sold by Charsi or Gheed.
 */
export const ITEM_BASES: readonly ItemBase[] = [
  // -- weapons -------------------------------------------------------------
  base({
    id: 'hand-axe',
    name: 'Hand Axe',
    category: 'weapon',
    slot: 'weapon',
    width: 1,
    height: 3,
    qualityLevel: 1,
    requiredLevel: 1,
    requiredStrength: 0,
    requiredDexterity: 0,
    damage: { min: 3, max: 6 },
    value: 40,
    durability: 28,
  }),
  base({
    id: 'short-sword',
    name: 'Short Sword',
    category: 'weapon',
    slot: 'weapon',
    width: 1,
    height: 3,
    qualityLevel: 1,
    requiredLevel: 1,
    requiredStrength: 0,
    requiredDexterity: 0,
    damage: { min: 2, max: 7 },
    value: 44,
    durability: 24,
  }),
  base({
    id: 'club',
    name: 'Club',
    category: 'weapon',
    slot: 'weapon',
    width: 1,
    height: 3,
    qualityLevel: 1,
    requiredLevel: 1,
    requiredStrength: 0,
    requiredDexterity: 0,
    damage: { min: 1, max: 6 },
    value: 20,
    durability: 24,
  }),
  base({
    id: 'double-axe',
    name: 'Double Axe',
    category: 'weapon',
    slot: 'weapon',
    width: 2,
    height: 3,
    qualityLevel: 5,
    requiredLevel: 4,
    requiredStrength: 43,
    requiredDexterity: 0,
    damage: { min: 5, max: 13 },
    value: 90,
    durability: 24,
  }),
  base({
    id: 'broad-sword',
    name: 'Broad Sword',
    category: 'weapon',
    slot: 'weapon',
    width: 2,
    height: 3,
    qualityLevel: 8,
    requiredLevel: 5,
    requiredStrength: 48,
    requiredDexterity: 0,
    damage: { min: 7, max: 14 },
    value: 110,
    durability: 32,
  }),
  base({
    id: 'war-axe',
    name: 'War Axe',
    category: 'weapon',
    slot: 'weapon',
    width: 2,
    height: 3,
    qualityLevel: 14,
    requiredLevel: 7,
    requiredStrength: 67,
    requiredDexterity: 0,
    damage: { min: 10, max: 18 },
    value: 175,
    durability: 26,
    twoHanded: true,
  }),
  // -- shields -------------------------------------------------------------
  base({
    id: 'buckler',
    name: 'Buckler',
    category: 'shield',
    slot: 'offhand',
    width: 2,
    height: 2,
    qualityLevel: 1,
    requiredLevel: 1,
    requiredStrength: 12,
    requiredDexterity: 0,
    defense: 4,
    block: 0.3,
    value: 22,
    durability: 12,
  }),
  base({
    id: 'small-shield',
    name: 'Small Shield',
    category: 'shield',
    slot: 'offhand',
    width: 2,
    height: 3,
    qualityLevel: 5,
    requiredLevel: 3,
    requiredStrength: 22,
    requiredDexterity: 0,
    defense: 9,
    block: 0.35,
    value: 50,
    durability: 16,
  }),
  // -- body armour ---------------------------------------------------------
  base({
    id: 'quilted-armor',
    name: 'Quilted Armor',
    category: 'armour',
    slot: 'body',
    width: 2,
    height: 3,
    qualityLevel: 1,
    requiredLevel: 1,
    requiredStrength: 12,
    requiredDexterity: 0,
    defense: 9,
    value: 36,
    durability: 20,
  }),
  base({
    id: 'leather-armor',
    name: 'Leather Armor',
    category: 'armour',
    slot: 'body',
    width: 2,
    height: 3,
    qualityLevel: 3,
    requiredLevel: 3,
    requiredStrength: 15,
    requiredDexterity: 0,
    defense: 15,
    value: 55,
    durability: 24,
  }),
  base({
    id: 'ring-mail',
    name: 'Ring Mail',
    category: 'armour',
    slot: 'body',
    width: 2,
    height: 3,
    qualityLevel: 11,
    requiredLevel: 6,
    requiredStrength: 36,
    requiredDexterity: 0,
    defense: 26,
    value: 110,
    durability: 26,
  }),
  // -- helms ---------------------------------------------------------------
  base({
    id: 'cap',
    name: 'Cap',
    category: 'helm',
    slot: 'head',
    width: 2,
    height: 2,
    qualityLevel: 1,
    requiredLevel: 1,
    requiredStrength: 0,
    requiredDexterity: 0,
    defense: 3,
    value: 15,
    durability: 12,
  }),
  base({
    id: 'skull-cap',
    name: 'Skull Cap',
    category: 'helm',
    slot: 'head',
    width: 2,
    height: 2,
    qualityLevel: 6,
    requiredLevel: 4,
    requiredStrength: 15,
    requiredDexterity: 0,
    defense: 8,
    value: 40,
    durability: 18,
  }),
  // -- gloves, boots, belts -------------------------------------------------
  base({
    id: 'leather-gloves',
    name: 'Leather Gloves',
    category: 'gloves',
    slot: 'gloves',
    width: 2,
    height: 2,
    qualityLevel: 1,
    requiredLevel: 1,
    requiredStrength: 0,
    requiredDexterity: 0,
    defense: 2,
    value: 15,
    durability: 12,
  }),
  base({
    id: 'heavy-gloves',
    name: 'Heavy Gloves',
    category: 'gloves',
    slot: 'gloves',
    width: 2,
    height: 2,
    qualityLevel: 7,
    requiredLevel: 4,
    requiredStrength: 0,
    requiredDexterity: 0,
    defense: 5,
    value: 30,
    durability: 14,
  }),
  base({
    id: 'boots',
    name: 'Boots',
    category: 'boots',
    slot: 'boots',
    width: 2,
    height: 2,
    qualityLevel: 1,
    requiredLevel: 1,
    requiredStrength: 0,
    requiredDexterity: 0,
    defense: 2,
    value: 15,
    durability: 12,
  }),
  base({
    id: 'heavy-boots',
    name: 'Heavy Boots',
    category: 'boots',
    slot: 'boots',
    width: 2,
    height: 2,
    qualityLevel: 7,
    requiredLevel: 4,
    requiredStrength: 18,
    requiredDexterity: 0,
    defense: 5,
    value: 32,
    durability: 14,
  }),
  base({
    id: 'sash',
    name: 'Sash',
    category: 'belt',
    slot: 'belt',
    width: 2,
    height: 1,
    qualityLevel: 1,
    requiredLevel: 1,
    requiredStrength: 0,
    requiredDexterity: 0,
    defense: 2,
    value: 18,
    durability: 12,
  }),
  base({
    id: 'leather-belt',
    name: 'Leather Belt',
    category: 'belt',
    slot: 'belt',
    width: 2,
    height: 1,
    qualityLevel: 6,
    requiredLevel: 3,
    requiredStrength: 20,
    requiredDexterity: 0,
    defense: 5,
    value: 36,
    durability: 16,
  }),
  // -- jewellery -----------------------------------------------------------
  base({
    id: 'ring',
    name: 'Ring',
    category: 'ring',
    slot: 'ring',
    width: 1,
    height: 1,
    qualityLevel: 1,
    requiredLevel: 1,
    requiredStrength: 0,
    requiredDexterity: 0,
    value: 90,
    durability: 0,
  }),
  base({
    id: 'amulet',
    name: 'Amulet',
    category: 'amulet',
    slot: 'amulet',
    width: 1,
    height: 1,
    qualityLevel: 1,
    requiredLevel: 1,
    requiredStrength: 0,
    requiredDexterity: 0,
    value: 120,
    durability: 0,
  }),
];

const BASES_BY_ID = new Map(ITEM_BASES.map((entry) => [entry.id, entry]));

export function findBase(id: string): ItemBase | null {
  return BASES_BY_ID.get(id) ?? null;
}

/** Bases that may drop at `itemLevel`, optionally restricted by category. */
export function eligibleBases(
  itemLevel: number,
  categories?: readonly ItemCategory[],
): ItemBase[] {
  const allowed = categories === undefined ? null : new Set(categories);
  return ITEM_BASES.filter(
    (entry) =>
      entry.qualityLevel <= itemLevel && (allowed === null || allowed.has(entry.category)),
  );
}

/* -------------------------------------------------------------------------- */
/* Affixes                                                                     */
/* -------------------------------------------------------------------------- */

/** One rolled statistic within an affix. */
export interface AffixMod {
  readonly stat: ModifierKey;
  readonly min: number;
  readonly max: number;
}

export type AffixKind = 'prefix' | 'suffix';

export interface Affix {
  readonly id: string;
  readonly kind: AffixKind;
  /** The word inserted into the item name. */
  readonly name: string;
  /** Minimum item level at which this affix can spawn. The level gate. */
  readonly level: number;
  /**
   * Relative spawn weight within the eligible pool. Higher is commoner; the
   * strong affixes carry small weights, which is the entire economy of a
   * Diablo drop.
   */
  readonly weight: number;
  /**
   * Exclusion group. At most one affix per group appears on an item, so a rare
   * cannot roll three flavours of "+defence" and present as a jackpot.
   */
  readonly group: string;
  /** Categories this affix may appear on. */
  readonly categories: readonly ItemCategory[];
  readonly mods: readonly AffixMod[];
}

function affix(definition: Affix): Affix {
  return definition;
}

const WEAPONS: readonly ItemCategory[] = ['weapon'];
const ANY_ARMOUR = ARMOUR_CATEGORIES;
const JEWELLERY: readonly ItemCategory[] = ['ring', 'amulet'];
const EVERYTHING: readonly ItemCategory[] = [
  'weapon',
  'shield',
  'armour',
  'helm',
  'gloves',
  'boots',
  'belt',
  'ring',
  'amulet',
];

/**
 * The affix table.
 *
 * Tiers within a group climb in both power and required level — `Sharp` (level
 * 1) through `Vicious` (level 20) all sit in the `damage%` group — so a deeper
 * item level does not merely add options, it *replaces* the weak options with
 * strong ones while the group rule keeps only one of them on any given item.
 * That is the shape of D2's table and the reason item level is worth caring
 * about.
 */
export const AFFIXES: readonly Affix[] = [
  /* -- prefixes: enhanced damage ------------------------------------------ */
  affix({
    id: 'sharp',
    kind: 'prefix',
    name: 'Sharp',
    level: 1,
    weight: 100,
    group: 'damage%',
    categories: WEAPONS,
    mods: [{ stat: 'enhancedDamage', min: 10, max: 20 }],
  }),
  affix({
    id: 'fine',
    kind: 'prefix',
    name: 'Fine',
    level: 5,
    weight: 70,
    group: 'damage%',
    categories: WEAPONS,
    mods: [{ stat: 'enhancedDamage', min: 21, max: 40 }],
  }),
  affix({
    id: 'warriors',
    kind: 'prefix',
    name: "Warrior's",
    level: 12,
    weight: 40,
    group: 'damage%',
    categories: WEAPONS,
    mods: [{ stat: 'enhancedDamage', min: 41, max: 60 }],
  }),
  affix({
    id: 'vicious',
    kind: 'prefix',
    name: 'Vicious',
    level: 20,
    weight: 18,
    group: 'damage%',
    categories: WEAPONS,
    mods: [{ stat: 'enhancedDamage', min: 61, max: 90 }],
  }),
  /* -- prefixes: flat damage ---------------------------------------------- */
  affix({
    id: 'jagged',
    kind: 'prefix',
    name: 'Jagged',
    level: 2,
    weight: 80,
    group: 'damage-flat',
    categories: WEAPONS,
    mods: [
      { stat: 'minDamage', min: 1, max: 2 },
      { stat: 'maxDamage', min: 2, max: 4 },
    ],
  }),
  affix({
    id: 'deadly',
    kind: 'prefix',
    name: 'Deadly',
    level: 9,
    weight: 45,
    group: 'damage-flat',
    categories: WEAPONS,
    mods: [
      { stat: 'minDamage', min: 3, max: 5 },
      { stat: 'maxDamage', min: 5, max: 9 },
    ],
  }),
  affix({
    id: 'ferocious',
    kind: 'prefix',
    name: 'Ferocious',
    level: 18,
    weight: 20,
    group: 'damage-flat',
    categories: WEAPONS,
    mods: [
      { stat: 'minDamage', min: 6, max: 9 },
      { stat: 'maxDamage', min: 10, max: 16 },
    ],
  }),
  /* -- prefixes: elemental damage ----------------------------------------- */
  affix({
    id: 'fiery',
    kind: 'prefix',
    name: 'Fiery',
    level: 3,
    weight: 55,
    group: 'element',
    categories: WEAPONS,
    mods: [
      { stat: 'fireMin', min: 1, max: 3 },
      { stat: 'fireMax', min: 4, max: 8 },
    ],
  }),
  affix({
    id: 'smoldering',
    kind: 'prefix',
    name: 'Smoldering',
    level: 11,
    weight: 30,
    group: 'element',
    categories: WEAPONS,
    mods: [
      { stat: 'fireMin', min: 4, max: 7 },
      { stat: 'fireMax', min: 9, max: 16 },
    ],
  }),
  affix({
    id: 'shivering',
    kind: 'prefix',
    name: 'Shivering',
    level: 4,
    weight: 50,
    group: 'element',
    categories: WEAPONS,
    mods: [
      { stat: 'coldMin', min: 1, max: 2 },
      { stat: 'coldMax', min: 3, max: 6 },
    ],
  }),
  affix({
    id: 'static',
    kind: 'prefix',
    name: 'Static',
    level: 6,
    weight: 45,
    group: 'element',
    categories: WEAPONS,
    mods: [
      { stat: 'lightningMin', min: 1, max: 1 },
      { stat: 'lightningMax', min: 6, max: 14 },
    ],
  }),
  /* -- prefixes: defence --------------------------------------------------- */
  affix({
    id: 'sturdy',
    kind: 'prefix',
    name: 'Sturdy',
    level: 1,
    weight: 100,
    group: 'defense%',
    categories: ANY_ARMOUR,
    mods: [{ stat: 'enhancedDefense', min: 10, max: 20 }],
  }),
  affix({
    id: 'strong',
    kind: 'prefix',
    name: 'Strong',
    level: 5,
    weight: 65,
    group: 'defense%',
    categories: ANY_ARMOUR,
    mods: [{ stat: 'enhancedDefense', min: 21, max: 40 }],
  }),
  affix({
    id: 'glorious',
    kind: 'prefix',
    name: 'Glorious',
    level: 15,
    weight: 25,
    group: 'defense%',
    categories: ANY_ARMOUR,
    mods: [{ stat: 'enhancedDefense', min: 41, max: 65 }],
  }),
  affix({
    id: 'stalwart',
    kind: 'prefix',
    name: 'Stalwart',
    level: 3,
    weight: 60,
    group: 'defense-flat',
    categories: ANY_ARMOUR,
    mods: [{ stat: 'defense', min: 3, max: 8 }],
  }),
  affix({
    id: 'guardians',
    kind: 'prefix',
    name: "Guardian's",
    level: 12,
    weight: 30,
    group: 'defense-flat',
    categories: ANY_ARMOUR,
    mods: [{ stat: 'defense', min: 9, max: 20 }],
  }),
  /* -- prefixes: attributes and life -------------------------------------- */
  affix({
    id: 'bronze',
    kind: 'prefix',
    name: 'Bronze',
    level: 1,
    weight: 90,
    group: 'attack-rating',
    categories: EVERYTHING,
    mods: [{ stat: 'attackRating', min: 10, max: 20 }],
  }),
  affix({
    id: 'iron',
    kind: 'prefix',
    name: 'Iron',
    level: 4,
    weight: 60,
    group: 'attack-rating',
    categories: EVERYTHING,
    mods: [{ stat: 'attackRating', min: 21, max: 40 }],
  }),
  affix({
    id: 'steel',
    kind: 'prefix',
    name: 'Steel',
    level: 10,
    weight: 35,
    group: 'attack-rating',
    categories: EVERYTHING,
    mods: [{ stat: 'attackRating', min: 41, max: 70 }],
  }),
  affix({
    id: 'hale',
    kind: 'prefix',
    name: 'Hale',
    level: 1,
    weight: 80,
    group: 'life',
    categories: EVERYTHING,
    mods: [{ stat: 'life', min: 5, max: 10 }],
  }),
  affix({
    id: 'robust',
    kind: 'prefix',
    name: 'Robust',
    level: 8,
    weight: 40,
    group: 'life',
    categories: EVERYTHING,
    mods: [{ stat: 'life', min: 11, max: 20 }],
  }),
  affix({
    id: 'lizards',
    kind: 'prefix',
    name: "Lizard's",
    level: 2,
    weight: 60,
    group: 'mana',
    categories: EVERYTHING,
    mods: [{ stat: 'mana', min: 5, max: 10 }],
  }),
  affix({
    id: 'snakes',
    kind: 'prefix',
    name: "Snake's",
    level: 9,
    weight: 30,
    group: 'mana',
    categories: EVERYTHING,
    mods: [{ stat: 'mana', min: 11, max: 20 }],
  }),
  /* -- suffixes: attributes ----------------------------------------------- */
  affix({
    id: 'of-the-ox',
    kind: 'suffix',
    name: 'of the Ox',
    level: 1,
    weight: 90,
    group: 'strength',
    categories: EVERYTHING,
    mods: [{ stat: 'strength', min: 1, max: 3 }],
  }),
  affix({
    id: 'of-strength',
    kind: 'suffix',
    name: 'of Strength',
    level: 7,
    weight: 45,
    group: 'strength',
    categories: EVERYTHING,
    mods: [{ stat: 'strength', min: 4, max: 7 }],
  }),
  affix({
    id: 'of-the-fox',
    kind: 'suffix',
    name: 'of the Fox',
    level: 1,
    weight: 90,
    group: 'dexterity',
    categories: EVERYTHING,
    mods: [{ stat: 'dexterity', min: 1, max: 3 }],
  }),
  affix({
    id: 'of-dexterity',
    kind: 'suffix',
    name: 'of Dexterity',
    level: 7,
    weight: 45,
    group: 'dexterity',
    categories: EVERYTHING,
    mods: [{ stat: 'dexterity', min: 4, max: 7 }],
  }),
  affix({
    id: 'of-the-jackal',
    kind: 'suffix',
    name: 'of the Jackal',
    level: 1,
    weight: 90,
    group: 'vitality',
    categories: EVERYTHING,
    mods: [{ stat: 'vitality', min: 1, max: 3 }],
  }),
  affix({
    id: 'of-vita',
    kind: 'suffix',
    name: 'of Vita',
    level: 10,
    weight: 35,
    group: 'vitality',
    categories: EVERYTHING,
    mods: [{ stat: 'vitality', min: 4, max: 8 }],
  }),
  affix({
    id: 'of-the-mind',
    kind: 'suffix',
    name: 'of the Mind',
    level: 3,
    weight: 70,
    group: 'energy',
    categories: EVERYTHING,
    mods: [{ stat: 'energy', min: 1, max: 3 }],
  }),
  /* -- suffixes: resistances ---------------------------------------------- */
  affix({
    id: 'of-flame',
    kind: 'suffix',
    name: 'of Flame',
    level: 2,
    weight: 70,
    group: 'resist-fire',
    categories: EVERYTHING,
    mods: [{ stat: 'resistFire', min: 5, max: 12 }],
  }),
  affix({
    id: 'of-fire',
    kind: 'suffix',
    name: 'of Fire',
    level: 9,
    weight: 35,
    group: 'resist-fire',
    categories: EVERYTHING,
    mods: [{ stat: 'resistFire', min: 13, max: 25 }],
  }),
  affix({
    id: 'of-frost',
    kind: 'suffix',
    name: 'of Frost',
    level: 2,
    weight: 70,
    group: 'resist-cold',
    categories: EVERYTHING,
    mods: [{ stat: 'resistCold', min: 5, max: 12 }],
  }),
  affix({
    id: 'of-shock',
    kind: 'suffix',
    name: 'of Shock',
    level: 2,
    weight: 70,
    group: 'resist-lightning',
    categories: EVERYTHING,
    mods: [{ stat: 'resistLightning', min: 5, max: 12 }],
  }),
  affix({
    id: 'of-blight',
    kind: 'suffix',
    name: 'of Blight',
    level: 2,
    weight: 70,
    group: 'resist-poison',
    categories: EVERYTHING,
    mods: [{ stat: 'resistPoison', min: 5, max: 12 }],
  }),
  affix({
    id: 'of-warding',
    kind: 'suffix',
    name: 'of Warding',
    level: 14,
    weight: 14,
    group: 'resist-all',
    categories: JEWELLERY,
    mods: [{ stat: 'resistAll', min: 4, max: 10 }],
  }),
  /* -- suffixes: combat utility -------------------------------------------- */
  affix({
    id: 'of-blocking',
    kind: 'suffix',
    name: 'of Blocking',
    level: 5,
    weight: 40,
    group: 'block',
    categories: ['shield'],
    mods: [{ stat: 'blockChance', min: 4, max: 10 }],
  }),
  affix({
    id: 'of-slaying',
    kind: 'suffix',
    name: 'of Slaying',
    level: 8,
    weight: 30,
    group: 'critical',
    categories: WEAPONS,
    mods: [{ stat: 'criticalChance', min: 3, max: 8 }],
  }),
  affix({
    id: 'of-butchery',
    kind: 'suffix',
    name: 'of Butchery',
    level: 16,
    weight: 12,
    group: 'critical',
    categories: WEAPONS,
    mods: [{ stat: 'criticalChance', min: 9, max: 15 }],
  }),
  affix({
    id: 'of-the-sentinel',
    kind: 'suffix',
    name: 'of the Sentinel',
    level: 6,
    weight: 35,
    group: 'reduction',
    categories: ANY_ARMOUR,
    mods: [{ stat: 'damageReduction', min: 1, max: 3 }],
  }),
  affix({
    id: 'of-the-bulwark',
    kind: 'suffix',
    name: 'of the Bulwark',
    level: 17,
    weight: 12,
    group: 'reduction',
    categories: ANY_ARMOUR,
    mods: [{ stat: 'damageReducedPercent', min: 2, max: 6 }],
  }),
  affix({
    id: 'of-stamina',
    kind: 'suffix',
    name: 'of Stamina',
    level: 1,
    weight: 60,
    group: 'stamina',
    categories: ['boots', 'belt', 'armour'],
    mods: [{ stat: 'stamina', min: 10, max: 25 }],
  }),
];

const AFFIXES_BY_ID = new Map(AFFIXES.map((entry) => [entry.id, entry]));

export function findAffix(id: string): Affix | null {
  return AFFIXES_BY_ID.get(id) ?? null;
}

/**
 * Affixes that may spawn on `category` at `itemLevel`.
 *
 * This is the level gate, and it is the only place it exists: every generation
 * path draws from this function, so an affix that is not eligible here cannot
 * appear on an item by any route.
 */
export function eligibleAffixes(
  category: ItemCategory,
  itemLevel: number,
  kind?: AffixKind,
): Affix[] {
  return AFFIXES.filter(
    (entry) =>
      entry.level <= itemLevel &&
      entry.categories.includes(category) &&
      (kind === undefined || entry.kind === kind),
  );
}

/* -------------------------------------------------------------------------- */
/* Uniques                                                                     */
/* -------------------------------------------------------------------------- */

export interface UniqueDefinition {
  readonly id: string;
  readonly name: string;
  readonly baseId: string;
  readonly level: number;
  readonly mods: readonly AffixMod[];
}

/** Act I uniques, with D2's own names and roughly D2's own numbers. */
export const UNIQUES: readonly UniqueDefinition[] = [
  {
    id: 'the-gnasher',
    name: 'The Gnasher',
    baseId: 'hand-axe',
    level: 5,
    mods: [
      { stat: 'enhancedDamage', min: 70, max: 90 },
      { stat: 'attackRating', min: 20, max: 30 },
    ],
  },
  {
    id: 'rixots-keen',
    name: "Rixot's Keen",
    baseId: 'short-sword',
    level: 4,
    mods: [
      { stat: 'enhancedDamage', min: 40, max: 60 },
      { stat: 'minDamage', min: 3, max: 5 },
      { stat: 'attackRating', min: 30, max: 45 },
    ],
  },
  {
    id: 'pelta-lunata',
    name: 'Pelta Lunata',
    baseId: 'buckler',
    level: 3,
    mods: [
      { stat: 'enhancedDefense', min: 50, max: 80 },
      { stat: 'blockChance', min: 10, max: 15 },
      { stat: 'energy', min: 3, max: 5 },
    ],
  },
  {
    id: 'nagelring',
    name: 'Nagelring',
    baseId: 'ring',
    level: 7,
    mods: [
      { stat: 'attackRating', min: 50, max: 75 },
      { stat: 'damageReduction', min: 2, max: 3 },
    ],
  },
  {
    id: 'the-eye-of-etlich',
    name: 'The Eye of Etlich',
    baseId: 'amulet',
    level: 8,
    mods: [
      { stat: 'life', min: 10, max: 20 },
      { stat: 'coldMin', min: 1, max: 2 },
      { stat: 'coldMax', min: 3, max: 6 },
      { stat: 'resistAll', min: 5, max: 10 },
    ],
  },
  {
    id: 'hotspur',
    name: 'Hotspur',
    baseId: 'boots',
    level: 5,
    mods: [
      { stat: 'enhancedDefense', min: 30, max: 50 },
      { stat: 'fireMin', min: 3, max: 3 },
      { stat: 'fireMax', min: 6, max: 6 },
      { stat: 'resistFire', min: 20, max: 30 },
      { stat: 'life', min: 10, max: 15 },
    ],
  },
];

/** Uniques whose level gate `itemLevel` clears, optionally for one base. */
export function eligibleUniques(itemLevel: number, baseId?: string): UniqueDefinition[] {
  return UNIQUES.filter(
    (entry) => entry.level <= itemLevel && (baseId === undefined || entry.baseId === baseId),
  );
}

/* -------------------------------------------------------------------------- */
/* Rare naming                                                                 */
/* -------------------------------------------------------------------------- */

/** First word of a rare's name. Flavour only — no mechanical meaning. */
export const RARE_NAME_FIRST: readonly string[] = [
  'Blood',
  'Bone',
  'Corpse',
  'Doom',
  'Dread',
  'Ghoul',
  'Grim',
  'Hell',
  'Rune',
  'Skull',
  'Storm',
  'Viper',
  'Wraith',
  'Cruel',
  'Empyrean',
  'Havoc',
];

/** Second word of a rare's name. */
export const RARE_NAME_SECOND: readonly string[] = [
  'Bite',
  'Song',
  'Gaze',
  'Shield',
  'Bane',
  'Wound',
  'Grasp',
  'Spirit',
  'Fang',
  'Cry',
  'Web',
  'Whorl',
  'Brand',
  'Sting',
  'Edge',
  'Ward',
];

/* -------------------------------------------------------------------------- */
/* Items                                                                       */
/* -------------------------------------------------------------------------- */

/** An affix as it actually landed on one item. */
export interface RolledAffix {
  readonly affixId: string;
  readonly kind: AffixKind;
  readonly name: string;
  /** The rolled value per modifier, in the affix's declared order. */
  readonly values: readonly { readonly stat: ModifierKey; readonly value: number }[];
}

/** A generated item instance. Immutable except for durability. */
export interface Item {
  /** Unique per instance. Derived from the seed, so it is reproducible. */
  readonly uid: string;
  readonly baseId: string;
  /** Full display name, affixes included. */
  readonly name: string;
  readonly quality: ItemQuality;
  readonly category: ItemCategory;
  readonly slot: ItemSlot;
  readonly width: number;
  readonly height: number;
  readonly itemLevel: number;
  readonly requiredLevel: number;
  readonly requiredStrength: number;
  readonly requiredDexterity: number;
  readonly affixes: readonly RolledAffix[];
  /** Aggregated modifiers from every affix. What equipment sums. */
  readonly mods: Modifiers;
  /** Weapon damage after nothing — the base's own range. */
  readonly damage?: DamageRange;
  /** Armour defence before `enhancedDefense`. */
  readonly defense?: number;
  /** Shield block chance in `[0, 1]`. */
  readonly block?: number;
  readonly value: number;
  readonly maxDurability: number;
  /** Mutable: swinging wears a weapon down, Charsi fixes it. */
  durability: number;
  /** For uniques, which definition produced it. */
  readonly uniqueId?: string;
  readonly seed: number;
  readonly twoHanded: boolean;
}

/** Colour for this item's name, honouring the D2 convention. */
export function itemColour(item: Item): string {
  return ITEM_COLOURS[item.quality];
}

/** Every modifier line an item tooltip should print, in table order. */
export function describeItem(item: Item): string[] {
  const lines: string[] = [];
  if (item.damage !== undefined) {
    lines.push(`Damage: ${item.damage.min} to ${item.damage.max}`);
  }
  if (item.defense !== undefined && item.defense > 0) {
    lines.push(`Defense: ${item.defense}`);
  }
  if (item.block !== undefined && item.block > 0) {
    lines.push(`Chance to Block: ${Math.round(item.block * 100)}%`);
  }
  if (item.maxDurability > 0) {
    lines.push(`Durability: ${item.durability} of ${item.maxDurability}`);
  }
  if (item.requiredLevel > 1) lines.push(`Required Level: ${item.requiredLevel}`);
  if (item.requiredStrength > 0) lines.push(`Required Strength: ${item.requiredStrength}`);
  for (const key of MODIFIER_KEYS) {
    const value = item.mods[key];
    if (value === undefined || value === 0) continue;
    lines.push(describeModifier(key, value));
  }
  return lines;
}

/* -------------------------------------------------------------------------- */
/* Generation                                                                  */
/* -------------------------------------------------------------------------- */

/** Inclusive integer roll. */
function rollInt(min: number, max: number, rng: Rng): number {
  const lo = Math.round(min);
  const hi = Math.max(lo, Math.round(max));
  if (hi === lo) return lo;
  return lo + Math.floor(rng() * (hi - lo + 1));
}

/** Weighted pick. Returns null for an empty or zero-weight pool. */
function pickWeighted<T extends { readonly weight: number }>(pool: readonly T[], rng: Rng): T | null {
  let total = 0;
  for (const entry of pool) total += Math.max(0, entry.weight);
  if (total <= 0) return null;
  let roll = rng() * total;
  for (const entry of pool) {
    roll -= Math.max(0, entry.weight);
    if (roll < 0) return entry;
  }
  return pool[pool.length - 1] ?? null;
}

function pickUniform<T>(pool: readonly T[], rng: Rng): T | null {
  if (pool.length === 0) return null;
  return pool[Math.min(pool.length - 1, Math.floor(rng() * pool.length))] ?? null;
}

/**
 * Roll a quality tier.
 *
 * The odds are D2-shaped: normal is the overwhelming default, magic is common
 * enough to be the texture of play, rare is a small thrill and unique is rare
 * enough that a player remembers the drop. `bias` scales the non-normal
 * chances, so a boss's drop table can be generous without a second formula.
 */
export function rollQuality(itemLevel: number, rng: Rng, bias = 1): ItemQuality {
  const scale = Math.max(0, bias);
  // Deeper items are slightly likelier to be interesting, which is what makes
  // clearing to the bottom of the Den worth doing.
  const depth = Math.min(1, Math.max(0, itemLevel) / 40);
  const unique = 0.006 * scale * (1 + depth);
  const rare = 0.05 * scale * (1 + depth);
  const magic = 0.26 * scale * (1 + depth * 0.5);

  const roll = rng();
  if (roll < unique) return 'unique';
  if (roll < unique + rare) return 'rare';
  if (roll < unique + rare + magic) return 'magic';
  return 'normal';
}

/** Roll one affix's modifiers into a {@link RolledAffix}. */
function rollAffix(entry: Affix, rng: Rng): RolledAffix {
  return {
    affixId: entry.id,
    kind: entry.kind,
    name: entry.name,
    values: entry.mods.map((mod) => ({ stat: mod.stat, value: rollInt(mod.min, mod.max, rng) })),
  };
}

/**
 * Draw `count` affixes without repeating a group.
 *
 * Groups are removed from the pool as they are consumed rather than rejected
 * after the fact, so a request for six affixes on an item with four eligible
 * groups returns four rather than looping.
 */
function drawAffixes(
  pool: readonly Affix[],
  count: number,
  rng: Rng,
  usedGroups: Set<string>,
): RolledAffix[] {
  const out: RolledAffix[] = [];
  let available = pool.filter((entry) => !usedGroups.has(entry.group));
  for (let i = 0; i < count && available.length > 0; i++) {
    const chosen = pickWeighted(available, rng);
    if (chosen === null) break;
    usedGroups.add(chosen.group);
    out.push(rollAffix(chosen, rng));
    available = available.filter((entry) => entry.group !== chosen.group);
  }
  return out;
}

function aggregate(affixes: readonly RolledAffix[]): Modifiers {
  const totals: ModifierTotals = zeroModifiers();
  for (const rolled of affixes) {
    for (const { stat, value } of rolled.values) totals[stat] += value;
  }
  const out: Partial<Record<ModifierKey, number>> = {};
  for (const key of MODIFIER_KEYS) {
    if (totals[key] !== 0) out[key] = totals[key];
  }
  return out;
}

/**
 * The level a character must reach to equip an item.
 *
 * D2 derives it from the strongest affix rather than storing it, so a lucky
 * low-level drop with a strong affix is a *goal* rather than an immediate
 * upgrade. Three quarters of the affix level, floored to the base requirement,
 * reproduces that feel at this act's compressed level range.
 */
export function requiredLevelFor(
  baseItem: ItemBase,
  affixes: readonly RolledAffix[],
  quality: ItemQuality,
): number {
  let highest = 0;
  for (const rolled of affixes) {
    const entry = AFFIXES_BY_ID.get(rolled.affixId);
    if (entry !== undefined && entry.level > highest) highest = entry.level;
  }
  const fromAffix = highest > 0 ? Math.ceil(highest * 0.75) : 0;
  const qualityBump = quality === 'rare' ? 1 : quality === 'unique' ? 2 : 0;
  return Math.max(1, baseItem.requiredLevel, fromAffix + qualityBump);
}

/** Gold value, scaled by quality and by how much the affixes are worth. */
export function computeValue(
  baseItem: ItemBase,
  affixes: readonly RolledAffix[],
  quality: ItemQuality,
): number {
  const qualityScale =
    quality === 'unique' ? 6 : quality === 'rare' ? 3.2 : quality === 'magic' ? 1.8 : 1;
  let affixValue = 0;
  for (const rolled of affixes) {
    for (const { value } of rolled.values) affixValue += Math.abs(value) * 4;
  }
  return Math.max(1, Math.round((baseItem.value + affixValue) * qualityScale));
}

/** Compose the display name from the base and its affixes. */
export function composeName(
  baseItem: ItemBase,
  quality: ItemQuality,
  affixes: readonly RolledAffix[],
  rareName?: string,
  uniqueName?: string,
): string {
  if (quality === 'unique' && uniqueName !== undefined) return `${uniqueName} ${baseItem.name}`;
  if (quality === 'rare' && rareName !== undefined) return `${rareName} ${baseItem.name}`;
  if (quality === 'magic') {
    const prefix = affixes.find((entry) => entry.kind === 'prefix');
    const suffix = affixes.find((entry) => entry.kind === 'suffix');
    const head = prefix === undefined ? baseItem.name : `${prefix.name} ${baseItem.name}`;
    return suffix === undefined ? head : `${head} ${suffix.name}`;
  }
  return baseItem.name;
}

export interface GenerateOptions {
  /** The only source of randomness. Same seed, same item, forever. */
  readonly seed: number;
  /** Gates which bases and which affixes may appear. */
  readonly itemLevel: number;
  /** Force a base rather than rolling one. */
  readonly baseId?: string;
  /** Force a quality rather than rolling one. */
  readonly quality?: ItemQuality;
  /** Restrict the base roll to these categories. */
  readonly categories?: readonly ItemCategory[];
  /** Multiplier on the magic/rare/unique odds. */
  readonly qualityBias?: number;
  /** Force a specific unique. Implies `quality: 'unique'`. */
  readonly uniqueId?: string;
}

/** Affix count bounds per quality. Asserted directly by the tests. */
export const AFFIX_BOUNDS: Readonly<Record<ItemQuality, { min: number; max: number }>> = {
  normal: { min: 0, max: 0 },
  magic: { min: 1, max: 2 },
  rare: { min: 2, max: 6 },
  unique: { min: 0, max: 0 },
};

/**
 * Generate one item.
 *
 * The RNG draw order is fixed — base, quality, then affixes — so that changing
 * *how many* affixes a rare gets does not change *which base* the same seed
 * produces. That property is what lets the affix table be retuned without
 * invalidating every seeded test in the suite.
 */
export function generateItem(options: GenerateOptions): Item {
  const itemLevel = Math.max(1, Math.round(options.itemLevel));
  const rng = mulberry32(options.seed);

  // 1. base
  let baseItem: ItemBase | null =
    options.baseId !== undefined ? (BASES_BY_ID.get(options.baseId) ?? null) : null;
  if (baseItem === null) {
    const pool = eligibleBases(itemLevel, options.categories);
    baseItem = pickUniform(pool.length > 0 ? pool : ITEM_BASES, rng);
  }
  if (baseItem === null) throw new Error('[ItemGenerator] no base item is available');

  // 2. quality
  let quality: ItemQuality =
    options.uniqueId !== undefined
      ? 'unique'
      : (options.quality ?? rollQuality(itemLevel, rng, options.qualityBias ?? 1));

  // 3. affixes
  const usedGroups = new Set<string>();
  let affixes: RolledAffix[] = [];
  let rareName: string | undefined;
  let uniqueName: string | undefined;
  let uniqueId: string | undefined;

  if (quality === 'unique') {
    const candidates =
      options.uniqueId !== undefined
        ? UNIQUES.filter((entry) => entry.id === options.uniqueId)
        : eligibleUniques(itemLevel, baseItem.id);
    const chosen = pickUniform(candidates, rng);
    if (chosen === null) {
      // No unique exists for this base at this level. Downgrade to rare rather
      // than emitting a "unique" with no unique properties — an item whose
      // colour lies is worse than a slightly commoner drop.
      quality = 'rare';
    } else {
      const source = BASES_BY_ID.get(chosen.baseId);
      if (source !== undefined) baseItem = source;
      uniqueName = chosen.name;
      uniqueId = chosen.id;
      affixes = [
        {
          affixId: chosen.id,
          kind: 'prefix',
          name: chosen.name,
          values: chosen.mods.map((mod) => ({
            stat: mod.stat,
            value: rollInt(mod.min, mod.max, rng),
          })),
        },
      ];
    }
  }

  if (quality === 'magic') {
    const prefixes = eligibleAffixes(baseItem.category, itemLevel, 'prefix');
    const suffixes = eligibleAffixes(baseItem.category, itemLevel, 'suffix');
    // D2: a magic item has a prefix, a suffix, or both — never two of a kind.
    // The three cases are rolled explicitly so "both" is genuinely commoner
    // than either single, which is what makes a two-affix blue feel ordinary
    // rather than special.
    const shape = rng();
    const wantPrefix = shape < 0.35 || shape >= 0.7;
    const wantSuffix = shape >= 0.35;
    if (wantPrefix) affixes.push(...drawAffixes(prefixes, 1, rng, usedGroups));
    if (wantSuffix) affixes.push(...drawAffixes(suffixes, 1, rng, usedGroups));
    if (affixes.length === 0) {
      // Both draws came back empty (a category with no eligible affixes at this
      // level). A blue item with no properties is a bug the player can see, so
      // it becomes what it actually is.
      quality = 'normal';
    }
  } else if (quality === 'rare') {
    const pool = eligibleAffixes(baseItem.category, itemLevel);
    const bounds = AFFIX_BOUNDS.rare;
    // Deeper items lean toward the top of the range without ever leaving it.
    const depth = Math.min(1, itemLevel / 30);
    const target = rollInt(bounds.min, bounds.min + Math.round((bounds.max - bounds.min) * (0.5 + depth * 0.5)), rng);
    affixes = drawAffixes(pool, Math.min(bounds.max, target), rng, usedGroups);
    const first = pickUniform(RARE_NAME_FIRST, rng) ?? 'Grim';
    const second = pickUniform(RARE_NAME_SECOND, rng) ?? 'Bite';
    rareName = `${first} ${second}`;
    if (affixes.length < bounds.min) {
      quality = affixes.length === 0 ? 'normal' : 'magic';
      rareName = undefined;
    }
  }

  const mods = aggregate(affixes);
  const requiredLevel =
    quality === 'unique' && uniqueId !== undefined
      ? Math.max(
          baseItem.requiredLevel,
          Math.ceil((UNIQUES.find((entry) => entry.id === uniqueId)?.level ?? 1) * 0.75),
        )
      : requiredLevelFor(baseItem, affixes, quality);

  const item: Item = {
    uid: `${baseItem.id}-${options.seed >>> 0}-${itemLevel}`,
    baseId: baseItem.id,
    name: composeName(baseItem, quality, affixes, rareName, uniqueName),
    quality,
    category: baseItem.category,
    slot: baseItem.slot,
    width: baseItem.width,
    height: baseItem.height,
    itemLevel,
    requiredLevel,
    requiredStrength: baseItem.requiredStrength,
    requiredDexterity: baseItem.requiredDexterity,
    affixes,
    mods,
    value: computeValue(baseItem, affixes, quality),
    maxDurability: baseItem.durability,
    durability: baseItem.durability,
    seed: options.seed >>> 0,
    twoHanded: baseItem.twoHanded === true,
    ...(baseItem.damage !== undefined ? { damage: baseItem.damage } : {}),
    ...(baseItem.defense !== undefined ? { defense: baseItem.defense } : {}),
    ...(baseItem.block !== undefined ? { block: baseItem.block } : {}),
    ...(uniqueId !== undefined ? { uniqueId } : {}),
  };
  return item;
}

/** Deep copy of an item. Used when moving between containers and on save. */
export function cloneItem(item: Item): Item {
  return {
    ...item,
    affixes: item.affixes.map((entry) => ({ ...entry, values: entry.values.map((v) => ({ ...v })) })),
    mods: { ...item.mods },
  };
}

/**
 * Sum the modifier bags of a set of items.
 *
 * A broken item — durability exhausted — contributes nothing, which is the
 * mechanism that makes Charsi's repair service matter rather than being a gold
 * sink with no consequence.
 */
export function sumItemModifiers(items: readonly Item[]): ModifierTotals {
  const totals = zeroModifiers();
  for (const item of items) {
    if (item.maxDurability > 0 && item.durability <= 0) continue;
    addModifiers(totals, item.mods);
  }
  return totals;
}

/** Whether an item still functions. Broken items keep their slot but do nothing. */
export function isBroken(item: Item): boolean {
  return item.maxDurability > 0 && item.durability <= 0;
}
