/**
 * @module rpg/Character
 *
 * The player's sheet as one object: statistics, grid inventory, equipment,
 * skills and gold, plus the single derivation that turns all of it into the
 * numbers combat reads.
 *
 * ### Why this exists rather than five loose objects
 *
 * Because the derivation has to happen in exactly one place. Life comes from
 * vitality *and* from a ring; attack rating comes from dexterity *and* from a
 * prefix *and* from Axe Mastery. If each subsystem applied its own contribution
 * to a stored total, equipping and unequipping the same ring twice would leave a
 * residue — the single most common bug in an RPG layer, and one that no test of
 * any individual subsystem can catch.
 *
 * So nothing is stored. {@link Character.derived} recomputes from
 * `stats + equipment.modifiers() + skills.modifiers()` and caches the result
 * behind a version counter that every mutation bumps. Equip a ring, unequip it,
 * and the sheet is bit-for-bit what it was.
 *
 * `version` is also the UI's change signal: a HUD or inventory screen compares
 * the number it last rendered against the current one and redraws only when
 * they differ, which is what keeps a DOM overlay off the frame budget.
 */

import type { DefenseStats, OffenseStats } from '../combat/DamageModel';
import { Equipment, InventoryGrid, meetsRequirements, type EquipmentSlot, type EquipResult, type InventorySnapshot, type EquipmentSnapshot } from './Inventory';
import { isBroken, type Item } from './ItemGenerator';
import { SkillTree, type SkillTreeSnapshot } from './SkillTree';
import {
  BARBARIAN,
  CharacterStats,
  defenseFrom,
  offenseFrom,
  sumModifiers,
  type CharacterStatsSnapshot,
  type ClassDefinition,
  type DerivedStats,
  type PrimaryStat,
} from './Stats';

export interface CharacterSnapshot {
  readonly name: string;
  readonly stats: CharacterStatsSnapshot;
  readonly inventory: InventorySnapshot;
  readonly equipment: EquipmentSnapshot;
  readonly skills: SkillTreeSnapshot;
  readonly gold: number;
}

/** Where an item ended up after {@link Character.acquire}. */
export type AcquireResult = 'inventory' | 'full';

export interface CharacterOptions {
  readonly name?: string;
  readonly definition?: ClassDefinition;
  readonly gold?: number;
}

export class Character {
  readonly stats: CharacterStats;
  readonly inventory: InventoryGrid;
  readonly equipment = new Equipment();
  readonly skills: SkillTree;

  #name: string;
  #gold: number;
  #version = 0;
  #cache: { version: number; derived: DerivedStats } | null = null;

  constructor(options: CharacterOptions = {}) {
    this.#name = options.name ?? 'Barbarian';
    this.stats = new CharacterStats(options.definition ?? BARBARIAN);
    this.inventory = new InventoryGrid();
    this.skills = new SkillTree(this.stats);
    this.#gold = Math.max(0, Math.round(options.gold ?? 0));
  }

  /**
   * The character's name.
   *
   * Mutable, and restored by {@link Character.load}. It was `readonly` until a
   * save round-trip test noticed that loading a saved game silently renamed the
   * hero back to the default — the sort of omission that only a round trip
   * built from a *played* character can see, because a fresh character's name
   * already is the default.
   */
  get name(): string {
    return this.#name;
  }

  /**
   * Monotonic change counter.
   *
   * Bumped by every mutation that could move a number on the character sheet.
   * Read it, render, and compare next frame.
   */
  get version(): number {
    return this.#version;
  }

  get gold(): number {
    return this.#gold;
  }

  /** The full sheet, recomputed only when something has actually changed. */
  get derived(): DerivedStats {
    const cache = this.#cache;
    if (cache !== null && cache.version === this.#version) return cache.derived;
    const weaponDamage = this.equipment.weaponDamage();
    const derived = this.stats.derive({
      modifiers: sumModifiers(this.equipment.modifiers(), this.skills.modifiers()),
      armourDefense: this.equipment.armourDefense(),
      ...(weaponDamage !== undefined ? { weaponDamage } : {}),
    });
    this.#cache = { version: this.#version, derived };
    return derived;
  }

  /** The character's swing, ready for `resolveAttack`. */
  offense(baseCritical = 0): OffenseStats {
    return offenseFrom(this.derived, baseCritical);
  }

  /** The character's guard, ready for `resolveAttack`. */
  defense(options: { readonly poise?: number } = {}): DefenseStats {
    return defenseFrom(this.derived, {
      baseBlock: this.equipment.blockChance(),
      ...(options.poise !== undefined ? { poise: options.poise } : {}),
    });
  }

  /** Force a recompute. Call after mutating an item in place (durability). */
  touch(): void {
    this.#version++;
  }

  /* -- gold ---------------------------------------------------------------- */

  addGold(amount: number): number {
    const value = Math.max(0, Math.round(amount));
    if (value === 0) return 0;
    this.#gold += value;
    this.#version++;
    return value;
  }

  /** All-or-nothing withdrawal. @returns whether the character could afford it. */
  spendGold(amount: number): boolean {
    const value = Math.max(0, Math.round(amount));
    if (value > this.#gold) return false;
    this.#gold -= value;
    this.#version++;
    return true;
  }

  /* -- items --------------------------------------------------------------- */

  /** Put an item in the bag. @returns whether there was room. */
  acquire(item: Item): AcquireResult {
    const placement = this.inventory.add(item);
    if (placement === null) return 'full';
    this.#version++;
    return 'inventory';
  }

  /** Whether the character satisfies an item's level and attribute requirements. */
  canEquip(item: Item): boolean {
    const derived = this.derived;
    return meetsRequirements(item, {
      level: derived.level,
      strength: derived.attributes.strength,
      dexterity: derived.attributes.dexterity,
    });
  }

  /**
   * Equip an item, taking it out of the bag if it is there and putting whatever
   * it displaced back in.
   *
   * A displaced item with nowhere to go is **not** deleted: the equip is rolled
   * back and refused. Losing an item to a full inventory is the one failure a
   * player never forgives.
   */
  equip(item: Item, slot?: EquipmentSlot): EquipResult {
    if (!this.canEquip(item)) {
      return { equipped: false, displaced: [], slot: slot ?? null, reason: 'wrong-slot' };
    }

    const wasInBag = this.inventory.contains(item);
    const previous = wasInBag ? this.inventory.placementOf(item) : null;
    if (wasInBag) this.inventory.remove(item);

    const result = slot === undefined ? this.equipment.equip(item) : this.equipment.equip(item, slot);
    if (!result.equipped) {
      if (previous !== null) this.inventory.place(item, previous.x, previous.y);
      else if (wasInBag) this.inventory.add(item);
      return result;
    }

    // Everything displaced must fit, or the whole operation is undone.
    const rehomed: Item[] = [];
    for (const displaced of result.displaced) {
      if (this.inventory.add(displaced) !== null) {
        rehomed.push(displaced);
        continue;
      }
      // Roll back: unequip what we just put on, restore the displaced items,
      // and put the original item back where it came from.
      if (result.slot !== null) this.equipment.unequip(result.slot);
      for (const restored of rehomed) this.inventory.remove(restored);
      for (const restored of result.displaced) {
        this.equipment.equip(restored, this.#slotForRestore(restored));
      }
      if (previous !== null) this.inventory.place(item, previous.x, previous.y);
      else if (wasInBag) this.inventory.add(item);
      return { equipped: false, displaced: [], slot: result.slot, reason: 'occupied' };
    }

    this.#version++;
    return result;
  }

  /** Take an item off and put it in the bag. @returns whether it fitted. */
  unequip(slot: EquipmentSlot): boolean {
    const item = this.equipment.get(slot);
    if (item === null) return false;
    if (!this.inventory.hasRoomFor(item)) return false;
    this.equipment.unequip(slot);
    this.inventory.add(item);
    this.#version++;
    return true;
  }

  /** Remove an item from wherever it is. @returns whether anything was removed. */
  discard(item: Item): boolean {
    if (this.inventory.remove(item)) {
      this.#version++;
      return true;
    }
    const slot = this.equipment.slotOf(item);
    if (slot !== null) {
      this.equipment.unequip(slot);
      this.#version++;
      return true;
    }
    return false;
  }

  /** Everything equipped that is worn out and doing nothing. */
  brokenItems(): Item[] {
    return this.equipment.items().filter(isBroken);
  }

  /* -- progression --------------------------------------------------------- */

  /** Award experience. @returns the level-up result, for the UI to announce. */
  addExperience(amount: number): ReturnType<CharacterStats['addExperience']> {
    const result = this.stats.addExperience(amount);
    this.#version++;
    return result;
  }

  spendStatPoint(stat: PrimaryStat, count = 1): boolean {
    const ok = this.stats.spendStatPoint(stat, count);
    if (ok) this.#version++;
    return ok;
  }

  investSkillPoint(skillId: string): boolean {
    const ok = this.skills.invest(skillId);
    if (ok) this.#version++;
    return ok;
  }

  setActiveSkill(skillId: string | null): boolean {
    const ok = this.skills.setActive(skillId);
    if (ok) this.#version++;
    return ok;
  }

  /* -- persistence --------------------------------------------------------- */

  toJSON(): CharacterSnapshot {
    return {
      name: this.#name,
      stats: this.stats.toJSON(),
      inventory: this.inventory.toJSON(),
      equipment: this.equipment.toJSON(),
      skills: this.skills.toJSON(),
      gold: this.#gold,
    };
  }

  load(snapshot: CharacterSnapshot): void {
    this.#name = snapshot.name;
    this.stats.load(snapshot.stats);
    this.inventory.load(snapshot.inventory);
    this.equipment.load(snapshot.equipment);
    this.skills.load(snapshot.skills);
    this.#gold = Math.max(0, Math.round(snapshot.gold));
    this.#version++;
  }

  /** Which slot a rolled-back item came off. Rings need the free-finger rule. */
  #slotForRestore(item: Item): EquipmentSlot | undefined {
    return this.equipment.preferredSlot(item) ?? undefined;
  }
}
