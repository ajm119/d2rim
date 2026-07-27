/**
 * @module rpg/RpgSystem
 *
 * The module that owns the character sheet at runtime and connects it to
 * everything else: experience from kills, loot into the bag, quest rewards into
 * the character, derived pools into combat's vitals, and the active skill into
 * the swing that is happening.
 *
 * ### The combat bridge, and why it is shaped like this
 *
 * `combat/CombatSystem` is frozen for this phase, and it resolves the player's
 * swing against a fixed `PLAYER_OFFENSE` constant. This module must not edit
 * it, so it works through the public surface instead:
 *
 * 1. `combat:swing` reports which move started. The active skill's mana is
 *    spent here, and Howl — an area skill with no single target — resolves here
 *    against every enemy in radius.
 * 2. `combat:hit` reports a landed blow. The character's *real* offence (gear,
 *    attributes, skill) is compared against the baseline the combat system just
 *    used, and the **difference** is delivered as a second damage packet
 *    through `CombatSystem.resolve` — the same public entry point every other
 *    source of damage in the game uses.
 *
 * The result is that equipping a weapon genuinely changes how fast a skeleton
 * dies, and investing in Bash genuinely changes it again, with no edit to a
 * combat file. Two consequences are worth stating plainly rather than hiding:
 *
 * - **Attack rating from gear cannot rescue a miss.** The base swing rolls to
 *   hit against `PLAYER_OFFENSE`'s rating; if it misses, no `combat:hit` fires
 *   and there is nothing to augment. Gear attack rating therefore shows on the
 *   character sheet and feeds the augmentation packet, but does not widen the
 *   base hit window.
 * - **Defensive modifiers do not reach incoming damage.** `PlayerCombatant`
 *   builds its `DefenseStats` from `PLAYER_DEFENSE_BASE`. Maximum life *does*
 *   flow through, because it is read from the live `Vitals` pool this module
 *   resizes — so `+life` is a real combat number — but defence rating and
 *   resistances are sheet-only for now.
 *
 * Both close the same way: `CombatSystem` asks the locator for
 * {@link OffenseProviderKey} / {@link DefenseProviderKey} instead of reading
 * its constants. Those keys are registered here already, so that change is one
 * line on the other side of the fence.
 */

import * as THREE from 'three/webgpu';

import { CombatantsKey, type Combatant, type CombatantRegistry, type IncomingHit } from '../combat/Combatant';
import { CombatKey, PLAYER_OFFENSE, type CombatSystem } from '../combat/CombatSystem';
import type { DamageRange, DamageSpread, DefenseStats, OffenseStats } from '../combat/DamageModel';
import { DAMAGE_TYPES } from '../combat/DamageModel';
import { serviceKey } from '../core/ServiceLocator';
import type { GameContext, GameModule } from '../core/types';
import { QuestSystem, QuestSystemKey, type RewardPayload } from '../quest/QuestSystem';
import { Character } from './Character';
import { generateItem, type Item } from './ItemGenerator';
import {
  dropTableFor,
  LootSystemKey,
  variantFromLabel,
  type LootReceiver,
  type LootSystem,
} from './Loot';
import { NO_SKILL_EFFECT } from './SkillTree';
import { experienceForKill } from './Stats';
import { VendorRegistry } from './Vendor';
import {
  AUTOSAVE_SLOT,
  captureSave,
  createSaveStore,
  type SaveData,
  type SaveStore,
} from './SaveGame';

/* -------------------------------------------------------------------------- */
/* Events                                                                      */
/* -------------------------------------------------------------------------- */

declare module '../core/EventBus' {
  interface GameEvents {
    /** Experience was awarded. `total` is the running total after the award. */
    'rpg:experience': { amount: number; total: number; level: number; progress: number };
    /** The character gained one or more levels. */
    'rpg:levelUp': { level: number; statPoints: number; skillPoints: number };
    /** Gold changed. `delta` is signed. */
    'rpg:gold': { gold: number; delta: number };
    /** Anything on the character sheet changed. Carries the version counter. */
    'rpg:characterChanged': { version: number; level: number };
    /** The active skill changed, by investment or by cycling. */
    'rpg:activeSkill': { skillId: string | null; points: number };
    /** An item entered the bag. */
    'rpg:itemAcquired': { name: string; quality: string };
    /** The game was saved, or a save was loaded. */
    'rpg:saved': { slot: string };
    'rpg:loaded': { slot: string };
  }
}

/* -------------------------------------------------------------------------- */
/* Service keys                                                                */
/* -------------------------------------------------------------------------- */

export const RpgSystemKey = serviceKey<RpgSystem>('rpg');

/**
 * The player's live offensive statistics.
 *
 * Registered so that `CombatSystem` can adopt them with a `tryGet` when it is
 * next opened for edit, replacing its `PLAYER_OFFENSE` constant. Nothing
 * currently reads it except this module's own bridge and the drive harness.
 */
export const OffenseProviderKey = serviceKey<{ offense(): OffenseStats }>('rpg.offense');

/** The player's live defensive statistics. Same rationale as above. */
export const DefenseProviderKey = serviceKey<{ defense(): DefenseStats }>('rpg.defense');

/* -------------------------------------------------------------------------- */
/* Options                                                                     */
/* -------------------------------------------------------------------------- */

export interface RpgSystemOptions {
  /** Starting gold. D2 starts a new character on nothing; 60 buys a repair. */
  readonly startingGold?: number;
  /** Give the character a starting weapon. Off makes level 1 bare-handed. */
  readonly startingKit?: boolean;
  /** Landed hits per point of weapon durability lost. */
  readonly hitsPerDurability?: number;
  /** Save store. Defaults to IndexedDB, falling back to memory. */
  readonly store?: SaveStore;
}

/* -------------------------------------------------------------------------- */
/* The module                                                                  */
/* -------------------------------------------------------------------------- */

export class RpgSystem implements GameModule, LootReceiver {
  readonly name = 'rpg';

  readonly character: Character;
  readonly quests = new QuestSystem();
  readonly vendors = new VendorRegistry();

  readonly #options: Required<Omit<RpgSystemOptions, 'store'>>;
  readonly #store: SaveStore;
  readonly #unsubscribe: Array<() => void> = [];
  readonly #scratch = new THREE.Vector3();
  readonly #scratchB = new THREE.Vector3();

  #ctx: GameContext | null = null;
  #combat: CombatSystem | null = null;
  #lastVersion = -1;
  #skillArmed = false;
  #augmenting = false;
  #hitsLanded = 0;
  #playTime = 0;
  #currentZone = 'encampment';

  constructor(options: RpgSystemOptions = {}) {
    this.#options = {
      startingGold: options.startingGold ?? 60,
      startingKit: options.startingKit ?? true,
      hitsPerDurability: Math.max(1, options.hitsPerDurability ?? 6),
    };
    this.character = new Character({ gold: this.#options.startingGold });
    this.#store = options.store ?? createSaveStore();
  }

  /* -- lifecycle ----------------------------------------------------------- */

  init(ctx: GameContext): void {
    this.#ctx = ctx;
    ctx.services.register(RpgSystemKey, this);
    ctx.services.register(QuestSystemKey, this.quests);
    ctx.services.register(OffenseProviderKey, { offense: () => this.offense() });
    ctx.services.register(DefenseProviderKey, { defense: () => this.defense() });

    this.quests.bindEvents(ctx.events);
    this.quests.setRewardHandler((payload) => this.#grantReward(payload));

    if (this.#options.startingKit) this.#grantStartingKit();

    this.#unsubscribe.push(
      ctx.events.on('combat:death', (payload) => {
        if (payload.faction === 'enemy') this.#onEnemyKilled(payload.label);
      }),
      ctx.events.on('combat:swing', (payload) => {
        if (payload.faction !== 'player') return;
        this.#onPlayerSwing();
      }),
      ctx.events.on('combat:hit', (payload) => this.#onCombatHit(payload)),
      ctx.events.on('zone:loaded', (payload) => {
        this.#currentZone = payload.zoneId;
        // Every zone arrival is a fresh visit for the merchants who live there.
        for (const vendor of this.vendors.all) vendor.refresh();
      }),
    );

    this.#emitChanged();
  }

  update(ctx: GameContext, dt: number): void {
    this.#playTime += dt;
    this.#bind(ctx);

    if (this.character.version !== this.#lastVersion) {
      this.#lastVersion = this.character.version;
      this.#applyDerivedPools(false);
      this.#emitChanged();
    }
  }

  dispose(): void {
    for (const off of this.#unsubscribe) off();
    this.#unsubscribe.length = 0;
    this.quests.bindEvents(null);
    this.quests.setRewardHandler(null);
    this.#store.close();
    const ctx = this.#ctx;
    if (ctx !== null) {
      ctx.services.unregister(DefenseProviderKey);
      ctx.services.unregister(OffenseProviderKey);
      ctx.services.unregister(QuestSystemKey);
      ctx.services.unregister(RpgSystemKey);
    }
    this.#ctx = null;
  }

  /* -- public surface ------------------------------------------------------ */

  /** The character's live offensive statistics, skill included. */
  offense(): OffenseStats {
    const base = this.character.offense(PLAYER_OFFENSE.criticalChance);
    const effect = this.character.skills.activeEffect();
    return {
      ...base,
      attackRating: base.attackRating + effect.attackRatingBonus,
      criticalChance: Math.min(1, base.criticalChance + effect.criticalBonus),
      damage: scaleSpread(base.damage, effect.damageScale),
    };
  }

  /** The character's live defensive statistics. */
  defense(): DefenseStats {
    return this.character.defense({ poise: 3 });
  }

  /** Seconds of play, for the save's header. */
  get playTime(): number {
    return this.#playTime;
  }

  get zoneId(): string {
    return this.#currentZone;
  }

  /** Award experience from script. Used by the drive harness. */
  awardExperience(amount: number): void {
    const result = this.character.addExperience(amount);
    const ctx = this.#ctx;
    ctx?.events.emit('rpg:experience', {
      amount,
      total: this.character.stats.experience,
      level: this.character.stats.level,
      progress: this.character.stats.progress,
    });
    if (result.levelsGained > 0) {
      // D2 refills both pools on level up, and it is the clearest possible
      // signal that something good just happened.
      this.#applyDerivedPools(true);
      ctx?.events.emit('rpg:levelUp', {
        level: result.level,
        statPoints: this.character.stats.statPoints,
        skillPoints: this.character.stats.skillPoints,
      });
    }
  }

  /** Invest a skill point and announce it. */
  investSkill(skillId: string): boolean {
    if (!this.character.investSkillPoint(skillId)) return false;
    this.#ctx?.events.emit('rpg:activeSkill', {
      skillId: this.character.skills.active,
      points: this.character.skills.pointsIn(this.character.skills.active ?? skillId),
    });
    return true;
  }

  /** Select the active skill and announce it. */
  selectSkill(skillId: string | null): boolean {
    if (!this.character.setActiveSkill(skillId)) return false;
    this.#ctx?.events.emit('rpg:activeSkill', {
      skillId,
      points: skillId === null ? 0 : this.character.skills.pointsIn(skillId),
    });
    return true;
  }

  /** Cycle to the next usable active skill. */
  cycleSkill(): string | null {
    const next = this.character.skills.cycleActive();
    this.character.touch();
    this.#ctx?.events.emit('rpg:activeSkill', {
      skillId: next,
      points: next === null ? 0 : this.character.skills.pointsIn(next),
    });
    return next;
  }

  /** Restore the character to full. Akara's healing. */
  healToFull(): void {
    const combat = this.#combat;
    if (combat === null) return;
    combat.vitals.health.refill();
    combat.vitals.mana.refill();
  }

  /* -- LootReceiver -------------------------------------------------------- */

  takeItem(item: Item): boolean {
    if (this.character.acquire(item) === 'full') return false;
    this.#ctx?.events.emit('rpg:itemAcquired', { name: item.name, quality: item.quality });
    return true;
  }

  takeGold(amount: number): void {
    const added = this.character.addGold(amount);
    this.#ctx?.events.emit('rpg:gold', { gold: this.character.gold, delta: added });
  }

  /* -- persistence --------------------------------------------------------- */

  /** Snapshot the game into `slot`. */
  async save(slot: string = AUTOSAVE_SLOT): Promise<SaveData> {
    const data = captureSave({
      character: this.character,
      quests: this.quests,
      zoneId: this.#currentZone,
      playTime: this.#playTime,
    });
    await this.#store.write(slot, data);
    this.#ctx?.events.emit('rpg:saved', { slot });
    return data;
  }

  /** Load `slot`. @returns whether anything was loaded. */
  async load(slot: string = AUTOSAVE_SLOT): Promise<boolean> {
    const data = await this.#store.read(slot);
    if (data === null) return false;
    this.character.load(data.character);
    this.quests.load(data.quests);
    this.#playTime = data.playTime;
    this.#applyDerivedPools(true);
    this.#emitChanged();
    this.#ctx?.events.emit('rpg:loaded', { slot });
    return true;
  }

  get store(): SaveStore {
    return this.#store;
  }

  /* -- internals ----------------------------------------------------------- */

  /**
   * Resolve combat and the loot system once they exist.
   *
   * Lazy for the same reason the combat system's own player binding is: module
   * `init` order cannot guarantee that everything this needs has registered.
   */
  #bind(ctx: GameContext): void {
    if (this.#combat === null) {
      this.#combat = ctx.services.tryGet<CombatSystem>(CombatKey) ?? null;
      if (this.#combat !== null) this.#applyDerivedPools(true);
    }
    const loot = ctx.services.tryGet<LootSystem>(LootSystemKey);
    if (loot !== undefined) loot.setReceiver(this);
  }

  /**
   * Push derived maxima into the live pools.
   *
   * `heal` refills afterwards, which is what a level up should do; a gear
   * change keeps the current value and only moves the ceiling, so swapping a
   * `+20 life` ring in and out is not an infinite heal.
   */
  #applyDerivedPools(heal: boolean): void {
    const combat = this.#combat;
    if (combat === null) return;
    const derived = this.character.derived;
    combat.vitals.health.setMax(derived.maxLife, false);
    combat.vitals.mana.setMax(derived.maxMana, false);
    if (heal) {
      combat.vitals.health.refill();
      combat.vitals.mana.refill();
    }
    this.#ctx?.events.emit('combat:vitals', {
      health: combat.vitals.health.value,
      healthMax: combat.vitals.health.max,
      mana: combat.vitals.mana.value,
      manaMax: combat.vitals.mana.max,
      stamina: combat.stamina,
      staminaMax: combat.staminaMax,
    });
  }

  #emitChanged(): void {
    this.#ctx?.events.emit('rpg:characterChanged', {
      version: this.character.version,
      level: this.character.stats.level,
    });
  }

  #onEnemyKilled(label: string): void {
    const table = dropTableFor(variantFromLabel(label));
    const award = experienceForKill(
      table.monsterLevel,
      this.character.stats.level,
      table.experience,
    );
    this.awardExperience(award);
  }

  /* -- the combat bridge --------------------------------------------------- */

  /**
   * Called when the player starts a move.
   *
   * Spends the active skill's mana, and resolves area skills — the ones with no
   * single target for a `combat:hit` to hang off — here and now.
   */
  #onPlayerSwing(): void {
    const combat = this.#combat;
    if (combat === null) return;
    const effect = this.character.skills.activeEffect();
    if (effect.manaCost > 0 && !combat.vitals.mana.spend(effect.manaCost)) {
      // Not enough mana: the swing still lands, it just lands as a plain swing.
      // Refusing the whole attack would be worse — the player pressed a button
      // and nothing happened, with no feedback saying why. `#skillArmed` is how
      // the hit path learns that this particular swing is unskilled.
      this.#skillArmed = false;
      return;
    }
    this.#skillArmed = true;
    if (effect.radius > 0) this.#resolveAreaSkill(effect.radius, effect.staggerScale);
  }

  /** Howl: every enemy inside `radius` is rocked back. */
  #resolveAreaSkill(radius: number, staggerScale: number): void {
    const combat = this.#combat;
    const ctx = this.#ctx;
    if (combat === null || ctx === null) return;
    const self = combat.self;
    if (self === null) return;

    const registry = ctx.services.tryGet<CombatantRegistry>(CombatantsKey);
    if (registry === undefined) return;
    const origin = self.footPosition(this.#scratch).clone();

    const offense = this.offense();
    for (const target of registry.hostileTo('player')) {
      const position = target.footPosition(this.#scratchB);
      if (position.distanceTo(origin) > radius) continue;
      const direction = position.clone().sub(origin).setY(0);
      if (direction.lengthSq() < 1e-6) direction.set(0, 0, 1);
      direction.normalize();
      const hit: IncomingHit = {
        source: self,
        offense: { ...offense, alwaysHits: true, unblockable: true, damage: HOWL_DAMAGE },
        move: { id: 'skill.howl', damageScale: 1, staggerScale, knockback: 1.2, unblockable: true },
        point: position.clone().setY(position.y + 1),
        normal: direction.clone().negate(),
        direction,
      };
      this.#withAugmentGuard(() => combat.resolve(self, target, hit));
    }
  }

  /**
   * Called on every landed blow, from either side.
   *
   * Only the player's own hits on an enemy are augmented, and only when the
   * augmentation is not itself the thing being reported — without that guard
   * this is an infinite loop, because `CombatSystem.resolve` emits the very
   * event that got us here.
   */
  #onCombatHit(payload: {
    attacker: number;
    target: number;
    outcome: { result: string; total: number };
  }): void {
    if (this.#augmenting) return;
    const combat = this.#combat;
    const ctx = this.#ctx;
    if (combat === null || ctx === null) return;
    const self = combat.self;
    if (self === null || payload.attacker !== self.id) return;
    // A blocked blow was already absorbed; augmenting it would sneak damage
    // past the shield the defender successfully raised.
    if (payload.outcome.result !== 'hit') return;

    const registry = ctx.services.tryGet<CombatantRegistry>(CombatantsKey);
    const target = registry?.all.find((entry) => entry.id === payload.target);
    if (target === undefined || !target.alive) return;

    this.#hitsLanded++;
    if (this.#hitsLanded % this.#options.hitsPerDurability === 0) this.#wearWeapon();

    const effect = this.#skillArmed ? this.character.skills.activeEffect() : NO_SKILL_EFFECT;
    const mine = this.#skillArmed ? this.offense() : this.character.offense(PLAYER_OFFENSE.criticalChance);

    // There used to be a third packet here: the *difference* between what the
    // character should have dealt and what `CombatSystem` had already dealt
    // resolving the swing against its `PLAYER_OFFENSE` constant. It was a
    // workaround for combat not being able to see the character sheet, and it
    // cost three things — gear attack rating could not rescue a base miss,
    // because the roll had already happened; the extra `resolve` drew from
    // combat's shared RNG, so a seeded encounter replayed differently with the
    // RPG layer loaded than without it; and every landed blow arrived as two
    // damage numbers.
    //
    // `CombatSystem` now resolves the player's swing against
    // `RpgOffenseKey.offense()` directly, so the base hit already carries the
    // character's real damage, attack rating and critical chance. Sending the
    // difference on top would double-count all of it.

    // Extra strikes remain, because they are not augmentation: Double Swing's
    // second blow is a blow, and it should read as one.
    for (let i = 0; i < effect.extraHits; i++) {
      this.#deliver(combat, self, target, mine, mine.damage, 'skill.extra');
    }
  }

  #deliver(
    combat: CombatSystem,
    self: Combatant,
    target: Combatant,
    offense: OffenseStats,
    damage: DamageSpread,
    moveId: string,
  ): void {
    const point = target.footPosition(this.#scratch).clone();
    point.y += target.hitHeight * 0.55;
    const direction = point.clone().sub(self.footPosition(this.#scratchB)).setY(0);
    if (direction.lengthSq() < 1e-6) direction.set(0, 0, 1);
    direction.normalize();

    const hit: IncomingHit = {
      source: self,
      offense: {
        ...offense,
        damage,
        // The base swing already made — and passed — its hit and block rolls.
        // Rolling again here would mean gear that adds damage also adds a
        // second chance to whiff, which is not what "+5 to maximum damage"
        // says on the tin.
        alwaysHits: true,
        unblockable: true,
      },
      move: { id: moveId, damageScale: 1, knockback: 0, staggerScale: 0, unblockable: true },
      point,
      normal: direction.clone().negate(),
      direction,
    };
    this.#withAugmentGuard(() => combat.resolve(self, target, hit));
  }

  #withAugmentGuard(action: () => void): void {
    this.#augmenting = true;
    try {
      action();
    } finally {
      this.#augmenting = false;
    }
  }

  #wearWeapon(): void {
    const weapon = this.character.equipment.get('weapon');
    if (weapon === null || weapon.maxDurability <= 0 || weapon.durability <= 0) return;
    weapon.durability -= 1;
    this.character.touch();
  }

  /* -- rewards and starting kit -------------------------------------------- */

  #grantReward(payload: RewardPayload): void {
    const rewards = payload.rewards;
    if (rewards.experience !== undefined) this.awardExperience(rewards.experience);
    if (rewards.gold !== undefined) this.takeGold(rewards.gold);
    if (rewards.skillPoints !== undefined) this.character.stats.grantSkillPoints(rewards.skillPoints);
    if (rewards.statPoints !== undefined) this.character.stats.grantStatPoints(rewards.statPoints);
    for (const request of rewards.items ?? []) {
      const item = generateItem({
        seed: request.seed,
        itemLevel: request.itemLevel,
        ...(request.baseId !== undefined ? { baseId: request.baseId } : {}),
      });
      this.takeItem(item);
    }
    this.character.touch();
  }

  /**
   * The clothes the Barbarian starts in.
   *
   * A hand axe and a cap, both plain. Deliberately unexciting: the first magic
   * drop has to feel like an upgrade, and it cannot if the starting kit already
   * has a prefix on it.
   */
  #grantStartingKit(): void {
    const axe = generateItem({ seed: 0x1a2b3c, itemLevel: 1, baseId: 'hand-axe', quality: 'normal' });
    const cap = generateItem({ seed: 0x4d5e6f, itemLevel: 1, baseId: 'cap', quality: 'normal' });
    this.character.acquire(axe);
    this.character.acquire(cap);
    this.character.equip(axe);
    this.character.equip(cap);
  }
}

/* -------------------------------------------------------------------------- */
/* Damage arithmetic helpers                                                   */
/* -------------------------------------------------------------------------- */

/** Howl's token damage. It exists so the hit registers; it is not the point. */
const HOWL_DAMAGE: DamageSpread = { physical: { min: 1, max: 2 } };

/** Multiply every range in a spread. Used to apply a skill's `damageScale`. */
export function scaleSpread(spread: DamageSpread, scale: number): DamageSpread {
  if (scale === 1) return spread;
  const out: Record<string, DamageRange> = {};
  for (const type of DAMAGE_TYPES) {
    const range = spread[type];
    if (range === undefined) continue;
    out[type] = {
      min: Math.max(0, Math.round(range.min * scale)),
      max: Math.max(0, Math.round(range.max * scale)),
    };
  }
  return out as DamageSpread;
}

/**
 * `mine - baseline`, per damage type, floored at zero.
 *
 * @returns null when the difference is empty, which is the signal to skip the
 * augmentation entirely. That matters: `mitigatePhysical` floors a landed blow
 * at 1 damage, so delivering an all-zero packet would quietly hand a
 * bare-handed level 1 character a free point of damage on every swing.
 */
export function differenceSpread(
  mine: DamageSpread,
  baseline: DamageSpread,
): DamageSpread | null {
  const out: Record<string, DamageRange> = {};
  let any = false;
  for (const type of DAMAGE_TYPES) {
    const range = mine[type];
    if (range === undefined) continue;
    const base = baseline[type];
    const min = Math.max(0, Math.round(range.min - (base?.min ?? 0)));
    const max = Math.max(min, Math.round(range.max - (base?.max ?? 0)));
    if (max <= 0) continue;
    out[type] = { min, max };
    any = true;
  }
  return any ? (out as DamageSpread) : null;
}
