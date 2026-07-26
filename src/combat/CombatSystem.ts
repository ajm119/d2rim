/**
 * @module combat/CombatSystem
 *
 * The melee loop: input to combo to animation to swept hitbox to damage.
 *
 * ### The three things that decide whether melee feels good
 *
 * 1. **Input buffering.** A press during the previous swing's recovery is
 *    remembered for {@link DEFAULT_BUFFER_TIME} and fires the moment the
 *    character can act. Without it the player has to time their input to a
 *    window they cannot see, and the game reads as unresponsive no matter how
 *    good the animation is. This is the single highest-value 40 lines in the
 *    file.
 * 2. **A real chain.** Successive presses walk a *table* of different clips —
 *    chop, diagonal slice, stab, then the two-handed spin as a finisher —
 *    rather than restarting one clip. Replaying one animation is the tell that
 *    an attack is a button and not a move.
 * 3. **Authored hit windows.** The damage window is two normalised times on the
 *    clip, delivered as `AnimationGraph` events. The axe deals damage when the
 *    axe is passing through the enemy, which is the only version of this that
 *    survives contact with a player who is watching.
 *
 * ### Ownership
 *
 * This module registers itself as the `combat` service, and `PlayerController`
 * stands its placeholder attack input down the moment that key exists. That
 * only works if this module is registered with the engine **before** the player
 * — see `main.ts`. Its own binding to the player is therefore lazy: it looks
 * the player up every frame until it finds one.
 *
 * ### Phase placement
 *
 * - `fixedUpdate`: the combo machine, vitals, death and respawn. Deterministic.
 * - `lateUpdate`: the hitbox sweep, because it must run after the animation
 *   mixer has advanced (`PlayerController.update`), or the blade segment
 *   sampled is one frame stale and every hit lands late.
 */

import * as THREE from 'three/webgpu';

import { PlayerKey, type PlayerController } from '../character/PlayerController';
import { serviceKey } from '../core/ServiceLocator';
import type { GameContext, GameModule } from '../core/types';
import { CombatantRegistry, CombatantsKey, type Combatant, type IncomingHit } from './Combatant';
import {
  mulberry32,
  resolveAttack,
  type AttackOutcome,
  type DefenseStats,
  type MoveModifiers,
  type OffenseStats,
  type Rng,
} from './DamageModel';
import { WeaponHitbox, resolveWeaponAnchor, type WeaponAnchor } from './Hitbox';
import { DEFAULT_VITALS, Vitals } from './Vitals';

declare module '../core/EventBus' {
  interface GameEvents {
    /** A move started. Fired on the frame the animation is triggered. */
    'combat:swing': { attacker: number; faction: string; moveId: string };
    /** A swing connected — including a miss roll, which still made contact. */
    'combat:hit': {
      attacker: number;
      target: number;
      targetLabel: string;
      outcome: AttackOutcome;
      point: THREE.Vector3;
      normal: THREE.Vector3;
      direction: THREE.Vector3;
      /** Fraction of the target's health the blow removed, `[0, 1]`. */
      severity: number;
    };
    /** A combatant's health reached zero. */
    'combat:death': { combatant: number; faction: string; label: string };
    /** The player's pools changed. Drives the HUD. */
    'combat:vitals': {
      health: number;
      healthMax: number;
      mana: number;
      manaMax: number;
      stamina: number;
      staminaMax: number;
    };
    /** The player died and is waiting to respawn. */
    'combat:playerDown': { position: THREE.Vector3 };
    'combat:respawn': { position: THREE.Vector3 };
  }
}

/**
 * Service id `combat`. It must be exactly this string: `PlayerController`
 * tests for it by name to disable its placeholder attack input.
 */
export const CombatKey = serviceKey<CombatSystem>('combat');

/* -------------------------------------------------------------------------- */
/* Moves                                                                       */
/* -------------------------------------------------------------------------- */

export type AttackKind = 'light' | 'heavy';

/** Which blade a move swings. Resolved to an anchor once, at bind time. */
export type WeaponSlot = '1h' | '2h' | 'limb';

export interface MeleeMove {
  readonly id: string;
  /** `AnimationGraph` semantic state, e.g. `attack.slice`. */
  readonly action: string;
  readonly layer: 'full' | 'upper';
  readonly weapon: WeaponSlot;
  /** Playback multiplier. Below 1 reads as heavier. */
  readonly speed: number;
  /** Damage window as normalised clip times, `[open, close]`. */
  readonly window: readonly [number, number];
  /**
   * Normalised window in which a queued press chains into {@link next}.
   * Opening it *before* the hit lands is deliberate: the player commits to the
   * follow-up while the current swing is still travelling, which is what makes
   * a chain feel like one continuous motion rather than three button presses.
   */
  readonly chain: readonly [number, number];
  /** Normalised time at which the move releases control. */
  readonly recovery: number;
  /** Stamina cost. Refused, not clipped, when the pool is short. */
  readonly stamina: number;
  readonly modifiers: MoveModifiers;
  /** Where each input kind goes from here. Null restarts that kind's chain. */
  readonly next: Readonly<Record<AttackKind, string | null>>;
  /** Multiplier on hit-stop duration. Finishers hang longer. */
  readonly hitStop: number;
}

export type MoveTable = ReadonlyMap<string, MeleeMove>;

function move(definition: MeleeMove): MeleeMove {
  return definition;
}

/**
 * The Barbarian's melee vocabulary.
 *
 * Light attacks play on the **upper body** so the player keeps their feet — a
 * third-person game where a swing roots you in place feels broken. The heavy
 * and the spin are full-body precisely because they take control away; that is
 * the cost that makes them read as committed, and it is why the spin is the
 * finisher rather than the opener.
 */
export const PLAYER_MOVES: MoveTable = new Map(
  [
    move({
      id: 'chop',
      action: 'attack',
      layer: 'upper',
      weapon: '1h',
      speed: 1.05,
      window: [0.3, 0.52],
      chain: [0.34, 0.98],
      recovery: 0.78,
      stamina: 8,
      modifiers: { damageScale: 1, knockback: 2.6, staggerScale: 1 },
      next: { light: 'slice', heavy: 'heavy' },
      hitStop: 1,
    }),
    move({
      id: 'slice',
      action: 'attack.slice',
      layer: 'upper',
      weapon: '1h',
      speed: 1.1,
      window: [0.26, 0.5],
      chain: [0.32, 0.98],
      recovery: 0.76,
      stamina: 9,
      modifiers: { damageScale: 1.1, attackRatingBonus: 12, knockback: 2.8 },
      next: { light: 'stab', heavy: 'heavy' },
      hitStop: 1,
    }),
    move({
      id: 'stab',
      action: 'attack.stab',
      layer: 'upper',
      weapon: '1h',
      speed: 1.15,
      window: [0.32, 0.54],
      chain: [0.4, 0.98],
      recovery: 0.8,
      stamina: 10,
      modifiers: { damageScale: 1.15, criticalBonus: 0.08, knockback: 2.2 },
      next: { light: 'spin', heavy: 'heavy' },
      hitStop: 1.1,
    }),
    move({
      id: 'spin',
      action: 'attack.spin',
      layer: 'full',
      weapon: '2h',
      speed: 0.95,
      // A wide window: the whole point of a spin is that it sweeps everything
      // around you, and a narrow window on a 360 turns it into a worse chop.
      window: [0.26, 0.72],
      chain: [0.7, 0.98],
      recovery: 0.9,
      stamina: 24,
      modifiers: {
        damageScale: 1.7,
        knockback: 5,
        staggerScale: 1.6,
        unblockable: true,
        attackRatingBonus: 25,
      },
      next: { light: null, heavy: null },
      hitStop: 1.6,
    }),
    move({
      id: 'heavy',
      action: 'attack.heavy',
      layer: 'full',
      weapon: '2h',
      speed: 0.9,
      window: [0.4, 0.62],
      chain: [0.5, 0.98],
      recovery: 0.88,
      stamina: 22,
      modifiers: { damageScale: 1.85, knockback: 4.4, staggerScale: 1.8, criticalBonus: 0.05 },
      next: { light: 'chop', heavy: 'spin' },
      hitStop: 1.5,
    }),
  ].map((entry) => [entry.id, entry]),
);

/** Where each input kind starts a fresh chain. */
export const PLAYER_CHAIN_ROOTS: Readonly<Record<AttackKind, string>> = {
  light: 'chop',
  heavy: 'heavy',
};

/* -------------------------------------------------------------------------- */
/* Combo state machine (pure)                                                  */
/* -------------------------------------------------------------------------- */

export type ComboPhase = 'idle' | 'swinging';

/** Seconds a queued press survives. Long enough to be forgiving, short enough
 *  that a mashed button does not fire three swings after the player stopped. */
export const DEFAULT_BUFFER_TIME = 0.38;

export interface ComboOptions {
  readonly bufferTime?: number;
  readonly roots?: Readonly<Record<AttackKind, string>>;
}

/**
 * The combo chain, as a state machine over normalised clip time.
 *
 * Deliberately free of three.js, the engine, and the animation graph: it takes
 * seconds in and returns "start this move" out, which is what makes the whole
 * chain — buffering, chaining, expiry, finishers — testable without a renderer.
 *
 * The caller is responsible for telling it how long the started clip actually
 * is (`setDuration`), because clip length lives in the GLB and this class must
 * not know about GLBs.
 */
export class ComboMachine {
  readonly #moves: MoveTable;
  readonly #roots: Readonly<Record<AttackKind, string>>;
  readonly #bufferTime: number;

  #current: MeleeMove | null = null;
  #elapsed = 0;
  #duration = 1;
  #chainIndex = 0;
  #buffered: AttackKind | null = null;
  #bufferLeft = 0;

  constructor(moves: MoveTable, options: ComboOptions = {}) {
    this.#moves = moves;
    this.#roots = options.roots ?? PLAYER_CHAIN_ROOTS;
    this.#bufferTime = Math.max(0, options.bufferTime ?? DEFAULT_BUFFER_TIME);
  }

  get phase(): ComboPhase {
    return this.#current === null ? 'idle' : 'swinging';
  }

  get current(): MeleeMove | null {
    return this.#current;
  }

  /** Progress through the current move in `[0, 1]`. 0 when idle. */
  get normalizedTime(): number {
    if (this.#current === null || this.#duration <= 0) return 0;
    return Math.min(1, this.#elapsed / this.#duration);
  }

  /** How many moves deep the current chain is. 0 for the opener. */
  get chainIndex(): number {
    return this.#chainIndex;
  }

  get buffered(): AttackKind | null {
    return this.#buffered;
  }

  get bufferRemaining(): number {
    return this.#buffered === null ? 0 : this.#bufferLeft;
  }

  /** Whether a queued press would chain right now rather than wait. */
  get chainWindowOpen(): boolean {
    const current = this.#current;
    if (current === null) return true;
    const n = this.normalizedTime;
    return n >= current.chain[0] && n <= current.chain[1];
  }

  /** Queue an attack. A second press simply replaces the queued one. */
  press(kind: AttackKind): void {
    this.#buffered = kind;
    this.#bufferLeft = this.#bufferTime;
  }

  /** Forget any queued press. Used when blocking or dying. */
  clearBuffer(): void {
    this.#buffered = null;
    this.#bufferLeft = 0;
  }

  /**
   * Tell the machine how long the clip it just handed out really is.
   *
   * Called immediately after {@link update} returns a move. If the caller never
   * calls it the machine falls back to a 1 s clip, which keeps a missing-clip
   * situation ticking rather than wedging the player in a swing forever.
   */
  setDuration(seconds: number): void {
    this.#duration = Number.isFinite(seconds) && seconds > 0.01 ? seconds : 1;
  }

  /** Abandon the current move without starting anything (stagger, death). */
  interrupt(): void {
    this.#current = null;
    this.#elapsed = 0;
    this.#chainIndex = 0;
    this.clearBuffer();
  }

  /**
   * Advance one tick.
   *
   * @returns the move to start this tick, or null. At most one per tick: two
   * swings in one frame is never what the player asked for.
   */
  update(dt: number): MeleeMove | null {
    const started = this.#advance(dt);
    // Buffer decay happens *after* the start attempt, never before. The other
    // order silently loses a press whenever one frame is longer than the buffer
    // window — rare on a good machine, routine on a bad one, and it presents as
    // "the game dropped my input" precisely when the game is already struggling.
    if (started === null && dt > 0 && this.#buffered !== null) {
      this.#bufferLeft -= dt;
      if (this.#bufferLeft <= 0) this.clearBuffer();
    }
    return started;
  }

  #advance(dt: number): MeleeMove | null {
    const current = this.#current;
    if (current !== null) {
      this.#elapsed += Math.max(0, dt);
      const n = this.normalizedTime;

      if (this.#buffered !== null && n >= current.chain[0] && n <= current.chain[1]) {
        const nextId = current.next[this.#buffered] ?? this.#roots[this.#buffered];
        const next = this.#moves.get(nextId);
        if (next !== undefined) {
          this.clearBuffer();
          return this.#begin(next, this.#chainIndex + 1);
        }
      }

      if (n < current.recovery) return null;
      // Recovery reached: the move is over. Fall through so a press buffered
      // during recovery starts its chain on this same tick rather than on the
      // next one — one frame of latency here is felt.
      this.#current = null;
      this.#elapsed = 0;
      this.#chainIndex = 0;
    }

    if (this.#buffered === null) return null;
    const rootId = this.#roots[this.#buffered];
    const root = this.#moves.get(rootId);
    if (root === undefined) {
      this.clearBuffer();
      return null;
    }
    this.clearBuffer();
    return this.#begin(root, 0);
  }

  #begin(next: MeleeMove, chainIndex: number): MeleeMove {
    this.#current = next;
    this.#elapsed = 0;
    this.#duration = 1;
    this.#chainIndex = chainIndex;
    return next;
  }
}

/* -------------------------------------------------------------------------- */
/* Player statistics                                                           */
/* -------------------------------------------------------------------------- */

/** A level-1 Barbarian with a hand axe, roughly D2's opening numbers. */
export const PLAYER_OFFENSE: OffenseStats = {
  level: 1,
  attackRating: 105,
  damage: { physical: { min: 9, max: 17 } },
  criticalChance: 0.09,
  criticalMultiplier: 2,
};

export const PLAYER_DEFENSE_BASE = {
  level: 1,
  defense: 42,
  resistances: { fire: 0, cold: 0, lightning: 0, poison: 0 },
  blockChance: 0.42,
  blockAbsorb: 1,
  physicalReduction: 0.05,
  flatReduction: 1,
  maxHealth: DEFAULT_VITALS.health.max,
  poise: 3,
} as const;

/* -------------------------------------------------------------------------- */
/* The module                                                                  */
/* -------------------------------------------------------------------------- */

export interface CombatSystemOptions {
  /** Seed for every roll this system makes. Fixed by default: replayable. */
  readonly seed?: number;
  /** Seconds the corpse lies there before respawning. Default 3.2. */
  readonly respawnDelay?: number;
  readonly moves?: MoveTable;
  readonly bufferTime?: number;
}

/**
 * The player's own {@link Combatant} face.
 *
 * Split out from the module so that "the thing skeletons swing at" is a small
 * object with a clear contract rather than the whole combat system, and so the
 * hitbox code has exactly one shape of target to reason about.
 */
class PlayerCombatant implements Combatant {
  readonly id: number;
  readonly faction = 'player' as const;
  readonly label = 'player';
  readonly vitals: Vitals;

  readonly #system: CombatSystem;
  readonly #player: PlayerController;

  constructor(id: number, system: CombatSystem, player: PlayerController, vitals: Vitals) {
    this.id = id;
    this.#system = system;
    this.#player = player;
    this.vitals = vitals;
  }

  get object(): THREE.Object3D | null {
    return this.#player.object;
  }

  get alive(): boolean {
    return this.vitals.alive;
  }

  get hitRadius(): number {
    return 0.4;
  }

  get hitHeight(): number {
    return Math.max(1.2, this.#player.height);
  }

  get defense(): DefenseStats {
    return {
      ...PLAYER_DEFENSE_BASE,
      blocking: this.#system.blocking,
      maxHealth: this.vitals.health.max,
    };
  }

  footPosition(out: THREE.Vector3): THREE.Vector3 {
    return out.copy(this.#player.position);
  }

  facing(out: THREE.Vector3): THREE.Vector3 {
    return this.#player.forward(out);
  }

  receiveHit(hit: IncomingHit): AttackOutcome {
    return this.#system.applyHitToPlayer(this, hit);
  }
}

export class CombatSystem implements GameModule {
  readonly name = 'combat';

  readonly #options: Required<CombatSystemOptions>;
  readonly #registry = new CombatantRegistry();
  readonly #rng: Rng;

  readonly #scratchA = new THREE.Vector3();
  readonly #scratchB = new THREE.Vector3();
  readonly #spawn = new THREE.Vector3();

  #ctx: GameContext | null = null;
  #player: PlayerController | null = null;
  #self: PlayerCombatant | null = null;
  #vitals: Vitals;
  #combo: ComboMachine;

  #hitboxes = new Map<WeaponSlot, WeaponHitbox>();
  #activeHitbox: WeaponHitbox | null = null;
  #activeMove: MeleeMove | null = null;

  #blocking = false;
  #blockHandleActive = false;
  #inputFrame = -1;
  #respawnTimer = 0;
  #detachAnimation: (() => void) | null = null;

  constructor(options: CombatSystemOptions = {}) {
    this.#options = {
      seed: options.seed ?? 0x5eed1234,
      respawnDelay: options.respawnDelay ?? 3.2,
      moves: options.moves ?? PLAYER_MOVES,
      bufferTime: options.bufferTime ?? DEFAULT_BUFFER_TIME,
    };
    this.#rng = mulberry32(this.#options.seed);
    this.#vitals = new Vitals(DEFAULT_VITALS);
    this.#combo = new ComboMachine(this.#options.moves, { bufferTime: this.#options.bufferTime });
  }

  /* -- public surface ----------------------------------------------------- */

  get targets(): CombatantRegistry {
    return this.#registry;
  }

  get vitals(): Vitals {
    return this.#vitals;
  }

  get combo(): ComboMachine {
    return this.#combo;
  }

  get blocking(): boolean {
    return this.#blocking;
  }

  /** The player's own combatant, once the player has loaded. */
  get self(): Combatant | null {
    return this.#self;
  }

  /** Shared RNG. Enemies roll from it so an encounter replays identically. */
  get rng(): Rng {
    return this.#rng;
  }

  /** Stamina is owned by `PlayerController`; this is the read-through. */
  get stamina(): number {
    return this.#player?.stamina ?? 0;
  }

  get staminaMax(): number {
    return this.#player?.staminaMax ?? 1;
  }

  /** Current move id, for the debug readout and the drive harness. */
  get moveId(): string | null {
    return this.#combo.current?.id ?? null;
  }

  /** Whether the player's damage window is open right now. */
  get hitWindowOpen(): boolean {
    return this.#activeHitbox?.isOpen ?? false;
  }

  /**
   * Queue an attack from script rather than from the keyboard.
   *
   * The drive harness and the encounter tests use this; it goes through exactly
   * the same buffer and chain the player's own press does, so a scripted combo
   * exercises the real state machine rather than a shortcut around it.
   */
  press(kind: AttackKind): void {
    this.#combo.press(kind);
  }

  /**
   * Deal damage to a combatant from an arbitrary source.
   *
   * The single entry point for *all* damage in the game — swings, spikes,
   * scripted events — so that feedback, death and the event stream have exactly
   * one place to hook.
   */
  resolve(attacker: Combatant | null, target: Combatant, hit: IncomingHit): AttackOutcome {
    const outcome = target.receiveHit(hit);
    const ctx = this.#ctx;
    if (ctx === null) return outcome;

    if (outcome.result !== 'miss') {
      const maxHealth = Math.max(1, target.defense.maxHealth);
      ctx.events.emit('combat:hit', {
        attacker: attacker?.id ?? 0,
        target: target.id,
        targetLabel: target.label,
        outcome,
        point: hit.point.clone(),
        normal: hit.normal.clone(),
        direction: hit.direction.clone(),
        severity: Math.min(1, outcome.total / maxHealth),
      });
    }
    return outcome;
  }

  /** Applied by {@link PlayerCombatant.receiveHit}. Not called directly. */
  applyHitToPlayer(self: PlayerCombatant, hit: IncomingHit): AttackOutcome {
    const facing = this.#facingTowards(self, hit.point);
    const outcome = resolveAttack(hit.offense, self.defense, this.#rng, {
      move: hit.move,
      facing,
    });
    if (outcome.result === 'miss') return outcome;

    const graph = this.#player?.animation ?? null;
    if (outcome.result === 'blocked') {
      graph?.playAction('block.hit', { layer: 'upper', fadeIn: 0.04 });
      return outcome;
    }

    const before = self.vitals.alive;
    const applied = self.vitals.applyDamage(outcome.total);
    this.#emitVitals();

    if (applied.killed && before) {
      this.#combo.interrupt();
      this.#activeHitbox?.cancel();
      this.#activeHitbox = null;
      this.#activeMove = null;
      this.#blocking = false;
      this.#respawnTimer = this.#options.respawnDelay;
      this.#player?.setEnabled(false);
      graph?.cancelActions();
      graph?.playAction('death', { layer: 'full', hold: true, fadeIn: 0.12 });
      this.#ctx?.events.emit('combat:playerDown', { position: this.#player?.position.clone() ?? new THREE.Vector3() });
      this.#ctx?.events.emit('combat:death', { combatant: self.id, faction: 'player', label: 'player' });
    } else if (outcome.staggered) {
      this.#combo.interrupt();
      this.#activeHitbox?.cancel();
      this.#activeHitbox = null;
      this.#activeMove = null;
      graph?.playAction('hit', { layer: 'upper', fadeIn: 0.03 });
    }
    return outcome;
  }

  /* -- lifecycle ---------------------------------------------------------- */

  init(ctx: GameContext): void {
    this.#ctx = ctx;
    ctx.services.register(CombatKey, this);
    ctx.services.register(CombatantsKey, this.#registry);
  }

  fixedUpdate(ctx: GameContext, dt: number): void {
    this.#bind(ctx);
    this.#capturePresses(ctx);

    const self = this.#self;
    if (self === null) return;

    this.#vitals.update(dt);

    if (!self.alive) {
      this.#respawnTimer -= dt;
      if (this.#respawnTimer <= 0) this.#respawn(ctx);
      return;
    }

    this.#updateBlock();

    const started = this.#combo.update(dt);
    if (started !== null) this.#startMove(ctx, started);
    if (this.#combo.current === null && this.#activeMove !== null) this.#endMove();
  }

  lateUpdate(): void {
    const self = this.#self;
    const hitbox = this.#activeHitbox;
    const move = this.#activeMove;

    // Every hitbox tracks every frame, open or not, so that the sweep always
    // has a truthful previous pose to sweep *from*. Skipping this while closed
    // is the classic version of this bug: the first frame after the window
    // opens sweeps from wherever the blade was a whole swing ago.
    for (const box of this.#hitboxes.values()) {
      if (box !== hitbox) box.track([]);
    }
    if (hitbox === null || move === null || self === null) return;

    const targets = this.#registry.hostileTo('player');
    for (const contact of hitbox.track(targets)) {
      const hit: IncomingHit = {
        source: self,
        offense: PLAYER_OFFENSE,
        move: { id: move.id, ...move.modifiers },
        point: contact.point,
        normal: contact.normal,
        direction: contact.travel,
      };
      this.resolve(self, contact.target, hit);
    }
  }

  dispose(): void {
    this.#detachAnimation?.();
    this.#detachAnimation = null;
    this.#registry.clear();
    this.#ctx?.services.unregister(CombatantsKey);
    this.#ctx?.services.unregister(CombatKey);
    this.#ctx = null;
  }

  /* -- binding ------------------------------------------------------------ */

  /**
   * Attach to the player the first frame it exists.
   *
   * Lazy because this module is registered *before* `PlayerController` (so that
   * the `combat` service exists when the player's `init` looks for it), which
   * means the player's model has not loaded yet at our own `init`.
   */
  #bind(ctx: GameContext): void {
    if (this.#self !== null) return;
    const player = ctx.services.tryGet<PlayerController>(PlayerKey);
    if (player === undefined || player.object === null) return;

    this.#player = player;
    this.#self = new PlayerCombatant(this.#registry.nextId(), this, player, this.#vitals);
    this.#registry.add(this.#self);
    this.#spawn.copy(player.position);

    const root = player.object;
    const anchors: Array<[WeaponSlot, WeaponAnchor | null]> = [
      ['1h', resolveWeaponAnchor(root, { meshNames: ['1H_Axe'], reach: 0.62 })],
      ['2h', resolveWeaponAnchor(root, { meshNames: ['2H_Axe', '1H_Axe'], reach: 0.85 })],
      ['limb', resolveWeaponAnchor(root, { reach: 0.5 })],
    ];
    for (const [slot, anchor] of anchors) {
      if (anchor === null) continue;
      this.#hitboxes.set(
        slot,
        new WeaponHitbox(anchor, { radius: slot === '2h' ? 0.2 : 0.16, overreach: 0.12 }),
      );
    }
    if (this.#hitboxes.size === 0) {
      console.warn('[combat] the player model has no usable weapon anchor; melee will never hit');
    }

    const graph = player.animation;
    if (graph !== null) {
      this.#detachAnimation = graph.onAnimationEvent((event) => this.#onAnimationEvent(event));
    }
    this.#emitVitals();
    ctx.events.emit('combat:vitals', this.#vitalsPayload());
  }

  /* -- input -------------------------------------------------------------- */

  /**
   * Read the rising edges once per rendered frame.
   *
   * `Input` clears its edge sets at the end of every *frame*, but `fixedUpdate`
   * can run zero times in a frame — which is exactly what happens during hit
   * stop, when the time scale is a fraction and the accumulator never fills. A
   * press during a hit stop would simply vanish. So this is called from both
   * `fixedUpdate` and (via `#bind`) the first fixed step of the frame, guarded
   * by the frame counter so a frame with three fixed steps still queues one
   * press.
   */
  #capturePresses(ctx: GameContext): void {
    if (this.#inputFrame === ctx.time.frame) return;
    this.#inputFrame = ctx.time.frame;
    if (this.#self === null || !this.#self.alive) return;
    if (ctx.input.wasPressed('Attack')) this.#combo.press('light');
    if (ctx.input.wasPressed('HeavyAttack')) this.#combo.press('heavy');
  }

  update(ctx: GameContext): void {
    // Safety net for frames with no fixed step (hit stop, long frames).
    this.#capturePresses(ctx);
  }

  #updateBlock(): void {
    const ctx = this.#ctx;
    const player = this.#player;
    if (ctx === null || player === null) return;

    const wants = ctx.input.isDown('Block') && this.#combo.phase === 'idle';
    if (wants === this.#blocking) return;
    this.#blocking = wants;

    const graph = player.animation;
    if (graph === null) return;
    if (wants) {
      this.#combo.clearBuffer();
      graph.playAction('block', { layer: 'upper', loop: true, fadeIn: 0.1 });
      this.#blockHandleActive = true;
    } else if (this.#blockHandleActive) {
      graph.cancelActions('upper', 0.14);
      this.#blockHandleActive = false;
    }
  }

  /* -- moves -------------------------------------------------------------- */

  #startMove(ctx: GameContext, next: MeleeMove): void {
    const player = this.#player;
    const graph = player?.animation ?? null;
    if (player === null || graph === null) {
      this.#combo.interrupt();
      return;
    }

    if (next.stamina > 0 && !player.consumeStamina(next.stamina)) {
      // Refused, not clipped: a swing that plays with no damage because the
      // pool ran dry mid-animation is worse than a swing that never starts.
      this.#combo.interrupt();
      return;
    }

    if (this.#blocking) {
      this.#blocking = false;
      this.#blockHandleActive = false;
    }

    const handle = graph.playAction(next.action, {
      layer: next.layer,
      speed: next.speed,
      fadeIn: next.layer === 'upper' ? 0.06 : 0.1,
      events: [
        { name: 'hit.open', at: next.window[0] },
        { name: 'hit.close', at: next.window[1] },
      ],
    });
    // The clip is played at `speed`, so the wall-clock duration the combo
    // machine must reason about is the clip length divided by that multiplier.
    this.#combo.setDuration(handle.duration / Math.max(0.05, next.speed));

    this.#activeMove = next;
    this.#activeHitbox = this.#hitboxes.get(next.weapon) ?? this.#hitboxes.get('limb') ?? null;
    this.#activeHitbox?.beginSwing();

    ctx.events.emit('combat:swing', {
      attacker: this.#self?.id ?? 0,
      faction: 'player',
      moveId: next.id,
    });
  }

  #endMove(): void {
    this.#activeHitbox?.cancel();
    this.#activeHitbox = null;
    this.#activeMove = null;
  }

  #onAnimationEvent(event: { action: string; name: string }): void {
    const move = this.#activeMove;
    if (move === null || event.action !== move.action) return;
    if (event.name === 'hit.open') this.#activeHitbox?.openWindow();
    else if (event.name === 'hit.close') this.#activeHitbox?.closeWindow();
  }

  /* -- death and respawn --------------------------------------------------- */

  #respawn(ctx: GameContext): void {
    const player = this.#player;
    this.#vitals.revive();
    this.#respawnTimer = 0;
    if (player !== null) {
      player.teleport(this.#spawn.x, this.#spawn.y, this.#spawn.z);
      player.setEnabled(true);
      player.animation?.cancelActions(undefined, 0.2);
    }
    this.#emitVitals();
    ctx.events.emit('combat:respawn', { position: this.#spawn.clone() });
  }

  /* -- helpers ------------------------------------------------------------- */

  #facingTowards(self: Combatant, point: THREE.Vector3): number {
    self.facing(this.#scratchA);
    self.footPosition(this.#scratchB);
    this.#scratchB.subVectors(point, this.#scratchB).setY(0);
    if (this.#scratchB.lengthSq() < 1e-8) return 1;
    return this.#scratchA.dot(this.#scratchB.normalize());
  }

  #vitalsPayload(): {
    health: number;
    healthMax: number;
    mana: number;
    manaMax: number;
    stamina: number;
    staminaMax: number;
  } {
    return {
      health: this.#vitals.health.value,
      healthMax: this.#vitals.health.max,
      mana: this.#vitals.mana.value,
      manaMax: this.#vitals.mana.max,
      stamina: this.stamina,
      staminaMax: this.staminaMax,
    };
  }

  #emitVitals(): void {
    this.#ctx?.events.emit('combat:vitals', this.#vitalsPayload());
  }
}
