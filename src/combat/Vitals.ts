/**
 * @module combat/Vitals
 *
 * Health, mana and stamina: three pools with the same shape and different
 * regeneration behaviour, plus the death bookkeeping that hangs off health.
 *
 * The one design decision worth stating is the **regeneration delay**. A pool
 * that starts refilling the instant it is touched removes all tension from
 * resource management, because there is no window in which the player is
 * actually short of anything. Every pool here therefore stalls for
 * `regenDelay` seconds after being spent or damaged, and only then ramps. That
 * single number is what makes stamina a decision rather than a formality.
 *
 * Pure and dependency-free, so the numbers can be tested without an engine.
 */

export interface PoolOptions {
  readonly max: number;
  /** Starting value. Defaults to `max`. */
  readonly value?: number;
  /** Units restored per second once regeneration is running. Default 0. */
  readonly regen?: number;
  /** Seconds of stillness required before regeneration resumes. Default 0. */
  readonly regenDelay?: number;
  /**
   * Seconds for regeneration to reach full rate after the delay expires.
   * A hard start on a fast pool is visible as a jump in the bar. Default 0.35.
   */
  readonly regenRamp?: number;
}

export class ResourcePool {
  #max: number;
  #value: number;
  #regen: number;
  #delay: number;
  #ramp: number;
  #idle = Infinity;

  constructor(options: PoolOptions) {
    this.#max = Math.max(0, options.max);
    this.#value = Math.min(this.#max, Math.max(0, options.value ?? options.max));
    this.#regen = Math.max(0, options.regen ?? 0);
    this.#delay = Math.max(0, options.regenDelay ?? 0);
    this.#ramp = Math.max(0, options.regenRamp ?? 0.35);
  }

  get value(): number {
    return this.#value;
  }

  get max(): number {
    return this.#max;
  }

  /** `[0, 1]`. An empty maximum reads as empty, never as `NaN`. */
  get fraction(): number {
    return this.#max <= 0 ? 0 : this.#value / this.#max;
  }

  get empty(): boolean {
    return this.#value <= 0;
  }

  get full(): boolean {
    return this.#value >= this.#max;
  }

  /** Seconds since the pool was last spent or damaged. */
  get idleTime(): number {
    return this.#idle;
  }

  /** Raise or lower the ceiling, keeping the current fraction. */
  setMax(max: number, keepFraction = true): void {
    const next = Math.max(0, max);
    const fraction = this.fraction;
    this.#max = next;
    this.#value = keepFraction ? next * fraction : Math.min(this.#value, next);
  }

  setRegen(perSecond: number): void {
    this.#regen = Math.max(0, perSecond);
  }

  /** Remove up to `amount`. @returns how much was actually removed. */
  drain(amount: number): number {
    if (amount <= 0) return 0;
    const removed = Math.min(this.#value, amount);
    this.#value -= removed;
    this.#idle = 0;
    return removed;
  }

  /** Restore up to `amount`. @returns how much was actually restored. */
  restore(amount: number): number {
    if (amount <= 0) return 0;
    const added = Math.min(this.#max - this.#value, amount);
    this.#value += added;
    return added;
  }

  /**
   * All-or-nothing withdrawal.
   *
   * The contract combat wants: a dodge that half-executes because stamina ran
   * out mid-animation is worse than a dodge that refuses to start.
   */
  spend(amount: number): boolean {
    if (amount <= 0) return true;
    if (this.#value < amount) return false;
    this.#value -= amount;
    this.#idle = 0;
    return true;
  }

  /** Restart the regeneration delay without changing the value. */
  interrupt(): void {
    this.#idle = 0;
  }

  /** Refill and clear the delay. Used on respawn. */
  refill(): void {
    this.#value = this.#max;
    this.#idle = Infinity;
  }

  /** Empty the pool immediately. */
  deplete(): void {
    this.#value = 0;
    this.#idle = 0;
  }

  update(dt: number): void {
    if (dt <= 0) return;
    this.#idle += dt;
    if (this.#regen <= 0 || this.#idle < this.#delay) return;
    const since = this.#idle - this.#delay;
    const ramp = this.#ramp <= 0 ? 1 : Math.min(1, since / this.#ramp);
    this.restore(this.#regen * ramp * dt);
  }
}

export interface VitalsOptions {
  readonly health: PoolOptions;
  readonly mana: PoolOptions;
  readonly stamina: PoolOptions;
}

/** Diablo II-ish defaults for a level 1 Barbarian. */
export const DEFAULT_VITALS: VitalsOptions = {
  health: { max: 120, regen: 1.6, regenDelay: 4.5, regenRamp: 1.2 },
  mana: { max: 40, regen: 1.1, regenDelay: 1.5 },
  stamina: { max: 100, regen: 22, regenDelay: 0.8, regenRamp: 0.4 },
};

/**
 * The three pools plus death state.
 *
 * `alive` is latched rather than derived from `health > 0`, so that a corpse
 * whose health pool is later refilled for a respawn does not spontaneously come
 * back to life halfway through its death animation.
 */
export class Vitals {
  readonly health: ResourcePool;
  readonly mana: ResourcePool;
  readonly stamina: ResourcePool;

  #alive = true;
  #timeOfDeath = 0;
  #clock = 0;

  constructor(options: VitalsOptions = DEFAULT_VITALS) {
    this.health = new ResourcePool(options.health);
    this.mana = new ResourcePool(options.mana);
    this.stamina = new ResourcePool(options.stamina);
  }

  get alive(): boolean {
    return this.#alive;
  }

  /** Seconds since death, or 0 while alive. */
  get timeDead(): number {
    return this.#alive ? 0 : this.#clock - this.#timeOfDeath;
  }

  /**
   * Apply damage.
   *
   * @returns `{ removed, killed }` — `killed` is true only on the transition,
   * so the caller can fire a death animation exactly once no matter how many
   * hits land in the same frame.
   */
  applyDamage(amount: number): { removed: number; killed: boolean } {
    if (!this.#alive || amount <= 0) return { removed: 0, killed: false };
    const removed = this.health.drain(amount);
    // Damage should also stall stamina recovery: being hit is not resting.
    this.stamina.interrupt();
    if (this.health.empty) {
      this.#alive = false;
      this.#timeOfDeath = this.#clock;
      return { removed, killed: true };
    }
    return { removed, killed: false };
  }

  /** Kill outright, bypassing damage. */
  kill(): boolean {
    if (!this.#alive) return false;
    this.health.deplete();
    this.#alive = false;
    this.#timeOfDeath = this.#clock;
    return true;
  }

  /** Full reset. Used on respawn. */
  revive(): void {
    this.health.refill();
    this.mana.refill();
    this.stamina.refill();
    this.#alive = true;
  }

  update(dt: number): void {
    if (dt > 0) this.#clock += dt;
    if (!this.#alive) return;
    this.health.update(dt);
    this.mana.update(dt);
    this.stamina.update(dt);
  }
}
