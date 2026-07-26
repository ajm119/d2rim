/**
 * @module core/Time
 *
 * Frame timing primitives, kept free of any DOM dependency so the fixed-step
 * logic can be unit tested directly.
 *
 * The engine runs the standard "fix your timestep" arrangement: simulation
 * advances in constant 60 Hz slices while rendering happens at whatever rate
 * the display allows, and {@link FixedStepAccumulator.alpha} carries the
 * leftover fraction so renderers can interpolate between the last two
 * simulation states instead of visibly stuttering.
 */

import type { TimeState } from './types';

/** Create a zeroed {@link TimeState}. */
export function createTimeState(): TimeState {
  return { elapsed: 0, delta: 0, frame: 0, scale: 1 };
}

export interface FixedStepOptions {
  /** Simulation rate in hertz. Default 60. */
  hz?: number;
  /**
   * Maximum simulation slices consumed in a single frame. Default 5.
   *
   * This is the spiral-of-death guard: if a frame takes longer than the
   * simulation it triggers, running "just one more step" makes the next frame
   * later still, forever. Past this cap the backlog is discarded and simulation
   * time deliberately falls behind wall-clock time.
   */
  maxSubSteps?: number;
  /**
   * Largest raw frame delta accepted, in seconds. Default 0.25.
   *
   * Backgrounded tabs, breakpoints and GC pauses produce multi-second deltas.
   * Clamping keeps a resumed tab from teleporting every physics body.
   */
  maxDelta?: number;
}

/**
 * Fixed-timestep accumulator with interpolation.
 *
 * Usage per frame:
 * ```ts
 * const steps = acc.begin(rawDeltaSeconds);
 * for (let i = 0; i < steps; i++) fixedUpdate(acc.fixedDelta);
 * render(acc.alpha);
 * ```
 */
export class FixedStepAccumulator {
  /** Constant simulation slice in seconds (e.g. 1/60). */
  readonly fixedDelta: number;
  readonly maxSubSteps: number;
  readonly maxDelta: number;

  #accumulator = 0;
  #alpha = 0;
  #clamped = false;
  #starved = false;

  constructor(options: FixedStepOptions = {}) {
    const hz = options.hz ?? 60;
    if (!Number.isFinite(hz) || hz <= 0) {
      throw new RangeError(`[FixedStepAccumulator] hz must be > 0, got ${hz}`);
    }
    this.fixedDelta = 1 / hz;
    this.maxSubSteps = Math.max(1, Math.floor(options.maxSubSteps ?? 5));
    this.maxDelta = options.maxDelta ?? 0.25;
  }

  /** Unconsumed simulation time, in seconds. Always `< fixedDelta` after {@link begin}. */
  get pending(): number {
    return this.#accumulator;
  }

  /**
   * Interpolation factor in `[0, 1)`: how far the render frame sits between the
   * last completed simulation step and the next one.
   */
  get alpha(): number {
    return this.#alpha;
  }

  /** Whether the last {@link begin} clamped its input against {@link maxDelta}. */
  get wasClamped(): boolean {
    return this.#clamped;
  }

  /** Whether the last {@link begin} hit {@link maxSubSteps} and dropped backlog. */
  get wasStarved(): boolean {
    return this.#starved;
  }

  /**
   * Accumulate one frame's worth of time and report how many fixed steps the
   * caller should now run.
   *
   * @param rawDelta unscaled seconds since the previous frame. Negative,
   *   `NaN` and infinite inputs are treated as zero rather than throwing —
   *   clock sources genuinely produce these across tab suspension.
   * @param scale time multiplier applied after clamping, so slow-motion cannot
   *   be used to smuggle in an unbounded step count.
   */
  begin(rawDelta: number, scale = 1): number {
    this.#clamped = false;
    this.#starved = false;

    let delta = Number.isFinite(rawDelta) && rawDelta > 0 ? rawDelta : 0;
    if (delta > this.maxDelta) {
      delta = this.maxDelta;
      this.#clamped = true;
    }

    this.#accumulator += delta * (Number.isFinite(scale) ? Math.max(0, scale) : 0);

    let steps = 0;
    while (this.#accumulator >= this.fixedDelta && steps < this.maxSubSteps) {
      this.#accumulator -= this.fixedDelta;
      steps++;
    }

    if (steps === this.maxSubSteps && this.#accumulator >= this.fixedDelta) {
      // Backlog we will never catch up on. Drop it so the next frame starts
      // clean instead of compounding.
      this.#accumulator = 0;
      this.#starved = true;
    }

    this.#alpha = this.#accumulator / this.fixedDelta;
    return steps;
  }

  /** Discard pending time. Use after a deliberate discontinuity (level load). */
  reset(): void {
    this.#accumulator = 0;
    this.#alpha = 0;
    this.#clamped = false;
    this.#starved = false;
  }
}

/**
 * Wall-clock source for the engine, wrapping a monotonic time function.
 *
 * The time source is injectable so headless runs and tests can drive frames
 * from a virtual clock rather than `performance.now()`.
 */
export class Clock {
  readonly #now: () => number;
  #last: number;

  constructor(now: () => number = () => performance.now()) {
    this.#now = now;
    this.#last = now();
  }

  /** Seconds elapsed since the previous call. The first call returns 0. */
  tick(): number {
    const t = this.#now();
    const delta = (t - this.#last) / 1000;
    this.#last = t;
    return Number.isFinite(delta) && delta > 0 ? delta : 0;
  }

  /** Re-anchor to now, discarding time spent paused. */
  reset(): void {
    this.#last = this.#now();
  }
}
