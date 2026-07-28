/**
 * @module core/EventBus
 *
 * Strongly typed publish/subscribe used for cross-module communication without
 * import cycles.
 *
 * The event catalogue lives in the {@link GameEvents} interface. Later modules
 * extend it by declaration merging rather than by editing this file:
 *
 * ```ts
 * declare module '../core/EventBus' {
 *   interface GameEvents {
 *     'combat:hit': { attacker: number; target: number; damage: number };
 *   }
 * }
 * ```
 *
 * After that augmentation, `events.emit('combat:hit', ...)` type-checks its
 * payload and `events.on('combat:hit', ...)` infers the handler argument.
 */

import type { RendererBackend } from './types';

/**
 * The stages a session passes through before the first frame.
 *
 * Ordered by when they happen, and each is separately worth naming because each
 * can be the one that stalls: `renderer` blocks on WebGPU device acquisition,
 * `modules` covers asset download and shader compilation, and `zone` is terrain
 * and collider construction.
 */
export type BootPhase = 'renderer' | 'modules' | 'zone' | 'ready';

/**
 * The event catalogue: event name -> payload type.
 *
 * Names are namespaced `domain:verb`. Only genuinely engine-level events belong
 * here; feature events are added by the feature's own module via declaration
 * merging so this file never becomes a dependency hub.
 */
export interface GameEvents {
  /**
   * Boot progress, for the loading screen.
   *
   * `completed`/`total` are 0 for phases with no meaningful item count, in which
   * case only `label` is worth showing.
   */
  'boot:phase': {
    phase: BootPhase;
    /** Human-readable description of what is happening right now. */
    label: string;
    completed: number;
    total: number;
  };
  /** Boot failed unrecoverably. The loading screen shows this instead of hanging. */
  'boot:failed': { phase: BootPhase; message: string };
  /** Fired once, after the renderer exists and every module has initialised. */
  'engine:ready': { backend: RendererBackend };
  /**
   * Pipeline warmup finished. `compiled` is how many render pipelines were
   * built during the loading screen rather than inside a frame.
   */
  'engine:warmup': { millis: number; compiled: number; programs: number };
  /** Fired on every accepted resize, after the renderer has been resized. */
  'engine:resize': { width: number; height: number; pixelRatio: number };
  /** Fired when the loop pauses or resumes (tab visibility, manual control). */
  'engine:pause': { paused: boolean };
  /**
   * Fired at the end of every stepped frame, after `lateUpdate`.
   *
   * `dt` is the clamped, time-scaled delta the frame simulated with — what
   * gameplay saw. `rawDt` is what the wall clock actually reported, unclamped
   * and unscaled — what the machine actually did. They differ by a factor of
   * three on an overloaded frame, and anything reporting performance must use
   * `rawDt`. `frameMs` is the engine's own end-to-end cost for the frame.
   */
  'engine:frame': { frame: number; dt: number; rawDt: number; frameMs: number };
  /** Fired when a module is registered with the engine. */
  'module:added': { name: string };
  /** Fired when a module is removed and disposed. */
  'module:removed': { name: string };
}

/** Any key present in the (possibly augmented) {@link GameEvents} catalogue. */
export type GameEventName = keyof GameEvents;

/** Handler signature for a given event name. */
export type EventHandler<K extends GameEventName> = (payload: GameEvents[K]) => void;

/** Returned by {@link EventBus.on}; calling it removes that subscription. */
export type Unsubscribe = () => void;

export class EventBus {
  /**
   * Handlers are stored in insertion-ordered `Set`s so that duplicate
   * registration of the same function is idempotent and removal is O(1).
   */
  readonly #handlers = new Map<GameEventName, Set<EventHandler<GameEventName>>>();

  /**
   * Depth of nested `emit` calls. While non-zero, mutations to a handler set
   * that is currently being iterated would be unsafe, so `emit` iterates a
   * snapshot instead (see below).
   */
  #emitDepth = 0;

  /**
   * Subscribe to `event`.
   *
   * Handlers added during an in-flight `emit` of the same event are *not*
   * invoked by that emit; handlers removed during it are not invoked either.
   *
   * @returns a disposer. Idempotent — calling it twice is harmless.
   */
  on<K extends GameEventName>(event: K, cb: EventHandler<K>): Unsubscribe {
    let set = this.#handlers.get(event);
    if (set === undefined) {
      set = new Set();
      this.#handlers.set(event, set);
    }
    set.add(cb as EventHandler<GameEventName>);

    let active = true;
    return () => {
      if (!active) return;
      active = false;
      this.off(event, cb);
    };
  }

  /** Subscribe for exactly one delivery, then auto-unsubscribe. */
  once<K extends GameEventName>(event: K, cb: EventHandler<K>): Unsubscribe {
    const dispose = this.on(event, ((payload: GameEvents[K]) => {
      dispose();
      cb(payload);
    }) as EventHandler<K>);
    return dispose;
  }

  /** Remove a specific handler. No-op if it was never registered. */
  off<K extends GameEventName>(event: K, cb: EventHandler<K>): void {
    const set = this.#handlers.get(event);
    if (set === undefined) return;
    set.delete(cb as EventHandler<GameEventName>);
    if (set.size === 0 && this.#emitDepth === 0) this.#handlers.delete(event);
  }

  /**
   * Synchronously deliver `payload` to every current subscriber.
   *
   * A handler that throws must not prevent the remaining handlers from running,
   * so errors are caught, reported, and swallowed. Silent failure would be
   * worse: a listener throwing in one system would stall unrelated systems.
   */
  emit<K extends GameEventName>(event: K, payload: GameEvents[K]): void {
    const set = this.#handlers.get(event);
    if (set === undefined || set.size === 0) return;

    // Snapshot so that subscribe/unsubscribe from inside a handler cannot
    // invalidate this iteration.
    const snapshot = Array.from(set) as EventHandler<K>[];
    this.#emitDepth++;
    try {
      for (const handler of snapshot) {
        // Skip handlers removed by an earlier handler in this same emit.
        if (!set.has(handler as EventHandler<GameEventName>)) continue;
        try {
          handler(payload);
        } catch (error) {
          console.error(`[EventBus] handler for "${String(event)}" threw:`, error);
        }
      }
    } finally {
      this.#emitDepth--;
    }
  }

  /** Number of handlers currently subscribed to `event`. */
  listenerCount(event: GameEventName): number {
    return this.#handlers.get(event)?.size ?? 0;
  }

  /** Remove every handler for `event`, or all handlers when `event` is omitted. */
  clear(event?: GameEventName): void {
    if (event === undefined) this.#handlers.clear();
    else this.#handlers.delete(event);
  }
}
