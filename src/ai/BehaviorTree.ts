/**
 * @module ai/BehaviorTree
 *
 * A compact behaviour tree: composites, decorators, leaves, and the memory
 * semantics that make the difference between a tree and a nested `if`.
 *
 * ### The semantics that matter, stated once
 *
 * A node returns `success`, `failure` or `running`. `running` means "I am
 * mid-action, tick me again next frame", and every composite in this file is a
 * **memory** composite: when a child returns `running`, the composite records
 * which child it was and resumes there on the next tick instead of re-testing
 * its earlier children.
 *
 * That choice is the whole design. Without memory, a `Sequence` of
 * `[Telegraph, Swing]` re-evaluates `Telegraph` every frame and the skeleton
 * winds up forever. With memory, an enemy that has committed to an attack stays
 * committed — which is exactly the property a player needs in order to read a
 * telegraph and dodge it. Re-planning is then an explicit act:
 * {@link Node.reset} clears the memory of a whole subtree, and the enemy calls
 * it when something genuinely changes (it took a hit, it lost its target).
 *
 * There is no blackboard type baked in: the tree is generic over it, so the
 * skeleton's blackboard is a plain interface in `enemies/Skeleton.ts` and the
 * type checker catches a condition that reads a field nobody writes.
 */

export type Status = 'success' | 'failure' | 'running';

/** What every node is handed on tick. */
export interface TickContext<B> {
  readonly blackboard: B;
  /** Seconds since the previous tick of this tree. */
  readonly dt: number;
  /** Seconds since the tree started running. Monotonic. */
  readonly time: number;
}

export abstract class Node<B> {
  /** Diagnostic name. Shows up in {@link BehaviorTree.trace}. */
  readonly name: string;

  constructor(name: string) {
    this.name = name;
  }

  abstract tick(ctx: TickContext<B>): Status;

  /** Clear this node's memory. Composites must propagate to their children. */
  reset(): void {
    // Leaves with no state need no implementation.
  }
}

/* -------------------------------------------------------------------------- */
/* Composites                                                                  */
/* -------------------------------------------------------------------------- */

abstract class Composite<B> extends Node<B> {
  readonly children: readonly Node<B>[];
  protected index = 0;

  constructor(name: string, children: readonly Node<B>[]) {
    super(name);
    this.children = children;
  }

  override reset(): void {
    this.index = 0;
    for (const child of this.children) child.reset();
  }
}

/**
 * Run children in order until one fails.
 *
 * `success` when every child succeeded; `failure` the moment one fails (and the
 * whole sequence resets, so the next tick starts from the top); `running` while
 * a child is running, resuming at that child next tick.
 *
 * An empty sequence succeeds — the vacuous truth is the useful default, because
 * it makes a sequence built from a filtered list of conditions degrade to "no
 * conditions, go ahead" rather than to a dead branch.
 */
export class Sequence<B> extends Composite<B> {
  constructor(children: readonly Node<B>[], name = 'sequence') {
    super(name, children);
  }

  tick(ctx: TickContext<B>): Status {
    while (this.index < this.children.length) {
      const child = this.children[this.index];
      if (child === undefined) break;
      const status = child.tick(ctx);
      if (status === 'running') return 'running';
      if (status === 'failure') {
        this.reset();
        return 'failure';
      }
      this.index++;
    }
    this.reset();
    return 'success';
  }
}

/**
 * Run children in order until one succeeds. The classic priority list.
 *
 * `failure` only when every child failed. An empty selector fails, which is the
 * mirror of the empty sequence and for the same reason.
 */
export class Selector<B> extends Composite<B> {
  constructor(children: readonly Node<B>[], name = 'selector') {
    super(name, children);
  }

  tick(ctx: TickContext<B>): Status {
    while (this.index < this.children.length) {
      const child = this.children[this.index];
      if (child === undefined) break;
      const status = child.tick(ctx);
      if (status === 'running') return 'running';
      if (status === 'success') {
        this.reset();
        return 'success';
      }
      this.index++;
    }
    this.reset();
    return 'failure';
  }
}

/**
 * A selector that re-evaluates from the top every tick, abandoning a running
 * lower-priority child when a higher-priority one becomes viable.
 *
 * This is how "stop chasing, you are being hit" works: stagger sits above chase
 * in the priority list, and the moment its condition passes it must take over
 * from a chase that is happily reporting `running`. The abandoned subtree is
 * reset so it does not resume mid-action later.
 */
export class PrioritySelector<B> extends Composite<B> {
  #running = -1;

  constructor(children: readonly Node<B>[], name = 'priority') {
    super(name, children);
  }

  tick(ctx: TickContext<B>): Status {
    for (let i = 0; i < this.children.length; i++) {
      const child = this.children[i];
      if (child === undefined) continue;
      const status = child.tick(ctx);
      if (status === 'failure') {
        // A child that was running and has now failed its own preconditions
        // must not keep stale memory around, or it resumes mid-action the next
        // time it is reached.
        if (this.#running === i) {
          this.#running = -1;
          child.reset();
        }
        continue;
      }
      if (this.#running !== -1 && this.#running !== i) {
        const previous = this.children[this.#running];
        previous?.reset();
      }
      this.#running = status === 'running' ? i : -1;
      if (status === 'running') return 'running';
      return 'success';
    }
    if (this.#running !== -1) {
      this.children[this.#running]?.reset();
      this.#running = -1;
    }
    return 'failure';
  }

  override reset(): void {
    this.#running = -1;
    super.reset();
  }
}

/* -------------------------------------------------------------------------- */
/* Decorators                                                                  */
/* -------------------------------------------------------------------------- */

abstract class Decorator<B> extends Node<B> {
  readonly child: Node<B>;

  constructor(name: string, child: Node<B>) {
    super(name);
    this.child = child;
  }

  override reset(): void {
    this.child.reset();
  }
}

/** Swap success and failure. `running` passes through untouched. */
export class Inverter<B> extends Decorator<B> {
  constructor(child: Node<B>, name = 'inverter') {
    super(name, child);
  }

  tick(ctx: TickContext<B>): Status {
    const status = this.child.tick(ctx);
    if (status === 'success') return 'failure';
    if (status === 'failure') return 'success';
    return 'running';
  }
}

/** Always report success once the child settles. Turns an optional step optional. */
export class Succeeder<B> extends Decorator<B> {
  constructor(child: Node<B>, name = 'succeeder') {
    super(name, child);
  }

  tick(ctx: TickContext<B>): Status {
    const status = this.child.tick(ctx);
    return status === 'running' ? 'running' : 'success';
  }
}

/**
 * Refuse to run the child again until `seconds` have passed since it last
 * *finished*.
 *
 * The clock starts when the child settles, not when it starts, so an attack
 * with a 2 s cooldown means two seconds of breathing room after the swing — the
 * form of the rule the designer means, and not the one the naive version gives.
 */
export class Cooldown<B> extends Decorator<B> {
  readonly #seconds: number;
  #remaining = 0;

  constructor(seconds: number, child: Node<B>, name = 'cooldown') {
    super(name, child);
    this.#seconds = Math.max(0, seconds);
  }

  /** Seconds left before the child may run again. */
  get remaining(): number {
    return this.#remaining;
  }

  tick(ctx: TickContext<B>): Status {
    if (this.#remaining > 0) {
      this.#remaining = Math.max(0, this.#remaining - ctx.dt);
      return 'failure';
    }
    const status = this.child.tick(ctx);
    if (status !== 'running') this.#remaining = this.#seconds;
    return status;
  }

  /** Start the cooldown without running the child. */
  trigger(): void {
    this.#remaining = this.#seconds;
  }

  override reset(): void {
    // The cooldown itself deliberately survives a reset: an enemy that is
    // re-planning must not get a free attack out of it.
    this.child.reset();
  }
}

/** Repeat the child until it fails, then succeed. Bounded to avoid a spin. */
export class UntilFail<B> extends Decorator<B> {
  readonly #maxIterations: number;

  constructor(child: Node<B>, maxIterations = 32, name = 'untilFail') {
    super(name, child);
    this.#maxIterations = Math.max(1, maxIterations);
  }

  tick(ctx: TickContext<B>): Status {
    for (let i = 0; i < this.#maxIterations; i++) {
      const status = this.child.tick(ctx);
      if (status === 'running') return 'running';
      if (status === 'failure') return 'success';
    }
    return 'success';
  }
}

/* -------------------------------------------------------------------------- */
/* Leaves                                                                      */
/* -------------------------------------------------------------------------- */

/** A stateless predicate. */
export class Condition<B> extends Node<B> {
  readonly #predicate: (ctx: TickContext<B>) => boolean;

  constructor(name: string, predicate: (ctx: TickContext<B>) => boolean) {
    super(name);
    this.#predicate = predicate;
  }

  tick(ctx: TickContext<B>): Status {
    return this.#predicate(ctx) ? 'success' : 'failure';
  }
}

/**
 * A leaf that does something and reports how it went.
 *
 * `onReset` exists so an action that grabbed a resource (an animation slot, a
 * reserved approach angle) can hand it back when a higher-priority branch takes
 * over mid-action.
 */
export class Action<B> extends Node<B> {
  readonly #run: (ctx: TickContext<B>) => Status;
  readonly #onReset: (() => void) | undefined;

  constructor(
    name: string,
    run: (ctx: TickContext<B>) => Status,
    onReset?: () => void,
  ) {
    super(name);
    this.#run = run;
    this.#onReset = onReset;
  }

  tick(ctx: TickContext<B>): Status {
    return this.#run(ctx);
  }

  override reset(): void {
    this.#onReset?.();
  }
}

/** Report `running` for `seconds`, then `success`. The pause between beats. */
export class Wait<B> extends Node<B> {
  readonly #seconds: number;
  #elapsed = 0;

  constructor(seconds: number, name = 'wait') {
    super(name);
    this.#seconds = Math.max(0, seconds);
  }

  get elapsed(): number {
    return this.#elapsed;
  }

  tick(ctx: TickContext<B>): Status {
    this.#elapsed += ctx.dt;
    if (this.#elapsed >= this.#seconds) {
      this.#elapsed = 0;
      return 'success';
    }
    return 'running';
  }

  override reset(): void {
    this.#elapsed = 0;
  }
}

/* -------------------------------------------------------------------------- */
/* The tree                                                                    */
/* -------------------------------------------------------------------------- */

/**
 * Root wrapper: owns the clock and the last status.
 *
 * Trivial, but it means an enemy holds one object rather than a node plus two
 * loose numbers it has to remember to advance.
 */
export class BehaviorTree<B> {
  readonly root: Node<B>;
  #time = 0;
  #status: Status = 'failure';

  constructor(root: Node<B>) {
    this.root = root;
  }

  get time(): number {
    return this.#time;
  }

  /** Status of the most recent tick. */
  get status(): Status {
    return this.#status;
  }

  tick(blackboard: B, dt: number): Status {
    this.#time += Math.max(0, dt);
    this.#status = this.root.tick({ blackboard, dt, time: this.#time });
    return this.#status;
  }

  reset(): void {
    this.root.reset();
    this.#status = 'failure';
  }
}

/* -------------------------------------------------------------------------- */
/* Sugar                                                                       */
/* -------------------------------------------------------------------------- */

export const bt = {
  sequence: <B>(name: string, ...children: Node<B>[]): Sequence<B> => new Sequence(children, name),
  selector: <B>(name: string, ...children: Node<B>[]): Selector<B> => new Selector(children, name),
  priority: <B>(name: string, ...children: Node<B>[]): PrioritySelector<B> =>
    new PrioritySelector(children, name),
  condition: <B>(name: string, predicate: (ctx: TickContext<B>) => boolean): Condition<B> =>
    new Condition(name, predicate),
  action: <B>(
    name: string,
    run: (ctx: TickContext<B>) => Status,
    onReset?: () => void,
  ): Action<B> => new Action(name, run, onReset),
  wait: <B>(seconds: number, name?: string): Wait<B> => new Wait(seconds, name),
  invert: <B>(child: Node<B>, name?: string): Inverter<B> => new Inverter(child, name),
  succeed: <B>(child: Node<B>, name?: string): Succeeder<B> => new Succeeder(child, name),
  cooldown: <B>(seconds: number, child: Node<B>, name?: string): Cooldown<B> =>
    new Cooldown(seconds, child, name),
};
