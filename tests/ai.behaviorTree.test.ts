import { describe, expect, it } from 'vitest';

import {
  Action,
  BehaviorTree,
  Condition,
  Cooldown,
  Inverter,
  Node,
  PrioritySelector,
  Selector,
  Sequence,
  Succeeder,
  UntilFail,
  Wait,
  bt,
  type Status,
  type TickContext,
} from '../src/ai/BehaviorTree';

interface Board {
  log: string[];
  flag: boolean;
}

function board(): Board {
  return { log: [], flag: false };
}

function ctx(blackboard: Board, dt = 1 / 60, time = 0): TickContext<Board> {
  return { blackboard, dt, time };
}

/** A leaf that returns a scripted sequence of statuses and logs every tick. */
class Scripted extends Node<Board> {
  #index = 0;
  #resets = 0;

  constructor(
    name: string,
    private readonly script: readonly Status[],
  ) {
    super(name);
  }

  get resets(): number {
    return this.#resets;
  }

  tick(tick: TickContext<Board>): Status {
    tick.blackboard.log.push(this.name);
    const status = this.script[Math.min(this.#index, this.script.length - 1)] ?? 'success';
    this.#index++;
    return status;
  }

  override reset(): void {
    this.#resets++;
    this.#index = 0;
  }
}

describe('Sequence', () => {
  it('succeeds when every child succeeds, in order', () => {
    const b = board();
    const node = new Sequence<Board>([
      new Scripted('a', ['success']),
      new Scripted('b', ['success']),
    ]);
    expect(node.tick(ctx(b))).toBe('success');
    expect(b.log).toEqual(['a', 'b']);
  });

  it('fails on the first failure and does not tick later children', () => {
    const b = board();
    const later = new Scripted('b', ['success']);
    const node = new Sequence<Board>([new Scripted('a', ['failure']), later]);
    expect(node.tick(ctx(b))).toBe('failure');
    expect(b.log).toEqual(['a']);
  });

  it('returns running and resumes at the running child, not from the top', () => {
    const b = board();
    const first = new Scripted('a', ['success']);
    const second = new Scripted('b', ['running', 'success']);
    const node = new Sequence<Board>([first, second]);

    expect(node.tick(ctx(b))).toBe('running');
    expect(b.log).toEqual(['a', 'b']);

    b.log.length = 0;
    expect(node.tick(ctx(b))).toBe('success');
    // `a` must NOT run again: that is what memory means.
    expect(b.log).toEqual(['b']);
  });

  it('resets its children when it completes', () => {
    const child = new Scripted('a', ['success']);
    const node = new Sequence<Board>([child]);
    node.tick(ctx(board()));
    expect(child.resets).toBeGreaterThan(0);
  });

  it('resets its children when it fails', () => {
    const child = new Scripted('a', ['failure']);
    const node = new Sequence<Board>([child]);
    node.tick(ctx(board()));
    expect(child.resets).toBeGreaterThan(0);
  });

  it('succeeds vacuously when empty', () => {
    expect(new Sequence<Board>([]).tick(ctx(board()))).toBe('success');
  });
});

describe('Selector', () => {
  it('succeeds on the first success and skips the rest', () => {
    const b = board();
    const node = new Selector<Board>([
      new Scripted('a', ['failure']),
      new Scripted('b', ['success']),
      new Scripted('c', ['success']),
    ]);
    expect(node.tick(ctx(b))).toBe('success');
    expect(b.log).toEqual(['a', 'b']);
  });

  it('fails only when every child fails', () => {
    const node = new Selector<Board>([
      new Scripted('a', ['failure']),
      new Scripted('b', ['failure']),
    ]);
    expect(node.tick(ctx(board()))).toBe('failure');
  });

  it('remembers a running child across ticks', () => {
    const b = board();
    const node = new Selector<Board>([
      new Scripted('a', ['failure']),
      new Scripted('b', ['running', 'success']),
    ]);
    expect(node.tick(ctx(b))).toBe('running');
    b.log.length = 0;
    expect(node.tick(ctx(b))).toBe('success');
    expect(b.log).toEqual(['b']);
  });

  it('fails vacuously when empty', () => {
    expect(new Selector<Board>([]).tick(ctx(board()))).toBe('failure');
  });
});

describe('PrioritySelector', () => {
  it('re-evaluates from the top every tick', () => {
    const b = board();
    const high = new Scripted('high', ['failure', 'running']);
    const low = new Scripted('low', ['running', 'running']);
    const node = new PrioritySelector<Board>([high, low]);

    expect(node.tick(ctx(b))).toBe('running');
    expect(b.log).toEqual(['high', 'low']);

    b.log.length = 0;
    expect(node.tick(ctx(b))).toBe('running');
    // `high` runs again even though `low` was running: that is the whole point.
    expect(b.log).toEqual(['high']);
  });

  it('resets the abandoned lower-priority child when a higher one takes over', () => {
    const high = new Scripted('high', ['failure', 'running']);
    const low = new Scripted('low', ['running']);
    const node = new PrioritySelector<Board>([high, low]);
    node.tick(ctx(board()));
    expect(low.resets).toBe(0);
    node.tick(ctx(board()));
    expect(low.resets).toBe(1);
  });

  it('clears its memory when every child fails', () => {
    const running = new Scripted('a', ['running', 'failure']);
    const node = new PrioritySelector<Board>([running]);
    expect(node.tick(ctx(board()))).toBe('running');
    expect(node.tick(ctx(board()))).toBe('failure');
    expect(running.resets).toBeGreaterThan(0);
  });

  it('reports success without holding on to the child', () => {
    const node = new PrioritySelector<Board>([new Scripted('a', ['success'])]);
    expect(node.tick(ctx(board()))).toBe('success');
  });
});

describe('decorators', () => {
  it('Inverter swaps success and failure', () => {
    expect(new Inverter<Board>(new Scripted('a', ['success'])).tick(ctx(board()))).toBe('failure');
    expect(new Inverter<Board>(new Scripted('a', ['failure'])).tick(ctx(board()))).toBe('success');
  });

  it('Inverter passes running through', () => {
    expect(new Inverter<Board>(new Scripted('a', ['running'])).tick(ctx(board()))).toBe('running');
  });

  it('Succeeder turns failure into success but keeps running', () => {
    expect(new Succeeder<Board>(new Scripted('a', ['failure'])).tick(ctx(board()))).toBe('success');
    expect(new Succeeder<Board>(new Scripted('a', ['running'])).tick(ctx(board()))).toBe('running');
  });

  it('Cooldown runs the child, then refuses for the cooldown period', () => {
    const child = new Scripted('a', ['success']);
    const node = new Cooldown<Board>(1, child);
    const b = board();

    expect(node.tick(ctx(b, 0))).toBe('success');
    expect(node.remaining).toBe(1);
    expect(node.tick(ctx(b, 0.5))).toBe('failure');
    expect(node.remaining).toBeCloseTo(0.5, 6);
    expect(node.tick(ctx(b, 0.5))).toBe('failure');
    expect(node.remaining).toBe(0);
    expect(node.tick(ctx(b, 0))).toBe('success');
  });

  it('Cooldown does not start its clock while the child is still running', () => {
    const node = new Cooldown<Board>(1, new Scripted('a', ['running', 'success']));
    expect(node.tick(ctx(board(), 0))).toBe('running');
    expect(node.remaining).toBe(0);
    expect(node.tick(ctx(board(), 0))).toBe('success');
    expect(node.remaining).toBe(1);
  });

  it('Cooldown survives a reset, so re-planning does not grant a free attack', () => {
    const node = new Cooldown<Board>(2, new Scripted('a', ['success']));
    node.tick(ctx(board(), 0));
    node.reset();
    expect(node.remaining).toBe(2);
    expect(node.tick(ctx(board(), 0))).toBe('failure');
  });

  it('Cooldown can be triggered without running the child', () => {
    const b = board();
    const node = new Cooldown<Board>(1, new Scripted('a', ['success']));
    node.trigger();
    expect(node.tick(ctx(b, 0))).toBe('failure');
    expect(b.log).toEqual([]);
  });

  it('UntilFail repeats until the child fails', () => {
    const b = board();
    const node = new UntilFail<Board>(new Scripted('a', ['success', 'success', 'failure']));
    expect(node.tick(ctx(b))).toBe('success');
    expect(b.log).toEqual(['a', 'a', 'a']);
  });

  it('UntilFail is bounded so a always-succeeding child cannot hang the frame', () => {
    const b = board();
    const node = new UntilFail<Board>(new Scripted('a', ['success']), 5);
    expect(node.tick(ctx(b))).toBe('success');
    expect(b.log.length).toBe(5);
  });
});

describe('leaves', () => {
  it('Condition maps a predicate onto success and failure', () => {
    const b = board();
    const node = new Condition<Board>('flag', (tick) => tick.blackboard.flag);
    expect(node.tick(ctx(b))).toBe('failure');
    b.flag = true;
    expect(node.tick(ctx(b))).toBe('success');
  });

  it('Action returns whatever it says and can clean up on reset', () => {
    let cleaned = 0;
    const node = new Action<Board>('act', () => 'running', () => cleaned++);
    expect(node.tick(ctx(board()))).toBe('running');
    node.reset();
    expect(cleaned).toBe(1);
  });

  it('Wait runs for the requested time and then succeeds exactly once', () => {
    const node = new Wait<Board>(0.5);
    const b = board();
    expect(node.tick(ctx(b, 0.2))).toBe('running');
    expect(node.tick(ctx(b, 0.2))).toBe('running');
    expect(node.tick(ctx(b, 0.2))).toBe('success');
    // Having succeeded, it re-arms.
    expect(node.tick(ctx(b, 0.2))).toBe('running');
  });

  it('Wait of zero succeeds immediately', () => {
    expect(new Wait<Board>(0).tick(ctx(board(), 0))).toBe('success');
  });

  it('Wait forgets its elapsed time on reset', () => {
    const node = new Wait<Board>(1);
    node.tick(ctx(board(), 0.9));
    node.reset();
    expect(node.elapsed).toBe(0);
    expect(node.tick(ctx(board(), 0.5))).toBe('running');
  });
});

describe('BehaviorTree', () => {
  it('accumulates time across ticks and exposes it to nodes', () => {
    const seen: number[] = [];
    const tree = new BehaviorTree<Board>(
      bt.action<Board>('probe', (tick) => {
        seen.push(tick.time);
        return 'success';
      }),
    );
    tree.tick(board(), 0.5);
    tree.tick(board(), 0.25);
    expect(seen).toEqual([0.5, 0.75]);
    expect(tree.time).toBeCloseTo(0.75, 6);
  });

  it('records the status of the last tick', () => {
    const tree = new BehaviorTree<Board>(bt.condition<Board>('no', () => false));
    expect(tree.status).toBe('failure');
    tree.tick(board(), 0);
    expect(tree.status).toBe('failure');
  });

  it('reset clears the whole subtree', () => {
    const child = new Scripted('a', ['running']);
    const tree = new BehaviorTree<Board>(new Sequence<Board>([child]));
    tree.tick(board(), 0);
    tree.reset();
    expect(child.resets).toBeGreaterThan(0);
  });
});

describe('a skeleton-shaped tree', () => {
  /**
   * The exact shape `enemies/Skeleton.ts` builds: attack above chase above
   * patrol, with attack committed once it has started.
   */
  function build(state: { inRange: boolean; aware: boolean; swinging: boolean }) {
    const log: string[] = [];
    const tree = new BehaviorTree<Board>(
      bt.priority<Board>(
        'root',
        bt.sequence<Board>(
          'attack',
          bt.condition<Board>('canAttack', () => state.swinging || state.inRange),
          bt.action<Board>('swing', () => {
            log.push('swing');
            state.swinging = true;
            return 'running';
          }),
        ),
        bt.sequence<Board>(
          'chase',
          bt.condition<Board>('aware', () => state.aware),
          bt.action<Board>('pursue', () => {
            log.push('chase');
            return 'running';
          }),
        ),
        bt.action<Board>('patrol', () => {
          log.push('patrol');
          return 'running';
        }),
      ),
    );
    return { tree, log };
  }

  it('patrols when it knows nothing', () => {
    const state = { inRange: false, aware: false, swinging: false };
    const { tree, log } = build(state);
    tree.tick(board(), 0.1);
    expect(log).toEqual(['patrol']);
  });

  it('switches from patrol to chase the moment it becomes aware', () => {
    const state = { inRange: false, aware: false, swinging: false };
    const { tree, log } = build(state);
    tree.tick(board(), 0.1);
    state.aware = true;
    tree.tick(board(), 0.1);
    expect(log).toEqual(['patrol', 'chase']);
  });

  it('pre-empts a running chase to attack once in range', () => {
    const state = { inRange: false, aware: true, swinging: false };
    const { tree, log } = build(state);
    tree.tick(board(), 0.1);
    state.inRange = true;
    tree.tick(board(), 0.1);
    expect(log).toEqual(['chase', 'swing']);
  });

  it('stays committed to a swing after the player leaves range', () => {
    const state = { inRange: true, aware: true, swinging: false };
    const { tree, log } = build(state);
    tree.tick(board(), 0.1);
    state.inRange = false;
    tree.tick(board(), 0.1);
    expect(log).toEqual(['swing', 'swing']);
  });
});
