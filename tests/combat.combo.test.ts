import { describe, expect, it } from 'vitest';

import {
  ComboMachine,
  DEFAULT_BUFFER_TIME,
  PLAYER_CHAIN_ROOTS,
  PLAYER_MOVES,
  type MeleeMove,
} from '../src/combat/CombatSystem';

/** Every move the machine hands out has to be started before it is timed. */
function start(machine: ComboMachine, duration = 1): MeleeMove | null {
  const move = machine.update(0);
  if (move !== null) machine.setDuration(duration);
  return move;
}

/** Advance in small steps, starting anything the machine hands out. */
function run(
  machine: ComboMachine,
  seconds: number,
  duration = 1,
  step = 1 / 60,
): MeleeMove[] {
  const started: MeleeMove[] = [];
  for (let t = 0; t < seconds; t += step) {
    const move = machine.update(step);
    if (move !== null) {
      machine.setDuration(duration);
      started.push(move);
    }
  }
  return started;
}

function machine(): ComboMachine {
  return new ComboMachine(PLAYER_MOVES, { roots: PLAYER_CHAIN_ROOTS });
}

describe('move table', () => {
  it('defines the documented chain: chop -> slice -> stab -> spin', () => {
    expect(PLAYER_MOVES.get('chop')?.next.light).toBe('slice');
    expect(PLAYER_MOVES.get('slice')?.next.light).toBe('stab');
    expect(PLAYER_MOVES.get('stab')?.next.light).toBe('spin');
  });

  it('gives every move a distinct clip so a chain never replays one animation', () => {
    const actions = [...PLAYER_MOVES.values()].map((move) => move.action);
    expect(new Set(actions).size).toBe(actions.length);
  });

  it('opens every chain window before the move ends, or nothing could chain', () => {
    for (const move of PLAYER_MOVES.values()) {
      expect(move.chain[0]).toBeLessThan(move.recovery);
      expect(move.chain[0]).toBeLessThanOrEqual(move.chain[1]);
    }
  });

  it('keeps every damage window inside the clip', () => {
    for (const move of PLAYER_MOVES.values()) {
      expect(move.window[0]).toBeGreaterThan(0);
      expect(move.window[0]).toBeLessThan(move.window[1]);
      expect(move.window[1]).toBeLessThanOrEqual(1);
    }
  });

  it('resolves both chain roots', () => {
    expect(PLAYER_MOVES.has(PLAYER_CHAIN_ROOTS.light)).toBe(true);
    expect(PLAYER_MOVES.has(PLAYER_CHAIN_ROOTS.heavy)).toBe(true);
  });
});

describe('ComboMachine', () => {
  it('starts idle and hands out nothing without input', () => {
    const combo = machine();
    expect(combo.phase).toBe('idle');
    expect(run(combo, 2)).toEqual([]);
  });

  it('starts the light root on the first press', () => {
    const combo = machine();
    combo.press('light');
    expect(start(combo)?.id).toBe('chop');
    expect(combo.phase).toBe('swinging');
    expect(combo.chainIndex).toBe(0);
  });

  it('starts the heavy root on a heavy press', () => {
    const combo = machine();
    combo.press('heavy');
    expect(start(combo)?.id).toBe('heavy');
  });

  it('consumes the buffered press when it starts a move', () => {
    const combo = machine();
    combo.press('light');
    start(combo);
    expect(combo.buffered).toBeNull();
  });

  it('ignores a press made too early in the swing, then chains when the window opens', () => {
    const combo = machine();
    combo.press('light');
    start(combo, 1);

    // Press immediately: the chain window on `chop` opens at 0.34.
    combo.press('light');
    const early = run(combo, 0.2);
    expect(early).toEqual([]);
    expect(combo.buffered).toBe('light');

    const later = run(combo, 0.3);
    expect(later.map((move) => move.id)).toEqual(['slice']);
  });

  it('walks the whole chain under continuous mashing', () => {
    const combo = machine();
    const ids: string[] = [];
    for (let t = 0; t < 6; t += 1 / 60) {
      combo.press('light');
      const move = combo.update(1 / 60);
      if (move !== null) {
        combo.setDuration(1);
        ids.push(move.id);
      }
    }
    expect(ids.slice(0, 4)).toEqual(['chop', 'slice', 'stab', 'spin']);
  });

  it('increments the chain index as it goes and resets on a fresh chain', () => {
    const combo = machine();
    combo.press('light');
    start(combo);
    expect(combo.chainIndex).toBe(0);
    combo.press('light');
    run(combo, 0.5);
    expect(combo.chainIndex).toBe(1);

    // Let it lapse entirely, then start again.
    run(combo, 3);
    combo.press('light');
    start(combo);
    expect(combo.chainIndex).toBe(0);
  });

  it('restarts the chain after the finisher, since the spin chains nowhere', () => {
    const combo = machine();
    combo.press('light');
    const ids: Array<string | undefined> = [start(combo, 1)?.id];
    // Press just inside each move's chain window, which is how a player who is
    // watching the animation actually plays it.
    for (let i = 0; i < 4; i++) {
      const current = combo.current;
      if (current === null) break;
      run(combo, Math.max(0, current.chain[0] - combo.normalizedTime));
      combo.press('light');
      ids.push(run(combo, 0.3)[0]?.id);
    }
    expect(ids).toEqual(['chop', 'slice', 'stab', 'spin', 'chop']);
  });

  it('branches into the heavy move mid-chain', () => {
    const combo = machine();
    combo.press('light');
    start(combo);
    combo.press('heavy');
    const started = run(combo, 0.6);
    expect(started.map((move) => move.id)).toEqual(['heavy']);
  });

  it('takes the spin as the heavy follow-up to a heavy', () => {
    expect(PLAYER_MOVES.get('heavy')?.next.heavy).toBe('spin');
    const combo = machine();
    combo.press('heavy');
    start(combo, 1);
    // Heavy chains from 0.5; press inside the buffer window rather than at the
    // very start of a long swing, which is what the buffer is for.
    run(combo, 0.45);
    combo.press('heavy');
    expect(run(combo, 0.2)[0]?.id).toBe('spin');
  });

  it('does not hold a press indefinitely across a long wind-up', () => {
    // A press at the top of the heavy is *not* meant to survive until its chain
    // window half a second later: buffering is forgiveness, not a queue.
    const combo = machine();
    combo.press('heavy');
    start(combo, 1);
    combo.press('heavy');
    expect(run(combo, 0.44)).toEqual([]);
    expect(combo.buffered).toBeNull();
  });

  it('buffers a press made during recovery and fires it the moment the move ends', () => {
    const combo = machine();
    combo.press('light');
    start(combo, 1);
    // Run past the chain window's close but before recovery.
    run(combo, 0.99 * 1);
    // `chop` recovers at 0.78 and chains until 0.98, so press at 0.99.
    combo.press('light');
    const started = run(combo, 0.1);
    expect(started.length).toBe(1);
  });

  it('expires a buffered press that never reaches a chain window', () => {
    // A move whose chain window never opens: the press has nowhere to go, so
    // the buffer must lapse rather than sit there and fire minutes later.
    const stubborn = new Map<string, MeleeMove>([
      [
        'long',
        {
          ...(PLAYER_MOVES.get('chop') as MeleeMove),
          id: 'long',
          chain: [0.99, 1],
          recovery: 0.999,
        },
      ],
    ]);
    const combo = new ComboMachine(stubborn, { roots: { light: 'long', heavy: 'long' } });
    combo.press('light');
    combo.update(0);
    combo.setDuration(10);
    combo.press('light');
    expect(combo.buffered).toBe('light');
    combo.update(DEFAULT_BUFFER_TIME + 0.01);
    expect(combo.buffered).toBeNull();
  });

  it('reports the remaining buffer time as it drains', () => {
    const combo = machine();
    combo.press('light');
    start(combo, 10);
    combo.press('light');
    const before = combo.bufferRemaining;
    expect(before).toBeCloseTo(DEFAULT_BUFFER_TIME, 6);
    combo.update(0.1);
    expect(combo.bufferRemaining).toBeCloseTo(before - 0.1, 6);
  });

  it('replaces a queued press rather than stacking two swings', () => {
    const combo = machine();
    combo.press('light');
    combo.press('heavy');
    expect(combo.buffered).toBe('heavy');
    expect(start(combo)?.id).toBe('heavy');
  });

  it('returns at most one move per tick even with a huge delta', () => {
    const combo = machine();
    combo.press('light');
    const first = combo.update(10);
    expect(first?.id).toBe('chop');
    combo.setDuration(1);
    combo.press('light');
    const second = combo.update(10);
    expect(second).not.toBeNull();
    expect(combo.update(10)).toBeNull();
  });

  it('returns to idle once the move recovers', () => {
    const combo = machine();
    combo.press('light');
    start(combo, 1);
    run(combo, 0.9);
    expect(combo.phase).toBe('idle');
    expect(combo.current).toBeNull();
  });

  it('reports normalised progress through the current move', () => {
    const combo = machine();
    combo.press('light');
    start(combo, 1);
    combo.update(0.25);
    expect(combo.normalizedTime).toBeCloseTo(0.25, 3);
    expect(combo.normalizedTime).toBeLessThanOrEqual(1);
  });

  it('says whether the chain window is open right now', () => {
    const combo = machine();
    expect(combo.chainWindowOpen).toBe(true);
    combo.press('light');
    start(combo, 1);
    expect(combo.chainWindowOpen).toBe(false);
    combo.update(0.4);
    expect(combo.chainWindowOpen).toBe(true);
  });

  it('drops everything on interrupt', () => {
    const combo = machine();
    combo.press('light');
    start(combo);
    combo.press('light');
    combo.interrupt();
    expect(combo.phase).toBe('idle');
    expect(combo.buffered).toBeNull();
    expect(combo.chainIndex).toBe(0);
  });

  it('falls back to a one-second clip when nobody supplies a duration', () => {
    const combo = machine();
    combo.press('light');
    expect(combo.update(0)?.id).toBe('chop');
    combo.update(0.5);
    expect(combo.normalizedTime).toBeCloseTo(0.5, 6);
  });

  it('survives a move table whose chain points at a missing move', () => {
    const broken = new Map(PLAYER_MOVES);
    const chop = broken.get('chop');
    if (chop !== undefined) broken.set('chop', { ...chop, next: { light: 'ghost', heavy: null } });
    const combo = new ComboMachine(broken, { roots: PLAYER_CHAIN_ROOTS });
    combo.press('light');
    start(combo, 1);
    combo.press('light');
    // Nothing to chain into: the press waits and starts a fresh chop instead.
    const started = run(combo, 1.2);
    expect(started.every((move) => broken.has(move.id))).toBe(true);
  });

  it('does nothing when both roots are missing', () => {
    const combo = new ComboMachine(new Map(), { roots: PLAYER_CHAIN_ROOTS });
    combo.press('light');
    expect(combo.update(0.016)).toBeNull();
    expect(combo.buffered).toBeNull();
  });
});
