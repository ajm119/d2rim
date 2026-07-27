/**
 * The quest state machine: legal transitions, and — more importantly — the
 * illegal ones.
 *
 * Most of this file asserts that something *does not* happen: accepting a quest
 * that was never offered, progressing one that was never accepted, claiming a
 * reward twice. Those are the failures that make a quest completable in the
 * wrong order, and none of them show up in a test that only walks the happy
 * path.
 */

import { describe, expect, it, vi } from 'vitest';

import { EventBus } from '../src/core/EventBus';
import {
  QUEST_STATE_ORDER,
  QuestSystem,
  questStateAtLeast,
  type QuestDefinition,
} from '../src/quest/QuestSystem';

const QUEST: QuestDefinition = {
  id: 'test-quest',
  title: 'A Test of Faith',
  summary: 'Kill three things, find one place, and speak to one person.',
  giver: 'akara',
  objectives: [
    { id: 'kill', type: 'kill', description: 'Slay three skeletons', target: 'skeleton', required: 3 },
    { id: 'reach', type: 'reach', description: 'Find the cave', target: 'cave', required: 1 },
    { id: 'talk', type: 'talk', description: 'Speak to Kashya', target: 'kashya', required: 1 },
  ],
  rewards: { experience: 500, gold: 50, skillPoints: 1 },
};

function system(): QuestSystem {
  const quests = new QuestSystem();
  quests.register(QUEST);
  return quests;
}

/** Drive a quest to `complete`. */
function complete(quests: QuestSystem): void {
  quests.offer(QUEST.id);
  quests.accept(QUEST.id);
  quests.notifyKill('skeleton', 3);
  quests.notifyReach('cave');
  quests.notifyTalk('kashya');
}

describe('state ordering', () => {
  it('orders the lifecycle', () => {
    expect(QUEST_STATE_ORDER).toEqual([
      'inactive',
      'offered',
      'active',
      'complete',
      'turnedIn',
      'rewarded',
    ]);
  });

  it('compares states by lifecycle position', () => {
    expect(questStateAtLeast('complete', 'active')).toBe(true);
    expect(questStateAtLeast('active', 'complete')).toBe(false);
    expect(questStateAtLeast('rewarded', 'rewarded')).toBe(true);
  });
});

describe('legal transitions', () => {
  it('walks inactive -> offered -> active -> complete -> turnedIn -> rewarded', () => {
    const quests = system();
    expect(quests.state(QUEST.id)).toBe('inactive');
    expect(quests.offer(QUEST.id)).toBe(true);
    expect(quests.state(QUEST.id)).toBe('offered');
    expect(quests.accept(QUEST.id)).toBe(true);
    expect(quests.state(QUEST.id)).toBe('active');

    quests.notifyKill('skeleton', 3);
    quests.notifyReach('cave');
    expect(quests.state(QUEST.id)).toBe('active');
    quests.notifyTalk('kashya');
    expect(quests.state(QUEST.id)).toBe('complete');

    expect(quests.turnIn(QUEST.id)).toBe(true);
    expect(quests.state(QUEST.id)).toBe('turnedIn');
    expect(quests.reward(QUEST.id)).toBe(true);
    expect(quests.state(QUEST.id)).toBe('rewarded');
  });

  it('returns a declined quest to inactive so it can be offered again', () => {
    const quests = system();
    quests.offer(QUEST.id);
    expect(quests.decline(QUEST.id)).toBe(true);
    expect(quests.state(QUEST.id)).toBe('inactive');
    expect(quests.offer(QUEST.id)).toBe(true);
  });

  it('completes a quest with no objectives the moment it is accepted', () => {
    const quests = new QuestSystem();
    quests.register({ ...QUEST, id: 'empty', objectives: [] });
    quests.offer('empty');
    quests.accept('empty');
    expect(quests.state('empty')).toBe('complete');
  });
});

describe('illegal transitions', () => {
  it('refuses to accept a quest that was never offered', () => {
    const quests = system();
    expect(quests.accept(QUEST.id)).toBe(false);
    expect(quests.state(QUEST.id)).toBe('inactive');
  });

  it('refuses to offer a quest that is already active', () => {
    const quests = system();
    quests.offer(QUEST.id);
    quests.accept(QUEST.id);
    expect(quests.offer(QUEST.id)).toBe(false);
    expect(quests.state(QUEST.id)).toBe('active');
  });

  it('refuses to turn in a quest whose objectives are unfinished', () => {
    const quests = system();
    quests.offer(QUEST.id);
    quests.accept(QUEST.id);
    quests.notifyKill('skeleton', 2);
    expect(quests.turnIn(QUEST.id)).toBe(false);
    expect(quests.state(QUEST.id)).toBe('active');
  });

  it('refuses to reward a quest that has not been turned in', () => {
    const quests = system();
    complete(quests);
    expect(quests.reward(QUEST.id)).toBe(false);
    expect(quests.state(QUEST.id)).toBe('complete');
  });

  it('pays a reward exactly once', () => {
    const quests = system();
    const handler = vi.fn();
    quests.setRewardHandler(handler);
    complete(quests);
    expect(quests.turnInAndReward(QUEST.id)).toBe(true);
    expect(handler).toHaveBeenCalledTimes(1);

    expect(quests.turnIn(QUEST.id)).toBe(false);
    expect(quests.reward(QUEST.id)).toBe(false);
    expect(handler).toHaveBeenCalledTimes(1);
  });

  it('advances the state before the reward handler runs, so a throw cannot pay twice', () => {
    const quests = system();
    quests.setRewardHandler(() => {
      throw new Error('reward exploded');
    });
    complete(quests);
    quests.turnIn(QUEST.id);
    expect(() => quests.reward(QUEST.id)).toThrow('reward exploded');
    expect(quests.state(QUEST.id)).toBe('rewarded');
    expect(quests.reward(QUEST.id)).toBe(false);
  });

  it('does nothing at all for an unknown quest id', () => {
    const quests = system();
    expect(quests.offer('nope')).toBe(false);
    expect(quests.accept('nope')).toBe(false);
    expect(quests.progress('nope', 'kill')).toBe(false);
    expect(quests.state('nope')).toBe('inactive');
    expect(quests.status('nope')).toBeNull();
  });
});

describe('progress', () => {
  it('banks nothing before the quest is accepted', () => {
    const quests = system();
    expect(quests.notifyKill('skeleton', 3)).toBe(0);
    quests.offer(QUEST.id);
    expect(quests.notifyKill('skeleton', 3)).toBe(0);
    quests.accept(QUEST.id);
    expect(quests.status(QUEST.id)?.objectives[0]?.current).toBe(0);
  });

  it('banks nothing after the quest is handed in', () => {
    const quests = system();
    complete(quests);
    quests.turnInAndReward(QUEST.id);
    expect(quests.notifyKill('skeleton', 1)).toBe(0);
  });

  it('clamps at the required count', () => {
    const quests = system();
    quests.offer(QUEST.id);
    quests.accept(QUEST.id);
    quests.notifyKill('skeleton', 99);
    expect(quests.status(QUEST.id)?.objectives[0]?.current).toBe(3);
    // And a further kill reports no advance rather than silently over-counting.
    expect(quests.notifyKill('skeleton')).toBe(0);
  });

  it('only matches objectives whose type and target both agree', () => {
    const quests = system();
    quests.offer(QUEST.id);
    quests.accept(QUEST.id);
    expect(quests.notifyKill('cave')).toBe(0); // right target, wrong type
    expect(quests.notifyKill('zombie')).toBe(0); // right type, wrong target
    expect(quests.notifyReach('cave')).toBe(1);
  });

  it('shows the count only when there is more than one to do', () => {
    const quests = system();
    quests.offer(QUEST.id);
    quests.accept(QUEST.id);
    const status = quests.status(QUEST.id);
    expect(status?.objectives[0]?.showCount).toBe(true);
    expect(status?.objectives[1]?.showCount).toBe(false);
  });
});

describe('events', () => {
  it('emits the whole lifecycle onto the bus', () => {
    const events = new EventBus();
    const seen: string[] = [];
    for (const name of [
      'quest:offered',
      'quest:accepted',
      'quest:progress',
      'quest:objectiveComplete',
      'quest:complete',
      'quest:turnedIn',
      'quest:rewarded',
    ] as const) {
      events.on(name, () => seen.push(name));
    }

    const quests = system();
    quests.bindEvents(events);
    complete(quests);
    quests.turnInAndReward(QUEST.id);

    expect(seen).toContain('quest:offered');
    expect(seen).toContain('quest:accepted');
    expect(seen).toContain('quest:progress');
    expect(seen).toContain('quest:objectiveComplete');
    expect(seen.indexOf('quest:complete')).toBeLessThan(seen.indexOf('quest:turnedIn'));
    expect(seen.indexOf('quest:turnedIn')).toBeLessThan(seen.indexOf('quest:rewarded'));
  });

  it('emits one progress event per increment, carrying the running count', () => {
    const events = new EventBus();
    const counts: number[] = [];
    events.on('quest:progress', (payload) => counts.push(payload.current));

    const quests = system();
    quests.bindEvents(events);
    quests.offer(QUEST.id);
    quests.accept(QUEST.id);
    quests.notifyKill('skeleton');
    quests.notifyKill('skeleton');
    expect(counts).toEqual([1, 2]);
  });

  it('carries the reward amounts on the rewarded event', () => {
    const events = new EventBus();
    const payloads: { skillPoints: number; experience: number }[] = [];
    events.on('quest:rewarded', (value) => payloads.push(value));

    const quests = system();
    quests.bindEvents(events);
    complete(quests);
    quests.turnInAndReward(QUEST.id);
    expect(payloads).toHaveLength(1);
    expect(payloads[0]?.skillPoints).toBe(1);
    expect(payloads[0]?.experience).toBe(500);
  });

  it('works with no bus attached at all', () => {
    const quests = system();
    quests.bindEvents(null);
    expect(() => complete(quests)).not.toThrow();
    expect(quests.state(QUEST.id)).toBe('complete');
  });
});

describe('tracking and status', () => {
  it('tracks a quest only while it is between accept and reward', () => {
    const quests = system();
    expect(quests.tracked()).toHaveLength(0);
    quests.offer(QUEST.id);
    expect(quests.tracked()).toHaveLength(0);
    quests.accept(QUEST.id);
    expect(quests.tracked()).toHaveLength(1);
    complete(quests);
    expect(quests.tracked()).toHaveLength(1);
    quests.turnInAndReward(QUEST.id);
    expect(quests.tracked()).toHaveLength(0);
  });
});

describe('persistence', () => {
  it('round-trips state and counters', () => {
    const quests = system();
    quests.offer(QUEST.id);
    quests.accept(QUEST.id);
    quests.notifyKill('skeleton', 2);
    const snapshot = quests.toJSON();

    const restored = system();
    restored.load(snapshot);
    expect(restored.toJSON()).toEqual(snapshot);
    expect(restored.state(QUEST.id)).toBe('active');
    expect(restored.status(QUEST.id)?.objectives[0]?.current).toBe(2);
    // And progress continues from where it left off.
    restored.notifyKill('skeleton');
    expect(restored.status(QUEST.id)?.objectives[0]?.current).toBe(3);
  });

  it('drops saved counters for objectives that no longer exist', () => {
    const quests = system();
    quests.load({
      quests: [{ id: QUEST.id, state: 'active', counters: { kill: 2, ghost: 9 } }],
    });
    const snapshot = quests.toJSON();
    const entry = snapshot.quests.find((q) => q.id === QUEST.id);
    expect(entry?.counters['ghost']).toBeUndefined();
    expect(entry?.counters['kill']).toBe(2);
  });

  it('ignores a saved quest whose definition has been removed', () => {
    const quests = system();
    expect(() =>
      quests.load({ quests: [{ id: 'deleted-quest', state: 'active', counters: {} }] }),
    ).not.toThrow();
    expect(quests.state('deleted-quest')).toBe('inactive');
  });

  it('keeps live progress when a definition is re-registered', () => {
    // This is the mechanism the Den of Evil quest relies on to adopt the
    // generator's real monster count mid-quest.
    const quests = system();
    quests.offer(QUEST.id);
    quests.accept(QUEST.id);
    quests.notifyKill('skeleton', 2);

    quests.register({
      ...QUEST,
      objectives: QUEST.objectives.map((objective) =>
        objective.id === 'kill' ? { ...objective, required: 20 } : objective,
      ),
    });
    const status = quests.status(QUEST.id);
    expect(status?.state).toBe('active');
    expect(status?.objectives[0]?.current).toBe(2);
    expect(status?.objectives[0]?.required).toBe(20);
  });

  it('resets everything for a new game', () => {
    const quests = system();
    complete(quests);
    quests.reset();
    expect(quests.state(QUEST.id)).toBe('inactive');
    expect(quests.status(QUEST.id)?.objectives.every((o) => o.current === 0)).toBe(true);
  });
});
