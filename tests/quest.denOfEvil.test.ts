/**
 * The Den of Evil quest, end to end.
 *
 * The assertion this file exists for is that the kill count is **truthful**:
 * the tracker's "5 of 20" must match the number of monsters the dungeon
 * generator actually placed, for any seed. So the count is taken from a real
 * `generateDungeon` layout rather than from a constant, and the quest is driven
 * to completion against that number.
 */

import { describe, expect, it } from 'vitest';

import { EventBus } from '../src/core/EventBus';
import {
  DEN_KILL_OBJECTIVE,
  DEN_LOCATION_TAG,
  DEN_MONSTER_TAG,
  DEN_OF_EVIL_QUEST_ID,
  DEN_REACH_OBJECTIVE,
  PROVISIONAL_MONSTER_COUNT,
  createDenOfEvilQuest,
} from '../src/quest/DenOfEvil';
import { QuestSystem } from '../src/quest/QuestSystem';
import { generateDungeon } from '../src/world/DungeonGenerator';

function system(count: number): QuestSystem {
  const quests = new QuestSystem();
  quests.register(createDenOfEvilQuest(count));
  return quests;
}

describe('the definition', () => {
  it('is given by Akara and handed back to Akara', () => {
    const definition = createDenOfEvilQuest(20);
    expect(definition.giver).toBe('akara');
    expect(definition.turnInTo).toBe('akara');
    expect(definition.title).toBe('Den of Evil');
  });

  it("pays a skill point, as D2's own version does", () => {
    expect(createDenOfEvilQuest(20).rewards.skillPoints).toBe(1);
  });

  it('has a reach objective and a kill objective, in that order', () => {
    const [first, second] = createDenOfEvilQuest(20).objectives;
    expect(first?.id).toBe(DEN_REACH_OBJECTIVE);
    expect(first?.type).toBe('reach');
    expect(first?.target).toBe(DEN_LOCATION_TAG);
    expect(second?.id).toBe(DEN_KILL_OBJECTIVE);
    expect(second?.type).toBe('kill');
    expect(second?.target).toBe(DEN_MONSTER_TAG);
  });

  it('takes its required count from the argument, not a constant', () => {
    for (const count of [1, 7, 23, 41]) {
      const objective = createDenOfEvilQuest(count).objectives[1];
      expect(objective?.required).toBe(count);
    }
    // And clamps a nonsense count rather than producing an uncompletable quest.
    expect(createDenOfEvilQuest(0).objectives[1]?.required).toBe(1);
  });

  it('shows the count on the kill objective and hides it on the reach', () => {
    const [reach, kill] = createDenOfEvilQuest(20).objectives;
    expect(reach?.showCount).toBe(false);
    expect(kill?.showCount).toBe(true);
  });
});

describe('the count matches the generated cave', () => {
  it('agrees with the dungeon generator for a range of seeds', () => {
    for (const seed of ['den-1', 'den-2', 42, 'act-one']) {
      const layout = generateDungeon({ seed });
      const spawns = layout.spawnPoints.length;
      expect(spawns).toBeGreaterThan(0);

      const quests = system(spawns);
      quests.offer(DEN_OF_EVIL_QUEST_ID);
      quests.accept(DEN_OF_EVIL_QUEST_ID);
      quests.notifyReach(DEN_LOCATION_TAG);

      // Kill exactly as many as the generator placed, and no more.
      for (let i = 0; i < spawns; i++) {
        expect(quests.state(DEN_OF_EVIL_QUEST_ID), `${seed} after ${i}`).toBe('active');
        quests.notifyKill(DEN_MONSTER_TAG);
      }
      expect(quests.state(DEN_OF_EVIL_QUEST_ID), `${seed}`).toBe('complete');
      const status = quests.status(DEN_OF_EVIL_QUEST_ID);
      expect(status?.objectives[1]?.current).toBe(spawns);
      expect(status?.objectives[1]?.required).toBe(spawns);
    }
  });

  it('adopts a new count mid-quest without losing progress', () => {
    // This is what happens when the player accepts the quest in the camp — with
    // the provisional count — and then walks into a cave containing a different
    // number of monsters.
    const quests = system(PROVISIONAL_MONSTER_COUNT);
    quests.offer(DEN_OF_EVIL_QUEST_ID);
    quests.accept(DEN_OF_EVIL_QUEST_ID);
    quests.notifyReach(DEN_LOCATION_TAG);
    quests.notifyKill(DEN_MONSTER_TAG, 4);

    const layout = generateDungeon({ seed: 'den-of-evil' });
    quests.register(createDenOfEvilQuest(layout.spawnPoints.length));

    const status = quests.status(DEN_OF_EVIL_QUEST_ID);
    expect(status?.state).toBe('active');
    expect(status?.objectives[1]?.current).toBe(4);
    expect(status?.objectives[1]?.required).toBe(layout.spawnPoints.length);

    quests.notifyKill(DEN_MONSTER_TAG, layout.spawnPoints.length - 4);
    expect(quests.state(DEN_OF_EVIL_QUEST_ID)).toBe('complete');
  });

  it('completes on the last monster and not one before', () => {
    const quests = system(5);
    quests.offer(DEN_OF_EVIL_QUEST_ID);
    quests.accept(DEN_OF_EVIL_QUEST_ID);
    quests.notifyReach(DEN_LOCATION_TAG);
    quests.notifyKill(DEN_MONSTER_TAG, 4);
    expect(quests.state(DEN_OF_EVIL_QUEST_ID)).toBe('active');
    quests.notifyKill(DEN_MONSTER_TAG);
    expect(quests.state(DEN_OF_EVIL_QUEST_ID)).toBe('complete');
  });

  it('does not complete on kills alone if the cave was never entered', () => {
    // Only reachable through a scripted path, but the ordering must hold.
    const quests = system(3);
    quests.offer(DEN_OF_EVIL_QUEST_ID);
    quests.accept(DEN_OF_EVIL_QUEST_ID);
    quests.notifyKill(DEN_MONSTER_TAG, 3);
    expect(quests.state(DEN_OF_EVIL_QUEST_ID)).toBe('active');
    quests.notifyReach(DEN_LOCATION_TAG);
    expect(quests.state(DEN_OF_EVIL_QUEST_ID)).toBe('complete');
  });
});

describe('the whole run', () => {
  it('goes from offer to reward and pays what it promised', () => {
    const events = new EventBus();
    const seen: string[] = [];
    events.on('quest:accepted', () => seen.push('accepted'));
    events.on('quest:complete', () => seen.push('complete'));
    events.on('quest:rewarded', (payload) =>
      seen.push(`rewarded:${payload.skillPoints}:${payload.experience}`),
    );

    const layout = generateDungeon({ seed: 'end-to-end' });
    const quests = system(layout.spawnPoints.length);
    quests.bindEvents(events);

    let paidSkillPoints = 0;
    quests.setRewardHandler((payload) => {
      paidSkillPoints += payload.rewards.skillPoints ?? 0;
    });

    quests.offer(DEN_OF_EVIL_QUEST_ID);
    quests.accept(DEN_OF_EVIL_QUEST_ID);
    quests.notifyReach(DEN_LOCATION_TAG);
    quests.notifyKill(DEN_MONSTER_TAG, layout.spawnPoints.length);
    expect(quests.turnInAndReward(DEN_OF_EVIL_QUEST_ID)).toBe(true);

    expect(quests.state(DEN_OF_EVIL_QUEST_ID)).toBe('rewarded');
    expect(paidSkillPoints).toBe(1);
    expect(seen).toEqual(['accepted', 'complete', 'rewarded:1:900']);
  });

  it('survives a save and load in the middle', () => {
    const layout = generateDungeon({ seed: 'mid-save' });
    const total = layout.spawnPoints.length;
    const half = Math.floor(total / 2);

    const first = system(total);
    first.offer(DEN_OF_EVIL_QUEST_ID);
    first.accept(DEN_OF_EVIL_QUEST_ID);
    first.notifyReach(DEN_LOCATION_TAG);
    first.notifyKill(DEN_MONSTER_TAG, half);
    const snapshot = first.toJSON();

    const second = system(total);
    second.load(snapshot);
    second.notifyKill(DEN_MONSTER_TAG, total - half);
    expect(second.state(DEN_OF_EVIL_QUEST_ID)).toBe('complete');
    expect(second.turnInAndReward(DEN_OF_EVIL_QUEST_ID)).toBe(true);
  });
});
