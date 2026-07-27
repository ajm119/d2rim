/**
 * Dialogue: the runner, the condition helpers, tree validation, and the camp
 * scripts themselves.
 *
 * The Akara tests are the ones that matter. She is required to say something
 * different before offering the Den of Evil, while it is running, once it is
 * finished and after it has been paid out, and the test drives the real quest
 * state machine to each of those points rather than stubbing a state — a stub
 * would pass even if the entry rules were wired to the wrong states.
 */

import { describe, expect, it, vi } from 'vitest';

import {
  AKARA_DIALOGUE,
  CAMP_DIALOGUE,
  CHARSI_DIALOGUE,
  dialogueForNpc,
} from '../src/quest/CampDialogue';
import { DEN_OF_EVIL_QUEST_ID, createDenOfEvilQuest } from '../src/quest/DenOfEvil';
import {
  DialogueRunner,
  allOf,
  anyOf,
  not,
  validateDialogueTree,
  whenFlag,
  whenQuest,
  whenQuestAtLeast,
  type DialogueContext,
  type DialogueTree,
} from '../src/quest/Dialogue';
import { QuestSystem } from '../src/quest/QuestSystem';

function context(quests: QuestSystem | null, flags: string[] = []): DialogueContext {
  return {
    quests,
    npcId: 'akara',
    characterLevel: 1,
    gold: 0,
    flags: new Set(flags),
  };
}

function denQuests(): QuestSystem {
  const quests = new QuestSystem();
  quests.register(createDenOfEvilQuest(5));
  return quests;
}

describe('condition helpers', () => {
  it('matches a quest against a set of states', () => {
    const quests = denQuests();
    const inactive = whenQuest(DEN_OF_EVIL_QUEST_ID, 'inactive');
    const active = whenQuest(DEN_OF_EVIL_QUEST_ID, 'active');
    expect(inactive(context(quests))).toBe(true);
    expect(active(context(quests))).toBe(false);

    quests.offer(DEN_OF_EVIL_QUEST_ID);
    quests.accept(DEN_OF_EVIL_QUEST_ID);
    expect(inactive(context(quests))).toBe(false);
    expect(active(context(quests))).toBe(true);
  });

  it('matches at-or-past a state', () => {
    const quests = denQuests();
    const past = whenQuestAtLeast(DEN_OF_EVIL_QUEST_ID, 'active');
    expect(past(context(quests))).toBe(false);
    quests.offer(DEN_OF_EVIL_QUEST_ID);
    expect(past(context(quests))).toBe(false);
    quests.accept(DEN_OF_EVIL_QUEST_ID);
    expect(past(context(quests))).toBe(true);
  });

  it('treats a missing quest system as an inactive quest', () => {
    expect(whenQuest('anything', 'inactive')(context(null))).toBe(true);
  });

  it('composes', () => {
    const yes = (): boolean => true;
    const no = (): boolean => false;
    expect(allOf(yes, yes)(context(null))).toBe(true);
    expect(allOf(yes, no)(context(null))).toBe(false);
    expect(anyOf(no, yes)(context(null))).toBe(true);
    expect(anyOf(no, no)(context(null))).toBe(false);
    expect(not(no)(context(null))).toBe(true);
    expect(whenFlag('hasBrokenGear')(context(null, ['hasBrokenGear']))).toBe(true);
    expect(whenFlag('hasBrokenGear')(context(null))).toBe(false);
  });
});

describe('tree validation', () => {
  it('passes every camp script', () => {
    for (const tree of CAMP_DIALOGUE) {
      expect(validateDialogueTree(tree), `${tree.id}`).toEqual([]);
    }
  });

  it('catches a dangling choice target', () => {
    const broken: DialogueTree = {
      id: 'broken',
      npcId: 'ghost',
      speaker: 'Ghost',
      entry: [{ node: 'start' }],
      nodes: [
        {
          id: 'start',
          text: 'Boo.',
          choices: [{ id: 'go', text: 'Where?', next: 'nowhere' }],
        },
      ],
    };
    expect(validateDialogueTree(broken)).toContain(
      'broken: choice "go" on "start" points at missing node "nowhere"',
    );
  });

  it('catches a conditional final entry rule', () => {
    const broken: DialogueTree = {
      id: 'broken',
      npcId: 'ghost',
      speaker: 'Ghost',
      entry: [{ node: 'start', condition: () => false }],
      nodes: [{ id: 'start', text: 'Boo.' }],
    };
    // With no unconditional fallback there is a state in which this NPC cannot
    // be spoken to at all.
    expect(validateDialogueTree(broken)).toContain(
      'broken: the final entry rule must be unconditional',
    );
  });

  it('catches a dangling continuation and duplicate choice ids', () => {
    const broken: DialogueTree = {
      id: 'broken',
      npcId: 'ghost',
      speaker: 'Ghost',
      entry: [{ node: 'start' }],
      nodes: [
        {
          id: 'start',
          text: 'Boo.',
          next: 'missing',
          choices: [
            { id: 'a', text: 'One', next: null },
            { id: 'a', text: 'Two', next: null },
          ],
        },
      ],
    };
    const problems = validateDialogueTree(broken);
    expect(problems.some((p) => p.includes('continues to missing node'))).toBe(true);
    expect(problems.some((p) => p.includes('duplicate choice id'))).toBe(true);
  });
});

describe('DialogueRunner', () => {
  it('opens on the first entry rule whose condition passes', () => {
    const runner = new DialogueRunner();
    const view = runner.start(CHARSI_DIALOGUE, context(null));
    expect(view?.nodeId).toBe('greeting');
    expect(view?.speaker).toBe('Charsi');
    expect(view?.text.length).toBeGreaterThan(0);
  });

  it('hides choices whose conditions fail', () => {
    const tree: DialogueTree = {
      id: 't',
      npcId: 'n',
      speaker: 'N',
      entry: [{ node: 'start' }],
      nodes: [
        {
          id: 'start',
          text: 'Pick.',
          choices: [
            { id: 'always', text: 'Always', next: null },
            { id: 'never', text: 'Never', next: null, condition: () => false },
          ],
        },
      ],
    };
    const runner = new DialogueRunner();
    const view = runner.start(tree, context(null));
    expect(view?.choices.map((choice) => choice.id)).toEqual(['always']);
  });

  it('reports actions rather than executing them', () => {
    const handler = vi.fn();
    const runner = new DialogueRunner();
    runner.setActionHandler(handler);
    runner.start(AKARA_DIALOGUE, context(denQuests()));
    runner.chooseById('ask-evil');
    expect(handler).toHaveBeenCalledWith(
      { kind: 'offerQuest', questId: DEN_OF_EVIL_QUEST_ID },
      expect.anything(),
    );
  });

  it('ends the conversation on a choice with no continuation', () => {
    const runner = new DialogueRunner();
    runner.start(CHARSI_DIALOGUE, context(null));
    expect(runner.chooseById('leave')).toBeNull();
    expect(runner.active).toBe(false);
    expect(runner.nodeId).toBeNull();
  });

  it('advances a node with no choices, and stops at the end', () => {
    const tree: DialogueTree = {
      id: 't',
      npcId: 'n',
      speaker: 'N',
      entry: [{ node: 'a' }],
      nodes: [
        { id: 'a', text: 'One.', next: 'b' },
        { id: 'b', text: 'Two.' },
      ],
    };
    const runner = new DialogueRunner();
    runner.start(tree, context(null));
    expect(runner.advance()?.nodeId).toBe('b');
    expect(runner.advance()).toBeNull();
  });

  it('ignores a choice index that is not on screen', () => {
    const runner = new DialogueRunner();
    const opening = runner.start(CHARSI_DIALOGUE, context(null));
    expect(runner.choose(99)?.nodeId).toBe(opening?.nodeId);
    expect(runner.active).toBe(true);
  });

  it('ends cleanly rather than throwing on a missing node', () => {
    const tree: DialogueTree = {
      id: 't',
      npcId: 'n',
      speaker: 'N',
      entry: [{ node: 'missing' }],
      nodes: [{ id: 'a', text: 'One.' }],
    };
    const runner = new DialogueRunner();
    expect(runner.start(tree, context(null))).toBeNull();
    expect(runner.active).toBe(false);
  });

  it('finds a tree by NPC id', () => {
    expect(dialogueForNpc('gheed')?.speaker).toBe('Gheed');
    expect(dialogueForNpc('diablo')).toBeNull();
  });
});

describe('Akara reacts to the quest state', () => {
  /** Open a conversation with Akara and report which node she starts on. */
  function opening(quests: QuestSystem): string | null {
    const runner = new DialogueRunner();
    // Actions are wired through to the real quest system, so the conversation
    // genuinely advances the quest rather than pretending to.
    runner.setActionHandler((action) => {
      if (action.kind === 'offerQuest') quests.offer(action.questId);
      if (action.kind === 'acceptQuest') quests.accept(action.questId);
      if (action.kind === 'turnInQuest') quests.turnInAndReward(action.questId);
    });
    return runner.start(AKARA_DIALOGUE, context(quests))?.nodeId ?? null;
  }

  it('opens differently at every stage of the quest', () => {
    const quests = denQuests();
    const seen = new Map<string, string>();

    seen.set('inactive', opening(quests) ?? '');
    quests.offer(DEN_OF_EVIL_QUEST_ID);
    seen.set('offered', opening(quests) ?? '');
    quests.accept(DEN_OF_EVIL_QUEST_ID);
    seen.set('active', opening(quests) ?? '');
    quests.notifyReach('den-of-evil');
    quests.notifyKill('den-monster', 5);
    seen.set('complete', opening(quests) ?? '');
    quests.turnInAndReward(DEN_OF_EVIL_QUEST_ID);
    seen.set('rewarded', opening(quests) ?? '');

    expect(seen.get('inactive')).toBe('greeting');
    expect(seen.get('offered')).toBe('offer');
    expect(seen.get('active')).toBe('active');
    expect(seen.get('complete')).toBe('complete');
    expect(seen.get('turned-in')).toBeUndefined();
    expect(seen.get('rewarded')).toBe('turned-in');

    // Every stage must be distinct, or the branching is decorative.
    expect(new Set(seen.values()).size).toBe(seen.size);
  });

  it('offers and accepts the quest through the conversation alone', () => {
    const quests = denQuests();
    const runner = new DialogueRunner();
    runner.setActionHandler((action) => {
      if (action.kind === 'offerQuest') quests.offer(action.questId);
      if (action.kind === 'acceptQuest') quests.accept(action.questId);
    });

    runner.start(AKARA_DIALOGUE, context(quests));
    runner.chooseById('ask-evil');
    expect(quests.state(DEN_OF_EVIL_QUEST_ID)).toBe('offered');
    runner.chooseById('accept');
    expect(quests.state(DEN_OF_EVIL_QUEST_ID)).toBe('active');
  });

  it('lets the player decline and be offered again later', () => {
    const quests = denQuests();
    const runner = new DialogueRunner();
    runner.setActionHandler((action) => {
      if (action.kind === 'offerQuest') quests.offer(action.questId);
      if (action.kind === 'declineQuest') quests.decline(action.questId);
    });
    runner.start(AKARA_DIALOGUE, context(quests));
    runner.chooseById('ask-evil');
    runner.chooseById('decline');
    expect(quests.state(DEN_OF_EVIL_QUEST_ID)).toBe('inactive');
    expect(opening(quests)).toBe('greeting');
  });

  it('turns the quest in from the complete branch', () => {
    const quests = denQuests();
    quests.offer(DEN_OF_EVIL_QUEST_ID);
    quests.accept(DEN_OF_EVIL_QUEST_ID);
    quests.notifyReach('den-of-evil');
    quests.notifyKill('den-monster', 5);

    const runner = new DialogueRunner();
    runner.setActionHandler((action) => {
      if (action.kind === 'turnInQuest') quests.turnInAndReward(action.questId);
    });
    const view = runner.start(AKARA_DIALOGUE, context(quests));
    expect(view?.nodeId).toBe('complete');
    runner.chooseById('turn-in');
    expect(quests.state(DEN_OF_EVIL_QUEST_ID)).toBe('rewarded');
  });

  it('offers healing at every stage of the quest', () => {
    const quests = denQuests();
    const runner = new DialogueRunner();
    for (const stage of ['inactive', 'active', 'rewarded'] as const) {
      if (stage === 'active') {
        quests.offer(DEN_OF_EVIL_QUEST_ID);
        quests.accept(DEN_OF_EVIL_QUEST_ID);
      }
      if (stage === 'rewarded') {
        quests.notifyReach('den-of-evil');
        quests.notifyKill('den-monster', 5);
        quests.turnInAndReward(DEN_OF_EVIL_QUEST_ID);
      }
      const view = runner.start(AKARA_DIALOGUE, context(quests));
      expect(view?.choices.some((choice) => choice.id === 'heal'), stage).toBe(true);
      runner.end();
    }
  });
});
