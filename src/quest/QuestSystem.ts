/**
 * @module quest/QuestSystem
 *
 * A general quest state machine: definitions, objectives, progress, rewards and
 * persistence, with no knowledge of any particular quest.
 *
 * ### The state machine
 *
 * ```
 * inactive ──offer──▶ offered ──accept──▶ active ──all objectives──▶ complete
 *                        │                   │                          │
 *                        └──decline──────────┘                       turnIn
 *                                 (back to inactive)                    │
 *                                                                    turnedIn
 *                                                                       │
 *                                                                    reward
 *                                                                       │
 *                                                                   rewarded
 * ```
 *
 * Six states rather than four because **turning a quest in and being paid for
 * it are different events**. The player walks back to Akara, the objective is
 * acknowledged, and *then* the reward lands — and if the reward is a skill
 * point, the UI has to be able to show that transition. Collapsing the two
 * makes "I turned it in and nothing happened" indistinguishable from "the
 * reward code threw".
 *
 * Every transition is validated. `accept` on an `inactive` quest fails and says
 * so, rather than quietly activating a quest the player was never offered —
 * which is exactly how a quest ends up completable before it is available.
 *
 * ### Progress
 *
 * Objectives are counters, always. A "reach the cave mouth" objective is a
 * counter with a required count of one; so is "talk to Akara". Modelling them
 * uniformly means the tracker UI, the save format and the completion test are
 * each written once.
 *
 * ### Rewards
 *
 * This module never touches a character. {@link QuestSystem.setRewardHandler}
 * takes a callback that is handed the reward payload, so the quest layer can be
 * unit tested with no RPG layer present and the RPG layer stays the only thing
 * that knows how to add experience.
 */

import { serviceKey } from '../core/ServiceLocator';
import type { EventBus } from '../core/EventBus';

/* -------------------------------------------------------------------------- */
/* Events                                                                      */
/* -------------------------------------------------------------------------- */

declare module '../core/EventBus' {
  interface GameEvents {
    /** A quest giver has offered a quest; the dialogue is showing the choice. */
    'quest:offered': { questId: string; title: string };
    /** The player accepted. The quest is now tracked. */
    'quest:accepted': { questId: string; title: string };
    /** An objective's counter moved. Fired once per increment, not per frame. */
    'quest:progress': {
      questId: string;
      objectiveId: string;
      current: number;
      required: number;
      description: string;
    };
    /** One objective reached its required count. */
    'quest:objectiveComplete': { questId: string; objectiveId: string; description: string };
    /** Every objective is done; the quest is ready to hand in. */
    'quest:complete': { questId: string; title: string };
    /** The player handed the quest in to its giver. */
    'quest:turnedIn': { questId: string; title: string };
    /** Rewards have been granted. */
    'quest:rewarded': {
      questId: string;
      title: string;
      experience: number;
      gold: number;
      skillPoints: number;
      statPoints: number;
    };
    /** An offered quest was declined and is available again. */
    'quest:declined': { questId: string };
  }
}

/* -------------------------------------------------------------------------- */
/* Definitions                                                                 */
/* -------------------------------------------------------------------------- */

export type QuestState = 'inactive' | 'offered' | 'active' | 'complete' | 'turnedIn' | 'rewarded';

/** Ordering for comparisons like "is this quest at least active". */
export const QUEST_STATE_ORDER: readonly QuestState[] = [
  'inactive',
  'offered',
  'active',
  'complete',
  'turnedIn',
  'rewarded',
];

/** Whether `state` is at or past `other` in the lifecycle. */
export function questStateAtLeast(state: QuestState, other: QuestState): boolean {
  return QUEST_STATE_ORDER.indexOf(state) >= QUEST_STATE_ORDER.indexOf(other);
}

export type ObjectiveType = 'kill' | 'reach' | 'talk';

export interface QuestObjective {
  readonly id: string;
  readonly type: ObjectiveType;
  /** Shown in the tracker, e.g. "Slay the evil within". */
  readonly description: string;
  /**
   * What the objective matches against: an enemy tag for `kill`, a location id
   * for `reach`, an NPC id for `talk`.
   */
  readonly target: string;
  /** How many are needed. Defaults to 1 for `reach` and `talk`. */
  readonly required: number;
  /**
   * Whether the tracker shows "3 of 20". False for objectives whose count is
   * always one, where "1 of 1" is noise.
   */
  readonly showCount?: boolean;
}

export interface QuestRewards {
  readonly experience?: number;
  readonly gold?: number;
  readonly skillPoints?: number;
  readonly statPoints?: number;
  /** Item generation requests, resolved by the reward handler. */
  readonly items?: readonly { readonly seed: number; readonly itemLevel: number; readonly baseId?: string; readonly quality?: string }[];
}

export interface QuestDefinition {
  readonly id: string;
  readonly title: string;
  /** One paragraph, shown in the quest log. */
  readonly summary: string;
  /** NPC id that offers the quest. */
  readonly giver: string;
  /** NPC id the quest is handed in to. Defaults to `giver`. */
  readonly turnInTo?: string;
  readonly objectives: readonly QuestObjective[];
  readonly rewards: QuestRewards;
}

/* -------------------------------------------------------------------------- */
/* Runtime state                                                               */
/* -------------------------------------------------------------------------- */

export interface ObjectiveProgress {
  readonly id: string;
  readonly current: number;
  readonly required: number;
  readonly complete: boolean;
  readonly description: string;
  readonly showCount: boolean;
}

export interface QuestStatus {
  readonly id: string;
  readonly title: string;
  readonly summary: string;
  readonly state: QuestState;
  readonly objectives: readonly ObjectiveProgress[];
  readonly giver: string;
  readonly turnInTo: string;
}

interface QuestRecord {
  readonly definition: QuestDefinition;
  state: QuestState;
  counters: Map<string, number>;
}

export interface QuestSnapshot {
  readonly quests: readonly {
    readonly id: string;
    readonly state: QuestState;
    readonly counters: Readonly<Record<string, number>>;
  }[];
}

/** What a reward handler is given. Items are requests, not instances. */
export interface RewardPayload {
  readonly questId: string;
  readonly title: string;
  readonly rewards: QuestRewards;
}

export type RewardHandler = (payload: RewardPayload) => void;

export const QuestSystemKey = serviceKey<QuestSystem>('quest.system');

/* -------------------------------------------------------------------------- */
/* The system                                                                  */
/* -------------------------------------------------------------------------- */

export class QuestSystem {
  readonly #quests = new Map<string, QuestRecord>();
  #events: EventBus | null = null;
  #reward: RewardHandler | null = null;

  /**
   * Attach an event bus.
   *
   * Optional, and checked on every emit: the quest system is exercised by unit
   * tests with no engine, and a state machine that requires a bus to advance is
   * a state machine that can only be tested through an engine.
   */
  bindEvents(events: EventBus | null): void {
    this.#events = events;
  }

  setRewardHandler(handler: RewardHandler | null): void {
    this.#reward = handler;
  }

  /** Register a definition. Re-registering the same id replaces the definition
   *  but keeps live progress, so a hot reload does not reset the player. */
  register(definition: QuestDefinition): void {
    const existing = this.#quests.get(definition.id);
    if (existing !== undefined) {
      this.#quests.set(definition.id, {
        definition,
        state: existing.state,
        counters: existing.counters,
      });
      return;
    }
    this.#quests.set(definition.id, {
      definition,
      state: 'inactive',
      counters: new Map(definition.objectives.map((objective) => [objective.id, 0])),
    });
  }

  /** Every registered quest id, in registration order. */
  get ids(): string[] {
    return Array.from(this.#quests.keys());
  }

  definition(questId: string): QuestDefinition | null {
    return this.#quests.get(questId)?.definition ?? null;
  }

  state(questId: string): QuestState {
    return this.#quests.get(questId)?.state ?? 'inactive';
  }

  /** Whether the quest is being tracked right now (accepted, not yet paid). */
  isTracked(questId: string): boolean {
    const state = this.state(questId);
    return state === 'active' || state === 'complete' || state === 'turnedIn';
  }

  /** Full status, for the quest log and the tracker. */
  status(questId: string): QuestStatus | null {
    const record = this.#quests.get(questId);
    if (record === undefined) return null;
    return {
      id: record.definition.id,
      title: record.definition.title,
      summary: record.definition.summary,
      state: record.state,
      giver: record.definition.giver,
      turnInTo: record.definition.turnInTo ?? record.definition.giver,
      objectives: record.definition.objectives.map((objective) => {
        const current = record.counters.get(objective.id) ?? 0;
        return {
          id: objective.id,
          current,
          required: objective.required,
          complete: current >= objective.required,
          description: objective.description,
          showCount: objective.showCount ?? objective.required > 1,
        };
      }),
    };
  }

  /** Every quest currently worth showing in the log. */
  tracked(): QuestStatus[] {
    const out: QuestStatus[] = [];
    for (const id of this.#quests.keys()) {
      if (!this.isTracked(id)) continue;
      const status = this.status(id);
      if (status !== null) out.push(status);
    }
    return out;
  }

  /* -- transitions --------------------------------------------------------- */

  /** `inactive -> offered`. */
  offer(questId: string): boolean {
    const record = this.#quests.get(questId);
    if (record === undefined || record.state !== 'inactive') return false;
    record.state = 'offered';
    this.#events?.emit('quest:offered', { questId, title: record.definition.title });
    return true;
  }

  /** `offered -> active`. Also completes the quest immediately if it has no
   *  objectives, which a pure "go and talk to X" quest legitimately does not. */
  accept(questId: string): boolean {
    const record = this.#quests.get(questId);
    if (record === undefined || record.state !== 'offered') return false;
    record.state = 'active';
    this.#events?.emit('quest:accepted', { questId, title: record.definition.title });
    this.#checkCompletion(record);
    return true;
  }

  /** `offered -> inactive`. The quest can be offered again later. */
  decline(questId: string): boolean {
    const record = this.#quests.get(questId);
    if (record === undefined || record.state !== 'offered') return false;
    record.state = 'inactive';
    this.#events?.emit('quest:declined', { questId });
    return true;
  }

  /** `complete -> turnedIn`. Refused unless every objective is done. */
  turnIn(questId: string): boolean {
    const record = this.#quests.get(questId);
    if (record === undefined || record.state !== 'complete') return false;
    record.state = 'turnedIn';
    this.#events?.emit('quest:turnedIn', { questId, title: record.definition.title });
    return true;
  }

  /**
   * `turnedIn -> rewarded`, granting the rewards through the handler.
   *
   * The state is advanced *before* the handler runs, so a handler that throws
   * cannot leave the quest in a state where the player can claim again.
   */
  reward(questId: string): boolean {
    const record = this.#quests.get(questId);
    if (record === undefined || record.state !== 'turnedIn') return false;
    record.state = 'rewarded';
    const rewards = record.definition.rewards;
    this.#reward?.({ questId, title: record.definition.title, rewards });
    this.#events?.emit('quest:rewarded', {
      questId,
      title: record.definition.title,
      experience: rewards.experience ?? 0,
      gold: rewards.gold ?? 0,
      skillPoints: rewards.skillPoints ?? 0,
      statPoints: rewards.statPoints ?? 0,
    });
    return true;
  }

  /** Convenience: `turnIn` then `reward`, the normal path through dialogue. */
  turnInAndReward(questId: string): boolean {
    if (!this.turnIn(questId)) return false;
    return this.reward(questId);
  }

  /* -- progress ------------------------------------------------------------ */

  /**
   * Add to an objective's counter.
   *
   * Only advances an `active` quest — progress recorded before the player
   * accepted, or after they handed it in, is silently ignored rather than
   * banked. Banking it means a player who kills half the Den before talking to
   * Akara gets a quest that is already half done, which reads as a bug.
   */
  progress(questId: string, objectiveId: string, amount = 1): boolean {
    const record = this.#quests.get(questId);
    if (record === undefined || record.state !== 'active') return false;
    const objective = record.definition.objectives.find((entry) => entry.id === objectiveId);
    if (objective === undefined) return false;

    const before = record.counters.get(objectiveId) ?? 0;
    if (before >= objective.required) return false;
    const after = Math.min(objective.required, before + Math.max(0, Math.round(amount)));
    if (after === before) return false;
    record.counters.set(objectiveId, after);

    this.#events?.emit('quest:progress', {
      questId,
      objectiveId,
      current: after,
      required: objective.required,
      description: objective.description,
    });
    if (after >= objective.required) {
      this.#events?.emit('quest:objectiveComplete', {
        questId,
        objectiveId,
        description: objective.description,
      });
    }
    this.#checkCompletion(record);
    return true;
  }

  /**
   * Report a world event to every active quest.
   *
   * The three notify helpers are the only surface gameplay code uses: the Den
   * of Evil quest does not subscribe to `combat:death` itself, it is told that
   * something with a given tag died and works out whether it cares.
   */
  notifyKill(tag: string, count = 1): number {
    return this.#notify('kill', tag, count);
  }

  notifyReach(locationId: string): number {
    return this.#notify('reach', locationId, 1);
  }

  notifyTalk(npcId: string): number {
    return this.#notify('talk', npcId, 1);
  }

  #notify(type: ObjectiveType, target: string, count: number): number {
    let advanced = 0;
    for (const record of this.#quests.values()) {
      if (record.state !== 'active') continue;
      for (const objective of record.definition.objectives) {
        if (objective.type !== type || objective.target !== target) continue;
        if (this.progress(record.definition.id, objective.id, count)) advanced++;
      }
    }
    return advanced;
  }

  #checkCompletion(record: QuestRecord): void {
    if (record.state !== 'active') return;
    for (const objective of record.definition.objectives) {
      if ((record.counters.get(objective.id) ?? 0) < objective.required) return;
    }
    record.state = 'complete';
    this.#events?.emit('quest:complete', {
      questId: record.definition.id,
      title: record.definition.title,
    });
  }

  /* -- persistence --------------------------------------------------------- */

  toJSON(): QuestSnapshot {
    const quests = Array.from(this.#quests.values()).map((record) => {
      const counters: Record<string, number> = {};
      for (const [id, value] of record.counters) counters[id] = value;
      return { id: record.definition.id, state: record.state, counters };
    });
    return { quests };
  }

  /**
   * Restore progress.
   *
   * Definitions must already be registered — they are code, not save data. A
   * saved quest whose definition has since been removed is skipped, and a saved
   * counter for an objective that no longer exists is dropped, so a content
   * change cannot make an old save unloadable.
   */
  load(snapshot: QuestSnapshot): void {
    for (const entry of snapshot.quests) {
      const record = this.#quests.get(entry.id);
      if (record === undefined) continue;
      record.state = QUEST_STATE_ORDER.includes(entry.state) ? entry.state : 'inactive';
      record.counters = new Map(
        record.definition.objectives.map((objective) => [
          objective.id,
          Math.max(0, Math.min(objective.required, Math.round(entry.counters[objective.id] ?? 0))),
        ]),
      );
    }
  }

  /** Reset everything to `inactive` with zeroed counters. New game. */
  reset(): void {
    for (const record of this.#quests.values()) {
      record.state = 'inactive';
      record.counters = new Map(record.definition.objectives.map((o) => [o.id, 0]));
    }
  }
}
