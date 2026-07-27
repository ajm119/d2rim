/**
 * @module quest/DenOfEvil
 *
 * Act I's first quest, end to end: Akara offers it, the objective is clearing
 * every monster in the Den, and returning to Akara pays a skill point — as it
 * does in Diablo II.
 *
 * ### The count is real
 *
 * The kill objective's required count is **not** a hand-written number. The Den
 * is procedurally generated; `world/DungeonGenerator` decides how many
 * skeletons it contains, and that number varies with the seed. So the quest
 * definition is registered with a provisional count and then *re-registered*
 * with the generator's actual spawn count the first time the Den loads —
 * `QuestSystem.register` deliberately preserves live progress, which is what
 * makes that safe mid-quest.
 *
 * The alternative, hard-coding twenty, produces the single most corrosive bug a
 * quest can have: a tracker that reads "18 of 20" in an empty cave. The player
 * cannot tell whether they missed two skeletons or the game is lying, and
 * either way they walk the whole dungeon again.
 *
 * ### Two objectives, two types
 *
 * "Find the Den of Evil" is a `reach` objective satisfied by the zone loading;
 * "Slay every monster within" is a `kill` objective driven by `combat:death`.
 * The second cannot progress before the first, because there is nothing to kill
 * outside the Den that carries the tag.
 *
 * ### What this module does not do
 *
 * It does not grant the reward. `QuestSystem` calls the reward handler that
 * `rpg/RpgSystem` installed, so "a skill point" is defined in exactly one place
 * — next to the character that receives it.
 */

import type { GameContext, GameModule } from '../core/types';
import { ZoneManagerKey, type ZoneManager } from '../world/ZoneManager';
import { QuestSystemKey, type QuestDefinition, type QuestSystem } from './QuestSystem';

/** The quest id. Referenced by the dialogue scripts and the UI. */
export const DEN_OF_EVIL_QUEST_ID = 'den-of-evil';

/** Objective ids. Stable across saves — do not rename without a migration. */
export const DEN_REACH_OBJECTIVE = 'find-den';
export const DEN_KILL_OBJECTIVE = 'clear-den';

/** The tag a Den monster's death is reported under. */
export const DEN_MONSTER_TAG = 'den-monster';

/** The location id the `reach` objective matches. */
export const DEN_LOCATION_TAG = 'den-of-evil';

/** Zone id of the cave, as registered with `ZoneManager`. */
export const DEN_ZONE_ID = 'denOfEvil';

/**
 * The count used before the cave has ever been generated.
 *
 * It is what the quest log shows while the player is still in the camp, and it
 * is deliberately close to the generator's typical output so the number does
 * not visibly jump the moment they walk in.
 */
export const PROVISIONAL_MONSTER_COUNT = 20;

/** Build the definition for a Den containing `monsterCount` monsters. */
export function createDenOfEvilQuest(monsterCount: number): QuestDefinition {
  const required = Math.max(1, Math.round(monsterCount));
  return {
    id: DEN_OF_EVIL_QUEST_ID,
    title: 'Den of Evil',
    summary:
      'Akara asks that the cave beneath the Blood Moor be cleared of every last one of its ' +
      'inhabitants. Nothing that walks in there may be left walking.',
    giver: 'akara',
    turnInTo: 'akara',
    objectives: [
      {
        id: DEN_REACH_OBJECTIVE,
        type: 'reach',
        description: 'Find the Den of Evil',
        target: DEN_LOCATION_TAG,
        required: 1,
        showCount: false,
      },
      {
        id: DEN_KILL_OBJECTIVE,
        type: 'kill',
        description: 'Slay every monster within',
        target: DEN_MONSTER_TAG,
        required,
        showCount: true,
      },
    ],
    rewards: {
      // D2's own reward for this quest. One point, and it matters at level 4.
      skillPoints: 1,
      experience: 900,
      gold: 120,
    },
  };
}

/**
 * Wires the quest to the world.
 *
 * Registered as a `GameModule` so it gets the same lifecycle as everything
 * else; it holds no scene state and disposes cleanly.
 */
export class DenOfEvilQuest implements GameModule {
  readonly name = 'quest.denOfEvil';

  readonly #unsubscribe: Array<() => void> = [];
  #ctx: GameContext | null = null;
  #quests: QuestSystem | null = null;
  #currentZone: string | null = null;
  #monsterCount = PROVISIONAL_MONSTER_COUNT;
  #countResolved = false;

  /** The number of monsters the quest is currently counting toward. */
  get monsterCount(): number {
    return this.#monsterCount;
  }

  /** Whether the count has been taken from a real generated cave yet. */
  get countResolved(): boolean {
    return this.#countResolved;
  }

  init(ctx: GameContext): void {
    this.#ctx = ctx;
    const quests = ctx.services.tryGet<QuestSystem>(QuestSystemKey);
    if (quests === undefined) {
      console.warn('[quest] no QuestSystem registered; the Den of Evil quest is inert');
      return;
    }
    this.#quests = quests;
    quests.register(createDenOfEvilQuest(this.#monsterCount));

    this.#unsubscribe.push(
      ctx.events.on('zone:loaded', (payload) => this.#onZoneLoaded(payload.zoneId)),
      ctx.events.on('combat:death', (payload) => {
        if (payload.faction !== 'enemy') return;
        this.reportKill();
      }),
    );

    // The zone may already be loaded when this module initialises — the manager
    // loads its start zone during its own `init`, which runs first.
    const zones = ctx.services.tryGet<ZoneManager>(ZoneManagerKey);
    const active = zones?.active?.zoneId ?? null;
    if (active !== null) this.#onZoneLoaded(active);
  }

  dispose(): void {
    for (const off of this.#unsubscribe) off();
    this.#unsubscribe.length = 0;
    this.#quests = null;
    this.#ctx = null;
  }

  /**
   * Report one Den monster slain.
   *
   * Public so the drive harness can advance the quest without staging twenty
   * real fights, and so a test can drive it with no engine at all. It applies
   * the same zone guard the event path does, which is what stops a Blood Moor
   * kill from counting.
   */
  reportKill(): boolean {
    if (this.#currentZone !== DEN_ZONE_ID) return false;
    return (this.#quests?.notifyKill(DEN_MONSTER_TAG) ?? 0) > 0;
  }

  /**
   * Adopt the generated cave's real spawn count.
   *
   * Re-registers the definition, which keeps the counters. Called on the first
   * Den load; subsequent loads of the same session re-read it, because the cave
   * is regenerated per load and a second visit can legitimately hold a
   * different number.
   */
  resolveMonsterCount(count: number): void {
    const resolved = Math.max(1, Math.round(count));
    this.#monsterCount = resolved;
    this.#countResolved = true;
    this.#quests?.register(createDenOfEvilQuest(resolved));
  }

  #onZoneLoaded(zoneId: string): void {
    this.#currentZone = zoneId;
    if (zoneId !== DEN_ZONE_ID) return;

    const zones = this.#ctx?.services.tryGet<ZoneManager>(ZoneManagerKey);
    const spawns = zones?.active?.enemySpawns?.length ?? 0;
    if (spawns > 0) this.resolveMonsterCount(spawns);

    // Arriving *is* the reach objective. Fired after the count is adopted so a
    // tracker that redraws on progress shows the real total immediately.
    this.#quests?.notifyReach(DEN_LOCATION_TAG);
  }
}
