/**
 * @module quest/NPC
 *
 * The five Rogue Encampment characters — Akara, Kashya, Charsi, Gheed and
 * Warriv — placed on the anchors the camp scene publishes, animated, and wired
 * to the dialogue system.
 *
 * ### Placement is the scene's business, not this module's
 *
 * `scene/RogueEncampment` stages the camp around where its inhabitants will
 * stand and exposes those spots as `CAMP_NPC_ANCHORS` with a `placeNpc(id,
 * object)` call. This module names an anchor and hands over a figure; it never
 * learns where the tents ended up. That is what lets the camp be re-laid-out
 * without touching a single NPC.
 *
 * `placeNpc` also parents the figure into the *zone's* subtree rather than the
 * scene root, so the NPCs die with the camp. An NPC parented to `ctx.scene`
 * survives travel and turns up standing in the middle of the Den of Evil.
 *
 * ### Animation
 *
 * Each NPC gets its own `AnimationGraph` over the shared 41-joint rig, fed a
 * zero velocity so it settles into `idle`, and played `interact` on being
 * spoken to. The graph is per-instance because a mixer is bound to one object;
 * the *clips* are shared, loaded once per asset key and cloned with
 * `SkeletonUtils`.
 *
 * ### Dialogue actions
 *
 * The runner reports actions; it does not perform them. This module installs a
 * handler that turns `acceptQuest` into a `QuestSystem` call and `openVendor`
 * into an event the UI picks up, resolving both through the `ServiceLocator` so
 * the whole thing degrades to conversation-only if the RPG layer is absent.
 */

import * as THREE from 'three/webgpu';
import { clone as cloneSkinned } from 'three/examples/jsm/utils/SkeletonUtils.js';

import { normaliseHeight } from '../ai/EnemyDirector';
import { AssetManagerKey, type AssetKey, type AssetManager } from '../assets/AssetManager';
import { AnimationGraph } from '../character/AnimationGraph';
import { PlayerKey, type PlayerController } from '../character/PlayerController';
import type { GameContext, GameModule } from '../core/types';
import { serviceKey } from '../core/ServiceLocator';
import { RpgSystemKey, type RpgSystem } from '../rpg/RpgSystem';
import { ZoneManagerKey, type ZoneManager } from '../world/ZoneManager';
import type { NpcAnchor } from '../world/Zone';
import { CAMP_DIALOGUE, dialogueForNpc } from './CampDialogue';
import {
  DialogueRunner,
  validateDialogueTree,
  type DialogueAction,
  type DialogueContext,
  type DialogueTree,
  type DialogueView,
} from './Dialogue';
import { QuestSystemKey, type QuestSystem } from './QuestSystem';

/* -------------------------------------------------------------------------- */
/* Events                                                                      */
/* -------------------------------------------------------------------------- */

declare module '../core/EventBus' {
  interface GameEvents {
    /** The NPC the player is close enough to talk to, or null when none. */
    'npc:prompt': { npcId: string | null; displayName: string; verb: string };
    /** A conversation opened, advanced, or closed (`view === null`). */
    'npc:dialogue': { npcId: string; view: DialogueView | null };
    /** The player asked to trade. The UI opens the vendor screen. */
    'npc:vendor': { npcId: string; vendorId: string };
    /** Every camp NPC has been built and placed. */
    'npc:ready': { count: number };
  }
}

/* -------------------------------------------------------------------------- */
/* Definitions                                                                 */
/* -------------------------------------------------------------------------- */

export type NpcService = 'quest' | 'vendor' | 'repair' | 'heal' | 'caravan';

export interface NpcDefinition {
  readonly id: string;
  readonly displayName: string;
  /** Anchor id in `CAMP_NPC_ANCHORS`. */
  readonly anchorId: string;
  /** Which rig to wear. All three share the Barbarian's 41-joint skeleton. */
  readonly asset: AssetKey;
  /** Metres, measured across the skinned meshes. */
  readonly height: number;
  readonly services: readonly NpcService[];
}

/**
 * The camp's inhabitants.
 *
 * Three rigs across five characters: the hooded rogue for the two Sisters, the
 * knight for the two who work with their hands, and the mage for Akara. It is
 * a small library and the reuse is visible if you look for it — but every one
 * of them stands in a distinct place, faces a distinct way and wears a distinct
 * height, which is what the eye actually reads at conversational distance.
 */
export const CAMP_NPCS: readonly NpcDefinition[] = [
  {
    id: 'akara',
    displayName: 'Akara',
    anchorId: 'akara',
    asset: 'character.mage',
    height: 1.68,
    services: ['quest', 'heal', 'vendor'],
  },
  {
    id: 'kashya',
    displayName: 'Kashya',
    anchorId: 'kashya',
    asset: 'character.rogue',
    height: 1.71,
    services: [],
  },
  {
    id: 'charsi',
    displayName: 'Charsi',
    anchorId: 'charsi',
    asset: 'character.knight',
    height: 1.74,
    services: ['vendor', 'repair'],
  },
  {
    id: 'gheed',
    displayName: 'Gheed',
    anchorId: 'gheed',
    asset: 'character.knight',
    height: 1.66,
    services: ['vendor'],
  },
  {
    id: 'warriv',
    displayName: 'Warriv',
    anchorId: 'warriv',
    asset: 'character.rogue',
    height: 1.76,
    services: ['caravan'],
  },
];

/* -------------------------------------------------------------------------- */
/* Instances                                                                   */
/* -------------------------------------------------------------------------- */

/** A placed, animated NPC. */
export interface NpcInstance {
  readonly definition: NpcDefinition;
  readonly object: THREE.Object3D;
  readonly graph: AnimationGraph | null;
  readonly position: THREE.Vector3;
  readonly anchor: NpcAnchor | null;
}

/** How close the player must stand to be offered a conversation, in metres. */
export const TALK_RADIUS = 3.2;

export const NpcSystemKey = serviceKey<NpcSystem>('quest.npcs');

/** The camp's zone id. NPCs are built when it loads and dropped when it does not. */
export const CAMP_ZONE_ID = 'encampment';

/** The subset of `RogueEncampment` this module needs. Structural, not nominal. */
interface NpcPlacer {
  anchor(id: string): NpcAnchor | null;
  placeNpc(id: string, object: THREE.Object3D): NpcAnchor | null;
}

function isPlacer(value: unknown): value is NpcPlacer {
  const candidate = value as Partial<NpcPlacer> | null;
  return (
    candidate !== null &&
    typeof candidate === 'object' &&
    typeof candidate.placeNpc === 'function' &&
    typeof candidate.anchor === 'function'
  );
}

export interface NpcSystemOptions {
  readonly definitions?: readonly NpcDefinition[];
  readonly trees?: readonly DialogueTree[];
  /** Skip model loading. Used by headless tests that only want the dialogue. */
  readonly models?: boolean;
}

export class NpcSystem implements GameModule {
  readonly name = 'quest.npcs';

  readonly runner = new DialogueRunner();

  readonly #definitions: readonly NpcDefinition[];
  readonly #trees: readonly DialogueTree[];
  readonly #models: boolean;
  readonly #instances: NpcInstance[] = [];
  readonly #unsubscribe: Array<() => void> = [];
  readonly #scratch = new THREE.Vector3();

  #ctx: GameContext | null = null;
  #building = false;
  #promptId: string | null = null;
  #talkingTo: string | null = null;

  constructor(options: NpcSystemOptions = {}) {
    this.#definitions = options.definitions ?? CAMP_NPCS;
    this.#trees = options.trees ?? CAMP_DIALOGUE;
    this.#models = options.models ?? true;
  }

  get npcs(): readonly NpcInstance[] {
    return this.#instances;
  }

  /** The NPC the player is currently talking to, or null. */
  get talkingTo(): string | null {
    return this.#talkingTo;
  }

  /** The NPC in range, or null. Drives the "Press E to talk" prompt. */
  get prompt(): string | null {
    return this.#promptId;
  }

  init(ctx: GameContext): void {
    this.#ctx = ctx;
    ctx.services.register(NpcSystemKey, this);
    this.runner.setActionHandler((action, context) => this.#handleAction(action, context));

    // Content validation at boot rather than at the moment a player takes an
    // untested branch. Five trees, one pass, and every dangling `next` in the
    // camp is on the console before the first frame.
    for (const tree of this.#trees) {
      for (const problem of validateDialogueTree(tree)) {
        console.warn(`[npc] dialogue problem: ${problem}`);
      }
    }

    this.#unsubscribe.push(
      ctx.events.on('zone:loaded', (payload) => {
        if (payload.zoneId === CAMP_ZONE_ID) void this.build(ctx);
        else this.#drop();
      }),
      ctx.events.on('zone:unloading', () => this.#drop()),
    );

    const zones = ctx.services.tryGet<ZoneManager>(ZoneManagerKey);
    if (zones?.active?.zoneId === CAMP_ZONE_ID) void this.build(ctx);
  }

  update(ctx: GameContext, dt: number): void {
    for (const npc of this.#instances) npc.graph?.update(dt);

    if (this.#talkingTo !== null) {
      // While a conversation is open the overlay owns the keyboard; the world
      // must not also read the interact key, or closing a dialogue with E
      // immediately reopens it.
      return;
    }

    const nearest = this.#nearestInRange();
    const nextId = nearest?.definition.id ?? null;
    if (nextId !== this.#promptId) {
      this.#promptId = nextId;
      ctx.events.emit('npc:prompt', {
        npcId: nextId,
        displayName: nearest?.definition.displayName ?? '',
        verb: 'talk to',
      });
    }
    if (nextId !== null && ctx.input.wasPressed('Interact')) this.talkTo(nextId);
  }

  dispose(): void {
    for (const off of this.#unsubscribe) off();
    this.#unsubscribe.length = 0;
    this.#drop();
    this.runner.setActionHandler(null);
    this.#ctx?.services.unregister(NpcSystemKey);
    this.#ctx = null;
  }

  /* -- construction -------------------------------------------------------- */

  /**
   * Load, clone and place every camp NPC.
   *
   * Idempotent and re-entrant safe: travelling camp -> moor -> camp calls this
   * again, and the `#building` guard stops two concurrent loads from producing
   * ten figures on five anchors.
   */
  async build(ctx: GameContext): Promise<number> {
    if (this.#building || this.#instances.length > 0) return this.#instances.length;
    this.#building = true;
    try {
      const zone = ctx.services.tryGet<ZoneManager>(ZoneManagerKey)?.active ?? null;
      const placer = isPlacer(zone) ? zone : null;
      if (placer === null) {
        console.warn('[npc] the active zone cannot place NPCs; the camp will be empty');
        return 0;
      }

      const assets = this.#models
        ? (ctx.services.tryGet<AssetManager>(AssetManagerKey) ?? null)
        : null;
      const sources = new Map<string, { scene: THREE.Object3D; clips: THREE.AnimationClip[] }>();
      if (assets !== null) {
        for (const key of new Set(this.#definitions.map((entry) => entry.asset))) {
          if (!assets.has(key)) {
            console.warn(`[npc] no asset "${key}"`);
            continue;
          }
          try {
            const gltf = await assets.loadGLTF(key);
            assets.pin(key);
            sources.set(key, { scene: gltf.scene, clips: [...gltf.animations] });
          } catch (error) {
            console.error(`[npc] could not load ${key}:`, error);
          }
        }
      }

      for (const definition of this.#definitions) {
        const source = sources.get(definition.asset);
        const root =
          source === undefined ? new THREE.Group() : cloneSkinned(source.scene);
        root.name = `npc.${definition.id}`;
        if (source !== undefined) normaliseHeight(root, definition.height);

        const anchor = placer.placeNpc(definition.anchorId, root);
        if (anchor === null) {
          console.warn(`[npc] no anchor "${definition.anchorId}" for ${definition.id}`);
          continue;
        }

        let graph: AnimationGraph | null = null;
        if (source !== undefined && source.clips.length > 0) {
          graph = new AnimationGraph(root, source.clips, { verbose: false });
          // Zero velocity, feet on the ground: the blend space settles on idle
          // and stays there, which is the whole animation budget for an NPC who
          // stands beside a tent.
          graph.setLocomotion({ x: 0, z: 0 }, true);
        }

        this.#instances.push({
          definition,
          object: root,
          graph,
          position: new THREE.Vector3(anchor.position.x, anchor.position.y, anchor.position.z),
          anchor,
        });
      }

      ctx.events.emit('npc:ready', { count: this.#instances.length });
      return this.#instances.length;
    } finally {
      this.#building = false;
    }
  }

  #drop(): void {
    if (this.#instances.length === 0) return;
    // The figures are parented into the zone's subtree, so the zone's own
    // teardown disposes them. All that is owed here is the mixers and the
    // references — holding a disposed `AnimationGraph` across a transition is
    // how a zone leak starts.
    for (const npc of this.#instances) npc.graph?.dispose();
    this.#instances.length = 0;
    this.#promptId = null;
    if (this.#talkingTo !== null) this.endDialogue();
  }

  /* -- conversation -------------------------------------------------------- */

  /** Look an NPC up by id. */
  find(npcId: string): NpcInstance | null {
    return this.#instances.find((entry) => entry.definition.id === npcId) ?? null;
  }

  /**
   * Open a conversation.
   *
   * Public and id-addressed rather than proximity-only, so the drive harness
   * and the tests can talk to Akara without walking to her.
   */
  talkTo(npcId: string): DialogueView | null {
    const ctx = this.#ctx;
    const tree = this.#trees.find((entry) => entry.npcId === npcId) ?? dialogueForNpc(npcId);
    if (ctx === null || tree === null) return null;

    this.#talkingTo = npcId;
    this.find(npcId)?.graph?.playAction('interact', { layer: 'upper', fadeIn: 0.12 });
    // Talking to someone is a quest objective type in its own right.
    ctx.services.tryGet<QuestSystem>(QuestSystemKey)?.notifyTalk(npcId);

    const view = this.runner.start(tree, this.context(npcId));
    ctx.events.emit('npc:dialogue', { npcId, view });
    if (view === null) this.#talkingTo = null;
    return view;
  }

  /** Take a choice by its index in the visible list. */
  choose(index: number): DialogueView | null {
    const npcId = this.#talkingTo;
    if (npcId === null) return null;
    const view = this.runner.choose(index);
    this.#ctx?.events.emit('npc:dialogue', { npcId, view });
    if (view === null) this.#talkingTo = null;
    return view;
  }

  /** Take a choice by id. Used by click handlers and tests. */
  chooseById(choiceId: string): DialogueView | null {
    const npcId = this.#talkingTo;
    if (npcId === null) return null;
    const view = this.runner.chooseById(choiceId);
    this.#ctx?.events.emit('npc:dialogue', { npcId, view });
    if (view === null) this.#talkingTo = null;
    return view;
  }

  /** Continue a node with no choices. */
  advance(): DialogueView | null {
    const npcId = this.#talkingTo;
    if (npcId === null) return null;
    const view = this.runner.advance();
    this.#ctx?.events.emit('npc:dialogue', { npcId, view });
    if (view === null) this.#talkingTo = null;
    return view;
  }

  endDialogue(): void {
    const npcId = this.#talkingTo;
    this.runner.end();
    this.#talkingTo = null;
    if (npcId !== null) this.#ctx?.events.emit('npc:dialogue', { npcId, view: null });
  }

  /** Build the context a condition or a dynamic line reads. */
  context(npcId: string): DialogueContext {
    const ctx = this.#ctx;
    const rpg = ctx?.services.tryGet<RpgSystem>(RpgSystemKey);
    const flags = new Set<string>();
    if (rpg !== undefined && rpg.character.brokenItems().length > 0) flags.add('hasBrokenGear');
    return {
      quests: ctx?.services.tryGet<QuestSystem>(QuestSystemKey) ?? null,
      npcId,
      characterLevel: rpg?.character.stats.level ?? 1,
      gold: rpg?.character.gold ?? 0,
      flags,
    };
  }

  /* -- internals ----------------------------------------------------------- */

  #handleAction(action: DialogueAction, context: DialogueContext): void {
    const ctx = this.#ctx;
    const quests = ctx?.services.tryGet<QuestSystem>(QuestSystemKey);
    const rpg = ctx?.services.tryGet<RpgSystem>(RpgSystemKey);

    switch (action.kind) {
      case 'offerQuest':
        quests?.offer(action.questId);
        break;
      case 'acceptQuest':
        // Offer first when the player skipped straight to accepting — a branch
        // that reaches `accept` without passing through `offer` is legal script
        // and must not silently do nothing.
        if (quests?.state(action.questId) === 'inactive') quests.offer(action.questId);
        quests?.accept(action.questId);
        break;
      case 'declineQuest':
        quests?.decline(action.questId);
        break;
      case 'turnInQuest':
        quests?.turnInAndReward(action.questId);
        break;
      case 'openVendor': {
        const vendor = rpg?.vendors.forNpc(action.npcId);
        if (vendor !== undefined && vendor !== null) {
          vendor.refresh();
          ctx?.events.emit('npc:vendor', { npcId: action.npcId, vendorId: vendor.id });
        }
        break;
      }
      case 'repair': {
        const vendor = rpg?.vendors.forNpc(context.npcId);
        if (vendor !== undefined && vendor !== null && rpg !== undefined) {
          vendor.repairAll(rpg.character);
        }
        break;
      }
      case 'heal':
        rpg?.healToFull();
        break;
      case 'travel':
        ctx?.services
          .tryGet<ZoneManager>(ZoneManagerKey)
          ?.travelTo(action.zoneId, action.entryPoint ?? null);
        break;
      case 'end':
        break;
      default:
        break;
    }
  }

  #nearestInRange(): NpcInstance | null {
    const ctx = this.#ctx;
    if (ctx === null || this.#instances.length === 0) return null;
    const player = ctx.services.tryGet<PlayerController>(PlayerKey);
    if (player === undefined) return null;

    this.#scratch.copy(player.position);
    let best: NpcInstance | null = null;
    let bestDistance = TALK_RADIUS * TALK_RADIUS;
    for (const npc of this.#instances) {
      const distance = npc.position.distanceToSquared(this.#scratch);
      if (distance <= bestDistance) {
        best = npc;
        bestDistance = distance;
      }
    }
    return best;
  }
}
