/**
 * @module world/ZoneManager
 *
 * Owns which area of Act I is loaded, and the transition between areas.
 *
 * ## Why the manager hosts zones rather than the engine
 *
 * {@link Engine} keys its module registry by name and runs the phases over a
 * live array. Loading a zone from inside a frame would therefore mean mutating
 * that array mid-iteration, and unloading one would mean removing a module whose
 * `dispose` runs while its neighbours are half-way through their own update.
 *
 * So zones are hosted here instead: `ZoneManager` is a single ordinary module,
 * and it forwards `fixedUpdate` / `update` / `lateUpdate` to whichever zone is
 * active. The engine sees one module with a stable lifetime, and swapping the
 * world underneath it is a normal operation rather than a special case.
 *
 * Zones are registered as **factories**, not instances. An unloaded zone is
 * disposed and gone; re-entering it constructs a fresh one. That is the only
 * arrangement where "unload" can mean what it says — a retained instance holds
 * its geometries, its materials and its `#disposables` alive for the whole
 * session, which is exactly the leak this module exists to prevent.
 *
 * ## What a transition has to clean up
 *
 * Five things leak across a naive transition, and all five are handled here
 * because a zone cannot be trusted to remember all of them:
 *
 * | resource        | how it is reclaimed                                      |
 * |-----------------|----------------------------------------------------------|
 * | geometries      | {@link disposeZoneTree} walks the detached subtree        |
 * | materials       | ditto, including array materials, skipping shared ones    |
 * | textures        | ditto, every texture slot on every disposed material      |
 * | physics colliders | the manager snapshots the collider registry across the build and removes exactly the delta |
 * | enemies         | the zone's {@link EnemyDirector} is disposed with the zone |
 *
 * Materials and textures that came out of {@link AssetManager} are *shared* and
 * must not be disposed — the next zone will ask for the same barrel. They are
 * identified by `userData.shared`, which {@link markShared} stamps, and skipped.
 *
 * ## Ordering
 *
 * `ZoneManager` must be registered after `PhysicsWorld` (its zones build
 * colliders during `init`) and before `PlayerController` (which asks the world
 * where the ground is during *its* `init`). Enemies are the exception: they need
 * `CombatSystem`, which registers after this module, so enemy spawning for the
 * start zone is deferred to the first `update`. Travelling later has no such
 * problem, because by then everything is up.
 */

import { EnemyDirector, type SpawnPoint } from '../ai/EnemyDirector';
import { CombatKey, type CombatSystem } from '../combat/CombatSystem';
import { serviceKey } from '../core/ServiceLocator';
import type { GameContext, GameModule } from '../core/types';
import { PhysicsWorldKey, type ColliderRecord, type PhysicsWorld } from '../physics/PhysicsWorld';
import { PlayerKey, type PlayerController } from '../character/PlayerController';
import { buildPortalColliders } from './Portal';
import {
  disposeZoneTree,
  measureZone,
  resolveEntryPoint,
  type Zone,
  type ZoneEntryPoint,
} from './Zone';

declare module '../core/EventBus' {
  interface GameEvents {
    /** A zone has begun loading. */
    'zone:loadStart': { zoneId: string };
    /** Progress within a load, `0..1`. `phase` is a human-readable stage name. */
    'zone:loadProgress': { zoneId: string; phase: string; progress: number };
    /** A zone finished loading and is now the active zone. */
    'zone:loaded': { zoneId: string; entryPoint: string | null; millis: number };
    /** A zone is about to be torn down. Last chance to release references to it. */
    'zone:unloading': { zoneId: string };
    /** A zone has been torn down and its resources reclaimed. */
    'zone:unloaded': { zoneId: string; collidersRemoved: number; disposed: number };
    /** A `travelTo` has begun; the screen is fading out. */
    'zone:travelStart': { from: string | null; to: string; entryPoint: string | null };
    /** A `travelTo` has completed; the screen has faded back in. */
    'zone:travelEnd': { zoneId: string; entryPoint: string | null };
  }
}

export const ZoneManagerKey = serviceKey<ZoneManager>('world.zones');

/** Constructs a fresh zone. Called once per load. */
export type ZoneFactory = () => Zone;

export interface ZoneManagerOptions {
  /** Zone id to load during `init`. */
  readonly startZone?: string;
  /** Entry point within the start zone. Defaults to its first. */
  readonly startEntry?: string;
  /**
   * Fade duration each way, in seconds. `0` disables the overlay entirely.
   *
   * Paced off the wall clock rather than off `ctx.time`, and deliberately: the
   * screen is black for the duration of an `await`, during which no frames are
   * being stepped at all, so a frame-driven fade would never advance. The fade
   * is presentation; nothing in the simulation reads it, so determinism is
   * untouched. Harnesses set `0` and skip the wait.
   */
  readonly fadeSeconds?: number;
  /** Spawn enemies. Harnesses that want an empty zone set `false`. */
  readonly enemies?: boolean;
}

/** What a completed load produced, for diagnostics and the drive harness. */
export interface ZoneLoadReport {
  readonly zoneId: string;
  readonly millis: number;
  /** Colliders the zone's `buildColliders` added. */
  readonly colliders: number;
  /** Meshes under the zone root. A stand-in for the draw-call count. */
  readonly renderables: number;
  /** Triangles under the zone root, counting instances. */
  readonly triangles: number;
  readonly enemies: number;
}

export class ZoneManager implements GameModule {
  readonly name = 'world.zones';

  readonly #factories = new Map<string, ZoneFactory>();
  readonly #options: Required<ZoneManagerOptions>;

  #ctx: GameContext | null = null;
  #zone: Zone | null = null;
  #director: EnemyDirector | null = null;
  /** Colliders created by the active zone, to be removed on unload. */
  #zoneColliders: ColliderRecord[] = [];
  #fade: HTMLElement | null = null;
  #travelling: Promise<void> | null = null;
  #report: ZoneLoadReport | null = null;
  /** Set when a zone loaded before `CombatSystem` existed; see the module header. */
  #pendingEnemies: readonly SpawnPoint[] | null = null;
  #pendingEntry: ZoneEntryPoint | null = null;
  #disposed = false;

  constructor(options: ZoneManagerOptions = {}) {
    this.#options = {
      startZone: options.startZone ?? '',
      startEntry: options.startEntry ?? '',
      fadeSeconds: options.fadeSeconds ?? 0.35,
      enemies: options.enemies ?? true,
    };
  }

  /* -- accessors ---------------------------------------------------------- */

  /** The live zone, or `null` before the first load and during a transition. */
  get active(): Zone | null {
    return this.#zone;
  }

  get activeId(): string | null {
    return this.#zone?.zoneId ?? null;
  }

  /** Ids of every registered zone, in registration order. */
  get registered(): string[] {
    return Array.from(this.#factories.keys());
  }

  /** What the last completed load produced. */
  get report(): ZoneLoadReport | null {
    return this.#report;
  }

  /** Whether a `travelTo` is in flight. */
  get travelling(): boolean {
    return this.#travelling !== null;
  }

  /** The enemy director for the active zone, if it has one. */
  get director(): EnemyDirector | null {
    return this.#director;
  }

  /** Colliders currently owned by the active zone. */
  get zoneColliders(): readonly ColliderRecord[] {
    return this.#zoneColliders;
  }

  /* -- registry ----------------------------------------------------------- */

  /**
   * Register a zone factory.
   *
   * Registering is free — no geometry is built and no asset is fetched until the
   * zone is actually travelled to — so every zone in the act can be registered
   * at boot without paying for any of them.
   */
  register(zoneId: string, factory: ZoneFactory): this {
    if (this.#factories.has(zoneId)) {
      throw new Error(`[ZoneManager] zone "${zoneId}" is already registered`);
    }
    this.#factories.set(zoneId, factory);
    return this;
  }

  has(zoneId: string): boolean {
    return this.#factories.has(zoneId);
  }

  /* -- lifecycle ---------------------------------------------------------- */

  async init(ctx: GameContext): Promise<void> {
    this.#ctx = ctx;
    ctx.services.register(ZoneManagerKey, this);
    this.#buildFade();

    const start = this.#options.startZone;
    if (start === '') return;
    if (!this.#factories.has(start)) {
      console.error(
        `[ZoneManager] start zone "${start}" is not registered ` +
          `(registered: ${this.registered.join(', ') || 'none'})`,
      );
      return;
    }
    await this.#load(start, this.#options.startEntry === '' ? null : this.#options.startEntry);
  }

  fixedUpdate(ctx: GameContext, dt: number): void {
    this.#zone?.fixedUpdate?.(ctx, dt);
    // The director has to be driven on all three phases, not two. Everything an
    // enemy *decides* — perception, the behaviour tree, steering, the attack
    // state machine — lives in `EnemyBase.fixedUpdate`; `update` only poses the
    // model and `lateUpdate` only sweeps the weapon. Without this line the
    // skeletons render, blend and cull correctly and never think, which reads
    // as "the AI is broken" rather than as a missing lifecycle call.
    this.#director?.fixedUpdate(ctx, dt);
  }

  update(ctx: GameContext, dt: number): void {
    // The start zone's enemies could not be spawned during `init` because
    // `CombatSystem` had not registered yet. This is the first moment it has.
    if (this.#pendingEnemies !== null) {
      const spawns = this.#pendingEnemies;
      this.#pendingEnemies = null;
      void this.#spawnEnemies(spawns);
    }
    if (this.#pendingEntry !== null) {
      const entry = this.#pendingEntry;
      this.#pendingEntry = null;
      this.#placePlayer(entry);
    }
    this.#zone?.update?.(ctx, dt);
    this.#director?.update(ctx, dt);
  }

  lateUpdate(ctx: GameContext, dt: number): void {
    this.#zone?.lateUpdate?.(ctx, dt);
    this.#director?.lateUpdate();
  }

  dispose(): void {
    this.#disposed = true;
    this.#unload();
    this.#fade?.remove();
    this.#fade = null;
    this.#ctx?.services.unregister(ZoneManagerKey);
    this.#ctx = null;
  }

  /* -- travel ------------------------------------------------------------- */

  /**
   * Fade out, unload the current zone, load `zoneId`, put the player on
   * `entryPointId`, fade in.
   *
   * Serialised: a second call while one is in flight awaits the first and then
   * runs, so a player mashing the interact key inside a portal volume gets one
   * transition rather than a race between two teardowns.
   */
  async travelTo(zoneId: string, entryPointId: string | null = null): Promise<void> {
    const previous = this.#travelling;
    const run = (async (): Promise<void> => {
      if (previous !== null) await previous.catch(() => undefined);
      await this.#travel(zoneId, entryPointId);
    })();
    this.#travelling = run;
    try {
      await run;
    } finally {
      if (this.#travelling === run) this.#travelling = null;
    }
  }

  async #travel(zoneId: string, entryPointId: string | null): Promise<void> {
    const ctx = this.#ctx;
    if (ctx === null || this.#disposed) return;
    if (!this.#factories.has(zoneId)) {
      console.error(`[ZoneManager] cannot travel to unregistered zone "${zoneId}"`);
      return;
    }

    const from = this.activeId;
    ctx.events.emit('zone:travelStart', { from, to: zoneId, entryPoint: entryPointId });

    // The player is frozen for the whole transition. Without this he keeps his
    // input for the frames either side of the teleport and arrives in the new
    // zone already walking, which reads as the controls having stuck.
    const player = ctx.services.tryGet<PlayerController>(PlayerKey);
    player?.setEnabled(false);

    await this.#setFade(1);
    this.#unload();
    await this.#load(zoneId, entryPointId);
    await this.#setFade(0);

    player?.setEnabled(true);
    ctx.events.emit('zone:travelEnd', { zoneId, entryPoint: entryPointId });
  }

  /* -- load / unload ------------------------------------------------------ */

  async #load(zoneId: string, entryPointId: string | null): Promise<void> {
    const ctx = this.#ctx;
    const factory = this.#factories.get(zoneId);
    if (ctx === null || factory === undefined) return;

    const startedAt = Date.now();
    ctx.events.emit('zone:loadStart', { zoneId });
    const progress = (phase: string, value: number): void => {
      ctx.events.emit('zone:loadProgress', { zoneId, phase, progress: value });
    };

    progress('construct', 0.05);
    const zone = factory();
    this.#zone = zone;

    progress('build', 0.15);
    await zone.init(ctx);
    progress('colliders', 0.7);

    /* -- physics --------------------------------------------------------- */

    const physics = ctx.services.tryGet<PhysicsWorld>(PhysicsWorldKey);
    let colliders = 0;
    if (physics !== undefined && physics.ready) {
      // Snapshot before, diff after. The zone builds through the same
      // `PhysicsWorld.addCollider` every other system uses, so the manager can
      // reclaim its colliders without the zone having to report them — and
      // therefore without a zone being able to forget to.
      const before = new Set(physics.colliders.map((record) => record.id));
      zone.buildColliders?.(physics, ctx);
      // Inside the window on purpose — see the header of `world/Portal`.
      buildPortalColliders(physics, zone.portals);
      this.#zoneColliders = physics.colliders.filter((record) => !before.has(record.id));
      colliders = this.#zoneColliders.length;
      // Nothing built above is visible to a raycast until the query pipeline is
      // rebuilt, and the very next thing that happens is the player being asked
      // to stand on it.
      physics.syncQueries();
    } else {
      console.error(`[ZoneManager] no physics world; "${zoneId}" will not be solid`);
    }

    progress('place', 0.85);

    /* -- player ---------------------------------------------------------- */

    const entry = resolveEntryPoint(zone, entryPointId);
    if (entry !== null) {
      // During boot the player module has not initialised yet, so there is
      // nothing to place; defer to the first update, by which time it exists.
      if (ctx.services.has(PlayerKey)) this.#placePlayer(entry);
      else this.#pendingEntry = entry;
    }

    /* -- enemies --------------------------------------------------------- */

    const spawns = zone.enemySpawns ?? [];
    if (this.#options.enemies && spawns.length > 0) {
      if (ctx.services.has(CombatKey)) await this.#spawnEnemies(spawns);
      else this.#pendingEnemies = spawns;
    }

    const measured = measureZone(zone.root);
    this.#report = {
      zoneId,
      millis: Date.now() - startedAt,
      colliders,
      renderables: measured.renderables,
      triangles: measured.triangles,
      enemies: this.#director?.enemies.length ?? 0,
    };
    progress('ready', 1);
    ctx.events.emit('zone:loaded', {
      zoneId,
      entryPoint: entry?.id ?? null,
      millis: this.#report.millis,
    });
    console.info(
      `[ZoneManager] "${zoneId}" ready in ${this.#report.millis} ms: ` +
        `${measured.renderables} meshes, ${measured.triangles} tris, ` +
        `${colliders} colliders, ${this.#report.enemies} enemies`,
    );
  }

  #unload(): void {
    const ctx = this.#ctx;
    const zone = this.#zone;
    if (ctx === null || zone === null) return;

    ctx.events.emit('zone:unloading', { zoneId: zone.zoneId });

    // Enemies first: they hold Rapier bodies and combat-target registrations,
    // and disposing the physics colliders underneath a live enemy is the kind of
    // teardown order that produces a WASM trap rather than an exception.
    this.#director?.dispose();
    this.#director = null;

    const physics = ctx.services.tryGet<PhysicsWorld>(PhysicsWorldKey);
    let removed = 0;
    if (physics !== undefined && physics.ready) {
      for (const record of this.#zoneColliders) {
        physics.removeCollider(record);
        removed++;
      }
      physics.syncQueries();
    }
    this.#zoneColliders = [];

    // The zone's own `dispose` releases the things only it knows about — light
    // handles taken from `LightingService`, fog volumes, uniforms. The tree walk
    // then reclaims the GPU resources, including anything the zone forgot.
    zone.dispose?.();
    const disposed = disposeZoneTree(zone.root);

    this.#zone = null;
    this.#pendingEnemies = null;
    this.#pendingEntry = null;
    ctx.events.emit('zone:unloaded', {
      zoneId: zone.zoneId,
      collidersRemoved: removed,
      disposed,
    });
  }

  /* -- helpers ------------------------------------------------------------ */

  async #spawnEnemies(spawns: readonly SpawnPoint[]): Promise<void> {
    const ctx = this.#ctx;
    if (ctx === null || this.#disposed) return;
    const combat = ctx.services.tryGet<CombatSystem>(CombatKey);
    if (combat === undefined) {
      console.warn('[ZoneManager] no combat system; the zone will be unpopulated');
      return;
    }
    const director = new EnemyDirector({ spawns });
    this.#director = director;
    await director.init(ctx);
  }

  /**
   * Put the player on an entry point.
   *
   * The authored `y` is only a hint: the standing height is re-resolved against
   * the physics world, and preferentially through `findClearSpot`, so an entry
   * point that a later art pass happened to bury under a crate still lands the
   * player on open ground next to it rather than inside it.
   */
  #placePlayer(entry: ZoneEntryPoint): void {
    const ctx = this.#ctx;
    if (ctx === null) return;
    const player = ctx.services.tryGet<PlayerController>(PlayerKey);
    if (player === undefined) return;

    const physics = ctx.services.tryGet<PhysicsWorld>(PhysicsWorldKey);
    let { x, y, z } = entry.position;
    if (physics !== undefined && physics.ready) {
      const clear = physics.findClearSpot(x, z, 0.42, player.height, 5);
      if (clear !== null) {
        x = clear.x;
        y = clear.y;
        z = clear.z;
      } else {
        const ground = physics.groundHeight(x, z);
        if (ground !== null) y = ground;
        else {
          console.warn(
            `[ZoneManager] entry point "${entry.id}" has no ground under it; ` +
              'using the authored height',
          );
        }
      }
    }
    // A few centimetres of clearance so the first character-controller move
    // resolves downward onto the collider instead of starting inside it.
    player.teleport(x, y + 0.06, z);
  }

  /* -- fade --------------------------------------------------------------- */

  #buildFade(): void {
    if (this.#options.fadeSeconds <= 0) return;
    if (typeof document === 'undefined') return;
    const element = document.createElement('div');
    element.setAttribute(
      'style',
      'position:fixed;inset:0;z-index:60;pointer-events:none;background:#000;opacity:0;' +
        `transition:opacity ${this.#options.fadeSeconds}s ease-in-out;`,
    );
    element.dataset['d2rim'] = 'zone-fade';
    document.body.appendChild(element);
    this.#fade = element;
  }

  /** Drive the overlay to `target` and wait out the transition. */
  async #setFade(target: number): Promise<void> {
    const seconds = this.#options.fadeSeconds;
    if (seconds <= 0) return;
    if (this.#fade !== null) this.#fade.style.opacity = String(target);
    await new Promise<void>((resolve) => {
      setTimeout(resolve, seconds * 1000);
    });
  }
}
