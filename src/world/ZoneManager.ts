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

import { EnemyDirector } from '../ai/EnemyDirector';
import { CombatKey } from '../combat/CombatSystem';
import { serviceKey } from '../core/ServiceLocator';
import type { GameContext, GameModule } from '../core/types';
import { PhysicsWorldKey, type ColliderRecord, type PhysicsWorld } from '../physics/PhysicsWorld';
import { PlayerKey, type PlayerController } from '../character/PlayerController';
import { buildPortalColliders } from './Portal';
import type { ColorGradeSettings } from '../render/post/ColorGrade';
import { PostStackKey, type PostStack } from '../render/post/PostStack';
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
  #report: Omit<ZoneLoadReport, 'enemies'> | null = null;
  /** Set when a zone loaded before `CombatSystem` existed; see the module header. */
  #pendingEnemies = false;
  #pendingEntry: ZoneEntryPoint | null = null;
  #disposed = false;
  /** Resolves when the active zone's encounter is placed. See `enemiesReady`. */
  #enemiesReady: Promise<void> = Promise.resolve();
  #resolveEnemies: (() => void) | null = null;
  /**
   * The frame's look before the active zone trimmed it.
   *
   * Captured at the moment of application rather than at boot, so that the
   * baseline is whatever the frame graph actually shipped with — including any
   * later art pass — and a zone's trim is always relative to it. `null` when no
   * trim is in effect, which is what makes {@link #restoreGrade} idempotent.
   */
  #gradeBaseline: { exposure: number; grade: Partial<ColorGradeSettings> } | null = null;

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

  /**
   * What the last completed load produced.
   *
   * `enemies` is read live rather than snapshotted at the end of the load. It
   * used to be snapshotted, and on the boot path — where spawning is deferred
   * until `CombatSystem` exists — the snapshot was taken before a single
   * skeleton had been placed, so the moor permanently reported `0 enemies`
   * while six of them were walking around in it.
   */
  get report(): ZoneLoadReport | null {
    if (this.#report === null) return null;
    return { ...this.#report, enemies: this.#director?.enemies.length ?? 0 };
  }

  /**
   * Resolves when the active zone's encounter has been placed.
   *
   * The thing a caller needs and could not previously get. Enemy models are
   * fetched over the network, and a fetch completes on a *macrotask*; a harness
   * that drives the world with `await engine.stepFrames(1)` in a loop only ever
   * yields microtasks, so the load could not finish for as long as it kept
   * stepping. Awaiting this promise unwinds the stack and lets the fetch land.
   *
   * Already resolved for a zone with no encounter table, so awaiting it is
   * always safe.
   */
  get enemiesReady(): Promise<void> {
    return this.#enemiesReady;
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
    // The start zone's enemies could not be placed during `init` because
    // `CombatSystem` had not registered yet, and their models may still have
    // been in flight. `populate` is idempotent and returns false until both are
    // true, so this retries rather than firing once and hoping.
    //
    // It used to fire once, as `void this.#spawnEnemies(spawns)` — an unawaited
    // promise nothing could observe. When it worked, nothing said so; when it
    // did not, nothing said that either.
    if (this.#pendingEnemies) {
      const director = this.#director;
      if (director === null || director.populate()) {
        this.#pendingEnemies = false;
        this.#settleEnemies();
      }
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

    /* -- look ------------------------------------------------------------ */

    // Before the player is placed, so the first frame of a new zone is already
    // graded for it rather than showing one frame of the previous area's key.
    this.#applyGrade(zone);

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
      // The director is constructed and its models are requested here whether
      // or not `CombatSystem` exists yet: the download is by far the slowest
      // part of entering a populated zone and it needs nothing but the asset
      // manager. Only the *placing* waits.
      const director = new EnemyDirector({ spawns });
      this.#director = director;
      this.#enemiesReady = new Promise<void>((resolve) => {
        this.#resolveEnemies = resolve;
      });
      const load = director.init(ctx).catch((error: unknown) => {
        console.error(`[ZoneManager] "${zoneId}" could not load its enemies:`, error);
      });
      if (ctx.services.has(CombatKey)) {
        await load;
        this.#settleEnemies();
      } else {
        this.#pendingEnemies = true;
        // The models usually land *after* `CombatSystem` has registered, in
        // which case the director places the encounter at the end of its own
        // `init` and there is nothing left for `update` to retry. Settle here
        // rather than making a caller step another frame to be told so — a
        // harness that awaits `enemiesReady` without stepping would otherwise
        // wait for a frame that is itself waiting for the await.
        void load.then(() => {
          if (this.#director === director && director.ready) {
            this.#pendingEnemies = false;
            this.#settleEnemies();
          }
        });
      }
    }

    const measured = measureZone(zone.root);
    this.#report = {
      zoneId,
      millis: Date.now() - startedAt,
      colliders,
      renderables: measured.renderables,
      triangles: measured.triangles,
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
        `${colliders} colliders, ${this.#director?.enemies.length ?? 0} enemies` +
        (this.#pendingEnemies ? ' (placing deferred until combat is up)' : ''),
    );
  }

  #unload(): void {
    const ctx = this.#ctx;
    const zone = this.#zone;
    if (ctx === null || zone === null) return;

    ctx.events.emit('zone:unloading', { zoneId: zone.zoneId });

    // Put the frame back first. A zone's exposure trim is the one resource that
    // is *not* reclaimed by disposing its subtree — it lives on the post stack,
    // which outlives every zone — so leaving it set would light the next area
    // with the last one's key.
    this.#restoreGrade();

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
    this.#pendingEnemies = false;
    this.#settleEnemies();
    this.#enemiesReady = Promise.resolve();
    this.#pendingEntry = null;
    ctx.events.emit('zone:unloaded', {
      zoneId: zone.zoneId,
      collidersRemoved: removed,
      disposed,
    });
  }

  /* -- per-zone look ------------------------------------------------------- */

  /**
   * Apply a zone's exposure and grade trim, remembering what it replaced.
   *
   * Deliberately a *trim* and not a preset: `stops` multiplies the locked key
   * rather than replacing it, and the grade patch is merged over the shipping
   * look rather than substituted for it. So the project keeps one place where
   * the frame's character is decided, and a zone only says how its own set
   * departs from it. Tuning the base look still moves all three areas together,
   * which is the property a per-zone *preset* would quietly destroy.
   */
  #applyGrade(zone: Zone): void {
    const post = this.#ctx?.services.tryGet<PostStack>(PostStackKey);
    const trim = zone.grade;
    if (post === undefined || trim === undefined) return;
    // A previous zone's trim must never become this one's baseline.
    this.#restoreGrade();

    const composite = post.composite;
    const grade = post.grade;
    const baseExposure = composite.exposure;
    const patch = trim.grade ?? {};
    const current = grade.settings;
    const baseline: Partial<ColorGradeSettings> = {};
    for (const key of Object.keys(patch) as (keyof ColorGradeSettings)[]) {
      Object.assign(baseline, { [key]: current[key] });
    }
    this.#gradeBaseline = { exposure: baseExposure, grade: baseline };

    if (trim.stops !== undefined && trim.stops !== 0) {
      composite.setExposure(baseExposure * 2 ** trim.stops);
    }
    if (Object.keys(patch).length > 0) grade.set(patch);
  }

  /** Undo {@link #applyGrade}. Safe to call when nothing is applied. */
  #restoreGrade(): void {
    const baseline = this.#gradeBaseline;
    if (baseline === null) return;
    this.#gradeBaseline = null;
    const post = this.#ctx?.services.tryGet<PostStack>(PostStackKey);
    if (post === undefined) return;
    post.composite.setExposure(baseline.exposure);
    if (Object.keys(baseline.grade).length > 0) post.grade.set(baseline.grade);
  }

  /* -- helpers ------------------------------------------------------------ */

  /**
   * Release anything awaiting {@link enemiesReady}.
   *
   * Called on success *and* on unload: a caller that awaits the encounter of a
   * zone the player has already left must not wait forever.
   */
  #settleEnemies(): void {
    this.#resolveEnemies?.();
    this.#resolveEnemies = null;
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
