/**
 * @module core/Engine
 *
 * Owns the frame loop, the module registry and the lifetime of every core
 * service. One `Engine` exists per page.
 *
 * ### Frame structure
 *
 * ```
 * rAF -> clock.tick -> accumulate
 *     -> fixedUpdate x N   (constant 60 Hz slices, deterministic simulation)
 *     -> update            (once, variable dt, presentation-rate logic)
 *     -> lateUpdate        (once, after every module's update has settled)
 *     -> input.endFrame    (consume per-frame edges and deltas)
 *     -> renderer.render   (awaited; WebGPU submission is asynchronous)
 * ```
 *
 * Simulation is decoupled from display rate so physics stays stable at any
 * refresh rate. {@link Engine.alpha} exposes the leftover fraction of a
 * simulation slice for interpolated rendering.
 *
 * ### Determinism
 *
 * {@link Engine.stepFrames} advances the world by an exact number of frames
 * with an exact delta, driving the loop from the caller instead of the wall
 * clock. That is what makes golden-image regression testing viable: the same
 * `stepFrames(n)` produces byte-identical output across runs.
 */

import * as THREE from 'three/webgpu';

import { EventBus, type BootPhase } from './EventBus';
import { FrameStats } from './FrameStats';
import { Input } from './Input';
import { ServiceLocator } from './ServiceLocator';
import { Clock, createTimeState, FixedStepAccumulator } from './Time';
import type { GameContext, GameModule, RendererHandle, TimeState } from './types';
import { createRenderer, type CreateRendererOptions } from '../render/RendererFactory';

/**
 * Reset three's per-frame render counters.
 *
 * Structurally typed rather than narrowed to a renderer class: `info` is not on
 * the frozen `RendererHandle` contract, and a test double or a future backend
 * may not have one. A missing `info` is not an error, it just means there is
 * nothing to report.
 */
function resetRendererInfo(renderer: RendererHandle): void {
  const info = (renderer.three as unknown as { info?: { reset?: () => void } }).info;
  info?.reset?.();
}

export interface EngineOptions {
  /** The canvas the renderer draws into. Also the input/pointer-lock target. */
  canvas: HTMLCanvasElement;
  /** Supply an existing scene/camera, otherwise sensible ones are created. */
  scene?: THREE.Scene;
  camera?: THREE.PerspectiveCamera;
  /** Simulation rate. Default 60 Hz — the contract's fixed timestep. */
  fixedHz?: number;
  /** Spiral-of-death guard: max simulation slices per frame. Default 5. */
  maxSubSteps?: number;
  /**
   * Largest accepted raw frame delta, in seconds.
   *
   * Defaults to `maxSubSteps / fixedHz` — 5/60 ≈ 83 ms — rather than to the
   * accumulator's own 0.25 s, and the coupling is the point.
   *
   * A clamp *larger* than the substep budget can never be honoured. Hand the
   * accumulator 250 ms with a five-slice cap and it consumes 83 ms, discards
   * the other 167 ms as unreachable backlog, and reports `starved`. Meanwhile
   * `TimeState.delta` — which drives animation, camera and every
   * presentation-rate system — was set from the same 250 ms. So the simulation
   * advances a third as far as the presentation thinks it did, every frame,
   * and the world visibly runs in slow motion while animations play at speed.
   * That is exactly what a 4 fps machine was doing.
   *
   * Clamping to what the substeps can actually consume makes the two agree:
   * a badly overloaded frame still runs slower than real time, but simulation
   * and presentation run slow *together*, nothing desynchronises, and no
   * physics body is ever integrated across a gap it did not step through.
   */
  maxDelta?: number;
  /** Upper bound on devicePixelRatio. Default 2. */
  pixelRatioCap?: number;
  /** Begin the rAF loop as soon as boot completes. Default true. */
  autoStart?: boolean;
  /** Forwarded to {@link createRenderer}. */
  renderer?: CreateRendererOptions;
  /** Injectable monotonic clock, in milliseconds. Defaults to `performance.now`. */
  now?: () => number;
  /**
   * Force a 1×1 `readPixels` after each frame and time it. Default false.
   *
   * See {@link Engine.gpuSyncEnabled} for what the number means and why it is
   * opt-in.
   */
  gpuSync?: boolean;
  /** Compile the scene's pipelines on the loading screen. Default true. */
  warmup?: boolean;
  /**
   * Log a console warning for any frame slower than this, in milliseconds.
   * Default 250. Set to `Infinity` to silence it.
   */
  slowFrameMs?: number;
  /**
   * Supply the event bus rather than letting the engine create one.
   *
   * Boot emits its first `boot:phase` synchronously, before the constructor
   * returns, so anything that wants to observe the *whole* of boot — the
   * loading screen, principally — has to be subscribed before the engine
   * exists. That is only possible if it and the engine share a bus that
   * predates both.
   */
  events?: EventBus;
}

type UpdatePhase = 'fixedUpdate' | 'update' | 'lateUpdate';

interface ModuleRecord {
  readonly module: GameModule;
  initialized: boolean;
  /** Phases this module has already thrown in; used to rate-limit logging. */
  readonly failedPhases: Set<UpdatePhase>;
}

export class Engine {
  /** Resolves once the renderer exists and every queued module has initialised. */
  readonly ready: Promise<void>;

  readonly scene: THREE.Scene;
  readonly camera: THREE.PerspectiveCamera;
  readonly events: EventBus;
  readonly services = new ServiceLocator();

  readonly #canvas: HTMLCanvasElement;
  readonly #time: TimeState = createTimeState();
  readonly #accumulator: FixedStepAccumulator;
  readonly #clock: Clock;
  readonly #modules: ModuleRecord[] = [];
  readonly #now: () => number;
  readonly #frameStats = new FrameStats(240);
  /**
   * GPU time for the most recent resolved timer query, in ms. Pushed in by
   * whoever is driving `resolveTimestampsAsync` — the engine does not poll the
   * renderer itself, because resolving a timestamp pool is an async round trip
   * that has no business inside the frame's critical path.
   */
  #gpuMs = 0;
  #pixelRatioCap: number;
  readonly #autoStart: boolean;
  readonly #rendererOptions: CreateRendererOptions;
  readonly #gpuSync: boolean;
  readonly #warmup: boolean;
  readonly #slowFrameMs: number;
  /** One-pixel scratch for the `?gpusync=1` readback. Allocated once. */
  readonly #syncPixel = new Uint8Array(4);
  #syncUnavailable = false;
  #slowFramesLogged = 0;

  #renderer: RendererHandle | null = null;
  #input: Input | null = null;
  #ctx: GameContext | null = null;

  #rafId = 0;
  #running = false;
  #paused = false;
  #manual = false;
  #stepInFlight = false;
  #disposed = false;
  #resizeObserver: ResizeObserver | null = null;
  #detachDom: Array<() => void> = [];

  constructor(options: EngineOptions) {
    this.events = options.events ?? new EventBus();
    this.#canvas = options.canvas;
    this.scene = options.scene ?? new THREE.Scene();
    this.camera =
      options.camera ??
      new THREE.PerspectiveCamera(60, 16 / 9, 0.1, 2000);
    this.#pixelRatioCap = options.pixelRatioCap ?? 2;
    this.#autoStart = options.autoStart ?? true;
    this.#rendererOptions = options.renderer ?? {};
    this.#now = options.now ?? (() => performance.now());
    this.#gpuSync = options.gpuSync ?? false;
    this.#warmup = options.warmup ?? true;
    this.#slowFrameMs = options.slowFrameMs ?? 250;
    this.#clock = new Clock(this.#now);
    const hz = options.fixedHz ?? 60;
    const maxSubSteps = options.maxSubSteps ?? 5;
    this.#accumulator = new FixedStepAccumulator({
      hz,
      maxSubSteps,
      // See `EngineOptions.maxDelta`: the clamp must not exceed the simulation
      // time the substep budget can actually consume, or presentation time and
      // simulation time diverge on every overloaded frame.
      maxDelta: options.maxDelta ?? maxSubSteps / hz,
    });

    this.ready = this.#boot();
  }

  // -- accessors ----------------------------------------------------------

  /**
   * The shared {@link GameContext}. Throws before {@link ready} resolves —
   * there is genuinely no renderer to hand out yet, and returning a partially
   * populated context would push the failure somewhere less obvious.
   */
  get context(): GameContext {
    if (this.#ctx === null) {
      throw new Error('[Engine] context is not available until `await engine.ready`');
    }
    return this.#ctx;
  }

  get renderer(): RendererHandle {
    if (this.#renderer === null) {
      throw new Error('[Engine] renderer is not available until `await engine.ready`');
    }
    return this.#renderer;
  }

  get input(): Input {
    if (this.#input === null) {
      throw new Error('[Engine] input is not available until `await engine.ready`');
    }
    return this.#input;
  }

  /** Live frame clock. Mutated in place each frame; read, do not cache. */
  get time(): TimeState {
    return this.#time;
  }

  /**
   * Rolling window of **raw** frame times — the honest instrument.
   *
   * `time.delta` is clamped and scaled and is therefore useless for measuring
   * how fast the game is running; see `core/FrameStats` for the full argument.
   * Anything reporting performance to a human reads this instead.
   */
  get frameStats(): FrameStats {
    return this.#frameStats;
  }

  /**
   * Supply GPU time (ms) for subsequent frames, from a timer query.
   *
   * Pushed rather than pulled because resolving a timestamp pool is an async
   * device round trip: the overlay resolves it off the critical path on its own
   * cadence and hands the answer back here, where it joins the same rolling
   * window as everything else. Zero (the default) simply means "no timer query
   * on this device", which is common and not an error.
   */
  setGpuFrameTime(milliseconds: number): void {
    this.#gpuMs = Number.isFinite(milliseconds) && milliseconds > 0 ? milliseconds : 0;
  }

  /**
   * Interpolation factor in `[0, 1)` between the last two simulation steps.
   *
   * Kept on the engine rather than in {@link TimeState} because the contract
   * fixes `TimeState` to exactly four fields.
   */
  get alpha(): number {
    return this.#accumulator.alpha;
  }

  /** Constant simulation slice, in seconds. */
  get fixedDelta(): number {
    return this.#accumulator.fixedDelta;
  }

  /**
   * The delta clamp actually in force, in seconds.
   *
   * Surfaced so the overlay can say "this frame hit the clamp" *without*
   * reporting the clamp as if it were the frame time — the failure mode that
   * made every slow machine report the same frame rate. See
   * {@link EngineOptions.maxDelta}.
   */
  get maxFrameDelta(): number {
    return this.#accumulator.maxDelta;
  }

  /** Whether the rAF loop is scheduled. */
  get running(): boolean {
    return this.#running;
  }

  /** Whether the loop is suspended by tab visibility. */
  get paused(): boolean {
    return this.#paused;
  }

  /** Registered module names, in execution order. */
  get moduleNames(): string[] {
    return this.#modules.map((record) => record.module.name);
  }

  // -- module registry ----------------------------------------------------

  /**
   * Register a module. Modules run in registration order within each phase.
   *
   * Before boot, modules are queued and initialised as part of {@link ready}.
   * After boot they are initialised immediately; an async `init` means the
   * module is skipped for update phases until it resolves, so a slow-loading
   * module can never observe a half-built world.
   */
  add(module: GameModule): this {
    if (this.#modules.some((record) => record.module.name === module.name)) {
      throw new Error(`[Engine] a module named "${module.name}" is already registered`);
    }
    const record: ModuleRecord = { module, initialized: false, failedPhases: new Set() };
    this.#modules.push(record);
    this.events.emit('module:added', { name: module.name });

    if (this.#ctx !== null) void this.#initModule(record);
    return this;
  }

  /** Remove and dispose a module. @returns whether it was registered. */
  remove(target: GameModule | string): boolean {
    const name = typeof target === 'string' ? target : target.name;
    const index = this.#modules.findIndex((record) => record.module.name === name);
    if (index === -1) return false;

    const [record] = this.#modules.splice(index, 1);
    if (record !== undefined) {
      try {
        record.module.dispose?.();
      } catch (error) {
        console.error(`[Engine] "${name}".dispose() threw:`, error);
      }
    }
    this.events.emit('module:removed', { name });
    return true;
  }

  /** Look up a registered module by name. */
  getModule<T extends GameModule = GameModule>(name: string): T | undefined {
    return this.#modules.find((record) => record.module.name === name)?.module as T | undefined;
  }

  // -- loop control -------------------------------------------------------

  /** Start (or resume) the rAF loop. Leaves manual-stepping mode. */
  start(): void {
    if (this.#disposed || this.#running) return;
    this.#running = true;
    this.#manual = false;
    this.#clock.reset();
    this.#accumulator.reset();
    this.#schedule();
  }

  /** Stop the rAF loop. State is preserved; `start()` resumes cleanly. */
  stop(): void {
    if (!this.#running) return;
    this.#running = false;
    if (this.#rafId !== 0) {
      cancelAnimationFrame(this.#rafId);
      this.#rafId = 0;
    }
  }

  /**
   * Advance exactly `count` frames using exactly `dt` seconds each, awaiting
   * the renderer between frames.
   *
   * This puts the engine into manual mode: the rAF loop is cancelled and stays
   * cancelled until {@link start} is called again, so a capture harness cannot
   * be raced by a stray animation frame. The wall clock is bypassed entirely,
   * which is what makes repeated runs byte-identical.
   *
   * Returns a promise because WebGPU submission is asynchronous; the contract
   * types `render` as `void | Promise<void>` for exactly this reason.
   */
  async stepFrames(count: number, dt: number = this.#accumulator.fixedDelta): Promise<void> {
    await this.ready;
    this.stop();
    this.#manual = true;

    const frames = Math.max(0, Math.floor(count));
    for (let i = 0; i < frames; i++) {
      await this.#stepOnce(dt);
    }
  }

  /** Tear everything down. The engine is unusable afterwards. */
  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    this.stop();

    for (const off of this.#detachDom) off();
    this.#detachDom = [];
    this.#resizeObserver?.disconnect();
    this.#resizeObserver = null;

    // Reverse order so modules tear down against still-live dependencies.
    for (let i = this.#modules.length - 1; i >= 0; i--) {
      const record = this.#modules[i];
      if (record === undefined) continue;
      try {
        record.module.dispose?.();
      } catch (error) {
        console.error(`[Engine] "${record.module.name}".dispose() threw:`, error);
      }
    }
    this.#modules.length = 0;

    this.#input?.dispose();
    this.#renderer?.three.dispose();
    this.services.clear();
    this.events.clear();
    this.#ctx = null;
  }

  // -- internals ----------------------------------------------------------

  async #boot(): Promise<void> {
    // Boot progress is emitted rather than logged because the phases have
    // wildly different durations — device init is milliseconds, shader
    // compilation and zone construction are seconds — and a progress bar that
    // does not say which one it is sitting in reads as a hang. The loading
    // screen turns these into words.
    const phase = (name: BootPhase, label: string, completed = 0, total = 0): void => {
      this.events.emit('boot:phase', { phase: name, label, completed, total });
    };

    try {
      phase('renderer', 'initialising renderer');
      this.#renderer = await createRenderer(this.#canvas, {
        ...this.#rendererOptions,
        pixelRatioCap: this.#rendererOptions.pixelRatioCap ?? this.#pixelRatioCap,
      });
      phase('renderer', `renderer ready — ${this.#renderer.backend}`);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.events.emit('boot:failed', { phase: 'renderer', message });
      throw error;
    }
    this.#input = new Input({ target: this.#canvas });

    this.#ctx = {
      engine: this,
      scene: this.scene,
      camera: this.camera,
      renderer: this.#renderer,
      input: this.#input,
      events: this.events,
      time: this.#time,
      services: this.services,
    };

    this.#attachDom();
    this.resize();

    // Sequential, not parallel: modules routinely depend on services registered
    // by earlier modules, and registration order is the declared contract.
    const queued = [...this.#modules];
    let index = 0;
    for (const record of queued) {
      phase('modules', record.module.name, index++, queued.length);
      await this.#initModule(record);
    }
    phase('modules', 'modules ready', queued.length, queued.length);

    await this.#warmPipelines(phase);

    phase('ready', 'entering the world', 1, 1);
    this.events.emit('engine:ready', { backend: this.#renderer.backend });
    if (this.#autoStart) this.start();
  }

  async #initModule(record: ModuleRecord): Promise<void> {
    if (record.initialized || this.#ctx === null) return;
    try {
      await record.module.init(this.#ctx);
      record.initialized = true;
    } catch (error) {
      console.error(`[Engine] "${record.module.name}".init() failed:`, error);
    }
  }

  #schedule(): void {
    this.#rafId = requestAnimationFrame(this.#onAnimationFrame);
  }

  readonly #onAnimationFrame = (): void => {
    if (!this.#running) return;
    this.#schedule();

    // A previous frame's GPU work has not resolved yet. Skipping keeps the
    // clock honest (the elapsed time folds into the next tick) instead of
    // queueing render calls faster than the device can retire them — which is
    // the difference between "slow" and "unrecoverable" on a software
    // rasteriser.
    if (this.#stepInFlight) return;

    const delta = this.#clock.tick();
    void this.#stepOnce(delta).catch((error: unknown) => {
      console.error('[Engine] frame failed:', error);
    });
  };

  /** Run one complete frame. `rawDelta` is unscaled, unclamped seconds. */
  async #stepOnce(rawDelta: number): Promise<void> {
    const ctx = this.#ctx;
    const renderer = this.#renderer;
    if (ctx === null || renderer === null || this.#disposed) return;

    this.#stepInFlight = true;
    const frameStart = this.#now();
    try {
      const input = this.#input;
      input?.beginFrame();

      const clamped = Math.min(Math.max(0, rawDelta), this.#accumulator.maxDelta);
      const scaled = clamped * this.#time.scale;

      this.#time.delta = scaled;
      this.#time.elapsed += scaled;
      this.#time.frame++;

      // three's node renderer resets its own counters from *its* animation
      // loop, which this engine does not use — it drives frames itself so that
      // `stepFrames` can be deterministic. Without this call `info.render`
      // accumulates monotonically for the life of the session, so "draw calls"
      // in the debug overlay would be "draw calls since boot" and the one
      // number a player can report about their own GPU would be meaningless.
      // three documents exactly this: apps with their own loop must reset once
      // per frame.
      resetRendererInfo(renderer);

      const updateStart = this.#now();
      const steps = this.#accumulator.begin(rawDelta, this.#time.scale);
      const fixedDelta = this.#accumulator.fixedDelta;
      for (let step = 0; step < steps; step++) {
        this.#runPhase('fixedUpdate', ctx, fixedDelta);
      }

      this.#runPhase('update', ctx, scaled);
      this.#runPhase('lateUpdate', ctx, scaled);

      input?.endFrame();
      const updateEnd = this.#now();

      await renderer.render(this.scene, this.camera);
      const renderEnd = this.#now();

      const syncMs = this.#gpuSyncProbe(renderer);

      // Recorded from the *raw* delta, never from `scaled`. This is the one
      // number a player on hardware this project cannot buy will read back to
      // us, and every clamp between the wall clock and the overlay is a way for
      // it to lie. See `core/FrameStats`.
      this.#frameStats.record({
        rawMs: rawDelta * 1000,
        updateMs: updateEnd - updateStart,
        renderMs: renderEnd - updateEnd,
        gpuMs: this.#gpuMs,
        syncMs,
        clamped: this.#accumulator.wasClamped,
        starved: this.#accumulator.wasStarved,
        frame: this.#time.frame,
      });

      this.#reportSlowFrame(rawDelta * 1000, updateEnd - updateStart, renderEnd - updateEnd, syncMs);

      this.events.emit('engine:frame', {
        frame: this.#time.frame,
        dt: scaled,
        rawDt: rawDelta,
        frameMs: this.#now() - frameStart,
      });
    } finally {
      this.#stepInFlight = false;
    }
  }

  /**
   * Compile the scene's pipelines while the loading screen is still up.
   *
   * The zone is fully built by this point — `ZoneManager` is a module and every
   * module's `init` has been awaited — so the scene graph the first frame will
   * draw is the scene graph compiled here. That ordering is the whole value of
   * doing it: a warmup before the zone exists would compile nothing.
   *
   * Failures are logged and swallowed. Warming pipelines is an optimisation;
   * refusing to boot because an optimisation failed would be a strictly worse
   * outcome than the stall it is trying to avoid.
   */
  async #warmPipelines(
    phase: (name: BootPhase, label: string, completed?: number, total?: number) => void,
  ): Promise<void> {
    const renderer = this.#renderer;
    if (renderer === null || typeof renderer.warmup !== 'function') return;
    if (!this.#warmup) {
      console.info('[Engine] pipeline warmup skipped (?warmup=0).');
      return;
    }
    phase('ready', 'compiling shaders', 0, 1);
    try {
      const result = await renderer.warmup(this.scene, this.camera);
      const compiled = result.programs - result.before;
      console.info(
        `[Engine] pipeline warmup: ${result.millis.toFixed(0)} ms, ` +
          `${compiled} pipeline${compiled === 1 ? '' : 's'} compiled ` +
          `(${result.programs} total). Anything compiled after this point is a ` +
          'material the scene did not contain at load — a lazily-loaded texture ' +
          'swap, a post-chain pass, or a hot-swapped archetype — and shows up as a ' +
          'single very slow frame. Compare against the first "slow frame" report.',
      );
      this.events.emit('engine:warmup', {
        millis: result.millis,
        compiled,
        programs: result.programs,
      });
    } catch (error) {
      console.warn('[Engine] pipeline warmup failed:', error);
    }
  }

  /** Whether `?gpusync=1` is arming the forced readback. */
  get gpuSyncEnabled(): boolean {
    return this.#gpuSync && !this.#syncUnavailable;
  }

  /**
   * Force the GPU pipeline to drain, and time how long that takes.
   *
   * ### Why this exists
   *
   * In WebGL, `render()` returns as soon as the commands are queued. The GPU
   * work happens later and the browser absorbs the wait somewhere the page
   * cannot see — typically before it will grant the next animation frame. That
   * is why the machine under investigation reports 29 ms of update plus 66 ms
   * of render against a 730 ms frame: the missing 630 ms is real, it is GPU
   * work, and no CPU timer in the page can attribute it.
   *
   * `gl.readPixels` cannot be answered until everything queued before it has
   * retired, so the call blocks the main thread for approximately the
   * outstanding GPU work. One pixel is read, because the transfer is irrelevant
   * — the *synchronisation* is the measurement. It is crude: it includes
   * driver round-trip overhead, it serialises CPU and GPU where they would
   * normally overlap, and it will make the frame rate worse while it is on.
   * All three are acceptable in exchange for converting invisible queued work
   * into a number, on a device whose driver refuses to expose a timer query.
   *
   * Behind `?gpusync=1` and off by default for exactly those reasons.
   *
   * @returns the stall in milliseconds, or 0 when the probe is off or the
   *   context does not offer a readback.
   */
  #gpuSyncProbe(renderer: RendererHandle): number {
    if (!this.#gpuSync || this.#syncUnavailable) return 0;
    const gl = (
      renderer.three as unknown as { getContext?: () => unknown }
    ).getContext?.() as WebGL2RenderingContext | null | undefined;
    if (gl === null || gl === undefined || typeof gl.readPixels !== 'function') {
      this.#syncUnavailable = true;
      console.warn(
        '[Engine] ?gpusync=1 asked for a forced readback, but this backend exposes no ' +
          'WebGL context to read from. Disabling the probe.',
      );
      return 0;
    }
    const start = this.#now();
    try {
      // Read from whatever is currently bound — after `render()` that is the
      // canvas's default framebuffer. Binding a target of our own would add an
      // extra state change to every frame for no extra information.
      gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, this.#syncPixel);
    } catch (error) {
      this.#syncUnavailable = true;
      console.warn('[Engine] ?gpusync=1 readback failed; disabling the probe:', error);
      return 0;
    }
    return this.#now() - start;
  }

  /**
   * Name a pathological frame in the console while it is still fresh.
   *
   * The overlay can say `worst 25625.1 ms over 91 frames`, which establishes
   * that a 25-second stall happened and nothing whatsoever about *when* or
   * *around what*. A console line stamped with the frame number, the phase
   * split and the draw count turns that into evidence: a multi-second stall in
   * the first few frames, with update and render both small, is shader
   * compilation or a synchronous texture upload inside the driver; the same
   * stall at frame 400 with a large `update` is a GC pause or a zone load.
   *
   * Rate-limited to twelve reports, because a machine at 1 fps trips the
   * threshold on every single frame and a console with ten thousand identical
   * warnings in it is not a diagnostic either.
   */
  #reportSlowFrame(rawMs: number, updateMs: number, renderMs: number, syncMs: number): void {
    if (rawMs < this.#slowFrameMs || this.#slowFramesLogged >= 12) return;
    this.#slowFramesLogged++;
    const info = (this.#renderer?.three as unknown as {
      info?: { render?: { drawCalls?: number; triangles?: number } };
    })?.info?.render;
    const unaccounted = Math.max(0, rawMs - updateMs - renderMs - syncMs);
    console.warn(
      `[Engine] slow frame #${this.#time.frame}: ${rawMs.toFixed(1)} ms ` +
        `(update ${updateMs.toFixed(1)} + render ${renderMs.toFixed(1)}` +
        (syncMs > 0 ? ` + gpusync ${syncMs.toFixed(1)}` : '') +
        ` + ${unaccounted.toFixed(1)} unaccounted), ` +
        `draws ${info?.drawCalls ?? '?'}, tris ${Math.round(info?.triangles ?? 0)}` +
        (this.#slowFramesLogged === 12 ? ' — further slow-frame reports suppressed' : ''),
    );
  }

  /**
   * Invoke one lifecycle phase across every initialised module.
   *
   * A module that throws is logged once per phase and then skipped silently, so
   * a single broken system degrades that system rather than killing the frame
   * loop — an uncaught throw inside rAF stops the game with no further output.
   */
  #runPhase(phase: UpdatePhase, ctx: GameContext, dt: number): void {
    for (const record of this.#modules) {
      if (!record.initialized) continue;
      const fn = record.module[phase];
      if (fn === undefined) continue;
      try {
        fn.call(record.module, ctx, dt);
      } catch (error) {
        if (!record.failedPhases.has(phase)) {
          record.failedPhases.add(phase);
          console.error(
            `[Engine] "${record.module.name}".${phase}() threw ` +
              `(further occurrences suppressed):`,
            error,
          );
        }
      }
    }
  }

  // -- DOM integration ----------------------------------------------------

  /**
   * Resize the renderer and camera to the canvas's current CSS size.
   *
   * Safe to call at any time; a zero-sized canvas (display:none, or a layout
   * that has not settled) is ignored rather than producing a degenerate
   * projection matrix.
   */
  /** The cap currently in force. See {@link setPixelRatioCap}. */
  get pixelRatioCap(): number {
    return this.#pixelRatioCap;
  }

  /**
   * Change the upper bound on `devicePixelRatio`, and resize to match.
   *
   * The engine holds the value but does not *choose* it: the choice belongs to
   * `render/RenderSettings`, because the cap is squared into the fragment count
   * and is therefore the single largest term in a quality tier's budget. A flat
   * cap of 2 on a Retina display quadruples the cost of every full-screen pass
   * relative to the same window at DPR 1, which is not a decision the engine
   * has any information to make.
   *
   * Resizing immediately rather than waiting for the next window resize is the
   * point: it makes a live `?quality=` change take effect on the next frame.
   */
  setPixelRatioCap(cap: number): void {
    const next = Math.max(0.5, cap);
    if (next === this.#pixelRatioCap) return;
    this.#pixelRatioCap = next;
    this.resize();
  }

  resize(): void {
    const renderer = this.#renderer;
    if (renderer === null) return;

    const width = this.#canvas.clientWidth || window.innerWidth;
    const height = this.#canvas.clientHeight || window.innerHeight;
    if (width <= 0 || height <= 0) return;

    const pixelRatio = Math.min(window.devicePixelRatio || 1, this.#pixelRatioCap);
    renderer.three.setPixelRatio(pixelRatio);
    renderer.setSize(width, height);

    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();

    this.events.emit('engine:resize', { width, height, pixelRatio });
  }

  #attachDom(): void {
    const on = <T extends Event>(
      target: EventTarget,
      type: string,
      handler: (event: T) => void,
    ): void => {
      const listener = handler as EventListener;
      target.addEventListener(type, listener);
      this.#detachDom.push(() => target.removeEventListener(type, listener));
    };

    // ResizeObserver catches layout-driven size changes that never fire a
    // window resize (sidebars, fullscreen transitions). The window listener
    // remains for devicePixelRatio changes when a window moves between
    // displays, which the observer does not report.
    if (typeof ResizeObserver !== 'undefined') {
      this.#resizeObserver = new ResizeObserver(() => this.resize());
      this.#resizeObserver.observe(this.#canvas);
    }
    on(window, 'resize', () => this.resize());

    on(document, 'visibilitychange', () => {
      if (document.hidden) {
        if (!this.#running) return;
        this.#paused = true;
        this.stop();
        this.events.emit('engine:pause', { paused: true });
      } else if (this.#paused) {
        this.#paused = false;
        // Do not resume a loop the caller deliberately put into manual mode.
        if (!this.#manual) this.start();
        this.events.emit('engine:pause', { paused: false });
      }
    });
  }
}
