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
  /** Largest accepted raw frame delta, in seconds. Default 0.25. */
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
  readonly #pixelRatioCap: number;
  readonly #autoStart: boolean;
  readonly #rendererOptions: CreateRendererOptions;

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
    this.#clock = new Clock(options.now ?? (() => performance.now()));
    this.#accumulator = new FixedStepAccumulator({
      hz: options.fixedHz ?? 60,
      maxSubSteps: options.maxSubSteps ?? 5,
      maxDelta: options.maxDelta ?? 0.25,
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

      const steps = this.#accumulator.begin(rawDelta, this.#time.scale);
      const fixedDelta = this.#accumulator.fixedDelta;
      for (let step = 0; step < steps; step++) {
        this.#runPhase('fixedUpdate', ctx, fixedDelta);
      }

      this.#runPhase('update', ctx, scaled);
      this.#runPhase('lateUpdate', ctx, scaled);

      input?.endFrame();

      await renderer.render(this.scene, this.camera);

      this.events.emit('engine:frame', { frame: this.#time.frame, dt: scaled });
    } finally {
      this.#stepInFlight = false;
    }
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
