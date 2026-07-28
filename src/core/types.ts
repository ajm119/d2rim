/**
 * @module core/types
 *
 * The architecture contract for d2rim. Every gameplay, render and UI module in
 * the project is written against the interfaces declared here, so this file is
 * deliberately dependency-light and change-averse.
 *
 * Two three.js entry points appear below, and the distinction matters:
 *
 * - `three/webgpu` is the *runtime* entry point. It contains the full core
 *   (Scene, PerspectiveCamera, materials, ...) plus the node/TSL renderer
 *   architecture, and it is the only three build the bundle ever loads.
 * - `three` (the classic build) is referenced with `import type` only, purely so
 *   that `RendererHandle.three` can name `WebGLRenderer` in its union exactly as
 *   the contract specifies. Type-only imports are erased at compile time, so no
 *   second copy of three ends up in the bundle and `instanceof` stays sound.
 */

import type * as THREE from 'three/webgpu';
import type { WebGLRenderer } from 'three';

import type { Engine } from './Engine';
import type { EventBus } from './EventBus';
import type { Input } from './Input';
import type { ServiceLocator } from './ServiceLocator';

/**
 * Everything a {@link GameModule} is allowed to reach for. Modules receive this
 * on `init` and on every update; they must not capture global state instead.
 *
 * All fields are readonly: the context object identity is stable for the whole
 * lifetime of the engine, so modules may safely retain a reference to it.
 */
export interface GameContext {
  readonly engine: Engine;
  readonly scene: THREE.Scene;
  readonly camera: THREE.PerspectiveCamera;
  readonly renderer: RendererHandle;
  readonly input: Input;
  readonly events: EventBus;
  readonly time: TimeState;
  readonly services: ServiceLocator;
}

/**
 * A unit of game behaviour registered with the {@link Engine}.
 *
 * Lifecycle per frame, for every registered module in registration order:
 *   1. `fixedUpdate` — zero or more times, at a constant `fixedDt` (60 Hz).
 *   2. `update`      — exactly once, with the variable frame delta.
 *   3. `lateUpdate`  — exactly once, after every module's `update` has run.
 *
 * `lateUpdate` is the correct place for anything that must observe the settled
 * state of other modules in the same frame (camera follow, IK, UI readouts).
 */
export interface GameModule {
  readonly name: string;
  init(ctx: GameContext): Promise<void> | void;
  update?(ctx: GameContext, dt: number): void;
  fixedUpdate?(ctx: GameContext, fixedDt: number): void;
  lateUpdate?(ctx: GameContext, dt: number): void;
  dispose?(): void;
}

/**
 * Mutable frame clock. The engine owns the single instance handed out through
 * {@link GameContext.time} and mutates it in place, so modules must read the
 * fields per frame rather than destructuring once and caching.
 *
 * - `elapsed` — seconds of scaled game time since the first stepped frame.
 * - `delta`   — scaled, clamped seconds since the previous frame.
 * - `frame`   — frames stepped since boot; the first rendered frame is 1.
 * - `scale`   — time multiplier. 0 freezes gameplay while rendering continues.
 */
export interface TimeState {
  elapsed: number;
  delta: number;
  frame: number;
  scale: number;
}

/** Which GPU API actually won at runtime. */
export type RendererBackend = 'webgpu' | 'webgl2';

/**
 * Backend-agnostic wrapper around the three.js renderer.
 *
 * Game code renders through this handle and never touches backend specifics, so
 * that the WebGPU and WebGL2 paths remain interchangeable. `render` is typed
 * `void | Promise<void>` because the WebGPU path is genuinely asynchronous;
 * callers that need frame-accurate sequencing (headless capture, `stepFrames`)
 * must await it.
 */
export interface RendererHandle {
  readonly backend: RendererBackend;
  readonly three: THREE.WebGPURenderer | WebGLRenderer;
  readonly capabilities: {
    compute: boolean;
    float32Filterable: boolean;
    maxSamples: number;
  };
  setSize(w: number, h: number): void;
  render(scene: THREE.Scene, camera: THREE.Camera): void | Promise<void>;

  /**
   * Additive extension (not part of the original contract). Renders one frame
   * into an offscreen render target and reads it back as tightly packed,
   * top-row-first RGBA8 bytes.
   *
   * This exists because WebGPU canvas presentation never reaches the compositor
   * in the headless container this project is developed in: the GPU renders
   * correctly but `page.screenshot()` captures pure black. Reading back from a
   * render target bypasses the swapchain entirely and is the only capture route
   * that works for both backends, so the harness can stay backend-agnostic.
   */
  captureFrame?(
    scene: THREE.Scene,
    camera: THREE.Camera,
    width?: number,
    height?: number,
  ): Promise<CapturedFrame>;

  /**
   * Additive extension. Resolve outstanding GPU timer queries and return the
   * most recent render pass duration in **milliseconds**, or `null` when the
   * device offers no timer query.
   *
   * Present only when timestamps were requested at construction, because the
   * query pool is not free: WebGL2 needs `EXT_disjoint_timer_query_webgl2` and
   * every render pass then brackets itself with `beginQuery`/`endQuery`. The
   * game asks for it behind `?stats=1` only.
   *
   * This is the one measurement that separates "the CPU cannot feed the GPU"
   * from "the GPU cannot keep up", and those two have opposite fixes. Without
   * it, a frame-time number can only say that something is slow.
   */
  resolveGpuTime?(): Promise<number | null>;

  /**
   * Why {@link resolveGpuTime} will or will not produce a number, decided once
   * at construction rather than inferred from a stream of nulls.
   *
   * ### Why this had to become an explicit state
   *
   * The overlay printed `gpu …` forever on the machine under investigation, and
   * that ellipsis was read — reasonably — as "still resolving". It was not. It
   * was a false negative with a very specific cause:
   *
   * `WebGLBackend.initTimestampQuery` returns early unless
   * `EXT_disjoint_timer_query_webgl2` is present, so the timestamp query *pool*
   * is never created. `Backend.resolveTimestampsAsync` then hits its
   * `if (!queryPool) return;` and resolves with `undefined` **without writing
   * `info.render.timestamp`** — which three initialises to `0` and leaves
   * there. `resolveGpuTime` read that 0, decided `Number.isFinite(0)` was a
   * successful measurement, and handed 0 to `FrameStats` every quarter second.
   * `FrameStats.gpuAvailable` only flips on a sample greater than zero, so it
   * stayed false, and the overlay's three-way `available / unavailable /
   * pending` display fell through to `pending` in perpetuity.
   *
   * A diagnostic that says "loading" when it means "impossible" is worse than
   * one that says nothing, because it costs the reader the time they spend
   * waiting for it. The extension is now probed directly at construction —
   * `renderer.hasFeature('timestamp-query')`, which three maps onto exactly
   * that extension for the WebGL2 backend — and the answer is reported.
   *
   * - `'available'`  — queries are on and a number is coming.
   * - `'unsupported'` — the device or browser has no timer query. Common:
   *   Chromium gates `EXT_disjoint_timer_query_webgl2` for the timing side
   *   channel it is, and hardened builds such as Brave gate it harder.
   * - `'off'` — timestamps were never requested (no `?stats=1`).
   */
  readonly gpuTimer?: GpuTimerState;

  /**
   * Additive extension. How many render pipelines the renderer has compiled so
   * far, or 0 where the backend does not expose a countable cache.
   *
   * The one measurement that distinguishes "a 25-second frame was shader
   * compilation" from "a 25-second frame was something else": if the count
   * climbs across the stall, it was compilation.
   */
  programCount?(): number;

  /**
   * Additive extension. Compile every pipeline the scene needs, up front.
   *
   * Called on the loading screen, where a multi-second stall is expected. See
   * the implementation in `render/RendererFactory` for why lazy compilation is
   * the leading explanation for a multi-second frame in an otherwise steady
   * (if slow) session.
   */
  warmup?(
    scene: THREE.Scene,
    camera: THREE.Camera,
  ): Promise<{ millis: number; programs: number; before: number }>;

  /**
   * Additive extension. What the driver calls itself, unmasked where possible.
   *
   * ### Why a string on the overlay is worth this much ceremony
   *
   * Every performance report this project receives comes from a machine it
   * cannot touch, and the first question about any of them — *is this hardware
   * at all?* — has never been answerable. A browser that has fallen back to a
   * software rasteriser (SwiftShader, or Chromium's SwANGLE) does not say so:
   * it reports a working WebGL2 context, renders the correct image, and takes
   * tens of milliseconds a frame to do it, with the cost sitting exactly where
   * a GPU-bound frame's cost sits. That is indistinguishable from "the scene is
   * too heavy" from the inside, and it is the explanation that survives every
   * round of removing work — which is the position this investigation has been
   * in for several rounds.
   *
   * `WEBGL_debug_renderer_info` answers it in one line: `Apple M4` is hardware,
   * `Google SwiftShader` / `SwANGLE` is not. Chromium has been unmasking the
   * *masked* `RENDERER` parameter for a while, so both are read and the more
   * specific one wins; a browser that withholds both (Brave with fingerprint
   * protection at its stricter settings does exactly this) yields `null`,
   * which is itself worth printing — a hardened browser is also a browser that
   * withholds `EXT_disjoint_timer_query_webgl2` and may be refusing other GPU
   * features silently.
   */
  readonly gpuDescription?: string | null;
}

/** See {@link RendererHandle.gpuTimer}. */
export type GpuTimerState = 'available' | 'unsupported' | 'off';

/** Raw RGBA8 pixel readback produced by {@link RendererHandle.captureFrame}. */
export interface CapturedFrame {
  readonly width: number;
  readonly height: number;
  /** `width * height * 4` bytes, row 0 is the top row of the image. */
  readonly pixels: Uint8Array;
}
