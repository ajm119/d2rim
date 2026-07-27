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
}

/** Raw RGBA8 pixel readback produced by {@link RendererHandle.captureFrame}. */
export interface CapturedFrame {
  readonly width: number;
  readonly height: number;
  /** `width * height * 4` bytes, row 0 is the top row of the image. */
  readonly pixels: Uint8Array;
}
