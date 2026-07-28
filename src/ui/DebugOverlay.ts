/**
 * @module ui/DebugOverlay
 *
 * Always-on corner readout: which backend won, frame rate, frame count and
 * resolution — plus, behind `?stats=1`, the numbers that decide whether a frame
 * is expensive.
 *
 * ### The overlay is the only instrument
 *
 * This project has no GPU. Every performance claim it can make about a real
 * machine has to come back from a player reading this box, so it is worth more
 * than its pixels: draw calls, triangles, resident texture bytes, render-target
 * bytes, which post passes ran, and how many WebGPU errors the device swallowed
 * are exactly the numbers that separate "heavy" from "pathological".
 *
 * ### It used to lie about the resolution
 *
 * `size` read `0x0` forever. The overlay learned the resolution only from the
 * `engine:resize` event, and it subscribes during `init` — which runs from
 * `engine.add`, *after* `Engine.#boot` has already called `resize()` once. The
 * first (and, for a player who never resizes the window, only) event was
 * therefore emitted before anybody was listening, and the zero default stood.
 * A diagnostic that reports zero when the renderer is fine costs more time than
 * no diagnostic at all, so `init` now reads the live size straight off the
 * renderer and treats the event purely as an update.
 *
 * Two implementation notes that matter more than they look:
 *
 * - The DOM is only written when a displayed value actually changes, and never
 *   more than a few times a second. Touching `textContent` every frame forces
 *   style recalculation inside the frame budget and would make the overlay a
 *   measurable part of what it is measuring.
 * - The memory walk is *not* per frame. It traverses the scene graph, which is
 *   far too expensive for a frame budget, so it runs on a 2 s cadence and the
 *   displayed value is up to two seconds stale. That is fine for bytes, which
 *   move slowly, and it is why draw calls and triangles are read from the
 *   renderer's own counters instead.
 *
 * ### It used to lie about the frame rate too, and that was worse
 *
 * The `fps` line was an exponential moving average of `ctx.time.delta`. That is
 * the *clamped* delta: the fixed-step accumulator truncates it at `maxDelta`
 * before gameplay ever sees it. On any machine slower than the clamp the
 * average converges on the clamp and sits there — the old default of 0.25 s
 * produced `fps 4.0 (250.0 ms)` on a 6 fps machine and on a 1 fps machine
 * alike, and it read *identically* across two different renderer backends,
 * which is how the saturation was finally spotted. Every performance decision
 * taken against that number was taken against a constant.
 *
 * The overlay now reports from `Engine.frameStats`, which samples the raw
 * unclamped wall-clock delta, and it shows:
 *
 * - `fps` — from the **median** raw frame time, not a mean and not an
 *   instantaneous reciprocal. p95 is shown beside it, because the gap between
 *   the two is what stutter actually is.
 * - `cpu` — the update/render split plus GPU time from a timer query where the
 *   device has one. Time in the frame that is in neither is the browser
 *   blocking on the GPU, which is the diagnosis, not a rounding error.
 * - `clamp` — how many frames in the window hit the delta clamp or dropped
 *   simulation backlog. Shown only when non-zero, and never confused with the
 *   frame time: it says the *world* is running behind, which is a different
 *   fact from how long a frame took.
 */

import { AssetManagerKey } from '../assets/AssetManager';
import type { FrameStatsSnapshot } from '../core/FrameStats';
import type { GameContext, GameModule, GpuTimerState } from '../core/types';
import { describeRenderFlags, renderFlags } from '../render/DebugFlags';
import { collectMemoryReport } from '../render/MemoryReport';
import { PostStackKey, type PostStack } from '../render/post/PostStack';
import { gpuErrorReport } from '../render/RendererFactory';
import { VolumetricsKey, type VolumetricsService } from '../render/Volumetrics';
import { getSwizzleShimState } from '../render/webgpuCompat';

const STYLE = `
position: fixed;
top: 12px;
left: 12px;
z-index: 10;
padding: 8px 11px;
border-radius: 6px;
border: 1px solid rgba(255, 255, 255, 0.10);
background: rgba(8, 10, 14, 0.62);
backdrop-filter: blur(6px);
color: #d8dde6;
font: 12px/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
letter-spacing: 0.02em;
white-space: pre;
pointer-events: none;
user-select: none;
text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6);
`;

/** Refresh interval in seconds. 4 Hz is readable without being distracting. */
const REFRESH_INTERVAL = 0.25;

/** How often the scene-walking memory report is recomputed, in seconds. */
const MEMORY_INTERVAL = 2;

/** `?stats=1` (or the pre-existing `?mem=1`) turns on the expanded readout. */
export function statsRequested(search?: string): boolean {
  const query = search ?? (typeof window === 'undefined' ? '' : window.location.search);
  const params = new URLSearchParams(query);
  return params.get('stats') === '1' || params.get('mem') === '1';
}

/** Inputs to {@link gpuTimeLabel}. Named so the call site reads as a sentence. */
export interface GpuLabelInput {
  /** Whether any frame in the window carried a real timer-query measurement. */
  readonly gpuAvailable: boolean;
  /** Median GPU milliseconds, meaningful only when `gpuAvailable`. */
  readonly gpuMs: number;
  /** What the renderer decided about timer queries at construction. */
  readonly state: GpuTimerState | undefined;
  /** Whether the expanded readout (and therefore the query pool) was asked for. */
  readonly statsRequested: boolean;
  /** Whether a resolve attempt has already come back empty. */
  readonly refused: boolean;
}

/**
 * What to print after `gpu`, and specifically *never* an ellipsis for a device
 * that will never answer.
 *
 * ### The bug this replaces
 *
 * The previous version fell through to `…` in every case that was not a
 * resolved measurement, and on the reporter's machine it printed `…` for the
 * entire session. It reads as "loading". It meant "impossible", and the chain
 * that produced it is worth writing down because every link looks reasonable:
 *
 * 1. `?stats=1` asks three for `trackTimestamp`.
 * 2. `WebGLBackend.initTimestampQuery` returns early without
 *    `EXT_disjoint_timer_query_webgl2`, so no timestamp query pool is created.
 * 3. `Backend.resolveTimestampsAsync` hits `if (!queryPool) return;` and
 *    resolves with `undefined`, **never writing `info.render.timestamp`**.
 * 4. three initialises that field to `0` and leaves it there.
 * 5. `resolveGpuTime` read the 0, found it finite, and reported a successful
 *    measurement of zero — four times a second, forever.
 * 6. `FrameStats.gpuAvailable` only flips on a sample greater than zero, so it
 *    stayed false, and the overlay's three-way display fell through to
 *    "pending".
 *
 * The renderer now settles the question once at construction (see
 * `RendererHandle.gpuTimer`), so the states are genuinely distinguishable and
 * only a device that *can* answer and has not yet is ever spelled with an
 * ellipsis. A diagnostic that says "loading" when it means "impossible" costs
 * the reader every minute they spend waiting for it.
 */
export function gpuTimeLabel(input: GpuLabelInput): string {
  if (input.gpuAvailable) return `${input.gpuMs.toFixed(1)} ms`;
  if (!input.statsRequested || input.state === 'off') return 'off (?stats=1)';
  if (input.state === 'unsupported' || input.refused) {
    return 'n/a — timer unavailable, try ?gpusync=1';
  }
  return 'pending…';
}

interface RendererSizeProbe {
  getSize?: (target: Vector2Like) => Vector2Like;
  getPixelRatio?: () => number;
  info?: { render?: { drawCalls?: number; calls?: number; triangles?: number } };
}

/**
 * The part of `THREE.Vector2` that `Renderer.getSize` actually uses.
 *
 * Written out rather than importing `three/webgpu` for one constructor: this
 * module is otherwise DOM-only, and `getSize` calls exactly one method on the
 * target. The first version of this passed a plain `{ x, y }` and three threw
 * `e.set is not a function` from inside `getSize`, which took the whole
 * overlay's `init` down with it — a diagnostic that removes itself the moment
 * it is needed.
 */
interface Vector2Like {
  x: number;
  y: number;
  set(x: number, y: number): Vector2Like;
}

function vector2Like(): Vector2Like {
  return {
    x: 0,
    y: 0,
    set(x: number, y: number): Vector2Like {
      this.x = x;
      this.y = y;
      return this;
    },
  };
}

/** Megabytes, to one decimal. */
function mb(bytes: number): string {
  return (bytes / (1024 * 1024)).toFixed(1);
}

export interface DebugOverlayOptions {
  /** Force the expanded readout on or off. Defaults to the URL flag. */
  readonly stats?: boolean;
}

export class DebugOverlay implements GameModule {
  readonly name = 'DebugOverlay';

  readonly #parent: HTMLElement;
  readonly #statsOption: boolean | undefined;
  #stats = false;
  #element: HTMLDivElement | null = null;
  #frames: FrameStatsSnapshot | null = null;
  /** True once a GPU timer query has been asked for and refused. */
  #gpuTimerUnavailable = false;
  #gpuResolveInFlight = false;
  /** Consecutive zero-millisecond resolves. See `#pollGpuTime`. */
  #gpuZeroStreak = 0;
  /** Pipelines compiled during the loading-screen warmup, or null if it did not run. */
  #warmedPrograms: number | null = null;
  #sinceRefresh = 0;
  #sinceMemory = MEMORY_INTERVAL;
  #lastText = '';
  #width = 0;
  #height = 0;
  #pixelRatio = 1;
  /** The raw `devicePixelRatio`, so the tier's cap can be seen doing its job. */
  #deviceRatio = typeof window === 'undefined' ? 1 : window.devicePixelRatio || 1;
  #unsubscribeResize: (() => void) | null = null;
  #unsubscribeFrame: (() => void) | null = null;
  #unsubscribeWarmup: (() => void) | null = null;
  #drawCalls = 0;
  #triangles = 0;

  // Cached memory figures, refreshed on the slow cadence.
  #textureMb = '—';
  #targetMb = '—';
  #textureFormat: string | null = null;

  constructor(parent: HTMLElement = document.body, options: DebugOverlayOptions = {}) {
    this.#parent = parent;
    this.#statsOption = options.stats;
  }

  init(ctx: GameContext): void {
    this.#stats = this.#statsOption ?? statsRequested();

    const element = document.createElement('div');
    element.id = 'd2rim-debug-overlay';
    element.setAttribute('style', STYLE);
    this.#parent.appendChild(element);
    this.#element = element;

    // Seed from the renderer, because the first `engine:resize` has already
    // been emitted by the time this module is initialised. Without this the
    // overlay reports `0x0` until the player happens to resize the window.
    this.#readSizeFromRenderer(ctx);

    // Subsequent resolution changes still arrive by event, so the overlay never
    // reads layout during a frame.
    this.#unsubscribeResize = ctx.events.on('engine:resize', ({ width, height, pixelRatio }) => {
      this.#width = width;
      this.#height = height;
      this.#pixelRatio = pixelRatio;
      if (typeof window !== 'undefined') this.#deviceRatio = window.devicePixelRatio || 1;
      // Force a redraw on the next update so a resize shows immediately.
      this.#sinceRefresh = REFRESH_INTERVAL;
    });

    // The warmup total, so the overlay can say how many pipelines were
    // compiled *after* boot — which is how many stalls the session has taken.
    this.#unsubscribeWarmup = ctx.events.on('engine:warmup', ({ programs }) => {
      this.#warmedPrograms = programs;
    });

    // Draw calls and triangles have to be sampled *after* the render, and a
    // module's `update`/`lateUpdate` both run before it. `Engine` resets the
    // counters at the top of each frame and emits `engine:frame` once the
    // render has been awaited, so this is the only moment in the frame when
    // they hold a complete, current total. Reading them from `update` instead
    // reports zero, every time, which is how the previous version of this
    // overlay would have lied about the second most important number on it.
    if (this.#stats) {
      this.#unsubscribeFrame = ctx.events.on('engine:frame', () => {
        const render = (ctx.renderer.three as unknown as RendererSizeProbe).info?.render;
        this.#drawCalls = render?.drawCalls ?? 0;
        this.#triangles = Math.round(render?.triangles ?? 0);
      });
    }
  }

  update(ctx: GameContext, dt: number): void {
    const element = this.#element;
    if (element === null) return;

    if (this.#stats) {
      this.#sinceMemory += dt;
      if (this.#sinceMemory >= MEMORY_INTERVAL) {
        this.#sinceMemory = 0;
        this.#refreshMemory(ctx);
      }
    }

    this.#sinceRefresh += dt;
    if (this.#sinceRefresh < REFRESH_INTERVAL) return;
    this.#sinceRefresh = 0;

    // Percentiles sort the window, so this happens on the display cadence and
    // never per frame — see `core/FrameStats`.
    this.#frames = ctx.engine.frameStats.snapshot();
    this.#pollGpuTime(ctx);

    // A resize can happen without an `engine:resize` in exotic cases (a canvas
    // resized by CSS with no observer fired). Re-reading here is a handful of
    // property reads four times a second and it keeps the number honest.
    if (this.#width === 0 || this.#height === 0) this.#readSizeFromRenderer(ctx);

    const text = this.#compose(ctx);
    if (text !== this.#lastText) {
      element.textContent = text;
      this.#lastText = text;
    }
  }

  dispose(): void {
    this.#unsubscribeResize?.();
    this.#unsubscribeResize = null;
    this.#unsubscribeFrame?.();
    this.#unsubscribeFrame = null;
    this.#unsubscribeWarmup?.();
    this.#unsubscribeWarmup = null;
    this.#element?.remove();
    this.#element = null;
  }

  /* -- internals --------------------------------------------------------- */

  #compose(ctx: GameContext): string {
    const { backend, capabilities } = ctx.renderer;
    const shim = getSwizzleShimState();
    const frames = this.#frames ?? ctx.engine.frameStats.snapshot();

    const lines = [
      `d2rim  ${backend}${backend === 'webgpu' && shim.startsWith('patched') ? ` (${shim})` : ''}`,
      // Median first, because that is the number a player means by "fps", then
      // the p95 beside it because the difference between them is the stutter.
      // Both come from raw wall-clock deltas: nothing on this line has been
      // through the fixed-step clamp. See the module header.
      `fps    ${frames.fps.toFixed(1).padStart(5)}  p50 ${frames.p50Ms.toFixed(1)} ms  ` +
        `p95 ${frames.p95Ms.toFixed(1)} ms (${frames.fpsLow.toFixed(1)} fps)`,
      // Which frame, not just how bad. A 25-second stall at frame 3 is a
      // shader compile or a texture upload; the same stall at frame 400 is not,
      // and the overlay used to make those two indistinguishable.
      `worst  ${frames.maxMs.toFixed(1)} ms at frame ${frames.maxFrame} ` +
        `(over ${frames.samples} frames)`,
      `frame  ${ctx.time.frame}`,
      // Two sizes, because only one of them costs anything.
      //
      // The CSS size is what the window is; the *drawing buffer* is what the
      // GPU fills, and on a Retina display those differ by a factor of two per
      // axis — four in fragments. Reporting only the CSS size is how a machine
      // ends up quietly rendering 4.7 megapixels while the overlay says 1280.
      // `device` is the raw `devicePixelRatio` and `@` is what survived the
      // tier's cap, so the two together say whether the cap is doing anything.
      `size   ${this.#width}x${this.#height} css`,
      `buffer ${Math.round(this.#width * this.#pixelRatio)}x` +
        `${Math.round(this.#height * this.#pixelRatio)} ` +
        `@${this.#pixelRatio.toFixed(2)}x (device ${this.#deviceRatio.toFixed(2)}x) ` +
        `${((this.#width * this.#pixelRatio * this.#height * this.#pixelRatio) / 1e6).toFixed(2)} Mpx`,
      `caps   compute=${capabilities.compute ? 'yes' : 'no'} msaa=${capabilities.maxSamples}x`,
      // Where the median frame went. `gpu` is a real timer query where the
      // device has one; `—` means the extension is absent, which is a fact
      // about the driver rather than a failure. Whatever is left over after
      // update+render is the browser blocking before it grants the next
      // animation frame, i.e. the frame is GPU-bound.
      `time   update ${frames.updateMs.toFixed(1)} ms  render ${frames.renderMs.toFixed(1)} ms  ` +
        `gpu ${gpuTimeLabel({
          gpuAvailable: frames.gpuAvailable,
          gpuMs: frames.gpuMs,
          state: ctx.renderer.gpuTimer,
          statsRequested: this.#stats,
          refused: this.#gpuTimerUnavailable,
        })}`,
    ];

    // What the frame cost that neither timer saw. On a GPU-bound frame this is
    // the browser blocking before it will grant the next animation frame, and
    // stating it as its own line is the difference between a reader inferring
    // the diagnosis and being handed it.
    const unaccounted = frames.p50Ms - frames.updateMs - frames.renderMs - frames.syncMs;
    if (frames.p50Ms > 0 && unaccounted > 1) {
      lines.push(
        `blocked ${unaccounted.toFixed(1)} ms/frame unaccounted ` +
          `(${((unaccounted / frames.p50Ms) * 100).toFixed(0)}% — GPU or compositor)`,
      );
    }
    if (frames.syncAvailable) {
      lines.push(`gpusync ${frames.syncMs.toFixed(1)} ms forced readback stall (?gpusync=1)`);
    }

    // The configuration this frame time was measured under. A screenshot with
    // no record of which systems were switched off is not a measurement, and
    // the whole point of the kill switches is that the numbers come back from
    // someone else's laptop.
    const flags = renderFlags();
    lines.push(`flags  ${describeRenderFlags(flags)}`);

    // The clamp, stated separately and only when it is doing something. This
    // is *not* the frame time; it says simulation time is falling behind wall
    // clock, so the world runs in slow motion. Conflating the two is the exact
    // bug this overlay was rebuilt to remove.
    if (frames.clampedFrames > 0 || frames.starvedFrames > 0) {
      const clamp = ctx.engine.maxFrameDelta;
      const p50Seconds = frames.p50Ms / 1000;
      const worldSpeed = p50Seconds > 0 ? Math.min(1, clamp / p50Seconds) : 1;
      lines.push(
        `clamp  ${frames.clampedFrames}/${frames.samples} frames hit the ` +
          `${(clamp * 1000).toFixed(0)} ms sim clamp` +
          (frames.starvedFrames > 0 ? `, ${frames.starvedFrames} dropped backlog` : '') +
          ` — world at ${worldSpeed.toFixed(2)}x speed`,
      );
    }

    if (!this.#stats) {
      // Errors are never hidden: a device swallowing validation errors is the
      // single most important thing this overlay can say, flag or no flag.
      const gpu = gpuErrorReport();
      if (gpu.count > 0) lines.push(`GPU ERRORS ${gpu.count}`);
      return lines.join('\n');
    }

    lines.push(
      `draws  ${this.#drawCalls}`,
      `tris   ${this.#triangles.toLocaleString('en-US')}`,
      `tex    ${this.#textureMb} MB${
        this.#textureFormat === null ? '' : `  ${this.#textureFormat}`
      }`,
      `rt     ${this.#targetMb} MB`,
    );

    const post = ctx.services.tryGet<PostStack>(PostStackKey);
    if (post !== undefined) {
      const stats = post.stats;
      lines.push(
        `post   ${post.quality} ${stats.active.length} passes / ${stats.passDraws} draws`,
        `       ${stats.active.join(' ')}`,
      );
    }

    const fog = ctx.services.tryGet<VolumetricsService>(VolumetricsKey);
    if (fog !== undefined) {
      const s = fog.stats;
      // Whether the fog is *composited*, which is a different question from
      // which mode it would use — and the one that decides whether it costs
      // anything. `LightShaftsPass` is the only consumer of
      // `Volumetrics.createResolveNode`, and it is a `high`+ pass, so at the
      // `medium` tier the ray march is never built and never runs. The overlay
      // reported `fog raymarch noise=on` regardless, which reads as "a
      // per-pixel ray march is running" and sent this investigation straight at
      // a pass that was not in the frame.
      const composited =
        post !== undefined && post.passes.some((pass) => pass.id === 'lightshafts');
      lines.push(
        `fog    ${s.mode} noise=${s.detailNoise ? 'on' : 'off'} ` +
          `dispatch=${s.froxelDispatches} scope=${s.errorScopeFramesLeft} ` +
          `composited=${composited ? 'yes' : 'NO (no consumer pass)'}` +
          (s.froxelFailed ? ' FELL BACK' : ''),
      );
    }

    // How many render pipelines three has compiled, and how many of those the
    // loading screen paid for.
    //
    // This is the line that identifies a compile stall. Lazy pipeline creation
    // means the frame that first draws a given material+lights combination
    // absorbs the whole compile synchronously, and with TSL node materials on
    // Apple hardware — where ANGLE has to translate the generated GLSL to Metal
    // and link it — that can be a substantial fraction of a second per program.
    // A session that reports `worst 25625.1 ms` with no idea what happened in
    // that frame is unactionable; a session where this counter jumps from 40 to
    // 90 across the same frame is a diagnosis. Watch it while walking around:
    // every increment is a stall that has already happened.
    const programs = ctx.renderer.programCount?.();
    if (programs !== undefined) {
      lines.push(
        `prog   ${programs} pipelines` +
          (this.#warmedPrograms === null
            ? ' (warmup skipped or unsupported)'
            : ` — ${this.#warmedPrograms} precompiled on the loading screen, ` +
              `${Math.max(0, programs - this.#warmedPrograms)} since`),
      );
    }

    const gpu = gpuErrorReport();
    lines.push(`gpuerr ${gpu.count}${gpu.first === null ? '' : ` — ${gpu.first.slice(0, 60)}`}`);

    return lines.join('\n');
  }

  /**
   * Drain the GPU timer-query pool and hand the answer to the engine.
   *
   * Off the frame path deliberately: resolving is an async device round trip,
   * and putting one inside `update` would mean the instrument stalls the thing
   * it measures. One resolve in flight at a time, on the overlay's 4 Hz
   * cadence, is enough to keep three's fixed-size query pool from overflowing
   * (it warns and drops queries when it does) while costing nothing.
   *
   * A device without the extension answers `null` once and is then never asked
   * again, so the overlay can say `n/a` rather than an ambiguous dash.
   */
  #pollGpuTime(ctx: GameContext): void {
    if (!this.#stats || this.#gpuTimerUnavailable || this.#gpuResolveInFlight) return;
    // Settled at construction, so a device with no extension is never asked at
    // all rather than being asked four times a second forever.
    if (ctx.renderer.gpuTimer !== undefined && ctx.renderer.gpuTimer !== 'available') {
      this.#gpuTimerUnavailable = true;
      return;
    }
    const resolve = ctx.renderer.resolveGpuTime;
    if (typeof resolve !== 'function') {
      this.#gpuTimerUnavailable = true;
      return;
    }
    this.#gpuResolveInFlight = true;
    void resolve
      .call(ctx.renderer)
      .then((milliseconds) => {
        if (milliseconds === null) {
          this.#gpuTimerUnavailable = true;
          return;
        }
        // A run of exact zeroes means the query pool is answering but never
        // producing data — three's WebGL pool returns its `lastValue` (0) when
        // the disjoint bit is set, which some drivers keep set permanently
        // under load. Twenty consecutive zeroes at 4 Hz is five seconds, which
        // is long enough to be certain and short enough that the reader is
        // still looking. Giving up and saying `n/a` is strictly better than
        // showing an ellipsis forever; that is the whole bug being fixed here.
        if (milliseconds <= 0) {
          this.#gpuZeroStreak++;
          if (this.#gpuZeroStreak >= 20) {
            this.#gpuTimerUnavailable = true;
            console.warn(
              '[DebugOverlay] the GPU timer query is present but has returned zero ' +
                '20 times running (a permanently-set GPU_DISJOINT bit does this). ' +
                'Reporting "n/a"; use ?gpusync=1 for a wall-clock measurement instead.',
            );
          }
          return;
        }
        this.#gpuZeroStreak = 0;
        ctx.engine.setGpuFrameTime(milliseconds);
      })
      .catch(() => {
        this.#gpuTimerUnavailable = true;
      })
      .finally(() => {
        this.#gpuResolveInFlight = false;
      });
  }

  #readSizeFromRenderer(ctx: GameContext): void {
    const probe = ctx.renderer.three as unknown as RendererSizeProbe;
    try {
      if (typeof probe.getSize === 'function') {
        const size = probe.getSize(vector2Like());
        if (size.x > 0 && size.y > 0) {
          this.#width = Math.round(size.x);
          this.#height = Math.round(size.y);
        }
      }
      if (typeof probe.getPixelRatio === 'function') {
        const ratio = probe.getPixelRatio();
        if (Number.isFinite(ratio) && ratio > 0) this.#pixelRatio = ratio;
      }
    } catch {
      // Never take the frame — or the rest of the overlay — down over a
      // diagnostic read. The `engine:resize` subscription is still the backstop.
    }
  }

  #refreshMemory(ctx: GameContext): void {
    try {
      const assets = ctx.services.tryGet(AssetManagerKey) ?? null;
      const report = collectMemoryReport(ctx.scene, { topTextures: 0, assets });
      this.#textureMb = mb(Math.max(report.textureBytes, report.residentAssetBytes));
      this.#targetMb = mb(report.renderTargetBytes);
      // The format the GPU actually got, not the one the budget assumed. The
      // difference between those two is a factor of two in resident texture
      // memory and it went unnoticed for a whole optimisation pass, so it is
      // now on the overlay where a player can read it back.
      this.#textureFormat = assets?.compressedFormat?.format ?? null;
    } catch {
      // A diagnostic must never be the thing that breaks the frame.
      this.#textureMb = 'err';
      this.#targetMb = 'err';
    }
  }
}
