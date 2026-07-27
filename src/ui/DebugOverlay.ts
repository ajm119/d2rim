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
 * - FPS is reported from an exponential moving average of frame deltas rather
 *   than an instantaneous reciprocal, which otherwise jitters so hard the
 *   number is unreadable.
 * - The memory walk is *not* per frame. It traverses the scene graph, which is
 *   far too expensive for a frame budget, so it runs on a 2 s cadence and the
 *   displayed value is up to two seconds stale. That is fine for bytes, which
 *   move slowly, and it is why draw calls and triangles are read from the
 *   renderer's own counters instead.
 */

import { AssetManagerKey } from '../assets/AssetManager';
import type { GameContext, GameModule } from '../core/types';
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
  #smoothedDelta = 1 / 60;
  #sinceRefresh = 0;
  #sinceMemory = MEMORY_INTERVAL;
  #lastText = '';
  #width = 0;
  #height = 0;
  #pixelRatio = 1;
  #unsubscribeResize: (() => void) | null = null;
  #unsubscribeFrame: (() => void) | null = null;
  #drawCalls = 0;
  #triangles = 0;

  // Cached memory figures, refreshed on the slow cadence.
  #textureMb = '—';
  #targetMb = '—';

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
      // Force a redraw on the next update so a resize shows immediately.
      this.#sinceRefresh = REFRESH_INTERVAL;
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

    if (dt > 0) {
      // EMA with a ~0.9 retention factor: responsive within a few frames,
      // stable enough to read.
      this.#smoothedDelta = this.#smoothedDelta * 0.9 + dt * 0.1;
    }

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
    this.#element?.remove();
    this.#element = null;
  }

  /* -- internals --------------------------------------------------------- */

  #compose(ctx: GameContext): string {
    const fps = this.#smoothedDelta > 0 ? 1 / this.#smoothedDelta : 0;
    const { backend, capabilities } = ctx.renderer;
    const shim = getSwizzleShimState();

    const lines = [
      `d2rim  ${backend}${backend === 'webgpu' && shim.startsWith('patched') ? ` (${shim})` : ''}`,
      `fps    ${fps.toFixed(1).padStart(5)}  (${(this.#smoothedDelta * 1000).toFixed(1)} ms)`,
      `frame  ${ctx.time.frame}`,
      `size   ${this.#width}x${this.#height} @${this.#pixelRatio.toFixed(2)}x`,
      `caps   compute=${capabilities.compute ? 'yes' : 'no'} msaa=${capabilities.maxSamples}x`,
    ];

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
      `tex    ${this.#textureMb} MB`,
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
      lines.push(
        `fog    ${s.mode} noise=${s.detailNoise ? 'on' : 'off'} ` +
          `dispatch=${s.froxelDispatches} scope=${s.errorScopeFramesLeft}` +
          (s.froxelFailed ? ' FELL BACK' : ''),
      );
    }

    const gpu = gpuErrorReport();
    lines.push(`gpuerr ${gpu.count}${gpu.first === null ? '' : ` — ${gpu.first.slice(0, 60)}`}`);

    return lines.join('\n');
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
      const report = collectMemoryReport(ctx.scene, {
        topTextures: 0,
        assets: ctx.services.tryGet(AssetManagerKey) ?? null,
      });
      this.#textureMb = mb(Math.max(report.textureBytes, report.residentAssetBytes));
      this.#targetMb = mb(report.renderTargetBytes);
    } catch {
      // A diagnostic must never be the thing that breaks the frame.
      this.#textureMb = 'err';
      this.#targetMb = 'err';
    }
  }
}
