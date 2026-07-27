/**
 * @module ui/DebugOverlay
 *
 * Always-on corner readout: which backend won, frame rate, frame count and
 * resolution.
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
 */

import type { GameContext, GameModule } from '../core/types';
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

export class DebugOverlay implements GameModule {
  readonly name = 'DebugOverlay';

  readonly #parent: HTMLElement;
  #element: HTMLDivElement | null = null;
  #smoothedDelta = 1 / 60;
  #sinceRefresh = 0;
  #lastText = '';
  #width = 0;
  #height = 0;
  #pixelRatio = 1;
  #unsubscribeResize: (() => void) | null = null;

  constructor(parent: HTMLElement = document.body) {
    this.#parent = parent;
  }

  init(ctx: GameContext): void {
    const element = document.createElement('div');
    element.id = 'd2rim-debug-overlay';
    element.setAttribute('style', STYLE);
    this.#parent.appendChild(element);
    this.#element = element;

    // Resolution comes from the resize event rather than being polled, so the
    // overlay never reads layout during a frame.
    this.#unsubscribeResize = ctx.events.on('engine:resize', ({ width, height, pixelRatio }) => {
      this.#width = width;
      this.#height = height;
      this.#pixelRatio = pixelRatio;
      // Force a redraw on the next update so a resize shows immediately.
      this.#sinceRefresh = REFRESH_INTERVAL;
    });
  }

  update(ctx: GameContext, dt: number): void {
    const element = this.#element;
    if (element === null) return;

    if (dt > 0) {
      // EMA with a ~0.9 retention factor: responsive within a few frames,
      // stable enough to read.
      this.#smoothedDelta = this.#smoothedDelta * 0.9 + dt * 0.1;
    }

    this.#sinceRefresh += dt;
    if (this.#sinceRefresh < REFRESH_INTERVAL) return;
    this.#sinceRefresh = 0;

    const fps = this.#smoothedDelta > 0 ? 1 / this.#smoothedDelta : 0;
    const { backend, capabilities } = ctx.renderer;
    const shim = getSwizzleShimState();

    const text = [
      `d2rim  ${backend}${backend === 'webgpu' && shim.startsWith('patched') ? ` (${shim})` : ''}`,
      `fps    ${fps.toFixed(1).padStart(5)}  (${(this.#smoothedDelta * 1000).toFixed(1)} ms)`,
      `frame  ${ctx.time.frame}`,
      `size   ${this.#width}x${this.#height} @${this.#pixelRatio.toFixed(2)}x`,
      `caps   compute=${capabilities.compute ? 'yes' : 'no'} msaa=${capabilities.maxSamples}x`,
    ].join('\n');

    if (text !== this.#lastText) {
      element.textContent = text;
      this.#lastText = text;
    }
  }

  dispose(): void {
    this.#unsubscribeResize?.();
    this.#unsubscribeResize = null;
    this.#element?.remove();
    this.#element = null;
  }
}
