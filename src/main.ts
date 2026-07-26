/**
 * @module main
 *
 * Application entry point. Wires the canvas to an {@link Engine}, registers the
 * baseline modules, and publishes `window.__d2rim` for headless tooling.
 */

import * as THREE from 'three/webgpu';

import { AssetManager } from './assets/AssetManager';
import { Engine } from './core/Engine';
import type { GameContext } from './core/types';
import { ReferenceScene } from './scene/ReferenceScene';
import { DebugOverlay } from './ui/DebugOverlay';

/**
 * The handle headless capture and automated tests drive the game through.
 *
 * The intended sequence is: `await window.__d2rim.ready`, then
 * `await engine.stepFrames(n)` for a deterministic world state, then capture.
 * `stepFrames` puts the engine into manual mode, so no stray animation frame
 * can advance the world between stepping and capturing.
 */
export interface D2RimGlobal {
  readonly engine: Engine;
  /** Throws until `ready` resolves; see {@link Engine.context}. */
  readonly ctx: GameContext;
  readonly ready: Promise<void>;
  /**
   * The live `three/webgpu` namespace.
   *
   * Exposed for `tools/capture` shot setup scripts, which are strings evaluated
   * in the page and therefore cannot import anything. Without it a shot can
   * only move the camera; with it a shot can restage lighting, swap materials
   * or move objects to isolate exactly the thing under review.
   *
   * This is the same module instance the game uses, so `instanceof` holds. It
   * costs no extra bundle weight — three is already loaded — and game code must
   * never read it back off `window`.
   */
  readonly three: typeof THREE;
}

declare global {
  interface Window {
    __d2rim?: D2RimGlobal;
  }
}

function requireCanvas(): HTMLCanvasElement {
  const canvas = document.getElementById('viewport');
  if (!(canvas instanceof HTMLCanvasElement)) {
    throw new Error('[d2rim] #viewport canvas is missing from the document');
  }
  return canvas;
}

/**
 * Report a fatal boot failure in the page itself.
 *
 * A blank canvas with an error buried in the console is the worst possible
 * failure mode for a graphics project, particularly in headless capture where
 * nobody is reading the console.
 */
function reportFatal(error: unknown): void {
  console.error('[d2rim] fatal boot failure:', error);
  const panel = document.createElement('pre');
  panel.setAttribute(
    'style',
    'position:fixed;inset:24px;z-index:100;overflow:auto;margin:0;padding:20px;' +
      'border-radius:8px;border:1px solid #6d2b2b;background:#140d0d;color:#ffb4b4;' +
      'font:13px/1.6 ui-monospace,Menlo,Consolas,monospace;white-space:pre-wrap;',
  );
  panel.textContent = `d2rim failed to start\n\n${
    error instanceof Error ? (error.stack ?? error.message) : String(error)
  }`;
  document.body.appendChild(panel);
}

const canvas = requireCanvas();

/**
 * `?autostart=0` boots the engine without starting the rAF loop.
 *
 * Capture harnesses need this: with the loop running, frames are rendered
 * between `await ready` and the first `stepFrames` call, so the world state at
 * capture time depends on how long the harness took to issue its next command.
 * Suppressing autostart makes the caller the sole source of frame advancement.
 */
const autoStart = new URLSearchParams(window.location.search).get('autostart') !== '0';

const engine = new Engine({ canvas, autoStart });

// Registration order is initialisation order. The AssetManager must come first
// so that it has registered itself as a service before any scene module's
// `init` runs and asks for it.
engine.add(new AssetManager());
engine.add(new ReferenceScene());
engine.add(new DebugOverlay());

const ready = engine.ready
  .then(() => {
    // Pointer lock requires a user gesture, so it is requested on click rather
    // than at boot. Harmless in headless runs, where no click ever arrives.
    canvas.addEventListener('click', () => engine.input.requestPointerLock());
    document.body.classList.add('d2rim-ready');
  })
  .catch((error: unknown) => {
    reportFatal(error);
    throw error;
  });

window.__d2rim = {
  engine,
  get ctx(): GameContext {
    return engine.context;
  },
  ready,
  three: THREE,
};

// Vite HMR: without an explicit teardown the old engine keeps its rAF loop and
// event listeners alive, and every edit leaks another renderer.
if (import.meta.hot) {
  import.meta.hot.dispose(() => {
    engine.dispose();
    delete window.__d2rim;
  });
}
