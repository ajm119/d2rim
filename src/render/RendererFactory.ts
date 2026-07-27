/**
 * @module render/RendererFactory
 *
 * Creates the one renderer the game uses, wrapped in a backend-agnostic
 * {@link RendererHandle}.
 *
 * ### Backend strategy
 *
 * WebGPU is attempted first. On any failure — no `navigator.gpu`, a null
 * adapter, a device-lost during `init()`, a driver that rejects the swapchain —
 * the factory falls back to WebGL2 and logs which backend won. The fallback is
 * a `WebGPURenderer` constructed with `forceWebGL: true`, i.e. three.js's own
 * `WebGLBackend`, rather than the classic `WebGLRenderer`. That choice is
 * deliberate and is the single implementation-level departure from the literal
 * wording of the architecture contract:
 *
 * 1. TSL node materials are a feature of the node renderer architecture. A
 *    classic `WebGLRenderer` cannot render them at all, so falling back to it
 *    would not "degrade effects" — it would fail to draw the game's materials.
 * 2. `WebGLBackend` is a genuine WebGL2 path. It reports
 *    `backend.isWebGLBackend`, and this container's GPU probe captured a
 *    correct lit, shadowed, tone-mapped frame through it.
 * 3. Loading the classic build alongside `three/webgpu` would put two copies of
 *    every core class in the bundle and quietly break `instanceof`.
 *
 * The public type is unchanged: `RendererHandle.three` remains
 * `WebGPURenderer | WebGLRenderer`, so a future direct-WebGLRenderer tier needs
 * no contract change.
 *
 * ### Forcing a backend
 *
 * `?backend=webgl2` (or `webgpu`) in the page URL pins the backend. Headless
 * capture uses this to exercise both paths from one build, and it is the
 * fastest way to reproduce a WebGL2-only bug on a WebGPU-capable machine.
 */

import * as THREE from 'three/webgpu';

import type { CapturedFrame, RendererBackend, RendererHandle } from '../core/types';
import { installWebGPUCompat } from './webgpuCompat';

export interface CreateRendererOptions {
  /** Force a backend instead of probing. Defaults to the URL param, else auto. */
  backend?: RendererBackend | 'auto';
  /** MSAA on the default framebuffer. Default true. */
  antialias?: boolean;
  /**
   * Upper bound on `devicePixelRatio`. Default 2 — beyond that the fill-rate
   * cost is severe and the visual gain is not perceptible.
   */
  pixelRatioCap?: number;
  /** Tone-mapping exposure. Default 1.0; the reference scene is authored for it. */
  exposure?: number;
}

/** Backend hint parsed from `?backend=` on the current URL. */
function backendFromUrl(): RendererBackend | 'auto' {
  if (typeof window === 'undefined') return 'auto';
  const value = new URLSearchParams(window.location.search).get('backend');
  if (value === 'webgl2' || value === 'webgpu') return value;
  return 'auto';
}

/** Structural probe for `Renderer.backend` without importing backend classes. */
function isWebGPUBackend(renderer: THREE.Renderer): boolean {
  const backend = renderer.backend as unknown as { isWebGPUBackend?: boolean };
  return backend?.isWebGPUBackend === true;
}

/**
 * Apply the project's colour-pipeline defaults.
 *
 * These are the settings the reference scene — and every scene after it — is
 * authored against, so they live here rather than in scene code. Getting them
 * wrong is the single most common cause of "why does it look washed out".
 */
function configureRenderer(renderer: THREE.Renderer, exposure: number): void {
  // ACES filmic: filmic highlight rolloff so a bright sun disc and specular
  // hits compress gracefully instead of clipping to white.
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = exposure;

  // Render linear, present sRGB. Every colour input (material colours, sRGB
  // textures) is decoded to linear on the way in.
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  renderer.shadowMap.enabled = true;
  // PCF-soft is the best quality/cost point that both backends support.
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
}

/** Probe optional device features without letting a throw escape. */
function safeHasFeature(renderer: THREE.Renderer, name: string): boolean {
  try {
    return renderer.hasFeature(name);
  } catch {
    return false;
  }
}

function detectCapabilities(
  renderer: THREE.Renderer,
  backend: RendererBackend,
): RendererHandle['capabilities'] {
  if (backend === 'webgpu') {
    return {
      // Compute shaders are the WebGPU-only capability the project gates GPU
      // particles and skinning on.
      compute: true,
      float32Filterable: safeHasFeature(renderer, 'float32-filterable'),
      // WebGPU mandates support for exactly 1 and 4 samples; 4 is the ceiling.
      maxSamples: 4,
    };
  }

  // Boundary cast: `getContext()` is typed `unknown` on the common Renderer
  // because its return type depends on the backend.
  const gl = renderer.getContext() as WebGL2RenderingContext | null;
  let maxSamples = 4;
  let float32Filterable = false;
  if (gl !== null && typeof gl.getParameter === 'function') {
    const value: unknown = gl.getParameter(gl.MAX_SAMPLES);
    if (typeof value === 'number' && Number.isFinite(value)) maxSamples = value;
    float32Filterable = gl.getExtension('OES_texture_float_linear') !== null;
  }
  return { compute: false, float32Filterable, maxSamples };
}

/**
 * Construct a renderer for `canvas`, preferring WebGPU.
 *
 * Always resolves — a total failure to obtain any GPU context is the only case
 * that rejects, and it is genuinely unrecoverable.
 */
export async function createRenderer(
  canvas: HTMLCanvasElement,
  options: CreateRendererOptions = {},
): Promise<RendererHandle> {
  const requested = options.backend ?? backendFromUrl();
  const antialias = options.antialias ?? true;
  const pixelRatioCap = options.pixelRatioCap ?? 2;
  const exposure = options.exposure ?? 1.0;

  // Must happen before any WebGPU device work: the shim wraps a prototype
  // method three.js calls during `init()`.
  const shimState = installWebGPUCompat();

  let renderer: THREE.WebGPURenderer | null = null;
  let backend: RendererBackend = 'webgl2';

  if (requested !== 'webgl2') {
    try {
      const candidate = new THREE.WebGPURenderer({ canvas, antialias, forceWebGL: false });
      await candidate.init();
      if (isWebGPUBackend(candidate)) {
        renderer = candidate;
        backend = 'webgpu';
      } else {
        // three.js silently fell back to its WebGL backend (typically a null
        // adapter). Discard and take the explicit path so the intent is clear.
        candidate.dispose();
      }
    } catch (error) {
      console.warn('[RendererFactory] WebGPU initialisation failed, falling back:', error);
    }
  }

  if (renderer === null) {
    const fallback = new THREE.WebGPURenderer({ canvas, antialias, forceWebGL: true });
    await fallback.init();
    renderer = fallback;
    backend = 'webgl2';
  }

  configureRenderer(renderer, exposure);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, pixelRatioCap));

  const capabilities = detectCapabilities(renderer, backend);

  console.info(
    `[RendererFactory] backend=${backend} requested=${requested} shim=${shimState} ` +
      `compute=${capabilities.compute} float32Filterable=${capabilities.float32Filterable} ` +
      `maxSamples=${capabilities.maxSamples}`,
  );

  const three = renderer;

  /** Reused across capture calls so repeated captures do not churn GPU memory. */
  let captureTarget: THREE.RenderTarget | null = null;

  const handle: RendererHandle = {
    backend,
    three,
    capabilities,

    setSize(w: number, h: number): void {
      // `updateStyle: false` — the canvas is sized by CSS to fill the window,
      // and letting three write inline width/height would fight that.
      three.setSize(Math.max(1, Math.floor(w)), Math.max(1, Math.floor(h)), false);
    },

    render(scene: THREE.Scene, camera: THREE.Camera): void {
      // `render()` rather than the deprecated `renderAsync()`: since r181 the
      // latter is just `await init(); render()`, and `init()` has already been
      // awaited above. Command submission is synchronous from the caller's
      // point of view; GPU completion is only ever waited on where it actually
      // matters, in `captureFrame`'s readback.
      three.render(scene, camera);
    },

    async captureFrame(
      scene: THREE.Scene,
      camera: THREE.Camera,
      width?: number,
      height?: number,
    ): Promise<CapturedFrame> {
      const w = Math.max(1, Math.floor(width ?? canvas.width));
      const h = Math.max(1, Math.floor(height ?? canvas.height));

      if (captureTarget === null) {
        captureTarget = new THREE.RenderTarget(w, h, {
          format: THREE.RGBAFormat,
          type: THREE.UnsignedByteType,
          // Linear-tagged, i.e. "store the bytes as written". The target is
          // designated the renderer's *output* target below, which makes three
          // run its output pass (tone mapping + the sRGB OETF) into it — so the
          // frame is encoded exactly once, in the shader, and this capture
          // matches what the canvas shows. Tagging it `SRGBColorSpace` would
          // instead select an sRGB texture format whose hardware writes encode
          // as well, and the readback would come back a stop bright. See the
          // note on `PostStack.#ensureCaptureTarget`.
          colorSpace: THREE.LinearSRGBColorSpace,
          samples: Math.min(4, capabilities.maxSamples),
        });
        captureTarget.depthTexture = new THREE.DepthTexture(w, h);
      } else if (captureTarget.width !== w || captureTarget.height !== h) {
        captureTarget.setSize(w, h);
      }

      const previousTarget = three.getRenderTarget();
      const previousOutput = three.getOutputRenderTarget();
      three.setOutputRenderTarget(captureTarget);
      three.setRenderTarget(captureTarget);
      three.render(scene, camera);
      three.setRenderTarget(previousTarget);
      three.setOutputRenderTarget(previousOutput);

      const data = await three.readRenderTargetPixelsAsync(captureTarget, 0, 0, w, h);
      const pixels = unpackRows(
        new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
        w,
        h,
      );

      // three's WebGPU readback is already top-row-first; the WebGL path goes
      // through `gl.readPixels`, whose origin is bottom-left, so those rows
      // arrive inverted relative to image convention.
      return {
        width: w,
        height: h,
        pixels: backend === 'webgl2' ? flipRowsVertically(pixels, w, h) : pixels,
      };
    },
  };

  return handle;
}

/**
 * Copy a possibly row-padded RGBA8 readback into a tightly packed buffer.
 *
 * WebGPU requires the `bytesPerRow` of a texture-to-buffer copy to be a multiple
 * of 256, and three's backend returns that padded buffer verbatim rather than
 * stripping the padding. So for any width where `width * 4` is not a multiple of
 * 256 — that is, any width that is not a multiple of 64 — each successive row
 * arrives shifted further along the buffer. Read naively, the image is sheared
 * diagonally: garbage, but plausible-looking garbage that a blank-frame guard
 * happily passes.
 *
 * The WebGL path always returns tight rows and takes the fast path below, so the
 * two backends stay byte-comparable at every capture size.
 *
 * Exported for `tests/readback.unpack.test.ts`; game code should not call it.
 */
export function unpackRows(raw: Uint8Array, width: number, height: number): Uint8Array {
  const tightStride = width * 4;
  const tightSize = tightStride * height;
  if (raw.byteLength === tightSize) return raw;

  const paddedStride = Math.ceil(tightStride / 256) * 256;
  const expected = (height - 1) * paddedStride + tightStride;
  if (raw.byteLength < expected) {
    throw new Error(
      `captureFrame: readback is ${raw.byteLength} bytes, too short for ${width}x${height} ` +
        `at either ${tightStride} or ${paddedStride} bytes per row.`,
    );
  }

  const packed = new Uint8Array(tightSize);
  for (let y = 0; y < height; y++) {
    const from = y * paddedStride;
    packed.set(raw.subarray(from, from + tightStride), y * tightStride);
  }
  return packed;
}

/** Reverse row order of a tightly packed RGBA8 buffer, in place. */
function flipRowsVertically(pixels: Uint8Array, width: number, height: number): Uint8Array {
  const stride = width * 4;
  const row = new Uint8Array(stride);
  for (let y = 0; y < (height >> 1); y++) {
    const top = y * stride;
    const bottom = (height - 1 - y) * stride;
    row.set(pixels.subarray(top, top + stride));
    pixels.copyWithin(top, bottom, bottom + stride);
    pixels.set(row, bottom);
  }
  return pixels;
}
