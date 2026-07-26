/**
 * @module render/webgpuCompat
 *
 * Browser-compatibility shims that must be installed *before* a
 * `WebGPURenderer` is constructed.
 *
 * ### The `swizzle` incompatibility
 *
 * three.js r185 sets `GPUTextureViewDescriptor.swizzle = 'rgba'` — a string,
 * matching the WebGPU spec at the time it was written. Chromium 141's Dawn
 * implements the revised spec where `GPUTextureComponentSwizzle` is a
 * *dictionary* (`{ r, g, b, a }`). Against that build, every single
 * `GPUTexture.createView()` call throws:
 *
 * ```
 * TypeError: Failed to execute 'createView' on 'GPUTexture': Failed to read the
 * 'swizzle' property from 'GPUTextureViewDescriptor': The provided value is not
 * of type 'GPUTextureComponentSwizzle'.
 * ```
 *
 * which makes WebGPU completely non-functional rather than merely degraded.
 *
 * The patch is self-latching: it passes descriptors through untouched until it
 * observes that exact failure, then converts the string form to the dictionary
 * form for the remainder of the session. On browsers that accept the string
 * (and on browsers that predate the property entirely) it never latches and
 * costs one extra function call per `createView`. That "detect, don't sniff"
 * approach means the shim will quietly become a no-op once three.js ships the
 * dictionary form, with no version checks to maintain.
 *
 * No `@webgpu/types` dependency is taken: the handful of WebGPU shapes touched
 * here are declared locally, which keeps the boundary casts confined to this
 * file.
 */

/** Minimal structural view of the descriptor field this shim rewrites. */
interface TextureViewDescriptorLike {
  swizzle?: unknown;
  [key: string]: unknown;
}

type CreateViewFn = (descriptor?: TextureViewDescriptorLike) => unknown;

interface GPUTextureCtorLike {
  prototype: { createView: CreateViewFn };
}

/** How the shim resolved, for logging and diagnostics. */
export type SwizzleShimState =
  | 'unsupported' // no WebGPU in this browser; nothing installed
  | 'installed' // installed, no incompatibility observed yet
  | 'passthrough' // browser accepted the string form; shim is inert
  | 'patched-to-dict' // rewriting string swizzle to { r, g, b, a }
  | 'patched-strip'; // dictionary form also rejected; dropping the field

let state: SwizzleShimState = 'unsupported';
let installed = false;

/** Current shim state. Surfaced in the debug overlay. */
export function getSwizzleShimState(): SwizzleShimState {
  return state;
}

/**
 * Convert `'rgba'` to `{ r: 'r', g: 'g', b: 'b', a: 'a' }`, or drop the field
 * entirely when `strip` is set.
 */
function rewrite(descriptor: TextureViewDescriptorLike, strip: boolean): TextureViewDescriptorLike {
  const next: TextureViewDescriptorLike = { ...descriptor };
  const swizzle = next['swizzle'];
  if (!strip && typeof swizzle === 'string' && swizzle.length === 4) {
    next['swizzle'] = {
      r: swizzle[0],
      g: swizzle[1],
      b: swizzle[2],
      a: swizzle[3],
    };
  } else {
    delete next['swizzle'];
  }
  return next;
}

/**
 * Install the WebGPU compatibility shims. Idempotent, and safe to call in
 * environments with no WebGPU at all (it becomes a no-op).
 *
 * @returns the resulting {@link SwizzleShimState}.
 */
export function installWebGPUCompat(): SwizzleShimState {
  if (installed) return state;
  installed = true;

  // Boundary cast: `GPUTexture` is a browser global that TypeScript's DOM lib
  // does not declare, and this project deliberately avoids the @webgpu/types
  // dependency. Everything past this point is typed against the local shapes.
  const gpuTexture = (globalThis as Record<string, unknown>)['GPUTexture'] as
    | GPUTextureCtorLike
    | undefined;

  if (gpuTexture?.prototype?.createView === undefined) {
    state = 'unsupported';
    return state;
  }

  const original = gpuTexture.prototype.createView;
  /** 0 = probing, 1 = rewrite to dictionary, 2 = strip the field. */
  let mode: 0 | 1 | 2 = 0;

  gpuTexture.prototype.createView = function patchedCreateView(
    this: unknown,
    descriptor?: TextureViewDescriptorLike,
  ): unknown {
    const hasStringSwizzle = typeof descriptor?.['swizzle'] === 'string';

    if (mode !== 0) {
      return original.call(
        this,
        hasStringSwizzle && descriptor !== undefined ? rewrite(descriptor, mode === 2) : descriptor,
      );
    }

    try {
      const view = original.call(this, descriptor);
      if (hasStringSwizzle) state = 'passthrough';
      return view;
    } catch (error) {
      // Only the swizzle type error is ours to handle; anything else is a real
      // renderer bug and must keep propagating.
      if (!(error instanceof TypeError) || !hasStringSwizzle || descriptor === undefined) {
        throw error;
      }
      try {
        const view = original.call(this, rewrite(descriptor, false));
        mode = 1;
        state = 'patched-to-dict';
        return view;
      } catch {
        const view = original.call(this, rewrite(descriptor, true));
        mode = 2;
        state = 'patched-strip';
        return view;
      }
    }
  };

  state = 'installed';
  return state;
}
