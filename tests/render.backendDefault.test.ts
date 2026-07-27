/**
 * Which backend a first-time visitor gets, and why it is not WebGPU.
 *
 * The deployed build defaulted to WebGPU, and on a real WebGPU machine that
 * produced a black viewport at 4 fps with a live DOM HUD on top of it. The
 * mechanism is documented at length on `render/RendererFactory` and on
 * `Volumetrics.bakeFogNoiseTexture`: three r185 uploads a `Data3DTexture` as
 * separate 2D slices and binds it through a 2D view; Dawn rejects the bind
 * group; a poisoned bind group poisons the command encoder; the whole frame's
 * command buffer is dropped at submit. Every frame, forever.
 *
 * These tests are the guard rail on that decision. If someone flips the default
 * back without fixing the upload, this file fails and says why.
 */

import { describe, expect, it } from 'vitest';

import { AUTO_BACKEND, AUTO_BACKEND_REASON, backendFromUrl } from '../src/render/RendererFactory';
import { detailNoiseEnabled } from '../src/render/Volumetrics';

describe('default backend', () => {
  it('is WebGL2, not WebGPU', () => {
    // The whole point of the change. WebGPU is a *correct* target for this
    // engine and a broken one in the shipping three.js version; those are
    // different statements, and only the second one decides the default.
    expect(AUTO_BACKEND).toBe('webgl2');
  });

  it('explains itself, in the console and in the code', () => {
    expect(AUTO_BACKEND_REASON).toMatch(/Data3DTexture/);
    expect(AUTO_BACKEND_REASON).toMatch(/\?backend=webgpu/);
  });
});

describe('backendFromUrl', () => {
  it('lets a player opt in to WebGPU explicitly', () => {
    expect(backendFromUrl('?backend=webgpu')).toBe('webgpu');
  });

  it('lets a player pin WebGL2 explicitly', () => {
    expect(backendFromUrl('?backend=webgl2')).toBe('webgl2');
  });

  it('reports "auto" for an absent or unknown value rather than guessing', () => {
    expect(backendFromUrl('')).toBe('auto');
    expect(backendFromUrl('?quality=high')).toBe('auto');
    // A typo must not silently become a backend choice.
    expect(backendFromUrl('?backend=vulkan')).toBe('auto');
  });
});

describe('detailNoiseEnabled', () => {
  it('drops the 3D noise volume on WebGPU by default', () => {
    // This is the specific texture Dawn rejects. It is inlined into *both* fog
    // paths — the compute injection kernel and the ray-march fragment — so
    // falling back from froxel to ray march does not escape it. Not binding it
    // at all is the only thing that does.
    expect(detailNoiseEnabled('auto', 'webgpu')).toBe(false);
  });

  it('keeps it on WebGL2, where the upload is correct', () => {
    expect(detailNoiseEnabled('auto', 'webgl2')).toBe(true);
  });

  it('honours an explicit override in both directions', () => {
    // `on` exists so that a future three.js with a fixed upload can be tested
    // against WebGPU without editing this policy; `off` exists so a WebGL2
    // capture can isolate the noise's contribution.
    expect(detailNoiseEnabled('on', 'webgpu')).toBe(true);
    expect(detailNoiseEnabled('off', 'webgl2')).toBe(false);
  });
});
