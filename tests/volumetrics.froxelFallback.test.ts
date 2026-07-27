/**
 * Does the froxel fallback latch, or does it retry a failing compute dispatch
 * every frame?
 *
 * This was a live hypothesis for the reported 4 fps (250 ms/frame) on WebGPU:
 * a per-frame `pushErrorScope`/`popErrorScope` pair around a dispatch that is
 * rejected every time would be genuinely pathological, and 250 ms is far more
 * consistent with something pathological than with something merely heavy.
 *
 * It is answerable without a GPU, which is why this file exists. A fake node
 * renderer whose `compute()` throws stands in for a driver that refuses the
 * dispatch, and the module's own counters say what happened afterwards.
 *
 * The answer is that it latches: `#failFroxel` returns early when already
 * failed, `#chooseMode` consults the same latch, and the error scope is bounded
 * to the first four frames regardless. So the fallback is *not* the mechanism
 * behind the frame time — see `render/RendererFactory`'s module header for what
 * is. These tests keep it that way.
 */

import * as THREE from 'three/webgpu';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { EventBus } from '../src/core/EventBus';
import { ServiceLocator } from '../src/core/ServiceLocator';
import type { GameContext, RendererHandle } from '../src/core/types';
import { VolumetricsModule } from '../src/render/Volumetrics';

/** How many times `renderer.compute` was reached. */
let computeCalls = 0;
/** How many `pushErrorScope` calls the fake device saw. */
let scopePushes = 0;

/**
 * A node renderer that refuses every dispatch, with a WebGPU-shaped device so
 * the module takes the error-scope path rather than the "no device" path.
 */
function createFailingContext(): GameContext {
  computeCalls = 0;
  scopePushes = 0;

  const device = {
    pushErrorScope: (): void => void scopePushes++,
    popErrorScope: async (): Promise<null> => null,
  };

  const three = {
    backend: { isWebGPUBackend: true, device },
    compute: (): never => {
      computeCalls++;
      throw new Error('dispatch refused');
    },
  };

  const renderer = {
    backend: 'webgpu',
    three,
    capabilities: { compute: true, float32Filterable: false, maxSamples: 4 },
    setSize: (): void => {},
    render: (): void => {},
  } as unknown as RendererHandle;

  return {
    engine: {} as never,
    scene: new THREE.Scene(),
    camera: new THREE.PerspectiveCamera(60, 16 / 9, 0.1, 1000),
    renderer,
    input: {} as never,
    events: new EventBus(),
    time: { elapsed: 0, delta: 1 / 60, frame: 0, scale: 1 },
    services: new ServiceLocator(),
  };
}

/**
 * Advance the frame clock in place.
 *
 * `GameContext.time` is readonly on the interface — deliberately, modules must
 * not reassign it — but the `TimeState` behind it is the engine's own mutable
 * record, so a test standing in for the engine writes through it the same way
 * the engine does.
 */
function advance(ctx: GameContext, frame: number): void {
  const time = ctx.time as { frame: number; elapsed: number };
  time.frame = frame;
  time.elapsed = frame / 60;
}

beforeEach(() => {
  vi.spyOn(console, 'warn').mockImplementation(() => undefined);
  vi.spyOn(console, 'info').mockImplementation(() => undefined);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('froxel fallback', () => {
  it('latches after the first failure instead of retrying every frame', () => {
    const ctx = createFailingContext();
    const fog = new VolumetricsModule({ quality: 'low' });
    fog.init(ctx);
    expect(fog.stats.mode).toBe('froxel');

    for (let frame = 0; frame < 120; frame++) {
      advance(ctx, frame);
      fog.lateUpdate(ctx, 1 / 60);
    }

    const stats = fog.stats;
    expect(stats.froxelFailed).toBe(true);
    expect(stats.mode).toBe('raymarch');
    // The number that settles it. One attempt across 120 frames, not 120.
    expect(stats.froxelDispatches).toBe(1);
    expect(computeCalls).toBe(1);

    fog.dispose();
  });

  it('spends its validation error scope in the first frames and then stops', () => {
    const ctx = createFailingContext();
    const fog = new VolumetricsModule({ quality: 'low' });
    fog.init(ctx);

    for (let frame = 0; frame < 120; frame++) {
      advance(ctx, frame);
      fog.lateUpdate(ctx, 1 / 60);
    }

    // At most the bounded watch window, and in practice one — the throw on the
    // first dispatch latches the fallback before the budget can be spent. What
    // must never happen is a push per frame.
    expect(scopePushes).toBeLessThanOrEqual(4);
    expect(fog.stats.errorScopeFramesLeft).toBeLessThanOrEqual(4);

    fog.dispose();
  });

  it('announces the fallback once, not once per frame', () => {
    const ctx = createFailingContext();
    const reasons: string[] = [];
    ctx.events.on('volumetrics:fallback', ({ reason }) => reasons.push(reason));

    const fog = new VolumetricsModule({ quality: 'low' });
    fog.init(ctx);
    for (let frame = 0; frame < 60; frame++) {
      advance(ctx, frame);
      fog.lateUpdate(ctx, 1 / 60);
    }

    expect(reasons).toHaveLength(1);
    fog.dispose();
  });
});

describe('detail noise on WebGPU', () => {
  it('is not bound, so no Data3DTexture can poison a command buffer', () => {
    const ctx = createFailingContext();
    const fog = new VolumetricsModule({ quality: 'low' });
    fog.init(ctx);
    expect(fog.stats.detailNoise).toBe(false);
    fog.dispose();
  });
});
