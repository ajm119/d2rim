/**
 * @module tests/physics.colliderRegistry
 *
 * The collider *registry* — `PhysicsWorld.#records` — against a real Rapier
 * world and a real {@link CharacterController}.
 *
 * Every other physics test in this suite is pure maths on exported functions,
 * which is why the leak this file guards survived them all: the bug was not in
 * a formula, it was in teardown ordering between two objects, and no amount of
 * testing `classifySlope` can see it. Rapier's WASM initialises fine under
 * Node, so the real thing is used here rather than a mock — a mocked
 * `removeRigidBody` would not actually free the collider, and the whole
 * question is what happens to the record when it does.
 */

import * as THREE from 'three/webgpu';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { EventBus } from '../src/core/EventBus';
import { ServiceLocator } from '../src/core/ServiceLocator';
import type { GameContext } from '../src/core/types';
import { CharacterController } from '../src/physics/CharacterController';
import { PhysicsWorld } from '../src/physics/PhysicsWorld';

function makeContext(): GameContext {
  return {
    engine: {} as GameContext['engine'],
    scene: new THREE.Scene(),
    camera: new THREE.PerspectiveCamera(),
    renderer: {} as GameContext['renderer'],
    input: {} as GameContext['input'],
    events: new EventBus(),
    time: { elapsed: 0, delta: 1 / 60, frame: 0, scale: 1 },
    services: new ServiceLocator(),
  };
}

let physics: PhysicsWorld;

beforeEach(async () => {
  physics = new PhysicsWorld();
  await physics.init(makeContext());
});

afterEach(() => {
  physics.dispose();
});

/** Registry size and Rapier's own collider count, which must agree. */
const counts = (): { records: number; rapier: number } => ({
  records: physics.colliders.length,
  rapier: physics.world.colliders.len(),
});

describe('CharacterController disposal', () => {
  it('registers exactly one character collider while it lives', () => {
    const before = counts();
    const controller = new CharacterController(physics);
    const during = counts();
    expect(during.records).toBe(before.records + 1);
    expect(during.rapier).toBe(before.rapier + 1);
    expect(physics.colliders.filter((r) => r.kind === 'character')).toHaveLength(1);
    controller.dispose();
  });

  it('leaves no record behind — the +1-per-despawn leak', () => {
    const before = counts();
    const controller = new CharacterController(physics);
    controller.dispose();
    const after = counts();
    // Both halves matter. Rapier dropping the collider was already true before
    // the fix; the registry keeping the record was the leak.
    expect(after.rapier).toBe(before.rapier);
    expect(after.records).toBe(before.records);
  });

  it('does not accumulate across a zone lap of spawns and despawns', () => {
    const before = counts();
    // 6 + 20 + 6: the Blood Moor, the Den of Evil and the Blood Moor again,
    // which is the lap `tools/verify-zones.mjs` drives and where +32 stale
    // records were measured.
    for (const count of [6, 20, 6]) {
      const wave = Array.from({ length: count }, () => new CharacterController(physics));
      expect(counts().records).toBe(before.records + count);
      for (const controller of wave) controller.dispose();
      expect(counts().records).toBe(before.records);
    }
    expect(counts()).toEqual(before);
  });

  it('is idempotent', () => {
    const before = counts();
    const controller = new CharacterController(physics);
    controller.dispose();
    controller.dispose();
    expect(counts()).toEqual(before);
  });

  it('leaves every surviving record pointing at a live collider', () => {
    const keep = new CharacterController(physics);
    const drop = new CharacterController(physics);
    drop.dispose();
    // `recordFor` is the reason the registry exists: a raycast hit has to be
    // turned back into "what did I hit". A stale record makes that answer wrong
    // rather than absent, once Rapier reissues the handle.
    expect(physics.recordFor(keep.collider)).not.toBeNull();
    for (const record of physics.colliders) {
      expect(physics.world.getCollider(record.collider.handle)).toBeDefined();
    }
    keep.dispose();
  });
});

describe('PhysicsWorld.forgetCollider', () => {
  it('drops a record without asking Rapier to remove the collider', () => {
    const controller = new CharacterController(physics);
    const record = physics.recordFor(controller.collider);
    expect(record).not.toBeNull();
    const rapierBefore = physics.world.colliders.len();

    expect(physics.forgetCollider(record)).toBe(true);
    expect(physics.recordFor(controller.collider)).toBeNull();
    // The collider itself is untouched: this method is bookkeeping only.
    expect(physics.world.colliders.len()).toBe(rapierBefore);

    controller.dispose();
  });

  it('is a no-op for a record it does not hold', () => {
    const controller = new CharacterController(physics);
    const record = physics.recordFor(controller.collider);
    expect(physics.forgetCollider(record)).toBe(true);
    // Second call: the handle may still be live, but the registry no longer
    // owns it, so nothing may be dropped on this caller's behalf.
    expect(physics.forgetCollider(record)).toBe(false);
    expect(physics.forgetCollider(null)).toBe(false);
    expect(physics.forgetCollider(undefined)).toBe(false);
    controller.dispose();
  });

  it('refuses to drop a record whose handle has been reissued', () => {
    const first = new CharacterController(physics);
    const stale = physics.recordFor(first.collider);
    expect(stale).not.toBeNull();
    first.dispose();

    // Rapier reuses handles. If a later collider lands on the freed handle, the
    // stale record must not be able to evict it.
    let reissued: CharacterController | null = null;
    for (let i = 0; i < 4 && reissued === null; i++) {
      const candidate = new CharacterController(physics);
      if (candidate.collider.handle === stale?.collider.handle) reissued = candidate;
      else candidate.dispose();
    }
    if (reissued === null) return; // handle was not reused; nothing to assert
    const live = physics.recordFor(reissued.collider);
    expect(physics.forgetCollider(stale)).toBe(false);
    expect(physics.recordFor(reissued.collider)).toBe(live);
    reissued.dispose();
  });
});
