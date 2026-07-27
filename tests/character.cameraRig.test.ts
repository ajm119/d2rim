/**
 * @module tests/character.cameraRig
 *
 * The third-person spring arm's collision behaviour, tested without a browser.
 *
 * ### What this is guarding against
 *
 * A cold boot into the Rogue Encampment shipped with the camera 0.38 m behind
 * the Barbarian: half the opening frame was the back of his head. The cause was
 * world-side — six decorative torch posts merged into one `Mesh` spanning a
 * 17.5 m ring, from which the collider pass derived a single 35 m box across
 * the whole camp at chest height — and it is fixed at the source. But the arm
 * had no floor, so *any* prop was able to do this, and the next one would have
 * done it somewhere nobody was looking.
 *
 * So the assertions below are about the rig's promises rather than about that
 * one collider: whatever the sphere cast reports, and wherever the player is
 * standing, the camera stays far enough back to be a camera. The measurement of
 * the real arm at the real spawns lives in `tools/verify-camera-arm.mjs`, which
 * needs a live physics world and therefore a browser.
 */

import * as THREE from 'three/webgpu';
import { beforeEach, describe, expect, it } from 'vitest';

import { EventBus } from '../src/core/EventBus';
import { ServiceLocator } from '../src/core/ServiceLocator';
import type { GameContext } from '../src/core/types';
import {
  CameraRig,
  MIN_ARM_FRACTION,
  damp,
  smootherstep,
} from '../src/character/CameraRig';
import { PlayerKey } from '../src/character/PlayerController';
import { PhysicsWorldKey } from '../src/physics/PhysicsWorld';
import {
  COLLISION_GROUPS,
  CollisionLayer,
  filterOf,
  membershipOf,
} from '../src/physics/Layers';

/* -------------------------------------------------------------------------- */
/* Fixtures                                                                   */
/* -------------------------------------------------------------------------- */

/**
 * The shortest arm any zone spawn is allowed to produce.
 *
 * Not a tuning constant — a legibility one, and it was measured off captures:
 * at 1.2 m the Barbarian's head and shoulder own the left half of a 1280x720
 * frame at 55° FOV, and the opening shot of a zone is exactly where that is
 * least affordable. 1.5 m sits just under the rig's own floor, so the assertion
 * fails on a real regression rather than on a tuning nudge.
 */
const READABLE_ARM = 1.5;

/**
 * The authored ground positions the three zones put the player down on.
 *
 * Hard-coded rather than read off the zone modules, deliberately: two of the
 * three resolve their entry points against built terrain or a generated
 * dungeon, neither of which exists without a GPU, and a test that silently
 * skipped those two would be worse than one that states its coordinates. They
 * are here to prove the rig behaves the same wherever it is placed — a spawn
 * that moves does not invalidate the assertion.
 */
const ZONE_SPAWNS = [
  { zone: 'encampment', entry: 'camp-centre', x: 1.4, y: 0.04, z: 8.2, yaw: Math.PI },
  { zone: 'encampment', entry: 'gate', x: 0.9, y: 0.05, z: 19.5, yaw: Math.PI },
  { zone: 'bloodMoor', entry: 'from-camp', x: 0, y: 0.1, z: 34.0, yaw: 0 },
  { zone: 'bloodMoor', entry: 'spawn', x: 3.1, y: 0.1, z: -5.2, yaw: 0 },
  { zone: 'denOfEvil', entry: 'cave-mouth', x: 0, y: 3.0, z: 0, yaw: 0 },
] as const;

interface FakeHit {
  distance: number;
  kind: string;
  label: string;
}

/** A physics world that reports exactly the hit a test asks it to. */
class FakePhysics {
  ready = true;
  hit: FakeHit | null = null;
  /** Every query the rig issued, so the filtering can be asserted on. */
  readonly queries: { layers: number | undefined; solidOnly: boolean | undefined }[] = [];

  sphereCast(
    _origin: THREE.Vector3,
    _direction: THREE.Vector3,
    _radius: number,
    _maxDistance: number,
    options: { layers?: number; solidOnly?: boolean } = {},
  ): {
    distance: number;
    point: THREE.Vector3;
    normal: THREE.Vector3;
    collider: unknown;
    record: { kind: string; label: string } | null;
  } | null {
    this.queries.push({ layers: options.layers, solidOnly: options.solidOnly });
    const hit = this.hit;
    if (hit === null) return null;
    return {
      distance: hit.distance,
      point: new THREE.Vector3(),
      normal: new THREE.Vector3(0, 0, 1),
      collider: null,
      record: { kind: hit.kind, label: hit.label },
    };
  }
}

/** Just enough `PlayerController` for the rig: a pose, a size and a velocity. */
class FakePlayer {
  height = 1.71;
  yaw = 0;
  pitch = 0;
  eyeHeight = 1.57;
  controller = null;
  readonly object = new THREE.Object3D();
  readonly position = new THREE.Vector3();
  readonly velocity = new THREE.Vector3();

  placeAt(x: number, y: number, z: number, yaw: number): void {
    this.position.set(x, y, z);
    this.object.position.set(x, y, z);
    this.yaw = yaw;
  }
}

function makeContext(): GameContext {
  return {
    engine: {} as GameContext['engine'],
    scene: new THREE.Scene(),
    camera: new THREE.PerspectiveCamera(),
    renderer: {} as GameContext['renderer'],
    input: { wasPressed: () => false } as unknown as GameContext['input'],
    events: new EventBus(),
    time: { elapsed: 0, delta: 1 / 60, frame: 0, scale: 1 },
    services: new ServiceLocator(),
  };
}

let ctx: GameContext;
let player: FakePlayer;
let physics: FakePhysics;
let rig: CameraRig;

function boot(options: ConstructorParameters<typeof CameraRig>[0] = {}): void {
  ctx = makeContext();
  player = new FakePlayer();
  physics = new FakePhysics();
  ctx.services.register(PlayerKey, player as never);
  ctx.services.register(PhysicsWorldKey, physics as never);
  rig = new CameraRig(options);
  rig.init(ctx);
}

/** Run the rig for `frames` at a fixed 60 Hz, collecting the arm each frame. */
function run(frames: number, dt = 1 / 60): number[] {
  const samples: number[] = [];
  for (let i = 0; i < frames; i++) {
    rig.lateUpdate(ctx, dt);
    samples.push(rig.armLength);
  }
  return samples;
}

/** The distance from the camera to the pivot it is orbiting. */
function separation(): number {
  const pivot = new THREE.Vector3(
    player.object.position.x,
    player.object.position.y + rig.pivotHeight,
    player.object.position.z,
  );
  return ctx.camera.position.distanceTo(pivot);
}

beforeEach(() => {
  boot();
});

/* -------------------------------------------------------------------------- */
/* Maths                                                                      */
/* -------------------------------------------------------------------------- */

describe('smoothing helpers', () => {
  it('damps toward the target independently of the frame rate', () => {
    // One 1/30 s step and two 1/60 s steps must land in the same place, or the
    // camera behaves differently at 30 and 144 fps.
    const coarse = damp(0, 1, 8, 1 / 30);
    const fine = damp(damp(0, 1, 8, 1 / 60), 1, 8, 1 / 60);
    expect(fine).toBeCloseTo(coarse, 12);
  });

  it('smootherstep is clamped and flat at both ends', () => {
    expect(smootherstep(-1)).toBe(0);
    expect(smootherstep(2)).toBe(1);
    expect(smootherstep(0.5)).toBeCloseTo(0.5, 12);
  });
});

/* -------------------------------------------------------------------------- */
/* The floor                                                                  */
/* -------------------------------------------------------------------------- */

describe('arm floor', () => {
  it('derives the floor from the character, not from a constant', () => {
    expect(rig.restLength).toBeCloseTo(player.height * 2.1, 6);
    expect(rig.minLength).toBeCloseTo(rig.restLength * MIN_ARM_FRACTION, 6);
    expect(rig.minLength).toBeGreaterThan(READABLE_ARM);
  });

  it('refuses to collapse onto the player when a prop is right behind him', () => {
    // 0.44 m is what the encampment's stray 35 m box actually reported.
    physics.hit = { distance: 0.44, kind: 'prop', label: 'camp.torchposts' };
    run(30);
    expect(rig.armLength).toBeGreaterThanOrEqual(rig.minLength - 1e-9);
    expect(rig.obstruction).toBe('camp.torchposts');
  });

  it('holds the floor even when the obstacle reports zero distance', () => {
    physics.hit = { distance: 0, kind: 'prop', label: 'inside.something' };
    run(30);
    expect(rig.armLength).toBeCloseTo(rig.minLength, 6);
  });

  it('clamps an over-large configured floor to the rest length', () => {
    boot({ distance: 2, minDistance: 9 });
    expect(rig.minLength).toBe(2);
  });

  for (const spawn of ZONE_SPAWNS) {
    it(`keeps a readable arm at ${spawn.zone}/${spawn.entry}`, () => {
      boot();
      player.placeAt(spawn.x, spawn.y, spawn.z, spawn.yaw);
      // The worst case a spawn can present: something solid touching the pivot.
      physics.hit = { distance: 0, kind: 'prop', label: 'worst-case' };
      run(90);
      expect(rig.armLength).toBeGreaterThanOrEqual(READABLE_ARM);
      // And the camera really is that far from him — the floor is on the arm,
      // but what the player sees is the eye-to-pivot separation, and the
      // shoulder offset means the two are not the same number.
      expect(separation()).toBeGreaterThanOrEqual(READABLE_ARM);
    });
  }
});

/* -------------------------------------------------------------------------- */
/* What may stop the arm                                                      */
/* -------------------------------------------------------------------------- */

describe('probe filtering', () => {
  it('queries only the camera layer, and only solids', () => {
    run(1);
    expect(physics.queries.length).toBeGreaterThan(0);
    for (const query of physics.queries) {
      expect(query.layers).toBe(CollisionLayer.Camera);
      expect(query.solidOnly).toBe(true);
    }
  });

  it('the camera layer cannot see the player or a trigger', () => {
    // The mask is the first line of defence, and Rapier's test is symmetric, so
    // assert it in the direction that actually decides the outcome: the probe's
    // filter against each collider's membership. If the player or a trigger is
    // ever given `Camera` membership, the arm starts colliding with the body it
    // is anchored to and this fails before anyone has to look at a frame.
    const probeFilter = CollisionLayer.Camera;
    expect(membershipOf(COLLISION_GROUPS.player) & probeFilter).toBe(0);
    expect(membershipOf(COLLISION_GROUPS.trigger) & probeFilter).toBe(0);
    // ...and the geometry it must see, still visible.
    expect(membershipOf(COLLISION_GROUPS.terrain) & probeFilter).not.toBe(0);
    expect(membershipOf(COLLISION_GROUPS.prop) & probeFilter).not.toBe(0);
    // The probe's own membership must be broad, not `Camera`: a query that
    // declares itself a member of the layer it wants to hit fails the other
    // half of the symmetric test against every collider in the world.
    expect(filterOf(COLLISION_GROUPS.terrain) & CollisionLayer.Camera).not.toBe(0);
  });

  for (const kind of ['character', 'trigger', 'projectile']) {
    it(`ignores a ${kind} hit even if one reaches the query`, () => {
      physics.hit = { distance: 0.2, kind, label: `stray.${kind}` };
      run(30);
      expect(rig.armLength).toBeCloseTo(rig.restLength, 4);
      expect(rig.obstruction).toBeNull();
    });
  }

  it('still stops on terrain and props', () => {
    physics.hit = { distance: 2.4, kind: 'prop', label: 'camp.tent.3' };
    run(30);
    expect(rig.armLength).toBeCloseTo(2.4 - 0.06, 4);
  });
});

/* -------------------------------------------------------------------------- */
/* Motion                                                                     */
/* -------------------------------------------------------------------------- */

describe('pull-in and recovery', () => {
  it('pulls in on the frame the obstacle appears', () => {
    run(5);
    expect(rig.armLength).toBeCloseTo(rig.restLength, 4);
    physics.hit = { distance: 2.0, kind: 'prop', label: 'wall' };
    run(1);
    expect(rig.armLength).toBeCloseTo(1.94, 4);
  });

  it('recovers smoothly: monotone, rate-limited, and with no terminal jolt', () => {
    physics.hit = { distance: 0, kind: 'prop', label: 'wall' };
    run(20);
    const collapsed = rig.armLength;
    physics.hit = null;
    const samples = run(240);

    const steps = samples.map((value, i) => value - (i === 0 ? collapsed : samples[i - 1]!));
    // Monotone out, never back in.
    for (const step of steps) expect(step).toBeGreaterThanOrEqual(-1e-9);
    // Rate-limited: no frame moves the camera further than `extendRate * dt`.
    for (const step of steps) expect(step).toBeLessThanOrEqual(5 / 60 + 1e-9);
    // Eased, not cut: the last frame before rest moves far less than the first,
    // which is exactly what a pure rate limit does not do.
    const first = steps[0]!;
    const last = steps.at(-1)!;
    expect(last).toBeLessThan(first * 0.25);
    expect(samples.at(-1)!).toBeCloseTo(rig.restLength, 3);
  });

  it('does not fire the camera backwards when a prop is only brushed past', () => {
    physics.hit = { distance: 1.5, kind: 'prop', label: 'tree' };
    run(30);
    const held = rig.armLength;
    physics.hit = null;
    // Two frames of freedom is 16 cm at the extend rate, and not a metre.
    run(2);
    expect(rig.armLength - held).toBeLessThanOrEqual((5 * 2) / 60 + 1e-9);
  });
});

/* -------------------------------------------------------------------------- */
/* Teleports                                                                  */
/* -------------------------------------------------------------------------- */

describe('zone travel', () => {
  it('snaps the pivot instead of flying the camera across the world', () => {
    run(10);
    const before = ctx.camera.position.clone();
    // A zone transition: same rig, entirely different coordinates.
    player.placeAt(6, -0.9, 11.5, 0);
    run(1);
    expect(ctx.camera.position.distanceTo(before)).toBeGreaterThan(3);
    // One frame later the camera is already framing the new position properly,
    // rather than a second of travel away from it.
    expect(separation()).toBeGreaterThan(rig.restLength * 0.9);
  });

  it('arrives at rest length even if the previous zone left the arm collapsed', () => {
    physics.hit = { distance: 0, kind: 'prop', label: 'old-zone-wall' };
    run(20);
    expect(rig.armLength).toBeCloseTo(rig.minLength, 6);
    physics.hit = null;
    player.placeAt(-1.5, -0.03, 31.5, 0);
    run(1);
    expect(rig.armLength).toBeCloseTo(rig.restLength, 4);
  });

  it('does not mistake ordinary sprinting for a teleport', () => {
    // The teleport branch resets the arm to rest length, so a collapsed arm is
    // the cleanest probe for whether it fired: hold the camera against a wall
    // and sprint. 7 m/s is 0.12 m per frame, nowhere near the threshold, so the
    // arm must stay pinned all the way along.
    physics.hit = { distance: 0.5, kind: 'prop', label: 'corridor' };
    run(20);
    const pinned = rig.armLength;
    player.velocity.set(0, 0, -7);
    for (let i = 0; i < 60; i++) {
      player.placeAt(0, 0, player.object.position.z - 7 / 60, 0);
      rig.lateUpdate(ctx, 1 / 60);
    }
    expect(rig.armLength).toBeCloseTo(pinned, 6);
  });
});

/* -------------------------------------------------------------------------- */
/* Mode blend                                                                 */
/* -------------------------------------------------------------------------- */

describe('mode', () => {
  it('starts third person and reaches first person over the transition', () => {
    expect(rig.mode).toBe('third');
    expect(rig.blend).toBe(0);
    rig.setMode('first');
    run(Math.ceil(0.38 * 60) + 2);
    expect(rig.blend).toBe(1);
  });

  it('emits on a real change only', () => {
    const seen: string[] = [];
    ctx.events.on('camera:mode', ({ mode }) => seen.push(mode));
    rig.setMode('third');
    rig.setMode('first');
    rig.setMode('first');
    expect(seen).toEqual(['first']);
  });
});
