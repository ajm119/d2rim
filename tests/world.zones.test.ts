/**
 * @module tests/world.zones
 *
 * The zone system's logic, tested without a browser.
 *
 * What a unit test can reach here is the *bookkeeping*: which zone is active,
 * what order the lifecycle runs in, whether disposal frees what it owns and
 * leaves alone what it does not, and whether the portal geometry maths agrees
 * with itself. What it cannot reach — colliders, grounding, leaks measured
 * against the renderer's own counters — is what `tools/verify-zones.mjs` is for,
 * and the split is deliberate: everything below runs in milliseconds and gates
 * every commit.
 */

import * as THREE from 'three/webgpu';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { EventBus } from '../src/core/EventBus';
import { ServiceLocator } from '../src/core/ServiceLocator';
import type { GameContext } from '../src/core/types';
import { fireFlicker } from '../src/scene/Fire';
import {
  disposeZoneTree,
  findAnchor,
  markShared,
  measureZone,
  resolveEntryPoint,
  type PortalSpec,
  type Zone,
} from '../src/world/Zone';
import {
  PORTAL_HEIGHT,
  PORTAL_RADIUS,
  isInsidePortal,
  portalAt,
  portalPromptText,
} from '../src/world/Portal';
import { ZoneManager } from '../src/world/ZoneManager';

/* -------------------------------------------------------------------------- */
/* Fixtures                                                                   */
/* -------------------------------------------------------------------------- */

/**
 * The slice of {@link GameContext} the zone system actually touches.
 *
 * The engine, renderer and input handles are genuinely unreachable from here —
 * they need a canvas and a GPU — so they are cast in. Everything the code under
 * test reads is real: a real `EventBus`, a real `ServiceLocator`, a real scene.
 */
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

interface TestZone extends Zone {
  readonly initCalls: number[];
  readonly disposeCalls: number[];
}

let sequence = 0;

/** A zone that records its own lifecycle and builds one real mesh. */
function makeZone(id: string, portals: PortalSpec[] = []): TestZone {
  const root = new THREE.Group();
  root.name = `zone.${id}`;
  const initCalls: number[] = [];
  const disposeCalls: number[] = [];

  return {
    name: `scene.${id}`,
    zoneId: id,
    displayName: id,
    root,
    initCalls,
    disposeCalls,
    entryPoints: [
      { id: 'start', position: { x: 1, y: 2, z: 3 } },
      { id: 'back', position: { x: -4, y: 0, z: 5 } },
    ],
    portals,
    npcAnchors: [
      {
        id: 'someone',
        displayName: 'Someone',
        role: 'test',
        position: { x: 0, y: 0, z: 0 },
        yaw: 0,
        clearRadius: 1,
        note: 'test',
      },
    ],
    init(ctx: GameContext): void {
      initCalls.push(sequence++);
      ctx.scene.add(root);
      const mesh = new THREE.Mesh(
        new THREE.BoxGeometry(1, 1, 1),
        new THREE.MeshBasicMaterial(),
      );
      mesh.name = `${id}.mesh`;
      root.add(mesh);
    },
    dispose(): void {
      disposeCalls.push(sequence++);
    },
  };
}

beforeEach(() => {
  sequence = 0;
});

/* -------------------------------------------------------------------------- */

describe('Zone helpers', () => {
  it('resolves a named entry point', () => {
    const zone = makeZone('a');
    expect(resolveEntryPoint(zone, 'back')?.id).toBe('back');
  });

  it('falls back to the first entry point and warns on an unknown id', () => {
    const zone = makeZone('a');
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    expect(resolveEntryPoint(zone, 'nowhere')?.id).toBe('start');
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });

  it('returns the first entry point when none is named', () => {
    expect(resolveEntryPoint(makeZone('a'), null)?.id).toBe('start');
  });

  it('looks NPC anchors up by id', () => {
    const zone = makeZone('a');
    expect(findAnchor(zone, 'someone')?.displayName).toBe('Someone');
    expect(findAnchor(zone, 'nobody')).toBeNull();
  });
});

describe('disposeZoneTree', () => {
  it('frees geometries, materials and their textures', () => {
    const root = new THREE.Group();
    const geometry = new THREE.BoxGeometry();
    const texture = new THREE.Texture();
    const material = new THREE.MeshBasicMaterial({ map: texture });
    root.add(new THREE.Mesh(geometry, material));

    const geometryDispose = vi.spyOn(geometry, 'dispose');
    const materialDispose = vi.spyOn(material, 'dispose');
    const textureDispose = vi.spyOn(texture, 'dispose');

    const count = disposeZoneTree(root);
    expect(geometryDispose).toHaveBeenCalledOnce();
    expect(materialDispose).toHaveBeenCalledOnce();
    expect(textureDispose).toHaveBeenCalledOnce();
    expect(count).toBeGreaterThanOrEqual(3);
    expect(root.children).toHaveLength(0);
  });

  it('leaves shared resources alone', () => {
    // The failure this prevents: a zone disposes a barrel it cloned, the
    // AssetManager cache still hands that geometry to the next zone, and the
    // next zone renders nothing with no error anywhere.
    const root = new THREE.Group();
    const geometry = markShared(new THREE.BoxGeometry());
    const texture = markShared(new THREE.Texture());
    const material = markShared(new THREE.MeshBasicMaterial({ map: texture }));
    root.add(new THREE.Mesh(geometry, material));

    const geometryDispose = vi.spyOn(geometry, 'dispose');
    const materialDispose = vi.spyOn(material, 'dispose');
    const textureDispose = vi.spyOn(texture, 'dispose');

    disposeZoneTree(root);
    expect(geometryDispose).not.toHaveBeenCalled();
    expect(materialDispose).not.toHaveBeenCalled();
    expect(textureDispose).not.toHaveBeenCalled();
  });

  it('disposes a geometry shared between two meshes exactly once', () => {
    const root = new THREE.Group();
    const geometry = new THREE.BoxGeometry();
    const material = new THREE.MeshBasicMaterial();
    root.add(new THREE.Mesh(geometry, material));
    root.add(new THREE.Mesh(geometry, material));
    const geometryDispose = vi.spyOn(geometry, 'dispose');
    disposeZoneTree(root);
    expect(geometryDispose).toHaveBeenCalledOnce();
  });

  it('disposes lights, which own a shadow map three cannot see', () => {
    const root = new THREE.Group();
    const light = new THREE.PointLight();
    root.add(light);
    const dispose = vi.spyOn(light, 'dispose');
    disposeZoneTree(root);
    expect(dispose).toHaveBeenCalledOnce();
  });

  it('handles array materials', () => {
    const root = new THREE.Group();
    const a = new THREE.MeshBasicMaterial();
    const b = new THREE.MeshBasicMaterial();
    root.add(new THREE.Mesh(new THREE.BoxGeometry(), [a, b]));
    const disposeA = vi.spyOn(a, 'dispose');
    const disposeB = vi.spyOn(b, 'dispose');
    disposeZoneTree(root);
    expect(disposeA).toHaveBeenCalledOnce();
    expect(disposeB).toHaveBeenCalledOnce();
  });
});

describe('measureZone', () => {
  it('counts meshes and triangles', () => {
    const root = new THREE.Group();
    // A box is 12 triangles.
    root.add(new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial()));
    const measured = measureZone(root);
    expect(measured.renderables).toBe(1);
    expect(measured.triangles).toBe(12);
  });

  it('multiplies instanced geometry by the instance count', () => {
    const root = new THREE.Group();
    root.add(
      new THREE.InstancedMesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial(), 10),
    );
    expect(measureZone(root).triangles).toBe(120);
  });
});

describe('ZoneManager', () => {
  it('registers zones and reports them in order', () => {
    const zones = new ZoneManager({ fadeSeconds: 0 });
    zones.register('a', () => makeZone('a'));
    zones.register('b', () => makeZone('b'));
    expect(zones.registered).toEqual(['a', 'b']);
    expect(zones.has('a')).toBe(true);
    expect(zones.has('c')).toBe(false);
  });

  it('refuses a duplicate registration rather than silently replacing', () => {
    const zones = new ZoneManager({ fadeSeconds: 0 });
    zones.register('a', () => makeZone('a'));
    expect(() => zones.register('a', () => makeZone('a'))).toThrow(/already registered/);
  });

  it('loads the start zone during init and publishes progress', async () => {
    const ctx = makeContext();
    const events: string[] = [];
    ctx.events.on('zone:loadStart', (p) => events.push(`start:${p.zoneId}`));
    ctx.events.on('zone:loadProgress', (p) => events.push(`progress:${p.phase}`));
    ctx.events.on('zone:loaded', (p) => events.push(`loaded:${p.zoneId}:${p.entryPoint}`));

    const zones = new ZoneManager({ startZone: 'a', fadeSeconds: 0, enemies: false });
    zones.register('a', () => makeZone('a'));
    await zones.init(ctx);

    expect(zones.activeId).toBe('a');
    expect(events[0]).toBe('start:a');
    expect(events.at(-1)).toBe('loaded:a:start');
    // Progress must be monotonic and reach 1, or a loading bar built on it goes
    // backwards.
    expect(events.filter((e) => e.startsWith('progress'))).toContain('progress:ready');
    expect(ctx.services.has('world.zones')).toBe(true);
  });

  it('reports what the load produced', async () => {
    const ctx = makeContext();
    const zones = new ZoneManager({ startZone: 'a', fadeSeconds: 0, enemies: false });
    zones.register('a', () => makeZone('a'));
    await zones.init(ctx);
    expect(zones.report?.zoneId).toBe('a');
    expect(zones.report?.renderables).toBe(1);
    expect(zones.report?.triangles).toBe(12);
  });

  it('does nothing but warn when the start zone is not registered', async () => {
    const ctx = makeContext();
    const error = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const zones = new ZoneManager({ startZone: 'missing', fadeSeconds: 0 });
    await zones.init(ctx);
    expect(zones.activeId).toBeNull();
    expect(error).toHaveBeenCalled();
    error.mockRestore();
  });

  it('unloads the previous zone before loading the next', async () => {
    const ctx = makeContext();
    const a = makeZone('a');
    const b = makeZone('b');
    const zones = new ZoneManager({ startZone: 'a', fadeSeconds: 0, enemies: false });
    zones.register('a', () => a);
    zones.register('b', () => b);
    await zones.init(ctx);

    const order: string[] = [];
    ctx.events.on('zone:travelStart', (p) => order.push(`travelStart:${p.from}->${p.to}`));
    ctx.events.on('zone:unloading', (p) => order.push(`unloading:${p.zoneId}`));
    ctx.events.on('zone:unloaded', (p) => order.push(`unloaded:${p.zoneId}`));
    ctx.events.on('zone:loadStart', (p) => order.push(`loadStart:${p.zoneId}`));
    ctx.events.on('zone:travelEnd', (p) => order.push(`travelEnd:${p.zoneId}`));

    await zones.travelTo('b', 'back');

    expect(order).toEqual([
      'travelStart:a->b',
      'unloading:a',
      'unloaded:a',
      'loadStart:b',
      'travelEnd:b',
    ]);
    expect(zones.activeId).toBe('b');
    expect(a.disposeCalls).toHaveLength(1);
    // The old zone's subtree must be off the scene graph, not merely hidden.
    expect(ctx.scene.children).not.toContain(a.root);
    expect(ctx.scene.children).toContain(b.root);
  });

  it('constructs a fresh instance on every load', async () => {
    const ctx = makeContext();
    const built: TestZone[] = [];
    const zones = new ZoneManager({ startZone: 'a', fadeSeconds: 0, enemies: false });
    zones.register('a', () => {
      const zone = makeZone('a');
      built.push(zone);
      return zone;
    });
    zones.register('b', () => makeZone('b'));
    await zones.init(ctx);
    await zones.travelTo('b');
    await zones.travelTo('a');
    // Two distinct objects, not one retained instance re-initialised: a retained
    // zone holds its geometries for the whole session, which is the leak the
    // factory registration exists to prevent.
    expect(built).toHaveLength(2);
    expect(built[0]).not.toBe(built[1]);
    expect(built[0]?.disposeCalls).toHaveLength(1);
  });

  it('serialises concurrent travel rather than racing two teardowns', async () => {
    const ctx = makeContext();
    const zones = new ZoneManager({ startZone: 'a', fadeSeconds: 0, enemies: false });
    zones.register('a', () => makeZone('a'));
    zones.register('b', () => makeZone('b'));
    zones.register('c', () => makeZone('c'));
    await zones.init(ctx);

    // What a player mashing E inside a portal volume produces.
    await Promise.all([zones.travelTo('b'), zones.travelTo('c'), zones.travelTo('b')]);
    expect(zones.activeId).toBe('b');
    expect(zones.travelling).toBe(false);
    expect(ctx.scene.children.filter((child) => child.name.startsWith('zone.'))).toHaveLength(1);
  });

  it('refuses to travel to an unregistered zone and stays where it is', async () => {
    const ctx = makeContext();
    const error = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const zones = new ZoneManager({ startZone: 'a', fadeSeconds: 0, enemies: false });
    zones.register('a', () => makeZone('a'));
    await zones.init(ctx);
    await zones.travelTo('nowhere');
    expect(zones.activeId).toBe('a');
    expect(error).toHaveBeenCalled();
    error.mockRestore();
  });

  it('forwards the update phases to the active zone only', async () => {
    const ctx = makeContext();
    const updates: string[] = [];
    const zone = makeZone('a');
    const withUpdates: Zone = {
      ...zone,
      update: () => updates.push('update'),
      fixedUpdate: () => updates.push('fixed'),
      lateUpdate: () => updates.push('late'),
      init: zone.init.bind(zone),
    };
    const zones = new ZoneManager({ startZone: 'a', fadeSeconds: 0, enemies: false });
    zones.register('a', () => withUpdates);
    await zones.init(ctx);

    zones.fixedUpdate(ctx, 1 / 60);
    zones.update(ctx, 1 / 60);
    zones.lateUpdate(ctx, 1 / 60);
    expect(updates).toEqual(['fixed', 'update', 'late']);

    zones.dispose();
    zones.fixedUpdate(ctx, 1 / 60);
    expect(updates).toEqual(['fixed', 'update', 'late']);
  });

  it('tears the active zone down on dispose', async () => {
    const ctx = makeContext();
    const zone = makeZone('a');
    const zones = new ZoneManager({ startZone: 'a', fadeSeconds: 0, enemies: false });
    zones.register('a', () => zone);
    await zones.init(ctx);
    zones.dispose();
    expect(zone.disposeCalls).toHaveLength(1);
    expect(ctx.scene.children).not.toContain(zone.root);
    expect(ctx.services.has('world.zones')).toBe(false);
  });
});

/* -------------------------------------------------------------------------- */

const PORTAL: PortalSpec = {
  id: 'den',
  targetZone: 'denOfEvil',
  targetEntry: 'cave-mouth',
  position: { x: 10, y: 2, z: -5 },
  label: 'the Den of Evil',
};

describe('Portal geometry', () => {
  it('writes the prompt the interact binding actually uses', () => {
    expect(portalPromptText(PORTAL)).toBe('Press E to enter the Den of Evil');
    expect(portalPromptText({ ...PORTAL, verb: 'climb back out to' })).toBe(
      'Press E to climb back out to the Den of Evil',
    );
  });

  it('accepts a point at the centre and rejects one outside the radius', () => {
    expect(isInsidePortal(PORTAL, 10, 2, -5)).toBe(true);
    expect(isInsidePortal(PORTAL, 10 + PORTAL_RADIUS - 0.01, 2, -5)).toBe(true);
    expect(isInsidePortal(PORTAL, 10 + PORTAL_RADIUS + 0.01, 2, -5)).toBe(false);
  });

  it('bounds the volume vertically, with slack below for sloping ground', () => {
    expect(isInsidePortal(PORTAL, 10, 2 + PORTAL_HEIGHT - 0.01, -5)).toBe(true);
    expect(isInsidePortal(PORTAL, 10, 2 + PORTAL_HEIGHT + 0.01, -5)).toBe(false);
    expect(isInsidePortal(PORTAL, 10, 2 - 1.1, -5)).toBe(true);
    expect(isInsidePortal(PORTAL, 10, 2 - 1.3, -5)).toBe(false);
  });

  it('honours a per-portal radius and height', () => {
    const wide = { ...PORTAL, radius: 6, height: 8 };
    expect(isInsidePortal(wide, 15, 2, -5)).toBe(true);
    expect(isInsidePortal(wide, 10, 9, -5)).toBe(true);
  });

  it('picks the nearest overlapping portal, not the first in the array', () => {
    const far: PortalSpec = { ...PORTAL, id: 'far', position: { x: 12, y: 2, z: -5 } };
    const near: PortalSpec = { ...PORTAL, id: 'near', position: { x: 10.2, y: 2, z: -5 } };
    expect(portalAt([far, near], 10, 2, -5)?.id).toBe('near');
    expect(portalAt([near, far], 10, 2, -5)?.id).toBe('near');
  });

  it('returns null when the player is in no portal', () => {
    expect(portalAt([PORTAL], 0, 0, 0)).toBeNull();
    expect(portalAt([], 10, 2, -5)).toBeNull();
  });
});

/* -------------------------------------------------------------------------- */
/* Enemy placement                                                            */
/* -------------------------------------------------------------------------- */

/**
 * The deferral this section covers, and why it was worth a test.
 *
 * `ZoneManager` is registered *before* `CombatSystem` (its zones build
 * colliders that the player needs during his own `init`), so when the start
 * zone loads there is no combat system to register enemies with. The old code
 * stashed the spawn table and fired `void this.#spawnEnemies(spawns)` from the
 * first `update`: an unawaited promise with no completion signal of any kind.
 *
 * Two things fell out of that and both were invisible.
 *
 *  1. `report.enemies` was snapshotted at the end of the load — always before
 *     a single skeleton existed — so a populated Blood Moor permanently
 *     reported `0 enemies`.
 *  2. Nothing could wait for the encounter. The skeleton GLBs arrive on a
 *     *macrotask*, and a harness driving the world with `await stepFrames(1)`
 *     only ever yields microtasks, so the fight could not appear for as long as
 *     the harness kept asking for frames. Sixty driven frames, sixty-seven
 *     seconds of wall clock, zero enemies — measured.
 */

/** A zone with an encounter table, and nothing else different. */
function makeCombatZone(id: string, spawns = 3): TestZone {
  const zone = makeZone(id);
  return {
    ...zone,
    enemySpawns: Array.from({ length: spawns }, (_, i) => ({
      variant: 'minion',
      x: i * 2,
      z: 0,
    })),
  };
}

/**
 * Whether a promise has *already* settled, decided across a macrotask boundary.
 *
 * `Promise.race([p.then(...), Promise.resolve('pending')])` does not work and
 * is worth naming: `Promise.resolve` settles a microtask earlier than any
 * `.then` chain, so that form reports every promise as pending. A `setTimeout`
 * runs after the microtask queue has drained, which is the question being
 * asked.
 */
async function settledNow(promise: Promise<unknown>): Promise<'ready' | 'pending'> {
  return Promise.race([
    promise.then(() => 'ready' as const),
    new Promise<'pending'>((resolve) => {
      setTimeout(() => resolve('pending'), 0);
    }),
  ]);
}

/** Just enough `AssetManager` for the director to decide it has no models. */
function stubAssets(): unknown {
  return { has: () => false, loadGLTF: async () => ({ scene: null, animations: [] }), pin: () => undefined };
}

describe('ZoneManager enemy placement', () => {
  it('resolves enemiesReady immediately for a zone with no encounter', async () => {
    const ctx = makeContext();
    const zones = new ZoneManager({ startZone: 'a', fadeSeconds: 0 });
    zones.register('a', () => makeZone('a'));
    await zones.init(ctx);
    // Must be already resolved, not merely resolvable: a caller awaiting the
    // encounter of a safe zone has to fall straight through.
    await expect(settledNow(zones.enemiesReady)).resolves.toBe('ready');
    expect(zones.director).toBeNull();
  });

  it('defers placement when combat is not up yet, and retries until it is', async () => {
    const ctx = makeContext();
    ctx.services.register('assets', stubAssets());
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const info = vi.spyOn(console, 'info').mockImplementation(() => undefined);

    const zones = new ZoneManager({ startZone: 'a', fadeSeconds: 0 });
    zones.register('a', () => makeCombatZone('a'));
    await zones.init(ctx);

    // The director exists and its models have been asked for, but nothing is
    // placed: this is the state the boot path is always in.
    expect(zones.director).not.toBeNull();
    expect(zones.director?.loaded).toBe(true);
    expect(zones.director?.ready).toBe(false);
    expect(zones.report?.enemies).toBe(0);

    // Ticking with combat still missing must not consume the deferral. The old
    // code fired once and gave up silently.
    zones.update(ctx, 1 / 60);
    expect(zones.director?.ready).toBe(false);

    ctx.services.register('physics.world', {});
    ctx.services.register('combat', {});
    zones.update(ctx, 1 / 60);
    expect(zones.director?.ready).toBe(true);
    await expect(settledNow(zones.enemiesReady)).resolves.toBe('ready');

    warn.mockRestore();
    info.mockRestore();
  });

  it('places immediately, and awaits placement, when combat already exists', async () => {
    const ctx = makeContext();
    ctx.services.register('assets', stubAssets());
    ctx.services.register('physics.world', {});
    ctx.services.register('combat', {});
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const info = vi.spyOn(console, 'info').mockImplementation(() => undefined);

    const zones = new ZoneManager({ startZone: 'a', fadeSeconds: 0 });
    zones.register('a', () => makeCombatZone('a'));
    await zones.init(ctx);

    // This is the travel path, and it is the one that already worked.
    expect(zones.director?.ready).toBe(true);
    await expect(settledNow(zones.enemiesReady)).resolves.toBe('ready');

    warn.mockRestore();
    info.mockRestore();
  });

  it('releases a pending enemiesReady when the zone is unloaded underneath it', async () => {
    const ctx = makeContext();
    ctx.services.register('assets', stubAssets());
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const info = vi.spyOn(console, 'info').mockImplementation(() => undefined);

    const zones = new ZoneManager({ startZone: 'a', fadeSeconds: 0 });
    zones.register('a', () => makeCombatZone('a'));
    zones.register('b', () => makeZone('b'));
    await zones.init(ctx);
    const waited = zones.enemiesReady;

    // A player who walks straight back out of the moor. Whoever was awaiting
    // that zone's encounter must not wait for ever.
    await zones.travelTo('b');
    await expect(settledNow(waited)).resolves.toBe('ready');

    warn.mockRestore();
    info.mockRestore();
  });

  it('skips enemies entirely when the manager is told to', async () => {
    const ctx = makeContext();
    ctx.services.register('assets', stubAssets());
    ctx.services.register('physics.world', {});
    ctx.services.register('combat', {});
    const zones = new ZoneManager({ startZone: 'a', fadeSeconds: 0, enemies: false });
    zones.register('a', () => makeCombatZone('a'));
    await zones.init(ctx);
    expect(zones.director).toBeNull();
    expect(zones.report?.enemies).toBe(0);
  });
});

describe('fireFlicker', () => {
  it('stays close to unity so a fire never reads as a strobe', () => {
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < 20_000; i++) {
      const value = fireFlicker(i * 0.0037);
      min = Math.min(min, value);
      max = Math.max(max, value);
    }
    expect(min).toBeGreaterThan(0.82);
    expect(max).toBeLessThan(1.18);
  });

  it('is a pure function of time, so captures reproduce', () => {
    expect(fireFlicker(12.5)).toBe(fireFlicker(12.5));
    expect(fireFlicker(12.5, 1.3)).toBe(fireFlicker(13.8));
  });

  it('decorrelates two fires given different phases', () => {
    // Torches in a row must not pulse in unison; a shared curve is the tell.
    let apart = 0;
    for (let i = 0; i < 500; i++) {
      const t = i * 0.05;
      if (Math.abs(fireFlicker(t, 0) - fireFlicker(t, 1.37)) > 0.02) apart++;
    }
    expect(apart).toBeGreaterThan(300);
  });
});

/* -------------------------------------------------------------------------- */
/* Per-zone exposure and grade                                                 */
/* -------------------------------------------------------------------------- */

/**
 * A stand-in for the two objects on `PostStack` a zone trim touches.
 *
 * Faked rather than constructed, because a real `PostStack` needs a GPU. What
 * is *not* faked is the thing under test: the manager's bookkeeping of what it
 * replaced and its obligation to put it back. The one property that matters is
 * that the post stack outlives every zone, so any trim a zone leaves behind
 * becomes the next zone's baseline — which is a bug that would show up as "the
 * Blood Moor is lit like a cave" three transitions later and be untraceable.
 */
function makePost(): {
  service: unknown;
  exposure: () => number;
  settings: () => Record<string, unknown>;
} {
  let exposure = 1.72;
  const settings: Record<string, unknown> = {
    contrast: 1.12,
    contrastPivot: 0.235,
    saturation: 0.94,
    temperature: -520,
  };
  return {
    service: {
      composite: {
        get exposure() {
          return exposure;
        },
        setExposure(value: number) {
          exposure = value;
        },
      },
      grade: {
        get settings() {
          return settings;
        },
        set(changes: Record<string, unknown>) {
          Object.assign(settings, changes);
        },
      },
    },
    exposure: () => exposure,
    settings: () => ({ ...settings }),
  };
}

describe('ZoneManager per-zone grade', () => {
  const gradedZone = (id: string, stops: number, grade: Record<string, unknown>): Zone => ({
    ...makeZone(id),
    grade: { stops, grade } as NonNullable<Zone['grade']>,
  });

  it('applies a zone trim as stops on top of the shipping key, not as a replacement', async () => {
    const ctx = makeContext();
    const post = makePost();
    ctx.services.register('render.post', post.service);
    const manager = new ZoneManager({ startZone: 'cave', fadeSeconds: 0 });
    manager.register('cave', () => gradedZone('cave', 1, { contrast: 0.94 }));
    await manager.init(ctx);

    // +1 stop doubles. An implementation that *set* the exposure rather than
    // scaling it would agree with a correct one only at a baseline of 1.0,
    // which is exactly the value the shipping frame graph does not use.
    expect(post.exposure()).toBeCloseTo(3.44, 5);
    expect(post.settings().contrast).toBe(0.94);
    // Untouched keys must survive: the trim is a patch, not a preset.
    expect(post.settings().saturation).toBe(0.94);
    expect(post.settings().temperature).toBe(-520);
  });

  it('puts the frame back exactly as it found it when the zone unloads', async () => {
    const ctx = makeContext();
    const post = makePost();
    ctx.services.register('render.post', post.service);
    const before = { exposure: post.exposure(), settings: post.settings() };

    const manager = new ZoneManager({ startZone: 'cave', fadeSeconds: 0 });
    manager.register('cave', () => gradedZone('cave', 0.5, { contrast: 0.8, saturation: 0.7 }));
    manager.register('field', () => makeZone('field'));
    await manager.init(ctx);
    expect(post.exposure()).not.toBeCloseTo(before.exposure, 5);

    await manager.travelTo('field');
    expect(post.exposure()).toBeCloseTo(before.exposure, 5);
    expect(post.settings()).toEqual(before.settings);
  });

  it('never lets one zone become the next zone one’s baseline', async () => {
    const ctx = makeContext();
    const post = makePost();
    ctx.services.register('render.post', post.service);
    const base = post.exposure();

    const manager = new ZoneManager({ startZone: 'a', fadeSeconds: 0 });
    manager.register('a', () => gradedZone('a', 1, { contrast: 0.5 }));
    manager.register('b', () => gradedZone('b', 2, { contrast: 0.25 }));
    await manager.init(ctx);
    await manager.travelTo('b');

    // b's two stops are measured from the *shipping* key, not from a's doubled
    // one. Compounding is the failure this test exists to catch: a lap of
    // three graded zones would otherwise walk the exposure away permanently.
    expect(post.exposure()).toBeCloseTo(base * 4, 5);
    expect(post.settings().contrast).toBe(0.25);

    await manager.travelTo('a');
    expect(post.exposure()).toBeCloseTo(base * 2, 5);
    expect(post.settings().contrast).toBe(0.5);
  });

  it('restores the frame when the manager itself is disposed', async () => {
    const ctx = makeContext();
    const post = makePost();
    ctx.services.register('render.post', post.service);
    const base = post.exposure();
    const baseSettings = post.settings();

    const manager = new ZoneManager({ startZone: 'cave', fadeSeconds: 0 });
    manager.register('cave', () => gradedZone('cave', 1.5, { vignette: 0.5 }));
    await manager.init(ctx);
    manager.dispose();

    expect(post.exposure()).toBeCloseTo(base, 5);
    expect(post.settings()).toEqual(baseSettings);
  });

  it('leaves the frame alone for a zone that declares no trim', async () => {
    const ctx = makeContext();
    const post = makePost();
    ctx.services.register('render.post', post.service);
    const base = post.exposure();

    const manager = new ZoneManager({ startZone: 'plain', fadeSeconds: 0 });
    manager.register('plain', () => makeZone('plain'));
    await manager.init(ctx);
    expect(post.exposure()).toBe(base);
    expect(post.settings()).toEqual({
      contrast: 1.12,
      contrastPivot: 0.235,
      saturation: 0.94,
      temperature: -520,
    });
  });
});
