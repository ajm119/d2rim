import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';

import {
  WeaponHitbox,
  closestPointsOnSegments,
  localBladeExtents,
  resolveWeaponAnchor,
  sweepSegmentAgainstCapsule,
  type CapsuleTarget,
  type WeaponAnchor,
} from '../src/combat/Hitbox';

const v = (x: number, y: number, z: number): THREE.Vector3 => new THREE.Vector3(x, y, z);

describe('closestPointsOnSegments', () => {
  it('finds the crossing point of two perpendicular segments', () => {
    const result = closestPointsOnSegments(v(-1, 0, 0), v(1, 0, 0), v(0, 0, -1), v(0, 0, 1));
    expect(result.distance).toBeCloseTo(0, 6);
    expect(result.pointA.length()).toBeCloseTo(0, 6);
  });

  it('measures the gap between parallel segments', () => {
    const result = closestPointsOnSegments(v(0, 0, 0), v(1, 0, 0), v(0, 2, 0), v(1, 2, 0));
    expect(result.distance).toBeCloseTo(2, 6);
  });

  it('clamps to the endpoints when the closest approach is off the segments', () => {
    const result = closestPointsOnSegments(v(0, 0, 0), v(1, 0, 0), v(5, 0, 0), v(6, 0, 0));
    expect(result.distance).toBeCloseTo(4, 6);
    expect(result.s).toBe(1);
    expect(result.t).toBe(0);
  });

  it('handles a degenerate first segment without producing NaN', () => {
    const result = closestPointsOnSegments(v(0, 1, 0), v(0, 1, 0), v(-1, 0, 0), v(1, 0, 0));
    expect(Number.isFinite(result.distance)).toBe(true);
    expect(result.distance).toBeCloseTo(1, 6);
  });

  it('handles two degenerate segments', () => {
    const result = closestPointsOnSegments(v(0, 0, 0), v(0, 0, 0), v(3, 0, 0), v(3, 0, 0));
    expect(result.distance).toBeCloseTo(3, 6);
  });

  it('is symmetric in its arguments', () => {
    const a = closestPointsOnSegments(v(-2, 1, 0), v(2, 1, 0), v(0, -1, -3), v(0, -1, 3));
    const b = closestPointsOnSegments(v(0, -1, -3), v(0, -1, 3), v(-2, 1, 0), v(2, 1, 0));
    expect(a.distance).toBeCloseTo(b.distance, 6);
  });
});

describe('sweepSegmentAgainstCapsule', () => {
  const base = v(0, 0.4, 0);
  const top = v(0, 1.4, 0);
  const radius = 0.4;

  it('misses when the blade never comes near', () => {
    const hit = sweepSegmentAgainstCapsule(
      v(5, 1, 0),
      v(5, 1, 1),
      v(5, 1, 0),
      v(5, 1, 1),
      base,
      top,
      radius,
    );
    expect(hit).toBeNull();
  });

  it('hits when the blade is overlapping in its current pose', () => {
    const hit = sweepSegmentAgainstCapsule(
      v(0.2, 1, -1),
      v(0.2, 1, 1),
      v(0.2, 1, -1),
      v(0.2, 1, 1),
      base,
      top,
      radius,
    );
    expect(hit).not.toBeNull();
  });

  it('catches a blade that tunnels clean past the target in one frame', () => {
    // Previous pose two metres to the left, current pose two metres to the
    // right: a single-frame overlap test at either end reports nothing.
    const prevA = v(-2, 1, 0);
    const prevB = v(-2, 1, 0.6);
    const currA = v(2, 1, 0);
    const currB = v(2, 1, 0.6);

    expect(
      sweepSegmentAgainstCapsule(currA, currB, currA, currB, base, top, radius),
    ).toBeNull();
    const swept = sweepSegmentAgainstCapsule(prevA, prevB, currA, currB, base, top, radius);
    expect(swept).not.toBeNull();
    expect(swept?.time).toBeGreaterThan(0);
    expect(swept?.time).toBeLessThan(1);
  });

  it('reports the earliest contact in the interval, not the deepest', () => {
    const hit = sweepSegmentAgainstCapsule(
      v(-2, 1, 0),
      v(-2, 1, 0.6),
      v(0, 1, 0),
      v(0, 1, 0.6),
      base,
      top,
      radius,
      0,
      10,
    );
    expect(hit).not.toBeNull();
    expect(hit?.time).toBeLessThan(1);
  });

  it('puts the contact point on the capsule surface', () => {
    const hit = sweepSegmentAgainstCapsule(
      v(-2, 1, 0),
      v(-2, 1, 0.6),
      v(0.3, 1, 0),
      v(0.3, 1, 0.6),
      base,
      top,
      radius,
    );
    expect(hit).not.toBeNull();
    if (hit === null) return;
    const axial = Math.hypot(hit.point.x, hit.point.z);
    expect(axial).toBeCloseTo(radius, 3);
    expect(hit.normal.length()).toBeCloseTo(1, 6);
  });

  it('misses a target that is out of reach vertically', () => {
    const hit = sweepSegmentAgainstCapsule(
      v(-2, 4, 0),
      v(2, 4, 0),
      v(-2, 4, 0),
      v(2, 4, 0),
      base,
      top,
      radius,
    );
    expect(hit).toBeNull();
  });

  it('honours the blade radius as extra forgiveness', () => {
    const start = v(-2, 1, 0.55);
    const end = v(2, 1, 0.55);
    expect(sweepSegmentAgainstCapsule(start, end, start, end, base, top, 0.4, 0)).toBeNull();
    expect(
      sweepSegmentAgainstCapsule(start, end, start, end, base, top, 0.4, 0.25),
    ).not.toBeNull();
  });

  it('always produces a finite normal even through the capsule axis', () => {
    const hit = sweepSegmentAgainstCapsule(
      v(0, 1, -1),
      v(0, 1, 1),
      v(0, 1, -1),
      v(0, 1, 1),
      base,
      top,
      radius,
    );
    expect(hit).not.toBeNull();
    expect(Number.isFinite(hit?.normal.x ?? NaN)).toBe(true);
    expect(hit?.normal.length()).toBeCloseTo(1, 6);
  });
});

/* -------------------------------------------------------------------------- */
/* WeaponHitbox                                                                */
/* -------------------------------------------------------------------------- */

/** A blade the test can teleport wherever it likes. */
class ScriptedAnchor implements WeaponAnchor {
  readonly label = 'test.blade';
  readonly length = 1;
  hilt = new THREE.Vector3();
  tip = new THREE.Vector3(0, 0, 1);

  place(hiltAt: THREE.Vector3, tipAt: THREE.Vector3): void {
    this.hilt.copy(hiltAt);
    this.tip.copy(tipAt);
  }

  sample(hilt: THREE.Vector3, tip: THREE.Vector3): boolean {
    hilt.copy(this.hilt);
    tip.copy(this.tip);
    return true;
  }
}

function dummy(id: number, x: number, z: number): CapsuleTarget {
  return {
    id,
    hitRadius: 0.4,
    hitHeight: 1.8,
    footPosition: (out) => out.set(x, 0, z),
  };
}

describe('WeaponHitbox', () => {
  it('reports nothing while the damage window is closed', () => {
    const anchor = new ScriptedAnchor();
    const hitbox = new WeaponHitbox(anchor, { radius: 0.1 });
    const target = dummy(1, 0, 0);
    anchor.place(v(-1, 1, 0), v(1, 1, 0));
    hitbox.beginSwing();
    expect(hitbox.track([target])).toEqual([]);
  });

  it('lands a hit once the window opens', () => {
    const anchor = new ScriptedAnchor();
    const hitbox = new WeaponHitbox(anchor, { radius: 0.1 });
    const target = dummy(1, 0, 0);
    hitbox.beginSwing();
    anchor.place(v(-2, 1, 0), v(-1, 1, 0));
    hitbox.track([target]);
    hitbox.openWindow();
    anchor.place(v(-0.4, 1, 0), v(0.4, 1, 0));
    expect(hitbox.track([target]).length).toBe(1);
  });

  it('never hits the same target twice in one swing', () => {
    const anchor = new ScriptedAnchor();
    const hitbox = new WeaponHitbox(anchor, { radius: 0.1 });
    const target = dummy(1, 0, 0);
    hitbox.beginSwing();
    hitbox.openWindow();
    anchor.place(v(-0.4, 1, 0), v(0.4, 1, 0));
    expect(hitbox.track([target]).length).toBe(1);
    expect(hitbox.track([target]).length).toBe(0);
    expect(hitbox.track([target]).length).toBe(0);
    expect(hitbox.hitCount).toBe(1);
  });

  it('hits the same target again on the next swing', () => {
    const anchor = new ScriptedAnchor();
    const hitbox = new WeaponHitbox(anchor, { radius: 0.1 });
    const target = dummy(1, 0, 0);
    anchor.place(v(-0.4, 1, 0), v(0.4, 1, 0));

    hitbox.beginSwing();
    hitbox.openWindow();
    expect(hitbox.track([target]).length).toBe(1);

    hitbox.beginSwing();
    hitbox.openWindow();
    expect(hitbox.track([target]).length).toBe(1);
  });

  it('hits several targets in one sweep, once each', () => {
    const anchor = new ScriptedAnchor();
    const hitbox = new WeaponHitbox(anchor, { radius: 0.1 });
    const targets = [dummy(1, -0.5, 0), dummy(2, 0.5, 0), dummy(3, 8, 0)];
    hitbox.beginSwing();
    hitbox.openWindow();
    anchor.place(v(-1, 1, 0), v(1, 1, 0));
    const contacts = hitbox.track(targets);
    expect(contacts.map((contact) => contact.target.id).sort()).toEqual([1, 2]);
  });

  it('still sweeps when the window opens and closes inside one frame', () => {
    const anchor = new ScriptedAnchor();
    const hitbox = new WeaponHitbox(anchor, { radius: 0.1 });
    const target = dummy(1, 0, 0);
    hitbox.beginSwing();
    anchor.place(v(-2, 1, 0), v(-1, 1, 0));
    hitbox.track([target]);
    // A frame long enough that both authored markers fire before the sweep.
    hitbox.openWindow();
    hitbox.closeWindow();
    anchor.place(v(-0.4, 1, 0), v(0.4, 1, 0));
    expect(hitbox.track([target]).length).toBe(1);
  });

  it('does not carve a hitbox from the origin on its very first sample', () => {
    const anchor = new ScriptedAnchor();
    const hitbox = new WeaponHitbox(anchor, { radius: 0.1 });
    // A target sitting at the world origin, and a blade that starts far away.
    const target = dummy(1, 0, 0);
    anchor.place(v(20, 1, 20), v(21, 1, 20));
    hitbox.beginSwing();
    hitbox.openWindow();
    expect(hitbox.track([target])).toEqual([]);
  });

  it('reports the swing direction with the contact', () => {
    const anchor = new ScriptedAnchor();
    const hitbox = new WeaponHitbox(anchor, { radius: 0.1 });
    const target = dummy(1, 0, 0);
    hitbox.beginSwing();
    anchor.place(v(-2, 1, 0), v(-1.2, 1, 0));
    hitbox.track([target]);
    hitbox.openWindow();
    anchor.place(v(-0.4, 1, 0), v(0.4, 1, 0));
    const contact = hitbox.track([target])[0];
    expect(contact).toBeDefined();
    expect(contact?.travel.x).toBeGreaterThan(0.5);
    expect(contact?.travel.length()).toBeCloseTo(1, 6);
  });

  it('clears its per-swing memory on cancel', () => {
    const anchor = new ScriptedAnchor();
    const hitbox = new WeaponHitbox(anchor, { radius: 0.1 });
    const target = dummy(1, 0, 0);
    anchor.place(v(-0.4, 1, 0), v(0.4, 1, 0));
    hitbox.beginSwing();
    hitbox.openWindow();
    hitbox.track([target]);
    expect(hitbox.hitCount).toBe(1);
    hitbox.cancel();
    expect(hitbox.hitCount).toBe(0);
    expect(hitbox.isOpen).toBe(false);
  });

  it('extends the blade by the overreach', () => {
    const anchor = new ScriptedAnchor();
    const target = dummy(1, 0, 0);
    const tight = new WeaponHitbox(anchor, { radius: 0.01, overreach: 0 });
    const loose = new WeaponHitbox(anchor, { radius: 0.01, overreach: 0.5 });
    anchor.place(v(-2, 1, 0), v(-0.45, 1, 0));

    for (const box of [tight, loose]) {
      box.beginSwing();
      box.openWindow();
    }
    // Both are primed from the same pose, so neither sweeps any distance; only
    // the reach differs.
    tight.track([target]);
    loose.track([target]);
    expect(tight.hitCount).toBe(0);
    expect(loose.hitCount).toBe(1);
  });
});

/* -------------------------------------------------------------------------- */
/* Anchors                                                                     */
/* -------------------------------------------------------------------------- */

describe('weapon anchors', () => {
  function rig(): THREE.Object3D {
    const root = new THREE.Object3D();
    const lower = new THREE.Object3D();
    lower.name = 'lowerarmr';
    lower.position.set(0, 1, 0);
    const hand = new THREE.Object3D();
    hand.name = 'handr';
    hand.position.set(0, 0, 0.4);
    lower.add(hand);
    root.add(lower);
    root.updateMatrixWorld(true);
    return root;
  }

  it('falls back to the forearm when there is no weapon mesh', () => {
    const anchor = resolveWeaponAnchor(rig(), { reach: 1 });
    expect(anchor).not.toBeNull();
    const hilt = new THREE.Vector3();
    const tip = new THREE.Vector3();
    anchor?.sample(hilt, tip);
    // Hand at (0, 1, 0.4); forearm points along +z, so the tip is a metre on.
    expect(hilt.z).toBeCloseTo(0.4, 6);
    expect(tip.z).toBeCloseTo(1.4, 6);
  });

  it('returns null when the model is not the rig this game runs on', () => {
    expect(resolveWeaponAnchor(new THREE.Object3D())).toBeNull();
  });

  it('prefers a named weapon mesh and spans its longest axis', () => {
    const root = rig();
    const axe = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.1, 1.2));
    axe.name = '1H_Axe';
    root.getObjectByName('handr')?.add(axe);
    root.updateMatrixWorld(true);

    const anchor = resolveWeaponAnchor(root, { meshNames: ['1H_Axe'] });
    expect(anchor?.label).toBe('1H_Axe');
    expect(anchor?.length).toBeCloseTo(1.2, 6);
  });

  it('measures blade extents along the longest local axis', () => {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(0.2, 1.5, 0.2));
    const extents = localBladeExtents(mesh);
    expect(extents).not.toBeNull();
    expect(extents?.hilt.y).toBeCloseTo(-0.75, 6);
    expect(extents?.tip.y).toBeCloseTo(0.75, 6);
  });

  it('refuses a node with no geometry', () => {
    expect(localBladeExtents(new THREE.Object3D())).toBeNull();
  });
});
