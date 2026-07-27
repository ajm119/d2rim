import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';

import {
  canAutostep,
  classifySlope,
  downhillDirection,
  projectOntoGroundPlane,
  shouldJump,
} from '../src/physics/CharacterController';

const degrees = (value: number): number => THREE.MathUtils.degToRad(value);

/** Normal of a slope that rises `angle` degrees toward −z. */
function slopeNormal(angle: number): THREE.Vector3 {
  return new THREE.Vector3(0, Math.cos(degrees(angle)), Math.sin(degrees(angle))).normalize();
}

describe('classifySlope', () => {
  const climb = degrees(50);
  const slide = degrees(55);

  it('reads flat ground as walkable and not sliding', () => {
    const flat = classifySlope(1, climb, slide);
    expect(flat.angle).toBeCloseTo(0);
    expect(flat.walkable).toBe(true);
    expect(flat.sliding).toBe(false);
  });

  it('recovers the slope angle from the normal', () => {
    for (const angle of [5, 17, 33, 49, 61]) {
      expect(classifySlope(slopeNormal(angle).y, climb, slide).angle).toBeCloseTo(degrees(angle));
    }
  });

  it('is walkable exactly up to the climb limit', () => {
    expect(classifySlope(slopeNormal(49.5).y, climb, slide).walkable).toBe(true);
    expect(classifySlope(slopeNormal(50.5).y, climb, slide).walkable).toBe(false);
  });

  it('slides only past the slide limit', () => {
    expect(classifySlope(slopeNormal(54).y, climb, slide).sliding).toBe(false);
    expect(classifySlope(slopeNormal(56).y, climb, slide).sliding).toBe(true);
  });

  it('survives a denormalised or overshooting normal', () => {
    expect(classifySlope(1.0000001, climb, slide).angle).toBeCloseTo(0);
    expect(Number.isNaN(classifySlope(-3, climb, slide).angle)).toBe(false);
  });
});

describe('projectOntoGroundPlane', () => {
  it('is the identity on flat ground', () => {
    const out = projectOntoGroundPlane(
      new THREE.Vector3(3, 0, -4),
      new THREE.Vector3(0, 1, 0),
    );
    expect(out.x).toBeCloseTo(3);
    expect(out.y).toBeCloseTo(0);
    expect(out.z).toBeCloseTo(-4);
  });

  it('preserves speed on a slope instead of shrinking it', () => {
    for (const angle of [10, 25, 40]) {
      const out = projectOntoGroundPlane(new THREE.Vector3(0, 0, -4), slopeNormal(angle));
      // The whole point: the character walks up a bank at the speed it asked
      // for, so the animation's stride matching stays honest.
      expect(out.length()).toBeCloseTo(4);
    }
  });

  it('lies in the slope plane, so downhill motion follows the ground', () => {
    const normal = slopeNormal(30);
    const out = projectOntoGroundPlane(new THREE.Vector3(0, 0, 4), normal);
    expect(out.dot(normal)).toBeCloseTo(0);
    // Moving toward +z on a slope that rises toward −z means going downhill.
    expect(out.y).toBeLessThan(0);
  });

  it('gains height going uphill and loses it going downhill, symmetrically', () => {
    const normal = slopeNormal(20);
    const up = projectOntoGroundPlane(new THREE.Vector3(0, 0, -3), normal);
    const down = projectOntoGroundPlane(new THREE.Vector3(0, 0, 3), normal);
    expect(up.y).toBeCloseTo(-down.y);
    expect(up.y).toBeGreaterThan(0);
  });

  it('returns zero for zero input', () => {
    expect(projectOntoGroundPlane(new THREE.Vector3(), slopeNormal(30)).lengthSq()).toBe(0);
  });
});

describe('downhillDirection', () => {
  it('points away from the uphill side', () => {
    const out = downhillDirection(slopeNormal(35));
    expect(out.length()).toBeCloseTo(1);
    expect(out.z).toBeGreaterThan(0);
    expect(out.y).toBe(0);
  });

  it('is zero on flat ground', () => {
    expect(downhillDirection(new THREE.Vector3(0, 1, 0)).lengthSq()).toBe(0);
  });
});

describe('shouldJump', () => {
  const coyote = 0.14;

  it('fires while grounded with a pending request', () => {
    expect(shouldJump({ timeSinceGrounded: 0, jumpSpent: false, bufferRemaining: 0.1 }, coyote))
      .toBe(true);
  });

  it('does nothing without a request', () => {
    expect(shouldJump({ timeSinceGrounded: 0, jumpSpent: false, bufferRemaining: 0 }, coyote))
      .toBe(false);
  });

  it('still fires just after walking off a ledge', () => {
    expect(shouldJump({ timeSinceGrounded: 0.1, jumpSpent: false, bufferRemaining: 0.1 }, coyote))
      .toBe(true);
  });

  it('stops forgiving once the coyote window closes', () => {
    expect(shouldJump({ timeSinceGrounded: 0.2, jumpSpent: false, bufferRemaining: 0.1 }, coyote))
      .toBe(false);
  });

  it('refuses a second jump in the same airtime', () => {
    expect(shouldJump({ timeSinceGrounded: 0.05, jumpSpent: true, bufferRemaining: 0.1 }, coyote))
      .toBe(false);
  });
});

describe('canAutostep', () => {
  const stepOffset = 0.42;
  const minWidth = 0.18;

  it('steps over rubble inside the offset', () => {
    expect(canAutostep(0.28, 0.4, stepOffset, minWidth)).toBe(true);
  });

  it('blocks on anything taller than the offset', () => {
    expect(canAutostep(0.6, 0.4, stepOffset, minWidth)).toBe(false);
  });

  it('refuses to mantle a rail with nothing to stand on beyond it', () => {
    expect(canAutostep(0.3, 0.05, stepOffset, minWidth)).toBe(false);
  });

  it('treats a flat floor as nothing to step over', () => {
    expect(canAutostep(0, 1, stepOffset, minWidth)).toBe(false);
  });

  it('accepts an obstacle exactly at the limits', () => {
    expect(canAutostep(stepOffset, minWidth, stepOffset, minWidth)).toBe(true);
  });
});
