import { describe, expect, it } from 'vitest';

import * as THREE from 'three/webgpu';

import { findBone, sanitizeBoneName } from '../src/character/BoneNames';
import { damp, smootherstep } from '../src/character/CameraRig';
import { isFootPlanted, shouldReleasePin, solveTwoBoneAngles } from '../src/character/FootIK';
import { approachAngle, wrapAngle } from '../src/character/PlayerController';

describe('wrapAngle', () => {
  it('leaves angles already in range alone', () => {
    expect(wrapAngle(0)).toBeCloseTo(0);
    expect(wrapAngle(1.2)).toBeCloseTo(1.2);
    expect(wrapAngle(-1.2)).toBeCloseTo(-1.2);
  });

  it('folds any angle into (-pi, pi]', () => {
    for (let turns = -6; turns <= 6; turns++) {
      const wrapped = wrapAngle(0.7 + turns * Math.PI * 2);
      expect(wrapped).toBeCloseTo(0.7);
      expect(wrapped).toBeGreaterThan(-Math.PI - 1e-9);
      expect(wrapped).toBeLessThanOrEqual(Math.PI + 1e-9);
    }
  });
});

describe('approachAngle', () => {
  it('arrives exactly when the step covers the gap', () => {
    expect(approachAngle(0, 0.5, 1)).toBeCloseTo(0.5);
  });

  it('moves by at most maxDelta', () => {
    expect(approachAngle(0, 3, 0.1)).toBeCloseTo(0.1);
    expect(approachAngle(0, -3, 0.1)).toBeCloseTo(-0.1);
  });

  it('takes the short way round the wrap point', () => {
    // From 170 degrees to -170 degrees is a 20 degree turn, not a 340 degree one.
    const from = (170 * Math.PI) / 180;
    const to = (-170 * Math.PI) / 180;
    const stepped = approachAngle(from, to, 0.1);
    expect(stepped).toBeGreaterThan(from);
    expect(wrapAngle(stepped - from)).toBeCloseTo(0.1);
  });

  it('converges rather than orbiting', () => {
    let angle = 0;
    for (let i = 0; i < 200; i++) angle = approachAngle(angle, 2.5, 0.05);
    expect(angle).toBeCloseTo(2.5);
  });
});

describe('camera smoothing', () => {
  it('is frame-rate independent', () => {
    const oneBigStep = damp(0, 10, 8, 0.1);
    let stepped = 0;
    for (let i = 0; i < 6; i++) stepped = damp(stepped, 10, 8, 0.1 / 6);
    expect(stepped).toBeCloseTo(oneBigStep, 6);
  });

  it('converges to the target and never overshoots', () => {
    let value = 0;
    for (let i = 0; i < 400; i++) {
      value = damp(value, 5, 12, 1 / 60);
      expect(value).toBeLessThanOrEqual(5 + 1e-9);
    }
    expect(value).toBeCloseTo(5, 4);
  });

  it('eases the mode transition with zero velocity at both ends', () => {
    expect(smootherstep(0)).toBe(0);
    expect(smootherstep(1)).toBe(1);
    expect(smootherstep(0.5)).toBeCloseTo(0.5);
    expect(smootherstep(-2)).toBe(0);
    expect(smootherstep(3)).toBe(1);
    // Nearly stationary at the ends: a linear blend ticks visibly on take-off.
    expect(smootherstep(0.02)).toBeLessThan(0.002);
    expect(1 - smootherstep(0.98)).toBeLessThan(0.002);
  });
});

describe('two-bone IK', () => {
  it('bends to a right angle for the classic 3-4-5 style case', () => {
    // Equal bones of length 1 reaching sqrt(2) form a right angle at the joint.
    const solved = solveTwoBoneAngles(1, 1, Math.SQRT2);
    expect(solved.joint).toBeCloseTo(Math.PI / 2);
    expect(solved.root).toBeCloseTo(Math.PI / 4);
    expect(solved.clamped).toBe(false);
  });

  it('straightens the chain when the target is out of reach', () => {
    const solved = solveTwoBoneAngles(0.5, 0.5, 4);
    expect(solved.joint).toBeCloseTo(Math.PI);
    expect(solved.root).toBe(0);
    expect(solved.clamped).toBe(true);
  });

  it('bends further as the target comes closer, monotonically', () => {
    let previous = Math.PI + 1;
    for (let reach = 1.9; reach > 0.3; reach -= 0.1) {
      const solved = solveTwoBoneAngles(1, 1, reach);
      expect(solved.joint).toBeLessThanOrEqual(previous + 1e-9);
      previous = solved.joint;
    }
  });

  it('never produces NaN, including at degenerate inputs', () => {
    for (const [upper, lower, reach] of [
      [1, 1, 0],
      [1, 1, 2],
      [0, 1, 1],
      [1, 0.2, 0.5],
      [1, 1, -1],
    ] as const) {
      const solved = solveTwoBoneAngles(upper, lower, reach);
      expect(Number.isFinite(solved.root)).toBe(true);
      expect(Number.isFinite(solved.joint)).toBe(true);
    }
  });

  it('satisfies the triangle it claims to solve', () => {
    const upper = 0.46;
    const lower = 0.44;
    const reach = 0.72;
    const { root, joint } = solveTwoBoneAngles(upper, lower, reach);
    // Reconstruct the far end from the two angles and check it lands on target.
    const kneeX = Math.cos(root) * upper;
    const kneeY = Math.sin(root) * upper;
    const direction = root + (Math.PI - joint);
    const footX = kneeX + Math.cos(direction) * lower;
    const footY = kneeY + Math.sin(direction) * lower;
    expect(Math.hypot(footX, footY)).toBeCloseTo(reach, 6);
  });
});

describe('bone name resolution', () => {
  it('reproduces three\'s node-name sanitisation', () => {
    // GLTFLoader deletes reserved characters rather than replacing them.
    expect(sanitizeBoneName('foot.l')).toBe('footl');
    expect(sanitizeBoneName('IK-foot.r')).toBe('IK-footr');
    expect(sanitizeBoneName('upper arm')).toBe('upper_arm');
    expect(sanitizeBoneName('chest')).toBe('chest');
  });

  it('finds a bone whether or not the loader stripped the dots', () => {
    const root = new THREE.Object3D();
    const dotted = new THREE.Object3D();
    dotted.name = 'foot.l';
    const stripped = new THREE.Object3D();
    stripped.name = 'upperlegr';
    root.add(dotted, stripped);

    expect(findBone(root, 'foot.l')).toBe(dotted);
    expect(findBone(root, 'upperleg.r')).toBe(stripped);
    expect(findBone(root, 'nothing.here')).toBeNull();
  });

  it('falls back to a normalised scan for underscore variants', () => {
    const root = new THREE.Object3D();
    const bone = new THREE.Object3D();
    bone.name = 'Upper_Leg_L';
    root.add(bone);
    expect(findBone(root, 'upperleg.l')).toBe(bone);
  });
});

/* -------------------------------------------------------------------------- */
/* Foot lock                                                                  */
/* -------------------------------------------------------------------------- */

/**
 * The stance test, against numbers taken from the shipped rig.
 *
 * Worth stating why this exists as a pure function at all: the foot lock is
 * only as good as its decision about *when* a foot is down, and that decision
 * is the one thing in the module that cannot be inspected from a screenshot. A
 * lock that plants a foot mid-swing drags the leg; one that never plants does
 * nothing at all; and both look like "the IK is broken" from the outside.
 *
 * The measured shape it is calibrated against, from
 * `tools/verify-footplant.mjs` on `Running_A`: body speed ~4 m/s, the stance
 * foot's animated world speed ~0.6x that and the swing foot's well over 1.5x,
 * with clearance oscillating between roughly zero and a peak of a few tens of
 * centimetres.
 */
const THRESHOLDS = {
  plantClearanceRatio: 0.22,
  releaseClearanceRatio: 0.6,
  plantClearanceFloor: 0.012,
  plantSpeedRatio: 0.85,
  plantSpeedFloor: 0.4,
};

/** A foot whose contact height sits 4 cm above zero — the rig's real bias. */
const foot = (clearance: number, animatedSpeed: number, bodySpeed = 4) => ({
  clearance,
  stanceFloor: 0.04,
  swingPeak: 0.34,
  animatedSpeed,
  bodySpeed,
});

describe('isFootPlanted', () => {
  it('plants a foot that is down and slow', () => {
    expect(isFootPlanted(foot(0.045, 2.4), THRESHOLDS)).toBe(true);
  });

  it('refuses a foot passing low over the ground at swing speed', () => {
    // The failure mode that matters most: at the bottom of a swing the foot is
    // briefly as low as a planted one, and pinning it there yanks the leg.
    expect(isFootPlanted(foot(0.045, 6.4), THRESHOLDS)).toBe(false);
  });

  it('refuses a foot held still in the air', () => {
    expect(isFootPlanted(foot(0.26, 0.1), THRESHOLDS)).toBe(false);
  });

  it('measures height from the foot own contact level, not from zero', () => {
    // The bug this replaced. `clearance` comes from a sole offset taken in the
    // bind pose, and this rig's is out by about 4 cm — the same order as the
    // whole walk cycle's lift. Judged against zero, a planted foot reads as
    // airborne and nothing ever plants.
    const biased = { clearance: 0.04, stanceFloor: 0.04, swingPeak: 0.1, animatedSpeed: 0.2, bodySpeed: 1 };
    expect(isFootPlanted(biased, THRESHOLDS)).toBe(true);
    expect(isFootPlanted({ ...biased, stanceFloor: 0 }, THRESHOLDS)).toBe(false);
  });

  it('holds a planted foot past the height that would not have started it', () => {
    // The hysteresis. Between the two ratios is the band where a foot stays
    // planted but would not newly plant; without it the decision chatters and
    // the lock spends its life blending.
    const marginal = { clearance: 0.16, stanceFloor: 0.04, swingPeak: 0.34, animatedSpeed: 0.2, bodySpeed: 1 };
    expect(isFootPlanted(marginal, THRESHOLDS, false)).toBe(false);
    expect(isFootPlanted(marginal, THRESHOLDS, true)).toBe(true);
  });

  it('ignores animated speed once the foot is already pinned', () => {
    // The pin is what the foot is following, so its animated speed stops being
    // evidence about the ground. Testing it in both directions releases every
    // pin the moment the clip accelerates underneath it.
    const fast = foot(0.045, 9);
    expect(isFootPlanted(fast, THRESHOLDS, false)).toBe(false);
    expect(isFootPlanted(fast, THRESHOLDS, true)).toBe(true);
  });

  it('plants both feet of a character standing still', () => {
    // Floor and peak have converged, so the amplitude floor decides. Without
    // it the idle pose gets no correction at all.
    for (const jitter of [0, 0.003]) {
      expect(
        isFootPlanted(
          { clearance: jitter, stanceFloor: 0, swingPeak: 0.004, animatedSpeed: 0.03, bodySpeed: 0 },
          THRESHOLDS,
        ),
      ).toBe(true);
    }
  });

  it('scales its clearance threshold with the gait rather than fixing it', () => {
    // 6 cm above contact is planted in a high-stepping run and airborne in a
    // shuffle. A single absolute threshold cannot express that.
    const low = { clearance: 0.06, stanceFloor: 0, swingPeak: 0.08, animatedSpeed: 0.2, bodySpeed: 1 };
    const high = { clearance: 0.06, stanceFloor: 0, swingPeak: 0.5, animatedSpeed: 0.2, bodySpeed: 1 };
    expect(isFootPlanted(low, THRESHOLDS)).toBe(false);
    expect(isFootPlanted(high, THRESHOLDS)).toBe(true);
  });

  it('treats a missing ground ray as not planted', () => {
    expect(
      isFootPlanted(
        {
          clearance: Number.POSITIVE_INFINITY,
          stanceFloor: 0,
          swingPeak: 0.3,
          animatedSpeed: 0,
          bodySpeed: 0,
        },
        THRESHOLDS,
        true,
      ),
    ).toBe(false);
  });
});

describe('shouldReleasePin', () => {
  const LEG = 0.9;

  it('holds a pin that is in contact, near the foot and in reach', () => {
    expect(shouldReleasePin(true, 0.08, 0.34, 0.7, LEG)).toBe(false);
  });

  it('releases as soon as the foot lifts', () => {
    expect(shouldReleasePin(false, 0.02, 0.34, 0.7, LEG)).toBe(true);
  });

  it('releases a pin the animation has walked away from', () => {
    // This is what makes turning safe: a character pivoting on the spot swings
    // the animated foot away from the pin, and the pin goes rather than the leg
    // fighting the rotation.
    expect(shouldReleasePin(true, 0.4, 0.34, 0.7, LEG)).toBe(true);
  });

  it('releases before the chain has to straighten', () => {
    expect(shouldReleasePin(true, 0.1, 0.34, LEG * 0.99, LEG)).toBe(true);
    expect(shouldReleasePin(true, 0.1, 0.34, LEG * 0.9, LEG)).toBe(false);
  });
});
