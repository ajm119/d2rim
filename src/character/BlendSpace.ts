/**
 * @module character/BlendSpace
 *
 * The locomotion blend space, as pure arithmetic.
 *
 * A 2D blend space is normally drawn as clips scattered on a velocity plane
 * with some triangulation between them. That generality is wasted here: the
 * Barbarian has exactly four directional clips (forward, back, strafe left,
 * strafe right) at each of two speed tiers, which is a separable problem. So
 * this is factored as
 *
 * ```
 *   weight(clip) = directionWeight(clip.direction) * tierWeight(clip.tier)
 * ```
 *
 * which is continuous everywhere, sums to one, needs no triangulation, and —
 * the reason it is in its own module — is testable without three.js, a mixer,
 * or a GPU.
 *
 * Local space convention throughout: `x` is the character's right, `z` is the
 * character's **forward**. Velocities arriving here have already been rotated
 * out of world space by the caller.
 */

/** The four authored directions in the ring. */
export type BlendDirection = 'forward' | 'back' | 'left' | 'right';

/** All directions, in a stable order. */
export const BLEND_DIRECTIONS: readonly BlendDirection[] = ['forward', 'back', 'left', 'right'];

/** The speed tiers the ring is authored at. */
export type BlendTier = 'idle' | 'walk' | 'run';

export type DirectionWeights = Readonly<Record<BlendDirection, number>>;
export type TierWeights = Readonly<Record<BlendTier, number>>;

/** Everything the animation graph needs to weight its locomotion actions. */
export interface BlendSample {
  /** Planar speed, m/s. */
  readonly speed: number;
  /** Direction weights, summing to 1 when moving and all zero when still. */
  readonly direction: DirectionWeights;
  /** Tier weights, always summing to 1. */
  readonly tier: TierWeights;
}

export interface BlendSpaceParams {
  /** Below this speed the character is idle. Default 0.12 m/s. */
  readonly idleThreshold: number;
  /** Ground speed the walk ring is blended fully in at. */
  readonly walkSpeed: number;
  /** Ground speed the run ring is blended fully in at. */
  readonly runSpeed: number;
}

export const DEFAULT_BLEND_PARAMS: BlendSpaceParams = {
  idleThreshold: 0.12,
  walkSpeed: 1.8,
  runSpeed: 4.4,
};

const ZERO_DIRECTION: DirectionWeights = { forward: 0, back: 0, left: 0, right: 0 };

/** Hermite ease used for every tier ramp. Exact at the endpoints, C¹ inside. */
export function smoothstep(edge0: number, edge1: number, x: number): number {
  if (edge1 <= edge0) return x < edge0 ? 0 : 1;
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

/**
 * Split a planar direction across the four authored directions.
 *
 * The weights are the L1-normalised components of the unit direction, which is
 * exactly bilinear interpolation around the diamond joining the four clips: a
 * heading 45 degrees off forward gets half forward and half strafe, and no
 * heading ever activates more than two clips. Blending three or four
 * simultaneously — which a naive angular falloff does — is what makes a
 * character's legs look boneless while turning.
 *
 * `x` is right-positive, `z` is forward-positive.
 */
export function directionWeights(x: number, z: number): DirectionWeights {
  const magnitude = Math.hypot(x, z);
  if (magnitude < 1e-9) return ZERO_DIRECTION;
  const nx = x / magnitude;
  const nz = z / magnitude;
  const norm = Math.abs(nx) + Math.abs(nz);
  return {
    forward: Math.max(0, nz) / norm,
    back: Math.max(0, -nz) / norm,
    right: Math.max(0, nx) / norm,
    left: Math.max(0, -nx) / norm,
  };
}

/**
 * Split a speed across idle / walk / run.
 *
 * Two ramps rather than one three-way blend, because idle must be fully gone
 * before the run ring starts to appear: a standing pose leaking into a sprint
 * at 20% weight shortens the stride and is the classic cause of a character
 * that looks like it is jogging on ice.
 *
 * Above `runSpeed` the run tier stays pinned at 1 — extra speed is spent on
 * playback rate, not on more weight, which is what keeps sprinting from
 * flattening into a pose.
 */
export function tierWeights(speed: number, params: BlendSpaceParams): TierWeights {
  const { idleThreshold, walkSpeed, runSpeed } = params;
  if (speed <= idleThreshold) return { idle: 1, walk: 0, run: 0 };
  if (speed <= walkSpeed) {
    const t = smoothstep(idleThreshold, walkSpeed, speed);
    return { idle: 1 - t, walk: t, run: 0 };
  }
  const t = smoothstep(walkSpeed, runSpeed, speed);
  return { idle: 0, walk: 1 - t, run: t };
}

/** Sample the blend space for a local-space planar velocity. */
export function sampleBlendSpace(
  localX: number,
  localZ: number,
  params: BlendSpaceParams = DEFAULT_BLEND_PARAMS,
): BlendSample {
  const speed = Math.hypot(localX, localZ);
  return {
    speed,
    direction: speed <= params.idleThreshold ? ZERO_DIRECTION : directionWeights(localX, localZ),
    tier: tierWeights(speed, params),
  };
}

/** One clip's contribution, after the two factors have been multiplied out. */
export interface BlendContribution {
  readonly direction: BlendDirection;
  readonly tier: Exclude<BlendTier, 'idle'>;
  readonly weight: number;
}

/**
 * Expand a sample into per-(direction, tier) weights, dropping anything below
 * `epsilon` so that near-zero actions can be stopped rather than left running
 * at a weight nobody can see but the CPU still pays for.
 */
export function blendContributions(sample: BlendSample, epsilon = 1e-4): BlendContribution[] {
  const out: BlendContribution[] = [];
  for (const tier of ['walk', 'run'] as const) {
    const tierWeight = sample.tier[tier];
    if (tierWeight <= epsilon) continue;
    for (const direction of BLEND_DIRECTIONS) {
      const weight = tierWeight * sample.direction[direction];
      if (weight > epsilon) out.push({ direction, tier, weight });
    }
  }
  return out;
}

/**
 * The distance the blended pose covers in one animation cycle, in metres.
 *
 * This is the number that eliminates foot sliding. Each authored clip travels
 * some real distance per cycle (measured from the feet — see
 * `AnimationGraph.measureStride`), and when several are blended the effective
 * distance is their weighted mean. Dividing the actual ground speed by it gives
 * the cycles per second the mixer must run at for the planted foot to stay
 * planted.
 *
 * Returns `fallback` when nothing is weighted, so callers never divide by zero.
 */
export function blendedStride(
  contributions: readonly BlendContribution[],
  strideOf: (contribution: BlendContribution) => number,
  fallback: number,
): number {
  let weighted = 0;
  let total = 0;
  for (const contribution of contributions) {
    const stride = strideOf(contribution);
    if (!Number.isFinite(stride) || stride <= 1e-4) continue;
    weighted += stride * contribution.weight;
    total += contribution.weight;
  }
  return total <= 1e-6 ? fallback : weighted / total;
}

/**
 * Cycles per second required for a blended pose to keep up with the ground.
 *
 * Clamped, because the relationship breaks down at the extremes: at a crawl the
 * ratio tends to zero and the animation freezes mid-step (worse than a small
 * slide), and a sprint pushed past the clip's authored cadence turns a run into
 * a cartoon. `min`/`max` are the honest admission that stride warping has a
 * usable range and the blend tiers have to cover the rest.
 */
export function cycleRate(
  speed: number,
  stride: number,
  min = 0.4,
  max = 1.75,
): number {
  if (stride <= 1e-4) return min;
  return Math.min(max, Math.max(min, speed / stride));
}
