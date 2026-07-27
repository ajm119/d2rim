/**
 * @module character/AnimationGraph
 *
 * The animation state machine over `THREE.AnimationMixer`. Everything about
 * how the Barbarian *feels* to control that is not physics lives here.
 *
 * ### Root motion: not used, and why
 *
 * The authored clips were inspected before this was written. Every locomotion
 * clip — `Walking_A`, `Walking_Backwards`, `Running_A`, the strafes — animates
 * the `root` node's translation with **zero** horizontal displacement across
 * the whole cycle; only a vertical bob is present. They are in-place clips. The
 * only clips with real root translation are the dodges (`Dodge_Forward` moves
 * the root 0.25 model units over 0.4 s).
 *
 * So the decision is: **in-place clips, driven by controller velocity**, and
 * the horizontal component of the root track is flattened at load time. That is
 * not a preference, it is what the asset supports — extracting root motion from
 * a clip that has none yields a character that never moves.
 *
 * The consequence is that foot sliding becomes this module's problem rather
 * than the animator's, and it is solved by
 * {@link AnimationGraph.measureStrideDistance}: each locomotion clip is played
 * once at load and the *backward travel of its feet while they are on the
 * ground* is measured, giving the real metres-per-cycle the pose covers. The
 * mixer is then advanced at `groundSpeed / blendedStride` cycles per second, so
 * the planted foot moves backward through the world at exactly the speed the
 * world moves past the character. Change the walk speed in `PlayerController`
 * and the feet still do not slide, because nothing here is a hand-tuned
 * constant.
 *
 * "While they are on the ground" is load-bearing and was the phase-3 bug: the
 * first version measured each foot's total peak-to-peak excursion instead, and
 * a run has a flight phase during which the body covers ground that no foot can
 * account for. It under-reported this rig's run cycle by 54% and the character
 * span its legs at twice the rate it needed. See
 * {@link plantedTravelPerCycle}.
 *
 * The dodges' root translation is measured and exposed on the action handle so
 * a future dodge can drive the capsule from the clip, but it is *not* applied
 * to the transform — the controller owns position, always, and two things
 * moving one character is how you get a player who desyncs from their collider.
 *
 * ### Layering
 *
 * `THREE.AnimationMixer` blends per *property binding*, not per clip. That is
 * the whole mechanism behind the upper-body layer: an action whose clip has had
 * every lower-body track stripped out simply has nothing to say about the legs,
 * so the locomotion blend keeps driving them at full weight while the arms are
 * fully overridden. No second mixer, no manual bone-by-bone blending.
 */

import * as THREE from 'three/webgpu';

import { findBone } from './BoneNames';
import {
  DEFAULT_BLEND_PARAMS,
  blendContributions,
  blendedStride,
  cycleRate,
  sampleBlendSpace,
  type BlendDirection,
  type BlendSample,
  type BlendSpaceParams,
} from './BlendSpace';

/* -------------------------------------------------------------------------- */
/* Semantic clip map                                                          */
/* -------------------------------------------------------------------------- */

/**
 * Semantic state -> candidate clip names, best first.
 *
 * Candidates, not a single name, because four character rigs share this graph
 * and their clip sets are similar but not identical. The list is resolved
 * against the clips actually present in the GLB at construction time and every
 * unresolved state is logged, so a missing clip degrades one state instead of
 * throwing.
 */
export const CLIP_CANDIDATES = {
  idle: ['Idle', 'Unarmed_Idle', '2H_Melee_Idle'],
  'walk.forward': ['Walking_A', 'Walking_B', 'Walking_C'],
  'walk.back': ['Walking_Backwards', 'Walking_B'],
  'walk.left': ['Walking_Strafe_Left', 'Running_Strafe_Left'],
  'walk.right': ['Walking_Strafe_Right', 'Running_Strafe_Right'],
  'run.forward': ['Running_A', 'Running_B', 'Walking_A'],
  'run.back': ['Running_Backwards', 'Walking_Backwards'],
  'run.left': ['Running_Strafe_Left', 'Walking_Strafe_Left'],
  'run.right': ['Running_Strafe_Right', 'Walking_Strafe_Right'],
  air: ['Jump_Idle', 'Jump_Full_Long', 'Idle'],
  'jump.start': ['Jump_Start', 'Jump_Full_Short'],
  'jump.land': ['Jump_Land'],
  attack: ['1H_Melee_Attack_Chop', 'Unarmed_Melee_Attack_Punch_A'],
  'attack.slice': ['1H_Melee_Attack_Slice_Diagonal', '1H_Melee_Attack_Slice_Horizontal'],
  // Distinct from `attack.slice` because the two slices sweep through very
  // different volumes: the diagonal comes down across the body, the horizontal
  // scythes level at chest height. An enemy that only owns "slice" cannot pick
  // the one whose blade actually passes through the target.
  'attack.sweep': ['1H_Melee_Attack_Slice_Horizontal', '1H_Melee_Attack_Slice_Diagonal'],
  'attack.stab': ['1H_Melee_Attack_Stab', '2H_Melee_Attack_Stab'],
  'attack.heavy': ['2H_Melee_Attack_Chop', '2H_Melee_Attack_Slice'],
  'attack.spin': ['2H_Melee_Attack_Spin', '2H_Melee_Attack_Spinning'],
  block: ['Blocking', 'Block'],
  'block.raise': ['Block'],
  'block.hit': ['Block_Hit'],
  'block.attack': ['Block_Attack'],
  hit: ['Hit_A', 'Hit_B'],
  'hit.b': ['Hit_B', 'Hit_A'],
  'dodge.forward': ['Dodge_Forward'],
  'dodge.back': ['Dodge_Backward'],
  'dodge.left': ['Dodge_Left'],
  'dodge.right': ['Dodge_Right'],
  death: ['Death_A', 'Death_B'],
  'death.b': ['Death_B', 'Death_A'],
  interact: ['Interact', 'Use_Item'],
  pickup: ['PickUp', 'Interact'],
  cheer: ['Cheer'],
} as const satisfies Record<string, readonly string[]>;

/** Every semantic state this graph knows how to resolve. */
export type ClipState = keyof typeof CLIP_CANDIDATES;

/** States driven by the locomotion blend rather than by `playAction`. */
const LOCOMOTION_STATES = new Set<string>([
  'idle',
  'walk.forward',
  'walk.back',
  'walk.left',
  'walk.right',
  'run.forward',
  'run.back',
  'run.left',
  'run.right',
  'air',
]);

/**
 * Which layer an action plays on when the caller does not say.
 *
 * Light attacks, blocks and flinches are upper-body so the player keeps
 * control of their feet while using them — a Skyrim-style game where a swing
 * roots you in place feels broken. Heavy swings, dodges and deaths are
 * full-body precisely *because* they take control away; that is the cost that
 * makes them read as committed.
 */
const DEFAULT_LAYER: Readonly<Record<string, ActionLayer>> = {
  attack: 'upper',
  'attack.slice': 'upper',
  'attack.stab': 'upper',
  block: 'upper',
  'block.raise': 'upper',
  'block.hit': 'upper',
  'block.attack': 'upper',
  hit: 'upper',
  'hit.b': 'upper',
  interact: 'upper',
  pickup: 'upper',
};

/**
 * Default hit-frame events, in normalised clip time.
 *
 * Combat reads these so a swing connects when the axe passes through the
 * target rather than when the button was pressed. Overridable per call.
 */
const DEFAULT_EVENTS: Readonly<Record<string, readonly ActionEventSpec[]>> = {
  attack: [{ name: 'hit', at: 0.42 }],
  'attack.slice': [{ name: 'hit', at: 0.4 }],
  'attack.stab': [{ name: 'hit', at: 0.45 }],
  'attack.heavy': [{ name: 'hit', at: 0.5 }],
  'attack.spin': [{ name: 'hit', at: 0.38 }, { name: 'hit', at: 0.66 }],
  'block.attack': [{ name: 'hit', at: 0.4 }],
};

/* -------------------------------------------------------------------------- */
/* Public types                                                               */
/* -------------------------------------------------------------------------- */

export type ActionLayer = 'full' | 'upper';

/** How an action ended. */
export type ActionResult = 'completed' | 'cancelled' | 'interrupted';

/** A callback point inside a clip, in normalised time. */
export interface ActionEventSpec {
  readonly name: string;
  /** Normalised clip time in `[0, 1]`. */
  readonly at: number;
}

/** What {@link AnimationGraph.onAnimationEvent} receives. */
export interface AnimationEvent {
  /** Semantic action name, or `'locomotion'` for footsteps. */
  readonly action: string;
  /** Event name: a custom marker, or `start` / `end` / `footstep`. */
  readonly name: string;
  /** Normalised clip time the event fired at. */
  readonly normalizedTime: number;
}

export interface PlayActionOptions {
  readonly layer?: ActionLayer;
  /** Blend-in seconds. Default 0.12 (0.08 for upper-body). */
  readonly fadeIn?: number;
  /** Blend-out seconds. Default 0.18. */
  readonly fadeOut?: number;
  /** Playback multiplier. Default 1. */
  readonly speed?: number;
  /** Loop until cancelled. Default false. */
  readonly loop?: boolean;
  /** Hold the final pose instead of blending out. Used for death. */
  readonly hold?: boolean;
  /** Replace an action already running on this layer. Default true. */
  readonly interrupt?: boolean;
  /** Event markers. Replaces the defaults for this action entirely. */
  readonly events?: readonly ActionEventSpec[];
  /** Force a specific clip name, bypassing the semantic map. */
  readonly clip?: string;
}

/** The handle {@link AnimationGraph.playAction} returns. */
export interface ActionHandle {
  readonly action: string;
  readonly clip: string;
  readonly layer: ActionLayer;
  /** Clip length in seconds at speed 1. */
  readonly duration: number;
  /**
   * Horizontal distance the clip's own root track travels, in metres.
   * Zero for every locomotion clip. Exposed so a dodge can be driven from the
   * authored motion later; nothing applies it today.
   */
  readonly rootMotion: number;
  /** Resolves when the action finishes, is cancelled, or is interrupted. */
  readonly finished: Promise<ActionResult>;
  /** Seconds elapsed. */
  readonly elapsed: number;
  /** Progress in `[0, 1]`. */
  readonly normalizedTime: number;
  readonly done: boolean;
  cancel(fadeOut?: number): void;
}

export interface AnimationGraphOptions {
  /** Blend-space thresholds. Defaults to {@link DEFAULT_BLEND_PARAMS}. */
  readonly blend?: BlendSpaceParams;
  /**
   * Bone that roots the upper-body mask. Everything at or below it in the
   * skeleton belongs to the upper layer. Default `chest`, falling back to
   * `spine` — masking from the chest leaves the lower spine to the locomotion
   * blend, so a running attack still has a torso that counter-rotates with the
   * stride instead of a rigid trunk bolted onto moving legs.
   */
  readonly upperBodyRoot?: string;
  /** Foot bones used for stride measurement. Default `foot.l` / `foot.r`. */
  readonly footBones?: readonly [string, string];
  /** Log the resolved clip table at construction. Default true. */
  readonly verbose?: boolean;
  /**
   * Hard bounds on locomotion cadence, in cycles per second.
   *
   * Stride matching is exact inside this band and degrades gracefully outside
   * it. The upper bound is the honest admission that an asset's clips have a
   * stride the game's speeds may not agree with: push it too high and a run
   * becomes a cartoon, cap it too low and the feet slide.
   *
   * Default `[0.5, 2.9]`. On the shipped Barbarian rig that covers the walk
   * (1.25 m/s over a 0.56 m cycle → 2.25 c/s) and the run (4.4 m/s over a
   * 1.97 m cycle → 2.23 c/s) exactly, and clips only at the top of the sprint,
   * where 6.0 m/s wants 3.05 c/s and gets 2.9 — a 5% slide bought in exchange
   * for the sprint not turning into a cartoon. 2.9 c/s is 2.3× the run clip's
   * authored rate, which is about as far as time-scaling a gait can be pushed
   * before the ankles start to read as vibration rather than as steps.
   */
  readonly cadenceRange?: readonly [number, number];
}

/* -------------------------------------------------------------------------- */
/* Clip surgery                                                               */
/* -------------------------------------------------------------------------- */

/** Split `chest.quaternion` into `['chest', 'quaternion']` given known nodes. */
export function splitTrackName(
  trackName: string,
  knownNodes: readonly string[],
): { node: string; property: string } | null {
  // Longest first: `hand.l` must win over `hand` for `hand.l.quaternion`.
  for (const node of knownNodes) {
    if (!trackName.startsWith(`${node}.`)) continue;
    const property = trackName.slice(node.length + 1);
    if (!property.includes('.')) return { node, property };
  }
  return null;
}

/**
 * Copy a clip keeping only the tracks whose node passes `predicate`.
 *
 * Tracks whose node cannot be identified are dropped from a masked clip: an
 * unmatched track is almost always a weapon-visibility or morph track, and
 * letting one leak into the upper-body layer would make an attack hide the axe
 * that the locomotion layer is still holding.
 */
export function filterClipTracks(
  clip: THREE.AnimationClip,
  knownNodes: readonly string[],
  predicate: (node: string) => boolean,
): THREE.AnimationClip {
  const tracks = clip.tracks.filter((track) => {
    const split = splitTrackName(track.name, knownNodes);
    return split !== null && predicate(split.node);
  });
  return new THREE.AnimationClip(`${clip.name}__upper`, clip.duration, tracks, clip.blendMode);
}

/**
 * Flatten the horizontal component of a node's position track.
 *
 * Called on every clip at load. Locomotion clips are unaffected (their root
 * track is already flat); it exists for the dodges, whose authored root travel
 * would otherwise slide the mesh out of its own capsule.
 */
export function stripRootMotion(clip: THREE.AnimationClip, rootNode: string): THREE.AnimationClip {
  const trackName = `${rootNode}.position`;
  const tracks = clip.tracks.map((track) => {
    if (track.name !== trackName) return track;
    const values = Float32Array.from(track.values);
    const x = values[0] ?? 0;
    const z = values[2] ?? 0;
    for (let i = 0; i + 2 < values.length; i += 3) {
      values[i] = x;
      values[i + 2] = z;
    }
    return new THREE.VectorKeyframeTrack(track.name, Array.from(track.times), Array.from(values));
  });
  return new THREE.AnimationClip(clip.name, clip.duration, tracks, clip.blendMode);
}

/** A sampled foot position. Only the planar components and the height matter. */
export interface FootSample {
  readonly x: number;
  readonly y: number;
  readonly z: number;
}

/**
 * Metres of ground one animation cycle covers, from a single foot's trace.
 *
 * ### What this number has to be, and what it is not
 *
 * The playback rate that keeps a foot planted is `groundSpeed / S`, where `S`
 * is the distance the *body* must travel in one cycle. `S` is fixed by one
 * constraint and one only: while a foot is on the ground it must be
 * stationary in the world, so the body has to advance at exactly the rate the
 * foot is sliding backwards through body space. So
 *
 * ```
 *   S = (backward travel of the foot, per unit of cycle phase, while planted)
 * ```
 *
 * The previous implementation used each foot's **peak-to-peak excursion**,
 * summed over both feet. That is only equal to `S` when each foot is on the
 * ground for exactly half the cycle. Measured on the shipped Barbarian rig:
 *
 * | clip        | Σ peak-to-peak | stance travel per cycle | error  |
 * |-------------|----------------|-------------------------|--------|
 * | `Walking_A` | 0.62 m         | 0.56 m                  | +12%   |
 * | `Running_A` | 0.93 m         | 1.97 m                  | **−53%** |
 *
 * The run is the killer. A run has a flight phase — for most of the cycle
 * *neither* foot is down, and the body covers that ground for free. Peak-to-peak
 * cannot see any of it, so it reported a 0.93 m stride for a cycle that really
 * carries the body 1.97 m, the graph then span the run clip at more than twice
 * the rate it needed, and the planted foot shot backwards. That is the bulk of
 * "severe foot sliding": a 2.1× error in one number.
 *
 * ### The estimator
 *
 * 1. Call a sample *planted* when the foot is in the bottom `plantBand` of its
 *    own vertical range. Relative, so it works on a clip with a 3 cm shuffle
 *    and on one with a 30 cm knee lift.
 * 2. Take the dominant travel direction across the planted samples. Whichever
 *    way the clip walks — forward, backwards, sideways — the planted foot's net
 *    motion points the opposite way, so this needs no axis convention.
 * 3. Report the **median** per-phase speed along that direction, over planted
 *    samples that are actually moving along it.
 *
 * The median is what makes step 1's threshold harmless. Samples caught either
 * side of the true plant are still descending or already lifting, and they are
 * both a minority and outliers; a mean would be dragged by them (which is the
 * failure the old comment on this function correctly predicted and then dodged
 * by measuring the wrong thing entirely).
 *
 * @param trace one cycle of foot positions in body space, metres. The last
 *   sample may be a duplicate of the first; a wrapped cycle is expected.
 * @param plantBand fraction of the foot's vertical range counted as planted.
 * @returns metres per cycle, or 0 when the trace has no usable stance.
 */
export function plantedTravelPerCycle(
  trace: readonly FootSample[],
  plantBand = 0.3,
): number {
  if (trace.length < 4) return 0;
  // The trace wraps: the final sample is the same pose as the first, so there
  // are `length - 1` real intervals and each spans `1 / (length - 1)` of phase.
  const intervals = trace.length - 1;

  let minY = Infinity;
  let maxY = -Infinity;
  for (const sample of trace) {
    if (sample.y < minY) minY = sample.y;
    if (sample.y > maxY) maxY = sample.y;
  }
  if (!Number.isFinite(minY) || !Number.isFinite(maxY)) return 0;
  const ceiling = minY + Math.max(1e-6, (maxY - minY) * plantBand);

  const deltas: Array<{ x: number; z: number }> = [];
  let sumX = 0;
  let sumZ = 0;
  for (let i = 0; i < intervals; i++) {
    const a = trace[i];
    const b = trace[i + 1];
    if (a === undefined || b === undefined || a.y > ceiling) continue;
    const delta = { x: b.x - a.x, z: b.z - a.z };
    deltas.push(delta);
    sumX += delta.x;
    sumZ += delta.z;
  }
  if (deltas.length < 2) return 0;

  const length = Math.hypot(sumX, sumZ);
  if (length < 1e-9) return 0;
  const ux = sumX / length;
  const uz = sumZ / length;

  const rates: number[] = [];
  for (const delta of deltas) {
    const along = (delta.x * ux + delta.z * uz) * intervals;
    if (along > 1e-6) rates.push(along);
  }
  if (rates.length === 0) return 0;
  rates.sort((a, b) => a - b);
  const middle = rates.length >> 1;
  const median =
    rates.length % 2 === 1
      ? (rates[middle] ?? 0)
      : ((rates[middle - 1] ?? 0) + (rates[middle] ?? 0)) / 2;
  return median > 0 ? median : 0;
}

/** Total horizontal travel of a node's position track, in track units. */
export function measureRootMotion(clip: THREE.AnimationClip, rootNode: string): number {
  const track = clip.tracks.find((candidate) => candidate.name === `${rootNode}.position`);
  if (track === undefined || track.values.length < 6) return 0;
  const values = track.values;
  const x0 = values[0] ?? 0;
  const z0 = values[2] ?? 0;
  const xn = values[values.length - 3] ?? 0;
  const zn = values[values.length - 1] ?? 0;
  return Math.hypot(xn - x0, zn - z0);
}

/* -------------------------------------------------------------------------- */
/* Internals                                                                  */
/* -------------------------------------------------------------------------- */

interface ClipEntry {
  readonly name: string;
  readonly action: THREE.AnimationAction;
  readonly duration: number;
  /** Metres of ground covered per animation cycle. */
  stride: number;
  weight: number;
}

type ActionPhase = 'in' | 'hold' | 'out' | 'done';

class ActiveAction implements ActionHandle {
  readonly action: string;
  readonly clip: string;
  readonly layer: ActionLayer;
  readonly duration: number;
  readonly rootMotion: number;
  readonly finished: Promise<ActionResult>;

  readonly mixerAction: THREE.AnimationAction | null;
  readonly fadeIn: number;
  readonly fadeOut: number;
  readonly loop: boolean;
  readonly hold: boolean;
  readonly speed: number;
  readonly events: readonly ActionEventSpec[];

  phase: ActionPhase = 'in';
  time = 0;
  weight = 0;
  outTimer = 0;
  firedEvents = new Set<number>();

  #resolve: (result: ActionResult) => void = () => undefined;

  constructor(init: {
    action: string;
    clip: string;
    layer: ActionLayer;
    duration: number;
    rootMotion: number;
    mixerAction: THREE.AnimationAction | null;
    fadeIn: number;
    fadeOut: number;
    loop: boolean;
    hold: boolean;
    speed: number;
    events: readonly ActionEventSpec[];
  }) {
    this.action = init.action;
    this.clip = init.clip;
    this.layer = init.layer;
    this.duration = init.duration;
    this.rootMotion = init.rootMotion;
    this.mixerAction = init.mixerAction;
    this.fadeIn = init.fadeIn;
    this.fadeOut = init.fadeOut;
    this.loop = init.loop;
    this.hold = init.hold;
    this.speed = init.speed;
    this.events = init.events;
    this.finished = new Promise<ActionResult>((resolve) => {
      this.#resolve = resolve;
    });
  }

  get elapsed(): number {
    return this.time;
  }

  get normalizedTime(): number {
    return this.duration <= 0 ? 1 : Math.min(1, this.time / this.duration);
  }

  get done(): boolean {
    return this.phase === 'done';
  }

  cancel(fadeOut?: number): void {
    if (this.phase === 'done' || this.phase === 'out') return;
    this.phase = 'out';
    this.outTimer = fadeOut ?? this.fadeOut;
  }

  settle(result: ActionResult): void {
    this.phase = 'done';
    this.#resolve(result);
  }
}

/* -------------------------------------------------------------------------- */
/* AnimationGraph                                                             */
/* -------------------------------------------------------------------------- */

export class AnimationGraph {
  readonly #root: THREE.Object3D;
  readonly #mixer: THREE.AnimationMixer;
  readonly #clips = new Map<string, THREE.AnimationClip>();
  readonly #upperClips = new Map<string, THREE.AnimationClip>();
  readonly #resolved = new Map<string, string>();
  readonly #entries = new Map<string, ClipEntry>();
  readonly #nodeNames: string[];
  readonly #upperBones: Set<string>;
  #blend: BlendSpaceParams;
  readonly #cadence: readonly [number, number];
  readonly #listeners = new Set<(event: AnimationEvent) => void>();

  #active: ActiveAction[] = [];
  #idle: ClipEntry | null = null;
  #air: ClipEntry | null = null;

  #phase = 0;
  #lastPhase = 0;
  #airBlend = 0;
  #locomotionWeight = 1;
  #sample: BlendSample = sampleBlendSpace(0, 0, DEFAULT_BLEND_PARAMS);
  #grounded = true;
  #speed = 0;
  #cycleRate = 0;
  #disposed = false;

  constructor(
    root: THREE.Object3D,
    clips: readonly THREE.AnimationClip[],
    options: AnimationGraphOptions = {},
  ) {
    this.#root = root;
    this.#mixer = new THREE.AnimationMixer(root);
    this.#blend = options.blend ?? DEFAULT_BLEND_PARAMS;
    this.#cadence = options.cadenceRange ?? [0.5, 2.9];

    this.#nodeNames = collectNodeNames(root);
    // Longest first so `splitTrackName` matches `hand.l` before `hand`.
    this.#nodeNames.sort((a, b) => b.length - a.length);

    const rootNode = this.#nodeNames.includes('root') ? 'root' : 'hips';
    for (const clip of clips) {
      if (clip.name === 'T-Pose') continue; // never a pose we want to blend to
      this.#clips.set(clip.name, stripRootMotion(clip, rootNode));
    }

    this.#upperBones = collectUpperBones(root, options.upperBodyRoot ?? 'chest');
    const verbose = options.verbose !== false;
    this.#resolveStates(verbose);
    this.#buildLocomotion(options.footBones ?? ['foot.l', 'foot.r']);

    if (verbose) this.#logTable(clips, rootNode);
  }

  /* -- introspection ------------------------------------------------------ */

  /** Every clip name found in the model, in GLB order. */
  get clipNames(): string[] {
    return Array.from(this.#clips.keys());
  }

  /** Semantic state -> resolved clip name. Unresolved states are absent. */
  get clipTable(): ReadonlyMap<string, string> {
    return this.#resolved;
  }

  /** The dominant locomotion state right now. Diagnostics and tests. */
  get state(): string {
    if (!this.#grounded) return 'air';
    const contributions = blendContributions(this.#sample);
    let best: { key: string; weight: number } = { key: 'idle', weight: this.#sample.tier.idle };
    for (const contribution of contributions) {
      const key = `${contribution.tier}.${contribution.direction}`;
      if (contribution.weight > best.weight) best = { key, weight: contribution.weight };
    }
    return best.key;
  }

  /** The blend-space thresholds in force. */
  get blendParams(): BlendSpaceParams {
    return this.#blend;
  }

  /**
   * Retune the blend thresholds after construction.
   *
   * The caller that owns movement speed is the only one that can know where the
   * walk and run rings should sit, and it cannot know until this graph has
   * measured the clips — so the dependency is resolved by handing the answer
   * back rather than by guessing at construction time.
   */
  setBlendParams(params: BlendSpaceParams): void {
    this.#blend = params;
  }

  /**
   * Measured metres of ground covered per animation cycle, by *semantic state*
   * rather than clip name, so callers do not need the clip table.
   */
  strideForState(state: string): number {
    const clipName = this.#resolved.get(state);
    if (clipName === undefined) return 0;
    return this.#entries.get(clipName)?.stride ?? 0;
  }

  /**
   * The ground speed a locomotion clip covers when played at its *authored*
   * rate, m/s — its measured stride over its own duration.
   *
   * This is the speed at which the clip needs no time-scaling at all, so it is
   * the right place to put the top of a blend ramp: below it the pose is being
   * blended down toward a slower gait, above it the pose is one clip and the
   * extra speed is spent entirely on playback rate. A ramp that instead runs
   * from the walk speed all the way to the run speed leaves the character in a
   * two-clip blend across almost his whole usable range, and a blended stance
   * plants worse than either clip does alone.
   */
  naturalSpeedForState(state: string): number {
    const clipName = this.#resolved.get(state);
    if (clipName === undefined) return 0;
    const entry = this.#entries.get(clipName);
    if (entry === undefined || entry.duration <= 1e-4) return 0;
    return entry.stride / entry.duration;
  }

  /** Measured metres-per-cycle for each locomotion clip. Diagnostics. */
  get strideTable(): ReadonlyMap<string, number> {
    const out = new Map<string, number>();
    for (const [key, entry] of this.#entries) out.set(key, entry.stride);
    return out;
  }

  /**
   * Current blend weight of every locomotion clip, by clip name.
   *
   * Diagnostics and tests: "which clips are playing and how much" is the only
   * question worth asking of a blend tree, and it is invisible from outside a
   * mixer otherwise.
   */
  get weights(): ReadonlyMap<string, number> {
    const out = new Map<string, number>();
    for (const [name, entry] of this.#entries) out.set(name, entry.weight);
    return out;
  }

  /** Cycles per second the locomotion blend is currently advancing at. */
  get cycleRate(): number {
    return this.#cycleRate;
  }

  /** Normalised position in the stride cycle, `[0, 1)`. */
  get phase(): number {
    return this.#phase;
  }

  /** Whether any action — or a specific one — is running. */
  isActionPlaying(name?: string): boolean {
    if (name === undefined) return this.#active.some((entry) => entry.phase !== 'done');
    return this.#active.some((entry) => entry.action === name && entry.phase !== 'done');
  }

  /** Handles for everything currently running, most recent last. */
  get activeActions(): readonly ActionHandle[] {
    return this.#active;
  }

  /* -- driving ------------------------------------------------------------ */

  /**
   * Feed the blend space.
   *
   * `velocityLocal` is the character's velocity **in its own space** — x right,
   * z forward — in metres per second. Converting from world space is the
   * caller's job because only the caller knows which yaw the body is using
   * (the movement heading, not the camera's).
   */
  setLocomotion(
    velocityLocal: { x: number; z: number },
    grounded: boolean,
  ): void {
    this.#sample = sampleBlendSpace(velocityLocal.x, velocityLocal.z, this.#blend);
    this.#speed = this.#sample.speed;
    this.#grounded = grounded;
  }

  /**
   * Advance the mixer. Call once per rendered frame with the frame delta —
   * this is presentation, not simulation, so it belongs in `update`, not
   * `fixedUpdate`.
   */
  update(dt: number): void {
    if (this.#disposed || dt < 0) return;

    // Airborne blend. Fast in (leaving the ground is instantaneous and reads as
    // such) and slower out, so landing settles instead of snapping.
    const airTarget = this.#grounded ? 0 : 1;
    const airRate = airTarget > this.#airBlend ? 1 / 0.09 : 1 / 0.16;
    this.#airBlend = approach(this.#airBlend, airTarget, airRate * dt);

    this.#updateActions(dt);
    this.#updateLocomotion(dt);
    this.#mixer.update(dt);
  }

  /**
   * Start a one-shot (or looping) action.
   *
   * The returned handle's `finished` promise always resolves — never rejects —
   * so combat code can `await` a swing without a try/catch around every hit.
   */
  playAction(name: string, options: PlayActionOptions = {}): ActionHandle {
    const clipName = options.clip ?? this.#resolved.get(name);
    const clip = clipName === undefined ? undefined : this.#clips.get(clipName);
    if (clipName === undefined || clip === undefined) {
      console.warn(`[AnimationGraph] no clip for action "${name}"; ignoring`);
      const stub = new ActiveAction({
        action: name,
        clip: '',
        layer: 'full',
        duration: 0,
        rootMotion: 0,
        mixerAction: null,
        fadeIn: 0,
        fadeOut: 0,
        loop: false,
        hold: false,
        speed: 1,
        events: [],
      });
      stub.settle('cancelled');
      return stub;
    }

    const layer = options.layer ?? DEFAULT_LAYER[name] ?? 'full';
    const interrupt = options.interrupt ?? true;
    if (interrupt) {
      for (const existing of this.#active) {
        if (existing.layer === layer && existing.phase !== 'done') existing.cancel(0.08);
      }
    }

    const source = layer === 'upper' ? this.#upperClip(clipName, clip) : clip;
    const mixerAction = this.#mixer.clipAction(source);
    mixerAction.reset();
    mixerAction.enabled = true;
    mixerAction.setLoop(
      options.loop === true ? THREE.LoopRepeat : THREE.LoopOnce,
      options.loop === true ? Infinity : 1,
    );
    mixerAction.clampWhenFinished = true;
    mixerAction.timeScale = options.speed ?? 1;
    mixerAction.setEffectiveWeight(0);
    mixerAction.play();

    const entry = new ActiveAction({
      action: name,
      clip: clipName,
      layer,
      duration: clip.duration,
      rootMotion: measureRootMotion(clip, 'root'),
      mixerAction,
      fadeIn: options.fadeIn ?? (layer === 'upper' ? 0.08 : 0.12),
      fadeOut: options.fadeOut ?? 0.18,
      loop: options.loop ?? false,
      hold: options.hold ?? false,
      speed: options.speed ?? 1,
      events: options.events ?? DEFAULT_EVENTS[name] ?? [],
    });
    this.#active.push(entry);
    this.#emit({ action: name, name: 'start', normalizedTime: 0 });
    return entry;
  }

  /** Cancel everything on a layer, or everything if no layer is given. */
  cancelActions(layer?: ActionLayer, fadeOut?: number): void {
    for (const entry of this.#active) {
      if (layer !== undefined && entry.layer !== layer) continue;
      entry.cancel(fadeOut);
    }
  }

  /** Subscribe to animation events. Returns an unsubscribe function. */
  onAnimationEvent(callback: (event: AnimationEvent) => void): () => void {
    this.#listeners.add(callback);
    return () => this.#listeners.delete(callback);
  }

  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    for (const entry of this.#active) if (entry.phase !== 'done') entry.settle('cancelled');
    this.#active = [];
    this.#mixer.stopAllAction();
    this.#mixer.uncacheRoot(this.#root);
    this.#listeners.clear();
  }

  /* -- construction ------------------------------------------------------- */

  #resolveStates(verbose: boolean): void {
    const missing: string[] = [];
    const table = CLIP_CANDIDATES as Record<string, readonly string[]>;
    for (const [state, candidates] of Object.entries(table)) {
      const found = candidates.find((candidate) => this.#clips.has(candidate));
      if (found === undefined) missing.push(state);
      else this.#resolved.set(state, found);
    }
    if (missing.length > 0 && verbose) {
      console.warn(`[AnimationGraph] unresolved states: ${missing.join(', ')}`);
    }
  }

  #upperClip(clipName: string, clip: THREE.AnimationClip): THREE.AnimationClip {
    const cached = this.#upperClips.get(clipName);
    if (cached !== undefined) return cached;
    const masked = filterClipTracks(clip, this.#nodeNames, (node) => this.#upperBones.has(node));
    this.#upperClips.set(clipName, masked);
    return masked;
  }

  #buildLocomotion(footBones: readonly [string, string]): void {
    // `findBone`, not `getObjectByName`: GLTFLoader deletes the dots from
    // `foot.l`, so the authored name never matches at runtime.
    const feet = footBones
      .map((name) => findBone(this.#root, name))
      .filter((bone): bone is THREE.Object3D => bone !== null);
    if (feet.length !== 2) {
      console.warn(
        `[AnimationGraph] foot bones ${footBones.join('/')} not found; ` +
          'stride matching falls back to authored constants and feet may slide',
      );
    }

    for (const state of LOCOMOTION_STATES) {
      const clipName = this.#resolved.get(state);
      if (clipName === undefined) continue;
      const clip = this.#clips.get(clipName);
      if (clip === undefined) continue;

      let entry = this.#findEntryByClip(clipName);
      if (entry === undefined) {
        const action = this.#mixer.clipAction(clip);
        action.setLoop(THREE.LoopRepeat, Infinity);
        action.enabled = true;
        action.setEffectiveWeight(0);
        // Time is assigned from the shared phase every frame, so the mixer must
        // not advance it on its own. This is what keeps the walk and the run
        // contacting the ground on the same beat through the whole blend.
        action.timeScale = 0;
        action.play();
        entry = {
          name: clipName,
          action,
          duration: clip.duration,
          stride: feet.length === 2 ? this.measureStrideDistance(clip, feet) : 0,
          weight: 0,
        };
        this.#entries.set(clipName, entry);
      }
      if (state === 'idle') this.#idle = entry;
      else if (state === 'air') this.#air = entry;
    }
  }

  #findEntryByClip(clipName: string): ClipEntry | undefined {
    return this.#entries.get(clipName);
  }

  /**
   * Measure how far a clip carries the body in one animation cycle.
   *
   * The clip is stepped through on a throwaway mixer and both feet are traced
   * in the character's own space, then handed to {@link plantedTravelPerCycle}.
   *
   * Returns 0 when the measurement is implausible, which makes the caller fall
   * back to the authored default rather than trusting a number that would set
   * the playback rate to something absurd.
   */
  measureStrideDistance(
    clip: THREE.AnimationClip,
    feet: readonly THREE.Object3D[],
    samples = 48,
  ): number {
    if (clip.duration <= 0 || feet.length === 0) return 0;

    const mixer = new THREE.AnimationMixer(this.#root);
    const action = mixer.clipAction(clip);
    action.play();
    action.setEffectiveWeight(1);
    action.timeScale = 0;

    const worldScale = this.#root.getWorldScale(new THREE.Vector3()).x || 1;
    const inverse = new THREE.Matrix4();
    const local = new THREE.Vector3();
    const traces: THREE.Vector3[][] = feet.map(() => []);

    for (let s = 0; s <= samples; s++) {
      action.time = (s / samples) * clip.duration;
      mixer.update(0);
      this.#root.updateMatrixWorld(true);
      inverse.copy(this.#root.matrixWorld).invert();
      for (let f = 0; f < feet.length; f++) {
        const foot = feet[f];
        const trace = traces[f];
        if (foot === undefined || trace === undefined) continue;
        local.setFromMatrixPosition(foot.matrixWorld).applyMatrix4(inverse);
        trace.push(local.clone().multiplyScalar(worldScale));
      }
    }

    mixer.stopAllAction();
    mixer.uncacheRoot(this.#root);

    let total = 0;
    let counted = 0;
    for (const trace of traces) {
      const rate = plantedTravelPerCycle(trace);
      if (rate <= 0) continue;
      total += rate;
      counted++;
    }
    if (counted === 0) return 0;
    const stride = total / counted;
    return stride >= 0.3 && stride <= 5 ? stride : 0;
  }

  #logTable(sourceClips: readonly THREE.AnimationClip[], rootNode: string): void {
    const strides = Array.from(this.#entries.values())
      .map((entry) => `${entry.name}=${entry.stride.toFixed(2)}m/cycle`)
      .join(', ');
    console.info(
      `[AnimationGraph] ${sourceClips.length} clips on "${this.#root.name || 'character'}"\n` +
        `  clips: ${sourceClips.map((clip) => clip.name).join(', ')}\n` +
        `  root node: ${rootNode} (horizontal translation flattened — in-place locomotion)\n` +
        `  upper-body mask: ${this.#upperBones.size} bones\n` +
        `  measured strides: ${strides || 'none'}`,
    );
  }

  /* -- per-frame ---------------------------------------------------------- */

  #updateActions(dt: number): void {
    let fullBodyWeight = 0;

    for (const entry of this.#active) {
      if (entry.phase === 'done') {
        // A held pose (death) stays settled but keeps suppressing locomotion.
        if (entry.layer === 'full') fullBodyWeight = Math.max(fullBodyWeight, entry.weight);
        continue;
      }

      entry.time += dt * entry.speed;

      // Events fire on the frame the playhead passes the marker, once each.
      const normalized = entry.normalizedTime;
      for (let i = 0; i < entry.events.length; i++) {
        const marker = entry.events[i];
        if (marker === undefined || entry.firedEvents.has(i)) continue;
        if (normalized >= marker.at) {
          entry.firedEvents.add(i);
          this.#emit({ action: entry.action, name: marker.name, normalizedTime: normalized });
        }
      }

      switch (entry.phase) {
        case 'in': {
          entry.weight = entry.fadeIn <= 0 ? 1 : Math.min(1, entry.weight + dt / entry.fadeIn);
          if (entry.weight >= 1) entry.phase = 'hold';
          break;
        }
        case 'hold': {
          entry.weight = 1;
          // Safety net. A non-looping action that somehow outlives its own clip
          // by a wide margin is a bug, and the symptom — locomotion pinned at
          // zero weight forever — is indistinguishable from a frozen character.
          // Better to end it and be wrong by a frame than to lock the player.
          if (!entry.loop && entry.time > entry.duration + entry.fadeOut + 0.5) {
            entry.phase = 'out';
            entry.outTimer = 0;
            break;
          }
          if (!entry.loop && entry.time >= entry.duration - entry.fadeOut) {
            if (entry.hold) {
              // Death and knockdown poses stay. The caller is responsible for
              // clearing them, because "get up" is a gameplay decision.
              entry.settle('completed');
              fullBodyWeight = Math.max(fullBodyWeight, entry.layer === 'full' ? 1 : 0);
              continue;
            }
            entry.phase = 'out';
            entry.outTimer = entry.fadeOut;
          }
          break;
        }
        case 'out': {
          entry.outTimer -= dt;
          entry.weight = entry.fadeOut <= 0 ? 0 : Math.max(0, entry.outTimer / entry.fadeOut);
          if (entry.outTimer <= 0) {
            entry.weight = 0;
            entry.mixerAction?.stop();
            entry.settle(entry.time >= entry.duration * 0.98 ? 'completed' : 'cancelled');
            this.#emit({
              action: entry.action,
              name: 'end',
              normalizedTime: entry.normalizedTime,
            });
          }
          break;
        }
        default:
          break;
      }

      entry.mixerAction?.setEffectiveWeight(entry.weight);
      if (entry.layer === 'full') fullBodyWeight = Math.max(fullBodyWeight, entry.weight);
    }

    this.#active = this.#active.filter((entry) => entry.phase !== 'done' || entry.weight > 0);
    this.#locomotionWeight = 1 - fullBodyWeight;
  }

  #updateLocomotion(dt: number): void {
    const contributions = blendContributions(this.#sample);
    const walkStride = this.#strideFor('walk', 'forward');
    const stride = blendedStride(
      contributions,
      (contribution) => this.#strideFor(contribution.tier, contribution.direction),
      walkStride,
    );

    this.#cycleRate = cycleRate(this.#speed, stride, this.#cadence[0], this.#cadence[1]);
    this.#lastPhase = this.#phase;
    if (this.#speed > this.#blend.idleThreshold) {
      this.#phase = (this.#phase + this.#cycleRate * dt) % 1;
      // Two contacts per cycle. Cheap, exact, and gives audio and the future
      // footstep VFX a hook that is locked to the pose rather than to a timer.
      if (crossed(this.#lastPhase, this.#phase, 0)) this.#footstep(0);
      if (crossed(this.#lastPhase, this.#phase, 0.5)) this.#footstep(0.5);
    }

    for (const entry of this.#entries.values()) entry.weight = 0;

    const ground = (1 - this.#airBlend) * this.#locomotionWeight;
    for (const contribution of contributions) {
      const entry = this.#entryFor(contribution.tier, contribution.direction);
      if (entry !== null) entry.weight += contribution.weight * ground;
    }
    if (this.#idle !== null) this.#idle.weight += this.#sample.tier.idle * ground;
    if (this.#air !== null) this.#air.weight += this.#airBlend * this.#locomotionWeight;

    for (const entry of this.#entries.values()) {
      entry.action.setEffectiveWeight(entry.weight);
      if (entry === this.#idle || entry === this.#air) {
        // Idle and the airborne loop have no stride to match, so they run on
        // their own clock. Advanced by hand because `timeScale` is 0 for every
        // locomotion action.
        entry.action.time = (entry.action.time + dt) % Math.max(entry.duration, 1e-3);
      } else {
        entry.action.time = this.#phase * entry.duration;
      }
    }
  }

  #entryFor(tier: 'walk' | 'run', direction: BlendDirection): ClipEntry | null {
    const clipName = this.#resolved.get(`${tier}.${direction}`);
    if (clipName === undefined) return null;
    return this.#entries.get(clipName) ?? null;
  }

  #strideFor(tier: 'walk' | 'run', direction: BlendDirection): number {
    const entry = this.#entryFor(tier, direction);
    if (entry === null || entry.stride <= 0) {
      // Authored fallbacks, used only when the measurement was rejected.
      return tier === 'run' ? 3.4 : 1.5;
    }
    return entry.stride;
  }

  #footstep(at: number): void {
    this.#emit({ action: 'locomotion', name: 'footstep', normalizedTime: at });
  }

  #emit(event: AnimationEvent): void {
    for (const listener of this.#listeners) {
      try {
        listener(event);
      } catch (error) {
        console.error(`[AnimationGraph] listener for "${event.name}" threw:`, error);
      }
    }
  }
}

/* -------------------------------------------------------------------------- */
/* Helpers                                                                    */
/* -------------------------------------------------------------------------- */

function collectNodeNames(root: THREE.Object3D): string[] {
  const names: string[] = [];
  root.traverse((object) => {
    if (object.name.length > 0) names.push(object.name);
  });
  return names;
}

function collectUpperBones(root: THREE.Object3D, maskRoot: string): Set<string> {
  const start = findBone(root, maskRoot) ?? findBone(root, 'spine');
  const bones = new Set<string>();
  if (start === null) {
    console.warn(`[AnimationGraph] no "${maskRoot}" bone; upper-body layering is disabled`);
    return bones;
  }
  start.traverse((object) => {
    if (object.name.length > 0) bones.add(object.name);
  });
  return bones;
}

/** Move `value` toward `target` by at most `maxDelta`. */
function approach(value: number, target: number, maxDelta: number): number {
  if (value < target) return Math.min(target, value + maxDelta);
  return Math.max(target, value - maxDelta);
}

/** Whether a wrapping phase passed `marker` between `from` and `to`. */
function crossed(from: number, to: number, marker: number): boolean {
  if (to >= from) return from < marker && marker <= to;
  return marker > from || marker <= to;
}
