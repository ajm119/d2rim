import { beforeEach, describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';

import {
  AnimationGraph,
  filterClipTracks,
  measureRootMotion,
  plantedTravelPerCycle,
  splitTrackName,
  stripRootMotion,
  type FootSample,
} from '../src/character/AnimationGraph';

/* -------------------------------------------------------------------------- */
/* A stand-in for the Barbarian's 41-joint rig                                */
/* -------------------------------------------------------------------------- */

/** Bone names and parents, mirroring the real GLB's naming exactly. */
const RIG: readonly (readonly [string, string | null])[] = [
  ['root', null],
  ['hips', 'root'],
  ['spine', 'hips'],
  ['chest', 'spine'],
  ['head', 'chest'],
  ['upperarm.l', 'chest'],
  ['lowerarm.l', 'upperarm.l'],
  ['hand.l', 'lowerarm.l'],
  ['upperarm.r', 'chest'],
  ['lowerarm.r', 'upperarm.r'],
  ['hand.r', 'lowerarm.r'],
  ['upperleg.l', 'hips'],
  ['lowerleg.l', 'upperleg.l'],
  ['foot.l', 'lowerleg.l'],
  ['upperleg.r', 'hips'],
  ['lowerleg.r', 'upperleg.r'],
  ['foot.r', 'lowerleg.r'],
];

function makeRig(): THREE.Object3D {
  const character = new THREE.Object3D();
  character.name = 'barbarian';
  const nodes = new Map<string, THREE.Object3D>();
  for (const [name, parent] of RIG) {
    const bone = new THREE.Object3D();
    bone.name = name;
    nodes.set(name, bone);
    (parent === null ? character : (nodes.get(parent) ?? character)).add(bone);
  }
  character.updateMatrixWorld(true);
  return character;
}

/**
 * A gait clip whose feet alternate: each foot is planted on the ground for half
 * the cycle while travelling `step` metres backwards, then swings forward with
 * its heel clear. One cycle therefore covers `2 * step` metres of ground, which
 * is the number `measureStrideDistance` has to recover.
 */
function makeGaitClip(name: string, duration: number, step: number): THREE.AnimationClip {
  const half = step / 2;
  const times = [0, 0.25, 0.5, 0.75, 1].map((t) => t * duration);
  const planted = [half, 0, -half];
  const swung = [-half, 0, half];
  const left = [
    0, 0, planted[0] ?? 0,
    0, 0, planted[1] ?? 0,
    0, 0, planted[2] ?? 0,
    0, 0.3, swung[1] ?? 0,
    0, 0, planted[0] ?? 0,
  ];
  const right = [
    0, 0, swung[0] ?? 0,
    0, 0.3, swung[1] ?? 0,
    0, 0, planted[0] ?? 0,
    0, 0, planted[1] ?? 0,
    0, 0, planted[2] ?? 0,
  ];
  return new THREE.AnimationClip(name, duration, [
    new THREE.VectorKeyframeTrack('foot.l.position', times, left),
    new THREE.VectorKeyframeTrack('foot.r.position', times, right),
    new THREE.QuaternionKeyframeTrack(
      'chest.quaternion',
      [0, duration],
      [0, 0, 0, 1, 0, 0.1, 0, 0.995],
    ),
  ]);
}

function makePoseClip(name: string, duration: number): THREE.AnimationClip {
  return new THREE.AnimationClip(name, duration, [
    new THREE.QuaternionKeyframeTrack(
      'chest.quaternion',
      [0, duration],
      [0, 0, 0, 1, 0, 0.2, 0, 0.98],
    ),
    new THREE.QuaternionKeyframeTrack(
      'upperleg.l.quaternion',
      [0, duration],
      [0, 0, 0, 1, 0.1, 0, 0, 0.995],
    ),
  ]);
}

function makeClips(): THREE.AnimationClip[] {
  return [
    makePoseClip('Idle', 1.0),
    makeGaitClip('Walking_A', 1.0, 0.75),
    makeGaitClip('Walking_Backwards', 1.0, 0.6),
    makeGaitClip('Running_A', 0.8, 1.7),
    makeGaitClip('Running_Strafe_Left', 0.8, 1.2),
    makeGaitClip('Running_Strafe_Right', 0.8, 1.2),
    makePoseClip('Jump_Idle', 0.5),
    makePoseClip('Jump_Start', 0.3),
    makePoseClip('Jump_Land', 0.4),
    makePoseClip('1H_Melee_Attack_Chop', 0.9),
    makePoseClip('1H_Melee_Attack_Slice_Diagonal', 0.8),
    makePoseClip('2H_Melee_Attack_Chop', 1.2),
    makePoseClip('Blocking', 1.0),
    makePoseClip('Hit_A', 0.6),
    makePoseClip('Death_A', 1.4),
    makePoseClip('T-Pose', 0.1),
  ];
}

function tick(graph: AnimationGraph, seconds: number, step = 1 / 60): void {
  for (let elapsed = 0; elapsed < seconds; elapsed += step) graph.update(step);
}

/* -------------------------------------------------------------------------- */

describe('clip surgery', () => {
  const nodes = RIG.map(([name]) => name).sort((a, b) => b.length - a.length);

  it('splits a track name even when the node name contains dots', () => {
    expect(splitTrackName('hand.l.quaternion', nodes)).toEqual({
      node: 'hand.l',
      property: 'quaternion',
    });
    expect(splitTrackName('chest.position', nodes)).toEqual({
      node: 'chest',
      property: 'position',
    });
    expect(splitTrackName('nonexistent.scale', nodes)).toBeNull();
  });

  it('keeps only the masked tracks', () => {
    const clip = makeGaitClip('Walking_A', 1, 0.75);
    const upper = filterClipTracks(clip, nodes, (node) => node === 'chest');
    expect(upper.tracks.map((track) => track.name)).toEqual(['chest.quaternion']);
    expect(upper.duration).toBe(clip.duration);
  });

  it('flattens horizontal root translation and leaves the bob alone', () => {
    const clip = new THREE.AnimationClip('Dodge_Forward', 0.4, [
      new THREE.VectorKeyframeTrack(
        'root.position',
        [0, 0.4],
        [0, 0, 0, 0, 0.05, 0.25],
      ),
    ]);
    expect(measureRootMotion(clip, 'root')).toBeCloseTo(0.25);
    const stripped = stripRootMotion(clip, 'root');
    expect(measureRootMotion(stripped, 'root')).toBe(0);
    const values = Array.from(stripped.tracks[0]?.values ?? []);
    expect(values.map((value) => Number(value.toFixed(4)))).toEqual([0, 0, 0, 0, 0.05, 0]);
  });
});

describe('AnimationGraph construction', () => {
  let graph: AnimationGraph;

  beforeEach(() => {
    graph = new AnimationGraph(makeRig(), makeClips(), { verbose: false });
  });

  it('enumerates the clips it was given and drops the T-pose', () => {
    expect(graph.clipNames).toContain('Walking_A');
    expect(graph.clipNames).toContain('Running_A');
    expect(graph.clipNames).not.toContain('T-Pose');
  });

  it('maps semantic states onto real clip names', () => {
    expect(graph.clipTable.get('idle')).toBe('Idle');
    expect(graph.clipTable.get('walk.forward')).toBe('Walking_A');
    expect(graph.clipTable.get('run.forward')).toBe('Running_A');
    expect(graph.clipTable.get('walk.back')).toBe('Walking_Backwards');
    expect(graph.clipTable.get('attack')).toBe('1H_Melee_Attack_Chop');
    expect(graph.clipTable.get('death')).toBe('Death_A');
  });

  it('recovers each clip stride from the motion of its planted feet', () => {
    // Two 0.75 m steps per cycle for the walk, two 1.7 m for the run.
    // NB these fixtures have each foot down for exactly half the cycle, which
    // is the one duty factor at which the old peak-to-peak estimator and the
    // stance-travel estimator agree — see `plantedTravelPerCycle`. That is
    // precisely why this suite passed while the real rig slid; the flight-phase
    // cases below are the ones that tell the two estimators apart.
    expect(graph.strideTable.get('Walking_A') ?? 0).toBeCloseTo(1.5, 2);
    expect(graph.strideTable.get('Running_A') ?? 0).toBeCloseTo(3.4, 2);
    expect(graph.strideTable.get('Running_Strafe_Left') ?? 0).toBeCloseTo(2.4, 2);
  });
});

/* -------------------------------------------------------------------------- */
/* Stride estimation                                                          */
/* -------------------------------------------------------------------------- */

/**
 * One cycle of a single foot, in body space.
 *
 * The foot is on the ground for `duty` of the cycle, travelling backwards at a
 * constant `stride` metres per unit phase, then swings forward with a lift. The
 * body therefore has to cover exactly `stride` metres per cycle for this foot
 * to stand still in the world — *whatever* `duty` is, which is the whole point.
 */
function footTrace(stride: number, duty: number, samples = 48, lift = 0.3): FootSample[] {
  const out: FootSample[] = [];
  for (let s = 0; s <= samples; s++) {
    const phi = s / samples;
    if (phi <= duty) {
      out.push({ x: 0, y: 0, z: (duty / 2 - phi) * stride });
    } else {
      const t = (phi - duty) / (1 - duty);
      out.push({ x: 0, y: Math.sin(Math.PI * t) * lift, z: (-duty / 2 + t * duty) * stride });
    }
  }
  return out;
}

/** Rotate a trace in the ground plane, so direction-independence is testable. */
function rotated(trace: readonly FootSample[], radians: number): FootSample[] {
  const cos = Math.cos(radians);
  const sin = Math.sin(radians);
  return trace.map((s) => ({ x: s.x * cos - s.z * sin, y: s.y, z: s.x * sin + s.z * cos }));
}

describe('plantedTravelPerCycle', () => {
  it('recovers the ground a cycle covers from a plain 50% duty gait', () => {
    expect(plantedTravelPerCycle(footTrace(1.5, 0.5))).toBeCloseTo(1.5, 2);
  });

  it('is not fooled by a flight phase, which is what broke the run', () => {
    // A run: each foot is down for a quarter of the cycle and travels 0.5 m
    // while it is, but the body covers 2.0 m — the rest is spent airborne.
    // Peak-to-peak excursion summed over two feet would report 2 * 0.5 = 1.0,
    // half the truth, and the mixer would then run at twice the needed rate.
    const trace = footTrace(2, 0.25);
    expect(plantedTravelPerCycle(trace)).toBeCloseTo(2, 1);

    let excursion = 0;
    for (const a of trace) for (const b of trace) excursion = Math.max(excursion, Math.abs(b.z - a.z));
    expect(excursion * 2).toBeLessThan(1.2);
  });

  it('reports the same number however the clip is oriented', () => {
    const forward = plantedTravelPerCycle(footTrace(1.5, 0.45));
    for (const angle of [Math.PI / 2, Math.PI, -Math.PI * 0.75, 0.37]) {
      expect(plantedTravelPerCycle(rotated(footTrace(1.5, 0.45), angle))).toBeCloseTo(forward, 3);
    }
  });

  it('scales linearly with the ground a cycle covers', () => {
    const one = plantedTravelPerCycle(footTrace(1, 0.4));
    const three = plantedTravelPerCycle(footTrace(3, 0.4));
    expect(three / one).toBeCloseTo(3, 1);
  });

  it('tolerates a threshold-straddling plant rather than averaging it away', () => {
    // A tiny lift puts most of the swing inside the plant band. The median is
    // what keeps those samples from dragging the answer; a mean would not.
    expect(plantedTravelPerCycle(footTrace(1.5, 0.5, 48, 0.02))).toBeCloseTo(1.5, 1);
  });

  it('gives up rather than guessing on a foot that never moves', () => {
    const still: FootSample[] = Array.from({ length: 24 }, () => ({ x: 0, y: 0, z: 0 }));
    expect(plantedTravelPerCycle(still)).toBe(0);
    expect(plantedTravelPerCycle([{ x: 0, y: 0, z: 0 }])).toBe(0);
  });
});

describe('locomotion state machine', () => {
  let graph: AnimationGraph;

  beforeEach(() => {
    graph = new AnimationGraph(makeRig(), makeClips(), { verbose: false });
  });

  it('is idle at rest and only idle', () => {
    graph.setLocomotion({ x: 0, z: 0 }, true);
    tick(graph, 0.5);
    expect(graph.state).toBe('idle');
    expect(graph.weights.get('Idle') ?? 0).toBeCloseTo(1, 2);
    expect(graph.weights.get('Running_A') ?? 0).toBe(0);
  });

  it('runs forward at speed', () => {
    graph.setLocomotion({ x: 0, z: 6 }, true);
    tick(graph, 0.5);
    expect(graph.state).toBe('run.forward');
    expect(graph.weights.get('Running_A') ?? 0).toBeCloseTo(1, 2);
    expect(graph.weights.get('Idle') ?? 0).toBe(0);
  });

  it('strafes without touching the forward clips', () => {
    graph.setLocomotion({ x: 6, z: 0 }, true);
    tick(graph, 0.5);
    expect(graph.state).toBe('run.right');
    expect(graph.weights.get('Running_Strafe_Right') ?? 0).toBeCloseTo(1, 2);
    expect(graph.weights.get('Running_A') ?? 0).toBe(0);
  });

  it('backpedals on the dedicated clip', () => {
    graph.setLocomotion({ x: 0, z: -1.9 }, true);
    tick(graph, 0.5);
    expect(graph.state).toBe('walk.back');
    expect(graph.weights.get('Walking_Backwards') ?? 0).toBeGreaterThan(0.9);
  });

  it('blends rather than snaps through the walk-to-run band', () => {
    graph.setLocomotion({ x: 0, z: 3 }, true);
    tick(graph, 0.3);
    const walk = graph.weights.get('Walking_A') ?? 0;
    const run = graph.weights.get('Running_A') ?? 0;
    expect(walk).toBeGreaterThan(0.05);
    expect(run).toBeGreaterThan(0.05);
    expect(walk + run).toBeCloseTo(1, 2);
  });

  it('drives cadence from ground speed so the feet keep up', () => {
    graph.setLocomotion({ x: 0, z: 6 }, true);
    tick(graph, 0.2);
    const fast = graph.cycleRate;
    graph.setLocomotion({ x: 0, z: 2 }, true);
    tick(graph, 0.2);
    expect(graph.cycleRate).toBeLessThan(fast);
  });

  it('advances a shared phase and wraps it', () => {
    graph.setLocomotion({ x: 0, z: 6 }, true);
    graph.update(0.1);
    const first = graph.phase;
    expect(first).toBeGreaterThan(0);
    tick(graph, 3);
    expect(graph.phase).toBeGreaterThanOrEqual(0);
    expect(graph.phase).toBeLessThan(1);
  });

  it('switches to the airborne pose when the ground goes away', () => {
    graph.setLocomotion({ x: 0, z: 5 }, true);
    tick(graph, 0.5);
    graph.setLocomotion({ x: 0, z: 5 }, false);
    tick(graph, 0.4);
    expect(graph.state).toBe('air');
    expect(graph.weights.get('Jump_Idle') ?? 0).toBeGreaterThan(0.9);
    expect(graph.weights.get('Running_A') ?? 0).toBeLessThan(0.1);
  });
});

describe('actions', () => {
  let graph: AnimationGraph;

  beforeEach(() => {
    graph = new AnimationGraph(makeRig(), makeClips(), { verbose: false });
    graph.setLocomotion({ x: 0, z: 5 }, true);
    tick(graph, 0.4);
  });

  it('reports what is playing and clears itself when finished', async () => {
    const handle = graph.playAction('attack');
    expect(handle.clip).toBe('1H_Melee_Attack_Chop');
    expect(handle.layer).toBe('upper');
    expect(graph.isActionPlaying('attack')).toBe(true);
    expect(graph.isActionPlaying('death')).toBe(false);

    tick(graph, 1.4);
    await expect(handle.finished).resolves.toBe('completed');
    expect(graph.isActionPlaying('attack')).toBe(false);
  });

  it('keeps the legs running under an upper-body action', () => {
    graph.playAction('attack');
    tick(graph, 0.3);
    expect(graph.weights.get('Running_A') ?? 0).toBeGreaterThan(0.9);
  });

  it('takes the whole body over for a committed action', () => {
    graph.playAction('attack.heavy');
    tick(graph, 0.4);
    expect(graph.weights.get('Running_A') ?? 0).toBeLessThan(0.15);
  });

  it('gives locomotion back after a full-body action ends', async () => {
    const handle = graph.playAction('attack.heavy');
    tick(graph, 1.6);
    await handle.finished;
    tick(graph, 0.3);
    expect(graph.weights.get('Running_A') ?? 0).toBeGreaterThan(0.9);
  });

  it('fires a hit event at the configured normalised time, exactly once', () => {
    const seen: { name: string; at: number }[] = [];
    graph.onAnimationEvent((event) => {
      if (event.name === 'hit') seen.push({ name: event.name, at: event.normalizedTime });
    });
    graph.playAction('attack', { events: [{ name: 'hit', at: 0.5 }] });
    tick(graph, 1.4);
    expect(seen).toHaveLength(1);
    expect(seen[0]?.at).toBeGreaterThanOrEqual(0.5);
    expect(seen[0]?.at).toBeLessThan(0.58);
  });

  it('resolves cancelled when interrupted part way through', async () => {
    const handle = graph.playAction('attack.heavy');
    tick(graph, 0.2);
    handle.cancel(0.05);
    tick(graph, 0.2);
    await expect(handle.finished).resolves.toBe('cancelled');
  });

  it('interrupts the previous action on the same layer', async () => {
    const first = graph.playAction('attack');
    tick(graph, 0.1);
    graph.playAction('attack.slice');
    tick(graph, 0.2);
    await expect(first.finished).resolves.toBe('cancelled');
    expect(graph.isActionPlaying('attack.slice')).toBe(true);
  });

  it('loops a block until it is cancelled', () => {
    const handle = graph.playAction('block', { loop: true });
    tick(graph, 3);
    expect(graph.isActionPlaying('block')).toBe(true);
    handle.cancel(0.05);
    tick(graph, 0.2);
    expect(graph.isActionPlaying('block')).toBe(false);
  });

  it('holds a death pose and keeps locomotion suppressed', async () => {
    const handle = graph.playAction('death', { layer: 'full', hold: true });
    tick(graph, 2);
    await expect(handle.finished).resolves.toBe('completed');
    tick(graph, 1);
    expect(graph.weights.get('Running_A') ?? 0).toBeLessThan(0.05);
  });

  it('degrades to a settled handle for an unknown action', async () => {
    const handle = graph.playAction('nope');
    await expect(handle.finished).resolves.toBe('cancelled');
    expect(graph.isActionPlaying('nope')).toBe(false);
  });

  it('reports footstep events twice per cycle', () => {
    let steps = 0;
    graph.onAnimationEvent((event) => {
      if (event.name === 'footstep') steps++;
    });
    graph.setLocomotion({ x: 0, z: 6 }, true);
    tick(graph, 2);
    expect(steps).toBeGreaterThanOrEqual(3);
  });
});
