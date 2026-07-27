/**
 * How far can an enemy's swing actually reach?
 *
 * Loads an enemy GLB straight from disk, walks each melee clip, builds the
 * implied blade exactly the way `combat/Hitbox.LimbAnchor` does at runtime
 * (hand bone, extended past the fist along the forearm by the profile reach),
 * and sweeps a player-sized capsule outward along the enemy's facing to find
 * the furthest separation at which the blade still touches it.
 *
 * The output is the source of the `window` and `reachDuringWindow` numbers in
 * `src/ai/enemies/Skeleton.ts`. Re-run it whenever an attack is retargeted onto
 * a different clip, and copy the numbers across — `tests/ai.skeleton.test.ts`
 * asserts that every attack's reach clears its variant's stand-off, so a swing
 * authored onto a clip that cannot reach fails the suite instead of silently
 * disarming the enemy for a whole phase, which is what happened in phase 3.
 *
 *   node tools/measure-enemy-reach.mjs [glb ...]
 *
 * Reads `phi:distance` pairs across the whole clip; `0` means "never touches at
 * any separation", which is what the old `1H_Melee_Attack_Chop` scored almost
 * everywhere.
 */
import { readFileSync } from 'node:fs';
import * as THREE from 'three';

// GLTFLoader reaches for `self` and for blob URLs when it decodes textures.
// Neither exists in node and neither matters here: only the skeleton is read.
globalThis.self = globalThis;
const { GLTFLoader } = await import('three/examples/jsm/loaders/GLTFLoader.js');

/** Matches `EnemyProfile.reach` for the skeletons. */
const REACH = 0.8;
/** Matches `PlayerCombatant.hitRadius` / `hitHeight` and the hitbox radius. */
const PLAYER_RADIUS = 0.4;
const PLAYER_HEIGHT = 1.85;
const BLADE_RADIUS = 0.13;
/** Matches the heights in `SKELETON_PROFILES`, close enough for every variant. */
const ENEMY_HEIGHT = 1.78;

const CLIPS = [
  '1H_Melee_Attack_Chop',
  '1H_Melee_Attack_Slice_Diagonal',
  '1H_Melee_Attack_Slice_Horizontal',
  '1H_Melee_Attack_Stab',
  '2H_Melee_Attack_Chop',
];

const files =
  process.argv.length > 2
    ? process.argv.slice(2)
    : [
        'public/assets/models/enemies/SkeletonWarrior.glb',
        'public/assets/models/enemies/SkeletonMinion.glb',
        'public/assets/models/enemies/SkeletonRogue.glb',
        'public/assets/models/enemies/SkeletonMage.glb',
      ];

/** Closest distance between a segment and the player capsule's axis segment. */
function segmentGap(hilt, tip, base, top) {
  let best = Infinity;
  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  for (let i = 0; i <= 24; i++) {
    a.lerpVectors(hilt, tip, i / 24);
    for (let j = 0; j <= 24; j++) {
      b.lerpVectors(base, top, j / 24);
      best = Math.min(best, a.distanceTo(b));
    }
  }
  return best;
}

for (const file of files) {
  const buffer = readFileSync(file);
  const gltf = await new Promise((resolve, reject) =>
    new GLTFLoader().parse(
      buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength),
      '',
      resolve,
      reject,
    ),
  );
  const root = gltf.scene;
  root.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(root);
  root.scale.setScalar(ENEMY_HEIGHT / (box.max.y - box.min.y));
  root.updateMatrixWorld(true);

  const bones = {};
  root.traverse((o) => (bones[o.name.toLowerCase().replace(/[^a-z0-9]/g, '')] = o));
  if (bones.handr === undefined || bones.lowerarmr === undefined) {
    console.log(`${file}: no hand.r / lowerarm.r — this rig can never land a hit`);
    continue;
  }

  console.log(`\n=== ${file}`);
  for (const name of CLIPS) {
    const clip = gltf.animations.find((c) => c.name === name);
    if (clip === undefined) continue;
    const mixer = new THREE.AnimationMixer(root);
    const action = mixer.clipAction(clip);
    action.play();
    action.setEffectiveWeight(1);
    action.timeScale = 0;

    const row = [];
    let peak = 0;
    for (let s = 0; s <= 40; s++) {
      const phi = s / 40;
      action.time = phi * clip.duration;
      mixer.update(0);
      root.updateMatrixWorld(true);
      const hilt = bones.handr.getWorldPosition(new THREE.Vector3());
      const elbow = bones.lowerarmr.getWorldPosition(new THREE.Vector3());
      const tip = hilt.clone().sub(elbow).normalize().multiplyScalar(REACH).add(hilt);
      let contact = 0;
      // The enemy faces +z, so sweep the capsule out along +z.
      for (let d = 2.4; d >= 0.2; d -= 0.02) {
        const base = new THREE.Vector3(0, PLAYER_RADIUS, d);
        const top = new THREE.Vector3(0, PLAYER_HEIGHT - PLAYER_RADIUS, d);
        if (segmentGap(hilt, tip, base, top) <= PLAYER_RADIUS + BLADE_RADIUS) {
          contact = d;
          break;
        }
      }
      peak = Math.max(peak, contact);
      row.push(`${phi.toFixed(2)}:${contact.toFixed(2)}`);
    }
    mixer.stopAllAction();
    mixer.uncacheRoot(root);
    console.log(`${name.padEnd(34)} peak ${peak.toFixed(2)} m | ${row.join(' ')}`);
  }
}
