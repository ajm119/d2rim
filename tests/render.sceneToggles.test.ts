/**
 * The scene-graph kill switches, and the classifier they stand on.
 *
 * The claim these tests have to defend is narrow and load-bearing: `?props=off`
 * removes props, *and nothing else*. A bisection is only worth running if each
 * step subtracts exactly one thing; a switch that also quietly took the terrain
 * with it would produce a large, encouraging speed-up and a completely wrong
 * conclusion.
 *
 * The names asserted here are the real names from `src/scene/*.ts` — `ground`,
 * `camp.ground`, `den.floor`, `camp.forge.anvil.iron`, `moor.denMouth.rock`.
 * A heuristic is only as good as the corpus it was checked against, so the
 * corpus is written down.
 */

import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';

import { parseRenderFlags } from '../src/render/DebugFlags';
import {
  SceneToggles,
  classifySceneObject,
  collectCharacterRoots,
} from '../src/render/SceneToggles';

/** A mesh with a real (if trivial) geometry and material, named as the zones name theirs. */
function mesh(name: string): THREE.Mesh {
  const object = new THREE.Mesh(
    new THREE.BoxGeometry(1, 1, 1),
    new THREE.MeshStandardMaterial({ color: 0x808080 }),
  );
  object.name = name;
  return object;
}

/** A minimal rigged figure: the structural signature the classifier keys on. */
function character(name: string): THREE.Object3D {
  const group = new THREE.Group();
  group.name = name;
  const bone = new THREE.Bone();
  const skeleton = new THREE.Skeleton([bone]);
  const skinned = new THREE.SkinnedMesh(
    new THREE.BoxGeometry(1, 1, 1),
    new THREE.MeshStandardMaterial(),
  );
  skinned.name = `${name}.body`;
  skinned.bind(skeleton);
  group.add(bone, skinned);
  return group;
}

/**
 * A stand-in for the shipped scenes, using their actual object names.
 *
 * `camp.ground` is the encampment's ground plane, `den.floor` the cave floor,
 * and the props are the real ones a bisecting player will be looking at.
 */
function buildScene(): {
  scene: THREE.Scene;
  ground: THREE.Mesh;
  floor: THREE.Mesh;
  anvil: THREE.Mesh;
  rock: THREE.Mesh;
  hero: THREE.Object3D;
} {
  const scene = new THREE.Scene();
  const ground = mesh('camp.ground');
  const floor = mesh('den.floor');
  const anvil = mesh('camp.forge.anvil.iron');
  const rock = mesh('moor.denMouth.rock');
  const hero = character('barbarian');

  // Props nested under a grouping node, exactly as `#buildForge` builds them.
  const forge = new THREE.Group();
  forge.name = 'camp.forge';
  forge.add(anvil);

  scene.add(ground, floor, forge, rock, hero, new THREE.DirectionalLight());
  return { scene, ground, floor, anvil, rock, hero };
}

/** How many objects the renderer would actually walk into a draw. */
function visibleMeshCount(root: THREE.Object3D): number {
  let count = 0;
  const walk = (object: THREE.Object3D): void => {
    if (!object.visible) return;
    if ((object as { isMesh?: boolean }).isMesh === true) count++;
    for (const child of object.children) walk(child);
  };
  for (const child of root.children) walk(child);
  return count;
}

describe('classifySceneObject', () => {
  it('reads the ground meshes the three zones actually build', () => {
    expect(classifySceneObject(mesh('ground'))).toBe('terrain');
    expect(classifySceneObject(mesh('camp.ground'))).toBe('terrain');
    expect(classifySceneObject(mesh('den.floor'))).toBe('terrain');
  });

  it('does not mistake cave walls and ceilings for terrain', () => {
    // Deliberate, and the reason the match is on the last name segment rather
    // than a substring: if `den.wallskin` and `den.ceiling` counted as terrain
    // then `?terrain=off` would empty the Den of Evil completely, and "the
    // frame got fast when I removed the entire room" is not a finding.
    expect(classifySceneObject(mesh('den.ceiling'))).toBe('prop');
    expect(classifySceneObject(mesh('den.wallskin'))).toBe('prop');
    expect(classifySceneObject(mesh('camp.palisade'))).toBe('prop');
  });

  it('classifies anything rigged as a character, whatever it is called', () => {
    const scene = new THREE.Scene();
    const hero = character('barbarian');
    scene.add(hero);
    const roots = collectCharacterRoots(scene);
    expect(classifySceneObject(hero, roots.has(hero))).toBe('character');
  });

  it('lets a zone override the guess through userData', () => {
    const odd = mesh('some.unusual.name');
    odd.userData.d2rimKind = 'terrain';
    expect(classifySceneObject(odd)).toBe('terrain');

    const permanent = mesh('sky.dome');
    permanent.userData.d2rimKind = 'ignore';
    expect(classifySceneObject(permanent)).toBe('ignore');
  });

  it('falls back to prop, which is the safe default', () => {
    // A misfiled prop costs one object out of a `?props=off` run. A misfiled
    // terrain or character costs the whole frame's baseline.
    expect(classifySceneObject(mesh('camp.cart'))).toBe('prop');
    expect(classifySceneObject(mesh(''))).toBe('prop');
  });
});

describe('collectCharacterRoots', () => {
  it('marks the rig and every ancestor of it', () => {
    const scene = new THREE.Scene();
    const outer = new THREE.Group();
    outer.name = 'actors';
    const hero = character('barbarian');
    outer.add(hero);
    scene.add(outer);

    const roots = collectCharacterRoots(scene);
    expect(roots.has(outer)).toBe(true);
    expect(roots.has(hero)).toBe(true);
  });

  it('marks nothing in a scene with no rigs', () => {
    const scene = new THREE.Scene();
    scene.add(mesh('camp.ground'), mesh('camp.cart'));
    expect(collectCharacterRoots(scene).size).toBe(0);
  });
});

describe('SceneToggles', () => {
  it('is a no-op — and says so — when nothing is switched off', () => {
    expect(SceneToggles.isNoOp(parseRenderFlags(''))).toBe(true);
    expect(SceneToggles.isNoOp(parseRenderFlags('?props=off'))).toBe(false);
    expect(SceneToggles.isNoOp(parseRenderFlags('?flat=1'))).toBe(false);
    // The post/fog/shadow switches are handled by the frame graph, not here.
    expect(SceneToggles.isNoOp(parseRenderFlags('?fog=off&post=off'))).toBe(true);
  });

  it('?props=off hides props and leaves terrain and the hero standing', () => {
    const { scene, ground, floor, anvil, rock, hero } = buildScene();
    const before = visibleMeshCount(scene);

    const toggles = new SceneToggles(parseRenderFlags('?props=off'));
    toggles.sweep(scene);

    expect(anvil.visible).toBe(false);
    expect(rock.visible).toBe(false);
    expect(ground.visible).toBe(true);
    expect(floor.visible).toBe(true);
    expect(hero.visible).toBe(true);
    // The measurable claim: two fewer objects reach the render list.
    expect(visibleMeshCount(scene)).toBe(before - 2);
  });

  it('?terrain=off hides only the ground', () => {
    const { scene, ground, floor, anvil, hero } = buildScene();
    new SceneToggles(parseRenderFlags('?terrain=off')).sweep(scene);
    expect(ground.visible).toBe(false);
    expect(floor.visible).toBe(false);
    expect(anvil.visible).toBe(true);
    expect(hero.visible).toBe(true);
  });

  it('?chars=off hides the whole rig, not just the skinned mesh', () => {
    const { scene, hero, ground, anvil } = buildScene();
    new SceneToggles(parseRenderFlags('?chars=off')).sweep(scene);
    // Hiding the group is what actually removes the skinning work: an
    // invisible ancestor takes the subtree out of the render list, so the
    // skeleton is never uploaded and the skinned draw never happens.
    expect(hero.visible).toBe(false);
    expect(ground.visible).toBe(true);
    expect(anvil.visible).toBe(true);
    // ground, den.floor, the anvil and the rock survive; the rig's skinned
    // mesh is the only thing that went.
    expect(visibleMeshCount(scene)).toBe(4);
  });

  it('never hides a grouping node under ?props=off', () => {
    // The failure this guards against: `camp.forge` is a `Group`, classifies as
    // a prop, and hiding it would take the anvil *and* anything else parented
    // to it out — which is right by accident here and wrong the moment a zone
    // parents the ground or an NPC to a shared group.
    const { scene, anvil } = buildScene();
    const forge = scene.getObjectByName('camp.forge');
    new SceneToggles(parseRenderFlags('?props=off')).sweep(scene);
    expect(forge?.visible).toBe(true);
    expect(anvil.visible).toBe(false);
  });

  it('composes: ?props=off&terrain=off&chars=off empties the frame', () => {
    const { scene } = buildScene();
    new SceneToggles(parseRenderFlags('?props=off&terrain=off&chars=off')).sweep(scene);
    expect(visibleMeshCount(scene)).toBe(0);
  });

  it('is idempotent, because it re-sweeps on a cadence', () => {
    const { scene, anvil } = buildScene();
    const toggles = new SceneToggles(parseRenderFlags('?props=off'));
    toggles.sweep(scene);
    const first = toggles.stats.hidden;
    toggles.sweep(scene);
    expect(toggles.stats.hidden).toBe(first);
    expect(anvil.visible).toBe(false);
  });

  it('picks up objects that appear after the first sweep', () => {
    // Zones stream props in and gameplay spawns enemies and loot, so a
    // one-shot sweep at boot would leave most of a session unswitched.
    const { scene } = buildScene();
    const toggles = new SceneToggles(parseRenderFlags('?props=off'));
    toggles.sweep(scene);
    const late = mesh('camp.torchpost');
    scene.add(late);
    expect(late.visible).toBe(true);
    toggles.sweep(scene);
    expect(late.visible).toBe(false);
  });

  it('restores everything it hid on dispose', () => {
    const { scene, anvil, ground } = buildScene();
    const toggles = new SceneToggles(parseRenderFlags('?props=off&terrain=off'));
    toggles.sweep(scene);
    expect(anvil.visible).toBe(false);
    toggles.dispose();
    expect(anvil.visible).toBe(true);
    expect(ground.visible).toBe(true);
  });

  it('does not un-hide something the game itself hid', () => {
    const { scene, anvil } = buildScene();
    anvil.visible = false;
    const toggles = new SceneToggles(parseRenderFlags('?props=off'));
    toggles.sweep(scene);
    toggles.dispose();
    // The module only tracks what it changed, so a prop the quest system
    // switched off stays off.
    expect(anvil.visible).toBe(false);
  });
});

describe('?flat=1', () => {
  it('swaps every material for an unlit one and keeps the geometry', () => {
    const { scene, ground, anvil } = buildScene();
    const before = visibleMeshCount(scene);

    const toggles = new SceneToggles(parseRenderFlags('?flat=1'));
    toggles.sweep(scene);

    // The point of the flag: same draws, same triangles, none of the shading.
    expect(visibleMeshCount(scene)).toBe(before);
    expect((ground.material as THREE.Material).type).toBe('MeshBasicNodeMaterial');
    expect((anvil.material as THREE.Material).type).toBe('MeshBasicNodeMaterial');
    expect(toggles.stats.flattened).toBeGreaterThan(0);
  });

  it('flattens skinned meshes too, so characters keep animating', () => {
    const { scene, hero } = buildScene();
    new SceneToggles(parseRenderFlags('?flat=1')).sweep(scene);
    const body = hero.getObjectByName('barbarian.body') as THREE.SkinnedMesh;
    expect((body.material as THREE.Material).type).toBe('MeshBasicNodeMaterial');
  });

  it('carries the base colour across, so the frame is still readable', () => {
    const scene = new THREE.Scene();
    const red = mesh('camp.cart');
    (red.material as THREE.MeshStandardMaterial).color.setRGB(0.9, 0.1, 0.1);
    scene.add(red);

    new SceneToggles(parseRenderFlags('?flat=1')).sweep(scene);
    const flat = red.material as THREE.MeshBasicNodeMaterial;
    expect(flat.color.r).toBeCloseTo(0.9, 5);
    expect(flat.color.b).toBeCloseTo(0.1, 5);
  });

  it('reuses one stand-in per source material rather than one per mesh', () => {
    // Instanced props share a material; minting a new stand-in per mesh would
    // turn a flat run into hundreds of extra pipelines and measure the swap
    // instead of the scene.
    const scene = new THREE.Scene();
    const shared = new THREE.MeshStandardMaterial({ color: 0x445566 });
    for (let i = 0; i < 5; i++) {
      const object = new THREE.Mesh(new THREE.BoxGeometry(), shared);
      object.name = `camp.timber.${i}`;
      scene.add(object);
    }

    new SceneToggles(parseRenderFlags('?flat=1')).sweep(scene);
    const materials = new Set(scene.children.map((child) => (child as THREE.Mesh).material));
    expect(materials.size).toBe(1);
  });

  it('restores the original materials on dispose', () => {
    const { scene, ground } = buildScene();
    const original = ground.material;
    const toggles = new SceneToggles(parseRenderFlags('?flat=1'));
    toggles.sweep(scene);
    expect(ground.material).not.toBe(original);
    toggles.dispose();
    expect(ground.material).toBe(original);
  });

  it('does not flatten what it has already hidden', () => {
    // `?minimal=1&props=off` must not pay to build stand-in materials for
    // objects that will never be drawn.
    const { scene, anvil } = buildScene();
    const toggles = new SceneToggles(parseRenderFlags('?flat=1&props=off'));
    toggles.sweep(scene);
    expect(anvil.visible).toBe(false);
    expect((anvil.material as THREE.Material).type).toBe('MeshStandardMaterial');
  });
});
