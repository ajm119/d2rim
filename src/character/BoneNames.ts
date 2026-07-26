/**
 * @module character/BoneNames
 *
 * Finding a bone by the name the artist gave it.
 *
 * `GLTFLoader` runs every node name through `PropertyBinding.sanitizeNodeName`
 * before putting it in the scene graph, because three's animation track syntax
 * is `node.property` and a node called `hand.l` would make that grammar
 * ambiguous. Sanitising **deletes** `. [ ] : /` outright, so the Barbarian's
 * `foot.l` is `footl` at runtime and `upperleg.r` is `upperlegr`.
 *
 * Nothing warns about this. `getObjectByName('foot.l')` simply returns
 * `undefined`, and the symptom is a foot IK system that silently does nothing
 * and a stride measurement that silently returns zero — both of which look
 * exactly like "the feature is off" rather than "the lookup failed".
 *
 * So every bone lookup in `src/character` goes through here, which tries the
 * authored name, then the sanitised form, then a last-resort normalised scan.
 */

import type * as THREE from 'three/webgpu';

/** The characters `PropertyBinding.sanitizeNodeName` deletes. */
const RESERVED = /[\[\]./:]/g;

/** Apply three's own node-name sanitisation. */
export function sanitizeBoneName(name: string): string {
  return name.replace(/\s/g, '_').replace(RESERVED, '');
}

/** Strip everything that is not alphanumeric and lowercase the rest. */
function normalise(name: string): string {
  return name.toLowerCase().replace(/[^a-z0-9]/g, '');
}

/**
 * Find a bone by its authored name, tolerating loader-side renaming.
 *
 * Tried in order: the name as given, the sanitised form, the underscore form
 * (some exporters use it), and finally a normalised scan of the whole subtree.
 * The scan is O(bones) and runs once at init, never per frame.
 */
export function findBone(root: THREE.Object3D, name: string): THREE.Object3D | null {
  const candidates = [name, sanitizeBoneName(name), name.replace(/\./g, '_')];
  for (const candidate of candidates) {
    const found = root.getObjectByName(candidate);
    if (found !== undefined) return found;
  }

  const wanted = normalise(name);
  let match: THREE.Object3D | null = null;
  root.traverse((object) => {
    if (match === null && normalise(object.name) === wanted) match = object;
  });
  return match;
}
