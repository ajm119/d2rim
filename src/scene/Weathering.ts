/**
 * @module scene/Weathering
 *
 * The project's single prop-weathering path: one loaded glTF material in, one
 * graded node material out.
 *
 * *Mechanism* lives here; *policy* does not. The grades themselves — how dark a
 * Blood Moor barrel is, how much chroma the Barbarian keeps, how cold a cave
 * wall reads — stay with the scene that makes the art-direction claim, because
 * they are that scene's opinion. What every scene shares is the pipeline the
 * opinion is expressed through, and duplicating that pipeline per zone is how a
 * project ends up with three subtly different definitions of "grimdark".
 *
 * Extracted from `scene/BloodMoor`, unchanged, when the Rogue Encampment and the
 * Den of Evil needed the same treatment with different numbers.
 */

import * as THREE from 'three/webgpu';
import {
  float,
  luminance,
  mix,
  mx_noise_float,
  normalWorldGeometry,
  positionWorld,
  saturate,
  smoothstep,
  texture,
  vec3,
  vec4,
} from 'three/tsl';

/**
 * The weathering multiplier applied to every prop albedo.
 *
 * Cold and dark, with the blue channel held highest so that what survives the
 * desaturation is the *shade* direction rather than the wood tone. Tuned
 * against the terrain: props land within about a stop of the mud they stand
 * on, which is what stops them clipping out of a frame exposed for the ground.
 */
/**
 * One weathering recipe. Two exist: one for the world, one for the subject.
 *
 * Separating them is the difference between "everything is desaturated" and an
 * *art direction*. A prop's job is to sit inside the palette and never be
 * looked at; the Barbarian's job is to be looked at, which means he gets the
 * same treatment at half the strength — enough that he belongs in the world,
 * not so much that he stops being the figure in it.
 */
export interface WeatheringGrade {
  /** How far the sampled albedo is dragged toward its own luminance, 0–1. */
  readonly desaturation: number;
  /**
   * Multiplied onto the *dark* end of the desaturated albedo.
   *
   * See {@link weatherMaterial}: the tint is a two-stop ramp driven by the
   * albedo's own luminance, not a constant. This is the cold stop.
   */
  readonly shadowTint: THREE.Color;
  /** The warm stop of the same ramp, multiplied onto the light end. */
  readonly lightTint: THREE.Color;
  /**
   * How hard residual red chroma is pulled back after grading, 0–1.
   *
   * 0 leaves the hue alone; 1 removes warm chroma entirely. The palette rule
   * is that the fire is the only warm source, so props get most of this and the
   * hero gets almost none.
   */
  readonly warmClamp: number;
  /** Peak strength of the world-space soot/grime multiply, 0–1. */
  readonly grime: number;
  /** Peak strength of the up-facing dust / edge-wear lift, 0–1. */
  readonly wear: number;
  /** Added to 0.6x the source roughness. Nothing out here is polished. */
  readonly roughnessFloor: number;
  readonly envMapIntensity: number;
}

/**
 * Rebuild one prop material as a weathered node material.
 *
 * The colour correction has to happen in the shader, not on `material.color`.
 * These packs put all of their colour in a palette atlas and leave the
 * base-colour *factor* white, so multiplying the factor — which is what this
 * function used to do — can darken a prop but can never desaturate one. And
 * saturation was the actual problem: the moor was full of salmon-pink barrels
 * and primary-green conifers, dark but fully chromatic, and the warm firelight
 * then pushed them the rest of the way into candy.
 *
 * So the material is promoted to a `MeshStandardNodeMaterial` and given an
 * explicit `colorNode`:
 *
 * ```
 *   albedo = texture(map) · material.color
 *   albedo  = texture(map) · material.color
 *   flat    = mix(albedo, luminance(albedo), desaturation)
 *   tinted  = flat · mix(shadowTint, lightTint, luminance(flat))
 *   grimed  = tinted · worldSootNoise + upFacingDust
 *   graded  = grey(grimed) + chroma(grimed) · warmClampMatrix
 * ```
 *
 * The reason this is safe — and the reason the previous author was right to be
 * nervous about it — is `TextureNode.generate`, which wraps every sample in
 * `colorSpaceToWorking(…, texture.colorSpace)`. The sRGB decode the built-in
 * map path applies is therefore applied here too, automatically. Skipping it
 * would roughly double the midtones, in the exact opposite direction from the
 * point of this pass, and would be invisible until measured.
 *
 * Promoting to a node material has a second, unplanned benefit: `#applyOcclusion`
 * only folds the GTAO buffer into `THREE.NodeMaterial` instances, so before
 * this change every prop in the scene was silently missing its contact
 * occlusion. Now they all have it.
 */
export function weatherMaterial(source: THREE.Material, grade: WeatheringGrade): THREE.Material {
  // Duck-typed rather than `instanceof`: three's own cross-realm type tags
  // cannot be defeated by a second copy of the library arriving through the
  // example loaders, and a material that fell through this check would pass
  // straight out unweathered — a bug whose only symptom is a colour that is
  // subtly wrong, three layers away from its cause.
  if (!isStandardLike(source)) return source.clone();

  const clone = source.clone();
  const node = new THREE.MeshStandardNodeMaterial();
  // The same key-copy `NodeLibrary.fromMaterial` performs, which is three's own
  // supported route from a loaded material to its node equivalent. Done against
  // a *clone* so nothing mutable (`normalScale`, `color`) is aliased back into
  // the `AssetManager`'s cache. The two casts are the boundary: `for…in` over a
  // material is untyped by construction.
  const from = clone as unknown as Record<string, unknown>;
  const to = node as unknown as Record<string, unknown>;
  for (const key in from) {
    // Identity fields must stay the node material's own, or three's caches key
    // two different materials to one entry.
    if (key === 'uuid' || key === 'id' || key === 'version' || key === 'type') continue;
    if (key in to) to[key] = from[key];
  }

  const map = node.map;
  const factor = new THREE.Color().copy(clone.color);
  const sampled = map === null ? vec4(1, 1, 1, 1) : texture(map);
  const albedo = sampled.rgb.mul(vec3(factor.r, factor.g, factor.b));

  // 1. Desaturate, lightly. See `WEATHERING_DESATURATION`.
  const flattened = mix(albedo, vec3(luminance(albedo)), float(grade.desaturation)).toVar(
    'weatherFlattened',
  );

  // 2. The two-stop tint. The ramp parameter is the albedo's *own* luminance,
  //    so a prop's dark half and its light half get different hues rather than
  //    different brightnesses of one hue. That is the entire difference between
  //    a limited palette and a monochrome one, and it costs one `mix`.
  // NB: every `toVar` name in this function is prefixed. TSL emits the name
  // verbatim as a GLSL identifier, and `flat` — the obvious name for the
  // desaturated albedo — is a reserved interpolation qualifier in GLSL ES 3.0.
  // It compiles silently on the WebGPU/WGSL path and hard-fails the WebGL2
  // fragment shader, which is a divergence between backends introduced by a
  // variable name.
  const key = saturate(luminance(flattened)).toVar('weatherKey');
  const tinted = flattened
    .mul(
      mix(
        vec3(grade.shadowTint.r, grade.shadowTint.g, grade.shadowTint.b),
        vec3(grade.lightTint.r, grade.lightTint.g, grade.lightTint.b),
        key,
      ),
    )
    .toVar('weatherTinted');

  // 3. Surface craft, in two world-space terms that cost four noise samples
  //    between them and give the kit props the painterly variation their flat
  //    atlas cells cannot.
  //
  //    - soot/grime: a two-octave world-space noise multiply. World space, not
  //      UV space, so the pattern does not repeat per instance — fifty crates
  //      cloned from one mesh stop reading as fifty copies of one crate.
  //    - dust/wear: up-facing surfaces get a slight cold lift. This is the
  //      cheapest legible substitute for curvature-driven edge wear: on the
  //      chunky, hard-edged, large-facet KayKit silhouettes the top faces *are*
  //      the worn edges, so an N·up term lands almost exactly where a curvature
  //      map would and needs no extra vertex data.
  const grimeSpace = positionWorld.mul(0.85);
  const soot = mx_noise_float(grimeSpace)
    .mul(0.5)
    .add(mx_noise_float(grimeSpace.mul(3.3).add(vec3(19, 4, 7))).mul(0.28))
    .add(0.5);
  const grimed = tinted
    .mul(mix(float(1), saturate(soot), float(grade.grime)))
    .add(
      vec3(0.055, 0.058, 0.066).mul(
        float(grade.wear).mul(smoothstep(0.35, 0.98, normalWorldGeometry.y)),
      ),
    )
    .toVar('weatherGrimed');

  // 4. The warm clamp. Mixing toward luminance preserves hue *direction*, so a
  //    salmon barrel desaturates into a paler salmon barrel and stays the most
  //    chromatic warm thing in a frame whose whole premise is that the fire is
  //    the only warm source. Scaling the red component of the residual chroma
  //    vector — not the red channel of the colour — pulls warm props toward the
  //    cold ambient while leaving cold ones and overall value untouched.
  const grey = luminance(grimed);
  const chroma = grimed.sub(vec3(grey));
  const graded = vec3(grey).add(
    vec3(
      chroma.x.mul(float(1 - grade.warmClamp)),
      chroma.y.mul(float(1 - grade.warmClamp * 0.35)),
      chroma.z,
    ),
  );

  // Alpha is forced to 1 and the material forced opaque. The kit atlases carry
  // padding cells with alpha < 1, and any prop whose source material happened
  // to arrive with `transparent: true` — which the key-copy above faithfully
  // preserves — then punched a hole through itself. Nothing in this prop set is
  // legitimately translucent, so the correct fix is to say so once here rather
  // than to audit fifty-seven glTF materials.
  node.colorNode = vec4(graded.max(vec3(0)), 1);
  node.transparent = false;
  node.opacity = 1;
  node.alphaTest = 0;
  node.depthWrite = true;
  // The factor is folded into the node above; leaving it on would apply twice.
  node.color.setRGB(1, 1, 1);

  node.roughness = THREE.MathUtils.clamp(clone.roughness * 0.6 + grade.roughnessFloor, 0.5, 1);
  node.metalness = Math.min(clone.metalness, 0.2);
  // Trimmed so a bright overcast sky does not relight the props past the
  // terrain they stand on.
  node.envMapIntensity = grade.envMapIntensity;
  node.name = `${source.name}.weathered`;
  clone.dispose();
  return node;
}

/** three's own cross-realm type tag for the standard/physical material family. */
function isStandardLike(
  material: THREE.Material,
): material is THREE.MeshStandardMaterial | THREE.MeshPhysicalMaterial {
  const tagged = material as unknown as {
    isMeshStandardMaterial?: boolean;
    isMeshPhysicalMaterial?: boolean;
  };
  return tagged.isMeshStandardMaterial === true || tagged.isMeshPhysicalMaterial === true;
}
