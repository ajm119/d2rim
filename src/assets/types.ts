/**
 * @module assets/types
 *
 * Vocabulary shared by the asset manifest, the generated registry and the
 * runtime {@link AssetManager}.
 *
 * This module is deliberately dependency-free (no three.js import) so that the
 * generated registry can import it without dragging the renderer into any tool
 * or test that only wants to reason about asset metadata.
 */

/**
 * What an asset *is*, semantically.
 *
 * The role — not the file extension — decides how the asset is decoded and,
 * critically, which colour space its texture is tagged with. Getting that wrong
 * is the single most common source of "why does the lighting look wrong":
 * a normal map interpreted as sRGB produces subtly inverted-looking surface
 * detail that is very hard to spot and impossible to compensate for downstream.
 */
export type AssetRole =
  /** Equirectangular HDR environment map. Loaded as float, used for IBL. */
  | 'hdri'
  /** Base colour / diffuse. The only texture role that is sRGB-encoded. */
  | 'albedo'
  /** Tangent-space normal map. Linear. */
  | 'normal'
  /** Linear scalar roughness, conventionally in the green channel. */
  | 'roughness'
  /** Linear scalar metalness, conventionally in the blue channel. */
  | 'metalness'
  /** Linear ambient-occlusion, conventionally in the red channel. */
  | 'ao'
  /** Emissive colour. sRGB-encoded, like albedo. */
  | 'emissive'
  /** Any linear single- or multi-channel mask (height, alpha, blend weights). */
  | 'mask'
  /** A glTF / GLB scene. */
  | 'gltf'
  /** Opaque bytes with no interpretation. */
  | 'binary';

/**
 * Licence clearance state.
 *
 * `core` assets have a licence documented in a file we fetched and archived.
 * `review-required` assets are usable for local prototyping but must not reach
 * a public build until a human resolves their redistribution rights; the
 * fetcher skips them unless explicitly asked, so they are normally absent from
 * disk even though their keys exist.
 */
export type AssetTier = 'core' | 'review-required';

/** Texture roles that are colour data and therefore sRGB-encoded on disk. */
export const SRGB_ROLES: ReadonlySet<AssetRole> = new Set<AssetRole>(['albedo', 'emissive']);

/** Roles that decode to a `THREE.Texture` rather than a scene or raw bytes. */
export const TEXTURE_ROLES: ReadonlySet<AssetRole> = new Set<AssetRole>([
  'albedo',
  'normal',
  'roughness',
  'metalness',
  'ao',
  'emissive',
  'mask',
]);

/**
 * Whether a role's texture data is colour (sRGB) or measurement (linear).
 *
 * Exposed as a function rather than leaving callers to test the set directly,
 * because this is the decision the rest of the engine should be asking about by
 * name.
 */
export function isColorRole(role: AssetRole): boolean {
  return SRGB_ROLES.has(role);
}
