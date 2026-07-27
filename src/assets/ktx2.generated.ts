/**
 * @module assets/ktx2.generated
 *
 * GENERATED FILE — do not edit.
 *
 * Produced by `tools/assets/encode-ktx2.mjs`. Regenerate with
 * `npm run assets:ktx2`.
 *
 * Maps a manifest texture path to its KTX2 / Basis Universal sibling. The JPEG
 * named on the left is still the licensed source of truth in
 * `registry.generated.ts`; the KTX2 on the right is a build artefact the
 * AssetManager substitutes when the GPU can transcode it.
 *
 * Encoded as ETC1S, which transcodes to BC1 on desktop: 0.5 bytes per pixel and
 * it stays compressed in VRAM, against 4 bytes per pixel for a decoded JPEG.
 *
 * Source set:     34.7 MB across 39 files
 * Compressed set: 26.7 MB
 */

/** Manifest texture path -> compressed sibling path. */
export const KTX2_VARIANTS: Readonly<Record<string, string>> = {
  'assets/textures/cloth-albedo.jpg': 'assets/textures-ktx2/cloth-albedo.ktx2',
  'assets/textures/cloth-normal.jpg': 'assets/textures-ktx2/cloth-normal.ktx2',
  'assets/textures/cloth-roughness.jpg': 'assets/textures-ktx2/cloth-roughness.ktx2',
  'assets/textures/metal-iron-albedo.jpg': 'assets/textures-ktx2/metal-iron-albedo.ktx2',
  'assets/textures/metal-iron-metalness.jpg': 'assets/textures-ktx2/metal-iron-metalness.ktx2',
  'assets/textures/metal-iron-normal.jpg': 'assets/textures-ktx2/metal-iron-normal.ktx2',
  'assets/textures/metal-iron-roughness.jpg': 'assets/textures-ktx2/metal-iron-roughness.ktx2',
  'assets/textures/rock-cliff-albedo.jpg': 'assets/textures-ktx2/rock-cliff-albedo.ktx2',
  'assets/textures/rock-cliff-ao.jpg': 'assets/textures-ktx2/rock-cliff-ao.ktx2',
  'assets/textures/rock-cliff-normal.jpg': 'assets/textures-ktx2/rock-cliff-normal.ktx2',
  'assets/textures/rock-cliff-roughness.jpg': 'assets/textures-ktx2/rock-cliff-roughness.ktx2',
  'assets/textures/rock-mossy-albedo.jpg': 'assets/textures-ktx2/rock-mossy-albedo.ktx2',
  'assets/textures/rock-mossy-ao.jpg': 'assets/textures-ktx2/rock-mossy-ao.ktx2',
  'assets/textures/rock-mossy-normal.jpg': 'assets/textures-ktx2/rock-mossy-normal.ktx2',
  'assets/textures/rock-mossy-roughness.jpg': 'assets/textures-ktx2/rock-mossy-roughness.ktx2',
  'assets/textures/stone-masonry-albedo.jpg': 'assets/textures-ktx2/stone-masonry-albedo.ktx2',
  'assets/textures/stone-masonry-ao.jpg': 'assets/textures-ktx2/stone-masonry-ao.ktx2',
  'assets/textures/stone-masonry-normal.jpg': 'assets/textures-ktx2/stone-masonry-normal.ktx2',
  'assets/textures/stone-masonry-roughness.jpg': 'assets/textures-ktx2/stone-masonry-roughness.ktx2',
  'assets/textures/terrain-grass-albedo.jpg': 'assets/textures-ktx2/terrain-grass-albedo.ktx2',
  'assets/textures/terrain-grass-ao.jpg': 'assets/textures-ktx2/terrain-grass-ao.ktx2',
  'assets/textures/terrain-grass-normal.jpg': 'assets/textures-ktx2/terrain-grass-normal.ktx2',
  'assets/textures/terrain-grass-roughness.jpg': 'assets/textures-ktx2/terrain-grass-roughness.ktx2',
  'assets/textures/terrain-mud-albedo.jpg': 'assets/textures-ktx2/terrain-mud-albedo.ktx2',
  'assets/textures/terrain-mud-ao.jpg': 'assets/textures-ktx2/terrain-mud-ao.ktx2',
  'assets/textures/terrain-mud-normal.jpg': 'assets/textures-ktx2/terrain-mud-normal.ktx2',
  'assets/textures/terrain-mud-roughness.jpg': 'assets/textures-ktx2/terrain-mud-roughness.ktx2',
  'assets/textures/wood-bark-albedo.jpg': 'assets/textures-ktx2/wood-bark-albedo.ktx2',
  'assets/textures/wood-bark-ao.jpg': 'assets/textures-ktx2/wood-bark-ao.ktx2',
  'assets/textures/wood-bark-normal.jpg': 'assets/textures-ktx2/wood-bark-normal.ktx2',
  'assets/textures/wood-bark-roughness.jpg': 'assets/textures-ktx2/wood-bark-roughness.ktx2',
  'assets/textures/wood-floor-albedo.jpg': 'assets/textures-ktx2/wood-floor-albedo.ktx2',
  'assets/textures/wood-floor-ao.jpg': 'assets/textures-ktx2/wood-floor-ao.ktx2',
  'assets/textures/wood-floor-normal.jpg': 'assets/textures-ktx2/wood-floor-normal.ktx2',
  'assets/textures/wood-floor-roughness.jpg': 'assets/textures-ktx2/wood-floor-roughness.ktx2',
  'assets/textures/wood-plank-albedo.jpg': 'assets/textures-ktx2/wood-plank-albedo.ktx2',
  'assets/textures/wood-plank-ao.jpg': 'assets/textures-ktx2/wood-plank-ao.ktx2',
  'assets/textures/wood-plank-normal.jpg': 'assets/textures-ktx2/wood-plank-normal.ktx2',
  'assets/textures/wood-plank-roughness.jpg': 'assets/textures-ktx2/wood-plank-roughness.ktx2',
};

/** Total on-disk size of the compressed set, in bytes. */
export const KTX2_TOTAL_BYTES = 27960063;
