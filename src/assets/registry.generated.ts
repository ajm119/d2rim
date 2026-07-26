/**
 * @module assets/registry.generated
 *
 * GENERATED FILE — do not edit.
 *
 * Produced by `tools/assets/fetch-assets.mjs` from `tools/assets/manifest.json`.
 * Regenerate with `npm run assets`. `tests/assets.registry.test.ts` fails if this
 * file and the manifest disagree, so the two cannot silently drift apart.
 *
 * Paths are relative to the served root and are resolved against
 * `import.meta.env.BASE_URL` at load time, so the game still works when it is
 * deployed under a sub-path.
 */

import type { AssetRole, AssetTier } from './types';

/** One entry in the generated asset registry. */
export interface GeneratedAssetEntry {
  /** Path relative to the served root, e.g. `assets/hdri/overcast.exr`. */
  readonly path: string;
  /** Drives colour space and filtering decisions in the AssetManager. */
  readonly role: AssetRole;
  /** `review-required` assets are absent unless explicitly fetched. */
  readonly tier: AssetTier;
  /** SPDX identifier of the licence this asset ships under. */
  readonly license: string;
  /** Byte length recorded at fetch time; `null` when never fetched. */
  readonly bytes: number | null;
  /** SHA-256 recorded at fetch time; `null` when never fetched. */
  readonly sha256: string | null;
}

export const GENERATED_ASSETS = {
  'env.overcast': {
    path: 'assets/hdri/overcast.exr',
    role: 'hdri',
    tier: 'core',
    license: 'CC0-1.0',
    bytes: 312862,
    sha256: '6fff311b5f3dad0910d8f334a6d9d88218ffd873d72c679ee1b7e4bccfa140e3',
  },
  'env.dusk': {
    path: 'assets/hdri/dusk.exr',
    role: 'hdri',
    tier: 'core',
    license: 'CC0-1.0',
    bytes: 148357,
    sha256: 'd09a4d7581f5454f13c3da3c323a231d00a34e60200de7654a895d2d41a7ceb7',
  },
  'env.cave': {
    path: 'assets/hdri/cave.exr',
    role: 'hdri',
    tier: 'core',
    license: 'CC0-1.0',
    bytes: 154431,
    sha256: '9509f1b917dbb4c98d63b92e9fd12fb79e3376d17ec3186e2de39543e32ce5ff',
  },
  'detail.normal.coarse': {
    path: 'assets/normals/detail-coarse.webp',
    role: 'normal',
    tier: 'core',
    license: 'CC0-1.0',
    bytes: 115088,
    sha256: 'f8606c84cfcb8ed0dff37ee457b37c4c50ba50347fef259ecefc945aa91a857d',
  },
  'detail.normal.fine': {
    path: 'assets/normals/detail-fine.webp',
    role: 'normal',
    tier: 'core',
    license: 'CC0-1.0',
    bytes: 66246,
    sha256: 'b3141d549811dbab3ed05683cbacf2f9775d02999dec6ac0767322d26bf34a29',
  },
  'detail.normal.organic': {
    path: 'assets/normals/detail-organic.webp',
    role: 'normal',
    tier: 'core',
    license: 'CC0-1.0',
    bytes: 94568,
    sha256: '01804607a0e340d338e47b1b1fb5f4ce3503f6f35261934b283067f3120bc6a9',
  },
  'character.robot': {
    path: 'assets/models/RobotExpressive.glb',
    role: 'gltf',
    tier: 'core',
    license: 'CC0-1.0',
    bytes: 463988,
    sha256: '047f5e5fb3bb6d378bd1df16ca6137f2a596c99b3a1b5690b4020c05aaf6f319',
  },
  'creature.fox': {
    path: 'assets/models/Fox.glb',
    role: 'gltf',
    tier: 'core',
    license: 'CC-BY-4.0',
    bytes: 162852,
    sha256: 'd97044e701822bac5a62696459b27d7b375aada5de8574ed4362edbba94771f7',
  },
  'character.bigvegas': {
    path: 'assets/models/review/BigVegas.glb',
    role: 'gltf',
    tier: 'review-required',
    license: 'LicenseRef-Mixamo-Unclear',
    bytes: null,
    sha256: null,
  },
  'animation.xbot': {
    path: 'assets/models/review/Xbot.glb',
    role: 'gltf',
    tier: 'review-required',
    license: 'LicenseRef-Mixamo-Unclear',
    bytes: null,
    sha256: null,
  },
} as const satisfies Record<string, GeneratedAssetEntry>;

/** Every semantic asset key known to the build. */
export type GeneratedAssetKey = keyof typeof GENERATED_ASSETS;
