#!/usr/bin/env node
/**
 * Transcode the shipped JPEG texture set to KTX2 / Basis Universal (ETC1S).
 *
 * ## Why
 *
 * A JPEG is small on disk and enormous in VRAM. It decodes to raw RGBA8, so
 * every 2048x2048 plate costs 16.8 MB on the GPU, or 22.4 MB once three
 * generates its mip chain. The project ships 40 of them: 836 MB of texture
 * memory before a single model, HDRI or render target is allocated. That is
 * what makes the deployed build fall over with Chromium's "Error code: 5",
 * which is an out-of-memory renderer crash rather than a driver fault.
 *
 * KTX2 with ETC1S transcodes to BC1 on desktop (0.5 bytes/px) and *stays
 * compressed on the GPU*, so the same plate costs 2.1 MB, or 2.8 MB with mips.
 * That is an 8x reduction, and it is a reduction in the number that actually
 * crashes the tab.
 *
 * It also happens to be smaller on the wire — measured 860 KB against the
 * 2.6 MB source JPEG for `terrain-grass-albedo` — because ETC1S output is
 * entropy-coded. So this is not the usual quality-for-size trade; it is better
 * on both axes at the cost of block-compression artefacts.
 *
 * ## The JPEGs remain the source of truth
 *
 * `manifest.json` and `registry.generated.ts` still describe the JPEGs, with
 * their licences and hashes. This tool writes `.ktx2` siblings plus a generated
 * lookup module, and the AssetManager prefers the compressed variant *when the
 * GPU can transcode it*, falling back to the JPEG otherwise. Nothing downstream
 * has to know which one it got.
 *
 * ## Encoder settings, and why
 *
 * - `-y_flip`: three's `KTX2Loader` forces `flipY = false` because a compressed
 *   texture cannot be flipped at upload time, while `TextureLoader` defaults to
 *   `flipY = true`. Baking the flip in is what makes the two paths produce the
 *   same image; without it every surface that swaps to KTX2 gets mirrored UVs
 *   and a normal map with an inverted green channel.
 * - `-normal_map` on normal maps: disables the perceptual/selector RDO tuning
 *   that assumes it is looking at colour. Applied to a normal map that tuning
 *   smears the very high-frequency detail the map exists to carry.
 * - `-linear` on every non-colour role: roughness, AO, metalness and normals are
 *   data, not radiance, so both the error metric and the mip filter must run in
 *   linear space.
 *
 * Usage:
 *   node tools/assets/encode-ktx2.mjs [--force] [--jobs N]
 */

import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const SRC_DIR = path.join(ROOT, 'public/assets/textures');
const OUT_DIR = path.join(ROOT, 'public/assets/textures-ktx2');
const GENERATED = path.join(ROOT, 'src/assets/ktx2.generated.ts');
const BASISU = path.join(ROOT, 'node_modules/basis_universal/bin/basisu');

const args = process.argv.slice(2);
const FORCE = args.includes('--force');
const jobsFlag = args.indexOf('--jobs');
const JOBS =
  jobsFlag >= 0 ? Number(args[jobsFlag + 1]) : Math.max(1, Math.min(4, os.cpus().length));

/**
 * Roles that carry colour, and therefore want sRGB-aware error metrics.
 * Everything else is data and must be encoded linearly.
 */
const COLOR_SUFFIXES = new Set(['albedo', 'emissive', 'diffuse', 'basecolor']);

/** Derive the role from the filename suffix, e.g. `rock-cliff-normal.jpg`. */
function roleOf(basename) {
  const stem = basename.replace(/\.[^.]+$/, '');
  const suffix = stem.slice(stem.lastIndexOf('-') + 1).toLowerCase();
  return suffix;
}

function encoderArgsFor(role) {
  const out = ['-ktx2', '-mipmap', '-y_flip', '-comp_level', '1'];
  if (role === 'normal') {
    // Normal maps need every bit of the quality budget and none of the
    // colour-oriented heuristics.
    out.push('-normal_map', '-linear', '-mip_linear', '-q', '255');
  } else if (COLOR_SUFFIXES.has(role)) {
    out.push('-q', '190');
  } else {
    out.push('-linear', '-mip_linear', '-q', '190');
  }
  return out;
}

function sha256(file) {
  return createHash('sha256').update(fs.readFileSync(file)).digest('hex');
}

function encode(srcPath, outPath, role) {
  return new Promise((resolve, reject) => {
    const child = spawn(
      BASISU,
      [...encoderArgsFor(role), '-output_file', outPath, srcPath],
      { stdio: ['ignore', 'ignore', 'pipe'] },
    );
    let stderr = '';
    child.stderr.on('data', (d) => (stderr += d));
    child.on('error', reject);
    child.on('close', (code) => {
      if (code === 0) resolve();
      else reject(new Error(`basisu exited ${code} for ${srcPath}\n${stderr}`));
    });
  });
}

async function main() {
  if (!fs.existsSync(BASISU)) {
    console.error(
      `[encode-ktx2] basisu not found at ${BASISU}.\n` +
        `Install it with:  npm install --no-save basis_universal@1.16.4-1`,
    );
    process.exit(1);
  }
  fs.chmodSync(BASISU, 0o755);
  fs.mkdirSync(OUT_DIR, { recursive: true });

  const sources = fs
    .readdirSync(SRC_DIR)
    .filter((f) => /\.(jpg|jpeg|png)$/i.test(f))
    .sort();

  const results = [];
  let index = 0;
  let encoded = 0;
  let skipped = 0;

  async function worker() {
    for (;;) {
      const i = index++;
      if (i >= sources.length) return;
      const file = sources[i];
      const srcPath = path.join(SRC_DIR, file);
      const outName = `${file.replace(/\.[^.]+$/, '')}.ktx2`;
      const outPath = path.join(OUT_DIR, outName);
      const role = roleOf(file);

      if (!FORCE && fs.existsSync(outPath)) {
        skipped++;
      } else {
        const started = Date.now();
        await encode(srcPath, outPath, role);
        encoded++;
        console.log(
          `[encode-ktx2] ${file} -> ${outName} ` +
            `(${(fs.statSync(srcPath).size / 1048576).toFixed(2)} MB -> ` +
            `${(fs.statSync(outPath).size / 1048576).toFixed(2)} MB, ` +
            `${((Date.now() - started) / 1000).toFixed(1)}s, role=${role})`,
        );
      }

      results.push({
        source: `assets/textures/${file}`,
        ktx2: `assets/textures-ktx2/${outName}`,
        bytes: fs.statSync(outPath).size,
        sha256: sha256(outPath),
      });
    }
  }

  await Promise.all(Array.from({ length: JOBS }, worker));
  results.sort((a, b) => a.source.localeCompare(b.source));

  const srcBytes = sources.reduce((n, f) => n + fs.statSync(path.join(SRC_DIR, f)).size, 0);
  const outBytes = results.reduce((n, r) => n + r.bytes, 0);

  const body = results
    .map((r) => `  '${r.source}': '${r.ktx2}',`)
    .join('\n');

  fs.writeFileSync(
    GENERATED,
    `/**
 * @module assets/ktx2.generated
 *
 * GENERATED FILE — do not edit.
 *
 * Produced by \`tools/assets/encode-ktx2.mjs\`. Regenerate with
 * \`npm run assets:ktx2\`.
 *
 * Maps a manifest texture path to its KTX2 / Basis Universal sibling. The JPEG
 * named on the left is still the licensed source of truth in
 * \`registry.generated.ts\`; the KTX2 on the right is a build artefact the
 * AssetManager substitutes when the GPU can transcode it.
 *
 * Encoded as ETC1S, which transcodes to BC1 on desktop: 0.5 bytes per pixel and
 * it stays compressed in VRAM, against 4 bytes per pixel for a decoded JPEG.
 *
 * Source set:     ${(srcBytes / 1048576).toFixed(1)} MB across ${sources.length} files
 * Compressed set: ${(outBytes / 1048576).toFixed(1)} MB
 */

/** Manifest texture path -> compressed sibling path. */
export const KTX2_VARIANTS: Readonly<Record<string, string>> = {
${body}
};

/** Total on-disk size of the compressed set, in bytes. */
export const KTX2_TOTAL_BYTES = ${outBytes};
`,
    'utf8',
  );

  console.log(
    `\n[encode-ktx2] ${encoded} encoded, ${skipped} up to date.\n` +
      `[encode-ktx2] on disk: ${(srcBytes / 1048576).toFixed(1)} MB JPEG -> ` +
      `${(outBytes / 1048576).toFixed(1)} MB KTX2\n` +
      `[encode-ktx2] wrote ${path.relative(ROOT, GENERATED)}`,
  );
}

main().catch((error) => {
  console.error('[encode-ktx2]', error);
  process.exit(1);
});
