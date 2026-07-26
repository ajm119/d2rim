#!/usr/bin/env node
/**
 * d2rim asset fetcher.
 *
 * Declarative, idempotent and resumable. Everything it does is driven by
 * `tools/assets/manifest.json`; this file contains no asset URLs of its own.
 *
 * ## What it guarantees
 *
 * - **Idempotent.** A second run downloads nothing. Files already on disk are
 *   matched against `assets.lock.json` by size (fast path) or by SHA-256
 *   (`--verify`), and skipped when they agree.
 * - **Resumable.** HTTP downloads stream into a `.part` file and are only
 *   renamed into place once complete, so an interrupted run never leaves a
 *   truncated asset that looks valid. A surviving `.part` is resumed with a
 *   Range request when the server supports it.
 * - **Integrity-checked.** npm tarballs are verified against the registry's own
 *   `dist.integrity` (SHA-512) before a single byte is extracted. Every emitted
 *   file is SHA-256'd, and a hash that disagrees with a manifest pin is a hard
 *   failure, never a warning.
 * - **Licence-auditable.** Each asset names the file at source that documents
 *   its licence (`licenseEvidence`). That file is downloaded and archived next
 *   to the asset, so the licence claim can be checked offline and years later,
 *   without trusting this script's summary of it.
 *
 * ## Licence tiers
 *
 * `core` assets have a licence documented in a fetchable file. They are
 * downloaded by default. `review-required` assets are reachable and useful but
 * their redistribution rights are unresolved; they are skipped unless the
 * operator explicitly passes `--include-review-required`, and they are marked
 * as unresolved in both attribution outputs. This is the mechanism that keeps
 * "licence compliance is non-negotiable" true by construction rather than by
 * remembering.
 *
 * ## Outputs
 *
 *   public/assets/**              the assets themselves
 *   public/assets/licenses/**     archived licence evidence
 *   public/ATTRIBUTIONS.md        human-readable credits
 *   public/ATTRIBUTIONS.json      machine-readable credits
 *   src/assets/registry.generated.ts   typed semantic-key -> path registry
 *   tools/assets/assets.lock.json      content hashes (commit this)
 *
 * ## Usage
 *
 *   node tools/assets/fetch-assets.mjs
 *   node tools/assets/fetch-assets.mjs --verify        # re-hash everything
 *   node tools/assets/fetch-assets.mjs --force         # ignore cache, refetch
 *   node tools/assets/fetch-assets.mjs --only=hdri     # substring filter on key
 *   node tools/assets/fetch-assets.mjs --include-review-required
 *   node tools/assets/fetch-assets.mjs --dry-run
 */

import { createHash } from 'node:crypto';
import { createWriteStream } from 'node:fs';
import { mkdir, readFile, rename, rm, stat, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';
import { gunzipSync } from 'node:zlib';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const MANIFEST_PATH = join(ROOT, 'tools/assets/manifest.json');
const LOCK_PATH = join(ROOT, 'tools/assets/assets.lock.json');

/* -------------------------------------------------------------------------- */
/* CLI                                                                         */
/* -------------------------------------------------------------------------- */

function parseArgs(argv) {
  const flags = {
    force: false,
    verify: false,
    dryRun: false,
    includeReviewRequired: false,
    only: null,
  };
  for (const arg of argv) {
    if (arg === '--force') flags.force = true;
    else if (arg === '--verify') flags.verify = true;
    else if (arg === '--dry-run') flags.dryRun = true;
    else if (arg === '--include-review-required') flags.includeReviewRequired = true;
    else if (arg.startsWith('--only=')) flags.only = arg.slice('--only='.length);
    else if (arg === '--help' || arg === '-h') {
      console.log(
        'usage: fetch-assets.mjs [--force] [--verify] [--dry-run] ' +
          '[--only=<substring>] [--include-review-required]',
      );
      process.exit(0);
    } else {
      throw new Error(`unknown flag: ${arg}`);
    }
  }
  return flags;
}

/* -------------------------------------------------------------------------- */
/* Logging                                                                     */
/* -------------------------------------------------------------------------- */

const useColor = process.stdout.isTTY === true && process.env.NO_COLOR === undefined;
const paint = (code, s) => (useColor ? `[${code}m${s}[0m` : s);
const dim = (s) => paint('2', s);
const green = (s) => paint('32', s);
const yellow = (s) => paint('33', s);
const red = (s) => paint('31', s);
const cyan = (s) => paint('36', s);

function humanBytes(n) {
  if (n === null || n === undefined) return '—';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}

/* -------------------------------------------------------------------------- */
/* Small filesystem helpers                                                    */
/* -------------------------------------------------------------------------- */

async function ensureDir(path) {
  await mkdir(path, { recursive: true });
}

async function fileSize(path) {
  try {
    const s = await stat(path);
    return s.isFile() ? s.size : null;
  } catch {
    return null;
  }
}

function sha256(buffer) {
  return createHash('sha256').update(buffer).digest('hex');
}

async function sha256File(path) {
  return sha256(await readFile(path));
}

async function readJson(path, fallback) {
  try {
    return JSON.parse(await readFile(path, 'utf8'));
  } catch (error) {
    if (fallback !== undefined && error.code === 'ENOENT') return fallback;
    throw error;
  }
}

/* -------------------------------------------------------------------------- */
/* HTTP                                                                        */
/* -------------------------------------------------------------------------- */

/**
 * Download `url` to `destPath`, resuming a partial `.part` file when possible.
 *
 * The `.part` indirection is what makes an interrupted run safe: the final path
 * only ever appears after the whole body has been written, so a half-downloaded
 * asset can never be mistaken for a complete one on the next run.
 */
async function downloadToFile(url, destPath, { retries = 3 } = {}) {
  await ensureDir(dirname(destPath));
  const partPath = `${destPath}.part`;

  for (let attempt = 1; attempt <= retries; attempt++) {
    const existing = (await fileSize(partPath)) ?? 0;
    const headers = {
      // Some proxies serve a cached error body without an explicit UA.
      'user-agent': 'd2rim-asset-fetcher/1.0 (+https://github.com/ajm119/d2rim)',
    };
    if (existing > 0) headers.range = `bytes=${existing}-`;

    try {
      const response = await fetch(url, { headers, redirect: 'follow' });

      // 416 means our .part is already >= the full length: it is stale garbage.
      if (response.status === 416) {
        await rm(partPath, { force: true });
        continue;
      }
      if (!response.ok) {
        throw new Error(`HTTP ${response.status} ${response.statusText}`);
      }
      if (response.body === null) {
        throw new Error('empty response body');
      }

      // The server honoured the range only if it answered 206. A 200 means it
      // is sending the whole file again, so the existing bytes must be dropped
      // rather than appended to.
      const resuming = existing > 0 && response.status === 206;
      if (existing > 0 && !resuming) await rm(partPath, { force: true });

      await pipeline(
        Readable.fromWeb(response.body),
        createWriteStream(partPath, { flags: resuming ? 'a' : 'w' }),
      );

      // Cross-check against Content-Length when the server sent an unranged
      // response; a truncated stream that ended cleanly is otherwise invisible.
      //
      // This is only meaningful for an identity-coded response. When the server
      // applies a transfer compression (raw.githubusercontent.com gzips text
      // assets such as .gltf, but not binaries like .glb/.bin), Content-Length
      // describes the COMPRESSED body while fetch hands us the DECOMPRESSED
      // stream, so the two legitimately disagree -- we would otherwise reject a
      // perfectly good file for being larger than advertised.
      const encoding = (response.headers.get('content-encoding') ?? '').trim().toLowerCase();
      const identityCoded = encoding === '' || encoding === 'identity';
      const declared = Number(response.headers.get('content-length'));
      if (!resuming && identityCoded && Number.isFinite(declared) && declared > 0) {
        const actual = await fileSize(partPath);
        if (actual !== declared) {
          throw new Error(`truncated download: got ${actual} of ${declared} bytes`);
        }
      }

      await rename(partPath, destPath);
      return;
    } catch (error) {
      if (attempt === retries) {
        throw new Error(`failed to download ${url}: ${error.message}`, { cause: error });
      }
      const backoff = 400 * attempt;
      console.log(dim(`      retry ${attempt}/${retries - 1} in ${backoff}ms (${error.message})`));
      await new Promise((r) => setTimeout(r, backoff));
    }
  }
}

/** Fetch a URL fully into a Buffer. Used for small metadata documents. */
async function fetchBuffer(url) {
  const response = await fetch(url, {
    headers: { 'user-agent': 'd2rim-asset-fetcher/1.0' },
    redirect: 'follow',
  });
  if (!response.ok) throw new Error(`HTTP ${response.status} for ${url}`);
  return Buffer.from(await response.arrayBuffer());
}

/* -------------------------------------------------------------------------- */
/* npm tarballs                                                                */
/* -------------------------------------------------------------------------- */

/**
 * Minimal USTAR reader.
 *
 * Implemented here rather than shelling out to `tar` or adding a dependency:
 * the format is 512-byte headers plus 512-byte-aligned payloads, npm tarballs
 * are plain USTAR, and this keeps the fetcher dependency-free so it can run
 * before `npm install` if it ever needs to.
 */
function untar(buffer) {
  const entries = new Map();
  let offset = 0;
  let longName = null;

  while (offset + 512 <= buffer.length) {
    const header = buffer.subarray(offset, offset + 512);
    // Two consecutive zero blocks terminate the archive; one is enough to stop.
    if (header[0] === 0) break;

    const readField = (start, length) =>
      header.subarray(start, start + length).toString('utf8').replace(/\0.*$/s, '').trim();

    let name = readField(0, 100);
    const size = parseInt(readField(124, 12) || '0', 8) || 0;
    const typeFlag = String.fromCharCode(header[156]);
    const prefix = readField(345, 155);
    if (prefix !== '') name = `${prefix}/${name}`;
    if (longName !== null) {
      name = longName;
      longName = null;
    }

    offset += 512;
    const data = buffer.subarray(offset, offset + size);
    offset += Math.ceil(size / 512) * 512;

    if (typeFlag === 'L') {
      // GNU long-name extension: this entry's payload is the next entry's name.
      longName = data.toString('utf8').replace(/\0.*$/s, '');
    } else if (typeFlag === '0' || typeFlag === '\0' || typeFlag === '') {
      entries.set(name, Buffer.from(data));
    }
  }
  return entries;
}

/** Verify an npm `dist.integrity` string (`<alg>-<base64>`) against bytes. */
function verifyIntegrity(buffer, integrity) {
  if (typeof integrity !== 'string' || !integrity.includes('-')) return;
  const [algorithm, expected] = [
    integrity.slice(0, integrity.indexOf('-')),
    integrity.slice(integrity.indexOf('-') + 1),
  ];
  const actual = createHash(algorithm).update(buffer).digest('base64');
  if (actual !== expected) {
    throw new Error(
      `npm integrity mismatch (${algorithm}): registry said ${expected}, got ${actual}`,
    );
  }
}

/**
 * Download and unpack an npm package, memoised per run and cached on disk.
 *
 * Resolving through the packument rather than guessing the tarball URL gets us
 * the registry's own integrity hash, which is a far stronger check than
 * anything we could invent.
 */
const npmCache = new Map();
async function loadNpmPackage(source, cacheDir, { force }) {
  const id = `${source.package}@${source.version}`;
  const memo = npmCache.get(id);
  if (memo !== undefined) return memo;

  const tarballPath = join(cacheDir, `${source.package.replace(/[@/]/g, '_')}-${source.version}.tgz`);

  const promise = (async () => {
    let tarball = null;
    if (!force) {
      try {
        tarball = await readFile(tarballPath);
      } catch {
        tarball = null;
      }
    }

    if (tarball === null) {
      const packumentUrl = `https://registry.npmjs.org/${source.package}`;
      console.log(dim(`      resolving ${id} via ${packumentUrl}`));
      const packument = JSON.parse((await fetchBuffer(packumentUrl)).toString('utf8'));
      const versionInfo = packument.versions?.[source.version];
      if (versionInfo === undefined) {
        throw new Error(`${source.package} has no version ${source.version} on the registry`);
      }
      tarball = await fetchBuffer(versionInfo.dist.tarball);
      verifyIntegrity(tarball, versionInfo.dist.integrity);
      console.log(
        dim(`      tarball ${humanBytes(tarball.length)}, integrity OK (${versionInfo.dist.integrity.split('-')[0]})`),
      );
      await ensureDir(cacheDir);
      await writeFile(tarballPath, tarball);
    }

    const entries = untar(gunzipSync(tarball));
    // npm tarballs root everything under `package/`; strip it so manifest
    // members read as the package's own paths.
    const stripped = new Map();
    for (const [name, data] of entries) {
      stripped.set(name.startsWith('package/') ? name.slice('package/'.length) : name, data);
    }
    return stripped;
  })();

  npmCache.set(id, promise);
  return promise;
}

/* -------------------------------------------------------------------------- */
/* Transforms                                                                  */
/* -------------------------------------------------------------------------- */

/**
 * Decode a `@pmndrs/assets`-style module back into the binary it wraps.
 *
 * Those packages ship every asset as `export default 'data:<mime>;base64,...'`
 * so that bundlers can inline them. We want the real file on disk, so the
 * payload is extracted and decoded.
 */
function dataUriModuleToBinary(source, label) {
  const text = source.toString('utf8');
  const match = /base64,\s*([A-Za-z0-9+/=\s]+?)['"`]/.exec(text);
  if (match === null) {
    throw new Error(`${label}: no base64 data URI payload found in module`);
  }
  const binary = Buffer.from(match[1].replace(/\s+/g, ''), 'base64');
  if (binary.length === 0) throw new Error(`${label}: decoded payload is empty`);
  return binary;
}

const TRANSFORMS = {
  'data-uri-module': dataUriModuleToBinary,
};

/* -------------------------------------------------------------------------- */
/* Licence evidence                                                            */
/* -------------------------------------------------------------------------- */

/**
 * Fetch and archive the file at source that documents an asset's licence.
 *
 * Reference syntax is `<sourceId>:<pathOrMember>`. Deduplicated per run, since
 * one LICENSE file typically covers many assets.
 */
const evidenceCache = new Map();
async function archiveLicenseEvidence(ref, manifest, outputRoot, cacheDir, flags) {
  if (typeof ref !== 'string' || ref === '') return null;
  const memo = evidenceCache.get(ref);
  if (memo !== undefined) return memo;

  const separator = ref.indexOf(':');
  const sourceId = ref.slice(0, separator);
  const member = ref.slice(separator + 1);
  const source = manifest.sources[sourceId];
  if (source === undefined) throw new Error(`licenseEvidence names unknown source "${sourceId}"`);

  const outPath = join(outputRoot, 'licenses', `${sourceId}__${member.replace(/[/\\]/g, '_')}`);
  const relPath = `licenses/${sourceId}__${member.replace(/[/\\]/g, '_')}`;

  const promise = (async () => {
    if (flags.dryRun) return relPath;
    let bytes;
    if (source.kind === 'npm') {
      const entries = await loadNpmPackage(source, cacheDir, { force: false });
      const data = entries.get(member);
      if (data === undefined) {
        throw new Error(`licence evidence "${member}" not present in ${source.package}`);
      }
      bytes = data;
    } else {
      bytes = await fetchBuffer(new URL(member, source.baseUrl).href);
    }
    await ensureDir(dirname(outPath));
    await writeFile(outPath, bytes);
    return relPath;
  })();

  evidenceCache.set(ref, promise);
  return promise;
}

/* -------------------------------------------------------------------------- */
/* Fetch one asset                                                             */
/* -------------------------------------------------------------------------- */

/** Resolve the canonical remote URL an asset came from, for the record. */
function resolveAssetUrl(asset, manifest) {
  if (asset.url !== undefined) return asset.url;
  const source = manifest.sources[asset.source];
  if (source === undefined) return null;
  if (source.kind === 'npm') {
    return `npm:${source.package}@${source.version}/${asset.member}`;
  }
  return new URL(asset.path, source.baseUrl).href;
}

/**
 * Bring one asset to its declared on-disk state.
 *
 * @returns a record describing what happened: `skipped`, `fetched` or `failed`.
 */
async function fetchAsset(asset, manifest, context) {
  const { outputRoot, cacheDir, flags, lock } = context;
  const outPath = join(outputRoot, asset.out);
  const url = resolveAssetUrl(asset, manifest);
  const locked = lock.assets?.[asset.key];

  /* --- fast path: already correct on disk ------------------------------- */
  if (!flags.force) {
    const size = await fileSize(outPath);
    if (size !== null && locked !== undefined && size === locked.bytes) {
      if (!flags.verify) {
        return { status: 'skipped', bytes: size, sha256: locked.sha256, url, outPath };
      }
      const hash = await sha256File(outPath);
      if (hash === locked.sha256) {
        return { status: 'skipped', bytes: size, sha256: hash, url, outPath, verified: true };
      }
      console.log(yellow(`      hash drift on disk — refetching`));
    }
  }

  if (flags.dryRun) {
    return { status: 'would-fetch', bytes: locked?.bytes ?? null, sha256: null, url, outPath };
  }

  /* --- produce the bytes ------------------------------------------------ */
  const source = asset.source === 'http' ? { kind: 'http' } : manifest.sources[asset.source];
  if (source === undefined) throw new Error(`asset "${asset.key}" names unknown source`);

  let bytes;
  if (source.kind === 'npm') {
    const entries = await loadNpmPackage(source, cacheDir, { force: flags.force });
    const raw = entries.get(asset.member);
    if (raw === undefined) {
      throw new Error(
        `member "${asset.member}" not found in ${source.package}@${source.version} ` +
          `(${entries.size} entries in tarball)`,
      );
    }
    bytes = raw;
  } else {
    // HTTP assets stream straight to their destination so the download is
    // resumable; transforms (which need the whole buffer) read it back.
    await downloadToFile(url, outPath);
    bytes = await readFile(outPath);
  }

  if (asset.transform !== undefined && asset.transform !== null) {
    const transform = TRANSFORMS[asset.transform];
    if (transform === undefined) throw new Error(`unknown transform "${asset.transform}"`);
    bytes = transform(bytes, asset.key);
  }

  await ensureDir(dirname(outPath));
  await writeFile(outPath, bytes);

  /* --- verify ----------------------------------------------------------- */
  const hash = sha256(bytes);
  if (typeof asset.sha256 === 'string' && asset.sha256 !== hash) {
    throw new Error(`sha256 pin mismatch: manifest expects ${asset.sha256}, got ${hash}`);
  }
  if (typeof asset.bytes === 'number' && asset.bytes !== bytes.length) {
    throw new Error(`size pin mismatch: manifest expects ${asset.bytes}, got ${bytes.length}`);
  }
  if (locked !== undefined && locked.sha256 !== hash && !flags.force) {
    console.log(
      yellow(
        `      warning: content changed upstream (lock ${locked.sha256.slice(0, 12)} -> ${hash.slice(0, 12)})`,
      ),
    );
  }

  return { status: 'fetched', bytes: bytes.length, sha256: hash, url, outPath };
}

/* -------------------------------------------------------------------------- */
/* Generated outputs                                                           */
/* -------------------------------------------------------------------------- */

function attributionRecord(asset, manifest, result, evidencePath) {
  const license = manifest.licenses[asset.license] ?? {};
  return {
    key: asset.key,
    title: asset.credit?.title ?? asset.key,
    author: asset.credit?.author ?? 'unknown',
    sourceUrl: asset.credit?.sourceUrl ?? result?.url ?? null,
    supportUrl: asset.credit?.supportUrl ?? null,
    retrievedFrom: result?.url ?? resolveAssetUrl(asset, manifest),
    license: {
      spdx: asset.license,
      name: license.name ?? asset.license,
      url: license.url ?? null,
      requiresAttribution: license.requiresAttribution ?? true,
      permitsCommercial: license.permitsCommercial ?? false,
      redistributable: license.redistributable ?? false,
      notes: license.notes ?? null,
    },
    licenseEvidence: evidencePath,
    tier: asset.tier,
    role: asset.role,
    localPath: `assets/${asset.out}`,
    bytes: result?.bytes ?? null,
    sha256: result?.sha256 ?? null,
    usage: asset.usage ?? null,
    present: result !== null && result.status !== 'would-fetch',
  };
}

function renderAttributionsMarkdown(records, manifest) {
  const lines = [];
  lines.push('# Asset attributions');
  lines.push('');
  lines.push(
    'This file is generated by `tools/assets/fetch-assets.mjs` from ' +
      '`tools/assets/manifest.json`. Do not edit it by hand — edit the manifest and re-run ' +
      '`npm run assets`.',
  );
  lines.push('');
  lines.push(
    'Every third-party asset used by d2rim is listed here with its exact source and licence. ' +
      'The file at source that documents each licence is archived under ' +
      '`assets/licenses/` so the claim can be verified offline.',
  );
  lines.push('');

  const present = records.filter((r) => r.tier === 'core');
  const review = records.filter((r) => r.tier === 'review-required');

  const byLicense = new Map();
  for (const record of present) {
    const list = byLicense.get(record.license.spdx) ?? [];
    list.push(record);
    byLicense.set(record.license.spdx, list);
  }

  lines.push('## Summary');
  lines.push('');
  lines.push('| Licence | Assets | Attribution required |');
  lines.push('| --- | --- | --- |');
  for (const [spdx, list] of byLicense) {
    const license = manifest.licenses[spdx] ?? {};
    lines.push(
      `| ${spdx} | ${list.length} | ${license.requiresAttribution === true ? 'yes' : 'no'} |`,
    );
  }
  lines.push('');

  for (const [spdx, list] of byLicense) {
    const license = manifest.licenses[spdx] ?? {};
    lines.push(`## ${license.name ?? spdx}`);
    lines.push('');
    lines.push(`SPDX identifier: \`${spdx}\`${license.url ? ` — <${license.url}>` : ''}`);
    lines.push('');
    if (license.requiresAttribution === true) {
      lines.push(
        '> This licence **requires attribution**. The credits below must be reproduced ' +
          'in any distributed build.',
      );
      lines.push('');
    }
    for (const record of list) {
      lines.push(`### ${record.title}`);
      lines.push('');
      lines.push(`- **Semantic key:** \`${record.key}\``);
      lines.push(`- **Author:** ${record.author}`);
      lines.push(`- **Source:** ${record.sourceUrl ?? '—'}`);
      lines.push(`- **Retrieved from:** \`${record.retrievedFrom}\``);
      lines.push(`- **Local path:** \`${record.localPath}\``);
      if (record.bytes !== null) {
        lines.push(`- **Size:** ${humanBytes(record.bytes)} (${record.bytes} bytes)`);
      }
      if (record.sha256 !== null) lines.push(`- **SHA-256:** \`${record.sha256}\``);
      if (record.licenseEvidence !== null) {
        lines.push(`- **Licence evidence:** \`${record.licenseEvidence}\``);
      }
      if (record.supportUrl !== null) lines.push(`- **Support the creator:** ${record.supportUrl}`);
      if (record.usage !== null) lines.push(`- **Used for:** ${record.usage}`);
      lines.push('');
    }
  }

  if (review.length > 0) {
    lines.push('## Held back pending licence review');
    lines.push('');
    lines.push(
      'These assets are reachable and technically useful but their redistribution rights are ' +
        '**unresolved**. They are not downloaded by a default `npm run assets` run, and they ' +
        'must not appear in a public build until a human has made a licensing decision. ' +
        'Fetch them for local evaluation with `--include-review-required`.',
    );
    lines.push('');
    for (const record of review) {
      lines.push(`### ${record.title} — NOT CLEARED`);
      lines.push('');
      lines.push(`- **Semantic key:** \`${record.key}\``);
      lines.push(`- **Author / origin:** ${record.author}`);
      lines.push(`- **Source:** ${record.sourceUrl ?? '—'}`);
      lines.push(`- **Declared licence:** ${record.license.spdx} — ${record.license.name}`);
      lines.push(`- **Present locally:** ${record.present ? 'yes' : 'no'}`);
      if (record.license.notes !== null) lines.push(`- **Why unresolved:** ${record.license.notes}`);
      if (record.usage !== null) lines.push(`- **Would be used for:** ${record.usage}`);
      lines.push('');
    }
  }

  lines.push('---');
  lines.push('');
  lines.push(`Generated ${new Date().toISOString()} by \`tools/assets/fetch-assets.mjs\`.`);
  lines.push('');
  return lines.join('\n');
}

/**
 * Emit the typed registry other modules import.
 *
 * Generating this rather than hand-maintaining it is what makes semantic keys
 * trustworthy: the key set, the on-disk paths and the licence metadata cannot
 * drift from the manifest, because they are the manifest.
 */
function renderRegistryModule(records) {
  const lines = [];
  lines.push('/**');
  lines.push(' * @module assets/registry.generated');
  lines.push(' *');
  lines.push(' * GENERATED FILE — do not edit.');
  lines.push(' *');
  lines.push(' * Produced by `tools/assets/fetch-assets.mjs` from `tools/assets/manifest.json`.');
  lines.push(' * Regenerate with `npm run assets`. `tests/assets.registry.test.ts` fails if this');
  lines.push(' * file and the manifest disagree, so the two cannot silently drift apart.');
  lines.push(' *');
  lines.push(' * Paths are relative to the served root and are resolved against');
  lines.push(' * `import.meta.env.BASE_URL` at load time, so the game still works when it is');
  lines.push(' * deployed under a sub-path.');
  lines.push(' */');
  lines.push('');
  lines.push("import type { AssetRole, AssetTier } from './types';");
  lines.push('');
  lines.push('/** One entry in the generated asset registry. */');
  lines.push('export interface GeneratedAssetEntry {');
  lines.push('  /** Path relative to the served root, e.g. `assets/hdri/overcast.exr`. */');
  lines.push('  readonly path: string;');
  lines.push('  /** Drives colour space and filtering decisions in the AssetManager. */');
  lines.push('  readonly role: AssetRole;');
  lines.push('  /** `review-required` assets are absent unless explicitly fetched. */');
  lines.push('  readonly tier: AssetTier;');
  lines.push('  /** SPDX identifier of the licence this asset ships under. */');
  lines.push('  readonly license: string;');
  lines.push('  /** Byte length recorded at fetch time; `null` when never fetched. */');
  lines.push('  readonly bytes: number | null;');
  lines.push('  /** SHA-256 recorded at fetch time; `null` when never fetched. */');
  lines.push('  readonly sha256: string | null;');
  lines.push('}');
  lines.push('');
  lines.push('export const GENERATED_ASSETS = {');
  for (const record of records) {
    lines.push(`  '${record.key}': {`);
    lines.push(`    path: '${record.localPath}',`);
    lines.push(`    role: '${record.role}',`);
    lines.push(`    tier: '${record.tier}',`);
    lines.push(`    license: '${record.license.spdx}',`);
    lines.push(`    bytes: ${record.bytes === null ? 'null' : record.bytes},`);
    lines.push(`    sha256: ${record.sha256 === null ? 'null' : `'${record.sha256}'`},`);
    lines.push('  },');
  }
  lines.push('} as const satisfies Record<string, GeneratedAssetEntry>;');
  lines.push('');
  lines.push('/** Every semantic asset key known to the build. */');
  lines.push('export type GeneratedAssetKey = keyof typeof GENERATED_ASSETS;');
  lines.push('');
  return lines.join('\n');
}

/* -------------------------------------------------------------------------- */
/* Main                                                                        */
/* -------------------------------------------------------------------------- */

async function main() {
  const flags = parseArgs(process.argv.slice(2));
  const manifest = await readJson(MANIFEST_PATH);
  const lock = await readJson(LOCK_PATH, { version: 1, assets: {} });

  const outputRoot = join(ROOT, manifest.outputDir);
  const cacheDir = join(ROOT, manifest.cacheDir);

  const selected = manifest.assets.filter((asset) => {
    if (flags.only !== null && !asset.key.includes(flags.only)) return false;
    if (asset.tier === 'review-required' && !flags.includeReviewRequired) return false;
    return true;
  });

  console.log(cyan('d2rim asset fetcher'));
  console.log(
    dim(
      `  manifest ${manifest.assets.length} assets — ${selected.length} selected` +
        `${flags.only !== null ? ` (--only=${flags.only})` : ''}` +
        `${flags.includeReviewRequired ? ' (including review-required)' : ''}` +
        `${flags.dryRun ? ' [dry run]' : ''}`,
    ),
  );
  console.log(dim(`  output   ${manifest.outputDir}`));
  console.log('');

  const results = new Map();
  const failures = [];

  for (const asset of selected) {
    const label = `${asset.key}`.padEnd(24);
    try {
      const result = await fetchAsset(asset, manifest, { outputRoot, cacheDir, flags, lock });
      results.set(asset.key, result);

      const marks = {
        fetched: green('fetched '),
        skipped: dim('cached  '),
        'would-fetch': yellow('pending '),
      };
      const note =
        result.status === 'skipped' && result.verified === true ? dim(' (hash verified)') : '';
      console.log(
        `  ${marks[result.status]} ${label} ${humanBytes(result.bytes).padStart(9)}  ` +
          `${dim(asset.license)}${note}`,
      );
    } catch (error) {
      failures.push({ key: asset.key, error });
      console.log(`  ${red('FAILED  ')} ${label} ${red(error.message)}`);
    }
  }

  /* --- licence evidence -------------------------------------------------- */
  console.log('');
  const evidencePaths = new Map();
  for (const asset of selected) {
    if (asset.licenseEvidence === undefined) continue;
    try {
      const path = await archiveLicenseEvidence(
        asset.licenseEvidence,
        manifest,
        outputRoot,
        cacheDir,
        flags,
      );
      evidencePaths.set(asset.key, path);
    } catch (error) {
      failures.push({ key: `${asset.key} (licence evidence)`, error });
      console.log(`  ${red('FAILED  ')} licence evidence for ${asset.key}: ${error.message}`);
    }
  }
  if (evidencePaths.size > 0) {
    console.log(dim(`  archived ${evidenceCache.size} licence evidence file(s)`));
  }

  /* --- generated outputs ------------------------------------------------- */
  // Records cover EVERY manifest asset, not just the selected ones, so the
  // generated key space is stable regardless of which flags a run used. Assets
  // that are not on disk carry null size/hash and the AssetManager reports a
  // precise error if something tries to load one.
  const records = manifest.assets.map((asset) =>
    attributionRecord(asset, manifest, results.get(asset.key) ?? null, evidencePaths.get(asset.key) ?? null),
  );

  if (!flags.dryRun) {
    // Preserve lock entries for assets this run did not touch (e.g. --only).
    const nextLock = { version: 1, generatedAt: new Date().toISOString(), assets: { ...lock.assets } };
    for (const [key, result] of results) {
      if (result.status === 'would-fetch') continue;
      nextLock.assets[key] = {
        url: result.url,
        out: manifest.assets.find((a) => a.key === key)?.out ?? null,
        bytes: result.bytes,
        sha256: result.sha256,
        fetchedAt: nextLock.assets[key]?.fetchedAt ?? new Date().toISOString(),
      };
    }
    await writeFile(LOCK_PATH, `${JSON.stringify(nextLock, null, 2)}\n`);

    const attributionsJson = {
      generatedAt: new Date().toISOString(),
      generator: 'tools/assets/fetch-assets.mjs',
      project: 'd2rim',
      licenses: manifest.licenses,
      assets: records,
    };
    await ensureDir(join(ROOT, 'public'));
    await writeFile(
      join(ROOT, manifest.generated.attributionsJson),
      `${JSON.stringify(attributionsJson, null, 2)}\n`,
    );
    await writeFile(
      join(ROOT, manifest.generated.attributionsMd),
      renderAttributionsMarkdown(records, manifest),
    );
    await ensureDir(dirname(join(ROOT, manifest.generated.registry)));
    await writeFile(join(ROOT, manifest.generated.registry), renderRegistryModule(records));

    console.log(
      dim(
        `  wrote ${manifest.generated.attributionsMd}, ${manifest.generated.attributionsJson}, ` +
          `${manifest.generated.registry}, tools/assets/assets.lock.json`,
      ),
    );
  }

  /* --- summary ----------------------------------------------------------- */
  const fetched = [...results.values()].filter((r) => r.status === 'fetched');
  const cached = [...results.values()].filter((r) => r.status === 'skipped');
  const totalBytes = [...results.values()].reduce((sum, r) => sum + (r.bytes ?? 0), 0);

  console.log('');
  console.log(
    `  ${green(`${fetched.length} fetched`)}, ${dim(`${cached.length} cached`)}, ` +
      `${failures.length > 0 ? red(`${failures.length} failed`) : '0 failed'} — ` +
      `${humanBytes(totalBytes)} on disk`,
  );

  if (failures.length > 0) {
    console.log('');
    console.log(red('Failures:'));
    for (const { key, error } of failures) console.log(red(`  ${key}: ${error.message}`));
    process.exitCode = 1;
  }
}

main().catch((error) => {
  console.error(red(`\nfetch-assets failed: ${error.message}`));
  if (error.cause !== undefined) console.error(error.cause);
  process.exitCode = 1;
});
