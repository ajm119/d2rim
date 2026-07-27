/**
 * @module tools/capture/cli
 *
 * Shared plumbing for the capture tools: argument parsing, Chromium discovery,
 * and the dev-server lifecycle.
 *
 * Kept separate so `capture.mjs` reads as the capture pipeline and nothing else,
 * and so a future tool (frame diffing, video capture) inherits the same proven
 * browser flags rather than copying a subtly different set.
 */

import { spawn } from 'node:child_process';
import { existsSync, statSync } from 'node:fs';
import { readdir } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

/** Repository root, derived from this file's location. */
export const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..');

/**
 * Chromium flags proven to work in this container by the GPU probe.
 *
 * `--enable-unsafe-webgpu` is the decisive one: without it
 * `navigator.gpu.requestAdapter()` returns null and the WebGPU path silently
 * becomes the WebGL2 path. The ANGLE flags pin software rasterisation
 * explicitly rather than relying on it being the default, so a capture run is
 * reproducible if the container image ever changes.
 */
export const CHROMIUM_ARGS = Object.freeze([
  '--no-sandbox',
  '--disable-dev-shm-usage',
  '--enable-unsafe-webgpu',
  '--use-gl=angle',
  '--use-angle=swiftshader',
]);

/**
 * Locate the pre-installed Chromium.
 *
 * The bundled browser's build number does not always match what the installed
 * Playwright expects, and `playwright install` must never run here, so the
 * binary is located explicitly.
 *
 * @returns {string} absolute path to the Chromium executable
 */
export function findChromium() {
  const candidates = [
    process.env.CHROMIUM_PATH,
    '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
    '/opt/pw-browsers/chromium/chrome-linux/chrome',
  ].filter(Boolean);

  const found = candidates.find((path) => existsSync(path));
  if (found === undefined) {
    throw new Error(
      `no Chromium binary found. Tried:\n  ${candidates.join('\n  ')}\n` +
        'Set CHROMIUM_PATH to override. Never run `playwright install` in this container.',
    );
  }
  return found;
}

/**
 * Minimal argv parser.
 *
 * Supports `--key=value`, `--key value`, bare `--flag` (true) and `--no-flag`
 * (false). Values are coerced to numbers when the spec says so, and unknown
 * keys are rejected — a typo'd `--warmup` that silently did nothing would be a
 * capture run that quietly measured the wrong thing.
 *
 * @param {string[]} argv
 * @param {Record<string, {type: 'string'|'number'|'boolean', default?: unknown, help: string}>} spec
 * @returns {{options: Record<string, unknown>, positionals: string[]}}
 */
export function parseArgs(argv, spec) {
  const options = {};
  for (const [key, definition] of Object.entries(spec)) {
    if (definition.default !== undefined) options[key] = definition.default;
  }
  const positionals = [];

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (!arg.startsWith('--')) {
      positionals.push(arg);
      continue;
    }

    let key = arg.slice(2);
    let value;
    const equals = key.indexOf('=');
    if (equals !== -1) {
      value = key.slice(equals + 1);
      key = key.slice(0, equals);
    }

    let negated = false;
    if (!(key in spec) && key.startsWith('no-') && key.slice(3) in spec) {
      key = key.slice(3);
      negated = true;
    }

    const definition = spec[key];
    if (definition === undefined) {
      throw new Error(`unknown option --${key}\n\n${formatHelp(spec)}`);
    }

    if (definition.type === 'boolean') {
      options[key] = value === undefined ? !negated : value !== 'false' && value !== '0';
      continue;
    }

    if (value === undefined) {
      value = argv[++i];
      if (value === undefined || value.startsWith('--')) {
        throw new Error(`option --${key} expects a value`);
      }
    }

    if (definition.type === 'number') {
      const parsed = Number(value);
      if (!Number.isFinite(parsed)) throw new Error(`option --${key} expects a number, got "${value}"`);
      options[key] = parsed;
    } else {
      options[key] = value;
    }
  }

  return { options, positionals };
}

/** Render an option spec as an aligned help block. */
export function formatHelp(spec) {
  const rows = Object.entries(spec).map(([key, definition]) => {
    const suffix = definition.type === 'boolean' ? '' : ` <${definition.type}>`;
    const shown = definition.default === undefined ? '' : ` (default: ${definition.default})`;
    return [`  --${key}${suffix}`, `${definition.help}${shown}`];
  });
  const width = Math.max(...rows.map(([left]) => left.length));
  return ['Options:', ...rows.map(([left, right]) => `${left.padEnd(width + 2)}${right}`)].join('\n');
}

/**
 * Newest modification time under `dir`, recursively.
 *
 * @param {string} dir
 * @returns {Promise<number>} epoch milliseconds, 0 if the directory is absent
 */
async function newestMtime(dir) {
  if (!existsSync(dir)) return 0;
  let newest = 0;
  const entries = await readdir(dir, { withFileTypes: true });
  for (const entry of entries) {
    const path = join(dir, entry.name);
    const mtime = entry.isDirectory() ? await newestMtime(path) : statSync(path).mtimeMs;
    if (mtime > newest) newest = mtime;
  }
  return newest;
}

/**
 * Whether `dist/` is missing or older than the sources that produce it.
 *
 * Capturing a stale build is a silent lie: the harness reports success, the
 * critic grades pixels from an hour ago, and the loop congratulates itself on a
 * change that was never compiled.
 *
 * @returns {Promise<{stale: boolean, reason: string}>}
 */
export async function isBuildStale() {
  const distIndex = join(ROOT, 'dist/index.html');
  if (!existsSync(distIndex)) return { stale: true, reason: 'dist/index.html does not exist' };

  const builtAt = statSync(distIndex).mtimeMs;
  const sourceAt = Math.max(
    await newestMtime(join(ROOT, 'src')),
    statSync(join(ROOT, 'index.html')).mtimeMs,
    statSync(join(ROOT, 'vite.config.ts')).mtimeMs,
  );

  return sourceAt > builtAt
    ? { stale: true, reason: `sources are newer than dist/ (by ${((sourceAt - builtAt) / 1000).toFixed(0)}s)` }
    : { stale: false, reason: 'dist/ is up to date' };
}

/**
 * Run a command to completion, streaming its output.
 *
 * @param {string} command
 * @param {string[]} args
 * @returns {Promise<void>} rejects on a nonzero exit code
 */
export function run(command, args) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(command, args, { cwd: ROOT, stdio: 'inherit' });
    child.on('error', reject);
    child.on('exit', (code) => {
      if (code === 0) resolvePromise();
      else reject(new Error(`${command} ${args.join(' ')} exited with code ${code}`));
    });
  });
}

/** Build the production bundle with Vite. */
export async function buildBundle() {
  await run(process.execPath, [join(ROOT, 'node_modules/vite/bin/vite.js'), 'build']);
}

/**
 * @typedef {object} ServerHandle
 * @property {string} url        origin the server is reachable at
 * @property {() => void} stop   idempotent shutdown
 */

/**
 * Start `vite preview` and resolve only once it actually answers a request.
 *
 * Polling for a real HTTP response rather than parsing stdout means the caller
 * never navigates to a socket that is still binding.
 *
 * @param {number} port
 * @param {number} [timeoutMs]
 * @returns {Promise<ServerHandle>}
 */
export async function startPreviewServer(port, timeoutMs = 30_000) {
  const child = spawn(
    process.execPath,
    [join(ROOT, 'node_modules/vite/bin/vite.js'), 'preview', '--port', String(port), '--strictPort'],
    { cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe'] },
  );

  let stderr = '';
  child.stderr.on('data', (chunk) => {
    stderr += String(chunk);
  });
  child.stdout.resume();

  const url = `http://127.0.0.1:${port}`;
  const stop = () => {
    if (!child.killed) child.kill('SIGTERM');
  };

  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (child.exitCode !== null) {
      throw new Error(`vite preview exited early (code ${child.exitCode}):\n${stderr}`);
    }
    try {
      const response = await fetch(url, { signal: AbortSignal.timeout(1000) });
      if (response.ok) return { url, stop };
    } catch {
      // Not listening yet.
    }
    await delay(200);
  }

  stop();
  throw new Error(`vite preview did not become reachable at ${url} within ${timeoutMs}ms\n${stderr}`);
}

/** Promise-based sleep. */
export function delay(ms) {
  return new Promise((resolvePromise) => setTimeout(resolvePromise, ms));
}

/** Seconds, one decimal, for progress lines. */
export function since(startedAt) {
  return `${((Date.now() - startedAt) / 1000).toFixed(1)}s`;
}
