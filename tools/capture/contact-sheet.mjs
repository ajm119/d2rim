/**
 * @module tools/capture/contact-sheet
 *
 * Composites a set of captures into one labelled grid image.
 *
 * ### Why this exists
 *
 * A critic agent reviewing the game's visuals pays a full image-read per file.
 * Seven shots is seven reads and seven separate impressions, which is both
 * expensive and worse at the actual job — most visual regressions (a colour
 * space flip, a lighting change, one shot that went flat) are obvious side by
 * side and invisible one at a time. A contact sheet turns the whole capture run
 * into a single glance.
 *
 * ### Layout rules
 *
 * - Images are letterboxed (`fit: contain`), never cropped. A contact sheet that
 *   silently crops is worse than useless: the critic grades a composition that
 *   does not exist.
 * - Every cell carries its shot id, description and pixel statistics, so a note
 *   like "cell 4 is too dark" maps back to a file without guesswork.
 * - Failed shots (per `report.json`) are banded in red rather than dropped, so
 *   a missing shot is visible instead of being silently absent.
 * - The finished sheet is itself put through the blank-frame guard.
 *
 * ### Usage
 *
 *   node tools/capture/contact-sheet.mjs --in captures/shots --out captures/contact-sheet.png
 *   node tools/capture/contact-sheet.mjs a.png b.png --out sheet.png --cols 2
 */

import { existsSync } from 'node:fs';
import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { basename, dirname, isAbsolute, join, relative, resolve } from 'node:path';

import sharp from 'sharp';

import { formatHelp, parseArgs, ROOT } from './cli.mjs';
import { analyzeImageFile, evaluateFrame, formatStats } from './frame-guard.mjs';

const OPTIONS = {
  in: { type: 'string', default: 'captures/shots', help: 'directory of PNGs to composite' },
  out: { type: 'string', default: 'captures/contact-sheet.png', help: 'output image path' },
  cols: { type: 'number', default: 3, help: 'grid columns' },
  'cell-width': { type: 'number', default: 620, help: 'width of each cell in pixels' },
  title: { type: 'string', default: 'd2rim capture contact sheet', help: 'header title' },
  help: { type: 'boolean', default: false, help: 'show this help' },
};

/** Dark neutral palette: the sheet must not bias judgement of the frames in it. */
const THEME = Object.freeze({
  page: { r: 14, g: 16, b: 20, alpha: 1 },
  letterbox: { r: 8, g: 9, b: 11, alpha: 1 },
  margin: 26,
  gap: 20,
  headerHeight: 74,
  labelHeight: 74,
  text: '#e6e9ef',
  muted: '#8b94a3',
  accent: '#c0563a',
  fail: '#ff6b6b',
});

/** Escape text for inclusion in SVG markup. */
function xml(text) {
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/**
 * Greedy word wrap using an average-glyph-width estimate.
 *
 * Exact text metrics would need a font-shaping pass; for a label strip a good
 * estimate that never overflows is worth more than precision.
 */
function wrap(text, maxWidth, fontSize, maxLines) {
  const perChar = fontSize * 0.52;
  const limit = Math.max(8, Math.floor(maxWidth / perChar));
  const words = String(text).split(/\s+/).filter(Boolean);
  const lines = [];
  let current = '';

  for (const word of words) {
    const candidate = current === '' ? word : `${current} ${word}`;
    if (candidate.length <= limit) {
      current = candidate;
    } else {
      if (current !== '') lines.push(current);
      current = word;
      if (lines.length === maxLines) break;
    }
  }
  if (current !== '' && lines.length < maxLines) lines.push(current);

  if (lines.length === maxLines && words.join(' ').length > lines.join(' ').length) {
    const last = lines[maxLines - 1];
    lines[maxLines - 1] = `${last.slice(0, Math.max(0, limit - 1))}…`;
  }
  return lines;
}

/**
 * Clip a single line to the cell width.
 *
 * SVG text does not wrap or clip on its own, so an over-long stats line simply
 * runs out past the label plate and over the neighbouring cell. `charRatio` is
 * the advance width as a fraction of font size — monospace runs wider than the
 * proportional face used elsewhere.
 */
function truncate(text, maxWidth, fontSize, charRatio = 0.6) {
  const limit = Math.max(4, Math.floor(maxWidth / (fontSize * charRatio)));
  return text.length <= limit ? text : `${text.slice(0, limit - 1)}…`;
}

/**
 * Resolve which images go on the sheet, in which order, with what labels.
 *
 * Declaration order from `report.json` wins over filesystem order: a shot list
 * is authored as a sequence (wide, then detail, then lighting), and alphabetical
 * order would scramble that narrative.
 */
async function collectEntries({ inputDir, positionals, outPath }) {
  if (positionals.length > 0) {
    return positionals.map((file) => ({
      file: isAbsolute(file) ? file : resolve(ROOT, file),
      id: basename(file).replace(/\.png$/i, ''),
      description: '',
      ok: true,
      stats: null,
    }));
  }

  const reportPath = join(inputDir, 'report.json');
  if (existsSync(reportPath)) {
    const report = JSON.parse(await readFile(reportPath, 'utf8'));
    const entries = [];
    for (const shot of report.shots ?? []) {
      const file = join(inputDir, `${shot.id}.png`);
      if (!existsSync(file)) continue;
      entries.push({
        file,
        id: shot.id,
        description: shot.description ?? '',
        ok: shot.ok !== false,
        stats: shot.stats ?? null,
      });
    }
    if (entries.length > 0) return entries;
  }

  const files = (await readdir(inputDir))
    .filter((name) => /\.png$/i.test(name))
    .filter((name) => join(inputDir, name) !== outPath)
    .sort();
  return files.map((name) => ({
    file: join(inputDir, name),
    id: name.replace(/\.png$/i, ''),
    description: '',
    ok: true,
    stats: null,
  }));
}

async function main() {
  const { options, positionals } = parseArgs(process.argv.slice(2), OPTIONS);
  if (options.help) {
    console.log('Composite d2rim captures into one labelled grid image.\n');
    console.log('Usage: node tools/capture/contact-sheet.mjs --in captures/shots --out sheet.png\n');
    console.log(formatHelp(OPTIONS));
    return 0;
  }

  const inputDir = isAbsolute(options.in) ? options.in : resolve(ROOT, options.in);
  const outPath = isAbsolute(options.out) ? options.out : resolve(ROOT, options.out);

  const entries = await collectEntries({ inputDir, positionals, outPath });
  if (entries.length === 0) {
    throw new Error(`no PNGs found in ${relative(ROOT, inputDir)} - run capture.mjs first`);
  }

  // The cell aspect follows the first image so the common case (a run of shots
  // at one resolution) letterboxes nothing at all.
  const first = await sharp(entries[0].file).metadata();
  const aspect = first.width / first.height;

  const cols = Math.max(1, Math.min(options.cols, entries.length));
  const rows = Math.ceil(entries.length / cols);
  const cellWidth = Math.max(160, Math.round(options['cell-width']));
  const imageHeight = Math.round(cellWidth / aspect);
  const cellHeight = imageHeight + THEME.labelHeight;

  const width = THEME.margin * 2 + cols * cellWidth + (cols - 1) * THEME.gap;
  const height =
    THEME.margin * 2 + THEME.headerHeight + rows * cellHeight + (rows - 1) * THEME.gap;

  console.log(`contact sheet: ${entries.length} shot(s), ${cols}x${rows}, ${width}x${height}`);

  const composites = [];
  const svgParts = [];

  svgParts.push(
    `<text x="${THEME.margin}" y="${THEME.margin + 26}" fill="${THEME.text}" ` +
      `font-family="sans-serif" font-size="24" font-weight="600">${xml(options.title)}</text>`,
    `<text x="${THEME.margin}" y="${THEME.margin + 50}" fill="${THEME.muted}" ` +
      `font-family="sans-serif" font-size="13">${xml(
        `${entries.length} shot(s) from ${relative(ROOT, inputDir)}  ·  ${new Date().toISOString()}`,
      )}</text>`,
  );

  for (const [index, entry] of entries.entries()) {
    const col = index % cols;
    const row = Math.floor(index / cols);
    const left = THEME.margin + col * (cellWidth + THEME.gap);
    const top = THEME.margin + THEME.headerHeight + row * (cellHeight + THEME.gap);

    const resized = await sharp(entry.file)
      .resize(cellWidth, imageHeight, { fit: 'contain', background: THEME.letterbox })
      .png()
      .toBuffer();
    composites.push({ input: resized, left, top });

    const labelTop = top + imageHeight;
    const statusColor = entry.ok ? THEME.accent : THEME.fail;

    // Label plate, plus a status bar down the left edge that reads at a glance.
    svgParts.push(
      `<rect x="${left}" y="${labelTop}" width="${cellWidth}" height="${THEME.labelHeight}" fill="#161a20"/>`,
      `<rect x="${left}" y="${labelTop}" width="4" height="${THEME.labelHeight}" fill="${statusColor}"/>`,
      `<text x="${left + 14}" y="${labelTop + 21}" fill="${THEME.text}" font-family="sans-serif" ` +
        `font-size="15" font-weight="600">${xml(`${index + 1}. ${entry.id}`)}${
          entry.ok ? '' : `<tspan fill="${THEME.fail}" font-size="13" dx="8">GUARD FAILED</tspan>`
        }</text>`,
    );

    const descriptionLines = wrap(entry.description, cellWidth - 28, 12.5, 2);
    descriptionLines.forEach((line, lineIndex) => {
      svgParts.push(
        `<text x="${left + 14}" y="${labelTop + 39 + lineIndex * 15}" fill="${THEME.muted}" ` +
          `font-family="sans-serif" font-size="12.5">${xml(line)}</text>`,
      );
    });

    const detail =
      entry.stats === null
        ? `${first.width}x${first.height}`
        : formatStats(entry.stats).replace(/\s{2,}/g, '  ');
    svgParts.push(
      `<text x="${left + 14}" y="${labelTop + THEME.labelHeight - 9}" fill="#6f7a8a" ` +
        `font-family="monospace" font-size="11">${xml(truncate(detail, cellWidth - 28, 11))}</text>`,
    );
  }

  const overlay = Buffer.from(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">${svgParts.join('')}</svg>`,
  );
  composites.push({ input: overlay, left: 0, top: 0 });

  const sheet = await sharp({
    create: { width, height, channels: 4, background: THEME.page },
  })
    .composite(composites)
    .png()
    .toBuffer();

  await mkdir(dirname(outPath), { recursive: true });
  await writeFile(outPath, sheet);

  // The sheet is a capture artifact too, and an all-black sheet built from good
  // frames (a composite that silently no-oped) would be just as misleading.
  // Decoded back off disk for the same reason capture.mjs does it: this also
  // proves the encode and the write.
  const stats = await analyzeImageFile(outPath);
  const verdict = evaluateFrame(stats);

  console.log(`  wrote ${relative(ROOT, outPath)}`);
  console.log(`  ${formatStats(stats)}`);

  if (!verdict.ok) {
    console.error('\nCONTACT SHEET IS DEGENERATE:');
    for (const reason of verdict.failures) console.error(`  ! ${reason}`);
    return 1;
  }

  const failedShots = entries.filter((entry) => !entry.ok);
  if (failedShots.length > 0) {
    console.log(
      `  note: ${failedShots.length} shot(s) are marked as guard failures in report.json ` +
        `(${failedShots.map((entry) => entry.id).join(', ')})`,
    );
  }
  return 0;
}

try {
  process.exitCode = await main();
} catch (error) {
  console.error(`\n${basename(process.argv[1])}: ${error.message}`);
  if (process.env.DEBUG) console.error(error.stack);
  process.exitCode = 1;
}
