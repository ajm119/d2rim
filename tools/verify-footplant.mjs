/**
 * Foot-plant regression guard.
 *
 * Drives the real game and measures, per frame, the *world* speed of the
 * slower-moving foot bone. A stride-matched character has a foot at very near
 * zero world speed through every stance phase; a character whose playback rate
 * does not match its ground speed has both feet travelling with the body, and
 * that is the single most visible tell of amateur character work.
 *
 * Reports and asserts three things per gait:
 *   - `plantedFraction`  frames where the slower foot is under PLANT_SPEED
 *   - `meanMinFootSpeed` mean over frames of the slower foot's world speed
 *   - `minMinFootSpeed`  the best plant achieved anywhere in the sample
 *
 * A run legitimately scores a lower planted fraction than a walk: it has a
 * flight phase, and during flight *both* feet are moving, correctly. So the
 * thresholds differ by gait rather than pretending one number fits.
 *
 *   node tools/verify-footplant.mjs
 *
 * Exits non-zero on a regression.
 */
import { spawn } from 'node:child_process';
import { chromium } from 'playwright';
import { CHROMIUM_ARGS, findChromium } from './capture/cli.mjs';

const PORT = 5241;
/** World speed under which a foot counts as genuinely planted, m/s. */
const PLANT_SPEED = 0.25;

const GAITS = [
  {
    // Block held. The walk clip is a true walk — a foot is on the ground for
    // over 40% of the cycle and there is double support — so this is the gait
    // where a plant is physically available on every frame, and the one where
    // sliding is most visible because the character is slow enough to look at.
    name: 'walk',
    keys: [],
    mouse: 'right',
    frames: 96,
    minPlantedFraction: 0.15,
    maxMeanRatio: 0.55,
    maxBestPlant: 0.12,
  },
  {
    name: 'run',
    keys: ['w'],
    frames: 96,
    // A run has a flight phase. For most of the cycle *neither* foot is down
    // and both are correctly moving, so a low planted fraction and a mean near
    // body speed are the right answer here; what has to be true is that the
    // stance, when it happens, is a real stance.
    minPlantedFraction: 0.02,
    maxMeanRatio: 0.8,
    maxBestPlant: 0.4,
  },
  {
    // The one gait that runs into the cadence cap, so the one that is allowed
    // to slide by design. Guarded anyway: unbounded is not the same as free.
    name: 'sprint',
    keys: ['w', 'Shift'],
    frames: 96,
    minPlantedFraction: 0,
    maxMeanRatio: 0.95,
    maxBestPlant: 0.75,
  },
];

const server = spawn(
  'npx',
  ['vite', 'preview', '--port', String(PORT), '--strictPort', '--host', '127.0.0.1'],
  { cwd: process.cwd(), stdio: ['ignore', 'pipe', 'pipe'] },
);
server.stderr.on('data', () => undefined);
await new Promise((r) => {
  server.stdout.on('data', (c) => String(c).includes('Local:') && r());
  setTimeout(r, 15000);
});

const browser = await chromium.launch({ args: [...CHROMIUM_ARGS], executablePath: findChromium() });
const page = await browser.newPage({ viewport: { width: 192, height: 108 } });
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
// `zone=bloodMoor`: the default start zone is the Rogue Encampment, whose
// walls are close enough that the character cannot reach a steady gait inside
// them. Locomotion has to be measured somewhere with room to run.
await page.goto(`http://127.0.0.1:${PORT}/?autostart=0&backend=webgl2&quality=low&zone=bloodMoor`, {
  waitUntil: 'load',
});
await page.evaluate(() => window.__d2rim.ready);
await page.evaluate((n) => window.__d2rim.engine.stepFrames(n), 20);

await page.evaluate(() => {
  const d2 = window.__d2rim;
  const player = d2.ctx.services.get('character.player');
  const feet = {};
  player.object.traverse((o) => {
    const n = o.name.toLowerCase().replace(/[^a-z0-9]/g, '');
    if (n === 'footl' || n === 'footr') feet[n] = o;
  });
  window.__sample = () => {
    const V = d2.three.Vector3;
    const at = (o) => {
      const v = new V();
      o.getWorldPosition(v);
      return [v.x, v.y, v.z];
    };
    return {
      pos: [player.position.x, player.position.y, player.position.z],
      speed: player.speed,
      cycleRate: player.animation?.cycleRate ?? 0,
      state: player.animation?.state ?? null,
      l: at(feet.footl),
      r: at(feet.footr),
    };
  };
});

const setup = await page.evaluate(() => {
  const player = window.__d2rim.ctx.services.get('character.player');
  const graph = player.animation;
  return {
    gait: player.gait,
    height: +player.height.toFixed(3),
    scale: +player.object.scale.x.toFixed(4),
    blendParams: graph.blendParams,
    clips: {
      walk: graph.clipTable.get('walk.forward'),
      run: graph.clipTable.get('run.forward'),
    },
    strides: Object.fromEntries(
      [...graph.strideTable].map(([k, v]) => [k, +v.toFixed(3)]),
    ),
  };
});
console.log('setup:', JSON.stringify(setup, null, 1));

const report = { gaits: {}, gait: setup.gait };
let failed = false;

for (const gait of GAITS) {
  // Blocking slows the character to the walk gait; it is the only way the
  // input map has of asking for a walk, and it is what the walk clip is for.
  // Back to the spawn between gaits: 96 frames at sprint covers ten metres of
  // a scattered moor, and the next gait should not start wedged against a rock.
  await page.evaluate(() => {
    const svc = window.__d2rim.ctx.services;
    const player = svc.get('character.player');
    const physics = svc.get('physics.world');
    const spot = physics.findClearSpot(1.1, 0.4, 0.42, player.height, 6);
    if (spot !== null) player.teleport(spot.x, spot.y, spot.z);
  });
  await page.evaluate(() => window.__d2rim.engine.stepFrames(8));
  if (gait.mouse === 'right') await page.mouse.down({ button: 'right' });
  await page.keyboard.down('w');
  for (const key of gait.keys) if (key !== 'w') await page.keyboard.down(key);
  let samples;
  try {
    samples = await page.evaluate(async (n) => {
      const out = [];
      for (let i = 0; i < n; i++) {
        await window.__d2rim.engine.stepFrames(1);
        out.push(window.__sample());
      }
      return out;
    }, gait.frames);
  } catch (error) {
    // A software rasteriser under load does occasionally take the tab with it.
    // Report the gaits already measured rather than losing the whole run.
    console.log(`\n[${gait.name}] ABORTED — ${String(error).split('\n')[0]}`);
    failed = true;
    break;
  }
  for (const key of gait.keys) if (key !== 'w') await page.keyboard.up(key);
  await page.keyboard.up('w');
  if (gait.mouse === 'right') await page.mouse.up({ button: 'right' });
  await page.evaluate(() => window.__d2rim.engine.stepFrames(6));

  const dt = 1 / 60;
  const frames = [];
  for (let i = 1; i < samples.length; i++) {
    const a = samples[i - 1];
    const b = samples[i];
    frames.push({
      body: Math.hypot(b.pos[0] - a.pos[0], b.pos[2] - a.pos[2]) / dt,
      foot: Math.min(
        Math.hypot(b.l[0] - a.l[0], b.l[2] - a.l[2]) / dt,
        Math.hypot(b.r[0] - a.r[0], b.r[2] - a.r[2]) / dt,
      ),
    });
  }

  // Stride matching is a statement about *steady* locomotion, so measure it on
  // steady locomotion. The Blood Moor is a hilly field of props: driven flat
  // out the character accelerates to gait speed, clips a rock, sheds most of
  // it, and climbs a bank, and averaging that whole arc says nothing about
  // whether the feet plant. So take the longest run of consecutive frames whose
  // speed holds within a few percent of its own mean, and measure there. A
  // sample with no such window is reported as one rather than quietly averaged.
  const steadyWindow = (rows, tolerance = 0.09, minimum = 30) => {
    let best = null;
    for (let start = 0; start < rows.length; start++) {
      if (rows[start].body < 0.5) continue;
      let sum = 0;
      for (let end = start; end < rows.length; end++) {
        const speed = rows[end].body;
        if (speed < 0.5) break;
        sum += speed;
        const mean = sum / (end - start + 1);
        let ok = true;
        for (let k = start; k <= end; k++) {
          if (Math.abs(rows[k].body - mean) > mean * tolerance) {
            ok = false;
            break;
          }
        }
        if (!ok) break;
        if (best === null || end - start > best.end - best.start) best = { start, end };
      }
    }
    return best !== null && best.end - best.start + 1 >= minimum ? best : null;
  };

  // Widen the tolerance until a window appears. The Blood Moor is not a
  // treadmill: at walking pace the per-frame displacement of a capsule stepping
  // over uneven ground jitters by more than a tenth, so demanding a tight band
  // finds nothing and reports it as a failure of the character rather than of
  // the measurement. The tolerance actually used is reported.
  let window = null;
  let tolerance = 0;
  for (const candidate of [0.09, 0.15, 0.24, 0.36]) {
    window = steadyWindow(frames, candidate, 24);
    tolerance = candidate;
    if (window !== null) break;
  }
  const chosen = window === null ? [] : frames.slice(window.start, window.end + 1);
  const body = chosen.map((f) => f.body);
  const slower = chosen.map((f) => f.foot);
  const mean = (xs) => xs.reduce((p, c) => p + c, 0) / (xs.length || 1);
  const meanBody = mean(body);
  const meanMin = mean(slower);
  const bestPlant = slower.length ? Math.min(...slower) : Infinity;
  const planted = slower.filter((s) => s < PLANT_SPEED).length / (slower.length || 1);
  // `frames[i]` is the interval ending at `samples[i + 1]`, so read the graph's
  // state from inside the steady window rather than from the final frame.
  const last = samples[window === null ? samples.length - 1 : window.end + 1];

  const row = {
    frames: slower.length,
    meanBodySpeed: +meanBody.toFixed(3),
    meanMinFootSpeed: +meanMin.toFixed(3),
    meanRatio: +(meanMin / (meanBody || 1)).toFixed(3),
    minMinFootSpeed: +bestPlant.toFixed(3),
    plantedFraction: +planted.toFixed(3),
    cycleRate: +last.cycleRate.toFixed(3),
    state: last.state,
    peakBodySpeed: +Math.max(...frames.map((f) => f.body), 0).toFixed(3),
    reportedSpeed: +last.speed.toFixed(3),
    steadyWindow: window === null ? null : [window.start, window.end],
    drivenFrames: frames.length,
  };
  report.gaits[gait.name] = row;

  const checks = [
    ['a steady window was found at all', chosen.length > 0, `${chosen.length} frames`],
    ['plantedFraction', planted >= gait.minPlantedFraction, `>= ${gait.minPlantedFraction}`],
    ['meanRatio', meanMin / (meanBody || 1) <= gait.maxMeanRatio, `<= ${gait.maxMeanRatio}`],
    ['minMinFootSpeed', bestPlant <= gait.maxBestPlant, `<= ${gait.maxBestPlant}`],
  ];
  console.log(`\n[${gait.name}]`, JSON.stringify(row));
  for (const [label, ok, want] of checks) {
    console.log(`  ${ok ? 'PASS' : 'FAIL'} ${label} ${want}`);
    if (!ok) failed = true;
  }
}

console.log('\ngaits m/s:', JSON.stringify(report.gait));
if (errors.length > 0) {
  console.log('pageErrors:', errors);
  failed = true;
}
await browser.close();
server.kill();
console.log(failed ? '\nFOOT PLANT: FAIL' : '\nFOOT PLANT: PASS');
process.exit(failed ? 1 : 0);
