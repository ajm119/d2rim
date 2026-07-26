# `tools/capture` — headless capture harness

Turns a declarative shot list into a directory of PNGs that a critic agent can
grade, plus a single contact sheet that shows all of them at once.

This is the backbone of the project's visual quality loop. Its one hard promise:

> **A broken renderer produces a failed run, never a plausible-looking image.**

Every capture is decoded back off disk and put through a blank-frame guard
before the tool is allowed to exit 0. Silently writing a black PNG is the worst
failure mode here, because everything downstream then grades a frame that was
never rendered — so it is made impossible rather than unlikely.

---

## Quick start

```bash
npm run shots                 # capture every shot in tools/capture/shots.json
npm run contact-sheet         # composite captures/shots/*.png into one sheet
```

Outputs:

| path | what |
|---|---|
| `captures/shots/<id>.png` | one PNG per shot |
| `captures/shots/report.json` | per-shot pixel statistics, pass/fail, timings |
| `captures/contact-sheet.png` | labelled grid of every shot |

A full seven-shot run takes about **4–5 minutes** in this container. That is
SwiftShader software rasterisation, not a signal about real-GPU performance.

### If you are a critic agent

Read **`captures/contact-sheet.png`** first — one image, every shot, labelled.
Then `Read` individual PNGs from `captures/shots/` for anything that needs a
closer look. `report.json` gives you the shot descriptions and pixel statistics
without opening any images.

---

## `capture.mjs`

```bash
node tools/capture/capture.mjs [options]
```

| option | default | meaning |
|---|---|---|
| `--shots <file>` | `tools/capture/shots.json` | shot list |
| `--out <dir>` | `captures/shots` | output directory |
| `--only <a,b>` | all | capture just these shot ids |
| `--width` / `--height` | per shot | override every shot's resolution |
| `--warmup <n>` | per shot | override every shot's warmup frames |
| `--backend <b>` | per shot | force `webgl2` or `webgpu` everywhere |
| `--url <origin>` | — | use a running server instead of `vite preview` |
| `--port <n>` | `4173` | preview server port |
| `--no-build` | — | do not rebuild `dist/` even if stale |
| `--keep-going` | off | capture all shots instead of stopping at the first failure |
| `--ignore-page-errors` | off | do not fail a shot on an uncaught page exception |

Exit code is `0` only if every requested shot was captured **and** passed the
guard.

### Fast iteration

Full runs are slow because every warmup frame is real GPU work. When tuning one
shot:

```bash
node tools/capture/capture.mjs --only backlit-rim --width 800 --height 450 --warmup 4
```

That is a few seconds instead of a minute. Re-running with `--only` **merges**
into the existing `report.json` rather than replacing it, so the other shots'
entries survive and `contact-sheet.mjs` still sees the full set.

### What a run actually does

```
build if stale → vite preview → chromium → per shot:
  set viewport → navigate ?autostart=0 → await __d2rim.ready
  → setup script → stepFrames(warmup) → pose script
  → capture → decode → BLANK-FRAME GUARD → write
```

**Determinism.** The page boots with `?autostart=0`, so the requestAnimationFrame
loop never runs and this tool is the only thing that advances time.
`engine.stepFrames(n)` uses a fixed synthetic delta and awaits the renderer each
frame. Two runs of the same shot therefore produce the same world state, which
is what makes golden-image diffing possible later. This is also why warmup is a
frame count and not a sleep.

**Staleness.** If `dist/` is older than `src/`, `index.html` or `vite.config.ts`,
the tool rebuilds before capturing. Grading a stale build is a silent lie, so
this is on by default.

---

## `shots.json`

```jsonc
{
  "defaults": { "width": 1600, "height": 900, "warmupFrames": 45, "backend": "webgl2" },
  "shots": [
    {
      "id": "wide-establishing",          // required, unique, filename-safe
      "description": "…",                 // shown in logs, report and sheet labels
      "url": "/",                         // resolved against the server origin
      "backend": "webgl2",                // or "webgpu"
      "width": 1600, "height": 900,
      "warmupFrames": 60,
      "mode": "readback",                 // or "screenshot"
      "setup": ["…"],                     // JS, runs BEFORE the warmup
      "pose":  ["…"],                     // JS, runs AFTER the warmup
      "guard": { "minEdgeDensity": 0.001 } // per-shot threshold overrides
    }
  ]
}
```

Scripts are strings, or **arrays of lines** joined with newlines (JSON has no
multi-line strings, and one-line scripts full of `\n` are unmaintainable).

### `setup` vs `pose` — read this before writing a shot

They are not interchangeable, and getting it wrong is the most likely way to
waste a run:

- **`setup` runs before the warmup frames.** Use it for scene state you want the
  simulation to then settle into — time of day, spawning something, swapping a
  material.
- **`pose` runs after the warmup, immediately before the capture render.** Use it
  for **the camera**. Scene modules re-drive the camera from `time.elapsed` on
  every update, so a camera set in `setup` is simply overwritten during the
  warmup. Nothing updates between `pose` and the capture, so a pose sticks
  exactly as written.

Scripts are async function bodies with these bindings injected:

```
d2rim  engine  ctx  scene  camera  renderer  time  services  events  THREE
```

`THREE` is the live `three/webgpu` namespace (`window.__d2rim.three`), so
`instanceof` holds against the game's own objects. Shot scripts are strings and
cannot `import`, which is why it is exposed.

```js
// typical pose
camera.position.set(-6.4, 2.9, -7.1);
camera.lookAt(0, 2.05, 0);
camera.fov = 36;
camera.updateProjectionMatrix();
```

### `mode`

- **`readback`** (default) — renders into an offscreen target and reads the
  pixels back. No DOM. Exact output resolution regardless of the viewport.
- **`screenshot`** — `page.screenshot()`, so DOM/HUD layers are included. Use it
  for UI review. A GPU sync is forced first, otherwise the compositor is dozens
  of frames behind and the screenshot is stale or times out.

`readback` is the default because it is the only route that works on **both**
backends here (see below), which keeps cross-backend comparison meaningful.

### Seeded baselines

| id | what it is for |
|---|---|
| `wide-establishing` | overall composition, horizon, aerial perspective |
| `hero-material-closeup` | polished metal, reflection sharpness, contact shadow |
| `roughness-sweep` | the PBR roughness ramp — first thing to break on a BRDF regression |
| `backlit-rim` | into-the-sun rim lighting, tone mapping, highlight rolloff |
| `shadow-detail` | PCF softness, acne, peter-panning, shadow-map aliasing |
| `hud-composite` | DOM overlay legibility over the rendered frame |
| `webgpu-backend-check` | proof the WebGPU path renders correctly |

Warmup counts are currently sized to "settle the sim and reach a known animation
pose". **When TAA, SSAO or any temporal accumulation lands, raise them to at
least the accumulation window**, or captures will be graded mid-convergence.

---

## The blank-frame guard (`frame-guard.mjs`)

Also usable standalone on any image:

```bash
npm run guard captures/shots/*.png
node tools/capture/frame-guard.mjs some-frame.png
```

Five near-orthogonal statistics, because no single one catches every degenerate
frame:

| statistic | catches | default |
|---|---|---|
| `uniqueColors` | solid fills, clear-colour-only frames | ≥ 512 |
| `meanLuminance` | all-black (device lost), blown-out white | 0.02 – 0.97 |
| `luminanceStdDev` | flat fills of *any* brightness, including mid grey | ≥ 0.015 |
| `edgeDensity` | smooth gradients with no geometry in them | ≥ 0.002 |
| `dominantBucketShare` | a frame that is 99 % sky with the scene missing | ≤ 0.92 |
| `opaqueShare` | a fully transparent PNG | ≥ 0.98 |

The combination matters. A clear sky gradient with nothing in front of it passes
the first three and fails the last two — that is "the renderer booted but the
scene never loaded", and a naive black-pixel check waves it straight through.

**Verified behaviour.** Against the real scene at 1600×900: ~70–130 k unique
colours, luminance σ 0.12–0.19, edge density 0.20–0.39 — every threshold sits
one to two orders of magnitude away, so the guard flags catastrophes and never
bikesheds a frame that merely got darker. Confirmed to fail: pure black, pure
white, mid grey, transparent, sky-gradient-only, sparse noise on black, an
emptied scene, and a camera buried under the ground plane (a uniform *non-black*
grey at luminance 0.245).

> If the guard fires, the render path is broken. **Inspect the PNG before
> touching the thresholds.** Loosening a threshold to make a run go green
> disables the one thing standing between this project and a quality loop that
> grades black images.

---

## `contact-sheet.mjs`

```bash
node tools/capture/contact-sheet.mjs --in captures/shots --out captures/contact-sheet.png
node tools/capture/contact-sheet.mjs a.png b.png --cols 2 --cell-width 800
```

| option | default |
|---|---|
| `--in <dir>` | `captures/shots` |
| `--out <file>` | `captures/contact-sheet.png` |
| `--cols <n>` | `3` |
| `--cell-width <px>` | `620` |
| `--title <text>` | `d2rim capture contact sheet` |

Ordering follows `report.json` (i.e. shot-list declaration order — a shot list is
authored as a sequence, and alphabetical order would scramble it). Images are
letterboxed, **never cropped**: a sheet that silently crops makes the critic
grade a composition that does not exist. Shots that failed the guard are banded
red rather than dropped, so a bad shot is visible instead of absent. The finished
sheet is itself run through the guard.

Uses `sharp` (installs cleanly here, prebuilt binary, no compilation).

---

## Container specifics

Chromium lives at `/opt/pw-browsers/chromium-1194/chrome-linux/chrome`
(override with `CHROMIUM_PATH`). **Never run `playwright install`.**

Proven flags, in `cli.mjs`:

```
--no-sandbox  --disable-dev-shm-usage  --enable-unsafe-webgpu
--use-gl=angle  --use-angle=swiftshader
```

`--enable-unsafe-webgpu` is decisive: without it `navigator.gpu.requestAdapter()`
returns null and the WebGPU path silently degrades into the WebGL2 path.

### WebGPU limitations here

Two container-specific quirks, neither of which is an engine bug:

1. **The WebGPU swapchain never reaches the headless compositor.** The GPU
   renders correctly, but `page.screenshot()` of a WebGPU canvas is pure black.
   Hence the readback capture route.
2. **Presenting to the WebGPU canvas eventually loses the device**
   (`A valid external Instance reference no longer exists`), which takes the
   readback down with it. So `warmupFrames` is **automatically clamped to 0** on
   WebGPU, with a warning. Animated shots must use `webgl2`.

Both paths render the same scene identically otherwise — compare
`wide-establishing` against `webgpu-backend-check` on the contact sheet.

---

## Extending

Adding a shot is a JSON edit; no code changes. Keep ids filename-safe
(`[a-z0-9-]`) and write a description worth reading — it is the label a critic
sees on the contact sheet, and it is the only context they get for the frame.

Unknown option names and unknown `guard` keys are **rejected**, not ignored: a
typo'd flag that silently did nothing would be a capture run that quietly
measured the wrong thing.
