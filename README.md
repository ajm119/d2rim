# d2rim

A Skyrim-style third/first-person re-imagining of Diablo II Act I, built in
three.js.

Barbarian, melee-focused, third-person by default with a first-person toggle.
The vertical slice runs Rogue Encampment (hub) → Blood Moor (outdoor) → Den of
Evil (cave dungeon, first quest).

---

## Quick start

```bash
npm install
npm run dev        # vite dev server on http://127.0.0.1:5173
npm run build      # typecheck + production build
npm run preview    # serve the production build
npm test           # vitest unit tests
npm run typecheck  # tsc, app + tooling configs
npm run capture    # headless screenshot of the reference scene
```

### URL parameters

| Parameter         | Effect                                                          |
| ----------------- | --------------------------------------------------------------- |
| `?backend=webgl2` | Force the WebGL2 backend (WebGPU is the default when available) |
| `?backend=webgpu` | Force WebGPU; fails over to WebGL2 if it cannot initialise      |
| `?autostart=0`    | Boot without starting the rAF loop; the caller steps frames     |
| `?quality=low`    | Cheapest tier: FXAA at 0.75 scale, no GTAO, no SSR, 2 cascades   |
| `?quality=medium` | FXAA, half-res GTAO, no SSR, 3 cascades                          |
| `?quality=high`   | Default. TAA(8), half-res GTAO and SSR, 4 x 2048 cascades        |
| `?quality=ultra`  | TAA(16) + sharpen, full-res GTAO, high SSR, 4 x 3072 cascades     |

One `?quality=` value drives every subsystem at once — see
`src/render/RenderSettings.ts`, which owns the ladder and is the reason there is
one knob rather than six.

---

## Architecture

### Tech stack

- **Vite + TypeScript** (strict, plus `noUncheckedIndexedAccess` and
  `exactOptionalPropertyTypes`), ES modules, no framework for game code.
- **three.js** via the `three/webgpu` and `three/tsl` entry points.
- **@dimforge/rapier3d-compat** for physics.
- **Native WebAudio** for audio.
- **vitest** for unit tests, **playwright** for headless capture.

### Module model

The engine is a registry of `GameModule`s driven by a fixed-timestep loop.
A module is a self-contained system (player controller, combat, terrain, UI)
that receives a `GameContext` and hooks whichever lifecycle phases it needs:

```
rAF → clock.tick → accumulate
    → fixedUpdate × N   constant 60 Hz slices; deterministic simulation
    → update            once, variable dt; presentation-rate logic
    → lateUpdate        once, after every module's update has settled
    → input.endFrame    consume per-frame edges and deltas
    → renderer.render
```

Simulation is decoupled from display rate so physics behaves identically at
60 Hz and 144 Hz. `engine.alpha` carries the leftover fraction of a simulation
slice for interpolated rendering.

Modules never import each other. They communicate two ways:

- **`EventBus`** — typed pub/sub for notifications. The `GameEvents` interface
  is extended by declaration merging, so a feature adds its own events without
  editing the core:

  ```ts
  declare module '../core/EventBus' {
    interface GameEvents {
      'combat:hit': { attacker: number; target: number; damage: number };
    }
  }
  ```

- **`ServiceLocator`** — typed registry for handing out object references.
  Keys carry their value type, so lookups need no cast:

  ```ts
  export const PhysicsKey = serviceKey<PhysicsWorld>('physics');
  services.register(PhysicsKey, world);
  const world = services.get(PhysicsKey); // PhysicsWorld
  ```

Together these keep the module graph acyclic, which is the property that lets
systems be added, removed and tested in isolation.

### Source layout

```
src/
  core/
    types.ts            The architecture contract: GameContext, GameModule,
                        TimeState, RendererHandle, RendererBackend
    Engine.ts           Frame loop, module registry, resize, visibility pause,
                        deterministic stepFrames()
    EventBus.ts         Typed pub/sub, augmentable via declaration merging
    ServiceLocator.ts   Typed service registry
    Input.ts            Action-mapped keyboard/mouse + pointer lock
    Time.ts             FixedStepAccumulator, Clock, TimeState factory
  render/
    RendererFactory.ts  WebGPU-first renderer with WebGL2 fallback
    webgpuCompat.ts     Browser compatibility shims (see below)
    FrameGraph.ts       THE INTEGRATION POINT. Assembles every render module in
                        dependency order, closes the seams between them, and
                        audits every ServiceLocator contract at boot
    RenderSettings.ts   The one quality ladder and the one weather state
    Sky.ts / Atmosphere.ts / TimeOfDay.ts    scattering sky, aerial perspective
    Lighting.ts / CascadedShadowMaps.ts / IBL.ts
    MaterialLibrary.ts + materials/          triplanar, blending, parallax, wetness
    Volumetrics.ts      froxel fog, shadowed in-scatter, local fog volumes
    post/               PostStack, TAA, GTAO, SSR, Bloom, Tonemap, ColorGrade,
                        LightShafts, Denoise, Motion
    ProceduralSky.ts    HDR equirectangular sky: background + IBL
    ProceduralTextures.ts  Seamless albedo/roughness/normal synthesis
  scene/
    BloodMoor.ts        The showcase scene: the Act I vertical slice
    ReferenceScene.ts   The phase-1 material chart, kept for regression isolation
  ui/
    DebugOverlay.ts     Backend / FPS / frame count readout
  main.ts               Boot, module registration, window.__d2rim
```

### Determinism

`Engine.stepFrames(n, dt?)` advances the world by an exact number of frames with
an exact delta, bypassing the wall clock and cancelling the rAF loop. Combined
with seeded procedural generation (no `Math.random` anywhere in asset
synthesis), the same step count reproduces the same frame, which makes
golden-image regression testing viable.

### Headless tooling

`window.__d2rim` exposes `{ engine, ctx, ready }`. The capture sequence is:

```js
await window.__d2rim.ready;
await window.__d2rim.engine.stepFrames(60);
// world state is now deterministic; capture
```

---

## Renderer

WebGPU is attempted first and WebGL2 is the fallback. Which one won is logged at
boot and shown in the debug overlay.

The fallback is a `WebGPURenderer` constructed with `forceWebGL: true` — three's
own `WebGLBackend` — rather than the classic `WebGLRenderer`. This is deliberate:

1. TSL node materials only exist in the node renderer architecture. A classic
   `WebGLRenderer` cannot draw them, so falling back to it would not degrade
   effects, it would fail to draw the game's materials at all.
2. `WebGLBackend` is a real WebGL2 path, verified rendering a full lit, shadowed,
   tone-mapped frame in this project's headless container.
3. Loading the classic build alongside `three/webgpu` would put two copies of
   every core class in the bundle and silently break `instanceof`.

`RendererHandle.three` is still typed `WebGPURenderer | WebGLRenderer`, so adding
a direct-`WebGLRenderer` tier later needs no contract change.

### Colour pipeline

Set once in `RendererFactory` and authored against everywhere else:
ACES filmic tone mapping at exposure 1.0, sRGB output, linear working space,
PCF-soft shadows.

### Compatibility shim

three.js r185 sets `GPUTextureViewDescriptor.swizzle` to the string `'rgba'`.
Chromium 141's Dawn implements the revised spec where that field is a dictionary,
so **every** `GPUTexture.createView()` throws and WebGPU is entirely dead.

`render/webgpuCompat.ts` installs a self-latching patch before the renderer is
constructed: descriptors pass through untouched until that exact failure is
observed, then the string is converted to `{ r, g, b, a }` for the rest of the
session. On browsers that accept the string it never latches, so the shim
becomes a no-op automatically once three.js ships the dictionary form — there
are no version checks to maintain.

---

## Showcase scene

`src/scene/BloodMoor.ts` is the visual baseline the rest of the project is
calibrated against: a composed view of Diablo II Act I's Blood Moor, built from
the project's real photoscanned material sets and real prop models.

It is composed rather than assembled. Three depth planes — foreground rubble
shelf, midground campfire and ruined wall, background treed ridge — with one
focal point (the fire, the only warm light in the frame) and one leading line
(the broken fence running from the lower-left corner back toward it). The
Barbarian stands between the wall and the fire, idling, which proves the
41-joint rig loads and animates and gives the moor a scale reference.

The ground is one mesh with a height-blended two-archetype material: wet mud
underneath, dead grass on top, coverage driven by world height and slope so mud
collects in the hollows the way water would sort it. Exposed cliff rock is
deliberately *not* a third splat layer — splatting can only produce rock lying
flat against the heightfield, so the escarpment is real displaced geometry
carrying the triplanar `rock` archetype. The ruined wall is boxed courses of the
parallax-mapped `wetStone` archetype, tiled per block from the block's own world
size so the coursing stays life-sized.

`TerrainField` is the single authority for ground height: the mesh builder and
every prop placement sample the same function, which is the only reason props sit
*on* the ground rather than hovering. Everything derives from one seed and every
animation reads `TimeState.elapsed`, so `stepFrames(n)` reproduces any frame.

`src/scene/ReferenceScene.ts` — the phase-1 material response chart — is kept
because it isolates BRDF regressions from art-direction changes, which the
showcase scene deliberately cannot.

---

## Frame graph

`src/render/FrameGraph.ts` is where six independently-authored subsystems become
one renderer. Read its module comment before touching render order: it documents
every seam it closes, including the two fog models (aerial perspective owns
beyond 96 m, the froxel volume owns nearer, and the near-ground mist term is
zeroed out of the atmosphere so it is not modelled twice), the single exposure
authority (`PostStack.composite` — the renderer's own tone mapping is forced
off and the frame graph asserts it), and the four `ServiceLocator` ids that more
than one module declares.

`auditServices` prints a connected/degraded table at boot. Nothing in this
renderer is allowed to silently no-op because a service was never registered.

---

## Headless capture

```bash
node scripts/capture.mjs --backend=webgl2 --frames=60
node scripts/capture.mjs --backend=webgpu --frames=0
```

Output lands in `captures/`.

Capture goes through `RendererHandle.captureFrame()` — render to an offscreen
target, then `readRenderTargetPixelsAsync` — for **both** backends rather than
`page.screenshot()`. Two independent reasons:

- WebGPU has no choice. Its canvas swapchain never reaches the headless
  compositor, so a screenshot is pure black even though the GPU produced correct
  pixels.
- On WebGL2 a screenshot *works*, but `render()` only queues GPU work. Under
  software rasterisation the CPU queues frames far faster than they retire, so a
  screenshot waits on a compositor dozens of frames behind and times out. The
  readback is a genuine sync point.

### Container-specific notes

The development container has no GPU: everything is SwiftShader software
rasterisation through ANGLE → Vulkan 1.3. Consequences worth knowing:

- **Frame rates here mean nothing** about real-GPU performance. Capture at
  960×540 and budget generous timeouts.
- **WebGL2 is the reliable headless backend.** It needs no special flags.
- **WebGPU renders correctly but cannot present.** Worse, presenting to the
  canvas eventually loses the device (`A valid external Instance reference no
  longer exists`), which takes the readback with it. So WebGPU captures must use
  `--frames=0` (and `?autostart=0`, which the harness sets), keeping all
  rendering on the offscreen path. Animated captures use WebGL2.
- Working Chromium flags:
  `--no-sandbox --disable-dev-shm-usage --enable-unsafe-webgpu --use-gl=angle --use-angle=swiftshader`.
  `--enable-unsafe-webgpu` is the only decisive one. Never run
  `playwright install`; the browser bundle is pre-provisioned and the harness
  points at it explicitly.
- Omitting `--enable-unsafe-webgpu` makes `requestAdapter()` return null, which
  exercises the WebGL2 fallback path for free — useful as a CI matrix.

---

## Controls

| Action                    | Binding                  |
| ------------------------- | ------------------------ |
| Move                      | `W` `A` `S` `D` / arrows |
| Sprint                    | `Shift`                  |
| Jump                      | `Space`                  |
| Attack                    | Left mouse               |
| Heavy attack              | Middle mouse / `R`       |
| Block                     | Right mouse              |
| Interact                  | `E`                      |
| Toggle first/third person | `F`                      |
| Inventory                 | `I` / `Tab`              |
| Skill tree                | `K`                      |
| Menu                      | `Escape`                 |

Input is action-mapped: game code asks `input.isDown('Attack')` and never sees a
key code, so rebinding and future gamepad support need no gameplay changes.

---

## Licence

MIT — see [LICENSE](./LICENSE).
