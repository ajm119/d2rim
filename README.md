# d2rim

A Skyrim-style third/first-person re-imagining of **Diablo II Act I**, built in
three.js and taken to a playable vertical slice.

You are the Barbarian. You start in the Rogue Encampment, take the Den of Evil
quest from Akara, walk out into the Blood Moor, fight your way through the
skeletons in it, clear the cave at the far end, and walk back to turn the quest
in for a skill point. Melee combat, D2's affix item generator, a grid inventory,
a skill tree, five NPCs with dialogue and vendors, and save/load through the
browser's IndexedDB.

It is a **vertical slice**, not a game. One class, one act, three zones, one
quest chain, and **no audio at all**. Read [Known defects](#known-defects)
before you form an opinion — it is complete and specific on purpose.

---

## Quick start

```bash
npm install          # no engines field is declared; developed and verified on Node 22.22
npm run dev          # vite dev server on http://127.0.0.1:5173
```

| Script                  | What it does                                                |
| ----------------------- | ----------------------------------------------------------- |
| `npm run dev`           | Vite dev server, `127.0.0.1:5173`                            |
| `npm run build`         | `npm run typecheck && vite build`                            |
| `npm run preview`       | Serves `dist/` on `127.0.0.1:4173`                           |
| `npm test`              | `vitest run` — 55 test files                                 |
| `npm run test:watch`    | Vitest in watch mode                                         |
| `npm run typecheck`     | `tsc --noEmit` over the app and the tooling tsconfigs        |
| `npm run assets`        | Fetch the third-party assets listed in `tools/assets/manifest.json` |
| `npm run assets:verify` | Re-verify every fetched asset by SHA-256                     |
| `npm run shots`         | Deterministic headless screenshot harness (`tools/capture/`) |
| `npm run contact-sheet` | Composite the captures into one labelled grid                |
| `npm run guard`         | Blank-frame guard over captured PNGs                         |
| `npm run capture`       | Older single-shot capture entry point (`scripts/capture.mjs`) |

The verification harnesses in `tools/*.mjs` are **not** wired to npm scripts.
They run against the built bundle:

```bash
npm run build && node tools/verify-kill.mjs
```

### URL parameters

| Parameter                                | Default        | Effect                                                        |
| ---------------------------------------- | -------------- | ------------------------------------------------------------- |
| `?backend=webgpu` \| `webgl2`            | auto-probe     | Force a renderer backend instead of probing for WebGPU         |
| `?quality=low` \| `medium` \| `high` \| `ultra` | `high`  | The whole render ladder at once — see `src/render/RenderSettings.ts` |
| `?zone=encampment` \| `bloodMoor` \| `denOfEvil` | `encampment` | Boot straight into a zone                                |
| `?autostart=0`                           | on             | Boot without starting the rAF loop; the caller steps frames    |
| `?enemies=0`                             | on             | Boot zones unpopulated                                         |
| `?fade=0`                                | `0.35` s       | Disable the zone-transition fade                               |

An unrecognised `?quality=` logs a warning and falls back to `high` rather than
failing quietly.

### Controls

Bindings are `KeyboardEvent.code`, so they are layout-independent — `KeyW` is
the same physical key on AZERTY. Source of truth: `DEFAULT_BINDINGS` in
`src/core/Input.ts`.

| Action                      | Binding                        |
| --------------------------- | ------------------------------ |
| Move                        | `W` `A` `S` `D` / arrow keys   |
| Sprint                      | `Shift` (either)               |
| Jump                        | `Space`                        |
| Attack (light chain)        | Left mouse                     |
| Heavy attack                | Middle mouse or `R`            |
| Block                       | Right mouse                    |
| Interact / talk             | `E`                            |
| Toggle first/third person   | `F`                            |
| Inventory                   | `I` or `Tab`                   |
| Skill tree                  | `K`                            |
| Pause menu                  | `Escape`                       |

Screen-specific input, handled by `UiManager` on a capturing listener so it wins
over gameplay:

- **Escape** closes the topmost open panel; with nothing open it opens the pause
  menu. Panel toggles are inert while a dialogue is open — the player is
  talking, not managing gear.
- **Inventory** is click-to-lift, click-to-place, not HTML5 drag: *click to lift
  an item, click again to place it, right-click to equip.*
- **Dialogue** advances on `Space` / `Enter`; `1`–`9` pick a numbered choice.
- **Skill tree** and **vendor** are pointer-only.

Pointer lock is requested on the first canvas click, because browsers require a
user gesture. Game code never sees a key code — it asks `input.isDown('Attack')`
— which is what makes rebinding and future gamepad support a non-event.

---

## Architecture

### The module contract

The engine is a registry of `GameModule`s. A module is a self-contained system
— the player controller, combat, a zone, one render pass, one UI screen — that
receives a `GameContext` and implements whichever lifecycle hooks it needs.
`src/core/types.ts` is the whole contract and is worth reading first.

```ts
interface GameModule {
  readonly name: string;                       // unique; Engine.add throws on a duplicate
  init(ctx: GameContext): Promise<void> | void;
  fixedUpdate?(ctx: GameContext, fixedDt: number): void;  // 0..N times, constant 60 Hz
  update?(ctx: GameContext, dt: number): void;            // exactly once, variable dt
  lateUpdate?(ctx: GameContext, dt: number): void;        // once, after every update
  dispose?(): void;
}
```

`GameContext` carries `engine`, `scene`, `camera`, `renderer`, `input`, `events`,
`time`, `services`. Its identity is stable for the engine's lifetime, so a module
may hold onto it.

`TimeState` is deliberately fixed to exactly four mutable fields — `elapsed`,
`delta`, `frame`, `scale` — and is mutated in place. Read it, do not cache it.
Setting `scale` to 0 freezes gameplay while rendering continues.

**Registration order is frame order.** Modules are initialised sequentially, not
in parallel, because they routinely depend on services registered by earlier
ones, and disposed in reverse. `main.ts` states each ordering constraint where
it matters: combat is registered *before* the player because `PlayerController`
stands its placeholder attack input down when it finds a `combat` service;
`CombatFeedback` is last of the gameplay modules because its `lateUpdate` adds
camera shake and must run after `CameraRig` has placed the camera.

### The loop

Fixed timestep, decoupled from display rate, so physics behaves the same at
60 Hz and 144 Hz.

```
rAF ─► input.beginFrame()
    ─► clamp raw delta to [0, 0.25 s], scale by time.scale, advance TimeState
    ─► fixedUpdate × N        constant 1/60 s slices, at most 5 per frame
    ─► update                 once, variable dt
    ─► lateUpdate             once, after every module's update has settled
    ─► input.endFrame()       clear per-frame edges and mouse deltas
    ─► await renderer.render(scene, camera)
    ─► emit 'engine:frame'
```

`maxSubSteps = 5` is the spiral-of-death guard. `Engine.alpha` carries the
leftover fraction between simulation steps for interpolated rendering; it lives
on the engine rather than in `TimeState` because the contract fixes that type at
four fields. A `#stepInFlight` guard drops a rAF callback whose predecessor's
GPU work has not resolved, so the CPU cannot queue renders faster than the
device retires them.

A module that throws is caught, logged once per phase, and then skipped: one
broken system degrades that system rather than killing the frame loop.

`Engine.stepFrames(count, dt?)` cancels the rAF loop, latches manual mode and
advances an exact number of frames with an exact delta, bypassing the wall
clock. Every harness in `tools/` is built on it, and it is the reason
`stepFrames(n)` reproduces the same frame across runs.

### EventBus and ServiceLocator

Modules never import each other. They communicate two ways, and between them the
module graph stays acyclic — which is the property that lets systems be added,
removed and tested in isolation.

**`EventBus`** — typed pub/sub, extended by declaration merging, so a feature
adds its own events without editing the core:

```ts
declare module '../core/EventBus' {
  interface GameEvents {
    'combat:hit': { attacker: number; target: number; /* … */ };
  }
}
```

Handlers live in insertion-ordered `Set`s; `emit` iterates a snapshot so a
handler may unsubscribe another mid-emit; a throwing handler is caught and does
not stop the rest.

**`ServiceLocator`** — typed registry, keyed by a phantom-branded key so lookups
need no cast:

```ts
export const PhysicsKey = serviceKey<PhysicsWorld>('physics.world');
services.register(PhysicsKey, world);
const world = services.get(PhysicsKey);   // inferred PhysicsWorld
```

`get` throws (listing what *is* registered) and `register` throws on a duplicate
id. `tryGet` is the optional-dependency route. This exists because
`CombatSystem` needs the player's transform, and importing `PlayerController` —
which imports combat — would be a cycle.

### The frame graph

`src/render/FrameGraph.ts` is where twelve independently-authored render modules
become one renderer. It builds them in a fixed order, closes the seams between
them, and audits 24 `ServiceLocator` contracts at boot, printing a
connected/degraded table. Nothing in the renderer is allowed to silently no-op
because a service was never registered.

There is **no dependency solver**. Ordering is an explicit array plus one
insertion hint (`addPass(pass, { before: 'post.taa' })`), and the nine
load-bearing constraints are written down in the module comment. This is a
deliberate choice: a solver would hide the constraints it satisfies.

Per-frame order:

```
update       TimeOfDay ──► Sky (sky-view LUT, environment probe, fog node)
lateUpdate   Lighting (key light + CSM cascade fit)
             GuideBuffer (shared depth+normal prepass, one draw)
             GTAO ──► IBL ──► SSR (hi-Z, against last frame's colour)
             Volumetrics (froxel scatter) ──► LightShafts
render       MOTION (jitter projection, bind velocity MRT)
             SHADOWS (CSM cascades) ──► OPAQUE ──► TRANSPARENT
             post.ssr        composite reflection delta      [HDR]
             lightshafts     composite volumetrics + shafts  [HDR]
             post.taa        temporal resolve                [HDR]
             post.bloom      producer — writes its own pyramid
             post.composite  EXPOSURE ─► AgX ─► GRADE        [HDR ─► LDR]
             post.fxaa       FXAA tiers only                 [LDR]
DOM          HUD, inventory, dialogue — composited by the browser
```

A pass is either a **chain** pass (consumes the running colour texture and
produces a new one) or a **producer** (reads it, writes only into buffers it
owns, and so costs no ping-pong slot — which is how bloom's six render targets
cost zero full-res scratch).

Two seams worth knowing about, both documented at length in `FrameGraph.ts`:

- **Two fog models, one atmosphere.** The froxel volume owns everything inside
  45 m and aerial perspective owns beyond it; the near-ground mist term is
  zeroed out of the atmosphere so it is not modelled twice.
- **One exposure authority.** `PostStack.composite` owns exposure, locked at
  1.72 with auto-exposure off; the renderer's own tone mapping is forced to
  `NoToneMapping` while the stack is installed, and the frame graph asserts it.
  The tone curve is **AgX** with a `grimdark` look, not ACES — ACES's hue skew
  turns the two colours this game lives on, orange firelight and blue-grey
  shade, into yellow and cyan.

### Zone lifecycle

A `Zone` is a `GameModule` that additionally owns a subtree (`root`), a travel
id, entry points, portals, optional NPC anchors and enemy spawns, an optional
colour grade, and an optional collider builder. All zone geometry is authored
**origin-centred in zone-local coordinates**; two zones are never live at once.

`ZoneManager` is one ordinary module that hosts the active zone and forwards the
three update phases to it. Zones are registered as **factories, not instances**,
so re-entering a zone constructs a fresh one. It must be registered after
`PhysicsWorld` and before `PlayerController`.

Load, with the progress phases it reports:

1. `construct` — call the factory
2. `build` — `await zone.init(ctx)`
3. `colliders` — snapshot physics collider ids, call `zone.buildColliders()`,
   build portal colliders, diff to learn which ones the zone owns
4. `place` — apply the zone's colour grade *before* placing the player, so the
   first frame of a new zone is already graded for it; resolve an entry point
5. enemies — construct the zone's `EnemyDirector` and start the model download
6. `ready` — emit `zone:loaded`

Unload reclaims five things and is tested for all five: geometries, materials
and textures (walking the detached subtree, and disposing lights' shadow maps),
the snapshotted physics colliders, and the zone's enemies. Anything from
`AssetManager`'s cache is stamped shared and skipped. The per-zone grade is
reverted on unload so a zone cannot leak its look into the next one.

The three zones are `encampment` (Rogue Encampment, the hub), `bloodMoor` (the
outdoor showcase scene) and `denOfEvil` (a procedurally generated cave, from
`src/world/DungeonGenerator.ts`).

### Renderer

WebGPU is probed first, WebGL2 is the fallback, and which one won is logged at
boot and shown in the debug overlay. The fallback is a `WebGPURenderer` built
with `forceWebGL: true` — three's own `WebGLBackend` — not the classic
`WebGLRenderer`, because TSL node materials only exist in the node renderer
architecture; falling back to the classic renderer would not degrade effects, it
would fail to draw the game's materials at all.

---

## What genuinely works

Every claim in this section has a harness behind it. Where a number appears, it
was measured by driving the built game, not by reading the source.

**The character.** A 41-joint rig with 76 animation clips, driven by a blend
space and an animation graph with layered upper/full-body actions. Third person
by default, first person on `F`.

**Foot planting.** A three-pass analytic IK solver pins the stance foot in world
space. Measured by `tools/verify-footplant.mjs`, before the solver and after it:

| gait   | body m/s | slower foot m/s | ratio        | planted frac  | held foot m/s |
| ------ | -------- | --------------- | ------------ | ------------- | ------------- |
| walk   | 0.93     | 0.62 → 0.162    | .67 → .174   | .18 → .875    | → 0.000       |
| run    | 3.98     | 2.47 → 2.381    | .62 → .598   | .06 → .246    | → 0.010       |
| sprint | 5.53     | 3.30 → 3.090    | .60 → .559   | .04 → .267    | → 0.001       |

**Melee combat, in both directions.** Swept hitboxes (the blade is a moving line
segment, sampled every frame and swept between poses with five substeps),
authored damage windows delivered as animation events, input buffering, and a
real five-move chain. The numbers are in [Combat balance](#combat-balance)
below.

**Enemies.** Four skeleton variants — minion, warrior, rogue, mage — on a
behaviour tree, with perception, a stand-off distance, pursuit, real attack
clips and real damage. Their reach numbers are measured, not guessed:
`tools/measure-enemy-reach.mjs` loads each GLB and sweeps a player capsule
outward to find the furthest separation at which each clip's implied blade still
touches during its authored window.

**Three connected zones**, with portals and travel in both directions, and
without leaking. `tools/verify-zones.mjs` walks camp → moor → den → moor → camp
and asserts that zone-owned colliders and character-collider records return to
baseline *exactly*, then walks a **second** lap and asserts renderer geometry and
texture counts are flat — because an absolute bound cannot tell cache warm-up
from a leak.

**The RPG layer.** D2's affix item generator, a grid inventory with real
placement, a skill tree, a quest chain, five NPCs with dialogue trees, vendors,
and save/load. `tools/verify-rpg.mjs` runs the whole loop end to end against the
browser's real IndexedDB: level up, spend a skill point and confirm a *combat*
number moved, equip an item and confirm the offence handed to the damage model
changed, accept the Den quest from Akara, clear the Den, turn it in, save,
reload, compare.

**Gear is load-bearing, not cosmetic.** `rpg.offense` / `rpg.defense` are
`ServiceLocator` providers that `CombatSystem` asks for instead of reading
constants, so an equipped ring's attack rating moves the roll the swing actually
makes, and an equipped plate's damage reduction cuts the damage a blow actually
applies. Both are asserted in `tools/verify-balance.mjs`.

### Combat balance

For most of this project's life every combat number it had was damage
**taken**. It could say how fast a passive Barbarian dies and could not say
whether he can kill anything, which is not a game. `tools/verify-kill.mjs`
closes that, and the way it does it matters: it teleports the fighters **once**
to stage the encounter and then lets both act, aiming the player by feeding
pointer pixels through `Input.nudgePointer` — the same accumulator a `mousemove`
writes to — capped at 8 rad/s so the harness cannot turn faster than a hand
could. A rotation is not a teleport, so the blade keeps moving between frames
and `WeaponHitbox`'s sweep stays a real sweep. Movement is real `KeyW` events;
attacks are real `mousedown` on the canvas.

**Outgoing** — a level-1 Barbarian in the starting kit, against a skeleton that
is moving, retreating and fighting back. One run each:

| variant | pool | swings | landed | mean blow | time to kill | player HP left |
| ------- | ---- | ------ | ------ | --------- | ------------ | -------------- |
| minion  | 46   | 2      | 2      | 29.0      | **1.00 s**   | 120 / 120      |
| warrior | 68   | 17     | 5      | 16.6      | **10.07 s**  | 53 / 120       |
| rogue   | 58   | 12     | 4      | 28.8      | **7.10 s**   | 85 / 120       |
| mage    | 52   | 8      | 3      | 17.3      | **5.47 s**   | 102 / 120      |

He kills all four. Across runs the warrior lands between 4.6 s and 10.1 s and
the minion between 1.0 s and 3.4 s; the scatter is real and comes from how the
skeleton chooses to reposition, not from the harness. Every skeleton takes
between two and five connecting blows, which is the number the balance is
actually built on and the one the harness asserts.

`swings` counts every press the combo machine accepted; `landed` counts the ones
that connected *and* passed the to-hit roll. The gap between them is the fight:
a mashing player whiffs while the skeleton repositions.

The harness reports its own residual aim error so that a bad result can be told
apart from a badly aimed one. It is **0.0°** for almost every duel — the 8 rad/s
turn cap simply never binds at duelling range. The one exception observed is the
mage, which moves the most: 4.3° of mean residual in one run, 0.0° in another.
Nothing here was measured by a Barbarian swinging at where a skeleton used to
be.

**Incoming**, measured separately by `tools/verify-balance.mjs`: one skeleton
kills a passive, non-blocking, non-moving player in **12.30 s**; three kill him
in **5.20 s**.

**A three-skeleton pack is winnable, and it is close.** Fought for real in both
directions, by a level-1 Barbarian in the starting kit who is given exactly one
rule of footwork — back off when two of them are on you *and* you are hurt:

> **won in 16.80 s** — all three killed, 32 swings, 10 landed, 79 damage taken,
> **49.3 / 120 health remaining**, 52% of frames spent giving ground.

That is the shape this encounter wanted: a fight the player finishes on under
half health, where standing still and mashing would have killed him (three
skeletons put a passive player down in 5.2 s) and where the footwork is what
makes the difference. The Blood Moor's arc of six skeletons is therefore a
fight rather than a queue.

**Gear.** Equipping a rolled weapon moves `CombatSystem.playerOffense()` — the
object the swing resolves against, not the character screen — and the harness
asserts it. Worth saying plainly, though: at character level 1 the Blood Moor
drop table does **not** reliably produce a weapon better than the starting Hand
Axe. The best magic weapon in one 600-drop search was a Sharp Club at 8–20
against the axe's 10–19: a higher maximum and a *lower* mean. The plumbing
works; the level-1 loot curve is flat.

---

## Known defects

This list is complete to the best of the project's knowledge and is not
softened. If something here reads as a dealbreaker for your use, it probably is.

### There is no audio. None.

`src/audio` was never built. There is no `AudioContext` anywhere in the source —
`'audio'` exists as an `AssetKind` and nothing consumes it. No footsteps, no
weapon impacts, no wind, no fire crackle, no monster vocalisations, no music, no
UI clicks. The game is silent.

It is not merely that the files are missing — **28 `.ogg` files are sitting in
`public/assets/audio/`** (footsteps, combat, foley, loot, UI, from the Kenney RPG
Audio pack, fetched and attributed) and no code anywhere loads or plays a single
one of them. What is missing is the entire playback layer: a mixer, positional
sources, an animation-event-to-sound binding, and ducking.

And the files that *are* there do not cover what the game most needs. Wind, fire
crackle and monster vocalisations would all have to be **synthesised**, because
no CC0 source for them is reachable through this environment's egress proxy. So
this is not a "drop in some files" gap; it is a subsystem plus original sound
design.

### `quality=ultra` fails the tier-parity gate, marginally

`tools/capture/capture.mjs` asserts that quality tiers change the *cost* of the
picture and not the picture: `quality-low` and `quality-ultra` must agree to
0.03 of mean luminance and 0.025 of near-black share. Ultra currently fails this
by a small margin. The suspect is tier-varying `shadowDistance` — 90 m at low
against 240 m at ultra (`src/render/RenderSettings.ts`) — which changes cascade
fit and therefore how much of the frame is in shadow at all. Not diagnosed to a
root cause; not fixed.

### WebGPU is dead on this three.js version, and the fix is not small

`webgpu-backend-check` fails. three.js r185 uploads a `Data3DTexture` as 2D
slices and then binds it through 2D views. Dawn rejects every submit that
touches the froxel fog volumes, so the whole frame fails on the WebGPU backend.
This is not a shim-able mistake like the `swizzle` string that
`render/webgpuCompat.ts` patches: a real fix needs the volumetrics rewritten
onto a **2D slice atlas with hand-written TSL trilinear sampling**. Until then,
**WebGL2 is the backend that works**, and it is what every harness and capture
uses.

### Feet still slide at speed

See the table above. The lock engages for about 88% of a walk cycle and only
about **25%** of a run or sprint, because this rig's run clip has a short contact
and the pin cannot be taken until the plant test has agreed for two frames. The
foot the lock *is* holding is genuinely still — hundredths of a metre per second
— but at a run the slower foot still moves at 0.56–0.60 of body speed. Walking
looks right. Running slides. Closing that gap is real remaining work.

### The level-1 loot curve is flat

Six hundred drops rolled off the Blood Moor table, filtered to magic-or-better
weapons the character can actually use, and the best of them was a Sharp Club at
8–20 physical against the starting Hand Axe's 10–19 — a higher maximum and a
lower mean. Affixes, generation, equipping and the path from an item to the
damage the swing carries are all verified and all work; what is missing is a
reason to care about a drop in the first area. The base-item and affix weights
want retuning against what the player starts holding.

### No item icons

There is no item art in this project. Every inventory item is drawn as a
quality-coloured plate with its name on it. At a 34-pixel grid cell that is
arguably more readable than a 32×32 sprite would be, but it is not what an
inventory is supposed to look like.

### Other things a new reader should not be misled by

- **One class, one act.** The Barbarian, and the first three areas of Act I.
  There is no Sorceress, no Amazon, no Act II, no Andariel, no waypoints, no
  town portal, no multiplayer, no difficulty tiers.
- **Attack rating from gear cannot rescue a miss.** The base swing rolls to hit
  against the provider's rating; if that roll misses, no `combat:hit` fires and
  the RPG bridge has nothing to augment. Gear AR shows on the sheet and feeds
  the augmentation packet, but it does not widen the base hit window.
- **The art is stylized low-poly, on purpose.** See
  [Assets](#assets-and-attribution).
- **Enemies do not respawn, and cannot be revived.** A zone's spawn table runs
  once on load; the Blood Moor's six skeletons are all the skeletons there will
  ever be until you leave and come back. And there is a latent bug behind that:
  `Vitals.revive()` restores a dead enemy's health pool but does not touch
  `EnemyBase`'s own state machine, so the "revived" enemy is still in
  `state === 'dead'`, keeps sinking, and is culled by `EnemyDirector` 4.6 s
  after it originally fell — while `alive` reads `true` the whole time. Nothing
  in the game currently revives an enemy, so nothing is visibly broken, but any
  future resurrect (a Necromancer, a boss phase) will hit this.
- **`src/physics/WorldColliders.ts` is dead code**, superseded by
  `ZoneManager`.
- **`npm run capture` and `npm run shots` are two different harnesses.**
  `scripts/capture.mjs` is the older single-shot entry point; `tools/capture/`
  is the deterministic seeded one.
- **`render.gbuffer.surface` is deliberately unsatisfiable.** This is a forward
  renderer, so SSR runs on scalar defaults (roughness 0.34, metalness 0) rather
  than a real G-buffer. The audit reports it as degraded because it is.
- **Two Mixamo-derived assets are held back** pending a licensing decision and
  are not fetched by a default `npm run assets`.
- **This container has no GPU.** Everything is SwiftShader software
  rasterisation through ANGLE → Vulkan. **Frame rates measured here mean
  nothing** and no performance claim in this repository should be read as one.
  The project has never run on real hardware.

---

## Verification harnesses

The main discipline of this project was **driving the real game rather than
trusting claims**. Every harness in `tools/` spawns `vite preview` over `dist/`
(never the dev server — HMR destroys the execution context mid-run), drives the
built bundle through `window.__d2rim`, and steps frames *inside the page*,
because a Playwright round trip per frame costs more than the frame does under
software rasterisation.

| Harness                     | What it proves                                                                                                                                 |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `verify-kill.mjs`           | **The player can win.** Time-to-kill, swings and landed hits for a Barbarian against each skeleton variant that is moving and fighting back, the same four fights with the best weapon 600 drops will yield, and a three-skeleton pack fight. Three independently booted phases; asserts a budget on all of it. |
| `verify-balance.mjs`        | The other direction: how long a passive player survives one skeleton and three, and that `rpg.offense`/`rpg.defense` are load-bearing.          |
| `verify-player-damage.mjs`  | The two-sided regression guard — a skeleton left alone takes health off the player, a player who refuses to fight dies, and a dead player respawns alive at full health. |
| `verify-combat-loop.mjs`    | The walk a player walks: boot at the camp with no flags, travel camp → moor → den, and prove each zone places its declared spawn table and that skeletons perceive, close, damage and die. |
| `verify-encounter.mjs`      | That the Blood Moor's six skeletons appear on **both** the walk-in and the boot-in path — which differ, and the deferred one was broken.        |
| `verify-zones.mjs`          | Standing, leaking and looking: that the player stands on each zone's real surface, that a full lap returns colliders and records to baseline exactly and a second lap leaves geometry and texture counts flat, and that each zone's capture survives the blank-frame guard. |
| `verify-rpg.mjs`            | A ten-step end-to-end pass over the whole RPG layer, including save/load through the browser's real IndexedDB.                                  |
| `verify-footplant.mjs`      | Per-frame **world** speed of the foot bones, asserted per gait — because foot sliding is the single most visible tell of amateur character work. |
| `verify-enemy-clips.mjs`    | That every skeleton's semantic attack state resolves to a real clip and that its hit events fire.                                              |
| `verify-enemy-damage.mjs`   | The separation at which an enemy swing actually connects with the player.                                                                      |
| `verify-enemy-blade.mjs`    | Where an enemy's blade actually is during its own hit window, relative to the player capsule.                                                   |
| `verify-standoff.mjs`       | That skeletons hold a stand-off instead of burrowing into the player.                                                                          |
| `verify-independent.mjs`    | A from-scratch re-verification written by a reviewing agent, sharing no code with the other harnesses.                                          |
| `verify-followup.mjs`       | A focused re-check of AI approach, the camera arm, foot plants and wall collision.                                                             |
| `measure-enemy-reach.mjs`   | Produces the `reachDuringWindow` numbers in `src/ai/enemies/Skeleton.ts` by sweeping a player capsule against each clip.                        |
| `combat-drive.mjs` / `combat-kill-check.mjs` | That one scripted swing damages a staged skeleton, and that a target at low health dies from a landed hit.                     |
| `scratch-drive.mjs`         | An ad-hoc character harness: holds real keys, steps frames, dumps a deterministic state snapshot.                                               |
| `capture/capture.mjs`       | Deterministic seeded screenshots, built so a broken renderer produces a **failed run** rather than a plausible-looking black image.             |
| `capture/frame-guard.mjs`   | Five near-orthogonal statistics over a decoded PNG, because the worst failure mode of a capture harness is quietly writing a black image.        |
| `capture/exposure-report.mjs` | A nine-percentile luma histogram, crushed/clipped share, and warm/cold hue separation.                                                        |
| `assets/fetch-assets.mjs`   | Idempotent, resumable, SHA-256-locked asset fetching, and the generator behind `public/ATTRIBUTIONS.md`.                                        |

Plus 55 Vitest files under `tests/`, running in a Node environment over the pure
logic: the damage model, the combo state machine, hitbox geometry, the item
generator, the dungeon generator, zone teardown, the render maths (atmosphere,
sky, CSM, GTAO, SSR, denoise, tonemap, colour grade) and the frame-graph wiring.

### A note on harness design

Four failures shaped how these are written, and all four are worth internalising
before adding another:

1. **A staging trick can suppress the thing it is measuring.** An early attempt
   at the player-side kill measurement teleported both fighters to fixed marks
   every frame to hold the duel geometry still. It recorded **27 swings and 0
   landed hits** — not because the game was broken, but because `WeaponHitbox`
   resolves a contact by sweeping the blade from its *previous* pose, and a
   per-frame teleport makes every sweep a zero-length segment. `verify-kill.mjs`
   teleports exactly once, to stage the encounter, and then aims with real
   pointer pixels through `Input.nudgePointer`.
2. **A capture that always succeeds is worse than no capture.** Hence the frame
   guard, the parity gate, and the rule that unknown option names are rejected
   rather than ignored.
3. **A harness can quietly consume the world it measures.** `EnemyDirector`
   splices a corpse out of its list once the corpse has finished sinking. A
   version of `verify-kill.mjs` that fought four duels and then a thirty-second
   pack fight destroyed six skeletons permanently, and every measurement after
   that reported "no fresh target of that variant" — which reads exactly like a
   broken spawn table. Fights now revive their target immediately, and the long
   fight goes last.
4. **A scripted player can be too careful to produce a result.** The pack fight
   gives the player one rule of footwork: back off when two skeletons are on
   him. Applied unconditionally, the player spent a third of the fight retreating
   at full health, took 28 damage in thirty seconds and the clock expired with
   one skeleton still up. That is a stalemate, and a stalemate answers neither
   "can he win" nor "is it dangerous". The rule is now conditioned on actually
   being hurt.

---

## Assets and attribution

**`public/ATTRIBUTIONS.md` is the authoritative list**, generated by
`tools/assets/fetch-assets.mjs` from `tools/assets/manifest.json`. Do not edit it
by hand. Every entry carries its author, source, retrieval URL, local path, byte
size and SHA-256, and the licence text at source is archived under
`public/assets/licenses/` so the claim can be verified offline.

146 catalogued entries. 144 ship: **143 CC0-1.0** and exactly one CC-BY-4.0 (the
Fox model, which doubles as the test case proving the attribution generator
handles a licence that actually requires attribution). The remaining two are
Mixamo-derived and are **held back pending a licensing decision** — they are not
downloaded by a default `npm run assets` and must not appear in a public build
until a human decides.

Principal sources: Kay Lousberg / **KayKit** (65 entries — characters, props,
dungeon kit), **Poly Haven** (32 — HDRIs and PBR textures), **Kenney** RPG Audio
(21, unused), **ambientCG** (11), `@pmndrs/assets` (6), and a handful from the
three.js and Khronos sample sets.

**Why the art direction is "stylized-AAA" and not photoreal.** The KayKit
character and prop art is stylized low-poly — chunky silhouettes, flat-ish
forms, no photoscanned detail. That is what is actually in the box, so the
renderer is aimed at making stylized geometry read as grimdark and expensive:
AgX tonemapping on a `grimdark` look, a locked exposure, per-zone grading,
volumetric fog, light shafts, GTAO, SSR and cascaded shadows doing the work that
surface detail would otherwise do. Photoreal was never available and pretending
otherwise would have produced a worse-looking game.

---

## Licence

MIT — see [LICENSE](./LICENSE). The licence covers this repository's code. Third
party assets carry their own licences; see `public/ATTRIBUTIONS.md`.
