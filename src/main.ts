/**
 * @module main
 *
 * Application entry point. Wires the canvas to an {@link Engine}, registers the
 * baseline modules, and publishes `window.__d2rim` for headless tooling.
 */

import * as THREE from 'three/webgpu';

import { AssetManager, AssetManagerKey } from './assets/AssetManager';
import { CameraRig } from './character/CameraRig';
import { FootIK } from './character/FootIK';
import { PlayerController } from './character/PlayerController';
import { CombatSystem } from './combat/CombatSystem';
import { CombatFeedback } from './combat/Feedback';
import { Engine } from './core/Engine';
import { EventBus } from './core/EventBus';
import type { GameContext } from './core/types';
import { PhysicsWorld } from './physics/PhysicsWorld';
import { DenOfEvilQuest } from './quest/DenOfEvil';
import { NpcSystem } from './quest/NPC';
import { LootSystem } from './rpg/Loot';
import { RpgSystem } from './rpg/RpgSystem';
import { buildFrameGraph, type FrameGraph } from './render/FrameGraph';
import { RENDER_TIERS, qualityFromUrl } from './render/RenderSettings';
import { collectMemoryReport, formatMemoryReport } from './render/MemoryReport';
import { BloodMoor } from './scene/BloodMoor';
import { DenOfEvil } from './scene/DenOfEvil';
import { RogueEncampment } from './scene/RogueEncampment';
import { CombatHud } from './ui/CombatHud';
import { DebugOverlay, statsRequested } from './ui/DebugOverlay';
import { DialogueOverlay } from './ui/DialogueOverlay';
import { InventoryScreen } from './ui/InventoryScreen';
import { LoadingScreen } from './ui/LoadingScreen';
import { PauseMenu } from './ui/PauseMenu';
import { RpgHud } from './ui/RpgHud';
import { SkillTreeScreen } from './ui/SkillTreeScreen';
import { UiManager } from './ui/UiManager';
import { VendorScreen } from './ui/VendorScreen';
import { PortalSystem } from './world/Portal';
import type { Zone } from './world/Zone';
import { ZoneManager } from './world/ZoneManager';

/**
 * The handle headless capture and automated tests drive the game through.
 *
 * The intended sequence is: `await window.__d2rim.ready`, then
 * `await engine.stepFrames(n)` for a deterministic world state, then capture.
 * `stepFrames` puts the engine into manual mode, so no stray animation frame
 * can advance the world between stepping and capturing.
 */
export interface D2RimGlobal {
  readonly engine: Engine;
  /** Throws until `ready` resolves; see {@link Engine.context}. */
  readonly ctx: GameContext;
  readonly ready: Promise<void>;
  /**
   * The live `three/webgpu` namespace.
   *
   * Exposed for `tools/capture` shot setup scripts, which are strings evaluated
   * in the page and therefore cannot import anything. Without it a shot can
   * only move the camera; with it a shot can restage lighting, swap materials
   * or move objects to isolate exactly the thing under review.
   *
   * This is the same module instance the game uses, so `instanceof` holds. It
   * costs no extra bundle weight — three is already loaded — and game code must
   * never read it back off `window`.
   */
  readonly three: typeof THREE;
  /**
   * The assembled renderer.
   *
   * Capture scripts and the debug console reach subsystems through this rather
   * than through the `ServiceLocator`, because it is typed: `d2rim.render.post`
   * is a `PostStack`, where `services.get('render.post')` needs a cast.
   */
  readonly render: FrameGraph;
  /**
   * The **active zone**.
   *
   * Was a `BloodMoor` instance for the whole session; is now whichever zone
   * `ZoneManager` currently has loaded, and is `null` while a transition is in
   * flight. Capture shots that reach for `d2rim.scene.field` or
   * `d2rim.scene.constructor.defaultCamera` therefore need the moor to be the
   * loaded zone — see `?zone=` below.
   */
  readonly scene: Zone | null;
  /** Zone registry and travel. `d2rim.zones.travelTo('denOfEvil')` from a shot. */
  readonly zones: ZoneManager;
}

declare global {
  interface Window {
    __d2rim?: D2RimGlobal;
  }
}

function requireCanvas(): HTMLCanvasElement {
  const canvas = document.getElementById('viewport');
  if (!(canvas instanceof HTMLCanvasElement)) {
    throw new Error('[d2rim] #viewport canvas is missing from the document');
  }
  return canvas;
}

/**
 * Report a fatal boot failure in the page itself.
 *
 * A blank canvas with an error buried in the console is the worst possible
 * failure mode for a graphics project, particularly in headless capture where
 * nobody is reading the console.
 */
function reportFatal(error: unknown): void {
  console.error('[d2rim] fatal boot failure:', error);
  const panel = document.createElement('pre');
  panel.setAttribute(
    'style',
    'position:fixed;inset:24px;z-index:100;overflow:auto;margin:0;padding:20px;' +
      'border-radius:8px;border:1px solid #6d2b2b;background:#140d0d;color:#ffb4b4;' +
      'font:13px/1.6 ui-monospace,Menlo,Consolas,monospace;white-space:pre-wrap;',
  );
  panel.textContent = `d2rim failed to start\n\n${
    error instanceof Error ? (error.stack ?? error.message) : String(error)
  }`;
  document.body.appendChild(panel);
}

const canvas = requireCanvas();

/**
 * `?autostart=0` boots the engine without starting the rAF loop.
 *
 * Capture harnesses need this: with the loop running, frames are rendered
 * between `await ready` and the first `stepFrames` call, so the world state at
 * capture time depends on how long the harness took to issue its next command.
 * Suppressing autostart makes the caller the sole source of frame advancement.
 */
const params = new URLSearchParams(window.location.search);
const autoStart = params.get('autostart') !== '0';

/**
 * `?zone=<id>` chooses the starting zone. Default: the Rogue Encampment.
 *
 * The act now has three zones and the hub is where a session begins, so `/`
 * boots into the camp. The one thing that changes for tooling is that
 * `d2rim.scene` is the *active* zone rather than always a `BloodMoor`: the shots
 * in `tools/capture/shots.json` reach for `d2rim.scene.field.heightAt` and
 * `d2rim.scene.constructor.defaultCamera`, so they need `?zone=bloodMoor`, which
 * restores exactly the previous boot state.
 */
const startZone = params.get('zone') ?? 'encampment';
/** `?enemies=0` boots the zones unpopulated, for capture and for drive tests. */
const enemies = params.get('enemies') !== '0';
/** `?fade=0` removes the transition wait, which a headless harness does not want. */
const fadeSeconds = params.get('fade') === '0' ? 0 : 0.35;

/**
 * The bus is created here rather than by the engine so that the loading screen
 * can subscribe *before* the engine exists. `Engine.#boot` emits its first
 * `boot:phase` synchronously from the constructor, so a screen wired up
 * afterwards would miss the renderer phase entirely — which is exactly the
 * phase that stalls when WebGPU adapter acquisition goes wrong.
 */
const events = new EventBus();
const loadingScreen = new LoadingScreen(events);

// The tier's pixel-ratio cap, resolved before the engine exists.
//
// `RenderSettings` owns this value and re-applies it in `init`, but the engine
// is constructed first and starts creating the renderer immediately, so a cap
// applied later would let the very first frame be built at the device's full
// ratio — on a Retina laptop, four times the fragments — and then resize. That
// is a visible hitch on precisely the machines the cap exists to protect, so
// the URL is read once here and the two agree from the first frame.
const bootTier = RENDER_TIERS[qualityFromUrl()];

const engine = new Engine({
  canvas,
  autoStart,
  events,
  pixelRatioCap: bootTier.pixelRatioCap,
  // GPU timer queries only when someone is looking. They cost a query pair per
  // render pass and, on some drivers, a pipeline flush — worth it to tell a
  // CPU-bound frame from a GPU-bound one, not worth it by default.
  renderer: { timestamps: statsRequested() },
});

// Registration order *is* frame order — see `render/FrameGraph.ts`, which owns
// the ordering constraints and the reasoning behind every one of them.
//
// Three fixed points bracket it:
//   - AssetManager first: everything downstream resolves textures and models
//     through it, and it must have registered before any `init` asks for it.
//   - the frame graph next: twelve render modules in dependency order.
//   - content last, so the zones resolve a fully-built renderer.
const render = buildFrameGraph();

/**
 * The act's zones, registered as factories.
 *
 * Registration is free — nothing is built and no asset is fetched until a zone
 * is travelled to — so all three are declared here and exactly one of them is
 * ever resident.
 *
 * `driveCamera: false` and `controlHero: false` hand the two things the Blood
 * Moor used to own — the camera pose and the Barbarian's mixer — to the gameplay
 * modules. `loadHero: false` is the zone-system addition: the player module owns
 * the figure and keeps it across transitions, so the scene must not build a
 * second one.
 */
const zones = new ZoneManager({ startZone, startEntry: '', fadeSeconds, enemies });
zones.register('encampment', () => new RogueEncampment({ settings: render.settings }));
zones.register(
  'bloodMoor',
  () =>
    new BloodMoor({
      settings: render.settings,
      driveCamera: false,
      controlHero: false,
      loadHero: false,
    }),
);
zones.register('denOfEvil', () => new DenOfEvil());

engine.add(new AssetManager());
for (const module of render.modules) engine.add(module);

// Gameplay, in dependency order. Physics first so its service exists before
// anything resolves it; the zone manager next, because its zones build
// colliders against that world during their own load and the player asks where
// the ground is immediately afterward; then the player, whose capsule needs a
// world to stand on; then foot IK, which re-poses the legs after the animation
// graph has run; then the camera, which must observe a settled character.
// `lateUpdate` order follows registration order, so this list *is* the frame
// order.
engine.add(new PhysicsWorld());
// Replaces the old `WorldColliders` module, which read one hard-coded scene
// module by name. Zones build their own colliders now, and the manager tracks
// and reclaims them per zone — a per-zone lifetime that a single global collider
// build cannot express.
engine.add(zones);
// Combat *before* the player: `PlayerController.init` checks for a registered
// `combat` service and stands its placeholder attack input down when it finds
// one, so the service has to exist by then. Combat binds to the player lazily
// in return, which is the price of that ordering and is paid in one method.
engine.add(new CombatSystem());
engine.add(new PlayerController());
engine.add(new FootIK());
engine.add(new CameraRig());
// Feedback last of the gameplay modules: its `lateUpdate` adds the camera
// shake, and it must run after `CameraRig` has placed the camera or the rig
// simply overwrites it.
engine.add(new CombatFeedback());
// Portals read the player's settled position, so they follow him. Their volumes
// are built by the zone manager, inside its collider-tracking window; this
// module only owns the prompt and the interact key.
engine.add(new PortalSystem());

// The RPG layer. `RpgSystem` after combat, because it binds to the combat
// service and resizes its vitals pools from the derived character sheet; loot
// after that, because the drop it creates is handed to the RPG system as its
// receiver; the quest and NPC modules last, because both resolve the quest
// system the RPG module registered.
engine.add(new RpgSystem());
engine.add(new LootSystem());
engine.add(new DenOfEvilQuest());
engine.add(new NpcSystem());

// UI. `UiManager` first: every screen registers itself with it during `init`,
// and it owns the overlay root they attach to.
engine.add(new UiManager());
engine.add(new CombatHud());
engine.add(new RpgHud());
engine.add(new InventoryScreen());
engine.add(new SkillTreeScreen());
engine.add(new VendorScreen());
engine.add(new DialogueOverlay());
engine.add(new PauseMenu());
engine.add(new DebugOverlay());

const ready = engine.ready
  .then(() => {
    // Pointer lock requires a user gesture, so it is requested on click rather
    // than at boot. Harmless in headless runs, where no click ever arrives.
    canvas.addEventListener('click', () => engine.input.requestPointerLock());
    document.body.classList.add('d2rim-ready');
  })
  .catch((error: unknown) => {
    // The loading screen is already on top of the page and already styled, so
    // it is the right surface for this. `reportFatal` stays as the backstop for
    // a failure early enough that the screen itself never got built.
    loadingScreen.fail(
      'failed to start',
      error instanceof Error ? (error.stack ?? error.message) : String(error),
    );
    reportFatal(error);
    throw error;
  });

/**
 * `window.__d2rimMemory()` — what the GPU is actually holding, in megabytes.
 *
 * This is the diagnostic that did not exist when the deployed build started
 * dying with Chromium's out-of-memory "Error code: 5", and its absence is why
 * the cause took measurement rather than reading to find. It is deliberately
 * callable from a plain browser console with no flags, because the machine that
 * matters is the player's and the only instrument available there is devtools.
 *
 * `?mem=1` additionally dumps a report to the console shortly after the first
 * frame, for the case where someone can reproduce a crash but cannot be talked
 * through typing a function name.
 */
function memoryReport(): ReturnType<typeof collectMemoryReport> {
  return collectMemoryReport(engine.scene, {
    renderer: engine.context.renderer.three as unknown as {
      info?: { render?: { drawCalls?: number; triangles?: number } };
    },
    // Without this the report misses the entire terrain and prop material set,
    // which is bound through TSL nodes rather than through `material.map`.
    assets: engine.context.services.tryGet(AssetManagerKey) ?? null,
  });
}

declare global {
  interface Window {
    __d2rimMemory?: () => ReturnType<typeof collectMemoryReport>;
  }
}
window.__d2rimMemory = memoryReport;

if (params.get('mem') === '1') {
  void ready.then(() => {
    // One frame of slack so lazily-created targets (TAA history, bloom
    // pyramid) exist before they are counted.
    setTimeout(() => console.info(formatMemoryReport(memoryReport())), 500);
  });
}

window.__d2rim = {
  engine,
  get ctx(): GameContext {
    return engine.context;
  },
  ready,
  three: THREE,
  render,
  get scene(): Zone | null {
    return zones.active;
  },
  zones,
};

// Vite HMR: without an explicit teardown the old engine keeps its rAF loop and
// event listeners alive, and every edit leaks another renderer.
if (import.meta.hot) {
  import.meta.hot.dispose(() => {
    engine.dispose();
    delete window.__d2rim;
  });
}
