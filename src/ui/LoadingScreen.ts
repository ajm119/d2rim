/**
 * @module ui/LoadingScreen
 *
 * The first thing anybody sees, and — until this existed — the longest anybody
 * looked at nothing.
 *
 * ## Why it is not a `GameModule`
 *
 * Every other screen in `src/ui` is a module, initialised by the engine and
 * attached to `UiManager`'s overlay root. This one cannot be, because the thing
 * it has to report on is *the engine starting up*. By the time module `init`
 * runs, the renderer has been acquired and half the assets are in flight —
 * precisely the interval the player spends staring at a black rectangle
 * wondering whether the page is broken.
 *
 * So it is a plain class, constructed in `main.ts` before the engine, against
 * an `EventBus` that predates both. It takes over the static `#boot` element
 * that `index.html` already ships, which means there is never a frame where
 * neither the curtain nor the screen is present.
 *
 * ## Real progress, not a fake animation
 *
 * The percentage is a weighted sum over four phases with fixed budgets. The
 * weights are not equal because the phases are not: on a cold cache the asset
 * download dominates everything else combined, and a bar that gave device
 * acquisition a quarter of its travel would sit at 25% for nine tenths of the
 * wait. Within a phase, progress is whatever real signal that phase publishes —
 * `assets:batch` counts, module init counts — and phases with no countable work
 * simply hold their floor until the next one starts.
 *
 * The number is also monotonic by construction. Asset batches overlap and
 * modules report their own counts, so the naive combination goes backwards;
 * a progress bar that goes backwards is read as a bug even when it is telling
 * the truth.
 *
 * ## Failure is visible
 *
 * A WebGPU device that never arrives, or an asset that 404s, used to leave the
 * curtain sitting at "initialising renderer" forever with the reason in a
 * console nobody has open. `boot:failed` and `assets:error` are rendered on
 * screen, in the page, with the message.
 */

import type { EventBus, BootPhase, Unsubscribe } from '../core/EventBus';

/** Fraction of the bar each phase owns, in order. Must sum to 1. */
export const PHASE_WEIGHTS: Readonly<Record<BootPhase, number>> = {
  // Device acquisition: fast when it works, and when it does not it fails
  // rather than crawls, so it earns very little of the bar.
  renderer: 0.08,
  // Module init is where asset download and shader compilation live. It is the
  // overwhelming majority of a cold start.
  modules: 0.62,
  // Terrain, props and collider construction for the starting zone.
  zone: 0.28,
  ready: 0.02,
};

export const PHASE_ORDER: readonly BootPhase[] = ['renderer', 'modules', 'zone', 'ready'];

/** Cumulative fraction at which a phase begins. */
export function phaseFloor(phase: BootPhase): number {
  let total = 0;
  for (const entry of PHASE_ORDER) {
    if (entry === phase) break;
    total += PHASE_WEIGHTS[entry];
  }
  return total;
}

/**
 * Turn a module name into something a player can read.
 *
 * `render.materials` means nothing to anyone outside this repository, but
 * "loading textures" tells the player both that progress is real and roughly
 * how much of it is left. Unmatched names fall back to a generic label rather
 * than leaking an internal identifier onto the screen.
 */
export function describeModule(name: string): string {
  // Keyed off the real `GameModule.name` values. Ordered most-specific first.
  const table: ReadonlyArray<readonly [RegExp, string]> = [
    [/^AssetManager$/, 'opening the asset registry'],
    [/^render\.settings$/, 'choosing quality settings'],
    [/^render\.materials$/, 'loading textures'],
    [/^(Sky|Lighting|IBL|TimeOfDay)$/, 'building the sky'],
    [/^render\.(post|gtao|ssr|volumetrics|lightShafts|bridges)$/, 'compiling shaders'],
    [/^render\./, 'compiling shaders'],
    [/^physics\./, 'starting the physics world'],
    [/^world\.zones$/, 'building terrain'],
    [/^world\./, 'opening the ways between zones'],
    [/^scene\./, 'building terrain'],
    [/^(combat|character\.|ai\.)/, 'waking the Barbarian'],
    [/^(rpg|quest\.)/, 'preparing the act'],
    [/^ui\.|Overlay$/, 'drawing the interface'],
  ];
  for (const [pattern, label] of table) if (pattern.test(name)) return label;
  return 'preparing the world';
}

/**
 * Player-facing name for a zone construction sub-phase.
 *
 * `ZoneManager` emits terse internal identifiers — `build`, `colliders`,
 * `place` — which are the right vocabulary for a log and the wrong one for a
 * loading screen. Zone construction is the second-longest phase of a cold
 * start, so these are the words a player reads for several seconds.
 */
export function describeZonePhase(phase: string): string {
  const table: Readonly<Record<string, string>> = {
    construct: 'raising the terrain',
    build: 'building the world',
    colliders: 'setting the ground underfoot',
    place: 'placing props and enemies',
    ready: 'entering the world',
  };
  return table[phase] ?? 'building the world';
}

export interface LoadingScreenOptions {
  /** Element to take over. Defaults to `#boot`. */
  readonly element?: HTMLElement | null;
  /** Remove the screen automatically once boot reports ready. Default true. */
  readonly autoHide?: boolean;
}

export class LoadingScreen {
  readonly #root: HTMLElement;
  readonly #bar: HTMLElement;
  readonly #percent: HTMLElement;
  readonly #label: HTMLElement;
  readonly #detail: HTMLElement;
  readonly #unsubscribes: Unsubscribe[] = [];

  #fraction = 0;
  #phase: BootPhase = 'renderer';
  #hidden = false;
  #failed = false;
  #autoHide = true;
  /** Assets that failed, deduplicated. Shown as a warning, not a fatal error. */
  readonly #assetErrors = new Set<string>();

  constructor(events: EventBus, options: LoadingScreenOptions = {}) {
    this.#autoHide = options.autoHide ?? true;
    const existing = options.element ?? document.getElementById('boot');
    this.#root = existing instanceof HTMLElement ? existing : document.createElement('div');
    if (this.#root.parentElement === null) document.body.appendChild(this.#root);

    this.#root.textContent = '';
    this.#root.classList.add('d2rim-loading');
    this.#root.setAttribute('role', 'status');
    this.#root.setAttribute('aria-live', 'polite');
    this.#injectStyles();

    const panel = document.createElement('div');
    panel.className = 'd2rim-loading__panel';

    const title = document.createElement('div');
    title.className = 'd2rim-loading__title';
    title.textContent = 'Diablo II — Act I';

    this.#label = document.createElement('div');
    this.#label.className = 'd2rim-loading__label';
    this.#label.textContent = 'initialising renderer';

    const track = document.createElement('div');
    track.className = 'd2rim-loading__track';
    this.#bar = document.createElement('div');
    this.#bar.className = 'd2rim-loading__bar';
    track.appendChild(this.#bar);

    const row = document.createElement('div');
    row.className = 'd2rim-loading__row';
    this.#percent = document.createElement('span');
    this.#percent.className = 'd2rim-loading__percent';
    this.#percent.textContent = '0%';
    this.#detail = document.createElement('span');
    this.#detail.className = 'd2rim-loading__detail';
    row.append(this.#percent, this.#detail);

    panel.append(title, track, row, this.#label);
    this.#root.appendChild(panel);

    this.#subscribe(events);
  }

  /** Current progress in `[0,1]`. Exposed for headless verification. */
  get fraction(): number {
    return this.#fraction;
  }

  get phase(): BootPhase {
    return this.#phase;
  }

  get hidden(): boolean {
    return this.#hidden;
  }

  /** The text a player is currently being shown. Exposed for tests. */
  get label(): string {
    return this.#label.textContent ?? '';
  }

  #subscribe(events: EventBus): void {
    const on = <K extends Parameters<EventBus['on']>[0]>(
      name: K,
      handler: Parameters<EventBus['on']>[1],
    ): void => {
      this.#unsubscribes.push(events.on(name, handler as never));
    };

    on('boot:phase', (payload) => {
      const { phase, label, completed, total } = payload as {
        phase: BootPhase;
        label: string;
        completed: number;
        total: number;
      };
      // Zone construction happens *inside* `world.zones`'s module init, so the
      // engine keeps emitting `modules` events after the zone has started
      // building. Without this guard the screen tells the player it is
      // "preparing the world" while it is three quarters of the way through
      // laying out terrain, and the label flickers between two narratives.
      // Phases only ever move forward, exactly like the percentage.
      if (PHASE_ORDER.indexOf(phase) < PHASE_ORDER.indexOf(this.#phase)) return;
      this.#phase = phase;
      const within = total > 0 ? completed / total : 0;
      this.#advance(phaseFloor(phase) + PHASE_WEIGHTS[phase] * within);
      // Module names are internal; everything else is already player-facing.
      this.#setLabel(phase === 'modules' && total > 0 ? describeModule(label) : label);
      this.#detail.textContent = total > 0 ? `${Math.min(completed, total)} / ${total}` : '';
    });

    on('boot:failed', (payload) => {
      const { phase, message } = payload as { phase: BootPhase; message: string };
      this.fail(`${phase} failed`, message);
    });

    // Asset batches run inside module init, so they refine the `modules` phase
    // rather than owning a phase of their own.
    on('assets:batch', (payload) => {
      const { completed, total, fraction } = payload as {
        completed: number;
        total: number;
        fraction: number;
      };
      if (this.#phase !== 'modules' || total === 0) return;
      this.#advance(phaseFloor('modules') + PHASE_WEIGHTS.modules * fraction);
      this.#setLabel('downloading assets');
      this.#detail.textContent = `${completed} / ${total}`;
    });

    on('assets:error', (payload) => {
      const { key, message } = payload as { key: string; message: string };
      // Not fatal on its own — `preload` deliberately collects failures and
      // carries on, and a missing prop is survivable where a missing device is
      // not. It is still shown, because silently loading a broken world is how
      // you get a bug report that says "some rocks are black".
      this.#assetErrors.add(key);
      console.warn(`[LoadingScreen] asset failed: ${key} — ${message}`);
      this.#detail.textContent = `${this.#assetErrors.size} asset${
        this.#assetErrors.size === 1 ? '' : 's'
      } failed`;
      this.#detail.classList.add('is-warning');
    });

    on('zone:loadStart', (payload) => {
      const { zoneId } = payload as { zoneId: string };
      this.#phase = 'zone';
      this.#advance(phaseFloor('zone'));
      this.#setLabel('building terrain');
      this.#detail.textContent = zoneId;
    });

    // The zone reports its own named sub-phases (heightfield, scatter,
    // colliders...), which is exactly the granularity this needs: zone
    // construction is seconds long and otherwise completely opaque.
    on('zone:loadProgress', (payload) => {
      const { phase, progress } = payload as { phase: string; progress: number };
      if (this.#phase !== 'zone') return;
      this.#advance(phaseFloor('zone') + PHASE_WEIGHTS.zone * Math.min(1, Math.max(0, progress)));
      this.#setLabel(describeZonePhase(phase));
    });

    on('zone:loaded', () => {
      this.#advance(phaseFloor('ready'));
      this.#setLabel('entering the world');
      this.#detail.textContent = '';
    });

    on('engine:ready', () => {
      if (this.#autoHide) this.finish();
    });
  }

  /** Progress is clamped monotonic: it may stall, but it may never retreat. */
  #advance(fraction: number): void {
    if (this.#failed) return;
    const next = Math.max(this.#fraction, Math.min(1, fraction));
    if (next === this.#fraction) return;
    this.#fraction = next;
    const percent = Math.round(next * 100);
    this.#bar.style.width = `${percent}%`;
    this.#percent.textContent = `${percent}%`;
  }

  #setLabel(text: string): void {
    if (this.#failed) return;
    this.#label.textContent = text;
  }

  /** Show a fatal error in the page rather than hanging at a percentage. */
  fail(heading: string, message: string): void {
    this.#failed = true;
    this.#root.classList.add('is-failed');
    this.#label.textContent = heading;
    this.#detail.textContent = '';
    this.#percent.textContent = '';
    const detail = document.createElement('pre');
    detail.className = 'd2rim-loading__error';
    detail.textContent = message;
    this.#root.querySelector('.d2rim-loading__panel')?.appendChild(detail);
  }

  /** Fade out and detach. Idempotent. */
  finish(): void {
    if (this.#hidden || this.#failed) return;
    this.#advance(1);
    this.#hidden = true;
    this.#root.classList.add('is-done');
    document.body.classList.add('d2rim-ready');
    const remove = (): void => {
      this.dispose();
      this.#root.remove();
    };
    // Matches the CSS transition; the timeout is the fallback for a browser
    // that never fires `transitionend` because the element was already hidden.
    this.#root.addEventListener('transitionend', remove, { once: true });
    setTimeout(remove, 900);
  }

  dispose(): void {
    for (const off of this.#unsubscribes) off();
    this.#unsubscribes.length = 0;
  }

  /**
   * Styles live here rather than in `index.html` so the screen is one
   * self-contained unit that a test can construct against a bare DOM.
   *
   * The look is deliberately of a piece with the game: parchment-and-ember on
   * near-black, a serif display face for the title, and a bar that glows rather
   * than fills flat. A default browser progress element in front of a grimdark
   * dungeon crawler announces "unfinished tech demo" before the player has seen
   * a single frame.
   */
  #injectStyles(): void {
    const id = 'd2rim-loading-styles';
    if (document.getElementById(id) !== null) return;
    const style = document.createElement('style');
    style.id = id;
    style.textContent = `
.d2rim-loading {
  position: fixed;
  inset: 0;
  z-index: 60;
  display: grid;
  place-items: center;
  background:
    radial-gradient(120% 90% at 50% 15%, #241611 0%, #0d0907 55%, #05070a 100%);
  transition: opacity 520ms ease;
  opacity: 1;
}
.d2rim-loading.is-done { opacity: 0; pointer-events: none; }
.d2rim-loading__panel {
  width: min(560px, 78vw);
  display: flex;
  flex-direction: column;
  gap: 14px;
  text-align: center;
}
.d2rim-loading__title {
  font: 600 26px/1.2 "Trajan Pro", "Cinzel", Georgia, "Times New Roman", serif;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: #c9a227;
  text-shadow: 0 0 18px rgba(201, 162, 39, 0.35), 0 2px 0 #000;
}
.d2rim-loading__track {
  position: relative;
  height: 6px;
  border-radius: 3px;
  background: #16100c;
  border: 1px solid #33241a;
  overflow: hidden;
}
.d2rim-loading__bar {
  height: 100%;
  width: 0%;
  border-radius: 3px;
  background: linear-gradient(90deg, #6d2b1a 0%, #c0563a 55%, #e8a04a 100%);
  box-shadow: 0 0 12px rgba(224, 120, 60, 0.6);
  transition: width 260ms ease;
}
.d2rim-loading__row {
  display: flex;
  justify-content: space-between;
  font: 11px/1.4 ui-monospace, Menlo, Consolas, monospace;
  letter-spacing: 0.16em;
  color: #7a6a5c;
}
.d2rim-loading__percent { color: #d8c9b4; }
.d2rim-loading__detail.is-warning { color: #d08a4a; }
.d2rim-loading__label {
  font: 12px/1.6 ui-monospace, Menlo, Consolas, monospace;
  letter-spacing: 0.24em;
  text-transform: uppercase;
  color: #8c7c6a;
  min-height: 1.6em;
}
.d2rim-loading.is-failed .d2rim-loading__label { color: #ff9a9a; }
.d2rim-loading.is-failed .d2rim-loading__bar {
  background: #6d2b2b;
  box-shadow: none;
}
.d2rim-loading__error {
  margin: 8px 0 0;
  padding: 12px 14px;
  max-height: 34vh;
  overflow: auto;
  text-align: left;
  border-radius: 6px;
  border: 1px solid #6d2b2b;
  background: #140d0d;
  color: #ffb4b4;
  font: 12px/1.6 ui-monospace, Menlo, Consolas, monospace;
  white-space: pre-wrap;
}
/* The old static curtain must not double up with this screen. */
body.d2rim-ready .d2rim-loading { opacity: 0; pointer-events: none; }
`;
    document.head.appendChild(style);
  }
}

