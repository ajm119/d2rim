/**
 * @module ui/PauseMenu
 *
 * Escape: resume, save, load, and the quest log.
 *
 * ### It really pauses
 *
 * Opening the menu sets `time.scale` to zero rather than merely hiding the
 * world behind a scrim. A "pause" menu that leaves skeletons swinging is worse
 * than no pause menu at all, because the player has been told they are safe.
 * The renderer keeps running — the engine's loop is unaffected by the time
 * scale — so the scene behind the panel is still live, just frozen.
 *
 * The previous scale is restored on close rather than assumed to be 1, because
 * `combat/Feedback` drives hit-stop by scaling time and a menu opened during a
 * hit-stop must not cancel it.
 *
 * ### Save and load are asynchronous and say so
 *
 * `IndexedDbSaveStore.write` resolves when the transaction commits. The button
 * reports the outcome, because a save button that gives no feedback is a save
 * button players press four times.
 *
 * ### Brightness
 *
 * The menu owns the one display setting the art direction cannot make for the
 * player: a brightness trim in stops. It writes through
 * `PostStack.composite.setExposureCompensation` — *not* `setExposure`, which
 * belongs to the art direction and is overwritten on every zone transition by
 * `ZoneManager`'s `ZoneGrade` trim — and persists to local storage immediately,
 * so it survives a crash as well as a quit. See `render/DisplaySettings`.
 *
 * The slider is live while the menu is open: the world is still rendering
 * behind the scrim (only `time.scale` is zero), so the player can see what they
 * are choosing rather than adjusting blind and closing the menu to find out.
 */

import type { GameContext, GameModule } from '../core/types';
import { QuestSystemKey, type QuestSystem } from '../quest/QuestSystem';
import {
  EXPOSURE_STOPS_MAX,
  EXPOSURE_STOPS_MIN,
  clampExposureStops,
  loadExposureStops,
  saveExposureStops,
} from '../render/DisplaySettings';
import { PostStackKey, type PostStack } from '../render/post/PostStack';
import { RpgSystemKey, type RpgSystem } from '../rpg/RpgSystem';
import { AUTOSAVE_SLOT } from '../rpg/SaveGame';
import { buttonStyle, clearChildren, el, hasDom, headingStyle, panelStyle, scrimStyle, screenRoot, UI, Z } from './theme';
import { UiManagerKey, type UiManager, type UiScreen } from './UiManager';

export class PauseMenu implements GameModule, UiScreen {
  readonly name = 'ui.menu';
  readonly id = 'menu' as const;

  readonly root: HTMLElement;

  #ctx: GameContext | null = null;
  #log: HTMLDivElement | null = null;
  #status: HTMLDivElement | null = null;
  #slider: HTMLInputElement | null = null;
  #readout: HTMLDivElement | null = null;
  #stops = 0;
  #savedScale = 1;

  constructor() {
    this.root = screenRoot(scrimStyle(Z.menu));
  }

  init(ctx: GameContext): void {
    this.#ctx = ctx;
    if (!hasDom()) return;
    this.#build();
    ctx.services.tryGet<UiManager>(UiManagerKey)?.register(this);
  }

  dispose(): void {
    this.root.remove();
    this.#ctx = null;
  }

  onOpen(): void {
    const ctx = this.#ctx;
    if (ctx === null) return;
    this.#savedScale = ctx.time.scale;
    ctx.time.scale = 0;
    ctx.events.emit('engine:pause', { paused: true });

    // Show what is actually in force, which is not necessarily what is stored:
    // `?exposure=` deliberately overrides the saved value without persisting.
    const live = ctx.services.tryGet<PostStack>(PostStackKey)?.composite.exposureCompensation;
    if (live !== undefined) {
      this.#stops = clampExposureStops(live);
      if (this.#slider !== null) this.#slider.value = String(this.#stops);
      this.#syncReadout();
    }
  }

  onClose(): void {
    const ctx = this.#ctx;
    if (ctx === null) return;
    ctx.time.scale = this.#savedScale;
    ctx.events.emit('engine:pause', { paused: false });
  }

  refresh(): void {
    const log = this.#log;
    const ctx = this.#ctx;
    if (log === null || ctx === null) return;
    clearChildren(log);

    const quests = ctx.services.tryGet<QuestSystem>(QuestSystemKey);
    const tracked = quests?.tracked() ?? [];
    if (tracked.length === 0) {
      log.appendChild(el('div', `color:${UI.textDim};font:13px/1.6 ${UI.font};`, 'No active quests.'));
      return;
    }
    for (const quest of tracked) {
      log.appendChild(el('div', `color:${UI.accent};font:600 14px/1.5 ${UI.font};`, quest.title));
      log.appendChild(
        el('div', `color:${UI.textDim};font:12px/1.6 ${UI.font};margin-bottom:6px;`, quest.summary),
      );
      for (const objective of quest.objectives) {
        const count = objective.showCount ? ` — ${objective.current} of ${objective.required}` : '';
        log.appendChild(
          el(
            'div',
            `color:${objective.complete ? UI.ok : UI.text};font:13px/1.6 ${UI.font};`,
            `${objective.complete ? '✓' : '·'} ${objective.description}${count}`,
          ),
        );
      }
    }
  }

  /* -- construction -------------------------------------------------------- */

  #build(): void {
    const panel = el(
      'div',
      panelStyle('width:min(560px,92vw);padding:24px 26px;display:flex;flex-direction:column;gap:14px;'),
    );
    panel.addEventListener('pointerdown', (event) => event.stopPropagation());
    this.root.addEventListener('pointerdown', () => this.#ui()?.close('menu'));

    panel.appendChild(el('div', headingStyle('margin:0;'), 'Paused'));

    const buttons = el('div', 'display:flex;gap:10px;flex-wrap:wrap;');
    buttons.append(
      this.#button('Resume', 'menu-resume', () => this.#ui()?.close('menu')),
      this.#button('Save', 'menu-save', () => void this.#save()),
      this.#button('Load', 'menu-load', () => void this.#load()),
      this.#button('Inventory', 'menu-inventory', () => {
        this.#ui()?.close('menu');
        this.#ui()?.open('inventory');
      }),
      this.#button('Skills', 'menu-skills', () => {
        this.#ui()?.close('menu');
        this.#ui()?.open('skills');
      }),
    );
    panel.appendChild(buttons);

    const status = el('div', `color:${UI.textDim};font:12px/1.6 ${UI.font};min-height:20px;`);
    this.#status = status;
    panel.appendChild(status);

    panel.appendChild(this.#buildBrightness());

    panel.appendChild(
      el('div', headingStyle(`margin-top:6px;border-top:1px solid ${UI.border};padding-top:14px;`), 'Quest log'),
    );
    const log = el('div', 'display:flex;flex-direction:column;gap:2px;');
    this.#log = log;
    panel.appendChild(log);

    this.root.appendChild(panel);
  }

  /**
   * The brightness row: label, slider, live readout, reset.
   *
   * Range and step come from `DisplaySettings` rather than being written here,
   * so the URL override, the persisted value and this control can never
   * disagree about what is representable.
   */
  #buildBrightness(): HTMLElement {
    const row = el(
      'div',
      `display:flex;align-items:center;gap:10px;flex-wrap:wrap;` +
        `border-top:1px solid ${UI.border};padding-top:14px;`,
    );
    row.appendChild(
      el('div', `color:${UI.text};font:600 13px/1.5 ${UI.font};min-width:78px;`, 'Brightness'),
    );

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = String(EXPOSURE_STOPS_MIN);
    slider.max = String(EXPOSURE_STOPS_MAX);
    slider.step = '0.05';
    slider.dataset['d2rim'] = 'menu-brightness';
    slider.setAttribute('style', 'flex:1 1 200px;min-width:160px;accent-color:' + UI.accent + ';');
    slider.setAttribute('aria-label', 'Brightness, in stops');
    this.#slider = slider;

    const readout = el(
      'div',
      `color:${UI.textDim};font:12px/1.6 ${UI.font};min-width:96px;text-align:right;`,
    );
    this.#readout = readout;

    // Seed from storage rather than from the post stack: the stack may not have
    // registered yet at `init` time, and storage is the authority the stack was
    // itself seeded from at boot.
    this.#stops = clampExposureStops(loadExposureStops());
    slider.value = String(this.#stops);

    const apply = (stops: number, persist: boolean): void => {
      this.#stops = clampExposureStops(stops);
      slider.value = String(this.#stops);
      this.#applyExposure();
      if (persist) saveExposureStops(this.#stops);
    };

    // `input` for the live preview, `change` for the commit. Persisting on
    // every `input` would write to local storage a hundred times per drag.
    slider.addEventListener('input', () => apply(Number.parseFloat(slider.value), false));
    slider.addEventListener('change', () => apply(Number.parseFloat(slider.value), true));

    row.append(slider, readout, this.#button('Reset', 'menu-brightness-reset', () => apply(0, true)));
    this.#syncReadout();
    return row;
  }

  /**
   * Push the trim into the post stack.
   *
   * Public in effect (the slider and `onOpen` both call it) because the stack
   * is resolved lazily: `PostStack` registers its service during its own `init`
   * and this module is registered after it, but a test may build the menu with
   * no renderer at all and must not crash for it.
   */
  #applyExposure(): void {
    this.#ctx?.services.tryGet<PostStack>(PostStackKey)?.composite.setExposureCompensation(
      this.#stops,
    );
    this.#syncReadout();
  }

  #syncReadout(): void {
    const readout = this.#readout;
    if (readout === null) return;
    const stops = this.#stops;
    readout.textContent =
      stops === 0
        ? 'default'
        : `${stops > 0 ? '+' : ''}${stops.toFixed(2)} EV  ×${(2 ** stops).toFixed(2)}`;
  }

  #button(label: string, id: string, onClick: () => void): HTMLButtonElement {
    const node = el('button', buttonStyle(), label);
    node.dataset['d2rim'] = id;
    node.addEventListener('click', (event) => {
      event.stopPropagation();
      onClick();
    });
    return node;
  }

  async #save(): Promise<void> {
    const rpg = this.#rpg();
    if (rpg === null) return;
    try {
      await rpg.save(AUTOSAVE_SLOT);
      this.#say('Game saved.');
    } catch (error) {
      this.#say(`Could not save: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  async #load(): Promise<void> {
    const rpg = this.#rpg();
    if (rpg === null) return;
    try {
      const loaded = await rpg.load(AUTOSAVE_SLOT);
      this.#say(loaded ? 'Game loaded.' : 'No save found.');
      this.refresh();
    } catch (error) {
      this.#say(`Could not load: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  #say(message: string): void {
    if (this.#status !== null) this.#status.textContent = message;
  }

  #ui(): UiManager | undefined {
    return this.#ctx?.services.tryGet<UiManager>(UiManagerKey);
  }

  #rpg(): RpgSystem | null {
    return this.#ctx?.services.tryGet<RpgSystem>(RpgSystemKey) ?? null;
  }
}
