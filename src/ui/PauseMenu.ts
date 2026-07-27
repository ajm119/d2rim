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
 */

import type { GameContext, GameModule } from '../core/types';
import { QuestSystemKey, type QuestSystem } from '../quest/QuestSystem';
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

    panel.appendChild(
      el('div', headingStyle(`margin-top:6px;border-top:1px solid ${UI.border};padding-top:14px;`), 'Quest log'),
    );
    const log = el('div', 'display:flex;flex-direction:column;gap:2px;');
    this.#log = log;
    panel.appendChild(log);

    this.root.appendChild(panel);
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
