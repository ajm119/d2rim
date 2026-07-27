/**
 * @module ui/DialogueOverlay
 *
 * The conversation box: a speaker, a line, and a numbered list of replies.
 *
 * ### Bottom third, not centre
 *
 * A conversation is something happening *in the world*, and the player should
 * still be looking at the person they are talking to. So the panel sits in the
 * bottom third with the scene above it left clear, rather than a centred modal
 * that hides the NPC the dialogue is about.
 *
 * ### Keyboard first
 *
 * Replies are numbered and 1-9 selects them; Space or Enter continues a line
 * with no replies; Escape leaves. The mouse works too, but a dialogue that can
 * only be driven with a pointer is a dialogue that stops the game dead every
 * time it opens — and it is untestable from a headless harness.
 */

import type { GameContext, GameModule } from '../core/types';
import { NpcSystemKey, type NpcSystem } from '../quest/NPC';
import type { DialogueView } from '../quest/Dialogue';
import { clearChildren, el, hasDom, panelStyle, UI, Z } from './theme';
import { UiManagerKey, type UiManager, type UiScreen } from './UiManager';

export class DialogueOverlay implements GameModule, UiScreen {
  readonly name = 'ui.dialogue';
  readonly id = 'dialogue' as const;

  readonly root: HTMLElement;

  #ctx: GameContext | null = null;
  #speaker: HTMLDivElement | null = null;
  #text: HTMLDivElement | null = null;
  #choices: HTMLDivElement | null = null;
  #view: DialogueView | null = null;
  readonly #disposers: Array<() => void> = [];

  constructor() {
    this.root = hasDom()
      ? el(
          'div',
          `position:fixed;inset:0;z-index:${Z.dialogue};pointer-events:auto;` +
            'display:flex;align-items:flex-end;justify-content:center;padding-bottom:6vh;' +
            'background:linear-gradient(to bottom,rgba(0,0,0,0) 40%,rgba(4,3,2,0.55) 100%);',
        )
      : ({ style: {} } as unknown as HTMLElement);
  }

  init(ctx: GameContext): void {
    this.#ctx = ctx;
    if (!hasDom()) return;
    this.#build();
    ctx.services.tryGet<UiManager>(UiManagerKey)?.register(this);

    const onKey = (event: KeyboardEvent): void => this.#onKeyDown(event);
    window.addEventListener('keydown', onKey);
    this.#disposers.push(() => window.removeEventListener('keydown', onKey));

    this.#disposers.push(
      ctx.events.on('npc:dialogue', (payload) => this.#onDialogue(payload.view)),
    );
  }

  dispose(): void {
    for (const off of this.#disposers) off();
    this.#disposers.length = 0;
    this.root.remove();
    this.#ctx = null;
  }

  onClose(): void {
    // Closing the panel by any route — Escape, the manager, a zone change —
    // must also end the conversation, or the runner is left mid-tree and the
    // next `talkTo` resumes from a node the player cannot see.
    this.#ctx?.services.tryGet<NpcSystem>(NpcSystemKey)?.endDialogue();
    this.#view = null;
  }

  refresh(): void {
    this.#render(this.#view);
  }

  /* -- construction -------------------------------------------------------- */

  #build(): void {
    this.root.style.display = 'none';

    const panel = el(
      'div',
      panelStyle('width:min(760px,92vw);padding:18px 22px 16px;'),
    );
    panel.dataset['d2rim'] = 'dialogue-panel';

    const speaker = el(
      'div',
      `color:${UI.accent};font:600 13px/1.2 ${UI.font};letter-spacing:0.16em;` +
        'text-transform:uppercase;margin-bottom:8px;',
    );
    const text = el('div', `color:${UI.text};font:15px/1.65 ${UI.font};margin-bottom:14px;`);
    const choices = el('div', 'display:flex;flex-direction:column;gap:6px;');

    panel.append(speaker, text, choices);
    this.root.appendChild(panel);
    this.#speaker = speaker;
    this.#text = text;
    this.#choices = choices;
  }

  /* -- rendering ----------------------------------------------------------- */

  #onDialogue(view: DialogueView | null): void {
    const ui = this.#ctx?.services.tryGet<UiManager>(UiManagerKey);
    this.#view = view;
    if (view === null) {
      ui?.close('dialogue');
      return;
    }
    this.#render(view);
    ui?.open('dialogue');
  }

  #render(view: DialogueView | null): void {
    const speaker = this.#speaker;
    const text = this.#text;
    const choices = this.#choices;
    if (speaker === null || text === null || choices === null || view === null) return;

    speaker.textContent = view.speaker;
    text.textContent = view.text;
    clearChildren(choices);

    if (view.choices.length === 0) {
      choices.appendChild(
        el(
          'div',
          `color:${UI.textDim};font:italic 13px/1.5 ${UI.font};`,
          'Press Space to continue',
        ),
      );
      return;
    }

    view.choices.forEach((choice, index) => {
      const row = el(
        'div',
        `display:flex;gap:10px;align-items:baseline;padding:7px 10px;border-radius:3px;` +
          `border:1px solid ${UI.border};background:${UI.panelRaised};cursor:pointer;`,
      );
      row.dataset['d2rim'] = `choice-${choice.id}`;
      row.append(
        el('span', `color:${UI.accent};font:600 12px/1.4 ${UI.fontMono};min-width:14px;`, `${index + 1}`),
        el('span', `color:${UI.text};font:14px/1.4 ${UI.font};`, choice.text),
      );
      row.addEventListener('pointerdown', (event) => {
        event.stopPropagation();
        this.#choose(index);
      });
      choices.appendChild(row);
    });
  }

  /* -- input --------------------------------------------------------------- */

  #onKeyDown(event: KeyboardEvent): void {
    if (this.#view === null) return;
    const ui = this.#ctx?.services.tryGet<UiManager>(UiManagerKey);
    if (ui === undefined || !ui.isOpen('dialogue')) return;

    if (event.code === 'Space' || event.code === 'Enter' || event.code === 'NumpadEnter') {
      event.preventDefault();
      if (this.#view.choices.length === 0) this.#advance();
      else this.#choose(0);
      return;
    }
    const digit = /^Digit([1-9])$/.exec(event.code);
    if (digit !== null) {
      event.preventDefault();
      this.#choose(Number(digit[1]) - 1);
    }
  }

  #choose(index: number): void {
    const npcs = this.#ctx?.services.tryGet<NpcSystem>(NpcSystemKey);
    if (npcs === undefined) return;
    const view = npcs.choose(index);
    if (view === null) this.#ctx?.services.tryGet<UiManager>(UiManagerKey)?.close('dialogue');
  }

  #advance(): void {
    const npcs = this.#ctx?.services.tryGet<NpcSystem>(NpcSystemKey);
    if (npcs === undefined) return;
    const view = npcs.advance();
    if (view === null) this.#ctx?.services.tryGet<UiManager>(UiManagerKey)?.close('dialogue');
  }
}
