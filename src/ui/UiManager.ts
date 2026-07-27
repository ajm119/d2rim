/**
 * @module ui/UiManager
 *
 * Owns the overlay root, the screen stack, and the one thing every game UI
 * gets wrong: who has the keyboard.
 *
 * ### The input contract
 *
 * While any screen is open, `ctx.input.enabled` is false, so no gameplay module
 * sees a key — the Barbarian does not swing his axe because the player pressed
 * a number to buy a sword. The manager therefore cannot use `ctx.input` for its
 * *own* hotkeys either, or it would disable the very keys that close the
 * screen. It listens to the DOM directly instead, which is also what lets it
 * `preventDefault` Tab before the browser moves focus off the canvas.
 *
 * When nothing is open the overlay root is `pointer-events: none` end to end,
 * so a closed UI is indistinguishable from no UI: clicks reach the canvas,
 * pointer lock works, and the mouse look is unimpeded.
 *
 * ### Stack, not toggle set
 *
 * Screens open onto a stack and Escape pops one. That is what makes "open the
 * inventory, click a vendor, press Escape" return to the inventory rather than
 * to the game — the behaviour every player already has in their fingers.
 * Dialogue is exclusive: opening a conversation closes the panels, because a
 * modal conversation with an inventory in front of it is nobody's design.
 */

import { serviceKey } from '../core/ServiceLocator';
import type { GameContext, GameModule } from '../core/types';
import { hasDom, UI, Z } from './theme';

/* -------------------------------------------------------------------------- */
/* Events                                                                      */
/* -------------------------------------------------------------------------- */

declare module '../core/EventBus' {
  interface GameEvents {
    /** A screen opened or closed. `open` lists the whole stack, bottom first. */
    'ui:screens': { open: readonly string[]; top: string | null };
  }
}

/** Every screen the overlay can show. */
export type ScreenId = 'inventory' | 'skills' | 'vendor' | 'dialogue' | 'menu';

/** Screens that take over the whole overlay rather than stacking. */
const EXCLUSIVE: ReadonlySet<ScreenId> = new Set<ScreenId>(['dialogue', 'menu']);

/** What a screen must provide to be managed. */
export interface UiScreen {
  readonly id: ScreenId;
  /** The screen's root node. The manager owns its visibility, nothing else. */
  readonly root: HTMLElement;
  /** Rebuild content. Called on open and whenever the manager is asked to. */
  refresh?(): void;
  onOpen?(): void;
  onClose?(): void;
}

export const UiManagerKey = serviceKey<UiManager>('ui');

export class UiManager implements GameModule {
  readonly name = 'ui.manager';

  readonly #screens = new Map<ScreenId, UiScreen>();
  readonly #stack: ScreenId[] = [];
  readonly #disposers: Array<() => void> = [];

  #ctx: GameContext | null = null;
  #root: HTMLDivElement | null = null;
  #inputWasEnabled = true;

  /** The overlay root every screen and the HUD attach to. */
  get root(): HTMLElement | null {
    return this.#root;
  }

  get openScreens(): readonly ScreenId[] {
    return this.#stack;
  }

  get top(): ScreenId | null {
    return this.#stack[this.#stack.length - 1] ?? null;
  }

  get anyOpen(): boolean {
    return this.#stack.length > 0;
  }

  isOpen(id: ScreenId): boolean {
    return this.#stack.includes(id);
  }

  init(ctx: GameContext): void {
    this.#ctx = ctx;
    ctx.services.register(UiManagerKey, this);
    if (!hasDom()) return;

    const root = document.createElement('div');
    root.id = 'd2rim-ui';
    root.setAttribute(
      'style',
      `position:fixed;inset:0;z-index:${Z.screen};pointer-events:none;` +
        `font:14px/1.5 ${UI.font};color:${UI.text};`,
    );
    document.body.appendChild(root);
    this.#root = root;

    const onKeyDown = (event: KeyboardEvent): void => this.#onKeyDown(event);
    window.addEventListener('keydown', onKeyDown, { capture: true });
    this.#disposers.push(() => window.removeEventListener('keydown', onKeyDown, { capture: true }));
  }

  dispose(): void {
    for (const off of this.#disposers) off();
    this.#disposers.length = 0;
    this.closeAll();
    this.#root?.remove();
    this.#root = null;
    this.#screens.clear();
    this.#ctx?.services.unregister(UiManagerKey);
    this.#ctx = null;
  }

  /* -- registration -------------------------------------------------------- */

  /** Attach a screen. Its root is hidden until the screen is opened. */
  register(screen: UiScreen): void {
    this.#screens.set(screen.id, screen);
    screen.root.style.display = 'none';
    this.#root?.appendChild(screen.root);
  }

  unregister(id: ScreenId): void {
    this.close(id);
    this.#screens.delete(id);
  }

  screen(id: ScreenId): UiScreen | null {
    return this.#screens.get(id) ?? null;
  }

  /* -- stack --------------------------------------------------------------- */

  open(id: ScreenId): boolean {
    const screen = this.#screens.get(id);
    if (screen === undefined) return false;
    if (this.#stack.includes(id)) {
      screen.refresh?.();
      return true;
    }
    if (EXCLUSIVE.has(id)) this.closeAll();

    this.#stack.push(id);
    screen.root.style.display = '';
    screen.onOpen?.();
    screen.refresh?.();
    this.#applyInputState();
    this.#announce();
    return true;
  }

  close(id: ScreenId): boolean {
    const index = this.#stack.indexOf(id);
    if (index === -1) return false;
    this.#stack.splice(index, 1);
    const screen = this.#screens.get(id);
    if (screen !== undefined) {
      screen.root.style.display = 'none';
      screen.onClose?.();
    }
    this.#applyInputState();
    this.#announce();
    return true;
  }

  toggle(id: ScreenId): boolean {
    return this.isOpen(id) ? !this.close(id) : this.open(id);
  }

  /** Pop the topmost screen. @returns whether anything was closed. */
  back(): boolean {
    const top = this.top;
    return top === null ? false : this.close(top);
  }

  closeAll(): void {
    while (this.#stack.length > 0) {
      const id = this.#stack.pop() as ScreenId;
      const screen = this.#screens.get(id);
      if (screen === undefined) continue;
      screen.root.style.display = 'none';
      screen.onClose?.();
    }
    this.#applyInputState();
    this.#announce();
  }

  /** Rebuild every open screen. Called when the character sheet changes. */
  refreshOpen(): void {
    for (const id of this.#stack) this.#screens.get(id)?.refresh?.();
  }

  /* -- internals ----------------------------------------------------------- */

  /**
   * Take or release the keyboard and the pointer.
   *
   * Pointer lock is released on open because a modal panel the player cannot
   * point at is a modal panel they cannot use; it is not re-acquired on close,
   * since re-locking without a fresh user gesture is rejected by the browser
   * anyway — the next click on the canvas restores it.
   */
  #applyInputState(): void {
    const ctx = this.#ctx;
    const root = this.#root;
    if (ctx === null) return;

    const open = this.anyOpen;
    if (root !== null) root.style.pointerEvents = open ? 'auto' : 'none';

    if (open) {
      if (ctx.input.enabled) this.#inputWasEnabled = true;
      ctx.input.enabled = false;
      ctx.input.exitPointerLock();
    } else {
      ctx.input.enabled = this.#inputWasEnabled;
    }
  }

  #announce(): void {
    this.#ctx?.events.emit('ui:screens', { open: [...this.#stack], top: this.top });
  }

  /**
   * The overlay's own hotkeys.
   *
   * Handled in the capture phase and before `Input` sees them, because Tab
   * moves browser focus and Escape exits pointer lock, and both defaults fire
   * before a bubbling listener would get a look in.
   */
  #onKeyDown(event: KeyboardEvent): void {
    if (event.repeat) return;
    const code = event.code;

    if (code === 'Escape') {
      // Escape closes what is open, or opens the pause menu when nothing is.
      event.preventDefault();
      if (this.anyOpen) this.back();
      else this.open('menu');
      return;
    }

    // Everything below is a panel toggle. While a conversation is running the
    // player is talking, not managing gear, so the toggles are inert.
    if (this.isOpen('dialogue')) return;

    if (code === 'KeyI' || code === 'Tab') {
      event.preventDefault();
      this.toggle('inventory');
      return;
    }
    if (code === 'KeyK') {
      event.preventDefault();
      this.toggle('skills');
    }
  }
}
