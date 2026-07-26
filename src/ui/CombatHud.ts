/**
 * @module ui/CombatHud
 *
 * Health, mana and stamina, in the Diablo II arrangement: two orbs flanking a
 * centre strip, with stamina as a bar under it.
 *
 * DOM rather than a render pass, deliberately. The HUD has to be legible at any
 * resolution, it changes only when a number changes, and putting it in the
 * frame graph would mean a full-screen pass and a font atlas for something that
 * three divs and a border-radius do better.
 *
 * Everything here reads from events; the HUD never polls combat and never
 * writes to it.
 */

import type { GameContext, GameModule } from '../core/types';

interface Meter {
  readonly fill: HTMLDivElement;
  readonly label: HTMLDivElement;
  value: number;
  max: number;
}

function orb(colour: string, glow: string, side: 'left' | 'right'): HTMLDivElement {
  const root = document.createElement('div');
  root.setAttribute(
    'style',
    `position:absolute;bottom:14px;${side}:18px;width:96px;height:96px;border-radius:50%;` +
      'overflow:hidden;background:radial-gradient(circle at 35% 30%,#2a2119,#0b0906);' +
      `box-shadow:inset 0 0 22px rgba(0,0,0,0.9),0 0 14px ${glow};` +
      'border:2px solid #2f2419;',
  );
  const fill = document.createElement('div');
  fill.setAttribute(
    'style',
    `position:absolute;left:0;right:0;bottom:0;height:100%;background:${colour};` +
      'transition:height 120ms linear;',
  );
  const label = document.createElement('div');
  label.setAttribute(
    'style',
    'position:absolute;inset:0;display:flex;align-items:center;justify-content:center;' +
      "color:#f4ece0;font:600 13px/1 'Trebuchet MS',Georgia,serif;text-shadow:0 1px 3px #000;",
  );
  root.append(fill, label);
  (root as HTMLDivElement & { fill?: HTMLDivElement }).fill = fill;
  (root as HTMLDivElement & { label?: HTMLDivElement }).label = label;
  return root;
}

export class CombatHud implements GameModule {
  readonly name = 'ui.combatHud';

  #root: HTMLDivElement | null = null;
  #health: Meter | null = null;
  #mana: Meter | null = null;
  #stamina: Meter | null = null;
  #banner: HTMLDivElement | null = null;
  #unsubscribe: Array<() => void> = [];

  init(ctx: GameContext): void {
    if (typeof document === 'undefined') return;

    const root = document.createElement('div');
    root.id = 'd2rim-hud';
    root.setAttribute('style', 'position:fixed;inset:0;pointer-events:none;z-index:35;');

    const healthOrb = orb('linear-gradient(#e0442e,#7d1b12)', 'rgba(200,50,30,0.35)', 'left');
    const manaOrb = orb('linear-gradient(#3f7fe0,#16306e)', 'rgba(60,110,220,0.35)', 'right');

    const staminaShell = document.createElement('div');
    staminaShell.setAttribute(
      'style',
      'position:absolute;bottom:26px;left:50%;transform:translateX(-50%);width:260px;height:10px;' +
        'background:#150f0a;border:1px solid #35281a;border-radius:5px;overflow:hidden;',
    );
    const staminaFill = document.createElement('div');
    staminaFill.setAttribute(
      'style',
      'height:100%;width:100%;background:linear-gradient(90deg,#c8a44a,#e8d089);' +
        'transition:width 90ms linear;',
    );
    staminaShell.appendChild(staminaFill);

    const banner = document.createElement('div');
    banner.setAttribute(
      'style',
      'position:absolute;top:38%;left:50%;transform:translate(-50%,-50%);opacity:0;' +
        "color:#e8503a;font:700 46px/1 'Trebuchet MS',Georgia,serif;letter-spacing:6px;" +
        'text-shadow:0 3px 14px #000;transition:opacity 220ms ease;',
    );
    banner.textContent = 'YOU HAVE DIED';

    root.append(healthOrb, manaOrb, staminaShell, banner);
    document.body.appendChild(root);

    this.#root = root;
    this.#banner = banner;
    this.#health = meterOf(healthOrb);
    this.#mana = meterOf(manaOrb);
    this.#stamina = { fill: staminaFill, label: staminaFill, value: 1, max: 1 };

    this.#unsubscribe.push(
      ctx.events.on('combat:vitals', (payload) => {
        this.#set(this.#health, payload.health, payload.healthMax, true);
        this.#set(this.#mana, payload.mana, payload.manaMax, true);
        this.#setBar(this.#stamina, payload.stamina, payload.staminaMax);
      }),
      ctx.events.on('player:stamina', (payload) => {
        this.#setBar(this.#stamina, payload.value, payload.max);
      }),
      ctx.events.on('combat:playerDown', () => {
        if (this.#banner !== null) this.#banner.style.opacity = '1';
      }),
      ctx.events.on('combat:respawn', () => {
        if (this.#banner !== null) this.#banner.style.opacity = '0';
      }),
    );
  }

  update(ctx: GameContext): void {
    // Stamina is owned by `PlayerController` and only emits on spend, so it is
    // polled here rather than pushed; regeneration is continuous and an event
    // per frame for a bar is not a trade worth making.
    const combat = ctx.services.tryGet<{ stamina: number; staminaMax: number }>('combat');
    if (combat !== undefined) this.#setBar(this.#stamina, combat.stamina, combat.staminaMax);
  }

  dispose(): void {
    for (const off of this.#unsubscribe) off();
    this.#unsubscribe = [];
    this.#root?.remove();
    this.#root = null;
  }

  #set(meter: Meter | null, value: number, max: number, showNumber: boolean): void {
    if (meter === null) return;
    meter.value = value;
    meter.max = max;
    const fraction = max > 0 ? Math.max(0, Math.min(1, value / max)) : 0;
    meter.fill.style.height = `${(fraction * 100).toFixed(1)}%`;
    if (showNumber) meter.label.textContent = `${Math.ceil(value)}`;
  }

  #setBar(meter: Meter | null, value: number, max: number): void {
    if (meter === null) return;
    const fraction = max > 0 ? Math.max(0, Math.min(1, value / max)) : 0;
    meter.fill.style.width = `${(fraction * 100).toFixed(1)}%`;
  }
}

function meterOf(root: HTMLDivElement): Meter {
  const decorated = root as HTMLDivElement & { fill?: HTMLDivElement; label?: HTMLDivElement };
  return {
    fill: decorated.fill ?? root,
    label: decorated.label ?? root,
    value: 1,
    max: 1,
  };
}
