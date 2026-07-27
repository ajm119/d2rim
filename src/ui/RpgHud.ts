/**
 * @module ui/RpgHud
 *
 * The RPG half of the heads-up display: the experience bar, the active skill,
 * the quest tracker, the interact prompt, and the floating labels on loot.
 *
 * Health and mana orbs live in `ui/CombatHud`, which was already reading the
 * combat event stream before an RPG layer existed. Splitting them is
 * deliberate: this module can be removed entirely and the combat sandbox still
 * has a working HUD.
 *
 * ### Loot labels
 *
 * Projected from world space every frame, which is the one part of this file
 * that costs anything. It is bounded three ways: labels are only built for
 * entries within {@link LABEL_RANGE}, the projection is a single
 * `Vector3.project` per entry, and the DOM is only touched when a label's
 * rounded pixel position actually changes. A floor with a dozen drops on it
 * costs a dozen matrix multiplies a frame.
 *
 * The label carries the item's Diablo quality colour, because on a dark cave
 * floor that colour *is* the pickup decision.
 *
 * ### Everything is event-driven
 *
 * With one exception. The loot labels have to be re-projected whenever the
 * camera moves, which is every frame, so they are polled in `lateUpdate` after
 * `CameraRig` has settled the camera. Reading a stale camera puts every label
 * one frame behind the world it is labelling, which reads as the labels
 * swimming.
 */

import * as THREE from 'three/webgpu';

import type { GameContext, GameModule } from '../core/types';
import { LABEL_RANGE, LootSystemKey, type GroundLoot, type LootSystem } from '../rpg/Loot';
import { RpgSystemKey, type RpgSystem } from '../rpg/RpgSystem';
import { findSkill } from '../rpg/SkillTree';
import { QuestSystemKey, type QuestSystem } from '../quest/QuestSystem';
import { clearChildren, el, hasDom, UI, Z } from './theme';

interface LabelHandle {
  readonly node: HTMLDivElement;
  x: number;
  y: number;
  visible: boolean;
}

export class RpgHud implements GameModule {
  readonly name = 'ui.rpgHud';

  readonly #unsubscribe: Array<() => void> = [];
  readonly #labels = new Map<number, LabelHandle>();
  readonly #projected = new THREE.Vector3();

  #root: HTMLDivElement | null = null;
  #xpFill: HTMLDivElement | null = null;
  #xpText: HTMLDivElement | null = null;
  #levelBadge: HTMLDivElement | null = null;
  #skillBox: HTMLDivElement | null = null;
  #skillName: HTMLDivElement | null = null;
  #skillPoints: HTMLDivElement | null = null;
  #tracker: HTMLDivElement | null = null;
  #prompt: HTMLDivElement | null = null;
  #toasts: HTMLDivElement | null = null;
  #labelLayer: HTMLDivElement | null = null;
  #goldReadout: HTMLDivElement | null = null;

  #ctx: GameContext | null = null;
  #lastVersion = -1;

  init(ctx: GameContext): void {
    this.#ctx = ctx;
    if (!hasDom()) return;

    const root = el(
      'div',
      `position:fixed;inset:0;z-index:${Z.hud};pointer-events:none;font:14px/1.4 ${UI.font};`,
    );
    root.id = 'd2rim-rpg-hud';

    this.#labelLayer = el('div', 'position:absolute;inset:0;');
    root.appendChild(this.#labelLayer);

    root.appendChild(this.#buildExperienceBar());
    root.appendChild(this.#buildSkillBox());
    root.appendChild(this.#buildTracker());
    root.appendChild(this.#buildPrompt());
    root.appendChild(this.#buildToasts());
    root.appendChild(this.#buildGold());

    document.body.appendChild(root);
    this.#root = root;

    this.#unsubscribe.push(
      ctx.events.on('rpg:characterChanged', () => this.refresh()),
      ctx.events.on('rpg:experience', () => this.refresh()),
      ctx.events.on('rpg:gold', () => this.refresh()),
      ctx.events.on('rpg:activeSkill', () => this.refresh()),
      ctx.events.on('rpg:levelUp', (payload) =>
        this.toast(`Welcome to level ${payload.level}`, UI.accent),
      ),
      ctx.events.on('rpg:itemAcquired', (payload) => this.toast(payload.name, UI.text)),
      ctx.events.on('loot:refused', () => this.toast('Your pack is full', UI.danger)),
      ctx.events.on('quest:accepted', (payload) => {
        this.toast(`New quest: ${payload.title}`, UI.accent);
        this.refresh();
      }),
      ctx.events.on('quest:progress', () => this.refresh()),
      ctx.events.on('quest:objectiveComplete', (payload) =>
        this.toast(`${payload.description} — done`, UI.ok),
      ),
      ctx.events.on('quest:complete', (payload) => {
        this.toast(`${payload.title}: return to Akara`, UI.ok);
        this.refresh();
      }),
      ctx.events.on('quest:rewarded', (payload) => {
        if (payload.skillPoints > 0) {
          this.toast(`+${payload.skillPoints} skill point`, UI.accent);
        }
        this.refresh();
      }),
      ctx.events.on('npc:prompt', (payload) => {
        this.setPrompt(
          payload.npcId === null ? null : `Press E to ${payload.verb} ${payload.displayName}`,
        );
      }),
      ctx.events.on('ui:screens', (payload) => {
        // The HUD is noise behind a full-screen panel, and the tracker in
        // particular sits exactly where an inventory tooltip wants to be.
        if (this.#root !== null) this.#root.style.opacity = payload.open.length > 0 ? '0' : '1';
      }),
    );

    this.refresh();
  }

  lateUpdate(ctx: GameContext): void {
    this.#syncLabels(ctx);
    const rpg = ctx.services.tryGet<RpgSystem>(RpgSystemKey);
    if (rpg !== undefined && rpg.character.version !== this.#lastVersion) {
      this.#lastVersion = rpg.character.version;
      this.refresh();
    }
  }

  dispose(): void {
    for (const off of this.#unsubscribe) off();
    this.#unsubscribe.length = 0;
    for (const label of this.#labels.values()) label.node.remove();
    this.#labels.clear();
    this.#root?.remove();
    this.#root = null;
    this.#ctx = null;
  }

  /* -- public surface ------------------------------------------------------ */

  /** Rebuild every readout from the live state. */
  refresh(): void {
    const ctx = this.#ctx;
    if (ctx === null) return;
    const rpg = ctx.services.tryGet<RpgSystem>(RpgSystemKey);
    if (rpg === undefined) return;

    const stats = rpg.character.stats;
    if (this.#xpFill !== null) {
      this.#xpFill.style.width = `${(stats.progress * 100).toFixed(1)}%`;
    }
    if (this.#xpText !== null) {
      this.#xpText.textContent =
        stats.experienceToNextLevel > 0
          ? `${stats.experience} xp · ${stats.experienceToNextLevel} to next`
          : `${stats.experience} xp · maximum level`;
    }
    if (this.#levelBadge !== null) this.#levelBadge.textContent = `LVL ${stats.level}`;
    if (this.#goldReadout !== null) {
      this.#goldReadout.textContent = `${rpg.character.gold} gold`;
    }

    const activeId = rpg.character.skills.active;
    const definition = activeId === null ? null : findSkill(activeId);
    if (this.#skillName !== null) {
      this.#skillName.textContent = definition?.name ?? 'Attack';
    }
    if (this.#skillBox !== null) {
      this.#skillBox.style.borderColor = definition === null ? UI.border : UI.accent;
    }
    if (this.#skillPoints !== null) {
      const unspent = stats.skillPoints;
      this.#skillPoints.textContent = unspent > 0 ? `${unspent} unspent (K)` : '';
      this.#skillPoints.style.color = unspent > 0 ? UI.accent : UI.textDim;
    }

    this.#renderTracker(ctx.services.tryGet<QuestSystem>(QuestSystemKey) ?? null);
  }

  /** Show the interact prompt, or hide it with `null`. */
  setPrompt(text: string | null): void {
    const node = this.#prompt;
    if (node === null) return;
    node.textContent = text ?? '';
    node.style.opacity = text === null ? '0' : '1';
  }

  /** Announce something transient in the middle of the screen. */
  toast(text: string, colour: string = UI.text): void {
    const host = this.#toasts;
    if (host === null) return;
    const node = el(
      'div',
      `color:${colour};font:600 15px/1.4 ${UI.font};letter-spacing:0.05em;` +
        'text-shadow:0 2px 6px #000;opacity:0;transition:opacity 180ms ease;' +
        'text-align:center;margin-bottom:4px;',
      text,
    );
    host.appendChild(node);
    // Two frames of delay so the transition has a start value to animate from;
    // setting opacity in the same task the node is appended in skips it.
    requestAnimationFrame(() => requestAnimationFrame(() => (node.style.opacity = '1')));
    window.setTimeout(() => {
      node.style.opacity = '0';
      window.setTimeout(() => node.remove(), 260);
    }, 2600);
  }

  /* -- construction -------------------------------------------------------- */

  #buildExperienceBar(): HTMLDivElement {
    const shell = el(
      'div',
      'position:absolute;left:128px;right:128px;bottom:6px;height:9px;' +
        `background:#0a0705;border:1px solid ${UI.border};border-radius:5px;overflow:hidden;`,
    );
    const fill = el(
      'div',
      `height:100%;width:0%;background:linear-gradient(90deg,#8a6a22,${UI.experience});` +
        'transition:width 220ms ease;',
    );
    shell.appendChild(fill);
    this.#xpFill = fill;

    const text = el(
      'div',
      `position:absolute;left:128px;bottom:19px;color:${UI.textDim};` +
        `font:11px/1 ${UI.fontMono};letter-spacing:0.06em;text-shadow:0 1px 3px #000;`,
      '0 xp',
    );
    this.#xpText = text;

    const badge = el(
      'div',
      `position:absolute;right:128px;bottom:19px;color:${UI.accent};` +
        `font:600 12px/1 ${UI.font};letter-spacing:0.14em;text-shadow:0 1px 3px #000;`,
      'LVL 1',
    );
    this.#levelBadge = badge;

    const group = el('div', 'position:absolute;inset:0;');
    group.append(shell, text, badge);
    return group;
  }

  #buildSkillBox(): HTMLDivElement {
    const group = el('div', 'position:absolute;inset:0;');
    const box = el(
      'div',
      'position:absolute;left:50%;bottom:44px;transform:translateX(-50%);' +
        `min-width:132px;padding:6px 12px;background:rgba(10,8,6,0.86);` +
        `border:1px solid ${UI.border};border-radius:4px;text-align:center;`,
    );
    const label = el(
      'div',
      `color:${UI.textDim};font:10px/1 ${UI.font};letter-spacing:0.18em;` +
        'text-transform:uppercase;margin-bottom:3px;',
      'Skill',
    );
    const name = el('div', `color:${UI.text};font:600 14px/1.2 ${UI.font};`, 'Attack');
    const points = el('div', `color:${UI.textDim};font:10px/1.3 ${UI.font};margin-top:2px;`, '');
    box.append(label, name, points);
    group.appendChild(box);
    this.#skillBox = box;
    this.#skillName = name;
    this.#skillPoints = points;
    return group;
  }

  #buildTracker(): HTMLDivElement {
    const tracker = el(
      'div',
      'position:absolute;right:18px;top:18px;min-width:200px;max-width:280px;' +
        `padding:10px 12px;background:rgba(10,8,6,0.82);border:1px solid ${UI.border};` +
        'border-radius:4px;display:none;',
    );
    this.#tracker = tracker;
    return tracker;
  }

  #buildPrompt(): HTMLDivElement {
    const node = el(
      'div',
      'position:absolute;left:50%;bottom:22%;transform:translateX(-50%);' +
        `padding:9px 18px;border-radius:4px;border:1px solid ${UI.border};` +
        `background:rgba(8,7,6,0.82);color:${UI.text};font:500 15px/1.3 ${UI.font};` +
        'letter-spacing:0.04em;text-shadow:0 1px 3px #000;opacity:0;' +
        'transition:opacity 140ms ease-out;white-space:nowrap;',
    );
    node.dataset['d2rim'] = 'npc-prompt';
    this.#prompt = node;
    return node;
  }

  #buildToasts(): HTMLDivElement {
    const node = el(
      'div',
      'position:absolute;left:50%;top:22%;transform:translateX(-50%);' +
        'display:flex;flex-direction:column;align-items:center;',
    );
    this.#toasts = node;
    return node;
  }

  #buildGold(): HTMLDivElement {
    const node = el(
      'div',
      `position:absolute;right:18px;bottom:44px;color:${UI.accent};` +
        `font:600 13px/1 ${UI.font};letter-spacing:0.06em;text-shadow:0 1px 3px #000;`,
      '0 gold',
    );
    this.#goldReadout = node;
    return node;
  }

  /* -- tracker ------------------------------------------------------------- */

  #renderTracker(quests: QuestSystem | null): void {
    const tracker = this.#tracker;
    if (tracker === null) return;
    const active = quests?.tracked() ?? [];
    if (active.length === 0) {
      tracker.style.display = 'none';
      return;
    }
    tracker.style.display = '';
    clearChildren(tracker);

    for (const quest of active) {
      tracker.appendChild(
        el(
          'div',
          `color:${UI.accent};font:600 12px/1.3 ${UI.font};letter-spacing:0.12em;` +
            'text-transform:uppercase;margin-bottom:6px;',
          quest.title,
        ),
      );
      for (const objective of quest.objectives) {
        const done = objective.complete;
        const count = objective.showCount ? ` ${objective.current} of ${objective.required}` : '';
        tracker.appendChild(
          el(
            'div',
            `color:${done ? UI.ok : UI.text};font:13px/1.5 ${UI.font};` +
              `${done ? 'text-decoration:line-through;opacity:0.75;' : ''}`,
            `${done ? '✓' : '·'} ${objective.description}${count}`,
          ),
        );
      }
      if (quest.state === 'complete') {
        tracker.appendChild(
          el(
            'div',
            `color:${UI.ok};font:italic 12px/1.5 ${UI.font};margin-top:6px;`,
            'Return to Akara',
          ),
        );
      }
    }
  }

  /* -- loot labels --------------------------------------------------------- */

  #syncLabels(ctx: GameContext): void {
    const layer = this.#labelLayer;
    if (layer === null) return;
    const loot = ctx.services.tryGet<LootSystem>(LootSystemKey);
    const entries = loot?.entries ?? [];

    const seen = new Set<number>();
    for (const entry of entries) {
      seen.add(entry.id);
      this.#syncLabel(ctx, layer, entry);
    }
    for (const [id, label] of this.#labels) {
      if (seen.has(id)) continue;
      label.node.remove();
      this.#labels.delete(id);
    }
  }

  #syncLabel(ctx: GameContext, layer: HTMLDivElement, entry: GroundLoot): void {
    let label = this.#labels.get(entry.id);
    if (label === undefined) {
      const node = el(
        'div',
        'position:absolute;transform:translate(-50%,-100%);white-space:nowrap;' +
          `padding:2px 7px;border-radius:3px;background:rgba(6,5,4,0.86);` +
          `border:1px solid rgba(0,0,0,0.8);font:600 12px/1.3 ${UI.font};` +
          `color:${entry.colour};text-shadow:0 1px 3px #000;`,
        entry.label,
      );
      layer.appendChild(node);
      label = { node, x: -1, y: -1, visible: false };
      this.#labels.set(entry.id, label);
    }

    this.#projected.copy(entry.position);
    this.#projected.y += 0.55;
    const distance = this.#projected.distanceTo(ctx.camera.position);
    this.#projected.project(ctx.camera);

    const behind = this.#projected.z > 1;
    const visible = !behind && distance <= LABEL_RANGE;
    if (visible !== label.visible) {
      label.visible = visible;
      label.node.style.display = visible ? '' : 'none';
    }
    if (!visible) return;

    const x = Math.round(((this.#projected.x + 1) / 2) * window.innerWidth);
    const y = Math.round(((1 - this.#projected.y) / 2) * window.innerHeight);
    if (x !== label.x || y !== label.y) {
      label.x = x;
      label.y = y;
      label.node.style.left = `${x}px`;
      label.node.style.top = `${y}px`;
    }
  }
}
