/**
 * @module ui/InventoryScreen
 *
 * The inventory: character sheet on the left, paper doll in the middle, and the
 * 10x4 grid along the bottom, with drag-and-drop between all three.
 *
 * ### Drag and drop, D2 style
 *
 * There is no HTML5 drag API here. The interaction is Diablo's: a click picks
 * the item **onto the cursor**, the cursor carries it, and a second click puts
 * it down. That is not nostalgia — a held-cursor model is the only one where
 * dropping onto an occupied cell can *swap*, because a swap needs somewhere to
 * put the displaced item, and the cursor is that somewhere.
 *
 * Two pure functions in `ui/theme` carry the arithmetic: {@link grabOffset}
 * records which cell of the item was grabbed, and {@link cellFromPoint} turns a
 * pointer position back into a top-left cell. They are separated out and unit
 * tested because an off-by-one in either puts every dropped item one cell from
 * where the player released it — a bug that is invisible in a screenshot and
 * infuriating in the hand.
 *
 * ### Rendering
 *
 * The whole panel is rebuilt on {@link InventoryScreen.refresh}, which happens
 * on open and whenever the character's version counter moves. Rebuilding beats
 * diffing here: the panel is a hundred nodes, it is only visible while the game
 * is paused behind it, and a diffing implementation would be the only place in
 * the project where an item could be shown in a stale position.
 *
 * Items have no icons — the project has no item art — so each is drawn as a
 * quality-coloured plate with its name. That is honest, and at a 34-pixel cell
 * it is also more readable than a 32x32 sprite would be.
 */

import type { GameContext, GameModule } from '../core/types';
import { describeItem, isBroken, itemColour, type Item } from '../rpg/ItemGenerator';
import {
  EQUIPMENT_SLOTS,
  EQUIPMENT_SLOT_LABELS,
  unmetRequirement,
  type EquipmentSlot,
} from '../rpg/Inventory';
import { RpgSystemKey, type RpgSystem } from '../rpg/RpgSystem';
import { PRIMARY_STATS, type PrimaryStat } from '../rpg/Stats';
import {
  CELL_SIZE,
  cellFromPoint,
  clearChildren,
  el,
  grabOffset,
  hasDom,
  headingStyle,
  panelStyle,
  scrimStyle,
  screenRoot,
  UI,
  Z,
} from './theme';
import { UiManagerKey, type UiManager, type UiScreen } from './UiManager';

/** Where a held item came from, so a refused drop can be put back. */
type HeldSource =
  | { readonly kind: 'grid'; readonly x: number; readonly y: number }
  | { readonly kind: 'equipment'; readonly slot: EquipmentSlot };

interface Held {
  readonly item: Item;
  readonly source: HeldSource;
  /** Which cell of the item the pointer grabbed. */
  readonly offset: { readonly x: number; readonly y: number };
}

export class InventoryScreen implements GameModule, UiScreen {
  readonly name = 'ui.inventory';
  readonly id = 'inventory' as const;

  readonly root: HTMLElement;

  #ctx: GameContext | null = null;
  #grid: HTMLDivElement | null = null;
  #dollHost: HTMLDivElement | null = null;
  #statsHost: HTMLDivElement | null = null;
  #tooltip: HTMLDivElement | null = null;
  #cursor: HTMLDivElement | null = null;
  #held: Held | null = null;
  #lastVersion = -1;
  readonly #disposers: Array<() => void> = [];

  constructor() {
    this.root = screenRoot(scrimStyle(Z.screen));
  }

  init(ctx: GameContext): void {
    this.#ctx = ctx;
    if (!hasDom()) return;
    this.#build();
    ctx.services.tryGet<UiManager>(UiManagerKey)?.register(this);

    const onMove = (event: PointerEvent): void => this.#moveCursor(event);
    window.addEventListener('pointermove', onMove);
    this.#disposers.push(() => window.removeEventListener('pointermove', onMove));

    this.#disposers.push(
      ctx.events.on('rpg:characterChanged', () => {
        if (this.root.style.display !== 'none') this.refresh();
      }),
    );
  }

  dispose(): void {
    for (const off of this.#disposers) off();
    this.#disposers.length = 0;
    this.#cursor?.remove();
    this.#tooltip?.remove();
    this.root.remove();
    this.#ctx = null;
  }

  onOpen(): void {
    this.#lastVersion = -1;
  }

  onClose(): void {
    // An item left on the cursor when the panel closes has to go somewhere, and
    // the floor is not it: put it back where it came from, or anywhere it fits.
    this.#returnHeld();
    // The tooltip lives on `document.body`, outside the panel the manager
    // hides, so it has to be dismissed explicitly. `clearChildren` during a
    // rebuild can also remove the node the pointer is over without firing
    // `pointerleave`, which would otherwise strand a tooltip on screen.
    this.#hideTooltip();
  }

  refresh(): void {
    const rpg = this.#rpg();
    if (rpg === null || !hasDom()) return;
    if (rpg.character.version === this.#lastVersion) return;
    this.#lastVersion = rpg.character.version;
    this.#hideTooltip();
    this.#renderStats(rpg);
    this.#renderDoll(rpg);
    this.#renderGrid(rpg);
  }

  /* -- construction -------------------------------------------------------- */

  #build(): void {
    const panel = el(
      'div',
      panelStyle(
        'width:min(940px,94vw);max-height:92vh;overflow:auto;padding:20px 22px;' +
          'display:grid;grid-template-columns:210px 1fr;gap:20px;',
      ),
    );
    panel.dataset['d2rim'] = 'inventory-panel';

    const left = el('div', '');
    left.appendChild(el('div', headingStyle(), 'Character'));
    const stats = el('div', '');
    this.#statsHost = stats;
    left.appendChild(stats);

    const right = el('div', 'display:flex;flex-direction:column;gap:16px;');
    const dollWrap = el('div', '');
    dollWrap.appendChild(el('div', headingStyle(), 'Equipped'));
    const doll = el('div', 'display:grid;grid-template-columns:repeat(5,1fr);gap:8px;');
    this.#dollHost = doll;
    dollWrap.appendChild(doll);

    const gridWrap = el('div', '');
    gridWrap.appendChild(el('div', headingStyle(), 'Inventory'));
    const grid = el(
      'div',
      `position:relative;width:${CELL_SIZE * 10}px;height:${CELL_SIZE * 4}px;` +
        `background:${UI.well};border:1px solid ${UI.border};` +
        `background-image:linear-gradient(${UI.border} 1px,transparent 1px),` +
        `linear-gradient(90deg,${UI.border} 1px,transparent 1px);` +
        `background-size:${CELL_SIZE}px ${CELL_SIZE}px;background-position:-1px -1px;` +
        'cursor:pointer;',
    );
    grid.dataset['d2rim'] = 'inventory-grid';
    grid.addEventListener('pointerdown', (event) => this.#onGridPointerDown(event));
    this.#grid = grid;
    gridWrap.appendChild(grid);

    const hint = el(
      'div',
      `color:${UI.textDim};font:12px/1.6 ${UI.font};margin-top:8px;`,
      'Click to lift an item, click again to place it. Right-click to equip. I or Tab closes.',
    );
    gridWrap.appendChild(hint);

    right.append(dollWrap, gridWrap);
    panel.append(left, right);
    this.root.appendChild(panel);

    // Clicking the scrim closes; clicking the panel must not.
    panel.addEventListener('pointerdown', (event) => event.stopPropagation());
    this.root.addEventListener('pointerdown', () => {
      this.#returnHeld();
      this.#ctx?.services.tryGet<UiManager>(UiManagerKey)?.close('inventory');
    });

    const tooltip = el(
      'div',
      panelStyle(
        `position:fixed;z-index:${Z.cursor};max-width:280px;padding:10px 12px;` +
          'pointer-events:none;display:none;',
      ),
    );
    document.body.appendChild(tooltip);
    this.#tooltip = tooltip;

    const cursor = el(
      'div',
      `position:fixed;z-index:${Z.cursor};pointer-events:none;display:none;` +
        'transform:translate(-50%,-50%);',
    );
    document.body.appendChild(cursor);
    this.#cursor = cursor;
  }

  /* -- rendering ----------------------------------------------------------- */

  #renderStats(rpg: RpgSystem): void {
    const host = this.#statsHost;
    if (host === null) return;
    clearChildren(host);

    const character = rpg.character;
    const derived = character.derived;
    const stats = character.stats;

    host.appendChild(
      el('div', `color:${UI.accent};font:600 16px/1.4 ${UI.font};`, `${character.name}`),
    );
    host.appendChild(
      el(
        'div',
        `color:${UI.textDim};font:12px/1.6 ${UI.font};margin-bottom:10px;`,
        `Level ${derived.level} ${character.stats.definition.displayName}`,
      ),
    );

    for (const stat of PRIMARY_STATS) {
      host.appendChild(this.#statRow(rpg, stat, derived.attributes[stat], derived.baseAttributes[stat]));
    }
    if (stats.statPoints > 0) {
      host.appendChild(
        el(
          'div',
          `color:${UI.accent};font:12px/1.8 ${UI.font};margin:4px 0 10px;`,
          `${stats.statPoints} stat point${stats.statPoints === 1 ? '' : 's'} to spend`,
        ),
      );
    }

    const rows: readonly [string, string][] = [
      ['Life', `${derived.maxLife}`],
      ['Mana', `${derived.maxMana}`],
      ['Stamina', `${derived.maxStamina}`],
      ['Attack Rating', `${derived.attackRating}`],
      ['Damage', `${derived.damage.min} – ${derived.damage.max}`],
      ['Defense', `${derived.defense}`],
      ['Deadly Strike', `${Math.round(derived.criticalChance * 100)}%`],
      ['Block', `${Math.round((character.equipment.blockChance() + derived.blockChance) * 100)}%`],
      ['Gold', `${character.gold}`],
    ];
    const table = el('div', `margin-top:10px;border-top:1px solid ${UI.border};padding-top:10px;`);
    for (const [label, value] of rows) {
      const row = el('div', 'display:flex;justify-content:space-between;gap:8px;line-height:1.7;');
      row.append(
        el('span', `color:${UI.textDim};font:12px/1.7 ${UI.font};`, label),
        el('span', `color:${UI.text};font:600 12px/1.7 ${UI.fontMono};`, value),
      );
      table.appendChild(row);
    }
    host.appendChild(table);
  }

  #statRow(
    rpg: RpgSystem,
    stat: PrimaryStat,
    total: number,
    base: number,
  ): HTMLDivElement {
    const row = el('div', 'display:flex;align-items:center;gap:8px;line-height:1.9;');
    const name = stat.charAt(0).toUpperCase() + stat.slice(1);
    row.appendChild(el('span', `flex:1;color:${UI.textDim};font:12px/1.9 ${UI.font};`, name));
    // Gear-boosted attributes print in the accent colour, D2's blue-number
    // convention translated into this palette.
    const bonus = total - base;
    row.appendChild(
      el(
        'span',
        `color:${bonus > 0 ? UI.accent : UI.text};font:600 13px/1.9 ${UI.fontMono};min-width:34px;text-align:right;`,
        `${total}`,
      ),
    );
    if (rpg.character.stats.statPoints > 0) {
      const button = el(
        'button',
        `background:${UI.panelRaised};border:1px solid ${UI.borderBright};color:${UI.accent};` +
          `width:22px;height:22px;border-radius:3px;cursor:pointer;font:600 14px/1 ${UI.font};`,
        '+',
      );
      button.dataset['d2rim'] = `spend-${stat}`;
      button.addEventListener('click', (event) => {
        event.stopPropagation();
        rpg.character.spendStatPoint(stat);
        this.refresh();
      });
      row.appendChild(button);
    }
    return row;
  }

  #renderDoll(rpg: RpgSystem): void {
    const host = this.#dollHost;
    if (host === null) return;
    clearChildren(host);

    for (const slot of EQUIPMENT_SLOTS) {
      const item = rpg.character.equipment.get(slot);
      const cell = el(
        'div',
        `position:relative;height:${CELL_SIZE * 2}px;background:${UI.well};` +
          `border:1px solid ${item === null ? UI.border : itemColour(item)};border-radius:3px;` +
          'display:flex;flex-direction:column;align-items:center;justify-content:center;' +
          'padding:4px;text-align:center;cursor:pointer;overflow:hidden;',
      );
      cell.dataset['d2rim'] = `slot-${slot}`;
      cell.appendChild(
        el(
          'div',
          `color:${UI.textDim};font:9px/1.2 ${UI.font};letter-spacing:0.12em;` +
            'text-transform:uppercase;',
          EQUIPMENT_SLOT_LABELS[slot],
        ),
      );
      if (item !== null) {
        cell.appendChild(
          el(
            'div',
            `color:${itemColour(item)};font:600 11px/1.3 ${UI.font};margin-top:4px;` +
              `${isBroken(item) ? 'text-decoration:line-through;opacity:0.6;' : ''}`,
            item.name,
          ),
        );
        this.#attachTooltip(cell, item);
      }
      cell.addEventListener('pointerdown', (event) => {
        event.stopPropagation();
        this.#onSlotPointerDown(slot);
      });
      host.appendChild(cell);
    }
  }

  #renderGrid(rpg: RpgSystem): void {
    const grid = this.#grid;
    if (grid === null) return;
    clearChildren(grid);

    for (const placement of rpg.character.inventory.placements) {
      const item = placement.item;
      const colour = itemColour(item);
      const node = el(
        'div',
        `position:absolute;left:${placement.x * CELL_SIZE}px;top:${placement.y * CELL_SIZE}px;` +
          `width:${item.width * CELL_SIZE - 2}px;height:${item.height * CELL_SIZE - 2}px;` +
          `margin:1px;background:linear-gradient(160deg,rgba(30,24,18,0.95),rgba(12,10,8,0.98));` +
          `border:1px solid ${colour};border-radius:3px;` +
          'display:flex;align-items:center;justify-content:center;padding:3px;' +
          `color:${colour};font:600 10px/1.2 ${UI.font};text-align:center;overflow:hidden;` +
          // Long names on narrow items ("Leather Gloves" in a 2-cell box) must
          // wrap rather than clip; a clipped name is a name the player cannot
          // read at exactly the moment they are deciding whether to keep it.
          'word-break:break-word;' +
          `${isBroken(item) ? 'opacity:0.55;' : ''}`,
        item.name,
      );
      node.dataset['d2rim'] = `item-${item.uid}`;
      this.#attachTooltip(node, item);
      grid.appendChild(node);
    }
  }

  /* -- interaction --------------------------------------------------------- */

  #onGridPointerDown(event: PointerEvent): void {
    event.stopPropagation();
    const rpg = this.#rpg();
    const grid = this.#grid;
    if (rpg === null || grid === null) return;

    const rect = grid.getBoundingClientRect();
    const point = { x: event.clientX, y: event.clientY };
    const inventory = rpg.character.inventory;

    if (this.#held === null) {
      const raw = cellFromPoint(point, rect, CELL_SIZE);
      const item = inventory.itemAt(raw.x, raw.y);
      if (item === null) return;

      if (event.button === 2) {
        // Right-click equips straight from the bag, D2's own shortcut.
        rpg.character.equip(item);
        this.refresh();
        return;
      }

      const placement = inventory.placementOf(item);
      if (placement === null) return;
      const itemRect = {
        left: rect.left + placement.x * CELL_SIZE,
        top: rect.top + placement.y * CELL_SIZE,
      };
      const offset = grabOffset(point, itemRect, CELL_SIZE, item);
      inventory.remove(item);
      rpg.character.touch();
      this.#hold({ item, source: { kind: 'grid', x: placement.x, y: placement.y }, offset });
      this.refresh();
      return;
    }

    const held = this.#held;
    const target = cellFromPoint(point, rect, CELL_SIZE, held.offset);
    const result = inventory.drop(held.item, target.x, target.y);
    if (!result.placed) return;

    rpg.character.touch();
    if (result.displaced !== null) {
      // The swapped item goes onto the cursor, exactly as it does in D2.
      this.#hold({
        item: result.displaced,
        source: { kind: 'grid', x: target.x, y: target.y },
        offset: { x: 0, y: 0 },
      });
    } else {
      this.#hold(null);
    }
    this.refresh();
  }

  #onSlotPointerDown(slot: EquipmentSlot): void {
    const rpg = this.#rpg();
    if (rpg === null) return;
    const character = rpg.character;

    if (this.#held === null) {
      const item = character.equipment.get(slot);
      if (item === null) return;
      if (!character.inventory.hasRoomFor(item)) return;
      character.equipment.unequip(slot);
      character.touch();
      this.#hold({ item, source: { kind: 'equipment', slot }, offset: { x: 0, y: 0 } });
      this.refresh();
      return;
    }

    const held = this.#held;
    if (!character.equipment.accepts(held.item, slot)) return;
    if (!character.canEquip(held.item)) return;

    const previous = character.equipment.get(slot);
    const result = character.equipment.equip(held.item, slot);
    if (!result.equipped) return;
    character.touch();

    // Anything the equip displaced that is not the item we knowingly swapped
    // with goes back into the bag; the swap target goes onto the cursor.
    let next: Held | null = null;
    for (const displaced of result.displaced) {
      if (displaced === previous && next === null) {
        next = { item: displaced, source: { kind: 'equipment', slot }, offset: { x: 0, y: 0 } };
        continue;
      }
      character.inventory.add(displaced);
    }
    this.#hold(next);
    this.refresh();
  }

  #hold(held: Held | null): void {
    this.#held = held;
    const cursor = this.#cursor;
    if (cursor === null) return;
    if (held === null) {
      cursor.style.display = 'none';
      clearChildren(cursor);
      return;
    }
    clearChildren(cursor);
    const colour = itemColour(held.item);
    cursor.appendChild(
      el(
        'div',
        `width:${held.item.width * CELL_SIZE}px;height:${held.item.height * CELL_SIZE}px;` +
          `background:rgba(12,10,8,0.92);border:1px solid ${colour};border-radius:3px;` +
          `color:${colour};font:600 10px/1.2 ${UI.font};display:flex;align-items:center;` +
          'justify-content:center;text-align:center;padding:3px;opacity:0.9;',
        held.item.name,
      ),
    );
    cursor.style.display = '';
  }

  /** Put a held item back where it came from, or anywhere at all. */
  #returnHeld(): void {
    const held = this.#held;
    const rpg = this.#rpg();
    if (held === null || rpg === null) {
      this.#hold(null);
      return;
    }
    const inventory = rpg.character.inventory;
    if (held.source.kind === 'grid') {
      if (!inventory.place(held.item, held.source.x, held.source.y)) inventory.add(held.item);
    } else if (!rpg.character.equip(held.item, held.source.slot).equipped) {
      inventory.add(held.item);
    }
    rpg.character.touch();
    this.#hold(null);
    this.refresh();
  }

  #moveCursor(event: PointerEvent): void {
    const cursor = this.#cursor;
    if (cursor === null || this.#held === null) return;
    cursor.style.left = `${event.clientX}px`;
    cursor.style.top = `${event.clientY}px`;
  }

  /* -- tooltip ------------------------------------------------------------- */

  #attachTooltip(node: HTMLElement, item: Item): void {
    node.addEventListener('pointerenter', (event) => this.#showTooltip(item, event));
    node.addEventListener('pointerleave', () => this.#hideTooltip());
  }

  #showTooltip(item: Item, event: PointerEvent): void {
    const tooltip = this.#tooltip;
    const rpg = this.#rpg();
    if (tooltip === null) return;
    clearChildren(tooltip);

    tooltip.appendChild(
      el('div', `color:${itemColour(item)};font:600 14px/1.4 ${UI.font};`, item.name),
    );
    for (const line of describeItem(item)) {
      tooltip.appendChild(el('div', `color:${UI.text};font:12px/1.6 ${UI.font};`, line));
    }
    if (rpg !== null) {
      const derived = rpg.character.derived;
      const unmet = unmetRequirement(item, {
        level: derived.level,
        strength: derived.attributes.strength,
        dexterity: derived.attributes.dexterity,
      });
      if (unmet !== null) {
        tooltip.appendChild(
          el(
            'div',
            `color:${UI.danger};font:600 12px/1.6 ${UI.font};margin-top:4px;`,
            `You cannot use this yet — ${unmet} too low`,
          ),
        );
      }
    }
    if (isBroken(item)) {
      tooltip.appendChild(
        el('div', `color:${UI.danger};font:600 12px/1.6 ${UI.font};`, 'Broken — see Charsi'),
      );
    }

    tooltip.style.display = '';
    tooltip.style.left = `${Math.min(event.clientX + 16, window.innerWidth - 300)}px`;
    tooltip.style.top = `${Math.min(event.clientY + 16, window.innerHeight - 220)}px`;
  }

  #hideTooltip(): void {
    if (this.#tooltip !== null) this.#tooltip.style.display = 'none';
  }

  #rpg(): RpgSystem | null {
    return this.#ctx?.services.tryGet<RpgSystem>(RpgSystemKey) ?? null;
  }
}
