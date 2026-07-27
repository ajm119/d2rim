/**
 * @module ui/VendorScreen
 *
 * The shop: the merchant's shelf on the left, the player's bag on the right,
 * and the gold that moves between them along the top.
 *
 * Two lists rather than a grid. The grid is the *inventory's* constraint — it
 * is where "is this sword worth two rings" gets asked — and reproducing it in
 * the shop would only mean the player has to solve a packing problem before
 * they can read a price. What a shop needs is a name, a price and a column of
 * modifiers, so that is what it shows.
 *
 * A purchase that will not fit is refused with a reason rather than silently
 * ignored, because "I clicked buy and nothing happened" is the shop bug players
 * report and never diagnose.
 */

import type { GameContext, GameModule } from '../core/types';
import { describeItem, itemColour, type Item } from '../rpg/ItemGenerator';
import { RpgSystemKey, type RpgSystem } from '../rpg/RpgSystem';
import type { Vendor } from '../rpg/Vendor';
import { buttonStyle, clearChildren, el, hasDom, headingStyle, panelStyle, scrimStyle, UI, Z } from './theme';
import { UiManagerKey, type UiManager, type UiScreen } from './UiManager';

export class VendorScreen implements GameModule, UiScreen {
  readonly name = 'ui.vendor';
  readonly id = 'vendor' as const;

  readonly root: HTMLElement;

  #ctx: GameContext | null = null;
  #title: HTMLDivElement | null = null;
  #goldLine: HTMLDivElement | null = null;
  #stockHost: HTMLDivElement | null = null;
  #bagHost: HTMLDivElement | null = null;
  #status: HTMLDivElement | null = null;
  #vendorId: string | null = null;
  readonly #disposers: Array<() => void> = [];

  constructor() {
    this.root = hasDom()
      ? el('div', scrimStyle(Z.screen))
      : ({ style: {} } as unknown as HTMLElement);
  }

  init(ctx: GameContext): void {
    this.#ctx = ctx;
    if (!hasDom()) return;
    this.#build();
    ctx.services.tryGet<UiManager>(UiManagerKey)?.register(this);
    this.#disposers.push(
      ctx.events.on('npc:vendor', (payload) => {
        this.#vendorId = payload.vendorId;
        ctx.services.tryGet<UiManager>(UiManagerKey)?.open('vendor');
        this.refresh();
      }),
    );
  }

  dispose(): void {
    for (const off of this.#disposers) off();
    this.#disposers.length = 0;
    this.root.remove();
    this.#ctx = null;
  }

  /** Open on a specific merchant. Used by the drive harness. */
  show(vendorId: string): void {
    this.#vendorId = vendorId;
    this.#ctx?.services.tryGet<UiManager>(UiManagerKey)?.open('vendor');
    this.refresh();
  }

  refresh(): void {
    const rpg = this.#rpg();
    const vendor = this.#vendor();
    if (rpg === null || vendor === null) return;

    if (this.#title !== null) this.#title.textContent = vendor.definition.displayName;
    if (this.#goldLine !== null) this.#goldLine.textContent = `${rpg.character.gold} gold`;

    this.#renderStock(rpg, vendor);
    this.#renderBag(rpg, vendor);
  }

  /* -- construction -------------------------------------------------------- */

  #build(): void {
    this.root.style.display = 'none';

    const panel = el(
      'div',
      panelStyle('width:min(900px,94vw);max-height:90vh;overflow:auto;padding:20px 22px;'),
    );
    panel.addEventListener('pointerdown', (event) => event.stopPropagation());
    this.root.addEventListener('pointerdown', () => {
      this.#ctx?.services.tryGet<UiManager>(UiManagerKey)?.close('vendor');
    });

    const head = el('div', 'display:flex;justify-content:space-between;align-items:baseline;margin-bottom:14px;');
    const title = el('div', headingStyle('margin:0;'), 'Merchant');
    const gold = el('div', `color:${UI.accent};font:600 14px/1 ${UI.fontMono};`, '0 gold');
    head.append(title, gold);
    this.#title = title;
    this.#goldLine = gold;

    const columns = el('div', 'display:grid;grid-template-columns:1fr 1fr;gap:18px;');
    const left = el('div', '');
    left.appendChild(el('div', headingStyle(), 'For sale'));
    const stock = el('div', 'display:flex;flex-direction:column;gap:6px;');
    left.appendChild(stock);
    this.#stockHost = stock;

    const right = el('div', '');
    right.appendChild(el('div', headingStyle(), 'Your pack'));
    const bag = el('div', 'display:flex;flex-direction:column;gap:6px;');
    right.appendChild(bag);
    this.#bagHost = bag;

    columns.append(left, right);

    const status = el('div', `color:${UI.textDim};font:12px/1.6 ${UI.font};margin-top:12px;min-height:20px;`);
    this.#status = status;

    const footer = el('div', 'display:flex;gap:10px;margin-top:12px;');
    const repair = el('button', buttonStyle(), 'Repair all');
    repair.dataset['d2rim'] = 'vendor-repair';
    repair.addEventListener('click', (event) => {
      event.stopPropagation();
      this.#repair();
    });
    const close = el('button', buttonStyle(), 'Leave');
    close.addEventListener('click', (event) => {
      event.stopPropagation();
      this.#ctx?.services.tryGet<UiManager>(UiManagerKey)?.close('vendor');
    });
    footer.append(repair, close);

    panel.append(head, columns, status, footer);
    this.root.appendChild(panel);
  }

  /* -- rendering ----------------------------------------------------------- */

  #renderStock(rpg: RpgSystem, vendor: Vendor): void {
    const host = this.#stockHost;
    if (host === null) return;
    clearChildren(host);
    vendor.stock.forEach((item, index) => {
      const price = vendor.priceToBuy(item);
      const affordable = rpg.character.gold >= price;
      host.appendChild(
        this.#row(item, `${price} g`, affordable ? UI.accent : UI.danger, 'Buy', () => {
          const result = vendor.buy(index, rpg.character);
          this.#say(
            result.result === 'ok'
              ? `Bought ${item.name} for ${result.gold} gold.`
              : refusal(result.result),
          );
          this.refresh();
        }),
      );
    });
    if (vendor.stock.length === 0) {
      host.appendChild(el('div', `color:${UI.textDim};font:13px/1.6 ${UI.font};`, 'Nothing in stock.'));
    }
  }

  #renderBag(rpg: RpgSystem, vendor: Vendor): void {
    const host = this.#bagHost;
    if (host === null) return;
    clearChildren(host);
    for (const item of rpg.character.inventory.items) {
      const price = vendor.priceToSell(item);
      const accepted = vendor.accepts(item);
      host.appendChild(
        this.#row(item, `${price} g`, accepted ? UI.text : UI.textDim, 'Sell', () => {
          const result = vendor.sell(item, rpg.character);
          this.#say(
            result.result === 'ok'
              ? `Sold ${item.name} for ${-result.gold} gold.`
              : refusal(result.result),
          );
          this.refresh();
        }),
      );
    }
    if (rpg.character.inventory.items.length === 0) {
      host.appendChild(el('div', `color:${UI.textDim};font:13px/1.6 ${UI.font};`, 'Your pack is empty.'));
    }
  }

  #row(
    item: Item,
    price: string,
    priceColour: string,
    action: string,
    onClick: () => void,
  ): HTMLDivElement {
    const row = el(
      'div',
      `display:flex;align-items:center;gap:10px;padding:7px 9px;border-radius:3px;` +
        `border:1px solid ${UI.border};background:${UI.panelRaised};`,
    );
    row.dataset['d2rim'] = `vendor-item-${item.uid}`;

    const info = el('div', 'flex:1;min-width:0;');
    info.appendChild(el('div', `color:${itemColour(item)};font:600 13px/1.4 ${UI.font};`, item.name));
    const lines = describeItem(item).slice(0, 3).join(' · ');
    if (lines.length > 0) {
      info.appendChild(
        el(
          'div',
          `color:${UI.textDim};font:11px/1.4 ${UI.font};overflow:hidden;text-overflow:ellipsis;white-space:nowrap;`,
          lines,
        ),
      );
    }

    const priceNode = el('div', `color:${priceColour};font:600 12px/1 ${UI.fontMono};min-width:56px;text-align:right;`, price);
    const button = el('button', buttonStyle('padding:5px 11px;font-size:12px;'), action);
    button.addEventListener('click', (event) => {
      event.stopPropagation();
      onClick();
    });

    row.append(info, priceNode, button);
    return row;
  }

  #repair(): void {
    const rpg = this.#rpg();
    const vendor = this.#vendor();
    if (rpg === null || vendor === null) return;
    const result = vendor.repairAll(rpg.character);
    this.#say(
      result.result === 'ok' ? `Everything repaired for ${result.gold} gold.` : refusal(result.result),
    );
    this.refresh();
  }

  #say(message: string): void {
    if (this.#status !== null) this.#status.textContent = message;
  }

  #vendor(): Vendor | null {
    const rpg = this.#rpg();
    if (rpg === null || this.#vendorId === null) return null;
    return rpg.vendors.get(this.#vendorId);
  }

  #rpg(): RpgSystem | null {
    return this.#ctx?.services.tryGet<RpgSystem>(RpgSystemKey) ?? null;
  }
}

/** Turn a refusal code into a sentence. */
function refusal(result: string): string {
  switch (result) {
    case 'cannot-afford':
      return 'You cannot afford that.';
    case 'inventory-full':
      return 'Your pack is full.';
    case 'not-accepted':
      return 'This merchant does not deal in that.';
    case 'nothing-to-repair':
      return 'Nothing needs repairing.';
    case 'no-repair-service':
      return 'This merchant does not repair.';
    case 'not-owned':
      return 'You are not carrying that.';
    default:
      return 'That is not possible.';
  }
}
