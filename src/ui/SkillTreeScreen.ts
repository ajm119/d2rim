/**
 * @module ui/SkillTreeScreen
 *
 * The Barbarian's skill tree: two tiers, four skills, and the prerequisite
 * lines between them.
 *
 * ### What the screen has to communicate
 *
 * Three things, in this order of importance:
 *
 * 1. **Can I spend a point here, and if not, why?** Every skill shows its own
 *    refusal reason in place — "Requires level 6", "Requires Bash" — rather
 *    than greying out silently. A greyed skill with no explanation is the most
 *    common way a tree screen wastes a player's time.
 * 2. **What does the next point actually do?** The summary line prints the
 *    numbers at the *current* investment and the per-point rate, so the
 *    decision is arithmetic rather than a guess.
 * 3. **What is selected right now?** The active skill is the one that changes
 *    the next swing, so it is bordered in the accent colour and can be changed
 *    with a click on any skill that has a point in it.
 *
 * The prerequisite arrow is drawn as a plain bordered element rather than an
 * SVG path: with four skills in a fixed two-by-two layout there is exactly one
 * arrow, and a layout engine for one arrow is not a good trade.
 */

import type { GameContext, GameModule } from '../core/types';
import { RpgSystemKey, type RpgSystem } from '../rpg/RpgSystem';
import { findSkill, type SkillDefinition } from '../rpg/SkillTree';
import { buttonStyle, clearChildren, el, hasDom, headingStyle, panelStyle, scrimStyle, UI, Z } from './theme';
import { UiManagerKey, type UiManager, type UiScreen } from './UiManager';

export class SkillTreeScreen implements GameModule, UiScreen {
  readonly name = 'ui.skills';
  readonly id = 'skills' as const;

  readonly root: HTMLElement;

  #ctx: GameContext | null = null;
  #body: HTMLDivElement | null = null;
  #header: HTMLDivElement | null = null;
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
      ctx.events.on('rpg:characterChanged', () => {
        if (this.root.style.display !== 'none') this.refresh();
      }),
    );
  }

  dispose(): void {
    for (const off of this.#disposers) off();
    this.#disposers.length = 0;
    this.root.remove();
    this.#ctx = null;
  }

  refresh(): void {
    const rpg = this.#rpg();
    const body = this.#body;
    const header = this.#header;
    if (rpg === null || body === null || header === null) return;

    const tree = rpg.character.skills;
    header.textContent =
      tree.available > 0
        ? `${tree.available} skill point${tree.available === 1 ? '' : 's'} to spend`
        : 'No skill points to spend';
    header.style.color = tree.available > 0 ? UI.accent : UI.textDim;

    clearChildren(body);
    const tiers = new Map<number, SkillDefinition[]>();
    for (const skill of tree.skills) {
      const list = tiers.get(skill.tier) ?? [];
      list.push(skill);
      tiers.set(skill.tier, list);
    }

    for (const [tier, skills] of [...tiers.entries()].sort((a, b) => a[0] - b[0])) {
      const row = el(
        'div',
        'display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin-bottom:14px;',
      );
      row.dataset['d2rim'] = `skill-tier-${tier}`;
      for (const skill of [...skills].sort((a, b) => a.column - b.column)) {
        row.appendChild(this.#skillCard(rpg, skill));
      }
      body.appendChild(row);
    }
  }

  /* -- construction -------------------------------------------------------- */

  #build(): void {
    this.root.style.display = 'none';

    const panel = el(
      'div',
      panelStyle('width:min(680px,94vw);max-height:92vh;overflow:auto;padding:22px 24px;'),
    );
    panel.addEventListener('pointerdown', (event) => event.stopPropagation());
    this.root.addEventListener('pointerdown', () => {
      this.#ctx?.services.tryGet<UiManager>(UiManagerKey)?.close('skills');
    });

    panel.appendChild(el('div', headingStyle(), 'Barbarian Skills'));
    const header = el('div', `color:${UI.textDim};font:13px/1.6 ${UI.font};margin-bottom:16px;`);
    this.#header = header;
    panel.appendChild(header);

    const body = el('div', '');
    this.#body = body;
    panel.appendChild(body);

    panel.appendChild(
      el(
        'div',
        `color:${UI.textDim};font:12px/1.7 ${UI.font};margin-top:6px;border-top:1px solid ${UI.border};padding-top:10px;`,
        'Click a skill to make it active. Click SPEND to invest a point. K closes.',
      ),
    );
    this.root.appendChild(panel);
  }

  #skillCard(rpg: RpgSystem, skill: SkillDefinition): HTMLDivElement {
    const tree = rpg.character.skills;
    const points = tree.pointsIn(skill.id);
    const check = tree.canInvest(skill.id);
    const isActive = tree.active === skill.id;
    const border = isActive ? UI.accent : points > 0 ? UI.borderBright : UI.border;

    const card = el(
      'div',
      `background:${UI.panelRaised};border:1px solid ${border};border-radius:4px;` +
        `padding:12px 14px;display:flex;flex-direction:column;gap:6px;` +
        `${points > 0 ? '' : 'opacity:0.86;'}`,
    );
    card.dataset['d2rim'] = `skill-${skill.id}`;

    const title = el('div', 'display:flex;justify-content:space-between;align-items:baseline;gap:8px;');
    title.append(
      el('span', `color:${points > 0 ? UI.accent : UI.text};font:600 15px/1.3 ${UI.font};`, skill.name),
      el(
        'span',
        `color:${UI.textDim};font:600 12px/1.3 ${UI.fontMono};`,
        `${points} / ${skill.maxPoints}`,
      ),
    );
    card.appendChild(title);
    card.appendChild(
      el(
        'div',
        `color:${UI.textDim};font:11px/1.4 ${UI.font};letter-spacing:0.1em;text-transform:uppercase;`,
        skill.kind === 'passive' ? 'Passive' : 'Active',
      ),
    );
    card.appendChild(el('div', `color:${UI.text};font:13px/1.5 ${UI.font};`, skill.description));
    card.appendChild(
      el('div', `color:${UI.accent};font:12px/1.5 ${UI.fontMono};`, skill.summaryAt(points)),
    );

    const footer = el('div', 'display:flex;gap:8px;align-items:center;margin-top:4px;');
    const spend = el('button', buttonStyle('padding:6px 12px;font-size:12px;'), 'Spend');
    spend.dataset['d2rim'] = `spend-${skill.id}`;
    if (!check.allowed) {
      spend.disabled = true;
      spend.style.opacity = '0.45';
      spend.style.cursor = 'default';
    }
    spend.addEventListener('click', (event) => {
      event.stopPropagation();
      rpg.investSkill(skill.id);
      this.refresh();
    });
    footer.appendChild(spend);

    if (!check.allowed) {
      footer.appendChild(
        el('span', `color:${UI.textDim};font:12px/1.5 ${UI.font};`, refusalText(check.reason, check.missing, skill)),
      );
    } else if (isActive) {
      footer.appendChild(el('span', `color:${UI.accent};font:12px/1.5 ${UI.font};`, 'Active'));
    }
    card.appendChild(footer);

    if (skill.kind === 'active' && points > 0) {
      card.style.cursor = 'pointer';
      card.addEventListener('pointerdown', (event) => {
        event.stopPropagation();
        rpg.selectSkill(skill.id);
        this.refresh();
      });
    } else {
      card.addEventListener('pointerdown', (event) => event.stopPropagation());
    }
    return card;
  }

  #rpg(): RpgSystem | null {
    return this.#ctx?.services.tryGet<RpgSystem>(RpgSystemKey) ?? null;
  }
}

/** Turn a refusal into the sentence the player needs. */
function refusalText(
  reason: string,
  missing: string | undefined,
  skill: SkillDefinition,
): string {
  switch (reason) {
    case 'level':
      return `Requires level ${skill.requiredLevel}`;
    case 'prerequisite':
      return `Requires ${findSkill(missing ?? '')?.name ?? missing ?? 'another skill'}`;
    case 'no-points':
      return 'No points available';
    case 'maxed':
      return 'Fully invested';
    default:
      return '';
  }
}
