/**
 * @module quest/Dialogue
 *
 * A branching dialogue system with a typed script format and quest-state
 * conditions.
 *
 * ### The format
 *
 * A {@link DialogueTree} is a flat map of {@link DialogueNode}s plus a list of
 * *entry rules*. Starting a conversation walks the entry rules in order and
 * takes the first whose condition passes — which is the whole mechanism behind
 * "Akara says something different before, during and after the Den of Evil".
 * The alternative, a single root node with a cascade of conditional choices,
 * puts the branching in the wrong place: the interesting variation is in what
 * she *opens with*, not in what the player can say back.
 *
 * Nodes are addressed by id and never by array position, so inserting a line
 * into the middle of a conversation cannot silently re-point a branch.
 *
 * ### Conditions
 *
 * A condition is a plain predicate over a {@link DialogueContext}. Building
 * them by hand is possible; {@link whenQuest} and friends exist so the common
 * ones read as English at the call site:
 *
 * ```ts
 * { condition: whenQuest('den-of-evil', 'complete'), next: 'akara.thanks' }
 * ```
 *
 * ### Actions
 *
 * A choice may carry an {@link DialogueAction}: accept a quest, hand one in,
 * open a vendor, repair, heal. The runner does not execute them — it *reports*
 * them, and the module wiring dialogue into the game decides what "open a
 * vendor" means. That keeps this file free of the RPG layer and makes the whole
 * branching system testable with no engine, no scene and no character.
 */

import type { QuestState, QuestSystem } from './QuestSystem';
import { questStateAtLeast } from './QuestSystem';

/* -------------------------------------------------------------------------- */
/* Context and conditions                                                      */
/* -------------------------------------------------------------------------- */

/** Everything a dialogue condition or dynamic line is allowed to read. */
export interface DialogueContext {
  /** Quest state. Optional so a tree can be exercised without a quest system. */
  readonly quests: QuestSystem | null;
  /** Who the player is talking to. */
  readonly npcId: string;
  readonly characterLevel: number;
  readonly gold: number;
  /** Arbitrary named booleans set by the game, e.g. `'hasBrokenGear'`. */
  readonly flags: ReadonlySet<string>;
}

export type DialogueCondition = (ctx: DialogueContext) => boolean;

/** True when the quest is in exactly one of `states`. */
export function whenQuest(questId: string, ...states: readonly QuestState[]): DialogueCondition {
  return (ctx) => {
    const state = ctx.quests?.state(questId) ?? 'inactive';
    return states.includes(state);
  };
}

/** True when the quest has reached `state` or gone past it. */
export function whenQuestAtLeast(questId: string, state: QuestState): DialogueCondition {
  return (ctx) => questStateAtLeast(ctx.quests?.state(questId) ?? 'inactive', state);
}

/** True when a named flag is set. */
export function whenFlag(flag: string): DialogueCondition {
  return (ctx) => ctx.flags.has(flag);
}

/** Logical helpers, so a script never needs an inline arrow with a `&&` in it. */
export function allOf(...conditions: readonly DialogueCondition[]): DialogueCondition {
  return (ctx) => conditions.every((condition) => condition(ctx));
}

export function anyOf(...conditions: readonly DialogueCondition[]): DialogueCondition {
  return (ctx) => conditions.some((condition) => condition(ctx));
}

export function not(condition: DialogueCondition): DialogueCondition {
  return (ctx) => !condition(ctx);
}

/* -------------------------------------------------------------------------- */
/* Script format                                                               */
/* -------------------------------------------------------------------------- */

/** What a choice does, beyond moving to another node. */
export type DialogueAction =
  | { readonly kind: 'offerQuest'; readonly questId: string }
  | { readonly kind: 'acceptQuest'; readonly questId: string }
  | { readonly kind: 'declineQuest'; readonly questId: string }
  | { readonly kind: 'turnInQuest'; readonly questId: string }
  | { readonly kind: 'openVendor'; readonly npcId: string }
  | { readonly kind: 'repair' }
  | { readonly kind: 'heal' }
  | { readonly kind: 'travel'; readonly zoneId: string; readonly entryPoint?: string }
  | { readonly kind: 'end' };

export interface DialogueChoice {
  readonly id: string;
  readonly text: string;
  /** Node to move to. Omitted or null ends the conversation. */
  readonly next?: string | null;
  readonly action?: DialogueAction;
  /** Hidden entirely when this fails, rather than shown greyed out. */
  readonly condition?: DialogueCondition;
}

export interface DialogueNode {
  readonly id: string;
  /** Display name of whoever is speaking. Defaults to the NPC's name. */
  readonly speaker?: string;
  /** The line. A function when it needs to read the context. */
  readonly text: string | ((ctx: DialogueContext) => string);
  readonly choices?: readonly DialogueChoice[];
  /**
   * Where a node with no choices goes when the player presses continue. Absent
   * means the conversation ends.
   */
  readonly next?: string;
  /** Fired when the node is *entered*. Used by "talk to X" objectives. */
  readonly onEnter?: DialogueAction;
}

/** Picks an opening node. Walked in order; first match wins. */
export interface DialogueEntry {
  readonly node: string;
  readonly condition?: DialogueCondition;
}

export interface DialogueTree {
  readonly id: string;
  readonly npcId: string;
  readonly speaker: string;
  readonly entry: readonly DialogueEntry[];
  readonly nodes: readonly DialogueNode[];
}

/* -------------------------------------------------------------------------- */
/* Runtime                                                                     */
/* -------------------------------------------------------------------------- */

/** What the UI needs to draw one beat of a conversation. */
export interface DialogueView {
  readonly nodeId: string;
  readonly speaker: string;
  readonly text: string;
  /** Choices whose conditions pass, in script order. */
  readonly choices: readonly { readonly id: string; readonly text: string }[];
  /** True when the only thing the player can do is continue or close. */
  readonly terminal: boolean;
}

/** Executes the side effect of an action. Returns whether it succeeded. */
export type DialogueActionHandler = (action: DialogueAction, ctx: DialogueContext) => void;

/**
 * A conversation in flight.
 *
 * One runner may be reused across NPCs; {@link start} resets it.
 */
export class DialogueRunner {
  #tree: DialogueTree | null = null;
  #node: DialogueNode | null = null;
  #context: DialogueContext | null = null;
  #handler: DialogueActionHandler | null = null;

  setActionHandler(handler: DialogueActionHandler | null): void {
    this.#handler = handler;
  }

  get active(): boolean {
    return this.#node !== null;
  }

  get tree(): DialogueTree | null {
    return this.#tree;
  }

  get nodeId(): string | null {
    return this.#node?.id ?? null;
  }

  /**
   * Begin a conversation.
   *
   * @returns the opening view, or null when no entry rule matched — which is a
   * content bug, and it is better to say nothing than to open an empty box.
   */
  start(tree: DialogueTree, ctx: DialogueContext): DialogueView | null {
    this.#tree = tree;
    this.#context = ctx;
    const entry = tree.entry.find((rule) => rule.condition === undefined || rule.condition(ctx));
    if (entry === undefined) {
      this.#node = null;
      return null;
    }
    return this.#enter(entry.node);
  }

  /** The current view, recomputed against the live context. */
  view(): DialogueView | null {
    const node = this.#node;
    const tree = this.#tree;
    const ctx = this.#context;
    if (node === null || tree === null || ctx === null) return null;
    const choices = (node.choices ?? []).filter(
      (choice) => choice.condition === undefined || choice.condition(ctx),
    );
    return {
      nodeId: node.id,
      speaker: node.speaker ?? tree.speaker,
      text: typeof node.text === 'function' ? node.text(ctx) : node.text,
      choices: choices.map((choice) => ({ id: choice.id, text: choice.text })),
      terminal: choices.length === 0,
    };
  }

  /**
   * Take a choice by its index within the *visible* list.
   *
   * Index rather than id because that is what a keyboard-driven overlay has;
   * {@link chooseById} exists for tests and for click handlers, which have the
   * id to hand and should not have to recompute the visible index.
   */
  choose(index: number): DialogueView | null {
    const visible = this.#visibleChoices();
    const choice = visible[index];
    if (choice === undefined) return this.view();
    return this.chooseById(choice.id);
  }

  chooseById(choiceId: string): DialogueView | null {
    const node = this.#node;
    const ctx = this.#context;
    if (node === null || ctx === null) return null;
    const choice = this.#visibleChoices().find((entry) => entry.id === choiceId);
    if (choice === undefined) return this.view();

    if (choice.action !== undefined) this.#handler?.(choice.action, ctx);
    if (choice.action?.kind === 'end' || choice.next === undefined || choice.next === null) {
      this.end();
      return null;
    }
    return this.#enter(choice.next);
  }

  /** Advance a node with no choices. Ends the conversation when there is no `next`. */
  advance(): DialogueView | null {
    const node = this.#node;
    if (node === null) return null;
    if (this.#visibleChoices().length > 0) return this.view();
    if (node.next === undefined) {
      this.end();
      return null;
    }
    return this.#enter(node.next);
  }

  end(): void {
    this.#node = null;
    this.#tree = null;
    this.#context = null;
  }

  #visibleChoices(): DialogueChoice[] {
    const node = this.#node;
    const ctx = this.#context;
    if (node === null || ctx === null) return [];
    return (node.choices ?? []).filter(
      (choice) => choice.condition === undefined || choice.condition(ctx),
    );
  }

  #enter(nodeId: string): DialogueView | null {
    const tree = this.#tree;
    const ctx = this.#context;
    if (tree === null || ctx === null) return null;
    const node = tree.nodes.find((entry) => entry.id === nodeId);
    if (node === undefined) {
      console.warn(`[Dialogue] tree "${tree.id}" has no node "${nodeId}"`);
      this.end();
      return null;
    }
    this.#node = node;
    if (node.onEnter !== undefined) this.#handler?.(node.onEnter, ctx);
    return this.view();
  }
}

/**
 * Structural validation of a tree.
 *
 * Every `next`, every entry rule and every choice target must name a node that
 * exists. Content bugs of this kind are invisible until a player takes exactly
 * that branch, which in a five-NPC camp can be weeks; a two-line test that calls
 * this on every tree catches all of them at once.
 *
 * @returns the problems found. Empty means the tree is sound.
 */
export function validateDialogueTree(tree: DialogueTree): string[] {
  const problems: string[] = [];
  const ids = new Set(tree.nodes.map((node) => node.id));

  if (tree.nodes.length !== ids.size) problems.push(`${tree.id}: duplicate node ids`);
  if (tree.entry.length === 0) problems.push(`${tree.id}: no entry rules`);
  // The last entry rule must be unconditional, or a state exists in which the
  // NPC simply cannot be talked to.
  const last = tree.entry[tree.entry.length - 1];
  if (last !== undefined && last.condition !== undefined) {
    problems.push(`${tree.id}: the final entry rule must be unconditional`);
  }
  for (const rule of tree.entry) {
    if (!ids.has(rule.node)) problems.push(`${tree.id}: entry points at missing node "${rule.node}"`);
  }
  for (const node of tree.nodes) {
    if (node.next !== undefined && !ids.has(node.next)) {
      problems.push(`${tree.id}: node "${node.id}" continues to missing node "${node.next}"`);
    }
    const choiceIds = new Set<string>();
    for (const choice of node.choices ?? []) {
      if (choiceIds.has(choice.id)) {
        problems.push(`${tree.id}: node "${node.id}" has duplicate choice id "${choice.id}"`);
      }
      choiceIds.add(choice.id);
      if (choice.next !== undefined && choice.next !== null && !ids.has(choice.next)) {
        problems.push(
          `${tree.id}: choice "${choice.id}" on "${node.id}" points at missing node "${choice.next}"`,
        );
      }
    }
  }
  return problems;
}
