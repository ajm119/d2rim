/**
 * @module ui/theme
 *
 * Shared visual tokens and DOM helpers for the overlay UI.
 *
 * ### Why the UI is DOM and not in-scene geometry
 *
 * Text. A quest tracker, an item tooltip and a skill description are dense,
 * small, and must stay sharp at every resolution; a font atlas rendered into
 * the frame graph is a worse version of what the browser already does
 * perfectly, and it would cost a full-screen pass plus a texture upload for
 * something three `<div>`s and a `border-radius` do better. It also keeps the
 * UI entirely out of the renderer's way — no depth interactions, no
 * post-processing eating the contrast, no tonemapping crushing the text.
 *
 * ### The legibility rule
 *
 * The game frame is dark, grimdark-graded and currently under-exposed. So every
 * panel here sits on an **opaque** backdrop rather than a translucent one, with
 * a light border and text at a contrast ratio that survives whatever is behind
 * it. A translucent D2-style panel looks better in a screenshot of a bright
 * scene and becomes unreadable the moment the player walks into a cave.
 *
 * ### The pointer-events rule
 *
 * The overlay root is `pointer-events: none` and only *open* panels turn it
 * back on. A UI layer that swallows clicks while closed is indistinguishable
 * from a broken camera, and it is the single most common overlay bug.
 */

/** The palette. Warm iron and parchment, in Diablo II's register. */
export const UI = {
  /** Panel background. Opaque on purpose — see the module header. */
  panel: '#100c09',
  panelRaised: '#191310',
  /** Cell and slot wells. */
  well: '#0a0806',
  border: '#4a3a26',
  borderBright: '#7a6136',
  /** Body text. */
  text: '#e6dcc8',
  textDim: '#9c8f78',
  /** Headings and gold. */
  accent: '#d9b262',
  /** Health, mana, experience. */
  health: '#c0392b',
  mana: '#3b6fd0',
  experience: '#c8a44a',
  /** Warnings, unmet requirements. */
  danger: '#d9603f',
  ok: '#7bbd68',
  /** The typeface used everywhere in the overlay. */
  font: "'Trebuchet MS', 'Segoe UI', Georgia, serif",
  fontMono: 'ui-monospace, Menlo, Consolas, monospace',
} as const;

/** One inventory cell, in CSS pixels. D2's is 29; this reads better at 1080p. */
export const CELL_SIZE = 34;

/** Stacking order. The HUD sits under the screens, which sit under dialogue. */
export const Z = {
  hud: 35,
  screen: 60,
  dialogue: 70,
  menu: 80,
  cursor: 95,
} as const;

/** `document` is absent in the unit-test environment; every helper guards it. */
export function hasDom(): boolean {
  return typeof document !== 'undefined';
}

/**
 * A screen's root element, or an inert stand-in when there is no DOM.
 *
 * The unit-test environment is Node: `document` does not exist, and a screen
 * module must still be constructible so that the parts of it that are not DOM
 * can be imported and exercised. The stand-in carries only the `style` object,
 * which is the entire surface `UiManager` touches before a screen's `init`
 * decides there is nothing to build.
 *
 * The cast is a genuine environment boundary and is deliberately confined to
 * this one function rather than repeated in every screen.
 */
export function screenRoot(style: string): HTMLElement {
  if (!hasDom()) return { style: {} } as unknown as HTMLElement;
  return el('div', style);
}

/** Create an element with an inline style string and optional text. */
export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  style: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  node.setAttribute('style', style);
  if (text !== undefined) node.textContent = text;
  return node;
}

/**
 * A framed panel.
 *
 * The double border — a bright inner line inside a dark outer one — is what
 * separates the panel from a dark background without a drop shadow that would
 * smear over the scene behind it.
 */
export function panelStyle(extra = ''): string {
  return (
    `background:${UI.panel};border:1px solid ${UI.borderBright};` +
    `box-shadow:0 0 0 1px #000, 0 12px 40px rgba(0,0,0,0.75);` +
    `color:${UI.text};font:14px/1.5 ${UI.font};border-radius:4px;` +
    `${extra}`
  );
}

/** A panel heading: small caps, letterspaced, in the accent colour. */
export function headingStyle(extra = ''): string {
  return (
    `color:${UI.accent};font:600 13px/1.2 ${UI.font};letter-spacing:0.16em;` +
    `text-transform:uppercase;margin:0 0 10px 0;${extra}`
  );
}

/** A clickable button in the panel idiom. */
export function buttonStyle(extra = ''): string {
  return (
    `background:${UI.panelRaised};border:1px solid ${UI.border};color:${UI.text};` +
    `font:13px/1 ${UI.font};padding:8px 14px;border-radius:3px;cursor:pointer;` +
    `letter-spacing:0.04em;${extra}`
  );
}

/**
 * A screen-filling scrim behind a modal panel.
 *
 * Dark enough that the panel reads as the focus, transparent enough that the
 * player can still see they are standing in a camp and not a menu.
 */
export function scrimStyle(z: number): string {
  return (
    `position:fixed;inset:0;z-index:${z};background:rgba(4,3,2,0.62);` +
    `display:flex;align-items:center;justify-content:center;pointer-events:auto;`
  );
}

/** Clamp a fraction into `[0, 1]`, treating `NaN` as 0. */
export function fraction(value: number, max: number): number {
  if (!Number.isFinite(value) || !Number.isFinite(max) || max <= 0) return 0;
  return Math.max(0, Math.min(1, value / max));
}

/** Remove every child of a node. Cheaper and safer than `innerHTML = ''`. */
export function clearChildren(node: Element): void {
  while (node.firstChild !== null) node.removeChild(node.firstChild);
}

/**
 * Grid coordinates for a point inside a grid element.
 *
 * Pure, exported and unit tested because drag-and-drop correctness lives
 * entirely in this function: an off-by-one here puts every dropped item one
 * cell from where the player let go, and it is invisible in a screenshot.
 *
 * `offsetX`/`offsetY` are the pointer's offset *within the dragged item*, in
 * cells, so that a 2x3 sword picked up by its tip drops where its tip is.
 */
export function cellFromPoint(
  point: { readonly x: number; readonly y: number },
  rect: { readonly left: number; readonly top: number },
  cellSize: number,
  offsetCells: { readonly x: number; readonly y: number } = { x: 0, y: 0 },
): { x: number; y: number } {
  const size = Math.max(1, cellSize);
  return {
    x: Math.floor((point.x - rect.left) / size) - offsetCells.x,
    y: Math.floor((point.y - rect.top) / size) - offsetCells.y,
  };
}

/**
 * Which cell of an item the pointer grabbed, in item-local cells.
 *
 * Clamped into the item so that a grab on the item's very edge — a pixel
 * outside its box after rounding — does not produce a negative offset that
 * shifts every subsequent drop.
 */
export function grabOffset(
  point: { readonly x: number; readonly y: number },
  itemRect: { readonly left: number; readonly top: number },
  cellSize: number,
  item: { readonly width: number; readonly height: number },
): { x: number; y: number } {
  const size = Math.max(1, cellSize);
  return {
    x: Math.max(0, Math.min(item.width - 1, Math.floor((point.x - itemRect.left) / size))),
    y: Math.max(0, Math.min(item.height - 1, Math.floor((point.y - itemRect.top) / size))),
  };
}
