/**
 * @module world/Portal
 *
 * Interactable transition volumes: the thing you walk into and press E on.
 *
 * A portal is deliberately split in two, because the two halves have different
 * owners and different lifetimes:
 *
 * - the **volume** is a Rapier sensor on {@link CollisionLayer.Trigger}, created
 *   by {@link ZoneManager} inside the same collider-tracking window as the rest
 *   of the zone's physics. That is not an implementation detail: a portal
 *   collider created later, from a `zone:loaded` handler, would fall outside the
 *   manager's before/after snapshot and would survive the zone that owns it —
 *   an invisible wall in the next zone, at the same coordinates, that nothing
 *   can explain.
 * - the **interaction** is {@link PortalSystem}, one long-lived module that
 *   watches the player, publishes a prompt and calls `travelTo`.
 *
 * ## Why the prompt uses a distance test and not the sensor
 *
 * The sensor exists so the volume is a real, queryable part of the world —
 * phase 5's quest system can ask what triggers a point is inside, and the camera
 * and AI layers correctly ignore it because `COLLISION_GROUPS.trigger` only
 * accepts characters. But Rapier's overlap queries in this project's wrapper
 * exclude sensors, and the prompt only ever needs to answer "is the player
 * within 2.2 m of this one point" for a handful of portals. That is a subtraction
 * per portal per frame. Routing it through the narrow phase would be more
 * machinery, more allocation and a wider blast radius for exactly the same
 * answer.
 */

import * as THREE from 'three/webgpu';

import { PlayerKey, type PlayerController } from '../character/PlayerController';
import type { GameContext, GameModule } from '../core/types';
import { COLLISION_GROUPS } from '../physics/Layers';
import { type ColliderRecord, type PhysicsWorld } from '../physics/PhysicsWorld';
import type { PortalSpec } from './Zone';

export type { PortalSpec } from './Zone';

declare module '../core/EventBus' {
  interface GameEvents {
    /** The active prompt changed. `text` is `null` when the prompt is hidden. */
    'portal:prompt': { portalId: string | null; text: string | null };
    /** The player triggered a portal. Travel begins on the same frame. */
    'portal:used': { portalId: string; targetZone: string; targetEntry: string };
  }
}

/**
 * The slice of {@link ZoneManager} the portal system uses, resolved by service
 * id rather than by import.
 *
 * `ZoneManager` imports {@link buildPortalColliders} from this module, so an
 * import back the other way would be a cycle — and a cycle whose failure mode is
 * a `ZoneManagerKey` that is `undefined` at evaluation time, which is a genuinely
 * awful thing to debug. Depending on the capability instead of the class is both
 * the cheaper fix and the more honest description of the dependency: the portal
 * system needs something that can travel, not a `ZoneManager` in particular.
 */
export interface PortalTravelHost {
  readonly active: { readonly portals: readonly PortalSpec[] } | null;
  readonly travelling: boolean;
  travelTo(zoneId: string, entryPointId: string | null): Promise<void>;
}

/** Service id of the travel host. Matches `ZoneManagerKey.id`. */
export const PORTAL_TRAVEL_SERVICE = 'world.zones';

/** Default horizontal radius of a portal volume, in metres. */
export const PORTAL_RADIUS = 2.2;
/** Default vertical extent above the volume's floor position, in metres. */
export const PORTAL_HEIGHT = 3;

/** The line shown when the player is standing in a portal. */
export function portalPromptText(spec: PortalSpec): string {
  return `Press E to ${spec.verb ?? 'enter'} ${spec.label}`;
}

/**
 * Create the sensor colliders for a zone's portals.
 *
 * Called by {@link ZoneManager} between `Zone.buildColliders` and
 * `PhysicsWorld.syncQueries`, so the returned records are inside the manager's
 * ownership window and are removed with the zone.
 */
export function buildPortalColliders(
  physics: PhysicsWorld,
  portals: readonly PortalSpec[],
): ColliderRecord[] {
  const out: ColliderRecord[] = [];
  for (const spec of portals) {
    const radius = spec.radius ?? PORTAL_RADIUS;
    const height = spec.height ?? PORTAL_HEIGHT;
    const desc = physics.rapier.ColliderDesc.cylinder(height / 2, radius)
      .setTranslation(spec.position.x, spec.position.y + height / 2, spec.position.z)
      .setSensor(true)
      .setCollisionGroups(COLLISION_GROUPS.trigger);
    out.push(
      physics.addCollider(desc, { kind: 'trigger', label: `portal.${spec.id}` }),
    );
  }
  return out;
}

/** Is a world position inside a portal's volume? */
export function isInsidePortal(
  spec: PortalSpec,
  x: number,
  y: number,
  z: number,
): boolean {
  const radius = spec.radius ?? PORTAL_RADIUS;
  const height = spec.height ?? PORTAL_HEIGHT;
  const dx = x - spec.position.x;
  const dz = z - spec.position.z;
  if (dx * dx + dz * dz > radius * radius) return false;
  // Generous downward slack: the reported player position is at his feet, but a
  // portal authored on sloping ground can sit a little above or below them.
  return y >= spec.position.y - 1.2 && y <= spec.position.y + height;
}

/**
 * Pick the portal the player is standing in, nearest first.
 *
 * Overlapping portals are a content bug, but resolving them by distance means
 * the bug produces "the wrong door sometimes" rather than "a door that depends
 * on array order", which is the difference between a reproducible report and an
 * unreproducible one.
 */
export function portalAt(
  portals: readonly PortalSpec[],
  x: number,
  y: number,
  z: number,
): PortalSpec | null {
  let best: PortalSpec | null = null;
  let bestDistance = Infinity;
  for (const spec of portals) {
    if (!isInsidePortal(spec, x, y, z)) continue;
    const d = (x - spec.position.x) ** 2 + (z - spec.position.z) ** 2;
    if (d < bestDistance) {
      bestDistance = d;
      best = spec;
    }
  }
  return best;
}

const PROMPT_STYLE =
  'position:fixed;left:50%;bottom:16%;transform:translateX(-50%);z-index:40;' +
  'pointer-events:none;padding:9px 18px;border-radius:4px;' +
  'border:1px solid rgba(214,184,132,0.30);background:rgba(8,7,6,0.72);' +
  'color:#e8d7b4;font:500 15px/1.3 ui-serif,Georgia,serif;letter-spacing:0.04em;' +
  'text-shadow:0 1px 3px rgba(0,0,0,0.9);opacity:0;transition:opacity 140ms ease-out;' +
  'white-space:nowrap;';

export interface PortalSystemOptions {
  /** Build the DOM prompt. Harnesses that only assert travel can turn it off. */
  readonly prompt?: boolean;
}

/**
 * Watches the player against the active zone's portals.
 *
 * Registered once, for the whole session: it reads the portal list off
 * {@link ZoneManager} every frame rather than caching it, so a zone change needs
 * no subscription and cannot leave a stale portal armed.
 */
export class PortalSystem implements GameModule {
  readonly name = 'world.portals';

  readonly #options: Required<PortalSystemOptions>;
  readonly #position = new THREE.Vector3();

  #ctx: GameContext | null = null;
  #element: HTMLElement | null = null;
  #current: PortalSpec | null = null;

  constructor(options: PortalSystemOptions = {}) {
    this.#options = { prompt: options.prompt ?? true };
  }

  /** The portal the player is standing in, or `null`. */
  get current(): PortalSpec | null {
    return this.#current;
  }

  /** The prompt line currently displayed, or `null`. */
  get promptText(): string | null {
    return this.#current === null ? null : portalPromptText(this.#current);
  }

  init(ctx: GameContext): void {
    this.#ctx = ctx;
    if (!this.#options.prompt || typeof document === 'undefined') return;
    const element = document.createElement('div');
    element.setAttribute('style', PROMPT_STYLE);
    element.dataset['d2rim'] = 'portal-prompt';
    document.body.appendChild(element);
    this.#element = element;
  }

  update(ctx: GameContext): void {
    const zones = ctx.services.tryGet<PortalTravelHost>(PORTAL_TRAVEL_SERVICE);
    const player = ctx.services.tryGet<PlayerController>(PlayerKey);
    const zone = zones?.active ?? null;

    if (zones === undefined || player === undefined || zone === null || zones.travelling) {
      this.#setPrompt(null);
      return;
    }

    this.#position.copy(player.position);
    const found = portalAt(zone.portals, this.#position.x, this.#position.y, this.#position.z);
    this.#setPrompt(found);

    if (found === null || !ctx.input.wasPressed('Interact')) return;
    ctx.events.emit('portal:used', {
      portalId: found.id,
      targetZone: found.targetZone,
      targetEntry: found.targetEntry,
    });
    // Hide the prompt before the fade starts, or it hangs over the black screen
    // for the whole transition advertising a door that is already behind you.
    this.#setPrompt(null);
    void zones.travelTo(found.targetZone, found.targetEntry);
  }

  dispose(): void {
    this.#element?.remove();
    this.#element = null;
    this.#current = null;
    this.#ctx = null;
  }

  /**
   * Trigger a portal directly, bypassing the input check.
   *
   * The drive harness uses this: pressing a key through the browser depends on
   * pointer lock, focus and the engine happening to be stepping, none of which a
   * headless assertion about zone transitions should be entangled with.
   */
  async use(spec: PortalSpec): Promise<void> {
    const zones = this.#ctx?.services.tryGet<PortalTravelHost>(PORTAL_TRAVEL_SERVICE);
    if (zones === undefined) return;
    this.#setPrompt(null);
    await zones.travelTo(spec.targetZone, spec.targetEntry);
  }

  #setPrompt(spec: PortalSpec | null): void {
    if (spec === this.#current) return;
    this.#current = spec;
    const text = spec === null ? null : portalPromptText(spec);
    if (this.#element !== null) {
      if (text !== null) this.#element.textContent = text;
      this.#element.style.opacity = text === null ? '0' : '1';
    }
    this.#ctx?.events.emit('portal:prompt', { portalId: spec?.id ?? null, text });
  }
}
