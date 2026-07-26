/**
 * @module core/ServiceLocator
 *
 * A tiny registry that lets modules find each other at runtime without
 * importing each other at compile time. This is what keeps the module graph
 * acyclic: `CombatModule` needs the player's transform, but importing
 * `PlayerModule` (which imports combat for damage events) would be a cycle.
 * Instead both sides agree on a {@link ServiceKey}.
 *
 * Keys are branded with their value type, so lookups are type-safe without a
 * cast at the call site:
 *
 * ```ts
 * export const PhysicsKey = serviceKey<PhysicsWorld>('physics');
 * services.register(PhysicsKey, world);
 * const world = services.get(PhysicsKey); // inferred as PhysicsWorld
 * ```
 */

declare const serviceBrand: unique symbol;

/**
 * An opaque, type-carrying service identifier.
 *
 * The phantom `[serviceBrand]` property exists only in the type system; it is
 * never present at runtime, which is why it is declared optional and readonly.
 */
export interface ServiceKey<T> {
  readonly id: string;
  readonly [serviceBrand]?: T;
}

/** Mint a typed service key. Call once per service, at module scope. */
export function serviceKey<T>(id: string): ServiceKey<T> {
  return { id };
}

/** Anything accepted as a lookup key: a typed key or a bare string. */
export type ServiceRef<T> = ServiceKey<T> | string;

function idOf(ref: ServiceRef<unknown>): string {
  return typeof ref === 'string' ? ref : ref.id;
}

export class ServiceLocator {
  readonly #services = new Map<string, unknown>();

  /**
   * Register `instance` under `key`.
   *
   * Re-registering an existing key throws rather than silently replacing:
   * duplicate registration is always a bug (a module initialised twice, or two
   * modules colliding on a name) and it is far cheaper to catch it at boot than
   * to debug a stale reference later. Use {@link unregister} first for an
   * intentional swap.
   *
   * @returns the instance, so registration can be chained onto construction.
   */
  register<T>(key: ServiceRef<T>, instance: T): T {
    const id = idOf(key);
    if (this.#services.has(id)) {
      throw new Error(`[ServiceLocator] service "${id}" is already registered`);
    }
    this.#services.set(id, instance);
    return instance;
  }

  /**
   * Look up a service, throwing if it is absent.
   *
   * Failing loudly is intentional: a missing service means a module ordering
   * bug, and returning `undefined` would push the failure to some unrelated
   * property access much later.
   */
  get<T>(key: ServiceRef<T>): T {
    const id = idOf(key);
    if (!this.#services.has(id)) {
      throw new Error(
        `[ServiceLocator] service "${id}" is not registered ` +
          `(registered: ${this.keys().join(', ') || 'none'})`,
      );
    }
    return this.#services.get(id) as T;
  }

  /** Look up a service, returning `undefined` when it is genuinely optional. */
  tryGet<T>(key: ServiceRef<T>): T | undefined {
    return this.#services.get(idOf(key)) as T | undefined;
  }

  /** Whether a service is currently registered. */
  has(key: ServiceRef<unknown>): boolean {
    return this.#services.has(idOf(key));
  }

  /** Remove a registration. @returns whether anything was removed. */
  unregister(key: ServiceRef<unknown>): boolean {
    return this.#services.delete(idOf(key));
  }

  /** All registered ids, in registration order. Primarily for diagnostics. */
  keys(): string[] {
    return Array.from(this.#services.keys());
  }

  /** Drop every registration. Used on engine teardown. */
  clear(): void {
    this.#services.clear();
  }
}
