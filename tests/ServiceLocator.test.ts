import { describe, expect, it } from 'vitest';

import { ServiceLocator, serviceKey } from '../src/core/ServiceLocator';

interface Physics {
  gravity: number;
}

const PhysicsKey = serviceKey<Physics>('physics');
const AudioKey = serviceKey<{ volume: number }>('audio');

describe('ServiceLocator', () => {
  it('registers and retrieves by typed key', () => {
    const services = new ServiceLocator();
    const physics: Physics = { gravity: -9.81 };

    services.register(PhysicsKey, physics);

    expect(services.get(PhysicsKey)).toBe(physics);
    expect(services.get(PhysicsKey).gravity).toBe(-9.81);
  });

  it('registers and retrieves by plain string key', () => {
    const services = new ServiceLocator();
    services.register<number>('frameBudget', 16);
    expect(services.get<number>('frameBudget')).toBe(16);
  });

  it('treats a typed key and its string id as the same slot', () => {
    const services = new ServiceLocator();
    const physics: Physics = { gravity: -1 };

    services.register(PhysicsKey, physics);

    expect(services.has('physics')).toBe(true);
    expect(services.get<Physics>('physics')).toBe(physics);
  });

  it('returns the instance from register() so it can be chained', () => {
    const services = new ServiceLocator();
    const physics = services.register(PhysicsKey, { gravity: -9.81 });
    expect(physics.gravity).toBe(-9.81);
  });

  it('throws on duplicate registration rather than silently replacing', () => {
    const services = new ServiceLocator();
    services.register(PhysicsKey, { gravity: -9.81 });

    expect(() => services.register(PhysicsKey, { gravity: 0 })).toThrow(/already registered/);
    expect(services.get(PhysicsKey).gravity).toBe(-9.81);
  });

  it('throws a diagnostic error when a service is missing', () => {
    const services = new ServiceLocator();
    services.register(AudioKey, { volume: 1 });

    expect(() => services.get(PhysicsKey)).toThrow(/"physics" is not registered/);
    // The message lists what *is* registered, which is what makes ordering
    // bugs debuggable.
    expect(() => services.get(PhysicsKey)).toThrow(/audio/);
  });

  it('reports "none" when nothing is registered at all', () => {
    const services = new ServiceLocator();
    expect(() => services.get(PhysicsKey)).toThrow(/none/);
  });

  it('tryGet returns undefined instead of throwing', () => {
    const services = new ServiceLocator();
    expect(services.tryGet(PhysicsKey)).toBeUndefined();

    const physics: Physics = { gravity: -9.81 };
    services.register(PhysicsKey, physics);
    expect(services.tryGet(PhysicsKey)).toBe(physics);
  });

  it('has() reflects registration state', () => {
    const services = new ServiceLocator();
    expect(services.has(PhysicsKey)).toBe(false);
    services.register(PhysicsKey, { gravity: -9.81 });
    expect(services.has(PhysicsKey)).toBe(true);
  });

  it('unregister removes a service and reports whether it did', () => {
    const services = new ServiceLocator();
    services.register(PhysicsKey, { gravity: -9.81 });

    expect(services.unregister(PhysicsKey)).toBe(true);
    expect(services.unregister(PhysicsKey)).toBe(false);
    expect(services.has(PhysicsKey)).toBe(false);
  });

  it('allows re-registration after unregister (intentional swap)', () => {
    const services = new ServiceLocator();
    services.register(PhysicsKey, { gravity: -9.81 });
    services.unregister(PhysicsKey);

    expect(() => services.register(PhysicsKey, { gravity: -3.72 })).not.toThrow();
    expect(services.get(PhysicsKey).gravity).toBe(-3.72);
  });

  it('lists keys in registration order and clears them', () => {
    const services = new ServiceLocator();
    services.register(PhysicsKey, { gravity: -9.81 });
    services.register(AudioKey, { volume: 0.8 });

    expect(services.keys()).toEqual(['physics', 'audio']);

    services.clear();
    expect(services.keys()).toEqual([]);
    expect(services.has(PhysicsKey)).toBe(false);
  });

  it('stores falsy values without confusing them for absence', () => {
    const services = new ServiceLocator();
    services.register<number>('zero', 0);

    expect(services.has('zero')).toBe(true);
    expect(services.get<number>('zero')).toBe(0);
  });
});
