import { describe, expect, it, vi } from 'vitest';

import { EventBus } from '../src/core/EventBus';

/**
 * The bus is the backbone of cross-module communication, so these tests focus
 * on the properties modules will actually rely on: exactly-once delivery,
 * disposer correctness, and safety when handlers mutate the subscriber set or
 * throw mid-emit.
 */
describe('EventBus', () => {
  it('delivers a payload to every subscriber', () => {
    const bus = new EventBus();
    const a = vi.fn();
    const b = vi.fn();

    bus.on('engine:frame', a);
    bus.on('engine:frame', b);
    bus.emit('engine:frame', { frame: 7, dt: 0.016 });

    expect(a).toHaveBeenCalledTimes(1);
    expect(a).toHaveBeenCalledWith({ frame: 7, dt: 0.016 });
    expect(b).toHaveBeenCalledTimes(1);
  });

  it('does nothing when an event has no subscribers', () => {
    const bus = new EventBus();
    expect(() => bus.emit('engine:pause', { paused: true })).not.toThrow();
    expect(bus.listenerCount('engine:pause')).toBe(0);
  });

  it('registers a handler only once even if subscribed twice', () => {
    const bus = new EventBus();
    const handler = vi.fn();

    bus.on('engine:pause', handler);
    bus.on('engine:pause', handler);
    bus.emit('engine:pause', { paused: true });

    expect(bus.listenerCount('engine:pause')).toBe(1);
    expect(handler).toHaveBeenCalledTimes(1);
  });

  it('returns a disposer that unsubscribes, and is idempotent', () => {
    const bus = new EventBus();
    const handler = vi.fn();

    const dispose = bus.on('engine:pause', handler);
    dispose();
    dispose();
    bus.emit('engine:pause', { paused: false });

    expect(handler).not.toHaveBeenCalled();
    expect(bus.listenerCount('engine:pause')).toBe(0);
  });

  it('removes a specific handler with off()', () => {
    const bus = new EventBus();
    const kept = vi.fn();
    const dropped = vi.fn();

    bus.on('engine:frame', kept);
    bus.on('engine:frame', dropped);
    bus.off('engine:frame', dropped);
    bus.emit('engine:frame', { frame: 1, dt: 0 });

    expect(kept).toHaveBeenCalledTimes(1);
    expect(dropped).not.toHaveBeenCalled();
  });

  it('delivers once() exactly one time', () => {
    const bus = new EventBus();
    const handler = vi.fn();

    bus.once('engine:frame', handler);
    bus.emit('engine:frame', { frame: 1, dt: 0 });
    bus.emit('engine:frame', { frame: 2, dt: 0 });

    expect(handler).toHaveBeenCalledTimes(1);
    expect(handler).toHaveBeenCalledWith({ frame: 1, dt: 0 });
    expect(bus.listenerCount('engine:frame')).toBe(0);
  });

  it('keeps delivering to remaining handlers when one throws', () => {
    const bus = new EventBus();
    const spy = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const after = vi.fn();

    bus.on('engine:frame', () => {
      throw new Error('handler exploded');
    });
    bus.on('engine:frame', after);

    expect(() => bus.emit('engine:frame', { frame: 1, dt: 0 })).not.toThrow();
    expect(after).toHaveBeenCalledTimes(1);
    expect(spy).toHaveBeenCalled();

    spy.mockRestore();
  });

  it('does not invoke a handler added during the same emit', () => {
    const bus = new EventBus();
    const late = vi.fn();

    bus.on('engine:frame', () => {
      bus.on('engine:frame', late);
    });
    bus.emit('engine:frame', { frame: 1, dt: 0 });

    expect(late).not.toHaveBeenCalled();

    bus.emit('engine:frame', { frame: 2, dt: 0 });
    expect(late).toHaveBeenCalledTimes(1);
  });

  it('does not invoke a handler removed by an earlier handler in the same emit', () => {
    const bus = new EventBus();
    const second = vi.fn();

    bus.on('engine:frame', () => bus.off('engine:frame', second));
    bus.on('engine:frame', second);
    bus.emit('engine:frame', { frame: 1, dt: 0 });

    expect(second).not.toHaveBeenCalled();
  });

  it('supports nested emits without losing deliveries', () => {
    const bus = new EventBus();
    const order: string[] = [];

    bus.on('engine:frame', () => {
      order.push('frame');
      bus.emit('engine:pause', { paused: true });
      order.push('frame-after-nested');
    });
    bus.on('engine:pause', () => order.push('pause'));

    bus.emit('engine:frame', { frame: 1, dt: 0 });

    expect(order).toEqual(['frame', 'pause', 'frame-after-nested']);
  });

  it('clears one event or all events', () => {
    const bus = new EventBus();
    bus.on('engine:frame', vi.fn());
    bus.on('engine:pause', vi.fn());

    bus.clear('engine:frame');
    expect(bus.listenerCount('engine:frame')).toBe(0);
    expect(bus.listenerCount('engine:pause')).toBe(1);

    bus.clear();
    expect(bus.listenerCount('engine:pause')).toBe(0);
  });
});
