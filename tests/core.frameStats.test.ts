/**
 * The frame-time instrument.
 *
 * This exists because the previous one was saturated: the overlay averaged the
 * *clamped* frame delta, so any machine slower than the clamp reported exactly
 * `1 / maxDelta` fps forever — `4.0 fps (250.0 ms)` on a 6 fps laptop and on a
 * 1 fps one alike, identically across two renderer backends. Every test here
 * pins some part of "the number must survive the trip from the wall clock to
 * the overlay unmodified".
 */

import { describe, expect, it } from 'vitest';

import { FrameStats, type FrameSample } from '../src/core/FrameStats';

function sample(rawMs: number, extra: Partial<FrameSample> = {}): FrameSample {
  return {
    rawMs,
    updateMs: 0,
    renderMs: 0,
    gpuMs: 0,
    syncMs: 0,
    clamped: false,
    starved: false,
    frame: 0,
    ...extra,
  };
}

describe('FrameStats', () => {
  it('is empty, not zero-fps-with-confidence, before any frame', () => {
    const stats = new FrameStats(16);
    const snap = stats.snapshot();
    expect(snap.samples).toBe(0);
    expect(snap.fps).toBe(0);
    expect(snap.p50Ms).toBe(0);
  });

  it('reports the median frame time, not the mean', () => {
    const stats = new FrameStats(16);
    // Nine cheap frames and one catastrophic one. The mean is 36 ms (28 fps);
    // the median is 16 ms. A player on this machine sees 60 fps with a hitch,
    // and the median is the honest description of the typical frame.
    for (let i = 0; i < 9; i++) stats.record(sample(16));
    stats.record(sample(216));

    const snap = stats.snapshot();
    expect(snap.p50Ms).toBe(16);
    expect(snap.fps).toBeCloseTo(1000 / 16, 6);
  });

  it('reports p95 and the worst frame, so stutter is visible at all', () => {
    const stats = new FrameStats(128);
    for (let i = 0; i < 95; i++) stats.record(sample(16));
    for (let i = 0; i < 5; i++) stats.record(sample(120));

    const snap = stats.snapshot();
    expect(snap.samples).toBe(100);
    expect(snap.p50Ms).toBe(16);
    // Nearest-rank p95 over 100 samples is the 95th, which is still a fast one;
    // the point is that it sits at the boundary rather than being averaged away.
    expect(snap.p95Ms).toBeGreaterThanOrEqual(16);
    expect(snap.maxMs).toBe(120);
    expect(snap.fpsLow).toBeLessThanOrEqual(snap.fps);
  });

  it('records the raw time even when the frame was clamped', () => {
    // The whole point. A 250 ms frame that the accumulator clamped to 83 ms is
    // still a 250 ms frame, and this instrument must say so.
    const stats = new FrameStats(8);
    for (let i = 0; i < 8; i++) stats.record(sample(250, { clamped: true, starved: true }));

    const snap = stats.snapshot();
    expect(snap.p50Ms).toBe(250);
    expect(snap.fps).toBeCloseTo(4, 6);
    expect(snap.clampedFrames).toBe(8);
    expect(snap.starvedFrames).toBe(8);
  });

  it('keeps the clamp count separate from the frame time', () => {
    const stats = new FrameStats(8);
    stats.record(sample(16));
    stats.record(sample(400, { clamped: true }));
    stats.record(sample(16));
    stats.record(sample(16));

    const snap = stats.snapshot();
    expect(snap.clampedFrames).toBe(1);
    // The clamped frame contributes its real cost to the worst-case figure and
    // is not allowed to drag the median.
    expect(snap.maxMs).toBe(400);
    expect(snap.p50Ms).toBe(16);
  });

  it('splits update, render and GPU time', () => {
    const stats = new FrameStats(8);
    for (let i = 0; i < 4; i++) {
      stats.record(sample(50, { updateMs: 3, renderMs: 12, gpuMs: 44 }));
    }
    const snap = stats.snapshot();
    expect(snap.updateMs).toBe(3);
    expect(snap.renderMs).toBe(12);
    expect(snap.gpuMs).toBe(44);
    expect(snap.gpuAvailable).toBe(true);
  });

  it('reports GPU time as unavailable rather than as zero milliseconds', () => {
    const stats = new FrameStats(8);
    for (let i = 0; i < 4; i++) stats.record(sample(16));
    const snap = stats.snapshot();
    expect(snap.gpuAvailable).toBe(false);
    expect(snap.gpuMs).toBe(0);
  });

  it('is a ring: old frames leave the window', () => {
    const stats = new FrameStats(4);
    for (let i = 0; i < 4; i++) stats.record(sample(200));
    expect(stats.snapshot().p50Ms).toBe(200);

    // A scene change to something cheap must show up within one window, not be
    // averaged against a minute of boot.
    for (let i = 0; i < 4; i++) stats.record(sample(10));
    const snap = stats.snapshot();
    expect(snap.samples).toBe(4);
    expect(snap.p50Ms).toBe(10);
    expect(snap.maxMs).toBe(10);
  });

  it('survives a misbehaving clock without poisoning the percentiles', () => {
    const stats = new FrameStats(8);
    stats.record(sample(Number.NaN));
    stats.record(sample(Number.POSITIVE_INFINITY));
    stats.record(sample(-5));
    stats.record(sample(20));

    const snap = stats.snapshot();
    expect(Number.isFinite(snap.p50Ms)).toBe(true);
    expect(Number.isFinite(snap.fps)).toBe(true);
    expect(snap.maxMs).toBe(20);
  });

  it('reset() empties the window', () => {
    const stats = new FrameStats(8);
    for (let i = 0; i < 8; i++) stats.record(sample(16));
    stats.reset();
    expect(stats.size).toBe(0);
    expect(stats.snapshot().samples).toBe(0);
  });
});
