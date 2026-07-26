/**
 * The frame graph's ordering constraints, asserted rather than commented.
 *
 * `buildFrameGraph` returns modules in `Engine.add` order, and registration
 * order *is* frame order — a module whose `init` resolves a service registered
 * by a module further down the list silently degrades instead of failing. That
 * class of bug is invisible in the image (a scene with no god rays looks like a
 * scene with the fog turned down), so it is pinned here instead.
 *
 * These tests construct modules but never `init` them: initialisation needs a
 * live renderer, and the property under test is the ordering of the list, not
 * the behaviour of any module in it.
 */

import { describe, expect, it } from 'vitest';

import {
  FRAME_GRAPH_SERVICE_IDS,
  buildFrameGraph,
  handoffDistance,
} from '../src/render/FrameGraph';
import { RENDER_QUALITIES, RENDER_TIERS, RenderSettings } from '../src/render/RenderSettings';

function orderOf(names: readonly string[]): Map<string, number> {
  return new Map(names.map((name, index) => [name, index]));
}

describe('frame graph assembly', () => {
  const graph = buildFrameGraph({ quality: 'high' });
  const names = graph.modules.map((module) => module.name);
  const index = orderOf(names);

  it('registers every module exactly once', () => {
    expect(new Set(names).size).toBe(names.length);
  });

  it.each([
    // [earlier, later, why]
    ['render.settings', 'render.materials', 'materials read quality and weather at init'],
    ['render.post', 'render.gtao', 'GTAO binds the motion vectors PostStack registers'],
    ['render.post', 'render.ssr', 'SSR binds the motion vectors PostStack registers'],
    ['TimeOfDay', 'Sky', 'Sky would otherwise construct a second clock'],
    ['Sky', 'Lighting', 'Sky registers render.sky.sun'],
    ['render.gtao', 'IBL', 'GTAO registers render.ao, which IBL folds in'],
    ['IBL', 'render.ssr', 'SSR resolves render.ibl for its miss fallback'],
    ['Lighting', 'render.bridges', 'the bridge reads LightingService.sunShadows'],
    ['render.bridges', 'render.volumetrics', 'the bridge provides all three volumetric services'],
    ['render.volumetrics', 'render.lightShafts', 'the shaft pass composites the froxel buffer'],
  ])('initialises %s before %s (%s)', (earlier, later) => {
    const a = index.get(earlier);
    const b = index.get(later);
    expect(a, `${earlier} is not in the graph`).toBeDefined();
    expect(b, `${later} is not in the graph`).toBeDefined();
    expect(a as number).toBeLessThan(b as number);
  });

  it('audits every service id it can name', () => {
    // The audit table is the project's only inventory of cross-module
    // contracts. An id that is registered but not audited degrades silently.
    expect(FRAME_GRAPH_SERVICE_IDS).toContain('render.sky.sun');
    expect(FRAME_GRAPH_SERVICE_IDS).toContain('lighting.celestial');
    expect(FRAME_GRAPH_SERVICE_IDS).toContain('render.ao');
    expect(FRAME_GRAPH_SERVICE_IDS).toContain('render.gbuffer.surface');
    expect(new Set(FRAME_GRAPH_SERVICE_IDS).size).toBe(FRAME_GRAPH_SERVICE_IDS.length);
  });

  it('hands the near field to the froxel volume and the far field to the atmosphere', () => {
    // The two fog models must tile, not overlap. `volumeDistance` is the seam.
    expect(graph.volumetrics.params.volumeDistance).toBe(handoffDistance);
    // ...and the atmosphere's near-ground mist term must be off, or the first
    // `handoffDistance` metres are fogged by both models at once.
    expect(graph.timeOfDay.mood.mistDensity).toBe(0);
  });
});

describe('quality tiers', () => {
  it('is monotonic in cost across every subsystem that has a numeric knob', () => {
    const ladder = RENDER_QUALITIES.map((quality) => RENDER_TIERS[quality]);
    for (let i = 1; i < ladder.length; i++) {
      const lower = ladder[i - 1];
      const higher = ladder[i];
      expect(lower).toBeDefined();
      expect(higher).toBeDefined();
      if (lower === undefined || higher === undefined) continue;
      expect(higher.shadowCascades).toBeGreaterThanOrEqual(lower.shadowCascades);
      expect(higher.shadowMapSize).toBeGreaterThanOrEqual(lower.shadowMapSize);
      expect(higher.shadowDistance).toBeGreaterThanOrEqual(lower.shadowDistance);
      expect(higher.skyViewWidth).toBeGreaterThanOrEqual(lower.skyViewWidth);
      expect(higher.scatterDensity).toBeGreaterThanOrEqual(lower.scatterDensity);
    }
  });

  it('propagates one tier choice to every subsystem', () => {
    const low = buildFrameGraph({ quality: 'low' });
    expect(low.settings.tier.gtao).toBe('off');
    expect(low.settings.tier.ssr).toBe('off');
    // `PostStack.quality` only settles at `init` (it can be `'auto'`), so the
    // tier the graph *asked* for is the thing under test here.
    expect(low.settings.tier.post).toBe('low');
    expect(low.settings.materialQuality).toBe('low');
  });

  it('defaults weather to a wet moor rather than to dry', () => {
    // The exact value is an art-direction call and moves; what must not move
    // is that *something* registers a weather provider and that it is not
    // zero. Before `RenderSettings` existed nothing did, and every material in
    // the project silently used its authored-dry default.
    const settings = new RenderSettings({ quality: 'high' });
    expect(settings.wetness).toBeGreaterThan(0.25);
    expect(settings.wetness).toBeLessThanOrEqual(1);
    expect(settings.puddleLevel).toBeGreaterThan(0);
  });

  it('clamps wetness to the unit range', () => {
    const settings = new RenderSettings({ quality: 'high' });
    settings.setWetness(4);
    expect(settings.wetness).toBe(1);
    settings.setWetness(-1);
    expect(settings.wetness).toBe(0);
  });
});
