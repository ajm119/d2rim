/**
 * The loading screen's arithmetic and vocabulary.
 *
 * The DOM half is verified in a real browser by the headless drive harness —
 * there is no jsdom in this project and adding one to assert that `appendChild`
 * works would not be worth the dependency. What is worth testing here is the
 * part that is easy to get subtly wrong and impossible to eyeball: the phase
 * weighting, which decides whether the bar spends its travel where the player
 * spends their wait, and the module-name vocabulary, which decides whether the
 * screen says "loading textures" or leaks `render.materials` at someone.
 */

import { describe, expect, it } from 'vitest';

import type { BootPhase } from '../src/core/EventBus';
import {
  describeModule,
  phaseFloor,
  PHASE_ORDER,
  PHASE_WEIGHTS,
} from '../src/ui/LoadingScreen';

describe('phase weights', () => {
  it('sum to exactly one, so the bar reaches 100%', () => {
    const total = PHASE_ORDER.reduce((sum, phase) => sum + PHASE_WEIGHTS[phase], 0);
    expect(total).toBeCloseTo(1, 10);
  });

  it('give module init the largest share, because that is the longest wait', () => {
    // Asset download and shader compilation both happen inside module init. If
    // any other phase ever outweighs it, the bar is lying about where the time
    // goes.
    const largest = [...PHASE_ORDER].sort((a, b) => PHASE_WEIGHTS[b] - PHASE_WEIGHTS[a])[0];
    expect(largest).toBe('modules');
  });

  it('start each phase where the previous one ended, with no gaps or overlaps', () => {
    let expected = 0;
    for (const phase of PHASE_ORDER) {
      expect(phaseFloor(phase)).toBeCloseTo(expected, 10);
      expected += PHASE_WEIGHTS[phase];
    }
    expect(expected).toBeCloseTo(1, 10);
  });

  it('never places a floor outside [0,1)', () => {
    for (const phase of PHASE_ORDER) {
      expect(phaseFloor(phase)).toBeGreaterThanOrEqual(0);
      expect(phaseFloor(phase)).toBeLessThan(1);
    }
  });

  it('covers every declared boot phase', () => {
    const declared: readonly BootPhase[] = ['renderer', 'modules', 'zone', 'ready'];
    expect([...PHASE_ORDER].sort()).toEqual([...declared].sort());
  });
});

describe('describeModule', () => {
  // These are the real `GameModule.name` values the engine emits, not
  // approximations of them. An earlier version of this table guessed at names
  // like `physics` and `CombatSystem`; the engine actually says `physics.world`
  // and `combat`, so every one of those guesses silently fell through to the
  // generic label. Asserting on the real strings is the only way this stays
  // honest.
  it.each([
    ['AssetManager', 'opening the asset registry'],
    ['render.settings', 'choosing quality settings'],
    ['render.materials', 'loading textures'],
    ['Sky', 'building the sky'],
    ['IBL', 'building the sky'],
    ['render.post', 'compiling shaders'],
    ['render.gtao', 'compiling shaders'],
    ['physics.world', 'starting the physics world'],
    ['world.zones', 'building terrain'],
    ['world.portals', 'opening the ways between zones'],
    ['scene.bloodMoor', 'building terrain'],
    ['combat', 'waking the Barbarian'],
    ['character.player', 'waking the Barbarian'],
    ['ai.director', 'waking the Barbarian'],
    ['rpg', 'preparing the act'],
    ['rpg.loot', 'preparing the act'],
    ['quest.denOfEvil', 'preparing the act'],
    ['ui.inventory', 'drawing the interface'],
    ['DebugOverlay', 'drawing the interface'],
  ])('describes %s as "%s"', (name, expected) => {
    expect(describeModule(name)).toBe(expected);
  });

  it('has a mapping for every module the app actually registers', () => {
    // The generic fallback exists so an unmapped name is never *shown* raw, but
    // relying on it for a module that ships today means the loading screen says
    // "preparing the world" for several seconds while something specific and
    // nameable is happening.
    const registered = [
      'AssetManager', 'render.settings', 'render.materials', 'render.post',
      'render.gtao', 'render.ssr', 'render.volumetrics', 'render.lightShafts',
      'render.bridges', 'Sky', 'Lighting', 'IBL', 'TimeOfDay',
      'physics.world', 'world.zones', 'world.portals',
      'combat', 'combat.feedback', 'character.player', 'character.footIK',
      'character.cameraRig', 'ai.director',
      'rpg', 'rpg.loot', 'quest.denOfEvil', 'quest.npcs',
      'ui.manager', 'ui.combatHud', 'ui.rpgHud', 'ui.inventory', 'ui.skills',
      'ui.vendor', 'ui.dialogue', 'ui.menu', 'DebugOverlay',
    ];
    const unmapped = registered.filter((name) => describeModule(name) === 'preparing the world');
    expect(unmapped).toEqual([]);
  });

  it('falls back to a generic phrase rather than leaking an internal name', () => {
    // The failure mode this prevents: a module added later shows up on the
    // loading screen as `SomeNewInternalThing`, which tells the player nothing
    // and looks like a crash.
    const label = describeModule('SomeModuleNobodyMapped');
    expect(label).toBe('preparing the world');
    expect(label).not.toContain('SomeModuleNobodyMapped');
  });

  it('never returns an empty label', () => {
    for (const name of ['', 'x', 'render.', 'UI', '???']) {
      expect(describeModule(name).length).toBeGreaterThan(0);
    }
  });
});
