import { describe, expect, it } from 'vitest';

import {
  ALL_LAYERS,
  COLLISION_GROUPS,
  CollisionLayer,
  filterOf,
  groupsInteract,
  interactionGroups,
  layerMask,
  membershipOf,
} from '../src/physics/Layers';

describe('interaction groups', () => {
  it('packs membership into the high half and filter into the low half', () => {
    const groups = interactionGroups(CollisionLayer.Player, ALL_LAYERS);
    expect(membershipOf(groups)).toBe(CollisionLayer.Player);
    expect(filterOf(groups)).toBe(ALL_LAYERS);
  });

  it('masks each half to 16 bits so one cannot corrupt the other', () => {
    const groups = interactionGroups(0xdeadbeef, 0xfeedface);
    expect(membershipOf(groups)).toBe(0xbeef);
    expect(filterOf(groups)).toBe(0xface);
    expect(groups).toBeGreaterThanOrEqual(0);
  });

  it('ORs layers into a mask', () => {
    expect(layerMask(CollisionLayer.Terrain, CollisionLayer.Prop)).toBe(0b11);
    expect(layerMask()).toBe(0);
  });

  it('requires both directions to agree', () => {
    const a = interactionGroups(CollisionLayer.Player, CollisionLayer.Terrain);
    const oneWay = interactionGroups(CollisionLayer.Terrain, CollisionLayer.Enemy);
    const bothWays = interactionGroups(CollisionLayer.Terrain, CollisionLayer.Player);
    expect(groupsInteract(a, oneWay)).toBe(false);
    expect(groupsInteract(a, bothWays)).toBe(true);
  });
});

describe('the game collision matrix', () => {
  it('lets the player stand on terrain and props', () => {
    expect(groupsInteract(COLLISION_GROUPS.player, COLLISION_GROUPS.terrain)).toBe(true);
    expect(groupsInteract(COLLISION_GROUPS.player, COLLISION_GROUPS.prop)).toBe(true);
  });

  it('lets enemies stand on the same world and collide with the player', () => {
    expect(groupsInteract(COLLISION_GROUPS.enemy, COLLISION_GROUPS.terrain)).toBe(true);
    expect(groupsInteract(COLLISION_GROUPS.enemy, COLLISION_GROUPS.player)).toBe(true);
  });

  it('keeps the camera arm off the player but on the world', () => {
    expect(groupsInteract(COLLISION_GROUPS.cameraProbe, COLLISION_GROUPS.player)).toBe(false);
    expect(groupsInteract(COLLISION_GROUPS.cameraProbe, COLLISION_GROUPS.terrain)).toBe(true);
    expect(groupsInteract(COLLISION_GROUPS.cameraProbe, COLLISION_GROUPS.prop)).toBe(true);
  });

  it('makes triggers see characters and nothing else', () => {
    expect(groupsInteract(COLLISION_GROUPS.trigger, COLLISION_GROUPS.player)).toBe(true);
    expect(groupsInteract(COLLISION_GROUPS.trigger, COLLISION_GROUPS.enemy)).toBe(true);
    expect(groupsInteract(COLLISION_GROUPS.trigger, COLLISION_GROUPS.terrain)).toBe(false);
    expect(groupsInteract(COLLISION_GROUPS.trigger, COLLISION_GROUPS.projectile)).toBe(false);
  });

  it('lets projectiles hit everything that can be hit', () => {
    for (const target of ['terrain', 'prop', 'player', 'enemy'] as const) {
      expect(groupsInteract(COLLISION_GROUPS.projectile, COLLISION_GROUPS[target])).toBe(true);
    }
  });
});
