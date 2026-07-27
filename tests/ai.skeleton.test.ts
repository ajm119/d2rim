import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';

import { BLOOD_MOOR_SPAWNS, normaliseHeight } from '../src/ai/EnemyDirector';
import {
  ARRIVAL_EASE,
  SKELETON_PROFILES,
  SKELETON_VARIANTS,
  STANDOFF_CLEARANCE,
  pursuitSpeed,
  standoffRadius,
} from '../src/ai/enemies/Skeleton';
import { CombatantRegistry, isFlashable, type Combatant } from '../src/combat/Combatant';
import { hitChance, resolveAttack, mulberry32 } from '../src/combat/DamageModel';
import { PLAYER_DEFENSE_BASE, PLAYER_OFFENSE } from '../src/combat/CombatSystem';

describe('skeleton profiles', () => {
  it('covers every declared variant', () => {
    for (const variant of SKELETON_VARIANTS) {
      expect(SKELETON_PROFILES[variant]).toBeDefined();
    }
  });

  it('points each variant at its own asset key', () => {
    const assets = SKELETON_VARIANTS.map((v) => SKELETON_PROFILES[v]?.asset);
    expect(new Set(assets).size).toBe(assets.length);
    for (const asset of assets) expect(asset).toMatch(/^enemy\.skeleton\./);
  });

  it('gives every variant at least one attack with a sane window', () => {
    for (const variant of SKELETON_VARIANTS) {
      const profile = SKELETON_PROFILES[variant];
      expect(profile?.attacks.length).toBeGreaterThan(0);
      for (const attack of profile?.attacks ?? []) {
        expect(attack.window[0]).toBeGreaterThan(0);
        expect(attack.window[0]).toBeLessThan(attack.window[1]);
        expect(attack.window[1]).toBeLessThanOrEqual(attack.recovery);
        expect(attack.telegraph).toBeGreaterThan(0.15);
      }
    }
  });

  it('makes the hardest-hitting attack the most heavily telegraphed', () => {
    const attacks = SKELETON_VARIANTS.flatMap((v) => SKELETON_PROFILES[v]?.attacks ?? []);
    const byId = new Map(attacks.map((a) => [a.id, a]));
    const heavy = byId.get('skeleton.heavy');
    const chop = byId.get('skeleton.chop');
    expect(heavy?.modifiers.damageScale ?? 0).toBeGreaterThan(chop?.modifiers.damageScale ?? 0);
    expect(heavy?.telegraph ?? 0).toBeGreaterThan(chop?.telegraph ?? 0);
  });

  it('makes every skeleton undead: poison immune, cold vulnerable', () => {
    for (const variant of SKELETON_VARIANTS) {
      const defense = SKELETON_PROFILES[variant]?.defense;
      expect(defense?.resistances?.poison).toBe(100);
      expect(defense?.maxResistances?.poison).toBe(100);
      expect(defense?.resistances?.cold ?? 0).toBeLessThan(0);
    }
  });

  it('gives the enemy a shorter reach than its attack range, so it must close', () => {
    for (const variant of SKELETON_VARIANTS) {
      const profile = SKELETON_PROFILES[variant];
      expect(profile?.reach).toBeLessThan(profile?.attackRange ?? 0);
    }
  });

  it('keeps chase faster than patrol so a pursuit reads as a pursuit', () => {
    for (const variant of SKELETON_VARIANTS) {
      const profile = SKELETON_PROFILES[variant];
      expect(profile?.chaseSpeed).toBeGreaterThan(profile?.walkSpeed ?? 0);
    }
  });

  it('hears further than it needs to see, but sees further than it hears', () => {
    for (const variant of SKELETON_VARIANTS) {
      const p = SKELETON_PROFILES[variant]?.perception;
      expect(p?.hearingRange).toBeGreaterThan(0);
      expect(p?.visionRange).toBeGreaterThan(p?.hearingRange ?? 0);
      expect(p?.loseRange).toBeGreaterThan(p?.visionRange ?? 0);
      expect(p?.visionHalfAngle).toBeLessThan(Math.PI);
    }
  });
});

describe('pursuitSpeed', () => {
  it('runs at full speed while the target is far away', () => {
    expect(pursuitSpeed(9, 1.48, 3.1)).toBeCloseTo(3.1);
  });

  it('stops once the stand-off ring is reached, so the enemy never enters the player', () => {
    expect(pursuitSpeed(1.48, 1.48, 3.1)).toBe(0);
    expect(pursuitSpeed(1.0, 1.48, 3.1)).toBe(0);
    expect(pursuitSpeed(0, 1.48, 3.1)).toBe(0);
  });

  it('eases down over the last stretch instead of stopping dead on the boundary', () => {
    const far = pursuitSpeed(1.48 + ARRIVAL_EASE, 1.48, 3.1);
    const near = pursuitSpeed(1.48 + ARRIVAL_EASE * 0.5, 1.48, 3.1);
    expect(far).toBeGreaterThan(near);
    expect(near).toBeGreaterThan(0);
  });

  it('never crawls: the floor is a quarter of the chase speed', () => {
    expect(pursuitSpeed(1.53, 1.48, 3.1)).toBeCloseTo(3.1 * 0.25);
  });

  it('tolerates a degenerate ease without dividing by zero', () => {
    expect(Number.isFinite(pursuitSpeed(2, 1.48, 3.1, 0))).toBe(true);
  });
});

describe('standoffRadius', () => {
  const PLAYER_HIT_RADIUS = 0.4;

  it('leaves the two capsules clear of each other rather than overlapping', () => {
    for (const variant of SKELETON_VARIANTS) {
      const profile = SKELETON_PROFILES[variant];
      expect(profile).toBeDefined();
      if (profile === undefined) continue;
      const gap = standoffRadius(profile) - PLAYER_HIT_RADIUS - profile.capsuleRadius;
      expect(gap).toBeCloseTo(STANDOFF_CLEARANCE);
      expect(gap).toBeGreaterThanOrEqual(-1e-9);
      // The *physical* capsules are smaller than the hit capsules, so touching
      // hit capsules still leaves daylight between the two bodies.
      const PLAYER_BODY_RADIUS = 0.304;
      expect(standoffRadius(profile) - PLAYER_BODY_RADIUS - profile.capsuleRadius).toBeGreaterThan(0);
    }
  });

  it('stops the pursuit before the enemy can enter the player', () => {
    for (const variant of SKELETON_VARIANTS) {
      const profile = SKELETON_PROFILES[variant];
      if (profile === undefined) continue;
      const standoff = standoffRadius(profile);
      expect(pursuitSpeed(standoff, standoff, profile.chaseSpeed)).toBe(0);
      expect(pursuitSpeed(standoff - 0.5, standoff, profile.chaseSpeed)).toBe(0);
    }
  });

  it('parks the enemy well inside its own attack range, so it keeps swinging', () => {
    for (const variant of SKELETON_VARIANTS) {
      const profile = SKELETON_PROFILES[variant];
      if (profile === undefined) continue;
      expect(standoffRadius(profile)).toBeLessThan(profile.attackRange);
    }
  });

  it('stops no further out than the swing can actually reach', () => {
    // Measured against the real rig: during the authored window the nearest
    // point of a skeleton's swinging arm sits 0.25 m outside the player's hit
    // capsule at a 1.10 m separation, and lengthening the blade does not move
    // it — the closest point is the elbow. So contact needs a stand-off of
    // about 0.85 m or less. A regression that widens this disarms the enemy,
    // which is invisible in a screenshot and fatal to the fight.
    const CONTACT_LIMIT = 1.1 - 0.25;
    for (const variant of SKELETON_VARIANTS) {
      const profile = SKELETON_PROFILES[variant];
      if (profile === undefined) continue;
      expect(standoffRadius(profile)).toBeLessThanOrEqual(CONTACT_LIMIT);
    }
  });
});

describe('the Blood Moor encounter', () => {
  it('names only variants that exist', () => {
    for (const spawn of BLOOD_MOOR_SPAWNS) {
      expect(SKELETON_PROFILES[spawn.variant]).toBeDefined();
    }
  });

  it('is a handful, and none of them starts on top of the player', () => {
    expect(BLOOD_MOOR_SPAWNS.length).toBeGreaterThanOrEqual(4);
    expect(BLOOD_MOOR_SPAWNS.length).toBeLessThanOrEqual(10);
    for (const spawn of BLOOD_MOOR_SPAWNS) {
      // The player spawns near (1.1, 0.4); nothing should be inside its
      // perception radius at boot or the game opens mid-fight.
      expect(Math.hypot(spawn.x - 1.1, spawn.z - 0.4)).toBeGreaterThan(6);
    }
  });

  it('does not stack two spawns on the same spot', () => {
    for (let i = 0; i < BLOOD_MOOR_SPAWNS.length; i++) {
      for (let j = i + 1; j < BLOOD_MOOR_SPAWNS.length; j++) {
        const a = BLOOD_MOOR_SPAWNS[i];
        const b = BLOOD_MOOR_SPAWNS[j];
        if (a === undefined || b === undefined) continue;
        expect(Math.hypot(a.x - b.x, a.z - b.z)).toBeGreaterThan(1.5);
      }
    }
  });
});

describe('normaliseHeight', () => {
  it('scales a skinned model to the requested height', () => {
    const root = new THREE.Object3D();
    const geometry = new THREE.BoxGeometry(0.5, 2, 0.5);
    const skinned = new THREE.SkinnedMesh(geometry, new THREE.MeshStandardMaterial());
    root.add(skinned);
    const scale = normaliseHeight(root, 1.5);
    expect(scale).toBeCloseTo(0.75, 6);
    expect(root.scale.y).toBeCloseTo(0.75, 6);
  });

  it('ignores loose props that are not skinned, which is the whole point', () => {
    const root = new THREE.Object3D();
    root.add(new THREE.SkinnedMesh(new THREE.BoxGeometry(0.5, 2, 0.5), new THREE.MeshStandardMaterial()));
    const prop = new THREE.Mesh(new THREE.BoxGeometry(0.2, 8, 0.2));
    prop.position.y = 5;
    root.add(prop);
    expect(normaliseHeight(root, 2)).toBeCloseTo(1, 6);
  });

  it('leaves a model with no skinned meshes alone', () => {
    const root = new THREE.Object3D();
    root.add(new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1)));
    expect(normaliseHeight(root, 5)).toBe(1);
    expect(root.scale.y).toBe(1);
  });
});

/* -------------------------------------------------------------------------- */
/* An encounter, without a renderer                                            */
/* -------------------------------------------------------------------------- */

class Dummy implements Combatant {
  health: number;
  hits = 0;
  flashed = 0;

  constructor(
    readonly id: number,
    readonly faction: 'player' | 'enemy',
    readonly label: string,
    private readonly stats: { defense: number; level: number; maxHealth: number },
  ) {
    this.health = stats.maxHealth;
  }

  readonly object = null;

  get alive(): boolean {
    return this.health > 0;
  }

  get hitRadius(): number {
    return 0.4;
  }

  get hitHeight(): number {
    return 1.8;
  }

  get defense() {
    return {
      level: this.stats.level,
      defense: this.stats.defense,
      maxHealth: this.stats.maxHealth,
      poise: 0,
    };
  }

  footPosition(out: THREE.Vector3): THREE.Vector3 {
    return out.set(0, 0, 0);
  }

  facing(out: THREE.Vector3): THREE.Vector3 {
    return out.set(0, 0, 1);
  }

  flash(): void {
    this.flashed++;
  }

  receiveHit(): never {
    throw new Error('unused');
  }
}

describe('a scripted duel, resolved through the shared model', () => {
  const rng = mulberry32(20250726);

  it('kills a minion in a plausible number of swings', () => {
    const profile = SKELETON_PROFILES.minion;
    expect(profile).toBeDefined();
    if (profile === undefined) return;

    let health = profile.maxHealth;
    let swings = 0;
    const defense = { ...profile.defense, maxHealth: profile.maxHealth };
    while (health > 0 && swings < 200) {
      const outcome = resolveAttack(PLAYER_OFFENSE, defense, rng);
      health -= outcome.total;
      swings++;
    }
    expect(health).toBeLessThanOrEqual(0);
    // Long enough to be a fight, short enough not to be a chore.
    expect(swings).toBeGreaterThanOrEqual(3);
    expect(swings).toBeLessThanOrEqual(12);
  });

  it('gives the player a reliable enough swing that a miss reads as spacing', () => {
    const profile = SKELETON_PROFILES.minion;
    const chance = hitChance(
      PLAYER_OFFENSE.attackRating,
      profile?.defense.defense ?? 0,
      PLAYER_OFFENSE.level,
      profile?.defense.level ?? 1,
    );
    expect(chance).toBeGreaterThan(0.75);
  });

  it('leaves the player able to survive a skeleton for a while', () => {
    const profile = SKELETON_PROFILES.warrior;
    expect(profile).toBeDefined();
    if (profile === undefined) return;

    let health = PLAYER_DEFENSE_BASE.maxHealth;
    let swings = 0;
    while (health > 0 && swings < 400) {
      health -= resolveAttack(profile.offense, { ...PLAYER_DEFENSE_BASE }, rng).total;
      swings++;
    }
    expect(swings).toBeGreaterThan(12);
  });

  it('makes cold noticeably better than fire against the undead', () => {
    const profile = SKELETON_PROFILES.minion;
    expect(profile).toBeDefined();
    if (profile === undefined) return;
    const defense = { ...profile.defense, maxHealth: profile.maxHealth };
    const flat = { min: 20, max: 20 };
    const always = { level: 5, attackRating: 0, criticalChance: 0, alwaysHits: true };

    const cold = resolveAttack({ ...always, damage: { cold: flat } }, defense, mulberry32(1));
    const fire = resolveAttack({ ...always, damage: { fire: flat } }, defense, mulberry32(1));
    expect(cold.total).toBeGreaterThan(fire.total);
  });
});

describe('CombatantRegistry', () => {
  const registry = new CombatantRegistry();
  const hero = new Dummy(registry.nextId(), 'player', 'hero', {
    defense: 60,
    level: 2,
    maxHealth: 120,
  });
  const skeleton = new Dummy(registry.nextId(), 'enemy', 'skeleton', {
    defense: 34,
    level: 2,
    maxHealth: 46,
  });

  it('mints unique ids', () => {
    expect(hero.id).not.toBe(skeleton.id);
  });

  it('only returns live combatants of the other faction', () => {
    registry.add(hero);
    registry.add(skeleton);
    expect(registry.hostileTo('player')).toEqual([skeleton]);
    expect(registry.hostileTo('enemy')).toEqual([hero]);
    skeleton.health = 0;
    expect(registry.hostileTo('player')).toEqual([]);
  });

  it('never adds the same combatant twice', () => {
    registry.add(hero);
    expect(registry.byFaction('player').length).toBe(1);
  });

  it('finds the player without knowing its id', () => {
    expect(registry.first('player')).toBe(hero);
  });

  it('removes cleanly and reports whether it did anything', () => {
    expect(registry.remove(skeleton)).toBe(true);
    expect(registry.remove(skeleton)).toBe(false);
    expect(registry.all.length).toBe(1);
  });

  it('detects which combatants can be flashed', () => {
    expect(isFlashable(hero)).toBe(true);
    expect(isFlashable({ ...hero, flash: undefined } as unknown as Combatant)).toBe(false);
  });
});
