/**
 * @module combat/Feedback
 *
 * The game-feel layer: hit stop, camera trauma, impact sparks, hit flash and
 * damage numbers. This is the difference between a hit that lands and a hit
 * that feels like nothing, and none of it changes a single number in the damage
 * model — which is exactly why it is a separate module.
 *
 * ### Hit stop
 *
 * On impact the whole simulation is slowed to a crawl for 60–100 ms scaled by
 * how much of the target's health the blow removed. **Slowed, not frozen**: the
 * engine's time scale multiplies the fixed-step accumulator, and a scale of
 * exactly zero would make elapsed real time unrecoverable from the scaled delta
 * — which would then make the hit-stop timer itself unmeasurable. A scale of
 * 0.06 is a 16× slowdown; it reads as a freeze and stays arithmetically sound.
 *
 * ### Camera shake: trauma, not jitter
 *
 * Random per-frame offsets look like a dropped camera, not an impact, because
 * white noise has no temporal coherence — every frame is unrelated to the last.
 * The model here is the standard one (Jonathan Cooper / Squirrel Eiserloh):
 *
 * - a scalar `trauma` in `[0, 1]` that **accumulates** across hits,
 * - which **decays** linearly with time,
 * - driving offsets as `trauma²` (so small hits barely register and big ones
 *   dominate, and the tail-off is smooth rather than linear),
 * - sampled from **smooth value noise** on three independent seeds, so the
 *   camera traces a continuous path instead of teleporting.
 *
 * The offset is applied in `lateUpdate` *after* `CameraRig` has placed the
 * camera, and is never accumulated into the rig's own state — the rig recomputes
 * its pose from scratch every frame, so the shake is transient by construction.
 * That means this module must be registered with the engine after the rig.
 *
 * ### Everything else
 *
 * Sparks are a single pooled `THREE.Points` cloud, oriented to the hit normal
 * and biased along the swing direction. Transient impact lights go through the
 * `Lighting` service, never `scene.add(new PointLight())` — the clustered
 * lighting pass owns the budget. Damage numbers are DOM, positioned by
 * projecting the world point: crisp at any resolution, free to animate, and
 * legible in a way a 3D text mesh at this scale is not.
 */

import * as THREE from 'three/webgpu';

import type { GameContext, GameModule } from '../core/types';
import { LightingKey, type LightHandle, type LightingService } from '../render/Lighting';
import { isFlashable, type Combatant } from './Combatant';
import { CombatKey, type CombatSystem } from './CombatSystem';

/* -------------------------------------------------------------------------- */
/* Pure feel maths                                                            */
/* -------------------------------------------------------------------------- */

/** Hit stop floor and ceiling, seconds. Below 60 ms nobody feels it; above
 *  100 ms the game stutters instead of punching. */
export const HIT_STOP_MIN = 0.06;
export const HIT_STOP_MAX = 0.1;

/**
 * Hit-stop duration for one impact.
 *
 * `severity` is the fraction of the target's health removed; `scale` is the
 * move's own multiplier, so a finisher hangs longer than a jab at the same
 * damage.
 */
export function hitStopDuration(severity: number, scale = 1): number {
  const t = Math.min(1, Math.max(0, severity) * 3);
  return Math.min(HIT_STOP_MAX * 1.8, (HIT_STOP_MIN + (HIT_STOP_MAX - HIT_STOP_MIN) * t) * scale);
}

/**
 * Smooth 1D value noise in `[-1, 1]`.
 *
 * Integer lattice, hashed, smoothstep-interpolated. Not Perlin's gradient
 * noise, and deliberately so: for a scalar shake channel the two are visually
 * indistinguishable and this is a dozen lines with no gradient table, no
 * permutation array, and exact reproducibility from a seed — which matters,
 * because a camera that shakes differently on every replay cannot be captured
 * and compared.
 */
export function valueNoise(seed: number, t: number): number {
  const i = Math.floor(t);
  const f = t - i;
  const smooth = f * f * (3 - 2 * f);
  const a = hashToUnit(seed, i);
  const b = hashToUnit(seed, i + 1);
  return a + (b - a) * smooth;
}

function hashToUnit(seed: number, n: number): number {
  let h = (Math.imul(n, 0x27d4eb2d) ^ Math.imul(seed + 0x9e3779b9, 0x85ebca6b)) >>> 0;
  h ^= h >>> 15;
  h = Math.imul(h, 0x2545f491) >>> 0;
  h ^= h >>> 13;
  return ((h >>> 0) / 2147483648) - 1;
}

/** Trauma to displacement: squared, which is the whole trick. */
export function traumaToShake(trauma: number): number {
  const t = Math.min(1, Math.max(0, trauma));
  return t * t;
}

/** Linear decay, clamped at zero. */
export function decayTrauma(trauma: number, dt: number, perSecond: number): number {
  return Math.max(0, trauma - perSecond * Math.max(0, dt));
}

/** Trauma added by one impact. Saturating, so a flurry does not white out. */
export function traumaForHit(severity: number, critical: boolean): number {
  const base = 0.18 + Math.min(0.42, Math.max(0, severity) * 1.2);
  return critical ? Math.min(1, base * 1.35) : base;
}

/* -------------------------------------------------------------------------- */
/* Particles                                                                   */
/* -------------------------------------------------------------------------- */

const PARTICLE_CAPACITY = 640;

/**
 * A single pooled spark cloud for the whole game.
 *
 * One `Points` object, one draw call, a ring-buffer allocator and CPU
 * integration. At this particle count the CPU cost is noise, and the
 * alternative — a GPU system — would need compute, which the WebGL2 fallback
 * path does not have.
 */
class SparkCloud {
  readonly points: THREE.Points;

  readonly #position: Float32Array;
  readonly #colour: Float32Array;
  readonly #velocity: Float32Array;
  readonly #life: Float32Array;
  readonly #maxLife: Float32Array;
  readonly #geometry: THREE.BufferGeometry;
  readonly #material: THREE.PointsMaterial;
  #cursor = 0;
  #live = 0;

  constructor() {
    this.#position = new Float32Array(PARTICLE_CAPACITY * 3);
    this.#colour = new Float32Array(PARTICLE_CAPACITY * 3);
    this.#velocity = new Float32Array(PARTICLE_CAPACITY * 3);
    this.#life = new Float32Array(PARTICLE_CAPACITY);
    this.#maxLife = new Float32Array(PARTICLE_CAPACITY);

    this.#geometry = new THREE.BufferGeometry();
    this.#geometry.setAttribute('position', new THREE.BufferAttribute(this.#position, 3));
    this.#geometry.setAttribute('color', new THREE.BufferAttribute(this.#colour, 3));
    this.#geometry.setDrawRange(0, 0);
    // A generous fixed sphere: the cloud is rebuilt every frame and recomputing
    // real bounds would cost more than the culling saves.
    this.#geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 500);

    this.#material = new THREE.PointsMaterial({
      size: 0.045,
      sizeAttenuation: true,
      vertexColors: true,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    this.points = new THREE.Points(this.#geometry, this.#material);
    this.points.name = 'combat.sparks';
    this.points.frustumCulled = false;
    this.points.renderOrder = 12;
  }

  get live(): number {
    return this.#live;
  }

  /**
   * Emit a burst.
   *
   * Sparks are thrown along the reflection of the swing about the surface
   * normal, with a cone of scatter: that is what makes a burst read as coming
   * *off* the target rather than out of it.
   */
  burst(
    origin: THREE.Vector3,
    normal: THREE.Vector3,
    travel: THREE.Vector3,
    count: number,
    colour: THREE.Color,
    speed: number,
    rng: () => number,
  ): void {
    const spray = new THREE.Vector3().copy(travel).reflect(normal).normalize();
    if (!Number.isFinite(spray.x)) spray.copy(normal);
    spray.lerp(normal, 0.45).normalize();

    for (let i = 0; i < count; i++) {
      const index = this.#cursor;
      this.#cursor = (this.#cursor + 1) % PARTICLE_CAPACITY;
      const p = index * 3;

      this.#position[p] = origin.x;
      this.#position[p + 1] = origin.y;
      this.#position[p + 2] = origin.z;

      const scatter = 0.75;
      const vx = spray.x + (rng() - 0.5) * scatter;
      const vy = spray.y + (rng() - 0.5) * scatter + 0.35;
      const vz = spray.z + (rng() - 0.5) * scatter;
      const magnitude = speed * (0.45 + rng() * 0.8);
      this.#velocity[p] = vx * magnitude;
      this.#velocity[p + 1] = vy * magnitude;
      this.#velocity[p + 2] = vz * magnitude;

      const heat = 0.55 + rng() * 0.6;
      this.#colour[p] = colour.r * heat;
      this.#colour[p + 1] = colour.g * heat;
      this.#colour[p + 2] = colour.b * heat;

      const life = 0.28 + rng() * 0.42;
      this.#life[index] = life;
      this.#maxLife[index] = life;
    }
  }

  update(dt: number): void {
    if (dt <= 0) return;
    let live = 0;
    let highWater = 0;
    for (let i = 0; i < PARTICLE_CAPACITY; i++) {
      const remaining = this.#life[i] ?? 0;
      if (remaining <= 0) continue;
      const next = remaining - dt;
      const p = i * 3;
      if (next <= 0) {
        this.#life[i] = 0;
        // Park dead sparks at the origin of the cloud with zero colour so a
        // stale vertex can never be drawn as a bright dot at (0,0,0).
        this.#colour[p] = 0;
        this.#colour[p + 1] = 0;
        this.#colour[p + 2] = 0;
        continue;
      }
      this.#life[i] = next;
      this.#velocity[p + 1] = (this.#velocity[p + 1] ?? 0) - 14 * dt;
      const drag = Math.max(0, 1 - 3.2 * dt);
      this.#velocity[p] = (this.#velocity[p] ?? 0) * drag;
      this.#velocity[p + 1] = (this.#velocity[p + 1] ?? 0) * drag;
      this.#velocity[p + 2] = (this.#velocity[p + 2] ?? 0) * drag;
      this.#position[p] = (this.#position[p] ?? 0) + (this.#velocity[p] ?? 0) * dt;
      this.#position[p + 1] = (this.#position[p + 1] ?? 0) + (this.#velocity[p + 1] ?? 0) * dt;
      this.#position[p + 2] = (this.#position[p + 2] ?? 0) + (this.#velocity[p + 2] ?? 0) * dt;

      const fade = next / Math.max(1e-4, this.#maxLife[i] ?? 1);
      const scale = fade * fade;
      this.#colour[p] = (this.#colour[p] ?? 0) * (0.985 + 0.015 * scale);
      live++;
      highWater = i + 1;
    }
    this.#live = live;
    this.#geometry.setDrawRange(0, highWater);
    this.#geometry.attributes.position!.needsUpdate = true;
    this.#geometry.attributes.color!.needsUpdate = true;
    this.#material.opacity = live > 0 ? 1 : 0;
  }

  dispose(): void {
    this.points.removeFromParent();
    this.#geometry.dispose();
    this.#material.dispose();
  }
}

/* -------------------------------------------------------------------------- */
/* Damage numbers                                                              */
/* -------------------------------------------------------------------------- */

interface FloatingNumber {
  readonly element: HTMLDivElement;
  readonly origin: THREE.Vector3;
  life: number;
  readonly maxLife: number;
  readonly drift: THREE.Vector3;
}

/* -------------------------------------------------------------------------- */
/* Impact lights                                                               */
/* -------------------------------------------------------------------------- */

interface ImpactLight {
  readonly handle: LightHandle;
  readonly peak: number;
  life: number;
  readonly maxLife: number;
}

/* -------------------------------------------------------------------------- */
/* The module                                                                  */
/* -------------------------------------------------------------------------- */

export interface FeedbackOptions {
  /** Peak positional shake in metres at trauma 1. Default 0.22. */
  readonly shakeAmplitude?: number;
  /** Peak rotational shake in radians at trauma 1. Default 0.035. */
  readonly shakeRoll?: number;
  /** Shake frequency in Hz. Default 22. */
  readonly shakeFrequency?: number;
  /** Trauma lost per second. Default 1.9. */
  readonly traumaDecay?: number;
  /** Disable the DOM layer (damage numbers, damage vignette). */
  readonly dom?: boolean;
  /** Disable hit stop, for capture runs that need a steady clock. */
  readonly hitStop?: boolean;
}

type Mutable<T> = { -readonly [K in keyof T]: T[K] };

export class CombatFeedback implements GameModule {
  readonly name = 'combat.feedback';

  /** Mutable because {@link CombatFeedback.setHitStop} toggles one field. */
  readonly #options: Mutable<Required<FeedbackOptions>>;
  readonly #sparks = new SparkCloud();
  readonly #numbers: FloatingNumber[] = [];
  readonly #lights: ImpactLight[] = [];
  readonly #offset = new THREE.Vector3();
  readonly #projected = new THREE.Vector3();
  readonly #colour = new THREE.Color();

  #ctx: GameContext | null = null;
  #lighting: LightingService | null = null;
  #container: HTMLDivElement | null = null;
  #vignette: HTMLDivElement | null = null;
  #unsubscribe: Array<() => void> = [];

  #trauma = 0;
  #shakeClock = 0;
  #hitStopLeft = 0;
  #vignetteLevel = 0;
  #rngState = 0x9e3779b9;

  constructor(options: FeedbackOptions = {}) {
    this.#options = {
      shakeAmplitude: options.shakeAmplitude ?? 0.22,
      shakeRoll: options.shakeRoll ?? 0.035,
      shakeFrequency: options.shakeFrequency ?? 22,
      traumaDecay: options.traumaDecay ?? 1.9,
      dom: options.dom ?? true,
      hitStop: options.hitStop ?? true,
    };
  }

  /* -- introspection (drive harness and tests) ---------------------------- */

  get trauma(): number {
    return this.#trauma;
  }

  get liveSparks(): number {
    return this.#sparks.live;
  }

  get hitStopRemaining(): number {
    return this.#hitStopLeft;
  }

  /** Current shake offset, for a capture that wants to assert it is non-zero. */
  get shakeOffset(): THREE.Vector3 {
    return this.#offset;
  }

  /**
   * Turn hit stop on or off at runtime.
   *
   * Capture and drive harnesses need this: a 16× slowdown on every impact means
   * a scripted encounter costs sixteen times the frames to reach the same
   * simulated state, and in a software rasteriser that is the difference
   * between a two-minute run and a forty-minute one.
   */
  setHitStop(enabled: boolean): void {
    this.#options.hitStop = enabled;
    if (!enabled && this.#hitStopLeft > 0) {
      this.#hitStopLeft = 0;
      if (this.#ctx !== null) this.#ctx.time.scale = 1;
    }
  }

  /* -- lifecycle ----------------------------------------------------------- */

  init(ctx: GameContext): void {
    this.#ctx = ctx;
    ctx.scene.add(this.#sparks.points);
    this.#lighting = ctx.services.tryGet<LightingService>(LightingKey) ?? null;

    if (this.#options.dom && typeof document !== 'undefined') this.#buildDom();

    this.#unsubscribe.push(
      ctx.events.on('combat:hit', (payload) => this.#onHit(ctx, payload)),
      ctx.events.on('combat:death', (payload) => this.#onDeath(payload)),
    );
  }

  update(ctx: GameContext, dt: number): void {
    // `dt` arrives already multiplied by the time scale this module owns, so
    // real elapsed seconds have to be recovered before anything that must keep
    // running during a hit stop is advanced.
    const scale = Math.max(1e-4, ctx.time.scale);
    const real = dt / scale;

    if (this.#hitStopLeft > 0) {
      this.#hitStopLeft = Math.max(0, this.#hitStopLeft - real);
      if (this.#hitStopLeft <= 0) ctx.time.scale = 1;
    }

    this.#shakeClock += real;
    this.#trauma = decayTrauma(this.#trauma, real, this.#options.traumaDecay);
    this.#sparks.update(dt);
    this.#updateLights(real);
    this.#updateNumbers(ctx, real);
    this.#updateVignette(real);
  }

  /**
   * Apply the shake.
   *
   * Runs in `lateUpdate` after `CameraRig` has placed the camera. Nothing is
   * written back into the rig, so the next frame starts from a clean pose and
   * the offsets cannot integrate into a drift.
   */
  lateUpdate(ctx: GameContext): void {
    const shake = traumaToShake(this.#trauma);
    if (shake <= 1e-4) {
      this.#offset.set(0, 0, 0);
      return;
    }
    const t = this.#shakeClock * this.#options.shakeFrequency;
    const amplitude = shake * this.#options.shakeAmplitude;
    this.#offset.set(
      valueNoise(11, t) * amplitude,
      valueNoise(29, t) * amplitude * 0.8,
      valueNoise(47, t) * amplitude * 0.5,
    );
    // Applied in camera space, so the shake is always lateral to the view
    // rather than to the world — a world-space shake reads as the *level*
    // moving when the player happens to be looking down an axis.
    this.#offset.applyQuaternion(ctx.camera.quaternion);
    ctx.camera.position.add(this.#offset);
    ctx.camera.rotateZ(valueNoise(83, t) * shake * this.#options.shakeRoll);
    ctx.camera.updateMatrixWorld();
  }

  dispose(): void {
    for (const off of this.#unsubscribe) off();
    this.#unsubscribe = [];
    for (const light of this.#lights) light.handle.release();
    this.#lights.length = 0;
    for (const entry of this.#numbers) entry.element.remove();
    this.#numbers.length = 0;
    this.#sparks.dispose();
    this.#container?.remove();
    this.#container = null;
    this.#vignette?.remove();
    this.#vignette = null;
    if (this.#ctx !== null) this.#ctx.time.scale = 1;
    this.#ctx = null;
  }

  /* -- reactions ------------------------------------------------------------ */

  #onHit(
    ctx: GameContext,
    payload: {
      target: number;
      outcome: { result: string; total: number; critical: boolean; applied: Record<string, number> };
      point: THREE.Vector3;
      normal: THREE.Vector3;
      direction: THREE.Vector3;
      severity: number;
    },
  ): void {
    const blocked = payload.outcome.result === 'blocked';
    const severity = payload.severity;

    // Hit stop, scaled by damage. A block still hangs, briefly — the impact was
    // real, the damage was not.
    if (this.#options.hitStop) {
      const duration = blocked ? HIT_STOP_MIN * 0.6 : hitStopDuration(severity, 1);
      if (duration > this.#hitStopLeft) {
        this.#hitStopLeft = duration;
        ctx.time.scale = 0.06;
      }
    }

    this.#trauma = Math.min(
      1,
      this.#trauma + (blocked ? 0.12 : traumaForHit(severity, payload.outcome.critical)),
    );

    const cold = (payload.outcome.applied.cold ?? 0) > payload.outcome.total * 0.4;
    const colour = blocked
      ? this.#colour.setHex(0xffd9a0)
      : cold
        ? this.#colour.setHex(0x8fd8ff)
        : this.#colour.setHex(0xff5a32);
    const count = blocked ? 8 : Math.round(10 + Math.min(18, severity * 55));
    this.#sparks.burst(
      payload.point,
      payload.normal,
      payload.direction,
      count,
      colour,
      blocked ? 3 : 4.5 + severity * 6,
      () => this.#random(),
    );
    // Kept modest on purpose. The post stack blooms this, and an impact light
    // bright enough to look right without bloom turns the target into a white
    // blob with it — which reads as a rendering fault, not as a hit.
    this.#addImpactLight(payload.point, colour, blocked ? 1.8 : 2.6 + severity * 5.5);

    const target = this.#findCombatant(ctx, payload.target);
    if (target !== null && isFlashable(target)) {
      target.flash(blocked ? 0xfff0d0 : 0xffffff, blocked ? 0.1 : 0.14 + severity * 0.2);
    }
    if (target !== null && target.faction === 'player' && !blocked) {
      this.#vignetteLevel = Math.min(1, this.#vignetteLevel + 0.35 + severity);
    }

    this.#spawnNumber(payload.point, payload.outcome.total, {
      blocked,
      critical: payload.outcome.critical,
      cold,
      friendly: target?.faction === 'player',
    });
  }

  #onDeath(payload: { faction: string }): void {
    this.#trauma = Math.min(1, this.#trauma + (payload.faction === 'player' ? 0.75 : 0.28));
  }

  #findCombatant(ctx: GameContext, id: number): Combatant | null {
    const combat = ctx.services.tryGet<CombatSystem>(CombatKey);
    if (combat === undefined) return null;
    return combat.targets.all.find((entry) => entry.id === id) ?? null;
  }

  /* -- impact lights -------------------------------------------------------- */

  /**
   * A short, bright flash at the contact point, through the lighting service.
   *
   * Never `scene.add(new PointLight())`: the renderer arbitrates a fixed light
   * budget every frame and a raw light bypasses the whole clustered pass.
   */
  #addImpactLight(point: THREE.Vector3, colour: THREE.Color, intensity: number): void {
    const lighting = this.#lighting;
    if (lighting === null || intensity <= 0) return;
    const handle = lighting.addLight({
      kind: 'point',
      position: { x: point.x, y: point.y, z: point.z },
      color: colour.getHex(),
      intensity,
      radius: 3.2,
      name: 'combat.impact',
    });
    this.#lights.push({ handle, peak: intensity, life: 0.12, maxLife: 0.12 });
  }

  #updateLights(dt: number): void {
    for (let i = this.#lights.length - 1; i >= 0; i--) {
      const light = this.#lights[i];
      if (light === undefined) continue;
      light.life -= dt;
      if (light.life <= 0) {
        light.handle.release();
        this.#lights.splice(i, 1);
        continue;
      }
      const fade = light.life / light.maxLife;
      light.handle.setIntensity(light.peak * fade * fade);
    }
  }

  /* -- DOM ------------------------------------------------------------------ */

  #buildDom(): void {
    const container = document.createElement('div');
    container.id = 'd2rim-combat-feedback';
    container.setAttribute(
      'style',
      'position:fixed;inset:0;pointer-events:none;z-index:40;overflow:hidden;' +
        "font:600 20px/1 'Trebuchet MS',Georgia,serif;",
    );
    document.body.appendChild(container);
    this.#container = container;

    const vignette = document.createElement('div');
    vignette.setAttribute(
      'style',
      'position:fixed;inset:0;pointer-events:none;z-index:39;opacity:0;' +
        'background:radial-gradient(ellipse at center,rgba(0,0,0,0) 45%,rgba(150,10,10,0.85) 100%);',
    );
    document.body.appendChild(vignette);
    this.#vignette = vignette;
  }

  #spawnNumber(
    point: THREE.Vector3,
    amount: number,
    style: { blocked: boolean; critical: boolean; cold: boolean; friendly: boolean },
  ): void {
    const container = this.#container;
    if (container === null) return;
    const element = document.createElement('div');
    const colour = style.blocked
      ? '#d8d2c0'
      : style.friendly
        ? '#ff6b5a'
        : style.critical
          ? '#ffd166'
          : style.cold
            ? '#9fe0ff'
            : '#f4ece0';
    const size = style.critical ? 30 : style.blocked ? 17 : 21;
    element.textContent = style.blocked ? 'block' : String(Math.max(1, Math.round(amount)));
    element.setAttribute(
      'style',
      `position:absolute;transform:translate(-50%,-50%);color:${colour};` +
        `font-size:${size}px;text-shadow:0 2px 6px rgba(0,0,0,0.9),0 0 12px rgba(0,0,0,0.7);` +
        'white-space:nowrap;will-change:transform,opacity;',
    );
    container.appendChild(element);
    this.#numbers.push({
      element,
      origin: point.clone(),
      life: 0.95,
      maxLife: 0.95,
      drift: new THREE.Vector3((this.#random() - 0.5) * 0.5, 1.35, (this.#random() - 0.5) * 0.5),
    });
  }

  #updateNumbers(ctx: GameContext, dt: number): void {
    if (this.#numbers.length === 0) return;
    const canvas = ctx.renderer.three.domElement as HTMLCanvasElement | undefined;
    const rect = canvas?.getBoundingClientRect();
    for (let i = this.#numbers.length - 1; i >= 0; i--) {
      const entry = this.#numbers[i];
      if (entry === undefined) continue;
      entry.life -= dt;
      if (entry.life <= 0) {
        entry.element.remove();
        this.#numbers.splice(i, 1);
        continue;
      }
      const age = 1 - entry.life / entry.maxLife;
      this.#projected
        .copy(entry.origin)
        .addScaledVector(entry.drift, age * 0.9 - age * age * 0.35)
        .project(ctx.camera);
      const visible = this.#projected.z < 1;
      const width = rect?.width ?? window.innerWidth;
      const height = rect?.height ?? window.innerHeight;
      const left = (rect?.left ?? 0) + (this.#projected.x * 0.5 + 0.5) * width;
      const top = (rect?.top ?? 0) + (-this.#projected.y * 0.5 + 0.5) * height;
      entry.element.style.left = `${left.toFixed(1)}px`;
      entry.element.style.top = `${top.toFixed(1)}px`;
      entry.element.style.opacity = visible ? (1 - age * age).toFixed(3) : '0';
    }
  }

  #updateVignette(dt: number): void {
    const vignette = this.#vignette;
    if (vignette === null) return;
    if (this.#vignetteLevel <= 0) return;
    this.#vignetteLevel = Math.max(0, this.#vignetteLevel - dt * 1.4);
    vignette.style.opacity = (this.#vignetteLevel * 0.8).toFixed(3);
  }

  /* -- rng ------------------------------------------------------------------ */

  /** A private, seeded stream so visuals never perturb the combat RNG. */
  #random(): number {
    this.#rngState = (this.#rngState + 0x6d2b79f5) >>> 0;
    let t = this.#rngState;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
}
