/**
 * @module scene/Fire
 *
 * A reusable firelight source: ember bed, flame plume and a flicker curve.
 *
 * The Blood Moor's campfire is authored inline, tuned against one composition
 * and one camera. This is the generalised version, because Act I needs the same
 * effect at five different scales — a camp bonfire, a forge's coal bed, torches
 * on posts, and the sparse guttering torches that are the only light in the Den
 * of Evil — and five copies of a 200-line shader is how a project ends up with
 * five subtly different fires.
 *
 * ## Three pieces, and why each one is needed
 *
 * - **the ember bed**, an unlit disc whose radial falloff goes from near-white
 *   through orange to almost black. It is the *bright* part: a fire's core is
 *   several stops above its flame, and without it the plume reads as an orange
 *   fog card.
 * - **the plume**, an open-ended cone with additive blending, alpha falling off
 *   toward the tip and a noise-driven horizontal wobble. Open-ended matters —
 *   a capped cone has a visible disc at the top when seen from above.
 * - **the flicker**, two incommensurable sine frequencies plus a third. A single
 *   frequency beats visibly; a random walk reads as a fault in the display. The
 *   same scalar drives the light intensity, the emissive strength and the plume's
 *   vertical scale, because a fire whose light and geometry flicker out of phase
 *   reads as a lamp with a picture of a fire on it.
 *
 * ## Determinism
 *
 * The animation clock is a `uniform` fed from `ctx.time.elapsed`, never TSL's
 * `time` node. That node reads the renderer's own wall clock, which under
 * `Engine.stepFrames(n)` is not the simulation clock — a fire driven by it makes
 * every capture a different picture. This is the whole determinism contract for
 * this effect, in one variable.
 *
 * ## Cost
 *
 * Two meshes and two materials per fire: a 20-segment disc (20 triangles) and a
 * 9x6 open cone (108 triangles). Both are unlit `MeshBasicNodeMaterial`, so
 * neither takes part in shadow, GTAO or IBL. A dozen fires in a zone is 24 draw
 * calls and under 1600 triangles, which is not the reason any frame is slow.
 */

import * as THREE from 'three/webgpu';
import {
  float,
  mix,
  mx_noise_float,
  oneMinus,
  positionLocal,
  sin,
  smoothstep,
  uniform,
  vec3,
  vec4,
} from 'three/tsl';

export interface FireOptions {
  /** Radius of the ember bed, in metres. The plume is sized from it. */
  readonly radius?: number;
  /** Plume height, in metres. Default `radius * 4`. */
  readonly height?: number;
  /**
   * Radiance multiplier for the ember bed. Values well above 1 are correct and
   * intended: this is the brightest thing in the act and the tonemapper is what
   * brings it back into range.
   */
  readonly intensity?: number;
  /** Phase offset so a row of torches does not pulse in unison. */
  readonly phase?: number;
  readonly name?: string;
}

/**
 * The flicker curve, as a pure function of time.
 *
 * Exported and pure so it can be asserted rather than eyeballed: the useful
 * property is that it stays inside roughly ±17% of unity, because a light that
 * swings 50% is a strobe and a light that swings 3% is a constant.
 */
export function fireFlicker(elapsed: number, phase = 0): number {
  const t = elapsed + phase;
  return 1 + 0.085 * Math.sin(t * 7.31) + 0.055 * Math.sin(t * 11.97 + 1.7) + 0.03 * Math.sin(t * 19.3);
}

export class Fire {
  /** Add this to the scene at the base of the fire. */
  readonly group = new THREE.Group();

  readonly #radius: number;
  readonly #height: number;
  readonly #phase: number;
  readonly #time = uniform(0);
  readonly #glow = uniform(1);
  readonly #plume: THREE.Mesh;
  readonly #owned: { dispose(): void }[] = [];

  #flicker = 1;

  constructor(options: FireOptions = {}) {
    const radius = options.radius ?? 0.42;
    this.#radius = radius;
    this.#height = options.height ?? radius * 4;
    this.#phase = options.phase ?? 0;
    const intensity = options.intensity ?? 1;
    this.group.name = options.name ?? 'fire';
    // Never a collider and never an occluder: it is a light-emitting card.
    this.group.userData['noCollide'] = true;

    /* -- ember bed ------------------------------------------------------- */

    const emberGeometry = new THREE.CircleGeometry(radius, 20);
    emberGeometry.rotateX(-Math.PI / 2);
    const radial = positionLocal.xz.length().div(float(radius)).clamp(0, 1);
    // Squared falloff, not linear: coals are a hot core with a fast edge, and a
    // linear ramp reads as a painted circle.
    const falloff = oneMinus(radial.mul(radial)).clamp(0, 1);
    // Slow crawling noise so individual coals brighten and dim independently.
    const coalSpace = positionLocal.mul(11).add(vec3(0, this.#time.mul(0.5), 0));
    const coals = mx_noise_float(coalSpace).mul(0.5).add(0.72).clamp(0.3, 1.3);
    const emberColour = mix(
      vec3(5.2, 1.7, 0.4),
      vec3(0.14, 0.02, 0.008),
      radial.pow(0.55),
    ).mul(coals).mul(this.#glow).mul(float(intensity));

    const emberMaterial = new THREE.MeshBasicNodeMaterial({
      transparent: true,
      depthWrite: false,
      toneMapped: true,
      side: THREE.FrontSide,
    });
    emberMaterial.name = 'fire.embers';
    emberMaterial.colorNode = vec4(emberColour, falloff);
    const embers = new THREE.Mesh(emberGeometry, emberMaterial);
    embers.name = 'fire.embers';
    embers.renderOrder = 4;
    embers.userData['noCollide'] = true;
    this.group.add(embers);
    this.#owned.push(emberGeometry, emberMaterial);

    /* -- plume ----------------------------------------------------------- */

    const plumeGeometry = new THREE.CylinderGeometry(
      radius * 0.03,
      radius * 0.72,
      this.#height,
      9,
      6,
      true,
    );
    plumeGeometry.translate(0, this.#height / 2, 0);

    const rise = positionLocal.y.div(float(this.#height)).clamp(0, 1);
    // Three-stop ramp: the base is the hottest and least saturated, the middle
    // is the orange everyone thinks of as "fire", the tip goes to smoke-dark.
    const plumeColour = mix(
      mix(vec3(1.18, 0.44, 0.115), vec3(0.68, 0.185, 0.025), smoothstep(0, 0.4, rise)),
      vec3(0.16, 0.05, 0.02),
      smoothstep(0.4, 1, rise),
    );
    // Licks travelling up the plume, so the silhouette moves rather than just
    // brightening. Two terms at different rates, again to avoid a visible beat.
    const lick = sin(rise.mul(9.4).sub(this.#time.mul(6.1)))
      .mul(0.5)
      .add(0.5)
      .mul(0.45)
      .add(0.55);
    const alpha = oneMinus(smoothstep(0.02, 0.86, rise)).mul(lick).mul(this.#glow).clamp(0, 1);

    const plumeMaterial = new THREE.MeshBasicNodeMaterial({
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
      toneMapped: true,
    });
    plumeMaterial.name = 'fire.plume';
    // World-space-ish noise on the local position: the plume leans and bulges
    // instead of being a rotationally symmetric cone, which is the single tell
    // that separates a fire from a traffic cone with a gradient on it.
    const wobbleSpace = positionLocal.mul(2.6).add(vec3(0, this.#time.mul(-1.9), this.#time.mul(0.4)));
    const wobble = mx_noise_float(wobbleSpace).mul(0.5).add(0.5);
    plumeMaterial.colorNode = vec4(plumeColour.mul(float(intensity)).mul(wobble.add(0.55)), alpha);

    const plume = new THREE.Mesh(plumeGeometry, plumeMaterial);
    plume.name = 'fire.plume';
    plume.renderOrder = 5;
    plume.userData['noCollide'] = true;
    this.group.add(plume);
    this.#plume = plume;
    this.#owned.push(plumeGeometry, plumeMaterial);
  }

  /** The current flicker multiplier, for driving a light alongside the geometry. */
  get flicker(): number {
    return this.#flicker;
  }

  get radius(): number {
    return this.#radius;
  }

  /**
   * Advance the effect. `elapsed` must be `ctx.time.elapsed`, never a wall clock.
   *
   * @returns the flicker multiplier, so the caller can apply it to its light in
   * the same statement and the two can never drift apart.
   */
  update(elapsed: number): number {
    const flicker = fireFlicker(elapsed, this.#phase);
    this.#flicker = flicker;
    this.#time.value = elapsed + this.#phase;
    this.#glow.value = flicker;
    // The plume also *breathes*: a fire's height varies, not just its brightness.
    // Y only, so the base stays welded to the ember bed.
    this.#plume.scale.set(1, 0.9 + 0.22 * flicker, 1);
    return flicker;
  }

  dispose(): void {
    for (const resource of this.#owned) resource.dispose();
    this.#owned.length = 0;
    this.group.removeFromParent();
    this.group.clear();
  }
}
