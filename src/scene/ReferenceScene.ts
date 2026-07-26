/**
 * @module scene/ReferenceScene
 *
 * The project's visual baseline.
 *
 * This is not a debug scene. It is the calibration target every later
 * environment is judged against, so it is lit and composed as if it were
 * shipping art: procedural HDR sky doing both background and image-based
 * lighting, one warm directional key casting soft shadows, no flat ambient
 * term anywhere, and ACES tone mapping at exposure 1.0.
 *
 * The subject matter is a material response chart — a metalness sweep and a
 * roughness sweep flanking a polished hero object on weathered ground. That
 * choice is deliberate: it exercises the full PBR path (dielectric and
 * conductor, mirror to matte, normal-mapped and not, shadow caster and
 * receiver) while still reading as a composed image rather than a test grid.
 * If the renderer regresses, it shows up here first and unmistakably.
 *
 * Everything animates from {@link TimeState.elapsed}, never from wall-clock
 * time, so `Engine.stepFrames(n)` reproduces any frame exactly.
 */

import * as THREE from 'three/webgpu';

import { AssetManagerKey } from '../assets/AssetManager';
import type { GameContext, GameModule } from '../core/types';
import { createProceduralSky, type ProceduralSky } from '../render/ProceduralSky';
import { createGroundMaterialMaps, type GroundMaterialMaps } from '../render/ProceduralTextures';

/**
 * The image-based lighting source.
 *
 * `env.overcast` is a real, fetched, CC0 HDRI (see `public/ATTRIBUTIONS.md`),
 * picked on measured radiance rather than on its filename: mean luminance 0.70,
 * blue/red ratio 1.80, and a peak-to-mean of only 57, meaning there is no hard
 * sun disc in it. That is genuinely diffuse overcast light, which is exactly the
 * cold flat key Diablo II's Act I lives under.
 */
const ENVIRONMENT_KEY = 'env.overcast';

/** Camera framing constants, kept together so the composition is tunable in one place. */
const CAMERA = {
  fov: 38,
  near: 0.1,
  far: 600,
  /** Orbit radius, height and look-at target. */
  radius: 15.0,
  height: 4.0,
  target: new THREE.Vector3(0, 1.35, 0),
  /** Degrees per second. Slow enough to read as a considered reveal. */
  orbitSpeed: 3.5,
  /** Starting azimuth, chosen so the key light rakes across the hero. */
  startAzimuth: -28,
} as const;

export class ReferenceScene implements GameModule {
  readonly name = 'ReferenceScene';

  #sky: ProceduralSky | null = null;
  #ground: GroundMaterialMaps | null = null;
  #root: THREE.Group | null = null;
  /** Objects with per-frame motion, paired with their phase offset. */
  #spinners: Array<{ mesh: THREE.Mesh; speed: number }> = [];
  readonly #disposables: Array<{ dispose(): void }> = [];
  /**
   * Set when this scene pinned an environment map in the asset cache. Held so
   * that `dispose` can release the pin — the texture itself belongs to the
   * AssetManager and must not be disposed here.
   */
  #pinnedEnvironment: { unpin(key: typeof ENVIRONMENT_KEY): void } | null = null;

  async init(ctx: GameContext): Promise<void> {
    const { scene, camera } = ctx;

    const sky = createProceduralSky({ sunElevation: 27, sunAzimuth: 42, cloudiness: 0.38 });
    this.#sky = sky;

    // `applyToScene` installs the procedural sky as *both* the visible
    // background and the IBL source. The background stays — a 1024x512 sky with
    // a real sun disc is a far better thing to look at than a 512x256 IBL plate.
    // The environment half is then replaced below by the fetched HDRI.
    sky.applyToScene(scene);
    await this.#applyFetchedEnvironment(ctx);

    // Slightly under 1 so shadowed faces keep some depth. At full strength the
    // sky fill competes with the key and the image loses its contrast range.
    scene.environmentIntensity = 0.85;

    // Aerial perspective. The colour is sampled from the sky's warm horizon so
    // distant ground melts into the haze instead of ending at a hard edge.
    scene.fog = new THREE.FogExp2(new THREE.Color(0.50, 0.40, 0.33), 0.008);

    const root = new THREE.Group();
    root.name = 'ReferenceScene';
    this.#root = root;
    scene.add(root);

    this.#buildLighting(root, sky);
    this.#buildGround(root);
    this.#buildMaterialChart(root);
    this.#buildHero(root);

    camera.fov = CAMERA.fov;
    camera.near = CAMERA.near;
    camera.far = CAMERA.far;
    camera.updateProjectionMatrix();
    this.#placeCamera(camera, 0);
  }

  update(ctx: GameContext, dt: number): void {
    this.#placeCamera(ctx.camera, ctx.time.elapsed);
    for (const { mesh, speed } of this.#spinners) {
      mesh.rotation.y += dt * speed;
    }
  }

  dispose(): void {
    const root = this.#root;
    if (root !== null) {
      root.removeFromParent();
      this.#root = null;
    }
    for (const item of this.#disposables) item.dispose();
    this.#disposables.length = 0;
    this.#sky?.dispose();
    this.#sky = null;
    this.#ground?.dispose();
    this.#ground = null;
    this.#spinners = [];
    // Release the pin only; the AssetManager owns and disposes the texture.
    this.#pinnedEnvironment?.unpin(ENVIRONMENT_KEY);
    this.#pinnedEnvironment = null;
  }

  // -- construction -------------------------------------------------------

  /** Track a resource so `dispose` can release GPU memory deterministically. */
  #own<T extends { dispose(): void }>(resource: T): T {
    this.#disposables.push(resource);
    return resource;
  }

  #placeCamera(camera: THREE.PerspectiveCamera, elapsed: number): void {
    const angle = THREE.MathUtils.degToRad(CAMERA.startAzimuth + elapsed * CAMERA.orbitSpeed);
    camera.position.set(
      Math.sin(angle) * CAMERA.radius,
      CAMERA.height,
      Math.cos(angle) * CAMERA.radius,
    );
    camera.lookAt(CAMERA.target);
  }

  /**
   * Replace the procedural placeholder environment with the fetched HDRI.
   *
   * The texture is owned by the {@link AssetManager}, not by this scene, so it
   * is pinned rather than tracked as a disposable: an environment map evicted
   * by an LRU sweep mid-frame would blank every reflection in the image.
   *
   * Degrading rather than throwing is deliberate. A fresh clone that has not run
   * `npm run assets` yet still boots and still renders a correct, lit frame —
   * it just falls back to the procedural sky's own IBL. A hard failure here
   * would turn a missing optional download into a black screen.
   */
  async #applyFetchedEnvironment(ctx: GameContext): Promise<void> {
    const assets = ctx.services.tryGet(AssetManagerKey);
    if (assets === undefined) {
      console.warn('[ReferenceScene] no AssetManager registered; keeping procedural environment');
      return;
    }

    try {
      const environment = await assets.loadEnvironment(ENVIRONMENT_KEY);
      assets.pin(ENVIRONMENT_KEY);
      ctx.scene.environment = environment;
      this.#pinnedEnvironment = assets;
      console.info(
        `[ReferenceScene] environment: ${ENVIRONMENT_KEY} ` +
          `(${assets.entry(ENVIRONMENT_KEY).license}, ${assets.url(ENVIRONMENT_KEY)})`,
      );
    } catch (error) {
      console.warn(
        `[ReferenceScene] could not load "${ENVIRONMENT_KEY}", falling back to the ` +
          `procedural sky environment. Run \`npm run assets\` to fetch it.`,
        error,
      );
    }
  }

  /**
   * One key light, and nothing else.
   *
   * All fill comes from the sky IBL, which is what gives shadowed faces a cool
   * blue bounce and lit faces a warm one. An `AmbientLight` would flatten
   * exactly that gradient — hence the contract's "no flat ambient".
   */
  #buildLighting(root: THREE.Group, sky: ProceduralSky): void {
    const sun = new THREE.DirectionalLight(sky.sunColor.clone(), 3.6);
    sun.position.copy(sky.sunDirection).multiplyScalar(60);
    sun.castShadow = true;

    sun.shadow.mapSize.set(2048, 2048);
    // Tight ortho frustum around the subject: shadow texel density is the
    // single biggest lever on shadow quality, and a loose frustum wastes it.
    const extent = 14;
    sun.shadow.camera.left = -extent;
    sun.shadow.camera.right = extent;
    sun.shadow.camera.top = extent;
    sun.shadow.camera.bottom = -extent;
    sun.shadow.camera.near = 1;
    sun.shadow.camera.far = 140;
    // Constant bias fights acne on flat surfaces; normal bias fights it on
    // curved ones without the peter-panning a large constant bias causes.
    sun.shadow.bias = -0.0004;
    sun.shadow.normalBias = 0.035;
    sun.shadow.camera.updateProjectionMatrix();

    root.add(sun);
    root.add(sun.target);
  }

  #buildGround(root: THREE.Group): void {
    const maps = createGroundMaterialMaps({ size: 512, repeat: 26 });
    this.#ground = maps;

    const geometry = this.#own(new THREE.PlaneGeometry(400, 400, 1, 1));
    const material = this.#own(
      new THREE.MeshStandardMaterial({
        map: maps.map,
        roughnessMap: maps.roughnessMap,
        normalMap: maps.normalMap,
        normalScale: new THREE.Vector2(0.85, 0.85),
        metalness: 0.0,
        roughness: 1.0,
        // `envMapIntensity` below 1 keeps the ground from picking up an
        // unrealistic amount of sky and going blue.
        envMapIntensity: 0.85,
      }),
    );

    const ground = new THREE.Mesh(geometry, material);
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    ground.name = 'Ground';
    root.add(ground);
  }

  /**
   * Two arcs of spheres: conductors sweeping roughness in front, dielectrics
   * sweeping roughness behind. Together they cover the corners of the BRDF that
   * break first when a renderer regresses.
   */
  #buildMaterialChart(root: THREE.Group): void {
    const geometry = this.#own(new THREE.SphereGeometry(0.62, 48, 32));
    const count = 5;
    const spread = 5.2;

    for (let i = 0; i < count; i++) {
      const t = i / (count - 1);
      const x = -spread + t * spread * 2;
      // Arrange along a shallow arc so the row does not read as a flat ruler.
      const z = -4.1 - Math.cos((t - 0.5) * Math.PI) * 0.9;

      const metal = this.#own(
        new THREE.MeshStandardMaterial({
          // Warm steel that drifts toward bronze along the sweep.
          color: new THREE.Color().setHSL(0.075, 0.18 + t * 0.22, 0.55),
          metalness: 1.0,
          roughness: 0.04 + t * 0.58,
          envMapIntensity: 1.0,
        }),
      );
      const sphere = new THREE.Mesh(geometry, metal);
      sphere.position.set(x, 0.62, z);
      sphere.castShadow = true;
      sphere.receiveShadow = true;
      root.add(sphere);
    }

    for (let i = 0; i < count; i++) {
      const t = i / (count - 1);
      const x = spread - t * spread * 2;
      const z = 3.6 + Math.cos((t - 0.5) * Math.PI) * 0.9;

      const dielectric = this.#own(
        new THREE.MeshStandardMaterial({
          color: new THREE.Color(0.62, 0.16, 0.13),
          metalness: 0.0,
          roughness: 0.06 + t * 0.85,
          envMapIntensity: 1.0,
        }),
      );
      const sphere = new THREE.Mesh(geometry, dielectric);
      sphere.position.set(x, 0.62, z);
      sphere.castShadow = true;
      sphere.receiveShadow = true;
      root.add(sphere);
    }
  }

  /**
   * The focal point: a polished torus knot on a rough stone plinth, flanked by
   * two brass posts. Mixed geometry complexity next to the smooth spheres keeps
   * the frame from looking synthetic.
   */
  #buildHero(root: THREE.Group): void {
    const plinthGeometry = this.#own(new THREE.BoxGeometry(2.6, 0.55, 2.6));
    const plinthMaterial = this.#own(
      new THREE.MeshStandardMaterial({
        color: new THREE.Color(0.16, 0.155, 0.15),
        metalness: 0.0,
        roughness: 0.82,
        envMapIntensity: 0.9,
      }),
    );
    const plinth = new THREE.Mesh(plinthGeometry, plinthMaterial);
    plinth.position.set(0, 0.275, 0);
    plinth.castShadow = true;
    plinth.receiveShadow = true;
    root.add(plinth);

    const knotGeometry = this.#own(new THREE.TorusKnotGeometry(0.78, 0.25, 220, 40));
    const knotMaterial = this.#own(
      new THREE.MeshStandardMaterial({
        color: new THREE.Color(0.78, 0.68, 0.46),
        metalness: 1.0,
        roughness: 0.17,
        envMapIntensity: 1.0,
      }),
    );
    const knot = new THREE.Mesh(knotGeometry, knotMaterial);
    knot.position.set(0, 1.72, 0);
    knot.castShadow = true;
    knot.receiveShadow = true;
    root.add(knot);
    this.#spinners.push({ mesh: knot, speed: 0.35 });

    const postGeometry = this.#own(new THREE.CylinderGeometry(0.17, 0.21, 2.3, 24));
    const postMaterial = this.#own(
      new THREE.MeshStandardMaterial({
        color: new THREE.Color(0.55, 0.42, 0.20),
        metalness: 0.95,
        roughness: 0.34,
        envMapIntensity: 1.0,
      }),
    );
    for (const x of [-4.0, 4.0]) {
      const post = new THREE.Mesh(postGeometry, postMaterial);
      post.position.set(x, 1.15, 0);
      post.castShadow = true;
      post.receiveShadow = true;
      root.add(post);
    }
  }
}
