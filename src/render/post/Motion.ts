/**
 * @module render/post/Motion
 *
 * Screen-space motion vectors, and the sub-pixel camera jitter that TAA needs.
 *
 * Everything temporal in this renderer — TAA, motion blur, screen-space
 * reflections, temporally-amortised AO — reprojects last frame's result into
 * this frame. Reprojection needs to know, for every pixel, where the surface it
 * shows *was* last frame. That is what the velocity buffer stores.
 *
 * ---
 *
 * ## What is written
 *
 * Velocity is stored as the **NDC-space delta** `p_now.xy − p_prev.xy`, in
 * `[-2, 2]`, exactly as three.js's own `VelocityNode` produces it:
 *
 * ```
 * clip_now  = P      · V      · M      · positionLocal
 * clip_prev = P_prev · V_prev · M_prev · positionPrevious
 * v         = clip_now.xy / clip_now.w  −  clip_prev.xy / clip_prev.w
 * ```
 *
 * The consumer converts to a texture-space offset with
 * {@link NDC_TO_UV} = `(0.5, −0.5)` and reads history at `uv − v * NDC_TO_UV`.
 * The Y flip is not cosmetic: NDC is Y-up, three's render-target texture space
 * is Y-down, and getting this wrong produces a TAA that looks *almost* right
 * while smearing every horizontal edge.
 *
 * Three classes of motion are covered, and the coverage is why this is a
 * separate module rather than three lines inside `PostStack`:
 *
 * 1. **Camera motion.** `P_prev`/`V_prev` are snapshotted once per frame.
 * 2. **Rigid object motion.** `M_prev` is snapshotted per object, after the
 *    draw, by `VelocityNode.updateAfter`.
 * 3. **Skinned/instanced deformation.** `positionPrevious` is a varying that
 *    `Skinning.js` and `Instance.js` overwrite with the vertex position under
 *    the *previous* bone palette / instance matrix. That only happens when
 *    `NodeBuilder.needsPreviousData()` is true, which in turn is only true when
 *    the renderer's MRT declares a `velocity` output. So the Barbarian and the
 *    Fallen get correct per-vertex velocity *for free*, but only if the scene
 *    is drawn through {@link MotionVectors.mrtNode}. Draw the scene without it
 *    and skinned characters silently get rigid-body velocity, which reads as
 *    ghost limbs trailing behind every animation.
 *
 * ## Jitter
 *
 * TAA converges by moving the sample point inside the pixel footprint every
 * frame and letting the temporal filter integrate the results. The offsets come
 * from a Halton(2, 3) low-discrepancy sequence (Halton 1964), which is the
 * standard choice — Karis, *"High Quality Temporal Supersampling"*, SIGGRAPH
 * 2014 Advances in Real-Time Rendering — because its partial sums stay evenly
 * distributed, so the image is well-sampled after *any* number of frames rather
 * than only after a full period. A random sequence clumps; a regular grid
 * aliases against the pixel grid.
 *
 * Jitter is applied through `Camera.setViewOffset`, which folds the offset into
 * the projection matrix, and is cleared again as soon as the scene draw
 * finishes so that gameplay code (picking, frustum culling, UI projection)
 * never observes a jittered camera. The *unjittered* projection is handed to
 * `VelocityNode.setProjectionMatrix` so that the jitter — which is not motion —
 * does not leak into the velocity buffer.
 *
 * ## History rejection
 *
 * A camera cut (teleport, cutscene, respawn) invalidates every reprojection at
 * once. Detecting it here rather than in TAA keeps the policy in one place:
 * {@link MotionVectors.historyValid} goes false for one frame whenever the
 * camera translates further than a heuristic threshold or rotates more than a
 * few degrees between frames, and consumers hard-reset their history.
 *
 * ## References
 *
 * - B. Karis, *"High Quality Temporal Supersampling"*, SIGGRAPH 2014.
 * - J. Jimenez, *"Filmic SMAA"*, SIGGRAPH 2016 (velocity buffer layout).
 * - three.js `src/nodes/accessors/VelocityNode.js` (the encoding this matches).
 */

import * as THREE from 'three/webgpu';
import { mrt, output, velocity } from 'three/tsl';

import { serviceKey } from '../../core/ServiceLocator';

/* ------------------------------------------------------------------------- *
 * Pure math — exported for tests
 * ------------------------------------------------------------------------- */

/**
 * Van der Corput radical inverse of `index` in `base`.
 *
 * Reflects the base-`b` digits of `index` about the radix point:
 * `123` in base 10 becomes `0.321`. Doing it with the iterative
 * multiply-accumulate below rather than by building a digit array keeps it
 * branch-light and exact for the small indices a jitter sequence uses.
 *
 * @param index one-based sample index; index 0 returns 0
 * @param base any integer >= 2, conventionally a prime
 */
export function radicalInverse(index: number, base: number): number {
  let result = 0;
  let denominator = 1;
  let i = index;
  while (i > 0) {
    denominator /= base;
    result += denominator * (i % base);
    i = Math.floor(i / base);
  }
  return result;
}

/**
 * A Halton(baseX, baseY) sequence of sub-pixel offsets in `[-0.5, 0.5)`.
 *
 * Returned flat as `[x0, y0, x1, y1, ...]` so it can be indexed without
 * allocating a `Vector2` per frame.
 *
 * Index 0 of the raw Halton sequence is the degenerate `(0, 0)`, which is a
 * *useful* sample (the unjittered pixel centre) but a terrible one to start
 * from, because the first frame of a fresh history would then be the only
 * unjittered frame. Sampling starts at raw index 1.
 */
export function haltonJitterSequence(count: number, baseX = 2, baseY = 3): Float32Array {
  const offsets = new Float32Array(count * 2);
  for (let i = 0; i < count; i++) {
    offsets[i * 2] = radicalInverse(i + 1, baseX) - 0.5;
    offsets[i * 2 + 1] = radicalInverse(i + 1, baseY) - 0.5;
  }
  return offsets;
}

/**
 * Mean of a jitter sequence, per axis. A well-formed sequence integrates to
 * (approximately) the pixel centre, otherwise TAA converges to an image shifted
 * off the true sample grid. Used by the test suite; cheap enough to assert on
 * in debug builds too.
 */
export function jitterSequenceMean(offsets: Float32Array): [number, number] {
  const count = offsets.length / 2;
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < count; i++) {
    sx += offsets[i * 2] ?? 0;
    sy += offsets[i * 2 + 1] ?? 0;
  }
  return [sx / count, sy / count];
}

/**
 * Scale factor from a velocity buffer's NDC delta to a texture-space offset.
 *
 * `x` maps `[-1, 1] -> [0, 1]`; `y` additionally flips because NDC is Y-up and
 * render-target texture space is Y-down.
 */
export const NDC_TO_UV: readonly [number, number] = [0.5, -0.5];

/* ------------------------------------------------------------------------- *
 * Service contract
 * ------------------------------------------------------------------------- */

/**
 * What this module publishes for every other temporal effect in the renderer.
 *
 * Registered under {@link MotionVectorsKey}. Motion blur, SSR and temporally
 * amortised AO should resolve this and degrade to a non-temporal path when it
 * is absent or when {@link MotionVectorProvider.velocityTexture} is `null`
 * (which is the case at the `low` quality tier, where the velocity attachment
 * is not allocated at all).
 *
 * Every matrix is a live reference owned by this module and rewritten in place
 * each frame. Copy, do not retain.
 */
export interface MotionVectorProvider {
  /**
   * RG (or RGBA on backends without two-channel float render targets)
   * half-float texture holding the NDC-space delta. `null` when motion vectors
   * are disabled for the current quality tier.
   */
  readonly velocityTexture: THREE.Texture | null;
  /** Depth attachment of the same render pass, or `null` before first render. */
  readonly depthTexture: THREE.DepthTexture | null;
  /** Multiply a sampled velocity by this to get a texture-space offset. */
  readonly ndcToUv: THREE.Vector2;
  /** Sub-pixel offset applied to the camera this frame, in pixels. */
  readonly jitter: THREE.Vector2;
  /** View matrix used for the previous frame's scene draw. */
  readonly previousViewMatrix: THREE.Matrix4;
  /** Unjittered projection matrix used for the previous frame's scene draw. */
  readonly previousProjectionMatrix: THREE.Matrix4;
  /** Unjittered projection matrix for the current frame. */
  readonly projectionMatrix: THREE.Matrix4;
  /** False for exactly one frame after a camera cut or a resize. */
  readonly historyValid: boolean;
  /** Monotonic count of scene draws issued through this module. */
  readonly frameIndex: number;
}

/** Service key for {@link MotionVectorProvider}. */
export const MotionVectorsKey = serviceKey<MotionVectorProvider>('render.motion');

/* ------------------------------------------------------------------------- *
 * Options
 * ------------------------------------------------------------------------- */

export interface MotionVectorsOptions {
  /**
   * Length of the Halton jitter sequence. 8 converges fast and is stable under
   * motion; 16 resolves finer detail on static shots at the cost of a longer
   * settle. Anything above 16 stops paying for itself once the neighbourhood
   * clamp is doing its job. Default 8.
   */
  sequenceLength?: number;
  /** Multiplier on the jitter amplitude, in pixels. Default 1. */
  jitterScale?: number;
  /**
   * Camera translation between frames, in world units, above which the history
   * is declared invalid. The default (8 units) is far beyond anything a
   * running player covers in one frame at 30 fps but well below a teleport.
   */
  cutTranslation?: number;
  /**
   * Camera rotation between frames, in degrees, above which the history is
   * declared invalid. 45 deg/frame is roughly 2700 deg/s — a cut, not a flick.
   */
  cutRotationDegrees?: number;
}

/* ------------------------------------------------------------------------- *
 * MotionVectors
 * ------------------------------------------------------------------------- */

/** Saved `Camera.view` state, restored after the jittered draw. */
interface CameraViewState {
  enabled: boolean;
  fullWidth: number;
  fullHeight: number;
  offsetX: number;
  offsetY: number;
  width: number;
  height: number;
}

/**
 * `PerspectiveCamera.view`, which @types/three declares as a nullable inline
 * object literal. Naming it keeps the save/restore code readable.
 */
type CameraView = CameraViewState | null;

/**
 * Owns the jitter sequence, the previous-frame camera matrices, and the MRT
 * declaration that makes three.js write velocity.
 *
 * This is *not* a `GameModule`: it has no independent frame of its own and must
 * be driven from inside {@link module:render/post/PostStack}'s scene draw, between
 * {@link beginSceneDraw} and {@link endSceneDraw}. `PostStack` registers it as
 * a service on behalf of everyone else.
 */
export class MotionVectors implements MotionVectorProvider {
  readonly ndcToUv = new THREE.Vector2(NDC_TO_UV[0], NDC_TO_UV[1]);
  readonly jitter = new THREE.Vector2(0, 0);
  readonly previousViewMatrix = new THREE.Matrix4();
  readonly previousProjectionMatrix = new THREE.Matrix4();
  readonly projectionMatrix = new THREE.Matrix4();

  velocityTexture: THREE.Texture | null = null;
  depthTexture: THREE.DepthTexture | null = null;
  historyValid = false;
  frameIndex = 0;

  #offsets: Float32Array;
  #sequenceLength: number;
  #jitterScale: number;
  #cutTranslationSq: number;
  #cutRotationCos: number;

  #jitterEnabled = true;
  #velocityEnabled = true;
  #forceReset = true;

  /** Previous frame's camera world matrix, for cut detection. */
  readonly #previousCameraWorld = new THREE.Matrix4();
  #hasPreviousCamera = false;

  /** Saved during the jittered draw so the camera is handed back untouched. */
  #savedView: CameraView = null;
  #jitterApplied = false;

  /**
   * Stable `Matrix4` instance handed to `VelocityNode`. It keeps the reference,
   * so it must be mutated in place rather than replaced.
   */
  readonly #unjitteredProjection = new THREE.Matrix4();

  /**
   * `mrt({ output, velocity })`. Built once — rebuilding it every frame would
   * invalidate every material's cached node graph and recompile the world.
   */
  readonly #mrt: THREE.Node;

  constructor(options: MotionVectorsOptions = {}) {
    this.#sequenceLength = Math.max(1, Math.floor(options.sequenceLength ?? 8));
    this.#jitterScale = options.jitterScale ?? 1;
    this.#offsets = haltonJitterSequence(this.#sequenceLength);
    const cut = options.cutTranslation ?? 8;
    this.#cutTranslationSq = cut * cut;
    this.#cutRotationCos = Math.cos(
      THREE.MathUtils.degToRad(options.cutRotationDegrees ?? 45),
    );
    this.#mrt = mrt({ output, velocity }) as unknown as THREE.Node;
  }

  /* -- configuration ----------------------------------------------------- */

  get sequenceLength(): number {
    return this.#sequenceLength;
  }

  /** Rebuilds the Halton table. Invalidates the history (the phase changes). */
  setSequenceLength(length: number): void {
    const next = Math.max(1, Math.floor(length));
    if (next === this.#sequenceLength) return;
    this.#sequenceLength = next;
    this.#offsets = haltonJitterSequence(next);
    this.#forceReset = true;
  }

  get jitterEnabled(): boolean {
    return this.#jitterEnabled;
  }

  /** Turn jitter off when nothing consumes it — it costs an image otherwise. */
  setJitterEnabled(enabled: boolean): void {
    if (enabled === this.#jitterEnabled) return;
    this.#jitterEnabled = enabled;
    this.#forceReset = true;
  }

  get velocityEnabled(): boolean {
    return this.#velocityEnabled;
  }

  /**
   * Whether the scene draw declares a velocity output.
   *
   * Turning this off removes a colour attachment *and* the previous-position
   * vertex work from every skinned material, which is the single biggest lever
   * available at the `low` tier.
   */
  setVelocityEnabled(enabled: boolean): void {
    if (enabled === this.#velocityEnabled) return;
    this.#velocityEnabled = enabled;
    this.#forceReset = true;
    if (!enabled) this.velocityTexture = null;
  }

  /** MRT declaration for the scene draw, or `null` when velocity is disabled. */
  get mrtNode(): THREE.Node | null {
    return this.#velocityEnabled ? this.#mrt : null;
  }

  /** Force one frame of history rejection (resize, scene swap, tier change). */
  reset(): void {
    this.#forceReset = true;
    this.#hasPreviousCamera = false;
  }

  /* -- per-frame driving ------------------------------------------------- */

  /**
   * Snapshot the previous frame's matrices, detect cuts, and jitter the camera.
   *
   * Must be paired with {@link endSceneDraw} — the camera is left in a modified
   * state in between, and an early return that skips the restore leaves the
   * game rendering through a half-pixel-offset projection forever.
   *
   * @param width  drawing-buffer width the scene is rendered at
   * @param height drawing-buffer height the scene is rendered at
   */
  beginSceneDraw(camera: THREE.PerspectiveCamera, width: number, height: number): void {
    // Previous-frame matrices come from the *end* of the last draw, before this
    // frame's camera update is folded in. `camera.matrixWorldInverse` is kept
    // current by the renderer, but reading it here (rather than in
    // `endSceneDraw`) would sample it after gameplay had already moved the
    // camera, so the snapshot is taken from what was stored last frame.
    camera.updateMatrixWorld();
    camera.updateProjectionMatrix();

    this.#detectCut(camera);

    this.#unjitteredProjection.copy(camera.projectionMatrix);
    this.projectionMatrix.copy(camera.projectionMatrix);

    // Tell VelocityNode to reproject with the unjittered projection. Jitter is
    // a sampling offset, not scene motion; leaving it in would make every pixel
    // report a half-pixel velocity and defeat the neighbourhood clamp.
    velocity.setProjectionMatrix(this.#unjitteredProjection);

    if (this.#jitterEnabled && width > 0 && height > 0) {
      const index = this.frameIndex % this.#sequenceLength;
      const jx = (this.#offsets[index * 2] ?? 0) * this.#jitterScale;
      const jy = (this.#offsets[index * 2 + 1] ?? 0) * this.#jitterScale;
      this.jitter.set(jx, jy);

      this.#savedView = cloneView(camera.view);
      camera.setViewOffset(width, height, jx, jy, width, height);
      this.#jitterApplied = true;
    } else {
      this.jitter.set(0, 0);
      this.#jitterApplied = false;
    }
  }

  /**
   * Undo the jitter and roll the frame forward.
   *
   * Called in a `finally` by {@link module:render/post/PostStack} so a throw
   * inside the scene draw cannot strand a jittered camera.
   */
  endSceneDraw(camera: THREE.PerspectiveCamera): void {
    if (this.#jitterApplied) {
      const saved = this.#savedView;
      if (saved !== null && saved.enabled) {
        camera.setViewOffset(
          saved.fullWidth,
          saved.fullHeight,
          saved.offsetX,
          saved.offsetY,
          saved.width,
          saved.height,
        );
      } else {
        camera.clearViewOffset();
      }
      this.#savedView = null;
      this.#jitterApplied = false;
    }

    velocity.setProjectionMatrix(null);

    this.previousViewMatrix.copy(camera.matrixWorldInverse);
    this.previousProjectionMatrix.copy(this.#unjitteredProjection);
    this.#previousCameraWorld.copy(camera.matrixWorld);
    this.#hasPreviousCamera = true;

    this.frameIndex++;
    this.historyValid = true;
    this.#forceReset = false;
  }

  /** Record the attachments the scene draw wrote into. */
  setTargets(velocityTexture: THREE.Texture | null, depthTexture: THREE.DepthTexture | null): void {
    this.velocityTexture = this.#velocityEnabled ? velocityTexture : null;
    this.depthTexture = depthTexture;
  }

  dispose(): void {
    this.velocityTexture = null;
    this.depthTexture = null;
  }

  /* -- internals --------------------------------------------------------- */

  /**
   * Compare this frame's camera against last frame's.
   *
   * Translation uses a squared distance to avoid a `sqrt`; rotation compares
   * the forward basis vectors with a dot product, which is monotonic in the
   * angle and needs no trig at runtime.
   */
  #detectCut(camera: THREE.PerspectiveCamera): void {
    if (this.#forceReset || !this.#hasPreviousCamera) {
      this.historyValid = false;
      return;
    }

    const now = camera.matrixWorld.elements;
    const was = this.#previousCameraWorld.elements;

    const dx = (now[12] ?? 0) - (was[12] ?? 0);
    const dy = (now[13] ?? 0) - (was[13] ?? 0);
    const dz = (now[14] ?? 0) - (was[14] ?? 0);
    if (dx * dx + dy * dy + dz * dz > this.#cutTranslationSq) {
      this.historyValid = false;
      return;
    }

    // Column 2 of a camera's world matrix is +Z, which for three's convention
    // points *backwards* along the view direction. Either sign works for an
    // angle comparison.
    const forwardDot =
      (now[8] ?? 0) * (was[8] ?? 0) + (now[9] ?? 0) * (was[9] ?? 0) + (now[10] ?? 0) * (was[10] ?? 0);
    this.historyValid = forwardDot >= this.#cutRotationCos;
  }
}

function cloneView(view: CameraView): CameraView {
  if (view === null) return null;
  return {
    enabled: view.enabled,
    fullWidth: view.fullWidth,
    fullHeight: view.fullHeight,
    offsetX: view.offsetX,
    offsetY: view.offsetY,
    width: view.width,
    height: view.height,
  };
}
