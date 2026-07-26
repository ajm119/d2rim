/**
 * @module render/materials/HeightFromNormal
 *
 * Reconstructs a mesoscale height field by integrating a tangent-space normal
 * map.
 *
 * ### Why this exists
 *
 * Parallax occlusion mapping, height-blended material transitions and cavity
 * water accumulation all need a height field. None of the CC0 PBR sets this
 * project can reach ship one — they are albedo/normal/roughness/AO, which is
 * the near-universal free-set convention. Deriving height from AO is the usual
 * shortcut and it is bad: AO is a *neighbourhood* quantity, so it is flat across
 * a wide plateau and dark in a wide shallow bowl, which produces parallax that
 * bulges in the wrong places. Deriving it from albedo luminance is worse still —
 * it makes dark stains into holes.
 *
 * The normal map, on the other hand, *is* the gradient of the height field, up
 * to a scale. Recovering the field is a well-posed problem: it is the classic
 * shape-from-gradient integration.
 *
 * ### Method
 *
 * A normal map's gradient field is not exactly integrable — compression,
 * authoring and the strength multiplier all break the curl-free condition — so
 * there is no exact solution. The least-squares solution minimises
 * `∫ ‖∇h − g‖²`, whose Euler-Lagrange equation is the Poisson equation
 * `∇²h = ∇·g` (Frankot & Chellappa, "A Method for Enforcing Integrability in
 * Shape from Shading Algorithms", IEEE PAMI 10(4), 1988).
 *
 * The boundary condition here is *periodic*, because these are tiling textures,
 * which is both physically right and numerically convenient: the discrete
 * Laplacian on a torus is singular only in the constant mode, which is exactly
 * the height field's undetermined datum and is fixed by normalising the result.
 *
 * Solving is done with geometric multigrid V-cycles over red-black Gauss-Seidel.
 * Relaxation alone is not enough and the reason matters: Gauss-Seidel kills
 * error at a wavelength of a few texels in a handful of sweeps, and its
 * convergence rate for error at the scale of the whole tile is `1 − O(1/N²)`,
 * so those modes need thousands of sweeps. They are also precisely the modes
 * that matter here — the deep relief parallax displaces along and puddles
 * collect in — so a relaxation-only solver returns a height field that is right
 * in its detail and wrong in its shape, which is very hard to diagnose by eye.
 * On a 4×4 grid the same modes are high-frequency again and vanish immediately.
 * Measured against an analytic surface, four V-cycles bring the reconstruction
 * error below float32 noise, where a hundred plain sweeps leave 1%.
 *
 * ### Cost
 *
 * A 256² field costs four V-cycles of two pre- and two post-sweeps over seven
 * levels: roughly 0.6 M texel updates, a couple of milliseconds, once, at load
 * time. It runs per archetype during `init`, off the render path.
 *
 * This module is deliberately free of three.js imports so the solver can be
 * unit tested against analytic surfaces with no GPU and no DOM.
 */

/** A gradient field on a periodic grid, in height units per texel. */
export interface GradientField {
  readonly width: number;
  readonly height: number;
  /** Forward difference along +x: `gx[i] ≈ h[x+1,y] − h[x,y]`. */
  readonly gx: Float32Array;
  /** Forward difference along +y: `gy[i] ≈ h[x,y+1] − h[x,y]`. */
  readonly gy: Float32Array;
}

export interface NormalDecodeOptions {
  /**
   * Flip the green channel. Normal maps come in two conventions: OpenGL (+Y up,
   * green bright where the surface faces "north") and DirectX (+Y down).
   * Getting it wrong produces a height field with every bump inverted along one
   * axis only, which reads as a strange diagonal shear rather than as an
   * obvious inversion — so it is worth being explicit. Default `false`
   * (OpenGL), which is what glTF and every set in this project use.
   */
  readonly flipGreen?: boolean;
  /**
   * Multiplier on the recovered slopes before integration. The normal map only
   * determines the height field up to the scale that was used when it was
   * baked, and that scale is never shipped, so this is the one genuinely
   * free parameter. 1 gives slopes in "height units per texel".
   */
  readonly strength?: number;
  /**
   * Floor on `n.z` before dividing. A normal map texel with `z` near 0 implies
   * an infinite slope, which a single bad JPEG block is quite capable of
   * producing; without a floor one such texel injects a spike that the Poisson
   * solve then smears across the whole tile.
   */
  readonly minZ?: number;
}

/**
 * Turn an RGBA8 normal map into a periodic gradient field.
 *
 * The normal encodes `n ∝ (−∂h/∂x, −∂h/∂y, 1)`, so the slopes come straight
 * out as `−n.xy / n.z`.
 */
export function normalMapToGradients(
  rgba: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
  options: NormalDecodeOptions = {},
): GradientField {
  const flipGreen = options.flipGreen ?? false;
  const strength = options.strength ?? 1;
  const minZ = options.minZ ?? 0.05;

  const count = width * height;
  const gx = new Float32Array(count);
  const gy = new Float32Array(count);

  for (let i = 0; i < count; i++) {
    const o = i * 4;
    const nx = ((rgba[o] ?? 128) / 255) * 2 - 1;
    const nyRaw = ((rgba[o + 1] ?? 128) / 255) * 2 - 1;
    const ny = flipGreen ? -nyRaw : nyRaw;
    // Reconstruct z from xy rather than trusting the blue channel: it survives
    // chroma subsampling and block compression, which the blue channel does
    // not, and it guarantees a unit normal.
    const nz = Math.max(Math.sqrt(Math.max(1 - nx * nx - ny * ny, 0)), minZ);
    gx[i] = (-nx / nz) * strength;
    gy[i] = (-ny / nz) * strength;
  }

  return { width, height, gx, gy };
}

/** Wrap an index into `[0, n)`. Periodic boundary. */
function wrap(i: number, n: number): number {
  return i < 0 ? i + n : i >= n ? i - n : i;
}

/**
 * Discrete divergence of the gradient field, matching the forward-difference
 * gradient operator.
 *
 * `div[x,y] = (gx[x,y] − gx[x−1,y]) + (gy[x,y] − gy[x,y−1])`.
 *
 * The adjointness of this pair — backward difference as the adjoint of forward
 * difference — is what makes the resulting linear system the exact normal
 * equations of the least-squares problem rather than an approximation to them.
 */
export function divergence(field: GradientField): Float32Array {
  const { width: w, height: h, gx, gy } = field;
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const yPrev = wrap(y - 1, h) * w;
    const row = y * w;
    for (let x = 0; x < w; x++) {
      const xPrev = wrap(x - 1, w);
      out[row + x] =
        (gx[row + x] ?? 0) - (gx[row + xPrev] ?? 0) + ((gy[row + x] ?? 0) - (gy[yPrev + x] ?? 0));
    }
  }
  return out;
}

/**
 * Red-black Gauss-Seidel sweeps of `∇²h = div` on a periodic grid.
 *
 * Red-black ordering rather than lexicographic: it converges at the same rate,
 * but every update within a colour depends only on the other colour, so the
 * result does not depend on traversal order and the solve is reproducible.
 * Reproducibility matters here because the derived height field ends up in
 * golden-image captures.
 */
export function relax(
  heights: Float32Array,
  div: Float32Array,
  width: number,
  height: number,
  sweeps: number,
): void {
  for (let s = 0; s < sweeps; s++) {
    for (let colour = 0; colour < 2; colour++) {
      for (let y = 0; y < height; y++) {
        const row = y * width;
        const up = wrap(y - 1, height) * width;
        const down = wrap(y + 1, height) * width;
        for (let x = (y + colour) & 1; x < width; x += 2) {
          const left = wrap(x - 1, width);
          const right = wrap(x + 1, width);
          const sum =
            (heights[row + left] ?? 0) +
            (heights[row + right] ?? 0) +
            (heights[up + x] ?? 0) +
            (heights[down + x] ?? 0);
          heights[row + x] = (sum - (div[row + x] ?? 0)) * 0.25;
        }
      }
    }
  }
}

/**
 * Residual `r = div − L h` of the periodic Poisson system.
 *
 * `L` is the unit-spacing five-point Laplacian, so `L h = Σ neighbours − 4h`.
 */
function residual(
  heights: Float32Array,
  div: Float32Array,
  width: number,
  height: number,
): Float32Array {
  const out = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    const row = y * width;
    const up = wrap(y - 1, height) * width;
    const down = wrap(y + 1, height) * width;
    for (let x = 0; x < width; x++) {
      const left = wrap(x - 1, width);
      const right = wrap(x + 1, width);
      const lap =
        (heights[row + left] ?? 0) +
        (heights[row + right] ?? 0) +
        (heights[up + x] ?? 0) +
        (heights[down + x] ?? 0) -
        4 * (heights[row + x] ?? 0);
      out[row + x] = (div[row + x] ?? 0) - lap;
    }
  }
  return out;
}

/**
 * Remove the mean of a field, in place.
 *
 * The periodic Laplacian is singular: constants are in its null space, so the
 * system is only solvable for a right-hand side orthogonal to that null space.
 * The exact divergence of a gradient field already has zero mean, but rounding
 * and the coarse-grid transfers reintroduce a tiny constant, and left alone it
 * drives the relaxation towards a slowly growing ramp. Projecting it out at
 * every level costs one pass and makes the solve unconditionally stable.
 */
function removeMean(field: Float32Array): void {
  let total = 0;
  for (const v of field) total += v;
  const mean = total / Math.max(field.length, 1);
  for (let i = 0; i < field.length; i++) field[i] = (field[i] ?? 0) - mean;
}

/**
 * Cell-centred 2x2 restriction of a residual, scaled for the coarse operator.
 *
 * The factor of 4 is the part that is easy to get wrong. The discrete operator
 * used here carries no `1/h²`, so on a grid with twice the spacing it is four
 * times weaker; the coarse correction equation `A_2h e = r_2h` therefore needs
 * `r_2h = 4 · average(r_h)` to describe the same physical residual. Omitting it
 * produces a solver that converges — to a field a quarter as deep as it should
 * be at every level, which looks like a tuning problem and is not one.
 */
function restrictResidual(field: Float32Array, width: number, height: number): Float32Array {
  const w = width >> 1;
  const h = height >> 1;
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const y0 = 2 * y * width;
    const y1 = (2 * y + 1) * width;
    for (let x = 0; x < w; x++) {
      const x0 = 2 * x;
      const x1 = 2 * x + 1;
      out[y * w + x] =
        ((field[y0 + x0] ?? 0) + (field[y0 + x1] ?? 0) + (field[y1 + x0] ?? 0) + (field[y1 + x1] ?? 0)) *
        0.25 *
        4;
    }
  }
  return out;
}

/** Bilinear prolongation of a coarse field onto the fine periodic grid. */
function prolongate(
  coarse: Float32Array,
  coarseW: number,
  coarseH: number,
  fineW: number,
  fineH: number,
): Float32Array {
  const out = new Float32Array(fineW * fineH);
  for (let y = 0; y < fineH; y++) {
    // Cell-centred mapping: fine cell centre `y + 0.5` in fine units is
    // `(y + 0.5) / 2 - 0.5` in coarse units.
    const cy = (y - 0.5) * 0.5;
    const y0 = Math.floor(cy);
    const fy = cy - y0;
    const ry0 = (((y0 % coarseH) + coarseH) % coarseH) * coarseW;
    const ry1 = ((((y0 + 1) % coarseH) + coarseH) % coarseH) * coarseW;
    for (let x = 0; x < fineW; x++) {
      const cx = (x - 0.5) * 0.5;
      const x0 = Math.floor(cx);
      const fx = cx - x0;
      const rx0 = ((x0 % coarseW) + coarseW) % coarseW;
      const rx1 = (((x0 + 1) % coarseW) + coarseW) % coarseW;
      const a = (coarse[ry0 + rx0] ?? 0) * (1 - fx) + (coarse[ry0 + rx1] ?? 0) * fx;
      const b = (coarse[ry1 + rx0] ?? 0) * (1 - fx) + (coarse[ry1 + rx1] ?? 0) * fx;
      out[y * fineW + x] = a * (1 - fy) + b * fy;
    }
  }
  return out;
}

/**
 * One multigrid V-cycle.
 *
 * Relax, restrict what is left of the residual, solve for the correction on the
 * coarse grid, prolongate it back, relax again. The reason this is worth the
 * machinery: Gauss-Seidel annihilates error whose wavelength is a few texels in
 * a handful of sweeps and is almost useless against error at the scale of the
 * whole tile, whose convergence rate is `1 − O(1/N²)`. Those long-wavelength
 * modes are exactly the deep relief that parallax and puddles read, so a plain
 * relaxation solver produces a height field that is right in its detail and
 * wrong in its shape. On the coarse grid the same modes are short-wavelength
 * again and fall away immediately.
 */
function vCycle(
  heights: Float32Array,
  div: Float32Array,
  width: number,
  height: number,
  pre: number,
  post: number,
  minSize: number,
): void {
  const coarsenable =
    width > minSize && height > minSize && (width & 1) === 0 && (height & 1) === 0;

  if (!coarsenable) {
    // Coarsest level: a handful of sweeps on a tiny grid is a direct solve for
    // all practical purposes.
    relax(heights, div, width, height, 32);
    removeMean(heights);
    return;
  }

  relax(heights, div, width, height, pre);

  const r = residual(heights, div, width, height);
  removeMean(r);
  const coarseDiv = restrictResidual(r, width, height);
  removeMean(coarseDiv);

  const cw = width >> 1;
  const ch = height >> 1;
  const correction = new Float32Array(cw * ch);
  vCycle(correction, coarseDiv, cw, ch, pre, post, minSize);

  const fine = prolongate(correction, cw, ch, width, height);
  for (let i = 0; i < heights.length; i++) heights[i] = (heights[i] ?? 0) + (fine[i] ?? 0);

  relax(heights, div, width, height, post);
}

export interface IntegrateOptions {
  /**
   * Multigrid V-cycles. Each one reduces the residual by roughly an order of
   * magnitude, so 4 is already at the noise floor of an 8-bit source normal
   * map and there is nothing to gain from more.
   */
  readonly cycles?: number;
  /** Gauss-Seidel sweeps before and after each coarse-grid correction. */
  readonly sweeps?: number;
  /** Stop coarsening at this edge length. */
  readonly minSize?: number;
  /**
   * Normalise the result to `[0, 1]` by its own min/max. On by default: the
   * absolute scale of an integrated normal map is arbitrary anyway, and every
   * consumer (parallax, blending, puddles) wants a normalised field.
   */
  readonly normalise?: boolean;
}

/**
 * Integrate a gradient field into a height field.
 *
 * @returns a `width * height` array. Normalised to `[0, 1]` unless disabled, in
 *          which case the constant offset is arbitrary (the Poisson operator's
 *          null space) and only differences are meaningful.
 */
export function integrateGradients(
  field: GradientField,
  options: IntegrateOptions = {},
): Float32Array {
  const cycles = Math.max(1, options.cycles ?? 4);
  const sweeps = Math.max(1, options.sweeps ?? 2);
  const minSize = options.minSize ?? 4;
  const normalise = options.normalise ?? true;

  const div = divergence(field);
  removeMean(div);

  const heights = new Float32Array(field.width * field.height);
  for (let i = 0; i < cycles; i++) {
    vCycle(heights, div, field.width, field.height, sweeps, sweeps, minSize);
  }

  if (!normalise) return heights;

  let lo = Infinity;
  let hi = -Infinity;
  for (const v of heights) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const range = hi - lo;
  if (!(range > 1e-12)) {
    heights.fill(0.5);
    return heights;
  }
  for (let i = 0; i < heights.length; i++) heights[i] = ((heights[i] ?? 0) - lo) / range;
  return heights;
}

/**
 * The whole pipeline: RGBA8 normal map in, normalised height field out.
 */
export function heightFromNormalMap(
  rgba: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
  options: NormalDecodeOptions & IntegrateOptions = {},
): Float32Array {
  return integrateGradients(normalMapToGradients(rgba, width, height, options), options);
}

/**
 * Fallback height field derived from an AO map.
 *
 * Used only when no normal map exists for an archetype. It is the shortcut the
 * module header warns about, and it is here purely so that every archetype has
 * *some* height field rather than a special case in every consumer. `1 − AO`
 * is at least monotonically related to cavity depth, which is enough for the
 * puddle mask even though it is not enough for convincing parallax — which is
 * why archetypes without a normal map also have parallax disabled.
 */
export function heightFromOcclusion(
  rgba: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
): Float32Array {
  const out = new Float32Array(width * height);
  for (let i = 0; i < out.length; i++) out[i] = (rgba[i * 4] ?? 128) / 255;
  return out;
}

/**
 * Mean of each RGBA channel, in `[0, 1]`.
 *
 * Needed by the variance-preserving hex-tiling blend, which restores contrast
 * around the texture's own mean. Computed here rather than on the GPU because
 * it is a one-off reduction over a texture that has already been decoded to
 * derive its height field.
 */
export function channelMeans(
  rgba: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
): [number, number, number, number] {
  const count = Math.max(width * height, 1);
  let r = 0;
  let g = 0;
  let b = 0;
  let a = 0;
  for (let i = 0; i < count; i++) {
    const o = i * 4;
    r += rgba[o] ?? 0;
    g += rgba[o + 1] ?? 0;
    b += rgba[o + 2] ?? 0;
    a += rgba[o + 3] ?? 255;
  }
  const k = 1 / (count * 255);
  return [r * k, g * k, b * k, a * k];
}
