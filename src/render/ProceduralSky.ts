/**
 * @module render/ProceduralSky
 *
 * Generates a high-dynamic-range equirectangular sky entirely in code, with no
 * asset download.
 *
 * The output serves two purposes at once:
 *
 * 1. **Background** — what the camera sees looking at the horizon.
 * 2. **Image-based lighting** — assigned to `scene.environment`, three.js runs
 *    it through PMREM to produce the pre-filtered radiance used for ambient
 *    diffuse and glossy reflections.
 *
 * Both jobs demand real HDR: the sun disc has to be hundreds of times brighter
 * than the sky around it, or metals reflect a dull grey smear instead of a hot
 * highlight, and ACES tone mapping has nothing to roll off. The texture is
 * therefore `HalfFloatType` in linear space, with radiance values well above 1.
 *
 * The model is a cheap analytic approximation rather than a physical sky:
 * a Rayleigh-ish zenith-to-horizon gradient, a Mie forward-scattering lobe
 * around the sun, a bright sun disc, a dark ground hemisphere with a bounce
 * term, and a thin band of fbm cirrus. It costs about 12 ms to build at
 * 1024x512 and is deterministic, so golden-image tests are unaffected.
 */

import * as THREE from 'three/webgpu';
import { equirectUV, texture as textureNode, vec2 } from 'three/tsl';

/**
 * `Scene.backgroundNode` is read by the node renderer — three's own
 * `RendererParameters` declares it — but @types/three r185 omits it from the
 * `Scene` class declaration. A narrow structural interface plus one cast is the
 * honest way to bridge that gap without reaching for `any`.
 */
interface SceneWithBackgroundNode {
  backgroundNode: THREE.Node | null;
}

export interface SkyOptions {
  /**
   * Equirect width in texels; height is half. Default 1024.
   *
   * The sky is a smooth gradient with one small disc, so 1024x512 is ample once
   * `backgroundBlurriness` is 0. Doubling it mainly costs PMREM time at boot.
   */
  width?: number;
  /** Sun elevation above the horizon, in degrees. Default 26. */
  sunElevation?: number;
  /** Sun compass azimuth, in degrees. Default 38. */
  sunAzimuth?: number;
  /** Overall radiance multiplier. Default 1. */
  intensity?: number;
  /** Cirrus cloud coverage in `[0, 1]`. Default 0.35. */
  cloudiness?: number;
  /** Deterministic noise seed. Default 1337. */
  seed?: number;
}

export interface ProceduralSky {
  /** Equirectangular HDR texture. */
  readonly texture: THREE.DataTexture;
  /** Unit vector pointing from the origin toward the sun. */
  readonly sunDirection: THREE.Vector3;
  /** Linear-space colour of direct sunlight, normalised to peak 1. */
  readonly sunColor: THREE.Color;
  /**
   * Install this sky as both the visible background and the IBL source.
   *
   * The two jobs take deliberately different routes:
   *
   * - **Background** goes through `scene.backgroundNode`, sampling the
   *   equirect directly in TSL. Assigning `scene.background` instead would
   *   route it through `CubeMapNode`, which re-projects the texture into a cube
   *   render target and loses both resolution and highlight range — the sun
   *   disc smears and the sky flattens to grey.
   * - **Environment** goes through `scene.environment`, which is exactly where
   *   PMREM pre-filtering belongs: glossy reflections and ambient diffuse need
   *   the blurred mip chain.
   */
  applyToScene(scene: THREE.Scene): void;
  dispose(): void;
}

/** Deterministic 32-bit hash -> `[0, 1)`. Used instead of `Math.random`. */
function hash2(x: number, y: number, seed: number): number {
  let h = (x * 374761393 + y * 668265263 + seed * 2147483647) | 0;
  h = (h ^ (h >>> 13)) * 1274126177;
  h = h ^ (h >>> 16);
  return (h >>> 0) / 4294967296;
}

function smoothstep(t: number): number {
  return t * t * (3 - 2 * t);
}

/**
 * Value noise on a lattice that wraps every `period` cells, so the result tiles
 * seamlessly around the equirect's horizontal seam.
 */
function valueNoise(x: number, y: number, period: number, seed: number): number {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const xf = x - xi;
  const yf = y - yi;

  const wrap = (v: number): number => ((v % period) + period) % period;
  const x0 = wrap(xi);
  const x1 = wrap(xi + 1);
  const y0 = wrap(yi);
  const y1 = wrap(yi + 1);

  const u = smoothstep(xf);
  const v = smoothstep(yf);

  const n00 = hash2(x0, y0, seed);
  const n10 = hash2(x1, y0, seed);
  const n01 = hash2(x0, y1, seed);
  const n11 = hash2(x1, y1, seed);

  return (n00 * (1 - u) + n10 * u) * (1 - v) + (n01 * (1 - u) + n11 * u) * v;
}

/** Fractal Brownian motion over {@link valueNoise}, normalised to `[0, 1]`. */
function fbm(x: number, y: number, period: number, seed: number, octaves = 4): number {
  let sum = 0;
  let amplitude = 0.5;
  let total = 0;
  let freq = 1;
  for (let i = 0; i < octaves; i++) {
    sum += valueNoise(x * freq, y * freq, period * freq, seed + i * 71) * amplitude;
    total += amplitude;
    amplitude *= 0.5;
    freq *= 2;
  }
  return sum / total;
}

/**
 * Build the sky.
 *
 * Radiance constants below are hand-tuned against ACES tone mapping at exposure
 * 1.0: the sky sits around 0.6-2.0, the sun glow reaches ~12, and the disc hits
 * ~320 so that polished metal produces a genuine specular hotspot without the
 * open sky clipping to white.
 */
export function createProceduralSky(options: SkyOptions = {}): ProceduralSky {
  const width = options.width ?? 1024;
  const height = width >> 1;
  const intensity = options.intensity ?? 1;
  const cloudiness = options.cloudiness ?? 0.35;
  const seed = options.seed ?? 1337;

  const elevation = THREE.MathUtils.degToRad(options.sunElevation ?? 26);
  const azimuth = THREE.MathUtils.degToRad(options.sunAzimuth ?? 38);
  const sunDirection = new THREE.Vector3(
    Math.cos(elevation) * Math.sin(azimuth),
    Math.sin(elevation),
    Math.cos(elevation) * Math.cos(azimuth),
  ).normalize();

  // Late-afternoon palette: cool zenith, warm horizon haze, warm sun.
  const zenith = new THREE.Color(0.10, 0.26, 0.62);
  const horizon = new THREE.Color(0.86, 0.56, 0.34);
  const glow = new THREE.Color(1.0, 0.62, 0.30);
  const sunRadiance = new THREE.Color(1.0, 0.87, 0.70);
  const ground = new THREE.Color(0.055, 0.048, 0.040);
  const bounce = new THREE.Color(0.24, 0.19, 0.14);

  const data = new Uint16Array(width * height * 4);
  const toHalf = THREE.DataUtils.toHalfFloat;

  // cos of the sun's angular radius. The real sun is ~0.27 deg; widened to 1.4
  // so it survives PMREM filtering and reads as a disc rather than a firefly.
  const cosSunRadius = Math.cos(THREE.MathUtils.degToRad(1.4));

  for (let y = 0; y < height; y++) {
    // Equirect: v maps to polar angle from +Y down to -Y.
    const v = (y + 0.5) / height;
    const phi = v * Math.PI;
    const sinPhi = Math.sin(phi);
    const dirY = Math.cos(phi);

    for (let x = 0; x < width; x++) {
      const u = (x + 0.5) / width;
      const theta = u * Math.PI * 2;
      const dirX = sinPhi * Math.sin(theta);
      const dirZ = sinPhi * Math.cos(theta);

      const cosSun = dirX * sunDirection.x + dirY * sunDirection.y + dirZ * sunDirection.z;
      const up = Math.max(0, dirY);

      let r: number;
      let g: number;
      let b: number;

      if (dirY >= 0) {
        // Sky hemisphere. A steep falloff pins the warm haze to the horizon
        // band; anything shallower washes the blue out of the whole upper
        // hemisphere and the result reads as overcast rather than late sun.
        const haze = Math.pow(1 - up, 7);
        r = zenith.r + (horizon.r - zenith.r) * haze;
        g = zenith.g + (horizon.g - zenith.g) * haze;
        b = zenith.b + (horizon.b - zenith.b) * haze;

        // Broad Mie forward-scatter lobe, plus a tighter inner lobe.
        const forward = Math.max(0, cosSun);
        const mie = Math.pow(forward, 6) * 0.30 + Math.pow(forward, 120) * 3.2;
        r += glow.r * mie;
        g += glow.g * mie;
        b += glow.b * mie;

        // Cirrus: only in the upper half, fading out at the zenith and horizon
        // so the band never looks like a hard edge.
        if (cloudiness > 0) {
          const band = Math.sin(Math.min(1, up * 2.4) * Math.PI);
          if (band > 0.001) {
            const n = fbm(u * 14, v * 14, 14, seed, 5);
            const cloud = Math.max(0, n - (1 - cloudiness) * 0.85) * band * 1.6;
            // Clouds are lit by the same warm key, so they brighten rather
            // than grey out the sky.
            r += cloud * 1.15;
            g += cloud * 1.02;
            b += cloud * 0.92;
          }
        }

        // The disc itself, added last so nothing dilutes it.
        if (cosSun > cosSunRadius) {
          r += sunRadiance.r * 320;
          g += sunRadiance.g * 320;
          b += sunRadiance.b * 320;
        }
      } else {
        // Ground hemisphere. Without a bounce term, everything facing downward
        // goes black and the render reads as "floating in a void".
        const t = Math.pow(1 + dirY, 6); // 1 at the horizon, 0 straight down
        r = ground.r + (bounce.r - ground.r) * t;
        g = ground.g + (bounce.g - ground.g) * t;
        b = ground.b + (bounce.b - ground.b) * t;
      }

      const i = (y * width + x) * 4;
      data[i] = toHalf(r * intensity);
      data[i + 1] = toHalf(g * intensity);
      data[i + 2] = toHalf(b * intensity);
      data[i + 3] = toHalf(1);
    }
  }

  const texture = new THREE.DataTexture(data, width, height, THREE.RGBAFormat, THREE.HalfFloatType);
  texture.mapping = THREE.EquirectangularReflectionMapping;
  // The data is already linear radiance; tagging it sRGB would double-decode it.
  texture.colorSpace = THREE.LinearSRGBColorSpace;
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearFilter;
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;

  return {
    texture,
    sunDirection,
    sunColor: sunRadiance.clone(),

    applyToScene(scene: THREE.Scene): void {
      // `equirectUV()` computes `v = asin(dir.y) / PI + 0.5`, putting the
      // zenith at v = 1. This texture is a `DataTexture` with `flipY = false`,
      // so its row 0 — the zenith — samples at v = 0. Left alone the sky
      // renders upside down: the dark ground hemisphere fills the top of the
      // frame with a hard bright seam at the horizon.
      //
      // The flip is applied here, to the background sampler only. The texture
      // data itself is the orientation PMREM already interprets correctly for
      // `scene.environment`, and changing it would fix the background by
      // breaking every reflection.
      const uv = equirectUV().mul(vec2(1, -1)).add(vec2(0, 1));
      (scene as unknown as SceneWithBackgroundNode).backgroundNode = textureNode(texture, uv);
      scene.environment = texture;
    },

    dispose(): void {
      texture.dispose();
    },
  };
}
