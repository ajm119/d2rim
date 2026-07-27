/**
 * @module render/MemoryReport
 *
 * Measures what the GPU is actually holding, and says it in megabytes.
 *
 * ## Why this module exists
 *
 * The deployed build was crashing Chromium with "Error code: 5" — an
 * out-of-memory renderer crash, not a driver fault. Nothing in the project
 * could answer the first question you have to ask in that situation: *how much
 * memory are we using, and on what?* Frame time was instrumented; bytes were
 * not. So the failure was invisible in every headless run, because a headless
 * software rasteriser has the whole of system RAM to fall over into and simply
 * never hits the wall a real GPU hits.
 *
 * This module is the answer to that. It is deliberately analytic rather than
 * empirical: it counts bytes from image dimensions and texture formats instead
 * of asking the driver, because there is no portable API that will tell you and
 * because an analytic count works identically under SwiftShader and on a real
 * GPU. That makes it something a test can assert on.
 *
 * ## Render targets are tracked, not discovered
 *
 * Textures and geometry are reachable by walking the scene graph. Render
 * targets are not — they are held in private fields of a dozen passes and there
 * is no global registry in three. So allocation sites call
 * {@link trackRenderTarget}, and the ledger removes them again by listening for
 * three's own `dispose` event. That means a target only has to be registered
 * once, at birth, and a pass that disposes correctly is accounted for
 * automatically. A pass that *leaks* keeps showing up in the report, which is
 * exactly the behaviour you want from a leak detector.
 */

import * as THREE from 'three/webgpu';

/* -------------------------------------------------------------------------- */
/* Render-target ledger                                                        */
/* -------------------------------------------------------------------------- */

const liveTargets = new Set<THREE.RenderTarget>();

/**
 * Register a render target so it appears in {@link collectMemoryReport}.
 *
 * Self-unregistering: three dispatches a `dispose` event from
 * `RenderTarget.dispose()`, so callers never have to pair this with an untrack.
 */
export function trackRenderTarget<T extends THREE.RenderTarget>(target: T, label?: string): T {
  if (liveTargets.has(target)) return target;
  if (label !== undefined && target.texture.name === '') target.texture.name = label;
  liveTargets.add(target);
  target.addEventListener('dispose', function onDispose() {
    liveTargets.delete(target);
    target.removeEventListener('dispose', onDispose);
  });
  return target;
}

/** Every render target currently alive. Diagnostics only. */
export function trackedRenderTargets(): readonly THREE.RenderTarget[] {
  return [...liveTargets];
}

/** Drop every tracked target without disposing. For test isolation only. */
export function resetRenderTargetLedger(): void {
  liveTargets.clear();
}

/* -------------------------------------------------------------------------- */
/* Byte accounting                                                             */
/* -------------------------------------------------------------------------- */

/** Bytes per pixel implied by a texture's type and format. */
export function bytesPerPixel(type: THREE.TextureDataType, format: THREE.AnyPixelFormat): number {
  const channels =
    format === THREE.RedFormat
      ? 1
      : format === THREE.RGFormat
        ? 2
        : format === THREE.RGBFormat
          ? 3
          : 4;
  const size =
    type === THREE.FloatType || type === THREE.UnsignedIntType || type === THREE.IntType
      ? 4
      : type === THREE.HalfFloatType || type === THREE.UnsignedShortType
        ? 2
        : 1;
  return channels * size;
}

/**
 * GPU bytes for one texture.
 *
 * A compressed texture reports the true size of its transcoded blocks, because
 * that is what the driver uploads and keeps — the entire point of shipping
 * KTX2. An uncompressed one is `w * h * bpp`, plus a third again when three
 * generates a mip chain.
 */
export function textureBytes(texture: THREE.Texture): number {
  const compressed = texture as THREE.Texture & {
    isCompressedTexture?: boolean;
    mipmaps?: ReadonlyArray<{ data?: { byteLength?: number } }> | null;
  };

  if (compressed.isCompressedTexture === true && compressed.mipmaps != null) {
    let total = 0;
    for (const level of compressed.mipmaps) total += level.data?.byteLength ?? 0;
    return total;
  }

  const image = texture.image as { width?: number; height?: number; depth?: number } | null;
  const width = image?.width ?? 0;
  const height = image?.height ?? 0;
  if (width === 0 || height === 0) return 0;

  const depth = image?.depth ?? 1;
  const faces = (texture as THREE.Texture & { isCubeTexture?: boolean }).isCubeTexture === true
    ? 6
    : 1;
  const mipFactor = texture.generateMipmaps ? 4 / 3 : 1;
  return Math.round(
    width * height * Math.max(1, depth) * faces * bytesPerPixel(texture.type, texture.format) * mipFactor,
  );
}

/** GPU bytes for one geometry: every attribute plus the index buffer. */
export function geometryBytes(geometry: THREE.BufferGeometry): number {
  let total = 0;
  for (const attribute of Object.values(geometry.attributes)) {
    const array = (attribute as THREE.BufferAttribute).array as ArrayLike<number> & {
      BYTES_PER_ELEMENT?: number;
    };
    total += array.length * (array.BYTES_PER_ELEMENT ?? 4);
  }
  const index = geometry.index;
  if (index !== null) {
    const array = index.array as ArrayLike<number> & { BYTES_PER_ELEMENT?: number };
    total += array.length * (array.BYTES_PER_ELEMENT ?? 4);
  }
  return total;
}

/* -------------------------------------------------------------------------- */
/* Report                                                                      */
/* -------------------------------------------------------------------------- */

export interface TextureBreakdown {
  readonly name: string;
  readonly width: number;
  readonly height: number;
  readonly bytes: number;
  readonly compressed: boolean;
}

export interface MemoryReport {
  /** Decoded/resident texture bytes reachable from the scene. */
  readonly textureBytes: number;
  readonly textureCount: number;
  /** How many of those are GPU-compressed (KTX2/Basis). */
  readonly compressedTextureCount: number;
  /** Bytes held by tracked render targets. */
  readonly renderTargetBytes: number;
  readonly renderTargetCount: number;
  /** Vertex and index buffer bytes reachable from the scene. */
  readonly geometryBytes: number;
  readonly geometryCount: number;
  readonly meshCount: number;
  readonly triangles: number;
  readonly drawCalls: number;
  /**
   * Bytes the `AssetManager` holds resident, when it was supplied.
   *
   * Overlaps `textureBytes` for anything bound through a plain material slot,
   * and covers the TSL-bound material set that the scene walk cannot see.
   */
  readonly residentAssetBytes: number;
  readonly residentAssetCount: number;
  /**
   * The number to watch. Render targets and geometry, plus whichever of the
   * two texture measurements is larger — they overlap, so adding them would
   * double-count, and the larger is the safer estimate.
   */
  readonly totalBytes: number;
  /** Largest textures first. Truncated by `topTextures`. */
  readonly largest: readonly TextureBreakdown[];
}

export interface MemoryReportOptions {
  /** How many entries to include in `largest`. Default 12. */
  readonly topTextures?: number;
  /** Renderer, for draw-call and triangle counters. */
  readonly renderer?: { info?: { render?: { drawCalls?: number; triangles?: number } } } | null;
  /**
   * The asset cache, for the textures a scene walk provably cannot find.
   *
   * This is not belt-and-braces, it is load-bearing. `MaterialLibrary` binds
   * every terrain and prop texture through TSL nodes — `material.colorNode`,
   * `normalNode` and friends — rather than through `material.map`. A traversal
   * that looks for `THREE.Texture` in a material's own properties therefore
   * finds *none of them*, and the first version of this report confidently
   * declared 117 MB for a scene whose material set alone was the thing under
   * investigation. Reading the cache's own byte count closes that gap.
   */
  readonly assets?: { stats(): { bytes: number; entries: number } } | null;
}

/**
 * Walk a scene and total what it costs.
 *
 * Textures and geometries are de-duplicated by identity, because a scene where
 * two hundred rocks share one albedo map costs one albedo map, and a report
 * that says otherwise sends you optimising the wrong thing.
 */
export function collectMemoryReport(
  scene: THREE.Object3D,
  options: MemoryReportOptions = {},
): MemoryReport {
  const seenTextures = new Set<THREE.Texture>();
  const seenGeometries = new Set<THREE.BufferGeometry>();
  const breakdown: TextureBreakdown[] = [];

  let texBytes = 0;
  let compressedCount = 0;
  let geoBytes = 0;
  let meshCount = 0;
  let triangles = 0;

  const addTexture = (texture: THREE.Texture | null | undefined): void => {
    if (!(texture instanceof THREE.Texture) || seenTextures.has(texture)) return;
    seenTextures.add(texture);
    const bytes = textureBytes(texture);
    texBytes += bytes;
    const compressed =
      (texture as THREE.Texture & { isCompressedTexture?: boolean }).isCompressedTexture === true;
    if (compressed) compressedCount++;
    const image = texture.image as { width?: number; height?: number } | null;
    breakdown.push({
      name: texture.name !== '' ? texture.name : (texture.source?.uuid ?? texture.uuid).slice(0, 8),
      width: image?.width ?? 0,
      height: image?.height ?? 0,
      bytes,
      compressed,
    });
  };

  // `scene.environment` and `scene.background` are the two textures that are
  // never reachable through a material and are routinely the largest in the
  // whole frame, so they are checked explicitly rather than traversed to.
  const sceneLike = scene as THREE.Object3D & {
    environment?: THREE.Texture | null;
    background?: THREE.Texture | THREE.Color | null;
  };
  addTexture(sceneLike.environment);
  if (sceneLike.background instanceof THREE.Texture) addTexture(sceneLike.background);

  scene.traverse((object) => {
    const mesh = object as THREE.Mesh;
    const geometry = mesh.geometry;
    if (geometry instanceof THREE.BufferGeometry) {
      meshCount++;
      if (!seenGeometries.has(geometry)) {
        seenGeometries.add(geometry);
        geoBytes += geometryBytes(geometry);
      }
      const index = geometry.index;
      const count = index !== null ? index.count : (geometry.attributes.position?.count ?? 0);
      const instances = (mesh as THREE.InstancedMesh).isInstancedMesh
        ? (mesh as THREE.InstancedMesh).count
        : 1;
      triangles += (count / 3) * instances;
    }

    const material = (mesh as Partial<THREE.Mesh>).material;
    if (material === undefined) return;
    for (const entry of Array.isArray(material) ? material : [material]) {
      for (const value of Object.values(entry as unknown as Record<string, unknown>)) {
        if (value instanceof THREE.Texture) addTexture(value);
      }
    }
  });

  for (const target of liveTargets) {
    // A target's own texture is not reachable from the scene graph, so it must
    // not be double-counted here — it is counted in the render-target total.
    seenTextures.add(target.texture);
  }

  let rtBytes = 0;
  for (const target of liveTargets) {
    const textures = (target as THREE.RenderTarget & { textures?: THREE.Texture[] }).textures;
    const attachments = textures !== undefined && textures.length > 0 ? textures : [target.texture];
    for (const texture of attachments) {
      rtBytes +=
        target.width *
        target.height *
        Math.max(1, target.depth ?? 1) *
        bytesPerPixel(texture.type, texture.format);
    }
    if (target.depthBuffer) rtBytes += target.width * target.height * 4;
  }

  breakdown.sort((a, b) => b.bytes - a.bytes);

  const render = options.renderer?.info?.render;
  const assetStats = options.assets?.stats();
  const residentAssetBytes = assetStats?.bytes ?? 0;

  return {
    textureBytes: texBytes,
    textureCount: breakdown.length,
    compressedTextureCount: compressedCount,
    renderTargetBytes: rtBytes,
    renderTargetCount: liveTargets.size,
    geometryBytes: geoBytes,
    geometryCount: seenGeometries.size,
    meshCount,
    triangles: Math.round(triangles),
    drawCalls: render?.drawCalls ?? 0,
    residentAssetBytes,
    residentAssetCount: assetStats?.entries ?? 0,
    totalBytes: Math.max(texBytes, residentAssetBytes) + rtBytes + geoBytes,
    largest: breakdown.slice(0, options.topTextures ?? 12),
  };
}

const MB = 1024 * 1024;

/** Human-readable multi-line rendering of a report. */
export function formatMemoryReport(report: MemoryReport): string {
  const mb = (bytes: number): string => `${(bytes / MB).toFixed(1)} MB`;
  const lines = [
    '[d2rim] GPU memory report',
    `  textures        ${mb(report.textureBytes).padStart(10)}  ` +
      `(${report.textureCount} unique in scene, ${report.compressedTextureCount} GPU-compressed)`,
    `  resident assets ${mb(report.residentAssetBytes).padStart(10)}  ` +
      `(${report.residentAssetCount} cached; includes TSL-bound material textures)`,
    `  render targets  ${mb(report.renderTargetBytes).padStart(10)}  (${report.renderTargetCount} live)`,
    `  geometry        ${mb(report.geometryBytes).padStart(10)}  (${report.geometryCount} unique)`,
    `  TOTAL           ${mb(report.totalBytes).padStart(10)}`,
    `  scene           ${report.meshCount} meshes, ${report.triangles.toLocaleString()} tris, ` +
      `${report.drawCalls} draw calls`,
  ];
  if (report.largest.length > 0) {
    lines.push('  largest textures:');
    for (const entry of report.largest) {
      lines.push(
        `    ${mb(entry.bytes).padStart(9)}  ${entry.width}x${entry.height}` +
          `${entry.compressed ? ' [compressed]' : ''}  ${entry.name}`,
      );
    }
  }
  return lines.join('\n');
}
