/**
 * @module render/DebugFlags
 *
 * Per-system kill switches, read once from the URL, so that a player on
 * hardware this project does not own can bisect a slow frame in two minutes.
 *
 * ## Why this exists
 *
 * The overlay can now report honest frame times, and on the machine that
 * matters they read:
 *
 * ```
 * fps 1.4  p50 730.4 ms  p95 895.5 ms
 * time update 29.4 ms  render 66.4 ms  gpu …
 * draws 574  tris 389,383  buffer 1.90 Mpx
 * ```
 *
 * 29.4 + 66.4 = 95.8 ms of CPU against a 730 ms frame. Six hundred and thirty
 * milliseconds a frame are spent somewhere no CPU timer can see — which, in a
 * pipeline where `render()` returns as soon as commands are queued, means the
 * browser blocking on the GPU before it will grant the next animation frame.
 * 574 draws and 389k triangles over 1.9 Mpx is a scene a MacBook Air M4 should
 * present at 60 fps, so *something specific* is pathological and the scene as a
 * whole is not.
 *
 * There is no GPU in this project's CI — it renders on SwiftShader across four
 * cores — so that "something" cannot be measured here. It can only be measured
 * there, by the player, and the only way to measure it there is to be able to
 * switch each candidate off independently and watch the number move. That is
 * what this module is: a set of composable URL flags, each of which genuinely
 * *removes* a system rather than turning its intensity down.
 *
 * ## The distinction that matters: off, not zero
 *
 * An intensity of zero still pays for the pass. A shader that multiplies by a
 * uniform `0` still samples its textures, still runs its loop, still costs its
 * fill. Every flag here is wired to the point where the work stops being
 * issued:
 *
 * - `?fog=off` sets the volumetrics quality to `off`, which makes
 *   `VolumetricsModule` choose mode `off`, skip baking the 3D noise volume, and
 *   return a pass-through resolve node — and separately drops the light-shaft
 *   pass, which is the only consumer of that node.
 * - `?shadows=off` never attaches the cascade node to the sun, so no shadow
 *   camera renders and no material compiles a shadow sampler into its graph.
 * - `?post=off` disables the whole chain: the scene target is blitted straight
 *   to the canvas through one tone-mapped copy, so bloom's pyramid, the
 *   metering reduction, the grade and FXAA are all simply never run.
 * - `?bloom=off` additionally removes the bloom *sample* from the composite's
 *   node graph, so the composite compiles without the sampler at all. See the
 *   trap note below.
 * - `?props=off` / `?terrain=off` / `?chars=off` set `visible = false` on whole
 *   subtrees, which three skips during render-list assembly — the draws do not
 *   happen and the counter proves it.
 * - `?flat=1` swaps every scene material for an unlit basic one, which deletes
 *   the terrain uber-shader, all IBL sampling and all shadow sampling from
 *   every fragment in the frame at once.
 *
 * ### The bloom trap, which is real and was nearly shipped again
 *
 * `PostStack`'s tier table carries a comment saying bloom must stay on at every
 * tier because `CompositePass` "compiles against an unbound sampler" and
 * produces a black frame. Half of that is true. `CompositePass` does guard the
 * *intensity* (`bloomTexture === null` sets it to 0) and it does keep sampling
 * `#uBloom` unconditionally — but `makeSourceNode` seeds every source with a
 * 1×1 opaque black `DataTexture`, so the sampler is bound and the fetch returns
 * zero. Disabling bloom is therefore already safe, and the black frame that was
 * blamed on it belongs to the single-cascade bug documented on the `low` tier.
 *
 * It is still wrong to *pay* for that fetch, so `?bloom=off` now goes further
 * and takes the term out of the graph: `CompositePass.#structureKey` includes
 * the bloom state, so switching it off rebuilds the material without the
 * sampler rather than multiplying its result by zero.
 *
 * ## Composability
 *
 * Flags are independent and may be combined freely — `?fog=off&shadows=off` is
 * meaningful and does both. `?minimal=1` is the one aggregate: it is exactly
 * `fog=off&shadows=off&post=off&flat=1`, i.e. the floor. An explicit flag
 * always outranks `minimal`, so `?minimal=1&post=on` is a legal way to ask
 * "everything off except the post chain".
 *
 * ## Values
 *
 * Every switch accepts `off`/`0`/`false`/`no` to disable and `on`/`1`/`true`/
 * `yes` to enable; `?fog=off` and `?fog=0` are the same flag. An unrecognised
 * value warns and is ignored rather than being silently treated as `off`, for
 * the same reason `?quality=` warns: a typo that quietly changes what is being
 * measured is worse than no flag at all.
 */

/** The resolved switch set. Every field is "is this system running". */
export interface RenderFlags {
  /** Volumetric fog: the raymarch/froxel volume and its 3D noise. */
  readonly fog: boolean;
  /** Sun cascaded shadow maps, and every material's shadow sampling. */
  readonly shadows: boolean;
  /** The post chain as a whole. False presents the scene through one copy. */
  readonly post: boolean;
  /** The bloom pyramid, and the bloom term in the composite. */
  readonly bloom: boolean;
  /** The FXAA resolve pass. */
  readonly fxaa: boolean;
  /** Scene props: everything that is neither terrain nor a character. */
  readonly props: boolean;
  /** The terrain / ground / floor meshes. */
  readonly terrain: boolean;
  /** Characters: skinned meshes and their rigs. */
  readonly chars: boolean;
  /** Scene materials. False swaps every one for an unlit basic material. */
  readonly lit: boolean;
  /** `?gpusync=1`: force a 1×1 readback after each frame to observe GPU stalls. */
  readonly gpuSync: boolean;
  /** `?warmup=0`: skip the pre-render pipeline warmup on the loading screen. */
  readonly warmup: boolean;
}

/** Everything on. The shipping configuration. */
export const DEFAULT_RENDER_FLAGS: RenderFlags = Object.freeze({
  fog: true,
  shadows: true,
  post: true,
  bloom: true,
  fxaa: true,
  props: true,
  terrain: true,
  chars: true,
  lit: true,
  gpuSync: false,
  warmup: true,
});

/** What `?minimal=1` means, before explicit flags are layered on top. */
const MINIMAL: Partial<RenderFlags> = Object.freeze({
  fog: false,
  shadows: false,
  post: false,
  bloom: false,
  fxaa: false,
  lit: false,
});

const TRUTHY = new Set(['1', 'on', 'true', 'yes']);
const FALSY = new Set(['0', 'off', 'false', 'no']);

/**
 * Parse one boolean-ish parameter.
 *
 * @returns `null` when the parameter is absent *or* unparseable, so the caller
 *   keeps whatever it already had. Unparseable additionally warns.
 */
function readSwitch(params: URLSearchParams, name: string): boolean | null {
  const raw = params.get(name);
  if (raw === null) return null;
  const value = raw.trim().toLowerCase();
  if (TRUTHY.has(value)) return true;
  if (FALSY.has(value)) return false;
  console.warn(
    `[DebugFlags] unknown ?${name}=${raw}; expected on|off (or 1|0). Ignoring, ` +
      `so ${name} keeps its current setting.`,
  );
  return null;
}

/**
 * Resolve the flag set from a query string.
 *
 * Pure, and exported separately from {@link renderFlags} so tests can drive it
 * without a `window`. Order is deliberate: `minimal` is applied first as a
 * baseline, then every explicit flag overrides it.
 */
export function parseRenderFlags(search: string): RenderFlags {
  const params = new URLSearchParams(search);
  const flags: {
    -readonly [K in keyof RenderFlags]: RenderFlags[K];
  } = { ...DEFAULT_RENDER_FLAGS };

  if (readSwitch(params, 'minimal') === true) Object.assign(flags, MINIMAL);

  /** Which switches the URL named explicitly, so implications never override them. */
  const explicit = new Set<keyof RenderFlags>();

  const named: readonly (readonly [string, keyof RenderFlags])[] = [
    ['fog', 'fog'],
    ['shadows', 'shadows'],
    ['post', 'post'],
    ['bloom', 'bloom'],
    ['fxaa', 'fxaa'],
    ['props', 'props'],
    ['terrain', 'terrain'],
    ['chars', 'chars'],
    ['lit', 'lit'],
  ];
  for (const [param, key] of named) {
    const value = readSwitch(params, param);
    if (value !== null) {
      flags[key] = value;
      explicit.add(key);
    }
  }

  // `?flat=1` is the alias a bisecting player will actually type: it reads as
  // "make it flat", i.e. *disable* lighting, so it is inverted relative to the
  // field it drives. Applied after `?lit=`, so the two agree when both appear.
  const flat = readSwitch(params, 'flat');
  if (flat !== null) {
    flags.lit = !flat;
    explicit.add('lit');
  }

  // The post chain carries its members with it, in both directions.
  //
  // Off is the obvious half: `?post=off` must report — and mean — that bloom
  // and FXAA are not running, or the overlay's flag line lies about what a
  // screenshot was measured under.
  //
  // On is the half that is easy to get wrong. `?minimal=1&post=on` is how a
  // bisection asks "everything off *except* the post chain", and a `minimal`
  // baseline that left bloom and FXAA switched off underneath would hand back a
  // third configuration that is neither of the two being compared. An explicit
  // `?post=on&bloom=off` still wins, because `explicit` records what the URL
  // actually said.
  if (flags.post && explicit.has('post')) {
    if (!explicit.has('bloom')) flags.bloom = DEFAULT_RENDER_FLAGS.bloom;
    if (!explicit.has('fxaa')) flags.fxaa = DEFAULT_RENDER_FLAGS.fxaa;
  }

  // A pass cannot run while the chain that hosts it does not, so the "off" half
  // of the implication is unconditional and outranks even an explicit
  // `?bloom=on`. That combination is not a configuration, it is a
  // contradiction, and honouring the `bloom=on` half would put bloom on the
  // overlay's "running" list for a frame in which no chain pass executed.
  if (!flags.post) {
    flags.bloom = false;
    flags.fxaa = false;
  }

  flags.gpuSync = readSwitch(params, 'gpusync') ?? false;
  flags.warmup = readSwitch(params, 'warmup') ?? true;

  return Object.freeze(flags);
}

/** True when every rendering system is in its shipping state. */
export function allFlagsDefault(flags: RenderFlags): boolean {
  return (
    flags.fog &&
    flags.shadows &&
    flags.post &&
    flags.bloom &&
    flags.fxaa &&
    flags.props &&
    flags.terrain &&
    flags.chars &&
    flags.lit
  );
}

/**
 * One line naming everything that is switched off.
 *
 * Goes to the console at boot *and* onto the debug overlay, because the
 * instrument here is a screenshot from someone else's laptop: a frame time with
 * no record of the configuration that produced it is not a measurement.
 */
export function describeRenderFlags(flags: RenderFlags): string {
  const off: string[] = [];
  if (!flags.fog) off.push('fog');
  if (!flags.shadows) off.push('shadows');
  if (!flags.post) off.push('post');
  else {
    if (!flags.bloom) off.push('bloom');
    if (!flags.fxaa) off.push('fxaa');
  }
  if (!flags.props) off.push('props');
  if (!flags.terrain) off.push('terrain');
  if (!flags.chars) off.push('chars');
  if (!flags.lit) off.push('lit');

  const extra: string[] = [];
  if (flags.gpuSync) extra.push('gpusync');
  if (!flags.warmup) extra.push('no-warmup');

  const disabled = off.length === 0 ? 'all systems on' : `off: ${off.join(' ')}`;
  return extra.length === 0 ? disabled : `${disabled}  [${extra.join(' ')}]`;
}

/* -------------------------------------------------------------------------- *
 * The process-wide instance
 * -------------------------------------------------------------------------- */

let cached: RenderFlags | null = null;

/**
 * The flags for this page load, parsed once.
 *
 * Cached because it is read from a dozen places during boot and the answer
 * cannot change: these are load-time switches, not settings. `setRenderFlags`
 * exists for tests and for the capture harness, which builds frame graphs
 * without a URL.
 */
export function renderFlags(): RenderFlags {
  if (cached !== null) return cached;
  const search = typeof window === 'undefined' ? '' : window.location.search;
  cached = parseRenderFlags(search);
  return cached;
}

/** Override the cached flags. Pass `null` to re-read the URL on next access. */
export function setRenderFlags(flags: RenderFlags | null): void {
  cached = flags;
}

/** Log the active set once, at boot. Returns what it logged, for tests. */
export function logRenderFlags(flags: RenderFlags = renderFlags()): string {
  const line = describeRenderFlags(flags);
  if (allFlagsDefault(flags) && !flags.gpuSync && flags.warmup) {
    console.info(
      `[DebugFlags] ${line}. Bisect a slow frame with ?minimal=1, then re-enable ` +
        'one at a time: ?fog=off ?shadows=off ?post=off ?bloom=off ?fxaa=off ' +
        '?props=off ?terrain=off ?chars=off ?flat=1 ?gpusync=1',
    );
  } else {
    console.warn(`[DebugFlags] NON-DEFAULT RENDER CONFIGURATION — ${line}`);
  }
  return line;
}
