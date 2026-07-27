/**
 * @module core/FrameStats
 *
 * A rolling window of **raw, unclamped** frame times, and the percentiles that
 * make them mean something.
 *
 * ### Why this file exists
 *
 * The debug overlay used to report frame time from an exponential moving
 * average of `TimeState.delta` — and `TimeState.delta` is the *clamped* delta,
 * bounded by the fixed-step accumulator's `maxDelta`. On a machine slower than
 * the clamp, that average converges to the clamp and stays there: every
 * struggling device in the world reports exactly `1 / maxDelta` fps, forever.
 * With the old default of `maxDelta = 0.25` that is `4.0 fps (250.0 ms)`, to
 * one decimal place, on a 6 fps machine and on a 0.5 fps machine alike. Two
 * backends were compared using that number and it read identically on both,
 * which is not a coincidence — it is the instrument being pinned against its
 * end stop.
 *
 * So the rule this module exists to enforce: **the number the player reports
 * back must be the number the wall clock produced**, before any clamping,
 * scaling or smoothing. The clamp is still shown, separately, as a count of how
 * many frames in the window hit it — that is diagnostic in its own right (it
 * says "simulation is running behind wall clock") but it must never be
 * mistaken for the frame time.
 *
 * ### Why percentiles rather than an average
 *
 * An average hides the shape. A 60 fps average with a 200 ms hitch every second
 * is unplayable and reads as "fine". p50 says what the typical frame costs and
 * p95 says what the bad frames cost, and the gap between them is the single
 * most useful thing an overlay can say about stutter. They are computed over a
 * fixed-size ring buffer (a few seconds of frames), so they respond to a scene
 * change within the window rather than being dragged around by boot.
 *
 * ### The CPU/GPU split
 *
 * Each sample also carries where the time went:
 *
 * - `updateMs` — every module's `fixedUpdate`/`update`/`lateUpdate`. Pure JS.
 * - `renderMs` — wall time of the awaited `renderer.render` call. On WebGL2
 *   this is command submission plus whatever implicit synchronisation the
 *   driver imposes; on WebGPU it includes the await on submission.
 * - `gpuMs` — genuine GPU time, from `EXT_disjoint_timer_query_webgl2` (or the
 *   WebGPU timestamp-query feature) via three's `resolveTimestampsAsync`.
 *   `0` when the extension is unavailable, which is common and not an error.
 *
 * `updateMs + renderMs` accounting for far less than the raw frame time is
 * itself the diagnosis: the missing time is the browser blocking on the GPU
 * before it will hand out the next animation frame, i.e. the frame is
 * GPU-bound. That inference is only available because the raw time is honest.
 */

/** One frame's worth of timing, in milliseconds. */
export interface FrameSample {
  /** Wall-clock time since the previous frame, unclamped and unscaled. */
  readonly rawMs: number;
  /** Time inside module update phases. */
  readonly updateMs: number;
  /** Wall time of the awaited render call. */
  readonly renderMs: number;
  /** GPU time from a timer query, or 0 when unavailable. */
  readonly gpuMs: number;
  /** Whether the fixed-step clamp truncated this frame's delta. */
  readonly clamped: boolean;
  /** Whether the fixed-step accumulator dropped simulation backlog. */
  readonly starved: boolean;
}

/** Aggregated view over the rolling window. All times in milliseconds. */
export interface FrameStatsSnapshot {
  /** How many frames the window currently holds. */
  readonly samples: number;
  /** The most recent raw frame time. */
  readonly lastMs: number;
  /** Median raw frame time — what a typical frame costs. */
  readonly p50Ms: number;
  /** 95th-percentile raw frame time — what the bad frames cost. */
  readonly p95Ms: number;
  /** Worst raw frame time in the window. */
  readonly maxMs: number;
  /** Frames per second implied by {@link p50Ms}. */
  readonly fps: number;
  /** Frames per second implied by {@link p95Ms}; the "1% low" style figure. */
  readonly fpsLow: number;
  /** Median time in module update phases. */
  readonly updateMs: number;
  /** Median wall time of the render call. */
  readonly renderMs: number;
  /** Median GPU time, or 0 when no timer query is available. */
  readonly gpuMs: number;
  /** Whether any sample in the window carried a GPU timing. */
  readonly gpuAvailable: boolean;
  /** How many frames in the window hit the fixed-step delta clamp. */
  readonly clampedFrames: number;
  /** How many frames in the window dropped simulation backlog. */
  readonly starvedFrames: number;
}

const EMPTY: FrameStatsSnapshot = {
  samples: 0,
  lastMs: 0,
  p50Ms: 0,
  p95Ms: 0,
  maxMs: 0,
  fps: 0,
  fpsLow: 0,
  updateMs: 0,
  renderMs: 0,
  gpuMs: 0,
  gpuAvailable: false,
  clampedFrames: 0,
  starvedFrames: 0,
};

/**
 * Nearest-rank percentile over an already-sorted ascending array.
 *
 * Nearest-rank rather than interpolated: with a window of a couple of hundred
 * frames the difference is well under the noise floor, and picking a real
 * observed sample means "p95 = 210 ms" names a frame that actually happened.
 */
function percentile(sorted: readonly number[], q: number): number {
  const n = sorted.length;
  if (n === 0) return 0;
  const rank = Math.ceil(q * n);
  const index = Math.min(n - 1, Math.max(0, rank - 1));
  return sorted[index] ?? 0;
}

/**
 * Fixed-capacity ring of {@link FrameSample}s with percentile queries.
 *
 * Sampling is O(1) per frame and allocation-free after construction: the ring
 * is preallocated and overwritten in place, because an instrument that
 * allocates once per frame is an instrument that shows up in its own numbers.
 * {@link snapshot} does sort, which is O(n log n) over the window — that is why
 * callers are expected to snapshot on a display cadence (a few hertz), never
 * per frame.
 */
export class FrameStats {
  readonly capacity: number;

  readonly #raw: Float64Array;
  readonly #update: Float64Array;
  readonly #render: Float64Array;
  readonly #gpu: Float64Array;
  readonly #clamped: Uint8Array;
  readonly #starved: Uint8Array;

  #count = 0;
  #cursor = 0;
  #lastMs = 0;

  constructor(capacity = 240) {
    this.capacity = Math.max(1, Math.floor(capacity));
    this.#raw = new Float64Array(this.capacity);
    this.#update = new Float64Array(this.capacity);
    this.#render = new Float64Array(this.capacity);
    this.#gpu = new Float64Array(this.capacity);
    this.#clamped = new Uint8Array(this.capacity);
    this.#starved = new Uint8Array(this.capacity);
  }

  /** Frames currently held, up to {@link capacity}. */
  get size(): number {
    return this.#count;
  }

  /**
   * Add one frame.
   *
   * Non-finite and negative times are stored as zero rather than rejected: a
   * clock that misbehaves across tab suspension must not be able to poison the
   * percentiles with a `NaN` that then sorts unpredictably.
   */
  record(sample: FrameSample): void {
    const clean = (value: number): number =>
      Number.isFinite(value) && value > 0 ? value : 0;

    const i = this.#cursor;
    this.#raw[i] = clean(sample.rawMs);
    this.#update[i] = clean(sample.updateMs);
    this.#render[i] = clean(sample.renderMs);
    this.#gpu[i] = clean(sample.gpuMs);
    this.#clamped[i] = sample.clamped ? 1 : 0;
    this.#starved[i] = sample.starved ? 1 : 0;

    this.#lastMs = clean(sample.rawMs);
    this.#cursor = (i + 1) % this.capacity;
    if (this.#count < this.capacity) this.#count++;
  }

  /** Aggregate the window. Sorts internally; call a few times a second, not per frame. */
  snapshot(): FrameStatsSnapshot {
    const n = this.#count;
    if (n === 0) return EMPTY;

    const raw: number[] = [];
    const update: number[] = [];
    const render: number[] = [];
    const gpu: number[] = [];
    let clampedFrames = 0;
    let starvedFrames = 0;
    let gpuAvailable = false;
    let maxMs = 0;

    for (let i = 0; i < n; i++) {
      const r = this.#raw[i] ?? 0;
      raw.push(r);
      if (r > maxMs) maxMs = r;
      update.push(this.#update[i] ?? 0);
      render.push(this.#render[i] ?? 0);
      const g = this.#gpu[i] ?? 0;
      if (g > 0) {
        gpuAvailable = true;
        gpu.push(g);
      }
      if (this.#clamped[i] === 1) clampedFrames++;
      if (this.#starved[i] === 1) starvedFrames++;
    }

    const ascending = (a: number, b: number): number => a - b;
    raw.sort(ascending);
    update.sort(ascending);
    render.sort(ascending);
    gpu.sort(ascending);

    const p50Ms = percentile(raw, 0.5);
    const p95Ms = percentile(raw, 0.95);

    return {
      samples: n,
      lastMs: this.#lastMs,
      p50Ms,
      p95Ms,
      maxMs,
      fps: p50Ms > 0 ? 1000 / p50Ms : 0,
      fpsLow: p95Ms > 0 ? 1000 / p95Ms : 0,
      updateMs: percentile(update, 0.5),
      renderMs: percentile(render, 0.5),
      gpuMs: percentile(gpu, 0.5),
      gpuAvailable,
      clampedFrames,
      starvedFrames,
    };
  }

  /** Drop every sample. Use after a deliberate discontinuity such as a zone load. */
  reset(): void {
    this.#count = 0;
    this.#cursor = 0;
    this.#lastMs = 0;
  }
}
