/**
 * @module render/DisplaySettings
 *
 * The player's brightness trim: one number, in stops, persisted across
 * sessions and settable from the pause menu or the URL.
 *
 * ## Why a player-facing exposure control is not optional
 *
 * The frame's key is an art decision and it is locked
 * (`FrameGraph` sets `autoExposure: false` — see the long note there, it is
 * right and it should stay). But a locked key is locked against *scene*
 * radiance, not against the panel it is eventually displayed on. Two players
 * looking at the same correctly-graded frame on a 250-nit laptop in daylight
 * and on a 600-nit monitor in a dark room are not looking at the same picture,
 * and no amount of grading in the renderer can reconcile them. A grimdark game
 * puts most of its information in the bottom third of the histogram, which is
 * exactly the part a bright room erases first.
 *
 * So the trim lives *outside* the art direction:
 *
 * - It is applied through `CompositePass.setExposureCompensation`, which is a
 *   multiplier on the final exposure and is deliberately separate from
 *   `setExposure`. `setExposure` is owned by the art direction and by the
 *   per-zone {@link module:world/Zone} `ZoneGrade` trims, which multiply it. If
 *   the player's slider wrote there instead, the next zone transition would
 *   overwrite it — `ZoneManager` restores the exposure it saw at zone entry —
 *   and the player's setting would silently evaporate on the first portal.
 * - It is in *stops*, because that is the unit the rest of the exposure system
 *   already speaks and because it is perceptually even: every notch is the same
 *   apparent change.
 *
 * ## Precedence
 *
 * `?exposure=` (diagnosis) > stored setting (the player) > 0 (the authored
 * look). The URL form deliberately does not persist: it is for answering "is
 * this dark or is it broken" in one page load without corrupting the player's
 * saved preference.
 */

/** Clamp for the trim, in stops. ±2 stops is a factor of 16 end to end. */
export const EXPOSURE_STOPS_MIN = -2;
export const EXPOSURE_STOPS_MAX = 2;

/** Where the trim is persisted. */
export const EXPOSURE_STORAGE_KEY = 'd2rim.settings.exposureStops';

export function clampExposureStops(stops: number): number {
  if (!Number.isFinite(stops)) return 0;
  return Math.min(EXPOSURE_STOPS_MAX, Math.max(EXPOSURE_STOPS_MIN, stops));
}

/**
 * `?exposure=<stops>`, or `null` when absent or unparseable.
 *
 * Stops, not a multiplier: `?exposure=1` is one stop brighter (×2), and
 * `?exposure=-0.5` is half a stop darker. An unparseable value warns rather
 * than silently doing nothing, because a typo in a diagnostic flag that quietly
 * changes nothing is how an hour disappears.
 */
export function exposureStopsFromUrl(search?: string): number | null {
  const query = search ?? (typeof window === 'undefined' ? '' : window.location.search);
  const raw = new URLSearchParams(query).get('exposure');
  if (raw === null) return null;
  const value = Number.parseFloat(raw);
  if (!Number.isFinite(value)) {
    console.warn(`[DisplaySettings] unknown ?exposure=${raw}; expected a number of stops.`);
    return null;
  }
  return clampExposureStops(value);
}

/** Read the persisted trim. `0` when nothing is stored or storage is blocked. */
export function loadExposureStops(): number {
  try {
    const raw = globalThis.localStorage?.getItem(EXPOSURE_STORAGE_KEY);
    if (raw === null || raw === undefined) return 0;
    return clampExposureStops(Number.parseFloat(raw));
  } catch {
    // Private browsing, a blocked third-party context, a disabled storage API.
    // A missing preference is not an error worth surfacing.
    return 0;
  }
}

/** Persist the trim. Silently a no-op when storage is unavailable. */
export function saveExposureStops(stops: number): void {
  try {
    globalThis.localStorage?.setItem(EXPOSURE_STORAGE_KEY, String(clampExposureStops(stops)));
  } catch {
    /* see loadExposureStops */
  }
}

/**
 * The trim to boot with, and a one-line explanation of where it came from.
 *
 * Logged rather than returned silently: when a player reports "it's too dark",
 * the first question is whether their slider is even doing anything, and the
 * console should already contain the answer.
 */
export function initialExposureStops(search?: string): number {
  const fromUrl = exposureStopsFromUrl(search);
  if (fromUrl !== null) {
    console.info(
      `[DisplaySettings] exposure trim ${fromUrl >= 0 ? '+' : ''}${fromUrl.toFixed(2)} stops ` +
        `(×${(2 ** fromUrl).toFixed(2)}) from ?exposure=. Not persisted.`,
    );
    return fromUrl;
  }
  const stored = loadExposureStops();
  if (stored !== 0) {
    console.info(
      `[DisplaySettings] exposure trim ${stored >= 0 ? '+' : ''}${stored.toFixed(2)} stops ` +
        `(×${(2 ** stored).toFixed(2)}) from saved settings. Change it in the pause menu.`,
    );
  }
  return stored;
}
