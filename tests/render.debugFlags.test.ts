/**
 * The per-system kill switches.
 *
 * These exist because the deployed build runs at 1.4 fps on a machine this
 * project cannot profile — no GPU in CI, SwiftShader on four cores — and the
 * only instrument available on the machine that matters is a player who can
 * edit a URL. So the flags have to be right *here*, where they can be asserted,
 * because the place they will be used is the place nobody can debug.
 *
 * Two properties matter more than the rest and are pinned hardest:
 *
 * 1. **Unknown values do not silently disable things.** `?fog=of` must not read
 *    as "fog off". A typo that quietly changes the configuration under
 *    measurement produces a number that means nothing and a conclusion that is
 *    worse than no conclusion.
 * 2. **`post=off` implies `bloom=off` and `fxaa=off`.** Both are members of the
 *    chain that is being switched off, and the overlay's flag line has to say
 *    so or a screenshot will claim bloom was running when it was not.
 */

import { describe, expect, it, vi, afterEach } from 'vitest';

import {
  DEFAULT_RENDER_FLAGS,
  allFlagsDefault,
  describeRenderFlags,
  logRenderFlags,
  parseRenderFlags,
  renderFlags,
  setRenderFlags,
} from '../src/render/DebugFlags';

afterEach(() => {
  setRenderFlags(null);
  vi.restoreAllMocks();
});

describe('parseRenderFlags', () => {
  it('leaves every system on for an empty query', () => {
    const flags = parseRenderFlags('');
    expect(flags).toEqual(DEFAULT_RENDER_FLAGS);
    expect(allFlagsDefault(flags)).toBe(true);
  });

  it('switches exactly one system off and leaves the rest alone', () => {
    const flags = parseRenderFlags('?fog=off');
    expect(flags.fog).toBe(false);
    // The whole value of the bisection is that the other eight are untouched.
    expect(flags.shadows).toBe(true);
    expect(flags.post).toBe(true);
    expect(flags.bloom).toBe(true);
    expect(flags.fxaa).toBe(true);
    expect(flags.props).toBe(true);
    expect(flags.terrain).toBe(true);
    expect(flags.chars).toBe(true);
    expect(flags.lit).toBe(true);
  });

  it('composes independent switches', () => {
    const flags = parseRenderFlags('?fog=off&shadows=off&chars=off');
    expect(flags.fog).toBe(false);
    expect(flags.shadows).toBe(false);
    expect(flags.chars).toBe(false);
    expect(flags.post).toBe(true);
    expect(flags.terrain).toBe(true);
  });

  it('accepts 0/1 as well as off/on, and is case-insensitive', () => {
    expect(parseRenderFlags('?fog=0').fog).toBe(false);
    expect(parseRenderFlags('?fog=OFF').fog).toBe(false);
    expect(parseRenderFlags('?fog=False').fog).toBe(false);
    expect(parseRenderFlags('?fog=no').fog).toBe(false);
    expect(parseRenderFlags('?fog=1').fog).toBe(true);
    expect(parseRenderFlags('?fog=yes').fog).toBe(true);
  });

  it('warns on an unparseable value and changes nothing', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const flags = parseRenderFlags('?fog=of');
    // The dangerous failure is reading "of" as falsy. Fog must stay on.
    expect(flags.fog).toBe(true);
    expect(warn).toHaveBeenCalledOnce();
    expect(String(warn.mock.calls[0]?.[0])).toContain('?fog=of');
  });

  it('turns bloom and fxaa off with the chain that contains them', () => {
    const flags = parseRenderFlags('?post=off');
    expect(flags.post).toBe(false);
    expect(flags.bloom).toBe(false);
    expect(flags.fxaa).toBe(false);
  });

  it('leaves the scene alone when only the post chain is disabled', () => {
    const flags = parseRenderFlags('?post=off');
    expect(flags.props).toBe(true);
    expect(flags.terrain).toBe(true);
    expect(flags.chars).toBe(true);
    expect(flags.lit).toBe(true);
    expect(flags.shadows).toBe(true);
  });
});

describe('?minimal=1', () => {
  it('is the floor: no fog, no shadows, no post, no lighting', () => {
    const flags = parseRenderFlags('?minimal=1');
    expect(flags.fog).toBe(false);
    expect(flags.shadows).toBe(false);
    expect(flags.post).toBe(false);
    expect(flags.bloom).toBe(false);
    expect(flags.fxaa).toBe(false);
    expect(flags.lit).toBe(false);
  });

  it('keeps the world visible, so the floor is still a picture of the scene', () => {
    // Deliberate. `?minimal=1` answers "is any single effect responsible"; it
    // must not also answer "is the geometry responsible", because then a slow
    // result would be ambiguous between the two. Hiding geometry is what
    // `?props=off` / `?terrain=off` / `?chars=off` are for, and they compose
    // with `minimal` for anyone who wants both.
    const flags = parseRenderFlags('?minimal=1');
    expect(flags.props).toBe(true);
    expect(flags.terrain).toBe(true);
    expect(flags.chars).toBe(true);
  });

  it('is overridden by an explicit flag, in either direction', () => {
    // "Everything off except the thing I want to measure" has to be one URL.
    const flags = parseRenderFlags('?minimal=1&post=on');
    expect(flags.post).toBe(true);
    expect(flags.fog).toBe(false);
    expect(flags.shadows).toBe(false);
    // `post=on` must not drag bloom/fxaa back down through the implication.
    expect(flags.bloom).toBe(true);
    expect(flags.fxaa).toBe(true);
  });

  it('composes with the scene switches for a truly empty frame', () => {
    const flags = parseRenderFlags('?minimal=1&props=off&terrain=off&chars=off');
    expect(flags.props).toBe(false);
    expect(flags.terrain).toBe(false);
    expect(flags.chars).toBe(false);
    expect(flags.lit).toBe(false);
  });
});

describe('?flat=1', () => {
  it('is the readable alias for lit=off', () => {
    expect(parseRenderFlags('?flat=1').lit).toBe(false);
    expect(parseRenderFlags('?lit=off').lit).toBe(false);
    expect(parseRenderFlags('?flat=0').lit).toBe(true);
  });
});

describe('probe flags', () => {
  it('are off unless asked for', () => {
    expect(parseRenderFlags('').gpuSync).toBe(false);
    expect(parseRenderFlags('?gpusync=1').gpuSync).toBe(true);
    // The warmup defaults *on*: it moves shader compilation onto the loading
    // screen, which is where a multi-second stall belongs.
    expect(parseRenderFlags('').warmup).toBe(true);
    expect(parseRenderFlags('?warmup=0').warmup).toBe(false);
  });

  it('does not let ?minimal=1 arm the probes', () => {
    // `minimal` is a *cheapness* aggregate. `gpusync` deliberately makes the
    // frame slower, so folding it in would poison exactly the measurement
    // `minimal` is used to take.
    expect(parseRenderFlags('?minimal=1').gpuSync).toBe(false);
  });
});

describe('describeRenderFlags', () => {
  it('says so plainly when nothing is switched off', () => {
    expect(describeRenderFlags(parseRenderFlags(''))).toBe('all systems on');
  });

  it('names every disabled system, because a screenshot has to carry it', () => {
    const line = describeRenderFlags(parseRenderFlags('?fog=off&shadows=off'));
    expect(line).toContain('fog');
    expect(line).toContain('shadows');
    expect(line.startsWith('off:')).toBe(true);
  });

  it('does not list bloom and fxaa separately once the whole chain is off', () => {
    // Reporting `off: post bloom fxaa` is noise: the members are implied by the
    // chain and repeating them makes the line harder to read at a glance on an
    // overlay that is already eleven lines tall.
    const line = describeRenderFlags(parseRenderFlags('?post=off'));
    expect(line).toContain('post');
    expect(line).not.toContain('bloom');
    expect(line).not.toContain('fxaa');
  });

  it('flags the probes separately from the kill switches', () => {
    const line = describeRenderFlags(parseRenderFlags('?gpusync=1'));
    expect(line).toContain('all systems on');
    expect(line).toContain('gpusync');
  });
});

describe('logRenderFlags', () => {
  it('warns loudly for a non-default configuration', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    logRenderFlags(parseRenderFlags('?minimal=1'));
    expect(warn).toHaveBeenCalledOnce();
    expect(String(warn.mock.calls[0]?.[0])).toContain('NON-DEFAULT');
  });

  it('advertises the switches on an ordinary boot', () => {
    const info = vi.spyOn(console, 'info').mockImplementation(() => undefined);
    logRenderFlags(parseRenderFlags(''));
    const line = String(info.mock.calls[0]?.[0]);
    expect(line).toContain('?minimal=1');
    expect(line).toContain('?fog=off');
    expect(line).toContain('?gpusync=1');
  });
});

describe('the cached instance', () => {
  it('is overridable, so a harness is never at the mercy of a stale URL', () => {
    const forced = parseRenderFlags('?fog=off');
    setRenderFlags(forced);
    expect(renderFlags()).toBe(forced);
    setRenderFlags(null);
    // Back to reading the environment, which under vitest has no query string.
    expect(renderFlags().fog).toBe(true);
  });
});

/**
 * The bisection flags added for the fixed-per-frame-stall investigation.
 *
 * `?scale=` is the load-bearing one. A player's `?minimal=1` run draws 77 calls
 * with unlit materials and still spends 58 ms a frame in a place no CPU timer
 * can see, and there is exactly one experiment that separates "bound by fill
 * or bandwidth" from "a fixed stall that has nothing to do with shading":
 * change the pixel count and nothing else. That experiment is only worth
 * running if the flag reports honestly, which is why an out-of-range value is
 * *refused* rather than clamped — a run that measured 0.1 while the reader
 * believed it measured 0.25 is worse than a run that never happened.
 */
describe('the resolution and shadow bisection flags', () => {
  it('leaves the tier in charge when they are absent', () => {
    const flags = parseRenderFlags('');
    expect(flags.renderScale).toBeNull();
    expect(flags.cascades).toBeNull();
    expect(flags.shadowDistance).toBeNull();
  });

  it('accepts a render scale inside the documented range', () => {
    expect(parseRenderFlags('?scale=0.5').renderScale).toBe(0.5);
    expect(parseRenderFlags('?scale=0.25').renderScale).toBe(0.25);
    expect(parseRenderFlags('?scale=1').renderScale).toBe(1);
  });

  it('refuses an out-of-range scale rather than clamping it', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    expect(parseRenderFlags('?scale=0.1').renderScale).toBeNull();
    expect(parseRenderFlags('?scale=2').renderScale).toBeNull();
    expect(warn).toHaveBeenCalledTimes(2);
  });

  it('refuses a scale that is not a number', () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    expect(parseRenderFlags('?scale=half').renderScale).toBeNull();
  });

  it('accepts an integer cascade count in [1, 4] and refuses anything else', () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    expect(parseRenderFlags('?cascades=1').cascades).toBe(1);
    expect(parseRenderFlags('?cascades=4').cascades).toBe(4);
    expect(parseRenderFlags('?cascades=0').cascades).toBeNull();
    expect(parseRenderFlags('?cascades=5').cascades).toBeNull();
    expect(parseRenderFlags('?cascades=1.5').cascades).toBeNull();
  });

  it('accepts a fractional shadow distance, because metres are not integers', () => {
    expect(parseRenderFlags('?shadowdist=55.5').shadowDistance).toBe(55.5);
  });

  it('names both on the overlay line, so a screenshot records what it measured', () => {
    const line = describeRenderFlags(parseRenderFlags('?scale=0.5&cascades=1'));
    expect(line).toContain('scale=0.5');
    expect(line).toContain('cascades=1');
  });
});

/**
 * Local (point/spot) shadow casters, which are off by default now.
 *
 * A shadow-casting point light is a cube map: six complete submissions of the
 * whole scene, every frame it is lit. The player's own bisection put the total
 * shadow bill at 574 → 161 draws against two sun cascades that this project's
 * headless trace prices at 154, so the local cube was costing more than both
 * cascades together — to shadow a warm pool of firelight under an overcast sky.
 */
describe('local shadow casters', () => {
  it('are off unless asked for', () => {
    expect(parseRenderFlags('').localShadows).toBe(false);
    expect(DEFAULT_RENDER_FLAGS.localShadows).toBe(false);
  });

  it('come back with ?localshadows=on', () => {
    expect(parseRenderFlags('?localshadows=on').localShadows).toBe(true);
    expect(parseRenderFlags('?localshadows=1').localShadows).toBe(true);
  });

  it('cannot survive ?shadows=off, which has to mean all shadows', () => {
    expect(parseRenderFlags('?shadows=off&localshadows=on').localShadows).toBe(false);
    expect(parseRenderFlags('?minimal=1&localshadows=on').localShadows).toBe(false);
  });

  it('are named on the overlay line when on, and silent when off', () => {
    expect(describeRenderFlags(parseRenderFlags('?localshadows=on'))).toContain('localshadows');
    expect(describeRenderFlags(parseRenderFlags(''))).not.toContain('localshadows');
  });

  it('do not make an otherwise-default configuration read as non-default', () => {
    expect(allFlagsDefault(parseRenderFlags(''))).toBe(true);
  });
});
