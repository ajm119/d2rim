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
