/**
 * The post-chain kill switches: `?post=off`, `?bloom=off`, `?fxaa=off`.
 *
 * ### Why these need tests of their own
 *
 * `PostStack.#applyTier` re-derives `bloom.enabled` and `fxaa.enabled` from the
 * tier table every time it runs — at `init`, and on every `setQuality`. Anything
 * that switched a pass off by assigning to the pass directly would therefore be
 * silently undone before the first frame, and the symptom would be "the flag
 * did nothing", which is indistinguishable from "the pass was not the problem".
 * On a bisection where every step is a hypothesis, that failure mode is
 * expensive: it does not produce a wrong answer, it produces a *confident* wrong
 * answer. Hence `setOverrides`, and hence these tests.
 *
 * ### And why the bloom one exists specifically
 *
 * `PostStack`'s tier table carries a standing comment that bloom must never be
 * switched off because the composite would "compile against an unbound sampler"
 * and produce a black frame. The sampler was never unbound — `makeSourceNode`
 * seeds it with a 1x1 opaque black `DataTexture` — so the fetch was always safe
 * and always wasted. `CompositePass` now takes bloom into its structure key, so
 * disabling the pyramid rebuilds the composite *without* the fetch rather than
 * multiplying its result by zero. The test below pins the rebuild, because a
 * structure key that silently stops discriminating is exactly how the term
 * would creep back in.
 */

import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';

import { parseRenderFlags } from '../src/render/DebugFlags';
import { BloomPass } from '../src/render/post/Bloom';
import { ColorGrade } from '../src/render/post/ColorGrade';
import { PostStack } from '../src/render/post/PostStack';
import { CompositePass } from '../src/render/post/Tonemap';
import type { PostFrame } from '../src/render/post/PostStack';

/**
 * A `PostFrame` that records blits instead of issuing them.
 *
 * Enough for the passes under test: `CompositePass.render` only reads sizes and
 * the input texture, and calls `frame.blit` exactly once for the composite
 * itself (plus five more when metering is on, which it is not here).
 */
function fakeFrame(): PostFrame & { blits: { material: THREE.Material; label: string }[] } {
  const blits: { material: THREE.Material; label: string }[] = [];
  const texture = new THREE.DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1);
  return {
    renderer: null as unknown as PostFrame['renderer'],
    capabilities: { backend: 'webgl2', float32Filterable: false, maxSamples: 4, compute: false },
    camera: new THREE.PerspectiveCamera(),
    quality: 'medium',
    width: 320,
    height: 180,
    deltaTime: 1 / 60,
    elapsed: 0,
    frameIndex: 0,
    motion: null as unknown as PostFrame['motion'],
    depthTexture: null,
    input: texture,
    output: null,
    blit(material: THREE.Material, _target: THREE.RenderTarget | null, label: string) {
      blits.push({ material, label });
    },
    blits,
  } as unknown as PostFrame & { blits: { material: THREE.Material; label: string }[] };
}

describe('PostStack.setOverrides', () => {
  it('switches bloom and fxaa off', () => {
    const stack = new PostStack({ quality: 'medium' });
    stack.setOverrides({ bloom: false, fxaa: false });
    expect(stack.bloom.enabled).toBe(false);
    expect(stack.fxaa.enabled).toBe(false);
    stack.dispose();
  });

  it('survives a tier change, which is the whole reason it exists', () => {
    // `setQuality` runs `#applyTier`, which assigns `bloom.enabled` straight
    // from the tier table. Without the override being re-applied afterwards the
    // flag would last exactly until the first quality change — and `?quality=`
    // is the other knob a bisecting player reaches for.
    const stack = new PostStack({ quality: 'medium' });
    stack.setOverrides({ bloom: false, fxaa: false });
    stack.setQuality('high');
    expect(stack.bloom.enabled).toBe(false);
    stack.setQuality('low');
    expect(stack.bloom.enabled).toBe(false);
    expect(stack.fxaa.enabled).toBe(false);
    stack.dispose();
  });

  it('disables the whole chain for ?post=off', () => {
    const stack = new PostStack({ quality: 'medium' });
    expect(stack.enabled).toBe(true);
    stack.setOverrides({ enabled: false });
    expect(stack.enabled).toBe(false);
    stack.dispose();
  });

  it('only ever subtracts: an override of `true` changes nothing', () => {
    // `setOverrides` is a kill switch, not a quality control. Whether a pass
    // *runs* stays the tier's decision (`#applyTier`) and the pass's own
    // `isAvailable`; all a flag can do is take one away. Otherwise `?bloom=on`
    // at `?quality=low` would be a way to assemble a configuration no tier
    // ships, and the bisection would be measuring a fiction.
    const stack = new PostStack({ quality: 'medium' });
    const bloomBefore = stack.bloom.enabled;
    const fxaaBefore = stack.fxaa.enabled;
    stack.setOverrides({ bloom: true, fxaa: true });
    expect(stack.bloom.enabled).toBe(bloomBefore);
    expect(stack.fxaa.enabled).toBe(fxaaBefore);
    stack.dispose();
  });

  it('translates the URL flags exactly', () => {
    const flags = parseRenderFlags('?bloom=off');
    const stack = new PostStack({ quality: 'medium' });
    const fxaaBefore = stack.fxaa.enabled;
    stack.setOverrides({ enabled: flags.post, bloom: flags.bloom, fxaa: flags.fxaa });
    expect(stack.enabled).toBe(true);
    expect(stack.bloom.enabled).toBe(false);
    // `?bloom=off` said nothing about FXAA, so FXAA is untouched.
    expect(stack.fxaa.enabled).toBe(fxaaBefore);
    stack.dispose();
  });
});

describe('CompositePass and the bloom term', () => {
  it('rebuilds the material when bloom is switched off', () => {
    // Not a cosmetic rebuild: the new material is compiled from a node graph
    // with no bloom sampler and no bloom fetch in it, which is what makes
    // `?bloom=off` a measurement rather than a multiply by zero.
    const bloom = new BloomPass({});
    const grade = new ColorGrade({});
    const composite = new CompositePass(bloom, grade, { autoExposure: false });

    const frame = fakeFrame();
    composite.render(frame);
    const withBloom = frame.blits.at(-1)?.material;
    expect(withBloom).toBeDefined();

    bloom.enabled = false;
    composite.render(frame);
    const withoutBloom = frame.blits.at(-1)?.material;

    expect(withoutBloom).toBeDefined();
    expect(withoutBloom).not.toBe(withBloom);

    composite.dispose();
    bloom.dispose();
  });

  it('keeps one material while bloom stays on', () => {
    // The structure key must discriminate on bloom without becoming unstable:
    // a key that changed every frame would recompile the frame's single
    // full-resolution pass on every frame, which is a far worse bug than the
    // one being fixed.
    const bloom = new BloomPass({});
    const grade = new ColorGrade({});
    const composite = new CompositePass(bloom, grade, { autoExposure: false });

    const frame = fakeFrame();
    composite.render(frame);
    const first = frame.blits.at(-1)?.material;
    composite.render(frame);
    composite.render(frame);
    expect(frame.blits.at(-1)?.material).toBe(first);

    composite.dispose();
    bloom.dispose();
  });

  it('still produces a composite draw with bloom off', () => {
    // The historical fear was a black frame. There is still exactly one
    // full-resolution draw and it still has a material; what changed is what is
    // inside it.
    const bloom = new BloomPass({});
    bloom.enabled = false;
    const grade = new ColorGrade({});
    const composite = new CompositePass(bloom, grade, { autoExposure: false });

    const frame = fakeFrame();
    composite.render(frame);

    expect(frame.blits).toHaveLength(1);
    expect(frame.blits[0]?.label).toBe('post.composite');

    composite.dispose();
    bloom.dispose();
  });
});
