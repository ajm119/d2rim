import { chromium } from 'playwright';
import { CHROMIUM_ARGS, findChromium, startPreviewServer } from '../cli.mjs';

const PROBE = `
  const d2rim = window.__d2rim;
  await d2rim.ready;
  await d2rim.engine.stepFrames(8);
  const three = d2rim.ctx.renderer.three;
  const seen = [];
  const orig = three.setRenderTarget.bind(three);
  three.setRenderTarget = function (t, a, b) {
    seen.push((t?.texture?.name ?? (t ? 'unnamed' : 'canvas')) + (a !== undefined ? ':' + a : ''));
    return orig(t, a, b);
  };
  await d2rim.engine.stepFrames(1);
  three.setRenderTarget = orig;
  const shadowed = [];
  d2rim.engine.scene.traverse((o) => {
    if (o.isLight && o.castShadow) shadowed.push({
      n: o.name, i: o.intensity,
      node: !!(o.shadow && o.shadow.shadowNode),
    });
  });
  let receivers = 0, total = 0;
  d2rim.engine.scene.traverse((o) => { if (o.isMesh) { total++; if (o.receiveShadow) receivers++; } });
  return { seen: [...new Set(seen)], order: seen, shadowed, receivers, total };
`;

const server = await startPreviewServer(4322);
const executablePath = findChromium();
const browser = await chromium.launch({ args: CHROMIUM_ARGS, ...(executablePath ? { executablePath } : {}) });
const page = await browser.newPage({ viewport: { width: 1528, height: 794 } });
const extra = process.argv[2] ?? '';
const url = `${server.url}/?autostart=0&enemies=0&fade=0&stats=1&backend=webgl2&quality=medium&zone=encampment${extra}`;
await page.goto(url, { waitUntil: 'load', timeout: 180000 });
const out = await page.evaluate(new Function(`return (async () => { ${PROBE} })();`));
console.log('unique targets:', out.seen.join(' | '));
console.log('order:', out.order.join(' '));
console.log('shadowed lights:', JSON.stringify(out.shadowed));
console.log('receivers:', out.receivers, '/', out.total);
await browser.close();
server.stop();
