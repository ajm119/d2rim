import { chromium } from 'playwright';
import { CHROMIUM_ARGS, findChromium, startPreviewServer } from '../cli.mjs';

const PROBE = `
  const d2rim = window.__d2rim;
  await d2rim.ready;
  await d2rim.engine.stepFrames(8);
  const three = d2rim.ctx.renderer.three;
  const THREE = d2rim.three;
  const out = {};

  const trace = async (label) => {
    const seen = [];
    const orig = three.setRenderTarget.bind(three);
    three.setRenderTarget = function (t, a, b) {
      seen.push((t?.texture?.name ?? (t ? 'unnamed' : 'canvas')));
      return orig(t, a, b);
    };
    const d0 = three.info.render.drawCalls;
    await d2rim.engine.stepFrames(1);
    three.setRenderTarget = orig;
    out[label] = { targets: [...new Set(seen)], draws: three.info.render.drawCalls };
  };

  await trace('base');

  // Force every mesh to receive shadows, and rebuild materials.
  d2rim.engine.scene.traverse((o) => { if (o.isMesh) o.receiveShadow = true; });
  d2rim.engine.scene.traverse((o) => { if (o.isMesh) { const m = Array.isArray(o.material) ? o.material : [o.material]; for (const x of m) x.needsUpdate = true; } });
  await trace('allReceive');

  return out;
`;

const server = await startPreviewServer(4323);
const executablePath = findChromium();
const browser = await chromium.launch({ args: CHROMIUM_ARGS, ...(executablePath ? { executablePath } : {}) });
const page = await browser.newPage({ viewport: { width: 1528, height: 794 } });
const url = `${server.url}/?autostart=0&enemies=0&fade=0&stats=1&backend=webgl2&quality=medium&zone=encampment&localshadows=on`;
await page.goto(url, { waitUntil: 'load', timeout: 180000 });
console.log(JSON.stringify(await page.evaluate(new Function(`return (async () => { ${PROBE} })();`)), null, 1));
await browser.close();
server.stop();
