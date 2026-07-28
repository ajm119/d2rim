import { chromium } from 'playwright';
import { CHROMIUM_ARGS, findChromium, startPreviewServer } from '/home/user/d2rim/tools/capture/cli.mjs';

const PROBE = `
  const d2rim = window.__d2rim;
  await d2rim.ready;
  await d2rim.engine.stepFrames(6);

  const three = d2rim.ctx.renderer.three;
  // Instrument: count renderer.render() invocations and the draws inside each.
  const submissions = [];
  const orig = three.render.bind(three);
  let depth = 0;
  three.render = function (scene, camera) {
    const before = three.info.render.drawCalls;
    depth++;
    const r = orig(scene, camera);
    depth--;
    submissions.push({
      depth,
      camera: camera?.type ?? '?',
      cam: camera?.name ?? '',
      scene: scene?.name ?? '',
      rt: three.getRenderTarget()?.texture?.name ?? (three.getRenderTarget() ? 'rt' : 'canvas'),
      draws: three.info.render.drawCalls - before,
    });
    return r;
  };
  await d2rim.engine.stepFrames(2);
  three.render = orig;

  const lighting = d2rim.ctx.services.tryGet('render.lighting');
  const lights = [];
  d2rim.engine.scene.traverse((o) => {
    if (o.isLight) lights.push({
      name: o.name, type: o.type, cast: o.castShadow, intensity: o.intensity,
      visible: o.visible, layers: o.layers.mask,
      shadowMap: o.shadow?.map ? 'map' : (o.shadow ? 'noMap' : '-'),
      autoUpdate: o.shadow?.autoUpdate, needsUpdate: o.shadow?.needsUpdate,
      y: Math.round(o.position.y),
    });
  });
  return {
    lights,
    submissions,
    lightingStats: lighting?.stats ?? null,
    total: three.info.render.drawCalls,
    tris: three.info.render.triangles,
  };
`;

const server = await startPreviewServer(4321);
const executablePath = findChromium();
const browser = await chromium.launch({ args: CHROMIUM_ARGS, ...(executablePath ? { executablePath } : {}) });
const page = await browser.newPage({ viewport: { width: 1528, height: 794 } });
const zones = process.argv[2] ?? 'encampment';
const extra = process.argv[3] ?? '';
for (const zone of zones.split(',')) {
  const url = `${server.url}/?autostart=0&enemies=0&fade=0&stats=1&backend=webgl2&quality=medium&zone=${zone}${extra}`;
  await page.goto(url, { waitUntil: 'load', timeout: 180000 });
  const out = await page.evaluate(new Function(`return (async () => { ${PROBE} })();`));
  console.log('###', zone, extra);
  console.log(JSON.stringify(out, null, 1));
}
await browser.close();
server.stop();
