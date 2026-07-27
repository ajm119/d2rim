/**
 * tools/verify-rpg.mjs
 *
 * Drives the RPG layer in a real browser and asserts the things a unit test
 * cannot: that the modules actually wire together inside the engine, that the
 * UI renders and is legible, and that the Den of Evil quest can be completed
 * end to end.
 *
 * What it does, in order:
 *
 *   1. boots into the Rogue Encampment and waits for the five NPCs to be placed
 *   2. awards experience until the character levels, and checks the pools grew
 *   3. spends a skill point and confirms a *combat* number moved
 *   4. generates an item, equips it, and confirms the offence handed to the
 *      damage model changed
 *   5. drops loot on the floor and picks it up through the loot system
 *   6. opens Akara's dialogue and accepts the Den of Evil quest through it
 *   7. travels to the Den, adopts the generator's real spawn count, and reports
 *      that many kills
 *   8. travels back and turns the quest in to Akara for its skill point
 *   9. saves and loads through the browser's real IndexedDB and compares
 *  10. captures the HUD, the inventory screen, the skill tree, the dialogue
 *      overlay and the vendor, as PNGs, for a human to read
 *
 * Usage: `node tools/verify-rpg.mjs [outDir] [--no-shots] [--quality=low|high]`
 * Assumes `dist/` is current; run `npm run build` first.
 */

import { mkdirSync } from 'node:fs';
import { spawn } from 'node:child_process';

import { chromium } from 'playwright';

import { CHROMIUM_ARGS, ROOT, findChromium } from './capture/cli.mjs';

const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const SHOTS = !process.argv.includes('--no-shots');
const OUT = args[0] ?? '/tmp/rpg';
const PORT = 5251;
const WIDTH = 1280;
const HEIGHT = 720;
const QUALITY =
  (process.argv.find((a) => a.startsWith('--quality=')) ?? '--quality=low').split('=')[1];

mkdirSync(OUT, { recursive: true });

const failures = [];
const check = (name, ok, detail) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail === undefined ? '' : ` — ${detail}`}`);
  if (!ok) failures.push(name);
};

/* -- server ---------------------------------------------------------------- */

const server = spawn(
  'npx',
  ['vite', 'preview', '--port', String(PORT), '--strictPort', '--host', '127.0.0.1'],
  { cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe'] },
);
server.stderr.on('data', () => undefined);
await new Promise((resolve) => {
  server.stdout.on('data', (chunk) => {
    if (String(chunk).includes('Local:')) resolve();
  });
  setTimeout(resolve, 15_000);
});

const browser = await chromium.launch({ args: [...CHROMIUM_ARGS], executablePath: findChromium() });
const page = await browser.newPage({ viewport: { width: WIDTH, height: HEIGHT } });
page.setDefaultNavigationTimeout(240_000);
page.setDefaultTimeout(240_000);
const logs = [];
page.on('console', (m) => logs.push(`${m.type()}: ${m.text()}`));
page.on('pageerror', (e) => logs.push(`pageerror: ${e.message}`));

const step = async (n) => {
  const startedAt = Date.now();
  await page.evaluate((frames) => window.__d2rim.engine.stepFrames(frames), n);
  return ((Date.now() - startedAt) / 1000).toFixed(1);
};

const shot = async (name) => {
  if (!SHOTS) return;
  await page.screenshot({ path: `${OUT}/${name}.png` });
  console.log(`  ...wrote ${OUT}/${name}.png`);
};

/* -- boot ------------------------------------------------------------------ */

// `enemies=0` keeps the zones unpopulated: this harness drives the quest by
// reporting kills directly, and twenty skeletons pathfinding on SwiftShader
// would triple the runtime while testing the AI rather than the RPG layer.
const url =
  `http://127.0.0.1:${PORT}/?autostart=0&backend=webgl2&quality=${QUALITY}` +
  `&fade=0&enemies=0&zone=encampment`;
process.stdout.write(`  ...loading ${url}\n`);
await page.goto(url, { waitUntil: 'load' });
await page.evaluate(() => window.__d2rim.ready);
await step(12);

/* -- 1. wiring ------------------------------------------------------------- */

const wiring = await page.evaluate(() => {
  const services = window.__d2rim.ctx.services;
  return {
    keys: services.keys(),
    modules: window.__d2rim.engine.moduleNames,
  };
});
check(
  'rpg services registered',
  ['rpg', 'quest.system', 'rpg.loot', 'quest.npcs', 'ui'].every((key) =>
    wiring.keys.includes(key),
  ),
  wiring.keys.filter((k) => k.startsWith('rpg') || k.startsWith('quest') || k === 'ui').join(', '),
);

/* -- 2. the camp NPCs ------------------------------------------------------ */

await page
  .waitForFunction(() => (window.__d2rim.ctx.services.get('quest.npcs').npcs.length ?? 0) >= 5, null, {
    timeout: 90_000,
  })
  .catch(() => undefined);
const npcs = await page.evaluate(() =>
  window.__d2rim.ctx.services.get('quest.npcs').npcs.map((npc) => ({
    id: npc.definition.id,
    x: Number(npc.position.x.toFixed(2)),
    z: Number(npc.position.z.toFixed(2)),
    animated: npc.graph !== null,
  })),
);
check('five camp NPCs placed', npcs.length === 5, npcs.map((n) => n.id).join(', '));
check(
  'every NPC has an animation graph',
  npcs.length > 0 && npcs.every((n) => n.animated),
  `${npcs.filter((n) => n.animated).length}/${npcs.length}`,
);
check(
  'NPCs stand on distinct anchors',
  new Set(npcs.map((n) => `${n.x},${n.z}`)).size === npcs.length,
);

/* -- 3. experience and levelling ------------------------------------------- */

const levelling = await page.evaluate(() => {
  const rpg = window.__d2rim.ctx.services.get('rpg');
  const combat = window.__d2rim.ctx.services.get('combat');
  const before = {
    level: rpg.character.stats.level,
    life: rpg.character.derived.maxLife,
    healthMax: combat.vitals.health.max,
    skillPoints: rpg.character.stats.skillPoints,
    statPoints: rpg.character.stats.statPoints,
  };
  rpg.awardExperience(3750); // level 4
  const after = {
    level: rpg.character.stats.level,
    life: rpg.character.derived.maxLife,
    healthMax: combat.vitals.health.max,
    skillPoints: rpg.character.stats.skillPoints,
    statPoints: rpg.character.stats.statPoints,
  };
  return { before, after };
});
check(
  'awarding experience levels the character',
  levelling.after.level === 4 && levelling.before.level === 1,
  `${levelling.before.level} -> ${levelling.after.level}`,
);
check(
  'levelling grants stat and skill points',
  levelling.after.skillPoints === 3 && levelling.after.statPoints === 15,
  `${levelling.after.statPoints} stat, ${levelling.after.skillPoints} skill`,
);
check(
  "the level-up reaches combat's own health pool",
  levelling.after.healthMax > levelling.before.healthMax &&
    levelling.after.healthMax === levelling.after.life,
  `${levelling.before.healthMax} -> ${levelling.after.healthMax}`,
);

/* -- 4. a skill point changes a combat number ------------------------------ */

const skill = await page.evaluate(() => {
  const rpg = window.__d2rim.ctx.services.get('rpg');
  const before = rpg.offense();
  const invested = rpg.investSkill('bash');
  const after = rpg.offense();
  return {
    invested,
    active: rpg.character.skills.active,
    points: rpg.character.skills.pointsIn('bash'),
    beforeMax: before.damage.physical?.max ?? 0,
    afterMax: after.damage.physical?.max ?? 0,
    beforeAr: before.attackRating,
    afterAr: after.attackRating,
  };
});
check('a skill point is spent', skill.invested && skill.points === 1, `bash ${skill.points}`);
check('investing selects the skill', skill.active === 'bash', String(skill.active));
check(
  'Bash raises the damage handed to the damage model',
  skill.afterMax > skill.beforeMax,
  `${skill.beforeMax} -> ${skill.afterMax}`,
);
check(
  'Bash raises the attack rating handed to the damage model',
  skill.afterAr > skill.beforeAr,
  `${skill.beforeAr} -> ${skill.afterAr}`,
);

/* -- 5. an equipped item changes a combat number --------------------------- */

const gear = await page.evaluate(() => {
  const rpg = window.__d2rim.ctx.services.get('rpg');
  // Items are generated through the loot system's own drop table, which is the
  // path the game itself uses — rather than by importing the generator, which
  // is not reachable by name from a production bundle.
  const loot = window.__d2rim.ctx.services.get('rpg.loot');

  const before = rpg.offense();
  const beforeLife = rpg.character.derived.maxLife;

  // Roll drops until something equippable turns up, then equip it.
  let equipped = null;
  for (let seed = 1; seed < 400 && equipped === null; seed++) {
    const rolled = loot.onEnemyDeath(9000 + seed, 'mage#0');
    for (const item of rolled.items) {
      if (!rpg.character.canEquip(item)) continue;
      // Insist on an item that carries at least one modifier. A plain white
      // item would satisfy "something got equipped" while proving nothing about
      // whether affixes reach the damage model.
      if (Object.keys(item.mods).length === 0) continue;
      rpg.character.acquire(item);
      if (rpg.character.equip(item).equipped) {
        equipped = { name: item.name, quality: item.quality, mods: item.mods };
        break;
      }
    }
  }
  loot.clear();

  const after = rpg.offense();
  return {
    equipped,
    beforeMax: before.damage.physical?.max ?? 0,
    afterMax: after.damage.physical?.max ?? 0,
    beforeAr: before.attackRating,
    afterAr: after.attackRating,
    beforeLife,
    afterLife: rpg.character.derived.maxLife,
    weapon: rpg.character.equipment.get('weapon')?.name ?? null,
  };
});
check(
  'a magic or better item was generated and equipped',
  gear.equipped !== null && Object.keys(gear.equipped.mods).length > 0,
  gear.equipped === null
    ? 'none'
    : `${gear.equipped.quality} ${gear.equipped.name} ${JSON.stringify(gear.equipped.mods)}`,
);
check(
  'the equipped item changed a combat number',
  gear.afterMax !== gear.beforeMax ||
    gear.afterAr !== gear.beforeAr ||
    gear.afterLife !== gear.beforeLife,
  `damage ${gear.beforeMax} -> ${gear.afterMax}, AR ${gear.beforeAr} -> ${gear.afterAr}, ` +
    `life ${gear.beforeLife} -> ${gear.afterLife}`,
);

/* -- 6. loot on the floor -------------------------------------------------- */

const loot = await page.evaluate(() => {
  const lootSystem = window.__d2rim.ctx.services.get('rpg.loot');
  const rpg = window.__d2rim.ctx.services.get('rpg');
  const combat = window.__d2rim.ctx.services.get('combat');
  const self = combat.self;
  const position = self.footPosition(new window.__d2rim.three.Vector3());

  lootSystem.clear();
  const gold = lootSystem.spawnGold(137, position.clone(), 0);
  const before = rpg.character.gold;
  const nearest = lootSystem.nearest();
  const picked = lootSystem.pickUpNearest();
  return {
    dropped: lootSystem.entries.length,
    label: gold.label,
    colour: gold.colour,
    hasMarker: gold.object !== null,
    nearestFound: nearest !== null,
    picked: picked !== null,
    goldBefore: before,
    goldAfter: rpg.character.gold,
    remaining: lootSystem.entries.length,
  };
});
check('loot lands on the floor with a marker', loot.hasMarker, loot.label);
check('the label carries a colour', /^#[0-9a-f]{6}$/i.test(loot.colour), loot.colour);
check('the nearest pickup is found', loot.nearestFound && loot.picked);
check(
  'picking up gold credits the character',
  loot.goldAfter === loot.goldBefore + 137 && loot.remaining === 0,
  `${loot.goldBefore} -> ${loot.goldAfter}`,
);

/* -- 7. the HUD ------------------------------------------------------------ */

await step(4);
const hud = await page.evaluate(() => {
  const root = document.getElementById('d2rim-rpg-hud');
  const text = root?.innerText ?? '';
  return {
    present: root !== null,
    text,
    orbs: document.getElementById('d2rim-hud') !== null,
  };
});
check('the RPG HUD is in the document', hud.present && hud.orbs);
check('the HUD shows the level and gold', /LVL \d+/.test(hud.text) && /gold/.test(hud.text), hud.text.replace(/\n/g, ' | ').slice(0, 120));
await shot('01-hud-camp');

/* -- 8. the inventory screen ----------------------------------------------- */

// Guarantee something is in the bag: run 2 of this harness equipped a body
// armour rather than a weapon, left the pack empty, and the drag assertion had
// nothing to pick up — a fixture that reported a failure the code did not have.
await page.evaluate(() => {
  const rpg = window.__d2rim.ctx.services.get('rpg');
  const loot = window.__d2rim.ctx.services.get('rpg.loot');
  for (let seed = 900; rpg.character.inventory.count < 2 && seed < 1400; seed++) {
    for (const item of loot.onEnemyDeath(seed, 'warrior#0').items) rpg.character.acquire(item);
  }
  loot.clear();
});
await page.evaluate(() => window.__d2rim.ctx.services.get('ui').open('inventory'));
await step(2);
const inventory = await page.evaluate(() => {
  const ui = window.__d2rim.ctx.services.get('ui');
  const grid = document.querySelector('[data-d2rim="inventory-grid"]');
  const rect = grid?.getBoundingClientRect();
  // Measure the panel by its own marker, not by walking up from the grid — the
  // first version of this check climbed two parents and landed on the panel's
  // right-hand *column*, which is not centred and never was supposed to be.
  const panel = document
    .querySelector('[data-d2rim="inventory-panel"]')
    ?.getBoundingClientRect();
  return {
    panelLeft: Math.round(panel?.left ?? -1),
    panelRight: Math.round(panel?.right ?? -1),
    panelTop: Math.round(panel?.top ?? -1),
    panelBottom: Math.round(panel?.bottom ?? -1),
    open: ui.isOpen('inventory'),
    inputDisabled: !window.__d2rim.ctx.input.enabled,
    gridPresent: grid !== null,
    gridWidth: Math.round(rect?.width ?? 0),
    gridHeight: Math.round(rect?.height ?? 0),
    itemNodes: grid?.children.length ?? 0,
    slots: document.querySelectorAll('[data-d2rim^="slot-"]').length,
    text: document.querySelector('#d2rim-ui')?.innerText.replace(/\n+/g, ' | ').slice(0, 200) ?? '',
  };
});
check('the inventory screen opens', inventory.open && inventory.gridPresent);
check('opening it takes the keyboard away from the game', inventory.inputDisabled);
check(
  'the grid is ten by four cells',
  // 34 px cells plus the panel's 1 px border on each side.
  inventory.gridWidth === 342 && inventory.gridHeight === 138,
  `${inventory.gridWidth}x${inventory.gridHeight}`,
);
check('ten equipment slots are drawn', inventory.slots === 10, String(inventory.slots));
check(
  'the panel is centred in the viewport',
  Math.abs(inventory.panelLeft - (1280 - inventory.panelRight)) <= 2 &&
    Math.abs(inventory.panelTop - (720 - inventory.panelBottom)) <= 2,
  `x ${inventory.panelLeft}..${inventory.panelRight}, y ${inventory.panelTop}..${inventory.panelBottom}`,
);
check('items are drawn in the grid', inventory.itemNodes > 0, `${inventory.itemNodes} item nodes`);
await shot('02-inventory');

/* -- 9. drag and drop ------------------------------------------------------ */

const drag = await page.evaluate(async () => {
  const rpg = window.__d2rim.ctx.services.get('rpg');
  const inventoryGrid = rpg.character.inventory;
  const first = inventoryGrid.placements[0];
  if (first === undefined) return { ran: false };

  const grid = document.querySelector('[data-d2rim="inventory-grid"]');
  const rect = grid.getBoundingClientRect();
  const cell = 34;
  const from = {
    x: rect.left + first.x * cell + 6,
    y: rect.top + first.y * cell + 6,
  };
  // Somewhere the item is not: the far right of the grid.
  const toCell = { x: 10 - first.item.width, y: 4 - first.item.height };
  const to = { x: rect.left + toCell.x * cell + 6, y: rect.top + toCell.y * cell + 6 };

  const send = (type, point) =>
    grid.dispatchEvent(
      new PointerEvent(type, {
        clientX: point.x,
        clientY: point.y,
        bubbles: true,
        button: 0,
      }),
    );

  send('pointerdown', from); // lift
  send('pointerdown', to); // place
  const placement = inventoryGrid.placementOf(first.item);
  return {
    ran: true,
    uid: first.item.uid,
    fromCell: { x: first.x, y: first.y },
    wanted: toCell,
    got: placement === null ? null : { x: placement.x, y: placement.y },
  };
});
check(
  'an item can be dragged to a new cell',
  drag.ran && drag.got !== null && drag.got.x === drag.wanted.x && drag.got.y === drag.wanted.y,
  drag.ran
    ? `${JSON.stringify(drag.fromCell)} -> ${JSON.stringify(drag.got)} (wanted ${JSON.stringify(drag.wanted)})`
    : 'no item to drag',
);
await shot('03-inventory-after-drag');

/* -- 10. the skill tree ---------------------------------------------------- */

await page.evaluate(() => {
  const ui = window.__d2rim.ctx.services.get('ui');
  ui.close('inventory');
  ui.open('skills');
});
await step(2);
const tree = await page.evaluate(() => {
  const cards = Array.from(document.querySelectorAll('[data-d2rim^="skill-"]'))
    .filter((node) => !node.dataset.d2rim.startsWith('skill-tier'))
    .map((node) => node.dataset.d2rim);
  return {
    open: window.__d2rim.ctx.services.get('ui').isOpen('skills'),
    cards,
    text: document.querySelector('#d2rim-ui')?.innerText.replace(/\n+/g, ' | ').slice(0, 240) ?? '',
  };
});
check('the skill tree screen opens with every skill', tree.open && tree.cards.length === 4, tree.cards.join(', '));
check('it names the points available', /skill point/i.test(tree.text), tree.text.slice(0, 90));
await shot('04-skill-tree');

/* -- 11. the quest, through Akara's dialogue ------------------------------- */

await page.evaluate(() => window.__d2rim.ctx.services.get('ui').closeAll());
await step(2);

const opened = await page.evaluate(() => {
  const npcSystem = window.__d2rim.ctx.services.get('quest.npcs');
  const view = npcSystem.talkTo('akara');
  return {
    speaker: view?.speaker ?? null,
    node: view?.nodeId ?? null,
    choices: view?.choices.map((c) => c.id) ?? [],
  };
});
check('Akara opens a conversation', opened.speaker === 'Akara' && opened.node === 'greeting', opened.choices.join(', '));
await step(2);
await shot('05-dialogue-akara');

const accepted = await page.evaluate(() => {
  const npcSystem = window.__d2rim.ctx.services.get('quest.npcs');
  const quests = window.__d2rim.ctx.services.get('quest.system');
  npcSystem.chooseById('ask-evil');
  const offered = quests.state('den-of-evil');
  npcSystem.chooseById('accept');
  const active = quests.state('den-of-evil');
  npcSystem.endDialogue();
  return { offered, active, status: quests.status('den-of-evil') };
});
check(
  'the quest is offered and accepted through dialogue alone',
  accepted.offered === 'offered' && accepted.active === 'active',
  `${accepted.offered} -> ${accepted.active}`,
);
await step(2);
const tracker = await page.evaluate(
  () => document.getElementById('d2rim-rpg-hud')?.innerText.replace(/\n+/g, ' | ') ?? '',
);
check('the tracker shows the quest', /Den of Evil/i.test(tracker), tracker.slice(0, 140));
await shot('06-hud-quest-tracked');

/* -- 12. into the Den ------------------------------------------------------ */

process.stdout.write('  ...travelling to the Den of Evil\n');
await page.evaluate(() => window.__d2rim.zones.travelTo('denOfEvil', null));
await page.waitForFunction(() => window.__d2rim.zones.active?.zoneId === 'denOfEvil', null, {
  timeout: 180_000,
});
await step(6);

const den = await page.evaluate(() => {
  const quests = window.__d2rim.ctx.services.get('quest.system');
  const module = window.__d2rim.engine.getModule('quest.denOfEvil');
  const zone = window.__d2rim.zones.active;
  return {
    zoneId: zone?.zoneId ?? null,
    spawnPoints: zone?.layout?.spawnPoints.length ?? 0,
    enemySpawns: zone?.enemySpawns?.length ?? 0,
    countResolved: module?.countResolved ?? false,
    monsterCount: module?.monsterCount ?? 0,
    status: quests.status('den-of-evil'),
  };
});
check('the Den loaded', den.zoneId === 'denOfEvil');
check(
  'the quest adopted the generator’s real spawn count',
  den.countResolved && den.monsterCount === den.enemySpawns && den.enemySpawns > 0,
  `${den.monsterCount} monsters (layout says ${den.spawnPoints})`,
);
check(
  'the reach objective completed on arrival',
  den.status?.objectives?.[0]?.complete === true,
  den.status?.objectives?.[0]?.description,
);
check(
  'the kill objective counts against the real total',
  den.status?.objectives?.[1]?.required === den.enemySpawns,
  `0 of ${den.status?.objectives?.[1]?.required}`,
);

const clearing = await page.evaluate((total) => {
  const module = window.__d2rim.engine.getModule('quest.denOfEvil');
  const quests = window.__d2rim.ctx.services.get('quest.system');
  const marks = [];
  for (let i = 0; i < total; i++) {
    module.reportKill();
    if (i === Math.floor(total / 2)) {
      marks.push(quests.status('den-of-evil').objectives[1].current);
    }
  }
  return {
    midway: marks[0] ?? 0,
    final: quests.status('den-of-evil').objectives[1].current,
    state: quests.state('den-of-evil'),
  };
}, den.enemySpawns);
check(
  'progress is truthful part-way through',
  clearing.midway === Math.floor(den.enemySpawns / 2) + 1,
  `${clearing.midway} of ${den.enemySpawns}`,
);
check(
  'clearing the Den completes the quest',
  clearing.state === 'complete' && clearing.final === den.enemySpawns,
  `${clearing.final} of ${den.enemySpawns}, state ${clearing.state}`,
);
await step(2);
await shot('07-den-quest-complete');

/* -- 13. back to Akara for the reward -------------------------------------- */

process.stdout.write('  ...travelling back to the camp\n');
await page.evaluate(() => window.__d2rim.zones.travelTo('encampment', null));
await page.waitForFunction(() => window.__d2rim.zones.active?.zoneId === 'encampment', null, {
  timeout: 180_000,
});
await page
  .waitForFunction(() => window.__d2rim.ctx.services.get('quest.npcs').npcs.length >= 5, null, {
    timeout: 90_000,
  })
  .catch(() => undefined);
await step(6);

const turnIn = await page.evaluate(() => {
  const npcSystem = window.__d2rim.ctx.services.get('quest.npcs');
  const quests = window.__d2rim.ctx.services.get('quest.system');
  const rpg = window.__d2rim.ctx.services.get('rpg');

  const before = {
    skillPoints: rpg.character.stats.skillPoints,
    gold: rpg.character.gold,
    experience: rpg.character.stats.experience,
  };
  const view = npcSystem.talkTo('akara');
  const node = view?.nodeId ?? null;
  npcSystem.chooseById('turn-in');
  const after = {
    skillPoints: rpg.character.stats.skillPoints,
    gold: rpg.character.gold,
    experience: rpg.character.stats.experience,
  };
  const state = quests.state('den-of-evil');
  npcSystem.endDialogue();
  return { node, before, after, state };
});
check(
  'Akara greets a finished quest differently',
  turnIn.node === 'complete',
  String(turnIn.node),
);
check(
  'turning in pays the skill point',
  turnIn.after.skillPoints === turnIn.before.skillPoints + 1,
  `${turnIn.before.skillPoints} -> ${turnIn.after.skillPoints}`,
);
check(
  'turning in pays the gold and experience',
  turnIn.after.gold > turnIn.before.gold && turnIn.after.experience > turnIn.before.experience,
  `${turnIn.before.gold}g -> ${turnIn.after.gold}g, ${turnIn.before.experience}xp -> ${turnIn.after.experience}xp`,
);
check('the quest reached "rewarded"', turnIn.state === 'rewarded', turnIn.state);

/* -- 14. the vendor -------------------------------------------------------- */

const vendor = await page.evaluate(() => {
  const rpg = window.__d2rim.ctx.services.get('rpg');
  const ui = window.__d2rim.ctx.services.get('ui');
  const charsi = rpg.vendors.get('charsi');
  charsi.refresh(7);
  rpg.character.addGold(5000);

  const before = { gold: rpg.character.gold, stock: charsi.stock.length, bag: rpg.character.inventory.count };
  const target = charsi.stock[0];
  const price = charsi.priceToBuy(target);
  const result = charsi.buy(0, rpg.character);

  window.__d2rim.ctx.events.emit('npc:vendor', { npcId: 'charsi', vendorId: 'charsi' });
  return {
    open: ui.isOpen('vendor'),
    result: result.result,
    name: target.name,
    price,
    before,
    after: { gold: rpg.character.gold, stock: charsi.stock.length, bag: rpg.character.inventory.count },
  };
});
check(
  'buying moves gold one way and the item the other',
  vendor.result === 'ok' &&
    vendor.after.gold === vendor.before.gold - vendor.price &&
    vendor.after.bag === vendor.before.bag + 1 &&
    vendor.after.stock === vendor.before.stock - 1,
  `${vendor.name} for ${vendor.price}g`,
);
check('the vendor screen opens on request', vendor.open);
await step(2);
await shot('08-vendor');

/* -- 15. the pause menu ---------------------------------------------------- */

await page.evaluate(() => {
  const ui = window.__d2rim.ctx.services.get('ui');
  ui.closeAll();
  ui.open('menu');
});
await step(2);
const menu = await page.evaluate(() => ({
  open: window.__d2rim.ctx.services.get('ui').isOpen('menu'),
  timeScale: window.__d2rim.ctx.time.scale,
  text: document.querySelector('#d2rim-ui')?.innerText.replace(/\n+/g, ' | ').slice(0, 200) ?? '',
}));
check('the pause menu opens and really pauses', menu.open && menu.timeScale === 0, `scale ${menu.timeScale}`);
check(
  'the quest log stops tracking a rewarded quest',
  /No active quests/i.test(menu.text),
  menu.text.slice(0, 120),
);
await shot('09-pause-menu');

/* -- 16. save and load through real IndexedDB ------------------------------ */

const save = await page.evaluate(async () => {
  const rpg = window.__d2rim.ctx.services.get('rpg');
  const before = JSON.stringify(rpg.character.toJSON());
  const quests = JSON.stringify(rpg.quests.toJSON());

  await rpg.save('verify-slot');

  // Wreck the live state so a load that does nothing cannot pass.
  rpg.character.addGold(99_999);
  rpg.character.stats.grantSkillPoints(50);
  const wrecked = JSON.stringify(rpg.character.toJSON());

  const loaded = await rpg.load('verify-slot');
  const after = JSON.stringify(rpg.character.toJSON());
  const questsAfter = JSON.stringify(rpg.quests.toJSON());

  const list = await rpg.store.list();

  // Read the record straight out of the browser's own IndexedDB, bypassing the
  // store class entirely. A minified production bundle renames the class, so
  // `constructor.name` proves nothing; the bytes being in the real database
  // prove everything.
  const raw = await new Promise((resolve) => {
    const open = indexedDB.open('d2rim', 1);
    open.onerror = () => resolve(null);
    open.onsuccess = () => {
      const db = open.result;
      const request = db.transaction('saves', 'readonly').objectStore('saves').get('verify-slot');
      request.onsuccess = () => resolve(request.result ?? null);
      request.onerror = () => resolve(null);
    };
  });

  return {
    loaded,
    matches: after === before,
    wreckedDiffers: wrecked !== before,
    questsMatch: questsAfter === quests,
    rawSlot: raw?.slot ?? null,
    rawIsJson: typeof raw?.json === 'string' && raw.json.startsWith('{'),
    rawMatches: typeof raw?.json === 'string' && JSON.parse(raw.json).character.gold === JSON.parse(before).gold,
    slots: list.map((entry) => entry.slot),
    level: JSON.parse(after).stats.level,
  };
});
check(
  'the save is really in the browser IndexedDB',
  save.rawSlot === 'verify-slot' && save.rawIsJson && save.rawMatches,
  `slot ${save.rawSlot}, json ${save.rawIsJson}, contents match ${save.rawMatches}`,
);
check('the load actually read something back', save.loaded && save.wreckedDiffers);
check('the character round-trips exactly', save.matches, `level ${save.level}`);
check('the quest state round-trips exactly', save.questsMatch);
check('the slot is listed', save.slots.includes('verify-slot'), save.slots.join(', '));

/* -- 17. the UI does not capture input when closed ------------------------- */

await page.evaluate(() => window.__d2rim.ctx.services.get('ui').closeAll());
await step(2);
const closed = await page.evaluate(() => {
  const root = document.getElementById('d2rim-ui');
  const centre = document.elementFromPoint(window.innerWidth / 2, window.innerHeight / 2);
  return {
    pointerEvents: root === null ? 'missing' : getComputedStyle(root).pointerEvents,
    inputEnabled: window.__d2rim.ctx.input.enabled,
    hitTest: centre?.tagName ?? 'none',
    timeScale: window.__d2rim.ctx.time.scale,
  };
});
check('the closed overlay does not swallow the pointer', closed.pointerEvents === 'none', closed.pointerEvents);
check('a click in the middle of the screen reaches the canvas', closed.hitTest === 'CANVAS', closed.hitTest);
check('the keyboard is handed back to the game', closed.inputEnabled);
check('closing the menu restores the time scale', closed.timeScale === 1, `scale ${closed.timeScale}`);

/* -- report ---------------------------------------------------------------- */

const errors = logs.filter((line) => line.startsWith('pageerror') || line.startsWith('error:'));
check('no page errors', errors.length === 0, errors.slice(0, 3).join(' / '));

console.log(`\n${failures.length === 0 ? 'ALL PASS' : `${failures.length} FAILURES`}`);
for (const failure of failures) console.log(`  - ${failure}`);

await browser.close();
server.kill();
process.exit(failures.length === 0 ? 0 : 1);
