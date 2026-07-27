/**
 * Save and load.
 *
 * The round-trip fixture is deliberately *rich*: a levelled character with
 * spent stat points, invested skills, a chosen active skill, a non-trivial
 * inventory layout including a swapped-in rare, worn equipment, damaged
 * durability, gold, and a quest part-way through its objectives. A save test
 * built from a fresh character passes against an implementation that persists
 * nothing at all, because a fresh character *is* the default.
 *
 * Both stores serialise through JSON, so a payload that would not survive
 * IndexedDB fails here too.
 */

import { describe, expect, it } from 'vitest';

import { Character } from '../src/rpg/Character';
import { generateItem } from '../src/rpg/ItemGenerator';
import {
  AUTOSAVE_SLOT,
  MemorySaveStore,
  SAVE_VERSION,
  applySave,
  captureSave,
  decodeSave,
  describeSave,
  encodeSave,
  indexedDbAvailable,
  isSaveData,
  createSaveStore,
} from '../src/rpg/SaveGame';
import { QuestSystem } from '../src/quest/QuestSystem';
import { createDenOfEvilQuest, DEN_OF_EVIL_QUEST_ID } from '../src/quest/DenOfEvil';

/** A character that has actually been played with. */
function playedCharacter(): Character {
  const character = new Character({ name: 'Bul-Kathos', gold: 743 });
  character.addExperience(14_175); // level 6
  character.spendStatPoint('strength', 12);
  character.spendStatPoint('vitality', 8);
  character.investSkillPoint('bash');
  character.investSkillPoint('bash');
  character.investSkillPoint('howl');
  character.setActiveSkill('howl');

  const sword = generateItem({ seed: 88, itemLevel: 5, baseId: 'short-sword', quality: 'magic' });
  const armour = generateItem({ seed: 99, itemLevel: 3, baseId: 'leather-armor', quality: 'normal' });
  const ring = generateItem({ seed: 12, itemLevel: 4, baseId: 'ring', quality: 'magic' });
  const spare = generateItem({ seed: 33, itemLevel: 5, baseId: 'buckler', quality: 'magic' });

  character.acquire(sword);
  character.acquire(armour);
  character.acquire(ring);
  character.equip(sword);
  character.equip(armour);
  character.equip(ring);

  // A non-default layout: put the spare somewhere other than first fit.
  character.inventory.place(spare, 6, 1);
  // And some wear, so durability is not at its default either.
  sword.durability = sword.maxDurability - 5;
  character.touch();
  return character;
}

function playedQuests(): QuestSystem {
  const quests = new QuestSystem();
  quests.register(createDenOfEvilQuest(17));
  quests.offer(DEN_OF_EVIL_QUEST_ID);
  quests.accept(DEN_OF_EVIL_QUEST_ID);
  quests.notifyReach('den-of-evil');
  quests.notifyKill('den-monster', 9);
  return quests;
}

describe('the fixture itself', () => {
  it('is not the default character, so the round trip has something to prove', () => {
    const played = playedCharacter();
    const fresh = new Character();
    expect(JSON.stringify(played.toJSON())).not.toBe(JSON.stringify(fresh.toJSON()));
    expect(played.stats.level).toBe(6);
    expect(played.equipment.items().length).toBeGreaterThanOrEqual(3);
    expect(played.inventory.count).toBeGreaterThan(0);
    expect(played.skills.totalInvested).toBe(3);
    expect(played.gold).toBe(743);
  });
});

describe('capture and apply', () => {
  it('captures every part of the game state', () => {
    const data = captureSave({
      character: playedCharacter(),
      quests: playedQuests(),
      zoneId: 'denOfEvil',
      entryPoint: 'cave-mouth',
      playTime: 912.6,
      now: () => 1_700_000_000_000,
    });
    expect(data.version).toBe(SAVE_VERSION);
    expect(data.savedAt).toBe(1_700_000_000_000);
    expect(data.playTime).toBe(913);
    expect(data.zoneId).toBe('denOfEvil');
    expect(data.entryPoint).toBe('cave-mouth');
    expect(data.character.stats.level).toBe(6);
    expect(data.quests.quests.length).toBeGreaterThan(0);
  });

  it('mutates nothing it captures', () => {
    const character = playedCharacter();
    const before = JSON.stringify(character.toJSON());
    captureSave({ character, quests: playedQuests(), zoneId: 'encampment' });
    expect(JSON.stringify(character.toJSON())).toBe(before);
  });

  it('restores a character to the exact same sheet', () => {
    const source = playedCharacter();
    const quests = playedQuests();
    const data = captureSave({ character: source, quests, zoneId: 'bloodMoor' });

    const target = new Character();
    const targetQuests = new QuestSystem();
    targetQuests.register(createDenOfEvilQuest(17));
    applySave(JSON.parse(JSON.stringify(data)), target, targetQuests);

    expect(target.toJSON()).toEqual(source.toJSON());
    expect(targetQuests.toJSON()).toEqual(quests.toJSON());
    // And the derived sheet — the thing combat actually reads — agrees.
    expect(target.derived).toEqual(source.derived);
  });

  it('restores the exact inventory layout, not merely the same items', () => {
    const source = playedCharacter();
    const data = captureSave({ character: source, quests: new QuestSystem(), zoneId: 'x' });

    const target = new Character();
    applySave(data, target, new QuestSystem());
    for (const placement of source.inventory.placements) {
      const restored = target.inventory.itemAt(placement.x, placement.y);
      expect(restored, `${placement.x},${placement.y}`).not.toBeNull();
      expect(restored?.uid).toBe(placement.item.uid);
    }
  });

  it('restores durability, so a worn weapon stays worn', () => {
    const source = playedCharacter();
    const worn = source.equipment.get('weapon');
    expect(worn?.durability).toBeLessThan(worn?.maxDurability ?? 0);

    const target = new Character();
    applySave(captureSave({ character: source, quests: new QuestSystem(), zoneId: 'x' }), target, new QuestSystem());
    expect(target.equipment.get('weapon')?.durability).toBe(worn?.durability);
  });
});

describe('encoding', () => {
  it('survives a JSON round trip byte for byte', () => {
    const data = captureSave({
      character: playedCharacter(),
      quests: playedQuests(),
      zoneId: 'denOfEvil',
      now: () => 1234,
    });
    const decoded = decodeSave(encodeSave(data));
    expect(decoded).not.toBeNull();
    expect(encodeSave(decoded as never)).toBe(encodeSave(data));
  });

  it('rejects unreadable text and foreign objects', () => {
    expect(decodeSave('not json at all')).toBeNull();
    expect(decodeSave('{"hello":"world"}')).toBeNull();
    expect(isSaveData(null)).toBe(false);
    expect(isSaveData({ version: SAVE_VERSION + 5 })).toBe(false);
  });

  it('summarises a slot without loading it', () => {
    const data = captureSave({
      character: playedCharacter(),
      quests: playedQuests(),
      zoneId: 'denOfEvil',
      playTime: 60,
      now: () => 555,
    });
    expect(describeSave('slot-1', data)).toEqual({
      slot: 'slot-1',
      savedAt: 555,
      playTime: 60,
      characterName: 'Bul-Kathos',
      level: 6,
      zoneId: 'denOfEvil',
    });
  });
});

describe('MemorySaveStore', () => {
  it('round-trips through the store exactly', async () => {
    const store = new MemorySaveStore();
    const source = playedCharacter();
    const quests = playedQuests();
    const data = captureSave({ character: source, quests, zoneId: 'denOfEvil' });

    await store.write(AUTOSAVE_SLOT, data);
    const read = await store.read(AUTOSAVE_SLOT);
    expect(read).not.toBeNull();

    const target = new Character();
    const targetQuests = new QuestSystem();
    targetQuests.register(createDenOfEvilQuest(17));
    applySave(read as never, target, targetQuests);

    expect(target.toJSON()).toEqual(source.toJSON());
    expect(targetQuests.toJSON()).toEqual(quests.toJSON());
  });

  it('serialises rather than holding a reference', async () => {
    // If the store held the object, mutating the character afterwards would
    // change what comes back out — and every round-trip test would pass for
    // free.
    const store = new MemorySaveStore();
    const character = playedCharacter();
    await store.write('s', captureSave({ character, quests: new QuestSystem(), zoneId: 'x' }));
    character.addGold(5000);

    const read = await store.read('s');
    expect(read?.character.gold).toBe(743);
  });

  it('returns null for a slot that was never written', async () => {
    const store = new MemorySaveStore();
    expect(await store.read('nope')).toBeNull();
  });

  it('lists slots newest first and deletes them', async () => {
    const store = new MemorySaveStore();
    const character = playedCharacter();
    const quests = new QuestSystem();
    await store.write('old', captureSave({ character, quests, zoneId: 'a', now: () => 100 }));
    await store.write('new', captureSave({ character, quests, zoneId: 'b', now: () => 200 }));

    const list = await store.list();
    expect(list.map((entry) => entry.slot)).toEqual(['new', 'old']);
    expect(await store.remove('old')).toBe(true);
    expect(await store.list()).toHaveLength(1);
    expect(store.size).toBe(1);
  });

  it('overwrites a slot rather than appending to it', async () => {
    const store = new MemorySaveStore();
    const character = playedCharacter();
    const quests = new QuestSystem();
    await store.write('s', captureSave({ character, quests, zoneId: 'a' }));
    character.addGold(100);
    await store.write('s', captureSave({ character, quests, zoneId: 'b' }));

    expect(store.size).toBe(1);
    expect((await store.read('s'))?.zoneId).toBe('b');
    expect((await store.read('s'))?.character.gold).toBe(843);
  });
});

describe('store selection', () => {
  it('falls back to memory where IndexedDB does not exist', () => {
    // The unit-test environment is Node, so this is the fallback path; in the
    // browser the drive harness exercises the IndexedDB one.
    expect(indexedDbAvailable()).toBe(false);
    expect(createSaveStore()).toBeInstanceOf(MemorySaveStore);
  });
});
