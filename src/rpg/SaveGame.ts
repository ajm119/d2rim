/**
 * @module rpg/SaveGame
 *
 * Persistence: the whole character — statistics, inventory layout, equipment,
 * skills, quest progress, gold and current zone — to IndexedDB, and back
 * identically.
 *
 * ### Round-trip exactness is the requirement
 *
 * Everything else here follows from it. The payload is a plain JSON document
 * with no class instances, no `Map`s and no `undefined`-valued keys, and both
 * store implementations serialise through `JSON.stringify`/`JSON.parse` — even
 * {@link MemorySaveStore}, which could trivially hold the object by reference.
 *
 * That last decision is the important one. A memory store that keeps a
 * reference makes the round-trip test pass unconditionally: it would pass with
 * a `Map` in the payload, with a `Date`, with a circular reference — with
 * anything at all, right up until the real IndexedDB path is exercised by a
 * player. Forcing the same serialisation through both means the fast test and
 * the real thing can only agree or both fail.
 *
 * ### IndexedDB rather than localStorage
 *
 * localStorage is synchronous — it blocks the frame — and caps out around 5 MB
 * of UTF-16. IndexedDB is asynchronous, an order of magnitude larger, and is
 * the right home for a save that will grow a stash and a mercenary later.
 *
 * ### Versioning
 *
 * {@link SAVE_VERSION} is written into every document and checked on read. A
 * save from a future version is refused rather than partially applied, because
 * a half-loaded character is far worse than a missing one.
 */

import type { CharacterSnapshot } from './Character';
import type { Character } from './Character';
import type { QuestSnapshot, QuestSystem } from '../quest/QuestSystem';

/* -------------------------------------------------------------------------- */
/* Document                                                                    */
/* -------------------------------------------------------------------------- */

/** Bumped whenever the payload shape changes incompatibly. */
export const SAVE_VERSION = 1;

export interface SaveData {
  readonly version: number;
  /** Epoch milliseconds. Written by {@link captureSave}. */
  readonly savedAt: number;
  /** Seconds of play. Displayed on the load screen. */
  readonly playTime: number;
  readonly character: CharacterSnapshot;
  readonly quests: QuestSnapshot;
  /** Zone the player was standing in. */
  readonly zoneId: string;
  /** Entry point to arrive at when the save is loaded. */
  readonly entryPoint: string | null;
}

/** Summary of a slot, for a load menu. */
export interface SaveSlotInfo {
  readonly slot: string;
  readonly savedAt: number;
  readonly playTime: number;
  readonly characterName: string;
  readonly level: number;
  readonly zoneId: string;
}

export interface CaptureOptions {
  readonly character: Character;
  readonly quests: QuestSystem;
  readonly zoneId: string;
  readonly entryPoint?: string | null;
  readonly playTime?: number;
  /** Injectable clock, so a test can assert on `savedAt`. */
  readonly now?: () => number;
}

/** Snapshot the live game into a document. Pure — mutates nothing. */
export function captureSave(options: CaptureOptions): SaveData {
  const now = options.now ?? (() => Date.now());
  return {
    version: SAVE_VERSION,
    savedAt: now(),
    playTime: Math.max(0, Math.round(options.playTime ?? 0)),
    character: options.character.toJSON(),
    quests: options.quests.toJSON(),
    zoneId: options.zoneId,
    entryPoint: options.entryPoint ?? null,
  };
}

/**
 * Apply a document to the live game.
 *
 * Quest *definitions* must already be registered — they are code, and a save
 * that could resurrect a deleted quest is a save that can wedge the game.
 */
export function applySave(data: SaveData, character: Character, quests: QuestSystem): void {
  character.load(data.character);
  quests.load(data.quests);
}

/** Summarise a document without loading it. */
export function describeSave(slot: string, data: SaveData): SaveSlotInfo {
  return {
    slot,
    savedAt: data.savedAt,
    playTime: data.playTime,
    characterName: data.character.name,
    level: data.character.stats.level,
    zoneId: data.zoneId,
  };
}

/**
 * Whether a parsed object is a save document this build can read.
 *
 * Structural rather than a version equality check, because a save file is
 * untrusted input: the store's job is to reject nonsense, not to trust that
 * whatever came out of the database has the shape it had going in.
 */
export function isSaveData(value: unknown): value is SaveData {
  if (value === null || typeof value !== 'object') return false;
  const candidate = value as Partial<SaveData>;
  return (
    typeof candidate.version === 'number' &&
    candidate.version <= SAVE_VERSION &&
    typeof candidate.savedAt === 'number' &&
    typeof candidate.zoneId === 'string' &&
    candidate.character !== undefined &&
    typeof candidate.character === 'object' &&
    candidate.quests !== undefined &&
    typeof candidate.quests === 'object'
  );
}

/* -------------------------------------------------------------------------- */
/* Stores                                                                      */
/* -------------------------------------------------------------------------- */

/** Where saves live. Two implementations: IndexedDB and memory. */
export interface SaveStore {
  read(slot: string): Promise<SaveData | null>;
  write(slot: string, data: SaveData): Promise<void>;
  list(): Promise<SaveSlotInfo[]>;
  remove(slot: string): Promise<boolean>;
  close(): void;
}

/** Serialise, and refuse anything that will not survive the trip. */
export function encodeSave(data: SaveData): string {
  const json = JSON.stringify(data);
  if (json === undefined) throw new Error('[SaveGame] the save payload is not serialisable');
  return json;
}

/** Parse and validate. Returns null for anything unreadable. */
export function decodeSave(json: string): SaveData | null {
  try {
    const parsed: unknown = JSON.parse(json);
    return isSaveData(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

/**
 * In-memory store that serialises exactly as the real one does.
 *
 * Used by tests and by the headless drive harness. See the module header for
 * why it round-trips through JSON rather than holding the object.
 */
export class MemorySaveStore implements SaveStore {
  readonly #slots = new Map<string, string>();

  read(slot: string): Promise<SaveData | null> {
    const json = this.#slots.get(slot);
    return Promise.resolve(json === undefined ? null : decodeSave(json));
  }

  write(slot: string, data: SaveData): Promise<void> {
    this.#slots.set(slot, encodeSave(data));
    return Promise.resolve();
  }

  async list(): Promise<SaveSlotInfo[]> {
    const out: SaveSlotInfo[] = [];
    for (const [slot, json] of this.#slots) {
      const data = decodeSave(json);
      if (data !== null) out.push(describeSave(slot, data));
    }
    out.sort((a, b) => b.savedAt - a.savedAt);
    return Promise.resolve(out);
  }

  remove(slot: string): Promise<boolean> {
    return Promise.resolve(this.#slots.delete(slot));
  }

  close(): void {
    /* nothing to release */
  }

  /** Slot count. Diagnostics only. */
  get size(): number {
    return this.#slots.size;
  }
}

/* -------------------------------------------------------------------------- */
/* IndexedDB                                                                   */
/* -------------------------------------------------------------------------- */

export const SAVE_DB_NAME = 'd2rim';
export const SAVE_DB_VERSION = 1;
export const SAVE_STORE_NAME = 'saves';

interface StoredRecord {
  slot: string;
  json: string;
  savedAt: number;
}

/** Promisify one IndexedDB request. */
function request<T>(source: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    source.onsuccess = () => resolve(source.result);
    source.onerror = () => reject(source.error ?? new Error('[SaveGame] IndexedDB request failed'));
  });
}

/** Whether IndexedDB exists in this environment. Node and some kiosks lack it. */
export function indexedDbAvailable(): boolean {
  return typeof indexedDB !== 'undefined';
}

/**
 * The real store.
 *
 * One object store keyed by slot name, holding the serialised document and a
 * denormalised `savedAt` so that listing slots does not have to parse every
 * save.
 */
export class IndexedDbSaveStore implements SaveStore {
  readonly #name: string;
  #db: IDBDatabase | null = null;
  #opening: Promise<IDBDatabase> | null = null;

  constructor(databaseName: string = SAVE_DB_NAME) {
    this.#name = databaseName;
  }

  async read(slot: string): Promise<SaveData | null> {
    const db = await this.#open();
    const transaction = db.transaction(SAVE_STORE_NAME, 'readonly');
    const record = await request<StoredRecord | undefined>(
      transaction.objectStore(SAVE_STORE_NAME).get(slot) as IDBRequest<StoredRecord | undefined>,
    );
    if (record === undefined) return null;
    return decodeSave(record.json);
  }

  async write(slot: string, data: SaveData): Promise<void> {
    const json = encodeSave(data);
    const db = await this.#open();
    const transaction = db.transaction(SAVE_STORE_NAME, 'readwrite');
    const record: StoredRecord = { slot, json, savedAt: data.savedAt };
    await request(transaction.objectStore(SAVE_STORE_NAME).put(record));
    // The write is not durable until the transaction commits, and a caller that
    // awaits `write` and then reloads the page must not lose the save.
    await new Promise<void>((resolve, reject) => {
      transaction.oncomplete = () => resolve();
      transaction.onabort = () =>
        reject(transaction.error ?? new Error('[SaveGame] save transaction aborted'));
      transaction.onerror = () =>
        reject(transaction.error ?? new Error('[SaveGame] save transaction failed'));
    });
  }

  async list(): Promise<SaveSlotInfo[]> {
    const db = await this.#open();
    const transaction = db.transaction(SAVE_STORE_NAME, 'readonly');
    const records = await request<StoredRecord[]>(
      transaction.objectStore(SAVE_STORE_NAME).getAll() as IDBRequest<StoredRecord[]>,
    );
    const out: SaveSlotInfo[] = [];
    for (const record of records) {
      const data = decodeSave(record.json);
      if (data !== null) out.push(describeSave(record.slot, data));
    }
    out.sort((a, b) => b.savedAt - a.savedAt);
    return out;
  }

  async remove(slot: string): Promise<boolean> {
    const db = await this.#open();
    const transaction = db.transaction(SAVE_STORE_NAME, 'readwrite');
    await request(transaction.objectStore(SAVE_STORE_NAME).delete(slot));
    return true;
  }

  close(): void {
    this.#db?.close();
    this.#db = null;
    this.#opening = null;
  }

  #open(): Promise<IDBDatabase> {
    const db = this.#db;
    if (db !== null) return Promise.resolve(db);
    const opening = this.#opening;
    if (opening !== null) return opening;
    if (!indexedDbAvailable()) {
      return Promise.reject(new Error('[SaveGame] IndexedDB is not available here'));
    }

    const promise = new Promise<IDBDatabase>((resolve, reject) => {
      const open = indexedDB.open(this.#name, SAVE_DB_VERSION);
      open.onupgradeneeded = () => {
        const database = open.result;
        if (!database.objectStoreNames.contains(SAVE_STORE_NAME)) {
          database.createObjectStore(SAVE_STORE_NAME, { keyPath: 'slot' });
        }
      };
      open.onsuccess = () => {
        this.#db = open.result;
        resolve(open.result);
      };
      open.onerror = () =>
        reject(open.error ?? new Error('[SaveGame] could not open the save database'));
    });
    this.#opening = promise;
    return promise;
  }
}

/** The store this build should use: IndexedDB when it exists, memory otherwise. */
export function createSaveStore(databaseName: string = SAVE_DB_NAME): SaveStore {
  return indexedDbAvailable() ? new IndexedDbSaveStore(databaseName) : new MemorySaveStore();
}

/** The slot the game autosaves into. */
export const AUTOSAVE_SLOT = 'autosave';
