/**
 * @module quest/CampDialogue
 *
 * The scripts for the five Rogue Encampment NPCs, written in the format
 * `quest/Dialogue` defines.
 *
 * Content rather than machinery: every branch here is data, and the only thing
 * this file knows how to do is name a quest id and a state. Akara is the one
 * that matters — she has five distinct openings depending on where the Den of
 * Evil stands, which is the whole reason the entry-rule mechanism exists — and
 * the other four are shorter because their jobs are shorter.
 *
 * Lines are D2-flavoured rather than transcribed. The point is that the player
 * can tell from Akara's first sentence whether they have a quest, are on one,
 * or have finished it, without opening the log.
 */

import { DEN_OF_EVIL_QUEST_ID } from './DenOfEvil';
import { whenQuest, type DialogueTree } from './Dialogue';

/* -------------------------------------------------------------------------- */
/* Akara — quest giver and healer                                              */
/* -------------------------------------------------------------------------- */

export const AKARA_DIALOGUE: DialogueTree = {
  id: 'akara',
  npcId: 'akara',
  speaker: 'Akara',
  // Entry rules in lifecycle order. The last is unconditional, so there is no
  // quest state in which Akara has nothing to say — `validateDialogueTree`
  // asserts exactly that.
  entry: [
    { node: 'complete', condition: whenQuest(DEN_OF_EVIL_QUEST_ID, 'complete') },
    { node: 'turned-in', condition: whenQuest(DEN_OF_EVIL_QUEST_ID, 'turnedIn', 'rewarded') },
    { node: 'active', condition: whenQuest(DEN_OF_EVIL_QUEST_ID, 'active') },
    { node: 'offer', condition: whenQuest(DEN_OF_EVIL_QUEST_ID, 'offered') },
    { node: 'greeting' },
  ],
  nodes: [
    {
      id: 'greeting',
      text:
        'Welcome, stranger. I am Akara, High Priestess of the Sightless Eye. ' +
        'Our order has fallen, and a great evil festers in the caves beneath the Blood Moor.',
      choices: [
        {
          id: 'ask-evil',
          text: 'What evil?',
          next: 'offer',
          action: { kind: 'offerQuest', questId: DEN_OF_EVIL_QUEST_ID },
        },
        { id: 'heal', text: 'Can you tend my wounds?', next: 'healed', action: { kind: 'heal' } },
        { id: 'trade', text: 'Do you have anything to trade?', next: null, action: { kind: 'openVendor', npcId: 'akara' } },
        { id: 'leave', text: 'Later.', next: null },
      ],
    },
    {
      id: 'offer',
      text:
        'The Den of Evil. Our own sisters, raised as corpses, and worse things beside them. ' +
        'Clear it — every last one — and the camp may sleep again.',
      choices: [
        {
          id: 'accept',
          text: 'I will clear the Den.',
          next: 'accepted',
          action: { kind: 'acceptQuest', questId: DEN_OF_EVIL_QUEST_ID },
        },
        {
          id: 'decline',
          text: 'Not yet.',
          next: null,
          action: { kind: 'declineQuest', questId: DEN_OF_EVIL_QUEST_ID },
        },
      ],
    },
    {
      id: 'accepted',
      text:
        'Then may the Eye watch over you. Take the eastern gate to the Blood Moor; ' +
        'the cave mouth lies beyond the ridge. Return to me when it is done.',
      choices: [{ id: 'go', text: 'It will be done.', next: null }],
    },
    {
      id: 'active',
      text:
        'The Den still breathes, stranger. Do not turn back until every one of them is still.',
      choices: [
        { id: 'heal', text: 'Tend my wounds first.', next: 'healed', action: { kind: 'heal' } },
        { id: 'trade', text: 'Show me your wares.', next: null, action: { kind: 'openVendor', npcId: 'akara' } },
        { id: 'leave', text: 'I go.', next: null },
      ],
    },
    {
      id: 'complete',
      text:
        'It is done — I feel it. The corruption beneath the Moor is lifted, and the camp ' +
        'breathes for the first time in a season. You have my thanks, and more than thanks.',
      choices: [
        {
          id: 'turn-in',
          text: 'The Den is cleared.',
          next: 'rewarded',
          action: { kind: 'turnInQuest', questId: DEN_OF_EVIL_QUEST_ID },
        },
      ],
    },
    {
      id: 'rewarded',
      text:
        'Take this knowledge, hard won by our order and freely given. You will find it in ' +
        'your own hands the next time you raise them.',
      choices: [{ id: 'thanks', text: 'My thanks, Priestess.', next: null }],
    },
    {
      id: 'turned-in',
      text: 'The Sightless Eye watches over you, champion. Rest, and be whole.',
      choices: [
        { id: 'heal', text: 'Tend my wounds.', next: 'healed', action: { kind: 'heal' } },
        { id: 'trade', text: 'Show me your wares.', next: null, action: { kind: 'openVendor', npcId: 'akara' } },
        { id: 'leave', text: 'Farewell.', next: null },
      ],
    },
    {
      id: 'healed',
      text: 'Be whole again. The Eye sees your wounds, and closes them.',
      choices: [{ id: 'ok', text: 'Thank you.', next: null }],
    },
  ],
};

/* -------------------------------------------------------------------------- */
/* Kashya — rogue captain                                                      */
/* -------------------------------------------------------------------------- */

export const KASHYA_DIALOGUE: DialogueTree = {
  id: 'kashya',
  npcId: 'kashya',
  speaker: 'Kashya',
  entry: [
    { node: 'after', condition: whenQuest(DEN_OF_EVIL_QUEST_ID, 'turnedIn', 'rewarded') },
    { node: 'during', condition: whenQuest(DEN_OF_EVIL_QUEST_ID, 'active', 'complete') },
    { node: 'greeting' },
  ],
  nodes: [
    {
      id: 'greeting',
      text:
        'I am Kashya, Captain of the Rogues — what is left of them. Keep clear of my ' +
        'watchpost and we will have no argument.',
      choices: [
        { id: 'ask', text: 'What happened here?', next: 'history' },
        { id: 'leave', text: 'Understood.', next: null },
      ],
    },
    {
      id: 'history',
      text:
        'Andariel took our monastery and our sisters both. Those you meet in the Moor wearing ' +
        'our colours are not our sisters any more. Put them down.',
      choices: [{ id: 'ok', text: 'I will.', next: null }],
    },
    {
      id: 'during',
      text:
        'So Akara has sent you into the Den. Good. My archers cannot hold the gate and clear ' +
        'a cave at once — do not make me regret sparing you the effort.',
      choices: [{ id: 'ok', text: 'It will hold.', next: null }],
    },
    {
      id: 'after',
      text:
        'The Den is quiet. I had not expected it. You have bought this camp a night of sleep, ' +
        'outlander — that is more than most manage.',
      choices: [{ id: 'ok', text: 'Rest well, Captain.', next: null }],
    },
  ],
};

/* -------------------------------------------------------------------------- */
/* Charsi — blacksmith                                                         */
/* -------------------------------------------------------------------------- */

export const CHARSI_DIALOGUE: DialogueTree = {
  id: 'charsi',
  npcId: 'charsi',
  speaker: 'Charsi',
  entry: [{ node: 'greeting' }],
  nodes: [
    {
      id: 'greeting',
      text:
        'Charsi, blacksmith to the Sisters. The forge is still hot, and I can put an edge back ' +
        'on most things that have lost one.',
      choices: [
        {
          id: 'trade',
          text: 'Show me your wares.',
          next: null,
          action: { kind: 'openVendor', npcId: 'charsi' },
        },
        { id: 'repair', text: 'Repair my equipment.', next: 'repaired', action: { kind: 'repair' } },
        { id: 'leave', text: 'Another time.', next: null },
      ],
    },
    {
      id: 'repaired',
      text: 'There. Good as it was — better than it deserves, after what you have put it through.',
      choices: [{ id: 'ok', text: 'Good work.', next: null }],
    },
  ],
};

/* -------------------------------------------------------------------------- */
/* Gheed — merchant                                                            */
/* -------------------------------------------------------------------------- */

export const GHEED_DIALOGUE: DialogueTree = {
  id: 'gheed',
  npcId: 'gheed',
  speaker: 'Gheed',
  entry: [{ node: 'greeting' }],
  nodes: [
    {
      id: 'greeting',
      text:
        'Ahh — a customer! Gheed, at your service, and my prices are the fairest you will find ' +
        'in a camp with precisely one merchant in it.',
      choices: [
        {
          id: 'trade',
          text: 'Let me see what you have.',
          next: null,
          action: { kind: 'openVendor', npcId: 'gheed' },
        },
        { id: 'haggle', text: 'Your prices are robbery.', next: 'haggle' },
        { id: 'leave', text: 'No.', next: null },
      ],
    },
    {
      id: 'haggle',
      text:
        'Robbery! In a camp besieged by the dead! My friend, the dead do not haggle, and neither ' +
        'do I. Now — was there something?',
      choices: [
        {
          id: 'trade',
          text: 'Fine. Show me.',
          next: null,
          action: { kind: 'openVendor', npcId: 'gheed' },
        },
        { id: 'leave', text: 'No.', next: null },
      ],
    },
  ],
};

/* -------------------------------------------------------------------------- */
/* Warriv — caravan                                                            */
/* -------------------------------------------------------------------------- */

export const WARRIV_DIALOGUE: DialogueTree = {
  id: 'warriv',
  npcId: 'warriv',
  speaker: 'Warriv',
  entry: [
    { node: 'after', condition: whenQuest(DEN_OF_EVIL_QUEST_ID, 'turnedIn', 'rewarded') },
    { node: 'greeting' },
  ],
  nodes: [
    {
      id: 'greeting',
      text:
        'Warriv, caravan master. My wagons run east when the road is safe, and the road has not ' +
        'been safe since the monastery fell.',
      choices: [
        { id: 'ask', text: 'When will it be safe?', next: 'when' },
        { id: 'leave', text: 'Safe travels.', next: null },
      ],
    },
    {
      id: 'when',
      text:
        'When the Den beneath the Moor stops emptying itself onto the road every night. Speak to ' +
        'Akara — she has been looking for someone with your bearing.',
      choices: [{ id: 'ok', text: 'I will.', next: null }],
    },
    {
      id: 'after',
      text:
        'The scouts say the Den is silent. If that holds another day I will have the wagons ' +
        'moving east by dawn. You have my thanks, and a seat if you want one.',
      choices: [{ id: 'ok', text: 'Perhaps later.', next: null }],
    },
  ],
};

/** Every camp script, keyed by NPC id. */
export const CAMP_DIALOGUE: readonly DialogueTree[] = [
  AKARA_DIALOGUE,
  KASHYA_DIALOGUE,
  CHARSI_DIALOGUE,
  GHEED_DIALOGUE,
  WARRIV_DIALOGUE,
];

export function dialogueForNpc(npcId: string): DialogueTree | null {
  return CAMP_DIALOGUE.find((tree) => tree.npcId === npcId) ?? null;
}
