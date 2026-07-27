/**
 * @module core/Input
 *
 * Action-mapped keyboard + mouse input with pointer lock.
 *
 * Game code asks about *actions* (`isDown('Attack')`), never about physical
 * keys. That indirection is what makes rebinding, alternate bindings and future
 * gamepad support possible without touching a single gameplay call site.
 *
 * Frame model: DOM events arrive asynchronously and are accumulated into
 * pending state. The engine calls {@link Input.beginFrame} before the update
 * phase to latch that state, and {@link Input.endFrame} after `lateUpdate` to
 * clear per-frame edges and mouse deltas. Every module therefore observes
 * identical input within a frame, regardless of when it runs.
 */

/** Every action the game can bind. Physical keys never appear in game code. */
export type InputAction =
  | 'MoveForward'
  | 'MoveBack'
  | 'MoveLeft'
  | 'MoveRight'
  | 'Sprint'
  | 'Jump'
  | 'Attack'
  | 'HeavyAttack'
  | 'Block'
  | 'Interact'
  | 'ToggleCamera'
  | 'Inventory'
  | 'SkillTree'
  | 'Menu';

/** All actions, in a stable order. Useful for binding UIs and diagnostics. */
export const INPUT_ACTIONS: readonly InputAction[] = [
  'MoveForward',
  'MoveBack',
  'MoveLeft',
  'MoveRight',
  'Sprint',
  'Jump',
  'Attack',
  'HeavyAttack',
  'Block',
  'Interact',
  'ToggleCamera',
  'Inventory',
  'SkillTree',
  'Menu',
];

/**
 * A binding token. Keyboard bindings use `KeyboardEvent.code` (layout
 * independent — `KeyW` is the same physical key on AZERTY). Mouse buttons use
 * the synthetic `Mouse0`/`Mouse1`/`Mouse2` codes for left/middle/right.
 */
export type BindingCode = string;

export type Bindings = Readonly<Record<InputAction, readonly BindingCode[]>>;

/** Default Barbarian control scheme: WASD melee, right-click block. */
export const DEFAULT_BINDINGS: Bindings = {
  MoveForward: ['KeyW', 'ArrowUp'],
  MoveBack: ['KeyS', 'ArrowDown'],
  MoveLeft: ['KeyA', 'ArrowLeft'],
  MoveRight: ['KeyD', 'ArrowRight'],
  Sprint: ['ShiftLeft', 'ShiftRight'],
  Jump: ['Space'],
  Attack: ['Mouse0'],
  HeavyAttack: ['Mouse1', 'KeyR'],
  Block: ['Mouse2'],
  Interact: ['KeyE'],
  ToggleCamera: ['KeyF'],
  Inventory: ['KeyI', 'Tab'],
  SkillTree: ['KeyK'],
  Menu: ['Escape'],
};

/**
 * Codes whose browser default is actively hostile to a game: Space scrolls,
 * Tab moves focus off the canvas, arrows scroll. Suppressed while input is
 * enabled and the canvas owns interaction.
 */
const SUPPRESS_DEFAULT = new Set<BindingCode>([
  'Space',
  'Tab',
  'ArrowUp',
  'ArrowDown',
  'ArrowLeft',
  'ArrowRight',
]);

/** Read-only 2D vector view handed out for mouse deltas. */
export interface ReadonlyVec2 {
  readonly x: number;
  readonly y: number;
}

export interface InputOptions {
  /** Element that receives pointer/wheel events and owns pointer lock. */
  target: HTMLElement;
  /** Keyboard events are global; defaults to `window`. */
  keyboardTarget?: EventTarget;
  bindings?: Partial<Bindings>;
}

export class Input {
  readonly #target: HTMLElement;
  readonly #keyboardTarget: EventTarget;
  readonly #disposers: Array<() => void> = [];

  /** action -> bound codes. */
  #bindings: Record<InputAction, BindingCode[]>;
  /** code -> actions, rebuilt whenever bindings change. O(1) event dispatch. */
  #codeToActions = new Map<BindingCode, InputAction[]>();

  /** Physically held codes, updated straight from the DOM events. */
  readonly #held = new Set<BindingCode>();
  /** Codes that went down since the last {@link endFrame}. */
  readonly #pressed = new Set<BindingCode>();
  /** Codes that went up since the last {@link endFrame}. */
  readonly #released = new Set<BindingCode>();

  #mouseDX = 0;
  #mouseDY = 0;
  #mouseX = 0;
  #mouseY = 0;
  #wheel = 0;
  #pointerLocked = false;
  #enabled = true;

  constructor(options: InputOptions) {
    this.#target = options.target;
    this.#keyboardTarget = options.keyboardTarget ?? window;
    this.#bindings = { ...DEFAULT_BINDINGS } as Record<InputAction, BindingCode[]>;
    for (const action of INPUT_ACTIONS) {
      const override = options.bindings?.[action];
      this.#bindings[action] = [...(override ?? DEFAULT_BINDINGS[action])];
    }
    this.#rebuildLookup();
    this.#attach();
  }

  // -- queries ------------------------------------------------------------

  /** Whether any code bound to `action` is currently held. */
  isDown(action: InputAction): boolean {
    if (!this.#enabled) return false;
    for (const code of this.#bindings[action]) if (this.#held.has(code)) return true;
    return false;
  }

  /** Whether `action` transitioned to held during this frame (rising edge). */
  wasPressed(action: InputAction): boolean {
    if (!this.#enabled) return false;
    for (const code of this.#bindings[action]) if (this.#pressed.has(code)) return true;
    return false;
  }

  /** Whether `action` was let go during this frame (falling edge). */
  wasReleased(action: InputAction): boolean {
    if (!this.#enabled) return false;
    for (const code of this.#bindings[action]) if (this.#released.has(code)) return true;
    return false;
  }

  /**
   * Accumulated pointer movement for this frame, in CSS pixels. Under pointer
   * lock this is raw relative motion and is unbounded; otherwise it is the
   * delta of the cursor position inside the target element.
   */
  get mouseDelta(): ReadonlyVec2 {
    return { x: this.#mouseDX, y: this.#mouseDY };
  }

  /** Cursor position within the target element, in CSS pixels. */
  get mousePosition(): ReadonlyVec2 {
    return { x: this.#mouseX, y: this.#mouseY };
  }

  /**
   * Add synthetic pointer motion to this frame's accumulated delta.
   *
   * This is the mouse, moved by something other than a hand. It exists because
   * a headless harness cannot aim: `mousemove` under pointer lock reports
   * `movementX`, and pointer lock needs a user gesture that a scripted run does
   * not have. Everything downstream — {@link mouseDelta}, `PlayerController`'s
   * look accumulator, the camera rig — is untouched, so a harness that turns
   * the player with this is exercising the same code path as a player who
   * turns with a mouse, at whatever rate it chooses to feed.
   *
   * Deliberately *not* a yaw setter. A setter would let a harness snap the
   * character to a facing no hand could produce, and any time-to-kill measured
   * that way would be a measurement of the harness. Feeding pixels keeps the
   * turn subject to the same sensitivity, the same per-frame accumulation and
   * the same `turnRate` limit on the body catching up to the camera.
   *
   * Cleared by {@link endFrame} like any other delta, so it must be fed on the
   * frame it is meant to apply to.
   */
  nudgePointer(dx: number, dy = 0): void {
    this.#mouseDX += dx;
    this.#mouseDY += dy;
  }

  /** Accumulated wheel movement for this frame. Positive scrolls away. */
  get wheelDelta(): number {
    return this.#wheel;
  }

  get pointerLocked(): boolean {
    return this.#pointerLocked;
  }

  /**
   * When disabled, every query reports neutral. Used while a modal UI owns the
   * keyboard, without tearing down listeners (so held keys stay tracked and do
   * not "stick" once input is re-enabled).
   */
  get enabled(): boolean {
    return this.#enabled;
  }

  set enabled(value: boolean) {
    if (this.#enabled === value) return;
    this.#enabled = value;
    if (!value) this.#clearFrameState();
  }

  /**
   * Signed axis from two actions, in `[-1, 1]`. Opposing keys cancel, which is
   * what players expect from simultaneous A+D.
   */
  axis(negative: InputAction, positive: InputAction): number {
    return (this.isDown(positive) ? 1 : 0) - (this.isDown(negative) ? 1 : 0);
  }

  /**
   * Movement intent in local space: `x` is strafe (right positive), `y` is
   * forward (forward positive). Normalised so diagonals are not faster.
   */
  moveVector(): ReadonlyVec2 {
    const x = this.axis('MoveLeft', 'MoveRight');
    const y = this.axis('MoveBack', 'MoveForward');
    if (x === 0 && y === 0) return { x: 0, y: 0 };
    const inv = 1 / Math.hypot(x, y);
    return { x: x * inv, y: y * inv };
  }

  // -- frame lifecycle ----------------------------------------------------

  /**
   * Called by the engine before the update phase. Present for symmetry and as
   * the hook for future latched-state handling (gamepad polling); currently a
   * no-op because DOM events already mutate state as they arrive.
   */
  beginFrame(): void {
    /* intentionally empty — see doc comment */
  }

  /** Called by the engine after `lateUpdate`; clears edges and deltas. */
  endFrame(): void {
    this.#pressed.clear();
    this.#released.clear();
    this.#mouseDX = 0;
    this.#mouseDY = 0;
    this.#wheel = 0;
  }

  // -- pointer lock -------------------------------------------------------

  /**
   * Request pointer lock on the target element. Must be called from a user
   * gesture; the browser rejects it otherwise, and the rejection is swallowed
   * because it is an expected, recoverable condition.
   */
  requestPointerLock(): void {
    const result: unknown = this.#target.requestPointerLock();
    if (result instanceof Promise) result.catch(() => undefined);
  }

  exitPointerLock(): void {
    if (document.pointerLockElement === this.#target) document.exitPointerLock();
  }

  // -- binding ------------------------------------------------------------

  /** Replace the codes bound to `action`. */
  rebind(action: InputAction, codes: readonly BindingCode[]): void {
    this.#bindings[action] = [...codes];
    this.#rebuildLookup();
  }

  /** Current bindings, as a defensive copy. */
  getBindings(): Bindings {
    const out = {} as Record<InputAction, readonly BindingCode[]>;
    for (const action of INPUT_ACTIONS) out[action] = [...this.#bindings[action]];
    return out;
  }

  dispose(): void {
    for (const off of this.#disposers) off();
    this.#disposers.length = 0;
    this.#held.clear();
    this.#clearFrameState();
  }

  // -- internals ----------------------------------------------------------

  #clearFrameState(): void {
    this.#pressed.clear();
    this.#released.clear();
    this.#mouseDX = 0;
    this.#mouseDY = 0;
    this.#wheel = 0;
  }

  #rebuildLookup(): void {
    this.#codeToActions = new Map();
    for (const action of INPUT_ACTIONS) {
      for (const code of this.#bindings[action]) {
        const list = this.#codeToActions.get(code);
        if (list === undefined) this.#codeToActions.set(code, [action]);
        else list.push(action);
      }
    }
  }

  #isBound(code: BindingCode): boolean {
    return this.#codeToActions.has(code);
  }

  #down(code: BindingCode): void {
    // Guard against auto-repeat producing a rising edge every keyboard tick.
    if (this.#held.has(code)) return;
    this.#held.add(code);
    this.#pressed.add(code);
  }

  #up(code: BindingCode): void {
    if (!this.#held.delete(code)) return;
    this.#released.add(code);
  }

  #on<T extends Event>(
    target: EventTarget,
    type: string,
    handler: (event: T) => void,
    options?: AddEventListenerOptions,
  ): void {
    const listener = handler as EventListener;
    target.addEventListener(type, listener, options);
    this.#disposers.push(() => target.removeEventListener(type, listener, options));
  }

  #attach(): void {
    this.#on<KeyboardEvent>(this.#keyboardTarget, 'keydown', (event) => {
      if (event.repeat) return;
      if (!this.#isBound(event.code)) return;
      if (SUPPRESS_DEFAULT.has(event.code)) event.preventDefault();
      this.#down(event.code);
    });

    this.#on<KeyboardEvent>(this.#keyboardTarget, 'keyup', (event) => {
      if (!this.#isBound(event.code)) return;
      if (SUPPRESS_DEFAULT.has(event.code)) event.preventDefault();
      this.#up(event.code);
    });

    // A held key that is still down when the window loses focus never delivers
    // its keyup, so the action would stay stuck on forever. Release everything.
    this.#on(window, 'blur', () => {
      for (const code of Array.from(this.#held)) this.#up(code);
    });

    this.#on<MouseEvent>(this.#target, 'mousedown', (event) => {
      const code = `Mouse${event.button}`;
      if (!this.#isBound(code)) return;
      event.preventDefault();
      this.#down(code);
    });

    // Mouse-up is tracked on the window: releasing outside the canvas must
    // still end the action.
    this.#on<MouseEvent>(window, 'mouseup', (event) => {
      this.#up(`Mouse${event.button}`);
    });

    // Right-click is a bound game action (Block), so the context menu must not
    // appear over the canvas.
    this.#on<MouseEvent>(this.#target, 'contextmenu', (event) => event.preventDefault());

    this.#on<MouseEvent>(this.#target, 'mousemove', (event) => {
      if (this.#pointerLocked) {
        this.#mouseDX += event.movementX;
        this.#mouseDY += event.movementY;
      } else {
        const rect = this.#target.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        this.#mouseDX += x - this.#mouseX;
        this.#mouseDY += y - this.#mouseY;
        this.#mouseX = x;
        this.#mouseY = y;
      }
    });

    this.#on<WheelEvent>(this.#target, 'wheel', (event) => {
      event.preventDefault();
      this.#wheel += event.deltaY;
    }, { passive: false });

    this.#on(document, 'pointerlockchange', () => {
      this.#pointerLocked = document.pointerLockElement === this.#target;
      // Movement accumulated across the mode switch is in the wrong coordinate
      // space (absolute vs relative); discard it rather than snap the camera.
      this.#mouseDX = 0;
      this.#mouseDY = 0;
    });
  }
}
