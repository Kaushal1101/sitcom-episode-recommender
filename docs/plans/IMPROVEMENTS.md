# Planned Improvements

---

## Task 1: Episode selection as a final choice

When the user is shown episode options (ITEM_MULTI or ITEM_YN), they currently
can only interact with them as a preference signal. The system should also allow
the user to indicate they want to watch a specific episode outright — bypassing
further questioning and completing the session immediately.

This requires changes to both the UI (a distinct "Watch this" action separate
from the preference pick) and the backend (routing directly to RECOMMEND + exit
when a final selection is made, rather than re-running the pipeline).

### Implementation plan

**Backend — `backend/api/main.py`**

1. Extend `AnswerRequest` with an optional `is_final_selection: bool = False` field.
2. In `submit_answer`, before calling `_apply_state_update`, check
   `request.is_final_selection`. If true:
   - Determine the chosen episode. For ITEM_MULTI, match `request.answer`
     against `decision.candidates`. For ITEM_YN, use `decision.candidates[0]`.
   - Still call `_apply_state_update` so the answer is appended to history.
   - Build and return a terminal `DecisionResponse` with `done=True`,
     `farewell="Enjoy <title>! 🎬"`, and the `episode_id` of the chosen episode.
     Delete the session.
   - Do **not** call `strategy.next_action`.

No changes needed to `SessionState`, `ConversationStrategy`, or
`preference_updater` — the backend short-circuit is purely in the API layer.

**Frontend — `frontend/src/App.jsx`**

3. For `ITEM_MULTI`, replace the single episode button with a two-button card:
   - **"That one"** — calls `handleAnswer(option)` as today (preference signal).
   - **"Watch this"** — calls a new `handleFinalSelection(option)` function.
4. For `ITEM_YN`, add a **"Yes, watch it"** button alongside the existing Yes/No.
   Pressing it calls `handleFinalSelection` with the episode label (or a sentinel
   the backend can recognise for the single candidate).
5. `handleFinalSelection(option)` is identical to `handleAnswer` except it posts
   `{ answer: option, is_final_selection: true }`.

**Frontend — `frontend/src/App.css`**

6. Style the "Watch this" button distinctly (filled accent colour) so it is
   visually separate from the preference pick.

---

## Task 2: Make attribute questions less confidence-impactful so YN questions surface

ATTRIBUTE_MULTI sets preference dimensions with strong values (0.8, 0.9, etc.)
in a single step. This causes confidence to overshoot the ATTRIBUTE_YN and
ITEM_YN bands in one jump, making those question types practically unreachable.

The fix should slow down how strongly a single ATTRIBUTE_MULTI answer moves
the confidence score so the system asks more questions before converging —
surfacing the yes/no question types as intended and producing a more natural
conversational flow.

### Implementation plan

**`backend/csm/preference_updater.py`**

1. Add a constant `ATTRIBUTE_MULTI_STEP: float = 0.5` near the top (alongside
   `YN_POSITIVE`, `YN_NEGATIVE`, `ITEM_STEP`).
2. In `_apply_attribute`, for the `ATTRIBUTE_MULTI` / `SOFT_CRITIQUE` branch,
   replace the direct-set:

   ```python
   _write_preference(state, intent, intent_options[answer])
   ```

   with a blended update that moves the current value half-way toward the target:

   ```python
   target = intent_options[answer]
   current = _read_preference(state, intent)
   blended = _clamp(current + ATTRIBUTE_MULTI_STEP * (target - current))
   _write_preference(state, intent, blended)
   ```

3. Add a `_read_preference(state, intent) -> float` helper (mirrors
   `_write_preference`) that returns the current scalar for a mood or tone dim.

**Why this works**: the formula `current + step * (target - current)` is the
same exponential-blend used for item updates. With `step=0.5`, a single answer
can at most move the dimension 50 % toward the target. A dimension that starts
at 0.0 with a target of 0.9 reaches 0.45 on the first answer, not 0.9. The
lower per-question signal means confidence builds more gradually, keeping it in
the ATTRIBUTE_YN and ITEM_* bands for multiple turns.

**Tuning note**: `ATTRIBUTE_MULTI_STEP=0.5` is a starting point. If YN
questions still don't appear in practice, lower it toward 0.3. If the system
feels sluggish, raise it back toward 0.7.

---

## Task 3: UI touchup

The current UI is functional but visually sparse. The goal is to make it feel
more polished without changing the layout structure or adding new features.

### Implementation plan

**`frontend/src/App.css`**

1. **Typography**: use a system font stack that includes Inter/Segoe UI for
   crisper text. Increase base `line-height` to 1.6 for readability.
2. **Colour palette**: introduce CSS custom properties (`--accent`, `--surface`,
   `--border`, `--text-muted`) so colours are consistent and easy to adjust.
   Use a soft indigo accent (`#6366f1`) matching the existing confidence bar.
3. **Container**: add a subtle `box-shadow` and `border-radius` to `#root` so
   the card floats off the background. Slightly increase `padding`.
4. **Buttons**: round corners to `border-radius: 6px`, add a `transition` on
   background and border, and give the primary/accent button (used for "Watch
   this" from Task 1) a filled indigo style with white text.
5. **Episode cards**: add a left accent border stripe (`border-left: 3px solid
   var(--accent)`) on `.episode-option` for visual hierarchy. Increase synopsis
   padding slightly.
6. **Loading state**: when `loading` is true, show a subtle spinner or pulsing
   opacity on the question text so the user knows something is happening.
7. **Header**: make `h1` slightly smaller (1.2rem) and use `font-weight: 600`
   with a muted letter-spacing to look less plain.
8. **Confidence panel**: style the `<details>` toggle with a small chevron icon
   (CSS only, `::after` pseudo-element) and use tabular-nums for the confidence
   value so it doesn't jump widths.

**`frontend/src/App.jsx`**

9. Add a `loading` CSS class to the question `<p>` when `loading` is true, so
   the stylesheet can apply a pulsing opacity animation without JS logic.
10. For the start screen, add a one-line tagline under the `<h1>` so the page
    doesn't feel empty before the first click.
