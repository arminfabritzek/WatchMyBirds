# WatchMyBirds UI Standard

This file defines the frontend conventions and the preferred
shared DOM structures for:
- Modals (including types/variants)
- Review-stage panels
- Thumbnails / Tiles (including types/variants)
- Action bars
- Image viewers

This is migration-aware guidance for the current app state. It is not a claim
that every legacy surface has already been converted.

## 0. Unified Image UX (binding, app-wide)

Every place in the app that shows a bird image — Stream, Gallery, Subgallery,
Species, Species Overview, Review Desk, Trash, Restore, detail modals, and any future surface — must present
the **same image UX**. This is a hard rule, not a guideline. It exists so that
shared concerns (favorite, change-species, view-details, hover
tooltips, keyboard navigation, bbox overlay behavior, zoom modal) can be added
or changed in **one place** and instantly apply everywhere.

**The contract:**

1. **One viewer component.** Every image is rendered through
   `templates/components/modal_image_viewer.html` (`render_image_viewer`)
   inside the canonical `wm-image-viewer` / `wm-modal__image` /
   shared viewer shells. No surface builds its own
   `<img>` + bbox overlay stack.
2. **One toolbox.** Every image-bearing surface hosts the shared
   `tile_toolbox` macro (§5) inside a `wm-toolbox-host` container. The toolbox
   is the only place per-image actions live (Favorite, View Details,
   Change Species, Adjust Bounding Box, Move to Trash, Restore,
   No birds in full image, …). Surfaces may *omit* an action when the route does not
   support it; they may **not** rename it, reorder it, or add a parallel
   surface-local button for the same action. The offered-box species control
   defined in §0e is the sole object-anchored exception. See §0b for which
   actions render as primary vs. inside the overflow on each surface.
3. **One action vocabulary.** The canonical verbs in §"Detection Action Frame
   Contract" are the only allowed labels. Adding a new image action means
   adding it to the toolbox macro and to this section, never as a one-off in a
   single template.
4. **One zoom path.** Clicking any image opens the same `wm-modal` viewer
   (detection-backed → `detection_modal.html`, image-only fallback →
   the shared image viewer). No surface ships its own lightbox.
5. **Action position is fixed.** Tile toolboxes sit at the bottom of the image
   on hover/focus (`wm-toolbox`). Detection detail modals use the fixed
   `wm-bird-editor` rail immediately below the photo; it is the sole modal
   toolbar and owns object editing plus one overflow menu. Surfaces do not
   add a second action row or menu.
6. **Per-image state surfaces are uniform.** Things like favorite stars,
   eligibility badges, and selection checkboxes use
   shared classes (`wm-tile__badge`, `wm-toolbox__fav`, …) and the same DOM
   position across surfaces.
7. **No new image-bearing surface ships without reusing this contract.**
   PRs that introduce a new image view must reuse `render_image_viewer` +
   `tile_toolbox`. If a real exception is needed, this section must be
   updated in the same change.
8. **View mode is an explicit choice.** Gallery grids use a segmented
   `Crop | Full` control. Detail viewers with an editable object use
   `Full | Focus` at the start of the shared editor rail. The order and
   position stay stable while the selected bird changes.

**Why this matters:** the explicit goal is that adding a new per-image
affordance becomes a single edit to the toolbox
macro plus the design-system CSS, and lights up Stream, Gallery, Review,
Trash, modals, and everything else at once. Any drift defeats that goal.

Favorite is the image preference; star ratings are not supported. Downloads
retain favorite labels and flags without numeric rating tags.

**Audit:** check this file against the codebase after touching
image-bearing surfaces.

## Current station history (binding)

`current_observations` is the single SQL view for current object identity and
admission. Startup drops and recreates it after schema setup. Gallery, Species,
Station Report, saved Live observations, homepage selections and counts read
this view; readers do not re-evaluate model thresholds or legacy assignments.
Manual objects participate without model scores or synthetic decisions.

Active objects with a current bird-presence answer `absent` are never admitted,
even if a species answer remains current. Otherwise a current confirmed or
corrected species answer wins; `unknown` and `wrong` never admit the proposed
species. Without a species answer, only the stored `decision_state = confirmed`
and a species-level decision admit an object. Stored historical rows with NULL
or empty `decision_level` count as species level. New captures always store a
level; a missing capture input still keeps the decision unresolved. Missing
`decision_state` values stay unresolved.
Image statements are training evidence and do not control display.

The capture policy owns box, species-confidence, uncertainty, display-score,
named-bird and geographic checks. Ticker, Telegram and MQTT use that same capture
decision; unresolved ticker events carry no species name. Geography excludes at
exactly zero occurrence; thresholds are unchanged. Settings affect new captures
only. No background job re-evaluates stored admission decisions.

Sharpness and CLIP rank photos, covers and event representatives only. They never
veto admission or send a confident bird to Review. All group covers use
`utils.cover_selection`: favorite, sharpness (`quality_gallery_ok != 0`, with
NULL treated as sharp), PTZ close-up (`preset` / `manual_drive`), AI eligibility,
aesthetic score, detection score, box quality, then recency. Exact ties retain
the selector's existing random or stable choice. Standard Gallery applies no
second score floor. Human answers bypass automatic thresholds. Refutations affect
their own object; historical hints do not blacklist a species or reopen Review.
Excluded automatic records expose the persisted decision reason. Open Review and
admitted observations do not overlap. This rule does not change Review's layout.
Confirmed archive species remain discoverable outside a selected period and
are excluded from that period's totals. The always-visible `Bird · species
unknown` link has an explicit empty state. Missing originals retain identity,
dates and history; surviving previews and placeholders are identified.
Localized and scientific names always use the same current species key.
Correction Details lists the preserved raw model species and classifier
selection separately from current human answers.

Ordinary Review uses the original `review_grid.html` event cards, day navigation,
selection controls and canonical detection modal. Its membership is the shared
current observation projection: every active unanswered excluded proposal.
The existing event/day view has Open and Done tabs. Both read
`current_observations`; Done includes every object with a current answer, including
presence-only and box-only answers, No bird, Wrong species and manual birds,
plus current image statements grouped by capture day. Missing photos use a
placeholder without removing the answer. No bird takes precedence over an older
species answer in the card caption. Tabs, day navigation and pagination preserve
the date selection. Withdrawn manual objects remain visible as withdrawn
records, with no detector actions or object editing.
Event buttons answer selected open objects, or all open objects in that card
when none are selected. They require no evidence-quality assessment. The server
checks the whole selection under one write transaction; completed, admitted,
or missing objects reject the entire operation. Manually added birds are edited
in the shared viewer. No-bird answers concern the selected object, never
unselected siblings or the whole image. Answered
objects with resolved species or absence leave the queue; an event disappears
once it has no open objects. Presence-only and box-only answers also appear in
Done while unresolved objects remain in Open.
The editor rail keeps the available modal width even when the image is narrow
or missing, so answer controls and notices remain readable.
Station Report uses the same cards and viewer, current identifications and
separate human/automatic days; total days use their union. Links preserve date
filters. The calendar shows positive records; empty cells mean
unknown, never absence. Knowledge links use existing references and own images;
no behavior claims or unsourced local species profiles are fabricated.

One bounded, per-application report worker computes snapshots. Opening shows
the last result with its local calculation time (as of HH:MM). A missing result
or one older than ten minutes starts a background calculation on opening, with
a loading note. The browser polls only until that opening calculation finishes;
the previous report stays visible. Afterward, captures, database commits, media
changes and elapsed time never hide or replace the open report. Refresh explicitly
starts a new calculation and replaces the result when ready, preserving filter
drafts, scrolling and expanded details. Errors retain the previous result and
offer Refresh. An in-report correction immediately updates the affected card
and marks totals "outdated, refresh" with the existing `alert alert-info`
status notice style used by the update notice. Show the notice below the report
controls and directly below "Species and recorded days". Both retain
`role="status"`, explain that the correction is saved but the species list and
totals still need recalculation, and disappear only after a successful Refresh.
The notice by the species list includes a Refresh button using the same action
and busy state as the upper button, including after report content is replaced.
Closing the viewer does not reload the report. Replaced editors release listeners
and observers. Local development runs this worker on its isolated copy; camera,
inference, integrations and retention jobs remain disabled.

Ordinary Review uses day navigation and complete visit groups from the shared
`build_bird_events` grouping rules. Group membership grants no approval. Each
object exposes its proposal and admission reason; the shared viewer offers
Confirm species, Change species, Bird · species unknown, and No bird. These
answers do not require box quality or event approval. Done includes object absence
answers, manual birds and image statements. An explicit object dismissal completes
that object. Whole-image statements are training facts and do not control display.
The migration withdraws contradictory image and object answers before Review opens;
ordinary answer writes reject bird-presence contradictions. A valid whole-image
absence still has a withdrawal notice in the shared viewer, including Answered.
It is an existing image answer, not an unresolved conflict.

## Training export

The Training export page shows waiting photo counts and recorded exports.
`Export waiting photos` records the waiting revisions and starts a streamed ZIP
download. The page immediately shows the new unconfirmed export. `Download again`
uses its recorded context, even after answers change. `Export received` confirms
only that export and displays its receipt time. Waiting answers and unconfirmed
exports protect photos; missing originals do not wait. Box-quality answers and
the optional box walkthrough stay available independently of export receipts.

Use the shared `btn` variants. Waiting counts and every unconfirmed export use a
prominent `alert alert-info` notice with `role="status"`; the notice explains that
photos remain protected until receipt is confirmed. Export actions wrap at narrow
widths. Creating and acknowledging exports require authenticated CSRF-protected
POSTs; downloads never acknowledge receipt.

## Live camera status

PTZ status polling pauses in hidden tabs. On Live, the tracking overlay reuses
the page's initial and 15-second status snapshots while Auto Cam or the overlay
is off. Its 200 ms polling runs only while both are enabled and the page is
visible, with at most one overlay request in flight. Successful Auto Cam changes
refresh the shared snapshot immediately. Manual controls remain available with
Auto Cam off. Status reads report app state; they do not send ONVIF commands.

## Homepage species cards

The Live page renders its camera shell and today's observations without waiting
for archive-wide statistics. Best of Species loads once in a separate request,
with a visible loading status, an empty state and an explicit Retry on failure.
Its species total updates the shared ticker when ready; an unknown total is
omitted rather than displayed as zero. The existing 30-second photo-pool cache
is shared by concurrent requests; current human answers are refreshed for every
response. Only dialogs for displayed covers/frames are sent. Archive content
sits outside the periodically refreshed today's-visitors section so that refresh
does not remove it or close an open archive viewer.

Today's Visitors and Best of Species use the same fixed four-image budget,
rendered as two equal columns and two rows without an image-count control.
Choose distinct source images using the existing favorite and quality priorities;
group photos receive no additional preference. Three available images reduce to
two; one image spans the card width with `object-fit: contain` so the existing
thumbnail is not cropped again. Image actions and detail modals stay shared.
Best of Species initially shows the first three species; native Show more
expands the remaining cards. Today's Visitors keeps all selected species visible.

## Gallery day navigation

The Gallery overview shows 30 days per server-rendered page, newest first.
Previous/Next links retain access to older days. Review totals describe visible
offered detections for each day and count current human facts only; pagination
does not change the shared image or detail-modal interactions.

## 0c. Detail Modal Companion Visibility (binding)

When the user opens an **image-detail modal** — clicked through from
Stream, Gallery, Subgallery, Species, Species Overview, Review queue,
Review event panel, or Trash — every active detection on the source
frame MUST render as a bbox overlay. The detection that opened the
modal is visually distinguished as `isCurrent`; companion detections
share the same overlay style.

**Why this rule:** tile / queue / story-board surfaces intentionally
collapse to one representative detection per image — that is by design
for surface clarity. The detail modal is the place where the user
expects the **full** information for that one image. Hiding companions
in the detail view is a regression: the user has no other affordance
to discover that the frame had additional birds.

**Required pieces (call-site checklist):**

1. The backend dict that drives the modal MUST include a `siblings`
   list of every active detection on the same source image, with
   `bbox_x` / `bbox_y` / `bbox_w` / `bbox_h` (frame-fraction
   coordinates) and `detection_id` per entry. The image's own
   detection appears in the list too.
2. The Jinja call to `render_image_viewer` MUST pass that list as
   `siblings=...`. The macro emits `data-siblings` JSON for the JS
   layer to read.
3. The JS auto-render path (`initBboxOverlay`) MUST force-on the
   overlay when `data-siblings.length > 1` regardless of the saved
   user preference. Single-detection frames continue to honour the
   saved pref.
4. The bbox toggle button continues to drive `data-siblings` from the
   action-bar `siblings | tojson` blob — already implemented in
   `gallery_utils.js:toggleBboxOverlay`.

**Visibility gate:** companions use their own SQL predicate
(`_companion_visibility_sql`), not the gallery's. The gallery additionally
applies the shared admission rule, including conservative quality safeguards only
for automatic records. Its decision-state gate applies to model-only records; current human species answers take precedence. A box
someone marked "bird present, species unknown" keeps `decision_state = 'unknown'`
on purpose (its species answer is complete) and MUST still render here. Do not
route companions through the gallery predicate.

**Out of scope of this rule:**

- Tile previews / thumbnails / story-board cards — these stay as
  one-cover-per-image (UI_STANDARD §0 contract preserved).
- Notifications and the Review queue's `MAX(score)` grouping —
  intentional summary surfaces.
- Review grid tile cells — those are queue/list cells,
  not detail modals.

**Closing the loop with §0:** §0 establishes that "shared concerns
... bbox overlay behavior ... applied everywhere"; this section is
the explicit form of that promise for the multi-detection case.

## 0d. Direct Bounding-Box Correction (binding)

Bounding-box correction is an ordinary image action inside the canonical
detail modal. It is not a separate labeling mode or page.

1. `Adjust box` starts from the active object's stored geometry. `Add missing
   bird` is the sole blank-canvas exception: an authenticated operator may
   draw one manual object on an existing stored image. It creates explicit
   manual object evidence, never a detector proposal or biological event.
2. `Adjust box` preserves the current `Full` or `Focus` view, and either view
   remains selectable during the edit. `Add missing bird` switches to `Full`
   so the complete drawing canvas is visible. Hovering and dragging a side
   moves that side; corners move two sides; dragging inside moves the whole box.
3. Geometry stays inside the image and retains a visible minimum size. Resize
   handles compensate for the Focus transform so their on-screen size remains
   stable. The shared math contract lives in `assets/js/bbox_editor_math.js`.
4. Nothing is persisted during dragging. `Save box` explicitly records a
   corrected, suitable box through `/api/labels/answer`; `Cancel` and `Escape`
   discard the edit.
5. Saving updates the active viewer, companion data, bbox toggle, and sibling
   selector without overwriting the immutable proposal snapshot.
6. Keyboard users can focus the box and move it with arrow keys; Shift uses a
   larger step. Buttons and the action glyph carry labels and native tooltips.
7. `Correction Details` is an optional overflow action for the active box. It
   lists the exact current facts and derives OD and CLS readiness separately
   from the same core reason-code contract used by dataset export. On images
   with multiple proposals it states how many boxes have answers and makes
   clear that other boxes are unchanged.
8. The fixed editor rail is ordered `Full | Focus`, species, `Adjust box`,
   `Add missing bird`, one overflow. During an edit the same rail replaces the
   two edit actions with `Cancel` and `Save bird`; Escape cancels. Species uses
   `WmSpeciesPicker` and remains draft state until Save succeeds.
9. Saving a new manual object asserts only object bird presence, its manual
   geometry, and either a catalog species identity or explicit unknown species.
   Later saves append revisions only for the changed geometry/species axes.
   They do not assert bbox quality, image completeness, event approval, or any
   sibling fact. Manual objects carry no detector/classifier model or score.
10. When `original_present = 0`, the shared editor shows a persistent
    `wm-training-export-notice` above its controls. It explains that corrections
    are saved but cannot be exported for training without the original. The
    notice never covers the image. Offered-box corrections remain available;
    manual-object creation and manual geometry updates require the original,
    while species-only updates to an existing manual object remain available.
    The dataset page uses
    the same notice style to count distinct labeled birds whose training views
    are excluded for missing media; these are separate from pending box verdicts.
11. In the Box Walkthrough, `Box fits` is also a direct primary action in the
    editor rail. It records the current box verdict through the shared editor
    action and advances only after the save succeeds.

12. Each box carries a compact badge at its lower-right corner: `?` for unchecked,
    a green check for confirmed, and a red `!` for incorrect. Its accessible name
    and tooltip spell out the status; the menu retains the full status text.
    Pending edits use a clock rather than a saved confirmation check. Manual
    objects use a pencil without inventing a box-quality fact. Badges retain
    their screen size in Focus view and avoid resize handles during editing.
    For authenticated operators, the offered-box badge is a keyboard-accessible
    button in browse mode: click to confirm suitable, click again to retract.
    It targets its own box, including companions, and never changes species.
    It is disabled while saving, stale, or editing; manual boxes stay read-only.
13. The overflow keeps two box actions (`Box fits`, `Box is wrong`) and a
    non-action status line. Pressing the active action clears that assessment.
    Browse-mode answers save immediately; edit-mode answers say `Unsaved` until
    Save succeeds. Cancel keeps the stored assessment. Geometry changes imply
    a suitable new box unless the user explicitly chooses another assessment.
14. Stored verdicts come only from current canonical human facts. They survive
    reload, companion selection and inter-modal updates. Draft intent is separate:
    a species-only save never repeats, changes or clears the stored box answer.
    Combined changes and retraction persist atomically through the answer API.

Every packaged JS/CSS include uses the shared `asset_version` template value,
read once from `utils.deploy_info.read_build_metadata()["git_commit"]` when the
pages blueprint is registered. A new build commit changes all asset URLs after
one page reload. Restart the application after installing a new build; tabs
already open need a reload. Required scripts and styles are served by the
appliance, including the pinned Bootstrap files in `assets/vendor/`. Typography
uses the existing system-font stacks; do not add remote font requests. The UI
must load and its local controls must work without internet access or a warm
browser cache. External links and explicitly enabled integrations may still
require internet access.

Pending boxes on an image already admitted to the gallery are visible companions:
a dashed outline, `Review` in the box label, and `In Review` in the bird selector.
They never admit another image or create species cards. Dismissed detector boxes
remain visible and selectable in detail modals with a dashed, muted outline and
an explicit `Dismissed` label. `Restore bird` appends a new object-presence answer
and supersedes any contradictory whole-image absence. It keeps the existing
species and box assessments, proposal, and history; it does not approve either
axis. Dismissed boxes do not count as active birds or gain gallery admission.
The gallery admission and quality filters continue to apply to cards, not companion
visibility. Gallery and Review edit the same identities and current facts; closing
an edited Review viewer refreshes the queue. Object actions do not approve events.

`Add missing bird` checks its saved-image geometry against current and dismissed
boxes before creating an identity. A match presents `Correct bird` and highlights
the candidate; ambiguous matches require a selection. `A different bird — keep
both` explicitly preserves genuinely overlapping birds. Overlap alone never merges
objects. Correcting a dismissed detection retains its identity and model proposal.
A new manual object has its own stable identity and no model scores. An explicit
bird correction replaces a contradictory earlier full-image absence fact while
preserving history; it does not assert image completeness.

The Full viewer fits the entire bitmap into the height left by the editor rail.
Its overlay uses that same image geometry at desktop and narrow widths.

## 0e. Offered-Box Species Confirmation (binding)

Gallery detail modals may attach one compact species control directly to each
offered detection box. This is an object selector and one-tap answer, not a
second toolbox or a blank-canvas labeling mode.

1. The primary segment shows the localized proposed species. Activating it
   records only object bird presence plus species confirmation for that
   detection through `/api/labels/answer`. It never confirms bbox quality,
   image-level bird presence, or sibling detections.
2. The secondary `▾` segment opens the existing shared species picker for that
   detection. It does not introduce a second picker implementation.
3. A check mark means canonical current human evidence exists. AI decision
   state and legacy review columns must never create the check appearance.
   Label colour and the confirm/retract action use that same canonical state.
   Homepage modal payloads refresh current human answers for both primary and
   companion detections on each request, independently of cached photo pools.
4. Confirmed and corrected known species use a fully green label, including
   the dropdown segment, with a white check and text.
   Species-unknown answers do not get a positive species confirmation colour.
   The frame turns green only for its own saved suitable-box verdict.
5. Gallery tiles remain read-only. They show independent human-assessment
   counts: `Species n/N · Boxes n/N`. A box marked wrong counts as assessed;
   species explicitly marked unknown counts as answered. These are counts of
   offered detections, not images, positive approvals or training eligibility.
   Do not show a shared completion check for independent axes. All fact-writing
   controls stay inside the detail modal or the established Review workflow.
6. Every segment is keyboard focusable, carries an accessible name, and keeps
   a practical touch target. Focus/Full transforms, resizing, and active-box
   switching must keep labels aligned with their boxes.

**Class vocabulary:**

| Class | Role |
|---|---|
| `wm-bbox-label-layer` | Viewer-aligned overlay layer that hosts offered-box species controls |
| `wm-bbox-label` | One species control aligned to a proposed detection box |
| `wm-bbox-label__confirm` | Primary segment that confirms the proposed species |
| `wm-bbox-label__status` | Read-only species/status segment when direct confirmation is unavailable |
| `wm-bbox-label__picker` | Secondary segment that opens the shared species picker |
| `is-reviewed` | Current canonical species evidence exists for this box |
| `is-corrected` | The current human species answer differs from the proposal |

## 0f. Review object answers (binding)

Review groups open objects by day and event. Event membership grants no approval.
Card actions answer selected open objects, or all open objects on the card when
none are selected, without an evidence-quality assessment. The shared viewer
resolves conflicts and keeps image-level statements distinct from object answers.
The diagnostic query and layout variants, rail panels and continuity-batch UI
have been removed. There is one Review layout.

The authenticated `/api/review/modal/<event_key>/<detection_id>` endpoint keeps
the shared detection editor available. `/api/review-thumb/<filename>` is shared
with Trash. Removed panel endpoints return 404.

## 0a. Hover Tooltip Convention (binding)

Every interactive control on an image-bearing surface — toolbox buttons,
review decision buttons, bbox toggles, navigation arrows, modal action-bar
buttons — must carry a **short, informative hover tooltip**. Tooltips are
discoverability, not noise.

**Rules:**

1. **Mechanism:** native `title="…"` attribute. No custom JS popovers, no
   auto-opening tooltips, no tooltip libraries. The browser's built-in delay
   keeps them unobtrusive.
2. **Content:** one short phrase (≤ ~60 chars) describing **what the button
   does**, in the same imperative voice as the action vocabulary
   (`Approve this event`, `Mark bounding box as wrong`, `Open in per-frame
   queue`). Not a restatement of the visible label.
3. **State-aware:** when a button is disabled, the tooltip explains *why*
   (`Login required`, `Event not yet eligible — needs ≥ 3 frames`).
4. **Accessibility parity:** every button also carries an `aria-label`
   (already required by the action frame contract). The `title` and
   `aria-label` may differ — `aria-label` is the canonical name of the
   action; `title` is the helpful one-liner.
5. **No emoji-only buttons without a tooltip.** If the visible glyph is
   ✓, ⋮, ↗, ‹, ›, ⌕, ☆, etc., a `title` is mandatory.
6. **One source per action.** When the same action appears in the toolbox
   macro and in a review decision rail, both surfaces use the same tooltip
   wording. Tooltip copy lives next to the action definition, not in
   per-template forks.

## 0b. Action Priority — Primary + Overflow (binding)

§0 establishes that every image-bearing surface uses the same
`tile_toolbox` macro with the same canonical action vocabulary. This
section adds the missing piece: **how many of those actions are
visible at once, and which ones**. Without this rule the toolbox would
grow monotonically as new actions land, and the most-used action would
stop being visually primary.

**The rule:**

1. **At most 3 primary actions per surface, with one explicit Review
   exception.** Two is the typical case; three is reserved for a genuinely
   distinct decide-action or restore-action. Review per-member tiles expose
   four primary actions (`Favorite`, `Change Species`, `Move to Trash`, and
   `No birds in full image`) because those high-frequency verdict actions must
   remain directly reachable during fast scanning and on touch devices. No
   other surface may exceed three without updating this rule and the table
   below in the same change.
2. **One overflow control.** A single `⋮` button (`wm-toolbox__more`)
   opens a dropdown (`wm-toolbox__menu`) that holds every action the
   surface supports but does not promote to primary.
3. **No nested overflow.** If a surface's overflow grows beyond ~6
   items, the right answer is to reconsider whether some of those
   items belong on that surface at all, not to nest the menu.
4. **New actions land in overflow by default.** A plan that wants to
   promote an action to primary on some surface MUST update the
   per-surface map in this section in the same change.
5. **No action becomes unreachable.** Every action the surface
   supports is either primary or in the overflow. There is no third
   bucket.
6. **Accessibility is part of the contract.** The `⋮` control must be
   keyboard-operable (Tab to focus, Enter / Space to open, Esc to
   close, Arrow keys to navigate items) and must announce as a menu
   to assistive technology. Tap-target size meets the existing
   convention (visible hit area ≥ 44×44 CSS px or equivalent).
7. **No persistence of menu state.** The overflow opens, closes, done.
   No `localStorage` flag, no per-page memory.

**Per-surface action map (binding):**

| Surface                          | Primary (always visible)                 | Overflow                                                                                |
|----------------------------------|------------------------------------------|-----------------------------------------------------------------------------------------|
| Stream                           | Favorite, View Details                   | Change Species, Move to Trash, No birds in full image                        |
| Gallery                          | Favorite, View Details                   | Change Species, Move to Trash, No birds in full image                        |
| Subgallery                       | Favorite, View Details                   | Change Species, Move to Trash, No birds in full image                        |
| Species / Species Overview       | Favorite, View Details                   | Change Species, Move to Trash                                                           |
| Review per-member tile (inside tile_toolbox) | Favorite, Change Species, Move to Trash, No birds in full image | View Details |
| Trash                            | Restore                                  | View Details, Change Species *(if exposed)* — Favorite is intentionally suppressed |
| Detail modals (`wm-bird-editor`) | Full / Focus, Species, Adjust box, Add missing bird | Bounding boxes, Full screen, Download, Open day, Favorite, Move to Trash, No birds in full image |

**Rules embedded in the table:**

- The order of primary actions in each row is the order they render
  in (per §0 point 5: bottom-of-image `wm-toolbox` or modal-footer
  `wm-toolbox--bar`).
- A surface MAY expose fewer actions than the table lists (omit per
  §0 point 2), but it MAY NOT promote an action to primary that the
  table places in overflow, without updating this section.
- The Review surface splits across two rows on purpose. The
  event-level decide-verbs (Approve Event, Move Event to Trash) live
  on the card header — they are primary by virtue of being in their
  own row above the tiles, not by toolbox-priority rules. The
  per-member tile toolbox inside the card promotes Change Species,
  Move to Trash, and No birds in full image out of the overflow alongside
  Favorite — four primary actions for the per-tile case. The
  hover-only reveal is overridden by the Review Grid CSS so the
  buttons are always visible: the per-tile verdict is the
  highest-frequency Review action and hover-only is too fragile on
  touch devices and during fast scans.
  Tile sizes 8/row and 16/row (contact-sheet mode) restore the
  hover-only reveal because the always-on chrome would otherwise
  cover most of the bird image.
- Detail modals inherit the primary/overflow split shown above, NOT
  the split of the surface that opened them. A modal opened from
  Stream and a modal opened from Trash both use the "Detail modals"
  row.
- **Detail modals** use the `wm-bird-editor` exception defined in §0d. The
  image remains primary, while selection and editing stay visible in one
  stable rail. Detection-only actions such as Favorite and Move to Trash are
  disabled when the active object was manually added.
- Orphans (top-level) and Restore are listed in `web/` as
  routes but do not render image tiles with the `tile_toolbox`
  macro. They are intentionally absent from this table. If a future
  plan adds toolbox-bearing tiles to one of these surfaces, that
  plan adds its row here in the same change.
- **Trash:** Favorite is intentionally suppressed on this surface —
  a discarded tile cannot meaningfully be favorited. The
  `restore-single` legacy button under each tile was removed on
  2026-05-14; the primary `↩` Restore in the toolbox is the only
  per-tile single-restore path. Bulk-restore via the page-header
  `Restore` button (Checkbox + multi-select) is a different
  workflow and remains.

**Audit:** check this table against the codebase after touching any
image-bearing surface. The table is the source of truth — if a
template disagrees with it, the template is wrong, unless the same
commit amends this table.

**Cross-reference:** the toolbox primary+overflow rule (introduced
2026-05-14) defines the migration sequence and the macro extension
that implements this contract.

Variants follow the BEM modifier `--` and are always set in addition to the base
class (for example `wm-modal wm-modal--form`, `wm-tile wm-tile--review`).

## Current Conventions

- `assets/design-system.css` is the authoritative source for shared UI
  primitives such as buttons, badges, review-stage controls, tiles, and modal
  subcomponents.
- Detection-bearing surfaces currently belong to one of two shells:
  `public shell` for Stream, Gallery, Species, and Species Overview, and
  `workbench shell` for Review and Trash. Layout density may differ, but shared
  detection components must not fork semantics or wording between shells.
- New shared buttons must use `btn` plus the design-system modifiers such as
  `btn--primary`, `btn--secondary`, `btn--danger`, `btn--accent`, `btn--success`,
  `btn--info`, `btn--outline-primary`, `btn--outline-danger`, `btn--sm`,
  `btn--lg`, and `btn--block`.
- Legacy Bootstrap-era button classes still exist in older surfaces such as
  `settings`, `edit`, `login`, and `partials/taskbar`. They are tolerated only
  as migration debt. Do not introduce new `btn btn-primary`,
  `btn btn-outline-*`, `btn-light`, or similar legacy-only patterns.
- Detection modal composition should continue to flow through
  `templates/components/detection_modal.html`, which already composes
  `modal_image_viewer.html`, `modal_detection_info.html`, and
  `modal_action_bar.html`.
- Review uses the shared event-card grid and detection modal (§6).
- Trash should follow the same `workbench shell` logic:
  summary bar above, left-side ops rail for batch/range/import controls, and
  the item surface on the right.
- Review and modal status text mapping must come from one shared helper or
  macro, not from duplicated inline label logic.
- If markup or decision logic repeats in 2 or more surfaces, extract a shared
  partial, macro, or Python helper instead of duplicating it again.
- `assets/js/gallery_utils.js` is still an oversized compatibility module.
  New unrelated behavior should go into a dedicated JS file rather than growing
  that file further.

## Detection Action Frame Contract

The shared detection action frame is the canonical control surface for
detection-bearing tiles, filmstrips, and modal/detail surfaces.

- Canonical action vocabulary:
  `View Details`, `Favorite`, `Change Species`, `Adjust Bounding Box`,
  `Correction Details`, `Move to Trash`, `Restore`, `Correct`, `Wrong`,
  `Approve`, `No birds in full image`
- Surfaces may omit actions only when the subject identity or route does not
  support them. They must not rename the same underlying action on another
  surface.
- New detection controls must use delegated `data-action` handlers. Do not add
  new inline `onclick` handlers for detection actions.
- Public surfaces must render the same frame for guests and authenticated users.
  Protected actions must stay visible in a disabled/login-required state rather
  than disappearing.
- Workbench surfaces may stay authenticated-only, but when they reuse the frame
  they should keep the same wording and ordering.
- Review-side utility panels may keep short section headings, but action labels
  themselves should stay canonical, for example `Change Species`,
  `BBox Confirm`, and `BBox Reject`.
- Review quick-species strips may stay on the local select/confirm path, but
  the Review species section must still expose one explicit route into the full
  species picker such as `Choose another species`.
- Review quick-species state should be legible at a glance: the default
  suggestion and the current selection should use distinct visual markers
  instead of relying on helper copy alone.
- Viewer/navigation controls such as view mode, close, next/previous, and download
  are not canonical detection actions. Previous, Next, More, and Close form the
  compact header navigation. The frequent `Focus | Full` choice sits directly
  on the image; lower-frequency viewer utilities live in More. They must not
  replace or duplicate the detection-action frame.
- In detail modals, object actions such as `Favorite`, `Change Species`, and
  `Adjust Bounding Box`
  belong to the image hover toolbox. The compact modal header stays a
  viewer/navigation surface and never duplicates object actions.

## Detection Presentation Anti-Drift Rules

`docs/UI_STANDARD.md` defines shared frontend composition rules and the
semantic source of truth for detection presentation.

To prevent future drift between surfaces:

- Detection badge meaning, species/title trust semantics, and review approval
  semantics must come from one shared source per concern.
- Templates may compose shared values, but they must not re-derive badge labels,
  manual-vs-AI meaning, or species/title trust rules from raw DB fields in new
  local inline logic.
- When a detection surface needs status text, species/title display values, or
  review-state display values, prefer a presenter/helper or shared macro input
  over branching directly in the template.
- New review actions should use `data-*` attributes plus delegated JS handlers.
  Do not add new inline `onclick="..."` handlers with serialized dynamic data.
- If a modal/detail footer exposes detection actions, it must consume the same
  action-frame vocabulary used by tile and filmstrip surfaces instead of
  inventing modal-only wording such as `Relabel` or `Delete`.
- If a semantic rule changes in one detection surface and should apply to other
  detection surfaces, the shared helper/macro contract must be updated first, or
  in the same change.
- Any PR that changes shared detection semantics must update this file if the
  contract, ownership, or allowed patterns changed.

## Review Checklist

- Use `assets/design-system.css` for shared primitives; page-local CSS should be
  limited to surface-specific layout.
- Prefer shared modal and review-stage compositions over building another local
  variant.
- Do not add new legacy Bootstrap button variants to templates.
- Do not duplicate decision-state or badge semantics in templates.
- Do not rename canonical detection actions per surface.
- Do not add new inline event handlers for review/detection interactions; use
  delegated handlers with `data-*` attributes.
- If a JS file is already a mixed-responsibility module, add new work in a
  dedicated file unless it is the same responsibility.
- If a template pattern repeats in 2 or more places, extract it before the
  third copy lands.
- If a shared detection semantic changed, verify whether
  `docs/UI_STANDARD.md` and the active detection-presentation workflow need the
  same update.

---

## 1. Modal Types (mandatory)

**Types**
- `wm-modal` (Detail/Review with Image-Viewer)
- `wm-modal wm-modal--form` (Settings forms, e.g., Add/Edit Camera)

### 1.1 Standard Modal (wm-modal) — two-stockwork layout

The detail modal is **header + body**. A compact navigation group lives in the
header (right slot); there is no separate footer row. Single-detection
frames render only the image in the body. Multi-detection frames add a
slim sibling-selector strip below the image — clicking a strip chip
*or* clicking a companion bbox on the canvas switches the active
detection, which retargets the header viewer state and every active
detection action in the image toolbox.

```html
<div class="modal fade gallery-modal wm-modal"
     id="modal-{{ group_id }}-{{ detection_id }}"
     tabindex="-1"
     aria-hidden="true"
     data-modal-group="{{ group_id }}"
     data-image-path="{{ image_path }}">

  <div class="modal-dialog modal-xl modal-dialog-centered modal-dialog-scrollable modal-fullscreen-md-down wm-modal__dialog">
    <div class="modal-content wm-modal__content">

      <div class="modal-header wm-modal__header">
        <div class="wm-modal__title">
          <span class="wm-modal__title-text">{{ title }}</span>
          <span class="wm-modal__title-sub">
            {{ date }} {{ time }} · <em>{{ species }}</em>
            · <span class="wm-modal__title-conf">{{ od_pct }}% / {{ cls_pct }}%</span>
            · {{ ai_status }}
          </span>
        </div>
        <div class="wm-modal__header-actions">
          <!-- modal_action_bar.html with in_header=true, show_maximize=true, show_close=true -->
          <!-- Prev · Next | More (Boxes / Full screen / Download / Day) · Close -->
        </div>
      </div>

      <div class="wm-modal__body">
        <div class="wm-modal__image wm-toolbox-host">
          <!-- render_image_viewer(show_view_mode=true) + tile_toolbox (surface='detail_modal') -->
        </div>
        {# Multi-bird frames only: #}
        <div class="wm-modal__sibling-strip">
          <!-- render_detection_info — slim chips, no per-card buttons -->
        </div>
      </div>

    </div>
  </div>
</div>
```

**Sizing.** `.wm-modal__dialog` is `max-width: min(1600px, 100vw - 4rem)`.
The image inside `.wm-modal__image .wm-image-viewer__img` is capped at
`max-height: calc(100vh - var(--wm-modal-chrome))` with
`object-fit: contain`, where `--wm-modal-chrome` defaults to `10rem`
(header + padding budget) and shrinks to `4rem` in maximized mode.

**Action hierarchy is binding.** Previous, Next, and Close are the only
always-visible header controls. The shared `wm-bird-editor` rail immediately
below the photo owns `Full | Focus`, species, box editing, manual-object
creation, and the one More menu. Boxes, Maximize, Download, Day, Favorite, and
eligible destructive actions live in that menu. No surface adds a parallel
image toolbox, header More menu, `.wm-modal__action` footer, or utility row.

**Active-detection contract (multi-bird).** Clicking either a
companion bbox on the canvas or a `.sibling-card` chip dispatches
`wmb:active-detection-change` on the modal root. The handler
(`gallery_utils.js`) updates `.wm-image-viewer[data-bbox-*]`, the
header `.bbox-toggle[data-detection-id]` + `data-current-bbox`, all active
image-toolbox actions carrying `[data-detection-id]`, redraws the
canvas, and re-applies smart-zoom if active. Surfaces consuming
`detection_modal.html` get this for free.

---

**Review-specific modifiers:**
- `wm-modal__image--solo` — applied to the modal image container when displaying
  a non-detection image without bbox overlays or sibling panels.

**Layout modifiers:**
- `wm-modal--maximized` — toggled by `toggleModalMaximize()` from the
  `⛶` button in the header action group. Dialog fills the viewport
  edge-to-edge; subtitle and sibling-strip hide. State is per-modal-
  open (reset on `hidden.bs.modal`); not persisted across opens.

**Retired classes (2026-05-23):**
- `wm-modal__action` — the separate footer row that hosted the action
  bar. Replaced by `.wm-modal__header-actions` inside the header.
  CSS rule is retained for one release as a no-op safety net for
  any external override; physical removal in a follow-up.
- `wm-modal__info` (as a default body element) — the rich info block
  below the image. Single-detection content moved into
  `.wm-modal__title-sub`; multi-detection content reduced to the
  `.wm-modal__sibling-strip` chip variant. The CSS rule remains;
  it is no longer rendered by `detection_modal.html`.

**New classes (2026-05-23):**
- `wm-modal__header-actions` — flex slot on the right side of
  `wm-modal__header`. Hosts the header-action-group variant of
  `modal_action_bar.html`.
- `wm-modal__title-conf` — span around the `87% / 70%` OD/CLS
  confidence pair in the subtitle. Uses `font-variant-numeric:
  tabular-nums` for stable column alignment.
- `wm-modal__sibling-strip` — slim horizontally-scrolling strip
  below the image on multi-detection frames. Hosts `.sibling-card`
  chips without per-card action buttons.
- `wm-modal--maximized` — full-viewport modifier (see above).
- `modal-action-bar--in-header` — compact navigation variant with Previous,
  Next, More, and Close. Its More menu owns viewer utilities and destinations;
  it never duplicates detection mutations from the image toolbox.
- `wm-view-mode-toggle--viewer` — direct `Focus | Full` segmented choice,
  centred near the bottom edge of detection-backed detail images.
- `wm-view-mode-toggle--gallery` — direct `Crop | Full` segmented choice for
  gallery-grid thumbnails.

---

### 1.2 Form Modal (wm-modal wm-modal--form)

```html
<div class="modal fade wm-modal wm-modal--form"
     id="modal-settings-{{ form_id }}"
     tabindex="-1"
     aria-hidden="true">

  <div class="modal-dialog modal-lg modal-dialog-centered wm-modal__dialog">
    <div class="modal-content wm-modal__content">

      <div class="modal-header wm-modal__header">
        <div class="wm-modal__title">
          <span class="wm-modal__title-text">{{ title }}</span>
          <span class="wm-modal__title-sub">{{ subtitle }}</span>
        </div>
        <button type="button" class="btn-close wm-modal__close" data-bs-dismiss="modal"></button>
      </div>

      <form class="wm-modal__form" method="post" action="{{ form_action }}">
        <div class="wm-modal__body">
          <div class="wm-modal__fields">
            <!-- Form fields -->
          </div>
        </div>

        <div class="wm-modal__action wm-modal__action--form">
          <div class="wm-modal__actions">
            <button type="button" class="btn btn--secondary" data-bs-dismiss="modal">Cancel</button>
            <button type="submit" class="btn btn--primary">Save</button>
          </div>
        </div>
      </form>

    </div>
  </div>
</div>
```

---

## 2. Tile Types (mandatory)

**Types**
- `wm-tile` (Standard, Gallery/Species/Stream)
- `wm-tile wm-tile--review` (Review/Orphans)
- `wm-tile wm-tile--bbox` (Thumbnail macro with bounding box)

### 2.1 Standard Tile (wm-tile)

```html
<div class="wm-tile" data-detection-id="{{ detection_id }}">
  <button type="button"
          class="wm-tile__button"
          data-bs-toggle="modal"
          data-bs-target="#modal-{{ group_id }}-{{ detection_id }}">

    <div class="wm-tile__media">
      <img class="wm-tile__image"
           src="{{ thumb_url }}"
           alt="{{ common_name }}">

      <span class="wm-tile__badge">{{ count }}</span>
    </div>
  </button>

  <div class="wm-tile__body">
    <span class="wm-tile__name">{{ common_name }}</span>
    <span class="wm-tile__latin">{{ latin_name }}</span>
  </div>
</div>
```

---

### 2.2 Review Tile (wm-tile wm-tile--review) — Legacy

> **Note:** This tile type has been replaced by the `review-grid`
> composition (see §6). The CSS class `wm-tile--review` still exists in
> `design-system.css` but the HTML structure below is no longer produced by any
> template. Retained here for reference only.

```html
<div class="wm-tile wm-tile--review" data-filename="{{ filename }}">
  <div class="wm-tile__select">
    <input class="form-check-input wm-tile__checkbox" type="checkbox" value="{{ filename }}">
  </div>

  <span class="wm-tile__badge wm-tile__badge--reason">{{ reason_label }}</span>

  <button type="button"
          class="wm-tile__button"
          data-bs-toggle="modal"
          data-bs-target="#modal-review-{{ filename|replace('.', '_') }}">
    <div class="wm-tile__media">
      <img class="wm-tile__image"
           src="{{ thumb_url }}"
           alt="{{ filename }}">
    </div>
  </button>

  <div class="wm-tile__body">
    <span class="wm-tile__meta">{{ formatted_date }}</span>
    <span class="wm-tile__name">{{ filename }}</span>
    <span class="wm-tile__size">{{ file_size_str }}</span>
  </div>

  <div class="wm-tile__actions">
    <!-- review actions -->
  </div>
</div>
```

---

### 2.3 BBox Tile (wm-tile wm-tile--bbox)

`data-bbox-*` values are percentages (0-100) from the detection record.

```html
<div class="wm-tile wm-tile--bbox"
     data-bbox-x="{{ bbox_x }}"
     data-bbox-y="{{ bbox_y }}"
     data-bbox-w="{{ bbox_w }}"
     data-bbox-h="{{ bbox_h }}">
  <button type="button"
          class="wm-tile__button"
          data-bs-toggle="modal"
          data-bs-target="#modal-{{ group_id }}-{{ detection_id }}">

    <div class="wm-tile__media wm-tile__media--bbox">
      <img class="wm-tile__image wm-tile__image--bbox"
           src="{{ thumb_url }}"
           alt="{{ common_name }}">
    </div>
  </button>

  <div class="wm-tile__body">
    <span class="wm-tile__name">{{ common_name }}</span>
    <span class="wm-tile__latin">{{ latin_name }}</span>
  </div>
</div>
```

---

## 3. Standard Action Bar

```html
<div class="modal-action-bar">
  <div class="modal-action-bar__group modal-action-bar__nav">
    <!-- Previous · Next -->
  </div>

  <div class="modal-action-bar__group">
    <!-- More menu · Close -->
  </div>
</div>
```

The detail-modal header deliberately exposes only four controls. The More menu
uses the same keyboard behavior as the shared toolbox overflow: Enter/Space
opens it, Escape closes it, and Arrow Up/Down moves through menu items.

---

## 4. Standard Image Viewer

```html
<div class="modal-image-viewer wm-image-viewer">
  <img class="wm-image-viewer__img bbox-base-image"
       src="{{ image_url }}"
       data-detection-id="{{ detection_id }}"
       role="button"
       data-bs-dismiss="modal">

  <canvas class="wm-image-viewer__overlay bbox-overlay"></canvas>

  <div class="wm-view-mode-toggle wm-view-mode-toggle--viewer smart-zoom-toggle"
       role="group" aria-label="Image view">
    <button data-view-mode="zoom" aria-pressed="true">Focus</button>
    <button data-view-mode="full" aria-pressed="false">Full</button>
  </div>
</div>
```

The segmented view control is rendered only when a valid bbox exists. Its
saved preference follows the viewer scope, and `Focus` is re-applied to the
active box after `wmb:active-detection-change`.

---

## 5. Tile Toolbox (wm-toolbox)

The tile toolbox is the shared action overlay for detection-bearing tiles,
filmstrip items, and modal image viewers. It is rendered by the
`tile_toolbox` macro in `templates/partials/tile_toolbox.html`.

**Host pattern:** Any container that hosts a toolbox adds the class
`wm-toolbox-host`. This enables hover/focus reveal behavior. Used on
`wm-tile`, `obs-filmstrip__item`, and `wm-modal__image`.

**Class vocabulary:**

| Class | Role |
|---|---|
| `wm-toolbox-host` | Container that reveals the toolbox on hover/focus |
| `wm-toolbox` | Toolbox root — positioned overlay inside the host |
| `wm-toolbox--bar` | Bar variant — horizontal strip layout |
| `wm-toolbox__btn` | Individual action button |
| `wm-toolbox__btn--toggled` | Pressed-state modifier for stateful buttons (zoom toggle, …) |
| `wm-toolbox__btn--locked` | Disabled / login-required state |
| `wm-toolbox__fav` | Favorite toggle button (special styling) |
| `wm-toolbox__menu` | Dropdown trigger (three-dot / more button) |
| `wm-toolbox__more` | Alias for the menu trigger |
| `wm-toolbox__dropdown` | Dropdown panel |
| `wm-toolbox__item` | Individual dropdown menu item |

**Stateful toolbox buttons:** any
toolbox button that carries an on/off state — the legacy Review smart-zoom toggle,
the bbox-overlay toggle — must mirror its state
on **all three** of the following at the same time:

1. `aria-pressed="true|false"` — for assistive tech.
2. `wm-toolbox__btn--toggled` class — for the visual pressed style.
3. State-aware `title=` — short imperative copy that describes what
   the **next** click will do (e.g. `Show full image` when zoomed in,
   `Zoom into bird` when full). When the action degrades (e.g. zoom
   intent persisted but no bbox on the current frame), the title
   becomes a state explanation (`No bounding box — zoom unavailable
   for this frame`).

`assets/js/gallery_utils.js:applySmartZoomToggleState` is the canonical state
implementation. It supports both the direct `Focus | Full` segmented control
in detail viewers and the legacy single-button Review control. Segmented
options expose one `aria-pressed="true"` choice; the legacy button retains the
pressed modifier, state-aware title, and emoji-label swap.

**Workspace-scoped persistence:** Review cards use `.wm-viewer-scope` and
`data-zoom-pref-key="wmb_review_zoom_pref"` so sibling cards share the same zoom
preference. Shared viewer controls retain their normal state behavior.

```html
<div class="wm-toolbox-host">
  <img class="wm-tile__image" src="..." alt="...">

  <div class="wm-toolbox">
    <button class="wm-toolbox__fav" data-action="favorite">...</button>
    <button class="wm-toolbox__btn" data-action="view-details">...</button>
    <div class="wm-toolbox__menu">
      <button class="wm-toolbox__more">...</button>
      <div class="wm-toolbox__dropdown">
        <button class="wm-toolbox__item" data-action="change-species">Change Species</button>
        <button class="wm-toolbox__item" data-action="move-trash">Move to Trash</button>
      </div>
    </div>
  </div>
</div>
```

---

## 6. Review Grid (review-grid, default since 2026-05-23)

The Review Desk renders as a **vertical stack of self-contained event
cards** (`templates/review_grid.html` +
`templates/components/review_grid_card.html`). One card per BirdEvent.
Each card carries its own header (date · time window · frame count ·
species and object-answer controls), its own tile grid
sized by the shared page slider, and its own
modal-detection markup for the shared lightbox path.

**Layout:** vertical event-card stack with a filter card above and a
day navigation above the cards. **No left side rail. No right
decision rail.** Per-card decision verbs live in the card header;
each action follows the card selection.

Review sizing keeps images primary: new Crop sessions start at 320px and
Full at 640px; saved per-mode sizes remain respected. Tiles shrink to the
available card width on narrow screens. Desktop event headers and day
controls use compact spacing; narrow or touch layouts keep controls at least
44px high. Shared image viewers, toolbox actions and image aspect ratios
remain unchanged.

Open and Done share numbered pagination. Lists of up to 20 pages show every
number; longer lists show the first and last pages, the current page and up to
two neighbours on each side, with ellipses for omitted ranges. The current page
is highlighted and marked with `aria-current="page"`. Previous/Next appear when
available. Longer lists also offer a labelled Go to page number field and Go
button; Enter submits the field. Native input validation accepts whole page
numbers from 1 to the last page. Links and the jump form preserve the tab,
selected day and date filters. Controls wrap on narrow screens, retain visible
keyboard focus, and use at least 44px targets
on narrow or touch layouts.

### Keyboard navigation (review_grid_keyboard.js)

The Review Grid supports hands-on-keyboard triage. The binding is
operator-defined and drives the **existing** card-header action buttons
via synthetic clicks — it re-implements no action, so disabled-state,
Smart-Mode scope, confirm dialogs and toasts are all inherited.

| Key | Action |
|---|---|
| `↑` / `↓` | Move the event cursor to the previous / next card. The active card shows `.review-grid__card--kbd-active` (a blue focus ring inside the hard-black event outline) and scrolls to centre. |
| `←` / `→` | Cycle the *armed* action within the active card: **Invert → Confirm species → Change species → Bird · species unknown → No bird(s) in crop(s) → Move to Trash**. The armed button shows `.review-grid__card-action--armed`. Disabled verbs (e.g. Approve before a species is picked) are skipped. |
| `Space` | Fire the armed action (clicks the armed button). |
| `Enter` | Open the active event's detail modal (first actionable tile). |

The handler stays inert while a text field, day navigation or pagination
control is focused, or any modal is open. Pagination links retain Enter
activation, and Go retains Enter/Space activation without firing a card action.
New per-card action verbs added to `review_grid_card.html`
must also be added to `ACTION_ORDER` in `review_grid_keyboard.js` if
they should be reachable from the `←/→` cursor.

The species picker (`assets/js/species_picker.js`) is matched in the
grid handler's modal-open guard (it is a bespoke overlay, not a
Bootstrap modal), so while it is open it owns the arrow keys and the
grid cursor stands still behind it.

**Picker keyboard binding** (active whenever `WmSpeciesPicker` is open,
e.g. after Relabel):

| Key | Action |
|---|---|
| `↑` / `↓` | Move a highlight cursor through the species list (`.wm-species-picker-item`), skipping section headers and hints. |
| `Space` / `Enter` | Pick the highlighted species. `Space` only picks once a cursor exists, so a space can still be typed into the search box before navigating. |
| `←` / `Esc` | Cancel — resolves to no change and returns to the event card. |

## 6bb. Review Grid card (review-grid-card-*)

`templates/components/review_grid_card.html` renders one event card. The header
shows the species proposal, date, time window and frame count. Controls are
Invert, Confirm species, Change species, Bird · species unknown,
No bird(s) in crop(s) and Move to Trash.
Selecting objects scopes an answer to that selection; without a selection it
answers every open object in the card. No bird(s) in crop(s) is an object answer,
including when tiles are displayed in Full mode; it never asserts image-wide absence.
The full-image no-bird action remains in the shared viewer with its safety gate.
Move to Trash confirms the selected objects, or all open objects in that event,
then uses `/api/trash/move`; it creates no bird-presence or species answer.

Cards keep the shared viewer, toolbox and species colour. Selection clears after
a successful answer or trash action and when leaving or restoring the page.
Cancelled or failed actions keep the current selection. Shift-click anchors
belong to their own event card.
`assets/js/review_grid.js` owns selection and per-mode tile sizing; the page emits
one shared detection modal per object. There are no evidence controls, continuity
batch actions or panel layouts.

## 6d. Species colour coding

`core.species_colours.assign_species_colours` maps sorted scientific names to
eight Wong palette slots. Page builders assign the colour map across the visible
objects. Cards carry `data-species-colour` and `--cell-species-colour`; canvas
boxes carry `speciesColour`. The CSS palette owns Light/Dark variants, and
`gallery_utils.js` reads it for canvas strokes. Responsive bbox labels retain
species colour when abbreviated. The removed diagnostic species-reference
image overlays have no current consumer.

## 6e. Subgallery Concurrent Visits

Two species in the same 30-minute window stay **two `BirdEvent`s**
in the data layer — the same-species split rule at `core/events.py`
is not weakened. The Subgallery groups them visually so the operator
can see at a glance that the visits happened at the same time.

**Binding architectural decision:** the grouping is UI-only. No new
database tables. No changes to `core/events.py`.

**Helper:** `core.gallery_core.group_concurrent_observations(observations, *, window_minutes=5.0)`
- Pure function, no DB calls, no mutation.
- Takes the observation list returned by `group_detections_into_observations`.
- Returns `list[list[dict]]` — each inner list is one visit window.
- Tolerance: `window_minutes=5` by default. An observation joins the
  current window when its `start_time` is within tolerance of the
  running max `end_time` of that window.
- Deterministic given the same input set, independent of input order.

**Template contract (`templates/subgallery.html`):**
- The Subgallery iterates `visit_windows`, not the flat `observations`
  list. `observations` stays in the template context for modal
  rendering (backwards compatibility).
- A visit window of **one** observation renders with **no** wrapper —
  the common case must not gain visual noise.
- A visit window of **two or more** observations is wrapped in a
  `.concurrent-visit` shell with a single header line
  (`HH:MM:SS – HH:MM:SS · N species`).
- Each observation inside a visit window remains its own `.wm-tile`,
  clickable, with its own filmstrip, its own toolbox, its own modal.
- Species colouring (§6d) inside a concurrent visit uses the same
  palette assignments as elsewhere — a window with Elster + Kohlmeise
  shows two different colours inside one shell, which is exactly the
  point.

**BEM block (`assets/design-system.css`):**
- `.concurrent-visit` — the outer container. Full-row child of
  `#subgallery-grid` via `grid-column: 1 / -1`.
- `.concurrent-visit__header` — single-line header with time range +
  species count.
- `.concurrent-visit__time` — tabular-nums time range.
- `.concurrent-visit__meta` — species count, separated by a middle dot.
- `.concurrent-visit__shell` — nested grid that mirrors
  `.gallery-grid`'s `minmax(240px, 1fr)` so tiles inside the shell
  line up visually with tiles outside.
- Background uses `--color-surface-2`, clearly lighter than the tile
  background, so tiles inside still read as individual cards.

**Sort behaviour:**
- The server-side route re-sorts visit windows using the active
  `sort_by` parameter applied to the first observation of each window
  (`time_desc` → `end_time` desc, `time_asc` → `start_time` asc,
  `score` → `max(best_score)` desc, `species` → common name asc).
- Ties resolve deterministically via the internal
  `(start_time, end_time, observation_id)` sort inside
  `group_concurrent_observations`.

**Pagination note (follow-up, not implemented yet):**
- Today the helper runs on the already-paged slice, so a visit window
  that would straddle a page boundary splits across pages. When the
  Subgallery gains pagination that cares about this, the rule becomes
  "paginate by visit window, not by observation".

---

## 6f. Canonical Human Corrections

Human correction happens in the ordinary image UX. There is no
`/admin/labeling` workspace and no training-language detour for routine image
actions. Existing proposals are corrected in their detail modal; whole-image
and species answers continue through their normal controls.

**Full-image no-bird safety gate (binding):**

- The canonical action label is `No birds in full image`.
- Every invocation displays the complete affected image before confirmation,
  even when the originating surface currently shows a crop. Multi-image
  actions display each complete image before the request is sent.
- The confirmation is never remembered in `localStorage`, `sessionStorage`, or
  runtime state. Every invocation requires a fresh explicit confirmation.
- `Cancel`, `Escape`, backdrop dismissal, an unavailable full-image preview,
  or an unavailable dialog implementation must fail closed and send no write.
- Only the affirmative button shown after the complete image may submit
  `image_bird_presence=absent` to `/api/labels/answer`. The transaction writes
  absence for each detected object and the image statement.
- Active manually added birds block confirmation. The dialog names the number
  to remove first, for example `Remove the 2 birds you added by hand first`.
- A current `No birds in full image` statement blocks adding a manual bird
  until that statement is withdrawn. Manual objects keep their own revisions.

**Component ownership:**

- `templates/partials/no_bird_confirm_dialog.html` owns the single shared
  dialog structure and is mounted once from `templates/base.html`.
- `assets/design-system.css` owns the `wm-no-bird-confirm` presentation,
  including viewport sizing and backdrop treatment. Dialog layout must not be
  rebuilt with JavaScript inline styles.
- `assets/js/tile_actions.js` resolves the safe full-image URL, populates the
  shared dialog, and owns its sequential multi-image and fail-closed behavior.

**Class vocabulary:**

| Class | Role |
|---|---|
| `wm-no-bird-confirm` | Shared native dialog and canonical `wm-modal` variant |
| `wm-no-bird-confirm__content` | Dialog content stack |
| `wm-no-bird-confirm__summary` | Frame-wide impact statement |
| `wm-no-bird-confirm__preview` | Complete affected frame preview |
| `wm-no-bird-confirm__hint` | Whole-image scope reminder |
| `wm-no-bird-confirm__actions` | Cancel and affirmative action row |

**Semantic contract:**

- Image bird presence and proposed-object bird presence are separate questions.
  Rejecting the outlined object must not record whole-image `No bird`.
- Box quality, corrected coordinates, and species identity are independent.
  Unanswered questions remain unknown and must not be inferred from another
  answer.
- A corrected box is rendered as the current answer without overwriting the
  original proposal snapshot; its interaction follows binding §0d.
- Species can be confirmed, corrected, marked wrong, or left unknown. A wrong
  proposal never forces a replacement guess.
- Unanswered axes remain unknown. Export or downstream consumers never create
  or modify human facts.
- Multiple proposals on one image are independent object subjects. The active
  box action and its Expert details retarget together; partial progress is
  shown as answered versus total proposed boxes without implying image-wide
  completion.

---

## Rules

1. Every modal structure uses a defined type: `wm-modal` or `wm-modal wm-modal--form`.
2. Standard image tiles use `wm-tile`, `wm-tile wm-tile--review` (legacy), or `wm-tile wm-tile--bbox`. Documented surface compositions such as Review Grid (§6/§6bb) use their own shared components; the legacy tile list is not exhaustive.
3. Review uses the `review-grid` event-card composition (§6 and §6bb).
4. Every detection-bearing browsing surface uses `tile_toolbox` (§5) for action overlays.
5. No template may build its own modal, tile, or toolbox structures.
6. Reuse the component families and surface-specific compositions documented here. This is not a global ban on every class without a `wm-` prefix.
7. Shared component styling belongs in the design system; preserve the documented structure and design tokens across surfaces.
8. Species colour slots (§6d) are deterministic across the visible page.
10. New surfaces that ship a bbox overlay must read the species slot from the host element's `data-species-colour` (or `box.speciesColour` on the canvas side), not from the legacy `BBOX_COLORS` rotation.
11. Storage usage (`/storage`) is an authenticated, non-image diagnostic page linked
    from the Settings retention panel. One explicit Measure storage action starts
    a background scan; normal page/status reads use the saved result. Show measured
    file categories, filesystem free space, SQLite reuse, overlapping protection
    reasons and their union separately. Show scan time, progress, partial errors,
    unavailable scopes and bounded growth history. Offer no cleanup action or
    automatic release of training protection on this page.
    The deletion preview and executor use the same `core/deletion_plan.py` rules.
    Show its six classes, training and receipt waits, unconfirmed exports, and
    next file deadlines. A prominent `alert alert-info` with `role="status"`
    shows whether automatic deletion was enabled at measurement time and the
    number of photos, files and bytes due within the following 14 days,
    including overdue files. Explain the fresh 14-day grace period for open
    cases and model non-birds; while off, assume activation today. Mark saved
    measurements clearly and request a new measurement after settings change.
    Missing deadline starts and incomplete scans use the same status pattern.
    Show a rough full-card forecast above the measurement, based on recent stored
    captures, measured free space minus the reserve, retained metadata and planned
    file expiry. Respect the deletion switch. Say not foreseeable when the model
    finds no lasting growth, and unavailable for insufficient history or incomplete
    scans. Settings changes require a new measurement. Below the single warning
    threshold in `core/storage_forecast.py`, use `alert alert-warning` with
    `role="status"` and suggest Delete everything in Settings or more storage.
12. Storage Retention lives in Settings (`#panelRetention`) using the standard
    `settings-panel` / `settings-field` shells. It has one Automatic deletion
    switch and the Old bird photos choices Keep previews / Delete everything.
    Opening the panel calculates the next 14 days with the same status notice
    as storage, explicitly including the existing Review backlog. The switch
    also gates Clean up now. Show the last run's removed file and row counts,
    bytes, missing files and errors. Original-serving routes return `410 Gone`
    for a retained image whose original was retired; history and answers remain
    available through surviving previews or placeholders.
13. A shared status notice above page content (`alert alert-info`, `role="status"`,
    polite live announcements) shows the authenticated owner when photo saving
    pauses or pressure cleanup removes files early. It states the blocking reason,
    continued live view, automatic resumption, and protection of favorites and
    pending training. Refresh it from cached runtime status; polling must not scan
    media or initiate cleanup. Explain in Settings and Storage that pressure cleanup
    may shorten the normal Review grace period while automatic deletion is on.

## 7. Guided Recovery (non-image settings workflow)

Guided USB recovery lives inside the existing **Data & Backups** settings panel.
The discovery callout uses `.recovery-discovery`; it never initiates recovery.
Snapshot rows expose `Review & restore`, which opens the canonical
`wm-modal wm-modal--form` preview. `.recovery-preview` is a responsive two-column
facts grid that collapses to one column below 640 px. The form must show source,
date, record counts, compatibility, required/available space, settings policy,
and replacement/checkpoint behavior before enabling its destructive action.

Both replacement and checkpoint acknowledgements are required. Merge remains a
separate archive action with distinct copy. After submission, the browser moves
to the independent runner page; that page owns progress, reconnect, failed-start
retry, and checkpoint rollback while the main Flask service is unavailable.


## Local development notice

The isolated local launcher adds one shared notice through `base.html`: Local
development, source snapshot time, local-only edits and disabled camera/jobs.
It appears on all rendered pages in this profile. Production does not inject
this context. Preserve the shared viewer/editor and existing authentication.

In the local profile, Live replaces camera playback with an immediate disabled
notice and omits its player and camera controls. Saved visits keep the canonical
image viewer and editing actions. No connection spinner or camera-retry action
should suggest that a live feed is expected in this profile.
