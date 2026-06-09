/*
 * Review Grid keyboard navigation.
 *
 * Operator-defined binding for hands-on-keyboard triage of the event
 * stack. Drives the EXISTING card-header action buttons via synthetic
 * clicks rather than re-implementing any action — so disabled-state,
 * Smart-Mode scope, confirm dialogs and toasts are all inherited from
 * review_grid.js for free.
 *
 *   ArrowUp / ArrowDown  → move the event cursor (prev / next card)
 *   ArrowLeft / ArrowRight → cycle the armed action within the active
 *                            card: Invert → Approve → Relabel → No Bird
 *                            → Trash
 *   Space                → fire the armed action
 *   Enter                → open the active event's detail modal
 *                          (first actionable tile)
 *
 * Convention: one document-level keydown listener, state held as the
 * active card element + an armed-action index. Cards are re-queried on
 * every keystroke so the navigation survives card/tile removal (the
 * grid mutates the DOM as events get actioned).
 */
(function () {
    'use strict';

    // Armed-action order is the operator's spec, left-to-right. Each
    // entry maps to the data-review-grid-action verb already wired in
    // review_grid.js.
    const ACTION_ORDER = [
        'invert_selection',
        'approve_event',
        'relabel_event',
        'no_bird_event',
        'trash_event'
    ];

    const APPROVE_INDEX = ACTION_ORDER.indexOf('approve_event');

    let activeCard = null;
    let armedIndex = APPROVE_INDEX; // default to Approve — the most common verb

    function allCards() {
        return Array.from(document.querySelectorAll('[data-review-grid-card]'));
    }

    function isTypingTarget(el) {
        if (!el) return false;
        const tag = el.tagName;
        return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT'
            || el.isContentEditable;
    }

    // The species picker is a bespoke overlay, not a Bootstrap modal,
    // so it needs its own selector to suppress grid keys while open.
    function aModalIsOpen() {
        return !!document.querySelector(
            '.modal.show, .wm-modal.show, [data-bs-backdrop].show, .wm-species-picker-overlay'
        );
    }

    function clearActive() {
        if (activeCard) activeCard.classList.remove('review-grid__card--kbd-active');
        activeCard = null;
    }

    function setActiveCard(card, scrollIntoView) {
        const cards = allCards();
        if (!cards.length) { clearActive(); return; }
        if (!card || cards.indexOf(card) === -1) card = cards[0];
        const movedToDifferentCard = activeCard !== card;
        if (activeCard && movedToDifferentCard) {
            activeCard.classList.remove('review-grid__card--kbd-active');
        }
        activeCard = card;
        activeCard.classList.add('review-grid__card--kbd-active');
        // Landing on a fresh card re-arms Approve, so a Trash/No-Bird
        // cursor never carries over into the next event — a wrong Space
        // there would be destructive.
        if (movedToDifferentCard) armedIndex = APPROVE_INDEX;
        renderArmedAction();
        if (scrollIntoView) {
            activeCard.scrollIntoView({ block: 'center', behavior: 'smooth' });
        }
    }

    function ensureActive() {
        const cards = allCards();
        if (!cards.length) { clearActive(); return false; }
        if (!activeCard || cards.indexOf(activeCard) === -1) {
            setActiveCard(cards[0], true);
        }
        return true;
    }

    function moveEventCursor(direction) {
        const cards = allCards();
        if (!cards.length) { clearActive(); return; }
        if (!activeCard || cards.indexOf(activeCard) === -1) {
            setActiveCard(cards[0], true);
            return;
        }
        const idx = cards.indexOf(activeCard);
        const next = Math.min(cards.length - 1, Math.max(0, idx + direction));
        setActiveCard(cards[next], true);
    }

    function actionButton(card, verb) {
        return card.querySelector('[data-review-grid-action="' + verb + '"]');
    }

    // Buttons that are disabled or absent (e.g. Approve before a species
    // is picked) are skipped so the armed cursor never lands on a
    // dead verb.
    function isArmable(card, verb) {
        const btn = actionButton(card, verb);
        return !!btn && !btn.disabled;
    }

    function renderArmedAction() {
        document.querySelectorAll('.review-grid__card-action--armed')
            .forEach(function (b) { b.classList.remove('review-grid__card-action--armed'); });
        if (!activeCard) return;
        // Snap the armed index onto the nearest armable verb so the
        // highlight is never on a disabled button.
        if (!isArmable(activeCard, ACTION_ORDER[armedIndex])) {
            const fallback = ACTION_ORDER.findIndex(function (v) {
                return isArmable(activeCard, v);
            });
            if (fallback !== -1) armedIndex = fallback;
        }
        const btn = actionButton(activeCard, ACTION_ORDER[armedIndex]);
        if (btn && !btn.disabled) {
            btn.classList.add('review-grid__card-action--armed');
        }
    }

    function cycleArmedAction(direction) {
        if (!ensureActive()) return;
        const n = ACTION_ORDER.length;
        // Walk in the requested direction until we land on an armable
        // verb, at most one full loop. (((x % n) + n) % n) keeps the
        // index positive for negative directions.
        for (let step = 1; step <= n; step++) {
            const raw = armedIndex + direction * step;
            const candidate = ((raw % n) + n) % n;
            if (isArmable(activeCard, ACTION_ORDER[candidate])) {
                armedIndex = candidate;
                renderArmedAction();
                return;
            }
        }
        // No armable verb at all — leave state as-is.
        renderArmedAction();
    }

    function fireArmedAction() {
        if (!ensureActive()) return;
        const verb = ACTION_ORDER[armedIndex];
        const btn = actionButton(activeCard, verb);
        if (btn && !btn.disabled) btn.click();
    }

    function openActiveModal() {
        if (!ensureActive()) return;
        // "Open the event" = open the first actionable tile's modal.
        const tile = activeCard.querySelector(
            '.review-grid__tile:not(.review-grid__tile--context) [data-tile-image][data-modal-target]'
        );
        if (!tile) return;
        const target = tile.dataset.modalTarget;
        const modalEl = target && document.querySelector(target);
        if (modalEl && window.bootstrap && window.bootstrap.Modal) {
            window.bootstrap.Modal.getOrCreateInstance(modalEl).show();
        }
    }

    function onKeydown(event) {
        if (event.defaultPrevented) return;
        if (event.metaKey || event.ctrlKey || event.altKey) return;
        if (isTypingTarget(event.target)) return;
        if (aModalIsOpen()) return;

        switch (event.key) {
            case 'ArrowUp':
                event.preventDefault();
                moveEventCursor(-1);
                break;
            case 'ArrowDown':
                event.preventDefault();
                moveEventCursor(1);
                break;
            case 'ArrowLeft':
                event.preventDefault();
                cycleArmedAction(-1);
                break;
            case 'ArrowRight':
                event.preventDefault();
                cycleArmedAction(1);
                break;
            case ' ':
            case 'Spacebar': // legacy key name
                event.preventDefault();
                fireArmedAction();
                break;
            case 'Enter':
                event.preventDefault();
                openActiveModal();
                break;
            default:
                break;
        }
    }

    // Keep the armed highlight in sync after the grid mutates a card
    // (selection change re-enables Approve, etc.).
    function onGridMutated() {
        if (activeCard && allCards().indexOf(activeCard) === -1) {
            // Active card was removed — advance to the nearest survivor.
            const cards = allCards();
            if (cards.length) setActiveCard(cards[0], false);
            else clearActive();
        } else {
            renderArmedAction();
        }
    }

    function bootstrap() {
        if (!document.querySelector('[data-review-grid-stack]')) return;
        document.addEventListener('keydown', onKeydown);
        document.addEventListener('change', onGridMutated);
        // Click on a card also moves the keyboard cursor there, so mouse
        // and keyboard agree on "which event is active".
        document.addEventListener('click', function (event) {
            const card = event.target.closest('[data-review-grid-card]');
            if (card) setActiveCard(card, false);
        });
        // Arm the first card so the very first keystroke has a target.
        ensureActive();
    }

    document.addEventListener('DOMContentLoaded', bootstrap);
    if (document.readyState !== 'loading') {
        bootstrap();
    }
})();
