/* Shared multi-bird editor for detection-backed detail modals. */
(function () {
    'use strict';

    const MIN_BOX = 0.018;
    const sharedObjectCache = new Map();

    function readJson(value, fallback) {
        try { return JSON.parse(value || ''); } catch (error) { return fallback; }
    }

    function clone(value) {
        return JSON.parse(JSON.stringify(value));
    }

    function objectKey(item) {
        if (item.object_key) return String(item.object_key);
        if (item.manual_object_id) return 'manual:' + String(item.manual_object_id);
        return 'detection:' + String(item.detection_id);
    }

    // An explicit "species unknown" clears the override, so the fallback chain
    // below would otherwise resurrect the classifier's withdrawn guess.
    const HUMAN_UNKNOWN_SOURCES = ['manual_unknown', 'manual_wrong'];

    function isHumanUnknownSpecies(item) {
        if (item.provenance === 'human_unknown') return true;
        if (item.human_review_state === 'reviewed_unknown') return true;
        const source = String(item.species_source || '').trim().toLowerCase();
        return HUMAN_UNKNOWN_SOURCES.indexOf(source) !== -1;
    }

    function normalize(item) {
        const manualId = item.manual_object_id == null ? null : Number(item.manual_object_id);
        const detectionId = item.detection_id == null ? null : Number(item.detection_id);
        const humanUnknown = isHumanUnknownSpecies(item);
        const speciesKey = humanUnknown ? null : (item.species_key || item.manual_species_override ||
            item.cls_class_name || item.od_class_name || null);
        const humanReviewState = item.human_review_state ||
            (humanUnknown ? 'reviewed_unknown' : 'unreviewed');
        let provenance = item.provenance;
        if (!provenance && humanUnknown) provenance = 'human_unknown';
        if (!provenance && manualId) provenance = 'manually_added';
        if (!provenance && humanReviewState === 'confirmed') provenance = 'human_confirmed';
        if (!provenance && humanReviewState !== 'unreviewed') provenance = 'manually_identified';
        if (!provenance) {
            provenance = item.species_source === 'manual' ? 'manually_identified' : 'model_proposal';
        }
        return {
            objectKey: objectKey(item),
            objectKind: manualId ? 'manual' : 'detection',
            manualObjectId: manualId,
            detectionId: detectionId,
            revision: item.revision == null ? null : Number(item.revision),
            speciesKey: speciesKey,
            commonName: item.common_name || (speciesKey ? speciesKey.replace(/_/g, ' ') : 'Bird · species unknown'),
            provenance: provenance,
            humanReviewState: humanReviewState,
            formattedDate: item.formatted_date || '',
            formattedTime: item.formatted_time || '',
            odClassName: item.od_class_name || null,
            odConfidence: item.od_confidence == null ? null : Number(item.od_confidence),
            clsClassName: item.cls_class_name || null,
            clsConfidence: item.cls_confidence == null ? null : Number(item.cls_confidence),
            isFavorite: Boolean(item.is_favorite),
            bbox: {
                x: Number(item.bbox_x), y: Number(item.bbox_y),
                w: Number(item.bbox_w), h: Number(item.bbox_h)
            }
        };
    }

    function validObject(item) {
        const box = item && item.bbox;
        return item && item.objectKey && box &&
            [box.x, box.y, box.w, box.h].every(Number.isFinite) &&
            box.w > 0 && box.h > 0;
    }

    function sameBox(a, b) {
        return ['x', 'y', 'w', 'h'].every(function (key) {
            return Math.abs(Number(a[key]) - Number(b[key])) < 0.0000001;
        });
    }

    // A species answer commits no geometry. Keep any box edit pending.
    function confirmedSpeciesState(saved, displayed) {
        const confirmed = clone(saved);
        const answer = displayed || saved;
        confirmed.speciesKey = answer.speciesKey;
        confirmed.commonName = answer.commonName;
        confirmed.humanReviewState = 'confirmed';
        confirmed.provenance = 'human_confirmed';
        const pending = displayed && !sameBox(saved.bbox, displayed.bbox)
            ? clone(confirmed) : null;
        if (pending) pending.bbox = clone(displayed.bbox);
        return { confirmed: confirmed, draft: pending };
    }

    // Judging an offered box is a question about what is already on screen,
    // so it is asked while simply viewing a bird — requiring an edit draft
    // first would hide it exactly where it is meant to be answered.
    function bboxVerdictView(item, inEditMode) {
        if (!item || item.objectKind === 'manual') {
            return { visible: false, active: null };
        }
        return { visible: true, active: item.bboxVerdict || null };
    }

    // View mode has no Save button, so a verdict given there posts itself.
    // In edit mode it joins the pending box change instead, keeping "dragging
    // the box is the verdict" a single save.
    function bboxVerdictClickMode(inEditMode) {
        return inEditMode ? 'draft' : 'post';
    }

    // Confirming a species writes immediately, so the second click is a
    // second write that undoes the first rather than a local reset.
    function speciesClickAction(item) {
        if (!item || !item.speciesKey) return 'none';
        const answered = item.provenance === 'human_confirmed'
            || item.provenance === 'manually_identified';
        return answered ? 'retract' : 'confirm';
    }

    // Pressing the active verdict clears it, so a mis-click returns the
    // axis to unanswered instead of locking in an answer nobody meant.
    function nextBboxVerdict(current, pressed) {
        return current === pressed ? null : pressed;
    }

    // One save, one body: every axis the person actually answered, and
    // nothing else. Redrawing a box is itself the verdict that the new box
    // is right; correcting a species says nothing about the geometry, so
    // that axis stays absent rather than being guessed at.
    function detectionAnswerBody(filename, draft, initial) {
        const body = { filename: filename, detection_id: draft.detectionId };
        const moved = !sameBox(draft.bbox, initial.bbox);
        if (moved) body.bbox_correction = draft.bbox;
        if (draft.speciesKey !== initial.speciesKey) {
            body.species_identity = draft.speciesKey ? 'corrected' : 'unknown';
            body.species_key = draft.speciesKey;
        }
        const verdict = draft.bboxVerdict;
        if (verdict === 'suitable' || verdict === 'unsuitable') {
            body.bbox_quality = verdict;
        } else if (moved) {
            body.bbox_quality = 'suitable';
        }
        return body;
    }

    if (typeof module === 'object' && module.exports) {
        module.exports = {
            confirmedSpeciesState: confirmedSpeciesState,
            detectionAnswerBody: detectionAnswerBody,
            nextBboxVerdict: nextBboxVerdict,
            speciesClickAction: speciesClickAction,
            bboxVerdictView: bboxVerdictView,
            bboxVerdictClickMode: bboxVerdictClickMode,
            normalize: normalize,
            isHumanUnknownSpecies: isHumanUnknownSpecies
        };
        return;
    }

    function provenanceLabel(item) {
        if (item.provenance === 'human_unknown') return 'Marked unknown by a person';
        if (item.provenance === 'manually_added') return 'Manually added';
        if (item.provenance === 'manually_identified') return 'Manually identified';
        if (item.provenance === 'human_confirmed') return 'Human confirmed';
        return 'AI proposal';
    }

    function createRequestId() {
        if (window.crypto && typeof window.crypto.randomUUID === 'function') {
            return window.crypto.randomUUID();
        }
        return 'manual-' + Date.now() + '-' + Math.random().toString(16).slice(2);
    }

    function setup(modal) {
        if (modal.dataset.birdEditorReady === 'true') {
            if (typeof modal._wmBirdEditorSync === 'function') {
                requestAnimationFrame(function () {
                    requestAnimationFrame(modal._wmBirdEditorSync);
                });
            }
            return;
        }
        const root = modal.querySelector('[data-bird-editor]');
        const viewer = modal.querySelector('.wm-image-viewer');
        const img = viewer && viewer.querySelector('.wm-image-viewer__img');
        const layer = viewer && viewer.querySelector('[data-bird-editor-layer]');
        if (!root || !viewer || !img || !layer || !window.WmBboxMath) return;
        modal.dataset.birdEditorReady = 'true';

        const currentRaw = readJson(viewer.dataset.currentDetection, {});
        const siblingRaw = readJson(viewer.dataset.siblings, []);
        const byKey = new Map();
        [currentRaw].concat(Array.isArray(siblingRaw) ? siblingRaw : []).forEach(function (item) {
            const normalized = normalize(item || {});
            if (validObject(normalized)) byKey.set(normalized.objectKey, normalized);
        });
        let objects = Array.from(byKey.values());
        const cachedObjects = sharedObjectCache.get(root.dataset.filename);
        if (Array.isArray(cachedObjects)) objects = cachedObjects.map(normalize).filter(validObject);
        let selectedKey = objectKey(currentRaw);
        if (!objects.some(function (item) { return item.objectKey === selectedKey; }) && objects.length) {
            selectedKey = objects[0].objectKey;
        }
        let mode = 'browse';
        let draft = null;
        let initialDraft = null;
        let gesture = null;
        let saving = false;
        let boxesVisible = true;
        let allowHide = false;
        let stale = false;
        let pendingObjects = null;

        const status = root.querySelector('[data-editor-status]');
        const objectSelect = root.querySelector('[data-editor-object-select]');
        const drawHint = root.querySelector('[data-editor-draw-hint]');
        const adjustButton = root.querySelector('[data-editor-action="adjust"]');
        const addButton = root.querySelector('[data-editor-action="add"]');
        const placeButton = root.querySelector('[data-editor-action="place"]');
        const cancelButton = root.querySelector('[data-editor-action="cancel"]');
        const saveButton = root.querySelector('[data-editor-action="save"]');
        const menuButton = root.querySelector('[data-editor-action="menu"]');
        const menu = menuButton && menuButton.nextElementSibling;
        const retractButton = root.querySelector('[data-editor-action="retract"]');
        const titleName = modal.querySelector('[data-editor-title-name]');
        const titleLink = modal.querySelector('[data-editor-title-link]');
        const titleSpecies = modal.querySelector('[data-editor-title-species]');
        const titleConfidence = modal.querySelector('[data-editor-title-confidence]');
        const titleProvenance = modal.querySelector('[data-editor-title-provenance]');

        function current() {
            return draft || objects.find(function (item) { return item.objectKey === selectedKey; }) || objects[0];
        }

        function hasChanges() {
            if (!draft) return false;
            if (draft.objectKind === 'manual' && !draft.manualObjectId) return true;
            return !!initialDraft && (
                !sameBox(draft.bbox, initialDraft.bbox) ||
                draft.speciesKey !== initialDraft.speciesKey ||
                (draft.bboxVerdict || null) !== (initialDraft.bboxVerdict || null)
            );
        }

        function renderBboxVerdict() {
            const view = bboxVerdictView(current(), mode === 'edit');
            root.querySelectorAll('[data-editor-action="bbox-verdict"]').forEach(
                function (button) {
                    const value = button.dataset.bboxVerdict;
                    const on = view.visible && view.active === value;
                    button.setAttribute('aria-pressed', on ? 'true' : 'false');
                    button.hidden = !view.visible || (button.hasAttribute('data-editor-walkthrough-confirm') && mode !== 'browse');
                    if (button.hasAttribute('data-editor-walkthrough-confirm')) button.disabled = saving || stale;
                    const icon = button.querySelector('[data-bbox-verdict-icon]');
                    if (icon) icon.textContent = on ? '\u2611' : '\u2610';
                }
            );
        }

        function percent(value) {
            return Number.isFinite(value) ? Math.round(value * 100) + '%' : '';
        }

        function updateHeader(item) {
            if (!item) return;
            if (titleName) titleName.textContent = item.commonName;
            if (titleSpecies) titleSpecies.textContent = item.speciesKey ? item.speciesKey.replace(/_/g, ' ') : '';
            if (titleLink) {
                titleLink.style.opacity = item.provenance === 'model_proposal' ? '0.65' : '';
                if (titleLink.tagName === 'A') {
                    if (item.speciesKey) {
                        titleLink.href = 'https://en.wikipedia.org/wiki/' + encodeURIComponent(item.speciesKey.replace(/_/g, ' '));
                        titleLink.setAttribute('aria-label', 'Open Wikipedia (new tab): ' + item.commonName);
                        titleLink.title = 'Open Wikipedia (new tab): ' + item.commonName;
                        titleLink.removeAttribute('aria-disabled');
                        titleLink.hidden = false;
                    } else {
                        // Leaving the previous bird's href in place would offer a
                        // confident link to a species nobody asserted.
                        titleLink.removeAttribute('href');
                        titleLink.setAttribute('aria-disabled', 'true');
                        titleLink.setAttribute('aria-label', item.commonName);
                        titleLink.title = 'No species to look up';
                    }
                }
            }
            if (titleConfidence) {
                let confidence = '';
                if (item.provenance === 'model_proposal') {
                    if (Number.isFinite(item.odConfidence)) confidence = 'OD ' + percent(item.odConfidence);
                    if (item.clsClassName && Number.isFinite(item.clsConfidence)) {
                        confidence += (confidence ? ' / ' : '') + 'CLS ' + percent(item.clsConfidence);
                    }
                } else if (item.objectKind === 'detection' && item.clsClassName && Number.isFinite(item.clsConfidence)) {
                    confidence = 'Original AI: ' + item.clsClassName.replace(/_/g, ' ') + ' CLS ' + percent(item.clsConfidence);
                }
                titleConfidence.textContent = confidence;
                titleConfidence.hidden = !confidence;
            }
            if (titleProvenance) titleProvenance.textContent = ' · ' + provenanceLabel(item);
        }

        function populateObjectSelect() {
            if (!objectSelect) return;
            const activeKey = draft ? draft.objectKey : selectedKey;
            objectSelect.replaceChildren();
            objects.forEach(function (item, index) {
                const option = document.createElement('option');
                option.value = item.objectKey;
                option.textContent = String(index + 1) + ' · ' + item.commonName;
                option.selected = item.objectKey === activeKey;
                objectSelect.appendChild(option);
            });
            if (draft && !objects.some(function (item) { return item.objectKey === draft.objectKey; })) {
                const option = document.createElement('option');
                option.value = draft.objectKey;
                option.textContent = String(objects.length + 1) + ' · ' + draft.commonName;
                option.selected = true;
                objectSelect.appendChild(option);
            }
            objectSelect.disabled = mode !== 'browse' || objects.length <= 1;
            objectSelect.title = objects.length <= 1 ? 'Only one bird in this image' : 'Switch the active bird';
        }

        function requireAuth() {
            if (root.dataset.canModerate === 'true') return true;
            if (typeof window.redirectToLogin === 'function') window.redirectToLogin();
            else if (window.wmToast) window.wmToast('Login required', 'info', 2600);
            return false;
        }

        function forceFull() {
            const full = root.querySelector('[data-action="set-smart-zoom"][data-view-mode="full"]');
            if (full && full.getAttribute('aria-pressed') !== 'true') full.click();
        }

        function geometry() {
            const rendered = typeof window.getWmRenderedImageGeometry === 'function'
                ? window.getWmRenderedImageGeometry(img)
                : { contentX: 0, contentY: 0, contentW: img.clientWidth, contentH: img.clientHeight };
            return Object.assign({}, rendered, {
                left: img.offsetLeft || 0,
                top: img.offsetTop || 0
            });
        }

        function syncLayer() {
            const g = geometry();
            if (img.clientWidth <= 0 || img.clientHeight <= 0 || g.contentW <= 0 || g.contentH <= 0) return;
            root.style.width = img.clientWidth + 'px';
            layer.style.left = g.left + 'px';
            layer.style.top = g.top + 'px';
            layer.style.width = img.clientWidth + 'px';
            layer.style.height = img.clientHeight + 'px';
            layer.style.transform = img.style.transform || '';
            layer.style.transformOrigin = img.style.transformOrigin || 'center center';
            positionShapes();
        }
        modal._wmBirdEditorSync = syncLayer;

        function currentZoomScale() {
            const match = String(img.style.transform || '').match(/scale\(([0-9.]+)\)/);
            if (!match) return 1;
            const scale = Number(match[1]);
            return Number.isFinite(scale) && scale > 0 ? scale : 1;
        }

        function positionShapeLabel(node, label, occupied, scale) {
            const viewport = viewer.getBoundingClientRect();
            const box = node.getBoundingClientRect();
            const width = label.offsetWidth;
            const height = label.offsetHeight;
            const gap = 10;
            const inset = 6;
            const candidates = [
                {
                    name: 'above', space: box.top - viewport.top,
                    fits: box.top - viewport.top >= height + gap + inset,
                    left: box.left + (box.width - width) / 2,
                    top: box.top - height - gap
                },
                {
                    name: 'below', space: viewport.bottom - box.bottom,
                    fits: viewport.bottom - box.bottom >= height + gap + inset,
                    left: box.left + (box.width - width) / 2,
                    top: box.bottom + gap
                },
                {
                    name: 'right', space: viewport.right - box.right,
                    fits: viewport.right - box.right >= width + gap + inset,
                    left: box.right + gap,
                    top: box.top + (box.height - height) / 2
                },
                {
                    name: 'left', space: box.left - viewport.left,
                    fits: box.left - viewport.left >= width + gap + inset,
                    left: box.left - width - gap,
                    top: box.top + (box.height - height) / 2
                }
            ];
            const maxLeft = Math.max(viewport.left + inset, viewport.right - width - inset);
            const maxTop = Math.max(viewport.top + inset, viewport.bottom - height - inset);
            const positioned = candidates.map(function (candidate) {
                const left = Math.min(Math.max(candidate.left, viewport.left + inset), maxLeft);
                const top = Math.min(Math.max(candidate.top, viewport.top + inset), maxTop);
                const bounds = { left: left, top: top, right: left + width, bottom: top + height };
                const overlapsBox = bounds.left < box.right + gap
                    && bounds.right > box.left - gap
                    && bounds.top < box.bottom + gap
                    && bounds.bottom > box.top - gap;
                const overlapsLabel = occupied.some(function (other) {
                    return bounds.left < other.right + 4
                        && bounds.right > other.left - 4
                        && bounds.top < other.bottom + 4
                        && bounds.bottom > other.top - 4;
                });
                return Object.assign({}, candidate, {
                    left: left, top: top, bounds: bounds,
                    overlapsBox: overlapsBox, overlapsLabel: overlapsLabel
                });
            });
            const previousPlacement = label.dataset.placement;
            const placement = positioned.find(function (candidate) {
                return candidate.name === previousPlacement
                    && !candidate.overlapsBox && !candidate.overlapsLabel;
            }) || positioned.find(function (candidate) {
                return candidate.fits && !candidate.overlapsBox && !candidate.overlapsLabel;
            }) || positioned.find(function (candidate) {
                return candidate.fits && !candidate.overlapsBox;
            }) || positioned.find(function (candidate) {
                return !candidate.overlapsBox && !candidate.overlapsLabel;
            }) || positioned.reduce(function (best, candidate) {
                return candidate.space > best.space ? candidate : best;
            });
            const inverse = 1 / scale;
            label.dataset.placement = placement.name;
            label.style.left = ((placement.left - box.left) / scale) + 'px';
            label.style.top = ((placement.top - box.top) / scale) + 'px';
            label.style.right = 'auto';
            label.style.bottom = 'auto';
            label.style.transformOrigin = '0 0';
            label.style.transform = 'scale(' + inverse + ')';
            occupied.push(placement.bounds);
        }

        function point(event) {
            const rect = layer.getBoundingClientRect();
            const g = geometry();
            const localWidth = layer.offsetWidth || img.clientWidth;
            const localHeight = layer.offsetHeight || img.clientHeight;
            const localX = (event.clientX - rect.left) * (localWidth / rect.width);
            const localY = (event.clientY - rect.top) * (localHeight / rect.height);
            return {
                x: Math.min(1, Math.max(0, (localX - g.contentX) / g.contentW)),
                y: Math.min(1, Math.max(0, (localY - g.contentY) / g.contentH))
            };
        }

        function positionShapes() {
            const g = geometry();
            const scale = currentZoomScale();
            const nodes = Array.from(layer.querySelectorAll('[data-editor-object-key]'));
            const inverse = 1 / scale;
            layer.style.setProperty('--wm-bird-editor-stroke', inverse + 'px');
            layer.style.setProperty('--wm-bird-editor-soft-stroke', (0.75 * inverse) + 'px');
            layer.style.setProperty('--wm-bird-editor-radius', (3 * inverse) + 'px');
            layer.style.setProperty('--wm-bird-editor-focus', (2 * inverse) + 'px');
            layer.style.setProperty('--wm-bird-editor-glow', (14 * inverse) + 'px');
            layer.style.setProperty('--wm-bird-editor-handle-scale', String(inverse));
            nodes.forEach(function (node) {
                const item = (draft && node.dataset.editorObjectKey === draft.objectKey)
                    ? draft
                    : objects.find(function (candidate) {
                        return candidate.objectKey === node.dataset.editorObjectKey;
                    });
                if (!item) return;
                node.style.left = (g.contentX + item.bbox.x * g.contentW) + 'px';
                node.style.top = (g.contentY + item.bbox.y * g.contentH) + 'px';
                node.style.width = (item.bbox.w * g.contentW) + 'px';
                node.style.height = (item.bbox.h * g.contentH) + 'px';
            });
            const occupied = [];
            nodes.forEach(function (node) {
                const label = node.querySelector('.wm-bird-editor__box-label');
                if (label) positionShapeLabel(node, label, occupied, scale);
            });
        }

        function buildShapes() {
            layer.replaceChildren();
            if (!boxesVisible && mode === 'browse') return;
            const visible = objects.map(function (item) {
                return draft && item.objectKey === draft.objectKey ? draft : item;
            });
            if (draft && !objects.some(function (item) { return item.objectKey === draft.objectKey; })) {
                visible.push(draft);
            }
            visible.forEach(function (item, index) {
                const selected = item.objectKey === (draft ? draft.objectKey : selectedKey);
                const shape = document.createElement('div');
                shape.className = 'wm-bird-editor__box' + (selected ? ' is-selected' : '');
                shape.dataset.editorObjectKey = item.objectKey;
                const body = document.createElement('button');
                body.type = 'button';
                body.className = 'wm-bird-editor__box-body';
                body.dataset.editorHandle = 'move';
                body.setAttribute('aria-label', 'Select ' + item.commonName);
                body.setAttribute('aria-pressed', selected ? 'true' : 'false');
                body.title = 'Select ' + item.commonName;
                shape.appendChild(body);

                const label = document.createElement('div');
                label.className = 'wm-bird-editor__box-label';
                const reviewed = item.humanReviewState !== 'unreviewed';
                const clickAction = speciesClickAction(item);
                if (item.objectKind === 'detection' && clickAction !== 'none') {
                    const undo = clickAction === 'retract';
                    const confirm = document.createElement('button');
                    confirm.type = 'button';
                    confirm.className = 'wm-bird-editor__box-label-name';
                    confirm.dataset.editorConfirm = item.objectKey;
                    confirm.textContent = (undo ? '✓ ' : '')
                        + String(index + 1) + ' · ' + item.commonName;
                    confirm.setAttribute(
                        'aria-label',
                        (undo ? 'Take back species: ' : 'Confirm species: ') + item.commonName
                    );
                    confirm.title = undo
                        ? 'Click again to take this confirmation back'
                        : 'Confirm this species';
                    label.appendChild(confirm);
                } else {
                    const name = document.createElement('span');
                    name.className = 'wm-bird-editor__box-label-name';
                    name.textContent = (reviewed ? '✓ ' : '') + String(index + 1) + ' · ' + item.commonName;
                    name.title = item.objectKind === 'manual' ? 'Manually added bird' : 'Species already answered';
                    label.appendChild(name);
                }
                if (root.dataset.canModerate === 'true') {
                    const picker = document.createElement('button');
                    picker.type = 'button';
                    picker.className = 'wm-bird-editor__box-label-picker';
                    picker.dataset.editorSpecies = item.objectKey;
                    picker.textContent = '▾';
                    picker.setAttribute('aria-label', 'Change species for ' + item.commonName);
                    picker.title = 'Choose a different species';
                    label.appendChild(picker);
                }
                shape.appendChild(label);

                if (selected && draft && mode === 'edit') {
                    ['n', 'e', 's', 'w', 'nw', 'ne', 'se', 'sw'].forEach(function (handle) {
                        const control = document.createElement('button');
                        control.type = 'button';
                        control.className = 'wm-bird-editor__handle wm-bird-editor__handle--' + handle;
                        control.dataset.editorHandle = handle;
                        control.setAttribute('aria-label', 'Resize ' + handle + ' edge');
                        control.title = 'Drag to resize · arrow keys to adjust';
                        shape.appendChild(control);
                    });
                }
                layer.appendChild(shape);
            });
            positionShapes();
        }

        function setActiveViewer(item) {
            if (!item) return;
            viewer.dataset.bboxX = String(item.bbox.x);
            viewer.dataset.bboxY = String(item.bbox.y);
            viewer.dataset.bboxW = String(item.bbox.w);
            viewer.dataset.bboxH = String(item.bbox.h);
            img.dataset.detectionId = item.detectionId == null ? '' : String(item.detectionId);
            const bboxToggle = root.querySelector('.bbox-toggle');
            if (bboxToggle) {
                bboxToggle.dataset.detectionId = item.detectionId == null ? '' : String(item.detectionId);
                bboxToggle.dataset.currentBbox = JSON.stringify({
                    x: item.bbox.x, y: item.bbox.y, w: item.bbox.w, h: item.bbox.h,
                    id: item.detectionId, name: item.commonName
                });
            }
            root.querySelectorAll('[data-action="favorite"], [data-action="move-trash"], [data-action="correction-details"]').forEach(function (control) {
                if (item.detectionId == null) {
                    control.dataset.editorUnavailable = 'true';
                    control.setAttribute('aria-disabled', 'true');
                    delete control.dataset.detectionId;
                } else {
                    delete control.dataset.editorUnavailable;
                    control.dataset.detectionId = String(item.detectionId);
                    if (root.dataset.canModerate === 'true') control.removeAttribute('aria-disabled');
                }
            });
            const favorite = root.querySelector('[data-action="favorite"]');
            if (favorite) {
                const icon = favorite.querySelector('[data-favorite-icon]');
                const label = favorite.querySelector('[data-favorite-label]');
                if (item.detectionId != null) {
                    if (icon) icon.textContent = item.isFavorite ? '⭐' : '☆';
                    if (label) label.textContent = item.isFavorite ? 'Unfavorite' : 'Favorite';
                    favorite.setAttribute('aria-pressed', item.isFavorite ? 'true' : 'false');
                    favorite.setAttribute('aria-label', item.isFavorite ? 'Remove from favorites' : 'Add to favorites');
                    favorite.title = item.isFavorite ? 'Unfavorite this detection' : 'Favorite this detection';
                } else {
                    // Favorites don't apply to manually added birds; the stale
                    // label from a previously selected detection must not imply
                    // a favorite state this control can no longer act on.
                    if (icon) icon.textContent = '☆';
                    if (label) label.textContent = 'Favorite';
                    favorite.removeAttribute('aria-pressed');
                    favorite.setAttribute('aria-label', 'Favorite (unavailable for a manually added bird)');
                    favorite.title = 'Favoriting is unavailable for a manually added bird';
                }
            }
            if (retractButton) retractButton.hidden = item.objectKind !== 'manual' || mode !== 'browse';
            updateHeader(item);
        }

        function render(message) {
            const item = current();
            const editing = mode !== 'browse';
            root.classList.toggle('is-editing', editing);
            root.classList.toggle('is-adding', mode === 'add');
            layer.classList.toggle('is-editing', editing);
            layer.classList.toggle('is-adding', mode === 'add');
            drawHint.hidden = mode !== 'add';
            adjustButton.hidden = editing;
            addButton.hidden = editing;
            placeButton.hidden = mode !== 'add';
            cancelButton.hidden = !editing;
            saveButton.hidden = !editing;
            menuButton.closest('.wm-bird-editor__menu').hidden = editing;
            saveButton.disabled = saving || stale || !hasChanges() || !draft || draft.bbox.w < MIN_BOX || draft.bbox.h < MIN_BOX;
            renderBboxVerdict();
            cancelButton.disabled = saving;
            adjustButton.disabled = saving;
            addButton.disabled = saving;
            menuButton.disabled = saving;
            if (item) {
                setActiveViewer(item);
            }
            if (message) status.textContent = message;
            else if (mode === 'add') status.textContent = 'Draw around the missing bird · or use Place box';
            else if (mode === 'edit') status.textContent = 'Drag the box or its edges · arrow keys adjust · Esc cancels';
            else if (item) status.textContent = item.commonName + ' · ' + provenanceLabel(item);
            populateObjectSelect();
            buildShapes();
            syncLayer();
        }

        function beginEdit() {
            if (saving || stale) return;
            if (!requireAuth()) return;
            const item = current();
            if (!item) return;
            initialDraft = clone(item);
            draft = clone(item);
            mode = 'edit';
            closeMenu();
            render();
            requestAnimationFrame(function () {
                const body = layer.querySelector('.is-selected .wm-bird-editor__box-body');
                if (body) body.focus({ preventScroll: true });
            });
        }

        function beginAdd() {
            if (saving || stale) return;
            if (!requireAuth()) return;
            forceFull();
            draft = null;
            initialDraft = null;
            mode = 'add';
            boxesVisible = true;
            closeMenu();
            render();
            placeButton.focus({ preventScroll: true });
        }

        function cancelEdit(message) {
            draft = null;
            initialDraft = null;
            gesture = null;
            saving = false;
            mode = 'browse';
            if (pendingObjects) {
                objects = pendingObjects.map(normalize).filter(validObject);
                viewer.dataset.siblings = JSON.stringify(pendingObjects);
                pendingObjects = null;
                if (!objects.some(function (item) { return item.objectKey === selectedKey; })) {
                    selectedKey = objects.length ? objects[0].objectKey : '';
                }
            }
            stale = false;
            render(message || 'Changes discarded');
            adjustButton.focus({ preventScroll: true });
        }

        function selectObject(key) {
            if (saving) return;
            if (mode !== 'browse' || !objects.some(function (item) { return item.objectKey === key; })) return;
            selectedKey = key;
            render();
            const focus = root.querySelector('[data-view-mode="zoom"]');
            if (focus && focus.getAttribute('aria-pressed') === 'true') {
                focus.click();
            }
        }

        function serializedObjects() {
            return objects.map(function (object) {
                return {
                    detection_id: object.detectionId,
                    manual_object_id: object.manualObjectId,
                    object_key: object.objectKey,
                    object_kind: object.objectKind,
                    revision: object.revision,
                    bbox_x: object.bbox.x, bbox_y: object.bbox.y,
                    bbox_w: object.bbox.w, bbox_h: object.bbox.h,
                    species_key: object.speciesKey,
                    common_name: object.commonName,
                    provenance: object.provenance,
                    human_review_state: object.humanReviewState,
                    od_class_name: object.odClassName,
                    od_confidence: object.odConfidence,
                    cls_class_name: object.clsClassName,
                    cls_confidence: object.clsConfidence,
                    is_favorite: object.isFavorite
                };
            });
        }

        function broadcastObjects() {
            const snapshot = serializedObjects();
            sharedObjectCache.set(root.dataset.filename, snapshot);
            document.dispatchEvent(new CustomEvent('wmb:bird-objects-updated', {
                detail: {
                    filename: root.dataset.filename,
                    source: modal,
                    objects: snapshot
                }
            }));
        }

        function openMenu() {
            if (!menu) return;
            const open = menu.classList.toggle('wm-toolbox__dropdown--open');
            menuButton.setAttribute('aria-expanded', open ? 'true' : 'false');
            if (open) {
                const first = menu.querySelector('[role="menuitem"]');
                if (first) first.focus({ preventScroll: true });
            }
        }

        function closeMenu() {
            if (!menu) return;
            menu.classList.remove('wm-toolbox__dropdown--open');
            menuButton.setAttribute('aria-expanded', 'false');
        }

        async function chooseSpecies(key) {
            if (saving || stale) return;
            if (!requireAuth() || typeof window.WmSpeciesPicker === 'undefined') return;
            const item = (key && objects.find(function (candidate) { return candidate.objectKey === key; })) || current();
            if (!item) return;
            if (mode !== 'browse' && (!draft || item.objectKey !== draft.objectKey)) {
                render('Save or Cancel before editing a different bird');
                return;
            }
            if (mode === 'browse') selectedKey = item.objectKey;
            const choice = await window.WmSpeciesPicker.pickSpecies({
                currentSpecies: item.speciesKey || '',
                detectionId: item.detectionId,
                mountEl: modal,
                title: 'Select species',
                allowUnknown: true,
                birdsOnly: item.objectKind === 'manual'
            });
            if (!choice || saving || stale) return;
            if (mode === 'browse') {
                initialDraft = clone(item);
                draft = clone(item);
                mode = 'edit';
            }
            draft.speciesKey = choice.scientific || null;
            draft.commonName = choice.common || 'Bird · species unknown';
            render('Species changed · Save to keep this edit');
        }

        async function confirmSpecies(key, button) {
            if (saving || stale) return;
            const saved = objects.find(function (candidate) { return candidate.objectKey === key; });
            if (!saved || (draft && draft.objectKey !== key)) return;
            const displayed = draft && draft.objectKey === key ? clone(draft) : null;
            const item = displayed || clone(saved);
            if (item.objectKind !== 'detection' || !item.speciesKey || !requireAuth()) return;
            saving = true;
            render('Confirming species…');
            let message;
            try {
                const response = await fetch('/api/labels/answer', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        filename: root.dataset.filename,
                        detection_id: item.detectionId,
                        object_bird_presence: 'present',
                        species_identity: 'confirmed',
                        species_key: item.speciesKey
                    })
                });
                const payload = await response.json().catch(function () { return {}; });
                if (!response.ok || payload.status !== 'success') {
                    throw new Error(payload.message || 'Unable to confirm species');
                }
                const result = confirmedSpeciesState(saved, displayed);
                updateObjects(result.confirmed);
                if (displayed) {
                    draft = result.draft;
                    initialDraft = draft ? clone(result.confirmed) : null;
                    mode = draft ? 'edit' : 'browse';
                }
                message = draft ? 'Species confirmed · box changes still need Save'
                    : result.confirmed.commonName + ' confirmed · other birds unchanged';
                broadcastObjects();
                if (window.wmToast) window.wmToast('Species confirmed', 'success', 2200);
            } catch (error) {
                message = error.message || 'Unable to confirm species';
                if (window.wmToast) window.wmToast(message, 'error', 4200);
            } finally {
                saving = false;
                render(message);
            }
        }

        async function retractSpecies(key) {
            if (saving || stale) return;
            const saved = objects.find(function (candidate) { return candidate.objectKey === key; });
            if (!saved || (draft && draft.objectKey !== key)) return;
            if (saved.objectKind !== 'detection' || !requireAuth()) return;
            saving = true;
            render('Taking the species back…');
            let message;
            try {
                const response = await fetch('/api/labels/species/retract', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        filename: root.dataset.filename,
                        detection_id: saved.detectionId
                    })
                });
                const payload = await response.json().catch(function () { return {}; });
                if (!response.ok || payload.status !== 'success') {
                    throw new Error(payload.message || 'Unable to take the species back');
                }
                const reverted = clone(saved);
                reverted.provenance = 'model_proposal';
                reverted.humanReviewState = 'unreviewed';
                updateObjects(reverted);
                message = reverted.commonName + ' · confirmation withdrawn';
                broadcastObjects();
                if (window.wmToast) window.wmToast('Confirmation withdrawn', 'success', 2200);
            } catch (error) {
                message = error.message || 'Unable to take the species back';
                if (window.wmToast) window.wmToast(message, 'error', 4200);
            } finally {
                saving = false;
                render(message);
            }
        }

        async function postBboxVerdict(item, verdict) {
            if (saving || stale || !requireAuth()) return;
            saving = true;
            render(verdict ? 'Saving box verdict…' : 'Clearing box verdict…');
            let message;
            try {
                const body = {
                    filename: root.dataset.filename,
                    detection_id: item.detectionId
                };
                let url = '/api/labels/answer';
                if (verdict) body.bbox_quality = verdict;
                else url = '/api/labels/bbox-quality/retract';
                const response = await fetch(url, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body)
                });
                const payload = await response.json().catch(function () { return {}; });
                if (!response.ok || payload.status !== 'success') {
                    throw new Error(payload.message || 'Unable to save the box verdict');
                }
                const next = clone(item);
                next.bboxVerdict = verdict;
                updateObjects(next);
                message = verdict === 'suitable' ? 'Box marked as fitting'
                    : verdict === 'unsuitable' ? 'Box marked as wrong'
                    : 'Box verdict cleared';
                broadcastObjects();
                if (window.wmToast) window.wmToast(message, 'success', 2200);
            } catch (error) {
                message = error.message || 'Unable to save the box verdict';
                if (window.wmToast) window.wmToast(message, 'error', 4200);
            } finally {
                saving = false;
                render(message);
            }
        }

        function updateObjects(item) {
            const index = objects.findIndex(function (candidate) {
                return candidate.objectKey === item.objectKey;
            });
            if (index >= 0) objects[index] = item;
            else objects.push(item);
            selectedKey = item.objectKey;
            viewer.dataset.siblings = JSON.stringify(serializedObjects());
        }

        function objectFromResponse(payload, commonName) {
            const object = payload.object;
            return normalize({
                manual_object_id: object.manual_object_id,
                object_key: 'manual:' + object.manual_object_id,
                object_kind: 'manual',
                revision: object.revision,
                bbox_x: object.bbox.x, bbox_y: object.bbox.y,
                bbox_w: object.bbox.w, bbox_h: object.bbox.h,
                species_key: object.species_key,
                common_name: commonName || (object.species_key ? object.species_key.replace(/_/g, ' ') : 'Bird · species unknown'),
                provenance: 'manually_added'
            });
        }

        async function save() {
            if (saving || !draft || stale || !hasChanges()) return;
            const savingDraft = clone(draft);
            const savingInitial = clone(initialDraft);
            saving = true;
            render('Saving…');
            try {
                let response;
                if (savingDraft.objectKind === 'manual' && savingDraft.manualObjectId) {
                    const body = {
                        filename: root.dataset.filename,
                        expected_revision: savingInitial.revision
                    };
                    if (!sameBox(savingDraft.bbox, savingInitial.bbox)) body.bbox = savingDraft.bbox;
                    if (savingDraft.speciesKey !== savingInitial.speciesKey) body.species_key = savingDraft.speciesKey;
                    response = await fetch('/api/manual-objects/' + encodeURIComponent(savingDraft.manualObjectId), {
                        method: 'PATCH',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(body)
                    });
                } else if (savingDraft.objectKind === 'manual') {
                    response = await fetch('/api/manual-objects', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            filename: root.dataset.filename,
                            bbox: savingDraft.bbox,
                            species_key: savingDraft.speciesKey,
                            request_id: savingDraft.requestId
                        })
                    });
                } else {
                    const body = detectionAnswerBody(
                        root.dataset.filename, savingDraft, savingInitial
                    );
                    response = await fetch('/api/labels/answer', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(body)
                    });
                }
                const payload = await response.json().catch(function () { return {}; });
                if (!response.ok || payload.status !== 'success') {
                    throw new Error(payload.message || 'Unable to save bird');
                }

                let saved;
                if (savingDraft.objectKind === 'manual') {
                    saved = objectFromResponse(payload, savingDraft.commonName);
                } else {
                    saved = clone(savingDraft);
                    // The verdict was an answer about this save, not a property
                    // of the bird; keeping it would re-send it on the next one.
                    delete saved.bboxVerdict;
                    if (savingDraft.speciesKey !== savingInitial.speciesKey) {
                        saved.provenance = 'manually_identified';
                        saved.humanReviewState = savingDraft.speciesKey ? 'corrected' : 'reviewed_unknown';
                    }
                }
                updateObjects(saved);
                draft = null;
                initialDraft = null;
                mode = 'browse';
                saving = false;
                stale = false;
                pendingObjects = null;
                render(saved.commonName + ' saved · ' + provenanceLabel(saved));
                broadcastObjects();
                if (window.wmToast) window.wmToast('Bird saved', 'success', 2200);
            } catch (error) {
                saving = false;
                render(error.message || 'Unable to save bird');
                if (window.wmToast) window.wmToast(error.message || 'Unable to save bird', 'error', 4200);
            }
        }

        async function retractManualObject() {
            const item = current();
            if (!item || item.objectKind !== 'manual' || saving || !requireAuth()) return;
            if (!window.confirm('Retract this manually added bird? The audit history will be kept.')) return;
            saving = true;
            render('Retracting…');
            try {
                const response = await fetch(
                    '/api/manual-objects/' + encodeURIComponent(item.manualObjectId) + '/retract',
                    {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            filename: root.dataset.filename,
                            expected_revision: item.revision
                        })
                    }
                );
                const payload = await response.json().catch(function () { return {}; });
                if (!response.ok || payload.status !== 'success') {
                    throw new Error(payload.message || 'Unable to retract bird');
                }
                objects = objects.filter(function (candidate) { return candidate.objectKey !== item.objectKey; });
                selectedKey = objects.length ? objects[0].objectKey : '';
                viewer.dataset.siblings = JSON.stringify(serializedObjects());
                saving = false;
                closeMenu();
                render('Manually added bird retracted · audit history kept');
                broadcastObjects();
                if (window.wmToast) window.wmToast('Added bird retracted', 'success', 2400);
            } catch (error) {
                saving = false;
                render(error.message || 'Unable to retract bird');
                if (window.wmToast) window.wmToast(error.message || 'Unable to retract bird', 'error', 4200);
            }
        }

        root.addEventListener('click', function (event) {
            if (saving) return;
            const control = event.target.closest('[data-editor-action]');
            if (!control) return;
            const action = control.dataset.editorAction;
            if (action === 'adjust') beginEdit();
            else if (action === 'add') beginAdd();
            else if (action === 'place') {
                draft = {
                    objectKey: 'draft:' + createRequestId(), objectKind: 'manual',
                    manualObjectId: null, detectionId: null, revision: null,
                    requestId: createRequestId(), speciesKey: null,
                    commonName: 'Bird · species unknown', provenance: 'manually_added',
                    bbox: { x: 0.435, y: 0.4, w: 0.13, h: 0.2 }
                };
                initialDraft = clone(draft);
                mode = 'edit';
                render('Move with arrow keys · focus an edge to resize');
                const body = layer.querySelector('.is-selected .wm-bird-editor__box-body');
                if (body) body.focus({ preventScroll: true });
            } else if (action === 'cancel') cancelEdit();
            else if (action === 'save') save();
            else if (action === 'menu') openMenu();
            else if (action === 'retract') retractManualObject();
            else if (action === 'bbox-verdict') {
                const item = current();
                if (!bboxVerdictView(item, mode === 'edit').visible) return;
                const next = nextBboxVerdict(
                    item.bboxVerdict || null, control.dataset.bboxVerdict
                );
                if (bboxVerdictClickMode(mode === 'edit') === 'draft') {
                    draft.bboxVerdict = next;
                    render();
                } else {
                    postBboxVerdict(item, next);
                }
            }
        });

        if (objectSelect) {
            objectSelect.addEventListener('change', function () {
                if (mode !== 'browse' || saving) {
                    populateObjectSelect();
                    render('Save or Cancel before switching birds');
                    return;
                }
                selectObject(objectSelect.value);
            });
        }

        layer.addEventListener('click', function (event) {
            const confirm = event.target.closest('[data-editor-confirm]');
            if (confirm) {
                event.preventDefault();
                event.stopPropagation();
                const key = confirm.dataset.editorConfirm;
                const target = objects.find(function (candidate) {
                    return candidate.objectKey === key;
                });
                const shown = draft && draft.objectKey === key ? draft : target;
                const action = speciesClickAction(shown);
                if (action === 'confirm') confirmSpecies(key, confirm);
                else if (action === 'retract') retractSpecies(key);
                return;
            }
            const species = event.target.closest('[data-editor-species]');
            if (species) {
                event.preventDefault();
                event.stopPropagation();
                chooseSpecies(species.dataset.editorSpecies);
                return;
            }
            const select = event.target.closest('[data-editor-object-key]');
            if (select && mode === 'browse') {
                selectObject(select.dataset.editorObjectKey);
            }
        });

        layer.addEventListener('pointerdown', function (event) {
            if (event.button !== 0 || saving || stale) return;
            const p = point(event);
            if (mode === 'add') {
                event.preventDefault();
                const requestId = createRequestId();
                draft = {
                    objectKey: 'draft:' + requestId, objectKind: 'manual',
                    manualObjectId: null, detectionId: null, revision: null,
                    requestId: requestId, speciesKey: null,
                    commonName: 'Bird · species unknown', provenance: 'manually_added',
                    bbox: { x: p.x, y: p.y, w: 0, h: 0 }
                };
                gesture = { type: 'draw', start: p, pointerId: event.pointerId };
                buildShapes();
            } else if (mode === 'edit' && draft) {
                const handle = event.target.closest('[data-editor-handle]');
                const shape = event.target.closest('[data-editor-object-key]');
                if (!handle || !shape || shape.dataset.editorObjectKey !== draft.objectKey) return;
                event.preventDefault();
                gesture = {
                    type: handle.dataset.editorHandle,
                    start: p,
                    startBox: clone(draft.bbox),
                    pointerId: event.pointerId
                };
            } else return;
            layer.setPointerCapture(event.pointerId);
            layer.classList.add('is-dragging');
        });

        layer.addEventListener('pointermove', function (event) {
            if (!gesture || gesture.pointerId !== event.pointerId || !draft) return;
            event.preventDefault();
            const p = point(event);
            const dx = p.x - gesture.start.x;
            const dy = p.y - gesture.start.y;
            if (gesture.type === 'draw') {
                const left = Math.min(gesture.start.x, p.x);
                const top = Math.min(gesture.start.y, p.y);
                draft.bbox = { x: left, y: top, w: Math.abs(p.x - gesture.start.x), h: Math.abs(p.y - gesture.start.y) };
            } else {
                draft.bbox = window.WmBboxMath.resizeBox(
                    gesture.startBox, gesture.type, dx, dy, MIN_BOX
                );
            }
            positionShapes();
        });

        function finishGesture(event) {
            if (!gesture || gesture.pointerId !== event.pointerId) return;
            const wasDraw = gesture.type === 'draw';
            gesture = null;
            layer.classList.remove('is-dragging');
            if (layer.hasPointerCapture(event.pointerId)) layer.releasePointerCapture(event.pointerId);
            if (wasDraw) {
                if (draft.bbox.w < MIN_BOX || draft.bbox.h < MIN_BOX) {
                    draft = null;
                    render('Draw a slightly larger box around the bird');
                } else {
                    initialDraft = clone(draft);
                    mode = 'edit';
                    render('Choose a species, or leave it explicitly unknown');
                    const picker = layer.querySelector('.is-selected [data-editor-species]');
                    if (picker) picker.focus({ preventScroll: true });
                }
            } else render('Box adjusted · Save to keep this edit');
        }
        layer.addEventListener('pointerup', finishGesture);
        layer.addEventListener('pointercancel', function (event) {
            if (!gesture || gesture.pointerId !== event.pointerId) return;
            if (gesture.type === 'draw') draft = null;
            else draft.bbox = gesture.startBox;
            gesture = null;
            layer.classList.remove('is-dragging');
            render();
        });

        layer.addEventListener('keydown', function (event) {
            if (saving || stale || !draft || !event.target.closest('[data-editor-object-key]')) return;
            if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) return;
            event.preventDefault();
            event.stopPropagation();
            const step = event.shiftKey ? 0.015 : 0.003;
            const dx = event.key === 'ArrowLeft' ? -step : event.key === 'ArrowRight' ? step : 0;
            const dy = event.key === 'ArrowUp' ? -step : event.key === 'ArrowDown' ? step : 0;
            const handle = event.target.dataset.editorHandle || 'move';
            draft.bbox = window.WmBboxMath.resizeBox(draft.bbox, handle, dx, dy, MIN_BOX);
            positionShapes();
            saveButton.disabled = saving || stale || !hasChanges();
            status.textContent = 'Box adjusted · Save to keep this edit';
        });

        modal.addEventListener('keydown', function (event) {
            if (event.key === 'Escape' && mode !== 'browse') {
                if (modal.querySelector('.wm-species-picker-overlay')) return;
                event.preventDefault();
                event.stopImmediatePropagation();
                if (saving) {
                    render('Save in progress · please wait');
                    return;
                }
                cancelEdit();
            }
        }, true);

        modal.addEventListener('hide.bs.modal', function (event) {
            if (saving) {
                event.preventDefault();
                render('Save in progress · please wait');
                return;
            }
            if (mode === 'browse' || allowHide) return;
            event.preventDefault();
            if (window.confirm('Discard unsaved bird changes?')) {
                allowHide = true;
                modal.dataset.birdEditorDiscardAccepted = 'true';
                cancelEdit();
                const instance = window.bootstrap && window.bootstrap.Modal.getInstance(modal);
                if (instance) instance.hide();
            }
        });
        modal.addEventListener('hidden.bs.modal', function () {
            allowHide = false;
            delete modal.dataset.birdEditorDiscardAccepted;
        });

        root.addEventListener('click', function (event) {
            const disabled = event.target.closest('[data-editor-unavailable="true"]');
            if (disabled) {
                event.preventDefault();
                event.stopImmediatePropagation();
                if (window.wmToast) window.wmToast('This action is unavailable for a manually added bird.', 'info', 3200);
            }
        }, true);

        function toggleBoxesVisible() {
            boxesVisible = !boxesVisible;
            buildShapes();
            closeMenu();
            menuButton.focus({ preventScroll: true });
        }
        // Exposed for tile_actions.js's shared dispatcher, which owns click
        // routing for data-action controls (see toggle-bbox-overlay there) and
        // must not also reach the legacy canvas overlay for this toolbar.
        modal._wmBirdEditorToggleBoxes = toggleBoxesVisible;

        document.addEventListener('pointerdown', function (event) {
            if (menu && menu.classList.contains('wm-toolbox__dropdown--open') && !event.target.closest('.wm-bird-editor__menu')) {
                closeMenu();
            }
        });
        window.addEventListener('resize', syncLayer);
        new MutationObserver(syncLayer).observe(img, { attributes: true, attributeFilter: ['style'] });
        function syncFinishedTransform(event) {
            if (event.target === img && event.propertyName === 'transform') syncLayer();
        }
        img.addEventListener('transitionend', syncFinishedTransform);
        img.addEventListener('load', syncLayer);
        /* Geometry can also change without touching the image's own style
           attribute or the window size - maximize toggles a class on an
           ancestor, and the modal's flex column reflows the image when the
           editor rail claims its height. A ResizeObserver catches every such
           case; the listeners above stay for browsers that fire them first. */
        if (typeof ResizeObserver === 'function') {
            new ResizeObserver(function () { syncLayer(); }).observe(img);
        }
        document.addEventListener('wmb:bird-objects-updated', function (event) {
            const detail = event.detail || {};
            if (detail.source === modal || detail.filename !== root.dataset.filename || !Array.isArray(detail.objects)) return;
            if (draft || saving) {
                pendingObjects = detail.objects;
                stale = true;
                render('This image changed in another editor · Cancel and reopen before saving');
                return;
            }
            objects = detail.objects.map(normalize).filter(validObject);
            if (!objects.some(function (item) { return item.objectKey === selectedKey; })) {
                selectedKey = objects.length ? objects[0].objectKey : '';
            }
            viewer.dataset.siblings = JSON.stringify(detail.objects);
            render('Updated from another view');
        });
        document.addEventListener('wmb:favorite-updated', function (event) {
            const detail = event.detail || {};
            const item = objects.find(function (candidate) {
                return candidate.detectionId === Number(detail.detectionId);
            });
            if (!item) return;
            item.isFavorite = Boolean(detail.isFavorite);
            if (item.objectKey === selectedKey) render();
        });
        render();
        requestAnimationFrame(function () {
            requestAnimationFrame(syncLayer);
        });
    }

    document.addEventListener('shown.bs.modal', function (event) {
        if (event.target.classList.contains('gallery-modal')) setup(event.target);
    });
    document.querySelectorAll('.gallery-modal').forEach(setup);
    // Non-modal container: the box walkthrough embeds the same editor in a
    // full page rather than a Bootstrap modal, so it is set up on load.
    document.querySelectorAll('.wm-box-walkthrough').forEach(setup);
})();
