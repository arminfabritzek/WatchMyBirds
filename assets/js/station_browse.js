(function () {
    'use strict';
    let changed = false;
    document.addEventListener('wmb:bird-objects-updated', function () {
        changed = true;
        document.querySelectorAll('[data-current-report]').forEach(function (node) { node.hidden = true; });
    });
    document.addEventListener('hidden.bs.modal', function () {
        if (changed) window.location.reload();
    });
}());
