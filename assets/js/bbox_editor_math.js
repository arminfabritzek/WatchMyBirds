(function (root, factory) {
    'use strict';
    var api = factory();
    if (typeof module === 'object' && module.exports) module.exports = api;
    if (root) root.WmBboxMath = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
    'use strict';

    function clamp(value, minimum, maximum) {
        return Math.min(Math.max(value, minimum), maximum);
    }

    function resizeBox(box, handle, deltaX, deltaY, minimumSize) {
        var minSize = clamp(Number(minimumSize) || 0.01, 0.001, 1);
        var left = Number(box.x);
        var top = Number(box.y);
        var right = left + Number(box.w);
        var bottom = top + Number(box.h);

        if (handle === 'move') {
            var width = right - left;
            var height = bottom - top;
            return {
                x: clamp(left + deltaX, 0, 1 - width),
                y: clamp(top + deltaY, 0, 1 - height),
                w: width,
                h: height
            };
        }

        if (handle.indexOf('w') !== -1) {
            left = clamp(left + deltaX, 0, right - minSize);
        } else if (handle.indexOf('e') !== -1) {
            right = clamp(right + deltaX, left + minSize, 1);
        }

        if (handle.indexOf('n') !== -1) {
            top = clamp(top + deltaY, 0, bottom - minSize);
        } else if (handle.indexOf('s') !== -1) {
            bottom = clamp(bottom + deltaY, top + minSize, 1);
        }

        return {
            x: left,
            y: top,
            w: right - left,
            h: bottom - top
        };
    }

    return { resizeBox: resizeBox };
});
