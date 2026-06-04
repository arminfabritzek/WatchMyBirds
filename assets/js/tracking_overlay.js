// Live auto-PTZ tracking overlay — draws the controller's current
// target box + state label + a centre crosshair over the stream.
//
// Draws ONLY. The <canvas id="trackingOverlay"> is positioned and sized
// to the video letterbox rectangle by the inline PTZ script in
// stream.html (syncTrackingOverlay), so this module works purely in
// canvas-local coordinates: a normalised box (x,y,w,h) maps to canvas
// pixels by a plain multiply — no letterbox math here.
//
// Data source: GET /api/v1/ptz/auto/status, which wraps the controller
// status under the `auto_ptz` key (read data.auto_ptz, NOT the root).
// The endpoint is login_required; this script is only included for
// moderators, so a 401 should not normally happen — but we degrade
// silently (draw nothing) if it does.
(function () {
    "use strict";

    // Controlled by the "Live Tracking Overlay" setting (Settings →
    // Detection & AI), delivered as overlay_enabled in the status poll
    // below. The poll always runs (moderator-only page), but the overlay
    // only shows + draws while the setting is on, so a live toggle change
    // takes effect within one poll without a page reload.
    var canvas = document.getElementById("trackingOverlay");
    if (!canvas || !canvas.getContext) return;
    var ctx = canvas.getContext("2d");

    // State → colour + label. States with no entry draw no box (idle,
    // overview, returning).
    var COLOURS = {
        acquiring: "#f5a623",   // amber
        tracking: "#2ecc71",    // green
        settling: "#b8860b",    // dimmed amber
        lost_grace: "#e67e22",  // orange
    };
    var LABEL = {
        acquiring: "ACQUIRING",
        tracking: "TRACKING",
        settling: "SETTLING",
        lost_grace: "LOST",
    };

    var TWEEN_MS = 200;     // Option B: glide the box to its new position
    var POLL_MS = 200;      // ~5 Hz status poll

    var enabled = false;    // mirrors the overlay_enabled setting
    var cur = null;         // currently drawn box {x,y,w,h} (canvas-normalised)
    var tgt = null;         // latest measured box, or null
    var tweenStart = 0;
    var state = "idle";
    var countdown = null;

    function setEnabled(on) {
        if (on === enabled) return;
        enabled = on;
        canvas.hidden = !on;
        if (!on) { tgt = null; cur = null; }
    }

    function poll() {
        fetch("/api/v1/ptz/auto/status", { credentials: "same-origin" })
            .then(function (r) {
                if (!r.ok) { state = "idle"; tgt = null; cur = null; return null; }
                return r.json();
            })
            .then(function (data) {
                if (!data) return;
                setEnabled(data.overlay_enabled === true);
                if (!enabled) return;
                var a = data.auto_ptz || {};
                state = a.state || "idle";
                countdown = (typeof a.seconds_until_return === "number")
                    ? a.seconds_until_return : null;
                var b = a.last_bbox;
                if (b && b.length === 4 && COLOURS[state]) {
                    tgt = { x: b[0], y: b[1], w: b[2], h: b[3] };
                    if (!cur) cur = { x: tgt.x, y: tgt.y, w: tgt.w, h: tgt.h };
                    tweenStart = (typeof performance !== "undefined")
                        ? performance.now() : 0;
                } else {
                    tgt = null;
                    cur = null;
                }
            })
            .catch(function () { state = "idle"; tgt = null; cur = null; });
    }

    function lerp(a, b, t) { return a + (b - a) * t; }

    function draw(now) {
        var W = canvas.width, H = canvas.height;
        ctx.clearRect(0, 0, W, H);

        // When the setting is off, draw nothing (the crosshair/box vanish
        // immediately on toggle-off) but keep the rAF loop alive so the
        // overlay reappears the instant the setting is turned back on.
        if (!enabled) {
            window.requestAnimationFrame(draw);
            return;
        }

        // Centre crosshair (default on) — the visual proof that the
        // camera is centring the bird.
        ctx.strokeStyle = "rgba(255,255,255,0.5)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(W / 2 - 10, H / 2);
        ctx.lineTo(W / 2 + 10, H / 2);
        ctx.moveTo(W / 2, H / 2 - 10);
        ctx.lineTo(W / 2, H / 2 + 10);
        ctx.stroke();

        if (tgt && cur && COLOURS[state]) {
            var t = (TWEEN_MS > 0) ? Math.min(1, (now - tweenStart) / TWEEN_MS) : 1;
            var x = lerp(cur.x, tgt.x, t);
            var y = lerp(cur.y, tgt.y, t);
            var w = lerp(cur.w, tgt.w, t);
            var h = lerp(cur.h, tgt.h, t);
            if (t >= 1) cur = { x: tgt.x, y: tgt.y, w: tgt.w, h: tgt.h };

            ctx.strokeStyle = COLOURS[state];
            ctx.lineWidth = 3;
            ctx.strokeRect(x * W, y * H, w * W, h * H);

            var label = LABEL[state] || "";
            if (state === "lost_grace" && countdown !== null) {
                label += " " + countdown + "s";
            }
            if (label) {
                ctx.font = "14px monospace";
                var tw = ctx.measureText(label).width;
                var lx = x * W;
                var ly = y * H;
                ctx.fillStyle = "rgba(0,0,0,0.6)";
                ctx.fillRect(lx, ly - 18, tw + 8, 16);
                ctx.fillStyle = COLOURS[state];
                ctx.fillText(label, lx + 4, ly - 5);
            }
        }
        window.requestAnimationFrame(draw);
    }

    // Start hidden; the first poll flips visibility via setEnabled once
    // the overlay_enabled setting is known.
    canvas.hidden = true;
    poll();
    setInterval(poll, POLL_MS);
    window.requestAnimationFrame(draw);
})();
