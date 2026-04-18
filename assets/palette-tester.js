/**
 * WatchMyBirds — Palette Switcher
 * Three modes: Day · Night · Custom (placeholder for later).
 * Persists choice via localStorage across all pages.
 * Toggle: click the floating pill or Ctrl+Shift+P.
 */
(function () {
  'use strict';

  var STORAGE_KEY = 'wmb-theme';

  var THEMES = {
    day: {
      label: 'Day',
      icon: '\u2600\uFE0F',
      colors: {
        '--color-bg': '#fbfcfb', '--color-surface': '#ffffff',
        '--color-surface-2': '#f3f8f4', '--color-surface-3': '#e8f0ea',
        '--color-primary': '#2e7d52', '--color-primary-dark': '#1b5e3a', '--color-primary-light': '#b8e8cc',
        '--color-accent': '#e07828', '--color-accent-dark': '#c06018', '--color-accent-light': '#fde8cc',
        '--color-secondary': '#4088cc', '--color-secondary-light': '#dceaf8',
        '--color-text': '#1a2e22', '--color-text-muted': '#4a6858', '--color-text-subtle': '#88aa98',
        '--color-border': '#d0e0d4', '--color-border-light': '#e4f0e8',
        '--color-success': '#2d9e48', '--color-warning': '#e0a020', '--color-danger': '#d94040', '--color-info': '#3088c0',
        '--color-impact': '#b08860', '--color-impact-light': '#faf0e0', '--color-impact-dark': '#7a5c3a', '--color-impact-ratio': '#5a8aaa',
      }
    },
    night: {
      label: 'Night',
      icon: '\uD83C\uDF19',
      colors: {
        '--color-bg': '#282e38', '--color-surface': '#323a44',
        '--color-surface-2': '#3a4450', '--color-surface-3': '#44505c',
        '--color-primary': '#6ac8b0', '--color-primary-dark': '#4aa890', '--color-primary-light': '#2a4840',
        '--color-accent': '#e0aa70', '--color-accent-dark': '#c08850', '--color-accent-light': '#3a3020',
        '--color-secondary': '#7ab0d0', '--color-secondary-light': '#2a3a48',
        '--color-text': '#e8ece8', '--color-text-muted': '#a8bab4', '--color-text-subtle': '#708880',
        '--color-border': '#485660', '--color-border-light': '#3a4850',
        '--color-success': '#58c878', '--color-warning': '#e0b848', '--color-danger': '#e06060', '--color-info': '#60b0d8',
        '--color-impact': '#d0b080', '--color-impact-light': '#343028', '--color-impact-dark': '#e8d8b0', '--color-impact-ratio': '#78a0c0',
      }
    },
  };

  // ── Apply immediately from localStorage (no flash) ──────────
  var overrideEl = document.createElement('style');
  overrideEl.id = 'wmb-theme-overrides';
  document.head.appendChild(overrideEl);

  var activeTheme = null;
  var customColors = null;

  function loadSaved() {
    try {
      var raw = localStorage.getItem(STORAGE_KEY);
      if (raw) return JSON.parse(raw);
    } catch (e) {}
    return null;
  }

  function saveState() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        theme: activeTheme,
        custom: customColors
      }));
    } catch (e) {}
  }

  function applyColors(colors, themeKey) {
    if (!colors) { overrideEl.textContent = ''; return; }
    var rules = Object.keys(colors).map(function (k) {
      return '  ' + k + ': ' + colors[k] + ' !important;';
    }).join('\n');
    var css = ':root {\n' + rules + '\n}';

    // Preserve the design-system app-bar background (#f4ecd9) across themes;
    // only adapt app-bar text/link colors in night mode so copy stays readable.
    if (themeKey === 'night') {
      css += '\n.app-bar__link { color: ' + colors['--color-text-muted'] + ' !important; }';
      css += '\n.app-bar__link:hover { color: ' + colors['--color-text'] + ' !important; }';
      css += '\n.app-bar__link--primary { color: ' + colors['--color-text'] + ' !important; }';
      css += '\n.app-bar__link.active { color: ' + colors['--color-primary'] + ' !important; }';
      css += '\n.app-bar__brand { color: ' + colors['--color-text'] + ' !important; }';
      css += '\n.app-bar__toggle { color: ' + colors['--color-text'] + ' !important; }';
    }

    overrideEl.textContent = css;
  }

  // Restore saved state before first paint
  var saved = loadSaved();
  if (saved) {
    activeTheme = saved.theme;
    customColors = saved.custom || null;
    if (activeTheme && THEMES[activeTheme]) {
      applyColors(THEMES[activeTheme].colors, activeTheme);
    }
  }

  // ── Build UI after DOM ready ────────────────────────────────
  function init() {

    // Inject CSS
    var css = document.createElement('style');
    css.textContent = [
      '#wmb-switcher{',
      '  position:fixed;bottom:20px;right:20px;z-index:99999;',
      '  display:flex;align-items:center;gap:0;',
      '  background:rgba(30,34,40,.92);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);',
      '  border-radius:28px;border:1px solid rgba(255,255,255,.08);',
      '  box-shadow:0 8px 32px rgba(0,0,0,.3),0 0 0 0 rgba(92,184,160,0);',
      '  padding:4px;transition:all .4s cubic-bezier(.34,1.56,.64,1);',
      '  font-family:system-ui,-apple-system,sans-serif;',
      '}',
      '#wmb-switcher:hover{box-shadow:0 8px 32px rgba(0,0,0,.4),0 0 0 3px rgba(92,184,160,.15)}',
      '#wmb-switcher.collapsed{border-radius:50%;padding:0}',
      '#wmb-switcher.collapsed .wmb-btn-theme{display:none}',
      '#wmb-switcher.collapsed .wmb-btn-toggle{border-radius:50%;width:48px;height:48px}',

      '.wmb-btn-theme{',
      '  border:none;cursor:pointer;font-size:13px;font-weight:600;',
      '  padding:8px 14px;border-radius:22px;',
      '  background:transparent;color:rgba(255,255,255,.5);',
      '  transition:all .25s ease;display:flex;align-items:center;gap:5px;',
      '  font-family:inherit;white-space:nowrap;',
      '}',
      '.wmb-btn-theme:hover{color:rgba(255,255,255,.8);background:rgba(255,255,255,.06)}',
      '.wmb-btn-theme.active{color:#fff;background:rgba(255,255,255,.12)}',
      '.wmb-btn-theme.active.is-day{background:rgba(46,125,82,.35);color:#a8dbbe}',
      '.wmb-btn-theme.active.is-night{background:rgba(92,184,160,.2);color:#90d8c4}',

      '.wmb-btn-toggle{',
      '  border:none;cursor:pointer;font-size:18px;',
      '  padding:8px 10px;border-radius:22px;',
      '  background:transparent;color:rgba(255,255,255,.4);',
      '  transition:all .3s cubic-bezier(.34,1.56,.64,1);',
      '  display:flex;align-items:center;justify-content:center;',
      '  font-family:inherit;',
      '}',
      '.wmb-btn-toggle:hover{color:#fff;transform:rotate(-15deg)}',

      /* Smooth page transitions when theme changes */
      '.wmb-transitioning,.wmb-transitioning *,.wmb-transitioning *::before,.wmb-transitioning *::after{',
      '  transition:background-color .5s ease,color .4s ease,border-color .4s ease,',
      '  box-shadow .4s ease,fill .4s ease,stroke .4s ease !important;',
      '}',
    ].join('\n');
    document.head.appendChild(css);

    // Build switcher pill
    var switcher = document.createElement('div');
    switcher.id = 'wmb-switcher';

    var buttons = {};
    ['day', 'night'].forEach(function (key) {
      var t = THEMES[key];
      var btn = document.createElement('button');
      btn.className = 'wmb-btn-theme is-' + key + (activeTheme === key ? ' active' : '');
      btn.innerHTML = '<span>' + t.icon + '</span><span>' + t.label + '</span>';
      btn.addEventListener('click', function () { switchTheme(key); });
      buttons[key] = btn;
      switcher.appendChild(btn);
    });

    // Collapse toggle
    var toggleBtn = document.createElement('button');
    toggleBtn.className = 'wmb-btn-toggle';
    toggleBtn.textContent = '\uD83C\uDFA8';
    toggleBtn.title = 'Palette Switcher (Ctrl+Shift+P)';
    toggleBtn.addEventListener('click', function () {
      switcher.classList.toggle('collapsed');
      try { localStorage.setItem('wmb-pill-collapsed', switcher.classList.contains('collapsed') ? '1' : '0'); } catch(e){}
    });
    switcher.appendChild(toggleBtn);

    // Restore collapsed state
    try { if (localStorage.getItem('wmb-pill-collapsed') === '1') switcher.classList.add('collapsed'); } catch(e){}

    document.body.appendChild(switcher);

    // Keyboard shortcut
    document.addEventListener('keydown', function (e) {
      if (e.ctrlKey && e.shiftKey && e.key === 'P') {
        e.preventDefault();
        switcher.classList.toggle('collapsed');
      }
    });

    // ── Switch Logic ──────────────────────────────────────────
    function switchTheme(key) {
      // Enable smooth transition
      document.documentElement.classList.add('wmb-transitioning');
      setTimeout(function () { document.documentElement.classList.remove('wmb-transitioning'); }, 600);

      // Deactivate all
      Object.keys(buttons).forEach(function (k) { buttons[k].classList.remove('active'); });
      buttons[key].classList.add('active');

      if (key === null || key === activeTheme) {
        // Toggle off: back to default CSS
        activeTheme = null;
        applyColors(null, null);
      } else {
        activeTheme = key;
        applyColors(THEMES[key].colors, key);
      }

      saveState();
    }

    // If no theme was saved, don't highlight any button (show original CSS)
    if (!activeTheme) {
      Object.keys(buttons).forEach(function (k) { buttons[k].classList.remove('active'); });
    }
  }

  // Run init
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
