var pr = Object.defineProperty;
var mr = (i, t, e) => t in i ? pr(i, t, { enumerable: !0, configurable: !0, writable: !0, value: e }) : i[t] = e;
var M = (i, t, e) => mr(i, typeof t != "symbol" ? t + "" : t, e);
function et() {
}
function fo(i) {
  return i();
}
function Ls() {
  return /* @__PURE__ */ Object.create(null);
}
function We(i) {
  i.forEach(fo);
}
function uo(i) {
  return typeof i == "function";
}
function Mi(i, t) {
  return i != i ? t == t : i !== t || i && typeof i == "object" || typeof i == "function";
}
function br(i) {
  return Object.keys(i).length === 0;
}
function D(i, t) {
  i.appendChild(t);
}
function Q(i, t, e) {
  i.insertBefore(t, e || null);
}
function Z(i) {
  i.parentNode && i.parentNode.removeChild(i);
}
function _r(i, t) {
  for (let e = 0; e < i.length; e += 1)
    i[e] && i[e].d(t);
}
function E(i) {
  return document.createElement(i);
}
function Ri(i) {
  return document.createElementNS("http://www.w3.org/2000/svg", i);
}
function ut(i) {
  return document.createTextNode(i);
}
function tt() {
  return ut(" ");
}
function C(i, t, e) {
  e == null ? i.removeAttribute(t) : i.getAttribute(t) !== e && i.setAttribute(t, e);
}
function xr(i) {
  return Array.from(i.childNodes);
}
function vt(i, t) {
  t = "" + t, i.data !== t && (i.data = /** @type {string} */
  t);
}
function A(i, t, e, s) {
  e == null ? i.style.removeProperty(t) : i.style.setProperty(t, e, "");
}
let Te;
function we(i) {
  Te = i;
}
function go() {
  if (!Te) throw new Error("Function called outside component initialization");
  return Te;
}
function ps(i) {
  go().$$.on_mount.push(i);
}
function yr(i) {
  go().$$.on_destroy.push(i);
}
const re = [], Gi = [];
let ae = [];
const Rs = [], po = /* @__PURE__ */ Promise.resolve();
let Zi = !1;
function mo() {
  Zi || (Zi = !0, po.then(bo));
}
function vr() {
  return mo(), po;
}
function Ji(i) {
  ae.push(i);
}
const Ei = /* @__PURE__ */ new Set();
let se = 0;
function bo() {
  if (se !== 0)
    return;
  const i = Te;
  do {
    try {
      for (; se < re.length; ) {
        const t = re[se];
        se++, we(t), kr(t.$$);
      }
    } catch (t) {
      throw re.length = 0, se = 0, t;
    }
    for (we(null), re.length = 0, se = 0; Gi.length; ) Gi.pop()();
    for (let t = 0; t < ae.length; t += 1) {
      const e = ae[t];
      Ei.has(e) || (Ei.add(e), e());
    }
    ae.length = 0;
  } while (re.length);
  for (; Rs.length; )
    Rs.pop()();
  Zi = !1, Ei.clear(), we(i);
}
function kr(i) {
  if (i.fragment !== null) {
    i.update(), We(i.before_update);
    const t = i.dirty;
    i.dirty = [-1], i.fragment && i.fragment.p(i.ctx, t), i.after_update.forEach(Ji);
  }
}
function Mr(i) {
  const t = [], e = [];
  ae.forEach((s) => i.indexOf(s) === -1 ? t.push(s) : e.push(s)), e.forEach((s) => s()), ae = t;
}
const si = /* @__PURE__ */ new Set();
let Gt;
function wr() {
  Gt = {
    r: 0,
    c: [],
    p: Gt
    // parent group
  };
}
function Sr() {
  Gt.r || We(Gt.c), Gt = Gt.p;
}
function le(i, t) {
  i && i.i && (si.delete(i), i.i(t));
}
function Se(i, t, e, s) {
  if (i && i.o) {
    if (si.has(i)) return;
    si.add(i), Gt.c.push(() => {
      si.delete(i), s && (e && i.d(1), s());
    }), i.o(t);
  } else s && s();
}
function Es(i) {
  return (i == null ? void 0 : i.length) !== void 0 ? i : Array.from(i);
}
function Fi(i) {
  i && i.c();
}
function ni(i, t, e) {
  const { fragment: s, after_update: n } = i.$$;
  s && s.m(t, e), Ji(() => {
    const o = i.$$.on_mount.map(fo).filter(uo);
    i.$$.on_destroy ? i.$$.on_destroy.push(...o) : We(o), i.$$.on_mount = [];
  }), n.forEach(Ji);
}
function oi(i, t) {
  const e = i.$$;
  e.fragment !== null && (Mr(e.after_update), We(e.on_destroy), e.fragment && e.fragment.d(t), e.on_destroy = e.fragment = null, e.ctx = []);
}
function Pr(i, t) {
  i.$$.dirty[0] === -1 && (re.push(i), mo(), i.$$.dirty.fill(0)), i.$$.dirty[t / 31 | 0] |= 1 << t % 31;
}
function wi(i, t, e, s, n, o, r = null, a = [-1]) {
  const l = Te;
  we(i);
  const c = i.$$ = {
    fragment: null,
    ctx: [],
    // state
    props: o,
    update: et,
    not_equal: n,
    bound: Ls(),
    // lifecycle
    on_mount: [],
    on_destroy: [],
    on_disconnect: [],
    before_update: [],
    after_update: [],
    context: new Map(t.context || (l ? l.$$.context : [])),
    // everything else
    callbacks: Ls(),
    dirty: a,
    skip_bound: !1,
    root: t.target || l.$$.root
  };
  r && r(c.root);
  let h = !1;
  if (c.ctx = e ? e(i, t.props || {}, (d, f, ...u) => {
    const g = u.length ? u[0] : f;
    return c.ctx && n(c.ctx[d], c.ctx[d] = g) && (!c.skip_bound && c.bound[d] && c.bound[d](g), h && Pr(i, d)), f;
  }) : [], c.update(), h = !0, We(c.before_update), c.fragment = s ? s(c.ctx) : !1, t.target) {
    if (t.hydrate) {
      const d = xr(t.target);
      c.fragment && c.fragment.l(d), d.forEach(Z);
    } else
      c.fragment && c.fragment.c();
    t.intro && le(i.$$.fragment), ni(i, t.target, t.anchor), bo();
  }
  we(l);
}
class Si {
  constructor() {
    /**
     * ### PRIVATE API
     *
     * Do not use, may change at any time
     *
     * @type {any}
     */
    M(this, "$$");
    /**
     * ### PRIVATE API
     *
     * Do not use, may change at any time
     *
     * @type {any}
     */
    M(this, "$$set");
  }
  /** @returns {void} */
  $destroy() {
    oi(this, 1), this.$destroy = et;
  }
  /**
   * @template {Extract<keyof Events, string>} K
   * @param {K} type
   * @param {((e: Events[K]) => void) | null | undefined} callback
   * @returns {() => void}
   */
  $on(t, e) {
    if (!uo(e))
      return et;
    const s = this.$$.callbacks[t] || (this.$$.callbacks[t] = []);
    return s.push(e), () => {
      const n = s.indexOf(e);
      n !== -1 && s.splice(n, 1);
    };
  }
  /**
   * @param {Partial<Props>} props
   * @returns {void}
   */
  $set(t) {
    this.$$set && !br(t) && (this.$$.skip_bound = !0, this.$$set(t), this.$$.skip_bound = !1);
  }
}
const Dr = "4";
typeof window < "u" && (window.__svelte || (window.__svelte = { v: /* @__PURE__ */ new Set() })).v.add(Dr);
function Ar(i) {
  var q, it, B, W, $, at;
  let t, e, s, n = Fs(
    /*summary*/
    (q = i[0]) == null ? void 0 : q.total_detections
  ) + "", o, r, a, l, c, h, d = (
    /*summary*/
    (((it = i[0]) == null ? void 0 : it.total_species) || 0) + ""
  ), f, u, g, p, m, b, _ = (
    /*summary*/
    (((W = (B = i[0]) == null ? void 0 : B.date_range) == null ? void 0 : W.first) || "—") + ""
  ), y, v, x, S, k, w, P = (
    /*summary*/
    (((at = ($ = i[0]) == null ? void 0 : $.date_range) == null ? void 0 : at.last) || "—") + ""
  ), T, L, R;
  return {
    c() {
      t = E("div"), e = E("div"), s = E("div"), o = ut(n), r = tt(), a = E("div"), a.textContent = "Total Detections", l = tt(), c = E("div"), h = E("div"), f = ut(d), u = tt(), g = E("div"), g.textContent = "Species Detected", p = tt(), m = E("div"), b = E("div"), y = ut(_), v = tt(), x = E("div"), x.textContent = "First Detection", S = tt(), k = E("div"), w = E("div"), T = ut(P), L = tt(), R = E("div"), R.textContent = "Last Detection", A(s, "font-size", "2rem"), A(s, "font-weight", "700"), A(s, "color", "#1a202c"), A(a, "font-size", "0.85rem"), A(a, "color", "#718096"), A(a, "margin-top", "0.25rem"), A(e, "background", "white"), A(e, "border-radius", "12px"), A(e, "padding", "1.25rem"), A(e, "box-shadow", "0 2px 8px rgba(0,0,0,0.06)"), A(e, "border-left", "4px solid #4caf50"), A(h, "font-size", "2rem"), A(h, "font-weight", "700"), A(h, "color", "#1a202c"), A(g, "font-size", "0.85rem"), A(g, "color", "#718096"), A(g, "margin-top", "0.25rem"), A(c, "background", "white"), A(c, "border-radius", "12px"), A(c, "padding", "1.25rem"), A(c, "box-shadow", "0 2px 8px rgba(0,0,0,0.06)"), A(c, "border-left", "4px solid #0288d1"), A(b, "font-size", "1.1rem"), A(b, "font-weight", "700"), A(b, "color", "#1a202c"), A(x, "font-size", "0.85rem"), A(x, "color", "#718096"), A(x, "margin-top", "0.25rem"), A(m, "background", "white"), A(m, "border-radius", "12px"), A(m, "padding", "1.25rem"), A(m, "box-shadow", "0 2px 8px rgba(0,0,0,0.06)"), A(m, "border-left", "4px solid #00897b"), A(w, "font-size", "1.1rem"), A(w, "font-weight", "700"), A(w, "color", "#1a202c"), A(R, "font-size", "0.85rem"), A(R, "color", "#718096"), A(R, "margin-top", "0.25rem"), A(k, "background", "white"), A(k, "border-radius", "12px"), A(k, "padding", "1.25rem"), A(k, "box-shadow", "0 2px 8px rgba(0,0,0,0.06)"), A(k, "border-left", "4px solid #ff9800"), A(t, "display", "grid"), A(t, "grid-template-columns", "repeat(4, 1fr)"), A(t, "gap", "1rem"), A(t, "margin-bottom", "1.5rem"), C(t, "class", "svelte-qy1gjj");
    },
    m(H, ht) {
      Q(H, t, ht), D(t, e), D(e, s), D(s, o), D(e, r), D(e, a), D(t, l), D(t, c), D(c, h), D(h, f), D(c, u), D(c, g), D(t, p), D(t, m), D(m, b), D(b, y), D(m, v), D(m, x), D(t, S), D(t, k), D(k, w), D(w, T), D(k, L), D(k, R);
    },
    p(H, [ht]) {
      var jt, Mt, Lt, bt, Rt, _t;
      ht & /*summary*/
      1 && n !== (n = Fs(
        /*summary*/
        (jt = H[0]) == null ? void 0 : jt.total_detections
      ) + "") && vt(o, n), ht & /*summary*/
      1 && d !== (d = /*summary*/
      (((Mt = H[0]) == null ? void 0 : Mt.total_species) || 0) + "") && vt(f, d), ht & /*summary*/
      1 && _ !== (_ = /*summary*/
      (((bt = (Lt = H[0]) == null ? void 0 : Lt.date_range) == null ? void 0 : bt.first) || "—") + "") && vt(y, _), ht & /*summary*/
      1 && P !== (P = /*summary*/
      (((_t = (Rt = H[0]) == null ? void 0 : Rt.date_range) == null ? void 0 : _t.last) || "—") + "") && vt(T, P);
    },
    i: et,
    o: et,
    d(H) {
      H && Z(t);
    }
  };
}
function Fs(i) {
  return i >= 1e3 ? (i / 1e3).toFixed(1) + "k" : (i == null ? void 0 : i.toString()) || "0";
}
function Cr(i, t, e) {
  let { summary: s } = t;
  return i.$$set = (n) => {
    "summary" in n && e(0, s = n.summary);
  }, [s];
}
class Or extends Si {
  constructor(t) {
    super(), wi(this, t, Cr, Ar, Mi, { summary: 0 });
  }
}
/*!
 * @kurkle/color v0.3.4
 * https://github.com/kurkle/color#readme
 * (c) 2024 Jukka Kurkela
 * Released under the MIT License
 */
function Ne(i) {
  return i + 0.5 | 0;
}
const Et = (i, t, e) => Math.max(Math.min(i, e), t);
function _e(i) {
  return Et(Ne(i * 2.55), 0, 255);
}
function Vt(i) {
  return Et(Ne(i * 255), 0, 255);
}
function Dt(i) {
  return Et(Ne(i / 2.55) / 100, 0, 1);
}
function Is(i) {
  return Et(Ne(i * 100), 0, 100);
}
const ft = { 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, A: 10, B: 11, C: 12, D: 13, E: 14, F: 15, a: 10, b: 11, c: 12, d: 13, e: 14, f: 15 }, Qi = [..."0123456789ABCDEF"], Tr = (i) => Qi[i & 15], Lr = (i) => Qi[(i & 240) >> 4] + Qi[i & 15], je = (i) => (i & 240) >> 4 === (i & 15), Rr = (i) => je(i.r) && je(i.g) && je(i.b) && je(i.a);
function Er(i) {
  var t = i.length, e;
  return i[0] === "#" && (t === 4 || t === 5 ? e = {
    r: 255 & ft[i[1]] * 17,
    g: 255 & ft[i[2]] * 17,
    b: 255 & ft[i[3]] * 17,
    a: t === 5 ? ft[i[4]] * 17 : 255
  } : (t === 7 || t === 9) && (e = {
    r: ft[i[1]] << 4 | ft[i[2]],
    g: ft[i[3]] << 4 | ft[i[4]],
    b: ft[i[5]] << 4 | ft[i[6]],
    a: t === 9 ? ft[i[7]] << 4 | ft[i[8]] : 255
  })), e;
}
const Fr = (i, t) => i < 255 ? t(i) : "";
function Ir(i) {
  var t = Rr(i) ? Tr : Lr;
  return i ? "#" + t(i.r) + t(i.g) + t(i.b) + Fr(i.a, t) : void 0;
}
const zr = /^(hsla?|hwb|hsv)\(\s*([-+.e\d]+)(?:deg)?[\s,]+([-+.e\d]+)%[\s,]+([-+.e\d]+)%(?:[\s,]+([-+.e\d]+)(%)?)?\s*\)$/;
function _o(i, t, e) {
  const s = t * Math.min(e, 1 - e), n = (o, r = (o + i / 30) % 12) => e - s * Math.max(Math.min(r - 3, 9 - r, 1), -1);
  return [n(0), n(8), n(4)];
}
function Br(i, t, e) {
  const s = (n, o = (n + i / 60) % 6) => e - e * t * Math.max(Math.min(o, 4 - o, 1), 0);
  return [s(5), s(3), s(1)];
}
function Vr(i, t, e) {
  const s = _o(i, 1, 0.5);
  let n;
  for (t + e > 1 && (n = 1 / (t + e), t *= n, e *= n), n = 0; n < 3; n++)
    s[n] *= 1 - t - e, s[n] += t;
  return s;
}
function Wr(i, t, e, s, n) {
  return i === n ? (t - e) / s + (t < e ? 6 : 0) : t === n ? (e - i) / s + 2 : (i - t) / s + 4;
}
function ms(i) {
  const e = i.r / 255, s = i.g / 255, n = i.b / 255, o = Math.max(e, s, n), r = Math.min(e, s, n), a = (o + r) / 2;
  let l, c, h;
  return o !== r && (h = o - r, c = a > 0.5 ? h / (2 - o - r) : h / (o + r), l = Wr(e, s, n, h, o), l = l * 60 + 0.5), [l | 0, c || 0, a];
}
function bs(i, t, e, s) {
  return (Array.isArray(t) ? i(t[0], t[1], t[2]) : i(t, e, s)).map(Vt);
}
function _s(i, t, e) {
  return bs(_o, i, t, e);
}
function Nr(i, t, e) {
  return bs(Vr, i, t, e);
}
function Hr(i, t, e) {
  return bs(Br, i, t, e);
}
function xo(i) {
  return (i % 360 + 360) % 360;
}
function jr(i) {
  const t = zr.exec(i);
  let e = 255, s;
  if (!t)
    return;
  t[5] !== s && (e = t[6] ? _e(+t[5]) : Vt(+t[5]));
  const n = xo(+t[2]), o = +t[3] / 100, r = +t[4] / 100;
  return t[1] === "hwb" ? s = Nr(n, o, r) : t[1] === "hsv" ? s = Hr(n, o, r) : s = _s(n, o, r), {
    r: s[0],
    g: s[1],
    b: s[2],
    a: e
  };
}
function $r(i, t) {
  var e = ms(i);
  e[0] = xo(e[0] + t), e = _s(e), i.r = e[0], i.g = e[1], i.b = e[2];
}
function Yr(i) {
  if (!i)
    return;
  const t = ms(i), e = t[0], s = Is(t[1]), n = Is(t[2]);
  return i.a < 255 ? `hsla(${e}, ${s}%, ${n}%, ${Dt(i.a)})` : `hsl(${e}, ${s}%, ${n}%)`;
}
const zs = {
  x: "dark",
  Z: "light",
  Y: "re",
  X: "blu",
  W: "gr",
  V: "medium",
  U: "slate",
  A: "ee",
  T: "ol",
  S: "or",
  B: "ra",
  C: "lateg",
  D: "ights",
  R: "in",
  Q: "turquois",
  E: "hi",
  P: "ro",
  O: "al",
  N: "le",
  M: "de",
  L: "yello",
  F: "en",
  K: "ch",
  G: "arks",
  H: "ea",
  I: "ightg",
  J: "wh"
}, Bs = {
  OiceXe: "f0f8ff",
  antiquewEte: "faebd7",
  aqua: "ffff",
  aquamarRe: "7fffd4",
  azuY: "f0ffff",
  beige: "f5f5dc",
  bisque: "ffe4c4",
  black: "0",
  blanKedOmond: "ffebcd",
  Xe: "ff",
  XeviTet: "8a2be2",
  bPwn: "a52a2a",
  burlywood: "deb887",
  caMtXe: "5f9ea0",
  KartYuse: "7fff00",
  KocTate: "d2691e",
  cSO: "ff7f50",
  cSnflowerXe: "6495ed",
  cSnsilk: "fff8dc",
  crimson: "dc143c",
  cyan: "ffff",
  xXe: "8b",
  xcyan: "8b8b",
  xgTMnPd: "b8860b",
  xWay: "a9a9a9",
  xgYF: "6400",
  xgYy: "a9a9a9",
  xkhaki: "bdb76b",
  xmagFta: "8b008b",
  xTivegYF: "556b2f",
  xSange: "ff8c00",
  xScEd: "9932cc",
  xYd: "8b0000",
  xsOmon: "e9967a",
  xsHgYF: "8fbc8f",
  xUXe: "483d8b",
  xUWay: "2f4f4f",
  xUgYy: "2f4f4f",
  xQe: "ced1",
  xviTet: "9400d3",
  dAppRk: "ff1493",
  dApskyXe: "bfff",
  dimWay: "696969",
  dimgYy: "696969",
  dodgerXe: "1e90ff",
  fiYbrick: "b22222",
  flSOwEte: "fffaf0",
  foYstWAn: "228b22",
  fuKsia: "ff00ff",
  gaRsbSo: "dcdcdc",
  ghostwEte: "f8f8ff",
  gTd: "ffd700",
  gTMnPd: "daa520",
  Way: "808080",
  gYF: "8000",
  gYFLw: "adff2f",
  gYy: "808080",
  honeyMw: "f0fff0",
  hotpRk: "ff69b4",
  RdianYd: "cd5c5c",
  Rdigo: "4b0082",
  ivSy: "fffff0",
  khaki: "f0e68c",
  lavFMr: "e6e6fa",
  lavFMrXsh: "fff0f5",
  lawngYF: "7cfc00",
  NmoncEffon: "fffacd",
  ZXe: "add8e6",
  ZcSO: "f08080",
  Zcyan: "e0ffff",
  ZgTMnPdLw: "fafad2",
  ZWay: "d3d3d3",
  ZgYF: "90ee90",
  ZgYy: "d3d3d3",
  ZpRk: "ffb6c1",
  ZsOmon: "ffa07a",
  ZsHgYF: "20b2aa",
  ZskyXe: "87cefa",
  ZUWay: "778899",
  ZUgYy: "778899",
  ZstAlXe: "b0c4de",
  ZLw: "ffffe0",
  lime: "ff00",
  limegYF: "32cd32",
  lRF: "faf0e6",
  magFta: "ff00ff",
  maPon: "800000",
  VaquamarRe: "66cdaa",
  VXe: "cd",
  VScEd: "ba55d3",
  VpurpN: "9370db",
  VsHgYF: "3cb371",
  VUXe: "7b68ee",
  VsprRggYF: "fa9a",
  VQe: "48d1cc",
  VviTetYd: "c71585",
  midnightXe: "191970",
  mRtcYam: "f5fffa",
  mistyPse: "ffe4e1",
  moccasR: "ffe4b5",
  navajowEte: "ffdead",
  navy: "80",
  Tdlace: "fdf5e6",
  Tive: "808000",
  TivedBb: "6b8e23",
  Sange: "ffa500",
  SangeYd: "ff4500",
  ScEd: "da70d6",
  pOegTMnPd: "eee8aa",
  pOegYF: "98fb98",
  pOeQe: "afeeee",
  pOeviTetYd: "db7093",
  papayawEp: "ffefd5",
  pHKpuff: "ffdab9",
  peru: "cd853f",
  pRk: "ffc0cb",
  plum: "dda0dd",
  powMrXe: "b0e0e6",
  purpN: "800080",
  YbeccapurpN: "663399",
  Yd: "ff0000",
  Psybrown: "bc8f8f",
  PyOXe: "4169e1",
  saddNbPwn: "8b4513",
  sOmon: "fa8072",
  sandybPwn: "f4a460",
  sHgYF: "2e8b57",
  sHshell: "fff5ee",
  siFna: "a0522d",
  silver: "c0c0c0",
  skyXe: "87ceeb",
  UXe: "6a5acd",
  UWay: "708090",
  UgYy: "708090",
  snow: "fffafa",
  sprRggYF: "ff7f",
  stAlXe: "4682b4",
  tan: "d2b48c",
  teO: "8080",
  tEstN: "d8bfd8",
  tomato: "ff6347",
  Qe: "40e0d0",
  viTet: "ee82ee",
  JHt: "f5deb3",
  wEte: "ffffff",
  wEtesmoke: "f5f5f5",
  Lw: "ffff00",
  LwgYF: "9acd32"
};
function Xr() {
  const i = {}, t = Object.keys(Bs), e = Object.keys(zs);
  let s, n, o, r, a;
  for (s = 0; s < t.length; s++) {
    for (r = a = t[s], n = 0; n < e.length; n++)
      o = e[n], a = a.replace(o, zs[o]);
    o = parseInt(Bs[r], 16), i[a] = [o >> 16 & 255, o >> 8 & 255, o & 255];
  }
  return i;
}
let $e;
function Ur(i) {
  $e || ($e = Xr(), $e.transparent = [0, 0, 0, 0]);
  const t = $e[i.toLowerCase()];
  return t && {
    r: t[0],
    g: t[1],
    b: t[2],
    a: t.length === 4 ? t[3] : 255
  };
}
const Kr = /^rgba?\(\s*([-+.\d]+)(%)?[\s,]+([-+.e\d]+)(%)?[\s,]+([-+.e\d]+)(%)?(?:[\s,/]+([-+.e\d]+)(%)?)?\s*\)$/;
function qr(i) {
  const t = Kr.exec(i);
  let e = 255, s, n, o;
  if (t) {
    if (t[7] !== s) {
      const r = +t[7];
      e = t[8] ? _e(r) : Et(r * 255, 0, 255);
    }
    return s = +t[1], n = +t[3], o = +t[5], s = 255 & (t[2] ? _e(s) : Et(s, 0, 255)), n = 255 & (t[4] ? _e(n) : Et(n, 0, 255)), o = 255 & (t[6] ? _e(o) : Et(o, 0, 255)), {
      r: s,
      g: n,
      b: o,
      a: e
    };
  }
}
function Gr(i) {
  return i && (i.a < 255 ? `rgba(${i.r}, ${i.g}, ${i.b}, ${Dt(i.a)})` : `rgb(${i.r}, ${i.g}, ${i.b})`);
}
const Ii = (i) => i <= 31308e-7 ? i * 12.92 : Math.pow(i, 1 / 2.4) * 1.055 - 0.055, ne = (i) => i <= 0.04045 ? i / 12.92 : Math.pow((i + 0.055) / 1.055, 2.4);
function Zr(i, t, e) {
  const s = ne(Dt(i.r)), n = ne(Dt(i.g)), o = ne(Dt(i.b));
  return {
    r: Vt(Ii(s + e * (ne(Dt(t.r)) - s))),
    g: Vt(Ii(n + e * (ne(Dt(t.g)) - n))),
    b: Vt(Ii(o + e * (ne(Dt(t.b)) - o))),
    a: i.a + e * (t.a - i.a)
  };
}
function Ye(i, t, e) {
  if (i) {
    let s = ms(i);
    s[t] = Math.max(0, Math.min(s[t] + s[t] * e, t === 0 ? 360 : 1)), s = _s(s), i.r = s[0], i.g = s[1], i.b = s[2];
  }
}
function yo(i, t) {
  return i && Object.assign(t || {}, i);
}
function Vs(i) {
  var t = { r: 0, g: 0, b: 0, a: 255 };
  return Array.isArray(i) ? i.length >= 3 && (t = { r: i[0], g: i[1], b: i[2], a: 255 }, i.length > 3 && (t.a = Vt(i[3]))) : (t = yo(i, { r: 0, g: 0, b: 0, a: 1 }), t.a = Vt(t.a)), t;
}
function Jr(i) {
  return i.charAt(0) === "r" ? qr(i) : jr(i);
}
class Le {
  constructor(t) {
    if (t instanceof Le)
      return t;
    const e = typeof t;
    let s;
    e === "object" ? s = Vs(t) : e === "string" && (s = Er(t) || Ur(t) || Jr(t)), this._rgb = s, this._valid = !!s;
  }
  get valid() {
    return this._valid;
  }
  get rgb() {
    var t = yo(this._rgb);
    return t && (t.a = Dt(t.a)), t;
  }
  set rgb(t) {
    this._rgb = Vs(t);
  }
  rgbString() {
    return this._valid ? Gr(this._rgb) : void 0;
  }
  hexString() {
    return this._valid ? Ir(this._rgb) : void 0;
  }
  hslString() {
    return this._valid ? Yr(this._rgb) : void 0;
  }
  mix(t, e) {
    if (t) {
      const s = this.rgb, n = t.rgb;
      let o;
      const r = e === o ? 0.5 : e, a = 2 * r - 1, l = s.a - n.a, c = ((a * l === -1 ? a : (a + l) / (1 + a * l)) + 1) / 2;
      o = 1 - c, s.r = 255 & c * s.r + o * n.r + 0.5, s.g = 255 & c * s.g + o * n.g + 0.5, s.b = 255 & c * s.b + o * n.b + 0.5, s.a = r * s.a + (1 - r) * n.a, this.rgb = s;
    }
    return this;
  }
  interpolate(t, e) {
    return t && (this._rgb = Zr(this._rgb, t._rgb, e)), this;
  }
  clone() {
    return new Le(this.rgb);
  }
  alpha(t) {
    return this._rgb.a = Vt(t), this;
  }
  clearer(t) {
    const e = this._rgb;
    return e.a *= 1 - t, this;
  }
  greyscale() {
    const t = this._rgb, e = Ne(t.r * 0.3 + t.g * 0.59 + t.b * 0.11);
    return t.r = t.g = t.b = e, this;
  }
  opaquer(t) {
    const e = this._rgb;
    return e.a *= 1 + t, this;
  }
  negate() {
    const t = this._rgb;
    return t.r = 255 - t.r, t.g = 255 - t.g, t.b = 255 - t.b, this;
  }
  lighten(t) {
    return Ye(this._rgb, 2, t), this;
  }
  darken(t) {
    return Ye(this._rgb, 2, -t), this;
  }
  saturate(t) {
    return Ye(this._rgb, 1, t), this;
  }
  desaturate(t) {
    return Ye(this._rgb, 1, -t), this;
  }
  rotate(t) {
    return $r(this._rgb, t), this;
  }
}
/*!
 * Chart.js v4.5.1
 * https://www.chartjs.org
 * (c) 2025 Chart.js Contributors
 * Released under the MIT License
 */
function wt() {
}
const Qr = /* @__PURE__ */ (() => {
  let i = 0;
  return () => i++;
})();
function F(i) {
  return i == null;
}
function Y(i) {
  if (Array.isArray && Array.isArray(i))
    return !0;
  const t = Object.prototype.toString.call(i);
  return t.slice(0, 7) === "[object" && t.slice(-6) === "Array]";
}
function I(i) {
  return i !== null && Object.prototype.toString.call(i) === "[object Object]";
}
function U(i) {
  return (typeof i == "number" || i instanceof Number) && isFinite(+i);
}
function dt(i, t) {
  return U(i) ? i : t;
}
function O(i, t) {
  return typeof i > "u" ? t : i;
}
const ta = (i, t) => typeof i == "string" && i.endsWith("%") ? parseFloat(i) / 100 : +i / t, vo = (i, t) => typeof i == "string" && i.endsWith("%") ? parseFloat(i) / 100 * t : +i;
function N(i, t, e) {
  if (i && typeof i.call == "function")
    return i.apply(e, t);
}
function V(i, t, e, s) {
  let n, o, r;
  if (Y(i))
    for (o = i.length, n = 0; n < o; n++)
      t.call(e, i[n], n);
  else if (I(i))
    for (r = Object.keys(i), o = r.length, n = 0; n < o; n++)
      t.call(e, i[r[n]], r[n]);
}
function pi(i, t) {
  let e, s, n, o;
  if (!i || !t || i.length !== t.length)
    return !1;
  for (e = 0, s = i.length; e < s; ++e)
    if (n = i[e], o = t[e], n.datasetIndex !== o.datasetIndex || n.index !== o.index)
      return !1;
  return !0;
}
function mi(i) {
  if (Y(i))
    return i.map(mi);
  if (I(i)) {
    const t = /* @__PURE__ */ Object.create(null), e = Object.keys(i), s = e.length;
    let n = 0;
    for (; n < s; ++n)
      t[e[n]] = mi(i[e[n]]);
    return t;
  }
  return i;
}
function ko(i) {
  return [
    "__proto__",
    "prototype",
    "constructor"
  ].indexOf(i) === -1;
}
function ea(i, t, e, s) {
  if (!ko(i))
    return;
  const n = t[i], o = e[i];
  I(n) && I(o) ? Re(n, o, s) : t[i] = mi(o);
}
function Re(i, t, e) {
  const s = Y(t) ? t : [
    t
  ], n = s.length;
  if (!I(i))
    return i;
  e = e || {};
  const o = e.merger || ea;
  let r;
  for (let a = 0; a < n; ++a) {
    if (r = s[a], !I(r))
      continue;
    const l = Object.keys(r);
    for (let c = 0, h = l.length; c < h; ++c)
      o(l[c], i, r, e);
  }
  return i;
}
function Pe(i, t) {
  return Re(i, t, {
    merger: ia
  });
}
function ia(i, t, e) {
  if (!ko(i))
    return;
  const s = t[i], n = e[i];
  I(s) && I(n) ? Pe(s, n) : Object.prototype.hasOwnProperty.call(t, i) || (t[i] = mi(n));
}
const Ws = {
  // Chart.helpers.core resolveObjectKey should resolve empty key to root object
  "": (i) => i,
  // default resolvers
  x: (i) => i.x,
  y: (i) => i.y
};
function sa(i) {
  const t = i.split("."), e = [];
  let s = "";
  for (const n of t)
    s += n, s.endsWith("\\") ? s = s.slice(0, -1) + "." : (e.push(s), s = "");
  return e;
}
function na(i) {
  const t = sa(i);
  return (e) => {
    for (const s of t) {
      if (s === "")
        break;
      e = e && e[s];
    }
    return e;
  };
}
function Wt(i, t) {
  return (Ws[t] || (Ws[t] = na(t)))(i);
}
function xs(i) {
  return i.charAt(0).toUpperCase() + i.slice(1);
}
const Ee = (i) => typeof i < "u", Nt = (i) => typeof i == "function", Ns = (i, t) => {
  if (i.size !== t.size)
    return !1;
  for (const e of i)
    if (!t.has(e))
      return !1;
  return !0;
};
function oa(i) {
  return i.type === "mouseup" || i.type === "click" || i.type === "contextmenu";
}
const z = Math.PI, j = 2 * z, ra = j + z, bi = Number.POSITIVE_INFINITY, aa = z / 180, K = z / 2, $t = z / 4, Hs = z * 2 / 3, Ft = Math.log10, kt = Math.sign;
function De(i, t, e) {
  return Math.abs(i - t) < e;
}
function js(i) {
  const t = Math.round(i);
  i = De(i, t, i / 1e3) ? t : i;
  const e = Math.pow(10, Math.floor(Ft(i))), s = i / e;
  return (s <= 1 ? 1 : s <= 2 ? 2 : s <= 5 ? 5 : 10) * e;
}
function la(i) {
  const t = [], e = Math.sqrt(i);
  let s;
  for (s = 1; s < e; s++)
    i % s === 0 && (t.push(s), t.push(i / s));
  return e === (e | 0) && t.push(e), t.sort((n, o) => n - o).pop(), t;
}
function ca(i) {
  return typeof i == "symbol" || typeof i == "object" && i !== null && !(Symbol.toPrimitive in i || "toString" in i || "valueOf" in i);
}
function he(i) {
  return !ca(i) && !isNaN(parseFloat(i)) && isFinite(i);
}
function ha(i, t) {
  const e = Math.round(i);
  return e - t <= i && e + t >= i;
}
function Mo(i, t, e) {
  let s, n, o;
  for (s = 0, n = i.length; s < n; s++)
    o = i[s][e], isNaN(o) || (t.min = Math.min(t.min, o), t.max = Math.max(t.max, o));
}
function gt(i) {
  return i * (z / 180);
}
function ys(i) {
  return i * (180 / z);
}
function $s(i) {
  if (!U(i))
    return;
  let t = 1, e = 0;
  for (; Math.round(i * t) / t !== i; )
    t *= 10, e++;
  return e;
}
function wo(i, t) {
  const e = t.x - i.x, s = t.y - i.y, n = Math.sqrt(e * e + s * s);
  let o = Math.atan2(s, e);
  return o < -0.5 * z && (o += j), {
    angle: o,
    distance: n
  };
}
function ts(i, t) {
  return Math.sqrt(Math.pow(t.x - i.x, 2) + Math.pow(t.y - i.y, 2));
}
function da(i, t) {
  return (i - t + ra) % j - z;
}
function nt(i) {
  return (i % j + j) % j;
}
function Fe(i, t, e, s) {
  const n = nt(i), o = nt(t), r = nt(e), a = nt(o - n), l = nt(r - n), c = nt(n - o), h = nt(n - r);
  return n === o || n === r || s && o === r || a > l && c < h;
}
function J(i, t, e) {
  return Math.max(t, Math.min(e, i));
}
function fa(i) {
  return J(i, -32768, 32767);
}
function Ct(i, t, e, s = 1e-6) {
  return i >= Math.min(t, e) - s && i <= Math.max(t, e) + s;
}
function vs(i, t, e) {
  e = e || ((r) => i[r] < t);
  let s = i.length - 1, n = 0, o;
  for (; s - n > 1; )
    o = n + s >> 1, e(o) ? n = o : s = o;
  return {
    lo: n,
    hi: s
  };
}
const Ot = (i, t, e, s) => vs(i, e, s ? (n) => {
  const o = i[n][t];
  return o < e || o === e && i[n + 1][t] === e;
} : (n) => i[n][t] < e), ua = (i, t, e) => vs(i, e, (s) => i[s][t] >= e);
function ga(i, t, e) {
  let s = 0, n = i.length;
  for (; s < n && i[s] < t; )
    s++;
  for (; n > s && i[n - 1] > e; )
    n--;
  return s > 0 || n < i.length ? i.slice(s, n) : i;
}
const So = [
  "push",
  "pop",
  "shift",
  "splice",
  "unshift"
];
function pa(i, t) {
  if (i._chartjs) {
    i._chartjs.listeners.push(t);
    return;
  }
  Object.defineProperty(i, "_chartjs", {
    configurable: !0,
    enumerable: !1,
    value: {
      listeners: [
        t
      ]
    }
  }), So.forEach((e) => {
    const s = "_onData" + xs(e), n = i[e];
    Object.defineProperty(i, e, {
      configurable: !0,
      enumerable: !1,
      value(...o) {
        const r = n.apply(this, o);
        return i._chartjs.listeners.forEach((a) => {
          typeof a[s] == "function" && a[s](...o);
        }), r;
      }
    });
  });
}
function Ys(i, t) {
  const e = i._chartjs;
  if (!e)
    return;
  const s = e.listeners, n = s.indexOf(t);
  n !== -1 && s.splice(n, 1), !(s.length > 0) && (So.forEach((o) => {
    delete i[o];
  }), delete i._chartjs);
}
function Po(i) {
  const t = new Set(i);
  return t.size === i.length ? i : Array.from(t);
}
const Do = function() {
  return typeof window > "u" ? function(i) {
    return i();
  } : window.requestAnimationFrame;
}();
function Ao(i, t) {
  let e = [], s = !1;
  return function(...n) {
    e = n, s || (s = !0, Do.call(window, () => {
      s = !1, i.apply(t, e);
    }));
  };
}
function ma(i, t) {
  let e;
  return function(...s) {
    return t ? (clearTimeout(e), e = setTimeout(i, t, s)) : i.apply(this, s), t;
  };
}
const ks = (i) => i === "start" ? "left" : i === "end" ? "right" : "center", st = (i, t, e) => i === "start" ? t : i === "end" ? e : (t + e) / 2, ba = (i, t, e, s) => i === (s ? "left" : "right") ? e : i === "center" ? (t + e) / 2 : t;
function Co(i, t, e) {
  const s = t.length;
  let n = 0, o = s;
  if (i._sorted) {
    const { iScale: r, vScale: a, _parsed: l } = i, c = i.dataset && i.dataset.options ? i.dataset.options.spanGaps : null, h = r.axis, { min: d, max: f, minDefined: u, maxDefined: g } = r.getUserBounds();
    if (u) {
      if (n = Math.min(
        // @ts-expect-error Need to type _parsed
        Ot(l, h, d).lo,
        // @ts-expect-error Need to fix types on _lookupByKey
        e ? s : Ot(t, h, r.getPixelForValue(d)).lo
      ), c) {
        const p = l.slice(0, n + 1).reverse().findIndex((m) => !F(m[a.axis]));
        n -= Math.max(0, p);
      }
      n = J(n, 0, s - 1);
    }
    if (g) {
      let p = Math.max(
        // @ts-expect-error Need to type _parsed
        Ot(l, r.axis, f, !0).hi + 1,
        // @ts-expect-error Need to fix types on _lookupByKey
        e ? 0 : Ot(t, h, r.getPixelForValue(f), !0).hi + 1
      );
      if (c) {
        const m = l.slice(p - 1).findIndex((b) => !F(b[a.axis]));
        p += Math.max(0, m);
      }
      o = J(p, n, s) - n;
    } else
      o = s - n;
  }
  return {
    start: n,
    count: o
  };
}
function Oo(i) {
  const { xScale: t, yScale: e, _scaleRanges: s } = i, n = {
    xmin: t.min,
    xmax: t.max,
    ymin: e.min,
    ymax: e.max
  };
  if (!s)
    return i._scaleRanges = n, !0;
  const o = s.xmin !== t.min || s.xmax !== t.max || s.ymin !== e.min || s.ymax !== e.max;
  return Object.assign(s, n), o;
}
const Xe = (i) => i === 0 || i === 1, Xs = (i, t, e) => -(Math.pow(2, 10 * (i -= 1)) * Math.sin((i - t) * j / e)), Us = (i, t, e) => Math.pow(2, -10 * i) * Math.sin((i - t) * j / e) + 1, Ae = {
  linear: (i) => i,
  easeInQuad: (i) => i * i,
  easeOutQuad: (i) => -i * (i - 2),
  easeInOutQuad: (i) => (i /= 0.5) < 1 ? 0.5 * i * i : -0.5 * (--i * (i - 2) - 1),
  easeInCubic: (i) => i * i * i,
  easeOutCubic: (i) => (i -= 1) * i * i + 1,
  easeInOutCubic: (i) => (i /= 0.5) < 1 ? 0.5 * i * i * i : 0.5 * ((i -= 2) * i * i + 2),
  easeInQuart: (i) => i * i * i * i,
  easeOutQuart: (i) => -((i -= 1) * i * i * i - 1),
  easeInOutQuart: (i) => (i /= 0.5) < 1 ? 0.5 * i * i * i * i : -0.5 * ((i -= 2) * i * i * i - 2),
  easeInQuint: (i) => i * i * i * i * i,
  easeOutQuint: (i) => (i -= 1) * i * i * i * i + 1,
  easeInOutQuint: (i) => (i /= 0.5) < 1 ? 0.5 * i * i * i * i * i : 0.5 * ((i -= 2) * i * i * i * i + 2),
  easeInSine: (i) => -Math.cos(i * K) + 1,
  easeOutSine: (i) => Math.sin(i * K),
  easeInOutSine: (i) => -0.5 * (Math.cos(z * i) - 1),
  easeInExpo: (i) => i === 0 ? 0 : Math.pow(2, 10 * (i - 1)),
  easeOutExpo: (i) => i === 1 ? 1 : -Math.pow(2, -10 * i) + 1,
  easeInOutExpo: (i) => Xe(i) ? i : i < 0.5 ? 0.5 * Math.pow(2, 10 * (i * 2 - 1)) : 0.5 * (-Math.pow(2, -10 * (i * 2 - 1)) + 2),
  easeInCirc: (i) => i >= 1 ? i : -(Math.sqrt(1 - i * i) - 1),
  easeOutCirc: (i) => Math.sqrt(1 - (i -= 1) * i),
  easeInOutCirc: (i) => (i /= 0.5) < 1 ? -0.5 * (Math.sqrt(1 - i * i) - 1) : 0.5 * (Math.sqrt(1 - (i -= 2) * i) + 1),
  easeInElastic: (i) => Xe(i) ? i : Xs(i, 0.075, 0.3),
  easeOutElastic: (i) => Xe(i) ? i : Us(i, 0.075, 0.3),
  easeInOutElastic(i) {
    return Xe(i) ? i : i < 0.5 ? 0.5 * Xs(i * 2, 0.1125, 0.45) : 0.5 + 0.5 * Us(i * 2 - 1, 0.1125, 0.45);
  },
  easeInBack(i) {
    return i * i * ((1.70158 + 1) * i - 1.70158);
  },
  easeOutBack(i) {
    return (i -= 1) * i * ((1.70158 + 1) * i + 1.70158) + 1;
  },
  easeInOutBack(i) {
    let t = 1.70158;
    return (i /= 0.5) < 1 ? 0.5 * (i * i * (((t *= 1.525) + 1) * i - t)) : 0.5 * ((i -= 2) * i * (((t *= 1.525) + 1) * i + t) + 2);
  },
  easeInBounce: (i) => 1 - Ae.easeOutBounce(1 - i),
  easeOutBounce(i) {
    return i < 1 / 2.75 ? 7.5625 * i * i : i < 2 / 2.75 ? 7.5625 * (i -= 1.5 / 2.75) * i + 0.75 : i < 2.5 / 2.75 ? 7.5625 * (i -= 2.25 / 2.75) * i + 0.9375 : 7.5625 * (i -= 2.625 / 2.75) * i + 0.984375;
  },
  easeInOutBounce: (i) => i < 0.5 ? Ae.easeInBounce(i * 2) * 0.5 : Ae.easeOutBounce(i * 2 - 1) * 0.5 + 0.5
};
function Ms(i) {
  if (i && typeof i == "object") {
    const t = i.toString();
    return t === "[object CanvasPattern]" || t === "[object CanvasGradient]";
  }
  return !1;
}
function Ks(i) {
  return Ms(i) ? i : new Le(i);
}
function zi(i) {
  return Ms(i) ? i : new Le(i).saturate(0.5).darken(0.1).hexString();
}
const _a = [
  "x",
  "y",
  "borderWidth",
  "radius",
  "tension"
], xa = [
  "color",
  "borderColor",
  "backgroundColor"
];
function ya(i) {
  i.set("animation", {
    delay: void 0,
    duration: 1e3,
    easing: "easeOutQuart",
    fn: void 0,
    from: void 0,
    loop: void 0,
    to: void 0,
    type: void 0
  }), i.describe("animation", {
    _fallback: !1,
    _indexable: !1,
    _scriptable: (t) => t !== "onProgress" && t !== "onComplete" && t !== "fn"
  }), i.set("animations", {
    colors: {
      type: "color",
      properties: xa
    },
    numbers: {
      type: "number",
      properties: _a
    }
  }), i.describe("animations", {
    _fallback: "animation"
  }), i.set("transitions", {
    active: {
      animation: {
        duration: 400
      }
    },
    resize: {
      animation: {
        duration: 0
      }
    },
    show: {
      animations: {
        colors: {
          from: "transparent"
        },
        visible: {
          type: "boolean",
          duration: 0
        }
      }
    },
    hide: {
      animations: {
        colors: {
          to: "transparent"
        },
        visible: {
          type: "boolean",
          easing: "linear",
          fn: (t) => t | 0
        }
      }
    }
  });
}
function va(i) {
  i.set("layout", {
    autoPadding: !0,
    padding: {
      top: 0,
      right: 0,
      bottom: 0,
      left: 0
    }
  });
}
const qs = /* @__PURE__ */ new Map();
function ka(i, t) {
  t = t || {};
  const e = i + JSON.stringify(t);
  let s = qs.get(e);
  return s || (s = new Intl.NumberFormat(i, t), qs.set(e, s)), s;
}
function He(i, t, e) {
  return ka(t, e).format(i);
}
const To = {
  values(i) {
    return Y(i) ? i : "" + i;
  },
  numeric(i, t, e) {
    if (i === 0)
      return "0";
    const s = this.chart.options.locale;
    let n, o = i;
    if (e.length > 1) {
      const c = Math.max(Math.abs(e[0].value), Math.abs(e[e.length - 1].value));
      (c < 1e-4 || c > 1e15) && (n = "scientific"), o = Ma(i, e);
    }
    const r = Ft(Math.abs(o)), a = isNaN(r) ? 1 : Math.max(Math.min(-1 * Math.floor(r), 20), 0), l = {
      notation: n,
      minimumFractionDigits: a,
      maximumFractionDigits: a
    };
    return Object.assign(l, this.options.ticks.format), He(i, s, l);
  },
  logarithmic(i, t, e) {
    if (i === 0)
      return "0";
    const s = e[t].significand || i / Math.pow(10, Math.floor(Ft(i)));
    return [
      1,
      2,
      3,
      5,
      10,
      15
    ].includes(s) || t > 0.8 * e.length ? To.numeric.call(this, i, t, e) : "";
  }
};
function Ma(i, t) {
  let e = t.length > 3 ? t[2].value - t[1].value : t[1].value - t[0].value;
  return Math.abs(e) >= 1 && i !== Math.floor(i) && (e = i - Math.floor(i)), e;
}
var Pi = {
  formatters: To
};
function wa(i) {
  i.set("scale", {
    display: !0,
    offset: !1,
    reverse: !1,
    beginAtZero: !1,
    bounds: "ticks",
    clip: !0,
    grace: 0,
    grid: {
      display: !0,
      lineWidth: 1,
      drawOnChartArea: !0,
      drawTicks: !0,
      tickLength: 8,
      tickWidth: (t, e) => e.lineWidth,
      tickColor: (t, e) => e.color,
      offset: !1
    },
    border: {
      display: !0,
      dash: [],
      dashOffset: 0,
      width: 1
    },
    title: {
      display: !1,
      text: "",
      padding: {
        top: 4,
        bottom: 4
      }
    },
    ticks: {
      minRotation: 0,
      maxRotation: 50,
      mirror: !1,
      textStrokeWidth: 0,
      textStrokeColor: "",
      padding: 3,
      display: !0,
      autoSkip: !0,
      autoSkipPadding: 3,
      labelOffset: 0,
      callback: Pi.formatters.values,
      minor: {},
      major: {},
      align: "center",
      crossAlign: "near",
      showLabelBackdrop: !1,
      backdropColor: "rgba(255, 255, 255, 0.75)",
      backdropPadding: 2
    }
  }), i.route("scale.ticks", "color", "", "color"), i.route("scale.grid", "color", "", "borderColor"), i.route("scale.border", "color", "", "borderColor"), i.route("scale.title", "color", "", "color"), i.describe("scale", {
    _fallback: !1,
    _scriptable: (t) => !t.startsWith("before") && !t.startsWith("after") && t !== "callback" && t !== "parser",
    _indexable: (t) => t !== "borderDash" && t !== "tickBorderDash" && t !== "dash"
  }), i.describe("scales", {
    _fallback: "scale"
  }), i.describe("scale.ticks", {
    _scriptable: (t) => t !== "backdropPadding" && t !== "callback",
    _indexable: (t) => t !== "backdropPadding"
  });
}
const te = /* @__PURE__ */ Object.create(null), es = /* @__PURE__ */ Object.create(null);
function Ce(i, t) {
  if (!t)
    return i;
  const e = t.split(".");
  for (let s = 0, n = e.length; s < n; ++s) {
    const o = e[s];
    i = i[o] || (i[o] = /* @__PURE__ */ Object.create(null));
  }
  return i;
}
function Bi(i, t, e) {
  return typeof t == "string" ? Re(Ce(i, t), e) : Re(Ce(i, ""), t);
}
class Sa {
  constructor(t, e) {
    this.animation = void 0, this.backgroundColor = "rgba(0,0,0,0.1)", this.borderColor = "rgba(0,0,0,0.1)", this.color = "#666", this.datasets = {}, this.devicePixelRatio = (s) => s.chart.platform.getDevicePixelRatio(), this.elements = {}, this.events = [
      "mousemove",
      "mouseout",
      "click",
      "touchstart",
      "touchmove"
    ], this.font = {
      family: "'Helvetica Neue', 'Helvetica', 'Arial', sans-serif",
      size: 12,
      style: "normal",
      lineHeight: 1.2,
      weight: null
    }, this.hover = {}, this.hoverBackgroundColor = (s, n) => zi(n.backgroundColor), this.hoverBorderColor = (s, n) => zi(n.borderColor), this.hoverColor = (s, n) => zi(n.color), this.indexAxis = "x", this.interaction = {
      mode: "nearest",
      intersect: !0,
      includeInvisible: !1
    }, this.maintainAspectRatio = !0, this.onHover = null, this.onClick = null, this.parsing = !0, this.plugins = {}, this.responsive = !0, this.scale = void 0, this.scales = {}, this.showLine = !0, this.drawActiveElementsOnTop = !0, this.describe(t), this.apply(e);
  }
  set(t, e) {
    return Bi(this, t, e);
  }
  get(t) {
    return Ce(this, t);
  }
  describe(t, e) {
    return Bi(es, t, e);
  }
  override(t, e) {
    return Bi(te, t, e);
  }
  route(t, e, s, n) {
    const o = Ce(this, t), r = Ce(this, s), a = "_" + e;
    Object.defineProperties(o, {
      [a]: {
        value: o[e],
        writable: !0
      },
      [e]: {
        enumerable: !0,
        get() {
          const l = this[a], c = r[n];
          return I(l) ? Object.assign({}, c, l) : O(l, c);
        },
        set(l) {
          this[a] = l;
        }
      }
    });
  }
  apply(t) {
    t.forEach((e) => e(this));
  }
}
var X = /* @__PURE__ */ new Sa({
  _scriptable: (i) => !i.startsWith("on"),
  _indexable: (i) => i !== "events",
  hover: {
    _fallback: "interaction"
  },
  interaction: {
    _scriptable: !1,
    _indexable: !1
  }
}, [
  ya,
  va,
  wa
]);
function Pa(i) {
  return !i || F(i.size) || F(i.family) ? null : (i.style ? i.style + " " : "") + (i.weight ? i.weight + " " : "") + i.size + "px " + i.family;
}
function _i(i, t, e, s, n) {
  let o = t[n];
  return o || (o = t[n] = i.measureText(n).width, e.push(n)), o > s && (s = o), s;
}
function Da(i, t, e, s) {
  s = s || {};
  let n = s.data = s.data || {}, o = s.garbageCollect = s.garbageCollect || [];
  s.font !== t && (n = s.data = {}, o = s.garbageCollect = [], s.font = t), i.save(), i.font = t;
  let r = 0;
  const a = e.length;
  let l, c, h, d, f;
  for (l = 0; l < a; l++)
    if (d = e[l], d != null && !Y(d))
      r = _i(i, n, o, r, d);
    else if (Y(d))
      for (c = 0, h = d.length; c < h; c++)
        f = d[c], f != null && !Y(f) && (r = _i(i, n, o, r, f));
  i.restore();
  const u = o.length / 2;
  if (u > e.length) {
    for (l = 0; l < u; l++)
      delete n[o[l]];
    o.splice(0, u);
  }
  return r;
}
function Yt(i, t, e) {
  const s = i.currentDevicePixelRatio, n = e !== 0 ? Math.max(e / 2, 0.5) : 0;
  return Math.round((t - n) * s) / s + n;
}
function Gs(i, t) {
  !t && !i || (t = t || i.getContext("2d"), t.save(), t.resetTransform(), t.clearRect(0, 0, i.width, i.height), t.restore());
}
function is(i, t, e, s) {
  Lo(i, t, e, s, null);
}
function Lo(i, t, e, s, n) {
  let o, r, a, l, c, h, d, f;
  const u = t.pointStyle, g = t.rotation, p = t.radius;
  let m = (g || 0) * aa;
  if (u && typeof u == "object" && (o = u.toString(), o === "[object HTMLImageElement]" || o === "[object HTMLCanvasElement]")) {
    i.save(), i.translate(e, s), i.rotate(m), i.drawImage(u, -u.width / 2, -u.height / 2, u.width, u.height), i.restore();
    return;
  }
  if (!(isNaN(p) || p <= 0)) {
    switch (i.beginPath(), u) {
      default:
        n ? i.ellipse(e, s, n / 2, p, 0, 0, j) : i.arc(e, s, p, 0, j), i.closePath();
        break;
      case "triangle":
        h = n ? n / 2 : p, i.moveTo(e + Math.sin(m) * h, s - Math.cos(m) * p), m += Hs, i.lineTo(e + Math.sin(m) * h, s - Math.cos(m) * p), m += Hs, i.lineTo(e + Math.sin(m) * h, s - Math.cos(m) * p), i.closePath();
        break;
      case "rectRounded":
        c = p * 0.516, l = p - c, r = Math.cos(m + $t) * l, d = Math.cos(m + $t) * (n ? n / 2 - c : l), a = Math.sin(m + $t) * l, f = Math.sin(m + $t) * (n ? n / 2 - c : l), i.arc(e - d, s - a, c, m - z, m - K), i.arc(e + f, s - r, c, m - K, m), i.arc(e + d, s + a, c, m, m + K), i.arc(e - f, s + r, c, m + K, m + z), i.closePath();
        break;
      case "rect":
        if (!g) {
          l = Math.SQRT1_2 * p, h = n ? n / 2 : l, i.rect(e - h, s - l, 2 * h, 2 * l);
          break;
        }
        m += $t;
      case "rectRot":
        d = Math.cos(m) * (n ? n / 2 : p), r = Math.cos(m) * p, a = Math.sin(m) * p, f = Math.sin(m) * (n ? n / 2 : p), i.moveTo(e - d, s - a), i.lineTo(e + f, s - r), i.lineTo(e + d, s + a), i.lineTo(e - f, s + r), i.closePath();
        break;
      case "crossRot":
        m += $t;
      case "cross":
        d = Math.cos(m) * (n ? n / 2 : p), r = Math.cos(m) * p, a = Math.sin(m) * p, f = Math.sin(m) * (n ? n / 2 : p), i.moveTo(e - d, s - a), i.lineTo(e + d, s + a), i.moveTo(e + f, s - r), i.lineTo(e - f, s + r);
        break;
      case "star":
        d = Math.cos(m) * (n ? n / 2 : p), r = Math.cos(m) * p, a = Math.sin(m) * p, f = Math.sin(m) * (n ? n / 2 : p), i.moveTo(e - d, s - a), i.lineTo(e + d, s + a), i.moveTo(e + f, s - r), i.lineTo(e - f, s + r), m += $t, d = Math.cos(m) * (n ? n / 2 : p), r = Math.cos(m) * p, a = Math.sin(m) * p, f = Math.sin(m) * (n ? n / 2 : p), i.moveTo(e - d, s - a), i.lineTo(e + d, s + a), i.moveTo(e + f, s - r), i.lineTo(e - f, s + r);
        break;
      case "line":
        r = n ? n / 2 : Math.cos(m) * p, a = Math.sin(m) * p, i.moveTo(e - r, s - a), i.lineTo(e + r, s + a);
        break;
      case "dash":
        i.moveTo(e, s), i.lineTo(e + Math.cos(m) * (n ? n / 2 : p), s + Math.sin(m) * p);
        break;
      case !1:
        i.closePath();
        break;
    }
    i.fill(), t.borderWidth > 0 && i.stroke();
  }
}
function Tt(i, t, e) {
  return e = e || 0.5, !t || i && i.x > t.left - e && i.x < t.right + e && i.y > t.top - e && i.y < t.bottom + e;
}
function Di(i, t) {
  i.save(), i.beginPath(), i.rect(t.left, t.top, t.right - t.left, t.bottom - t.top), i.clip();
}
function Ai(i) {
  i.restore();
}
function Aa(i, t, e, s, n) {
  if (!t)
    return i.lineTo(e.x, e.y);
  if (n === "middle") {
    const o = (t.x + e.x) / 2;
    i.lineTo(o, t.y), i.lineTo(o, e.y);
  } else n === "after" != !!s ? i.lineTo(t.x, e.y) : i.lineTo(e.x, t.y);
  i.lineTo(e.x, e.y);
}
function Ca(i, t, e, s) {
  if (!t)
    return i.lineTo(e.x, e.y);
  i.bezierCurveTo(s ? t.cp1x : t.cp2x, s ? t.cp1y : t.cp2y, s ? e.cp2x : e.cp1x, s ? e.cp2y : e.cp1y, e.x, e.y);
}
function Oa(i, t) {
  t.translation && i.translate(t.translation[0], t.translation[1]), F(t.rotation) || i.rotate(t.rotation), t.color && (i.fillStyle = t.color), t.textAlign && (i.textAlign = t.textAlign), t.textBaseline && (i.textBaseline = t.textBaseline);
}
function Ta(i, t, e, s, n) {
  if (n.strikethrough || n.underline) {
    const o = i.measureText(s), r = t - o.actualBoundingBoxLeft, a = t + o.actualBoundingBoxRight, l = e - o.actualBoundingBoxAscent, c = e + o.actualBoundingBoxDescent, h = n.strikethrough ? (l + c) / 2 : c;
    i.strokeStyle = i.fillStyle, i.beginPath(), i.lineWidth = n.decorationWidth || 2, i.moveTo(r, h), i.lineTo(a, h), i.stroke();
  }
}
function La(i, t) {
  const e = i.fillStyle;
  i.fillStyle = t.color, i.fillRect(t.left, t.top, t.width, t.height), i.fillStyle = e;
}
function ee(i, t, e, s, n, o = {}) {
  const r = Y(t) ? t : [
    t
  ], a = o.strokeWidth > 0 && o.strokeColor !== "";
  let l, c;
  for (i.save(), i.font = n.string, Oa(i, o), l = 0; l < r.length; ++l)
    c = r[l], o.backdrop && La(i, o.backdrop), a && (o.strokeColor && (i.strokeStyle = o.strokeColor), F(o.strokeWidth) || (i.lineWidth = o.strokeWidth), i.strokeText(c, e, s, o.maxWidth)), i.fillText(c, e, s, o.maxWidth), Ta(i, e, s, c, o), s += Number(n.lineHeight);
  i.restore();
}
function Ie(i, t) {
  const { x: e, y: s, w: n, h: o, radius: r } = t;
  i.arc(e + r.topLeft, s + r.topLeft, r.topLeft, 1.5 * z, z, !0), i.lineTo(e, s + o - r.bottomLeft), i.arc(e + r.bottomLeft, s + o - r.bottomLeft, r.bottomLeft, z, K, !0), i.lineTo(e + n - r.bottomRight, s + o), i.arc(e + n - r.bottomRight, s + o - r.bottomRight, r.bottomRight, K, 0, !0), i.lineTo(e + n, s + r.topRight), i.arc(e + n - r.topRight, s + r.topRight, r.topRight, 0, -K, !0), i.lineTo(e + r.topLeft, s);
}
const Ra = /^(normal|(\d+(?:\.\d+)?)(px|em|%)?)$/, Ea = /^(normal|italic|initial|inherit|unset|(oblique( -?[0-9]?[0-9]deg)?))$/;
function Fa(i, t) {
  const e = ("" + i).match(Ra);
  if (!e || e[1] === "normal")
    return t * 1.2;
  switch (i = +e[2], e[3]) {
    case "px":
      return i;
    case "%":
      i /= 100;
      break;
  }
  return t * i;
}
const Ia = (i) => +i || 0;
function ws(i, t) {
  const e = {}, s = I(t), n = s ? Object.keys(t) : t, o = I(i) ? s ? (r) => O(i[r], i[t[r]]) : (r) => i[r] : () => i;
  for (const r of n)
    e[r] = Ia(o(r));
  return e;
}
function Ro(i) {
  return ws(i, {
    top: "y",
    right: "x",
    bottom: "y",
    left: "x"
  });
}
function Jt(i) {
  return ws(i, [
    "topLeft",
    "topRight",
    "bottomLeft",
    "bottomRight"
  ]);
}
function rt(i) {
  const t = Ro(i);
  return t.width = t.left + t.right, t.height = t.top + t.bottom, t;
}
function G(i, t) {
  i = i || {}, t = t || X.font;
  let e = O(i.size, t.size);
  typeof e == "string" && (e = parseInt(e, 10));
  let s = O(i.style, t.style);
  s && !("" + s).match(Ea) && (console.warn('Invalid font style specified: "' + s + '"'), s = void 0);
  const n = {
    family: O(i.family, t.family),
    lineHeight: Fa(O(i.lineHeight, t.lineHeight), e),
    size: e,
    style: s,
    weight: O(i.weight, t.weight),
    string: ""
  };
  return n.string = Pa(n), n;
}
function xe(i, t, e, s) {
  let n, o, r;
  for (n = 0, o = i.length; n < o; ++n)
    if (r = i[n], r !== void 0 && r !== void 0)
      return r;
}
function za(i, t, e) {
  const { min: s, max: n } = i, o = vo(t, (n - s) / 2), r = (a, l) => e && a === 0 ? 0 : a + l;
  return {
    min: r(s, -Math.abs(o)),
    max: r(n, o)
  };
}
function Ht(i, t) {
  return Object.assign(Object.create(i), t);
}
function Ss(i, t = [
  ""
], e, s, n = () => i[0]) {
  const o = e || i;
  typeof s > "u" && (s = zo("_fallback", i));
  const r = {
    [Symbol.toStringTag]: "Object",
    _cacheable: !0,
    _scopes: i,
    _rootScopes: o,
    _fallback: s,
    _getTarget: n,
    override: (a) => Ss([
      a,
      ...i
    ], t, o, s)
  };
  return new Proxy(r, {
    /**
    * A trap for the delete operator.
    */
    deleteProperty(a, l) {
      return delete a[l], delete a._keys, delete i[0][l], !0;
    },
    /**
    * A trap for getting property values.
    */
    get(a, l) {
      return Fo(a, l, () => Ya(l, t, i, a));
    },
    /**
    * A trap for Object.getOwnPropertyDescriptor.
    * Also used by Object.hasOwnProperty.
    */
    getOwnPropertyDescriptor(a, l) {
      return Reflect.getOwnPropertyDescriptor(a._scopes[0], l);
    },
    /**
    * A trap for Object.getPrototypeOf.
    */
    getPrototypeOf() {
      return Reflect.getPrototypeOf(i[0]);
    },
    /**
    * A trap for the in operator.
    */
    has(a, l) {
      return Js(a).includes(l);
    },
    /**
    * A trap for Object.getOwnPropertyNames and Object.getOwnPropertySymbols.
    */
    ownKeys(a) {
      return Js(a);
    },
    /**
    * A trap for setting property values.
    */
    set(a, l, c) {
      const h = a._storage || (a._storage = n());
      return a[l] = h[l] = c, delete a._keys, !0;
    }
  });
}
function de(i, t, e, s) {
  const n = {
    _cacheable: !1,
    _proxy: i,
    _context: t,
    _subProxy: e,
    _stack: /* @__PURE__ */ new Set(),
    _descriptors: Eo(i, s),
    setContext: (o) => de(i, o, e, s),
    override: (o) => de(i.override(o), t, e, s)
  };
  return new Proxy(n, {
    /**
    * A trap for the delete operator.
    */
    deleteProperty(o, r) {
      return delete o[r], delete i[r], !0;
    },
    /**
    * A trap for getting property values.
    */
    get(o, r, a) {
      return Fo(o, r, () => Va(o, r, a));
    },
    /**
    * A trap for Object.getOwnPropertyDescriptor.
    * Also used by Object.hasOwnProperty.
    */
    getOwnPropertyDescriptor(o, r) {
      return o._descriptors.allKeys ? Reflect.has(i, r) ? {
        enumerable: !0,
        configurable: !0
      } : void 0 : Reflect.getOwnPropertyDescriptor(i, r);
    },
    /**
    * A trap for Object.getPrototypeOf.
    */
    getPrototypeOf() {
      return Reflect.getPrototypeOf(i);
    },
    /**
    * A trap for the in operator.
    */
    has(o, r) {
      return Reflect.has(i, r);
    },
    /**
    * A trap for Object.getOwnPropertyNames and Object.getOwnPropertySymbols.
    */
    ownKeys() {
      return Reflect.ownKeys(i);
    },
    /**
    * A trap for setting property values.
    */
    set(o, r, a) {
      return i[r] = a, delete o[r], !0;
    }
  });
}
function Eo(i, t = {
  scriptable: !0,
  indexable: !0
}) {
  const { _scriptable: e = t.scriptable, _indexable: s = t.indexable, _allKeys: n = t.allKeys } = i;
  return {
    allKeys: n,
    scriptable: e,
    indexable: s,
    isScriptable: Nt(e) ? e : () => e,
    isIndexable: Nt(s) ? s : () => s
  };
}
const Ba = (i, t) => i ? i + xs(t) : t, Ps = (i, t) => I(t) && i !== "adapters" && (Object.getPrototypeOf(t) === null || t.constructor === Object);
function Fo(i, t, e) {
  if (Object.prototype.hasOwnProperty.call(i, t) || t === "constructor")
    return i[t];
  const s = e();
  return i[t] = s, s;
}
function Va(i, t, e) {
  const { _proxy: s, _context: n, _subProxy: o, _descriptors: r } = i;
  let a = s[t];
  return Nt(a) && r.isScriptable(t) && (a = Wa(t, a, i, e)), Y(a) && a.length && (a = Na(t, a, i, r.isIndexable)), Ps(t, a) && (a = de(a, n, o && o[t], r)), a;
}
function Wa(i, t, e, s) {
  const { _proxy: n, _context: o, _subProxy: r, _stack: a } = e;
  if (a.has(i))
    throw new Error("Recursion detected: " + Array.from(a).join("->") + "->" + i);
  a.add(i);
  let l = t(o, r || s);
  return a.delete(i), Ps(i, l) && (l = Ds(n._scopes, n, i, l)), l;
}
function Na(i, t, e, s) {
  const { _proxy: n, _context: o, _subProxy: r, _descriptors: a } = e;
  if (typeof o.index < "u" && s(i))
    return t[o.index % t.length];
  if (I(t[0])) {
    const l = t, c = n._scopes.filter((h) => h !== l);
    t = [];
    for (const h of l) {
      const d = Ds(c, n, i, h);
      t.push(de(d, o, r && r[i], a));
    }
  }
  return t;
}
function Io(i, t, e) {
  return Nt(i) ? i(t, e) : i;
}
const Ha = (i, t) => i === !0 ? t : typeof i == "string" ? Wt(t, i) : void 0;
function ja(i, t, e, s, n) {
  for (const o of t) {
    const r = Ha(e, o);
    if (r) {
      i.add(r);
      const a = Io(r._fallback, e, n);
      if (typeof a < "u" && a !== e && a !== s)
        return a;
    } else if (r === !1 && typeof s < "u" && e !== s)
      return null;
  }
  return !1;
}
function Ds(i, t, e, s) {
  const n = t._rootScopes, o = Io(t._fallback, e, s), r = [
    ...i,
    ...n
  ], a = /* @__PURE__ */ new Set();
  a.add(s);
  let l = Zs(a, r, e, o || e, s);
  return l === null || typeof o < "u" && o !== e && (l = Zs(a, r, o, l, s), l === null) ? !1 : Ss(Array.from(a), [
    ""
  ], n, o, () => $a(t, e, s));
}
function Zs(i, t, e, s, n) {
  for (; e; )
    e = ja(i, t, e, s, n);
  return e;
}
function $a(i, t, e) {
  const s = i._getTarget();
  t in s || (s[t] = {});
  const n = s[t];
  return Y(n) && I(e) ? e : n || {};
}
function Ya(i, t, e, s) {
  let n;
  for (const o of t)
    if (n = zo(Ba(o, i), e), typeof n < "u")
      return Ps(i, n) ? Ds(e, s, i, n) : n;
}
function zo(i, t) {
  for (const e of t) {
    if (!e)
      continue;
    const s = e[i];
    if (typeof s < "u")
      return s;
  }
}
function Js(i) {
  let t = i._keys;
  return t || (t = i._keys = Xa(i._scopes)), t;
}
function Xa(i) {
  const t = /* @__PURE__ */ new Set();
  for (const e of i)
    for (const s of Object.keys(e).filter((n) => !n.startsWith("_")))
      t.add(s);
  return Array.from(t);
}
function Bo(i, t, e, s) {
  const { iScale: n } = i, { key: o = "r" } = this._parsing, r = new Array(s);
  let a, l, c, h;
  for (a = 0, l = s; a < l; ++a)
    c = a + e, h = t[c], r[a] = {
      r: n.parse(Wt(h, o), c)
    };
  return r;
}
const Ua = Number.EPSILON || 1e-14, fe = (i, t) => t < i.length && !i[t].skip && i[t], Vo = (i) => i === "x" ? "y" : "x";
function Ka(i, t, e, s) {
  const n = i.skip ? t : i, o = t, r = e.skip ? t : e, a = ts(o, n), l = ts(r, o);
  let c = a / (a + l), h = l / (a + l);
  c = isNaN(c) ? 0 : c, h = isNaN(h) ? 0 : h;
  const d = s * c, f = s * h;
  return {
    previous: {
      x: o.x - d * (r.x - n.x),
      y: o.y - d * (r.y - n.y)
    },
    next: {
      x: o.x + f * (r.x - n.x),
      y: o.y + f * (r.y - n.y)
    }
  };
}
function qa(i, t, e) {
  const s = i.length;
  let n, o, r, a, l, c = fe(i, 0);
  for (let h = 0; h < s - 1; ++h)
    if (l = c, c = fe(i, h + 1), !(!l || !c)) {
      if (De(t[h], 0, Ua)) {
        e[h] = e[h + 1] = 0;
        continue;
      }
      n = e[h] / t[h], o = e[h + 1] / t[h], a = Math.pow(n, 2) + Math.pow(o, 2), !(a <= 9) && (r = 3 / Math.sqrt(a), e[h] = n * r * t[h], e[h + 1] = o * r * t[h]);
    }
}
function Ga(i, t, e = "x") {
  const s = Vo(e), n = i.length;
  let o, r, a, l = fe(i, 0);
  for (let c = 0; c < n; ++c) {
    if (r = a, a = l, l = fe(i, c + 1), !a)
      continue;
    const h = a[e], d = a[s];
    r && (o = (h - r[e]) / 3, a[`cp1${e}`] = h - o, a[`cp1${s}`] = d - o * t[c]), l && (o = (l[e] - h) / 3, a[`cp2${e}`] = h + o, a[`cp2${s}`] = d + o * t[c]);
  }
}
function Za(i, t = "x") {
  const e = Vo(t), s = i.length, n = Array(s).fill(0), o = Array(s);
  let r, a, l, c = fe(i, 0);
  for (r = 0; r < s; ++r)
    if (a = l, l = c, c = fe(i, r + 1), !!l) {
      if (c) {
        const h = c[t] - l[t];
        n[r] = h !== 0 ? (c[e] - l[e]) / h : 0;
      }
      o[r] = a ? c ? kt(n[r - 1]) !== kt(n[r]) ? 0 : (n[r - 1] + n[r]) / 2 : n[r - 1] : n[r];
    }
  qa(i, n, o), Ga(i, o, t);
}
function Ue(i, t, e) {
  return Math.max(Math.min(i, e), t);
}
function Ja(i, t) {
  let e, s, n, o, r, a = Tt(i[0], t);
  for (e = 0, s = i.length; e < s; ++e)
    r = o, o = a, a = e < s - 1 && Tt(i[e + 1], t), o && (n = i[e], r && (n.cp1x = Ue(n.cp1x, t.left, t.right), n.cp1y = Ue(n.cp1y, t.top, t.bottom)), a && (n.cp2x = Ue(n.cp2x, t.left, t.right), n.cp2y = Ue(n.cp2y, t.top, t.bottom)));
}
function Qa(i, t, e, s, n) {
  let o, r, a, l;
  if (t.spanGaps && (i = i.filter((c) => !c.skip)), t.cubicInterpolationMode === "monotone")
    Za(i, n);
  else {
    let c = s ? i[i.length - 1] : i[0];
    for (o = 0, r = i.length; o < r; ++o)
      a = i[o], l = Ka(c, a, i[Math.min(o + 1, r - (s ? 0 : 1)) % r], t.tension), a.cp1x = l.previous.x, a.cp1y = l.previous.y, a.cp2x = l.next.x, a.cp2y = l.next.y, c = a;
  }
  t.capBezierPoints && Ja(i, e);
}
function As() {
  return typeof window < "u" && typeof document < "u";
}
function Cs(i) {
  let t = i.parentNode;
  return t && t.toString() === "[object ShadowRoot]" && (t = t.host), t;
}
function xi(i, t, e) {
  let s;
  return typeof i == "string" ? (s = parseInt(i, 10), i.indexOf("%") !== -1 && (s = s / 100 * t.parentNode[e])) : s = i, s;
}
const Ci = (i) => i.ownerDocument.defaultView.getComputedStyle(i, null);
function tl(i, t) {
  return Ci(i).getPropertyValue(t);
}
const el = [
  "top",
  "right",
  "bottom",
  "left"
];
function Qt(i, t, e) {
  const s = {};
  e = e ? "-" + e : "";
  for (let n = 0; n < 4; n++) {
    const o = el[n];
    s[o] = parseFloat(i[t + "-" + o + e]) || 0;
  }
  return s.width = s.left + s.right, s.height = s.top + s.bottom, s;
}
const il = (i, t, e) => (i > 0 || t > 0) && (!e || !e.shadowRoot);
function sl(i, t) {
  const e = i.touches, s = e && e.length ? e[0] : i, { offsetX: n, offsetY: o } = s;
  let r = !1, a, l;
  if (il(n, o, i.target))
    a = n, l = o;
  else {
    const c = t.getBoundingClientRect();
    a = s.clientX - c.left, l = s.clientY - c.top, r = !0;
  }
  return {
    x: a,
    y: l,
    box: r
  };
}
function Kt(i, t) {
  if ("native" in i)
    return i;
  const { canvas: e, currentDevicePixelRatio: s } = t, n = Ci(e), o = n.boxSizing === "border-box", r = Qt(n, "padding"), a = Qt(n, "border", "width"), { x: l, y: c, box: h } = sl(i, e), d = r.left + (h && a.left), f = r.top + (h && a.top);
  let { width: u, height: g } = t;
  return o && (u -= r.width + a.width, g -= r.height + a.height), {
    x: Math.round((l - d) / u * e.width / s),
    y: Math.round((c - f) / g * e.height / s)
  };
}
function nl(i, t, e) {
  let s, n;
  if (t === void 0 || e === void 0) {
    const o = i && Cs(i);
    if (!o)
      t = i.clientWidth, e = i.clientHeight;
    else {
      const r = o.getBoundingClientRect(), a = Ci(o), l = Qt(a, "border", "width"), c = Qt(a, "padding");
      t = r.width - c.width - l.width, e = r.height - c.height - l.height, s = xi(a.maxWidth, o, "clientWidth"), n = xi(a.maxHeight, o, "clientHeight");
    }
  }
  return {
    width: t,
    height: e,
    maxWidth: s || bi,
    maxHeight: n || bi
  };
}
const It = (i) => Math.round(i * 10) / 10;
function ol(i, t, e, s) {
  const n = Ci(i), o = Qt(n, "margin"), r = xi(n.maxWidth, i, "clientWidth") || bi, a = xi(n.maxHeight, i, "clientHeight") || bi, l = nl(i, t, e);
  let { width: c, height: h } = l;
  if (n.boxSizing === "content-box") {
    const f = Qt(n, "border", "width"), u = Qt(n, "padding");
    c -= u.width + f.width, h -= u.height + f.height;
  }
  return c = Math.max(0, c - o.width), h = Math.max(0, s ? c / s : h - o.height), c = It(Math.min(c, r, l.maxWidth)), h = It(Math.min(h, a, l.maxHeight)), c && !h && (h = It(c / 2)), (t !== void 0 || e !== void 0) && s && l.height && h > l.height && (h = l.height, c = It(Math.floor(h * s))), {
    width: c,
    height: h
  };
}
function Qs(i, t, e) {
  const s = t || 1, n = It(i.height * s), o = It(i.width * s);
  i.height = It(i.height), i.width = It(i.width);
  const r = i.canvas;
  return r.style && (e || !r.style.height && !r.style.width) && (r.style.height = `${i.height}px`, r.style.width = `${i.width}px`), i.currentDevicePixelRatio !== s || r.height !== n || r.width !== o ? (i.currentDevicePixelRatio = s, r.height = n, r.width = o, i.ctx.setTransform(s, 0, 0, s, 0, 0), !0) : !1;
}
const rl = function() {
  let i = !1;
  try {
    const t = {
      get passive() {
        return i = !0, !1;
      }
    };
    As() && (window.addEventListener("test", null, t), window.removeEventListener("test", null, t));
  } catch {
  }
  return i;
}();
function tn(i, t) {
  const e = tl(i, t), s = e && e.match(/^(\d+)(\.\d+)?px$/);
  return s ? +s[1] : void 0;
}
function qt(i, t, e, s) {
  return {
    x: i.x + e * (t.x - i.x),
    y: i.y + e * (t.y - i.y)
  };
}
function al(i, t, e, s) {
  return {
    x: i.x + e * (t.x - i.x),
    y: s === "middle" ? e < 0.5 ? i.y : t.y : s === "after" ? e < 1 ? i.y : t.y : e > 0 ? t.y : i.y
  };
}
function ll(i, t, e, s) {
  const n = {
    x: i.cp2x,
    y: i.cp2y
  }, o = {
    x: t.cp1x,
    y: t.cp1y
  }, r = qt(i, n, e), a = qt(n, o, e), l = qt(o, t, e), c = qt(r, a, e), h = qt(a, l, e);
  return qt(c, h, e);
}
const cl = function(i, t) {
  return {
    x(e) {
      return i + i + t - e;
    },
    setWidth(e) {
      t = e;
    },
    textAlign(e) {
      return e === "center" ? e : e === "right" ? "left" : "right";
    },
    xPlus(e, s) {
      return e - s;
    },
    leftForLtr(e, s) {
      return e - s;
    }
  };
}, hl = function() {
  return {
    x(i) {
      return i;
    },
    setWidth(i) {
    },
    textAlign(i) {
      return i;
    },
    xPlus(i, t) {
      return i + t;
    },
    leftForLtr(i, t) {
      return i;
    }
  };
};
function ce(i, t, e) {
  return i ? cl(t, e) : hl();
}
function Wo(i, t) {
  let e, s;
  (t === "ltr" || t === "rtl") && (e = i.canvas.style, s = [
    e.getPropertyValue("direction"),
    e.getPropertyPriority("direction")
  ], e.setProperty("direction", t, "important"), i.prevTextDirection = s);
}
function No(i, t) {
  t !== void 0 && (delete i.prevTextDirection, i.canvas.style.setProperty("direction", t[0], t[1]));
}
function Ho(i) {
  return i === "angle" ? {
    between: Fe,
    compare: da,
    normalize: nt
  } : {
    between: Ct,
    compare: (t, e) => t - e,
    normalize: (t) => t
  };
}
function en({ start: i, end: t, count: e, loop: s, style: n }) {
  return {
    start: i % e,
    end: t % e,
    loop: s && (t - i + 1) % e === 0,
    style: n
  };
}
function dl(i, t, e) {
  const { property: s, start: n, end: o } = e, { between: r, normalize: a } = Ho(s), l = t.length;
  let { start: c, end: h, loop: d } = i, f, u;
  if (d) {
    for (c += l, h += l, f = 0, u = l; f < u && r(a(t[c % l][s]), n, o); ++f)
      c--, h--;
    c %= l, h %= l;
  }
  return h < c && (h += l), {
    start: c,
    end: h,
    loop: d,
    style: i.style
  };
}
function jo(i, t, e) {
  if (!e)
    return [
      i
    ];
  const { property: s, start: n, end: o } = e, r = t.length, { compare: a, between: l, normalize: c } = Ho(s), { start: h, end: d, loop: f, style: u } = dl(i, t, e), g = [];
  let p = !1, m = null, b, _, y;
  const v = () => l(n, y, b) && a(n, y) !== 0, x = () => a(o, b) === 0 || l(o, y, b), S = () => p || v(), k = () => !p || x();
  for (let w = h, P = h; w <= d; ++w)
    _ = t[w % r], !_.skip && (b = c(_[s]), b !== y && (p = l(b, n, o), m === null && S() && (m = a(b, n) === 0 ? w : P), m !== null && k() && (g.push(en({
      start: m,
      end: w,
      loop: f,
      count: r,
      style: u
    })), m = null), P = w, y = b));
  return m !== null && g.push(en({
    start: m,
    end: d,
    loop: f,
    count: r,
    style: u
  })), g;
}
function $o(i, t) {
  const e = [], s = i.segments;
  for (let n = 0; n < s.length; n++) {
    const o = jo(s[n], i.points, t);
    o.length && e.push(...o);
  }
  return e;
}
function fl(i, t, e, s) {
  let n = 0, o = t - 1;
  if (e && !s)
    for (; n < t && !i[n].skip; )
      n++;
  for (; n < t && i[n].skip; )
    n++;
  for (n %= t, e && (o += n); o > n && i[o % t].skip; )
    o--;
  return o %= t, {
    start: n,
    end: o
  };
}
function ul(i, t, e, s) {
  const n = i.length, o = [];
  let r = t, a = i[t], l;
  for (l = t + 1; l <= e; ++l) {
    const c = i[l % n];
    c.skip || c.stop ? a.skip || (s = !1, o.push({
      start: t % n,
      end: (l - 1) % n,
      loop: s
    }), t = r = c.stop ? l : null) : (r = l, a.skip && (t = l)), a = c;
  }
  return r !== null && o.push({
    start: t % n,
    end: r % n,
    loop: s
  }), o;
}
function gl(i, t) {
  const e = i.points, s = i.options.spanGaps, n = e.length;
  if (!n)
    return [];
  const o = !!i._loop, { start: r, end: a } = fl(e, n, o, s);
  if (s === !0)
    return sn(i, [
      {
        start: r,
        end: a,
        loop: o
      }
    ], e, t);
  const l = a < r ? a + n : a, c = !!i._fullLoop && r === 0 && a === n - 1;
  return sn(i, ul(e, r, l, c), e, t);
}
function sn(i, t, e, s) {
  return !s || !s.setContext || !e ? t : pl(i, t, e, s);
}
function pl(i, t, e, s) {
  const n = i._chart.getContext(), o = nn(i.options), { _datasetIndex: r, options: { spanGaps: a } } = i, l = e.length, c = [];
  let h = o, d = t[0].start, f = d;
  function u(g, p, m, b) {
    const _ = a ? -1 : 1;
    if (g !== p) {
      for (g += l; e[g % l].skip; )
        g -= _;
      for (; e[p % l].skip; )
        p += _;
      g % l !== p % l && (c.push({
        start: g % l,
        end: p % l,
        loop: m,
        style: b
      }), h = b, d = p % l);
    }
  }
  for (const g of t) {
    d = a ? d : g.start;
    let p = e[d % l], m;
    for (f = d + 1; f <= g.end; f++) {
      const b = e[f % l];
      m = nn(s.setContext(Ht(n, {
        type: "segment",
        p0: p,
        p1: b,
        p0DataIndex: (f - 1) % l,
        p1DataIndex: f % l,
        datasetIndex: r
      }))), ml(m, h) && u(d, f - 1, g.loop, h), p = b, h = m;
    }
    d < f - 1 && u(d, f - 1, g.loop, h);
  }
  return c;
}
function nn(i) {
  return {
    backgroundColor: i.backgroundColor,
    borderCapStyle: i.borderCapStyle,
    borderDash: i.borderDash,
    borderDashOffset: i.borderDashOffset,
    borderJoinStyle: i.borderJoinStyle,
    borderWidth: i.borderWidth,
    borderColor: i.borderColor
  };
}
function ml(i, t) {
  if (!t)
    return !1;
  const e = [], s = function(n, o) {
    return Ms(o) ? (e.includes(o) || e.push(o), e.indexOf(o)) : o;
  };
  return JSON.stringify(i, s) !== JSON.stringify(t, s);
}
function Ke(i, t, e) {
  return i.options.clip ? i[e] : t[e];
}
function bl(i, t) {
  const { xScale: e, yScale: s } = i;
  return e && s ? {
    left: Ke(e, t, "left"),
    right: Ke(e, t, "right"),
    top: Ke(s, t, "top"),
    bottom: Ke(s, t, "bottom")
  } : t;
}
function Yo(i, t) {
  const e = t._clip;
  if (e.disabled)
    return !1;
  const s = bl(t, i.chartArea);
  return {
    left: e.left === !1 ? 0 : s.left - (e.left === !0 ? 0 : e.left),
    right: e.right === !1 ? i.width : s.right + (e.right === !0 ? 0 : e.right),
    top: e.top === !1 ? 0 : s.top - (e.top === !0 ? 0 : e.top),
    bottom: e.bottom === !1 ? i.height : s.bottom + (e.bottom === !0 ? 0 : e.bottom)
  };
}
/*!
 * Chart.js v4.5.1
 * https://www.chartjs.org
 * (c) 2025 Chart.js Contributors
 * Released under the MIT License
 */
class _l {
  constructor() {
    this._request = null, this._charts = /* @__PURE__ */ new Map(), this._running = !1, this._lastDate = void 0;
  }
  _notify(t, e, s, n) {
    const o = e.listeners[n], r = e.duration;
    o.forEach((a) => a({
      chart: t,
      initial: e.initial,
      numSteps: r,
      currentStep: Math.min(s - e.start, r)
    }));
  }
  _refresh() {
    this._request || (this._running = !0, this._request = Do.call(window, () => {
      this._update(), this._request = null, this._running && this._refresh();
    }));
  }
  _update(t = Date.now()) {
    let e = 0;
    this._charts.forEach((s, n) => {
      if (!s.running || !s.items.length)
        return;
      const o = s.items;
      let r = o.length - 1, a = !1, l;
      for (; r >= 0; --r)
        l = o[r], l._active ? (l._total > s.duration && (s.duration = l._total), l.tick(t), a = !0) : (o[r] = o[o.length - 1], o.pop());
      a && (n.draw(), this._notify(n, s, t, "progress")), o.length || (s.running = !1, this._notify(n, s, t, "complete"), s.initial = !1), e += o.length;
    }), this._lastDate = t, e === 0 && (this._running = !1);
  }
  _getAnims(t) {
    const e = this._charts;
    let s = e.get(t);
    return s || (s = {
      running: !1,
      initial: !0,
      items: [],
      listeners: {
        complete: [],
        progress: []
      }
    }, e.set(t, s)), s;
  }
  listen(t, e, s) {
    this._getAnims(t).listeners[e].push(s);
  }
  add(t, e) {
    !e || !e.length || this._getAnims(t).items.push(...e);
  }
  has(t) {
    return this._getAnims(t).items.length > 0;
  }
  start(t) {
    const e = this._charts.get(t);
    e && (e.running = !0, e.start = Date.now(), e.duration = e.items.reduce((s, n) => Math.max(s, n._duration), 0), this._refresh());
  }
  running(t) {
    if (!this._running)
      return !1;
    const e = this._charts.get(t);
    return !(!e || !e.running || !e.items.length);
  }
  stop(t) {
    const e = this._charts.get(t);
    if (!e || !e.items.length)
      return;
    const s = e.items;
    let n = s.length - 1;
    for (; n >= 0; --n)
      s[n].cancel();
    e.items = [], this._notify(t, e, Date.now(), "complete");
  }
  remove(t) {
    return this._charts.delete(t);
  }
}
var St = /* @__PURE__ */ new _l();
const on = "transparent", xl = {
  boolean(i, t, e) {
    return e > 0.5 ? t : i;
  },
  color(i, t, e) {
    const s = Ks(i || on), n = s.valid && Ks(t || on);
    return n && n.valid ? n.mix(s, e).hexString() : t;
  },
  number(i, t, e) {
    return i + (t - i) * e;
  }
};
class yl {
  constructor(t, e, s, n) {
    const o = e[s];
    n = xe([
      t.to,
      n,
      o,
      t.from
    ]);
    const r = xe([
      t.from,
      o,
      n
    ]);
    this._active = !0, this._fn = t.fn || xl[t.type || typeof r], this._easing = Ae[t.easing] || Ae.linear, this._start = Math.floor(Date.now() + (t.delay || 0)), this._duration = this._total = Math.floor(t.duration), this._loop = !!t.loop, this._target = e, this._prop = s, this._from = r, this._to = n, this._promises = void 0;
  }
  active() {
    return this._active;
  }
  update(t, e, s) {
    if (this._active) {
      this._notify(!1);
      const n = this._target[this._prop], o = s - this._start, r = this._duration - o;
      this._start = s, this._duration = Math.floor(Math.max(r, t.duration)), this._total += o, this._loop = !!t.loop, this._to = xe([
        t.to,
        e,
        n,
        t.from
      ]), this._from = xe([
        t.from,
        n,
        e
      ]);
    }
  }
  cancel() {
    this._active && (this.tick(Date.now()), this._active = !1, this._notify(!1));
  }
  tick(t) {
    const e = t - this._start, s = this._duration, n = this._prop, o = this._from, r = this._loop, a = this._to;
    let l;
    if (this._active = o !== a && (r || e < s), !this._active) {
      this._target[n] = a, this._notify(!0);
      return;
    }
    if (e < 0) {
      this._target[n] = o;
      return;
    }
    l = e / s % 2, l = r && l > 1 ? 2 - l : l, l = this._easing(Math.min(1, Math.max(0, l))), this._target[n] = this._fn(o, a, l);
  }
  wait() {
    const t = this._promises || (this._promises = []);
    return new Promise((e, s) => {
      t.push({
        res: e,
        rej: s
      });
    });
  }
  _notify(t) {
    const e = t ? "res" : "rej", s = this._promises || [];
    for (let n = 0; n < s.length; n++)
      s[n][e]();
  }
}
class Xo {
  constructor(t, e) {
    this._chart = t, this._properties = /* @__PURE__ */ new Map(), this.configure(e);
  }
  configure(t) {
    if (!I(t))
      return;
    const e = Object.keys(X.animation), s = this._properties;
    Object.getOwnPropertyNames(t).forEach((n) => {
      const o = t[n];
      if (!I(o))
        return;
      const r = {};
      for (const a of e)
        r[a] = o[a];
      (Y(o.properties) && o.properties || [
        n
      ]).forEach((a) => {
        (a === n || !s.has(a)) && s.set(a, r);
      });
    });
  }
  _animateOptions(t, e) {
    const s = e.options, n = kl(t, s);
    if (!n)
      return [];
    const o = this._createAnimations(n, s);
    return s.$shared && vl(t.options.$animations, s).then(() => {
      t.options = s;
    }, () => {
    }), o;
  }
  _createAnimations(t, e) {
    const s = this._properties, n = [], o = t.$animations || (t.$animations = {}), r = Object.keys(e), a = Date.now();
    let l;
    for (l = r.length - 1; l >= 0; --l) {
      const c = r[l];
      if (c.charAt(0) === "$")
        continue;
      if (c === "options") {
        n.push(...this._animateOptions(t, e));
        continue;
      }
      const h = e[c];
      let d = o[c];
      const f = s.get(c);
      if (d)
        if (f && d.active()) {
          d.update(f, h, a);
          continue;
        } else
          d.cancel();
      if (!f || !f.duration) {
        t[c] = h;
        continue;
      }
      o[c] = d = new yl(f, t, c, h), n.push(d);
    }
    return n;
  }
  update(t, e) {
    if (this._properties.size === 0) {
      Object.assign(t, e);
      return;
    }
    const s = this._createAnimations(t, e);
    if (s.length)
      return St.add(this._chart, s), !0;
  }
}
function vl(i, t) {
  const e = [], s = Object.keys(t);
  for (let n = 0; n < s.length; n++) {
    const o = i[s[n]];
    o && o.active() && e.push(o.wait());
  }
  return Promise.all(e);
}
function kl(i, t) {
  if (!t)
    return;
  let e = i.options;
  if (!e) {
    i.options = t;
    return;
  }
  return e.$shared && (i.options = e = Object.assign({}, e, {
    $shared: !1,
    $animations: {}
  })), e;
}
function rn(i, t) {
  const e = i && i.options || {}, s = e.reverse, n = e.min === void 0 ? t : 0, o = e.max === void 0 ? t : 0;
  return {
    start: s ? o : n,
    end: s ? n : o
  };
}
function Ml(i, t, e) {
  if (e === !1)
    return !1;
  const s = rn(i, e), n = rn(t, e);
  return {
    top: n.end,
    right: s.end,
    bottom: n.start,
    left: s.start
  };
}
function wl(i) {
  let t, e, s, n;
  return I(i) ? (t = i.top, e = i.right, s = i.bottom, n = i.left) : t = e = s = n = i, {
    top: t,
    right: e,
    bottom: s,
    left: n,
    disabled: i === !1
  };
}
function Uo(i, t) {
  const e = [], s = i._getSortedDatasetMetas(t);
  let n, o;
  for (n = 0, o = s.length; n < o; ++n)
    e.push(s[n].index);
  return e;
}
function an(i, t, e, s = {}) {
  const n = i.keys, o = s.mode === "single";
  let r, a, l, c;
  if (t === null)
    return;
  let h = !1;
  for (r = 0, a = n.length; r < a; ++r) {
    if (l = +n[r], l === e) {
      if (h = !0, s.all)
        continue;
      break;
    }
    c = i.values[l], U(c) && (o || t === 0 || kt(t) === kt(c)) && (t += c);
  }
  return !h && !s.all ? 0 : t;
}
function Sl(i, t) {
  const { iScale: e, vScale: s } = t, n = e.axis === "x" ? "x" : "y", o = s.axis === "x" ? "x" : "y", r = Object.keys(i), a = new Array(r.length);
  let l, c, h;
  for (l = 0, c = r.length; l < c; ++l)
    h = r[l], a[l] = {
      [n]: h,
      [o]: i[h]
    };
  return a;
}
function Vi(i, t) {
  const e = i && i.options.stacked;
  return e || e === void 0 && t.stack !== void 0;
}
function Pl(i, t, e) {
  return `${i.id}.${t.id}.${e.stack || e.type}`;
}
function Dl(i) {
  const { min: t, max: e, minDefined: s, maxDefined: n } = i.getUserBounds();
  return {
    min: s ? t : Number.NEGATIVE_INFINITY,
    max: n ? e : Number.POSITIVE_INFINITY
  };
}
function Al(i, t, e) {
  const s = i[t] || (i[t] = {});
  return s[e] || (s[e] = {});
}
function ln(i, t, e, s) {
  for (const n of t.getMatchingVisibleMetas(s).reverse()) {
    const o = i[n.index];
    if (e && o > 0 || !e && o < 0)
      return n.index;
  }
  return null;
}
function cn(i, t) {
  const { chart: e, _cachedMeta: s } = i, n = e._stacks || (e._stacks = {}), { iScale: o, vScale: r, index: a } = s, l = o.axis, c = r.axis, h = Pl(o, r, s), d = t.length;
  let f;
  for (let u = 0; u < d; ++u) {
    const g = t[u], { [l]: p, [c]: m } = g, b = g._stacks || (g._stacks = {});
    f = b[c] = Al(n, h, p), f[a] = m, f._top = ln(f, r, !0, s.type), f._bottom = ln(f, r, !1, s.type);
    const _ = f._visualValues || (f._visualValues = {});
    _[a] = m;
  }
}
function Wi(i, t) {
  const e = i.scales;
  return Object.keys(e).filter((s) => e[s].axis === t).shift();
}
function Cl(i, t) {
  return Ht(i, {
    active: !1,
    dataset: void 0,
    datasetIndex: t,
    index: t,
    mode: "default",
    type: "dataset"
  });
}
function Ol(i, t, e) {
  return Ht(i, {
    active: !1,
    dataIndex: t,
    parsed: void 0,
    raw: void 0,
    element: e,
    index: t,
    mode: "default",
    type: "data"
  });
}
function ue(i, t) {
  const e = i.controller.index, s = i.vScale && i.vScale.axis;
  if (s) {
    t = t || i._parsed;
    for (const n of t) {
      const o = n._stacks;
      if (!o || o[s] === void 0 || o[s][e] === void 0)
        return;
      delete o[s][e], o[s]._visualValues !== void 0 && o[s]._visualValues[e] !== void 0 && delete o[s]._visualValues[e];
    }
  }
}
const Ni = (i) => i === "reset" || i === "none", hn = (i, t) => t ? i : Object.assign({}, i), Tl = (i, t, e) => i && !t.hidden && t._stacked && {
  keys: Uo(e, !0),
  values: null
};
class pt {
  constructor(t, e) {
    this.chart = t, this._ctx = t.ctx, this.index = e, this._cachedDataOpts = {}, this._cachedMeta = this.getMeta(), this._type = this._cachedMeta.type, this.options = void 0, this._parsing = !1, this._data = void 0, this._objectData = void 0, this._sharedOptions = void 0, this._drawStart = void 0, this._drawCount = void 0, this.enableOptionSharing = !1, this.supportsDecimation = !1, this.$context = void 0, this._syncList = [], this.datasetElementType = new.target.datasetElementType, this.dataElementType = new.target.dataElementType, this.initialize();
  }
  initialize() {
    const t = this._cachedMeta;
    this.configure(), this.linkScales(), t._stacked = Vi(t.vScale, t), this.addElements(), this.options.fill && !this.chart.isPluginEnabled("filler") && console.warn("Tried to use the 'fill' option without the 'Filler' plugin enabled. Please import and register the 'Filler' plugin and make sure it is not disabled in the options");
  }
  updateIndex(t) {
    this.index !== t && ue(this._cachedMeta), this.index = t;
  }
  linkScales() {
    const t = this.chart, e = this._cachedMeta, s = this.getDataset(), n = (d, f, u, g) => d === "x" ? f : d === "r" ? g : u, o = e.xAxisID = O(s.xAxisID, Wi(t, "x")), r = e.yAxisID = O(s.yAxisID, Wi(t, "y")), a = e.rAxisID = O(s.rAxisID, Wi(t, "r")), l = e.indexAxis, c = e.iAxisID = n(l, o, r, a), h = e.vAxisID = n(l, r, o, a);
    e.xScale = this.getScaleForId(o), e.yScale = this.getScaleForId(r), e.rScale = this.getScaleForId(a), e.iScale = this.getScaleForId(c), e.vScale = this.getScaleForId(h);
  }
  getDataset() {
    return this.chart.data.datasets[this.index];
  }
  getMeta() {
    return this.chart.getDatasetMeta(this.index);
  }
  getScaleForId(t) {
    return this.chart.scales[t];
  }
  _getOtherScale(t) {
    const e = this._cachedMeta;
    return t === e.iScale ? e.vScale : e.iScale;
  }
  reset() {
    this._update("reset");
  }
  _destroy() {
    const t = this._cachedMeta;
    this._data && Ys(this._data, this), t._stacked && ue(t);
  }
  _dataCheck() {
    const t = this.getDataset(), e = t.data || (t.data = []), s = this._data;
    if (I(e)) {
      const n = this._cachedMeta;
      this._data = Sl(e, n);
    } else if (s !== e) {
      if (s) {
        Ys(s, this);
        const n = this._cachedMeta;
        ue(n), n._parsed = [];
      }
      e && Object.isExtensible(e) && pa(e, this), this._syncList = [], this._data = e;
    }
  }
  addElements() {
    const t = this._cachedMeta;
    this._dataCheck(), this.datasetElementType && (t.dataset = new this.datasetElementType());
  }
  buildOrUpdateElements(t) {
    const e = this._cachedMeta, s = this.getDataset();
    let n = !1;
    this._dataCheck();
    const o = e._stacked;
    e._stacked = Vi(e.vScale, e), e.stack !== s.stack && (n = !0, ue(e), e.stack = s.stack), this._resyncElements(t), (n || o !== e._stacked) && (cn(this, e._parsed), e._stacked = Vi(e.vScale, e));
  }
  configure() {
    const t = this.chart.config, e = t.datasetScopeKeys(this._type), s = t.getOptionScopes(this.getDataset(), e, !0);
    this.options = t.createResolver(s, this.getContext()), this._parsing = this.options.parsing, this._cachedDataOpts = {};
  }
  parse(t, e) {
    const { _cachedMeta: s, _data: n } = this, { iScale: o, _stacked: r } = s, a = o.axis;
    let l = t === 0 && e === n.length ? !0 : s._sorted, c = t > 0 && s._parsed[t - 1], h, d, f;
    if (this._parsing === !1)
      s._parsed = n, s._sorted = !0, f = n;
    else {
      Y(n[t]) ? f = this.parseArrayData(s, n, t, e) : I(n[t]) ? f = this.parseObjectData(s, n, t, e) : f = this.parsePrimitiveData(s, n, t, e);
      const u = () => d[a] === null || c && d[a] < c[a];
      for (h = 0; h < e; ++h)
        s._parsed[h + t] = d = f[h], l && (u() && (l = !1), c = d);
      s._sorted = l;
    }
    r && cn(this, f);
  }
  parsePrimitiveData(t, e, s, n) {
    const { iScale: o, vScale: r } = t, a = o.axis, l = r.axis, c = o.getLabels(), h = o === r, d = new Array(n);
    let f, u, g;
    for (f = 0, u = n; f < u; ++f)
      g = f + s, d[f] = {
        [a]: h || o.parse(c[g], g),
        [l]: r.parse(e[g], g)
      };
    return d;
  }
  parseArrayData(t, e, s, n) {
    const { xScale: o, yScale: r } = t, a = new Array(n);
    let l, c, h, d;
    for (l = 0, c = n; l < c; ++l)
      h = l + s, d = e[h], a[l] = {
        x: o.parse(d[0], h),
        y: r.parse(d[1], h)
      };
    return a;
  }
  parseObjectData(t, e, s, n) {
    const { xScale: o, yScale: r } = t, { xAxisKey: a = "x", yAxisKey: l = "y" } = this._parsing, c = new Array(n);
    let h, d, f, u;
    for (h = 0, d = n; h < d; ++h)
      f = h + s, u = e[f], c[h] = {
        x: o.parse(Wt(u, a), f),
        y: r.parse(Wt(u, l), f)
      };
    return c;
  }
  getParsed(t) {
    return this._cachedMeta._parsed[t];
  }
  getDataElement(t) {
    return this._cachedMeta.data[t];
  }
  applyStack(t, e, s) {
    const n = this.chart, o = this._cachedMeta, r = e[t.axis], a = {
      keys: Uo(n, !0),
      values: e._stacks[t.axis]._visualValues
    };
    return an(a, r, o.index, {
      mode: s
    });
  }
  updateRangeFromParsed(t, e, s, n) {
    const o = s[e.axis];
    let r = o === null ? NaN : o;
    const a = n && s._stacks[e.axis];
    n && a && (n.values = a, r = an(n, o, this._cachedMeta.index)), t.min = Math.min(t.min, r), t.max = Math.max(t.max, r);
  }
  getMinMax(t, e) {
    const s = this._cachedMeta, n = s._parsed, o = s._sorted && t === s.iScale, r = n.length, a = this._getOtherScale(t), l = Tl(e, s, this.chart), c = {
      min: Number.POSITIVE_INFINITY,
      max: Number.NEGATIVE_INFINITY
    }, { min: h, max: d } = Dl(a);
    let f, u;
    function g() {
      u = n[f];
      const p = u[a.axis];
      return !U(u[t.axis]) || h > p || d < p;
    }
    for (f = 0; f < r && !(!g() && (this.updateRangeFromParsed(c, t, u, l), o)); ++f)
      ;
    if (o) {
      for (f = r - 1; f >= 0; --f)
        if (!g()) {
          this.updateRangeFromParsed(c, t, u, l);
          break;
        }
    }
    return c;
  }
  getAllParsedValues(t) {
    const e = this._cachedMeta._parsed, s = [];
    let n, o, r;
    for (n = 0, o = e.length; n < o; ++n)
      r = e[n][t.axis], U(r) && s.push(r);
    return s;
  }
  getMaxOverflow() {
    return !1;
  }
  getLabelAndValue(t) {
    const e = this._cachedMeta, s = e.iScale, n = e.vScale, o = this.getParsed(t);
    return {
      label: s ? "" + s.getLabelForValue(o[s.axis]) : "",
      value: n ? "" + n.getLabelForValue(o[n.axis]) : ""
    };
  }
  _update(t) {
    const e = this._cachedMeta;
    this.update(t || "default"), e._clip = wl(O(this.options.clip, Ml(e.xScale, e.yScale, this.getMaxOverflow())));
  }
  update(t) {
  }
  draw() {
    const t = this._ctx, e = this.chart, s = this._cachedMeta, n = s.data || [], o = e.chartArea, r = [], a = this._drawStart || 0, l = this._drawCount || n.length - a, c = this.options.drawActiveElementsOnTop;
    let h;
    for (s.dataset && s.dataset.draw(t, o, a, l), h = a; h < a + l; ++h) {
      const d = n[h];
      d.hidden || (d.active && c ? r.push(d) : d.draw(t, o));
    }
    for (h = 0; h < r.length; ++h)
      r[h].draw(t, o);
  }
  getStyle(t, e) {
    const s = e ? "active" : "default";
    return t === void 0 && this._cachedMeta.dataset ? this.resolveDatasetElementOptions(s) : this.resolveDataElementOptions(t || 0, s);
  }
  getContext(t, e, s) {
    const n = this.getDataset();
    let o;
    if (t >= 0 && t < this._cachedMeta.data.length) {
      const r = this._cachedMeta.data[t];
      o = r.$context || (r.$context = Ol(this.getContext(), t, r)), o.parsed = this.getParsed(t), o.raw = n.data[t], o.index = o.dataIndex = t;
    } else
      o = this.$context || (this.$context = Cl(this.chart.getContext(), this.index)), o.dataset = n, o.index = o.datasetIndex = this.index;
    return o.active = !!e, o.mode = s, o;
  }
  resolveDatasetElementOptions(t) {
    return this._resolveElementOptions(this.datasetElementType.id, t);
  }
  resolveDataElementOptions(t, e) {
    return this._resolveElementOptions(this.dataElementType.id, e, t);
  }
  _resolveElementOptions(t, e = "default", s) {
    const n = e === "active", o = this._cachedDataOpts, r = t + "-" + e, a = o[r], l = this.enableOptionSharing && Ee(s);
    if (a)
      return hn(a, l);
    const c = this.chart.config, h = c.datasetElementScopeKeys(this._type, t), d = n ? [
      `${t}Hover`,
      "hover",
      t,
      ""
    ] : [
      t,
      ""
    ], f = c.getOptionScopes(this.getDataset(), h), u = Object.keys(X.elements[t]), g = () => this.getContext(s, n, e), p = c.resolveNamedOptions(f, u, g, d);
    return p.$shared && (p.$shared = l, o[r] = Object.freeze(hn(p, l))), p;
  }
  _resolveAnimations(t, e, s) {
    const n = this.chart, o = this._cachedDataOpts, r = `animation-${e}`, a = o[r];
    if (a)
      return a;
    let l;
    if (n.options.animation !== !1) {
      const h = this.chart.config, d = h.datasetAnimationScopeKeys(this._type, e), f = h.getOptionScopes(this.getDataset(), d);
      l = h.createResolver(f, this.getContext(t, s, e));
    }
    const c = new Xo(n, l && l.animations);
    return l && l._cacheable && (o[r] = Object.freeze(c)), c;
  }
  getSharedOptions(t) {
    if (t.$shared)
      return this._sharedOptions || (this._sharedOptions = Object.assign({}, t));
  }
  includeOptions(t, e) {
    return !e || Ni(t) || this.chart._animationsDisabled;
  }
  _getSharedOptions(t, e) {
    const s = this.resolveDataElementOptions(t, e), n = this._sharedOptions, o = this.getSharedOptions(s), r = this.includeOptions(e, o) || o !== n;
    return this.updateSharedOptions(o, e, s), {
      sharedOptions: o,
      includeOptions: r
    };
  }
  updateElement(t, e, s, n) {
    Ni(n) ? Object.assign(t, s) : this._resolveAnimations(e, n).update(t, s);
  }
  updateSharedOptions(t, e, s) {
    t && !Ni(e) && this._resolveAnimations(void 0, e).update(t, s);
  }
  _setStyle(t, e, s, n) {
    t.active = n;
    const o = this.getStyle(e, n);
    this._resolveAnimations(e, s, n).update(t, {
      options: !n && this.getSharedOptions(o) || o
    });
  }
  removeHoverStyle(t, e, s) {
    this._setStyle(t, s, "active", !1);
  }
  setHoverStyle(t, e, s) {
    this._setStyle(t, s, "active", !0);
  }
  _removeDatasetHoverStyle() {
    const t = this._cachedMeta.dataset;
    t && this._setStyle(t, void 0, "active", !1);
  }
  _setDatasetHoverStyle() {
    const t = this._cachedMeta.dataset;
    t && this._setStyle(t, void 0, "active", !0);
  }
  _resyncElements(t) {
    const e = this._data, s = this._cachedMeta.data;
    for (const [a, l, c] of this._syncList)
      this[a](l, c);
    this._syncList = [];
    const n = s.length, o = e.length, r = Math.min(o, n);
    r && this.parse(0, r), o > n ? this._insertElements(n, o - n, t) : o < n && this._removeElements(o, n - o);
  }
  _insertElements(t, e, s = !0) {
    const n = this._cachedMeta, o = n.data, r = t + e;
    let a;
    const l = (c) => {
      for (c.length += e, a = c.length - 1; a >= r; a--)
        c[a] = c[a - e];
    };
    for (l(o), a = t; a < r; ++a)
      o[a] = new this.dataElementType();
    this._parsing && l(n._parsed), this.parse(t, e), s && this.updateElements(o, t, e, "reset");
  }
  updateElements(t, e, s, n) {
  }
  _removeElements(t, e) {
    const s = this._cachedMeta;
    if (this._parsing) {
      const n = s._parsed.splice(t, e);
      s._stacked && ue(s, n);
    }
    s.data.splice(t, e);
  }
  _sync(t) {
    if (this._parsing)
      this._syncList.push(t);
    else {
      const [e, s, n] = t;
      this[e](s, n);
    }
    this.chart._dataChanges.push([
      this.index,
      ...t
    ]);
  }
  _onDataPush() {
    const t = arguments.length;
    this._sync([
      "_insertElements",
      this.getDataset().data.length - t,
      t
    ]);
  }
  _onDataPop() {
    this._sync([
      "_removeElements",
      this._cachedMeta.data.length - 1,
      1
    ]);
  }
  _onDataShift() {
    this._sync([
      "_removeElements",
      0,
      1
    ]);
  }
  _onDataSplice(t, e) {
    e && this._sync([
      "_removeElements",
      t,
      e
    ]);
    const s = arguments.length - 2;
    s && this._sync([
      "_insertElements",
      t,
      s
    ]);
  }
  _onDataUnshift() {
    this._sync([
      "_insertElements",
      0,
      arguments.length
    ]);
  }
}
M(pt, "defaults", {}), M(pt, "datasetElementType", null), M(pt, "dataElementType", null);
function Ll(i, t) {
  if (!i._cache.$bar) {
    const e = i.getMatchingVisibleMetas(t);
    let s = [];
    for (let n = 0, o = e.length; n < o; n++)
      s = s.concat(e[n].controller.getAllParsedValues(i));
    i._cache.$bar = Po(s.sort((n, o) => n - o));
  }
  return i._cache.$bar;
}
function Rl(i) {
  const t = i.iScale, e = Ll(t, i.type);
  let s = t._length, n, o, r, a;
  const l = () => {
    r === 32767 || r === -32768 || (Ee(a) && (s = Math.min(s, Math.abs(r - a) || s)), a = r);
  };
  for (n = 0, o = e.length; n < o; ++n)
    r = t.getPixelForValue(e[n]), l();
  for (a = void 0, n = 0, o = t.ticks.length; n < o; ++n)
    r = t.getPixelForTick(n), l();
  return s;
}
function El(i, t, e, s) {
  const n = e.barThickness;
  let o, r;
  return F(n) ? (o = t.min * e.categoryPercentage, r = e.barPercentage) : (o = n * s, r = 1), {
    chunk: o / s,
    ratio: r,
    start: t.pixels[i] - o / 2
  };
}
function Fl(i, t, e, s) {
  const n = t.pixels, o = n[i];
  let r = i > 0 ? n[i - 1] : null, a = i < n.length - 1 ? n[i + 1] : null;
  const l = e.categoryPercentage;
  r === null && (r = o - (a === null ? t.end - t.start : a - o)), a === null && (a = o + o - r);
  const c = o - (o - Math.min(r, a)) / 2 * l;
  return {
    chunk: Math.abs(a - r) / 2 * l / s,
    ratio: e.barPercentage,
    start: c
  };
}
function Il(i, t, e, s) {
  const n = e.parse(i[0], s), o = e.parse(i[1], s), r = Math.min(n, o), a = Math.max(n, o);
  let l = r, c = a;
  Math.abs(r) > Math.abs(a) && (l = a, c = r), t[e.axis] = c, t._custom = {
    barStart: l,
    barEnd: c,
    start: n,
    end: o,
    min: r,
    max: a
  };
}
function Ko(i, t, e, s) {
  return Y(i) ? Il(i, t, e, s) : t[e.axis] = e.parse(i, s), t;
}
function dn(i, t, e, s) {
  const n = i.iScale, o = i.vScale, r = n.getLabels(), a = n === o, l = [];
  let c, h, d, f;
  for (c = e, h = e + s; c < h; ++c)
    f = t[c], d = {}, d[n.axis] = a || n.parse(r[c], c), l.push(Ko(f, d, o, c));
  return l;
}
function Hi(i) {
  return i && i.barStart !== void 0 && i.barEnd !== void 0;
}
function zl(i, t, e) {
  return i !== 0 ? kt(i) : (t.isHorizontal() ? 1 : -1) * (t.min >= e ? 1 : -1);
}
function Bl(i) {
  let t, e, s, n, o;
  return i.horizontal ? (t = i.base > i.x, e = "left", s = "right") : (t = i.base < i.y, e = "bottom", s = "top"), t ? (n = "end", o = "start") : (n = "start", o = "end"), {
    start: e,
    end: s,
    reverse: t,
    top: n,
    bottom: o
  };
}
function Vl(i, t, e, s) {
  let n = t.borderSkipped;
  const o = {};
  if (!n) {
    i.borderSkipped = o;
    return;
  }
  if (n === !0) {
    i.borderSkipped = {
      top: !0,
      right: !0,
      bottom: !0,
      left: !0
    };
    return;
  }
  const { start: r, end: a, reverse: l, top: c, bottom: h } = Bl(i);
  n === "middle" && e && (i.enableBorderRadius = !0, (e._top || 0) === s ? n = c : (e._bottom || 0) === s ? n = h : (o[fn(h, r, a, l)] = !0, n = c)), o[fn(n, r, a, l)] = !0, i.borderSkipped = o;
}
function fn(i, t, e, s) {
  return s ? (i = Wl(i, t, e), i = un(i, e, t)) : i = un(i, t, e), i;
}
function Wl(i, t, e) {
  return i === t ? e : i === e ? t : i;
}
function un(i, t, e) {
  return i === "start" ? t : i === "end" ? e : i;
}
function Nl(i, { inflateAmount: t }, e) {
  i.inflateAmount = t === "auto" ? e === 1 ? 0.33 : 0 : t;
}
class ri extends pt {
  parsePrimitiveData(t, e, s, n) {
    return dn(t, e, s, n);
  }
  parseArrayData(t, e, s, n) {
    return dn(t, e, s, n);
  }
  parseObjectData(t, e, s, n) {
    const { iScale: o, vScale: r } = t, { xAxisKey: a = "x", yAxisKey: l = "y" } = this._parsing, c = o.axis === "x" ? a : l, h = r.axis === "x" ? a : l, d = [];
    let f, u, g, p;
    for (f = s, u = s + n; f < u; ++f)
      p = e[f], g = {}, g[o.axis] = o.parse(Wt(p, c), f), d.push(Ko(Wt(p, h), g, r, f));
    return d;
  }
  updateRangeFromParsed(t, e, s, n) {
    super.updateRangeFromParsed(t, e, s, n);
    const o = s._custom;
    o && e === this._cachedMeta.vScale && (t.min = Math.min(t.min, o.min), t.max = Math.max(t.max, o.max));
  }
  getMaxOverflow() {
    return 0;
  }
  getLabelAndValue(t) {
    const e = this._cachedMeta, { iScale: s, vScale: n } = e, o = this.getParsed(t), r = o._custom, a = Hi(r) ? "[" + r.start + ", " + r.end + "]" : "" + n.getLabelForValue(o[n.axis]);
    return {
      label: "" + s.getLabelForValue(o[s.axis]),
      value: a
    };
  }
  initialize() {
    this.enableOptionSharing = !0, super.initialize();
    const t = this._cachedMeta;
    t.stack = this.getDataset().stack;
  }
  update(t) {
    const e = this._cachedMeta;
    this.updateElements(e.data, 0, e.data.length, t);
  }
  updateElements(t, e, s, n) {
    const o = n === "reset", { index: r, _cachedMeta: { vScale: a } } = this, l = a.getBasePixel(), c = a.isHorizontal(), h = this._getRuler(), { sharedOptions: d, includeOptions: f } = this._getSharedOptions(e, n);
    for (let u = e; u < e + s; u++) {
      const g = this.getParsed(u), p = o || F(g[a.axis]) ? {
        base: l,
        head: l
      } : this._calculateBarValuePixels(u), m = this._calculateBarIndexPixels(u, h), b = (g._stacks || {})[a.axis], _ = {
        horizontal: c,
        base: p.base,
        enableBorderRadius: !b || Hi(g._custom) || r === b._top || r === b._bottom,
        x: c ? p.head : m.center,
        y: c ? m.center : p.head,
        height: c ? m.size : Math.abs(p.size),
        width: c ? Math.abs(p.size) : m.size
      };
      f && (_.options = d || this.resolveDataElementOptions(u, t[u].active ? "active" : n));
      const y = _.options || t[u].options;
      Vl(_, y, b, r), Nl(_, y, h.ratio), this.updateElement(t[u], u, _, n);
    }
  }
  _getStacks(t, e) {
    const { iScale: s } = this._cachedMeta, n = s.getMatchingVisibleMetas(this._type).filter((h) => h.controller.options.grouped), o = s.options.stacked, r = [], a = this._cachedMeta.controller.getParsed(e), l = a && a[s.axis], c = (h) => {
      const d = h._parsed.find((u) => u[s.axis] === l), f = d && d[h.vScale.axis];
      if (F(f) || isNaN(f))
        return !0;
    };
    for (const h of n)
      if (!(e !== void 0 && c(h)) && ((o === !1 || r.indexOf(h.stack) === -1 || o === void 0 && h.stack === void 0) && r.push(h.stack), h.index === t))
        break;
    return r.length || r.push(void 0), r;
  }
  _getStackCount(t) {
    return this._getStacks(void 0, t).length;
  }
  _getAxisCount() {
    return this._getAxis().length;
  }
  getFirstScaleIdForIndexAxis() {
    const t = this.chart.scales, e = this.chart.options.indexAxis;
    return Object.keys(t).filter((s) => t[s].axis === e).shift();
  }
  _getAxis() {
    const t = {}, e = this.getFirstScaleIdForIndexAxis();
    for (const s of this.chart.data.datasets)
      t[O(this.chart.options.indexAxis === "x" ? s.xAxisID : s.yAxisID, e)] = !0;
    return Object.keys(t);
  }
  _getStackIndex(t, e, s) {
    const n = this._getStacks(t, s), o = e !== void 0 ? n.indexOf(e) : -1;
    return o === -1 ? n.length - 1 : o;
  }
  _getRuler() {
    const t = this.options, e = this._cachedMeta, s = e.iScale, n = [];
    let o, r;
    for (o = 0, r = e.data.length; o < r; ++o)
      n.push(s.getPixelForValue(this.getParsed(o)[s.axis], o));
    const a = t.barThickness;
    return {
      min: a || Rl(e),
      pixels: n,
      start: s._startPixel,
      end: s._endPixel,
      stackCount: this._getStackCount(),
      scale: s,
      grouped: t.grouped,
      ratio: a ? 1 : t.categoryPercentage * t.barPercentage
    };
  }
  _calculateBarValuePixels(t) {
    const { _cachedMeta: { vScale: e, _stacked: s, index: n }, options: { base: o, minBarLength: r } } = this, a = o || 0, l = this.getParsed(t), c = l._custom, h = Hi(c);
    let d = l[e.axis], f = 0, u = s ? this.applyStack(e, l, s) : d, g, p;
    u !== d && (f = u - d, u = d), h && (d = c.barStart, u = c.barEnd - c.barStart, d !== 0 && kt(d) !== kt(c.barEnd) && (f = 0), f += d);
    const m = !F(o) && !h ? o : f;
    let b = e.getPixelForValue(m);
    if (this.chart.getDataVisibility(t) ? g = e.getPixelForValue(f + u) : g = b, p = g - b, Math.abs(p) < r) {
      p = zl(p, e, a) * r, d === a && (b -= p / 2);
      const _ = e.getPixelForDecimal(0), y = e.getPixelForDecimal(1), v = Math.min(_, y), x = Math.max(_, y);
      b = Math.max(Math.min(b, x), v), g = b + p, s && !h && (l._stacks[e.axis]._visualValues[n] = e.getValueForPixel(g) - e.getValueForPixel(b));
    }
    if (b === e.getPixelForValue(a)) {
      const _ = kt(p) * e.getLineWidthForValue(a) / 2;
      b += _, p -= _;
    }
    return {
      size: p,
      base: b,
      head: g,
      center: g + p / 2
    };
  }
  _calculateBarIndexPixels(t, e) {
    const s = e.scale, n = this.options, o = n.skipNull, r = O(n.maxBarThickness, 1 / 0);
    let a, l;
    const c = this._getAxisCount();
    if (e.grouped) {
      const h = o ? this._getStackCount(t) : e.stackCount, d = n.barThickness === "flex" ? Fl(t, e, n, h * c) : El(t, e, n, h * c), f = this.chart.options.indexAxis === "x" ? this.getDataset().xAxisID : this.getDataset().yAxisID, u = this._getAxis().indexOf(O(f, this.getFirstScaleIdForIndexAxis())), g = this._getStackIndex(this.index, this._cachedMeta.stack, o ? t : void 0) + u;
      a = d.start + d.chunk * g + d.chunk / 2, l = Math.min(r, d.chunk * d.ratio);
    } else
      a = s.getPixelForValue(this.getParsed(t)[s.axis], t), l = Math.min(r, e.min * e.ratio);
    return {
      base: a - l / 2,
      head: a + l / 2,
      center: a,
      size: l
    };
  }
  draw() {
    const t = this._cachedMeta, e = t.vScale, s = t.data, n = s.length;
    let o = 0;
    for (; o < n; ++o)
      this.getParsed(o)[e.axis] !== null && !s[o].hidden && s[o].draw(this._ctx);
  }
}
M(ri, "id", "bar"), M(ri, "defaults", {
  datasetElementType: !1,
  dataElementType: "bar",
  categoryPercentage: 0.8,
  barPercentage: 0.9,
  grouped: !0,
  animations: {
    numbers: {
      type: "number",
      properties: [
        "x",
        "y",
        "base",
        "width",
        "height"
      ]
    }
  }
}), M(ri, "overrides", {
  scales: {
    _index_: {
      type: "category",
      offset: !0,
      grid: {
        offset: !0
      }
    },
    _value_: {
      type: "linear",
      beginAtZero: !0
    }
  }
});
class ai extends pt {
  initialize() {
    this.enableOptionSharing = !0, super.initialize();
  }
  parsePrimitiveData(t, e, s, n) {
    const o = super.parsePrimitiveData(t, e, s, n);
    for (let r = 0; r < o.length; r++)
      o[r]._custom = this.resolveDataElementOptions(r + s).radius;
    return o;
  }
  parseArrayData(t, e, s, n) {
    const o = super.parseArrayData(t, e, s, n);
    for (let r = 0; r < o.length; r++) {
      const a = e[s + r];
      o[r]._custom = O(a[2], this.resolveDataElementOptions(r + s).radius);
    }
    return o;
  }
  parseObjectData(t, e, s, n) {
    const o = super.parseObjectData(t, e, s, n);
    for (let r = 0; r < o.length; r++) {
      const a = e[s + r];
      o[r]._custom = O(a && a.r && +a.r, this.resolveDataElementOptions(r + s).radius);
    }
    return o;
  }
  getMaxOverflow() {
    const t = this._cachedMeta.data;
    let e = 0;
    for (let s = t.length - 1; s >= 0; --s)
      e = Math.max(e, t[s].size(this.resolveDataElementOptions(s)) / 2);
    return e > 0 && e;
  }
  getLabelAndValue(t) {
    const e = this._cachedMeta, s = this.chart.data.labels || [], { xScale: n, yScale: o } = e, r = this.getParsed(t), a = n.getLabelForValue(r.x), l = o.getLabelForValue(r.y), c = r._custom;
    return {
      label: s[t] || "",
      value: "(" + a + ", " + l + (c ? ", " + c : "") + ")"
    };
  }
  update(t) {
    const e = this._cachedMeta.data;
    this.updateElements(e, 0, e.length, t);
  }
  updateElements(t, e, s, n) {
    const o = n === "reset", { iScale: r, vScale: a } = this._cachedMeta, { sharedOptions: l, includeOptions: c } = this._getSharedOptions(e, n), h = r.axis, d = a.axis;
    for (let f = e; f < e + s; f++) {
      const u = t[f], g = !o && this.getParsed(f), p = {}, m = p[h] = o ? r.getPixelForDecimal(0.5) : r.getPixelForValue(g[h]), b = p[d] = o ? a.getBasePixel() : a.getPixelForValue(g[d]);
      p.skip = isNaN(m) || isNaN(b), c && (p.options = l || this.resolveDataElementOptions(f, u.active ? "active" : n), o && (p.options.radius = 0)), this.updateElement(u, f, p, n);
    }
  }
  resolveDataElementOptions(t, e) {
    const s = this.getParsed(t);
    let n = super.resolveDataElementOptions(t, e);
    n.$shared && (n = Object.assign({}, n, {
      $shared: !1
    }));
    const o = n.radius;
    return e !== "active" && (n.radius = 0), n.radius += O(s && s._custom, o), n;
  }
}
M(ai, "id", "bubble"), M(ai, "defaults", {
  datasetElementType: !1,
  dataElementType: "point",
  animations: {
    numbers: {
      type: "number",
      properties: [
        "x",
        "y",
        "borderWidth",
        "radius"
      ]
    }
  }
}), M(ai, "overrides", {
  scales: {
    x: {
      type: "linear"
    },
    y: {
      type: "linear"
    }
  }
});
function Hl(i, t, e) {
  let s = 1, n = 1, o = 0, r = 0;
  if (t < j) {
    const a = i, l = a + t, c = Math.cos(a), h = Math.sin(a), d = Math.cos(l), f = Math.sin(l), u = (y, v, x) => Fe(y, a, l, !0) ? 1 : Math.max(v, v * e, x, x * e), g = (y, v, x) => Fe(y, a, l, !0) ? -1 : Math.min(v, v * e, x, x * e), p = u(0, c, d), m = u(K, h, f), b = g(z, c, d), _ = g(z + K, h, f);
    s = (p - b) / 2, n = (m - _) / 2, o = -(p + b) / 2, r = -(m + _) / 2;
  }
  return {
    ratioX: s,
    ratioY: n,
    offsetX: o,
    offsetY: r
  };
}
class Zt extends pt {
  constructor(t, e) {
    super(t, e), this.enableOptionSharing = !0, this.innerRadius = void 0, this.outerRadius = void 0, this.offsetX = void 0, this.offsetY = void 0;
  }
  linkScales() {
  }
  parse(t, e) {
    const s = this.getDataset().data, n = this._cachedMeta;
    if (this._parsing === !1)
      n._parsed = s;
    else {
      let o = (l) => +s[l];
      if (I(s[t])) {
        const { key: l = "value" } = this._parsing;
        o = (c) => +Wt(s[c], l);
      }
      let r, a;
      for (r = t, a = t + e; r < a; ++r)
        n._parsed[r] = o(r);
    }
  }
  _getRotation() {
    return gt(this.options.rotation - 90);
  }
  _getCircumference() {
    return gt(this.options.circumference);
  }
  _getRotationExtents() {
    let t = j, e = -j;
    for (let s = 0; s < this.chart.data.datasets.length; ++s)
      if (this.chart.isDatasetVisible(s) && this.chart.getDatasetMeta(s).type === this._type) {
        const n = this.chart.getDatasetMeta(s).controller, o = n._getRotation(), r = n._getCircumference();
        t = Math.min(t, o), e = Math.max(e, o + r);
      }
    return {
      rotation: t,
      circumference: e - t
    };
  }
  update(t) {
    const e = this.chart, { chartArea: s } = e, n = this._cachedMeta, o = n.data, r = this.getMaxBorderWidth() + this.getMaxOffset(o) + this.options.spacing, a = Math.max((Math.min(s.width, s.height) - r) / 2, 0), l = Math.min(ta(this.options.cutout, a), 1), c = this._getRingWeight(this.index), { circumference: h, rotation: d } = this._getRotationExtents(), { ratioX: f, ratioY: u, offsetX: g, offsetY: p } = Hl(d, h, l), m = (s.width - r) / f, b = (s.height - r) / u, _ = Math.max(Math.min(m, b) / 2, 0), y = vo(this.options.radius, _), v = Math.max(y * l, 0), x = (y - v) / this._getVisibleDatasetWeightTotal();
    this.offsetX = g * y, this.offsetY = p * y, n.total = this.calculateTotal(), this.outerRadius = y - x * this._getRingWeightOffset(this.index), this.innerRadius = Math.max(this.outerRadius - x * c, 0), this.updateElements(o, 0, o.length, t);
  }
  _circumference(t, e) {
    const s = this.options, n = this._cachedMeta, o = this._getCircumference();
    return e && s.animation.animateRotate || !this.chart.getDataVisibility(t) || n._parsed[t] === null || n.data[t].hidden ? 0 : this.calculateCircumference(n._parsed[t] * o / j);
  }
  updateElements(t, e, s, n) {
    const o = n === "reset", r = this.chart, a = r.chartArea, c = r.options.animation, h = (a.left + a.right) / 2, d = (a.top + a.bottom) / 2, f = o && c.animateScale, u = f ? 0 : this.innerRadius, g = f ? 0 : this.outerRadius, { sharedOptions: p, includeOptions: m } = this._getSharedOptions(e, n);
    let b = this._getRotation(), _;
    for (_ = 0; _ < e; ++_)
      b += this._circumference(_, o);
    for (_ = e; _ < e + s; ++_) {
      const y = this._circumference(_, o), v = t[_], x = {
        x: h + this.offsetX,
        y: d + this.offsetY,
        startAngle: b,
        endAngle: b + y,
        circumference: y,
        outerRadius: g,
        innerRadius: u
      };
      m && (x.options = p || this.resolveDataElementOptions(_, v.active ? "active" : n)), b += y, this.updateElement(v, _, x, n);
    }
  }
  calculateTotal() {
    const t = this._cachedMeta, e = t.data;
    let s = 0, n;
    for (n = 0; n < e.length; n++) {
      const o = t._parsed[n];
      o !== null && !isNaN(o) && this.chart.getDataVisibility(n) && !e[n].hidden && (s += Math.abs(o));
    }
    return s;
  }
  calculateCircumference(t) {
    const e = this._cachedMeta.total;
    return e > 0 && !isNaN(t) ? j * (Math.abs(t) / e) : 0;
  }
  getLabelAndValue(t) {
    const e = this._cachedMeta, s = this.chart, n = s.data.labels || [], o = He(e._parsed[t], s.options.locale);
    return {
      label: n[t] || "",
      value: o
    };
  }
  getMaxBorderWidth(t) {
    let e = 0;
    const s = this.chart;
    let n, o, r, a, l;
    if (!t) {
      for (n = 0, o = s.data.datasets.length; n < o; ++n)
        if (s.isDatasetVisible(n)) {
          r = s.getDatasetMeta(n), t = r.data, a = r.controller;
          break;
        }
    }
    if (!t)
      return 0;
    for (n = 0, o = t.length; n < o; ++n)
      l = a.resolveDataElementOptions(n), l.borderAlign !== "inner" && (e = Math.max(e, l.borderWidth || 0, l.hoverBorderWidth || 0));
    return e;
  }
  getMaxOffset(t) {
    let e = 0;
    for (let s = 0, n = t.length; s < n; ++s) {
      const o = this.resolveDataElementOptions(s);
      e = Math.max(e, o.offset || 0, o.hoverOffset || 0);
    }
    return e;
  }
  _getRingWeightOffset(t) {
    let e = 0;
    for (let s = 0; s < t; ++s)
      this.chart.isDatasetVisible(s) && (e += this._getRingWeight(s));
    return e;
  }
  _getRingWeight(t) {
    return Math.max(O(this.chart.data.datasets[t].weight, 1), 0);
  }
  _getVisibleDatasetWeightTotal() {
    return this._getRingWeightOffset(this.chart.data.datasets.length) || 1;
  }
}
M(Zt, "id", "doughnut"), M(Zt, "defaults", {
  datasetElementType: !1,
  dataElementType: "arc",
  animation: {
    animateRotate: !0,
    animateScale: !1
  },
  animations: {
    numbers: {
      type: "number",
      properties: [
        "circumference",
        "endAngle",
        "innerRadius",
        "outerRadius",
        "startAngle",
        "x",
        "y",
        "offset",
        "borderWidth",
        "spacing"
      ]
    }
  },
  cutout: "50%",
  rotation: 0,
  circumference: 360,
  radius: "100%",
  spacing: 0,
  indexAxis: "r"
}), M(Zt, "descriptors", {
  _scriptable: (t) => t !== "spacing",
  _indexable: (t) => t !== "spacing" && !t.startsWith("borderDash") && !t.startsWith("hoverBorderDash")
}), M(Zt, "overrides", {
  aspectRatio: 1,
  plugins: {
    legend: {
      labels: {
        generateLabels(t) {
          const e = t.data, { labels: { pointStyle: s, textAlign: n, color: o, useBorderRadius: r, borderRadius: a } } = t.legend.options;
          return e.labels.length && e.datasets.length ? e.labels.map((l, c) => {
            const d = t.getDatasetMeta(0).controller.getStyle(c);
            return {
              text: l,
              fillStyle: d.backgroundColor,
              fontColor: o,
              hidden: !t.getDataVisibility(c),
              lineDash: d.borderDash,
              lineDashOffset: d.borderDashOffset,
              lineJoin: d.borderJoinStyle,
              lineWidth: d.borderWidth,
              strokeStyle: d.borderColor,
              textAlign: n,
              pointStyle: s,
              borderRadius: r && (a || d.borderRadius),
              index: c
            };
          }) : [];
        }
      },
      onClick(t, e, s) {
        s.chart.toggleDataVisibility(e.index), s.chart.update();
      }
    }
  }
});
class li extends pt {
  initialize() {
    this.enableOptionSharing = !0, this.supportsDecimation = !0, super.initialize();
  }
  update(t) {
    const e = this._cachedMeta, { dataset: s, data: n = [], _dataset: o } = e, r = this.chart._animationsDisabled;
    let { start: a, count: l } = Co(e, n, r);
    this._drawStart = a, this._drawCount = l, Oo(e) && (a = 0, l = n.length), s._chart = this.chart, s._datasetIndex = this.index, s._decimated = !!o._decimated, s.points = n;
    const c = this.resolveDatasetElementOptions(t);
    this.options.showLine || (c.borderWidth = 0), c.segment = this.options.segment, this.updateElement(s, void 0, {
      animated: !r,
      options: c
    }, t), this.updateElements(n, a, l, t);
  }
  updateElements(t, e, s, n) {
    const o = n === "reset", { iScale: r, vScale: a, _stacked: l, _dataset: c } = this._cachedMeta, { sharedOptions: h, includeOptions: d } = this._getSharedOptions(e, n), f = r.axis, u = a.axis, { spanGaps: g, segment: p } = this.options, m = he(g) ? g : Number.POSITIVE_INFINITY, b = this.chart._animationsDisabled || o || n === "none", _ = e + s, y = t.length;
    let v = e > 0 && this.getParsed(e - 1);
    for (let x = 0; x < y; ++x) {
      const S = t[x], k = b ? S : {};
      if (x < e || x >= _) {
        k.skip = !0;
        continue;
      }
      const w = this.getParsed(x), P = F(w[u]), T = k[f] = r.getPixelForValue(w[f], x), L = k[u] = o || P ? a.getBasePixel() : a.getPixelForValue(l ? this.applyStack(a, w, l) : w[u], x);
      k.skip = isNaN(T) || isNaN(L) || P, k.stop = x > 0 && Math.abs(w[f] - v[f]) > m, p && (k.parsed = w, k.raw = c.data[x]), d && (k.options = h || this.resolveDataElementOptions(x, S.active ? "active" : n)), b || this.updateElement(S, x, k, n), v = w;
    }
  }
  getMaxOverflow() {
    const t = this._cachedMeta, e = t.dataset, s = e.options && e.options.borderWidth || 0, n = t.data || [];
    if (!n.length)
      return s;
    const o = n[0].size(this.resolveDataElementOptions(0)), r = n[n.length - 1].size(this.resolveDataElementOptions(n.length - 1));
    return Math.max(s, o, r) / 2;
  }
  draw() {
    const t = this._cachedMeta;
    t.dataset.updateControlPoints(this.chart.chartArea, t.iScale.axis), super.draw();
  }
}
M(li, "id", "line"), M(li, "defaults", {
  datasetElementType: "line",
  dataElementType: "point",
  showLine: !0,
  spanGaps: !1
}), M(li, "overrides", {
  scales: {
    _index_: {
      type: "category"
    },
    _value_: {
      type: "linear"
    }
  }
});
class Oe extends pt {
  constructor(t, e) {
    super(t, e), this.innerRadius = void 0, this.outerRadius = void 0;
  }
  getLabelAndValue(t) {
    const e = this._cachedMeta, s = this.chart, n = s.data.labels || [], o = He(e._parsed[t].r, s.options.locale);
    return {
      label: n[t] || "",
      value: o
    };
  }
  parseObjectData(t, e, s, n) {
    return Bo.bind(this)(t, e, s, n);
  }
  update(t) {
    const e = this._cachedMeta.data;
    this._updateRadius(), this.updateElements(e, 0, e.length, t);
  }
  getMinMax() {
    const t = this._cachedMeta, e = {
      min: Number.POSITIVE_INFINITY,
      max: Number.NEGATIVE_INFINITY
    };
    return t.data.forEach((s, n) => {
      const o = this.getParsed(n).r;
      !isNaN(o) && this.chart.getDataVisibility(n) && (o < e.min && (e.min = o), o > e.max && (e.max = o));
    }), e;
  }
  _updateRadius() {
    const t = this.chart, e = t.chartArea, s = t.options, n = Math.min(e.right - e.left, e.bottom - e.top), o = Math.max(n / 2, 0), r = Math.max(s.cutoutPercentage ? o / 100 * s.cutoutPercentage : 1, 0), a = (o - r) / t.getVisibleDatasetCount();
    this.outerRadius = o - a * this.index, this.innerRadius = this.outerRadius - a;
  }
  updateElements(t, e, s, n) {
    const o = n === "reset", r = this.chart, l = r.options.animation, c = this._cachedMeta.rScale, h = c.xCenter, d = c.yCenter, f = c.getIndexAngle(0) - 0.5 * z;
    let u = f, g;
    const p = 360 / this.countVisibleElements();
    for (g = 0; g < e; ++g)
      u += this._computeAngle(g, n, p);
    for (g = e; g < e + s; g++) {
      const m = t[g];
      let b = u, _ = u + this._computeAngle(g, n, p), y = r.getDataVisibility(g) ? c.getDistanceFromCenterForValue(this.getParsed(g).r) : 0;
      u = _, o && (l.animateScale && (y = 0), l.animateRotate && (b = _ = f));
      const v = {
        x: h,
        y: d,
        innerRadius: 0,
        outerRadius: y,
        startAngle: b,
        endAngle: _,
        options: this.resolveDataElementOptions(g, m.active ? "active" : n)
      };
      this.updateElement(m, g, v, n);
    }
  }
  countVisibleElements() {
    const t = this._cachedMeta;
    let e = 0;
    return t.data.forEach((s, n) => {
      !isNaN(this.getParsed(n).r) && this.chart.getDataVisibility(n) && e++;
    }), e;
  }
  _computeAngle(t, e, s) {
    return this.chart.getDataVisibility(t) ? gt(this.resolveDataElementOptions(t, e).angle || s) : 0;
  }
}
M(Oe, "id", "polarArea"), M(Oe, "defaults", {
  dataElementType: "arc",
  animation: {
    animateRotate: !0,
    animateScale: !0
  },
  animations: {
    numbers: {
      type: "number",
      properties: [
        "x",
        "y",
        "startAngle",
        "endAngle",
        "innerRadius",
        "outerRadius"
      ]
    }
  },
  indexAxis: "r",
  startAngle: 0
}), M(Oe, "overrides", {
  aspectRatio: 1,
  plugins: {
    legend: {
      labels: {
        generateLabels(t) {
          const e = t.data;
          if (e.labels.length && e.datasets.length) {
            const { labels: { pointStyle: s, color: n } } = t.legend.options;
            return e.labels.map((o, r) => {
              const l = t.getDatasetMeta(0).controller.getStyle(r);
              return {
                text: o,
                fillStyle: l.backgroundColor,
                strokeStyle: l.borderColor,
                fontColor: n,
                lineWidth: l.borderWidth,
                pointStyle: s,
                hidden: !t.getDataVisibility(r),
                index: r
              };
            });
          }
          return [];
        }
      },
      onClick(t, e, s) {
        s.chart.toggleDataVisibility(e.index), s.chart.update();
      }
    }
  },
  scales: {
    r: {
      type: "radialLinear",
      angleLines: {
        display: !1
      },
      beginAtZero: !0,
      grid: {
        circular: !0
      },
      pointLabels: {
        display: !1
      },
      startAngle: 0
    }
  }
});
class ss extends Zt {
}
M(ss, "id", "pie"), M(ss, "defaults", {
  cutout: 0,
  rotation: 0,
  circumference: 360,
  radius: "100%"
});
class ci extends pt {
  getLabelAndValue(t) {
    const e = this._cachedMeta.vScale, s = this.getParsed(t);
    return {
      label: e.getLabels()[t],
      value: "" + e.getLabelForValue(s[e.axis])
    };
  }
  parseObjectData(t, e, s, n) {
    return Bo.bind(this)(t, e, s, n);
  }
  update(t) {
    const e = this._cachedMeta, s = e.dataset, n = e.data || [], o = e.iScale.getLabels();
    if (s.points = n, t !== "resize") {
      const r = this.resolveDatasetElementOptions(t);
      this.options.showLine || (r.borderWidth = 0);
      const a = {
        _loop: !0,
        _fullLoop: o.length === n.length,
        options: r
      };
      this.updateElement(s, void 0, a, t);
    }
    this.updateElements(n, 0, n.length, t);
  }
  updateElements(t, e, s, n) {
    const o = this._cachedMeta.rScale, r = n === "reset";
    for (let a = e; a < e + s; a++) {
      const l = t[a], c = this.resolveDataElementOptions(a, l.active ? "active" : n), h = o.getPointPositionForValue(a, this.getParsed(a).r), d = r ? o.xCenter : h.x, f = r ? o.yCenter : h.y, u = {
        x: d,
        y: f,
        angle: h.angle,
        skip: isNaN(d) || isNaN(f),
        options: c
      };
      this.updateElement(l, a, u, n);
    }
  }
}
M(ci, "id", "radar"), M(ci, "defaults", {
  datasetElementType: "line",
  dataElementType: "point",
  indexAxis: "r",
  showLine: !0,
  elements: {
    line: {
      fill: "start"
    }
  }
}), M(ci, "overrides", {
  aspectRatio: 1,
  scales: {
    r: {
      type: "radialLinear"
    }
  }
});
class hi extends pt {
  getLabelAndValue(t) {
    const e = this._cachedMeta, s = this.chart.data.labels || [], { xScale: n, yScale: o } = e, r = this.getParsed(t), a = n.getLabelForValue(r.x), l = o.getLabelForValue(r.y);
    return {
      label: s[t] || "",
      value: "(" + a + ", " + l + ")"
    };
  }
  update(t) {
    const e = this._cachedMeta, { data: s = [] } = e, n = this.chart._animationsDisabled;
    let { start: o, count: r } = Co(e, s, n);
    if (this._drawStart = o, this._drawCount = r, Oo(e) && (o = 0, r = s.length), this.options.showLine) {
      this.datasetElementType || this.addElements();
      const { dataset: a, _dataset: l } = e;
      a._chart = this.chart, a._datasetIndex = this.index, a._decimated = !!l._decimated, a.points = s;
      const c = this.resolveDatasetElementOptions(t);
      c.segment = this.options.segment, this.updateElement(a, void 0, {
        animated: !n,
        options: c
      }, t);
    } else this.datasetElementType && (delete e.dataset, this.datasetElementType = !1);
    this.updateElements(s, o, r, t);
  }
  addElements() {
    const { showLine: t } = this.options;
    !this.datasetElementType && t && (this.datasetElementType = this.chart.registry.getElement("line")), super.addElements();
  }
  updateElements(t, e, s, n) {
    const o = n === "reset", { iScale: r, vScale: a, _stacked: l, _dataset: c } = this._cachedMeta, h = this.resolveDataElementOptions(e, n), d = this.getSharedOptions(h), f = this.includeOptions(n, d), u = r.axis, g = a.axis, { spanGaps: p, segment: m } = this.options, b = he(p) ? p : Number.POSITIVE_INFINITY, _ = this.chart._animationsDisabled || o || n === "none";
    let y = e > 0 && this.getParsed(e - 1);
    for (let v = e; v < e + s; ++v) {
      const x = t[v], S = this.getParsed(v), k = _ ? x : {}, w = F(S[g]), P = k[u] = r.getPixelForValue(S[u], v), T = k[g] = o || w ? a.getBasePixel() : a.getPixelForValue(l ? this.applyStack(a, S, l) : S[g], v);
      k.skip = isNaN(P) || isNaN(T) || w, k.stop = v > 0 && Math.abs(S[u] - y[u]) > b, m && (k.parsed = S, k.raw = c.data[v]), f && (k.options = d || this.resolveDataElementOptions(v, x.active ? "active" : n)), _ || this.updateElement(x, v, k, n), y = S;
    }
    this.updateSharedOptions(d, n, h);
  }
  getMaxOverflow() {
    const t = this._cachedMeta, e = t.data || [];
    if (!this.options.showLine) {
      let a = 0;
      for (let l = e.length - 1; l >= 0; --l)
        a = Math.max(a, e[l].size(this.resolveDataElementOptions(l)) / 2);
      return a > 0 && a;
    }
    const s = t.dataset, n = s.options && s.options.borderWidth || 0;
    if (!e.length)
      return n;
    const o = e[0].size(this.resolveDataElementOptions(0)), r = e[e.length - 1].size(this.resolveDataElementOptions(e.length - 1));
    return Math.max(n, o, r) / 2;
  }
}
M(hi, "id", "scatter"), M(hi, "defaults", {
  datasetElementType: !1,
  dataElementType: "point",
  showLine: !1,
  fill: !1
}), M(hi, "overrides", {
  interaction: {
    mode: "point"
  },
  scales: {
    x: {
      type: "linear"
    },
    y: {
      type: "linear"
    }
  }
});
var jl = /* @__PURE__ */ Object.freeze({
  __proto__: null,
  BarController: ri,
  BubbleController: ai,
  DoughnutController: Zt,
  LineController: li,
  PieController: ss,
  PolarAreaController: Oe,
  RadarController: ci,
  ScatterController: hi
});
function Xt() {
  throw new Error("This method is not implemented: Check that a complete date adapter is provided.");
}
class Os {
  constructor(t) {
    M(this, "options");
    this.options = t || {};
  }
  /**
  * Override default date adapter methods.
  * Accepts type parameter to define options type.
  * @example
  * Chart._adapters._date.override<{myAdapterOption: string}>({
  *   init() {
  *     console.log(this.options.myAdapterOption);
  *   }
  * })
  */
  static override(t) {
    Object.assign(Os.prototype, t);
  }
  // eslint-disable-next-line @typescript-eslint/no-empty-function
  init() {
  }
  formats() {
    return Xt();
  }
  parse() {
    return Xt();
  }
  format() {
    return Xt();
  }
  add() {
    return Xt();
  }
  diff() {
    return Xt();
  }
  startOf() {
    return Xt();
  }
  endOf() {
    return Xt();
  }
}
var $l = {
  _date: Os
};
function Yl(i, t, e, s) {
  const { controller: n, data: o, _sorted: r } = i, a = n._cachedMeta.iScale, l = i.dataset && i.dataset.options ? i.dataset.options.spanGaps : null;
  if (a && t === a.axis && t !== "r" && r && o.length) {
    const c = a._reversePixels ? ua : Ot;
    if (s) {
      if (n._sharedOptions) {
        const h = o[0], d = typeof h.getRange == "function" && h.getRange(t);
        if (d) {
          const f = c(o, t, e - d), u = c(o, t, e + d);
          return {
            lo: f.lo,
            hi: u.hi
          };
        }
      }
    } else {
      const h = c(o, t, e);
      if (l) {
        const { vScale: d } = n._cachedMeta, { _parsed: f } = i, u = f.slice(0, h.lo + 1).reverse().findIndex((p) => !F(p[d.axis]));
        h.lo -= Math.max(0, u);
        const g = f.slice(h.hi).findIndex((p) => !F(p[d.axis]));
        h.hi += Math.max(0, g);
      }
      return h;
    }
  }
  return {
    lo: 0,
    hi: o.length - 1
  };
}
function Oi(i, t, e, s, n) {
  const o = i.getSortedVisibleDatasetMetas(), r = e[t];
  for (let a = 0, l = o.length; a < l; ++a) {
    const { index: c, data: h } = o[a], { lo: d, hi: f } = Yl(o[a], t, r, n);
    for (let u = d; u <= f; ++u) {
      const g = h[u];
      g.skip || s(g, c, u);
    }
  }
}
function Xl(i) {
  const t = i.indexOf("x") !== -1, e = i.indexOf("y") !== -1;
  return function(s, n) {
    const o = t ? Math.abs(s.x - n.x) : 0, r = e ? Math.abs(s.y - n.y) : 0;
    return Math.sqrt(Math.pow(o, 2) + Math.pow(r, 2));
  };
}
function ji(i, t, e, s, n) {
  const o = [];
  return !n && !i.isPointInArea(t) || Oi(i, e, t, function(a, l, c) {
    !n && !Tt(a, i.chartArea, 0) || a.inRange(t.x, t.y, s) && o.push({
      element: a,
      datasetIndex: l,
      index: c
    });
  }, !0), o;
}
function Ul(i, t, e, s) {
  let n = [];
  function o(r, a, l) {
    const { startAngle: c, endAngle: h } = r.getProps([
      "startAngle",
      "endAngle"
    ], s), { angle: d } = wo(r, {
      x: t.x,
      y: t.y
    });
    Fe(d, c, h) && n.push({
      element: r,
      datasetIndex: a,
      index: l
    });
  }
  return Oi(i, e, t, o), n;
}
function Kl(i, t, e, s, n, o) {
  let r = [];
  const a = Xl(e);
  let l = Number.POSITIVE_INFINITY;
  function c(h, d, f) {
    const u = h.inRange(t.x, t.y, n);
    if (s && !u)
      return;
    const g = h.getCenterPoint(n);
    if (!(!!o || i.isPointInArea(g)) && !u)
      return;
    const m = a(t, g);
    m < l ? (r = [
      {
        element: h,
        datasetIndex: d,
        index: f
      }
    ], l = m) : m === l && r.push({
      element: h,
      datasetIndex: d,
      index: f
    });
  }
  return Oi(i, e, t, c), r;
}
function $i(i, t, e, s, n, o) {
  return !o && !i.isPointInArea(t) ? [] : e === "r" && !s ? Ul(i, t, e, n) : Kl(i, t, e, s, n, o);
}
function gn(i, t, e, s, n) {
  const o = [], r = e === "x" ? "inXRange" : "inYRange";
  let a = !1;
  return Oi(i, e, t, (l, c, h) => {
    l[r] && l[r](t[e], n) && (o.push({
      element: l,
      datasetIndex: c,
      index: h
    }), a = a || l.inRange(t.x, t.y, n));
  }), s && !a ? [] : o;
}
var ql = {
  modes: {
    index(i, t, e, s) {
      const n = Kt(t, i), o = e.axis || "x", r = e.includeInvisible || !1, a = e.intersect ? ji(i, n, o, s, r) : $i(i, n, o, !1, s, r), l = [];
      return a.length ? (i.getSortedVisibleDatasetMetas().forEach((c) => {
        const h = a[0].index, d = c.data[h];
        d && !d.skip && l.push({
          element: d,
          datasetIndex: c.index,
          index: h
        });
      }), l) : [];
    },
    dataset(i, t, e, s) {
      const n = Kt(t, i), o = e.axis || "xy", r = e.includeInvisible || !1;
      let a = e.intersect ? ji(i, n, o, s, r) : $i(i, n, o, !1, s, r);
      if (a.length > 0) {
        const l = a[0].datasetIndex, c = i.getDatasetMeta(l).data;
        a = [];
        for (let h = 0; h < c.length; ++h)
          a.push({
            element: c[h],
            datasetIndex: l,
            index: h
          });
      }
      return a;
    },
    point(i, t, e, s) {
      const n = Kt(t, i), o = e.axis || "xy", r = e.includeInvisible || !1;
      return ji(i, n, o, s, r);
    },
    nearest(i, t, e, s) {
      const n = Kt(t, i), o = e.axis || "xy", r = e.includeInvisible || !1;
      return $i(i, n, o, e.intersect, s, r);
    },
    x(i, t, e, s) {
      const n = Kt(t, i);
      return gn(i, n, "x", e.intersect, s);
    },
    y(i, t, e, s) {
      const n = Kt(t, i);
      return gn(i, n, "y", e.intersect, s);
    }
  }
};
const qo = [
  "left",
  "top",
  "right",
  "bottom"
];
function ge(i, t) {
  return i.filter((e) => e.pos === t);
}
function pn(i, t) {
  return i.filter((e) => qo.indexOf(e.pos) === -1 && e.box.axis === t);
}
function pe(i, t) {
  return i.sort((e, s) => {
    const n = t ? s : e, o = t ? e : s;
    return n.weight === o.weight ? n.index - o.index : n.weight - o.weight;
  });
}
function Gl(i) {
  const t = [];
  let e, s, n, o, r, a;
  for (e = 0, s = (i || []).length; e < s; ++e)
    n = i[e], { position: o, options: { stack: r, stackWeight: a = 1 } } = n, t.push({
      index: e,
      box: n,
      pos: o,
      horizontal: n.isHorizontal(),
      weight: n.weight,
      stack: r && o + r,
      stackWeight: a
    });
  return t;
}
function Zl(i) {
  const t = {};
  for (const e of i) {
    const { stack: s, pos: n, stackWeight: o } = e;
    if (!s || !qo.includes(n))
      continue;
    const r = t[s] || (t[s] = {
      count: 0,
      placed: 0,
      weight: 0,
      size: 0
    });
    r.count++, r.weight += o;
  }
  return t;
}
function Jl(i, t) {
  const e = Zl(i), { vBoxMaxWidth: s, hBoxMaxHeight: n } = t;
  let o, r, a;
  for (o = 0, r = i.length; o < r; ++o) {
    a = i[o];
    const { fullSize: l } = a.box, c = e[a.stack], h = c && a.stackWeight / c.weight;
    a.horizontal ? (a.width = h ? h * s : l && t.availableWidth, a.height = n) : (a.width = s, a.height = h ? h * n : l && t.availableHeight);
  }
  return e;
}
function Ql(i) {
  const t = Gl(i), e = pe(t.filter((c) => c.box.fullSize), !0), s = pe(ge(t, "left"), !0), n = pe(ge(t, "right")), o = pe(ge(t, "top"), !0), r = pe(ge(t, "bottom")), a = pn(t, "x"), l = pn(t, "y");
  return {
    fullSize: e,
    leftAndTop: s.concat(o),
    rightAndBottom: n.concat(l).concat(r).concat(a),
    chartArea: ge(t, "chartArea"),
    vertical: s.concat(n).concat(l),
    horizontal: o.concat(r).concat(a)
  };
}
function mn(i, t, e, s) {
  return Math.max(i[e], t[e]) + Math.max(i[s], t[s]);
}
function Go(i, t) {
  i.top = Math.max(i.top, t.top), i.left = Math.max(i.left, t.left), i.bottom = Math.max(i.bottom, t.bottom), i.right = Math.max(i.right, t.right);
}
function tc(i, t, e, s) {
  const { pos: n, box: o } = e, r = i.maxPadding;
  if (!I(n)) {
    e.size && (i[n] -= e.size);
    const d = s[e.stack] || {
      size: 0,
      count: 1
    };
    d.size = Math.max(d.size, e.horizontal ? o.height : o.width), e.size = d.size / d.count, i[n] += e.size;
  }
  o.getPadding && Go(r, o.getPadding());
  const a = Math.max(0, t.outerWidth - mn(r, i, "left", "right")), l = Math.max(0, t.outerHeight - mn(r, i, "top", "bottom")), c = a !== i.w, h = l !== i.h;
  return i.w = a, i.h = l, e.horizontal ? {
    same: c,
    other: h
  } : {
    same: h,
    other: c
  };
}
function ec(i) {
  const t = i.maxPadding;
  function e(s) {
    const n = Math.max(t[s] - i[s], 0);
    return i[s] += n, n;
  }
  i.y += e("top"), i.x += e("left"), e("right"), e("bottom");
}
function ic(i, t) {
  const e = t.maxPadding;
  function s(n) {
    const o = {
      left: 0,
      top: 0,
      right: 0,
      bottom: 0
    };
    return n.forEach((r) => {
      o[r] = Math.max(t[r], e[r]);
    }), o;
  }
  return s(i ? [
    "left",
    "right"
  ] : [
    "top",
    "bottom"
  ]);
}
function ye(i, t, e, s) {
  const n = [];
  let o, r, a, l, c, h;
  for (o = 0, r = i.length, c = 0; o < r; ++o) {
    a = i[o], l = a.box, l.update(a.width || t.w, a.height || t.h, ic(a.horizontal, t));
    const { same: d, other: f } = tc(t, e, a, s);
    c |= d && n.length, h = h || f, l.fullSize || n.push(a);
  }
  return c && ye(n, t, e, s) || h;
}
function qe(i, t, e, s, n) {
  i.top = e, i.left = t, i.right = t + s, i.bottom = e + n, i.width = s, i.height = n;
}
function bn(i, t, e, s) {
  const n = e.padding;
  let { x: o, y: r } = t;
  for (const a of i) {
    const l = a.box, c = s[a.stack] || {
      placed: 0,
      weight: 1
    }, h = a.stackWeight / c.weight || 1;
    if (a.horizontal) {
      const d = t.w * h, f = c.size || l.height;
      Ee(c.start) && (r = c.start), l.fullSize ? qe(l, n.left, r, e.outerWidth - n.right - n.left, f) : qe(l, t.left + c.placed, r, d, f), c.start = r, c.placed += d, r = l.bottom;
    } else {
      const d = t.h * h, f = c.size || l.width;
      Ee(c.start) && (o = c.start), l.fullSize ? qe(l, o, n.top, f, e.outerHeight - n.bottom - n.top) : qe(l, o, t.top + c.placed, f, d), c.start = o, c.placed += d, o = l.right;
    }
  }
  t.x = o, t.y = r;
}
var ot = {
  addBox(i, t) {
    i.boxes || (i.boxes = []), t.fullSize = t.fullSize || !1, t.position = t.position || "top", t.weight = t.weight || 0, t._layers = t._layers || function() {
      return [
        {
          z: 0,
          draw(e) {
            t.draw(e);
          }
        }
      ];
    }, i.boxes.push(t);
  },
  removeBox(i, t) {
    const e = i.boxes ? i.boxes.indexOf(t) : -1;
    e !== -1 && i.boxes.splice(e, 1);
  },
  configure(i, t, e) {
    t.fullSize = e.fullSize, t.position = e.position, t.weight = e.weight;
  },
  update(i, t, e, s) {
    if (!i)
      return;
    const n = rt(i.options.layout.padding), o = Math.max(t - n.width, 0), r = Math.max(e - n.height, 0), a = Ql(i.boxes), l = a.vertical, c = a.horizontal;
    V(i.boxes, (p) => {
      typeof p.beforeLayout == "function" && p.beforeLayout();
    });
    const h = l.reduce((p, m) => m.box.options && m.box.options.display === !1 ? p : p + 1, 0) || 1, d = Object.freeze({
      outerWidth: t,
      outerHeight: e,
      padding: n,
      availableWidth: o,
      availableHeight: r,
      vBoxMaxWidth: o / 2 / h,
      hBoxMaxHeight: r / 2
    }), f = Object.assign({}, n);
    Go(f, rt(s));
    const u = Object.assign({
      maxPadding: f,
      w: o,
      h: r,
      x: n.left,
      y: n.top
    }, n), g = Jl(l.concat(c), d);
    ye(a.fullSize, u, d, g), ye(l, u, d, g), ye(c, u, d, g) && ye(l, u, d, g), ec(u), bn(a.leftAndTop, u, d, g), u.x += u.w, u.y += u.h, bn(a.rightAndBottom, u, d, g), i.chartArea = {
      left: u.left,
      top: u.top,
      right: u.left + u.w,
      bottom: u.top + u.h,
      height: u.h,
      width: u.w
    }, V(a.chartArea, (p) => {
      const m = p.box;
      Object.assign(m, i.chartArea), m.update(u.w, u.h, {
        left: 0,
        top: 0,
        right: 0,
        bottom: 0
      });
    });
  }
};
class Zo {
  acquireContext(t, e) {
  }
  releaseContext(t) {
    return !1;
  }
  addEventListener(t, e, s) {
  }
  removeEventListener(t, e, s) {
  }
  getDevicePixelRatio() {
    return 1;
  }
  getMaximumSize(t, e, s, n) {
    return e = Math.max(0, e || t.width), s = s || t.height, {
      width: e,
      height: Math.max(0, n ? Math.floor(e / n) : s)
    };
  }
  isAttached(t) {
    return !0;
  }
  updateConfig(t) {
  }
}
class sc extends Zo {
  acquireContext(t) {
    return t && t.getContext && t.getContext("2d") || null;
  }
  updateConfig(t) {
    t.options.animation = !1;
  }
}
const di = "$chartjs", nc = {
  touchstart: "mousedown",
  touchmove: "mousemove",
  touchend: "mouseup",
  pointerenter: "mouseenter",
  pointerdown: "mousedown",
  pointermove: "mousemove",
  pointerup: "mouseup",
  pointerleave: "mouseout",
  pointerout: "mouseout"
}, _n = (i) => i === null || i === "";
function oc(i, t) {
  const e = i.style, s = i.getAttribute("height"), n = i.getAttribute("width");
  if (i[di] = {
    initial: {
      height: s,
      width: n,
      style: {
        display: e.display,
        height: e.height,
        width: e.width
      }
    }
  }, e.display = e.display || "block", e.boxSizing = e.boxSizing || "border-box", _n(n)) {
    const o = tn(i, "width");
    o !== void 0 && (i.width = o);
  }
  if (_n(s))
    if (i.style.height === "")
      i.height = i.width / (t || 2);
    else {
      const o = tn(i, "height");
      o !== void 0 && (i.height = o);
    }
  return i;
}
const Jo = rl ? {
  passive: !0
} : !1;
function rc(i, t, e) {
  i && i.addEventListener(t, e, Jo);
}
function ac(i, t, e) {
  i && i.canvas && i.canvas.removeEventListener(t, e, Jo);
}
function lc(i, t) {
  const e = nc[i.type] || i.type, { x: s, y: n } = Kt(i, t);
  return {
    type: e,
    chart: t,
    native: i,
    x: s !== void 0 ? s : null,
    y: n !== void 0 ? n : null
  };
}
function yi(i, t) {
  for (const e of i)
    if (e === t || e.contains(t))
      return !0;
}
function cc(i, t, e) {
  const s = i.canvas, n = new MutationObserver((o) => {
    let r = !1;
    for (const a of o)
      r = r || yi(a.addedNodes, s), r = r && !yi(a.removedNodes, s);
    r && e();
  });
  return n.observe(document, {
    childList: !0,
    subtree: !0
  }), n;
}
function hc(i, t, e) {
  const s = i.canvas, n = new MutationObserver((o) => {
    let r = !1;
    for (const a of o)
      r = r || yi(a.removedNodes, s), r = r && !yi(a.addedNodes, s);
    r && e();
  });
  return n.observe(document, {
    childList: !0,
    subtree: !0
  }), n;
}
const ze = /* @__PURE__ */ new Map();
let xn = 0;
function Qo() {
  const i = window.devicePixelRatio;
  i !== xn && (xn = i, ze.forEach((t, e) => {
    e.currentDevicePixelRatio !== i && t();
  }));
}
function dc(i, t) {
  ze.size || window.addEventListener("resize", Qo), ze.set(i, t);
}
function fc(i) {
  ze.delete(i), ze.size || window.removeEventListener("resize", Qo);
}
function uc(i, t, e) {
  const s = i.canvas, n = s && Cs(s);
  if (!n)
    return;
  const o = Ao((a, l) => {
    const c = n.clientWidth;
    e(a, l), c < n.clientWidth && e();
  }, window), r = new ResizeObserver((a) => {
    const l = a[0], c = l.contentRect.width, h = l.contentRect.height;
    c === 0 && h === 0 || o(c, h);
  });
  return r.observe(n), dc(i, o), r;
}
function Yi(i, t, e) {
  e && e.disconnect(), t === "resize" && fc(i);
}
function gc(i, t, e) {
  const s = i.canvas, n = Ao((o) => {
    i.ctx !== null && e(lc(o, i));
  }, i);
  return rc(s, t, n), n;
}
class pc extends Zo {
  acquireContext(t, e) {
    const s = t && t.getContext && t.getContext("2d");
    return s && s.canvas === t ? (oc(t, e), s) : null;
  }
  releaseContext(t) {
    const e = t.canvas;
    if (!e[di])
      return !1;
    const s = e[di].initial;
    [
      "height",
      "width"
    ].forEach((o) => {
      const r = s[o];
      F(r) ? e.removeAttribute(o) : e.setAttribute(o, r);
    });
    const n = s.style || {};
    return Object.keys(n).forEach((o) => {
      e.style[o] = n[o];
    }), e.width = e.width, delete e[di], !0;
  }
  addEventListener(t, e, s) {
    this.removeEventListener(t, e);
    const n = t.$proxies || (t.$proxies = {}), r = {
      attach: cc,
      detach: hc,
      resize: uc
    }[e] || gc;
    n[e] = r(t, e, s);
  }
  removeEventListener(t, e) {
    const s = t.$proxies || (t.$proxies = {}), n = s[e];
    if (!n)
      return;
    ({
      attach: Yi,
      detach: Yi,
      resize: Yi
    }[e] || ac)(t, e, n), s[e] = void 0;
  }
  getDevicePixelRatio() {
    return window.devicePixelRatio;
  }
  getMaximumSize(t, e, s, n) {
    return ol(t, e, s, n);
  }
  isAttached(t) {
    const e = t && Cs(t);
    return !!(e && e.isConnected);
  }
}
function mc(i) {
  return !As() || typeof OffscreenCanvas < "u" && i instanceof OffscreenCanvas ? sc : pc;
}
class mt {
  constructor() {
    M(this, "x");
    M(this, "y");
    M(this, "active", !1);
    M(this, "options");
    M(this, "$animations");
  }
  tooltipPosition(t) {
    const { x: e, y: s } = this.getProps([
      "x",
      "y"
    ], t);
    return {
      x: e,
      y: s
    };
  }
  hasValue() {
    return he(this.x) && he(this.y);
  }
  getProps(t, e) {
    const s = this.$animations;
    if (!e || !s)
      return this;
    const n = {};
    return t.forEach((o) => {
      n[o] = s[o] && s[o].active() ? s[o]._to : this[o];
    }), n;
  }
}
M(mt, "defaults", {}), M(mt, "defaultRoutes");
function bc(i, t) {
  const e = i.options.ticks, s = _c(i), n = Math.min(e.maxTicksLimit || s, s), o = e.major.enabled ? yc(t) : [], r = o.length, a = o[0], l = o[r - 1], c = [];
  if (r > n)
    return vc(t, c, o, r / n), c;
  const h = xc(o, t, n);
  if (r > 0) {
    let d, f;
    const u = r > 1 ? Math.round((l - a) / (r - 1)) : null;
    for (Ge(t, c, h, F(u) ? 0 : a - u, a), d = 0, f = r - 1; d < f; d++)
      Ge(t, c, h, o[d], o[d + 1]);
    return Ge(t, c, h, l, F(u) ? t.length : l + u), c;
  }
  return Ge(t, c, h), c;
}
function _c(i) {
  const t = i.options.offset, e = i._tickSize(), s = i._length / e + (t ? 0 : 1), n = i._maxLength / e;
  return Math.floor(Math.min(s, n));
}
function xc(i, t, e) {
  const s = kc(i), n = t.length / e;
  if (!s)
    return Math.max(n, 1);
  const o = la(s);
  for (let r = 0, a = o.length - 1; r < a; r++) {
    const l = o[r];
    if (l > n)
      return l;
  }
  return Math.max(n, 1);
}
function yc(i) {
  const t = [];
  let e, s;
  for (e = 0, s = i.length; e < s; e++)
    i[e].major && t.push(e);
  return t;
}
function vc(i, t, e, s) {
  let n = 0, o = e[0], r;
  for (s = Math.ceil(s), r = 0; r < i.length; r++)
    r === o && (t.push(i[r]), n++, o = e[n * s]);
}
function Ge(i, t, e, s, n) {
  const o = O(s, 0), r = Math.min(O(n, i.length), i.length);
  let a = 0, l, c, h;
  for (e = Math.ceil(e), n && (l = n - s, e = l / Math.floor(l / e)), h = o; h < 0; )
    a++, h = Math.round(o + a * e);
  for (c = Math.max(o, 0); c < r; c++)
    c === h && (t.push(i[c]), a++, h = Math.round(o + a * e));
}
function kc(i) {
  const t = i.length;
  let e, s;
  if (t < 2)
    return !1;
  for (s = i[0], e = 1; e < t; ++e)
    if (i[e] - i[e - 1] !== s)
      return !1;
  return s;
}
const Mc = (i) => i === "left" ? "right" : i === "right" ? "left" : i, yn = (i, t, e) => t === "top" || t === "left" ? i[t] + e : i[t] - e, vn = (i, t) => Math.min(t || i, i);
function kn(i, t) {
  const e = [], s = i.length / t, n = i.length;
  let o = 0;
  for (; o < n; o += s)
    e.push(i[Math.floor(o)]);
  return e;
}
function wc(i, t, e) {
  const s = i.ticks.length, n = Math.min(t, s - 1), o = i._startPixel, r = i._endPixel, a = 1e-6;
  let l = i.getPixelForTick(n), c;
  if (!(e && (s === 1 ? c = Math.max(l - o, r - l) : t === 0 ? c = (i.getPixelForTick(1) - l) / 2 : c = (l - i.getPixelForTick(n - 1)) / 2, l += n < t ? c : -c, l < o - a || l > r + a)))
    return l;
}
function Sc(i, t) {
  V(i, (e) => {
    const s = e.gc, n = s.length / 2;
    let o;
    if (n > t) {
      for (o = 0; o < n; ++o)
        delete e.data[s[o]];
      s.splice(0, n);
    }
  });
}
function me(i) {
  return i.drawTicks ? i.tickLength : 0;
}
function Mn(i, t) {
  if (!i.display)
    return 0;
  const e = G(i.font, t), s = rt(i.padding);
  return (Y(i.text) ? i.text.length : 1) * e.lineHeight + s.height;
}
function Pc(i, t) {
  return Ht(i, {
    scale: t,
    type: "scale"
  });
}
function Dc(i, t, e) {
  return Ht(i, {
    tick: e,
    index: t,
    type: "tick"
  });
}
function Ac(i, t, e) {
  let s = ks(i);
  return (e && t !== "right" || !e && t === "right") && (s = Mc(s)), s;
}
function Cc(i, t, e, s) {
  const { top: n, left: o, bottom: r, right: a, chart: l } = i, { chartArea: c, scales: h } = l;
  let d = 0, f, u, g;
  const p = r - n, m = a - o;
  if (i.isHorizontal()) {
    if (u = st(s, o, a), I(e)) {
      const b = Object.keys(e)[0], _ = e[b];
      g = h[b].getPixelForValue(_) + p - t;
    } else e === "center" ? g = (c.bottom + c.top) / 2 + p - t : g = yn(i, e, t);
    f = a - o;
  } else {
    if (I(e)) {
      const b = Object.keys(e)[0], _ = e[b];
      u = h[b].getPixelForValue(_) - m + t;
    } else e === "center" ? u = (c.left + c.right) / 2 - m + t : u = yn(i, e, t);
    g = st(s, r, n), d = e === "left" ? -K : K;
  }
  return {
    titleX: u,
    titleY: g,
    maxWidth: f,
    rotation: d
  };
}
class ie extends mt {
  constructor(t) {
    super(), this.id = t.id, this.type = t.type, this.options = void 0, this.ctx = t.ctx, this.chart = t.chart, this.top = void 0, this.bottom = void 0, this.left = void 0, this.right = void 0, this.width = void 0, this.height = void 0, this._margins = {
      left: 0,
      right: 0,
      top: 0,
      bottom: 0
    }, this.maxWidth = void 0, this.maxHeight = void 0, this.paddingTop = void 0, this.paddingBottom = void 0, this.paddingLeft = void 0, this.paddingRight = void 0, this.axis = void 0, this.labelRotation = void 0, this.min = void 0, this.max = void 0, this._range = void 0, this.ticks = [], this._gridLineItems = null, this._labelItems = null, this._labelSizes = null, this._length = 0, this._maxLength = 0, this._longestTextCache = {}, this._startPixel = void 0, this._endPixel = void 0, this._reversePixels = !1, this._userMax = void 0, this._userMin = void 0, this._suggestedMax = void 0, this._suggestedMin = void 0, this._ticksLength = 0, this._borderValue = 0, this._cache = {}, this._dataLimitsCached = !1, this.$context = void 0;
  }
  init(t) {
    this.options = t.setContext(this.getContext()), this.axis = t.axis, this._userMin = this.parse(t.min), this._userMax = this.parse(t.max), this._suggestedMin = this.parse(t.suggestedMin), this._suggestedMax = this.parse(t.suggestedMax);
  }
  parse(t, e) {
    return t;
  }
  getUserBounds() {
    let { _userMin: t, _userMax: e, _suggestedMin: s, _suggestedMax: n } = this;
    return t = dt(t, Number.POSITIVE_INFINITY), e = dt(e, Number.NEGATIVE_INFINITY), s = dt(s, Number.POSITIVE_INFINITY), n = dt(n, Number.NEGATIVE_INFINITY), {
      min: dt(t, s),
      max: dt(e, n),
      minDefined: U(t),
      maxDefined: U(e)
    };
  }
  getMinMax(t) {
    let { min: e, max: s, minDefined: n, maxDefined: o } = this.getUserBounds(), r;
    if (n && o)
      return {
        min: e,
        max: s
      };
    const a = this.getMatchingVisibleMetas();
    for (let l = 0, c = a.length; l < c; ++l)
      r = a[l].controller.getMinMax(this, t), n || (e = Math.min(e, r.min)), o || (s = Math.max(s, r.max));
    return e = o && e > s ? s : e, s = n && e > s ? e : s, {
      min: dt(e, dt(s, e)),
      max: dt(s, dt(e, s))
    };
  }
  getPadding() {
    return {
      left: this.paddingLeft || 0,
      top: this.paddingTop || 0,
      right: this.paddingRight || 0,
      bottom: this.paddingBottom || 0
    };
  }
  getTicks() {
    return this.ticks;
  }
  getLabels() {
    const t = this.chart.data;
    return this.options.labels || (this.isHorizontal() ? t.xLabels : t.yLabels) || t.labels || [];
  }
  getLabelItems(t = this.chart.chartArea) {
    return this._labelItems || (this._labelItems = this._computeLabelItems(t));
  }
  beforeLayout() {
    this._cache = {}, this._dataLimitsCached = !1;
  }
  beforeUpdate() {
    N(this.options.beforeUpdate, [
      this
    ]);
  }
  update(t, e, s) {
    const { beginAtZero: n, grace: o, ticks: r } = this.options, a = r.sampleSize;
    this.beforeUpdate(), this.maxWidth = t, this.maxHeight = e, this._margins = s = Object.assign({
      left: 0,
      right: 0,
      top: 0,
      bottom: 0
    }, s), this.ticks = null, this._labelSizes = null, this._gridLineItems = null, this._labelItems = null, this.beforeSetDimensions(), this.setDimensions(), this.afterSetDimensions(), this._maxLength = this.isHorizontal() ? this.width + s.left + s.right : this.height + s.top + s.bottom, this._dataLimitsCached || (this.beforeDataLimits(), this.determineDataLimits(), this.afterDataLimits(), this._range = za(this, o, n), this._dataLimitsCached = !0), this.beforeBuildTicks(), this.ticks = this.buildTicks() || [], this.afterBuildTicks();
    const l = a < this.ticks.length;
    this._convertTicksToLabels(l ? kn(this.ticks, a) : this.ticks), this.configure(), this.beforeCalculateLabelRotation(), this.calculateLabelRotation(), this.afterCalculateLabelRotation(), r.display && (r.autoSkip || r.source === "auto") && (this.ticks = bc(this, this.ticks), this._labelSizes = null, this.afterAutoSkip()), l && this._convertTicksToLabels(this.ticks), this.beforeFit(), this.fit(), this.afterFit(), this.afterUpdate();
  }
  configure() {
    let t = this.options.reverse, e, s;
    this.isHorizontal() ? (e = this.left, s = this.right) : (e = this.top, s = this.bottom, t = !t), this._startPixel = e, this._endPixel = s, this._reversePixels = t, this._length = s - e, this._alignToPixels = this.options.alignToPixels;
  }
  afterUpdate() {
    N(this.options.afterUpdate, [
      this
    ]);
  }
  beforeSetDimensions() {
    N(this.options.beforeSetDimensions, [
      this
    ]);
  }
  setDimensions() {
    this.isHorizontal() ? (this.width = this.maxWidth, this.left = 0, this.right = this.width) : (this.height = this.maxHeight, this.top = 0, this.bottom = this.height), this.paddingLeft = 0, this.paddingTop = 0, this.paddingRight = 0, this.paddingBottom = 0;
  }
  afterSetDimensions() {
    N(this.options.afterSetDimensions, [
      this
    ]);
  }
  _callHooks(t) {
    this.chart.notifyPlugins(t, this.getContext()), N(this.options[t], [
      this
    ]);
  }
  beforeDataLimits() {
    this._callHooks("beforeDataLimits");
  }
  determineDataLimits() {
  }
  afterDataLimits() {
    this._callHooks("afterDataLimits");
  }
  beforeBuildTicks() {
    this._callHooks("beforeBuildTicks");
  }
  buildTicks() {
    return [];
  }
  afterBuildTicks() {
    this._callHooks("afterBuildTicks");
  }
  beforeTickToLabelConversion() {
    N(this.options.beforeTickToLabelConversion, [
      this
    ]);
  }
  generateTickLabels(t) {
    const e = this.options.ticks;
    let s, n, o;
    for (s = 0, n = t.length; s < n; s++)
      o = t[s], o.label = N(e.callback, [
        o.value,
        s,
        t
      ], this);
  }
  afterTickToLabelConversion() {
    N(this.options.afterTickToLabelConversion, [
      this
    ]);
  }
  beforeCalculateLabelRotation() {
    N(this.options.beforeCalculateLabelRotation, [
      this
    ]);
  }
  calculateLabelRotation() {
    const t = this.options, e = t.ticks, s = vn(this.ticks.length, t.ticks.maxTicksLimit), n = e.minRotation || 0, o = e.maxRotation;
    let r = n, a, l, c;
    if (!this._isVisible() || !e.display || n >= o || s <= 1 || !this.isHorizontal()) {
      this.labelRotation = n;
      return;
    }
    const h = this._getLabelSizes(), d = h.widest.width, f = h.highest.height, u = J(this.chart.width - d, 0, this.maxWidth);
    a = t.offset ? this.maxWidth / s : u / (s - 1), d + 6 > a && (a = u / (s - (t.offset ? 0.5 : 1)), l = this.maxHeight - me(t.grid) - e.padding - Mn(t.title, this.chart.options.font), c = Math.sqrt(d * d + f * f), r = ys(Math.min(Math.asin(J((h.highest.height + 6) / a, -1, 1)), Math.asin(J(l / c, -1, 1)) - Math.asin(J(f / c, -1, 1)))), r = Math.max(n, Math.min(o, r))), this.labelRotation = r;
  }
  afterCalculateLabelRotation() {
    N(this.options.afterCalculateLabelRotation, [
      this
    ]);
  }
  afterAutoSkip() {
  }
  beforeFit() {
    N(this.options.beforeFit, [
      this
    ]);
  }
  fit() {
    const t = {
      width: 0,
      height: 0
    }, { chart: e, options: { ticks: s, title: n, grid: o } } = this, r = this._isVisible(), a = this.isHorizontal();
    if (r) {
      const l = Mn(n, e.options.font);
      if (a ? (t.width = this.maxWidth, t.height = me(o) + l) : (t.height = this.maxHeight, t.width = me(o) + l), s.display && this.ticks.length) {
        const { first: c, last: h, widest: d, highest: f } = this._getLabelSizes(), u = s.padding * 2, g = gt(this.labelRotation), p = Math.cos(g), m = Math.sin(g);
        if (a) {
          const b = s.mirror ? 0 : m * d.width + p * f.height;
          t.height = Math.min(this.maxHeight, t.height + b + u);
        } else {
          const b = s.mirror ? 0 : p * d.width + m * f.height;
          t.width = Math.min(this.maxWidth, t.width + b + u);
        }
        this._calculatePadding(c, h, m, p);
      }
    }
    this._handleMargins(), a ? (this.width = this._length = e.width - this._margins.left - this._margins.right, this.height = t.height) : (this.width = t.width, this.height = this._length = e.height - this._margins.top - this._margins.bottom);
  }
  _calculatePadding(t, e, s, n) {
    const { ticks: { align: o, padding: r }, position: a } = this.options, l = this.labelRotation !== 0, c = a !== "top" && this.axis === "x";
    if (this.isHorizontal()) {
      const h = this.getPixelForTick(0) - this.left, d = this.right - this.getPixelForTick(this.ticks.length - 1);
      let f = 0, u = 0;
      l ? c ? (f = n * t.width, u = s * e.height) : (f = s * t.height, u = n * e.width) : o === "start" ? u = e.width : o === "end" ? f = t.width : o !== "inner" && (f = t.width / 2, u = e.width / 2), this.paddingLeft = Math.max((f - h + r) * this.width / (this.width - h), 0), this.paddingRight = Math.max((u - d + r) * this.width / (this.width - d), 0);
    } else {
      let h = e.height / 2, d = t.height / 2;
      o === "start" ? (h = 0, d = t.height) : o === "end" && (h = e.height, d = 0), this.paddingTop = h + r, this.paddingBottom = d + r;
    }
  }
  _handleMargins() {
    this._margins && (this._margins.left = Math.max(this.paddingLeft, this._margins.left), this._margins.top = Math.max(this.paddingTop, this._margins.top), this._margins.right = Math.max(this.paddingRight, this._margins.right), this._margins.bottom = Math.max(this.paddingBottom, this._margins.bottom));
  }
  afterFit() {
    N(this.options.afterFit, [
      this
    ]);
  }
  isHorizontal() {
    const { axis: t, position: e } = this.options;
    return e === "top" || e === "bottom" || t === "x";
  }
  isFullSize() {
    return this.options.fullSize;
  }
  _convertTicksToLabels(t) {
    this.beforeTickToLabelConversion(), this.generateTickLabels(t);
    let e, s;
    for (e = 0, s = t.length; e < s; e++)
      F(t[e].label) && (t.splice(e, 1), s--, e--);
    this.afterTickToLabelConversion();
  }
  _getLabelSizes() {
    let t = this._labelSizes;
    if (!t) {
      const e = this.options.ticks.sampleSize;
      let s = this.ticks;
      e < s.length && (s = kn(s, e)), this._labelSizes = t = this._computeLabelSizes(s, s.length, this.options.ticks.maxTicksLimit);
    }
    return t;
  }
  _computeLabelSizes(t, e, s) {
    const { ctx: n, _longestTextCache: o } = this, r = [], a = [], l = Math.floor(e / vn(e, s));
    let c = 0, h = 0, d, f, u, g, p, m, b, _, y, v, x;
    for (d = 0; d < e; d += l) {
      if (g = t[d].label, p = this._resolveTickFontOptions(d), n.font = m = p.string, b = o[m] = o[m] || {
        data: {},
        gc: []
      }, _ = p.lineHeight, y = v = 0, !F(g) && !Y(g))
        y = _i(n, b.data, b.gc, y, g), v = _;
      else if (Y(g))
        for (f = 0, u = g.length; f < u; ++f)
          x = g[f], !F(x) && !Y(x) && (y = _i(n, b.data, b.gc, y, x), v += _);
      r.push(y), a.push(v), c = Math.max(y, c), h = Math.max(v, h);
    }
    Sc(o, e);
    const S = r.indexOf(c), k = a.indexOf(h), w = (P) => ({
      width: r[P] || 0,
      height: a[P] || 0
    });
    return {
      first: w(0),
      last: w(e - 1),
      widest: w(S),
      highest: w(k),
      widths: r,
      heights: a
    };
  }
  getLabelForValue(t) {
    return t;
  }
  getPixelForValue(t, e) {
    return NaN;
  }
  getValueForPixel(t) {
  }
  getPixelForTick(t) {
    const e = this.ticks;
    return t < 0 || t > e.length - 1 ? null : this.getPixelForValue(e[t].value);
  }
  getPixelForDecimal(t) {
    this._reversePixels && (t = 1 - t);
    const e = this._startPixel + t * this._length;
    return fa(this._alignToPixels ? Yt(this.chart, e, 0) : e);
  }
  getDecimalForPixel(t) {
    const e = (t - this._startPixel) / this._length;
    return this._reversePixels ? 1 - e : e;
  }
  getBasePixel() {
    return this.getPixelForValue(this.getBaseValue());
  }
  getBaseValue() {
    const { min: t, max: e } = this;
    return t < 0 && e < 0 ? e : t > 0 && e > 0 ? t : 0;
  }
  getContext(t) {
    const e = this.ticks || [];
    if (t >= 0 && t < e.length) {
      const s = e[t];
      return s.$context || (s.$context = Dc(this.getContext(), t, s));
    }
    return this.$context || (this.$context = Pc(this.chart.getContext(), this));
  }
  _tickSize() {
    const t = this.options.ticks, e = gt(this.labelRotation), s = Math.abs(Math.cos(e)), n = Math.abs(Math.sin(e)), o = this._getLabelSizes(), r = t.autoSkipPadding || 0, a = o ? o.widest.width + r : 0, l = o ? o.highest.height + r : 0;
    return this.isHorizontal() ? l * s > a * n ? a / s : l / n : l * n < a * s ? l / s : a / n;
  }
  _isVisible() {
    const t = this.options.display;
    return t !== "auto" ? !!t : this.getMatchingVisibleMetas().length > 0;
  }
  _computeGridLineItems(t) {
    const e = this.axis, s = this.chart, n = this.options, { grid: o, position: r, border: a } = n, l = o.offset, c = this.isHorizontal(), d = this.ticks.length + (l ? 1 : 0), f = me(o), u = [], g = a.setContext(this.getContext()), p = g.display ? g.width : 0, m = p / 2, b = function(W) {
      return Yt(s, W, p);
    };
    let _, y, v, x, S, k, w, P, T, L, R, q;
    if (r === "top")
      _ = b(this.bottom), k = this.bottom - f, P = _ - m, L = b(t.top) + m, q = t.bottom;
    else if (r === "bottom")
      _ = b(this.top), L = t.top, q = b(t.bottom) - m, k = _ + m, P = this.top + f;
    else if (r === "left")
      _ = b(this.right), S = this.right - f, w = _ - m, T = b(t.left) + m, R = t.right;
    else if (r === "right")
      _ = b(this.left), T = t.left, R = b(t.right) - m, S = _ + m, w = this.left + f;
    else if (e === "x") {
      if (r === "center")
        _ = b((t.top + t.bottom) / 2 + 0.5);
      else if (I(r)) {
        const W = Object.keys(r)[0], $ = r[W];
        _ = b(this.chart.scales[W].getPixelForValue($));
      }
      L = t.top, q = t.bottom, k = _ + m, P = k + f;
    } else if (e === "y") {
      if (r === "center")
        _ = b((t.left + t.right) / 2);
      else if (I(r)) {
        const W = Object.keys(r)[0], $ = r[W];
        _ = b(this.chart.scales[W].getPixelForValue($));
      }
      S = _ - m, w = S - f, T = t.left, R = t.right;
    }
    const it = O(n.ticks.maxTicksLimit, d), B = Math.max(1, Math.ceil(d / it));
    for (y = 0; y < d; y += B) {
      const W = this.getContext(y), $ = o.setContext(W), at = a.setContext(W), H = $.lineWidth, ht = $.color, jt = at.dash || [], Mt = at.dashOffset, Lt = $.tickWidth, bt = $.tickColor, Rt = $.tickBorderDash || [], _t = $.tickBorderDashOffset;
      v = wc(this, y, l), v !== void 0 && (x = Yt(s, v, H), c ? S = w = T = R = x : k = P = L = q = x, u.push({
        tx1: S,
        ty1: k,
        tx2: w,
        ty2: P,
        x1: T,
        y1: L,
        x2: R,
        y2: q,
        width: H,
        color: ht,
        borderDash: jt,
        borderDashOffset: Mt,
        tickWidth: Lt,
        tickColor: bt,
        tickBorderDash: Rt,
        tickBorderDashOffset: _t
      }));
    }
    return this._ticksLength = d, this._borderValue = _, u;
  }
  _computeLabelItems(t) {
    const e = this.axis, s = this.options, { position: n, ticks: o } = s, r = this.isHorizontal(), a = this.ticks, { align: l, crossAlign: c, padding: h, mirror: d } = o, f = me(s.grid), u = f + h, g = d ? -h : u, p = -gt(this.labelRotation), m = [];
    let b, _, y, v, x, S, k, w, P, T, L, R, q = "middle";
    if (n === "top")
      S = this.bottom - g, k = this._getXAxisLabelAlignment();
    else if (n === "bottom")
      S = this.top + g, k = this._getXAxisLabelAlignment();
    else if (n === "left") {
      const B = this._getYAxisLabelAlignment(f);
      k = B.textAlign, x = B.x;
    } else if (n === "right") {
      const B = this._getYAxisLabelAlignment(f);
      k = B.textAlign, x = B.x;
    } else if (e === "x") {
      if (n === "center")
        S = (t.top + t.bottom) / 2 + u;
      else if (I(n)) {
        const B = Object.keys(n)[0], W = n[B];
        S = this.chart.scales[B].getPixelForValue(W) + u;
      }
      k = this._getXAxisLabelAlignment();
    } else if (e === "y") {
      if (n === "center")
        x = (t.left + t.right) / 2 - u;
      else if (I(n)) {
        const B = Object.keys(n)[0], W = n[B];
        x = this.chart.scales[B].getPixelForValue(W);
      }
      k = this._getYAxisLabelAlignment(f).textAlign;
    }
    e === "y" && (l === "start" ? q = "top" : l === "end" && (q = "bottom"));
    const it = this._getLabelSizes();
    for (b = 0, _ = a.length; b < _; ++b) {
      y = a[b], v = y.label;
      const B = o.setContext(this.getContext(b));
      w = this.getPixelForTick(b) + o.labelOffset, P = this._resolveTickFontOptions(b), T = P.lineHeight, L = Y(v) ? v.length : 1;
      const W = L / 2, $ = B.color, at = B.textStrokeColor, H = B.textStrokeWidth;
      let ht = k;
      r ? (x = w, k === "inner" && (b === _ - 1 ? ht = this.options.reverse ? "left" : "right" : b === 0 ? ht = this.options.reverse ? "right" : "left" : ht = "center"), n === "top" ? c === "near" || p !== 0 ? R = -L * T + T / 2 : c === "center" ? R = -it.highest.height / 2 - W * T + T : R = -it.highest.height + T / 2 : c === "near" || p !== 0 ? R = T / 2 : c === "center" ? R = it.highest.height / 2 - W * T : R = it.highest.height - L * T, d && (R *= -1), p !== 0 && !B.showLabelBackdrop && (x += T / 2 * Math.sin(p))) : (S = w, R = (1 - L) * T / 2);
      let jt;
      if (B.showLabelBackdrop) {
        const Mt = rt(B.backdropPadding), Lt = it.heights[b], bt = it.widths[b];
        let Rt = R - Mt.top, _t = 0 - Mt.left;
        switch (q) {
          case "middle":
            Rt -= Lt / 2;
            break;
          case "bottom":
            Rt -= Lt;
            break;
        }
        switch (k) {
          case "center":
            _t -= bt / 2;
            break;
          case "right":
            _t -= bt;
            break;
          case "inner":
            b === _ - 1 ? _t -= bt : b > 0 && (_t -= bt / 2);
            break;
        }
        jt = {
          left: _t,
          top: Rt,
          width: bt + Mt.width,
          height: Lt + Mt.height,
          color: B.backdropColor
        };
      }
      m.push({
        label: v,
        font: P,
        textOffset: R,
        options: {
          rotation: p,
          color: $,
          strokeColor: at,
          strokeWidth: H,
          textAlign: ht,
          textBaseline: q,
          translation: [
            x,
            S
          ],
          backdrop: jt
        }
      });
    }
    return m;
  }
  _getXAxisLabelAlignment() {
    const { position: t, ticks: e } = this.options;
    if (-gt(this.labelRotation))
      return t === "top" ? "left" : "right";
    let n = "center";
    return e.align === "start" ? n = "left" : e.align === "end" ? n = "right" : e.align === "inner" && (n = "inner"), n;
  }
  _getYAxisLabelAlignment(t) {
    const { position: e, ticks: { crossAlign: s, mirror: n, padding: o } } = this.options, r = this._getLabelSizes(), a = t + o, l = r.widest.width;
    let c, h;
    return e === "left" ? n ? (h = this.right + o, s === "near" ? c = "left" : s === "center" ? (c = "center", h += l / 2) : (c = "right", h += l)) : (h = this.right - a, s === "near" ? c = "right" : s === "center" ? (c = "center", h -= l / 2) : (c = "left", h = this.left)) : e === "right" ? n ? (h = this.left + o, s === "near" ? c = "right" : s === "center" ? (c = "center", h -= l / 2) : (c = "left", h -= l)) : (h = this.left + a, s === "near" ? c = "left" : s === "center" ? (c = "center", h += l / 2) : (c = "right", h = this.right)) : c = "right", {
      textAlign: c,
      x: h
    };
  }
  _computeLabelArea() {
    if (this.options.ticks.mirror)
      return;
    const t = this.chart, e = this.options.position;
    if (e === "left" || e === "right")
      return {
        top: 0,
        left: this.left,
        bottom: t.height,
        right: this.right
      };
    if (e === "top" || e === "bottom")
      return {
        top: this.top,
        left: 0,
        bottom: this.bottom,
        right: t.width
      };
  }
  drawBackground() {
    const { ctx: t, options: { backgroundColor: e }, left: s, top: n, width: o, height: r } = this;
    e && (t.save(), t.fillStyle = e, t.fillRect(s, n, o, r), t.restore());
  }
  getLineWidthForValue(t) {
    const e = this.options.grid;
    if (!this._isVisible() || !e.display)
      return 0;
    const n = this.ticks.findIndex((o) => o.value === t);
    return n >= 0 ? e.setContext(this.getContext(n)).lineWidth : 0;
  }
  drawGrid(t) {
    const e = this.options.grid, s = this.ctx, n = this._gridLineItems || (this._gridLineItems = this._computeGridLineItems(t));
    let o, r;
    const a = (l, c, h) => {
      !h.width || !h.color || (s.save(), s.lineWidth = h.width, s.strokeStyle = h.color, s.setLineDash(h.borderDash || []), s.lineDashOffset = h.borderDashOffset, s.beginPath(), s.moveTo(l.x, l.y), s.lineTo(c.x, c.y), s.stroke(), s.restore());
    };
    if (e.display)
      for (o = 0, r = n.length; o < r; ++o) {
        const l = n[o];
        e.drawOnChartArea && a({
          x: l.x1,
          y: l.y1
        }, {
          x: l.x2,
          y: l.y2
        }, l), e.drawTicks && a({
          x: l.tx1,
          y: l.ty1
        }, {
          x: l.tx2,
          y: l.ty2
        }, {
          color: l.tickColor,
          width: l.tickWidth,
          borderDash: l.tickBorderDash,
          borderDashOffset: l.tickBorderDashOffset
        });
      }
  }
  drawBorder() {
    const { chart: t, ctx: e, options: { border: s, grid: n } } = this, o = s.setContext(this.getContext()), r = s.display ? o.width : 0;
    if (!r)
      return;
    const a = n.setContext(this.getContext(0)).lineWidth, l = this._borderValue;
    let c, h, d, f;
    this.isHorizontal() ? (c = Yt(t, this.left, r) - r / 2, h = Yt(t, this.right, a) + a / 2, d = f = l) : (d = Yt(t, this.top, r) - r / 2, f = Yt(t, this.bottom, a) + a / 2, c = h = l), e.save(), e.lineWidth = o.width, e.strokeStyle = o.color, e.beginPath(), e.moveTo(c, d), e.lineTo(h, f), e.stroke(), e.restore();
  }
  drawLabels(t) {
    if (!this.options.ticks.display)
      return;
    const s = this.ctx, n = this._computeLabelArea();
    n && Di(s, n);
    const o = this.getLabelItems(t);
    for (const r of o) {
      const a = r.options, l = r.font, c = r.label, h = r.textOffset;
      ee(s, c, 0, h, l, a);
    }
    n && Ai(s);
  }
  drawTitle() {
    const { ctx: t, options: { position: e, title: s, reverse: n } } = this;
    if (!s.display)
      return;
    const o = G(s.font), r = rt(s.padding), a = s.align;
    let l = o.lineHeight / 2;
    e === "bottom" || e === "center" || I(e) ? (l += r.bottom, Y(s.text) && (l += o.lineHeight * (s.text.length - 1))) : l += r.top;
    const { titleX: c, titleY: h, maxWidth: d, rotation: f } = Cc(this, l, e, a);
    ee(t, s.text, 0, 0, o, {
      color: s.color,
      maxWidth: d,
      rotation: f,
      textAlign: Ac(a, e, n),
      textBaseline: "middle",
      translation: [
        c,
        h
      ]
    });
  }
  draw(t) {
    this._isVisible() && (this.drawBackground(), this.drawGrid(t), this.drawBorder(), this.drawTitle(), this.drawLabels(t));
  }
  _layers() {
    const t = this.options, e = t.ticks && t.ticks.z || 0, s = O(t.grid && t.grid.z, -1), n = O(t.border && t.border.z, 0);
    return !this._isVisible() || this.draw !== ie.prototype.draw ? [
      {
        z: e,
        draw: (o) => {
          this.draw(o);
        }
      }
    ] : [
      {
        z: s,
        draw: (o) => {
          this.drawBackground(), this.drawGrid(o), this.drawTitle();
        }
      },
      {
        z: n,
        draw: () => {
          this.drawBorder();
        }
      },
      {
        z: e,
        draw: (o) => {
          this.drawLabels(o);
        }
      }
    ];
  }
  getMatchingVisibleMetas(t) {
    const e = this.chart.getSortedVisibleDatasetMetas(), s = this.axis + "AxisID", n = [];
    let o, r;
    for (o = 0, r = e.length; o < r; ++o) {
      const a = e[o];
      a[s] === this.id && (!t || a.type === t) && n.push(a);
    }
    return n;
  }
  _resolveTickFontOptions(t) {
    const e = this.options.ticks.setContext(this.getContext(t));
    return G(e.font);
  }
  _maxDigits() {
    const t = this._resolveTickFontOptions(0).lineHeight;
    return (this.isHorizontal() ? this.width : this.height) / t;
  }
}
class Ze {
  constructor(t, e, s) {
    this.type = t, this.scope = e, this.override = s, this.items = /* @__PURE__ */ Object.create(null);
  }
  isForType(t) {
    return Object.prototype.isPrototypeOf.call(this.type.prototype, t.prototype);
  }
  register(t) {
    const e = Object.getPrototypeOf(t);
    let s;
    Lc(e) && (s = this.register(e));
    const n = this.items, o = t.id, r = this.scope + "." + o;
    if (!o)
      throw new Error("class does not have id: " + t);
    return o in n || (n[o] = t, Oc(t, r, s), this.override && X.override(t.id, t.overrides)), r;
  }
  get(t) {
    return this.items[t];
  }
  unregister(t) {
    const e = this.items, s = t.id, n = this.scope;
    s in e && delete e[s], n && s in X[n] && (delete X[n][s], this.override && delete te[s]);
  }
}
function Oc(i, t, e) {
  const s = Re(/* @__PURE__ */ Object.create(null), [
    e ? X.get(e) : {},
    X.get(t),
    i.defaults
  ]);
  X.set(t, s), i.defaultRoutes && Tc(t, i.defaultRoutes), i.descriptors && X.describe(t, i.descriptors);
}
function Tc(i, t) {
  Object.keys(t).forEach((e) => {
    const s = e.split("."), n = s.pop(), o = [
      i
    ].concat(s).join("."), r = t[e].split("."), a = r.pop(), l = r.join(".");
    X.route(o, n, l, a);
  });
}
function Lc(i) {
  return "id" in i && "defaults" in i;
}
class Rc {
  constructor() {
    this.controllers = new Ze(pt, "datasets", !0), this.elements = new Ze(mt, "elements"), this.plugins = new Ze(Object, "plugins"), this.scales = new Ze(ie, "scales"), this._typedRegistries = [
      this.controllers,
      this.scales,
      this.elements
    ];
  }
  add(...t) {
    this._each("register", t);
  }
  remove(...t) {
    this._each("unregister", t);
  }
  addControllers(...t) {
    this._each("register", t, this.controllers);
  }
  addElements(...t) {
    this._each("register", t, this.elements);
  }
  addPlugins(...t) {
    this._each("register", t, this.plugins);
  }
  addScales(...t) {
    this._each("register", t, this.scales);
  }
  getController(t) {
    return this._get(t, this.controllers, "controller");
  }
  getElement(t) {
    return this._get(t, this.elements, "element");
  }
  getPlugin(t) {
    return this._get(t, this.plugins, "plugin");
  }
  getScale(t) {
    return this._get(t, this.scales, "scale");
  }
  removeControllers(...t) {
    this._each("unregister", t, this.controllers);
  }
  removeElements(...t) {
    this._each("unregister", t, this.elements);
  }
  removePlugins(...t) {
    this._each("unregister", t, this.plugins);
  }
  removeScales(...t) {
    this._each("unregister", t, this.scales);
  }
  _each(t, e, s) {
    [
      ...e
    ].forEach((n) => {
      const o = s || this._getRegistryForType(n);
      s || o.isForType(n) || o === this.plugins && n.id ? this._exec(t, o, n) : V(n, (r) => {
        const a = s || this._getRegistryForType(r);
        this._exec(t, a, r);
      });
    });
  }
  _exec(t, e, s) {
    const n = xs(t);
    N(s["before" + n], [], s), e[t](s), N(s["after" + n], [], s);
  }
  _getRegistryForType(t) {
    for (let e = 0; e < this._typedRegistries.length; e++) {
      const s = this._typedRegistries[e];
      if (s.isForType(t))
        return s;
    }
    return this.plugins;
  }
  _get(t, e, s) {
    const n = e.get(t);
    if (n === void 0)
      throw new Error('"' + t + '" is not a registered ' + s + ".");
    return n;
  }
}
var yt = /* @__PURE__ */ new Rc();
class Ec {
  constructor() {
    this._init = void 0;
  }
  notify(t, e, s, n) {
    if (e === "beforeInit" && (this._init = this._createDescriptors(t, !0), this._notify(this._init, t, "install")), this._init === void 0)
      return;
    const o = n ? this._descriptors(t).filter(n) : this._descriptors(t), r = this._notify(o, t, e, s);
    return e === "afterDestroy" && (this._notify(o, t, "stop"), this._notify(this._init, t, "uninstall"), this._init = void 0), r;
  }
  _notify(t, e, s, n) {
    n = n || {};
    for (const o of t) {
      const r = o.plugin, a = r[s], l = [
        e,
        n,
        o.options
      ];
      if (N(a, l, r) === !1 && n.cancelable)
        return !1;
    }
    return !0;
  }
  invalidate() {
    F(this._cache) || (this._oldCache = this._cache, this._cache = void 0);
  }
  _descriptors(t) {
    if (this._cache)
      return this._cache;
    const e = this._cache = this._createDescriptors(t);
    return this._notifyStateChanges(t), e;
  }
  _createDescriptors(t, e) {
    const s = t && t.config, n = O(s.options && s.options.plugins, {}), o = Fc(s);
    return n === !1 && !e ? [] : zc(t, o, n, e);
  }
  _notifyStateChanges(t) {
    const e = this._oldCache || [], s = this._cache, n = (o, r) => o.filter((a) => !r.some((l) => a.plugin.id === l.plugin.id));
    this._notify(n(e, s), t, "stop"), this._notify(n(s, e), t, "start");
  }
}
function Fc(i) {
  const t = {}, e = [], s = Object.keys(yt.plugins.items);
  for (let o = 0; o < s.length; o++)
    e.push(yt.getPlugin(s[o]));
  const n = i.plugins || [];
  for (let o = 0; o < n.length; o++) {
    const r = n[o];
    e.indexOf(r) === -1 && (e.push(r), t[r.id] = !0);
  }
  return {
    plugins: e,
    localIds: t
  };
}
function Ic(i, t) {
  return !t && i === !1 ? null : i === !0 ? {} : i;
}
function zc(i, { plugins: t, localIds: e }, s, n) {
  const o = [], r = i.getContext();
  for (const a of t) {
    const l = a.id, c = Ic(s[l], n);
    c !== null && o.push({
      plugin: a,
      options: Bc(i.config, {
        plugin: a,
        local: e[l]
      }, c, r)
    });
  }
  return o;
}
function Bc(i, { plugin: t, local: e }, s, n) {
  const o = i.pluginScopeKeys(t), r = i.getOptionScopes(s, o);
  return e && t.defaults && r.push(t.defaults), i.createResolver(r, n, [
    ""
  ], {
    scriptable: !1,
    indexable: !1,
    allKeys: !0
  });
}
function ns(i, t) {
  const e = X.datasets[i] || {};
  return ((t.datasets || {})[i] || {}).indexAxis || t.indexAxis || e.indexAxis || "x";
}
function Vc(i, t) {
  let e = i;
  return i === "_index_" ? e = t : i === "_value_" && (e = t === "x" ? "y" : "x"), e;
}
function Wc(i, t) {
  return i === t ? "_index_" : "_value_";
}
function wn(i) {
  if (i === "x" || i === "y" || i === "r")
    return i;
}
function Nc(i) {
  if (i === "top" || i === "bottom")
    return "x";
  if (i === "left" || i === "right")
    return "y";
}
function os(i, ...t) {
  if (wn(i))
    return i;
  for (const e of t) {
    const s = e.axis || Nc(e.position) || i.length > 1 && wn(i[0].toLowerCase());
    if (s)
      return s;
  }
  throw new Error(`Cannot determine type of '${i}' axis. Please provide 'axis' or 'position' option.`);
}
function Sn(i, t, e) {
  if (e[t + "AxisID"] === i)
    return {
      axis: t
    };
}
function Hc(i, t) {
  if (t.data && t.data.datasets) {
    const e = t.data.datasets.filter((s) => s.xAxisID === i || s.yAxisID === i);
    if (e.length)
      return Sn(i, "x", e[0]) || Sn(i, "y", e[0]);
  }
  return {};
}
function jc(i, t) {
  const e = te[i.type] || {
    scales: {}
  }, s = t.scales || {}, n = ns(i.type, t), o = /* @__PURE__ */ Object.create(null);
  return Object.keys(s).forEach((r) => {
    const a = s[r];
    if (!I(a))
      return console.error(`Invalid scale configuration for scale: ${r}`);
    if (a._proxy)
      return console.warn(`Ignoring resolver passed as options for scale: ${r}`);
    const l = os(r, a, Hc(r, i), X.scales[a.type]), c = Wc(l, n), h = e.scales || {};
    o[r] = Pe(/* @__PURE__ */ Object.create(null), [
      {
        axis: l
      },
      a,
      h[l],
      h[c]
    ]);
  }), i.data.datasets.forEach((r) => {
    const a = r.type || i.type, l = r.indexAxis || ns(a, t), h = (te[a] || {}).scales || {};
    Object.keys(h).forEach((d) => {
      const f = Vc(d, l), u = r[f + "AxisID"] || f;
      o[u] = o[u] || /* @__PURE__ */ Object.create(null), Pe(o[u], [
        {
          axis: f
        },
        s[u],
        h[d]
      ]);
    });
  }), Object.keys(o).forEach((r) => {
    const a = o[r];
    Pe(a, [
      X.scales[a.type],
      X.scale
    ]);
  }), o;
}
function tr(i) {
  const t = i.options || (i.options = {});
  t.plugins = O(t.plugins, {}), t.scales = jc(i, t);
}
function er(i) {
  return i = i || {}, i.datasets = i.datasets || [], i.labels = i.labels || [], i;
}
function $c(i) {
  return i = i || {}, i.data = er(i.data), tr(i), i;
}
const Pn = /* @__PURE__ */ new Map(), ir = /* @__PURE__ */ new Set();
function Je(i, t) {
  let e = Pn.get(i);
  return e || (e = t(), Pn.set(i, e), ir.add(e)), e;
}
const be = (i, t, e) => {
  const s = Wt(t, e);
  s !== void 0 && i.add(s);
};
class Yc {
  constructor(t) {
    this._config = $c(t), this._scopeCache = /* @__PURE__ */ new Map(), this._resolverCache = /* @__PURE__ */ new Map();
  }
  get platform() {
    return this._config.platform;
  }
  get type() {
    return this._config.type;
  }
  set type(t) {
    this._config.type = t;
  }
  get data() {
    return this._config.data;
  }
  set data(t) {
    this._config.data = er(t);
  }
  get options() {
    return this._config.options;
  }
  set options(t) {
    this._config.options = t;
  }
  get plugins() {
    return this._config.plugins;
  }
  update() {
    const t = this._config;
    this.clearCache(), tr(t);
  }
  clearCache() {
    this._scopeCache.clear(), this._resolverCache.clear();
  }
  datasetScopeKeys(t) {
    return Je(t, () => [
      [
        `datasets.${t}`,
        ""
      ]
    ]);
  }
  datasetAnimationScopeKeys(t, e) {
    return Je(`${t}.transition.${e}`, () => [
      [
        `datasets.${t}.transitions.${e}`,
        `transitions.${e}`
      ],
      [
        `datasets.${t}`,
        ""
      ]
    ]);
  }
  datasetElementScopeKeys(t, e) {
    return Je(`${t}-${e}`, () => [
      [
        `datasets.${t}.elements.${e}`,
        `datasets.${t}`,
        `elements.${e}`,
        ""
      ]
    ]);
  }
  pluginScopeKeys(t) {
    const e = t.id, s = this.type;
    return Je(`${s}-plugin-${e}`, () => [
      [
        `plugins.${e}`,
        ...t.additionalOptionScopes || []
      ]
    ]);
  }
  _cachedScopes(t, e) {
    const s = this._scopeCache;
    let n = s.get(t);
    return (!n || e) && (n = /* @__PURE__ */ new Map(), s.set(t, n)), n;
  }
  getOptionScopes(t, e, s) {
    const { options: n, type: o } = this, r = this._cachedScopes(t, s), a = r.get(e);
    if (a)
      return a;
    const l = /* @__PURE__ */ new Set();
    e.forEach((h) => {
      t && (l.add(t), h.forEach((d) => be(l, t, d))), h.forEach((d) => be(l, n, d)), h.forEach((d) => be(l, te[o] || {}, d)), h.forEach((d) => be(l, X, d)), h.forEach((d) => be(l, es, d));
    });
    const c = Array.from(l);
    return c.length === 0 && c.push(/* @__PURE__ */ Object.create(null)), ir.has(e) && r.set(e, c), c;
  }
  chartOptionScopes() {
    const { options: t, type: e } = this;
    return [
      t,
      te[e] || {},
      X.datasets[e] || {},
      {
        type: e
      },
      X,
      es
    ];
  }
  resolveNamedOptions(t, e, s, n = [
    ""
  ]) {
    const o = {
      $shared: !0
    }, { resolver: r, subPrefixes: a } = Dn(this._resolverCache, t, n);
    let l = r;
    if (Uc(r, e)) {
      o.$shared = !1, s = Nt(s) ? s() : s;
      const c = this.createResolver(t, s, a);
      l = de(r, s, c);
    }
    for (const c of e)
      o[c] = l[c];
    return o;
  }
  createResolver(t, e, s = [
    ""
  ], n) {
    const { resolver: o } = Dn(this._resolverCache, t, s);
    return I(e) ? de(o, e, void 0, n) : o;
  }
}
function Dn(i, t, e) {
  let s = i.get(t);
  s || (s = /* @__PURE__ */ new Map(), i.set(t, s));
  const n = e.join();
  let o = s.get(n);
  return o || (o = {
    resolver: Ss(t, e),
    subPrefixes: e.filter((a) => !a.toLowerCase().includes("hover"))
  }, s.set(n, o)), o;
}
const Xc = (i) => I(i) && Object.getOwnPropertyNames(i).some((t) => Nt(i[t]));
function Uc(i, t) {
  const { isScriptable: e, isIndexable: s } = Eo(i);
  for (const n of t) {
    const o = e(n), r = s(n), a = (r || o) && i[n];
    if (o && (Nt(a) || Xc(a)) || r && Y(a))
      return !0;
  }
  return !1;
}
var Kc = "4.5.1";
const qc = [
  "top",
  "bottom",
  "left",
  "right",
  "chartArea"
];
function An(i, t) {
  return i === "top" || i === "bottom" || qc.indexOf(i) === -1 && t === "x";
}
function Cn(i, t) {
  return function(e, s) {
    return e[i] === s[i] ? e[t] - s[t] : e[i] - s[i];
  };
}
function On(i) {
  const t = i.chart, e = t.options.animation;
  t.notifyPlugins("afterRender"), N(e && e.onComplete, [
    i
  ], t);
}
function Gc(i) {
  const t = i.chart, e = t.options.animation;
  N(e && e.onProgress, [
    i
  ], t);
}
function sr(i) {
  return As() && typeof i == "string" ? i = document.getElementById(i) : i && i.length && (i = i[0]), i && i.canvas && (i = i.canvas), i;
}
const fi = {}, Tn = (i) => {
  const t = sr(i);
  return Object.values(fi).filter((e) => e.canvas === t).pop();
};
function Zc(i, t, e) {
  const s = Object.keys(i);
  for (const n of s) {
    const o = +n;
    if (o >= t) {
      const r = i[n];
      delete i[n], (e > 0 || o > t) && (i[o + e] = r);
    }
  }
}
function Jc(i, t, e, s) {
  return !e || i.type === "mouseout" ? null : s ? t : i;
}
class At {
  static register(...t) {
    yt.add(...t), Ln();
  }
  static unregister(...t) {
    yt.remove(...t), Ln();
  }
  constructor(t, e) {
    const s = this.config = new Yc(e), n = sr(t), o = Tn(n);
    if (o)
      throw new Error("Canvas is already in use. Chart with ID '" + o.id + "' must be destroyed before the canvas with ID '" + o.canvas.id + "' can be reused.");
    const r = s.createResolver(s.chartOptionScopes(), this.getContext());
    this.platform = new (s.platform || mc(n))(), this.platform.updateConfig(s);
    const a = this.platform.acquireContext(n, r.aspectRatio), l = a && a.canvas, c = l && l.height, h = l && l.width;
    if (this.id = Qr(), this.ctx = a, this.canvas = l, this.width = h, this.height = c, this._options = r, this._aspectRatio = this.aspectRatio, this._layers = [], this._metasets = [], this._stacks = void 0, this.boxes = [], this.currentDevicePixelRatio = void 0, this.chartArea = void 0, this._active = [], this._lastEvent = void 0, this._listeners = {}, this._responsiveListeners = void 0, this._sortedMetasets = [], this.scales = {}, this._plugins = new Ec(), this.$proxies = {}, this._hiddenIndices = {}, this.attached = !1, this._animationsDisabled = void 0, this.$context = void 0, this._doResize = ma((d) => this.update(d), r.resizeDelay || 0), this._dataChanges = [], fi[this.id] = this, !a || !l) {
      console.error("Failed to create chart: can't acquire context from the given item");
      return;
    }
    St.listen(this, "complete", On), St.listen(this, "progress", Gc), this._initialize(), this.attached && this.update();
  }
  get aspectRatio() {
    const { options: { aspectRatio: t, maintainAspectRatio: e }, width: s, height: n, _aspectRatio: o } = this;
    return F(t) ? e && o ? o : n ? s / n : null : t;
  }
  get data() {
    return this.config.data;
  }
  set data(t) {
    this.config.data = t;
  }
  get options() {
    return this._options;
  }
  set options(t) {
    this.config.options = t;
  }
  get registry() {
    return yt;
  }
  _initialize() {
    return this.notifyPlugins("beforeInit"), this.options.responsive ? this.resize() : Qs(this, this.options.devicePixelRatio), this.bindEvents(), this.notifyPlugins("afterInit"), this;
  }
  clear() {
    return Gs(this.canvas, this.ctx), this;
  }
  stop() {
    return St.stop(this), this;
  }
  resize(t, e) {
    St.running(this) ? this._resizeBeforeDraw = {
      width: t,
      height: e
    } : this._resize(t, e);
  }
  _resize(t, e) {
    const s = this.options, n = this.canvas, o = s.maintainAspectRatio && this.aspectRatio, r = this.platform.getMaximumSize(n, t, e, o), a = s.devicePixelRatio || this.platform.getDevicePixelRatio(), l = this.width ? "resize" : "attach";
    this.width = r.width, this.height = r.height, this._aspectRatio = this.aspectRatio, Qs(this, a, !0) && (this.notifyPlugins("resize", {
      size: r
    }), N(s.onResize, [
      this,
      r
    ], this), this.attached && this._doResize(l) && this.render());
  }
  ensureScalesHaveIDs() {
    const e = this.options.scales || {};
    V(e, (s, n) => {
      s.id = n;
    });
  }
  buildOrUpdateScales() {
    const t = this.options, e = t.scales, s = this.scales, n = Object.keys(s).reduce((r, a) => (r[a] = !1, r), {});
    let o = [];
    e && (o = o.concat(Object.keys(e).map((r) => {
      const a = e[r], l = os(r, a), c = l === "r", h = l === "x";
      return {
        options: a,
        dposition: c ? "chartArea" : h ? "bottom" : "left",
        dtype: c ? "radialLinear" : h ? "category" : "linear"
      };
    }))), V(o, (r) => {
      const a = r.options, l = a.id, c = os(l, a), h = O(a.type, r.dtype);
      (a.position === void 0 || An(a.position, c) !== An(r.dposition)) && (a.position = r.dposition), n[l] = !0;
      let d = null;
      if (l in s && s[l].type === h)
        d = s[l];
      else {
        const f = yt.getScale(h);
        d = new f({
          id: l,
          type: h,
          ctx: this.ctx,
          chart: this
        }), s[d.id] = d;
      }
      d.init(a, t);
    }), V(n, (r, a) => {
      r || delete s[a];
    }), V(s, (r) => {
      ot.configure(this, r, r.options), ot.addBox(this, r);
    });
  }
  _updateMetasets() {
    const t = this._metasets, e = this.data.datasets.length, s = t.length;
    if (t.sort((n, o) => n.index - o.index), s > e) {
      for (let n = e; n < s; ++n)
        this._destroyDatasetMeta(n);
      t.splice(e, s - e);
    }
    this._sortedMetasets = t.slice(0).sort(Cn("order", "index"));
  }
  _removeUnreferencedMetasets() {
    const { _metasets: t, data: { datasets: e } } = this;
    t.length > e.length && delete this._stacks, t.forEach((s, n) => {
      e.filter((o) => o === s._dataset).length === 0 && this._destroyDatasetMeta(n);
    });
  }
  buildOrUpdateControllers() {
    const t = [], e = this.data.datasets;
    let s, n;
    for (this._removeUnreferencedMetasets(), s = 0, n = e.length; s < n; s++) {
      const o = e[s];
      let r = this.getDatasetMeta(s);
      const a = o.type || this.config.type;
      if (r.type && r.type !== a && (this._destroyDatasetMeta(s), r = this.getDatasetMeta(s)), r.type = a, r.indexAxis = o.indexAxis || ns(a, this.options), r.order = o.order || 0, r.index = s, r.label = "" + o.label, r.visible = this.isDatasetVisible(s), r.controller)
        r.controller.updateIndex(s), r.controller.linkScales();
      else {
        const l = yt.getController(a), { datasetElementType: c, dataElementType: h } = X.datasets[a];
        Object.assign(l, {
          dataElementType: yt.getElement(h),
          datasetElementType: c && yt.getElement(c)
        }), r.controller = new l(this, s), t.push(r.controller);
      }
    }
    return this._updateMetasets(), t;
  }
  _resetElements() {
    V(this.data.datasets, (t, e) => {
      this.getDatasetMeta(e).controller.reset();
    }, this);
  }
  reset() {
    this._resetElements(), this.notifyPlugins("reset");
  }
  update(t) {
    const e = this.config;
    e.update();
    const s = this._options = e.createResolver(e.chartOptionScopes(), this.getContext()), n = this._animationsDisabled = !s.animation;
    if (this._updateScales(), this._checkEventBindings(), this._updateHiddenIndices(), this._plugins.invalidate(), this.notifyPlugins("beforeUpdate", {
      mode: t,
      cancelable: !0
    }) === !1)
      return;
    const o = this.buildOrUpdateControllers();
    this.notifyPlugins("beforeElementsUpdate");
    let r = 0;
    for (let c = 0, h = this.data.datasets.length; c < h; c++) {
      const { controller: d } = this.getDatasetMeta(c), f = !n && o.indexOf(d) === -1;
      d.buildOrUpdateElements(f), r = Math.max(+d.getMaxOverflow(), r);
    }
    r = this._minPadding = s.layout.autoPadding ? r : 0, this._updateLayout(r), n || V(o, (c) => {
      c.reset();
    }), this._updateDatasets(t), this.notifyPlugins("afterUpdate", {
      mode: t
    }), this._layers.sort(Cn("z", "_idx"));
    const { _active: a, _lastEvent: l } = this;
    l ? this._eventHandler(l, !0) : a.length && this._updateHoverStyles(a, a, !0), this.render();
  }
  _updateScales() {
    V(this.scales, (t) => {
      ot.removeBox(this, t);
    }), this.ensureScalesHaveIDs(), this.buildOrUpdateScales();
  }
  _checkEventBindings() {
    const t = this.options, e = new Set(Object.keys(this._listeners)), s = new Set(t.events);
    (!Ns(e, s) || !!this._responsiveListeners !== t.responsive) && (this.unbindEvents(), this.bindEvents());
  }
  _updateHiddenIndices() {
    const { _hiddenIndices: t } = this, e = this._getUniformDataChanges() || [];
    for (const { method: s, start: n, count: o } of e) {
      const r = s === "_removeElements" ? -o : o;
      Zc(t, n, r);
    }
  }
  _getUniformDataChanges() {
    const t = this._dataChanges;
    if (!t || !t.length)
      return;
    this._dataChanges = [];
    const e = this.data.datasets.length, s = (o) => new Set(t.filter((r) => r[0] === o).map((r, a) => a + "," + r.splice(1).join(","))), n = s(0);
    for (let o = 1; o < e; o++)
      if (!Ns(n, s(o)))
        return;
    return Array.from(n).map((o) => o.split(",")).map((o) => ({
      method: o[1],
      start: +o[2],
      count: +o[3]
    }));
  }
  _updateLayout(t) {
    if (this.notifyPlugins("beforeLayout", {
      cancelable: !0
    }) === !1)
      return;
    ot.update(this, this.width, this.height, t);
    const e = this.chartArea, s = e.width <= 0 || e.height <= 0;
    this._layers = [], V(this.boxes, (n) => {
      s && n.position === "chartArea" || (n.configure && n.configure(), this._layers.push(...n._layers()));
    }, this), this._layers.forEach((n, o) => {
      n._idx = o;
    }), this.notifyPlugins("afterLayout");
  }
  _updateDatasets(t) {
    if (this.notifyPlugins("beforeDatasetsUpdate", {
      mode: t,
      cancelable: !0
    }) !== !1) {
      for (let e = 0, s = this.data.datasets.length; e < s; ++e)
        this.getDatasetMeta(e).controller.configure();
      for (let e = 0, s = this.data.datasets.length; e < s; ++e)
        this._updateDataset(e, Nt(t) ? t({
          datasetIndex: e
        }) : t);
      this.notifyPlugins("afterDatasetsUpdate", {
        mode: t
      });
    }
  }
  _updateDataset(t, e) {
    const s = this.getDatasetMeta(t), n = {
      meta: s,
      index: t,
      mode: e,
      cancelable: !0
    };
    this.notifyPlugins("beforeDatasetUpdate", n) !== !1 && (s.controller._update(e), n.cancelable = !1, this.notifyPlugins("afterDatasetUpdate", n));
  }
  render() {
    this.notifyPlugins("beforeRender", {
      cancelable: !0
    }) !== !1 && (St.has(this) ? this.attached && !St.running(this) && St.start(this) : (this.draw(), On({
      chart: this
    })));
  }
  draw() {
    let t;
    if (this._resizeBeforeDraw) {
      const { width: s, height: n } = this._resizeBeforeDraw;
      this._resizeBeforeDraw = null, this._resize(s, n);
    }
    if (this.clear(), this.width <= 0 || this.height <= 0 || this.notifyPlugins("beforeDraw", {
      cancelable: !0
    }) === !1)
      return;
    const e = this._layers;
    for (t = 0; t < e.length && e[t].z <= 0; ++t)
      e[t].draw(this.chartArea);
    for (this._drawDatasets(); t < e.length; ++t)
      e[t].draw(this.chartArea);
    this.notifyPlugins("afterDraw");
  }
  _getSortedDatasetMetas(t) {
    const e = this._sortedMetasets, s = [];
    let n, o;
    for (n = 0, o = e.length; n < o; ++n) {
      const r = e[n];
      (!t || r.visible) && s.push(r);
    }
    return s;
  }
  getSortedVisibleDatasetMetas() {
    return this._getSortedDatasetMetas(!0);
  }
  _drawDatasets() {
    if (this.notifyPlugins("beforeDatasetsDraw", {
      cancelable: !0
    }) === !1)
      return;
    const t = this.getSortedVisibleDatasetMetas();
    for (let e = t.length - 1; e >= 0; --e)
      this._drawDataset(t[e]);
    this.notifyPlugins("afterDatasetsDraw");
  }
  _drawDataset(t) {
    const e = this.ctx, s = {
      meta: t,
      index: t.index,
      cancelable: !0
    }, n = Yo(this, t);
    this.notifyPlugins("beforeDatasetDraw", s) !== !1 && (n && Di(e, n), t.controller.draw(), n && Ai(e), s.cancelable = !1, this.notifyPlugins("afterDatasetDraw", s));
  }
  isPointInArea(t) {
    return Tt(t, this.chartArea, this._minPadding);
  }
  getElementsAtEventForMode(t, e, s, n) {
    const o = ql.modes[e];
    return typeof o == "function" ? o(this, t, s, n) : [];
  }
  getDatasetMeta(t) {
    const e = this.data.datasets[t], s = this._metasets;
    let n = s.filter((o) => o && o._dataset === e).pop();
    return n || (n = {
      type: null,
      data: [],
      dataset: null,
      controller: null,
      hidden: null,
      xAxisID: null,
      yAxisID: null,
      order: e && e.order || 0,
      index: t,
      _dataset: e,
      _parsed: [],
      _sorted: !1
    }, s.push(n)), n;
  }
  getContext() {
    return this.$context || (this.$context = Ht(null, {
      chart: this,
      type: "chart"
    }));
  }
  getVisibleDatasetCount() {
    return this.getSortedVisibleDatasetMetas().length;
  }
  isDatasetVisible(t) {
    const e = this.data.datasets[t];
    if (!e)
      return !1;
    const s = this.getDatasetMeta(t);
    return typeof s.hidden == "boolean" ? !s.hidden : !e.hidden;
  }
  setDatasetVisibility(t, e) {
    const s = this.getDatasetMeta(t);
    s.hidden = !e;
  }
  toggleDataVisibility(t) {
    this._hiddenIndices[t] = !this._hiddenIndices[t];
  }
  getDataVisibility(t) {
    return !this._hiddenIndices[t];
  }
  _updateVisibility(t, e, s) {
    const n = s ? "show" : "hide", o = this.getDatasetMeta(t), r = o.controller._resolveAnimations(void 0, n);
    Ee(e) ? (o.data[e].hidden = !s, this.update()) : (this.setDatasetVisibility(t, s), r.update(o, {
      visible: s
    }), this.update((a) => a.datasetIndex === t ? n : void 0));
  }
  hide(t, e) {
    this._updateVisibility(t, e, !1);
  }
  show(t, e) {
    this._updateVisibility(t, e, !0);
  }
  _destroyDatasetMeta(t) {
    const e = this._metasets[t];
    e && e.controller && e.controller._destroy(), delete this._metasets[t];
  }
  _stop() {
    let t, e;
    for (this.stop(), St.remove(this), t = 0, e = this.data.datasets.length; t < e; ++t)
      this._destroyDatasetMeta(t);
  }
  destroy() {
    this.notifyPlugins("beforeDestroy");
    const { canvas: t, ctx: e } = this;
    this._stop(), this.config.clearCache(), t && (this.unbindEvents(), Gs(t, e), this.platform.releaseContext(e), this.canvas = null, this.ctx = null), delete fi[this.id], this.notifyPlugins("afterDestroy");
  }
  toBase64Image(...t) {
    return this.canvas.toDataURL(...t);
  }
  bindEvents() {
    this.bindUserEvents(), this.options.responsive ? this.bindResponsiveEvents() : this.attached = !0;
  }
  bindUserEvents() {
    const t = this._listeners, e = this.platform, s = (o, r) => {
      e.addEventListener(this, o, r), t[o] = r;
    }, n = (o, r, a) => {
      o.offsetX = r, o.offsetY = a, this._eventHandler(o);
    };
    V(this.options.events, (o) => s(o, n));
  }
  bindResponsiveEvents() {
    this._responsiveListeners || (this._responsiveListeners = {});
    const t = this._responsiveListeners, e = this.platform, s = (l, c) => {
      e.addEventListener(this, l, c), t[l] = c;
    }, n = (l, c) => {
      t[l] && (e.removeEventListener(this, l, c), delete t[l]);
    }, o = (l, c) => {
      this.canvas && this.resize(l, c);
    };
    let r;
    const a = () => {
      n("attach", a), this.attached = !0, this.resize(), s("resize", o), s("detach", r);
    };
    r = () => {
      this.attached = !1, n("resize", o), this._stop(), this._resize(0, 0), s("attach", a);
    }, e.isAttached(this.canvas) ? a() : r();
  }
  unbindEvents() {
    V(this._listeners, (t, e) => {
      this.platform.removeEventListener(this, e, t);
    }), this._listeners = {}, V(this._responsiveListeners, (t, e) => {
      this.platform.removeEventListener(this, e, t);
    }), this._responsiveListeners = void 0;
  }
  updateHoverStyle(t, e, s) {
    const n = s ? "set" : "remove";
    let o, r, a, l;
    for (e === "dataset" && (o = this.getDatasetMeta(t[0].datasetIndex), o.controller["_" + n + "DatasetHoverStyle"]()), a = 0, l = t.length; a < l; ++a) {
      r = t[a];
      const c = r && this.getDatasetMeta(r.datasetIndex).controller;
      c && c[n + "HoverStyle"](r.element, r.datasetIndex, r.index);
    }
  }
  getActiveElements() {
    return this._active || [];
  }
  setActiveElements(t) {
    const e = this._active || [], s = t.map(({ datasetIndex: o, index: r }) => {
      const a = this.getDatasetMeta(o);
      if (!a)
        throw new Error("No dataset found at index " + o);
      return {
        datasetIndex: o,
        element: a.data[r],
        index: r
      };
    });
    !pi(s, e) && (this._active = s, this._lastEvent = null, this._updateHoverStyles(s, e));
  }
  notifyPlugins(t, e, s) {
    return this._plugins.notify(this, t, e, s);
  }
  isPluginEnabled(t) {
    return this._plugins._cache.filter((e) => e.plugin.id === t).length === 1;
  }
  _updateHoverStyles(t, e, s) {
    const n = this.options.hover, o = (l, c) => l.filter((h) => !c.some((d) => h.datasetIndex === d.datasetIndex && h.index === d.index)), r = o(e, t), a = s ? t : o(t, e);
    r.length && this.updateHoverStyle(r, n.mode, !1), a.length && n.mode && this.updateHoverStyle(a, n.mode, !0);
  }
  _eventHandler(t, e) {
    const s = {
      event: t,
      replay: e,
      cancelable: !0,
      inChartArea: this.isPointInArea(t)
    }, n = (r) => (r.options.events || this.options.events).includes(t.native.type);
    if (this.notifyPlugins("beforeEvent", s, n) === !1)
      return;
    const o = this._handleEvent(t, e, s.inChartArea);
    return s.cancelable = !1, this.notifyPlugins("afterEvent", s, n), (o || s.changed) && this.render(), this;
  }
  _handleEvent(t, e, s) {
    const { _active: n = [], options: o } = this, r = e, a = this._getActiveElements(t, n, s, r), l = oa(t), c = Jc(t, this._lastEvent, s, l);
    s && (this._lastEvent = null, N(o.onHover, [
      t,
      a,
      this
    ], this), l && N(o.onClick, [
      t,
      a,
      this
    ], this));
    const h = !pi(a, n);
    return (h || e) && (this._active = a, this._updateHoverStyles(a, n, e)), this._lastEvent = c, h;
  }
  _getActiveElements(t, e, s, n) {
    if (t.type === "mouseout")
      return [];
    if (!s)
      return e;
    const o = this.options.hover;
    return this.getElementsAtEventForMode(t, o.mode, o, n);
  }
}
M(At, "defaults", X), M(At, "instances", fi), M(At, "overrides", te), M(At, "registry", yt), M(At, "version", Kc), M(At, "getChart", Tn);
function Ln() {
  return V(At.instances, (i) => i._plugins.invalidate());
}
function Qc(i, t, e) {
  const { startAngle: s, x: n, y: o, outerRadius: r, innerRadius: a, options: l } = t, { borderWidth: c, borderJoinStyle: h } = l, d = Math.min(c / r, nt(s - e));
  if (i.beginPath(), i.arc(n, o, r - c / 2, s + d / 2, e - d / 2), a > 0) {
    const f = Math.min(c / a, nt(s - e));
    i.arc(n, o, a + c / 2, e - f / 2, s + f / 2, !0);
  } else {
    const f = Math.min(c / 2, r * nt(s - e));
    if (h === "round")
      i.arc(n, o, f, e - z / 2, s + z / 2, !0);
    else if (h === "bevel") {
      const u = 2 * f * f, g = -u * Math.cos(e + z / 2) + n, p = -u * Math.sin(e + z / 2) + o, m = u * Math.cos(s + z / 2) + n, b = u * Math.sin(s + z / 2) + o;
      i.lineTo(g, p), i.lineTo(m, b);
    }
  }
  i.closePath(), i.moveTo(0, 0), i.rect(0, 0, i.canvas.width, i.canvas.height), i.clip("evenodd");
}
function th(i, t, e) {
  const { startAngle: s, pixelMargin: n, x: o, y: r, outerRadius: a, innerRadius: l } = t;
  let c = n / a;
  i.beginPath(), i.arc(o, r, a, s - c, e + c), l > n ? (c = n / l, i.arc(o, r, l, e + c, s - c, !0)) : i.arc(o, r, n, e + K, s - K), i.closePath(), i.clip();
}
function eh(i) {
  return ws(i, [
    "outerStart",
    "outerEnd",
    "innerStart",
    "innerEnd"
  ]);
}
function ih(i, t, e, s) {
  const n = eh(i.options.borderRadius), o = (e - t) / 2, r = Math.min(o, s * t / 2), a = (l) => {
    const c = (e - Math.min(o, l)) * s / 2;
    return J(l, 0, Math.min(o, c));
  };
  return {
    outerStart: a(n.outerStart),
    outerEnd: a(n.outerEnd),
    innerStart: J(n.innerStart, 0, r),
    innerEnd: J(n.innerEnd, 0, r)
  };
}
function oe(i, t, e, s) {
  return {
    x: e + i * Math.cos(t),
    y: s + i * Math.sin(t)
  };
}
function vi(i, t, e, s, n, o) {
  const { x: r, y: a, startAngle: l, pixelMargin: c, innerRadius: h } = t, d = Math.max(t.outerRadius + s + e - c, 0), f = h > 0 ? h + s + e + c : 0;
  let u = 0;
  const g = n - l;
  if (s) {
    const B = h > 0 ? h - s : 0, W = d > 0 ? d - s : 0, $ = (B + W) / 2, at = $ !== 0 ? g * $ / ($ + s) : g;
    u = (g - at) / 2;
  }
  const p = Math.max(1e-3, g * d - e / z) / d, m = (g - p) / 2, b = l + m + u, _ = n - m - u, { outerStart: y, outerEnd: v, innerStart: x, innerEnd: S } = ih(t, f, d, _ - b), k = d - y, w = d - v, P = b + y / k, T = _ - v / w, L = f + x, R = f + S, q = b + x / L, it = _ - S / R;
  if (i.beginPath(), o) {
    const B = (P + T) / 2;
    if (i.arc(r, a, d, P, B), i.arc(r, a, d, B, T), v > 0) {
      const H = oe(w, T, r, a);
      i.arc(H.x, H.y, v, T, _ + K);
    }
    const W = oe(R, _, r, a);
    if (i.lineTo(W.x, W.y), S > 0) {
      const H = oe(R, it, r, a);
      i.arc(H.x, H.y, S, _ + K, it + Math.PI);
    }
    const $ = (_ - S / f + (b + x / f)) / 2;
    if (i.arc(r, a, f, _ - S / f, $, !0), i.arc(r, a, f, $, b + x / f, !0), x > 0) {
      const H = oe(L, q, r, a);
      i.arc(H.x, H.y, x, q + Math.PI, b - K);
    }
    const at = oe(k, b, r, a);
    if (i.lineTo(at.x, at.y), y > 0) {
      const H = oe(k, P, r, a);
      i.arc(H.x, H.y, y, b - K, P);
    }
  } else {
    i.moveTo(r, a);
    const B = Math.cos(P) * d + r, W = Math.sin(P) * d + a;
    i.lineTo(B, W);
    const $ = Math.cos(T) * d + r, at = Math.sin(T) * d + a;
    i.lineTo($, at);
  }
  i.closePath();
}
function sh(i, t, e, s, n) {
  const { fullCircles: o, startAngle: r, circumference: a } = t;
  let l = t.endAngle;
  if (o) {
    vi(i, t, e, s, l, n);
    for (let c = 0; c < o; ++c)
      i.fill();
    isNaN(a) || (l = r + (a % j || j));
  }
  return vi(i, t, e, s, l, n), i.fill(), l;
}
function nh(i, t, e, s, n) {
  const { fullCircles: o, startAngle: r, circumference: a, options: l } = t, { borderWidth: c, borderJoinStyle: h, borderDash: d, borderDashOffset: f, borderRadius: u } = l, g = l.borderAlign === "inner";
  if (!c)
    return;
  i.setLineDash(d || []), i.lineDashOffset = f, g ? (i.lineWidth = c * 2, i.lineJoin = h || "round") : (i.lineWidth = c, i.lineJoin = h || "bevel");
  let p = t.endAngle;
  if (o) {
    vi(i, t, e, s, p, n);
    for (let m = 0; m < o; ++m)
      i.stroke();
    isNaN(a) || (p = r + (a % j || j));
  }
  g && th(i, t, p), l.selfJoin && p - r >= z && u === 0 && h !== "miter" && Qc(i, t, p), o || (vi(i, t, e, s, p, n), i.stroke());
}
class ve extends mt {
  constructor(e) {
    super();
    M(this, "circumference");
    M(this, "endAngle");
    M(this, "fullCircles");
    M(this, "innerRadius");
    M(this, "outerRadius");
    M(this, "pixelMargin");
    M(this, "startAngle");
    this.options = void 0, this.circumference = void 0, this.startAngle = void 0, this.endAngle = void 0, this.innerRadius = void 0, this.outerRadius = void 0, this.pixelMargin = 0, this.fullCircles = 0, e && Object.assign(this, e);
  }
  inRange(e, s, n) {
    const o = this.getProps([
      "x",
      "y"
    ], n), { angle: r, distance: a } = wo(o, {
      x: e,
      y: s
    }), { startAngle: l, endAngle: c, innerRadius: h, outerRadius: d, circumference: f } = this.getProps([
      "startAngle",
      "endAngle",
      "innerRadius",
      "outerRadius",
      "circumference"
    ], n), u = (this.options.spacing + this.options.borderWidth) / 2, g = O(f, c - l), p = Fe(r, l, c) && l !== c, m = g >= j || p, b = Ct(a, h + u, d + u);
    return m && b;
  }
  getCenterPoint(e) {
    const { x: s, y: n, startAngle: o, endAngle: r, innerRadius: a, outerRadius: l } = this.getProps([
      "x",
      "y",
      "startAngle",
      "endAngle",
      "innerRadius",
      "outerRadius"
    ], e), { offset: c, spacing: h } = this.options, d = (o + r) / 2, f = (a + l + h + c) / 2;
    return {
      x: s + Math.cos(d) * f,
      y: n + Math.sin(d) * f
    };
  }
  tooltipPosition(e) {
    return this.getCenterPoint(e);
  }
  draw(e) {
    const { options: s, circumference: n } = this, o = (s.offset || 0) / 4, r = (s.spacing || 0) / 2, a = s.circular;
    if (this.pixelMargin = s.borderAlign === "inner" ? 0.33 : 0, this.fullCircles = n > j ? Math.floor(n / j) : 0, n === 0 || this.innerRadius < 0 || this.outerRadius < 0)
      return;
    e.save();
    const l = (this.startAngle + this.endAngle) / 2;
    e.translate(Math.cos(l) * o, Math.sin(l) * o);
    const c = 1 - Math.sin(Math.min(z, n || 0)), h = o * c;
    e.fillStyle = s.backgroundColor, e.strokeStyle = s.borderColor, sh(e, this, h, r, a), nh(e, this, h, r, a), e.restore();
  }
}
M(ve, "id", "arc"), M(ve, "defaults", {
  borderAlign: "center",
  borderColor: "#fff",
  borderDash: [],
  borderDashOffset: 0,
  borderJoinStyle: void 0,
  borderRadius: 0,
  borderWidth: 2,
  offset: 0,
  spacing: 0,
  angle: void 0,
  circular: !0,
  selfJoin: !1
}), M(ve, "defaultRoutes", {
  backgroundColor: "backgroundColor"
}), M(ve, "descriptors", {
  _scriptable: !0,
  _indexable: (e) => e !== "borderDash"
});
function nr(i, t, e = t) {
  i.lineCap = O(e.borderCapStyle, t.borderCapStyle), i.setLineDash(O(e.borderDash, t.borderDash)), i.lineDashOffset = O(e.borderDashOffset, t.borderDashOffset), i.lineJoin = O(e.borderJoinStyle, t.borderJoinStyle), i.lineWidth = O(e.borderWidth, t.borderWidth), i.strokeStyle = O(e.borderColor, t.borderColor);
}
function oh(i, t, e) {
  i.lineTo(e.x, e.y);
}
function rh(i) {
  return i.stepped ? Aa : i.tension || i.cubicInterpolationMode === "monotone" ? Ca : oh;
}
function or(i, t, e = {}) {
  const s = i.length, { start: n = 0, end: o = s - 1 } = e, { start: r, end: a } = t, l = Math.max(n, r), c = Math.min(o, a), h = n < r && o < r || n > a && o > a;
  return {
    count: s,
    start: l,
    loop: t.loop,
    ilen: c < l && !h ? s + c - l : c - l
  };
}
function ah(i, t, e, s) {
  const { points: n, options: o } = t, { count: r, start: a, loop: l, ilen: c } = or(n, e, s), h = rh(o);
  let { move: d = !0, reverse: f } = s || {}, u, g, p;
  for (u = 0; u <= c; ++u)
    g = n[(a + (f ? c - u : u)) % r], !g.skip && (d ? (i.moveTo(g.x, g.y), d = !1) : h(i, p, g, f, o.stepped), p = g);
  return l && (g = n[(a + (f ? c : 0)) % r], h(i, p, g, f, o.stepped)), !!l;
}
function lh(i, t, e, s) {
  const n = t.points, { count: o, start: r, ilen: a } = or(n, e, s), { move: l = !0, reverse: c } = s || {};
  let h = 0, d = 0, f, u, g, p, m, b;
  const _ = (v) => (r + (c ? a - v : v)) % o, y = () => {
    p !== m && (i.lineTo(h, m), i.lineTo(h, p), i.lineTo(h, b));
  };
  for (l && (u = n[_(0)], i.moveTo(u.x, u.y)), f = 0; f <= a; ++f) {
    if (u = n[_(f)], u.skip)
      continue;
    const v = u.x, x = u.y, S = v | 0;
    S === g ? (x < p ? p = x : x > m && (m = x), h = (d * h + v) / ++d) : (y(), i.lineTo(v, x), g = S, d = 0, p = m = x), b = x;
  }
  y();
}
function rs(i) {
  const t = i.options, e = t.borderDash && t.borderDash.length;
  return !i._decimated && !i._loop && !t.tension && t.cubicInterpolationMode !== "monotone" && !t.stepped && !e ? lh : ah;
}
function ch(i) {
  return i.stepped ? al : i.tension || i.cubicInterpolationMode === "monotone" ? ll : qt;
}
function hh(i, t, e, s) {
  let n = t._path;
  n || (n = t._path = new Path2D(), t.path(n, e, s) && n.closePath()), nr(i, t.options), i.stroke(n);
}
function dh(i, t, e, s) {
  const { segments: n, options: o } = t, r = rs(t);
  for (const a of n)
    nr(i, o, a.style), i.beginPath(), r(i, t, a, {
      start: e,
      end: e + s - 1
    }) && i.closePath(), i.stroke();
}
const fh = typeof Path2D == "function";
function uh(i, t, e, s) {
  fh && !t.options.segment ? hh(i, t, e, s) : dh(i, t, e, s);
}
class zt extends mt {
  constructor(t) {
    super(), this.animated = !0, this.options = void 0, this._chart = void 0, this._loop = void 0, this._fullLoop = void 0, this._path = void 0, this._points = void 0, this._segments = void 0, this._decimated = !1, this._pointsUpdated = !1, this._datasetIndex = void 0, t && Object.assign(this, t);
  }
  updateControlPoints(t, e) {
    const s = this.options;
    if ((s.tension || s.cubicInterpolationMode === "monotone") && !s.stepped && !this._pointsUpdated) {
      const n = s.spanGaps ? this._loop : this._fullLoop;
      Qa(this._points, s, t, n, e), this._pointsUpdated = !0;
    }
  }
  set points(t) {
    this._points = t, delete this._segments, delete this._path, this._pointsUpdated = !1;
  }
  get points() {
    return this._points;
  }
  get segments() {
    return this._segments || (this._segments = gl(this, this.options.segment));
  }
  first() {
    const t = this.segments, e = this.points;
    return t.length && e[t[0].start];
  }
  last() {
    const t = this.segments, e = this.points, s = t.length;
    return s && e[t[s - 1].end];
  }
  interpolate(t, e) {
    const s = this.options, n = t[e], o = this.points, r = $o(this, {
      property: e,
      start: n,
      end: n
    });
    if (!r.length)
      return;
    const a = [], l = ch(s);
    let c, h;
    for (c = 0, h = r.length; c < h; ++c) {
      const { start: d, end: f } = r[c], u = o[d], g = o[f];
      if (u === g) {
        a.push(u);
        continue;
      }
      const p = Math.abs((n - u[e]) / (g[e] - u[e])), m = l(u, g, p, s.stepped);
      m[e] = t[e], a.push(m);
    }
    return a.length === 1 ? a[0] : a;
  }
  pathSegment(t, e, s) {
    return rs(this)(t, this, e, s);
  }
  path(t, e, s) {
    const n = this.segments, o = rs(this);
    let r = this._loop;
    e = e || 0, s = s || this.points.length - e;
    for (const a of n)
      r &= o(t, this, a, {
        start: e,
        end: e + s - 1
      });
    return !!r;
  }
  draw(t, e, s, n) {
    const o = this.options || {};
    (this.points || []).length && o.borderWidth && (t.save(), uh(t, this, s, n), t.restore()), this.animated && (this._pointsUpdated = !1, this._path = void 0);
  }
}
M(zt, "id", "line"), M(zt, "defaults", {
  borderCapStyle: "butt",
  borderDash: [],
  borderDashOffset: 0,
  borderJoinStyle: "miter",
  borderWidth: 3,
  capBezierPoints: !0,
  cubicInterpolationMode: "default",
  fill: !1,
  spanGaps: !1,
  stepped: !1,
  tension: 0
}), M(zt, "defaultRoutes", {
  backgroundColor: "backgroundColor",
  borderColor: "borderColor"
}), M(zt, "descriptors", {
  _scriptable: !0,
  _indexable: (t) => t !== "borderDash" && t !== "fill"
});
function Rn(i, t, e, s) {
  const n = i.options, { [e]: o } = i.getProps([
    e
  ], s);
  return Math.abs(t - o) < n.radius + n.hitRadius;
}
class ui extends mt {
  constructor(e) {
    super();
    M(this, "parsed");
    M(this, "skip");
    M(this, "stop");
    this.options = void 0, this.parsed = void 0, this.skip = void 0, this.stop = void 0, e && Object.assign(this, e);
  }
  inRange(e, s, n) {
    const o = this.options, { x: r, y: a } = this.getProps([
      "x",
      "y"
    ], n);
    return Math.pow(e - r, 2) + Math.pow(s - a, 2) < Math.pow(o.hitRadius + o.radius, 2);
  }
  inXRange(e, s) {
    return Rn(this, e, "x", s);
  }
  inYRange(e, s) {
    return Rn(this, e, "y", s);
  }
  getCenterPoint(e) {
    const { x: s, y: n } = this.getProps([
      "x",
      "y"
    ], e);
    return {
      x: s,
      y: n
    };
  }
  size(e) {
    e = e || this.options || {};
    let s = e.radius || 0;
    s = Math.max(s, s && e.hoverRadius || 0);
    const n = s && e.borderWidth || 0;
    return (s + n) * 2;
  }
  draw(e, s) {
    const n = this.options;
    this.skip || n.radius < 0.1 || !Tt(this, s, this.size(n) / 2) || (e.strokeStyle = n.borderColor, e.lineWidth = n.borderWidth, e.fillStyle = n.backgroundColor, is(e, n, this.x, this.y));
  }
  getRange() {
    const e = this.options || {};
    return e.radius + e.hitRadius;
  }
}
M(ui, "id", "point"), /**
* @type {any}
*/
M(ui, "defaults", {
  borderWidth: 1,
  hitRadius: 1,
  hoverBorderWidth: 1,
  hoverRadius: 4,
  pointStyle: "circle",
  radius: 3,
  rotation: 0
}), /**
* @type {any}
*/
M(ui, "defaultRoutes", {
  backgroundColor: "backgroundColor",
  borderColor: "borderColor"
});
function rr(i, t) {
  const { x: e, y: s, base: n, width: o, height: r } = i.getProps([
    "x",
    "y",
    "base",
    "width",
    "height"
  ], t);
  let a, l, c, h, d;
  return i.horizontal ? (d = r / 2, a = Math.min(e, n), l = Math.max(e, n), c = s - d, h = s + d) : (d = o / 2, a = e - d, l = e + d, c = Math.min(s, n), h = Math.max(s, n)), {
    left: a,
    top: c,
    right: l,
    bottom: h
  };
}
function Bt(i, t, e, s) {
  return i ? 0 : J(t, e, s);
}
function gh(i, t, e) {
  const s = i.options.borderWidth, n = i.borderSkipped, o = Ro(s);
  return {
    t: Bt(n.top, o.top, 0, e),
    r: Bt(n.right, o.right, 0, t),
    b: Bt(n.bottom, o.bottom, 0, e),
    l: Bt(n.left, o.left, 0, t)
  };
}
function ph(i, t, e) {
  const { enableBorderRadius: s } = i.getProps([
    "enableBorderRadius"
  ]), n = i.options.borderRadius, o = Jt(n), r = Math.min(t, e), a = i.borderSkipped, l = s || I(n);
  return {
    topLeft: Bt(!l || a.top || a.left, o.topLeft, 0, r),
    topRight: Bt(!l || a.top || a.right, o.topRight, 0, r),
    bottomLeft: Bt(!l || a.bottom || a.left, o.bottomLeft, 0, r),
    bottomRight: Bt(!l || a.bottom || a.right, o.bottomRight, 0, r)
  };
}
function mh(i) {
  const t = rr(i), e = t.right - t.left, s = t.bottom - t.top, n = gh(i, e / 2, s / 2), o = ph(i, e / 2, s / 2);
  return {
    outer: {
      x: t.left,
      y: t.top,
      w: e,
      h: s,
      radius: o
    },
    inner: {
      x: t.left + n.l,
      y: t.top + n.t,
      w: e - n.l - n.r,
      h: s - n.t - n.b,
      radius: {
        topLeft: Math.max(0, o.topLeft - Math.max(n.t, n.l)),
        topRight: Math.max(0, o.topRight - Math.max(n.t, n.r)),
        bottomLeft: Math.max(0, o.bottomLeft - Math.max(n.b, n.l)),
        bottomRight: Math.max(0, o.bottomRight - Math.max(n.b, n.r))
      }
    }
  };
}
function Xi(i, t, e, s) {
  const n = t === null, o = e === null, a = i && !(n && o) && rr(i, s);
  return a && (n || Ct(t, a.left, a.right)) && (o || Ct(e, a.top, a.bottom));
}
function bh(i) {
  return i.topLeft || i.topRight || i.bottomLeft || i.bottomRight;
}
function _h(i, t) {
  i.rect(t.x, t.y, t.w, t.h);
}
function Ui(i, t, e = {}) {
  const s = i.x !== e.x ? -t : 0, n = i.y !== e.y ? -t : 0, o = (i.x + i.w !== e.x + e.w ? t : 0) - s, r = (i.y + i.h !== e.y + e.h ? t : 0) - n;
  return {
    x: i.x + s,
    y: i.y + n,
    w: i.w + o,
    h: i.h + r,
    radius: i.radius
  };
}
class gi extends mt {
  constructor(t) {
    super(), this.options = void 0, this.horizontal = void 0, this.base = void 0, this.width = void 0, this.height = void 0, this.inflateAmount = void 0, t && Object.assign(this, t);
  }
  draw(t) {
    const { inflateAmount: e, options: { borderColor: s, backgroundColor: n } } = this, { inner: o, outer: r } = mh(this), a = bh(r.radius) ? Ie : _h;
    t.save(), (r.w !== o.w || r.h !== o.h) && (t.beginPath(), a(t, Ui(r, e, o)), t.clip(), a(t, Ui(o, -e, r)), t.fillStyle = s, t.fill("evenodd")), t.beginPath(), a(t, Ui(o, e)), t.fillStyle = n, t.fill(), t.restore();
  }
  inRange(t, e, s) {
    return Xi(this, t, e, s);
  }
  inXRange(t, e) {
    return Xi(this, t, null, e);
  }
  inYRange(t, e) {
    return Xi(this, null, t, e);
  }
  getCenterPoint(t) {
    const { x: e, y: s, base: n, horizontal: o } = this.getProps([
      "x",
      "y",
      "base",
      "horizontal"
    ], t);
    return {
      x: o ? (e + n) / 2 : e,
      y: o ? s : (s + n) / 2
    };
  }
  getRange(t) {
    return t === "x" ? this.width / 2 : this.height / 2;
  }
}
M(gi, "id", "bar"), M(gi, "defaults", {
  borderSkipped: "start",
  borderWidth: 0,
  borderRadius: 0,
  inflateAmount: "auto",
  pointStyle: void 0
}), M(gi, "defaultRoutes", {
  backgroundColor: "backgroundColor",
  borderColor: "borderColor"
});
var xh = /* @__PURE__ */ Object.freeze({
  __proto__: null,
  ArcElement: ve,
  BarElement: gi,
  LineElement: zt,
  PointElement: ui
});
const as = [
  "rgb(54, 162, 235)",
  "rgb(255, 99, 132)",
  "rgb(255, 159, 64)",
  "rgb(255, 205, 86)",
  "rgb(75, 192, 192)",
  "rgb(153, 102, 255)",
  "rgb(201, 203, 207)"
  // grey
], En = /* @__PURE__ */ as.map((i) => i.replace("rgb(", "rgba(").replace(")", ", 0.5)"));
function ar(i) {
  return as[i % as.length];
}
function lr(i) {
  return En[i % En.length];
}
function yh(i, t) {
  return i.borderColor = ar(t), i.backgroundColor = lr(t), ++t;
}
function vh(i, t) {
  return i.backgroundColor = i.data.map(() => ar(t++)), t;
}
function kh(i, t) {
  return i.backgroundColor = i.data.map(() => lr(t++)), t;
}
function Mh(i) {
  let t = 0;
  return (e, s) => {
    const n = i.getDatasetMeta(s).controller;
    n instanceof Zt ? t = vh(e, t) : n instanceof Oe ? t = kh(e, t) : n && (t = yh(e, t));
  };
}
function Fn(i) {
  let t;
  for (t in i)
    if (i[t].borderColor || i[t].backgroundColor)
      return !0;
  return !1;
}
function wh(i) {
  return i && (i.borderColor || i.backgroundColor);
}
function Sh() {
  return X.borderColor !== "rgba(0,0,0,0.1)" || X.backgroundColor !== "rgba(0,0,0,0.1)";
}
var Ph = {
  id: "colors",
  defaults: {
    enabled: !0,
    forceOverride: !1
  },
  beforeLayout(i, t, e) {
    if (!e.enabled)
      return;
    const { data: { datasets: s }, options: n } = i.config, { elements: o } = n, r = Fn(s) || wh(n) || o && Fn(o) || Sh();
    if (!e.forceOverride && r)
      return;
    const a = Mh(i);
    s.forEach(a);
  }
};
function Dh(i, t, e, s, n) {
  const o = n.samples || s;
  if (o >= e)
    return i.slice(t, t + e);
  const r = [], a = (e - 2) / (o - 2);
  let l = 0;
  const c = t + e - 1;
  let h = t, d, f, u, g, p;
  for (r[l++] = i[h], d = 0; d < o - 2; d++) {
    let m = 0, b = 0, _;
    const y = Math.floor((d + 1) * a) + 1 + t, v = Math.min(Math.floor((d + 2) * a) + 1, e) + t, x = v - y;
    for (_ = y; _ < v; _++)
      m += i[_].x, b += i[_].y;
    m /= x, b /= x;
    const S = Math.floor(d * a) + 1 + t, k = Math.min(Math.floor((d + 1) * a) + 1, e) + t, { x: w, y: P } = i[h];
    for (u = g = -1, _ = S; _ < k; _++)
      g = 0.5 * Math.abs((w - m) * (i[_].y - P) - (w - i[_].x) * (b - P)), g > u && (u = g, f = i[_], p = _);
    r[l++] = f, h = p;
  }
  return r[l++] = i[c], r;
}
function Ah(i, t, e, s) {
  let n = 0, o = 0, r, a, l, c, h, d, f, u, g, p;
  const m = [], b = t + e - 1, _ = i[t].x, v = i[b].x - _;
  for (r = t; r < t + e; ++r) {
    a = i[r], l = (a.x - _) / v * s, c = a.y;
    const x = l | 0;
    if (x === h)
      c < g ? (g = c, d = r) : c > p && (p = c, f = r), n = (o * n + a.x) / ++o;
    else {
      const S = r - 1;
      if (!F(d) && !F(f)) {
        const k = Math.min(d, f), w = Math.max(d, f);
        k !== u && k !== S && m.push({
          ...i[k],
          x: n
        }), w !== u && w !== S && m.push({
          ...i[w],
          x: n
        });
      }
      r > 0 && S !== u && m.push(i[S]), m.push(a), h = x, o = 0, g = p = c, d = f = u = r;
    }
  }
  return m;
}
function cr(i) {
  if (i._decimated) {
    const t = i._data;
    delete i._decimated, delete i._data, Object.defineProperty(i, "data", {
      configurable: !0,
      enumerable: !0,
      writable: !0,
      value: t
    });
  }
}
function In(i) {
  i.data.datasets.forEach((t) => {
    cr(t);
  });
}
function Ch(i, t) {
  const e = t.length;
  let s = 0, n;
  const { iScale: o } = i, { min: r, max: a, minDefined: l, maxDefined: c } = o.getUserBounds();
  return l && (s = J(Ot(t, o.axis, r).lo, 0, e - 1)), c ? n = J(Ot(t, o.axis, a).hi + 1, s, e) - s : n = e - s, {
    start: s,
    count: n
  };
}
var Oh = {
  id: "decimation",
  defaults: {
    algorithm: "min-max",
    enabled: !1
  },
  beforeElementsUpdate: (i, t, e) => {
    if (!e.enabled) {
      In(i);
      return;
    }
    const s = i.width;
    i.data.datasets.forEach((n, o) => {
      const { _data: r, indexAxis: a } = n, l = i.getDatasetMeta(o), c = r || n.data;
      if (xe([
        a,
        i.options.indexAxis
      ]) === "y" || !l.controller.supportsDecimation)
        return;
      const h = i.scales[l.xAxisID];
      if (h.type !== "linear" && h.type !== "time" || i.options.parsing)
        return;
      let { start: d, count: f } = Ch(l, c);
      const u = e.threshold || 4 * s;
      if (f <= u) {
        cr(n);
        return;
      }
      F(r) && (n._data = c, delete n.data, Object.defineProperty(n, "data", {
        configurable: !0,
        enumerable: !0,
        get: function() {
          return this._decimated;
        },
        set: function(p) {
          this._data = p;
        }
      }));
      let g;
      switch (e.algorithm) {
        case "lttb":
          g = Dh(c, d, f, s, e);
          break;
        case "min-max":
          g = Ah(c, d, f, s);
          break;
        default:
          throw new Error(`Unsupported decimation algorithm '${e.algorithm}'`);
      }
      n._decimated = g;
    });
  },
  destroy(i) {
    In(i);
  }
};
function Th(i, t, e) {
  const s = i.segments, n = i.points, o = t.points, r = [];
  for (const a of s) {
    let { start: l, end: c } = a;
    c = Ti(l, c, n);
    const h = ls(e, n[l], n[c], a.loop);
    if (!t.segments) {
      r.push({
        source: a,
        target: h,
        start: n[l],
        end: n[c]
      });
      continue;
    }
    const d = $o(t, h);
    for (const f of d) {
      const u = ls(e, o[f.start], o[f.end], f.loop), g = jo(a, n, u);
      for (const p of g)
        r.push({
          source: p,
          target: f,
          start: {
            [e]: zn(h, u, "start", Math.max)
          },
          end: {
            [e]: zn(h, u, "end", Math.min)
          }
        });
    }
  }
  return r;
}
function ls(i, t, e, s) {
  if (s)
    return;
  let n = t[i], o = e[i];
  return i === "angle" && (n = nt(n), o = nt(o)), {
    property: i,
    start: n,
    end: o
  };
}
function Lh(i, t) {
  const { x: e = null, y: s = null } = i || {}, n = t.points, o = [];
  return t.segments.forEach(({ start: r, end: a }) => {
    a = Ti(r, a, n);
    const l = n[r], c = n[a];
    s !== null ? (o.push({
      x: l.x,
      y: s
    }), o.push({
      x: c.x,
      y: s
    })) : e !== null && (o.push({
      x: e,
      y: l.y
    }), o.push({
      x: e,
      y: c.y
    }));
  }), o;
}
function Ti(i, t, e) {
  for (; t > i; t--) {
    const s = e[t];
    if (!isNaN(s.x) && !isNaN(s.y))
      break;
  }
  return t;
}
function zn(i, t, e, s) {
  return i && t ? s(i[e], t[e]) : i ? i[e] : t ? t[e] : 0;
}
function hr(i, t) {
  let e = [], s = !1;
  return Y(i) ? (s = !0, e = i) : e = Lh(i, t), e.length ? new zt({
    points: e,
    options: {
      tension: 0
    },
    _loop: s,
    _fullLoop: s
  }) : null;
}
function Bn(i) {
  return i && i.fill !== !1;
}
function Rh(i, t, e) {
  let n = i[t].fill;
  const o = [
    t
  ];
  let r;
  if (!e)
    return n;
  for (; n !== !1 && o.indexOf(n) === -1; ) {
    if (!U(n))
      return n;
    if (r = i[n], !r)
      return !1;
    if (r.visible)
      return n;
    o.push(n), n = r.fill;
  }
  return !1;
}
function Eh(i, t, e) {
  const s = Bh(i);
  if (I(s))
    return isNaN(s.value) ? !1 : s;
  let n = parseFloat(s);
  return U(n) && Math.floor(n) === n ? Fh(s[0], t, n, e) : [
    "origin",
    "start",
    "end",
    "stack",
    "shape"
  ].indexOf(s) >= 0 && s;
}
function Fh(i, t, e, s) {
  return (i === "-" || i === "+") && (e = t + e), e === t || e < 0 || e >= s ? !1 : e;
}
function Ih(i, t) {
  let e = null;
  return i === "start" ? e = t.bottom : i === "end" ? e = t.top : I(i) ? e = t.getPixelForValue(i.value) : t.getBasePixel && (e = t.getBasePixel()), e;
}
function zh(i, t, e) {
  let s;
  return i === "start" ? s = e : i === "end" ? s = t.options.reverse ? t.min : t.max : I(i) ? s = i.value : s = t.getBaseValue(), s;
}
function Bh(i) {
  const t = i.options, e = t.fill;
  let s = O(e && e.target, e);
  return s === void 0 && (s = !!t.backgroundColor), s === !1 || s === null ? !1 : s === !0 ? "origin" : s;
}
function Vh(i) {
  const { scale: t, index: e, line: s } = i, n = [], o = s.segments, r = s.points, a = Wh(t, e);
  a.push(hr({
    x: null,
    y: t.bottom
  }, s));
  for (let l = 0; l < o.length; l++) {
    const c = o[l];
    for (let h = c.start; h <= c.end; h++)
      Nh(n, r[h], a);
  }
  return new zt({
    points: n,
    options: {}
  });
}
function Wh(i, t) {
  const e = [], s = i.getMatchingVisibleMetas("line");
  for (let n = 0; n < s.length; n++) {
    const o = s[n];
    if (o.index === t)
      break;
    o.hidden || e.unshift(o.dataset);
  }
  return e;
}
function Nh(i, t, e) {
  const s = [];
  for (let n = 0; n < e.length; n++) {
    const o = e[n], { first: r, last: a, point: l } = Hh(o, t, "x");
    if (!(!l || r && a)) {
      if (r)
        s.unshift(l);
      else if (i.push(l), !a)
        break;
    }
  }
  i.push(...s);
}
function Hh(i, t, e) {
  const s = i.interpolate(t, e);
  if (!s)
    return {};
  const n = s[e], o = i.segments, r = i.points;
  let a = !1, l = !1;
  for (let c = 0; c < o.length; c++) {
    const h = o[c], d = r[h.start][e], f = r[h.end][e];
    if (Ct(n, d, f)) {
      a = n === d, l = n === f;
      break;
    }
  }
  return {
    first: a,
    last: l,
    point: s
  };
}
class dr {
  constructor(t) {
    this.x = t.x, this.y = t.y, this.radius = t.radius;
  }
  pathSegment(t, e, s) {
    const { x: n, y: o, radius: r } = this;
    return e = e || {
      start: 0,
      end: j
    }, t.arc(n, o, r, e.end, e.start, !0), !s.bounds;
  }
  interpolate(t) {
    const { x: e, y: s, radius: n } = this, o = t.angle;
    return {
      x: e + Math.cos(o) * n,
      y: s + Math.sin(o) * n,
      angle: o
    };
  }
}
function jh(i) {
  const { chart: t, fill: e, line: s } = i;
  if (U(e))
    return $h(t, e);
  if (e === "stack")
    return Vh(i);
  if (e === "shape")
    return !0;
  const n = Yh(i);
  return n instanceof dr ? n : hr(n, s);
}
function $h(i, t) {
  const e = i.getDatasetMeta(t);
  return e && i.isDatasetVisible(t) ? e.dataset : null;
}
function Yh(i) {
  return (i.scale || {}).getPointPositionForValue ? Uh(i) : Xh(i);
}
function Xh(i) {
  const { scale: t = {}, fill: e } = i, s = Ih(e, t);
  if (U(s)) {
    const n = t.isHorizontal();
    return {
      x: n ? s : null,
      y: n ? null : s
    };
  }
  return null;
}
function Uh(i) {
  const { scale: t, fill: e } = i, s = t.options, n = t.getLabels().length, o = s.reverse ? t.max : t.min, r = zh(e, t, o), a = [];
  if (s.grid.circular) {
    const l = t.getPointPositionForValue(0, o);
    return new dr({
      x: l.x,
      y: l.y,
      radius: t.getDistanceFromCenterForValue(r)
    });
  }
  for (let l = 0; l < n; ++l)
    a.push(t.getPointPositionForValue(l, r));
  return a;
}
function Ki(i, t, e) {
  const s = jh(t), { chart: n, index: o, line: r, scale: a, axis: l } = t, c = r.options, h = c.fill, d = c.backgroundColor, { above: f = d, below: u = d } = h || {}, g = n.getDatasetMeta(o), p = Yo(n, g);
  s && r.points.length && (Di(i, e), Kh(i, {
    line: r,
    target: s,
    above: f,
    below: u,
    area: e,
    scale: a,
    axis: l,
    clip: p
  }), Ai(i));
}
function Kh(i, t) {
  const { line: e, target: s, above: n, below: o, area: r, scale: a, clip: l } = t, c = e._loop ? "angle" : t.axis;
  i.save();
  let h = o;
  o !== n && (c === "x" ? (Vn(i, s, r.top), qi(i, {
    line: e,
    target: s,
    color: n,
    scale: a,
    property: c,
    clip: l
  }), i.restore(), i.save(), Vn(i, s, r.bottom)) : c === "y" && (Wn(i, s, r.left), qi(i, {
    line: e,
    target: s,
    color: o,
    scale: a,
    property: c,
    clip: l
  }), i.restore(), i.save(), Wn(i, s, r.right), h = n)), qi(i, {
    line: e,
    target: s,
    color: h,
    scale: a,
    property: c,
    clip: l
  }), i.restore();
}
function Vn(i, t, e) {
  const { segments: s, points: n } = t;
  let o = !0, r = !1;
  i.beginPath();
  for (const a of s) {
    const { start: l, end: c } = a, h = n[l], d = n[Ti(l, c, n)];
    o ? (i.moveTo(h.x, h.y), o = !1) : (i.lineTo(h.x, e), i.lineTo(h.x, h.y)), r = !!t.pathSegment(i, a, {
      move: r
    }), r ? i.closePath() : i.lineTo(d.x, e);
  }
  i.lineTo(t.first().x, e), i.closePath(), i.clip();
}
function Wn(i, t, e) {
  const { segments: s, points: n } = t;
  let o = !0, r = !1;
  i.beginPath();
  for (const a of s) {
    const { start: l, end: c } = a, h = n[l], d = n[Ti(l, c, n)];
    o ? (i.moveTo(h.x, h.y), o = !1) : (i.lineTo(e, h.y), i.lineTo(h.x, h.y)), r = !!t.pathSegment(i, a, {
      move: r
    }), r ? i.closePath() : i.lineTo(e, d.y);
  }
  i.lineTo(e, t.first().y), i.closePath(), i.clip();
}
function qi(i, t) {
  const { line: e, target: s, property: n, color: o, scale: r, clip: a } = t, l = Th(e, s, n);
  for (const { source: c, target: h, start: d, end: f } of l) {
    const { style: { backgroundColor: u = o } = {} } = c, g = s !== !0;
    i.save(), i.fillStyle = u, qh(i, r, a, g && ls(n, d, f)), i.beginPath();
    const p = !!e.pathSegment(i, c);
    let m;
    if (g) {
      p ? i.closePath() : Nn(i, s, f, n);
      const b = !!s.pathSegment(i, h, {
        move: p,
        reverse: !0
      });
      m = p && b, m || Nn(i, s, d, n);
    }
    i.closePath(), i.fill(m ? "evenodd" : "nonzero"), i.restore();
  }
}
function qh(i, t, e, s) {
  const n = t.chart.chartArea, { property: o, start: r, end: a } = s || {};
  if (o === "x" || o === "y") {
    let l, c, h, d;
    o === "x" ? (l = r, c = n.top, h = a, d = n.bottom) : (l = n.left, c = r, h = n.right, d = a), i.beginPath(), e && (l = Math.max(l, e.left), h = Math.min(h, e.right), c = Math.max(c, e.top), d = Math.min(d, e.bottom)), i.rect(l, c, h - l, d - c), i.clip();
  }
}
function Nn(i, t, e, s) {
  const n = t.interpolate(e, s);
  n && i.lineTo(n.x, n.y);
}
var Gh = {
  id: "filler",
  afterDatasetsUpdate(i, t, e) {
    const s = (i.data.datasets || []).length, n = [];
    let o, r, a, l;
    for (r = 0; r < s; ++r)
      o = i.getDatasetMeta(r), a = o.dataset, l = null, a && a.options && a instanceof zt && (l = {
        visible: i.isDatasetVisible(r),
        index: r,
        fill: Eh(a, r, s),
        chart: i,
        axis: o.controller.options.indexAxis,
        scale: o.vScale,
        line: a
      }), o.$filler = l, n.push(l);
    for (r = 0; r < s; ++r)
      l = n[r], !(!l || l.fill === !1) && (l.fill = Rh(n, r, e.propagate));
  },
  beforeDraw(i, t, e) {
    const s = e.drawTime === "beforeDraw", n = i.getSortedVisibleDatasetMetas(), o = i.chartArea;
    for (let r = n.length - 1; r >= 0; --r) {
      const a = n[r].$filler;
      a && (a.line.updateControlPoints(o, a.axis), s && a.fill && Ki(i.ctx, a, o));
    }
  },
  beforeDatasetsDraw(i, t, e) {
    if (e.drawTime !== "beforeDatasetsDraw")
      return;
    const s = i.getSortedVisibleDatasetMetas();
    for (let n = s.length - 1; n >= 0; --n) {
      const o = s[n].$filler;
      Bn(o) && Ki(i.ctx, o, i.chartArea);
    }
  },
  beforeDatasetDraw(i, t, e) {
    const s = t.meta.$filler;
    !Bn(s) || e.drawTime !== "beforeDatasetDraw" || Ki(i.ctx, s, i.chartArea);
  },
  defaults: {
    propagate: !0,
    drawTime: "beforeDatasetDraw"
  }
};
const Hn = (i, t) => {
  let { boxHeight: e = t, boxWidth: s = t } = i;
  return i.usePointStyle && (e = Math.min(e, t), s = i.pointStyleWidth || Math.min(s, t)), {
    boxWidth: s,
    boxHeight: e,
    itemHeight: Math.max(t, e)
  };
}, Zh = (i, t) => i !== null && t !== null && i.datasetIndex === t.datasetIndex && i.index === t.index;
class jn extends mt {
  constructor(t) {
    super(), this._added = !1, this.legendHitBoxes = [], this._hoveredItem = null, this.doughnutMode = !1, this.chart = t.chart, this.options = t.options, this.ctx = t.ctx, this.legendItems = void 0, this.columnSizes = void 0, this.lineWidths = void 0, this.maxHeight = void 0, this.maxWidth = void 0, this.top = void 0, this.bottom = void 0, this.left = void 0, this.right = void 0, this.height = void 0, this.width = void 0, this._margins = void 0, this.position = void 0, this.weight = void 0, this.fullSize = void 0;
  }
  update(t, e, s) {
    this.maxWidth = t, this.maxHeight = e, this._margins = s, this.setDimensions(), this.buildLabels(), this.fit();
  }
  setDimensions() {
    this.isHorizontal() ? (this.width = this.maxWidth, this.left = this._margins.left, this.right = this.width) : (this.height = this.maxHeight, this.top = this._margins.top, this.bottom = this.height);
  }
  buildLabels() {
    const t = this.options.labels || {};
    let e = N(t.generateLabels, [
      this.chart
    ], this) || [];
    t.filter && (e = e.filter((s) => t.filter(s, this.chart.data))), t.sort && (e = e.sort((s, n) => t.sort(s, n, this.chart.data))), this.options.reverse && e.reverse(), this.legendItems = e;
  }
  fit() {
    const { options: t, ctx: e } = this;
    if (!t.display) {
      this.width = this.height = 0;
      return;
    }
    const s = t.labels, n = G(s.font), o = n.size, r = this._computeTitleHeight(), { boxWidth: a, itemHeight: l } = Hn(s, o);
    let c, h;
    e.font = n.string, this.isHorizontal() ? (c = this.maxWidth, h = this._fitRows(r, o, a, l) + 10) : (h = this.maxHeight, c = this._fitCols(r, n, a, l) + 10), this.width = Math.min(c, t.maxWidth || this.maxWidth), this.height = Math.min(h, t.maxHeight || this.maxHeight);
  }
  _fitRows(t, e, s, n) {
    const { ctx: o, maxWidth: r, options: { labels: { padding: a } } } = this, l = this.legendHitBoxes = [], c = this.lineWidths = [
      0
    ], h = n + a;
    let d = t;
    o.textAlign = "left", o.textBaseline = "middle";
    let f = -1, u = -h;
    return this.legendItems.forEach((g, p) => {
      const m = s + e / 2 + o.measureText(g.text).width;
      (p === 0 || c[c.length - 1] + m + 2 * a > r) && (d += h, c[c.length - (p > 0 ? 0 : 1)] = 0, u += h, f++), l[p] = {
        left: 0,
        top: u,
        row: f,
        width: m,
        height: n
      }, c[c.length - 1] += m + a;
    }), d;
  }
  _fitCols(t, e, s, n) {
    const { ctx: o, maxHeight: r, options: { labels: { padding: a } } } = this, l = this.legendHitBoxes = [], c = this.columnSizes = [], h = r - t;
    let d = a, f = 0, u = 0, g = 0, p = 0;
    return this.legendItems.forEach((m, b) => {
      const { itemWidth: _, itemHeight: y } = Jh(s, e, o, m, n);
      b > 0 && u + y + 2 * a > h && (d += f + a, c.push({
        width: f,
        height: u
      }), g += f + a, p++, f = u = 0), l[b] = {
        left: g,
        top: u,
        col: p,
        width: _,
        height: y
      }, f = Math.max(f, _), u += y + a;
    }), d += f, c.push({
      width: f,
      height: u
    }), d;
  }
  adjustHitBoxes() {
    if (!this.options.display)
      return;
    const t = this._computeTitleHeight(), { legendHitBoxes: e, options: { align: s, labels: { padding: n }, rtl: o } } = this, r = ce(o, this.left, this.width);
    if (this.isHorizontal()) {
      let a = 0, l = st(s, this.left + n, this.right - this.lineWidths[a]);
      for (const c of e)
        a !== c.row && (a = c.row, l = st(s, this.left + n, this.right - this.lineWidths[a])), c.top += this.top + t + n, c.left = r.leftForLtr(r.x(l), c.width), l += c.width + n;
    } else {
      let a = 0, l = st(s, this.top + t + n, this.bottom - this.columnSizes[a].height);
      for (const c of e)
        c.col !== a && (a = c.col, l = st(s, this.top + t + n, this.bottom - this.columnSizes[a].height)), c.top = l, c.left += this.left + n, c.left = r.leftForLtr(r.x(c.left), c.width), l += c.height + n;
    }
  }
  isHorizontal() {
    return this.options.position === "top" || this.options.position === "bottom";
  }
  draw() {
    if (this.options.display) {
      const t = this.ctx;
      Di(t, this), this._draw(), Ai(t);
    }
  }
  _draw() {
    const { options: t, columnSizes: e, lineWidths: s, ctx: n } = this, { align: o, labels: r } = t, a = X.color, l = ce(t.rtl, this.left, this.width), c = G(r.font), { padding: h } = r, d = c.size, f = d / 2;
    let u;
    this.drawTitle(), n.textAlign = l.textAlign("left"), n.textBaseline = "middle", n.lineWidth = 0.5, n.font = c.string;
    const { boxWidth: g, boxHeight: p, itemHeight: m } = Hn(r, d), b = function(S, k, w) {
      if (isNaN(g) || g <= 0 || isNaN(p) || p < 0)
        return;
      n.save();
      const P = O(w.lineWidth, 1);
      if (n.fillStyle = O(w.fillStyle, a), n.lineCap = O(w.lineCap, "butt"), n.lineDashOffset = O(w.lineDashOffset, 0), n.lineJoin = O(w.lineJoin, "miter"), n.lineWidth = P, n.strokeStyle = O(w.strokeStyle, a), n.setLineDash(O(w.lineDash, [])), r.usePointStyle) {
        const T = {
          radius: p * Math.SQRT2 / 2,
          pointStyle: w.pointStyle,
          rotation: w.rotation,
          borderWidth: P
        }, L = l.xPlus(S, g / 2), R = k + f;
        Lo(n, T, L, R, r.pointStyleWidth && g);
      } else {
        const T = k + Math.max((d - p) / 2, 0), L = l.leftForLtr(S, g), R = Jt(w.borderRadius);
        n.beginPath(), Object.values(R).some((q) => q !== 0) ? Ie(n, {
          x: L,
          y: T,
          w: g,
          h: p,
          radius: R
        }) : n.rect(L, T, g, p), n.fill(), P !== 0 && n.stroke();
      }
      n.restore();
    }, _ = function(S, k, w) {
      ee(n, w.text, S, k + m / 2, c, {
        strikethrough: w.hidden,
        textAlign: l.textAlign(w.textAlign)
      });
    }, y = this.isHorizontal(), v = this._computeTitleHeight();
    y ? u = {
      x: st(o, this.left + h, this.right - s[0]),
      y: this.top + h + v,
      line: 0
    } : u = {
      x: this.left + h,
      y: st(o, this.top + v + h, this.bottom - e[0].height),
      line: 0
    }, Wo(this.ctx, t.textDirection);
    const x = m + h;
    this.legendItems.forEach((S, k) => {
      n.strokeStyle = S.fontColor, n.fillStyle = S.fontColor;
      const w = n.measureText(S.text).width, P = l.textAlign(S.textAlign || (S.textAlign = r.textAlign)), T = g + f + w;
      let L = u.x, R = u.y;
      l.setWidth(this.width), y ? k > 0 && L + T + h > this.right && (R = u.y += x, u.line++, L = u.x = st(o, this.left + h, this.right - s[u.line])) : k > 0 && R + x > this.bottom && (L = u.x = L + e[u.line].width + h, u.line++, R = u.y = st(o, this.top + v + h, this.bottom - e[u.line].height));
      const q = l.x(L);
      if (b(q, R, S), L = ba(P, L + g + f, y ? L + T : this.right, t.rtl), _(l.x(L), R, S), y)
        u.x += T + h;
      else if (typeof S.text != "string") {
        const it = c.lineHeight;
        u.y += fr(S, it) + h;
      } else
        u.y += x;
    }), No(this.ctx, t.textDirection);
  }
  drawTitle() {
    const t = this.options, e = t.title, s = G(e.font), n = rt(e.padding);
    if (!e.display)
      return;
    const o = ce(t.rtl, this.left, this.width), r = this.ctx, a = e.position, l = s.size / 2, c = n.top + l;
    let h, d = this.left, f = this.width;
    if (this.isHorizontal())
      f = Math.max(...this.lineWidths), h = this.top + c, d = st(t.align, d, this.right - f);
    else {
      const g = this.columnSizes.reduce((p, m) => Math.max(p, m.height), 0);
      h = c + st(t.align, this.top, this.bottom - g - t.labels.padding - this._computeTitleHeight());
    }
    const u = st(a, d, d + f);
    r.textAlign = o.textAlign(ks(a)), r.textBaseline = "middle", r.strokeStyle = e.color, r.fillStyle = e.color, r.font = s.string, ee(r, e.text, u, h, s);
  }
  _computeTitleHeight() {
    const t = this.options.title, e = G(t.font), s = rt(t.padding);
    return t.display ? e.lineHeight + s.height : 0;
  }
  _getLegendItemAt(t, e) {
    let s, n, o;
    if (Ct(t, this.left, this.right) && Ct(e, this.top, this.bottom)) {
      for (o = this.legendHitBoxes, s = 0; s < o.length; ++s)
        if (n = o[s], Ct(t, n.left, n.left + n.width) && Ct(e, n.top, n.top + n.height))
          return this.legendItems[s];
    }
    return null;
  }
  handleEvent(t) {
    const e = this.options;
    if (!ed(t.type, e))
      return;
    const s = this._getLegendItemAt(t.x, t.y);
    if (t.type === "mousemove" || t.type === "mouseout") {
      const n = this._hoveredItem, o = Zh(n, s);
      n && !o && N(e.onLeave, [
        t,
        n,
        this
      ], this), this._hoveredItem = s, s && !o && N(e.onHover, [
        t,
        s,
        this
      ], this);
    } else s && N(e.onClick, [
      t,
      s,
      this
    ], this);
  }
}
function Jh(i, t, e, s, n) {
  const o = Qh(s, i, t, e), r = td(n, s, t.lineHeight);
  return {
    itemWidth: o,
    itemHeight: r
  };
}
function Qh(i, t, e, s) {
  let n = i.text;
  return n && typeof n != "string" && (n = n.reduce((o, r) => o.length > r.length ? o : r)), t + e.size / 2 + s.measureText(n).width;
}
function td(i, t, e) {
  let s = i;
  return typeof t.text != "string" && (s = fr(t, e)), s;
}
function fr(i, t) {
  const e = i.text ? i.text.length : 0;
  return t * e;
}
function ed(i, t) {
  return !!((i === "mousemove" || i === "mouseout") && (t.onHover || t.onLeave) || t.onClick && (i === "click" || i === "mouseup"));
}
var id = {
  id: "legend",
  _element: jn,
  start(i, t, e) {
    const s = i.legend = new jn({
      ctx: i.ctx,
      options: e,
      chart: i
    });
    ot.configure(i, s, e), ot.addBox(i, s);
  },
  stop(i) {
    ot.removeBox(i, i.legend), delete i.legend;
  },
  beforeUpdate(i, t, e) {
    const s = i.legend;
    ot.configure(i, s, e), s.options = e;
  },
  afterUpdate(i) {
    const t = i.legend;
    t.buildLabels(), t.adjustHitBoxes();
  },
  afterEvent(i, t) {
    t.replay || i.legend.handleEvent(t.event);
  },
  defaults: {
    display: !0,
    position: "top",
    align: "center",
    fullSize: !0,
    reverse: !1,
    weight: 1e3,
    onClick(i, t, e) {
      const s = t.datasetIndex, n = e.chart;
      n.isDatasetVisible(s) ? (n.hide(s), t.hidden = !0) : (n.show(s), t.hidden = !1);
    },
    onHover: null,
    onLeave: null,
    labels: {
      color: (i) => i.chart.options.color,
      boxWidth: 40,
      padding: 10,
      generateLabels(i) {
        const t = i.data.datasets, { labels: { usePointStyle: e, pointStyle: s, textAlign: n, color: o, useBorderRadius: r, borderRadius: a } } = i.legend.options;
        return i._getSortedDatasetMetas().map((l) => {
          const c = l.controller.getStyle(e ? 0 : void 0), h = rt(c.borderWidth);
          return {
            text: t[l.index].label,
            fillStyle: c.backgroundColor,
            fontColor: o,
            hidden: !l.visible,
            lineCap: c.borderCapStyle,
            lineDash: c.borderDash,
            lineDashOffset: c.borderDashOffset,
            lineJoin: c.borderJoinStyle,
            lineWidth: (h.width + h.height) / 4,
            strokeStyle: c.borderColor,
            pointStyle: s || c.pointStyle,
            rotation: c.rotation,
            textAlign: n || c.textAlign,
            borderRadius: r && (a || c.borderRadius),
            datasetIndex: l.index
          };
        }, this);
      }
    },
    title: {
      color: (i) => i.chart.options.color,
      display: !1,
      position: "center",
      text: ""
    }
  },
  descriptors: {
    _scriptable: (i) => !i.startsWith("on"),
    labels: {
      _scriptable: (i) => ![
        "generateLabels",
        "filter",
        "sort"
      ].includes(i)
    }
  }
};
class Ts extends mt {
  constructor(t) {
    super(), this.chart = t.chart, this.options = t.options, this.ctx = t.ctx, this._padding = void 0, this.top = void 0, this.bottom = void 0, this.left = void 0, this.right = void 0, this.width = void 0, this.height = void 0, this.position = void 0, this.weight = void 0, this.fullSize = void 0;
  }
  update(t, e) {
    const s = this.options;
    if (this.left = 0, this.top = 0, !s.display) {
      this.width = this.height = this.right = this.bottom = 0;
      return;
    }
    this.width = this.right = t, this.height = this.bottom = e;
    const n = Y(s.text) ? s.text.length : 1;
    this._padding = rt(s.padding);
    const o = n * G(s.font).lineHeight + this._padding.height;
    this.isHorizontal() ? this.height = o : this.width = o;
  }
  isHorizontal() {
    const t = this.options.position;
    return t === "top" || t === "bottom";
  }
  _drawArgs(t) {
    const { top: e, left: s, bottom: n, right: o, options: r } = this, a = r.align;
    let l = 0, c, h, d;
    return this.isHorizontal() ? (h = st(a, s, o), d = e + t, c = o - s) : (r.position === "left" ? (h = s + t, d = st(a, n, e), l = z * -0.5) : (h = o - t, d = st(a, e, n), l = z * 0.5), c = n - e), {
      titleX: h,
      titleY: d,
      maxWidth: c,
      rotation: l
    };
  }
  draw() {
    const t = this.ctx, e = this.options;
    if (!e.display)
      return;
    const s = G(e.font), o = s.lineHeight / 2 + this._padding.top, { titleX: r, titleY: a, maxWidth: l, rotation: c } = this._drawArgs(o);
    ee(t, e.text, 0, 0, s, {
      color: e.color,
      maxWidth: l,
      rotation: c,
      textAlign: ks(e.align),
      textBaseline: "middle",
      translation: [
        r,
        a
      ]
    });
  }
}
function sd(i, t) {
  const e = new Ts({
    ctx: i.ctx,
    options: t,
    chart: i
  });
  ot.configure(i, e, t), ot.addBox(i, e), i.titleBlock = e;
}
var nd = {
  id: "title",
  _element: Ts,
  start(i, t, e) {
    sd(i, e);
  },
  stop(i) {
    const t = i.titleBlock;
    ot.removeBox(i, t), delete i.titleBlock;
  },
  beforeUpdate(i, t, e) {
    const s = i.titleBlock;
    ot.configure(i, s, e), s.options = e;
  },
  defaults: {
    align: "center",
    display: !1,
    font: {
      weight: "bold"
    },
    fullSize: !0,
    padding: 10,
    position: "top",
    text: "",
    weight: 2e3
  },
  defaultRoutes: {
    color: "color"
  },
  descriptors: {
    _scriptable: !0,
    _indexable: !1
  }
};
const Qe = /* @__PURE__ */ new WeakMap();
var od = {
  id: "subtitle",
  start(i, t, e) {
    const s = new Ts({
      ctx: i.ctx,
      options: e,
      chart: i
    });
    ot.configure(i, s, e), ot.addBox(i, s), Qe.set(i, s);
  },
  stop(i) {
    ot.removeBox(i, Qe.get(i)), Qe.delete(i);
  },
  beforeUpdate(i, t, e) {
    const s = Qe.get(i);
    ot.configure(i, s, e), s.options = e;
  },
  defaults: {
    align: "center",
    display: !1,
    font: {
      weight: "normal"
    },
    fullSize: !0,
    padding: 0,
    position: "top",
    text: "",
    weight: 1500
  },
  defaultRoutes: {
    color: "color"
  },
  descriptors: {
    _scriptable: !0,
    _indexable: !1
  }
};
const ke = {
  average(i) {
    if (!i.length)
      return !1;
    let t, e, s = /* @__PURE__ */ new Set(), n = 0, o = 0;
    for (t = 0, e = i.length; t < e; ++t) {
      const a = i[t].element;
      if (a && a.hasValue()) {
        const l = a.tooltipPosition();
        s.add(l.x), n += l.y, ++o;
      }
    }
    return o === 0 || s.size === 0 ? !1 : {
      x: [
        ...s
      ].reduce((a, l) => a + l) / s.size,
      y: n / o
    };
  },
  nearest(i, t) {
    if (!i.length)
      return !1;
    let e = t.x, s = t.y, n = Number.POSITIVE_INFINITY, o, r, a;
    for (o = 0, r = i.length; o < r; ++o) {
      const l = i[o].element;
      if (l && l.hasValue()) {
        const c = l.getCenterPoint(), h = ts(t, c);
        h < n && (n = h, a = l);
      }
    }
    if (a) {
      const l = a.tooltipPosition();
      e = l.x, s = l.y;
    }
    return {
      x: e,
      y: s
    };
  }
};
function xt(i, t) {
  return t && (Y(t) ? Array.prototype.push.apply(i, t) : i.push(t)), i;
}
function Pt(i) {
  return (typeof i == "string" || i instanceof String) && i.indexOf(`
`) > -1 ? i.split(`
`) : i;
}
function rd(i, t) {
  const { element: e, datasetIndex: s, index: n } = t, o = i.getDatasetMeta(s).controller, { label: r, value: a } = o.getLabelAndValue(n);
  return {
    chart: i,
    label: r,
    parsed: o.getParsed(n),
    raw: i.data.datasets[s].data[n],
    formattedValue: a,
    dataset: o.getDataset(),
    dataIndex: n,
    datasetIndex: s,
    element: e
  };
}
function $n(i, t) {
  const e = i.chart.ctx, { body: s, footer: n, title: o } = i, { boxWidth: r, boxHeight: a } = t, l = G(t.bodyFont), c = G(t.titleFont), h = G(t.footerFont), d = o.length, f = n.length, u = s.length, g = rt(t.padding);
  let p = g.height, m = 0, b = s.reduce((v, x) => v + x.before.length + x.lines.length + x.after.length, 0);
  if (b += i.beforeBody.length + i.afterBody.length, d && (p += d * c.lineHeight + (d - 1) * t.titleSpacing + t.titleMarginBottom), b) {
    const v = t.displayColors ? Math.max(a, l.lineHeight) : l.lineHeight;
    p += u * v + (b - u) * l.lineHeight + (b - 1) * t.bodySpacing;
  }
  f && (p += t.footerMarginTop + f * h.lineHeight + (f - 1) * t.footerSpacing);
  let _ = 0;
  const y = function(v) {
    m = Math.max(m, e.measureText(v).width + _);
  };
  return e.save(), e.font = c.string, V(i.title, y), e.font = l.string, V(i.beforeBody.concat(i.afterBody), y), _ = t.displayColors ? r + 2 + t.boxPadding : 0, V(s, (v) => {
    V(v.before, y), V(v.lines, y), V(v.after, y);
  }), _ = 0, e.font = h.string, V(i.footer, y), e.restore(), m += g.width, {
    width: m,
    height: p
  };
}
function ad(i, t) {
  const { y: e, height: s } = t;
  return e < s / 2 ? "top" : e > i.height - s / 2 ? "bottom" : "center";
}
function ld(i, t, e, s) {
  const { x: n, width: o } = s, r = e.caretSize + e.caretPadding;
  if (i === "left" && n + o + r > t.width || i === "right" && n - o - r < 0)
    return !0;
}
function cd(i, t, e, s) {
  const { x: n, width: o } = e, { width: r, chartArea: { left: a, right: l } } = i;
  let c = "center";
  return s === "center" ? c = n <= (a + l) / 2 ? "left" : "right" : n <= o / 2 ? c = "left" : n >= r - o / 2 && (c = "right"), ld(c, i, t, e) && (c = "center"), c;
}
function Yn(i, t, e) {
  const s = e.yAlign || t.yAlign || ad(i, e);
  return {
    xAlign: e.xAlign || t.xAlign || cd(i, t, e, s),
    yAlign: s
  };
}
function hd(i, t) {
  let { x: e, width: s } = i;
  return t === "right" ? e -= s : t === "center" && (e -= s / 2), e;
}
function dd(i, t, e) {
  let { y: s, height: n } = i;
  return t === "top" ? s += e : t === "bottom" ? s -= n + e : s -= n / 2, s;
}
function Xn(i, t, e, s) {
  const { caretSize: n, caretPadding: o, cornerRadius: r } = i, { xAlign: a, yAlign: l } = e, c = n + o, { topLeft: h, topRight: d, bottomLeft: f, bottomRight: u } = Jt(r);
  let g = hd(t, a);
  const p = dd(t, l, c);
  return l === "center" ? a === "left" ? g += c : a === "right" && (g -= c) : a === "left" ? g -= Math.max(h, f) + n : a === "right" && (g += Math.max(d, u) + n), {
    x: J(g, 0, s.width - t.width),
    y: J(p, 0, s.height - t.height)
  };
}
function ti(i, t, e) {
  const s = rt(e.padding);
  return t === "center" ? i.x + i.width / 2 : t === "right" ? i.x + i.width - s.right : i.x + s.left;
}
function Un(i) {
  return xt([], Pt(i));
}
function fd(i, t, e) {
  return Ht(i, {
    tooltip: t,
    tooltipItems: e,
    type: "tooltip"
  });
}
function Kn(i, t) {
  const e = t && t.dataset && t.dataset.tooltip && t.dataset.tooltip.callbacks;
  return e ? i.override(e) : i;
}
const ur = {
  beforeTitle: wt,
  title(i) {
    if (i.length > 0) {
      const t = i[0], e = t.chart.data.labels, s = e ? e.length : 0;
      if (this && this.options && this.options.mode === "dataset")
        return t.dataset.label || "";
      if (t.label)
        return t.label;
      if (s > 0 && t.dataIndex < s)
        return e[t.dataIndex];
    }
    return "";
  },
  afterTitle: wt,
  beforeBody: wt,
  beforeLabel: wt,
  label(i) {
    if (this && this.options && this.options.mode === "dataset")
      return i.label + ": " + i.formattedValue || i.formattedValue;
    let t = i.dataset.label || "";
    t && (t += ": ");
    const e = i.formattedValue;
    return F(e) || (t += e), t;
  },
  labelColor(i) {
    const e = i.chart.getDatasetMeta(i.datasetIndex).controller.getStyle(i.dataIndex);
    return {
      borderColor: e.borderColor,
      backgroundColor: e.backgroundColor,
      borderWidth: e.borderWidth,
      borderDash: e.borderDash,
      borderDashOffset: e.borderDashOffset,
      borderRadius: 0
    };
  },
  labelTextColor() {
    return this.options.bodyColor;
  },
  labelPointStyle(i) {
    const e = i.chart.getDatasetMeta(i.datasetIndex).controller.getStyle(i.dataIndex);
    return {
      pointStyle: e.pointStyle,
      rotation: e.rotation
    };
  },
  afterLabel: wt,
  afterBody: wt,
  beforeFooter: wt,
  footer: wt,
  afterFooter: wt
};
function lt(i, t, e, s) {
  const n = i[t].call(e, s);
  return typeof n > "u" ? ur[t].call(e, s) : n;
}
class cs extends mt {
  constructor(t) {
    super(), this.opacity = 0, this._active = [], this._eventPosition = void 0, this._size = void 0, this._cachedAnimations = void 0, this._tooltipItems = [], this.$animations = void 0, this.$context = void 0, this.chart = t.chart, this.options = t.options, this.dataPoints = void 0, this.title = void 0, this.beforeBody = void 0, this.body = void 0, this.afterBody = void 0, this.footer = void 0, this.xAlign = void 0, this.yAlign = void 0, this.x = void 0, this.y = void 0, this.height = void 0, this.width = void 0, this.caretX = void 0, this.caretY = void 0, this.labelColors = void 0, this.labelPointStyles = void 0, this.labelTextColors = void 0;
  }
  initialize(t) {
    this.options = t, this._cachedAnimations = void 0, this.$context = void 0;
  }
  _resolveAnimations() {
    const t = this._cachedAnimations;
    if (t)
      return t;
    const e = this.chart, s = this.options.setContext(this.getContext()), n = s.enabled && e.options.animation && s.animations, o = new Xo(this.chart, n);
    return n._cacheable && (this._cachedAnimations = Object.freeze(o)), o;
  }
  getContext() {
    return this.$context || (this.$context = fd(this.chart.getContext(), this, this._tooltipItems));
  }
  getTitle(t, e) {
    const { callbacks: s } = e, n = lt(s, "beforeTitle", this, t), o = lt(s, "title", this, t), r = lt(s, "afterTitle", this, t);
    let a = [];
    return a = xt(a, Pt(n)), a = xt(a, Pt(o)), a = xt(a, Pt(r)), a;
  }
  getBeforeBody(t, e) {
    return Un(lt(e.callbacks, "beforeBody", this, t));
  }
  getBody(t, e) {
    const { callbacks: s } = e, n = [];
    return V(t, (o) => {
      const r = {
        before: [],
        lines: [],
        after: []
      }, a = Kn(s, o);
      xt(r.before, Pt(lt(a, "beforeLabel", this, o))), xt(r.lines, lt(a, "label", this, o)), xt(r.after, Pt(lt(a, "afterLabel", this, o))), n.push(r);
    }), n;
  }
  getAfterBody(t, e) {
    return Un(lt(e.callbacks, "afterBody", this, t));
  }
  getFooter(t, e) {
    const { callbacks: s } = e, n = lt(s, "beforeFooter", this, t), o = lt(s, "footer", this, t), r = lt(s, "afterFooter", this, t);
    let a = [];
    return a = xt(a, Pt(n)), a = xt(a, Pt(o)), a = xt(a, Pt(r)), a;
  }
  _createItems(t) {
    const e = this._active, s = this.chart.data, n = [], o = [], r = [];
    let a = [], l, c;
    for (l = 0, c = e.length; l < c; ++l)
      a.push(rd(this.chart, e[l]));
    return t.filter && (a = a.filter((h, d, f) => t.filter(h, d, f, s))), t.itemSort && (a = a.sort((h, d) => t.itemSort(h, d, s))), V(a, (h) => {
      const d = Kn(t.callbacks, h);
      n.push(lt(d, "labelColor", this, h)), o.push(lt(d, "labelPointStyle", this, h)), r.push(lt(d, "labelTextColor", this, h));
    }), this.labelColors = n, this.labelPointStyles = o, this.labelTextColors = r, this.dataPoints = a, a;
  }
  update(t, e) {
    const s = this.options.setContext(this.getContext()), n = this._active;
    let o, r = [];
    if (!n.length)
      this.opacity !== 0 && (o = {
        opacity: 0
      });
    else {
      const a = ke[s.position].call(this, n, this._eventPosition);
      r = this._createItems(s), this.title = this.getTitle(r, s), this.beforeBody = this.getBeforeBody(r, s), this.body = this.getBody(r, s), this.afterBody = this.getAfterBody(r, s), this.footer = this.getFooter(r, s);
      const l = this._size = $n(this, s), c = Object.assign({}, a, l), h = Yn(this.chart, s, c), d = Xn(s, c, h, this.chart);
      this.xAlign = h.xAlign, this.yAlign = h.yAlign, o = {
        opacity: 1,
        x: d.x,
        y: d.y,
        width: l.width,
        height: l.height,
        caretX: a.x,
        caretY: a.y
      };
    }
    this._tooltipItems = r, this.$context = void 0, o && this._resolveAnimations().update(this, o), t && s.external && s.external.call(this, {
      chart: this.chart,
      tooltip: this,
      replay: e
    });
  }
  drawCaret(t, e, s, n) {
    const o = this.getCaretPosition(t, s, n);
    e.lineTo(o.x1, o.y1), e.lineTo(o.x2, o.y2), e.lineTo(o.x3, o.y3);
  }
  getCaretPosition(t, e, s) {
    const { xAlign: n, yAlign: o } = this, { caretSize: r, cornerRadius: a } = s, { topLeft: l, topRight: c, bottomLeft: h, bottomRight: d } = Jt(a), { x: f, y: u } = t, { width: g, height: p } = e;
    let m, b, _, y, v, x;
    return o === "center" ? (v = u + p / 2, n === "left" ? (m = f, b = m - r, y = v + r, x = v - r) : (m = f + g, b = m + r, y = v - r, x = v + r), _ = m) : (n === "left" ? b = f + Math.max(l, h) + r : n === "right" ? b = f + g - Math.max(c, d) - r : b = this.caretX, o === "top" ? (y = u, v = y - r, m = b - r, _ = b + r) : (y = u + p, v = y + r, m = b + r, _ = b - r), x = y), {
      x1: m,
      x2: b,
      x3: _,
      y1: y,
      y2: v,
      y3: x
    };
  }
  drawTitle(t, e, s) {
    const n = this.title, o = n.length;
    let r, a, l;
    if (o) {
      const c = ce(s.rtl, this.x, this.width);
      for (t.x = ti(this, s.titleAlign, s), e.textAlign = c.textAlign(s.titleAlign), e.textBaseline = "middle", r = G(s.titleFont), a = s.titleSpacing, e.fillStyle = s.titleColor, e.font = r.string, l = 0; l < o; ++l)
        e.fillText(n[l], c.x(t.x), t.y + r.lineHeight / 2), t.y += r.lineHeight + a, l + 1 === o && (t.y += s.titleMarginBottom - a);
    }
  }
  _drawColorBox(t, e, s, n, o) {
    const r = this.labelColors[s], a = this.labelPointStyles[s], { boxHeight: l, boxWidth: c } = o, h = G(o.bodyFont), d = ti(this, "left", o), f = n.x(d), u = l < h.lineHeight ? (h.lineHeight - l) / 2 : 0, g = e.y + u;
    if (o.usePointStyle) {
      const p = {
        radius: Math.min(c, l) / 2,
        pointStyle: a.pointStyle,
        rotation: a.rotation,
        borderWidth: 1
      }, m = n.leftForLtr(f, c) + c / 2, b = g + l / 2;
      t.strokeStyle = o.multiKeyBackground, t.fillStyle = o.multiKeyBackground, is(t, p, m, b), t.strokeStyle = r.borderColor, t.fillStyle = r.backgroundColor, is(t, p, m, b);
    } else {
      t.lineWidth = I(r.borderWidth) ? Math.max(...Object.values(r.borderWidth)) : r.borderWidth || 1, t.strokeStyle = r.borderColor, t.setLineDash(r.borderDash || []), t.lineDashOffset = r.borderDashOffset || 0;
      const p = n.leftForLtr(f, c), m = n.leftForLtr(n.xPlus(f, 1), c - 2), b = Jt(r.borderRadius);
      Object.values(b).some((_) => _ !== 0) ? (t.beginPath(), t.fillStyle = o.multiKeyBackground, Ie(t, {
        x: p,
        y: g,
        w: c,
        h: l,
        radius: b
      }), t.fill(), t.stroke(), t.fillStyle = r.backgroundColor, t.beginPath(), Ie(t, {
        x: m,
        y: g + 1,
        w: c - 2,
        h: l - 2,
        radius: b
      }), t.fill()) : (t.fillStyle = o.multiKeyBackground, t.fillRect(p, g, c, l), t.strokeRect(p, g, c, l), t.fillStyle = r.backgroundColor, t.fillRect(m, g + 1, c - 2, l - 2));
    }
    t.fillStyle = this.labelTextColors[s];
  }
  drawBody(t, e, s) {
    const { body: n } = this, { bodySpacing: o, bodyAlign: r, displayColors: a, boxHeight: l, boxWidth: c, boxPadding: h } = s, d = G(s.bodyFont);
    let f = d.lineHeight, u = 0;
    const g = ce(s.rtl, this.x, this.width), p = function(w) {
      e.fillText(w, g.x(t.x + u), t.y + f / 2), t.y += f + o;
    }, m = g.textAlign(r);
    let b, _, y, v, x, S, k;
    for (e.textAlign = r, e.textBaseline = "middle", e.font = d.string, t.x = ti(this, m, s), e.fillStyle = s.bodyColor, V(this.beforeBody, p), u = a && m !== "right" ? r === "center" ? c / 2 + h : c + 2 + h : 0, v = 0, S = n.length; v < S; ++v) {
      for (b = n[v], _ = this.labelTextColors[v], e.fillStyle = _, V(b.before, p), y = b.lines, a && y.length && (this._drawColorBox(e, t, v, g, s), f = Math.max(d.lineHeight, l)), x = 0, k = y.length; x < k; ++x)
        p(y[x]), f = d.lineHeight;
      V(b.after, p);
    }
    u = 0, f = d.lineHeight, V(this.afterBody, p), t.y -= o;
  }
  drawFooter(t, e, s) {
    const n = this.footer, o = n.length;
    let r, a;
    if (o) {
      const l = ce(s.rtl, this.x, this.width);
      for (t.x = ti(this, s.footerAlign, s), t.y += s.footerMarginTop, e.textAlign = l.textAlign(s.footerAlign), e.textBaseline = "middle", r = G(s.footerFont), e.fillStyle = s.footerColor, e.font = r.string, a = 0; a < o; ++a)
        e.fillText(n[a], l.x(t.x), t.y + r.lineHeight / 2), t.y += r.lineHeight + s.footerSpacing;
    }
  }
  drawBackground(t, e, s, n) {
    const { xAlign: o, yAlign: r } = this, { x: a, y: l } = t, { width: c, height: h } = s, { topLeft: d, topRight: f, bottomLeft: u, bottomRight: g } = Jt(n.cornerRadius);
    e.fillStyle = n.backgroundColor, e.strokeStyle = n.borderColor, e.lineWidth = n.borderWidth, e.beginPath(), e.moveTo(a + d, l), r === "top" && this.drawCaret(t, e, s, n), e.lineTo(a + c - f, l), e.quadraticCurveTo(a + c, l, a + c, l + f), r === "center" && o === "right" && this.drawCaret(t, e, s, n), e.lineTo(a + c, l + h - g), e.quadraticCurveTo(a + c, l + h, a + c - g, l + h), r === "bottom" && this.drawCaret(t, e, s, n), e.lineTo(a + u, l + h), e.quadraticCurveTo(a, l + h, a, l + h - u), r === "center" && o === "left" && this.drawCaret(t, e, s, n), e.lineTo(a, l + d), e.quadraticCurveTo(a, l, a + d, l), e.closePath(), e.fill(), n.borderWidth > 0 && e.stroke();
  }
  _updateAnimationTarget(t) {
    const e = this.chart, s = this.$animations, n = s && s.x, o = s && s.y;
    if (n || o) {
      const r = ke[t.position].call(this, this._active, this._eventPosition);
      if (!r)
        return;
      const a = this._size = $n(this, t), l = Object.assign({}, r, this._size), c = Yn(e, t, l), h = Xn(t, l, c, e);
      (n._to !== h.x || o._to !== h.y) && (this.xAlign = c.xAlign, this.yAlign = c.yAlign, this.width = a.width, this.height = a.height, this.caretX = r.x, this.caretY = r.y, this._resolveAnimations().update(this, h));
    }
  }
  _willRender() {
    return !!this.opacity;
  }
  draw(t) {
    const e = this.options.setContext(this.getContext());
    let s = this.opacity;
    if (!s)
      return;
    this._updateAnimationTarget(e);
    const n = {
      width: this.width,
      height: this.height
    }, o = {
      x: this.x,
      y: this.y
    };
    s = Math.abs(s) < 1e-3 ? 0 : s;
    const r = rt(e.padding), a = this.title.length || this.beforeBody.length || this.body.length || this.afterBody.length || this.footer.length;
    e.enabled && a && (t.save(), t.globalAlpha = s, this.drawBackground(o, t, n, e), Wo(t, e.textDirection), o.y += r.top, this.drawTitle(o, t, e), this.drawBody(o, t, e), this.drawFooter(o, t, e), No(t, e.textDirection), t.restore());
  }
  getActiveElements() {
    return this._active || [];
  }
  setActiveElements(t, e) {
    const s = this._active, n = t.map(({ datasetIndex: a, index: l }) => {
      const c = this.chart.getDatasetMeta(a);
      if (!c)
        throw new Error("Cannot find a dataset at index " + a);
      return {
        datasetIndex: a,
        element: c.data[l],
        index: l
      };
    }), o = !pi(s, n), r = this._positionChanged(n, e);
    (o || r) && (this._active = n, this._eventPosition = e, this._ignoreReplayEvents = !0, this.update(!0));
  }
  handleEvent(t, e, s = !0) {
    if (e && this._ignoreReplayEvents)
      return !1;
    this._ignoreReplayEvents = !1;
    const n = this.options, o = this._active || [], r = this._getActiveElements(t, o, e, s), a = this._positionChanged(r, t), l = e || !pi(r, o) || a;
    return l && (this._active = r, (n.enabled || n.external) && (this._eventPosition = {
      x: t.x,
      y: t.y
    }, this.update(!0, e))), l;
  }
  _getActiveElements(t, e, s, n) {
    const o = this.options;
    if (t.type === "mouseout")
      return [];
    if (!n)
      return e.filter((a) => this.chart.data.datasets[a.datasetIndex] && this.chart.getDatasetMeta(a.datasetIndex).controller.getParsed(a.index) !== void 0);
    const r = this.chart.getElementsAtEventForMode(t, o.mode, o, s);
    return o.reverse && r.reverse(), r;
  }
  _positionChanged(t, e) {
    const { caretX: s, caretY: n, options: o } = this, r = ke[o.position].call(this, t, e);
    return r !== !1 && (s !== r.x || n !== r.y);
  }
}
M(cs, "positioners", ke);
var ud = {
  id: "tooltip",
  _element: cs,
  positioners: ke,
  afterInit(i, t, e) {
    e && (i.tooltip = new cs({
      chart: i,
      options: e
    }));
  },
  beforeUpdate(i, t, e) {
    i.tooltip && i.tooltip.initialize(e);
  },
  reset(i, t, e) {
    i.tooltip && i.tooltip.initialize(e);
  },
  afterDraw(i) {
    const t = i.tooltip;
    if (t && t._willRender()) {
      const e = {
        tooltip: t
      };
      if (i.notifyPlugins("beforeTooltipDraw", {
        ...e,
        cancelable: !0
      }) === !1)
        return;
      t.draw(i.ctx), i.notifyPlugins("afterTooltipDraw", e);
    }
  },
  afterEvent(i, t) {
    if (i.tooltip) {
      const e = t.replay;
      i.tooltip.handleEvent(t.event, e, t.inChartArea) && (t.changed = !0);
    }
  },
  defaults: {
    enabled: !0,
    external: null,
    position: "average",
    backgroundColor: "rgba(0,0,0,0.8)",
    titleColor: "#fff",
    titleFont: {
      weight: "bold"
    },
    titleSpacing: 2,
    titleMarginBottom: 6,
    titleAlign: "left",
    bodyColor: "#fff",
    bodySpacing: 2,
    bodyFont: {},
    bodyAlign: "left",
    footerColor: "#fff",
    footerSpacing: 2,
    footerMarginTop: 6,
    footerFont: {
      weight: "bold"
    },
    footerAlign: "left",
    padding: 6,
    caretPadding: 2,
    caretSize: 5,
    cornerRadius: 6,
    boxHeight: (i, t) => t.bodyFont.size,
    boxWidth: (i, t) => t.bodyFont.size,
    multiKeyBackground: "#fff",
    displayColors: !0,
    boxPadding: 0,
    borderColor: "rgba(0,0,0,0)",
    borderWidth: 0,
    animation: {
      duration: 400,
      easing: "easeOutQuart"
    },
    animations: {
      numbers: {
        type: "number",
        properties: [
          "x",
          "y",
          "width",
          "height",
          "caretX",
          "caretY"
        ]
      },
      opacity: {
        easing: "linear",
        duration: 200
      }
    },
    callbacks: ur
  },
  defaultRoutes: {
    bodyFont: "font",
    footerFont: "font",
    titleFont: "font"
  },
  descriptors: {
    _scriptable: (i) => i !== "filter" && i !== "itemSort" && i !== "external",
    _indexable: !1,
    callbacks: {
      _scriptable: !1,
      _indexable: !1
    },
    animation: {
      _fallback: !1
    },
    animations: {
      _fallback: "animation"
    }
  },
  additionalOptionScopes: [
    "interaction"
  ]
}, gd = /* @__PURE__ */ Object.freeze({
  __proto__: null,
  Colors: Ph,
  Decimation: Oh,
  Filler: Gh,
  Legend: id,
  SubTitle: od,
  Title: nd,
  Tooltip: ud
});
const pd = (i, t, e, s) => (typeof t == "string" ? (e = i.push(t) - 1, s.unshift({
  index: e,
  label: t
})) : isNaN(t) && (e = null), e);
function md(i, t, e, s) {
  const n = i.indexOf(t);
  if (n === -1)
    return pd(i, t, e, s);
  const o = i.lastIndexOf(t);
  return n !== o ? e : n;
}
const bd = (i, t) => i === null ? null : J(Math.round(i), 0, t);
function qn(i) {
  const t = this.getLabels();
  return i >= 0 && i < t.length ? t[i] : i;
}
class hs extends ie {
  constructor(t) {
    super(t), this._startValue = void 0, this._valueRange = 0, this._addedLabels = [];
  }
  init(t) {
    const e = this._addedLabels;
    if (e.length) {
      const s = this.getLabels();
      for (const { index: n, label: o } of e)
        s[n] === o && s.splice(n, 1);
      this._addedLabels = [];
    }
    super.init(t);
  }
  parse(t, e) {
    if (F(t))
      return null;
    const s = this.getLabels();
    return e = isFinite(e) && s[e] === t ? e : md(s, t, O(e, t), this._addedLabels), bd(e, s.length - 1);
  }
  determineDataLimits() {
    const { minDefined: t, maxDefined: e } = this.getUserBounds();
    let { min: s, max: n } = this.getMinMax(!0);
    this.options.bounds === "ticks" && (t || (s = 0), e || (n = this.getLabels().length - 1)), this.min = s, this.max = n;
  }
  buildTicks() {
    const t = this.min, e = this.max, s = this.options.offset, n = [];
    let o = this.getLabels();
    o = t === 0 && e === o.length - 1 ? o : o.slice(t, e + 1), this._valueRange = Math.max(o.length - (s ? 0 : 1), 1), this._startValue = this.min - (s ? 0.5 : 0);
    for (let r = t; r <= e; r++)
      n.push({
        value: r
      });
    return n;
  }
  getLabelForValue(t) {
    return qn.call(this, t);
  }
  configure() {
    super.configure(), this.isHorizontal() || (this._reversePixels = !this._reversePixels);
  }
  getPixelForValue(t) {
    return typeof t != "number" && (t = this.parse(t)), t === null ? NaN : this.getPixelForDecimal((t - this._startValue) / this._valueRange);
  }
  getPixelForTick(t) {
    const e = this.ticks;
    return t < 0 || t > e.length - 1 ? null : this.getPixelForValue(e[t].value);
  }
  getValueForPixel(t) {
    return Math.round(this._startValue + this.getDecimalForPixel(t) * this._valueRange);
  }
  getBasePixel() {
    return this.bottom;
  }
}
M(hs, "id", "category"), M(hs, "defaults", {
  ticks: {
    callback: qn
  }
});
function _d(i, t) {
  const e = [], { bounds: n, step: o, min: r, max: a, precision: l, count: c, maxTicks: h, maxDigits: d, includeBounds: f } = i, u = o || 1, g = h - 1, { min: p, max: m } = t, b = !F(r), _ = !F(a), y = !F(c), v = (m - p) / (d + 1);
  let x = js((m - p) / g / u) * u, S, k, w, P;
  if (x < 1e-14 && !b && !_)
    return [
      {
        value: p
      },
      {
        value: m
      }
    ];
  P = Math.ceil(m / x) - Math.floor(p / x), P > g && (x = js(P * x / g / u) * u), F(l) || (S = Math.pow(10, l), x = Math.ceil(x * S) / S), n === "ticks" ? (k = Math.floor(p / x) * x, w = Math.ceil(m / x) * x) : (k = p, w = m), b && _ && o && ha((a - r) / o, x / 1e3) ? (P = Math.round(Math.min((a - r) / x, h)), x = (a - r) / P, k = r, w = a) : y ? (k = b ? r : k, w = _ ? a : w, P = c - 1, x = (w - k) / P) : (P = (w - k) / x, De(P, Math.round(P), x / 1e3) ? P = Math.round(P) : P = Math.ceil(P));
  const T = Math.max($s(x), $s(k));
  S = Math.pow(10, F(l) ? T : l), k = Math.round(k * S) / S, w = Math.round(w * S) / S;
  let L = 0;
  for (b && (f && k !== r ? (e.push({
    value: r
  }), k < r && L++, De(Math.round((k + L * x) * S) / S, r, Gn(r, v, i)) && L++) : k < r && L++); L < P; ++L) {
    const R = Math.round((k + L * x) * S) / S;
    if (_ && R > a)
      break;
    e.push({
      value: R
    });
  }
  return _ && f && w !== a ? e.length && De(e[e.length - 1].value, a, Gn(a, v, i)) ? e[e.length - 1].value = a : e.push({
    value: a
  }) : (!_ || w === a) && e.push({
    value: w
  }), e;
}
function Gn(i, t, { horizontal: e, minRotation: s }) {
  const n = gt(s), o = (e ? Math.sin(n) : Math.cos(n)) || 1e-3, r = 0.75 * t * ("" + i).length;
  return Math.min(t / o, r);
}
class ki extends ie {
  constructor(t) {
    super(t), this.start = void 0, this.end = void 0, this._startValue = void 0, this._endValue = void 0, this._valueRange = 0;
  }
  parse(t, e) {
    return F(t) || (typeof t == "number" || t instanceof Number) && !isFinite(+t) ? null : +t;
  }
  handleTickRangeOptions() {
    const { beginAtZero: t } = this.options, { minDefined: e, maxDefined: s } = this.getUserBounds();
    let { min: n, max: o } = this;
    const r = (l) => n = e ? n : l, a = (l) => o = s ? o : l;
    if (t) {
      const l = kt(n), c = kt(o);
      l < 0 && c < 0 ? a(0) : l > 0 && c > 0 && r(0);
    }
    if (n === o) {
      let l = o === 0 ? 1 : Math.abs(o * 0.05);
      a(o + l), t || r(n - l);
    }
    this.min = n, this.max = o;
  }
  getTickLimit() {
    const t = this.options.ticks;
    let { maxTicksLimit: e, stepSize: s } = t, n;
    return s ? (n = Math.ceil(this.max / s) - Math.floor(this.min / s) + 1, n > 1e3 && (console.warn(`scales.${this.id}.ticks.stepSize: ${s} would result generating up to ${n} ticks. Limiting to 1000.`), n = 1e3)) : (n = this.computeTickLimit(), e = e || 11), e && (n = Math.min(e, n)), n;
  }
  computeTickLimit() {
    return Number.POSITIVE_INFINITY;
  }
  buildTicks() {
    const t = this.options, e = t.ticks;
    let s = this.getTickLimit();
    s = Math.max(2, s);
    const n = {
      maxTicks: s,
      bounds: t.bounds,
      min: t.min,
      max: t.max,
      precision: e.precision,
      step: e.stepSize,
      count: e.count,
      maxDigits: this._maxDigits(),
      horizontal: this.isHorizontal(),
      minRotation: e.minRotation || 0,
      includeBounds: e.includeBounds !== !1
    }, o = this._range || this, r = _d(n, o);
    return t.bounds === "ticks" && Mo(r, this, "value"), t.reverse ? (r.reverse(), this.start = this.max, this.end = this.min) : (this.start = this.min, this.end = this.max), r;
  }
  configure() {
    const t = this.ticks;
    let e = this.min, s = this.max;
    if (super.configure(), this.options.offset && t.length) {
      const n = (s - e) / Math.max(t.length - 1, 1) / 2;
      e -= n, s += n;
    }
    this._startValue = e, this._endValue = s, this._valueRange = s - e;
  }
  getLabelForValue(t) {
    return He(t, this.chart.options.locale, this.options.ticks.format);
  }
}
class ds extends ki {
  determineDataLimits() {
    const { min: t, max: e } = this.getMinMax(!0);
    this.min = U(t) ? t : 0, this.max = U(e) ? e : 1, this.handleTickRangeOptions();
  }
  computeTickLimit() {
    const t = this.isHorizontal(), e = t ? this.width : this.height, s = gt(this.options.ticks.minRotation), n = (t ? Math.sin(s) : Math.cos(s)) || 1e-3, o = this._resolveTickFontOptions(0);
    return Math.ceil(e / Math.min(40, o.lineHeight / n));
  }
  getPixelForValue(t) {
    return t === null ? NaN : this.getPixelForDecimal((t - this._startValue) / this._valueRange);
  }
  getValueForPixel(t) {
    return this._startValue + this.getDecimalForPixel(t) * this._valueRange;
  }
}
M(ds, "id", "linear"), M(ds, "defaults", {
  ticks: {
    callback: Pi.formatters.numeric
  }
});
const Be = (i) => Math.floor(Ft(i)), Ut = (i, t) => Math.pow(10, Be(i) + t);
function Zn(i) {
  return i / Math.pow(10, Be(i)) === 1;
}
function Jn(i, t, e) {
  const s = Math.pow(10, e), n = Math.floor(i / s);
  return Math.ceil(t / s) - n;
}
function xd(i, t) {
  const e = t - i;
  let s = Be(e);
  for (; Jn(i, t, s) > 10; )
    s++;
  for (; Jn(i, t, s) < 10; )
    s--;
  return Math.min(s, Be(i));
}
function yd(i, { min: t, max: e }) {
  t = dt(i.min, t);
  const s = [], n = Be(t);
  let o = xd(t, e), r = o < 0 ? Math.pow(10, Math.abs(o)) : 1;
  const a = Math.pow(10, o), l = n > o ? Math.pow(10, n) : 0, c = Math.round((t - l) * r) / r, h = Math.floor((t - l) / a / 10) * a * 10;
  let d = Math.floor((c - h) / Math.pow(10, o)), f = dt(i.min, Math.round((l + h + d * Math.pow(10, o)) * r) / r);
  for (; f < e; )
    s.push({
      value: f,
      major: Zn(f),
      significand: d
    }), d >= 10 ? d = d < 15 ? 15 : 20 : d++, d >= 20 && (o++, d = 2, r = o >= 0 ? 1 : r), f = Math.round((l + h + d * Math.pow(10, o)) * r) / r;
  const u = dt(i.max, f);
  return s.push({
    value: u,
    major: Zn(u),
    significand: d
  }), s;
}
class fs extends ie {
  constructor(t) {
    super(t), this.start = void 0, this.end = void 0, this._startValue = void 0, this._valueRange = 0;
  }
  parse(t, e) {
    const s = ki.prototype.parse.apply(this, [
      t,
      e
    ]);
    if (s === 0) {
      this._zero = !0;
      return;
    }
    return U(s) && s > 0 ? s : null;
  }
  determineDataLimits() {
    const { min: t, max: e } = this.getMinMax(!0);
    this.min = U(t) ? Math.max(0, t) : null, this.max = U(e) ? Math.max(0, e) : null, this.options.beginAtZero && (this._zero = !0), this._zero && this.min !== this._suggestedMin && !U(this._userMin) && (this.min = t === Ut(this.min, 0) ? Ut(this.min, -1) : Ut(this.min, 0)), this.handleTickRangeOptions();
  }
  handleTickRangeOptions() {
    const { minDefined: t, maxDefined: e } = this.getUserBounds();
    let s = this.min, n = this.max;
    const o = (a) => s = t ? s : a, r = (a) => n = e ? n : a;
    s === n && (s <= 0 ? (o(1), r(10)) : (o(Ut(s, -1)), r(Ut(n, 1)))), s <= 0 && o(Ut(n, -1)), n <= 0 && r(Ut(s, 1)), this.min = s, this.max = n;
  }
  buildTicks() {
    const t = this.options, e = {
      min: this._userMin,
      max: this._userMax
    }, s = yd(e, this);
    return t.bounds === "ticks" && Mo(s, this, "value"), t.reverse ? (s.reverse(), this.start = this.max, this.end = this.min) : (this.start = this.min, this.end = this.max), s;
  }
  getLabelForValue(t) {
    return t === void 0 ? "0" : He(t, this.chart.options.locale, this.options.ticks.format);
  }
  configure() {
    const t = this.min;
    super.configure(), this._startValue = Ft(t), this._valueRange = Ft(this.max) - Ft(t);
  }
  getPixelForValue(t) {
    return (t === void 0 || t === 0) && (t = this.min), t === null || isNaN(t) ? NaN : this.getPixelForDecimal(t === this.min ? 0 : (Ft(t) - this._startValue) / this._valueRange);
  }
  getValueForPixel(t) {
    const e = this.getDecimalForPixel(t);
    return Math.pow(10, this._startValue + e * this._valueRange);
  }
}
M(fs, "id", "logarithmic"), M(fs, "defaults", {
  ticks: {
    callback: Pi.formatters.logarithmic,
    major: {
      enabled: !0
    }
  }
});
function us(i) {
  const t = i.ticks;
  if (t.display && i.display) {
    const e = rt(t.backdropPadding);
    return O(t.font && t.font.size, X.font.size) + e.height;
  }
  return 0;
}
function vd(i, t, e) {
  return e = Y(e) ? e : [
    e
  ], {
    w: Da(i, t.string, e),
    h: e.length * t.lineHeight
  };
}
function Qn(i, t, e, s, n) {
  return i === s || i === n ? {
    start: t - e / 2,
    end: t + e / 2
  } : i < s || i > n ? {
    start: t - e,
    end: t
  } : {
    start: t,
    end: t + e
  };
}
function kd(i) {
  const t = {
    l: i.left + i._padding.left,
    r: i.right - i._padding.right,
    t: i.top + i._padding.top,
    b: i.bottom - i._padding.bottom
  }, e = Object.assign({}, t), s = [], n = [], o = i._pointLabels.length, r = i.options.pointLabels, a = r.centerPointLabels ? z / o : 0;
  for (let l = 0; l < o; l++) {
    const c = r.setContext(i.getPointLabelContext(l));
    n[l] = c.padding;
    const h = i.getPointPosition(l, i.drawingArea + n[l], a), d = G(c.font), f = vd(i.ctx, d, i._pointLabels[l]);
    s[l] = f;
    const u = nt(i.getIndexAngle(l) + a), g = Math.round(ys(u)), p = Qn(g, h.x, f.w, 0, 180), m = Qn(g, h.y, f.h, 90, 270);
    Md(e, t, u, p, m);
  }
  i.setCenterPoint(t.l - e.l, e.r - t.r, t.t - e.t, e.b - t.b), i._pointLabelItems = Pd(i, s, n);
}
function Md(i, t, e, s, n) {
  const o = Math.abs(Math.sin(e)), r = Math.abs(Math.cos(e));
  let a = 0, l = 0;
  s.start < t.l ? (a = (t.l - s.start) / o, i.l = Math.min(i.l, t.l - a)) : s.end > t.r && (a = (s.end - t.r) / o, i.r = Math.max(i.r, t.r + a)), n.start < t.t ? (l = (t.t - n.start) / r, i.t = Math.min(i.t, t.t - l)) : n.end > t.b && (l = (n.end - t.b) / r, i.b = Math.max(i.b, t.b + l));
}
function wd(i, t, e) {
  const s = i.drawingArea, { extra: n, additionalAngle: o, padding: r, size: a } = e, l = i.getPointPosition(t, s + n + r, o), c = Math.round(ys(nt(l.angle + K))), h = Cd(l.y, a.h, c), d = Dd(c), f = Ad(l.x, a.w, d);
  return {
    visible: !0,
    x: l.x,
    y: h,
    textAlign: d,
    left: f,
    top: h,
    right: f + a.w,
    bottom: h + a.h
  };
}
function Sd(i, t) {
  if (!t)
    return !0;
  const { left: e, top: s, right: n, bottom: o } = i;
  return !(Tt({
    x: e,
    y: s
  }, t) || Tt({
    x: e,
    y: o
  }, t) || Tt({
    x: n,
    y: s
  }, t) || Tt({
    x: n,
    y: o
  }, t));
}
function Pd(i, t, e) {
  const s = [], n = i._pointLabels.length, o = i.options, { centerPointLabels: r, display: a } = o.pointLabels, l = {
    extra: us(o) / 2,
    additionalAngle: r ? z / n : 0
  };
  let c;
  for (let h = 0; h < n; h++) {
    l.padding = e[h], l.size = t[h];
    const d = wd(i, h, l);
    s.push(d), a === "auto" && (d.visible = Sd(d, c), d.visible && (c = d));
  }
  return s;
}
function Dd(i) {
  return i === 0 || i === 180 ? "center" : i < 180 ? "left" : "right";
}
function Ad(i, t, e) {
  return e === "right" ? i -= t : e === "center" && (i -= t / 2), i;
}
function Cd(i, t, e) {
  return e === 90 || e === 270 ? i -= t / 2 : (e > 270 || e < 90) && (i -= t), i;
}
function Od(i, t, e) {
  const { left: s, top: n, right: o, bottom: r } = e, { backdropColor: a } = t;
  if (!F(a)) {
    const l = Jt(t.borderRadius), c = rt(t.backdropPadding);
    i.fillStyle = a;
    const h = s - c.left, d = n - c.top, f = o - s + c.width, u = r - n + c.height;
    Object.values(l).some((g) => g !== 0) ? (i.beginPath(), Ie(i, {
      x: h,
      y: d,
      w: f,
      h: u,
      radius: l
    }), i.fill()) : i.fillRect(h, d, f, u);
  }
}
function Td(i, t) {
  const { ctx: e, options: { pointLabels: s } } = i;
  for (let n = t - 1; n >= 0; n--) {
    const o = i._pointLabelItems[n];
    if (!o.visible)
      continue;
    const r = s.setContext(i.getPointLabelContext(n));
    Od(e, r, o);
    const a = G(r.font), { x: l, y: c, textAlign: h } = o;
    ee(e, i._pointLabels[n], l, c + a.lineHeight / 2, a, {
      color: r.color,
      textAlign: h,
      textBaseline: "middle"
    });
  }
}
function gr(i, t, e, s) {
  const { ctx: n } = i;
  if (e)
    n.arc(i.xCenter, i.yCenter, t, 0, j);
  else {
    let o = i.getPointPosition(0, t);
    n.moveTo(o.x, o.y);
    for (let r = 1; r < s; r++)
      o = i.getPointPosition(r, t), n.lineTo(o.x, o.y);
  }
}
function Ld(i, t, e, s, n) {
  const o = i.ctx, r = t.circular, { color: a, lineWidth: l } = t;
  !r && !s || !a || !l || e < 0 || (o.save(), o.strokeStyle = a, o.lineWidth = l, o.setLineDash(n.dash || []), o.lineDashOffset = n.dashOffset, o.beginPath(), gr(i, e, r, s), o.closePath(), o.stroke(), o.restore());
}
function Rd(i, t, e) {
  return Ht(i, {
    label: e,
    index: t,
    type: "pointLabel"
  });
}
class Me extends ki {
  constructor(t) {
    super(t), this.xCenter = void 0, this.yCenter = void 0, this.drawingArea = void 0, this._pointLabels = [], this._pointLabelItems = [];
  }
  setDimensions() {
    const t = this._padding = rt(us(this.options) / 2), e = this.width = this.maxWidth - t.width, s = this.height = this.maxHeight - t.height;
    this.xCenter = Math.floor(this.left + e / 2 + t.left), this.yCenter = Math.floor(this.top + s / 2 + t.top), this.drawingArea = Math.floor(Math.min(e, s) / 2);
  }
  determineDataLimits() {
    const { min: t, max: e } = this.getMinMax(!1);
    this.min = U(t) && !isNaN(t) ? t : 0, this.max = U(e) && !isNaN(e) ? e : 0, this.handleTickRangeOptions();
  }
  computeTickLimit() {
    return Math.ceil(this.drawingArea / us(this.options));
  }
  generateTickLabels(t) {
    ki.prototype.generateTickLabels.call(this, t), this._pointLabels = this.getLabels().map((e, s) => {
      const n = N(this.options.pointLabels.callback, [
        e,
        s
      ], this);
      return n || n === 0 ? n : "";
    }).filter((e, s) => this.chart.getDataVisibility(s));
  }
  fit() {
    const t = this.options;
    t.display && t.pointLabels.display ? kd(this) : this.setCenterPoint(0, 0, 0, 0);
  }
  setCenterPoint(t, e, s, n) {
    this.xCenter += Math.floor((t - e) / 2), this.yCenter += Math.floor((s - n) / 2), this.drawingArea -= Math.min(this.drawingArea / 2, Math.max(t, e, s, n));
  }
  getIndexAngle(t) {
    const e = j / (this._pointLabels.length || 1), s = this.options.startAngle || 0;
    return nt(t * e + gt(s));
  }
  getDistanceFromCenterForValue(t) {
    if (F(t))
      return NaN;
    const e = this.drawingArea / (this.max - this.min);
    return this.options.reverse ? (this.max - t) * e : (t - this.min) * e;
  }
  getValueForDistanceFromCenter(t) {
    if (F(t))
      return NaN;
    const e = t / (this.drawingArea / (this.max - this.min));
    return this.options.reverse ? this.max - e : this.min + e;
  }
  getPointLabelContext(t) {
    const e = this._pointLabels || [];
    if (t >= 0 && t < e.length) {
      const s = e[t];
      return Rd(this.getContext(), t, s);
    }
  }
  getPointPosition(t, e, s = 0) {
    const n = this.getIndexAngle(t) - K + s;
    return {
      x: Math.cos(n) * e + this.xCenter,
      y: Math.sin(n) * e + this.yCenter,
      angle: n
    };
  }
  getPointPositionForValue(t, e) {
    return this.getPointPosition(t, this.getDistanceFromCenterForValue(e));
  }
  getBasePosition(t) {
    return this.getPointPositionForValue(t || 0, this.getBaseValue());
  }
  getPointLabelPosition(t) {
    const { left: e, top: s, right: n, bottom: o } = this._pointLabelItems[t];
    return {
      left: e,
      top: s,
      right: n,
      bottom: o
    };
  }
  drawBackground() {
    const { backgroundColor: t, grid: { circular: e } } = this.options;
    if (t) {
      const s = this.ctx;
      s.save(), s.beginPath(), gr(this, this.getDistanceFromCenterForValue(this._endValue), e, this._pointLabels.length), s.closePath(), s.fillStyle = t, s.fill(), s.restore();
    }
  }
  drawGrid() {
    const t = this.ctx, e = this.options, { angleLines: s, grid: n, border: o } = e, r = this._pointLabels.length;
    let a, l, c;
    if (e.pointLabels.display && Td(this, r), n.display && this.ticks.forEach((h, d) => {
      if (d !== 0 || d === 0 && this.min < 0) {
        l = this.getDistanceFromCenterForValue(h.value);
        const f = this.getContext(d), u = n.setContext(f), g = o.setContext(f);
        Ld(this, u, l, r, g);
      }
    }), s.display) {
      for (t.save(), a = r - 1; a >= 0; a--) {
        const h = s.setContext(this.getPointLabelContext(a)), { color: d, lineWidth: f } = h;
        !f || !d || (t.lineWidth = f, t.strokeStyle = d, t.setLineDash(h.borderDash), t.lineDashOffset = h.borderDashOffset, l = this.getDistanceFromCenterForValue(e.reverse ? this.min : this.max), c = this.getPointPosition(a, l), t.beginPath(), t.moveTo(this.xCenter, this.yCenter), t.lineTo(c.x, c.y), t.stroke());
      }
      t.restore();
    }
  }
  drawBorder() {
  }
  drawLabels() {
    const t = this.ctx, e = this.options, s = e.ticks;
    if (!s.display)
      return;
    const n = this.getIndexAngle(0);
    let o, r;
    t.save(), t.translate(this.xCenter, this.yCenter), t.rotate(n), t.textAlign = "center", t.textBaseline = "middle", this.ticks.forEach((a, l) => {
      if (l === 0 && this.min >= 0 && !e.reverse)
        return;
      const c = s.setContext(this.getContext(l)), h = G(c.font);
      if (o = this.getDistanceFromCenterForValue(this.ticks[l].value), c.showLabelBackdrop) {
        t.font = h.string, r = t.measureText(a.label).width, t.fillStyle = c.backdropColor;
        const d = rt(c.backdropPadding);
        t.fillRect(-r / 2 - d.left, -o - h.size / 2 - d.top, r + d.width, h.size + d.height);
      }
      ee(t, a.label, 0, -o, h, {
        color: c.color,
        strokeColor: c.textStrokeColor,
        strokeWidth: c.textStrokeWidth
      });
    }), t.restore();
  }
  drawTitle() {
  }
}
M(Me, "id", "radialLinear"), M(Me, "defaults", {
  display: !0,
  animate: !0,
  position: "chartArea",
  angleLines: {
    display: !0,
    lineWidth: 1,
    borderDash: [],
    borderDashOffset: 0
  },
  grid: {
    circular: !1
  },
  startAngle: 0,
  ticks: {
    showLabelBackdrop: !0,
    callback: Pi.formatters.numeric
  },
  pointLabels: {
    backdropColor: void 0,
    backdropPadding: 2,
    display: !0,
    font: {
      size: 10
    },
    callback(t) {
      return t;
    },
    padding: 5,
    centerPointLabels: !1
  }
}), M(Me, "defaultRoutes", {
  "angleLines.color": "borderColor",
  "pointLabels.color": "color",
  "ticks.color": "color"
}), M(Me, "descriptors", {
  angleLines: {
    _fallback: "grid"
  }
});
const Li = {
  millisecond: {
    common: !0,
    size: 1,
    steps: 1e3
  },
  second: {
    common: !0,
    size: 1e3,
    steps: 60
  },
  minute: {
    common: !0,
    size: 6e4,
    steps: 60
  },
  hour: {
    common: !0,
    size: 36e5,
    steps: 24
  },
  day: {
    common: !0,
    size: 864e5,
    steps: 30
  },
  week: {
    common: !1,
    size: 6048e5,
    steps: 4
  },
  month: {
    common: !0,
    size: 2628e6,
    steps: 12
  },
  quarter: {
    common: !1,
    size: 7884e6,
    steps: 4
  },
  year: {
    common: !0,
    size: 3154e7
  }
}, ct = /* @__PURE__ */ Object.keys(Li);
function to(i, t) {
  return i - t;
}
function eo(i, t) {
  if (F(t))
    return null;
  const e = i._adapter, { parser: s, round: n, isoWeekday: o } = i._parseOpts;
  let r = t;
  return typeof s == "function" && (r = s(r)), U(r) || (r = typeof s == "string" ? e.parse(r, s) : e.parse(r)), r === null ? null : (n && (r = n === "week" && (he(o) || o === !0) ? e.startOf(r, "isoWeek", o) : e.startOf(r, n)), +r);
}
function io(i, t, e, s) {
  const n = ct.length;
  for (let o = ct.indexOf(i); o < n - 1; ++o) {
    const r = Li[ct[o]], a = r.steps ? r.steps : Number.MAX_SAFE_INTEGER;
    if (r.common && Math.ceil((e - t) / (a * r.size)) <= s)
      return ct[o];
  }
  return ct[n - 1];
}
function Ed(i, t, e, s, n) {
  for (let o = ct.length - 1; o >= ct.indexOf(e); o--) {
    const r = ct[o];
    if (Li[r].common && i._adapter.diff(n, s, r) >= t - 1)
      return r;
  }
  return ct[e ? ct.indexOf(e) : 0];
}
function Fd(i) {
  for (let t = ct.indexOf(i) + 1, e = ct.length; t < e; ++t)
    if (Li[ct[t]].common)
      return ct[t];
}
function so(i, t, e) {
  if (!e)
    i[t] = !0;
  else if (e.length) {
    const { lo: s, hi: n } = vs(e, t), o = e[s] >= t ? e[s] : e[n];
    i[o] = !0;
  }
}
function Id(i, t, e, s) {
  const n = i._adapter, o = +n.startOf(t[0].value, s), r = t[t.length - 1].value;
  let a, l;
  for (a = o; a <= r; a = +n.add(a, 1, s))
    l = e[a], l >= 0 && (t[l].major = !0);
  return t;
}
function no(i, t, e) {
  const s = [], n = {}, o = t.length;
  let r, a;
  for (r = 0; r < o; ++r)
    a = t[r], n[a] = r, s.push({
      value: a,
      major: !1
    });
  return o === 0 || !e ? s : Id(i, s, n, e);
}
class Ve extends ie {
  constructor(t) {
    super(t), this._cache = {
      data: [],
      labels: [],
      all: []
    }, this._unit = "day", this._majorUnit = void 0, this._offsets = {}, this._normalized = !1, this._parseOpts = void 0;
  }
  init(t, e = {}) {
    const s = t.time || (t.time = {}), n = this._adapter = new $l._date(t.adapters.date);
    n.init(e), Pe(s.displayFormats, n.formats()), this._parseOpts = {
      parser: s.parser,
      round: s.round,
      isoWeekday: s.isoWeekday
    }, super.init(t), this._normalized = e.normalized;
  }
  parse(t, e) {
    return t === void 0 ? null : eo(this, t);
  }
  beforeLayout() {
    super.beforeLayout(), this._cache = {
      data: [],
      labels: [],
      all: []
    };
  }
  determineDataLimits() {
    const t = this.options, e = this._adapter, s = t.time.unit || "day";
    let { min: n, max: o, minDefined: r, maxDefined: a } = this.getUserBounds();
    function l(c) {
      !r && !isNaN(c.min) && (n = Math.min(n, c.min)), !a && !isNaN(c.max) && (o = Math.max(o, c.max));
    }
    (!r || !a) && (l(this._getLabelBounds()), (t.bounds !== "ticks" || t.ticks.source !== "labels") && l(this.getMinMax(!1))), n = U(n) && !isNaN(n) ? n : +e.startOf(Date.now(), s), o = U(o) && !isNaN(o) ? o : +e.endOf(Date.now(), s) + 1, this.min = Math.min(n, o - 1), this.max = Math.max(n + 1, o);
  }
  _getLabelBounds() {
    const t = this.getLabelTimestamps();
    let e = Number.POSITIVE_INFINITY, s = Number.NEGATIVE_INFINITY;
    return t.length && (e = t[0], s = t[t.length - 1]), {
      min: e,
      max: s
    };
  }
  buildTicks() {
    const t = this.options, e = t.time, s = t.ticks, n = s.source === "labels" ? this.getLabelTimestamps() : this._generate();
    t.bounds === "ticks" && n.length && (this.min = this._userMin || n[0], this.max = this._userMax || n[n.length - 1]);
    const o = this.min, r = this.max, a = ga(n, o, r);
    return this._unit = e.unit || (s.autoSkip ? io(e.minUnit, this.min, this.max, this._getLabelCapacity(o)) : Ed(this, a.length, e.minUnit, this.min, this.max)), this._majorUnit = !s.major.enabled || this._unit === "year" ? void 0 : Fd(this._unit), this.initOffsets(n), t.reverse && a.reverse(), no(this, a, this._majorUnit);
  }
  afterAutoSkip() {
    this.options.offsetAfterAutoskip && this.initOffsets(this.ticks.map((t) => +t.value));
  }
  initOffsets(t = []) {
    let e = 0, s = 0, n, o;
    this.options.offset && t.length && (n = this.getDecimalForValue(t[0]), t.length === 1 ? e = 1 - n : e = (this.getDecimalForValue(t[1]) - n) / 2, o = this.getDecimalForValue(t[t.length - 1]), t.length === 1 ? s = o : s = (o - this.getDecimalForValue(t[t.length - 2])) / 2);
    const r = t.length < 3 ? 0.5 : 0.25;
    e = J(e, 0, r), s = J(s, 0, r), this._offsets = {
      start: e,
      end: s,
      factor: 1 / (e + 1 + s)
    };
  }
  _generate() {
    const t = this._adapter, e = this.min, s = this.max, n = this.options, o = n.time, r = o.unit || io(o.minUnit, e, s, this._getLabelCapacity(e)), a = O(n.ticks.stepSize, 1), l = r === "week" ? o.isoWeekday : !1, c = he(l) || l === !0, h = {};
    let d = e, f, u;
    if (c && (d = +t.startOf(d, "isoWeek", l)), d = +t.startOf(d, c ? "day" : r), t.diff(s, e, r) > 1e5 * a)
      throw new Error(e + " and " + s + " are too far apart with stepSize of " + a + " " + r);
    const g = n.ticks.source === "data" && this.getDataTimestamps();
    for (f = d, u = 0; f < s; f = +t.add(f, a, r), u++)
      so(h, f, g);
    return (f === s || n.bounds === "ticks" || u === 1) && so(h, f, g), Object.keys(h).sort(to).map((p) => +p);
  }
  getLabelForValue(t) {
    const e = this._adapter, s = this.options.time;
    return s.tooltipFormat ? e.format(t, s.tooltipFormat) : e.format(t, s.displayFormats.datetime);
  }
  format(t, e) {
    const n = this.options.time.displayFormats, o = this._unit, r = e || n[o];
    return this._adapter.format(t, r);
  }
  _tickFormatFunction(t, e, s, n) {
    const o = this.options, r = o.ticks.callback;
    if (r)
      return N(r, [
        t,
        e,
        s
      ], this);
    const a = o.time.displayFormats, l = this._unit, c = this._majorUnit, h = l && a[l], d = c && a[c], f = s[e], u = c && d && f && f.major;
    return this._adapter.format(t, n || (u ? d : h));
  }
  generateTickLabels(t) {
    let e, s, n;
    for (e = 0, s = t.length; e < s; ++e)
      n = t[e], n.label = this._tickFormatFunction(n.value, e, t);
  }
  getDecimalForValue(t) {
    return t === null ? NaN : (t - this.min) / (this.max - this.min);
  }
  getPixelForValue(t) {
    const e = this._offsets, s = this.getDecimalForValue(t);
    return this.getPixelForDecimal((e.start + s) * e.factor);
  }
  getValueForPixel(t) {
    const e = this._offsets, s = this.getDecimalForPixel(t) / e.factor - e.end;
    return this.min + s * (this.max - this.min);
  }
  _getLabelSize(t) {
    const e = this.options.ticks, s = this.ctx.measureText(t).width, n = gt(this.isHorizontal() ? e.maxRotation : e.minRotation), o = Math.cos(n), r = Math.sin(n), a = this._resolveTickFontOptions(0).size;
    return {
      w: s * o + a * r,
      h: s * r + a * o
    };
  }
  _getLabelCapacity(t) {
    const e = this.options.time, s = e.displayFormats, n = s[e.unit] || s.millisecond, o = this._tickFormatFunction(t, 0, no(this, [
      t
    ], this._majorUnit), n), r = this._getLabelSize(o), a = Math.floor(this.isHorizontal() ? this.width / r.w : this.height / r.h) - 1;
    return a > 0 ? a : 1;
  }
  getDataTimestamps() {
    let t = this._cache.data || [], e, s;
    if (t.length)
      return t;
    const n = this.getMatchingVisibleMetas();
    if (this._normalized && n.length)
      return this._cache.data = n[0].controller.getAllParsedValues(this);
    for (e = 0, s = n.length; e < s; ++e)
      t = t.concat(n[e].controller.getAllParsedValues(this));
    return this._cache.data = this.normalize(t);
  }
  getLabelTimestamps() {
    const t = this._cache.labels || [];
    let e, s;
    if (t.length)
      return t;
    const n = this.getLabels();
    for (e = 0, s = n.length; e < s; ++e)
      t.push(eo(this, n[e]));
    return this._cache.labels = this._normalized ? t : this.normalize(t);
  }
  normalize(t) {
    return Po(t.sort(to));
  }
}
M(Ve, "id", "time"), M(Ve, "defaults", {
  bounds: "data",
  adapters: {},
  time: {
    parser: !1,
    unit: !1,
    round: !1,
    isoWeekday: !1,
    minUnit: "millisecond",
    displayFormats: {}
  },
  ticks: {
    source: "auto",
    callback: !1,
    major: {
      enabled: !1
    }
  }
});
function ei(i, t, e) {
  let s = 0, n = i.length - 1, o, r, a, l;
  e ? (t >= i[s].pos && t <= i[n].pos && ({ lo: s, hi: n } = Ot(i, "pos", t)), { pos: o, time: a } = i[s], { pos: r, time: l } = i[n]) : (t >= i[s].time && t <= i[n].time && ({ lo: s, hi: n } = Ot(i, "time", t)), { time: o, pos: a } = i[s], { time: r, pos: l } = i[n]);
  const c = r - o;
  return c ? a + (l - a) * (t - o) / c : a;
}
class gs extends Ve {
  constructor(t) {
    super(t), this._table = [], this._minPos = void 0, this._tableRange = void 0;
  }
  initOffsets() {
    const t = this._getTimestampsForTable(), e = this._table = this.buildLookupTable(t);
    this._minPos = ei(e, this.min), this._tableRange = ei(e, this.max) - this._minPos, super.initOffsets(t);
  }
  buildLookupTable(t) {
    const { min: e, max: s } = this, n = [], o = [];
    let r, a, l, c, h;
    for (r = 0, a = t.length; r < a; ++r)
      c = t[r], c >= e && c <= s && n.push(c);
    if (n.length < 2)
      return [
        {
          time: e,
          pos: 0
        },
        {
          time: s,
          pos: 1
        }
      ];
    for (r = 0, a = n.length; r < a; ++r)
      h = n[r + 1], l = n[r - 1], c = n[r], Math.round((h + l) / 2) !== c && o.push({
        time: c,
        pos: r / (a - 1)
      });
    return o;
  }
  _generate() {
    const t = this.min, e = this.max;
    let s = super.getDataTimestamps();
    return (!s.includes(t) || !s.length) && s.splice(0, 0, t), (!s.includes(e) || s.length === 1) && s.push(e), s.sort((n, o) => n - o);
  }
  _getTimestampsForTable() {
    let t = this._cache.all || [];
    if (t.length)
      return t;
    const e = this.getDataTimestamps(), s = this.getLabelTimestamps();
    return e.length && s.length ? t = this.normalize(e.concat(s)) : t = e.length ? e : s, t = this._cache.all = t, t;
  }
  getDecimalForValue(t) {
    return (ei(this._table, t) - this._minPos) / this._tableRange;
  }
  getValueForPixel(t) {
    const e = this._offsets, s = this.getDecimalForPixel(t) / e.factor - e.end;
    return ei(this._table, s * this._tableRange + this._minPos, !0);
  }
}
M(gs, "id", "timeseries"), M(gs, "defaults", Ve.defaults);
var zd = /* @__PURE__ */ Object.freeze({
  __proto__: null,
  CategoryScale: hs,
  LinearScale: ds,
  LogarithmicScale: fs,
  RadialLinearScale: Me,
  TimeScale: Ve,
  TimeSeriesScale: gs
});
const Bd = [
  jl,
  xh,
  gd,
  zd
];
function Vd(i) {
  let t, e, s, n = (
    /*peakHour*/
    i[3] && oo(i)
  );
  return {
    c() {
      n && n.c(), t = tt(), e = E("div"), s = E("canvas"), C(e, "class", "canvas-container svelte-3r4dde");
    },
    m(o, r) {
      n && n.m(o, r), Q(o, t, r), Q(o, e, r), D(e, s), i[4](s);
    },
    p(o, r) {
      /*peakHour*/
      o[3] ? n ? n.p(o, r) : (n = oo(o), n.c(), n.m(t.parentNode, t)) : n && (n.d(1), n = null);
    },
    d(o) {
      o && (Z(t), Z(e)), n && n.d(o), i[4](null);
    }
  };
}
function Wd(i) {
  let t, e;
  return {
    c() {
      t = E("div"), e = ut(
        /*error*/
        i[2]
      ), C(t, "class", "error svelte-3r4dde");
    },
    m(s, n) {
      Q(s, t, n), D(t, e);
    },
    p(s, n) {
      n & /*error*/
      4 && vt(
        e,
        /*error*/
        s[2]
      );
    },
    d(s) {
      s && Z(t);
    }
  };
}
function Nd(i) {
  let t;
  return {
    c() {
      t = E("div"), t.innerHTML = '<div class="spinner-small svelte-3r4dde"></div>', C(t, "class", "loading svelte-3r4dde");
    },
    m(e, s) {
      Q(e, t, s);
    },
    p: et,
    d(e) {
      e && Z(t);
    }
  };
}
function oo(i) {
  let t, e, s;
  return {
    c() {
      t = E("div"), e = ut("Peak Activity: "), s = ut(
        /*peakHour*/
        i[3]
      ), C(t, "class", "info-badge svelte-3r4dde");
    },
    m(n, o) {
      Q(n, t, o), D(t, e), D(t, s);
    },
    p(n, o) {
      o & /*peakHour*/
      8 && vt(
        s,
        /*peakHour*/
        n[3]
      );
    },
    d(n) {
      n && Z(t);
    }
  };
}
function Hd(i) {
  let t;
  function e(o, r) {
    return (
      /*loading*/
      o[1] ? Nd : (
        /*error*/
        o[2] ? Wd : Vd
      )
    );
  }
  let s = e(i), n = s(i);
  return {
    c() {
      t = E("div"), n.c(), C(t, "class", "chart-wrapper svelte-3r4dde");
    },
    m(o, r) {
      Q(o, t, r), n.m(t, null);
    },
    p(o, [r]) {
      s === (s = e(o)) && n ? n.p(o, r) : (n.d(1), n = s(o), n && (n.c(), n.m(t, null)));
    },
    i: et,
    o: et,
    d(o) {
      o && Z(t), n.d();
    }
  };
}
function ro(i) {
  if (i == null) return "-";
  const t = Math.floor(i), e = Math.round((i - t) * 60);
  return `${t.toString().padStart(2, "0")}:${e.toString().padStart(2, "0")}`;
}
function jd(i, t, e) {
  At.register(...Bd);
  let s, n, o = !0, r = null, a = null;
  ps(async () => {
    try {
      const c = await fetch("/api/analytics/time-of-day");
      if (!c.ok) throw new Error("Failed to load data");
      const h = await c.json(), d = h.points, f = h.histogram || [];
      if (e(3, a = ro(h.peak_hour)), e(1, o = !1), await vr(), s) {
        const g = s.getContext("2d").createLinearGradient(0, 0, 0, 400);
        g.addColorStop(0, "rgba(76, 175, 80, 0.4)"), g.addColorStop(1, "rgba(76, 175, 80, 0.05)"), n = new At(
          s,
          {
            type: "scatter",
            // Base type, datasets override
            data: {
              datasets: [
                {
                  type: "line",
                  label: "Activity Density",
                  data: d,
                  borderColor: "#2e7d32",
                  borderWidth: 3,
                  backgroundColor: g,
                  fill: !0,
                  pointRadius: 0,
                  // No dots on line
                  tension: 0.4,
                  // Smooth curve
                  xAxisID: "x"
                },
                {
                  type: "bar",
                  // Underlying histogram
                  label: "Raw Distribution",
                  data: f.map((p) => ({ x: p.x, y: p.y })),
                  backgroundColor: "rgba(2, 136, 209, 0.2)",
                  barPercentage: 1,
                  categoryPercentage: 1,
                  xAxisID: "x"
                }
              ]
            },
            options: {
              responsive: !0,
              maintainAspectRatio: !1,
              resizeDelay: 0,
              plugins: {
                legend: { display: !1 },
                tooltip: {
                  mode: "index",
                  intersect: !1,
                  callbacks: {
                    title: (p) => {
                      const m = p[0].parsed.x;
                      return ro(m);
                    }
                  }
                }
              },
              layout: {
                padding: { top: 12, right: 16, bottom: 10, left: 10 }
              },
              scales: {
                x: {
                  type: "linear",
                  min: 0,
                  max: 24,
                  grid: { display: !1 },
                  ticks: {
                    stepSize: 4,
                    // 0, 4, 8, 12, 16, 20, 24
                    callback: (p) => p === 24 ? "00:00" : p + ":00",
                    font: { size: 10 },
                    color: "#718096",
                    padding: 6
                  }
                },
                y: {
                  display: !1,
                  // Hide Y axis as requested
                  beginAtZero: !0,
                  grace: "12%"
                }
              }
            }
          }
        );
      }
    } catch (c) {
      e(2, r = c.message), e(1, o = !1);
    }
  }), yr(() => {
    n && n.destroy();
  });
  function l(c) {
    Gi[c ? "unshift" : "push"](() => {
      s = c, e(0, s);
    });
  }
  return [s, o, r, a, l];
}
class $d extends Si {
  constructor(t) {
    super(), wi(this, t, jd, Hd, Mi, {});
  }
}
function ao(i, t, e) {
  const s = i.slice();
  return s[3] = t[e], s;
}
function Yd(i) {
  let t, e, s, n, o = Es(
    /*series*/
    i[2]
  ), r = [];
  for (let a = 0; a < o.length; a += 1)
    r[a] = lo(ao(i, o, a));
  return {
    c() {
      t = E("div"), e = E("div"), e.innerHTML = '<div class="col-name svelte-1alnngh">Species</div> <div class="col-graph svelte-1alnngh">Activity Pattern (24h)</div> <div class="col-peak svelte-1alnngh">Peak</div>', s = tt(), n = E("div");
      for (let a = 0; a < r.length; a += 1)
        r[a].c();
      C(e, "class", "table-header svelte-1alnngh"), C(n, "class", "table-body"), C(t, "class", "table-container svelte-1alnngh");
    },
    m(a, l) {
      Q(a, t, l), D(t, e), D(t, s), D(t, n);
      for (let c = 0; c < r.length; c += 1)
        r[c] && r[c].m(n, null);
    },
    p(a, l) {
      if (l & /*formatTime, series, generateSparkline*/
      4) {
        o = Es(
          /*series*/
          a[2]
        );
        let c;
        for (c = 0; c < o.length; c += 1) {
          const h = ao(a, o, c);
          r[c] ? r[c].p(h, l) : (r[c] = lo(h), r[c].c(), r[c].m(n, null));
        }
        for (; c < r.length; c += 1)
          r[c].d(1);
        r.length = o.length;
      }
    },
    d(a) {
      a && Z(t), _r(r, a);
    }
  };
}
function Xd(i) {
  let t;
  return {
    c() {
      t = E("div"), t.textContent = "No data available", C(t, "class", "message svelte-1alnngh");
    },
    m(e, s) {
      Q(e, t, s);
    },
    p: et,
    d(e) {
      e && Z(t);
    }
  };
}
function Ud(i) {
  let t, e;
  return {
    c() {
      t = E("div"), e = ut(
        /*error*/
        i[1]
      ), C(t, "class", "message error svelte-1alnngh");
    },
    m(s, n) {
      Q(s, t, n), D(t, e);
    },
    p(s, n) {
      n & /*error*/
      2 && vt(
        e,
        /*error*/
        s[1]
      );
    },
    d(s) {
      s && Z(t);
    }
  };
}
function Kd(i) {
  let t;
  return {
    c() {
      t = E("div"), t.textContent = "Loading...", C(t, "class", "message svelte-1alnngh");
    },
    m(e, s) {
      Q(e, t, s);
    },
    p: et,
    d(e) {
      e && Z(t);
    }
  };
}
function lo(i) {
  let t, e, s = (
    /*item*/
    i[3].species.replace(/_/g, " ") + ""
  ), n, o, r, a, l, c, h, d, f, u, g, p = co(
    /*item*/
    i[3].peak_hour
  ) + "", m, b;
  return {
    c() {
      t = E("div"), e = E("div"), n = ut(s), r = tt(), a = E("div"), l = Ri("svg"), c = Ri("path"), d = Ri("path"), u = tt(), g = E("div"), m = ut(p), b = tt(), C(e, "class", "col-name svelte-1alnngh"), C(e, "title", o = /*item*/
      i[3].species), C(c, "d", h = ii(
        /*item*/
        i[3].points
      )), C(c, "fill", "none"), C(c, "stroke", "#4caf50"), C(c, "stroke-width", "1.5"), C(c, "vector-effect", "non-scaling-stroke"), C(d, "d", f = `${ii(
        /*item*/
        i[3].points
      )} L 200 30 L 0 30 Z`), C(d, "fill", "#4caf50"), C(d, "fill-opacity", "0.1"), C(d, "stroke", "none"), C(l, "class", "sparkline svelte-1alnngh"), C(l, "viewBox", "0 0 200 30"), C(l, "preserveAspectRatio", "none"), C(l, "width", "100%"), C(l, "height", "30"), C(a, "class", "col-graph svelte-1alnngh"), C(g, "class", "col-peak svelte-1alnngh"), C(t, "class", "table-row svelte-1alnngh");
    },
    m(_, y) {
      Q(_, t, y), D(t, e), D(e, n), D(t, r), D(t, a), D(a, l), D(l, c), D(l, d), D(t, u), D(t, g), D(g, m), D(t, b);
    },
    p(_, y) {
      y & /*series*/
      4 && s !== (s = /*item*/
      _[3].species.replace(/_/g, " ") + "") && vt(n, s), y & /*series*/
      4 && o !== (o = /*item*/
      _[3].species) && C(e, "title", o), y & /*series*/
      4 && h !== (h = ii(
        /*item*/
        _[3].points
      )) && C(c, "d", h), y & /*series*/
      4 && f !== (f = `${ii(
        /*item*/
        _[3].points
      )} L 200 30 L 0 30 Z`) && C(d, "d", f), y & /*series*/
      4 && p !== (p = co(
        /*item*/
        _[3].peak_hour
      ) + "") && vt(m, p);
    },
    d(_) {
      _ && Z(t);
    }
  };
}
function qd(i) {
  let t, e, s;
  function n(a, l) {
    return (
      /*loading*/
      a[0] ? Kd : (
        /*error*/
        a[1] ? Ud : (
          /*series*/
          a[2].length === 0 ? Xd : Yd
        )
      )
    );
  }
  let o = n(i), r = o(i);
  return {
    c() {
      t = E("div"), e = E("h4"), e.textContent = "Activity Patterns by Species", s = tt(), r.c(), C(e, "class", "alt-header svelte-1alnngh"), C(t, "class", "alt-view-container svelte-1alnngh");
    },
    m(a, l) {
      Q(a, t, l), D(t, e), D(t, s), r.m(t, null);
    },
    p(a, [l]) {
      o === (o = n(a)) && r ? r.p(a, l) : (r.d(1), r = o(a), r && (r.c(), r.m(t, null)));
    },
    i: et,
    o: et,
    d(a) {
      a && Z(t), r.d();
    }
  };
}
function co(i) {
  if (i == null) return "-";
  const t = Math.floor(i), e = Math.round((i - t) * 60);
  return `${t.toString().padStart(2, "0")}:${e.toString().padStart(2, "0")}`;
}
function ii(i) {
  if (!i || i.length < 2) return "";
  const t = 200, e = 30, s = 0, n = 24;
  return i.map((o, r) => {
    const a = (o.x - s) / (n - s) * t, l = e - o.y * e;
    return `${r === 0 ? "M" : "L"} ${a.toFixed(1)} ${l.toFixed(1)}`;
  }).join(" ");
}
function Gd(i, t, e) {
  let s = !0, n = null, o = [];
  return ps(async () => {
    try {
      const r = await fetch("/api/analytics/species-activity");
      if (!r.ok) throw new Error("Failed to load data");
      e(2, o = await r.json()), e(0, s = !1);
    } catch (r) {
      e(1, n = r.message), e(0, s = !1);
    }
  }), [s, n, o];
}
class Zd extends Si {
  constructor(t) {
    super(), wi(this, t, Gd, qd, Mi, {});
  }
}
function Jd(i) {
  let t, e, s, n, o, r, a, l, c, h, d;
  return t = new Or({ props: { summary: (
    /*summary*/
    i[0]
  ) } }), a = new $d({}), h = new Zd({}), {
    c() {
      Fi(t.$$.fragment), e = tt(), s = E("div"), n = E("div"), o = E("h3"), o.textContent = "Activity by Time of Day", r = tt(), Fi(a.$$.fragment), l = tt(), c = E("div"), Fi(h.$$.fragment), C(o, "class", "chart-title svelte-fczub0"), C(n, "class", "chart-card chart-card--full svelte-fczub0"), C(c, "class", "chart-card chart-card--full svelte-fczub0"), C(s, "class", "charts-section svelte-fczub0");
    },
    m(f, u) {
      ni(t, f, u), Q(f, e, u), Q(f, s, u), D(s, n), D(n, o), D(n, r), ni(a, n, null), D(s, l), D(s, c), ni(h, c, null), d = !0;
    },
    p(f, u) {
      const g = {};
      u & /*summary*/
      1 && (g.summary = /*summary*/
      f[0]), t.$set(g);
    },
    i(f) {
      d || (le(t.$$.fragment, f), le(a.$$.fragment, f), le(h.$$.fragment, f), d = !0);
    },
    o(f) {
      Se(t.$$.fragment, f), Se(a.$$.fragment, f), Se(h.$$.fragment, f), d = !1;
    },
    d(f) {
      f && (Z(e), Z(s)), oi(t, f), oi(a), oi(h);
    }
  };
}
function Qd(i) {
  let t, e, s;
  return {
    c() {
      t = E("div"), e = E("p"), s = ut(
        /*error*/
        i[2]
      ), C(t, "class", "error-container svelte-fczub0");
    },
    m(n, o) {
      Q(n, t, o), D(t, e), D(e, s);
    },
    p(n, o) {
      o & /*error*/
      4 && vt(
        s,
        /*error*/
        n[2]
      );
    },
    i: et,
    o: et,
    d(n) {
      n && Z(t);
    }
  };
}
function tf(i) {
  let t;
  return {
    c() {
      t = E("div"), t.innerHTML = '<div class="spinner svelte-fczub0"></div> <p>Loading analytics...</p>', C(t, "class", "loading-container svelte-fczub0");
    },
    m(e, s) {
      Q(e, t, s);
    },
    p: et,
    i: et,
    o: et,
    d(e) {
      e && Z(t);
    }
  };
}
function ef(i) {
  let t, e, s, n, o, r;
  const a = [tf, Qd, Jd], l = [];
  function c(h, d) {
    return (
      /*loading*/
      h[1] ? 0 : (
        /*error*/
        h[2] ? 1 : 2
      )
    );
  }
  return n = c(i), o = l[n] = a[n](i), {
    c() {
      t = E("div"), e = E("header"), e.innerHTML = '<h1 class="svelte-fczub0">Analytics</h1> <p class="subtitle svelte-fczub0">All-Time Detection Statistics</p>', s = tt(), o.c(), C(e, "class", "dashboard-header svelte-fczub0"), C(t, "class", "analytics-dashboard svelte-fczub0");
    },
    m(h, d) {
      Q(h, t, d), D(t, e), D(t, s), l[n].m(t, null), r = !0;
    },
    p(h, [d]) {
      let f = n;
      n = c(h), n === f ? l[n].p(h, d) : (wr(), Se(l[f], 1, 1, () => {
        l[f] = null;
      }), Sr(), o = l[n], o ? o.p(h, d) : (o = l[n] = a[n](h), o.c()), le(o, 1), o.m(t, null));
    },
    i(h) {
      r || (le(o), r = !0);
    },
    o(h) {
      Se(o), r = !1;
    },
    d(h) {
      h && Z(t), l[n].d();
    }
  };
}
function sf(i, t, e) {
  let s = null, n = !0, o = null;
  return ps(async () => {
    try {
      const r = await fetch("/api/analytics/summary");
      if (!r.ok) throw new Error("Failed to load summary");
      e(0, s = await r.json());
    } catch (r) {
      e(2, o = r.message);
    } finally {
      e(1, n = !1);
    }
  }), [s, n, o];
}
class nf extends Si {
  constructor(t) {
    super(), wi(this, t, sf, ef, Mi, {});
  }
}
const ho = document.querySelector("[data-analytics-dashboard]");
ho && new nf({
  target: ho
});
