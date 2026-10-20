"use strict";
// Pipeline viewer front end. Data: /api/* (viz.server); one trace per photo (viz.traces).

const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const css = (v) => getComputedStyle(document.body).getPropertyValue(v).trim();
const pct = (p) => (p == null ? "–" : p >= 0.995 ? "100%" : p < 0.005 && p > 0 ? "<1%" : `${Math.round(p * 100)}%`);

const S = { sets: [], set: null, index: [], featured: [], filter: "all", q: "", trace: null, stage: "decide", lam: "0", map: null, layers: {}, marks: {}, hidden: new Set(),
  mode: "triage", errors: {}, tier: "all", bench: "all" };
const TRIAGE_SETS = ["full", "dev3k"];  // sets with error triage (viz.errors)
const TIER_HELP = {  // tier names and one-line definitions (viz/errors.py)
  1: ["Policy close call", "Another tool choice (the other λ, or a fixed tool set) already answers within 25 km"],
  2: ["Coin flip", "A right area is on the table with at least half the chosen area's answer-head probability"],
  3: ["Near miss", "Answer 25–50 km off: right region, wrong point"],
  4: ["Right area ranked low", "A right area is on the table, with under half the chosen area's probability"],
  5: ["In pool, not an area", "A retrieval candidate lies within 25 km, but no area answers there"],
  6: ["Retrieval miss", "No candidate within 25 km anywhere"],
};

// ---------- formatting ----------
function fmtKm(km) {
  if (km == null || !isFinite(km)) return "–";
  if (km < 1) return `${Math.round(km * 1000)} m`;
  if (km < 10) return `${km.toFixed(1)} km`;
  return `${Math.round(km).toLocaleString()} km`;
}
function kmBadge(km, label = "") {
  if (km == null) return label ? `<span class="km"><span class="lab">${esc(label)}</span> –</span>` : "";
  const [cls, ic, t] = km < 25 ? ["good", "✓", "within 25 km"] : km < 200 ? ["warn", "~", "25–200 km off"] : ["bad", "✗", "over 200 km off"];
  return `<span class="km ${cls}" title="${t}">${label ? `<span class="lab">${esc(label)}</span>` : ""}<i>${ic}</i>${fmtKm(km)}</span>`;
}
const imgUrl = (im, s = 0) => (im.kind === "bench" ? `/img/bench/${im.bench}/${encodeURIComponent(im.file)}` : im.kind === "mp16" ? `/img/mp16/${encodeURIComponent(im.id)}` : `/img/upload/${im.id}`) + (s ? `?s=${s}` : "");
const candImg = (c, s = 256) => (c.mp16 ? `/img/mp16/${encodeURIComponent(c.mp16)}?s=${s}` : "");

function areaNames(areas) {
  let c = 0;
  return areas.map((a) => (a.source === "pinpoint" ? `A${a.rank + 1}` : `C${++c}`));
}
const areaColor = (a) => (a.source === "pinpoint" ? css("--series-1") : css("--series-3"));

// ---------- tooltip ----------
const tip = $("#tip");
document.addEventListener("mouseover", (e) => {
  const el = e.target.closest("[data-tip]");
  if (!el) return (tip.hidden = true);
  tip.innerHTML = el.dataset.tip;
  tip.hidden = false;
});
document.addEventListener("mousemove", (e) => {
  if (tip.hidden) return;
  const x = Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 8);
  tip.style.left = `${x}px`;
  tip.style.top = `${e.clientY + 16}px`;
});

// ---------- browser ----------
async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

async function init() {
  const saved = localStorage.getItem("viz-theme");
  if (saved) document.documentElement.dataset.theme = saved;
  [S.sets, S.featured] = await Promise.all([api("/api/sets"), api("/api/featured")]);
  $("#set-tabs").onclick = (e) => { const b = e.target.closest(".tab"); if (b) selectSet(b.dataset.set); };
  $("#mode").onclick = (e) => { const b = e.target.closest("[data-m]"); if (b) setMode(b.dataset.m); };
  $("#bench-chips").onclick = (e) => {
    const b = e.target.closest(".chip");
    if (!b) return;
    if (b.dataset.reg) S.regress = !S.regress;
    else S.bench = b.dataset.b;
    renderList();
  };
  $("#tierbar").onclick = $("#tierlist").onclick = (e) => {
    const b = e.target.closest("[data-tier]");
    if (b) { S.tier = S.tier === b.dataset.tier ? "all" : b.dataset.tier; renderList(); }
  };
  $("#filters").onclick = (e) => {
    const b = e.target.closest(".chip");
    if (!b) return;
    S.filter = b.dataset.f;
    document.querySelectorAll("#filters .chip").forEach((c) => c.classList.toggle("on", c === b));
    renderList();
  };
  $("#search").oninput = (e) => { S.q = e.target.value.toLowerCase(); renderList(); };
  $("#theme-btn").onclick = toggleTheme;
  setupUpload();
  const [set, i] = location.hash.slice(1).split("/");
  if (set && !TRIAGE_SETS.includes(set)) S.mode = "browse";
  await Promise.all(TRIAGE_SETS.filter((n) => S.sets.some((s) => s.name === n)).map(async (n) => {  // error counts for the tabs
    S.errors[n] = await api(`/api/errors/${n}`).catch(() => null);
    if (S.errors[n]) S.errors[n].byI = Object.fromEntries(S.errors[n].errors.map((r) => [String(r.i), r]));
  }));
  renderTabs();
  if (S.mode === "triage") {
    await selectSet(set || "full");
    const first = filteredErrors()[0];
    if (set && i != null) openTrace(set, i);
    else if (first) openTrace(S.set, first.i);
    return;
  }
  await selectSet(S.featured.length && !set ? "featured" : set || "full");
  if (set && i != null) openTrace(set, i);
  else if (S.featured.length) openTrace(S.featured[0].set, S.featured[0].i);
}

function renderTabs() {
  const all = [{ name: "featured", title: "Featured", n: S.featured.length }, ...S.sets];
  const tabs = S.mode === "triage" ? S.sets.filter((t) => TRIAGE_SETS.includes(t.name)) : all;
  const label = { featured: "Featured", full: "Test", dev3k: "Dev", upload: "Your photos" };
  $("#set-tabs").innerHTML = tabs
    .map((t) => `<button class="tab" data-set="${t.name}" title="${esc(t.title)}">${esc(label[t.name] || t.name)}<span class="n">${S.mode === "triage" && S.errors[t.name] ? `${S.errors[t.name].n_errors} errors` : t.n}</span></button>`)
    .join("");
  document.querySelectorAll("#set-tabs .tab").forEach((t) => t.classList.toggle("on", t.dataset.set === S.set));
  document.querySelectorAll("#mode button").forEach((b) => b.classList.toggle("on", b.dataset.m === S.mode));
  $("#triage").hidden = S.mode !== "triage";
  $("#filters").hidden = S.mode === "triage";
}

async function setMode(m) {
  S.mode = m;
  if (m === "triage" && !TRIAGE_SETS.includes(S.set)) S.set = "full";
  renderTabs();
  await selectSet(S.set || "full");
  if (S.trace) renderHero();
}

async function selectSet(name) {
  S.set = name;
  document.querySelectorAll("#set-tabs .tab").forEach((t) => t.classList.toggle("on", t.dataset.set === name));
  if (TRIAGE_SETS.includes(name) && !S.errors[name]) {
    S.errors[name] = await api(`/api/errors/${name}`).catch(() => null);
    if (S.errors[name]) S.errors[name].byI = Object.fromEntries(S.errors[name].errors.map((r) => [String(r.i), r]));
    renderTabs();
  }
  if (S.mode === "triage") return renderList();
  if (name === "featured") {
    const idx = {};
    for (const f of S.featured) idx[f.set] ??= await api(`/api/index/${f.set}`).catch(() => []);
    S.index = S.featured.map((f) => ({ ...(idx[f.set].find((r) => String(r.i) === String(f.i)) || { i: f.i }), set: f.set, note: f.note }));
  } else {
    S.index = (await api(`/api/index/${name}`)).map((r) => ({ ...r, set: name }));
  }
  renderList();
}

function passes(r) {
  const q = S.q;
  if (q && !(`${r.image_id} ${r.truth_place || ""} ${r.note || ""}`.toLowerCase().includes(q))) return false;
  const rel = r.released_km != null && r.released_km < 25, fin = r.final_km != null && r.final_km < 25;
  return { all: true, fixed: !rel && fin, broke: rel && !fin, right: rel && fin, wrong: r.released_km != null && !rel && !fin }[S.filter];
}

// ---------- error triage ----------
const tierColor = (k) => `var(--tier-${k})`;
function filteredErrors(ignoreTier = false) {
  const E = S.errors[S.set];
  if (!E) return [];
  const q = S.q;
  return E.errors.filter((r) => (S.bench === "all" || r.benchmark === S.bench) && (ignoreTier || S.tier === "all" || String(r.tier) === S.tier) &&
    (!S.regress || r.regression) &&
    (!q || `${r.image_id} ${r.truth_place} ${r.final_place} ${r.why}`.toLowerCase().includes(q)));
}
function renderTriage() {
  const E = S.errors[S.set];
  if (!E) {
    $("#tierbar").innerHTML = "";
    $("#tierlist").innerHTML = `<div class="muted small" style="padding:6px">No error triage for this set.</div>`;
    return [];
  }
  const benches = [...new Set(E.errors.map((r) => r.benchmark))].sort();
  const nReg = E.errors.filter((r) => r.regression && (S.bench === "all" || r.benchmark === S.bench)).length;
  $("#bench-chips").innerHTML = ["all", ...benches].map((b) => `<button data-b="${b}" class="chip ${S.bench === b ? "on" : ""}">${b === "all" ? "All benchmarks" : esc(b)}</button>`).join("") +
    `<button data-reg="1" class="chip ${S.regress ? "on" : ""}" data-tip="Errors where the released Pinpoint's or the retrained reranker's top-1 was within 25 km">Regressions only · ${nReg}</button>`;
  const base = filteredErrors(true);
  const nPhotos = S.bench === "all" ? E.n_photos : null;
  const counts = Object.keys(TIER_HELP).map((k) => [k, base.filter((r) => String(r.tier) === k).length]);
  $("#tierbar").innerHTML = counts.map(([k, n]) => (n ? `<div data-tier="${k}" class="${S.tier !== "all" && S.tier !== k ? "dim" : ""}" style="flex:${n};background:${tierColor(k)}" data-tip="<b>${esc(TIER_HELP[k][0])}</b>: ${n.toLocaleString()} errors<br>${esc(TIER_HELP[k][1])}"></div>` : "")).join("");
  $("#tierlist").innerHTML = counts.map(([k, n]) => `<button data-tier="${k}" class="${S.tier === k ? "on" : ""}" data-tip="${esc(TIER_HELP[k][1])}">
      <span class="tier-dot" style="background:${tierColor(k)}"></span><span>${k}. ${esc(TIER_HELP[k][0])}</span>
      <span class="n">${n.toLocaleString()}${nPhotos ? ` · ${(100 * n / nPhotos).toFixed(1)}%` : ""}</span></button>`).join("");
  return filteredErrors();
}
function renderTriageList() {
  const rows = renderTriage();
  const E = S.errors[S.set];
  $("#count").textContent = E ? `${rows.length.toLocaleString()} error${rows.length === 1 ? "" : "s"}, most fixable first${S.bench === "all" && S.tier === "all" && !S.q ? ` · ${(100 * (1 - E.n_errors / E.n_photos)).toFixed(1)}% of ${E.n_photos.toLocaleString()} photos right` : ""}` : "";
  const shown = rows.slice(0, 400);
  $("#list").innerHTML =
    shown.map((r) => {
      const thumb = S.set === "dev3k" ? `/img/mp16/${encodeURIComponent(r.image_id)}?s=120` : `/img/bench/${r.benchmark}/${encodeURIComponent(r.image_id)}?s=120`;
      return `<div class="item" data-set="${S.set}" data-i="${r.i}"><img loading="lazy" src="${thumb}" alt="">
        <div><div class="t" title="${esc(r.truth_place)}">${esc(r.truth_place || r.image_id)}</div>
        <div class="b"><span class="tier-badge"><span class="tier-dot" style="background:${tierColor(r.tier)}"></span>${esc(TIER_HELP[r.tier][0])}</span>${kmBadge(r.final_km, "Agent")}</div>
        <div class="why">${esc(r.why)}</div></div></div>`;
    }).join("") + (rows.length > shown.length ? `<div class="muted small" style="padding:8px">Showing the 400 most fixable — pick a tier or search to narrow.</div>` : "");
  $("#list").onclick = (e) => { const it = e.target.closest(".item"); if (it) openTrace(it.dataset.set, it.dataset.i); };
  markActive();
}

function renderList() {
  if (S.mode === "triage") return renderTriageList();
  const rows = S.index.filter(passes);
  $("#count").textContent = `${rows.length.toLocaleString()} photo${rows.length === 1 ? "" : "s"}`;
  const shown = rows.slice(0, 400);
  $("#list").innerHTML =
    shown
      .map((r) => {
        const thumb = r.benchmark === "upload" ? `/img/upload/${r.i}?s=120` : r.set === "dev3k" ? `/img/mp16/${encodeURIComponent(r.image_id)}?s=120` : `/img/bench/${r.benchmark}/${encodeURIComponent(r.image_id)}?s=120`;
        const title = r.note || r.truth_place || r.final_place || r.image_id;
        return `<div class="item" data-set="${r.set}" data-i="${r.i}"><img loading="lazy" src="${thumb}" alt="">
        <div><div class="t" title="${esc(title)}">${esc(title)}</div>
        <div class="b">${kmBadge(r.released_km, "Pinpoint")}${kmBadge(r.final_km, "Agent")}</div></div></div>`;
      })
      .join("") + (rows.length > shown.length ? `<div class="muted small" style="padding:8px">Showing the first 400 — search to narrow.</div>` : "");
  $("#list").onclick = (e) => { const it = e.target.closest(".item"); if (it) openTrace(it.dataset.set, it.dataset.i); };
  markActive();
}
function markActive() {
  document.querySelectorAll("#list .item").forEach((el) => el.classList.toggle("on", S.trace && el.dataset.set === S.trace.set && String(el.dataset.i) === String(S.trace.i)));
}

// ---------- trace ----------
async function openTrace(set, i) {
  const t = await api(`/api/trace/${set}/${i}`);
  if (TRIAGE_SETS.includes(set) && !S.errors[set]) {
    S.errors[set] = await api(`/api/errors/${set}`).catch(() => null);
    if (S.errors[set]) S.errors[set].byI = Object.fromEntries(S.errors[set].errors.map((r) => [String(r.i), r]));
  }
  t.err = S.errors[set]?.byI?.[String(i)] || null;
  S.trace = t;
  history.replaceState(null, "", `#${set}/${i}`);
  $("#empty").hidden = true;
  $("#view").hidden = false;
  t.names = areaNames(t.areas);
  if (!t.agent[S.lam]) S.lam = Object.keys(t.agent)[0];
  if (S.mode === "triage" && t.err) S.stage = { 1: "decide", 2: "decide", 3: "zoom", 4: "decide", 5: "retrieve", 6: "retrieve" }[t.err.tier];  // where the miss shows
  renderHero();
  renderMapBase();
  renderStages();
  markActive();
}

function renderHero() {
  const t = S.trace;
  $("#photo").src = imgUrl(t.image, 1200);
  $("#photo-id").textContent = `${{ full: "test set", dev3k: "dev set", upload: "uploaded" }[t.set] || t.set} · ${t.benchmark} · ${t.image_id}`;
  $("#truth-place").textContent = t.truth ? t.truth.place || "Unknown place" : "Location unknown";
  $("#truth-coord").textContent = t.truth ? `True location ${t.truth.lat.toFixed(4)}, ${t.truth.lon.toFixed(4)}` : "No true location given — distances are not shown.";
  const note = $("#triage-note"), er = t.err;
  note.hidden = !er;
  if (er) {
    note.style.borderLeftColor = tierColor(er.tier);
    const pos = S.errors[t.set]?.errors.indexOf(er);
    note.innerHTML = `<div class="h"><span class="tier-dot" style="background:${tierColor(er.tier)}"></span>Error tier ${er.tier}: ${esc(TIER_HELP[er.tier][0])}
        <span class="muted small" style="font-weight:400">#${(pos ?? 0) + 1} of ${S.errors[t.set]?.n_errors ?? "?"} by fixability · λ 0 answer</span></div>
      <div>${esc(er.why)}</div>${er.flags.length ? `<ul>${er.flags.map((f) => `<li>${esc(f)}</li>`).join("")}</ul>` : ""}
      <div class="muted small">${esc(TIER_HELP[er.tier][1])}.</div>`;
  }
  const fin = t.agent[S.lam].final;
  const rel = t.released.candidates[0] || {}, rt = t.retrained.candidates[0] || {};
  const tile = (k, sw, km, place, cls = "") =>
    `<div class="tile ${cls}"><div class="k">${sw}${k}</div><div class="v">${t.truth ? fmtKmTile(km) : "–"}</div><div class="p" title="${esc(place)}">${esc(place || "")}</div></div>`;
  $("#tiles").innerHTML =
    tile("Released Pinpoint top-1", `<span class="swatch" style="background:var(--series-2)"></span>`, rel.km, t.released.top1_place) +
    tile("Retrained reranker top-1", `<span class="swatch" style="background:var(--series-1)"></span>`, rt.km, t.retrained.top1_place) +
    tile(`Agent answer (λ ${S.lam})`, `<span class="mk-answer" style="width:9px;height:9px;border-width:1px;display:inline-block"></span>`, fin.km, fin.place, "final");
  const steps = t.agent[S.lam].steps;
  const acts = steps.slice(0, -1).map((s) => (s.action === "zoom" ? `zoom ${t.names[s.slot]}` : s.action));
  $("#trail").innerHTML =
    `<span class="muted">Agent:</span> <span class="act">glance</span>` +
    acts.map((a) => `<span class="arrow">→</span><span class="act">${esc(a)}</span>`).join("") +
    `<span class="arrow">→</span><span class="act"><b>answer ${t.names[fin.slot]}</b>${fin.zoomed ? " (zoomed point)" : ""}</span>`;
}
function fmtKmTile(km) {
  if (km == null) return "–";
  const ic = km < 25 ? `<span style="color:var(--good)">✓</span>` : km < 200 ? `<span style="color:var(--warning)">~</span>` : `<span style="color:var(--critical)">✗</span>`;
  const [n, u] = fmtKm(km).split(" ");
  return `${ic} ${n}<small> ${u} off</small>`;
}

// ---------- map ----------
function ensureMap() {
  if (S.map) return;
  S.map = L.map("map", { worldCopyJump: true, zoomControl: true, attributionControl: true }).setView([20, 0], 2);
  S.tiles = L.tileLayer(tileUrl(), { attribution: "Tiles &copy; Esri &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap contributors", maxZoom: 16 }).addTo(S.map);
}
function tileUrl() {  // Esri's keyless light / dark gray canvas
  const dark = document.documentElement.dataset.theme === "dark" || (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
  return `https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_${dark ? "Dark" : "Light"}_Gray_Base/MapServer/tile/{z}/{y}/{x}`;
}
function applyTiles() {
  if (S.tiles) S.tiles.setUrl(tileUrl());
}

function dot(lat, lon, color, r, tipHtml, opts = {}) {
  return L.circleMarker([lat, lon], { radius: r, color: css("--surface-1"), weight: 2, fillColor: color, fillOpacity: opts.fo ?? 0.9, ...opts }).bindTooltip(tipHtml, { direction: "top" });
}
function candTip(c, extra = "") {
  return `<b>${esc(c.place || "")}</b><br>${c.lat.toFixed(3)}, ${c.lon.toFixed(3)}${c.km != null ? `<br>${fmtKm(c.km)} from truth` : ""}${extra}`;
}

// layers per stage: which groups are shown
const STAGE_LAYERS = {
  retrieve: ["released", "pool"],
  areas: ["areas", "pool"],
  read: ["clues", "areas"],
  recognise: ["areas"],
  zoom: ["zoomc", "areas"],
  decide: ["areas", "answers"],
};
const LAYER_INFO = {
  released: ["Released Pinpoint (12)", () => css("--series-2"), "dot"],
  pool: ["Retrained pool (size = set prob.)", () => css("--series-1"), "dot"],
  areas: ["Areas (25 km)", () => css("--series-1"), "ring"],
  clues: ["Clue places (text search)", () => css("--series-3"), "dot"],
  zoomc: ["Zoomed-search candidates", () => css("--series-1"), "dot"],
  answers: ["Each area's answer point", () => css("--text-secondary"), "dot"],
};

function renderMapBase() {
  ensureMap();
  const t = S.trace;
  for (const g of Object.values(S.layers)) g.remove();
  S.layers = {};
  S.marks = {};
  const add = (g, key, m) => { (S.layers[g] ??= L.layerGroup()).addLayer(m); if (key) S.marks[key] = m; };

  t.released.candidates.forEach((c, k) => add("released", `rel:${k}`, dot(c.lat, c.lon, css("--series-2"), 6, candTip(c, `<br>released rank ${k + 1}`))));
  const pmax = Math.max(...t.retrained.candidates.map((c) => c.p || 0), 1e-6);
  t.retrained.candidates.forEach((c, k) =>
    add("pool", `pool:${k}`, dot(c.lat, c.lon, css("--series-1"), 3 + 9 * Math.sqrt((c.p || 0) / pmax), candTip(c, `<br>pool rank ${k + 1} · set prob. ${pct(c.p)}`), { fo: 0.75 })));
  t.areas.forEach((a, s) => {
    const col = areaColor(a);
    const ring = L.circle([a.center[0], a.center[1]], { radius: 25000, color: col, weight: 2, fillColor: col, fillOpacity: 0.08 })
      .bindTooltip(`<b>${t.names[s]}</b> · ${esc(a.place || "")}<br>${a.source === "pinpoint" ? "Pinpoint agreement area" : "clue place"}`, { direction: "top" });
    add("areas", `area:${s}`, ring);
    add("areas", null, L.marker([a.center[0], a.center[1]], { icon: L.divIcon({ className: "", html: `<div style="font-weight:700;font-size:12px;color:${col};text-shadow:0 0 3px var(--surface-1),0 0 3px var(--surface-1);transform:translate(10px,-18px)">${t.names[s]}</div>` }), interactive: false }));
    a.candidates.forEach((c, k) => add("zoomc", `zoom:${s}:${k}`, dot(c.lat, c.lon, col, k === a.zoomed.cand_index ? 8 : 4.5, candTip(c, `<br>${t.names[s]} zoom rank ${k + 1}${k === a.zoomed.cand_index ? " · <b>zoom pick</b>" : ""}`), { fo: 0.8 })));
    if (a.unzoomed.lat != null) add("answers", `ans:${s}:u`, dot(a.unzoomed.lat, a.unzoomed.lon, css("--text-secondary"), 5, `<b>${t.names[s]}</b> answer without zoom<br>${fmtKm(a.unzoomed.km)} from truth`));
    if (a.zoomed.lat != null) add("answers", `ans:${s}:z`, dot(a.zoomed.lat, a.zoomed.lon, css("--text-primary"), 5, `<b>${t.names[s]}</b> answer after zoom<br>${fmtKm(a.zoomed.km)} from truth`));
  });
  (t.clues.evidence || []).forEach((e, k) => {
    if (e.lat == null) return;
    add("clues", `clue:${k}`, dot(e.lat, e.lon, css("--series-3"), e.used ? 7 : 4.5,
      `<b>“${esc(e.query)}”</b> (${esc(e.source)})<br>${esc((e.region || []).filter(Boolean).reverse().join(", "))}<br>share ${pct(e.share)}${e.p_yes != null ? ` · judge P(yes) ${pct(e.p_yes)}` : ""}`, { fo: e.used ? 0.9 : 0.5 }));
  });
  // always on: truth and the agent's answer
  S.fixedLayer?.remove();
  S.fixedLayer = L.layerGroup().addTo(S.map);
  const fin = t.agent[S.lam].final;
  if (t.truth) {
    S.fixedLayer.addLayer(L.marker([t.truth.lat, t.truth.lon], { icon: L.divIcon({ className: "", html: `<div class="mk-truth">★</div>`, iconSize: [22, 22], iconAnchor: [11, 11] }), zIndexOffset: 1000 })
      .bindTooltip(`<b>True location</b><br>${esc(t.truth.place)}`, { direction: "top" }));
    if (fin.lat != null) S.fixedLayer.addLayer(L.polyline([[t.truth.lat, t.truth.lon], [fin.lat, fin.lon]], { color: css("--text-primary"), weight: 1.5, dashArray: "4 4", opacity: 0.7 }));
  }
  const er = t.err;  // error triage: ring the right area, or mark the right pool candidate
  if (er && er.right_slot != null && t.areas[er.right_slot]) {
    const a = t.areas[er.right_slot];
    S.fixedLayer.addLayer(L.circleMarker([a.center[0], a.center[1]], { radius: 18, color: css("--good"), weight: 3, dashArray: "5 4", fill: false })
      .bindTooltip(`<b>${t.names[er.right_slot]}</b> was right (${esc(a.place || "")})<br>answer head ${pct(er.p_right)} vs ${pct(er.p_chosen)} for ${esc(er.chosen)}`, { direction: "top" }));
  } else if (er && er.pool_right_rank) {
    const c = t.retrained.candidates[er.pool_right_rank - 1];
    S.fixedLayer.addLayer(L.circleMarker([c.lat, c.lon], { radius: 14, color: css("--good"), weight: 3, dashArray: "5 4", fill: false })
      .bindTooltip(`<b>Right pool candidate</b> #${er.pool_right_rank}<br>${esc(c.place || "")} · ${fmtKm(c.km)} from truth · set prob. ${pct(c.p)}`, { direction: "top" }));
  }
  if (fin.lat != null)
    S.fixedLayer.addLayer(L.marker([fin.lat, fin.lon], { icon: L.divIcon({ className: "", html: `<div class="mk-answer"></div>`, iconSize: [18, 18], iconAnchor: [9, 9] }), zIndexOffset: 900 })
      .bindTooltip(`<b>Agent answer</b> (${t.names[fin.slot]})<br>${esc(fin.place || "")}${fin.km != null ? `<br>${fmtKm(fin.km)} from truth` : ""}`, { direction: "top" }));
}

function showLayers(stage) {
  const want = STAGE_LAYERS[stage];
  for (const [g, layer] of Object.entries(S.layers)) {
    const on = want.includes(g) && !S.hidden.has(g);
    if (on && !S.map.hasLayer(layer)) layer.addTo(S.map);
    if (!on && S.map.hasLayer(layer)) layer.remove();
  }
  const t = S.trace;
  $("#legend").innerHTML =
    (t.truth ? `<span><span class="mk-truth" style="font-size:15px">★</span> True location</span>` : "") +
    `<span><span class="mk-answer" style="width:10px;height:10px;border-width:1px"></span> Agent answer</span>` +
    want.filter((g) => S.layers[g]).map((g) => {
      const [label, col, kind] = LAYER_INFO[g];
      const sw = kind === "ring" ? `<span class="swatch" style="border:2px solid ${col()};background:transparent"></span>` : `<span class="swatch" style="background:${col()}"></span>`;
      return `<button data-layer="${g}" class="${S.hidden.has(g) ? "off" : ""}" title="Show / hide">${sw}${label}</button>`;
    }).join("");
  $("#legend").onclick = (e) => {
    const b = e.target.closest("[data-layer]");
    if (!b) return;
    const g = b.dataset.layer;
    S.hidden.has(g) ? S.hidden.delete(g) : S.hidden.add(g);
    showLayers(S.stage);
  };
  fitStage(stage);
}

function fitStage(stage) {
  const t = S.trace, pts = [];
  if (t.truth) pts.push([t.truth.lat, t.truth.lon]);
  const fin = t.agent[S.lam].final;
  if (fin.lat != null) pts.push([fin.lat, fin.lon]);
  if (t.err?.right_slot != null && t.areas[t.err.right_slot]) pts.push(t.areas[t.err.right_slot].center);
  else if (t.err?.pool_right_rank) { const c = t.retrained.candidates[t.err.pool_right_rank - 1]; pts.push([c.lat, c.lon]); }
  if (stage === "retrieve") { t.released.candidates.slice(0, 3).forEach((c) => pts.push([c.lat, c.lon])); t.retrained.candidates.slice(0, 5).forEach((c) => pts.push([c.lat, c.lon])); }
  else if (stage === "read") { (t.clues.evidence || []).filter((e) => e.lat != null && e.used).forEach((e) => pts.push([e.lat, e.lon])); t.areas.forEach((a) => pts.push(a.center)); }
  else if (stage === "zoom" && S.zoomFocus != null && t.areas[S.zoomFocus]) { t.areas[S.zoomFocus].candidates.forEach((c) => pts.push([c.lat, c.lon])); }
  else t.areas.forEach((a) => pts.push(a.center));
  if (!pts.length) return;
  S.map.invalidateSize();
  if (pts.length === 1) S.map.setView(pts[0], 9);
  else S.map.fitBounds(L.latLngBounds(pts).pad(0.25), { maxZoom: 12 });
}

function hlMark(key, on) {
  const m = S.marks[key];
  if (!m) return;
  if (on) { m.openTooltip(); m.setStyle?.({ weight: 3, color: css("--text-primary") }); m.bringToFront?.(); }
  else { m.closeTooltip(); m.setStyle?.({ weight: 2, color: m instanceof L.Circle ? m.options.fillColor : css("--surface-1") }); }
}
document.addEventListener("mouseover", (e) => { const el = e.target.closest("[data-mark]"); if (el) el.dataset.mark.split(" ").forEach((k) => hlMark(k, true)); });
document.addEventListener("mouseout", (e) => { const el = e.target.closest("[data-mark]"); if (el) el.dataset.mark.split(" ").forEach((k) => hlMark(k, false)); });

// ---------- stages ----------
const STAGES = [
  ["retrieve", "Retrieve"],
  ["areas", "Areas"],
  ["read", "Read"],
  ["recognise", "Recognise"],
  ["zoom", "Zoom"],
  ["decide", "Decide"],
];
function renderStages() {
  $("#stages").innerHTML = STAGES.map(([k, l], j) => `<button class="stage ${k === S.stage ? "on" : ""}" data-s="${k}"><b>${j + 1}</b>${l}</button>`).join("");
  $("#stages").onclick = (e) => { const b = e.target.closest(".stage"); if (b) { S.stage = b.dataset.s; renderStages(); } };
  const fn = { retrieve: pRetrieve, areas: pAreas, read: pRead, recognise: pRecognise, zoom: pZoom, decide: pDecide }[S.stage];
  $("#panel").innerHTML = fn(S.trace);
  bindPanel();
  showLayers(S.stage);
}

function candCard(c, key, rankText, valueHtml, pick = false) {
  const img = candImg(c);
  return `<div class="cand ${pick ? "pick" : ""}" data-mark="${key}" data-tip="${esc(`<b>${esc(c.place || "?")}</b><br>MP16 ${esc(c.mp16 || "?")}<br>photographer ${esc(c.author || "?")}`)}">
    ${img ? `<img loading="lazy" src="${img}" alt="">` : `<div style="aspect-ratio:1;background:var(--surface-2)"></div>`}
    <div class="meta"><div class="place">${esc(c.place || "–")}</div>
    <div class="line"><span class="rank">${rankText}</span>${kmBadge(c.km)}</div>${valueHtml || ""}</div></div>`;
}
const bar = (p, cls = "") => `<div class="bar ${cls}"><div style="width:${Math.max(0, Math.min(1, p || 0)) * 100}%"></div></div>`;

function pRetrieve(t) {
  const rel = t.released.candidates, pool = t.retrained.candidates;
  const best = (cs) => cs.reduce((m, c) => (c.km != null && c.km < m ? c.km : m), Infinity);
  const n = S.retrieveAll ? pool.length : Math.min(24, pool.length);
  return `<p class="lead">SigLIP2 embeds the photo and Pinpoint's attention reranker scores visually similar geotagged MP16 photos. The <b>released</b> model returns 12 candidates;
    the <b>retrained</b> reranker (fine-tuned on bucket 99 with wide pools, no photographer exclusion) returns a wider pool whose set layer turns scores into probabilities.
    Hover a photo to find it on the map.</p>
    <h3><span class="swatch" style="background:var(--series-2)"></span> Released Pinpoint · 12 candidates ${t.truth ? `<span class="muted small">closest ${fmtKm(best(rel))}</span>` : ""}</h3>
    <div class="strip">${rel.map((c, k) => candCard(c, `rel:${k}`, `#${k + 1}`, `<div class="line"><span class="muted">score</span><span class="rank">${c.score?.toFixed(3) ?? "–"}</span></div>`)).join("")}</div>
    <h3><span class="swatch" style="background:var(--series-1)"></span> Retrained reranker pool · ${pool.length} candidates ${t.truth ? `<span class="muted small">closest ${fmtKm(best(pool))}</span>` : ""}</h3>
    <div class="strip">${pool.slice(0, n).map((c, k) => candCard(c, `pool:${k}`, `#${k + 1}`, `<div class="line"><span class="muted">set prob.</span><span class="rank">${pct(c.p)}</span></div>${bar(c.p / (pool[0].p || 1))}`)).join("")}</div>
    ${pool.length > 24 ? `<p><button class="btn ghost" data-act="pool-all">${S.retrieveAll ? "Show top 24" : `Show all ${pool.length}`}</button></p>` : ""}`;
}

function pAreas(t) {
  const fin = t.agent[S.lam].final;
  return `<p class="lead">Candidates within 25 km of each other are grouped into <b>agreement areas</b> (A1 = most support); clue places the judge trusts become extra
    hypotheses (C·). Each area carries pre-zoom evidence the agent sees for free. An area answers with the best pool candidate inside it, or with its zoomed pick once zoomed.</p>
  <table class="t"><thead><tr><th>Area</th><th>Place</th><th>Source</th><th class="num">Pool support</th><th class="num">Gallery photos in box</th>
  <th class="num">km to top-1</th><th class="num">Set-layer mass</th><th>Answer (no zoom)</th><th>Answer (zoomed)</th></tr></thead><tbody>
  ${t.areas.map((a, s) => `<tr class="${s === fin.slot ? "sel" : ""}" data-mark="area:${s} ans:${s}:u ans:${s}:z">
    <td><span class="area-tag"><span class="swatch" style="background:${areaColor(a)}"></span>${t.names[s]}${s === fin.slot ? ` <span class="muted small">← answer</span>` : ""}</span></td>
    <td>${esc(a.place)}</td><td>${a.source === "pinpoint" ? "Pinpoint" : "clue"}</td>
    <td class="num">${pct(a.pre.support)}</td><td class="num">${a.pre.density != null ? Math.round(a.pre.density).toLocaleString() : "–"}</td>
    <td class="num">${fmtKm(a.pre.d_top1)}</td><td class="num">${pct(a.pre.set_mass)}</td>
    <td>${t.truth ? kmBadge(a.unzoomed.km) : "–"}</td><td>${t.truth ? kmBadge(a.zoomed.km) : "–"}</td></tr>`).join("")}
  </tbody></table>`;
}

function pRead(t) {
  const o = t.clues.observe || {}, ev = t.clues.evidence || [], qs = t.clues.queries || [];
  const toks = (xs, cls = "") => (xs && xs.length ? xs.map((x) => `<span class="token ${cls}">${esc(x)}</span>`).join("") : `<span class="muted">none</span>`);
  const clueAreas = t.areas.map((a, s) => [a, s]).filter(([a]) => a.source === "clue");
  const readRow = (a) => `${a.pre.read_n ? `${a.pre.read_n} read clue${a.pre.read_n > 1 ? "s" : ""}, best P(yes) ${pct(a.pre.read_p)}` : ""}${a.pre.recall_n ? `${a.pre.read_n ? "; " : ""}${a.pre.recall_n} search phrase${a.pre.recall_n > 1 ? "s" : ""}` : ""}` || `<span class="muted">no clue nearby</span>`;
  return `<p class="lead"><b>Read</b> is a costly action (2 units). Qwen3.5-4B reads the photo (visible text, place names, scene features, a country guess) and writes search phrases.
    Each is looked up in an offline text search over GeoNames, Overture places and YFCC titles and tags. A fine-tuned 4B judge scores whether each found place fits the photo.
    Read reveals each area's clue columns and the clue-place hypotheses.</p>
  ${Object.keys(o).length ? `<div class="obs">
    <div class="box"><div class="k">Text read</div>${toks(o.text, "text")}</div>
    <div class="box"><div class="k">Place names</div>${toks(o.place_names)}</div>
    <div class="box"><div class="k">Scene features</div>${toks(o.features)}</div>
    <div class="box"><div class="k">Guesses</div>country <b>${esc(o.country || "–")}</b> · language ${esc(o.language || "–")} · script ${esc(o.script || "–")} · driving ${esc(o.driving_side || "–")}</div>
    <div class="box"><div class="k">Search phrases</div>${toks(qs)}</div></div>` : `<p class="muted">No clue reading for this photo${t.set === "upload" ? " (the 4B server was not running)" : ""}.</p>`}
  <h3>Text-search places</h3>
  ${ev.length ? `<table class="t"><thead><tr><th>Query</th><th>From</th><th>Top place found</th><th class="num">Matches</th><th>Cluster share</th><th>Judge P(yes)</th><th>Judged</th>${t.truth ? "<th>Distance</th>" : ""}</tr></thead><tbody>
    ${ev.map((e, k) => `<tr data-mark="clue:${k}"><td><b>${esc(e.query)}</b></td><td class="muted">${esc(e.source)}</td>
      <td>${esc((e.region || []).filter(Boolean).reverse().join(", ") || "–")}</td><td class="num">${e.matches?.toLocaleString() ?? "–"}</td>
      <td><div class="flex">${bar(e.share, "alt")}<span class="small">${pct(e.share)}</span></div></td>
      <td>${e.p_yes != null ? `<div class="flex">${bar(e.p_yes, "alt")}<span class="small">${pct(e.p_yes)}</span></div>` : `<span class="muted">–</span>`}</td>
      <td>${e.used ? "yes" : `<span class="muted">no</span>`}</td>${t.truth ? `<td>${e.lat != null ? kmBadge(dist(t.truth, e)) : "–"}</td>` : ""}</tr>`).join("")}
    </tbody></table>` : `<p class="muted">No text-search hits.</p>`}
  <h3>What reading changes for each area</h3>
  <table class="t"><thead><tr><th>Area</th><th>Place</th><th>Clue evidence inside its 25 km box</th></tr></thead><tbody>
  ${t.areas.map((a, s) => `<tr data-mark="area:${s}"><td><span class="area-tag"><span class="swatch" style="background:${areaColor(a)}"></span>${t.names[s]}</span></td><td>${esc(a.place)}</td><td>${readRow(a)}</td></tr>`).join("")}
  </tbody></table>
  ${clueAreas.length ? `<p class="muted small">Clue-place hypotheses added by reading: ${clueAreas.map(([a, s]) => `${t.names[s]} (${esc(a.place)})`).join(", ")}.</p>` : ""}`;
}
function dist(a, b) {
  const R = Math.PI / 180, h = Math.sin(((b.lat - a.lat) * R) / 2) ** 2 + Math.cos(a.lat * R) * Math.cos(b.lat * R) * Math.sin(((b.lon - a.lon) * R) / 2) ** 2;
  return 2 * 6371 * Math.asin(Math.min(1, Math.sqrt(h)));
}

function pRecognise(t) {
  const mx = Math.max(...t.areas.map((a) => a.megaloc.best || 0), 1e-6);
  return `<p class="lead"><b>Recognise</b> (1 unit) embeds the photo with MegaLoc, a same-place descriptor, and compares it with each area's zoomed-search photos.
    Four numbers per area: best match, top-3 mean, and each relative to the photo's other areas. A high, isolated match suggests the same spot is in the gallery.</p>
  <table class="t"><thead><tr><th>Area</th><th>Place</th><th>Best MegaLoc cosine</th><th class="num">Top-3 mean</th><th class="num">Best vs other areas</th><th class="num">Top-3 vs others</th>${t.truth ? "<th>Zoomed answer</th>" : ""}</tr></thead><tbody>
  ${t.areas.map((a, s) => `<tr data-mark="area:${s}"><td><span class="area-tag"><span class="swatch" style="background:${areaColor(a)}"></span>${t.names[s]}</span></td><td>${esc(a.place)}</td>
    <td><div class="flex">${bar((a.megaloc.best || 0) / mx)}<span class="small">${a.megaloc.best?.toFixed(3) ?? "–"}</span></div></td>
    <td class="num">${a.megaloc.top3?.toFixed(3) ?? "–"}</td><td class="num">${signed(a.megaloc.best_vs_others)}</td><td class="num">${signed(a.megaloc.top3_vs_others)}</td>
    ${t.truth ? `<td>${kmBadge(a.zoomed.km)}</td>` : ""}</tr>`).join("")}
  </tbody></table>`;
}
const signed = (x) => (x == null ? "–" : `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(3)}`);

function pZoom(t) {
  return `<p class="lead"><b>Zoom</b> (1 unit per area) re-runs the retrained Pinpoint restricted to a ~25 km box around the area. The agent sees the restricted search's
    reranker states and statistics, and a zoomed area answers with the gated kernel pick (the densest high-scoring spot, outlined). Hover an area name to focus the map on it.</p>
  <div class="zoom-grid">${t.areas.map((a, s) => `<div><div class="zoom-head" data-zoom="${s}">
      <span class="area-tag" data-mark="area:${s}"><span class="swatch" style="background:${areaColor(a)}"></span>${t.names[s]} · ${esc(a.place)}</span>
      <span class="muted small">${a.candidates.length} candidates · top score ${a.post.top1?.toFixed(3) ?? "–"} · ${a.post.photographers ?? "–"} photographers · spread ${fmtKm(a.post.r_spread)}</span>
      ${t.truth ? kmBadge(a.zoomed.km, "zoom pick") : ""}</div>
      <div class="strip">${a.candidates.map((c, k) => candCard(c, `zoom:${s}:${k}`, `#${k + 1}`, `<div class="line"><span class="muted">${k === a.zoomed.cand_index ? "<b>zoom pick</b>" : "score"}</span><span class="rank">${c.score?.toFixed(3) ?? "–"}</span></div>${c.megaloc != null ? `<div class="line"><span class="muted">MegaLoc</span><span class="rank">${c.megaloc.toFixed(3)}</span></div>` : ""}`, k === a.zoomed.cand_index)).join("")}</div></div>`).join("")}</div>`;
}

function pDecide(t) {
  const ag = t.agent[S.lam], steps = ag.steps, names = t.names;
  const lams = Object.keys(t.agent);
  const kmOf = (s, zoomed) => (zoomed ? t.areas[s].zoomed.km : t.areas[s].unzoomed.km);
  const stepCard = (st, j) => {
    const final = st.action === "answer";
    const chips = `<div class="state-chips">${st.read ? `<span class="token">read ✓</span>` : ""}${st.recognise ? `<span class="token">recognised ✓</span>` : ""}${st.zoomed.map((z, s) => (z ? `<span class="token">${names[s]} zoomed</span>` : "")).join("")}${!st.read && !st.recognise && !st.zoomed.some(Boolean) ? `<span class="token">glance only</span>` : ""}</div>`;
    const act = final ? "" : st.action === "zoom" ? `zoom ${names[st.slot]}` : st.action;
    const prow = (label, p, chosen) => `<div class="prow ${chosen ? "chosen" : ""}"><span class="l">${label}</span>${bar(p, chosen ? "pick" : "")}<span class="v">${pct(p)}</span></div>`;
    const policy = final ? `<div class="muted small">The agent stopped. The answer head picks <b>${names[st.slot]}</b>${st.zoomed[st.slot] ? " and answers with its zoomed point" : " and answers with the best pool candidate inside it"}.</div>` :
      [...st.p_zoom.map((p, s) => (st.zoomed[s] || p == null ? "" : prow(`zoom ${names[s]}`, p, st.action === "zoom" && st.slot === s))),
       st.read ? "" : prow("read", st.p_read, st.action === "read"), st.recognise ? "" : prow("recognise", st.p_recognise, st.action === "recognise"), prow("stop", st.p_stop, st.action === "stop")].join("");
    const ansMax = Math.max(...st.answer.map((x) => x || 0));
    const answer = st.answer.map((p, s) => (p == null || (!st.read && t.areas[s].source === "clue") ? "" :
      `<div class="prow wide ${p === ansMax ? "chosen" : ""}" data-mark="area:${s}"><span class="l">${names[s]} ${t.truth ? kmBadge(kmOf(s, st.zoomed[s])) : ""}</span>${bar(p, p === ansMax ? "pick" : "")}<span class="v">${pct(p)}</span></div>`)).join("");
    return `<div class="step ${final ? "final" : ""}"><div class="num">${final ? "✓" : j + 1}</div>
      <div><h4>${final ? "Answer" : `Policy → <span>${esc(act)}</span>`} <span class="muted">${final ? "" : "(action probabilities)"}</span></h4>${chips}${policy}</div>
      <div><h4>Answer head <span class="muted">· which area, if it stopped now</span></h4>${answer}</div></div>`;
  };
  const fixedCols = Object.keys(t.fixed);
  const fixedKm = (name, s) => (name === "zoom all" || name === "all tools" ? t.areas[s].zoomed.km : t.areas[s].unzoomed.km);
  const argmax = (xs) => xs.reduce((b, x, i) => ((x ?? -1) > (xs[b] ?? -1) ? i : b), 0);
  return `<div class="flex" style="justify-content:space-between"><p class="lead" style="margin:0">A REINFORCE policy over the frozen answer head chooses tools until it stops;
    return = [answer within 25 km] − λ × cost (zoom 1, recognise 1, read 2). λ 0 is the 51.2% system on the test set, λ 0.005 the cheaper one.</p>
    ${lams.length > 1 ? `<div class="seg" id="lam">${lams.map((l) => `<button data-lam="${l}" class="${l === S.lam ? "on" : ""}">λ ${l}</button>`).join("")}</div>` : ""}</div>
  <div class="steps" style="margin-top:14px">${steps.map(stepCard).join("")}</div>
  <h3>Same photo, fixed tool sets (no policy)</h3>
  <p class="lead">Answer-head probabilities per area when given a fixed set of tools. Bottom row: the distance of the area it would pick.</p>
  <table class="t"><thead><tr><th>Area</th>${fixedCols.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>
  ${t.areas.map((a, s) => `<tr data-mark="area:${s}"><td><span class="area-tag"><span class="swatch" style="background:${areaColor(a)}"></span>${names[s]}</span> <span class="muted small">${esc(a.place)}</span></td>
    ${fixedCols.map((c) => { const xs = t.fixed[c], p = xs[s]; const on = argmax(xs) === s; return `<td><div class="flex">${bar(p, on ? "pick" : "")}<span class="small" style="${on ? "font-weight:700" : ""}">${(a.source === "clue" && (c === "glance" || c === "recognise" || c === "zoom all")) ? "–" : pct(p)}</span></div></td>`; }).join("")}</tr>`).join("")}
  ${t.truth ? `<tr><td class="muted">picked area's answer</td>${fixedCols.map((c) => `<td>${kmBadge(fixedKm(c, argmax(t.fixed[c])))}</td>`).join("")}</tr>` : ""}
  </tbody></table>`;
}

function bindPanel() {
  const lam = $("#lam");
  if (lam) lam.onclick = (e) => { const b = e.target.closest("[data-lam]"); if (b) { S.lam = b.dataset.lam; renderHero(); renderMapBase(); renderStages(); } };
  const more = $("[data-act=pool-all]");
  if (more) more.onclick = () => { S.retrieveAll = !S.retrieveAll; renderStages(); };
  document.querySelectorAll("[data-zoom]").forEach((el) => (el.onmouseenter = () => { S.zoomFocus = +el.dataset.zoom; fitStage("zoom"); }));
}

// ---------- theme ----------
function toggleTheme() {
  const cur = document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const next = cur === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("viz-theme", next);
  applyTiles();
  if (S.trace) { renderMapBase(); renderStages(); }
}

// ---------- upload ----------
function setupUpload() {
  const dlg = $("#upload-dlg"), status = $("#upload-status");
  if (!S.sets.some((s) => s.name === "upload" && s.live)) {  // cached mode: browse saved traces only
    $("#upload-btn").hidden = true;
    $("#empty").textContent = "Pick a photo on the left.";
    return;
  }
  $("#upload-btn").onclick = () => {
    status.textContent = "";
    dlg.showModal();
  };
  $("#upload-go").onclick = async (e) => {
    e.preventDefault();
    const file = $("#upload-file").files[0];
    if (!file) return (status.textContent = "Choose a photo first.");
    const lat = $("#upload-lat").value, lon = $("#upload-lon").value;
    $("#upload-go").disabled = true;
    status.textContent = "Uploading…";
    try {
      const r = await fetch(`/api/upload?name=${encodeURIComponent(file.name)}&lat=${lat}&lon=${lon}`, { method: "POST", body: file });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || r.status);
      for (;;) {
        await new Promise((res) => setTimeout(res, 1500));
        const s = await api(`/api/job/${j.job}`);
        status.textContent = s.message || s.state;
        if (s.state === "done") {
          dlg.close();
          S.sets = await api("/api/sets");
          await init_tabs_refresh();
          await selectSet("upload");
          openTrace("upload", s.id);
          break;
        }
        if (s.state === "error") break;
      }
    } catch (err) {
      status.textContent = `Failed: ${err.message}`;
    } finally {
      $("#upload-go").disabled = false;
    }
  };
}
async function init_tabs_refresh() {
  const up = S.sets.find((s) => s.name === "upload");
  const tab = document.querySelector('#set-tabs .tab[data-set="upload"] .n');
  if (tab && up) tab.textContent = up.n;
}

init().catch((e) => { $("#empty").textContent = `Could not load: ${e.message}`; });
