(function () {
  "use strict";

  let DATA = null;
  let sortCol = "metric";
  let sortAsc = true;
  const expanded = new Set();

  const $ = (id) => document.getElementById(id);

  function fmtSec(v) {
    if (v == null || Number.isNaN(v)) return "—";
    if (v >= 100) return v.toFixed(1) + " s";
    return v.toFixed(2) + " s";
  }

  function fmtSpeedup(v) {
    if (v == null || Number.isNaN(v)) return "—";
    return v.toFixed(2) + "×";
  }

  function fmtSci(v) {
    if (v == null || Number.isNaN(v)) return "—";
    return v.toExponential(2);
  }

  function esc(s) {
    return String(s ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function corePillClass(label) {
    const l = (label || "").toLowerCase();
    if (l.includes("p-core") || l.includes("prime") || l.includes("big")) return "pill-pcore";
    if (l.includes("e-core") || l.includes("little")) return "pill-ecore";
    return "pill-core";
  }

  function getMetric(entry, workload) {
    if (workload === "probe_geomean") {
      return {
        wall: entry.probe_geomean_s,
        speedup: entry.probe_geomean_speedup,
        secondary: entry.hero_wall_s != null ? `${fmtSec(entry.hero_wall_s)} (${fmtSpeedup(entry.hero_speedup)})` : "—",
        pass: entry.all_pass,
        accText: entry.all_pass ? `worst ${fmtSci(entry.worst_rel_rms)}` : "FAIL",
      };
    }
    const d = entry.decks && entry.decks[workload];
    if (!d) return null;
    const parts = [];
    if (d.tran_s != null) parts.push(`tran ${d.tran_s.toFixed(2)}s`);
    if (d.load_s != null) parts.push(`load ${d.load_s.toFixed(2)}s`);
    if (d.factor_s != null) parts.push(`fac ${d.factor_s.toFixed(2)}s`);
    let accText = `rms ${fmtSci(d.rel_rms)}`;
    if (d.sar_codes_count > 0) {
      accText = `${d.sar_codes_count}/${d.sar_codes_count} codes${d.sar_codes_match ? "" : " MISMATCH"}`;
    } else if (d.pll_code != null) {
      accText = `code ${d.pll_code} · ${d.pll_fout_mhz != null ? d.pll_fout_mhz.toFixed(1) + " MHz" : ""}`;
    } else if (d.sdm_bits_count > 0) {
      accText = `${d.sdm_bits_count}/${d.sdm_bits_count} PDM bits${d.sdm_bits_match ? "" : " MISMATCH"}`;
    } else if (d.exact_grid) {
      accText += " · exact";
    }
    return {
      wall: d.wall_s,
      speedup: d.speedup_vs_ref,
      secondary: parts.join(" / ") || "—",
      pass: d.pass,
      accText,
    };
  }

  function populateCoreClasses() {
    const sel = $("coreclass-select");
    const classes = new Set();
    DATA.entries.forEach((e) => {
      if (e.core_class && e.core_class.label) classes.add(e.core_class.label);
    });
    Array.from(classes)
      .sort()
      .forEach((lbl) => {
        const opt = document.createElement("option");
        opt.value = lbl;
        opt.textContent = lbl;
        sel.appendChild(opt);
      });
  }

  function renderKPIs() {
    const passEntries = DATA.entries.filter((e) => e.all_pass);

    const bestProbe = passEntries
      .filter((e) => e.probe_geomean_s != null)
      .sort((a, b) => a.probe_geomean_s - b.probe_geomean_s)[0];
    if (bestProbe) {
      $("kpi-probe-val").textContent = `${fmtSec(bestProbe.probe_geomean_s)} (${fmtSpeedup(bestProbe.probe_geomean_speedup)})`;
      $("kpi-probe-sub").textContent = `${bestProbe.host.cpu_model} [${bestProbe.core_class.label}] · ${bestProbe.build.name} (${bestProbe.solver.toUpperCase()})`;
    }

    const bestHero = passEntries
      .filter((e) => e.hero_wall_s != null)
      .sort((a, b) => a.hero_wall_s - b.hero_wall_s)[0];
    if (bestHero) {
      $("kpi-hero-val").textContent = `${fmtSec(bestHero.hero_wall_s)} (${fmtSpeedup(bestHero.hero_speedup)})`;
      $("kpi-hero-sub").textContent = `${bestHero.host.cpu_model} [${bestHero.core_class.label}] · ${bestHero.build.name} (${bestHero.solver.toUpperCase()})`;
    }

    let maxSp = 0;
    let maxSpLabel = "";
    passEntries.forEach((e) => {
      Object.entries(e.decks || {}).forEach(([dk, d]) => {
        if (d.speedup_vs_ref && d.speedup_vs_ref > maxSp) {
          maxSp = d.speedup_vs_ref;
          maxSpLabel = `${dk} · ${e.build.name} (${e.solver.toUpperCase()})`;
        }
      });
    });
    if (maxSp > 0) {
      $("kpi-speedup-val").textContent = fmtSpeedup(maxSp);
      $("kpi-speedup-sub").textContent = maxSpLabel;
    }

    const cpuClasses = new Set(DATA.entries.map((e) => `${e.host.cpu_model}::${e.core_class.label}`));
    $("kpi-count-val").textContent = String(DATA.entries.length);
    $("kpi-count-sub").textContent = `${passEntries.length}/${DATA.entries.length} PASS · ${cpuClasses.size} CPU/core class${cpuClasses.size === 1 ? "" : "es"}`;
  }

  function renderWorkloadBanner(workload) {
    const el = $("workload-banner");
    if (workload === "probe_geomean") {
      const refGeomean = Math.exp(
        DATA.probe_decks.reduce((acc, d) => acc + Math.log(DATA.reference_baseline.wall_s[d]), 0) /
          DATA.probe_decks.length
      );
      const deckListHtml = DATA.probe_decks.map((d) => `<code>${esc(d)}</code>`).join(", ");
      el.innerHTML =
        `<div><strong>Probe Suite Geomean</strong> — Geometric mean of single-core wall times across all ${DATA.probe_decks.length} GF180 post-PEX circuits (${deckListHtml}).</div>` +
        `<div>Reference Baseline (Stock Sparse 1c): <strong>${refGeomean.toFixed(2)} s</strong></div>`;
    } else {
      const m = DATA.deck_meta[workload] || {};
      const refS = DATA.reference_baseline.wall_s[workload];
      el.innerHTML =
        `<div><strong>${esc(m.title || workload)}</strong> — ${esc(m.description || "")} (<code>${m.eqns || "?"} eqns</code>, source: <code>${esc(m.source || "")}</code>)</div>` +
        `<div>Reference Baseline (Stock Sparse 1c): <strong>${fmtSec(refS)}</strong></div>`;
    }
  }

  function renderBarChart(rows, workload) {
    const chart = $("bar-chart");
    $("chart-title").textContent =
      workload === "probe_geomean"
        ? "Single-Core Probe Geomean Wall Time (Lower is Faster)"
        : `Single-Core Wall Time — ${workload} (Lower is Faster)`;

    if (!rows.length) {
      chart.innerHTML = `<div class="empty">No matching single-core entries for the current filters.</div>`;
      return;
    }

    const maxWall = Math.max(...rows.map((r) => r.m.wall || 0), 0.01);
    chart.innerHTML = rows
      .slice(0, 12)
      .map(({ e, m }) => {
        const pct = Math.max(2, (m.wall / maxWall) * 100);
        const isStock = (e.build.name || "").toLowerCase().includes("stock");
        let cls = "klu-std";
        if (isStock) cls = "stock";
        else if (e.solver === "sparse") cls = "sparse";
        else if (e.build.pdk_mode === "precompiled_uselib") cls = "klu-uselib";
        const pdkTag = e.build.pdk_mode === "precompiled_uselib" ? " · .uselib" : "";
        return `
          <div class="bar-row">
            <div class="bar-label" title="${esc(e.host.cpu_model)} [${esc(e.core_class.label)}] — ${esc(e.build.name)} (${e.solver})">
              ${esc(e.build.name)} <small>(${e.solver.toUpperCase()}${pdkTag}) · ${esc(e.core_class.label)}</small>
            </div>
            <div class="bar-track">
              <div class="bar-fill ${cls}" style="width:${pct.toFixed(1)}%"></div>
            </div>
            <div class="bar-metric">${fmtSec(m.wall)} <span style="color:var(--green)">(${fmtSpeedup(m.speedup)})</span></div>
          </div>`;
      })
      .join("");
  }

  function renderDetailRow(e) {
    const ff = (e.build && e.build.feature_flags) || {};
    const optGen = ff.NGSPICE_OPT_GENERAL == null ? "n/a" : String(ff.NGSPICE_OPT_GENERAL);
    const optZen4 = ff.NGSPICE_OPT_ZEN4 == null ? "n/a" : String(ff.NGSPICE_OPT_ZEN4);
    const isa = (e.host.isa_features || []).join(", ") || "—";
    const deckOrder = [DATA.hero_deck, ...DATA.probe_decks];
    const deckRows = deckOrder
      .filter((dk) => e.decks && e.decks[dk])
      .map((dk) => {
        const d = e.decks[dk];
        let acc = fmtSci(d.rel_rms);
        if (d.sar_codes_count > 0) {
          acc = `${d.sar_codes_count}/${d.sar_codes_count} codes (EOC Δt ${d.eoc_max_dt_ps != null ? d.eoc_max_dt_ps.toFixed(1) + " ps" : "—"})`;
        } else if (d.pll_code != null) {
          acc = `AFC code ${d.pll_code} · ${d.pll_fout_mhz != null ? d.pll_fout_mhz.toFixed(2) + " MHz" : "—"} · VC ${d.pll_vc_final_v != null ? d.pll_vc_final_v.toFixed(3) + " V" : "—"}`;
        } else if (d.sdm_bits_count > 0) {
          acc = `${d.sdm_bits_count}/${d.sdm_bits_count} PDM bits (Δt ${d.sdm_pdm_max_dt_ps != null ? d.sdm_pdm_max_dt_ps.toFixed(1) + " ps" : "—"})`;
        } else if (d.exact_grid) {
          acc += " (exact grid)";
        }
        return `
          <tr>
            <td><code>${esc(dk)}</code></td>
            <td class="num">${fmtSec(d.wall_s)}</td>
            <td class="num speedup-val">${fmtSpeedup(d.speedup_vs_ref)}</td>
            <td class="num">${fmtSec(d.tran_s)}</td>
            <td class="num">${fmtSec(d.load_s)}</td>
            <td class="num">${fmtSec(d.factor_s)}</td>
            <td class="num">${fmtSec(d.solve_s)}</td>
            <td class="num">${fmtSec(d.trunc_s)}</td>
            <td class="num">${d.tot_iter != null ? d.tot_iter.toLocaleString() : "—"}</td>
            <td><span class="pill ${d.pass ? "pill-pass" : "pill-fail"}">${d.pass ? "PASS" : "FAIL"}</span> ${esc(acc)}</td>
          </tr>`;
      })
      .join("");

    return `
      <tr class="detail-row">
        <td colspan="8">
          <div class="detail-grid">
            <div class="detail-box">
              <h4>Build &amp; Compile Flags</h4>
              <div class="kv-list">
                <div class="kv-k">Build Name</div><div class="kv-v">${esc(e.build.name)}</div>
                <div class="kv-k">ngspice Version</div><div class="kv-v">${esc(e.build.ngspice_version || "—")}</div>
                <div class="kv-k">Repo / Ref</div><div class="kv-v">${esc(e.build.repo || "—")} ${e.build.git_ref ? `@ ${esc(e.build.git_ref)}` : ""}</div>
                <div class="kv-k">Compiler</div><div class="kv-v">${esc(e.build.compiler || "—")}</div>
                <div class="kv-k">Feature Flags</div><div class="kv-v">NGSPICE_OPT_GENERAL=${esc(optGen)}, NGSPICE_OPT_ZEN4=${esc(optZen4)}</div>
                <div class="kv-k">PDK Mode</div><div class="kv-v">${esc(e.build.pdk_mode)}${e.build.pdk_precompile_s != null ? ` (precompile ${(e.build.pdk_precompile_s * 1000).toFixed(1)} ms)` : ""}</div>
                <div class="kv-k">./configure</div><div class="kv-v">${esc(e.build.configure_flags || "—")}</div>
                <div class="kv-k">CFLAGS</div><div class="kv-v">${esc(e.build.cflags || "—")}</div>
                <div class="kv-k">LDFLAGS</div><div class="kv-v">${esc(e.build.ldflags || "—")}</div>
                ${e.build.notes ? `<div class="kv-k">Notes</div><div class="kv-v">${esc(e.build.notes)}</div>` : ""}
              </div>
            </div>
            <div class="detail-box">
              <h4>Processor, Core Class &amp; Host</h4>
              <div class="kv-list">
                <div class="kv-k">CPU Model</div><div class="kv-v">${esc(e.host.cpu_model)}</div>
                <div class="kv-k">Core Class</div><div class="kv-v">${esc(e.core_class.label)} (${e.core_class.hybrid ? "Hybrid / big.LITTLE" : "Homogeneous"})</div>
                <div class="kv-k">Pinned CPU ID</div><div class="kv-v">${e.core_class.pin_cpu != null ? `cpu${e.core_class.pin_cpu} (1 core / 1 thread)` : "1 core / 1 thread"}</div>
                <div class="kv-k">Max Core Freq</div><div class="kv-v">${e.core_class.max_freq_mhz != null ? `${e.core_class.max_freq_mhz} MHz` : "—"}</div>
                <div class="kv-k">Caches</div><div class="kv-v">L1d ${esc(e.host.l1d_cache || "—")} / L2 ${esc(e.core_class.l2_cache || e.host.l2_cache || "—")} / L3 ${esc(e.host.l3_cache || "—")}</div>
                <div class="kv-k">ISA Features</div><div class="kv-v">${esc(isa)}</div>
                <div class="kv-k">OS / Arch</div><div class="kv-v">${esc(e.host.os)} (${esc(e.host.arch)})</div>
                <div class="kv-k">Memory</div><div class="kv-v">${e.host.ram_gb != null ? `${e.host.ram_gb} GiB` : "—"}</div>
                <div class="kv-k">Submission</div><div class="kv-v">${esc(e.submission_id)} (${esc(e.timestamp_utc || "")})</div>
              </div>
            </div>
          </div>
          <table class="subtable">
            <thead>
              <tr>
                <th>Deck</th>
                <th class="num">Wall</th>
                <th class="num">Speedup</th>
                <th class="num">Tran</th>
                <th class="num">Load</th>
                <th class="num">Factor</th>
                <th class="num">Solve</th>
                <th class="num">Trunc</th>
                <th class="num">Iters</th>
                <th>Accuracy vs Golden</th>
              </tr>
            </thead>
            <tbody>${deckRows}</tbody>
          </table>
        </td>
      </tr>`;
  }

  function render() {
    const workload = $("workload-select").value;
    const solver = $("solver-select").value;
    const pdk = $("pdk-select").value;
    const coreClass = $("coreclass-select").value;
    const q = $("search-input").value.trim().toLowerCase();

    renderWorkloadBanner(workload);
    $("col-secondary-hdr").textContent =
      workload === "probe_geomean" ? "Hero 16-Cyc (1c)" : "Tran / Load / Factor";

    let rows = DATA.entries
      .map((e) => ({ e, m: getMetric(e, workload) }))
      .filter(({ e, m }) => {
        if (!m || m.wall == null) return false;
        if (solver !== "all" && e.solver !== solver) return false;
        if (pdk !== "all" && e.build.pdk_mode !== pdk) return false;
        if (coreClass !== "all" && e.core_class.label !== coreClass) return false;
        if (q) {
          const hay = [
            e.host.cpu_model,
            e.core_class.label,
            e.build.name,
            e.build.compiler,
            e.build.cflags,
            e.build.configure_flags,
            e.build.git_ref,
            e.solver,
          ]
            .join(" ")
            .toLowerCase();
          if (!hay.includes(q)) return false;
        }
        return true;
      });

    rows.sort((a, b) => {
      let cmp = 0;
      if (sortCol === "metric") cmp = a.m.wall - b.m.wall;
      else if (sortCol === "speedup") cmp = (b.m.speedup || 0) - (a.m.speedup || 0);
      else if (sortCol === "cpu") cmp = a.e.host.cpu_model.localeCompare(b.e.host.cpu_model);
      else if (sortCol === "build") cmp = a.e.build.name.localeCompare(b.e.build.name);
      else if (sortCol === "solver") cmp = a.e.solver.localeCompare(b.e.solver);
      return sortAsc ? cmp : -cmp;
    });

    renderBarChart(rows, workload);

    const tbody = $("leaderboard-body");
    if (!rows.length) {
      tbody.innerHTML = `<tr><td colspan="8" class="empty">No matching single-core entries for the selected filters.</td></tr>`;
      return;
    }

    tbody.innerHTML = rows
      .map(({ e, m }, idx) => {
        const isOpen = expanded.has(e.entry_id);
        const rank = idx + 1;
        const rankCls = rank <= 3 ? `rank-${rank}` : "";
        const ff = (e.build && e.build.feature_flags) || {};
        const pills = [];
        if (ff.NGSPICE_OPT_GENERAL === 1) pills.push(`<span class="pill pill-opt">OPT_GENERAL</span>`);
        if (ff.NGSPICE_OPT_ZEN4 === 1) pills.push(`<span class="pill pill-opt">OPT_ZEN4</span>`);
        if (e.build.pdk_mode === "precompiled_uselib") pills.push(`<span class="pill pill-uselib">.uselib</span>`);

        const mainRow = `
          <tr class="main-row ${isOpen ? "expanded" : ""}" data-id="${esc(e.entry_id)}">
            <td class="rank-num ${rankCls}"><span class="chev">${isOpen ? "▾" : "▸"}</span>${rank}</td>
            <td>
              <div class="cpu-name">${esc(e.host.cpu_model)}</div>
              <div class="subline">
                <span class="pill ${corePillClass(e.core_class.label)}">${esc(e.core_class.label)}</span>
                ${e.core_class.max_freq_mhz ? `${(e.core_class.max_freq_mhz / 1000).toFixed(2)} GHz` : ""}
                ${e.core_class.l2_cache ? `· L2 ${esc(e.core_class.l2_cache)}` : ""}
              </div>
            </td>
            <td>
              <div class="cpu-name">${esc(e.build.name)}</div>
              <div class="subline">${pills.join("")}${esc(e.build.compiler || "")}</div>
            </td>
            <td><span class="pill pill-solver">${esc(e.solver)}</span></td>
            <td class="num"><strong>${fmtSec(m.wall)}</strong></td>
            <td class="num speedup-val">${fmtSpeedup(m.speedup)}</td>
            <td class="num" style="color:var(--muted);font-size:12px">${esc(m.secondary)}</td>
            <td>
              <span class="pill ${m.pass ? "pill-pass" : "pill-fail"}">${m.pass ? "PASS" : "FAIL"}</span>
              <div class="subline">${esc(m.accText)}</div>
            </td>
          </tr>`;
        return isOpen ? mainRow + renderDetailRow(e) : mainRow;
      })
      .join("");

    tbody.querySelectorAll("tr.main-row").forEach((tr) => {
      tr.addEventListener("click", () => {
        const id = tr.getAttribute("data-id");
        if (expanded.has(id)) expanded.delete(id);
        else expanded.add(id);
        render();
      });
    });
  }

  function initThemeToggle() {
    const btn = $("theme");
    if (!btn) return;
    btn.addEventListener("click", () => {
      const root = document.documentElement;
      const dark = root.dataset.theme
        ? root.dataset.theme === "dark"
        : window.matchMedia("(prefers-color-scheme: dark)").matches;
      root.dataset.theme = dark ? "light" : "dark";
      try {
        localStorage.setItem("ngspice-bench-theme", root.dataset.theme);
      } catch (_) {}
    });
  }

  async function init() {
    initThemeToggle();
    try {
      const resp = await fetch("data/leaderboard.json", { cache: "no-cache" });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      DATA = await resp.json();
      populateCoreClasses();
      const hash = window.location.hash.replace(/^#/, "");
      const wsel = $("workload-select");
      if (hash && Array.from(wsel.options).some((o) => o.value === hash)) {
        wsel.value = hash;
      }
      renderKPIs();
      render();
      $("footer-meta").textContent = `ngspice-bench · Single-Core Leaderboard · Updated ${DATA.generated_utc}`;

      ["workload-select", "solver-select", "pdk-select", "coreclass-select"].forEach((id) =>
        $(id).addEventListener("change", render)
      );
      $("search-input").addEventListener("input", render);
      document.querySelectorAll("th.sortable").forEach((th) => {
        th.addEventListener("click", () => {
          const col = th.getAttribute("data-sort");
          if (sortCol === col) sortAsc = !sortAsc;
          else {
            sortCol = col;
            sortAsc = col !== "speedup";
          }
          render();
        });
      });
    } catch (err) {
      $("leaderboard-body").innerHTML = `<tr><td colspan="8" class="empty">Failed to load leaderboard.json: ${esc(err.message)}</td></tr>`;
    }
  }

  init();
})();
