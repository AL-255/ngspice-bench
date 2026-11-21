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
    if (l.includes("super") || l.includes("p-core") || l.includes("prime") || l.includes("big")) return "pill-pcore";
    if (l.includes("e-core") || l.includes("little")) return "pill-ecore";
    return "pill-core";
  }

  function renderCommitBadge(e, showBranch) {
    const shortCommit = e.short_commit || "—";
    if (e.commit_url) {
      const tip = `View commit ${e.commit || shortCommit} on GitHub` + (e.git_branch ? ` (${e.git_branch})` : "");
      const branchHtml = showBranch && e.git_branch ? ` <span class="commit-branch">${esc(e.git_branch)}</span>` : "";
      return `<a class="commit-link" href="${esc(e.commit_url)}" target="_blank" rel="noopener" title="${esc(tip)}"><code class="commit-code">${esc(shortCommit)}</code></a>${branchHtml}`;
    }
    if (e.commit) {
      const branchHtml = showBranch && e.git_branch ? ` <span class="commit-branch">${esc(e.git_branch)}</span>` : "";
      return `<code class="commit-code" title="${esc(`Commit ${e.commit}`)}">${esc(shortCommit)}</code>${branchHtml}`;
    }
    const tip = (e.build && e.build.ngspice_version) || "Commercial / non-git release";
    return `<code class="commit-code commit-nongit" title="${esc(tip)}">${esc(shortCommit)}</code>`;
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
      if (d.sdm_bits_match) {
        accText = `${d.sdm_bits_count}/${d.sdm_bits_count} PDM bits`;
      } else if (d.sdm_dac_corr != null) {
        accText = `DAC r=${Number(d.sdm_dac_corr).toFixed(4)} (${d.sdm_bits_count} cyc)`;
      } else {
        accText = `${d.sdm_bits_count} cyc MISMATCH`;
      }
    } else if (d.opa_tasks_count > 0) {
      accText = `${d.opa_tasks_count.toLocaleString()} tasks · rms ${fmtSci(d.rel_rms)}`;
    } else if (d.riscv_cycles_count > 0) {
      accText = `${d.riscv_cycles_count}/${d.riscv_cycles_count} cyc${d.riscv_cycles_match ? "" : " MISMATCH"} · rms ${fmtSci(d.rel_rms)}`;
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
      const refLabel = m.reference_label || "Stock Sparse 1c";
      const eqStr = m.eqns != null ? `${m.eqns.toLocaleString()} eqns` : "? eqns";
      el.innerHTML =
        `<div><strong>${esc(m.title || workload)}</strong> — ${esc(m.description || "")} (<code>${eqStr}</code>, source: <code>${esc(m.source || "")}</code>)</div>` +
        `<div>Reference Baseline (${esc(refLabel)}): <strong>${fmtSec(refS)}</strong></div>`;
    }
  }

  function shortCpuName(cpu) {
    return String(cpu || "Unknown CPU")
      .replace(/\s+with\s+Radeon\s+Graphics$/i, "")
      .replace(/\s+Processor$/i, "")
      .replace(/\s+\d+-Core$/i, "")
      .trim();
  }

  function enabledOptFlagsLabel(build) {
    if (build && build.simulator === "spectre") {
      return build.preset_label || "Spectre X";
    }
    const ff = (build && build.feature_flags) || {};
    const flags = [];
    if (ff.NGSPICE_OPT_GENERAL === 1) flags.push("OPT_GENERAL");
    if (ff.NGSPICE_OPT_ZEN4 === 1) flags.push("OPT_ZEN4");
    if (ff.NGSPICE_OPT_APPLE === 1) flags.push("OPT_APPLE");
    return flags.length ? flags.join(" + ") : "Stock";
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

    const visible = rows.slice(0, 14);
    const maxWall = Math.max(...visible.map((r) => r.m.wall || 0), 0.01);

    chart.innerHTML = visible
      .map(({ e, m }) => {
        const pct = Math.max(2, ((m.wall || 0) / maxWall) * 100);
        const isStock = (e.build.name || "").toLowerCase().includes("stock");
        const isSpectre = e.solver === "spectrex" || e.build.simulator === "spectre";
        let cls = "klu-std";
        if (isSpectre) cls = "spectre";
        else if (isStock) cls = "stock";
        else if (e.solver === "sparse") cls = "sparse";
        else if (e.build.pdk_mode === "precompiled_uselib") cls = "klu-uselib";
        const cpuTier = `${shortCpuName(e.host.cpu_model)} · ${e.core_class.label}`;
        const optFlags = enabledOptFlagsLabel(e.build);
        const algo = e.solver === "klu" ? "KLU" : e.solver === "sparse" ? "Sparse" : e.solver === "spectrex" ? "Spectre X" : e.solver.toUpperCase();
        const pdkTag =
          e.build.pdk_mode === "precompiled_uselib"
            ? " · .uselib"
            : e.build.pdk_mode === "commercial_spectre"
            ? " · Comm. PDK"
            : "";
        const line2 = isSpectre ? `${optFlags}${pdkTag}` : `${optFlags} · ${algo}${pdkTag}`;
        return `
          <div class="bar-row">
            <div class="bar-label" title="${esc(e.host.cpu_model)} [${esc(e.core_class.label)}] — ${esc(e.build.name)} (${e.solver}${e.short_commit ? `, ${esc(e.short_commit)}` : ""})">
              <div class="bar-label-main">${esc(cpuTier)}</div>
              <div class="bar-label-sub">${esc(line2)}</div>
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
    const isSpectre = e.solver === "spectrex" || e.build.simulator === "spectre";
    const ff = (e.build && e.build.feature_flags) || {};
    const optGen = ff.NGSPICE_OPT_GENERAL == null ? "n/a" : String(ff.NGSPICE_OPT_GENERAL);
    const optZen4 = ff.NGSPICE_OPT_ZEN4 == null ? "n/a" : String(ff.NGSPICE_OPT_ZEN4);
    const optApple = ff.NGSPICE_OPT_APPLE == null ? null : String(ff.NGSPICE_OPT_APPLE);
    const ffStr = isSpectre
      ? Object.entries(ff).map(([k, v]) => `${k}=${v}`).join(", ") || "Commercial binary"
      : `NGSPICE_OPT_GENERAL=${optGen}, NGSPICE_OPT_ZEN4=${optZen4}` + (optApple != null ? `, NGSPICE_OPT_APPLE=${optApple}` : "");
    const isa = (e.host.isa_features || []).join(", ") || "—";
    const deckOrder = [DATA.hero_deck, ...DATA.probe_decks, ...(DATA.scaling_decks || [])];
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
          acc = d.sdm_bits_match
            ? `${d.sdm_bits_count}/${d.sdm_bits_count} PDM bits (Δt ${d.sdm_pdm_max_dt_ps != null ? d.sdm_pdm_max_dt_ps.toFixed(1) + " ps" : "—"})`
            : `DAC r=${d.sdm_dac_corr != null ? Number(d.sdm_dac_corr).toFixed(4) : "—"} (${d.sdm_bits_count} cyc)`;
        } else if (d.riscv_cycles_count > 0) {
          acc = `${d.riscv_cycles_count}/${d.riscv_cycles_count} cyc (bit-exact) · rms ${fmtSci(d.rel_rms)}`;
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

    const repoHtml =
      e.build.repo && /^https?:\/\//.test(e.build.repo)
        ? `<a class="inline-link" href="${esc(e.build.repo)}" target="_blank" rel="noopener">${esc(e.build.repo)}</a>`
        : esc(e.build.repo || "—");
    const refHtml = e.build.git_ref ? ` <span class="muted-ref">(${esc(e.build.git_ref)})</span>` : "";

    return `
      <tr class="detail-row">
        <td colspan="9">
          <div class="detail-grid">
            <div class="detail-box">
              <h4>Build &amp; Compile Flags</h4>
              <div class="kv-list">
                <div class="kv-k">Build Name</div><div class="kv-v">${esc(e.build.name)}</div>
                <div class="kv-k">Simulator Version</div><div class="kv-v">${esc(e.build.ngspice_version || "—")}</div>
                <div class="kv-k">Commit / Release</div><div class="kv-v">${renderCommitBadge(e, true)}</div>
                <div class="kv-k">Repo / Ref</div><div class="kv-v">${repoHtml}${refHtml}</div>
                <div class="kv-k">Compiler</div><div class="kv-v">${esc(e.build.compiler || "—")}</div>
                <div class="kv-k">Feature Flags</div><div class="kv-v">${esc(ffStr)}</div>
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
                ${e.host.platform ? `<div class="kv-k">Platform</div><div class="kv-v">${esc(e.host.platform)}</div>` : ""}
                <div class="kv-k">Core Class</div><div class="kv-v">${esc(e.core_class.label)} (${e.core_class.hybrid ? "Hybrid / big.LITTLE" : "Homogeneous"})</div>
                <div class="kv-k">Pinned CPU ID</div><div class="kv-v">${e.core_class.pin_cpu != null ? `cpu${e.core_class.pin_cpu} (1 core / 1 thread)` : "1 core / 1 thread"}</div>
                <div class="kv-k">Max Core Freq</div><div class="kv-v">${e.core_class.max_freq_mhz != null ? `${e.core_class.max_freq_mhz} MHz` : "—"}</div>
                <div class="kv-k">Caches</div><div class="kv-v">L1d ${esc(e.core_class.l1d_cache || e.host.l1d_cache || "—")} / L2 ${esc(e.core_class.l2_cache || e.host.l2_cache || "—")} / L3 ${esc(e.host.l3_cache || "—")}</div>
                <div class="kv-k">ISA Features</div><div class="kv-v">${esc(isa)}</div>
                <div class="kv-k">OS / Arch</div><div class="kv-v">${esc(e.host.os)} (${esc(e.host.arch)})</div>
                <div class="kv-k">Memory</div><div class="kv-v">${esc(e.host.memory || (e.host.ram_gb != null ? `${e.host.ram_gb} GiB` : "—"))}</div>
                ${e.host.storage ? `<div class="kv-k">Storage (SSD)</div><div class="kv-v">${esc(e.host.storage)}</div>` : ""}
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
            e.host.platform || "",
            e.core_class.label,
            e.build.name,
            e.build.compiler,
            e.build.cflags,
            e.build.configure_flags,
            e.build.git_ref,
            e.commit || "",
            e.short_commit || "",
            e.git_branch || "",
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
      else if (sortCol === "commit") cmp = String(a.e.short_commit || "").localeCompare(String(b.e.short_commit || ""));
      else if (sortCol === "solver") cmp = a.e.solver.localeCompare(b.e.solver);
      return sortAsc ? cmp : -cmp;
    });

    renderBarChart(rows, workload);

    const tbody = $("leaderboard-body");
    if (!rows.length) {
      tbody.innerHTML = `<tr><td colspan="9" class="empty">No matching single-core entries for the selected filters.</td></tr>`;
      return;
    }

    tbody.innerHTML = rows
      .map(({ e, m }, idx) => {
        const isOpen = expanded.has(e.entry_id);
        const rank = idx + 1;
        const rankCls = rank <= 3 ? `rank-${rank}` : "";
        const ff = (e.build && e.build.feature_flags) || {};
        const pills = [];
        if (e.build.simulator === "spectre") pills.push(`<span class="pill pill-spectre">${esc(e.build.preset_label || "Spectre X")}</span>`);
        if (e.build.pdk_mode === "commercial_spectre") pills.push(`<span class="pill pill-commercial">Commercial GF180 PDK</span>`);
        if (ff.NGSPICE_OPT_GENERAL === 1) pills.push(`<span class="pill pill-opt">OPT_GENERAL</span>`);
        if (ff.NGSPICE_OPT_ZEN4 === 1) pills.push(`<span class="pill pill-opt">OPT_ZEN4</span>`);
        if (ff.NGSPICE_OPT_APPLE === 1) pills.push(`<span class="pill pill-opt">OPT_APPLE</span>`);
        if (e.build.pdk_mode === "precompiled_uselib") pills.push(`<span class="pill pill-uselib">.uselib</span>`);

        const subMeta = [
          e.host.platform ? esc(e.host.platform) : "",
          e.core_class.max_freq_mhz ? `${(e.core_class.max_freq_mhz / 1000).toFixed(2)} GHz` : "",
          e.core_class.l2_cache ? `L2 ${esc(e.core_class.l2_cache)}` : "",
        ].filter(Boolean).join(" · ");

        const mainRow = `
          <tr class="main-row ${isOpen ? "expanded" : ""}" data-id="${esc(e.entry_id)}">
            <td class="rank-num ${rankCls}"><span class="chev">${isOpen ? "▾" : "▸"}</span>${rank}</td>
            <td>
              <div class="cpu-name">${esc(e.host.cpu_model)}</div>
              <div class="subline">
                <span class="pill ${corePillClass(e.core_class.label)}">${esc(e.core_class.label)}</span>
                ${subMeta}
              </div>
            </td>
            <td>
              <div class="cpu-name">${esc(e.build.name)}</div>
              <div class="subline">${pills.join("")}${esc(e.build.compiler || "")}</div>
            </td>
            <td class="col-commit">${renderCommitBadge(e, false)}</td>
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
      tr.addEventListener("click", (ev) => {
        if (ev.target.closest("a")) return;
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
      $("leaderboard-body").innerHTML = `<tr><td colspan="9" class="empty">Failed to load leaderboard.json: ${esc(err.message)}</td></tr>`;
    }
  }

  init();
})();
