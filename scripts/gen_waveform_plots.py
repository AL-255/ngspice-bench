#!/usr/bin/env python3
"""Generate warm light-theme SVG transient waveform plots aligned with aicdb/hub-gemini4 palette."""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DECKS_DIR = REPO_ROOT / "decks"
GOLDEN_DIR = REPO_ROOT / "golden"
OUT_DIR = REPO_ROOT / "docs" / "img"

sys.path.insert(0, str(SCRIPT_DIR))
from compare_raw import (  # noqa: E402
    decode_pll_metrics,
    decode_sar12_codes,
    decode_sar12_weighted,
    decode_sdm_bits,
    parse_raw,
    reconstruct_sdm_ideal_dac,
)

# Palette from aicdb/hub-gemini4 (style.css + blocks.js COLORWAY)
COLORS = {
    "red": "#d9423a",      # brand red
    "blue": "#2f7fd1",     # colorway[1]
    "green": "#2ca02c",    # colorway[2]
    "orange": "#ff7f0e",   # colorway[3]
    "purple": "#9467bd",   # colorway[4]
    "cyan": "#17becf",     # colorway[5]
    "brown": "#a47148",    # colorway[8]
}


def parse_deck_pwl(deck_path: Path, src_name: str, t_end: float) -> Tuple[np.ndarray, np.ndarray]:
    """Parse an ideal PWL voltage source from a SPICE deck for exact input-signal plotting."""
    txt = deck_path.read_text()
    m = re.search(rf"^{re.escape(src_name)}\s+\S+\s+\S+\s+PWL\(([^)]+)\)", txt, re.M | re.I)
    if not m:
        return np.array([]), np.array([])
    nums = [float(x) for x in m.group(1).split()]
    ts = nums[0::2]
    vs = nums[1::2]
    if ts and ts[-1] < t_end:
        ts.append(t_end)
        vs.append(vs[-1])
    return np.asarray(ts, dtype=float), np.asarray(vs, dtype=float)


def compute_inst_freq_mhz(t: np.ndarray, v: np.ndarray, vth: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
    """Compute instantaneous frequency (MHz) from consecutive rising threshold crossings."""
    idx = np.where((v[:-1] < vth) & (v[1:] >= vth))[0]
    if len(idx) < 3:
        return np.array([]), np.array([])
    crs = np.array(
        [t[i] + (vth - v[i]) / (v[i + 1] - v[i]) * (t[i + 1] - t[i]) for i in idx],
        dtype=float,
    )
    dt = np.diff(crs)
    valid = dt > 1e-12
    f_mhz = 1.0 / dt[valid] / 1e6
    t_mid = 0.5 * (crs[:-1][valid] + crs[1:][valid])
    return t_mid, f_mhz


def decimate_minmax(t: np.ndarray, y: np.ndarray, max_bins: int = 720) -> Tuple[np.ndarray, np.ndarray]:
    """Envelope-preserving min/max decimation so narrow switching pulses are never lost."""
    n = len(t)
    if n <= max_bins * 2:
        return t, y
    edges = np.linspace(0, n, max_bins + 1, dtype=int)
    ts: List[float] = []
    ys: List[float] = []
    for i in range(max_bins):
        a, b = edges[i], edges[i + 1]
        if a >= b:
            continue
        seg = y[a:b]
        imin = a + int(np.argmin(seg))
        imax = a + int(np.argmax(seg))
        if imin <= imax:
            ts.extend((float(t[imin]), float(t[imax])))
            ys.extend((float(y[imin]), float(y[imax])))
        else:
            ts.extend((float(t[imax]), float(t[imin])))
            ys.extend((float(y[imax]), float(y[imin])))
    return np.asarray(ts), np.asarray(ys)


def fmt_num(v: float) -> str:
    av = abs(v)
    if av == 0:
        return "0"
    if av >= 1000:
        return f"{v:.0f}"
    if av >= 10:
        return f"{v:.1f}"
    if av >= 0.1:
        return f"{v:.2f}"
    return f"{v:.2g}"


def render_svg(
    title: str,
    subtitle: str,
    t_scale: float,
    t_unit: str,
    t_max_raw: float,
    panels: List[Dict],
    out_path: Path,
) -> None:
    W = 920
    margin_l = 66
    margin_r = 22
    header_h = 66
    panel_h = 122
    panel_gap = 40
    footer_h = 34
    plot_w = W - margin_l - margin_r
    H = header_h + len(panels) * panel_h + (len(panels) - 1) * panel_gap + footer_h

    default_t_max = t_max_raw * t_scale
    e = html.escape
    lines: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="{e(title)}">',
        "<style>",
        "  .bg { fill: #fffaf6; }",
        "  .panel-bg { fill: #ffffff; stroke: #f0dccd; stroke-width: 1; }",
        "  .grid { stroke: #f3e4d8; stroke-width: 1; }",
        "  .refline { stroke: #c4372f; stroke-width: 1.1; stroke-dasharray: 5 4; opacity: 0.7; }",
        "  .title { fill: #2b1a17; font-weight: 700; font-size: 13.5px; font-family: Inter,system-ui,-apple-system,'Segoe UI',sans-serif; }",
        "  .sub { fill: #7a5c52; font-size: 11px; font-family: ui-monospace,SFMono-Regular,Menlo,monospace; }",
        "  .ptitle { fill: #2b1a17; font-weight: 600; font-size: 11.5px; font-family: Inter,system-ui,-apple-system,'Segoe UI',sans-serif; }",
        "  .tick { fill: #7a5c52; font-size: 10.5px; font-family: ui-monospace,SFMono-Regular,Menlo,monospace; }",
        "  .legend { fill: #2b1a17; font-size: 11px; font-family: ui-monospace,SFMono-Regular,Menlo,monospace; }",
        "</style>",
        f'<rect class="bg" x="0" y="0" width="{W}" height="{H}" rx="8"/>',
        f'<text class="title" x="{margin_l}" y="22">{e(title)}</text>',
        f'<text class="sub" x="{margin_l}" y="39">{e(subtitle)}</text>',
    ]

    for pidx, panel in enumerate(panels):
        y0 = header_h + pidx * (panel_h + panel_gap)
        x0_val, x1_val = panel.get("xlim", (0.0, default_t_max))
        x_span = max(x1_val - x0_val, 1e-12)
        show_xticks = (pidx == len(panels) - 1) or (panel.get("xlim") is not None)

        lines.append(f'<text class="ptitle" x="{margin_l}" y="{y0 - 8}">{e(panel["title"])}</text>')

        lx = W - margin_r
        for tr in reversed(panel["traces"]):
            lbl = tr["label"]
            approx_w = len(lbl) * 6.8 + 20
            lx -= approx_w
            lines.append(
                f'<line x1="{lx:.1f}" y1="{y0 - 11.5:.1f}" x2="{lx + 12:.1f}" y2="{y0 - 11.5:.1f}" '
                f'stroke="{tr["color"]}" stroke-width="2.2" stroke-linecap="round"/>'
            )
            lines.append(f'<text class="legend" x="{lx + 16:.1f}" y="{y0 - 8}">{e(lbl)}</text>')
            lx -= 10

        lines.append(f'<rect class="panel-bg" x="{margin_l}" y="{y0}" width="{plot_w}" height="{panel_h}" rx="4"/>')

        if panel.get("ylim") is not None:
            ymin, ymax = panel["ylim"]
        else:
            all_y = np.concatenate([tr["y"] for tr in panel["traces"] if len(tr["y"]) > 0])
            ymin = float(np.min(all_y))
            ymax = float(np.max(all_y))
            pad = 0.08 * max(ymax - ymin, 1e-12)
            ymin -= pad
            ymax += pad
        if ymax <= ymin:
            ymax = ymin + 1.0

        for gi in range(5):
            frac = gi / 4.0
            gy = y0 + panel_h - frac * panel_h
            val = ymin + frac * (ymax - ymin)
            if 0 < gi < 4:
                lines.append(f'<line class="grid" x1="{margin_l}" y1="{gy:.1f}" x2="{margin_l + plot_w}" y2="{gy:.1f}"/>')
            if gi in (0, 2, 4):
                lines.append(f'<text class="tick" x="{margin_l - 7}" y="{gy + 3.5:.1f}" text-anchor="end">{fmt_num(val)}</text>')

        for xi in range(6):
            frac = xi / 5.0
            gx = margin_l + frac * plot_w
            if 0 < xi < 5:
                lines.append(f'<line class="grid" x1="{gx:.1f}" y1="{y0}" x2="{gx:.1f}" y2="{y0 + panel_h}"/>')
            if show_xticks:
                tval = x0_val + frac * x_span
                tick_lbl = f"{fmt_num(tval)} {t_unit}".strip()
                lines.append(
                    f'<text class="tick" x="{gx:.1f}" y="{y0 + panel_h + 15}" text-anchor="middle">{tick_lbl}</text>'
                )

        if panel.get("hline") is not None:
            hy = y0 + panel_h - ((panel["hline"] - ymin) / (ymax - ymin)) * panel_h
            lines.append(f'<line class="refline" x1="{margin_l}" y1="{hy:.1f}" x2="{margin_l + plot_w}" y2="{hy:.1f}"/>')

        clip_id = f"clip_{pidx}"
        lines.append(
            f'<clipPath id="{clip_id}"><rect x="{margin_l}" y="{y0}" width="{plot_w}" height="{panel_h}"/></clipPath>'
        )
        lines.append(f'<g clip-path="url(#{clip_id})">')
        for tr in panel["traces"]:
            tt, yy = tr["t"] * t_scale, tr["y"]
            if len(tt) == 0:
                continue
            if tr.get("step", False):
                pts: List[str] = []
                for k in range(len(tt)):
                    px = margin_l + ((float(tt[k]) - x0_val) / x_span) * plot_w
                    py = y0 + panel_h - ((float(yy[k]) - ymin) / (ymax - ymin)) * panel_h
                    if k > 0:
                        prev_py = y0 + panel_h - ((float(yy[k - 1]) - ymin) / (ymax - ymin)) * panel_h
                        pts.append(f"{px:.1f},{prev_py:.1f}")
                    pts.append(f"{px:.1f},{py:.1f}")
                last_py = y0 + panel_h - ((float(yy[-1]) - ymin) / (ymax - ymin)) * panel_h
                pts.append(f"{margin_l + plot_w:.1f},{last_py:.1f}")
            else:
                tt_d, yy_d = decimate_minmax(tt, yy, max_bins=650)
                xs = margin_l + ((tt_d - x0_val) / x_span) * plot_w
                ys = y0 + panel_h - ((yy_d - ymin) / (ymax - ymin)) * panel_h
                pts = [f"{xs[k]:.1f},{ys[k]:.1f}" for k in range(len(xs))]
            sw = tr.get("width", 1.5)
            op = tr.get("opacity", 0.95)
            dash = f' stroke-dasharray="{tr["dash"]}"' if tr.get("dash") else ""
            lines.append(
                f'<polyline fill="none" stroke="{tr["color"]}" stroke-width="{sw}" stroke-opacity="{op}"{dash} '
                f'stroke-linejoin="round" points="{" ".join(pts)}"/>'
            )
        lines.append("</g>")

    lines.append("</svg>")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path.name} ({out_path.stat().st_size / 1024:.1f} KiB)")


def _mask_window(t: np.ndarray, y: np.ndarray, t_min: float, t_max: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    m = t >= t_min if t_max is None else ((t >= t_min) & (t <= t_max))
    return t[m], y[m]


def render_opa1k_bode_svg(
    npz_path: Path,
    manifest_path: Path,
    out_path: Path,
    n_pvt_tasks: int = 16000,
) -> None:
    """Render a 2x2 op-amp portfolio figure: Open-Loop Bode Mag & Phase, 10 mV Step, and UGBW vs IDD."""
    data = np.load(npz_path)
    manifest = json.loads(manifest_path.read_text())

    freq = data["freq"].astype(float)
    logf = np.log10(freq)
    aol_db = data["aol_db"].astype(float)
    aol_ph = data["aol_ph"].astype(float)
    idd_ua = np.maximum(data["idd_a"].astype(float), 1e-12) * 1e6
    tr_dv = data["tr_dv_mv"].astype(float)
    n_cells = aol_db.shape[0]
    t_norm = np.linspace(0.0, 1.0, tr_dv.shape[1])

    cl_pf = np.array([float(m["cl_pf"]) for m in manifest])
    vdd_v = np.array([float(m["vdd"]) for m in manifest])
    archs = [m["arch"] for m in manifest]

    # Exact 0 dB UGBW & Phase Margin across all 1,000 cells
    ugbw = np.full(n_cells, np.nan)
    pm_deg = np.full(n_cells, np.nan)
    for i in range(n_cells):
        y = aol_db[i]
        cross = np.where((y[:-1] >= 0.0) & (y[1:] < 0.0))[0]
        if len(cross) > 0:
            k = int(cross[0])
            frac = (0.0 - y[k]) / (y[k + 1] - y[k])
            ugbw[i] = 10.0 ** (logf[k] + frac * (logf[k + 1] - logf[k]))
            ph_u = aol_ph[i, k] + frac * (aol_ph[i, k + 1] - aol_ph[i, k])
            pm_deg[i] = 180.0 + ph_u

    a0_med = float(np.median(aol_db[:, 0]))
    ugbw_p5, ugbw_med, ugbw_p95 = np.nanpercentile(ugbw, [5, 50, 95])
    pm_p5, pm_med, pm_p95 = np.nanpercentile(pm_deg, [5, 50, 95])
    ifom_med = float(np.nanmedian((ugbw / 1e6) * cl_pf / (idd_ua / 1e3)))
    tr_end_med = float(np.median(tr_dv[:, -1]))
    tr_pk_p95 = float(np.percentile(np.max(tr_dv, axis=1), 95))

    # Stratified sample of 300 cells (25 per topology, evenly spaced by UGBW) + global extremes
    by_arch: Dict[str, List[int]] = {}
    for i, a in enumerate(archs):
        by_arch.setdefault(a, []).append(i)
    sample_set = set()
    for a in sorted(by_arch):
        idxs = sorted(by_arch[a], key=lambda k: ugbw[k] if np.isfinite(ugbw[k]) else 0.0)
        pick = np.linspace(0, len(idxs) - 1, min(25, len(idxs)), dtype=int)
        for p in pick:
            sample_set.add(idxs[int(p)])
    sample_idxs = sorted(sample_set)

    # All-1,000 exact percentiles
    db_p10, db_p50, db_p90 = np.percentile(aol_db, [10, 50, 90], axis=0)
    ph_p10, ph_p50, ph_p90 = np.percentile(aol_ph, [10, 50, 90], axis=0)
    tr_p10, tr_p50, tr_p90 = np.percentile(tr_dv, [10, 50, 90], axis=0)

    W, H = 960, 672
    pw, ph = 408.0, 228.0
    x_left, x_right = 64.0, 534.0
    y_top, y_bot = 88.0, 396.0

    def sx(v: float, x0: float, xmin: float, xmax: float) -> float:
        return x0 + ((v - xmin) / (xmax - xmin)) * pw

    def sy(v: float, y0: float, ymin: float, ymax: float) -> float:
        return y0 + ph - ((v - ymin) / (ymax - ymin)) * ph

    def build_compound_path(
        xs_arr: np.ndarray,
        ys_mat: np.ndarray,
        indices: List[int],
        x0: float,
        y0: float,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        stride: int = 1,
    ) -> str:
        xs_s = xs_arr[::stride]
        px = [x0 + ((float(v) - xmin) / (xmax - xmin)) * pw for v in xs_s]
        segs: List[str] = []
        for idx in indices:
            ys_s = ys_mat[idx, ::stride]
            py = [y0 + ph - ((float(v) - ymin) / (ymax - ymin)) * ph for v in ys_s]
            cmds = [f"M{px[0]:.1f},{py[0]:.1f}"] + [f"L{px[k]:.1f},{py[k]:.1f}" for k in range(1, len(px))]
            segs.append("".join(cmds))
        return "".join(segs)

    def build_band_polygon(
        xs_arr: np.ndarray,
        lo_arr: np.ndarray,
        hi_arr: np.ndarray,
        x0: float,
        y0: float,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
    ) -> str:
        top_pts = [f"{sx(float(xs_arr[k]), x0, xmin, xmax):.1f},{sy(float(hi_arr[k]), y0, ymin, ymax):.1f}" for k in range(len(xs_arr))]
        bot_pts = [f"{sx(float(xs_arr[k]), x0, xmin, xmax):.1f},{sy(float(lo_arr[k]), y0, ymin, ymax):.1f}" for k in range(len(xs_arr) - 1, -1, -1)]
        return " ".join(top_pts + bot_pts)

    def build_polyline(
        xs_arr: np.ndarray,
        ys_arr: np.ndarray,
        x0: float,
        y0: float,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
    ) -> str:
        return " ".join(
            f"{sx(float(xs_arr[k]), x0, xmin, xmax):.1f},{sy(float(ys_arr[k]), y0, ymin, ymax):.1f}"
            for k in range(len(xs_arr))
        )

    def fmt_hz_short(hz: float) -> str:
        if hz >= 1e9:
            return f"{hz/1e9:.0f} GHz"
        if hz >= 1e6:
            return f"{hz/1e6:.2f} MHz" if hz < 1e7 else f"{hz/1e6:.0f} MHz"
        if hz >= 1e3:
            return f"{hz/1e3:.0f} kHz"
        return f"{hz:.0f} Hz"

    e = html.escape
    title = "opa1k_pvt16 — 1,000 Post-PEX Op-Amps × 16 PVT Corners (Open-Loop Bode, Step & FoM)"
    subtitle = (
        f"1,000 GF180 cells · 12 topologies · A0 med {a0_med:.1f} dB · "
        f"UGBW {fmt_hz_short(ugbw_p5)}–{fmt_hz_short(ugbw_p95)} (med {fmt_hz_short(ugbw_med)}) · "
        f"PM med {pm_med:.1f}° · {n_pvt_tasks:,} PVT tasks"
    )

    lines: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="{e(title)}">',
        "<style>",
        "  .bg { fill: #fffaf6; }",
        "  .panel-bg { fill: #ffffff; stroke: #f0dccd; stroke-width: 1; }",
        "  .grid { stroke: #f3e4d8; stroke-width: 1; }",
        "  .guide { stroke: #d8c2b2; stroke-width: 1; stroke-dasharray: 4 4; }",
        "  .ref-red { stroke: #c4372f; stroke-width: 1.2; stroke-dasharray: 5 4; opacity: 0.78; }",
        "  .ref-green { stroke: #2ca02c; stroke-width: 1.1; stroke-dasharray: 4 4; opacity: 0.75; }",
        "  .title { fill: #2b1a17; font-weight: 700; font-size: 13.5px; font-family: Inter,system-ui,-apple-system,'Segoe UI',sans-serif; }",
        "  .sub { fill: #7a5c52; font-size: 10.8px; font-family: ui-monospace,SFMono-Regular,Menlo,monospace; }",
        "  .ptitle { fill: #2b1a17; font-weight: 600; font-size: 11.5px; font-family: Inter,system-ui,-apple-system,'Segoe UI',sans-serif; }",
        "  .axlbl { fill: #7a5c52; font-size: 10.5px; font-family: Inter,system-ui,-apple-system,'Segoe UI',sans-serif; }",
        "  .tick { fill: #7a5c52; font-size: 10px; font-family: ui-monospace,SFMono-Regular,Menlo,monospace; }",
        "  .legend { fill: #2b1a17; font-size: 10px; font-family: ui-monospace,SFMono-Regular,Menlo,monospace; }",
        "  .badge-bg { fill: #fffaf6; fill-opacity: 0.92; stroke: #ebd5c5; stroke-width: 1; }",
        "  .badge-txt { fill: #4a332d; font-size: 9.8px; font-family: ui-monospace,SFMono-Regular,Menlo,monospace; }",
        "  .guide-txt { fill: #9b7e72; font-size: 9px; font-family: ui-monospace,SFMono-Regular,Menlo,monospace; }",
        "</style>",
        f'<rect class="bg" x="0" y="0" width="{W}" height="{H}" rx="8"/>',
        f'<text class="title" x="{x_left:.0f}" y="24">{e(title)}</text>',
        f'<text class="sub" x="{x_left:.0f}" y="42">{e(subtitle)}</text>',
    ]

    sample_200f = [i for i in sample_idxs if cl_pf[i] < 1.0]
    sample_2p = [i for i in sample_idxs if cl_pf[i] >= 1.0]

    def draw_legend_inline(x_right_edge: float, y_base: float, items: List[Tuple[str, str, str]]) -> None:
        lx = x_right_edge
        for lbl, color, dash in reversed(items):
            approx_w = len(lbl) * 6.0 + 18
            lx -= approx_w
            da = f' stroke-dasharray="{dash}"' if dash else ""
            lines.append(
                f'<line x1="{lx:.1f}" y1="{y_base - 3.5:.1f}" x2="{lx + 11:.1f}" y2="{y_base - 3.5:.1f}" '
                f'stroke="{color}" stroke-width="2.1"{da} stroke-linecap="round"/>'
            )
            lines.append(f'<text class="legend" x="{lx + 15:.1f}" y="{y_base:.1f}">{e(lbl)}</text>')
            lx -= 8

    # ---------- PANEL 1 (Top-Left): Open-Loop Bode Magnitude ----------
    f_min, f_max = 0.0, 9.0
    db_min, db_max = -40.0, 120.0
    lines.append(f'<text class="ptitle" x="{x_left:.0f}" y="{y_top - 10:.0f}">Open-Loop Bode Magnitude |A_ol(f)| (dB)</text>')
    draw_legend_inline(
        x_left + pw,
        y_top - 10,
        [("200fF", COLORS["blue"], ""), ("2pF", COLORS["orange"], ""), ("Med(1k)", COLORS["red"], "")],
    )
    lines.append(f'<rect class="panel-bg" x="{x_left:.0f}" y="{y_top:.0f}" width="{pw:.0f}" height="{ph:.0f}" rx="4"/>')
    for db_tick in (-40, 0, 40, 80, 120):
        gy = sy(db_tick, y_top, db_min, db_max)
        if db_min < db_tick < db_max:
            lines.append(f'<line class="grid" x1="{x_left:.0f}" y1="{gy:.1f}" x2="{x_left + pw:.0f}" y2="{gy:.1f}"/>')
        lines.append(f'<text class="tick" x="{x_left - 7:.0f}" y="{gy + 3.5:.1f}" text-anchor="end">{db_tick}</text>')
    dec_labels = {0: "1", 2: "100", 4: "10k", 6: "1M", 8: "100M", 9: "1G"}
    for dec in range(10):
        gx = sx(float(dec), x_left, f_min, f_max)
        if 0 < dec < 9:
            lines.append(f'<line class="grid" x1="{gx:.1f}" y1="{y_top:.0f}" x2="{gx:.1f}" y2="{y_top + ph:.0f}"/>')
        if dec in dec_labels:
            lines.append(f'<text class="tick" x="{gx:.1f}" y="{y_top + ph + 14:.0f}" text-anchor="middle">{dec_labels[dec]}</text>')
    lines.append(f'<text class="axlbl" x="{x_left + pw / 2:.1f}" y="{y_top + ph + 29:.0f}" text-anchor="middle">Frequency (Hz)</text>')
    y_0db = sy(0.0, y_top, db_min, db_max)
    lines.append(f'<line class="ref-red" x1="{x_left:.0f}" y1="{y_0db:.1f}" x2="{x_left + pw:.0f}" y2="{y_0db:.1f}"/>')

    lines.append(f'<clipPath id="clip_mag"><rect x="{x_left:.0f}" y="{y_top:.0f}" width="{pw:.0f}" height="{ph:.0f}"/></clipPath>')
    lines.append('<g clip-path="url(#clip_mag)">')
    p_mag_2p = build_compound_path(logf, aol_db, sample_2p, x_left, y_top, f_min, f_max, db_min, db_max, stride=3)
    p_mag_200 = build_compound_path(logf, aol_db, sample_200f, x_left, y_top, f_min, f_max, db_min, db_max, stride=3)
    lines.append(f'<path fill="none" stroke="{COLORS["orange"]}" stroke-width="0.65" stroke-opacity="0.22" d="{p_mag_2p}"/>')
    lines.append(f'<path fill="none" stroke="{COLORS["blue"]}" stroke-width="0.65" stroke-opacity="0.22" d="{p_mag_200}"/>')
    lines.append(
        f'<polygon fill="{COLORS["red"]}" fill-opacity="0.12" points="'
        f'{build_band_polygon(logf, db_p10, db_p90, x_left, y_top, f_min, f_max, db_min, db_max)}"/>'
    )
    lines.append(
        f'<polyline fill="none" stroke="{COLORS["red"]}" stroke-width="2.2" stroke-linejoin="round" points="'
        f'{build_polyline(logf, db_p50, x_left, y_top, f_min, f_max, db_min, db_max)}"/>'
    )
    lines.append(f'<rect class="badge-bg" x="{x_left + 8:.0f}" y="{y_top + ph - 26:.0f}" width="174" height="19" rx="3"/>')
    lines.append(
        f'<text class="badge-txt" x="{x_left + 14:.0f}" y="{y_top + ph - 13:.0f}">'
        f'0 dB UGBW: {fmt_hz_short(ugbw_p5)}–{fmt_hz_short(ugbw_p95)}</text>'
    )
    lines.append("</g>")

    # ---------- PANEL 2 (Top-Right): Open-Loop Bode Phase ----------
    ph_min, ph_max = -225.0, 15.0
    lines.append(f'<text class="ptitle" x="{x_right:.0f}" y="{y_top - 10:.0f}">Open-Loop Bode Phase ∠A_ol(f) (deg)</text>')
    draw_legend_inline(
        x_right + pw,
        y_top - 10,
        [("60° PM", COLORS["green"], "4 3"), ("-180°", COLORS["red"], "4 3"), ("Med(1k)", COLORS["red"], "")],
    )
    lines.append(f'<rect class="panel-bg" x="{x_right:.0f}" y="{y_top:.0f}" width="{pw:.0f}" height="{ph:.0f}" rx="4"/>')
    for ph_tick in (-225, -180, -135, -90, -45, 0):
        gy = sy(ph_tick, y_top, ph_min, ph_max)
        if ph_min < ph_tick < ph_max:
            lines.append(f'<line class="grid" x1="{x_right:.0f}" y1="{gy:.1f}" x2="{x_right + pw:.0f}" y2="{gy:.1f}"/>')
        lines.append(f'<text class="tick" x="{x_right - 7:.0f}" y="{gy + 3.5:.1f}" text-anchor="end">{ph_tick}°</text>')
    for dec in range(10):
        gx = sx(float(dec), x_right, f_min, f_max)
        if 0 < dec < 9:
            lines.append(f'<line class="grid" x1="{gx:.1f}" y1="{y_top:.0f}" x2="{gx:.1f}" y2="{y_top + ph:.0f}"/>')
        if dec in dec_labels:
            lines.append(f'<text class="tick" x="{gx:.1f}" y="{y_top + ph + 14:.0f}" text-anchor="middle">{dec_labels[dec]}</text>')
    lines.append(f'<text class="axlbl" x="{x_right + pw / 2:.1f}" y="{y_top + ph + 29:.0f}" text-anchor="middle">Frequency (Hz)</text>')
    y_m180 = sy(-180.0, y_top, ph_min, ph_max)
    y_m120 = sy(-120.0, y_top, ph_min, ph_max)
    lines.append(f'<line class="ref-red" x1="{x_right:.0f}" y1="{y_m180:.1f}" x2="{x_right + pw:.0f}" y2="{y_m180:.1f}"/>')
    lines.append(f'<line class="ref-green" x1="{x_right:.0f}" y1="{y_m120:.1f}" x2="{x_right + pw:.0f}" y2="{y_m120:.1f}"/>')

    lines.append(f'<clipPath id="clip_ph"><rect x="{x_right:.0f}" y="{y_top:.0f}" width="{pw:.0f}" height="{ph:.0f}"/></clipPath>')
    lines.append('<g clip-path="url(#clip_ph)">')
    p_ph_2p = build_compound_path(logf, aol_ph, sample_2p, x_right, y_top, f_min, f_max, ph_min, ph_max, stride=3)
    p_ph_200 = build_compound_path(logf, aol_ph, sample_200f, x_right, y_top, f_min, f_max, ph_min, ph_max, stride=3)
    lines.append(f'<path fill="none" stroke="{COLORS["orange"]}" stroke-width="0.65" stroke-opacity="0.22" d="{p_ph_2p}"/>')
    lines.append(f'<path fill="none" stroke="{COLORS["blue"]}" stroke-width="0.65" stroke-opacity="0.22" d="{p_ph_200}"/>')
    lines.append(
        f'<polygon fill="{COLORS["red"]}" fill-opacity="0.12" points="'
        f'{build_band_polygon(logf, ph_p10, ph_p90, x_right, y_top, f_min, f_max, ph_min, ph_max)}"/>'
    )
    lines.append(
        f'<polyline fill="none" stroke="{COLORS["red"]}" stroke-width="2.2" stroke-linejoin="round" points="'
        f'{build_polyline(logf, ph_p50, x_right, y_top, f_min, f_max, ph_min, ph_max)}"/>'
    )
    lines.append(f'<rect class="badge-bg" x="{x_right + 8:.0f}" y="{y_top + ph - 26:.0f}" width="188" height="19" rx="3"/>')
    lines.append(
        f'<text class="badge-txt" x="{x_right + 14:.0f}" y="{y_top + ph - 13:.0f}">'
        f'PM med {pm_med:.1f}° (P5–P95: {pm_p5:.0f}°–{pm_p95:.0f}°)</text>'
    )
    lines.append("</g>")

    # ---------- PANEL 3 (Bottom-Left): Closed-Loop 10 mV Step Response ----------
    tr_min, tr_max = -1.0, 15.0
    lines.append(f'<text class="ptitle" x="{x_left:.0f}" y="{y_bot - 10:.0f}">10 mV Step Response Δv_out (mV)</text>')
    draw_legend_inline(
        x_left + pw,
        y_bot - 10,
        [("200fF", COLORS["blue"], ""), ("2pF", COLORS["orange"], ""), ("Med(1k)", COLORS["green"], "")],
    )
    lines.append(f'<rect class="panel-bg" x="{x_left:.0f}" y="{y_bot:.0f}" width="{pw:.0f}" height="{ph:.0f}" rx="4"/>')
    for v_tick in (0, 5, 10, 15):
        gy = sy(float(v_tick), y_bot, tr_min, tr_max)
        if tr_min < v_tick < tr_max:
            lines.append(f'<line class="grid" x1="{x_left:.0f}" y1="{gy:.1f}" x2="{x_left + pw:.0f}" y2="{gy:.1f}"/>')
        lines.append(f'<text class="tick" x="{x_left - 7:.0f}" y="{gy + 3.5:.1f}" text-anchor="end">{v_tick}</text>')
    for xi in range(6):
        frac = xi / 5.0
        gx = sx(frac, x_left, 0.0, 1.0)
        if 0 < xi < 5:
            lines.append(f'<line class="grid" x1="{gx:.1f}" y1="{y_bot:.0f}" x2="{gx:.1f}" y2="{y_bot + ph:.0f}"/>')
        lines.append(f'<text class="tick" x="{gx:.1f}" y="{y_bot + ph + 14:.0f}" text-anchor="middle">{frac:.1f}</text>')
    lines.append(f'<text class="axlbl" x="{x_left + pw / 2:.1f}" y="{y_bot + ph + 29:.0f}" text-anchor="middle">Normalized Time (t / t_stop)</text>')
    y_10mv = sy(10.0, y_bot, tr_min, tr_max)
    lines.append(f'<line class="ref-red" x1="{x_left:.0f}" y1="{y_10mv:.1f}" x2="{x_left + pw:.0f}" y2="{y_10mv:.1f}"/>')

    lines.append(f'<clipPath id="clip_step"><rect x="{x_left:.0f}" y="{y_bot:.0f}" width="{pw:.0f}" height="{ph:.0f}"/></clipPath>')
    lines.append('<g clip-path="url(#clip_step)">')
    p_tr_2p = build_compound_path(t_norm, tr_dv, sample_2p, x_left, y_bot, 0.0, 1.0, tr_min, tr_max, stride=2)
    p_tr_200 = build_compound_path(t_norm, tr_dv, sample_200f, x_left, y_bot, 0.0, 1.0, tr_min, tr_max, stride=2)
    lines.append(f'<path fill="none" stroke="{COLORS["orange"]}" stroke-width="0.65" stroke-opacity="0.22" d="{p_tr_2p}"/>')
    lines.append(f'<path fill="none" stroke="{COLORS["blue"]}" stroke-width="0.65" stroke-opacity="0.22" d="{p_tr_200}"/>')
    lines.append(
        f'<polygon fill="{COLORS["green"]}" fill-opacity="0.14" points="'
        f'{build_band_polygon(t_norm, tr_p10, tr_p90, x_left, y_bot, 0.0, 1.0, tr_min, tr_max)}"/>'
    )
    lines.append(
        f'<polyline fill="none" stroke="{COLORS["green"]}" stroke-width="2.2" stroke-linejoin="round" points="'
        f'{build_polyline(t_norm, tr_p50, x_left, y_bot, 0.0, 1.0, tr_min, tr_max)}"/>'
    )
    lines.append(f'<rect class="badge-bg" x="{x_left + pw - 238:.0f}" y="{y_bot + ph - 26:.0f}" width="230" height="19" rx="3"/>')
    lines.append(
        f'<text class="badge-txt" x="{x_left + pw - 232:.0f}" y="{y_bot + ph - 13:.0f}">'
        f'Settled med {tr_end_med:.2f} mV · P95 pk {tr_pk_p95:.1f} mV</text>'
    )
    lines.append("</g>")

    # ---------- PANEL 4 (Bottom-Right): UGBW vs I_DD Portfolio Scatter (All 1,000 Cells) ----------
    ix_min, ix_max = -3.0, 3.0  # log10(I_DD / uA): 1 nA .. 1 mA
    uy_min, uy_max = 3.0, 9.0   # log10(UGBW / Hz): 1 kHz .. 1 GHz
    lines.append(f'<text class="ptitle" x="{x_right:.0f}" y="{y_bot - 10:.0f}">Portfolio Trade-Off: UGBW (Hz) vs I_DD (All 1,000 Cells)</text>')
    lines.append(f'<rect class="panel-bg" x="{x_right:.0f}" y="{y_bot:.0f}" width="{pw:.0f}" height="{ph:.0f}" rx="4"/>')
    u_ticks = {3: "1k", 4: "10k", 5: "100k", 6: "1M", 7: "10M", 8: "100M", 9: "1G"}
    for u_dec, u_lbl in u_ticks.items():
        gy = sy(float(u_dec), y_bot, uy_min, uy_max)
        if uy_min < u_dec < uy_max:
            lines.append(f'<line class="grid" x1="{x_right:.0f}" y1="{gy:.1f}" x2="{x_right + pw:.0f}" y2="{gy:.1f}"/>')
        lines.append(f'<text class="tick" x="{x_right - 7:.0f}" y="{gy + 3.5:.1f}" text-anchor="end">{u_lbl}</text>')
    i_ticks = {-3: "1n", -2: "10n", -1: "100n", 0: "1µ", 1: "10µ", 2: "100µ", 3: "1m"}
    for i_dec, i_lbl in i_ticks.items():
        gx = sx(float(i_dec), x_right, ix_min, ix_max)
        if ix_min < i_dec < ix_max:
            lines.append(f'<line class="grid" x1="{gx:.1f}" y1="{y_bot:.0f}" x2="{gx:.1f}" y2="{y_bot + ph:.0f}"/>')
        lines.append(f'<text class="tick" x="{gx:.1f}" y="{y_bot + ph + 14:.0f}" text-anchor="middle">{i_lbl}</text>')
    lines.append(f'<text class="axlbl" x="{x_right + pw / 2:.1f}" y="{y_bot + ph + 29:.0f}" text-anchor="middle">Quiescent Supply Current I_DD (A)</text>')

    lines.append(f'<clipPath id="clip_fom"><rect x="{x_right:.0f}" y="{y_bot:.0f}" width="{pw:.0f}" height="{ph:.0f}"/></clipPath>')
    lines.append('<g clip-path="url(#clip_fom)">')
    # Constant UGBW/I_DD guide diagonals: log10(UGBW) = log10(I_DD_uA) + c
    for c_val, c_lbl in ((5.0, "0.1 GHz/mA"), (6.0, "1 GHz/mA"), (7.0, "10 GHz/mA")):
        x1 = max(ix_min, uy_min - c_val)
        y1 = x1 + c_val
        x2 = min(ix_max, uy_max - c_val)
        y2 = x2 + c_val
        px1, py1 = sx(x1, x_right, ix_min, ix_max), sy(y1, y_bot, uy_min, uy_max)
        px2, py2 = sx(x2, x_right, ix_min, ix_max), sy(y2, y_bot, uy_min, uy_max)
        lines.append(f'<line class="guide" x1="{px1:.1f}" y1="{py1:.1f}" x2="{px2:.1f}" y2="{py2:.1f}"/>')
    # Label outer guides cleanly in open white space away from scatter points
    lines.append(f'<text class="guide-txt" x="{sx(1.55, x_right, ix_min, ix_max):.1f}" y="{sy(8.62, y_bot, uy_min, uy_max):.1f}">10 GHz/mA</text>')
    lines.append(f'<text class="guide-txt" x="{sx(0.90, x_right, ix_min, ix_max):.1f}" y="{sy(5.35, y_bot, uy_min, uy_max):.1f}">0.1 GHz/mA</text>')

    # Plot ALL 1,000 individual cells colored by (CL, VDD)
    class_spec = [
        ("200 fF · 1.8 V", COLORS["blue"], (cl_pf < 1.0) & (vdd_v < 2.5)),
        ("200 fF · 3.3 V", COLORS["cyan"], (cl_pf < 1.0) & (vdd_v >= 2.5)),
        ("2 pF · 1.8 V", COLORS["orange"], (cl_pf >= 1.0) & (vdd_v < 2.5)),
        ("2 pF · 3.3 V", COLORS["red"], (cl_pf >= 1.0) & (vdd_v >= 2.5)),
    ]
    log_idd = np.log10(idd_ua)
    log_ugbw = np.log10(np.where(np.isfinite(ugbw), ugbw, np.nan))
    for _, col, mask in class_spec:
        idxs = np.where(mask & np.isfinite(log_ugbw))[0]
        circles = [
            f'<circle cx="{sx(float(log_idd[k]), x_right, ix_min, ix_max):.1f}" '
            f'cy="{sy(float(log_ugbw[k]), y_bot, uy_min, uy_max):.1f}" r="2.0"/>'
            for k in idxs
        ]
        lines.append(f'<g fill="{col}" fill-opacity="0.62">' + "".join(circles) + "</g>")

    # Legend box in guaranteed-empty top-left corner of Panel 4
    bx, by = x_right + 10, y_bot + 10
    lines.append(f'<rect class="badge-bg" x="{bx:.0f}" y="{by:.0f}" width="118" height="68" rx="4"/>')
    for ci, (lbl, col, _) in enumerate(class_spec):
        cy = by + 13 + ci * 15
        lines.append(f'<circle cx="{bx + 11:.0f}" cy="{cy - 3:.0f}" r="3.2" fill="{col}" fill-opacity="0.85"/>')
        lines.append(f'<text class="legend" x="{bx + 20:.0f}" y="{cy:.0f}">{e(lbl)}</text>')

    # FoM summary box in guaranteed-empty bottom-right corner of Panel 4
    fx, fy = x_right + pw - 168, y_bot + ph - 38
    lines.append(f'<rect class="badge-bg" x="{fx:.0f}" y="{fy:.0f}" width="160" height="30" rx="4"/>')
    lines.append(f'<text class="badge-txt" x="{fx + 7:.0f}" y="{fy + 12:.0f}">IFoM med {ifom_med:.0f} MHz·pF/mA</text>')
    lines.append(f'<text class="badge-txt" x="{fx + 7:.0f}" y="{fy + 24:.0f}">12 topos · 16 PVT corners</text>')
    lines.append("</g>")

    lines.append("</svg>")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path.name} ({out_path.stat().st_size / 1024:.1f} KiB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="Optional substring filter (e.g. opa1k)")
    args = ap.parse_args()
    only = args.only.lower()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def want(name: str) -> bool:
        return (not only) or (only in name.lower())

    if only == "opa1k":
        opa_bode_npz = GOLDEN_DIR / "opa1k_bode_step_1k.npz"
        opa_manifest = REPO_ROOT / "circuits" / "opa1000" / "manifest.json"
        render_opa1k_bode_svg(
            npz_path=opa_bode_npz,
            manifest_path=opa_manifest,
            out_path=OUT_DIR / "opa1k_pvt16.svg",
            n_pvt_tasks=16000,
        )
        return

    # 1. sar12_hero_16cyc (continuous 1-cycle sinewave input + 12-bit redundant SAR output overlay!)
    d = parse_raw(GOLDEN_DIR / "sar12_hero_16cyc_baseline.raw.gz")
    t = d["time"]
    sar_w = decode_sar12_weighted(d)
    codes = sar_w["codes_15b"]
    t_per = 412.5e-9
    # Align staircase to sampling instant (falling edge of phi_s ~ 28ns into each cycle)
    code_t = np.array([c * t_per + 28e-9 for c in range(len(codes))], dtype=float)
    v_rec = sar_w["v_rec"]
    if "v(vip)" in d and "v(vin)" in d:
        vip_t, vip_v, vin_t, vin_v = t, d["v(vip)"], t, d["v(vin)"]
        vdiff_t, vdiff_v = t, d["v(vip)"] - d["v(vin)"]
    else:
        vip_t, vip_v = parse_deck_pwl(DECKS_DIR / "sar12_hero_16cyc.cir", "Vvip", float(t[-1]))
        vin_t, vin_v = parse_deck_pwl(DECKS_DIR / "sar12_hero_16cyc.cir", "Vvin", float(t[-1]))
        vdiff_t, vdiff_v = vip_t, vip_v - vin_v
    vin_at_s = np.interp(code_t, vdiff_t, vdiff_v)
    sar_corr = float(np.corrcoef(vin_at_s, v_rec)[0, 1]) if len(v_rec) > 2 else 1.0
    render_svg(
        title="sar12_hero_16cyc — 12-bit Redundant SAR ADC (16 Conversion Cycles, 151.5 kHz Sinewave)",
        subtitle=f"0 to 6.60 µs · {len(t):,} pts · {len(codes)}/16 codes · r={sar_corr:.6f}",
        t_scale=1e6,
        t_unit="µs",
        t_max_raw=float(t[-1]),
        panels=[
            {
                "title": "Differential Analog Input Sinewave v(vip) & v(vin) (V, 151.5 kHz)",
                "ylim": (0.0, 1.85),
                "traces": [
                    {"t": vip_t, "y": vip_v, "label": "v(vip)", "color": COLORS["blue"], "width": 1.8},
                    {"t": vin_t, "y": vin_v, "label": "v(vin)", "color": COLORS["purple"], "width": 1.6, "opacity": 0.9},
                ],
            },
            {
                "title": f"12-bit Redundant SAR Output vs Input (V, r={sar_corr:.6f})",
                "ylim": (-1.8, 1.8),
                "traces": [
                    {"t": vdiff_t, "y": vdiff_v, "label": "v(vip)-v(vin)", "color": COLORS["blue"], "width": 1.5, "dash": "5 3", "opacity": 0.85},
                    {"t": code_t, "y": v_rec, "label": "SAR 12b decoded", "color": COLORS["green"], "step": True, "width": 2.1},
                ],
            },
            {
                "title": "Sampling Strobe & End-of-Conversion (V)",
                "ylim": (-0.15, 2.0),
                "traces": [
                    {"t": t, "y": d["v(phi_s)"], "label": "v(phi_s)", "color": COLORS["blue"], "width": 1.4},
                    {"t": t, "y": d["v(eoc)"], "label": "v(eoc)", "color": COLORS["orange"], "width": 1.3, "opacity": 0.9},
                ],
            },
            {
                "title": "Supply Current -i(Vvdd) (mA)",
                "ylim": (-1.2, 7.2),
                "traces": [
                    {"t": t, "y": -d["i(vvdd)"] * 1e3, "label": "-i(Vvdd) [mA]", "color": COLORS["red"], "width": 1.25},
                ],
            },
        ],
        out_path=OUT_DIR / "sar12_hero_16cyc.svg",
    )

    # 2. sar12_probe_2cyc (continuous sinewave input + async bit cycling!)
    d = parse_raw(GOLDEN_DIR / "sar12_probe_2cyc_baseline.raw.gz")
    t = d["time"]
    sar2_w = decode_sar12_weighted(d)
    codes2_12b = sar2_w["codes_12b"]
    if "v(vip)" in d and "v(vin)" in d:
        vip_t, vip_v, vin_t, vin_v = t, d["v(vip)"], t, d["v(vin)"]
    else:
        vip_t, vip_v = parse_deck_pwl(DECKS_DIR / "sar12_probe_2cyc.cir", "Vvip", float(t[-1]))
        vin_t, vin_v = parse_deck_pwl(DECKS_DIR / "sar12_probe_2cyc.cir", "Vvin", float(t[-1]))
    render_svg(
        title="sar12_probe_2cyc — 12-bit Redundant SAR ADC (2 Cycles, Asynchronous Bit Cycling)",
        subtitle=f"0 to 825 ns · {len(t):,} pts · 12b codes {codes2_12b}",
        t_scale=1e9,
        t_unit="ns",
        t_max_raw=float(t[-1]),
        panels=[
            {
                "title": "Differential Analog Input Sinewave v(vip) & v(vin) (V, 151.5 kHz)",
                "ylim": (0.0, 1.85),
                "traces": [
                    {"t": vip_t, "y": vip_v, "label": "v(vip)", "color": COLORS["blue"], "width": 1.8},
                    {"t": vin_t, "y": vin_v, "label": "v(vin)", "color": COLORS["purple"], "width": 1.6, "opacity": 0.9},
                ],
            },
            {
                "title": "Sample, Async Comparator Clock & EOC (V)",
                "ylim": (-0.15, 2.0),
                "traces": [
                    {"t": t, "y": d["v(phi_s)"], "label": "v(phi_s)", "color": COLORS["blue"], "width": 1.5},
                    {"t": t, "y": d["v(comp_clk)"], "label": "v(comp_clk)", "color": COLORS["purple"], "width": 1.2, "opacity": 0.9},
                    {"t": t, "y": d["v(eoc)"], "label": "v(eoc)", "color": COLORS["orange"], "width": 1.5},
                ],
            },
            {
                "title": "Representative SAR Digital Outputs — MSB to LSB (V)",
                "ylim": (-0.15, 2.0),
                "traces": [
                    {"t": t, "y": d["v(bit_0)"], "label": "v(bit_0) MSB", "color": COLORS["green"], "width": 1.5},
                    {"t": t, "y": d["v(bit_2)"], "label": "v(bit_2)", "color": COLORS["blue"], "width": 1.3},
                    {"t": t, "y": d["v(bit_7)"], "label": "v(bit_7)", "color": COLORS["orange"], "width": 1.3},
                    {"t": t, "y": d["v(bit_14)"], "label": "v(bit_14) LSB", "color": COLORS["red"], "width": 1.25},
                ],
            },
            {
                "title": "Supply Current -i(Vvdd) (mA)",
                "ylim": (-1.2, 7.2),
                "traces": [
                    {"t": t, "y": -d["i(vvdd)"] * 1e3, "label": "-i(Vvdd) [mA]", "color": COLORS["red"], "width": 1.25},
                ],
            },
        ],
        out_path=OUT_DIR / "sar12_probe_2cyc.svg",
    )

    # 3. ti12_full_300ns
    d = parse_raw(GOLDEN_DIR / "ti12_full_300ns_baseline.raw.gz")
    t = d["time"]
    ti_t, ti_i = _mask_window(t, -d["i(vvdd)"] * 1e3, 0.5e-9)
    render_svg(
        title="ti12_full_300ns — Full-Chip 3-ch TI 12b SAR ADC + PLL (7,221 Equations)",
        subtitle=f"0 to 300 ns · {len(t):,} pts · Gear",
        t_scale=1e9,
        t_unit="ns",
        t_max_raw=float(t[-1]),
        panels=[
            {
                "title": "Differential Analog Input & Track/Hold Output (V)",
                "ylim": (0.0, 1.48),
                "traces": [
                    {"t": t, "y": d["v(vip)"], "label": "v(vip)", "color": COLORS["blue"], "width": 1.6},
                    {"t": t, "y": d["v(vin)"], "label": "v(vin)", "color": COLORS["purple"], "width": 1.6},
                    {"t": t, "y": d["v(vsh_p)"], "label": "v(vsh_p)", "color": COLORS["green"], "width": 1.7},
                ],
            },
            {
                "title": "Reference Clock fref & Internal Sample Node phi_s (V)",
                "ylim": (-0.1, 1.95),
                "traces": [
                    {"t": t, "y": d["v(fref)"], "label": "v(fref)", "color": COLORS["orange"], "width": 1.4},
                    {"t": t, "y": d["v(phi_s)"], "label": "v(phi_s)", "color": COLORS["blue"], "width": 1.5},
                ],
            },
            {
                "title": "Supply Current -i(VVDD) (mA, t >= 0.5 ns post-UIC startup)",
                "traces": [
                    {"t": ti_t, "y": ti_i, "label": "-i(VVDD) [mA]", "color": COLORS["red"], "width": 1.3},
                ],
            },
        ],
        out_path=OUT_DIR / "ti12_full_300ns.svg",
    )

    # 4. pll_ring_150m_4us (closed-loop startup from code 0 -> half full-scale code 31 + VC lock!)
    pll_raw = GOLDEN_DIR / "pll_ring_150m_4us_baseline.raw.gz"
    if pll_raw.exists():
        d = parse_raw(pll_raw)
        t = d["time"]
        pm = decode_pll_metrics(d)
        code = np.zeros(len(t), dtype=float)
        for b in range(6):
            code += (d[f"v(code_{b})"] > 0.9).astype(float) * (1 << b)
        # Mask brief t < 5ns UIC reset settle for clean code display
        ct, cy = _mask_window(t, code, 10e-9)
        ft, fy = compute_inst_freq_mhz(t, d["v(out)"])
        ft, fy = _mask_window(ft, fy, 120e-9)
        zt, z_out = _mask_window(t, d["v(out)"], 3.90e-6, 4.00e-6)
        _, z_fb = _mask_window(t, d["v(fb)"], 3.90e-6, 4.00e-6)
        _, z_ref = _mask_window(t, d["v(ref_clk)"], 3.90e-6, 4.00e-6)
        render_svg(
            title="pll_ring_150m_4us — Wide-Tuning 150 MHz Ring PLL Closed-Loop Startup (Code 0 to Half-Scale)",
            subtitle=f"0 to 4.0 µs · {len(t):,} pts · AFC code 0->{pm['code_final']} · f_lock={pm['fout_mhz']:.2f} MHz",
            t_scale=1e6,
            t_unit="µs",
            t_max_raw=float(t[-1]),
            panels=[
                {
                    "title": "6-bit SAR AFC Band Code (0..63, Dashed Line = Half Full-Scale 32)",
                    "ylim": (-3.0, 42.0),
                    "hline": 32.0,
                    "traces": [
                        {"t": ct, "y": cy, "label": f"AFC code (final={pm['code_final']})", "color": COLORS["green"], "width": 2.0},
                    ],
                },
                {
                    "title": "VCO Control Voltage v(vc) & AFC Done Flag (V)",
                    "ylim": (-0.1, 1.95),
                    "traces": [
                        {"t": t, "y": d["v(vc)"], "label": "v(vc)", "color": COLORS["blue"], "width": 1.8},
                        {"t": t, "y": d["v(afc_done)"], "label": "v(afc_done)", "color": COLORS["orange"], "width": 1.4, "opacity": 0.85},
                    ],
                },
                {
                    "title": "Instantaneous Output Frequency f_out(t) (MHz, Dashed = 152.7 MHz Target)",
                    "ylim": (60.0, 185.0),
                    "hline": 152.68,
                    "traces": [
                        {"t": ft, "y": fy, "label": "f_out [MHz]", "color": COLORS["red"], "width": 1.7},
                    ],
                },
                {
                    "title": "Zoomed Phase-Locked Clocks (3.90 to 4.00 µs): v(out) = 4x v(ref_clk) (V)",
                    "xlim": (3.90, 4.00),
                    "ylim": (-0.15, 2.0),
                    "traces": [
                        {"t": zt, "y": z_out, "label": "v(out) 152.7M", "color": COLORS["blue"], "width": 1.3, "opacity": 0.75},
                        {"t": zt, "y": z_fb, "label": "v(fb) 38.2M", "color": COLORS["green"], "width": 1.7},
                        {"t": zt, "y": z_ref, "label": "v(ref_clk) 38.2M", "color": COLORS["orange"], "width": 1.5, "opacity": 0.9},
                    ],
                },
            ],
            out_path=OUT_DIR / "pll_ring_150m_4us.svg",
        )

    # 5. sdm16_120cyc (with ideal-DAC + Sinc3 PDM recovery overlay!)
    sdm_raw = GOLDEN_DIR / "sdm16_120cyc_baseline.raw.gz"
    if sdm_raw.exists():
        d = parse_raw(sdm_raw)
        t = d["time"]
        bits = decode_sdm_bits(d)
        dac_rec = reconstruct_sdm_ideal_dac(d)
        si_t, si_i = _mask_window(t, -d["i(vvdd)"] * 1e3, 20e-9)
        pdm_sig = "v(xdut.dac_p)" if "v(xdut.dac_p)" in d else "v(xdut.pdm_bit)"
        pdm_lbl = "v(dac_p) latched PDM" if "v(xdut.dac_p)" in d else "v(pdm_bit)"
        vin_diff = d["v(vin_p)"] - d["v(vin_n)"]
        corr_str = f"r={dac_rec['corr']:.4f}, RMS={dac_rec['rms_err_mv']:.1f} mV" if dac_rec else ""
        p1_traces = [
            {"t": t, "y": vin_diff, "label": "vin_p - vin_n", "color": COLORS["blue"], "width": 1.6, "dash": "5 3"},
        ]
        if dac_rec:
            p1_traces.append(
                {"t": dac_rec["t_rec"], "y": dac_rec["v_rec"], "label": f"Ideal-DAC + Sinc3 (r={dac_rec['corr']:.4f})", "color": COLORS["green"], "width": 2.1}
            )
        render_svg(
            title="sdm16_120cyc — 16-bit Switched-Cap CIFB Sigma-Delta Modulator (120 Cycles, Ideal-DAC Verified)",
            subtitle=f"0 to 37.5 µs · {len(t):,} pts · {len(bits)} PDM bits · {corr_str}",
            t_scale=1e6,
            t_unit="µs",
            t_max_raw=float(t[-1]),
            panels=[
                {
                    "title": "Input vs Ideal-DAC + Sinc3 Recovered PDM (V, 26.7 kHz)",
                    "ylim": (-1.05, 1.05),
                    "traces": p1_traces,
                },
                {
                    "title": "1st & 2nd Switched-Cap Integrator Outputs with SC-CMFB (V)",
                    "ylim": (0.0, 1.95),
                    "traces": [
                        {"t": t, "y": d["v(xdut.vout1_p)"], "label": "v(vout1_p)", "color": COLORS["blue"], "width": 1.35},
                        {"t": t, "y": d["v(xdut.vout2_p)"], "label": "v(vout2_p)", "color": COLORS["green"], "width": 1.35},
                    ],
                },
                {
                    "title": "Noise-Shaped 1-bit PDM Output Bitstream (V, 120 Clock Cycles)",
                    "ylim": (-0.15, 2.0),
                    "traces": [
                        {"t": t, "y": d[pdm_sig], "label": pdm_lbl, "color": COLORS["orange"], "width": 1.2},
                    ],
                },
                {
                    "title": "Supply Current -i(Vvdd) (mA)",
                    "traces": [
                        {"t": si_t, "y": si_i, "label": "-i(Vvdd) [mA]", "color": COLORS["red"], "width": 1.15},
                    ],
                },
            ],
            out_path=OUT_DIR / "sdm16_120cyc.svg",
        )

    # 6. opa1k_pvt16 (1,000 distinct GF180 post-PEX op-amps x 16 PVT corners = 16,000 scattered tasks!)
    opa_bode_npz = GOLDEN_DIR / "opa1k_bode_step_1k.npz"
    opa_manifest = REPO_ROOT / "circuits" / "opa1000" / "manifest.json"
    if want("opa1k_pvt16") and opa_bode_npz.exists() and opa_manifest.exists():
        render_opa1k_bode_svg(
            npz_path=opa_bode_npz,
            manifest_path=opa_manifest,
            out_path=OUT_DIR / "opa1k_pvt16.svg",
            n_pvt_tasks=16000,
        )
    elif want("opa1k_pvt16") and (GOLDEN_DIR / "opa1k_pvt16_baseline.raw.gz").exists():
        opa_raw = GOLDEN_DIR / "opa1k_pvt16_baseline.raw.gz"
        d = parse_raw(opa_raw)
        n_des = 1000
        n_cor = len(d["idx_v"]) // n_des
        idx_x = np.arange(n_des, dtype=float)
        # Reshape (16 corners, 1000 designs); corner 4 is (27C, 0.95*VDD, 0.9*Ib), corner 5 is (27C, 0.95, 1.1), etc.
        # Compute nominal-T (27C, corners 4..7) mean plus full 16-corner min/max envelopes across all 1,000 designs!
        idd_mat = np.maximum(d["idd_dc"].reshape(n_cor, n_des), 1e-12) * 1e6  # uA
        vout_mat = d["vout_dc"].reshape(n_cor, n_des)
        acdc_mat = d["ac_dc"].reshape(n_cor, n_des)
        acpk_mat = d["ac_pk"].reshape(n_cor, n_des)
        dv_mat = (d["tr_v1"] - d["tr_v0"]).reshape(n_cor, n_des) * 1e3  # mV
        pk_mat = (d["tr_pk"] - d["tr_v0"]).reshape(n_cor, n_des) * 1e3  # mV

        idd_med = np.log10(np.median(idd_mat, axis=0))
        idd_min = np.log10(np.min(idd_mat, axis=0))
        idd_max = np.log10(np.max(idd_mat, axis=0))
        vout_med = np.median(vout_mat, axis=0)
        vout_min = np.min(vout_mat, axis=0)
        vout_max = np.max(vout_mat, axis=0)
        acdc_med = np.median(acdc_mat, axis=0)
        acpk_med = np.clip(np.median(acpk_mat, axis=0), 0.5, 3.0)
        dv_med = np.clip(np.median(dv_mat, axis=0), 0.0, 20.0)
        pk_med = np.clip(np.median(pk_mat, axis=0), 0.0, 25.0)

        render_svg(
            title="opa1k_pvt16 — 1,000 Post-PEX Op-Amps × 16 PVT Corners (16,000 Scattered OP+AC+Tran Tasks)",
            subtitle=f"Design Index 0..999 (12 Topologies) · {n_cor * n_des:,} tasks · T=[-40,27,85,125]°C · ±5% VDD · ±10% Ibias",
            t_scale=1.0,
            t_unit="",
            t_max_raw=float(n_des - 1),
            panels=[
                {
                    "title": "DC Supply Current log10(I_DD / µA) Across 1,000 Designs (16-Corner PVT Envelope)",
                    "ylim": (-2.5, 3.2),
                    "traces": [
                        {"t": idx_x, "y": idd_max, "label": "max PVT", "color": COLORS["orange"], "width": 1.0, "opacity": 0.65},
                        {"t": idx_x, "y": idd_min, "label": "min PVT", "color": COLORS["purple"], "width": 1.0, "opacity": 0.65},
                        {"t": idx_x, "y": idd_med, "label": "median log10(I_DD/µA)", "color": COLORS["blue"], "width": 1.35},
                    ],
                },
                {
                    "title": "DC Output Bias Point v(out) Across 1,000 Designs (1.8 V & 3.3 V, 16-Corner PVT)",
                    "ylim": (0.2, 2.6),
                    "traces": [
                        {"t": idx_x, "y": vout_max, "label": "+5% VDD", "color": COLORS["orange"], "width": 1.0, "opacity": 0.65},
                        {"t": idx_x, "y": vout_min, "label": "-5% VDD", "color": COLORS["purple"], "width": 1.0, "opacity": 0.65},
                        {"t": idx_x, "y": vout_med, "label": "median v(out) [V]", "color": COLORS["green"], "width": 1.3},
                    ],
                },
                {
                    "title": "Closed-Loop AC DC-Gain |H_cl(1 Hz)| & Peak |H_cl(f)| (Median Across 16 PVT Corners)",
                    "ylim": (0.85, 2.65),
                    "hline": 1.0,
                    "traces": [
                        {"t": idx_x, "y": acpk_med, "label": "max|H_cl(f)| peaking", "color": COLORS["red"], "width": 1.15, "opacity": 0.85},
                        {"t": idx_x, "y": acdc_med, "label": "|H_cl(1 Hz)| DC gain", "color": COLORS["blue"], "width": 1.4},
                    ],
                },
                {
                    "title": "Closed-Loop 10 mV Step Settling & Peak Overshoot (mV, Dashed = 10 mV Target)",
                    "ylim": (4.0, 22.0),
                    "hline": 10.0,
                    "traces": [
                        {"t": idx_x, "y": pk_med, "label": "step peak (mV)", "color": COLORS["orange"], "width": 1.15, "opacity": 0.85},
                        {"t": idx_x, "y": dv_med, "label": "settled step (mV)", "color": COLORS["green"], "width": 1.35},
                    ],
                },
            ],
            out_path=OUT_DIR / "opa1k_pvt16.svg",
        )


if __name__ == "__main__":
    main()
