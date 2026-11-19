#!/usr/bin/env python3
"""Generate warm light-theme SVG transient waveform plots aligned with aicdb/hub-gemini4 palette."""
from __future__ import annotations

import html
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
from compare_raw import decode_pll_metrics, decode_sar12_codes, decode_sdm_bits, parse_raw  # noqa: E402

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
    header_h = 46
    panel_h = 120
    panel_gap = 38
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
        "  .title { fill: #2b1a17; font: 700 13.5px Inter,system-ui,-apple-system,'Segoe UI',sans-serif; }",
        "  .sub { fill: #7a5c52; font: 11px ui-monospace,SFMono-Regular,Menlo,monospace; }",
        "  .ptitle { fill: #2b1a17; font: 600 11.5px Inter,system-ui,-apple-system,'Segoe UI',sans-serif; }",
        "  .tick { fill: #7a5c52; font: 10.5px ui-monospace,SFMono-Regular,Menlo,monospace; }",
        "  .legend { fill: #2b1a17; font: 11px ui-monospace,SFMono-Regular,Menlo,monospace; }",
        "</style>",
        f'<rect class="bg" x="0" y="0" width="{W}" height="{H}" rx="8"/>',
        f'<text class="title" x="{margin_l}" y="22">{e(title)}</text>',
        f'<text class="sub" x="{W - margin_r}" y="22" text-anchor="end">{e(subtitle)}</text>',
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
                lines.append(
                    f'<text class="tick" x="{gx:.1f}" y="{y0 + panel_h + 15}" text-anchor="middle">{fmt_num(tval)} {t_unit}</text>'
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
            lines.append(
                f'<polyline fill="none" stroke="{tr["color"]}" stroke-width="{sw}" stroke-opacity="{op}" '
                f'stroke-linejoin="round" points="{" ".join(pts)}"/>'
            )
        lines.append("</g>")

    lines.append("</svg>")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path.name} ({out_path.stat().st_size / 1024:.1f} KiB)")


def _mask_window(t: np.ndarray, y: np.ndarray, t_min: float, t_max: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    m = t >= t_min if t_max is None else ((t >= t_min) & (t <= t_max))
    return t[m], y[m]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. sar12_hero_16cyc (with differential analog input v(vip)/v(vin)!)
    d = parse_raw(GOLDEN_DIR / "sar12_hero_16cyc_baseline.raw.gz")
    t = d["time"]
    codes = decode_sar12_codes(d)
    t_per = 412.5e-9
    code_t = np.array([(c + 0.98) * t_per for c in range(len(codes))], dtype=float)
    code_y = np.array(codes, dtype=float)
    vip_t, vip_v = parse_deck_pwl(DECKS_DIR / "sar12_hero_16cyc.cir", "Vvip", float(t[-1]))
    vin_t, vin_v = parse_deck_pwl(DECKS_DIR / "sar12_hero_16cyc.cir", "Vvin", float(t[-1]))
    render_svg(
        title="sar12_hero_16cyc — 12-bit Redundant SAR ADC (16 Conversion Cycles, Golden Stock ngspice-47)",
        subtitle=f"0 to 6.60 µs · {len(t):,} pts · {len(codes)}/16 codes",
        t_scale=1e6,
        t_unit="µs",
        t_max_raw=float(t[-1]),
        panels=[
            {
                "title": "Differential Analog Input Signal v(vip) & v(vin) (V)",
                "ylim": (0.0, 1.85),
                "traces": [
                    {"t": vip_t, "y": vip_v, "label": "v(vip)", "color": COLORS["blue"], "width": 1.8},
                    {"t": vin_t, "y": vin_v, "label": "v(vin)", "color": COLORS["purple"], "width": 1.6, "opacity": 0.9},
                ],
            },
            {
                "title": "Decoded 15-bit Redundant SAR Output Code (Sampled at EOC)",
                "ylim": (-1500, 34500),
                "traces": [
                    {"t": code_t, "y": code_y, "label": "SAR code (0..32767)", "color": COLORS["green"], "step": True, "width": 2.1},
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

    # 2. sar12_probe_2cyc (with differential analog input v(vip)/v(vin)!)
    d = parse_raw(GOLDEN_DIR / "sar12_probe_2cyc_baseline.raw.gz")
    t = d["time"]
    codes2 = decode_sar12_codes(d)
    vip_t, vip_v = parse_deck_pwl(DECKS_DIR / "sar12_probe_2cyc.cir", "Vvip", float(t[-1]))
    vin_t, vin_v = parse_deck_pwl(DECKS_DIR / "sar12_probe_2cyc.cir", "Vvin", float(t[-1]))
    render_svg(
        title="sar12_probe_2cyc — 12-bit Redundant SAR ADC (2 Cycles, Asynchronous Bit Cycling)",
        subtitle=f"0 to 825 ns · {len(t):,} pts · codes {codes2}",
        t_scale=1e9,
        t_unit="ns",
        t_max_raw=float(t[-1]),
        panels=[
            {
                "title": "Differential Analog Input Signal v(vip) & v(vin) (V)",
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

    # 5. sdm16_120cyc
    sdm_raw = GOLDEN_DIR / "sdm16_120cyc_baseline.raw.gz"
    if sdm_raw.exists():
        d = parse_raw(sdm_raw)
        t = d["time"]
        bits = decode_sdm_bits(d)
        si_t, si_i = _mask_window(t, -d["i(vvdd)"] * 1e3, 20e-9)
        pdm_sig = "v(xdut.dac_p)" if "v(xdut.dac_p)" in d else "v(xdut.pdm_bit)"
        pdm_lbl = "v(dac_p) latched PDM" if "v(xdut.dac_p)" in d else "v(pdm_bit)"
        render_svg(
            title="sdm16_120cyc — 16-bit Switched-Cap CIFB Sigma-Delta Modulator (120 Cycles, In-Band Sine)",
            subtitle=f"0 to 37.5 µs · {len(t):,} pts · {len(bits)} post-reset PDM bits",
            t_scale=1e6,
            t_unit="µs",
            t_max_raw=float(t[-1]),
            panels=[
                {
                    "title": "Differential In-Band Sinusoidal Input v(vin_p) & v(vin_n) (±0.15 V @ 26.7 kHz)",
                    "ylim": (0.65, 1.15),
                    "traces": [
                        {"t": t, "y": d["v(vin_p)"], "label": "v(vin_p)", "color": COLORS["blue"], "width": 1.8},
                        {"t": t, "y": d["v(vin_n)"], "label": "v(vin_n)", "color": COLORS["purple"], "width": 1.6, "opacity": 0.9},
                    ],
                },
                {
                    "title": "1st & 2nd Switched-Cap Integrator Outputs (V)",
                    "ylim": (-0.15, 2.0),
                    "traces": [
                        {"t": t, "y": d["v(xdut.vout1_p)"], "label": "v(vout1_p)", "color": COLORS["blue"], "width": 1.35},
                        {"t": t, "y": d["v(xdut.vout2_p)"], "label": "v(vout2_p)", "color": COLORS["green"], "width": 1.35},
                    ],
                },
                {
                    "title": "Noise-Shaped PDM Output Bitstream (V, 120 Clock Cycles)",
                    "ylim": (-0.15, 2.0),
                    "traces": [
                        {"t": t, "y": d[pdm_sig], "label": pdm_lbl, "color": COLORS["orange"], "width": 1.2},
                    ],
                },
                {
                    "title": "Supply Current -i(Vvdd) (mA, t >= 20 ns post-UIC startup)",
                    "traces": [
                        {"t": si_t, "y": si_i, "label": "-i(Vvdd) [mA]", "color": COLORS["red"], "width": 1.15},
                    ],
                },
            ],
            out_path=OUT_DIR / "sdm16_120cyc.svg",
        )


if __name__ == "__main__":
    main()
