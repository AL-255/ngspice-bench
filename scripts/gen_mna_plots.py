#!/usr/bin/env python3
"""Generate warm light-theme MNA matrix sparsity SVG plots and print exact device counts."""
from __future__ import annotations

import base64
import html
import os
import re
import struct
import subprocess
import tempfile
import zlib
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DECKS_DIR = REPO_ROOT / "decks"
CIRCUITS_DIR = REPO_ROOT / "circuits"
PDK_DIR = REPO_ROOT / "pdk"
OUT_DIR = REPO_ROOT / "docs" / "img"
NGSPICE_BIN = Path(
    os.environ.get(
        "NGSPICE_BIN",
        "/Users/yukidama/openmpw/ngspice-opt/build-opt-apple-pgo/src/ngspice",
    )
)

# Warm stylesheet palette matching style.css and gen_waveform_plots.py
BG_CANVAS = "#fffaf6"
BG_PLOT = "#ffffff"
BORDER_COL = "#f0dccd"
TEXT_COL = "#2b1a17"
MUTED_COL = "#7a5c52"
COL_ORANGE = (0xFF, 0x7F, 0x0E)   # #ff7f0e
COL_BURNT = (0xC2, 0x41, 0x0C)    # #c2410c
COL_BRAND = (0xD9, 0x42, 0x3A)    # #d9423a
COL_DEEP = (0x7F, 0x1D, 0x1D)     # #7f1d1d
COL_BLUE = "#2f7fd1"


def encode_png_rgb(rgb: np.ndarray) -> bytes:
    """Encode an HxWx3 uint8 array to a lossless PNG byte string using stdlib zlib."""
    h, w, c = rgb.shape
    assert c == 3 and rgb.dtype == np.uint8

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack("!I", len(data))
            + tag
            + data
            + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
    ihdr = struct.pack("!IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, level=9))
        + chunk(b"IEND", b"")
    )


def _lerp_rgb(c0: Tuple[int, int, int], c1: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    return (
        int(round(c0[0] + (c1[0] - c0[0]) * t)),
        int(round(c0[1] + (c1[1] - c0[1]) * t)),
        int(round(c0[2] + (c1[2] - c0[2]) * t)),
    )


def density_to_rgb(t: float) -> Tuple[int, int, int]:
    """Map normalized log-density t in [0, 1] to warm orange -> burnt orange -> deep crimson."""
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return _lerp_rgb(COL_ORANGE, COL_BURNT, t * 2.0)
    return _lerp_rgb(COL_BURNT, COL_DEEP, (t - 0.5) * 2.0)


def extract_deck_mna_and_devices(deck_name: str) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, Counter]:
    """Run ngspice with KLU mdump + listing expand to get exact unfactored MNA matrix & device counts."""
    txt = (DECKS_DIR / f"{deck_name}.cir").read_text()
    txt = txt.replace("../pdk/", str(PDK_DIR.resolve()) + "/")
    txt = txt.replace("../circuits/", str(CIRCUITS_DIR.resolve()) + "/")

    lines: List[str] = []
    for l in txt.splitlines():
        ls = l.strip().lower()
        if ls.startswith("option ") and "sparse" in ls:
            continue
        if ls.startswith("tran "):
            lines.append("option klu")
            lines.append("tran 1p 1p uic")
            lines.append("mdump")
            lines.append("listing expand > expanded.txt")
        elif ls.startswith("write ") or ls.startswith("rusage"):
            continue
        else:
            lines.append(l)

    with tempfile.TemporaryDirectory(prefix="ngmna_") as tmp:
        wd = Path(tmp)
        (wd / ".spiceinit").write_text("set ngbehavior=hsa\nset num_threads=1\n")
        (wd / "run.cir").write_text("\n".join(lines) + "\n")
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        r = subprocess.run(
            [str(NGSPICE_BIN), "-b", str(wd / "run.cir"), "-o", str(wd / "out.log")],
            cwd=str(wd),
            env=env,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"ngspice failed for {deck_name}: {r.stderr}")

        rows_l: List[int] = []
        cols_l: List[int] = []
        vals_l: List[float] = []
        pat = re.compile(r"^Row:\s*(\d+)\s+Col:\s*(\d+)\s+Value:\s*([+-]?[0-9.eE+-]+)")
        for line in (wd / "out.log").read_text(errors="replace").splitlines():
            m = pat.match(line.strip())
            if m:
                rows_l.append(int(m.group(1)))
                cols_l.append(int(m.group(2)))
                vals_l.append(float(m.group(3)))

        cnt: Counter = Counter()
        for line in (wd / "expanded.txt").read_text(errors="replace").splitlines():
            m = re.match(r"^\s*\d+\s*:\s*([a-zA-Z])", line)
            if m:
                cnt[m.group(1).lower()] += 1

    rows = np.asarray(rows_l, dtype=np.int32)
    cols = np.asarray(cols_l, dtype=np.int32)
    vals = np.asarray(vals_l, dtype=np.float64)
    n = int(max(rows.max(), cols.max())) if len(rows) else 0
    return n, rows, cols, vals, cnt


def extract_opa_mna_and_stats() -> Tuple[str, int, np.ndarray, np.ndarray, np.ndarray, Counter, Dict[str, int]]:
    """Inspect all 1,000 opa1000 circuits and dump exact MNA of median-size post-PEX op-amp."""
    opa_files = sorted((CIRCUITS_DIR / "opa1000").glob("p_*.cir"))
    # First find median by raw element count (excluding top-level Xopamp_dut wrapper)
    sizes: List[Tuple[int, Path]] = []
    for p in opa_files:
        n_el = 0
        for line in p.read_text().splitlines():
            ls = line.strip()
            if not ls or ls.startswith("*") or ls.startswith("."):
                continue
            if ls.lower().startswith("xopamp") or ls.lower().startswith("xdut"):
                continue
            n_el += 1
        sizes.append((n_el, p))
    sizes.sort(key=lambda x: x[0])
    min_p = sizes[0][1]
    med_p = sizes[len(sizes) // 2][1]
    max_p = sizes[-1][1]

    def _run_opa_one(p: Path) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, Counter]:
        with tempfile.TemporaryDirectory(prefix="ngmna_opa_") as tmp:
            wd = Path(tmp)
            (wd / ".spiceinit").write_text("set ngbehavior=hsa\nset num_threads=1\n")
            pdk_inc = (DECKS_DIR / "pdk_opa.inc").read_text().replace("../pdk/", str(PDK_DIR.resolve()) + "/")
            (wd / "pdk_opa.inc").write_text(pdk_inc)
            cir_txt = p.read_text()
            cir_txt = re.sub(r"(?im)^\s*\.end\s*$", "", cir_txt)
            cir_txt += (
                "\n.control\n"
                "option klu\n"
                "op\n"
                "mdump\n"
                "listing expand > expanded.txt\n"
                "quit\n"
                ".endc\n"
                ".end\n"
            )
            (wd / "run.cir").write_text(cir_txt)
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            r = subprocess.run(
                [str(NGSPICE_BIN), "-b", str(wd / "run.cir"), "-o", str(wd / "out.log")],
                cwd=str(wd),
                env=env,
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                raise RuntimeError(f"ngspice failed for {p.name}: {r.stderr}")
            rows_l, cols_l, vals_l = [], [], []
            pat = re.compile(r"^Row:\s*(\d+)\s+Col:\s*(\d+)\s+Value:\s*([+-]?[0-9.eE+-]+)")
            for line in (wd / "out.log").read_text(errors="replace").splitlines():
                m = pat.match(line.strip())
                if m:
                    rows_l.append(int(m.group(1)))
                    cols_l.append(int(m.group(2)))
                    vals_l.append(float(m.group(3)))
            cnt: Counter = Counter()
            for line in (wd / "expanded.txt").read_text(errors="replace").splitlines():
                m = re.match(r"^\s*\d+\s*:\s*([a-zA-Z])", line)
                if m:
                    cnt[m.group(1).lower()] += 1
        rows = np.asarray(rows_l, dtype=np.int32)
        cols = np.asarray(cols_l, dtype=np.int32)
        vals = np.asarray(vals_l, dtype=np.float64)
        n = int(max(rows.max(), cols.max())) if len(rows) else 0
        return n, rows, cols, vals, cnt

    n_min, _, _, _, c_min = _run_opa_one(min_p)
    n_med, r_med, c_med, v_med, cnt_med = _run_opa_one(med_p)
    n_max, _, _, _, c_max = _run_opa_one(max_p)

    # Extract design title from first comment line of med_p
    hdr = med_p.read_text().splitlines()[0].lstrip("* ").strip()
    stats = {
        "min_devs": sum(c_min.values()),
        "med_devs": sum(cnt_med.values()),
        "max_devs": sum(c_max.values()),
        "min_n": n_min,
        "med_n": n_med,
        "max_n": n_max,
        "total_1k_approx": sum(s for s, _ in sizes),
    }
    return hdr, n_med, r_med, c_med, v_med, cnt_med, stats


def fmt_dev_breakdown(cnt: Counter) -> str:
    order = [("m", "MOS"), ("r", "R"), ("c", "C"), ("b", "B"), ("e", "E"), ("v", "V"), ("i", "I"), ("d", "D")]
    parts = [f"{cnt[k]:,} {lbl}" for k, lbl in order if cnt.get(k, 0) > 0]
    other = sum(v for k, v in cnt.items() if k not in {x[0] for x in order})
    if other:
        parts.append(f"{other:,} other")
    return " · ".join(parts)


def write_mna_svg(
    out_path: Path,
    title: str,
    subtitle_extra: str,
    n: int,
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
    dev_cnt: Counter,
    sparse_fill: int | None = None,
    fill_label: str = "Sp fill",
) -> None:
    nnz = len(rows)
    density_pct = (100.0 * nnz / (n * n)) if n > 0 else 0.0
    total_devs = sum(dev_cnt.values())

    W, H = 460, 466
    px0, py0, psize = 76, 86, 300

    # Tick positions
    tick_vals = [1, max(1, n // 4), max(1, n // 2), max(1, (3 * n) // 4), n]
    ticks_svg: List[str] = []
    for tv in tick_vals:
        frac = (tv - 1) / max(n - 1, 1)
        x = px0 + frac * psize
        y = py0 + frac * psize
        lbl = f"{tv:,}"
        # Top ticks
        ticks_svg.append(f'<line x1="{x:.1f}" y1="{py0 - 4}" x2="{x:.1f}" y2="{py0}" stroke="{MUTED_COL}" stroke-width="1"/>')
        ticks_svg.append(f'<text x="{x:.1f}" y="{py0 - 8}" text-anchor="middle" font-size="10" font-family="ui-monospace,Menlo,monospace" fill="{MUTED_COL}">{lbl}</text>')
        # Left ticks
        ticks_svg.append(f'<line x1="{px0 - 4}" y1="{y:.1f}" x2="{px0}" y2="{y:.1f}" stroke="{MUTED_COL}" stroke-width="1"/>')
        ticks_svg.append(f'<text x="{px0 - 8}" y="{y + 3.5:.1f}" text-anchor="end" font-size="10" font-family="ui-monospace,Menlo,monospace" fill="{MUTED_COL}">{lbl}</text>')
        # Subtle interior gridlines for interior ticks
        if tv not in (1, n):
            ticks_svg.append(f'<line x1="{x:.1f}" y1="{py0}" x2="{x:.1f}" y2="{py0 + psize}" stroke="{BORDER_COL}" stroke-width="0.7" stroke-dasharray="2,3"/>')
            ticks_svg.append(f'<line x1="{px0}" y1="{y:.1f}" x2="{px0 + psize}" y2="{y:.1f}" stroke="{BORDER_COL}" stroke-width="0.7" stroke-dasharray="2,3"/>')

    if n <= 200:
        # Exact vector rendering for small op-amp MNA matrix!
        cell = psize / n
        rsize = max(1.6, cell * 0.92)
        rects: List[str] = []
        n_branch = max(1, dev_cnt.get("v", 0))
        for r, c, v in zip(rows, cols, vals):
            x = px0 + (c - 1) * cell
            y = py0 + (r - 1) * cell
            if r == c:
                fill = "#991b1b"
            elif abs(abs(v) - 1.0) < 1e-12 and (r > n - n_branch or c > n - n_branch):
                fill = COL_BLUE
            else:
                fill = "#ff7f0e"
            rects.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{rsize:.2f}" height="{rsize:.2f}" fill="{fill}"/>')
        matrix_body = "\n    ".join(rects)
        legend_svg = (
            f'<rect x="{px0}" y="{py0 + psize + 16}" width="10" height="10" rx="2" fill="#991b1b"/>'
            f'<text x="{px0 + 15}" y="{py0 + psize + 25}" font-size="10.5" fill="{MUTED_COL}">Diagonal</text>'
            f'<rect x="{px0 + 82}" y="{py0 + psize + 16}" width="10" height="10" rx="2" fill="#ff7f0e"/>'
            f'<text x="{px0 + 97}" y="{py0 + psize + 25}" font-size="10.5" fill="{MUTED_COL}">Off-diag (G + sC)</text>'
            f'<rect x="{px0 + 206}" y="{py0 + psize + 16}" width="10" height="10" rx="2" fill="{COL_BLUE}"/>'
            f'<text x="{px0 + 221}" y="{py0 + psize + 25}" font-size="10.5" fill="{MUTED_COL}">V-src branch (±1)</text>'
        )
    else:
        # Crisp binned lossless PNG spy plot embedded in SVG
        G = 280
        grid = np.zeros((G, G), dtype=np.int32)
        ri = np.clip(((rows - 1) * G) // n, 0, G - 1)
        ci = np.clip(((cols - 1) * G) // n, 0, G - 1)
        np.add.at(grid, (ri, ci), 1)
        max_bin = int(grid.max()) if grid.size else 1

        rgb = np.full((G, G, 3), (0xFF, 0xFF, 0xFF), dtype=np.uint8)
        nz_y, nz_x = np.nonzero(grid)
        log_denom = np.log1p(max_bin) if max_bin > 1 else 1.0
        for y, x in zip(nz_y, nz_x):
            t = float(np.log1p(grid[y, x]) / log_denom) if max_bin > 1 else 0.35
            rgb[y, x] = density_to_rgb(t)

        b64 = base64.b64encode(encode_png_rgb(rgb)).decode("ascii")
        matrix_body = (
            f'<image x="{px0}" y="{py0}" width="{psize}" height="{psize}" '
            f'image-rendering="pixelated" preserveAspectRatio="none" '
            f'href="data:image/png;base64,{b64}"/>'
        )
        legend_svg = (
            f'<defs><linearGradient id="mnaGrad" x1="0%" y1="0%" x2="100%" y2="0%">'
            f'<stop offset="0%" stop-color="#ff7f0e"/>'
            f'<stop offset="50%" stop-color="#c2410c"/>'
            f'<stop offset="100%" stop-color="#7f1d1d"/>'
            f'</linearGradient></defs>'
            f'<text x="{px0}" y="{py0 + psize + 24}" font-size="10.5" fill="{MUTED_COL}">Bin NNZ density (log):</text>'
            f'<text x="{px0 + 132}" y="{py0 + psize + 24}" font-size="10" font-family="ui-monospace,Menlo,monospace" fill="{MUTED_COL}">1</text>'
            f'<rect x="{px0 + 144}" y="{py0 + psize + 15}" width="108" height="10" rx="3" fill="url(#mnaGrad)" stroke="{BORDER_COL}"/>'
            f'<text x="{px0 + 258}" y="{py0 + psize + 24}" font-size="10" font-family="ui-monospace,Menlo,monospace" fill="{MUTED_COL}">{max_bin:,}/px</text>'
        )

    fill_str = f" · {sparse_fill:,} {fill_label}" if sparse_fill is not None else ""
    sub1 = f"{n:,} × {n:,} MNA · {nnz:,} orig NNZ ({density_pct:.2f}%){fill_str}"
    sub2 = f"{total_devs:,} devs ({fmt_dev_breakdown(dev_cnt)})"
    if subtitle_extra:
        sub2 = f"{subtitle_extra}: {sub2}"

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="{html.escape(title)} MNA matrix sparsity">
  <rect width="{W}" height="{H}" rx="10" fill="{BG_CANVAS}"/>
  <text x="20" y="27" font-size="14" font-weight="700" font-family="Inter,system-ui,-apple-system,sans-serif" fill="{TEXT_COL}">{html.escape(title)}</text>
  <text x="20" y="45" font-size="10.5" font-family="ui-monospace,Menlo,monospace" fill="{MUTED_COL}">{html.escape(sub1)}</text>

  <!-- Matrix plot area -->
  <rect x="{px0}" y="{py0}" width="{psize}" height="{psize}" fill="{BG_PLOT}" stroke="{BORDER_COL}" stroke-width="1.2"/>
  {matrix_body}
  <g>
    {"".join(ticks_svg)}
  </g>
  <rect x="{px0}" y="{py0}" width="{psize}" height="{psize}" fill="none" stroke="{BORDER_COL}" stroke-width="1.2"/>

  <!-- Axis captions -->
  <text x="{px0 + psize / 2:.1f}" y="{py0 - 24}" text-anchor="middle" font-size="10.5" font-weight="600" font-family="Inter,system-ui,sans-serif" fill="{MUTED_COL}">Column index j (MNA variables)</text>
  <text x="20" y="{py0 + psize / 2:.1f}" text-anchor="middle" font-size="10.5" font-weight="600" font-family="Inter,system-ui,sans-serif" fill="{MUTED_COL}" transform="rotate(-90 20 {py0 + psize / 2:.1f})">Row index i (KCL + branch KVL)</text>

  <!-- Legend & device footer -->
  {legend_svg}
  <text x="{W / 2:.1f}" y="{H - 16}" text-anchor="middle" font-size="9.5" font-family="ui-monospace,Menlo,monospace" fill="{TEXT_COL}">{html.escape(sub2)}</text>
</svg>
'''
    out_path.write_text(svg)
    print(f"Wrote {out_path.name} ({out_path.stat().st_size / 1024:.1f} KiB) | {sub1} | {sub2}")


def main() -> None:
    import sys
    only = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    specs = [
        ("sar12_probe_2cyc", "sar12_mna.svg", "MNA Matrix — sar12 (12b Redundant SAR ADC)", 12459, "Sp fill"),
        ("ti12_full_300ns", "ti12_full_300ns_mna.svg", "MNA Matrix — ti12_full_300ns (3-ch TI SAR + PLL)", 30954, "Sp fill"),
        ("pll_ring_150m_4us", "pll_ring_150m_4us_mna.svg", "MNA Matrix — pll_ring_150m_4us (150 MHz Ring PLL)", 2498, "Sp fill"),
        ("sdm16_120cyc", "sdm16_120cyc_mna.svg", "MNA Matrix — sdm16_120cyc (16b SC-CIFB SDM)", 30530, "Sp fill"),
        ("riscv_picorv32_120ns", "riscv_picorv32_120ns_mna.svg", "MNA Matrix — riscv_picorv32_120ns (GF180 PicoRV32)", 2174667, "KLU fill"),
    ]
    for deck, fname, title, fill_cnt, flbl in specs:
        if only and only not in deck.lower():
            continue
        n, r, c, v, cnt = extract_deck_mna_and_devices(deck)
        write_mna_svg(OUT_DIR / fname, title, "", n, r, c, v, cnt, sparse_fill=fill_cnt, fill_label=flbl)

    if not only or "opa1k" in only:
        hdr, n_med, r_med, c_med, v_med, cnt_med, stats = extract_opa_mna_and_stats()
        print("opa1k_pvt16 stats:", stats, "median header:", hdr, "median cnt:", dict(cnt_med))
        write_mna_svg(
            OUT_DIR / "opa1k_pvt16_mna.svg",
            "MNA Matrix — opa1k_pvt16 (Median Op-Amp: p_0221)",
            "Median op-amp",
            n_med,
            r_med,
            c_med,
            v_med,
            cnt_med,
            sparse_fill=None,
        )


if __name__ == "__main__":
    main()
