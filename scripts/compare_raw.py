#!/usr/bin/env python3
"""Compare candidate ngspice raw waveforms against golden baseline (.raw or .raw.gz)."""
from __future__ import annotations

import gzip
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _read_raw_bytes(path: Path) -> bytes:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as fh:
            return fh.read()
    return path.read_bytes()


def parse_raw(path: Path) -> Dict[str, np.ndarray]:
    data = _read_raw_bytes(path)
    m = re.search(rb"\nBinary:\r?\n", data)
    if m is None:
        raise ValueError(f"No Binary section in {path}")
    hdr = data[: m.start()].decode("latin-1")
    body = data[m.end() :]
    nvars = int(re.search(r"No\. Variables:\s*(\d+)", hdr).group(1))
    npts = int(re.search(r"No\. Points:\s*(\d+)", hdr).group(1))
    vars_ = re.findall(r"^\s*\d+\s+(\S+)\s+", hdr, re.M)
    arr = np.frombuffer(body[: nvars * npts * 8], dtype=np.float64).reshape(npts, nvars)
    return {vars_[i].lower(): arr[:, i] for i in range(nvars)}


SAR12_REDUNDANT_WEIGHTS = [
    2048.009, 1024.035, 511.978, 512.022, 255.98, 127.95,
    63.999, 63.973, 31.987, 15.994, 7.998, 7.996, 3.999, 1.999, 1.0,
]


def decode_sar12_codes(d: Dict[str, np.ndarray], vdd: float = 1.8, t_per: float = 412.5e-9) -> List[int]:
    """Extract 15-bit redundant SAR bit patterns at the end of each conversion cycle."""
    if "v(bit_0)" not in d or "v(bit_14)" not in d:
        return []
    t = d["time"]
    n_cyc = int(round(float(t[-1]) / t_per))
    codes: List[int] = []
    vth = 0.5 * vdd
    for c in range(n_cyc):
        t_sample = (c + 0.98) * t_per
        idx = int(np.argmin(np.abs(t - t_sample)))
        code = 0
        for b in range(15):
            if float(d[f"v(bit_{b})"][idx]) > vth:
                code |= 1 << (14 - b)
        codes.append(code)
    return codes


def decode_sar12_weighted(
    d: Dict[str, np.ndarray], vdd: float = 1.8, t_per: float = 412.5e-9, fs_diff: float = 1.725
) -> Dict[str, Any]:
    """Decode 15-bit redundant SAR outputs into calibrated 12-bit codes and differential voltages."""
    raw_bits = decode_sar12_codes(d, vdd=vdd, t_per=t_per)
    w_sum = sum(SAR12_REDUNDANT_WEIGHTS)
    codes_12b: List[float] = []
    v_rec: List[float] = []
    for code in raw_bits:
        w_val = sum(
            w for b, w in enumerate(SAR12_REDUNDANT_WEIGHTS) if (code >> (14 - b)) & 1
        )
        codes_12b.append(round(w_val * 4095.0 / w_sum, 1))
        v_rec.append((2.0 * w_val / w_sum - 1.0) * fs_diff)
    return {"codes_15b": raw_bits, "codes_12b": codes_12b, "v_rec": np.asarray(v_rec, dtype=float)}


def decode_sdm_bits(d: Dict[str, np.ndarray], vdd: float = 1.8) -> List[int]:
    """Extract settled post-reset PDM DAC output bits during phi2 of each clock cycle."""
    sig = "v(xdut.dac_p)" if "v(xdut.dac_p)" in d else ("v(xdut.pdm_bit)" if "v(xdut.pdm_bit)" in d else None)
    if sig is None or "v(clk)" not in d:
        return []
    t = d["time"]
    ts = 312.5e-9
    n_cyc = int(round(float(t[-1]) / ts))
    vth = 0.5 * vdd
    bits: List[int] = []
    for k in range(5, n_cyc):
        t_sample = (k + 0.85) * ts
        if t_sample > float(t[-1]):
            break
        idx = int(np.argmin(np.abs(t - t_sample)))
        bits.append(1 if float(d[sig][idx]) > vth else 0)
    return bits


def reconstruct_sdm_ideal_dac(d: Dict[str, np.ndarray], vdd: float = 1.8) -> Dict[str, Any]:
    """Recover the analog input from the 1-bit PDM bitstream via an ideal ±VDD DAC + zero-phase Sinc3 LPF."""
    bits = decode_sdm_bits(d, vdd=vdd)
    if len(bits) < 30 or "v(vin_p)" not in d or "v(vin_n)" not in d:
        return {}
    t = d["time"]
    ts = 312.5e-9
    times = np.array([(k + 0.85) * ts for k in range(5, 5 + len(bits))], dtype=float)
    dac_v = np.array([vdd if b else -vdd for b in bits], dtype=float)
    box = np.ones(9, dtype=float) / 9.0
    h = np.convolve(np.convolve(box, box), box)  # 25-tap zero-phase Sinc3 FIR, unity DC gain
    rec = np.convolve(dac_v, h, mode="same")
    vin_diff = np.interp(times, t, d["v(vin_p)"] - d["v(vin_n)"])
    pad = 13
    best_corr = -1.0
    best_lag = 0
    best_rms_mv = float("inf")
    for lag in range(5):
        a_r = rec[pad + lag : len(rec) - pad]
        a_v = vin_diff[pad : len(rec) - pad - lag]
        if len(a_r) < 10:
            continue
        c = float(np.corrcoef(a_r, a_v)[0, 1])
        if c > best_corr:
            best_corr = c
            best_lag = lag
            best_rms_mv = float(np.sqrt(np.mean((a_r - a_v) ** 2)) * 1e3)
    # Time-aligned valid interior window for plotting
    t_valid = times[pad : len(rec) - pad - best_lag]
    v_valid = rec[pad + best_lag : len(rec) - pad]
    return {
        "t_rec": t_valid,
        "v_rec": v_valid,
        "corr": round(best_corr, 5),
        "lag_cyc": best_lag,
        "rms_err_mv": round(best_rms_mv, 2),
    }


def decode_pll_metrics(d: Dict[str, np.ndarray], vdd: float = 1.8) -> Dict[str, Any]:
    """Extract closed-loop PLL startup metrics: 6-bit SAR AFC code, AFC done time, locked VC & f_out."""
    if "v(afc_done)" not in d or "v(code_5)" not in d or "v(vc)" not in d:
        return {}
    t = d["time"]
    vth = 0.5 * vdd
    code = np.zeros(len(t), dtype=int)
    for b in range(6):
        code += (d[f"v(code_{b})"] > vth).astype(int) << b
    m_after = t > 0.5e-6
    afc_idx = np.where(d["v(afc_done)"][m_after] > vth)[0]
    t_afc = float(t[m_after][afc_idx[0]]) if len(afc_idx) else -1.0
    m_vc = t >= float(t[-1]) - 0.4e-6
    vc_final = float(np.mean(d["v(vc)"][m_vc]))
    m_f = t >= float(t[-1]) - 0.8e-6
    tw, vw = t[m_f], d["v(out)"][m_f]
    up = np.where((vw[:-1] < vth) & (vw[1:] >= vth))[0]
    if len(up) >= 3:
        crs = [float(tw[i] + (vth - vw[i]) / (vw[i + 1] - vw[i]) * (tw[i + 1] - tw[i])) for i in up]
        fout_mhz = (len(crs) - 1) / (crs[-1] - crs[0]) / 1e6
    else:
        fout_mhz = 0.0
    return {
        "code_final": int(code[-1]),
        "t_afc_s": t_afc,
        "vc_final_v": vc_final,
        "fout_mhz": fout_mhz,
    }


def _rising_edges(d: Dict[str, np.ndarray], sig: str, vth: float = 0.9) -> List[float]:
    tt, vv = d["time"], d[sig]
    idx = np.where((vv[:-1] < vth) & (vv[1:] >= vth))[0]
    return [float(tt[i] + (vth - vv[i]) / (vv[i + 1] - vv[i]) * (tt[i + 1] - tt[i])) for i in idx]


def compare_waveforms(ref_path: Path, cand_path: Path, tol_rms: float = 1e-3) -> Dict[str, Any]:
    ref = parse_raw(ref_path)
    cand = parse_raw(cand_path)
    t_ref = ref["time"]
    t_cand = cand["time"]
    t_min = max(float(t_ref[0]), float(t_cand[0]))
    t_max = min(float(t_ref[-1]), float(t_cand[-1]))
    t_cov = abs(float(t_cand[-1]) - float(t_ref[-1])) / max(abs(float(t_ref[-1])), 1e-30)
    exact_grid = (len(t_ref) == len(t_cand)) and bool(np.allclose(t_ref, t_cand, rtol=1e-12, atol=1e-20))
    t_eval = t_ref if exact_grid else np.linspace(t_min, t_max, min(8000, max(len(t_ref), 1000)))

    per_vec: Dict[str, Dict[str, float]] = {}
    worst_rel_rms = 0.0
    worst_vec = ""
    for k in ref:
        if k == "time" or k not in cand:
            continue
        vr = ref[k] if exact_grid else np.interp(t_eval, t_ref, ref[k])
        vc = cand[k] if exact_grid else np.interp(t_eval, t_cand, cand[k])
        if not np.all(np.isfinite(vc)):
            return {
                "pass": False,
                "error": f"Non-finite values in {k}",
                "worst_rel_rms": float("inf"),
                "exact_grid": exact_grid,
            }
        diff = np.abs(vc - vr)
        max_abs = float(np.max(diff))
        rms = float(np.sqrt(np.mean(diff * diff)))
        scale = max(float(np.max(np.abs(vr))), 1e-9)
        rel_rms = rms / scale
        per_vec[k] = {"max_abs": max_abs, "rms": rms, "rel_rms": rel_rms}
        if rel_rms > worst_rel_rms:
            worst_rel_rms = rel_rms
            worst_vec = k

    ref_codes = decode_sar12_codes(ref)
    cand_codes = decode_sar12_codes(cand)
    codes_match = (ref_codes == cand_codes) if ref_codes else True

    ref_sdm = decode_sdm_bits(ref)
    cand_sdm = decode_sdm_bits(cand)
    sdm_match = (ref_sdm == cand_sdm) if ref_sdm else True
    sdm_pdm_max_dt_ps = 0.0
    sdm_dac_corr = None

    ref_pll = decode_pll_metrics(ref)
    cand_pll = decode_pll_metrics(cand)
    pll_code_match = True
    pll_fout_rel_err = 0.0
    pll_vc_err_mv = 0.0

    eoc_max_dt_ps = 0.0
    if ref_codes:
        er = _rising_edges(ref, "v(eoc)")
        ec = _rising_edges(cand, "v(eoc)")
        if len(er) != len(ec):
            eoc_max_dt_ps = float("inf")
        elif er:
            eoc_max_dt_ps = max(abs(a - b) * 1e12 for a, b in zip(er, ec))
        t_post = np.linspace(max(t_min, 5e-9), t_max, min(8000, max(len(t_ref), 1000)))
        post_worst = 0.0
        for k in ref:
            if k == "time" or k not in cand:
                continue
            vr = np.interp(t_post, t_ref, ref[k])
            vc = np.interp(t_post, t_cand, cand[k])
            rms = float(np.sqrt(np.mean((vc - vr) ** 2)))
            scale = max(float(np.max(np.abs(vr))), 1e-9)
            post_worst = max(post_worst, rms / scale)
        worst_rel_rms = post_worst
        passed = (t_cov < 1e-6) and codes_match and (eoc_max_dt_ps < 500.0) and (post_worst < 0.05)
    elif ref_pll:
        pll_code_match = (cand_pll.get("code_final") == ref_pll.get("code_final"))
        t_afc_diff_ns = abs(cand_pll.get("t_afc_s", -1.0) - ref_pll.get("t_afc_s", -1.0)) * 1e9
        pll_vc_err_mv = abs(cand_pll.get("vc_final_v", 0.0) - ref_pll.get("vc_final_v", 0.0)) * 1e3
        f_ref_m = max(ref_pll.get("fout_mhz", 0.0), 1e-6)
        pll_fout_rel_err = abs(cand_pll.get("fout_mhz", 0.0) - f_ref_m) / f_ref_m
        vc_rms = per_vec.get("v(vc)", {}).get("rel_rms", float("inf"))
        worst_rel_rms = vc_rms
        worst_vec = "v(vc)"
        passed = (
            (t_cov < 1e-6)
            and pll_code_match
            and (t_afc_diff_ns < 20.0)
            and (pll_vc_err_mv < 10.0)
            and (pll_fout_rel_err < 0.002)
            and (vc_rms <= 0.02)
        )
    elif ref_sdm:
        edge_sig = "v(xdut.dac_p)" if "v(xdut.dac_p)" in ref else "v(xdut.pdm_bit)"
        pr = _rising_edges(ref, edge_sig)
        pc = _rising_edges(cand, edge_sig)
        if len(pr) != len(pc):
            sdm_pdm_max_dt_ps = float("inf")
        elif pr:
            sdm_pdm_max_dt_ps = max(abs(a - b) * 1e12 for a, b in zip(pr, pc))
        switching_nodes = {
            "v(xdut.pdm_bit)", "v(xdut.dac_p)", "v(xdut.dac_n)",
            "v(xdut.phi1)", "v(xdut.phi2)", "v(dout_0)", "v(dout_valid)",
            "v(clk)", "i(vvdd)",
        }
        analog_worst = max(
            (v["rel_rms"] for k, v in per_vec.items() if k not in switching_nodes),
            default=0.0,
        )
        worst_rel_rms = analog_worst
        worst_vec = max(
            ((v["rel_rms"], k) for k, v in per_vec.items() if k not in switching_nodes),
            default=(0.0, ""),
        )[1]
        sdm_dac = reconstruct_sdm_ideal_dac(cand)
        sdm_dac_corr = sdm_dac.get("corr", 0.0) if sdm_dac else 0.0
        passed = (
            (t_cov < 1e-6)
            and sdm_match
            and (sdm_pdm_max_dt_ps < 500.0)
            and (analog_worst <= 0.01)
            and (sdm_dac_corr >= 0.99)
        )
    else:
        sdm_dac_corr = None
        passed = (t_cov < 1e-6) and (worst_rel_rms <= tol_rms) and codes_match

    return {
        "pass": passed,
        "exact_grid": exact_grid,
        "npts_ref": int(len(t_ref)),
        "npts_cand": int(len(t_cand)),
        "t_end_rel_diff": t_cov,
        "worst_rel_rms": worst_rel_rms,
        "worst_vec": worst_vec,
        "sar_codes_ref": ref_codes,
        "sar_codes_cand": cand_codes,
        "sar_codes_match": codes_match,
        "eoc_max_dt_ps": eoc_max_dt_ps,
        "sdm_bits_count": len(ref_sdm),
        "sdm_bits_match": sdm_match,
        "sdm_pdm_max_dt_ps": sdm_pdm_max_dt_ps,
        "sdm_dac_corr": sdm_dac_corr if ref_sdm else None,
        "pll_code_ref": ref_pll.get("code_final") if ref_pll else None,
        "pll_code_cand": cand_pll.get("code_final") if cand_pll else None,
        "pll_code_match": pll_code_match,
        "pll_fout_mhz": round(cand_pll.get("fout_mhz", 0.0), 3) if cand_pll else None,
        "pll_vc_final_v": round(cand_pll.get("vc_final_v", 0.0), 4) if cand_pll else None,
        "pll_vc_err_mv": round(pll_vc_err_mv, 3) if ref_pll else None,
        "per_vec": per_vec,
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: compare_raw.py <ref.raw[.gz]> <cand.raw[.gz]> [tol_rms]")
        sys.exit(2)
    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3
    res = compare_waveforms(Path(sys.argv[1]), Path(sys.argv[2]), tol_rms=tol)
    print(
        f"Compare {Path(sys.argv[2]).name} vs {Path(sys.argv[1]).name}: "
        f"{'PASS' if res['pass'] else 'FAIL'} | exact_grid={res['exact_grid']} | "
        f"npts={res['npts_cand']}/{res['npts_ref']} | worst_rel_rms={res['worst_rel_rms']:.3e} ({res['worst_vec']})"
    )
    if res.get("sar_codes_ref"):
        print(f"  SAR codes match={res['sar_codes_match']}: ref={res['sar_codes_ref']} cand={res['sar_codes_cand']}")
    if res.get("pll_code_ref") is not None:
        print(f"  PLL code={res['pll_code_cand']}/{res['pll_code_ref']} fout={res['pll_fout_mhz']} MHz vc={res['pll_vc_final_v']} V")
    sys.exit(0 if res["pass"] else 1)
