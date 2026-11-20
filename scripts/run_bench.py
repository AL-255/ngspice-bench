#!/usr/bin/env python3
"""
ngspice-bench: Single-Core Benchmark Runner & Core-Class Profiler.

Runs the standard GF180 post-PEX transient benchmark suite on a single pinned
CPU core (num_threads=1, OMP_NUM_THREADS=1), automatically detects homogeneous
vs hybrid/big.LITTLE core classes (Intel P/E/LP-E, AMD Zen/Zenc, ARM big.LITTLE,
Apple Silicon P/E), verifies every waveform against stock ngspice-47 golden
references, and archives a structured JSON submission under results/.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DECKS_DIR = REPO_ROOT / "decks"
GOLDEN_DIR = REPO_ROOT / "golden"
RESULTS_DIR = REPO_ROOT / "results"
PDK_DIR = REPO_ROOT / "pdk" / "gf180mcu" / "gf180mcuD"

sys.path.insert(0, str(SCRIPT_DIR))
from compare_raw import compare_waveforms  # noqa: E402

PROBE_DECKS = [
    "opa1k_pvt16",
    "sdm16_120cyc",
    "pll_ring_150m_4us",
    "ti12_full_300ns",
    "sar12_probe_2cyc",
]
HERO_DECKS = ["sar12_hero_16cyc"]
ALL_DECKS = PROBE_DECKS + HERO_DECKS

PDK_DIRECTIVE_PAT = re.compile(
    r'^\s*\.(include|lib)\s+"?\.\./pdk/gf180mcu/gf180mcuD/[^"\n]+"?.*$',
    re.MULTILINE,
)


def _read_text_safe(p: Path) -> str:
    try:
        return p.read_text().strip()
    except OSError:
        return ""


def _parse_cpulist(spec: str) -> List[int]:
    cpus: List[int] = []
    for part in spec.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.extend(range(int(lo), int(hi) + 1))
        else:
            cpus.append(int(part))
    return sorted(set(cpus))


def detect_core_classes() -> List[Dict[str, Any]]:
    """Detect distinct CPU core classes (homogeneous vs Intel hybrid / ARM big.LITTLE / AMD hybrid)."""
    sys_name = platform.system()
    if sys_name == "Linux":
        # 1. Intel Hybrid PMU masks (/sys/devices/cpu_core/cpus, /sys/devices/cpu_atom/cpus, cpu_lowpower)
        intel_map = [
            ("P-core", Path("/sys/devices/cpu_core/cpus")),
            ("E-core", Path("/sys/devices/cpu_atom/cpus")),
            ("LP-E-core", Path("/sys/devices/cpu_lowpower/cpus")),
        ]
        found_intel: List[Tuple[str, List[int]]] = []
        for label, p in intel_map:
            if p.exists():
                ids = _parse_cpulist(_read_text_safe(p))
                if ids:
                    found_intel.append((label, ids))
        if len(found_intel) >= 2:
            out: List[Dict[str, Any]] = []
            for label, ids in found_intel:
                c0 = ids[0]
                max_khz = _read_text_safe(Path(f"/sys/devices/system/cpu/cpu{c0}/cpufreq/cpuinfo_max_freq"))
                l2 = _read_text_safe(Path(f"/sys/devices/system/cpu/cpu{c0}/cache/index2/size"))
                out.append({
                    "class_id": label.lower().replace("-", "_"),
                    "label": label,
                    "hybrid": True,
                    "cpus": ids,
                    "pin_cpu": c0,
                    "max_freq_mhz": round(int(max_khz) / 1000.0, 1) if max_khz.isdigit() else None,
                    "l2_cache": l2 or None,
                })
            return out

        # 2. General Linux detection via (cpu_capacity, cpuinfo_max_freq bucket, L2 size, ARM CPU part)
        cpu_dirs = sorted(
            [p for p in Path("/sys/devices/system/cpu").glob("cpu[0-9]*") if p.name[3:].isdigit()],
            key=lambda p: int(p.name[3:]),
        )
        # Parse /proc/cpuinfo per-processor CPU part (ARM) if available
        part_by_cpu: Dict[int, str] = {}
        proc_txt = _read_text_safe(Path("/proc/cpuinfo"))
        cur_id: Optional[int] = None
        for line in proc_txt.splitlines():
            if ":" not in line:
                continue
            k, v = [x.strip() for x in line.split(":", 1)]
            if k == "processor" and v.isdigit():
                cur_id = int(v)
            elif k == "CPU part" and cur_id is not None:
                part_by_cpu[cur_id] = v

        groups: Dict[Tuple[Any, ...], List[int]] = {}
        meta_by_sig: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        for d in cpu_dirs:
            cid = int(d.name[3:])
            online = _read_text_safe(d / "online")
            if online == "0":
                continue
            cap_s = _read_text_safe(d / "cpu_capacity")
            cap = int(cap_s) if cap_s.isdigit() else None
            freq_s = _read_text_safe(d / "cpufreq/cpuinfo_max_freq")
            freq_mhz = round(int(freq_s) / 1000.0, 1) if freq_s.isdigit() else None
            # Bucket frequency to 100 MHz to ignore minor per-core turbo bin differences (e.g. AMD preferred core)
            freq_bucket = round(freq_mhz / 250.0) * 250 if freq_mhz is not None else None
            l2 = _read_text_safe(d / "cache/index2/size") or None
            arm_part = part_by_cpu.get(cid)
            sig = (cap, arm_part, l2, freq_bucket if (cap is not None or arm_part is not None) else None)
            groups.setdefault(sig, []).append(cid)
            if sig not in meta_by_sig or (freq_mhz or 0) > (meta_by_sig[sig].get("max_freq_mhz") or 0):
                meta_by_sig[sig] = {
                    "capacity": cap,
                    "arm_part": arm_part,
                    "l2_cache": l2,
                    "max_freq_mhz": freq_mhz,
                }

        if len(groups) >= 2:
            # Sort descending by (capacity, max_freq_mhz) -> Prime / Big / Little
            ordered = sorted(
                groups.items(),
                key=lambda kv: (
                    meta_by_sig[kv[0]].get("capacity") or 0,
                    meta_by_sig[kv[0]].get("max_freq_mhz") or 0.0,
                ),
                reverse=True,
            )
            labels = (
                ["Big", "Little"]
                if len(ordered) == 2
                else (["Prime", "Big", "Little"] if len(ordered) == 3 else [f"Class-{i+1}" for i in range(len(ordered))])
            )
            out = []
            for idx, (sig, ids) in enumerate(ordered):
                m = meta_by_sig[sig]
                lbl = labels[idx]
                out.append({
                    "class_id": lbl.lower(),
                    "label": lbl,
                    "hybrid": True,
                    "cpus": ids,
                    "pin_cpu": ids[0],
                    "max_freq_mhz": m.get("max_freq_mhz"),
                    "l2_cache": m.get("l2_cache"),
                    "capacity": m.get("capacity"),
                    "arm_part": m.get("arm_part"),
                })
            return out

        all_ids = [cid for ids in groups.values() for cid in ids] or [0]
        # Pick CPU 0 (or highest max_freq core on homogeneous CPUs)
        best_cpu = all_ids[0]
        best_mhz = None
        for cid in all_ids:
            fs = _read_text_safe(Path(f"/sys/devices/system/cpu/cpu{cid}/cpufreq/cpuinfo_max_freq"))
            if fs.isdigit():
                mhz = round(int(fs) / 1000.0, 1)
                if best_mhz is None or mhz > best_mhz:
                    best_mhz = mhz
                    best_cpu = cid
        l2 = _read_text_safe(Path(f"/sys/devices/system/cpu/cpu{best_cpu}/cache/index2/size")) or None
        return [{
            "class_id": "homogeneous",
            "label": "Homogeneous",
            "hybrid": False,
            "cpus": all_ids,
            "pin_cpu": best_cpu,
            "max_freq_mhz": best_mhz,
            "l2_cache": l2,
        }]

    if sys_name == "Darwin":
        # Apple Silicon core-tier detection via sysctl hw.nperflevels
        def _sysctl(key: str) -> str:
            r = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True)
            return r.stdout.strip() if r.returncode == 0 else ""

        nlevels_s = _sysctl("hw.nperflevels")
        if nlevels_s.isdigit() and int(nlevels_s) >= 2:
            nlevels = int(nlevels_s)
            out = []
            higher_cores = 0
            for lv in range(nlevels):
                name = _sysctl(f"hw.perflevel{lv}.name") or ("P-core" if lv == 0 else "E-core")
                ncores = _sysctl(f"hw.perflevel{lv}.physicalcpu")
                l1d = _sysctl(f"hw.perflevel{lv}.l1dcachesize")
                l2 = _sysctl(f"hw.perflevel{lv}.l2cachesize")
                if lv == 0:
                    qos = "default"
                elif lv == nlevels - 1:
                    qos = "background"
                else:
                    # On 3-tier Apple Silicon (Super / Performance / Efficiency), hold the
                    # higher-tier cores with zero-memory register spinners at default QoS
                    # while running the benchmark under taskpolicy -c utility.
                    qos = "utility_pinned"
                out.append({
                    "class_id": name.lower().replace(" ", "_"),
                    "label": "P-core" if "performance" in name.lower() else ("E-core" if "efficiency" in name.lower() else name),
                    "hybrid": True,
                    "cpus": list(range(int(ncores))) if ncores.isdigit() else [0],
                    "pin_cpu": None,
                    "macos_qos": qos,
                    "higher_tier_cores": higher_cores if qos == "utility_pinned" else 0,
                    "l1d_cache": f"{int(l1d)//1024}K" if l1d.isdigit() else None,
                    "l2_cache": f"{int(l2)//1024}K" if l2.isdigit() else None,
                })
                if ncores.isdigit():
                    higher_cores += int(ncores)
            return out

    return [{
        "class_id": "homogeneous",
        "label": "Homogeneous",
        "hybrid": False,
        "cpus": [0],
        "pin_cpu": 0 if hasattr(os, "sched_setaffinity") else None,
    }]


def detect_host_info(
    core_classes: List[Dict[str, Any]],
    platform_override: str = "",
    memory_override: str = "",
    storage_override: str = "",
) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()}",
        "arch": platform.machine(),
        "platform": platform_override or None,
        "machine_model": None,
        "cpu_model": platform.processor() or "Unknown CPU",
        "logical_cpus": os.cpu_count() or 1,
        "physical_cores": None,
        "l1d_cache": None,
        "l2_cache": None,
        "l3_cache": None,
        "ram_gb": None,
        "memory": memory_override or None,
        "storage": storage_override or None,
        "isa_features": [],
        "is_hybrid": len(core_classes) > 1,
        "detected_core_classes": [c["label"] for c in core_classes],
    }

    if platform.system() == "Linux":
        vendor = _read_text_safe(Path("/sys/devices/virtual/dmi/id/sys_vendor"))
        prod = _read_text_safe(Path("/sys/devices/virtual/dmi/id/product_name"))
        if prod:
            info["machine_model"] = f"{vendor} {prod}".strip() if vendor and vendor.lower() not in prod.lower() else prod
            if not info["platform"]:
                info["platform"] = info["machine_model"]
        cpuinfo = _read_text_safe(Path("/proc/cpuinfo"))
        m = re.search(r"^model name\s*:\s*(.+)$", cpuinfo, re.M)
        if m:
            info["cpu_model"] = re.sub(r"\s+", " ", m.group(1)).strip()
        fm = re.search(r"^(?:flags|Features)\s*:\s*(.+)$", cpuinfo, re.M)
        if fm:
            raw_flags = set(fm.group(1).split())
            interesting = [
                "avx512f", "avx512dq", "avx512vl", "avx512bw", "avx2", "fma",
                "sse4_2", "asimd", "neon", "sve", "sve2", "fp16",
            ]
            info["isa_features"] = [f for f in interesting if f in raw_flags]
        # Physical cores from lscpu if available
        if shutil.which("lscpu"):
            r = subprocess.run(["lscpu"], capture_output=True, text=True)
            if r.returncode == 0:
                cps = re.search(r"Core\(s\) per socket:\s*(\d+)", r.stdout)
                skt = re.search(r"Socket\(s\):\s*(\d+)", r.stdout)
                if cps and skt:
                    info["physical_cores"] = int(cps.group(1)) * int(skt.group(1))
        for idx, key in [(0, "l1d_cache"), (2, "l2_cache"), (3, "l3_cache")]:
            sz = _read_text_safe(Path(f"/sys/devices/system/cpu/cpu0/cache/index{idx}/size"))
            if sz:
                info[key] = sz
        meminfo = _read_text_safe(Path("/proc/meminfo"))
        mm = re.search(r"^MemTotal:\s*(\d+)\s*kB", meminfo, re.M)
        if mm:
            info["ram_gb"] = round(int(mm.group(1)) / (1024 * 1024), 1)
    elif platform.system() == "Darwin":
        def _sys(k: str) -> str:
            rr = subprocess.run(["sysctl", "-n", k], capture_output=True, text=True)
            return rr.stdout.strip() if rr.returncode == 0 else ""
        brand = _sys("machdep.cpu.brand_string")
        if brand:
            info["cpu_model"] = brand
        hw_model = _sys("hw.model")
        if hw_model:
            info["machine_model"] = hw_model
        sw_ver = subprocess.run(["sw_vers", "-productVersion"], capture_output=True, text=True)
        if sw_ver.returncode == 0 and sw_ver.stdout.strip():
            info["os"] = f"macOS {sw_ver.stdout.strip()} (Darwin {platform.release()})"
        if not info["platform"]:
            # Best-effort hardware model detection via system_profiler (falls back to hw.model)
            model_name, model_num = "", ""
            try:
                sp = subprocess.run(["system_profiler", "SPHardwareDataType"], capture_output=True, text=True, timeout=5)
                if sp.returncode == 0:
                    mn = re.search(r"Model Name:\s*(.+)", sp.stdout)
                    mnum = re.search(r"Model Number:\s*(.+)", sp.stdout)
                    if mn and mn.group(1).strip() and mn.group(1).strip().lower() != "unknown":
                        model_name = mn.group(1).strip()
                    if mnum and mnum.group(1).strip() and mnum.group(1).strip().lower() != "unknown":
                        model_num = mnum.group(1).strip()
            except Exception:
                pass
            parts = [p for p in [hw_model, model_num] if p]
            if model_name and parts:
                info["platform"] = f"{model_name} ({', '.join(parts)})"
            elif model_name:
                info["platform"] = model_name
            elif parts:
                info["platform"] = ", ".join(parts)
        pcpu = _sys("hw.physicalcpu")
        if pcpu.isdigit():
            info["physical_cores"] = int(pcpu)
        l1d = _sys("hw.perflevel0.l1dcachesize") or _sys("hw.l1dcachesize")
        if l1d.isdigit():
            info["l1d_cache"] = f"{int(l1d)//1024}K"
        l2 = _sys("hw.perflevel0.l2cachesize") or _sys("hw.l2cachesize")
        if l2.isdigit():
            info["l2_cache"] = f"{int(l2)//1024}K"
        mem = _sys("hw.memsize")
        if mem.isdigit():
            info["ram_gb"] = round(int(mem) / (1024**3), 1)
        if not info["memory"] or not info["storage"]:
            try:
                sp_ms = subprocess.run(
                    ["system_profiler", "SPMemoryDataType", "SPNVMeDataType"],
                    capture_output=True, text=True, timeout=8,
                )
                if sp_ms.returncode == 0:
                    txt = sp_ms.stdout
                    if not info["memory"]:
                        msz = re.search(r"Memory:\s*(\d+\s*[GM]B)", txt)
                        mtype = re.search(r"Type:\s*([^\n]+)", txt)
                        mmfg = re.search(r"Manufacturer:\s*([^\n]+)", txt)
                        mparts = [
                            x.group(1).strip() for x in (msz, mtype) if x and x.group(1).strip().lower() != "unknown"
                        ]
                        if mparts:
                            s = " ".join(mparts)
                            if mmfg and mmfg.group(1).strip().lower() not in ("unknown", ""):
                                s += f" ({mmfg.group(1).strip()})"
                            info["memory"] = s
                    if not info["storage"]:
                        smod = re.search(r"Model:\s*([^\n]+)", txt)
                        scap = re.search(r"Capacity:\s*([0-9.]+\s*[TG]B)", txt)
                        if smod and smod.group(1).strip().lower() != "unknown":
                            s = smod.group(1).strip()
                            if scap:
                                s += f" ({scap.group(1).strip()} NVMe)"
                            info["storage"] = s
            except Exception:
                pass
        feats = []
        for fname, skey in [
            ("neon", "hw.optional.AdvSIMD"),
            ("fp16", "hw.optional.arm.FEAT_FP16"),
            ("bf16", "hw.optional.arm.FEAT_BF16"),
            ("sme", "hw.optional.arm.FEAT_SME"),
            ("sme2", "hw.optional.arm.FEAT_SME2"),
        ]:
            if _sys(skey) == "1":
                feats.append(fname)
        info["isa_features"] = feats

    return info


def detect_ngspice_version(ngspice_bin: Path) -> str:
    try:
        r = subprocess.run([str(ngspice_bin), "-v"], capture_output=True, text=True, timeout=10)
        txt = (r.stdout + "\n" + r.stderr).strip()
        m = re.search(r"ngspice-\d+[^\n]*", txt)
        if m:
            return m.group(0).strip()
        return txt.splitlines()[0].strip() if txt else "unknown"
    except Exception:
        return "unknown"


def parse_rusage(log_text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    patterns = {
        "total_elapsed_s": r"Total elapsed time \(seconds\) =\s*([0-9.eE+-]+)",
        "total_analysis_s": r"Total analysis time \(seconds\) =\s*([0-9.eE+-]+)",
        "tran_analysis_s": r"Transient analysis time =\s*([0-9.eE+-]+)",
        "tran_load_s": r"Transient load time =\s*([0-9.eE+-]+)",
        "tran_factor_s": r"Transient factor time =\s*([0-9.eE+-]+)",
        "tran_solve_s": r"Transient solve time =\s*([0-9.eE+-]+)",
        "tran_trunc_s": r"Transient trunc time =\s*([0-9.eE+-]+)",
        "reorder_s": r"Matrix reorder time =\s*([0-9.eE+-]+)",
        "tot_iter": r"Total iterations =\s*(\d+)",
        "tran_iter": r"Transient iterations =\s*(\d+)",
        "tran_pts": r"Transient timepoints =\s*(\d+)",
        "accepted_pts": r"Accepted timepoints =\s*(\d+)",
        "rejected_pts": r"Rejected timepoints =\s*(\d+)",
        "eqns": r"Circuit Equations =\s*(\d+)",
    }
    for k, pat in patterns.items():
        m = re.search(pat, log_text)
        if m:
            out[k] = int(m.group(1)) if k.endswith(("iter", "pts", "eqns")) else float(m.group(1))
    return out


def precompile_pdk(ngspice_bin: Path, out_nglib: Path, pin_cpu: Optional[int]) -> float:
    pdk_deck = DECKS_DIR / "pdk_gf180_typ.spice"
    cmd = [str(ngspice_bin), "--precompile", str(out_nglib), str(pdk_deck)]
    if pin_cpu is not None and shutil.which("taskset"):
        cmd = ["taskset", "-c", str(pin_cpu)] + cmd
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=str(DECKS_DIR), capture_output=True, text=True)
    dt_s = time.perf_counter() - t0
    if r.returncode != 0 or not out_nglib.exists():
        raise RuntimeError(f"--precompile failed (rc={r.returncode}): {r.stderr or r.stdout}")
    return round(dt_s, 4)


def _ensure_reg_spinner() -> Optional[Path]:
    """Build a zero-memory serial register spinner used to hold higher-tier cores on 3-tier Apple Silicon."""
    spin_bin = Path(tempfile.gettempdir()) / "ngbench_reg_spin"
    if spin_bin.exists():
        return spin_bin
    cc = shutil.which("cc") or shutil.which("clang")
    if not cc:
        return None
    src = Path(tempfile.gettempdir()) / "ngbench_reg_spin.c"
    src.write_text(
        "#include <stdint.h>\n"
        "int main(void) {\n"
        "    uint64_t x = 1;\n"
        "    for (;;) { __asm__ volatile(\"add %0, %0, #1\" : \"+r\"(x)); }\n"
        "    return 0;\n"
        "}\n"
    )
    r = subprocess.run([cc, "-O2", str(src), "-o", str(spin_bin)], capture_output=True)
    return spin_bin if r.returncode == 0 and spin_bin.exists() else None


def run_one(
    ngspice_bin: Path,
    deck_name: str,
    solver: str,
    work_dir: Path,
    tol_rms: float,
    pin_cpu: Optional[int],
    macos_qos: Optional[str],
    nglib_path: Optional[Path],
    higher_tier_cores: int = 0,
) -> Dict[str, Any]:
    src_cir = DECKS_DIR / f"{deck_name}.cir"
    text = src_cir.read_text()

    # Resolve relative ../pdk and ../circuits paths cleanly against DECKS_DIR so work_dir can be anywhere
    text = text.replace("../pdk/", str((REPO_ROOT / "pdk").resolve()) + "/")
    text = text.replace("../circuits/", str((REPO_ROOT / "circuits").resolve()) + "/")

    is_opa_bundle = (deck_name == "opa1k_pvt16")
    if is_opa_bundle:
        pdk_dir = (REPO_ROOT / "pdk/gf180mcu/gf180mcuD/libs.tech/ngspice").resolve()
        if nglib_path is not None:
            inc_lines = [f'.uselib "{nglib_path.resolve()}"']
        else:
            inc_lines = [
                f'.include "{pdk_dir}/design.ngspice"',
                ".param sw_stat_global=0",
                ".param sw_stat_mismatch=0",
                f'.lib "{pdk_dir}/sm141064.ngspice" typical',
                f'.lib "{pdk_dir}/sm141064.ngspice" res_typical',
                f'.lib "{pdk_dir}/sm141064.ngspice" mimcap_typical',
                f'.lib "{pdk_dir}/sm141064.ngspice" cap_mim',
                f'.lib "{pdk_dir}/sm141064.ngspice" moscap_typical',
            ]
        if solver == "klu":
            inc_lines.append(".option klu")
        (work_dir / "pdk_opa.inc").write_text("\n".join(inc_lines) + "\n")
    elif nglib_path is not None:
        # Replace all GF180 PDK .include/.lib lines with a single .uselib directive
        pdk_abs_pat = re.compile(
            r'^\s*\.(include|lib)\s+"?' + re.escape(str((REPO_ROOT / "pdk").resolve())) + r'/[^"\n]+"?.*$',
            re.MULTILINE,
        )
        replaced = False
        new_lines = []
        for line in text.splitlines():
            if pdk_abs_pat.match(line):
                if not replaced:
                    new_lines.append(f'.uselib "{nglib_path.resolve()}"')
                    replaced = True
            else:
                new_lines.append(line)
        text = "\n".join(new_lines) + "\n"

    inject = ["set num_threads=1"]
    if solver == "klu":
        inject.append("option klu")

    raw_out = work_dir / f"{deck_name}_{solver}_1t.raw"
    log_out = work_dir / f"{deck_name}_{solver}_1t.log"
    cir_out = work_dir / f"{deck_name}_{solver}_1t.cir"
    if not is_opa_bundle:
        text = re.sub(r"^tran ", "\n".join(inject) + "\ntran ", text, count=1, flags=re.M)
    text = re.sub(r"^write \S+", f"write {raw_out}", text, count=1, flags=re.M)
    cir_out.write_text(text)

    (work_dir / ".spiceinit").write_text(
        "set ngbehavior=hsa\nset filetype=binary\nset num_threads=1\n"
    )

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OMP_PROC_BIND"] = "true"

    cmd = [str(ngspice_bin), "-b", str(cir_out), "-o", str(log_out)]
    spinners: List[subprocess.Popen] = []
    if pin_cpu is not None and platform.system() == "Linux" and shutil.which("taskset"):
        cmd = ["taskset", "-c", str(pin_cpu)] + cmd
    elif macos_qos == "background" and platform.system() == "Darwin" and shutil.which("taskpolicy"):
        cmd = ["taskpolicy", "-c", "background"] + cmd
    elif macos_qos == "utility_pinned" and platform.system() == "Darwin" and shutil.which("taskpolicy"):
        cmd = ["taskpolicy", "-c", "utility"] + cmd
        if higher_tier_cores > 0:
            spin_bin = _ensure_reg_spinner()
            if spin_bin is not None:
                for _ in range(higher_tier_cores):
                    spinners.append(subprocess.Popen([str(spin_bin)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
                time.sleep(0.1)

    try:
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(work_dir), env=env, capture_output=True, text=True)
        wall_s = time.perf_counter() - t0
    finally:
        for sp in spinners:
            sp.terminate()
        for sp in spinners:
            try:
                sp.wait(timeout=2.0)
            except Exception:
                sp.kill()

    log_text = log_out.read_text(errors="replace") if log_out.exists() else ""
    rusage = parse_rusage(log_text)

    golden_raw = GOLDEN_DIR / f"{deck_name}_baseline.raw.gz"
    if not golden_raw.exists():
        golden_raw = GOLDEN_DIR / f"{deck_name}_baseline.raw"

    acc: Dict[str, Any] = {}
    if proc.returncode == 0 and raw_out.exists() and golden_raw.exists():
        acc = compare_waveforms(golden_raw, raw_out, tol_rms=tol_rms)
        if acc.get("pass"):
            acc.pop("per_vec", None)

    return {
        "deck": deck_name,
        "solver": solver,
        "num_threads": 1,
        "rc": proc.returncode,
        "wall_s": round(wall_s, 4),
        "rusage": rusage,
        "accuracy": acc,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="ngspice-bench single-core benchmark runner")
    ap.add_argument("--bin", required=True, help="Path to ngspice executable")
    ap.add_argument("--submission-id", required=True, help="Unique submission slug (saved to results/<id>.json)")
    ap.add_argument("--build-name", required=True, help="Human-readable ngspice build label")
    ap.add_argument("--suite", choices=["probe", "hero", "all"], default="all")
    ap.add_argument("--decks", nargs="+", help="Explicit deck list (overrides --suite)")
    ap.add_argument("--solvers", nargs="+", default=["klu", "sparse"], choices=["klu", "sparse"])
    ap.add_argument("--uselib", action="store_true", help="Precompile GF180 PDK once via --precompile and run with .uselib")
    ap.add_argument("--core-class", help="Explicit core class label (e.g. 'P-core', 'E-core', 'Zen 4', 'Homogeneous')")
    ap.add_argument("--platform", default="", help="Host platform / chassis model (e.g. 'MacBook Pro (Mac18,1, MHWV4LL/A)')")
    ap.add_argument("--memory", default="", help="Host memory details (e.g. '32 GB LPDDR5 (Samsung)')")
    ap.add_argument("--storage", default="", help="Host SSD / storage details (e.g. 'APPLE SSD AP1024Z (1 TB NVMe)')")
    ap.add_argument("--pin-cpu", type=int, help="Pin single-core run to specific Linux logical CPU ID")
    ap.add_argument("--all-core-classes", action="store_true", help="Run separately on every detected core class (for big.LITTLE / hybrid CPUs)")
    ap.add_argument("--compiler", default="", help="Compiler name/version (e.g. 'GCC 13.3.0')")
    ap.add_argument("--ngspice-repo", default="", help="Source repository URL")
    ap.add_argument("--ngspice-ref", default="", help="Git branch/tag/commit hash")
    ap.add_argument("--configure-flags", default="", help="./configure options used to build ngspice")
    ap.add_argument("--cflags", default="", help="CFLAGS used to build ngspice")
    ap.add_argument("--ldflags", default="", help="LDFLAGS used to build ngspice")
    ap.add_argument("--opt-general", type=int, choices=[0, 1], default=None, help="NGSPICE_OPT_GENERAL flag value (0 or 1)")
    ap.add_argument("--opt-zen4", type=int, choices=[0, 1], default=None, help="NGSPICE_OPT_ZEN4 flag value (0 or 1)")
    ap.add_argument("--opt-apple", type=int, choices=[0, 1], default=None, help="NGSPICE_OPT_APPLE flag value (0 or 1)")
    ap.add_argument("--notes", default="", help="Free-form notes about build or hardware")
    ap.add_argument("--tol-rms", type=float, default=1e-3)
    ap.add_argument("--results-dir", default="", help="Override output directory for submission JSON (default: repo results/)")
    ap.add_argument("--no-rebuild-leaderboard", action="store_true")
    args = ap.parse_args()

    if not (PDK_DIR / "libs.tech/ngspice/sm141064.ngspice").exists():
        print(
            "ERROR: GF180 PDK submodule not initialized! Run:\n"
            "  git submodule update --init --depth 1",
            file=sys.stderr,
        )
        sys.exit(2)

    ng_bin = Path(args.bin).resolve()
    decks = args.decks if args.decks else (PROBE_DECKS if args.suite == "probe" else (HERO_DECKS if args.suite == "hero" else ALL_DECKS))

    detected_classes = detect_core_classes()
    host_info = detect_host_info(
        detected_classes,
        platform_override=args.platform,
        memory_override=args.memory,
        storage_override=args.storage,
    )
    ng_ver = detect_ngspice_version(ng_bin)

    if args.all_core_classes:
        target_classes = []
        for cls in detected_classes:
            if cls.get("macos_qos") == "unselectable":
                print(
                    f"Skipping core class '{cls['label']}' (class_id={cls['class_id']}): "
                    "macOS XNU QoS cannot pin a single thread exclusively to intermediate perflevel clusters.",
                    file=sys.stderr,
                )
                continue
            target_classes.append(cls)
    elif args.core_class or args.pin_cpu is not None:
        base = detected_classes[0].copy()
        if args.core_class:
            want = args.core_class.strip().lower()
            for cls in detected_classes:
                if cls["label"].lower() == want or cls["class_id"].lower() == want:
                    base = cls.copy()
                    break
            base["label"] = args.core_class
            base["class_id"] = re.sub(r"[^a-z0-9]+", "_", args.core_class.lower()).strip("_")
        if args.pin_cpu is not None:
            base["pin_cpu"] = args.pin_cpu
        target_classes = [base]
    else:
        target_classes = [detected_classes[0]]

    out_results_dir = Path(args.results_dir).expanduser().resolve() if args.results_dir else RESULTS_DIR
    out_results_dir.mkdir(parents=True, exist_ok=True)
    overall_ok = True

    for cls in target_classes:
        sub_id = args.submission_id if len(target_classes) == 1 else f"{args.submission_id}_{cls['class_id']}"
        pin_cpu = cls.get("pin_cpu")
        macos_qos = cls.get("macos_qos")
        higher_tier_cores = int(cls.get("higher_tier_cores") or 0)
        print(
            f"\n=== ngspice-bench (1-core) | id={sub_id} | cpu={host_info['cpu_model']} "
            f"| core_class={cls['label']} (pin_cpu={pin_cpu}) | uselib={args.uselib} ==="
        )
        with tempfile.TemporaryDirectory(prefix="ngspice_bench_") as tmp:
            work_dir = Path(tmp)
            nglib_path: Optional[Path] = None
            precompile_s: Optional[float] = None
            if args.uselib:
                nglib_path = work_dir / "pdk_gf180_typ.nglib"
                precompile_s = precompile_pdk(ng_bin, nglib_path, pin_cpu)
                print(f"  Precompiled GF180 PDK -> {nglib_path.stat().st_size / 1024:.1f} KiB in {precompile_s*1000:.1f} ms")

            runs: List[Dict[str, Any]] = []
            all_ok = True
            for d in decks:
                for s in args.solvers:
                    r = run_one(ng_bin, d, s, work_dir, args.tol_rms, pin_cpu, macos_qos, nglib_path, higher_tier_cores)
                    runs.append(r)
                    ru = r["rusage"]
                    ac = r["accuracy"]
                    ok = (r["rc"] == 0) and bool(ac.get("pass", False))
                    if not ok:
                        all_ok = False
                        overall_ok = False
                    print(
                        f"  [{'PASS' if ok else 'FAIL'}] {d:22s} {s:6s} 1c | wall={r['wall_s']:7.2f}s "
                        f"tran={ru.get('tran_analysis_s', 0):6.2f}s load={ru.get('tran_load_s', 0):6.2f}s "
                        f"fac={ru.get('tran_factor_s', 0):5.2f}s slv={ru.get('tran_solve_s', 0):5.2f}s "
                        f"trn={ru.get('tran_trunc_s', 0):5.2f}s iter={ru.get('tot_iter', 0)} | "
                        f"rel_rms={ac.get('worst_rel_rms', -1):.2e}"
                    )
                    sys.stdout.flush()

        payload = {
            "schema_version": 1,
            "submission_id": sub_id,
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "mode": "single_core",
            "all_ok": all_ok,
            "host": host_info,
            "core_class": {
                "class_id": cls["class_id"],
                "label": cls["label"],
                "hybrid": cls.get("hybrid", False),
                "pin_cpu": pin_cpu,
                "max_freq_mhz": cls.get("max_freq_mhz"),
                "l1d_cache": cls.get("l1d_cache") or host_info.get("l1d_cache"),
                "l2_cache": cls.get("l2_cache") or host_info.get("l2_cache"),
                "capacity": cls.get("capacity"),
                "arm_part": cls.get("arm_part"),
            },
            "build": {
                "name": args.build_name,
                "ngspice_version": ng_ver,
                "repo": args.ngspice_repo,
                "git_ref": args.ngspice_ref,
                "compiler": args.compiler,
                "configure_flags": args.configure_flags,
                "cflags": args.cflags,
                "ldflags": args.ldflags,
                "feature_flags": {
                    "NGSPICE_OPT_GENERAL": args.opt_general,
                    "NGSPICE_OPT_ZEN4": args.opt_zen4,
                    **({"NGSPICE_OPT_APPLE": args.opt_apple} if args.opt_apple is not None else {}),
                },
                "pdk_mode": "precompiled_uselib" if args.uselib else "standard",
                "pdk_precompile_s": precompile_s,
                "notes": args.notes,
            },
            "runs": runs,
        }
        out_path = out_results_dir / f"{sub_id}.json"
        out_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Saved submission -> {out_path} (all_ok={all_ok})")

    if not args.no_rebuild_leaderboard:
        subprocess.run([sys.executable, str(SCRIPT_DIR / "build_leaderboard.py")], check=True)

    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
