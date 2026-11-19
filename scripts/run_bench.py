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
    "sdm16_probe_8cyc",
    "pll1p2g_probe_20ns",
    "ti12_full_probe_30ns",
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
        # Apple Silicon P-core / E-core detection via sysctl hw.nperflevels
        def _sysctl(key: str) -> str:
            r = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True)
            return r.stdout.strip() if r.returncode == 0 else ""

        nlevels = _sysctl("hw.nperflevels")
        if nlevels.isdigit() and int(nlevels) >= 2:
            out = []
            for lv in range(int(nlevels)):
                name = _sysctl(f"hw.perflevel{lv}.name") or (f"P-core" if lv == 0 else f"E-core")
                ncores = _sysctl(f"hw.perflevel{lv}.physicalcpu")
                l2 = _sysctl(f"hw.perflevel{lv}.l2cachesize")
                out.append({
                    "class_id": name.lower().replace(" ", "_"),
                    "label": "P-core" if "performance" in name.lower() else ("E-core" if "efficiency" in name.lower() else name),
                    "hybrid": True,
                    "cpus": list(range(int(ncores))) if ncores.isdigit() else [0],
                    "pin_cpu": None,
                    "macos_qos": "default" if lv == 0 else "background",
                    "l2_cache": f"{int(l2)//1024}K" if l2.isdigit() else None,
                })
            return out

    return [{
        "class_id": "homogeneous",
        "label": "Homogeneous",
        "hybrid": False,
        "cpus": [0],
        "pin_cpu": 0 if hasattr(os, "sched_setaffinity") else None,
    }]


def detect_host_info(core_classes: List[Dict[str, Any]]) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()}",
        "arch": platform.machine(),
        "cpu_model": platform.processor() or "Unknown CPU",
        "logical_cpus": os.cpu_count() or 1,
        "physical_cores": None,
        "l1d_cache": None,
        "l2_cache": None,
        "l3_cache": None,
        "ram_gb": None,
        "isa_features": [],
        "is_hybrid": len(core_classes) > 1,
        "detected_core_classes": [c["label"] for c in core_classes],
    }

    if platform.system() == "Linux":
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
        r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            info["cpu_model"] = r.stdout.strip()
        r2 = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        if r2.returncode == 0 and r2.stdout.strip().isdigit():
            info["ram_gb"] = round(int(r2.stdout.strip()) / (1024**3), 1)

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


def run_one(
    ngspice_bin: Path,
    deck_name: str,
    solver: str,
    work_dir: Path,
    tol_rms: float,
    pin_cpu: Optional[int],
    macos_qos: Optional[str],
    nglib_path: Optional[Path],
) -> Dict[str, Any]:
    src_cir = DECKS_DIR / f"{deck_name}.cir"
    text = src_cir.read_text()

    # Resolve relative ../pdk and ../circuits paths cleanly against DECKS_DIR so work_dir can be anywhere
    text = text.replace("../pdk/", str((REPO_ROOT / "pdk").resolve()) + "/")
    text = text.replace("../circuits/", str((REPO_ROOT / "circuits").resolve()) + "/")

    if nglib_path is not None:
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
    if pin_cpu is not None and platform.system() == "Linux" and shutil.which("taskset"):
        cmd = ["taskset", "-c", str(pin_cpu)] + cmd
    elif macos_qos == "background" and platform.system() == "Darwin" and shutil.which("taskpolicy"):
        cmd = ["taskpolicy", "-b"] + cmd

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(work_dir), env=env, capture_output=True, text=True)
    wall_s = time.perf_counter() - t0

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
    ap.add_argument("--notes", default="", help="Free-form notes about build or hardware")
    ap.add_argument("--tol-rms", type=float, default=1e-3)
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
    host_info = detect_host_info(detected_classes)
    ng_ver = detect_ngspice_version(ng_bin)

    if args.all_core_classes:
        target_classes = detected_classes
    elif args.core_class or args.pin_cpu is not None:
        base = detected_classes[0].copy()
        if args.core_class:
            base["label"] = args.core_class
            base["class_id"] = re.sub(r"[^a-z0-9]+", "_", args.core_class.lower()).strip("_")
        if args.pin_cpu is not None:
            base["pin_cpu"] = args.pin_cpu
        target_classes = [base]
    else:
        target_classes = [detected_classes[0]]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    overall_ok = True

    for cls in target_classes:
        sub_id = args.submission_id if len(target_classes) == 1 else f"{args.submission_id}_{cls['class_id']}"
        pin_cpu = cls.get("pin_cpu")
        macos_qos = cls.get("macos_qos")
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
                    r = run_one(ng_bin, d, s, work_dir, args.tol_rms, pin_cpu, macos_qos, nglib_path)
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
                },
                "pdk_mode": "precompiled_uselib" if args.uselib else "standard",
                "pdk_precompile_s": precompile_s,
                "notes": args.notes,
            },
            "runs": runs,
        }
        out_path = RESULTS_DIR / f"{sub_id}.json"
        out_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Saved submission -> {out_path} (all_ok={all_ok})")

    if not args.no_rebuild_leaderboard:
        subprocess.run([sys.executable, str(SCRIPT_DIR / "build_leaderboard.py")], check=True)

    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
