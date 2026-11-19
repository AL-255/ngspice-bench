#!/usr/bin/env python3
"""Aggregate results/*.json into docs/data/leaderboard.json (single-core leaderboard)."""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
OUT_PATH = REPO_ROOT / "docs" / "data" / "leaderboard.json"

PROBE_DECKS = [
    "sdm16_probe_8cyc",
    "pll1p2g_probe_20ns",
    "ti12_full_probe_30ns",
    "sar12_probe_2cyc",
]
HERO_DECK = "sar12_hero_16cyc"

DECK_META = {
    "sar12_hero_16cyc": {
        "title": "12-bit SAR ADC (16 Cycles, Hero)",
        "category": "hero",
        "eqns": 1559,
        "iters_approx": 206500,
        "pdk": "GF180MCU D",
        "source": "aicdb/sar-gen-gemini4",
        "description": "12-bit redundant post-PEX SAR ADC running 16 full conversion cycles (>10 min on stock ngspice-47 Sparse 1c). Verified bit-exact on all 16 output codes.",
    },
    "sar12_probe_2cyc": {
        "title": "12-bit SAR ADC (2 Cycles)",
        "category": "probe",
        "eqns": 1559,
        "iters_approx": 26000,
        "pdk": "GF180MCU D",
        "source": "aicdb/sar-gen-gemini4",
        "description": "2-cycle probe of the 12-bit post-PEX redundant SAR ADC.",
    },
    "ti12_full_probe_30ns": {
        "title": "Full-Chip 3-ch TI 12b SAR + PLL (30 ns)",
        "category": "probe",
        "eqns": 7221,
        "iters_approx": 2121,
        "pdk": "GF180MCU D",
        "source": "aicdb/ti-gen-gemini4",
        "description": "7,221-equation post-PEX full-chip time-interleaved 12-bit ADC + PLL.",
    },
    "pll1p2g_probe_20ns": {
        "title": "1.2 GHz All-Digital PLL (20 ns)",
        "category": "probe",
        "eqns": 1307,
        "iters_approx": 2866,
        "pdk": "GF180MCU D",
        "source": "aicdb/pll-gen-gemini4",
        "description": "Post-PEX 1.2 GHz all-digital PLL transient simulation.",
    },
    "sdm16_probe_8cyc": {
        "title": "16-bit Switched-Cap SDM (8 Cycles)",
        "category": "probe",
        "eqns": 3609,
        "iters_approx": 5596,
        "pdk": "GF180MCU D",
        "source": "aicdb/sdm-gen-gemini4",
        "description": "Post-PEX 16-bit switched-capacitor CIFB sigma-delta modulator.",
    },
}

# Reference baseline (stock ngspice-47 Sparse 1-core on AMD Ryzen 9 7940HX) for normalized speedup scores
REFERENCE_BASELINE_WALL_S = {
    "sar12_hero_16cyc": 628.48,
    "sar12_probe_2cyc": 98.94,
    "ti12_full_probe_30ns": 61.07,
    "pll1p2g_probe_20ns": 11.00,
    "sdm16_probe_8cyc": 8.29,
}


def geomean(vals: List[float]) -> Optional[float]:
    if not vals or any(v <= 0 for v in vals):
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def build() -> None:
    submissions: List[Dict[str, Any]] = []
    entries: List[Dict[str, Any]] = []

    for path in sorted(RESULTS_DIR.glob("*.json")):
        sub = json.loads(path.read_text())
        if sub.get("mode", "single_core") != "single_core":
            continue
        submissions.append(sub)

        # Group runs within a submission by solver (klu vs sparse) -> one leaderboard row per (submission, solver)
        by_solver: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for r in sub.get("runs", []):
            if r.get("num_threads", 1) != 1:
                continue
            by_solver.setdefault(r["solver"], {})[r["deck"]] = r

        for solver, deck_map in sorted(by_solver.items()):
            probe_times = [deck_map[d]["wall_s"] for d in PROBE_DECKS if d in deck_map]
            probe_speedups = [
                REFERENCE_BASELINE_WALL_S[d] / deck_map[d]["wall_s"]
                for d in PROBE_DECKS
                if d in deck_map and deck_map[d]["wall_s"] > 0
            ]
            has_all_probes = len(probe_times) == len(PROBE_DECKS)
            probe_total_s = round(sum(probe_times), 4) if has_all_probes else None
            probe_geomean_s = round(geomean(probe_times), 4) if has_all_probes else None
            probe_geomean_speedup = round(geomean(probe_speedups), 3) if has_all_probes else None

            hero_run = deck_map.get(HERO_DECK)
            hero_wall_s = hero_run["wall_s"] if hero_run else None
            hero_speedup = (
                round(REFERENCE_BASELINE_WALL_S[HERO_DECK] / hero_wall_s, 3)
                if hero_wall_s and hero_wall_s > 0
                else None
            )

            all_pass = all(
                (r.get("rc") == 0 and bool(r.get("accuracy", {}).get("pass", False)))
                for r in deck_map.values()
            )
            worst_rel_rms = max(
                (
                    r.get("accuracy", {}).get("worst_rel_rms", 0.0)
                    for r in deck_map.values()
                ),
                default=0.0,
            )

            # Compact per-deck summary for the UI
            decks_summary: Dict[str, Any] = {}
            for dname, r in deck_map.items():
                ru = r.get("rusage", {})
                ac = r.get("accuracy", {})
                ref_s = REFERENCE_BASELINE_WALL_S.get(dname)
                decks_summary[dname] = {
                    "wall_s": r["wall_s"],
                    "speedup_vs_ref": round(ref_s / r["wall_s"], 3) if (ref_s and r["wall_s"] > 0) else None,
                    "tran_s": ru.get("tran_analysis_s"),
                    "load_s": ru.get("tran_load_s"),
                    "factor_s": ru.get("tran_factor_s"),
                    "solve_s": ru.get("tran_solve_s"),
                    "trunc_s": ru.get("tran_trunc_s"),
                    "tot_iter": ru.get("tot_iter"),
                    "eqns": ru.get("eqns"),
                    "pass": (r.get("rc") == 0 and bool(ac.get("pass", False))),
                    "rel_rms": ac.get("worst_rel_rms"),
                    "exact_grid": ac.get("exact_grid"),
                    "sar_codes_match": ac.get("sar_codes_match"),
                    "sar_codes_count": len(ac.get("sar_codes_cand", [])),
                    "eoc_max_dt_ps": ac.get("eoc_max_dt_ps"),
                }

            entries.append({
                "entry_id": f"{sub['submission_id']}__{solver}",
                "submission_id": sub["submission_id"],
                "timestamp_utc": sub.get("timestamp_utc"),
                "solver": solver,
                "num_threads": 1,
                "all_pass": all_pass,
                "worst_rel_rms": worst_rel_rms,
                "host": sub["host"],
                "core_class": sub["core_class"],
                "build": sub["build"],
                "probe_total_s": probe_total_s,
                "probe_geomean_s": probe_geomean_s,
                "probe_geomean_speedup": probe_geomean_speedup,
                "hero_wall_s": hero_wall_s,
                "hero_speedup": hero_speedup,
                "decks": decks_summary,
            })

    # Default sort: fastest probe_geomean_s (or hero_wall_s) first
    entries.sort(
        key=lambda e: (
            0 if e["all_pass"] else 1,
            e["probe_geomean_s"] if e["probe_geomean_s"] is not None else 1e9,
            e["hero_wall_s"] if e["hero_wall_s"] is not None else 1e9,
        )
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(
            {
                "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "mode": "single_core",
                "reference_baseline": {
                    "description": "Stock ngspice-47 (Sparse 1-core) on AMD Ryzen 9 7940HX (Zen 4)",
                    "wall_s": REFERENCE_BASELINE_WALL_S,
                },
                "probe_decks": PROBE_DECKS,
                "hero_deck": HERO_DECK,
                "deck_meta": DECK_META,
                "entry_count": len(entries),
                "entries": entries,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Built leaderboard: {OUT_PATH} ({len(entries)} single-core entries from {len(submissions)} submissions)")


if __name__ == "__main__":
    build()
