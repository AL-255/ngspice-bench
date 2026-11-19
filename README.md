# ngspice-bench

**Standard single-core transient SPICE benchmark suite and live GitHub Pages leaderboard for `ngspice` builds across CPUs, core classes, and compile flags.**

Live Leaderboard: **https://al-255.github.io/ngspice-bench/**

---

## Benchmark Suite

All 5 workloads are real post-layout-extraction (post-PEX) mixed-signal circuits using the open-source GlobalFoundries 180nm MCU PDK (`gf180mcuD`, included via the [`wafer-space/gf180mcu`](https://github.com/wafer-space/gf180mcu) git submodule), derived from [`aicdb`](https://github.com/aicdb):

| Deck | Suite | Equations | Approx. Newton Iters | Description |
|---|---|---:|---:|---|
| `sar12_hero_16cyc` | `hero` | 1,559 | ~208,490 | 12-bit redundant post-PEX SAR ADC, 16 full conversion cycles (>10 min on stock `ngspice-47` Sparse 1c). Verified bit-exact on all 16 digital output codes and EOC timing. |
| `sar12_probe_2cyc` | `probe` | 1,559 | ~27,301 | 2-cycle probe of the 12-bit post-PEX redundant SAR ADC. |
| `ti12_full_300ns` | `probe` | 7,221 | ~17,619 | Full-chip 3-channel time-interleaved 12-bit SAR ADC + PLL (300 ns transient). |
| `pll_ring_150m_4us` | `probe` | 1,015 | ~153,944 | Closed-loop 7-stage wide-tuning (65.8–218.6 MHz, 3.3×) Type-II ring PLL with 6-bit SAR AFC + 161 GF180 stdcells; 4.0 µs startup from code `0` → half full-scale code `31` & 153.25 MHz lock. |
| `sdm16_120cyc` | `probe` | 3,609 | ~55,547 | 16-bit switched-capacitor CIFB sigma-delta modulator (post-PEX, nominal op-amp biases, unsaturated ±0.15 V @ 26.7 kHz sine, 37.5 µs / 120 cycles, bit-exact on all 115 post-reset latched PDM bits). |

---

## Benchmarking Policy

1. **Single-Core Only:** Every leaderboard entry runs strictly single-threaded (`num_threads=1`, `OMP_NUM_THREADS=1`), pinned to one logical CPU (`taskset -c <cpu>` on Linux).
2. **Heterogeneous / big.LITTLE Core Classes Reported Separately:** On hybrid CPUs (Intel P-core / E-core / LP-E-core, AMD Zen / Zen-c, ARM big.LITTLE, Apple Silicon P-core / E-core), each distinct core class is benchmarked separately and gets its own leaderboard entry.
3. **Strict Golden Waveform Verification:** Every run's binary `.raw` output is compared against stock `ngspice-47` golden waveforms (`golden/*.raw.gz`) via `scripts/compare_raw.py`.
   - `ti12_full_300ns` requires relative RMS waveform error `<= 1e-3` across all saved signals.
   - `pll_ring_150m_4us` requires exact final 6-bit SAR AFC code (`31`), `afc_done` timing skew `< 20 ns`, final `v(vc)` error `< 10 mV`, locked `f_out` error `< 0.2%`, and `v(vc)` relative RMS `<= 0.02`.
   - `sdm16_120cyc` requires **bit-exact** latched `dac_p` PDM output across all `115/115` post-reset clock cycles, PDM edge skew `< 500 ps`, and relative RMS `<= 0.01` across all analog signals.
   - SAR ADC circuits require **bit-exact** output codes on every conversion cycle (`16/16` on `sar12_hero_16cyc`, `2/2` on `sar12_probe_2cyc`), EOC edge skew `< 500 ps`, and post-reset waveform agreement.
   - Builds using `-ffast-math` or unsafe approximations that fail golden verification are marked **FAIL**.

---

## Quick Start & Run Guide

### 1. Clone and Initialize the GF180 PDK Submodule

```sh
git clone git@github.com:AL-255/ngspice-bench.git
cd ngspice-bench
git submodule update --init --depth 1
```

> **Tip:** Only the `gf180mcuD/libs.tech/ngspice` and `gf180mcuD/libs.ref/gf180mcu_fd_sc_mcu7t5v0/spice` directories (~1.7 MB) are read during simulation.

### 2. Prerequisites

- Python 3.9+ with `numpy` (`pip install numpy`)
- `ngspice` compiled with `--enable-klu` (and optionally `--enable-openmp`)
- Linux `taskset` (from `util-linux`, preinstalled on most distros) for single-core CPU pinning, or macOS `taskpolicy` for Apple Silicon P-core / E-core selection

Verify detected CPU & core classes on your machine:

```sh
python3 -c 'import sys, json; sys.path.insert(0, "scripts"); import run_bench; c = run_bench.detect_core_classes(); print(json.dumps({"core_classes": c, "host": run_bench.detect_host_info(c)}, indent=2))'
```

### 3. Running on a Homogeneous CPU

Run the full single-core suite (`probe` + `hero`) across both `klu` and `sparse`:

```sh
python3 scripts/run_bench.py \
  --bin /path/to/ngspice \
  --submission-id mycpu_ngspice47_std \
  --build-name "ngspice-47 (O3 native)" \
  --suite all \
  --solvers klu sparse \
  --compiler "GCC 13.3.0" \
  --ngspice-repo "https://github.com/AL-255/ngspice-sf-mirror" \
  --ngspice-ref "opt/zen4-transient@53747dbda" \
  --configure-flags="--without-x --with-readline=no --disable-debug --enable-openmp --enable-klu" \
  --cflags="-O3 -march=native -fno-math-errno -fno-trapping-math -flto" \
  --ldflags="-flto -O3 -march=native" \
  --opt-general 1 \
  --opt-zen4 0 \
  --notes "Optional notes about cooling, governor, RAM speed, etc."
```

For a faster probe-only test (~10–60 s depending on build/CPU), pass `--suite probe`.

### 4. Running on Hybrid / big.LITTLE CPUs (Intel P/E, AMD Zen/Zen-c, ARM big.LITTLE, Apple Silicon)

Pass `--all-core-classes` to automatically detect every core class, pin to a representative core of each class, and generate **separate submissions** (`results/<id>_p_core.json`, `results/<id>_e_core.json`, etc.):

```sh
python3 scripts/run_bench.py \
  --bin /path/to/ngspice \
  --submission-id i9_14900k_ngspice47 \
  --build-name "ngspice-47 (GCC 14)" \
  --suite all \
  --solvers klu \
  --all-core-classes \
  --compiler "GCC 14.2" \
  --cflags "-O3 -march=native"
```

You can also explicitly target a specific core class and Linux CPU ID:

```sh
# Benchmark P-core (e.g., cpu0)
python3 scripts/run_bench.py --bin /path/to/ngspice --submission-id mycpu_pcore --build-name "ngspice-47" --core-class "P-core" --pin-cpu 0 --suite all --solvers klu

# Benchmark E-core (e.g., cpu16)
python3 scripts/run_bench.py --bin /path/to/ngspice --submission-id mycpu_ecore --build-name "ngspice-47" --core-class "E-core" --pin-cpu 16 --suite all --solvers klu
```

### 5. Running with Explicit PDK Precompilation (`.uselib`)

For `ngspice` builds that support `--precompile` and `.uselib` (such as `opt/zen4-transient`), add `--uselib`. The runner precompiles `decks/pdk_gf180_typ.spice` once into a temporary `.nglib` blob (recording precompile time) and substitutes `.uselib` in each testbench while keeping all top-level netlists uncached:

```sh
python3 scripts/run_bench.py \
  --bin /path/to/ngspice \
  --submission-id mycpu_opt_uselib \
  --build-name "ngspice-47 opt/zen4-transient (.uselib)" \
  --suite all \
  --solvers klu \
  --uselib \
  --opt-general 1 \
  --opt-zen4 1
```

### 6. Building Portable vs Zen 4 Variants of `ngspice`

If building from [`AL-255/ngspice-sf-mirror`](https://github.com/AL-255/ngspice-sf-mirror) (`opt/zen4-transient`), optimizations are controlled by two independent compile-time flags:

- **Portable general optimizations** (`NGSPICE_OPT_GENERAL`): `--enable-opt-general` / `--disable-opt-general` (portable C99/OpenMP — x86-64, ARM64, RISC-V)
- **AMD Zen 4 / AVX-512 SIMD kernels** (`NGSPICE_OPT_ZEN4`): `--enable-opt-zen4` / `--disable-opt-zen4` (auto-enabled iff `__AVX512F__`)

### 7. Rebuilding the Leaderboard & Submitting Results

`scripts/run_bench.py` automatically runs `scripts/build_leaderboard.py` after saving `results/<submission_id>.json`, updating `docs/data/leaderboard.json`.

To regenerate the leaderboard manually:

```sh
python3 scripts/build_leaderboard.py
```

To preview the leaderboard locally:

```sh
python3 -m http.server -d docs 8080
```

To submit your results, open a Pull Request adding `results/<submission_id>.json` and the regenerated `docs/data/leaderboard.json`.

---

## Repository Layout

```
ngspice-bench/
├── pdk/gf180mcu/           # Git submodule: wafer-space/gf180mcu (GF180MCU D PDK)
├── circuits/               # 4 GF180 post-PEX netlists derived from aicdb
├── decks/                  # Portable relative-path SPICE testbenches + PDK bundle
├── golden/                 # Stock ngspice-47 golden waveforms (.raw.gz)
├── scripts/
│   ├── run_bench.py        # Single-core benchmark runner + hybrid core-class detector
│   ├── compare_raw.py      # Golden waveform + SAR digital code accuracy verifier
│   └── build_leaderboard.py# Aggregates results/*.json -> docs/data/leaderboard.json
├── results/                # Verified single-core submission JSONs
└── docs/                   # Static GitHub Pages leaderboard
```

## Attribution & Licenses

- **GF180MCU PDK:** [wafer-space/gf180mcu](https://github.com/wafer-space/gf180mcu) (Apache-2.0, GlobalFoundries PDK Authors), linked via git submodule.
- **Post-PEX Circuits:** Derived from [aicdb](https://github.com/aicdb) (`sar-gen-gemini4`, `ti-gen-gemini4`, `pll-gen-opus5.5`, `sdm-gen-gemini4`).
