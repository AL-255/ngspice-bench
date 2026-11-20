#!/usr/bin/env bash
# Build a fully self-contained multi-ISA x86_64 Linux AppImage for ngspice-bench:
#   - 4 Ubuntu 22.04 (glibc 2.35, GCC 12) ngspice binaries:
#       ngspice-opt-v4 (AVX-512), ngspice-opt-v3 (AVX2/FMA), ngspice-opt-v1 (SSE2),
#       ngspice-stock (upstream ngspice-47@a80f6e3e9)
#   - Standalone portable Python 3.12 + NumPy (zero host Python/NumPy needed)
#   - Complete GF180 post-PEX benchmark payload (decks, circuits, 1.8 MB PDK SPICE
#     subset, golden waveforms, scripts)
#   - Official AppImage Type-2 runtime + zstd SquashFS
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
NGSPICE_REPO="${NGSPICE_REPO:-$(cd "$BENCH_REPO/../ngspice-sf-mirror" && pwd)}"
DIST_DIR="${DIST_DIR:-$(cd "$BENCH_REPO/.." && pwd)/dist}"
WORK_DIR="${WORK_DIR:-/tmp/ngspice_appimage_build}"
CACHE_DIR="${CACHE_DIR:-/tmp/ngspice_appimage_cache}"

DOCKER_IMG="ngspice-bench-builder:ubuntu2204"
RUNTIME_URL="https://github.com/AppImage/type2-runtime/releases/download/continuous/runtime-x86_64"
PY_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20261110/cpython-3.12.16%2B20261110-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"

mkdir -p "$DIST_DIR" "$WORK_DIR" "$CACHE_DIR"
APPDIR="$WORK_DIR/AppDir"
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/lib" "$APPDIR/usr/share/ngspice-bench"

echo "=== [1/5] Building Ubuntu 22.04 Docker builder image ==="
docker build -t "$DOCKER_IMG" -f "$SCRIPT_DIR/Dockerfile.ubuntu2204" "$SCRIPT_DIR"

echo "=== [2/5] Compiling 4 x86_64 ngspice binaries inside Ubuntu 22.04 (glibc 2.35) ==="
BIN_OUT="$WORK_DIR/bins"
rm -rf "$BIN_OUT"
mkdir -p "$BIN_OUT"
docker run --rm \
    -u "$(id -u):$(id -g)" \
    -v "$NGSPICE_REPO:/src/ngspice-sf-mirror:ro" \
    -v "$SCRIPT_DIR/build_in_container.sh:/work/build_in_container.sh:ro" \
    -v "$BIN_OUT:/out" \
    "$DOCKER_IMG" \
    bash /work/build_in_container.sh

cp -a "$BIN_OUT/bin/." "$APPDIR/usr/bin/"
cp -a "$BIN_OUT/lib/." "$APPDIR/usr/lib/"
cp -a "$BIN_OUT/BUILD_INFO.env" "$APPDIR/usr/BUILD_INFO.env"

echo "=== [3/5] Preparing standalone portable Python 3.12 + NumPy ==="
PY_TAR="$CACHE_DIR/cpython-3.12-x86_64-stripped.tar.gz"
if [ ! -f "$PY_TAR" ]; then
    curl -fsSL -o "$PY_TAR" "$PY_URL"
fi
tar -xzf "$PY_TAR" -C "$APPDIR/usr"
# Install numpy into bundled python and strip bloat (pip/tests/tcl/idle/pycache)
"$APPDIR/usr/python/bin/python3" -m pip install --no-cache-dir --quiet numpy
rm -rf \
    "$APPDIR/usr/python/lib/python3.12/test" \
    "$APPDIR/usr/python/lib/python3.12/idlelib" \
    "$APPDIR/usr/python/lib/python3.12/tkinter" \
    "$APPDIR/usr/python/lib/python3.12/turtledemo" \
    "$APPDIR/usr/python/lib/python3.12/ensurepip" \
    "$APPDIR/usr/python/lib/python3.12/site-packages/pip"* \
    "$APPDIR/usr/python/lib/python3.12/site-packages/numpy/tests" \
    "$APPDIR/usr/python/lib/python3.12/site-packages/numpy/_core/tests" \
    "$APPDIR/usr/python/lib/python3.12/site-packages/numpy/linalg/tests" \
    "$APPDIR/usr/python/lib/python3.12/site-packages/numpy/fft/tests" \
    "$APPDIR/usr/python/lib/python3.12/site-packages/numpy/random/tests" \
    "$APPDIR/usr/python/lib/python3.12/site-packages/numpy/ma/tests" \
    "$APPDIR/usr/python/lib/python3.12/site-packages/numpy/polynomial/tests" \
    "$APPDIR/usr/python/lib/python3.12/site-packages/numpy/f2py" \
    "$APPDIR/usr/python/include" \
    "$APPDIR/usr/python/share" \
    "$APPDIR/usr/python/lib/libpython3.12.a" \
    "$APPDIR/usr/python/lib/python3.12/config-3.12-x86_64-linux-gnu"
find "$APPDIR/usr/python" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
find "$APPDIR/usr/python" -name '*.pyc' -delete 2>/dev/null || true
find "$APPDIR/usr/python" -name '*.so*' -type f -exec strip --strip-unneeded {} + 2>/dev/null || true
echo "  Bundled Python+NumPy uncompressed size: $(du -sh "$APPDIR/usr/python" | cut -f1)"

echo "=== [4/5] Copying ngspice-bench payload (clean decks, circuits, golden, scripts, 1.8 MB PDK subset) ==="
B_DST="$APPDIR/usr/share/ngspice-bench"
mkdir -p "$B_DST/decks" "$B_DST/circuits" "$B_DST/golden" "$B_DST/scripts"
mkdir -p "$B_DST/pdk/gf180mcu/gf180mcuD/libs.tech" "$B_DST/pdk/gf180mcu/gf180mcuD/libs.ref/gf180mcu_fd_sc_mcu7t5v0"

# Decks: only tracked deck/spice files (never stray .raw/.log files)
for f in "$BENCH_REPO/decks/"*.cir "$BENCH_REPO/decks/"*.spice "$BENCH_REPO/decks/"*.inc; do
    [ -e "$f" ] && cp -a "$f" "$B_DST/decks/"
done
cp -a "$BENCH_REPO/circuits/." "$B_DST/circuits/"
cp -a "$BENCH_REPO/golden/"*_baseline.raw.gz "$B_DST/golden/"
cp -a "$BENCH_REPO/scripts/run_bench.py" "$BENCH_REPO/scripts/compare_raw.py" "$BENCH_REPO/scripts/build_leaderboard.py" "$B_DST/scripts/"
cp -a "$BENCH_REPO/pdk/gf180mcu/gf180mcuD/libs.tech/ngspice" "$B_DST/pdk/gf180mcu/gf180mcuD/libs.tech/"
cp -a "$BENCH_REPO/pdk/gf180mcu/gf180mcuD/libs.ref/gf180mcu_fd_sc_mcu7t5v0/spice" "$B_DST/pdk/gf180mcu/gf180mcuD/libs.ref/gf180mcu_fd_sc_mcu7t5v0/"

# AppRun, desktop entry, icon
cp "$SCRIPT_DIR/AppRun" "$APPDIR/AppRun"
chmod +x "$APPDIR/AppRun"

cat > "$APPDIR/ngspice-bench.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=ngspice-bench
Comment=Single-core GF180 post-PEX SPICE benchmark suite
Exec=AppRun
Icon=ngspice-bench
Terminal=true
Categories=Development;Science;Electronics;
EOF

cat > "$APPDIR/ngspice-bench.svg" <<'EOF'
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128">
  <rect width="128" height="128" rx="24" fill="#111827"/>
  <path d="M18 82 L38 82 L50 30 L66 98 L82 48 L94 82 L110 82"
        fill="none" stroke="#f97316" stroke-width="10" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
EOF
ln -sf ngspice-bench.svg "$APPDIR/.DirIcon"

echo "  AppDir uncompressed breakdown:"
du -sh "$APPDIR/usr/bin" "$APPDIR/usr/lib" "$APPDIR/usr/python" "$B_DST" "$APPDIR"

echo "=== [5/5] Packing Type-2 AppImage (zstd SquashFS + official runtime-x86_64) ==="
RUNTIME="$CACHE_DIR/runtime-x86_64"
if [ ! -f "$RUNTIME" ]; then
    curl -fsSL -o "$RUNTIME" "$RUNTIME_URL"
    chmod +x "$RUNTIME"
fi

SQFS="$WORK_DIR/ngspice-bench.squashfs"
rm -f "$SQFS"
mksquashfs "$APPDIR" "$SQFS" -root-owned -noappend -comp zstd -Xcompression-level 15 -b 256K -quiet

# shellcheck disable=SC1091
source "$APPDIR/usr/BUILD_INFO.env"
OPT_SHORT="${NGSPICE_OPT_REF#*@}"
OUT_IMG="$DIST_DIR/ngspice-bench-${OPT_SHORT}-x86_64.AppImage"
OUT_LATEST="$DIST_DIR/ngspice-bench-x86_64.AppImage"

cat "$RUNTIME" "$SQFS" > "$OUT_IMG"
chmod +x "$OUT_IMG"
cp -f "$OUT_IMG" "$OUT_LATEST"
(cd "$DIST_DIR" && sha256sum "$(basename "$OUT_IMG")" "$(basename "$OUT_LATEST")" > SHA256SUMS.txt)

echo "=== SUCCESS ==="
ls -lh "$OUT_IMG" "$OUT_LATEST"
cat "$DIST_DIR/SHA256SUMS.txt"
