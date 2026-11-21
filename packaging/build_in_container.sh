#!/usr/bin/env bash
# Runs inside ubuntu:22.04 container (glibc 2.35, GCC 12) to build all 4 x86_64
# ngspice binaries for the AppImage with maximum Linux distro portability.
set -euo pipefail

SRC_OPT="${SRC_OPT:-/src/ngspice-sf-mirror}"
OUT_DIR="${OUT_DIR:-/out}"
STOCK_REF="${STOCK_REF:-a80f6e3e9}"
NPROC="${NPROC:-$(nproc)}"

mkdir -p "$OUT_DIR/bin" "$OUT_DIR/lib" "$OUT_DIR/lib/ngspice" "$OUT_DIR/share/ngspice/scripts"

COMMON_CONF=(
    --prefix=/usr
    --without-x
    --with-readline=no
    --disable-debug
    --enable-xspice
    --enable-cider
    --enable-openmp
    --enable-klu
    --enable-osdi
)

COMMON_LD="-static-libstdc++ -static-libgcc"
OPT_BASE_FLAGS="-O3 -fno-math-errno -fno-trapping-math -fno-semantic-interposition -flto=auto"

# Create a clean copy of the optimized source tree (with ./autogen.sh already run)
rm -rf /tmp/src-opt /tmp/src-stock
mkdir -p /tmp/src-opt /tmp/src-stock

echo "=== Preparing clean optimized source tree ==="
git -C "$SRC_OPT" archive HEAD | tar -x -C /tmp/src-opt
(cd /tmp/src-opt && ./autogen.sh >/dev/null 2>&1)

echo "=== Preparing clean stock ngspice-47 ($STOCK_REF) source tree ==="
git -C "$SRC_OPT" archive "$STOCK_REF" | tar -x -C /tmp/src-stock
(cd /tmp/src-stock && ./autogen.sh >/dev/null 2>&1)

build_variant() {
    local name="$1"
    local src="$2"
    local cflags="$3"
    local ldflags="$4"
    shift 4
    local extra_conf=("$@")

    echo "=== Building $name ==="
    local bdir="/tmp/build-$name"
    rm -rf "$bdir"
    mkdir -p "$bdir"
    (
        cd "$bdir"
        "$src/configure" "${COMMON_CONF[@]}" "${extra_conf[@]}" \
            CC=gcc-12 CXX=g++-12 \
            CFLAGS="$cflags" \
            CXXFLAGS="$cflags" \
            LDFLAGS="$ldflags" >/dev/null
        make -j"$NPROC" >/dev/null
        cp src/ngspice "$OUT_DIR/bin/$name"
        patchelf --set-rpath '$ORIGIN/../lib' "$OUT_DIR/bin/$name"
        strip --strip-unneeded "$OUT_DIR/bin/$name"
        if [ "$name" = "ngspice-opt-v1" ]; then
            make DESTDIR=/tmp/stage-inst install >/dev/null
            cp -a /tmp/stage-inst/usr/lib/ngspice/*.cm "$OUT_DIR/lib/ngspice/" 2>/dev/null || true
            strip --strip-unneeded "$OUT_DIR/lib/ngspice/"*.cm 2>/dev/null || true
        fi
    )
    rm -rf "$bdir" /tmp/stage-inst
}

# 1. AVX-512 (x86-64-v4: AMD Zen 4/5, Intel Skylake-X/Ice Lake/Sapphire Rapids+)
build_variant ngspice-opt-v4 /tmp/src-opt \
    "$OPT_BASE_FLAGS -march=x86-64-v4" \
    "-flto=auto -O3 -march=x86-64-v4 $COMMON_LD" \
    --enable-opt-general --enable-opt-zen4

# 2. AVX2 + FMA (x86-64-v3: AMD Zen 1/2/3, Intel Haswell..Arrow Lake)
build_variant ngspice-opt-v3 /tmp/src-opt \
    "$OPT_BASE_FLAGS -march=x86-64-v3" \
    "-flto=auto -O3 -march=x86-64-v3 $COMMON_LD" \
    --enable-opt-general --disable-opt-zen4

# 3. Portable baseline x86-64 (SSE2 only, zero AVX/AVX-512)
build_variant ngspice-opt-v1 /tmp/src-opt \
    "$OPT_BASE_FLAGS -march=x86-64 -mno-avx -mno-avx512f" \
    "-flto=auto -O3 -march=x86-64 $COMMON_LD" \
    --enable-opt-general --disable-opt-zen4

# 4. Upstream Stock ngspice-47 reference
build_variant ngspice-stock /tmp/src-stock \
    "-O2 -march=x86-64" \
    "-O2 -march=x86-64 $COMMON_LD"

# Bundle libgomp.so.1 from Ubuntu 22.04 so host OpenMP runtime is never required
GOMP_PATH="$(gcc-12 -print-file-name=libgomp.so.1)"
cp -L "$GOMP_PATH" "$OUT_DIR/lib/libgomp.so.1"
strip --strip-unneeded "$OUT_DIR/lib/libgomp.so.1"

echo "=== Verifying ISA isolation & GLIBC symbol versions ==="
for b in ngspice-opt-v4 ngspice-opt-v3 ngspice-opt-v1 ngspice-stock; do
    p="$OUT_DIR/bin/$b"
    zmm=$(objdump -d "$p" | grep -c '%zmm' || true)
    ymm=$(objdump -d "$p" | grep -c '%ymm' || true)
    max_glibc=$(readelf -V "$p" | grep -oE 'GLIBC_[0-9]+\.[0-9]+(\.[0-9]+)?' | sort -Vu | tail -n 1)
    echo "  $b: size=$(du -h "$p" | cut -f1)  max_glibc=$max_glibc  %zmm=$zmm  %ymm=$ymm"
    ldd "$p" | sed 's/^/    /'
done

# Strict ISA safety gates
test "$(objdump -d "$OUT_DIR/bin/ngspice-opt-v3" | grep -c '%zmm' || true)" -eq 0
test "$(objdump -d "$OUT_DIR/bin/ngspice-opt-v1" | grep -c '%zmm' || true)" -eq 0
test "$(objdump -d "$OUT_DIR/bin/ngspice-opt-v1" | grep -c '%ymm' || true)" -eq 0
test "$(objdump -d "$OUT_DIR/bin/ngspice-stock"  | grep -c '%zmm' || true)" -eq 0
test "$(objdump -d "$OUT_DIR/bin/ngspice-stock"  | grep -c '%ymm' || true)" -eq 0

# Record build metadata
OPT_SHA="$(git -C "$SRC_OPT" rev-parse --short=9 HEAD)"
cat > "$OUT_DIR/BUILD_INFO.env" <<EOF
NGSPICE_OPT_REF="opt/zen4-transient@${OPT_SHA}"
NGSPICE_STOCK_REF="ngspice-47@${STOCK_REF}"
NGSPICE_COMPILER="$(gcc-12 --version | head -n 1)"
NGSPICE_GLIBC="$(ldd --version | head -n 1)"
EOF
cat "$OUT_DIR/BUILD_INFO.env"
echo "=== Container build complete ==="
