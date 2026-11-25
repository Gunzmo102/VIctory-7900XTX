#!/usr/bin/env bash
# Builds the Windows executable on Linux with the repository's pinned Rust.
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
target=x86_64-pc-windows-gnu
profile=${1:-diag}
case "$profile" in
    diag) features=(--features diagnostics) ;;
    release|dev) features=() ;;
    *) printf 'Usage: %s [diag|release|dev]\n' "$0" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
    printf 'Usage: %s [diag|release|dev]\n' "$0" >&2
    exit 2
fi

for tool in rustup x86_64-w64-mingw32-gcc x86_64-w64-mingw32-ar; do
    if ! command -v "$tool" >/dev/null; then
        printf 'Missing %s. Install rustup and the x86-64 MinGW toolchain.\n' "$tool" >&2
        exit 1
    fi
done

cd -- "$repo"
toolchain=$(sed -n 's/^channel = "\([^"]*\)"$/\1/p' rust-toolchain.toml)
if [[ -z "$toolchain" ]]; then
    printf 'Could not read the Rust channel from rust-toolchain.toml.\n' >&2
    exit 1
fi
rustup toolchain install "$toolchain" --profile minimal --target "$target"
cargo "+$toolchain" build --locked -p victory --target "$target" --profile "$profile" \
    --target-dir "$repo/target" "${features[@]}"
