#!/usr/bin/env bash
# Runs an existing Windows build through Proton.
set -euo pipefail

if (( $# == 0 )) || [[ "$1" == --help ]]; then
    printf 'Usage: %s <title.exe> [victory run options]\n' "$0"
    printf 'Set PROTON to the path to the Proton script and STEAM_COMPAT_CLIENT_INSTALL_PATH to the Steam installation.\n'
    if (( $# == 0 )); then exit 2; fi
    exit 0
fi

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
: "${PROTON:?Set PROTON to the path to the installed Proton script}"
: "${STEAM_COMPAT_CLIENT_INSTALL_PATH:?Set STEAM_COMPAT_CLIENT_INSTALL_PATH to the Steam installation}"

profile=${VICTORY_BUILD_PROFILE:-diag}
case "$profile" in
    diag|release) directory=$profile ;;
    dev) directory=debug ;;
    *) printf 'Invalid VICTORY_BUILD_PROFILE: %s (use diag, release or dev).\n' "$profile" >&2; exit 2 ;;
esac
victory="$repo/target/x86_64-pc-windows-gnu/$directory/victory.exe"

for path in "$PROTON" "$victory" "$1"; do
    if [[ ! -f "$path" ]]; then
        printf 'File not found: %s\n' "$path" >&2
        exit 1
    fi
done
if [[ ! -x "$PROTON" || ! -d "$STEAM_COMPAT_CLIENT_INSTALL_PATH" ]]; then
    printf 'Proton must be executable and the Steam installation must exist.\n' >&2
    exit 1
fi

proton=$(realpath -e -- "$PROTON")
title=$(realpath -e -- "$1")
shift
export STEAM_COMPAT_CLIENT_INSTALL_PATH
STEAM_COMPAT_CLIENT_INSTALL_PATH=$(realpath -e -- "$STEAM_COMPAT_CLIENT_INSTALL_PATH")
export STEAM_COMPAT_DATA_PATH="${STEAM_COMPAT_DATA_PATH:-${XDG_DATA_HOME:-$HOME/.local/share}/victory/compatdata}"
mkdir -p -- "$STEAM_COMPAT_DATA_PATH"
STEAM_COMPAT_DATA_PATH=$(realpath -e -- "$STEAM_COMPAT_DATA_PATH")

cd -- "$repo"
# Proton's default Z: drive exposes absolute Linux paths to the Windows app.
exec "$proton" run "$victory" run "Z:${title//\//\\}" --execute "$@"
