#!/usr/bin/env sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
mkdir -p "$root/dist"
cd "$root"
zip -r "${1:-dist/PipeSD-edge.zip}" edge shared
