#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
STEREOVLA_ROOT="${STEREOVLA_ROOT:-$REPO_ROOT/thirdparty/StereoVLA}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/stereo/bin/python}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/${USER:-stereovla}_pip_cache}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] StereoVLA python not found: $PYTHON_BIN" >&2
  echo "[ERROR] Set PYTHON_BIN=/path/to/python or create/use the prefix env at $REPO_ROOT/stereo" >&2
  exit 1
fi
if [[ ! -f "$STEREOVLA_ROOT/requirements.txt" ]]; then
  echo "[ERROR] Missing requirements: $STEREOVLA_ROOT/requirements.txt" >&2
  exit 1
fi

mkdir -p "$PIP_CACHE_DIR"

echo "[INFO] python=$PYTHON_BIN"
"$PYTHON_BIN" - <<'PY'
import sys, platform
print("[INFO] version=" + sys.version.replace("\n", " "))
print("[INFO] platform=" + platform.platform())
PY

echo "[INFO] installing $STEREOVLA_ROOT/requirements.txt"
cd "$STEREOVLA_ROOT"
PIP_CACHE_DIR="$PIP_CACHE_DIR" "$PYTHON_BIN" -m pip install -r requirements.txt
