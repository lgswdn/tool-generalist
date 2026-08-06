#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export START_SERVERS=0
exec "$SCRIPT_DIR/eval_2servers.sh" "$@"
