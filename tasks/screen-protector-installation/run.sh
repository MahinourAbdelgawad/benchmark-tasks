#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
exec /data/IsaacLab-Arena/.venv/bin/python -Bu demo.py --condition "${1:-clean}"
