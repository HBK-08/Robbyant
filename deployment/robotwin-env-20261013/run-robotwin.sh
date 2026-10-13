#!/usr/bin/env bash
set -euo pipefail
source /home/lmy/Robbyant/deployment/robotwin-env-20261013/robotwin-env.sh
cd /home/lmy/Robbyant/RoboTwin
exec "$@"
