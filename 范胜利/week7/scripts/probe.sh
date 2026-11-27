#!/usr/bin/env bash
# 单句探针包装脚本：自动 cd 到 src、用带 torch 的 Python，避免路径/环境踩坑
# 用法：
#   bash scripts/probe.sh --text "漳州发展中标马洲污水处理厂PPP项目，中标金额3.2亿元" --methods rule,bilstm
set -euo pipefail
cd "$(dirname "$0")/../src"
PYTHON="${PYTHON:-/opt/anaconda3/envs/py312/bin/python}"
exec "$PYTHON" eval/probe_sentence.py "$@"
