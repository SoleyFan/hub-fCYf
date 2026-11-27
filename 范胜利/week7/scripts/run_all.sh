#!/usr/bin/env bash
# week7 一键复现（sample 数据版）
#
# 用法：
#   bash scripts/run_all.sh
#   PYTHON=/path/to/python BERT_PATH=/path/to/bert-base-chinese bash scripts/run_all.sh
#
# 说明：
#   - 默认用 sample 数据（800/300）跑规则、BiLSTM-CRF；设置 BERT_PATH 后追加 BERT+Linear；
#   - BERT+CRF 在纯 CPU 上约 3~5 倍 BERT+Linear 的单步成本，默认不跑（见 README 的说明）；
#   - LLM 链路需要 API Key 或本地 Qwen3-0.6B，未在脚本里自动跑，指标使用已保存预测复评。
#
#   BERT_PATH="/path/to/bert-base-chinese" bash scripts/run_all.sh

set -euo pipefail

cd "$(dirname "$0")/../src"
PYTHON="${PYTHON:-python3}"
DATA_DIR="${DATA_DIR:-../data/sample}"
FULL_DIR="${FULL_DIR:-}"
BERT_PATH="${BERT_PATH:-}"

echo "== 0. 数据体检 =="
if [[ -n "$FULL_DIR" ]]; then
  "$PYTHON" tools/audit_data.py --data_dir "$DATA_DIR" --full_dir "$FULL_DIR"
else
  "$PYTHON" tools/audit_data.py --data_dir "$DATA_DIR"
fi

echo "== 1. L0 规则 =="
"$PYTHON" rule/run_rule.py --data_dir "$DATA_DIR" --name rule_sample

echo "== 2. L1 BiLSTM-CRF（CPU 约 18 分钟）=="
"$PYTHON" bilstm_crf/train.py --data_dir "$DATA_DIR" --device cpu \
  --epochs 6 --max_len 256 --batch_size 32 --lr 3e-3 --warmup_steps 20 \
  --run_name bilstm_crf_sample

if [[ -n "$BERT_PATH" ]]; then
  echo "== 3. L2 BERT+Linear（CPU 约 45 分钟）=="
  "$PYTHON" bert/train.py --data_dir "$DATA_DIR" --device cpu --decoder linear \
    --bert_path "$BERT_PATH" --bert_layers 12 --epochs 3 --max_len 256 --batch_size 16 \
    --run_name bert_linear_sample
else
  echo "== 3. 跳过 BERT+Linear（未设置 BERT_PATH）=="
fi

echo "== 4. 同批复评（sample ∩ LLM n200 = 200 句）=="
"$PYTHON" eval/score_subset.py

echo "== 5. 报告图 =="
"$PYTHON" eval/make_report_figures.py

echo "完成：训练产物在 src/outputs/；汇总表/图在 week7/outputs/"
