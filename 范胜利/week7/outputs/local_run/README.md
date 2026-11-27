# 本地重训产物（sample dev 300 句）

这里只放**本次在 sample 上真正重训**的指标、日志与权重；预测文件统一在
`../predictions/local/`（避免同一份预测存两遍）。

| 方法 | n | 触发词 F1 | 论元 F1 | 事件级 F1 | 训练 | 延迟 |
|---|---:|---:|---:|---:|---:|---:|
| 规则（sample 重训） | 300 | 0.6791 | 0.1558 | 0.0169 | 0（统计词典） | 0.3 ms |
| BiLSTM-CRF（6 epoch） | 300 | 0.7623 | 0.3629 | 0.0472 | 1,080 s（CPU） | 50.7 ms |
| BERT+Linear（3 epoch） | 300 | 0.0000 | 0.0000 | 0.0000 | 2,696 s（CPU） | 238.7 ms |

> BERT+Linear 在 800 句 / 135 步、lr=2e-5 下内层验证 span F1 始终为 0（全 O 坍缩），
> 未收敛，不参与方法排名。同批 BiLSTM 用 3e-3、6 epoch 即可到论元 0.363，
> 说明小数据下「学习率 × 步数」比「模型是否预训练」更先决定能不能学起来。

文件：

- `metrics_*.json`：完整指标（含 per_type / per_role / 非法 BIO）
- `logs/`：逐 epoch loss / 内层验证 F1 / 控制台日志
- `checkpoints/bilstm_crf_sample.pt`：BiLSTM 权重（4.4 MB）
- 预测：`../predictions/local/pred_*_sample.jsonl`
