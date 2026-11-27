# 预测文件（predictions）

所有方法统一输出 `{"id","sent_id","events":[...]}` 的 JSONL；评测键是 **id + sent_id**
（DuEE-Fin 的 `id` 是公告 id，会重复，不能只用 id 对齐）。

| 子目录 | 内容 | 来源 |
|---|---|---|
| `disc/` | 规则、BiLSTM-CRF、BERT+Linear、BERT+CRF、硬约束、两段式，以及消融/冒烟预测（16 个文件） | 全量数据训练后的已保存预测，用于 sample 子集复评 |
| `llm/` | API zero-shot / few-shot、本地 0.6B zero-shot、LoRA SFT（4 个文件） | 全量/固定 n=200 的已保存预测 |
| `local/` | 规则、BiLSTM-CRF、BERT+Linear 在 sample 上重训后的预测（3 个文件） | 本次本地重训（sample 800 句） |

说明：

- `disc/` 与 `llm/` 是**全量训练模型**的预测，不是 sample 重训；报告中标注为“已保存参考预测”。
- `local/` 才是本次在用户提供的 sample 数据上真正训练出的预测。
- 统一评测脚本：`src/common/evaluate_events.py`；复评入口：`src/eval/score_subset.py`。
