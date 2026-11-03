# 历史归档

当前每库仅一个协议，以[GT索引](../results/gt_index.json)为准：HML作者20FPS/263D、AIST++真实SMPL-X30、FineDance统一LODGE、BEAT2固定EMAGE2024兼容speaker2 TEST15。

此处保存先前源码、结果和验证回执；不进入当前默认评分、配方或报告。源码以`.py.txt`保存供追溯，没有可运行兼容入口。原分数、输入哈希和失败证据保持不变。[migration.json](migration.json)记录原路径、归档路径、SHA、原因、来源提交及替代入口。

| 原位置 | 归档位置 | 当前替代 |
|---|---|---|
| `score_worker.py`旧源码 | [beat_public2025/score_worker.py.txt](beat_public2025/score_worker.py.txt) | 当前score仅调用固定BEAT2024；AIST++走aist-smplx |
| `aist_smplx.py`旧源码 | [aist_smpl_control/aist_smplx.py.txt](aist_smpl_control/aist_smplx.py.txt) | 当前只评真实SMPL-X域 |
| `adapters.py`旧源码 | [aist_smpl_control/adapters.py.txt](aist_smpl_control/adapters.py.txt) | 删除已禁用的SMPL输出分支 |
| `results/aist_smplx30.json`完整SMPL对照 | [完整比较](aist_smpl_control/results/aist_smplx30_full_comparison.json) | [当前SMPL-X结果](../results/aist_smplx30.json)保留同一实测数值 |
| `results/beat_full_result.json` | [旧public2025/full265结果](beat_public2025/results/beat_full_result.json) | [BEAT2024 TEST15](../results/beat2024_speaker2_gt.json) |
| `results/beat_speaker2_result.json` | [旧public2025/speaker2结果](beat_public2025/results/beat_speaker2_result.json) | 同上；不混入其FGD |
| `results/public_gt_migration.json` | [前次发布回执](previous_release/results/public_gt_migration.json) | 当前验证见[VALIDATION](../VALIDATION.md) |

AIST++ TEST20＋VAL20、FineDance TEST19仍是当前同协议的补充诊断，不因集合不同而归档。完整BEAT2官方成员catalog也保留原划分；当前评测固定其speaker2成员。

归档仅涉及本公开包；原始实验服务器的旧Run与其冻结快照不被搬移或重写。
