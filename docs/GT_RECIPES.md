# 与已报告GT对应的四套入口

每库仅一个当前协议，下面四条是唯一当前配方；不能按分数选择版本。原始结果SHA、完整精度和统计重复保存在`results/`；[gt_index.json](../results/gt_index.json)绑定配方、脚本与结果。每个源的名字和角色见[gt_eval_members.json](../metadata/gt_eval_members.json)，输入SHA与格式见[members.jsonl](../metadata/members.jsonl)。历史对照仅在[legacy](../legacy/README.md)追溯。

| 数据集 | 入口脚本 → 作者实现 | GT口径 | 对应结果 |
|---|---|---|---|
| HumanML3D | [score_worker.py](../score_worker.py) → T2M `final_evaluations.py` | 作者20 FPS/263D、TEST4198、20次、epoch28、batch32、unit4、Div300 | [hml_gt20.json](../results/hml_gt20.json)，取`candidate_native_gt` |
| AIST++ | [aist_smplx.py](../aist_smplx.py) → Bailando指标＋显式30 FPS时间 | 两端真实SMPL-X30、完整1363参考、TEST20；40条补充单列 | [aist_smplx30.json](../results/aist_smplx30.json)，取`smplx_test20_vs_full1363` |
| FineDance | [lodge_gt.py](../lodge_gt.py) → LODGE `metrics_finedance.py` | 作者发布202源/404份特征，原normalization和特征常数 | [finedance_lodge202_gt.json](../results/finedance_lodge202_gt.json)，Divk9.728692/Divg7.441713 |
| BEAT2 | [beat2024.py](../beat2024.py) → EMAGE2024 `utils/metric.py` | speaker2 TEST15、旧调用顺序/时间裁剪/音频；body＋hands | [beat2024_speaker2_gt.json](../results/beat2024_speaker2_gt.json)，BC0.686717/L1div13.070166 |

```bash
python evaluate.py gt --list
python evaluate.py gt --dataset HumanML3D --output runs/hml_gt20
python evaluate.py gt --dataset 'AIST++' --output runs/aist_smplx30
python evaluate.py gt --dataset FineDance --output runs/lodge_gt202
python evaluate.py gt --dataset BEAT2 --output runs/beat2024_gt15
```

先按[README](../README.md)配置合法取得的资产。BEAT2旧版须先执行`python evaluate.py prepare-beat2024`；原作者文件及normalizer逐字节验SHA，不改指标。每次使用新输出目录。

## FineDance统一LODGE

GT主表用作者发布的202源特征池。`lodge-gt`逐个校验404个文件，无缺失/额外文件后调用作者`quantized_metrics(pool,pool)`，直接报告返回的`div_k_gt`/`div_m_gt`。自身FID只作为identity诊断；论文GT FID未给，保持空值。

评模型时也使用同一份参考池：

```bash
python evaluate.py score --dataset FineDance --split test \
  --manifest membership_v1/manifest.jsonl --arm native_gt \
  --reference-features assets/lodge_published_reference_v1 \
  --output runs/lodge_loader19_diagnostic
```

这是19条loader有效GT的诊断，不是202池GT行。原动作须走LODGE预处理、grounding、ROW6D、FK、1024帧cap；不换成FineNet指标。生成/重建可用同一`score`入口及独立arm，要求每条对应的预测文件、预处理起帧和完整时间支持。推理源码20源与loader19源差`120`，两份原名单保留，不宣称论文最终输出成员已精确确认。

## BEAT2版本与显示单位

2024兼容入口保留L1div原地改写后BC复用的历史行为、0.3阈值、整秒/8帧/64+60k时间支持，以及body＋hands。15源共28680动作帧、26880 BC帧。raw BC0.686717133568按主表raw×10显示为6.867171335676；距论文6.896约0.42%，L1div13.070166距13.074约0.029%，已接受近似复现。

当前固定入口仅BC/L1div，不提供版本选择项。先前public2025的FGD及BC/L1div已移到[legacy](../legacy/README.md)，不进入当前报告。重建/生成的手须来自对应输出，不用GT手填充。

## HumanML3D和AIST++

HML归档完整20次而非13次中间均值；4198个源经作者caption loader成为4648条，batch32/drop_last后每轮4640条。GT两条分支分别采样caption/crop，结果分支固定为`candidate_native_gt`，不择优选分支。

AIST++为用户指定的SMPL-X30协议：原生60 FPS直接`[::2]`，不插值；默认shape/手、世界pelvis保持、固定24点。论文SMPL60数值单列，不能直接作相同口径比较。TEST20＋VAL20的40条结果只作补充，角色不改为40条TEST。

2026-11-06起所有当前AIST入口共用`assets/aist_smplx30_reference1363`，逐文件验SHA及1363成员。
参考源按作者camera mapping减ignore确定，整段提特征；训练仍只用TRAIN952。
旧TRAIN952结果保留在`legacy/aist_train952/`，不得与新分数混表。60FPS参考特征、原生SMPL特征均不可代替本池。

仅BEAT2评手。其他三库body，默认指姿态、保留腕/臂。原生GT、SOMA往返、Tokenizer重建、生成必须分开打分；本发布不把GT结果当SOMA重建验收。
