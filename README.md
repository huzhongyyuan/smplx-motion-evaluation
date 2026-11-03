# Four-dataset motion evaluation

四数据集动作评测：HumanML3D、AIST++、FineDance、BEAT2。提供自有CPU包装脚本、固定作者版本、成员与SHA元数据、GT实测结果、SMPL-X格式适配和协议说明。

完整中文说明：[数据集评测说明.md](数据集评测说明.md)。数据获取：[docs/DATA_ACCESS.md](docs/DATA_ACCESS.md)。第三方条款：[THIRD_PARTY.md](THIRD_PARTY.md)。

**每个数据集只有一个当前评测协议。** 四套入口与已报告GT一一对应，命令、结果分支、成员名单和结果SHA见[GT脚本索引](docs/GT_RECIPES.md)及机器可读的[gt_index.json](results/gt_index.json)。验证范围见[VALIDATION.md](VALIDATION.md)。表中数值均为GT评测；SOMA/Tokenizer往返另验。旧BEAT2公开版、AIST++的SMPL对照和旧发布回执已归档到[legacy](legacy/README.md)，不参与当前评分或报告。

## 表示与主指标

| 数据集 | 评分输入 | 实现 | 范围 |
|---|---|---|---|
| HumanML3D | 作者20 FPS/263D | T2M | body |
| AIST++ | 真实SMPL-X30，固定24点 | Bailando公式＋显式30 FPS时间参数 | body；与SMPL60论文协议分开 |
| FineDance | LODGE发布202源GT特征池；动作侧用原预处理/FK | LODGE原版 | body；TEST19诊断单列 |
| BEAT2 | SMPL-X30、speaker2 TEST15 | 固定EMAGE2024兼容BC/L1 | body＋hands；当前协议不含FGD |

**不是四库全都原生SMPL-X。** 只有BEAT2评手指；其他三库手指默认，保留手腕和手臂。

## 安装与合法获取资产

Python3.10；数值核心固定NumPy1.23.5、SciPy1.10.1、Torch2.6.0。完整实测环境快照见`docs/measured-environment.txt`，它是环境记录，不要求安装其中所有无关包。

```bash
git clone https://github.com/huzhongyyuan/smplx-motion-evaluation.git
cd smplx-motion-evaluation
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python evaluate.py fetch --list-only
python evaluate.py fetch
```

95份上游源码及成员证据按作者URL和revision下载后逐个验SHA，已实测全新获取成功；不下载无关训练资产或示例动作。上游各自许可适用；尤其Bailando为非商业条款，LODGE未找到明确再分发许可。此仓库不重新打包它们。

从作者渠道获取原始动作/文本/音频、T2M权重与GloVe、EMAGE资产、LODGE预计算特征；SMPL/SMPL-X模型需自行按其条款取得。按[数据获取说明](docs/DATA_ACCESS.md)整理路径并填写`config.local.json`，可从`config.example.json`复制。

```bash
python prepare_local_assets.py --help
python evaluate.py configure --config config.local.json
python evaluate.py prepare-beat2024
python evaluate.py check
```

`configure`校验30,582个源的存在、作者HML263D的SHA和关键资产SHA，生成本机绝对路径manifest；这些本机路径、数据和模型均被Git忽略。配置不修改原始数据。

## 运行

```bash
# 查看每个入口对应哪份GT、什么协议及历史结果
python evaluate.py gt --list

python evaluate.py gt --dataset HumanML3D --output runs/hml_gt20
python evaluate.py gt --dataset 'AIST++' --output runs/aist_smplx30
python evaluate.py gt --dataset FineDance --output runs/lodge_gt202
python evaluate.py gt --dataset BEAT2 --output runs/beat2024_gt15
```

详细speaker2、HumanML30 FPS诊断、条件计划、SMPL-X适配及SOMA/Tokenizer验收见中文说明和`--help`。每次使用全新输出目录，不覆盖结果；不自动重复启动失败Run。

## 当前GT结果

| 协议 | 已有结果 | 说明 |
|---|---|---|
| AIST++ SMPL-X30 TEST20 / TRAIN952 | FIDk28.568973、FIDg16.844502、Divk8.295857、Divg8.088570 | 30 FPS适配协议；非原SMPL60论文复现 |
| FineDance作者202源参考池 | Divk9.728692、Divg7.441713 | 对作者发布特征重算，按论文精度为9.73/7.44；未证明202条原动作全量重提一致 |
| BEAT2 EMAGE2024兼容speaker2 TEST15 | BC raw0.686717（×10为6.867171）、L1div13.070166 | 用户接受近似复现；不含旧版FGD |
| HumanML3D作者20 FPS TEST4198，20次 | R@1/2/3 0.512672/0.702338/0.797683；FID0.001632；Matching2.971407；Div9.550798 | 最终20轮，95%CI与每轮结果已归档 |

同协议补充诊断：AIST++的TEST20＋VAL20为40条，FineDance的loader有效TEST为19条；这些是样本集合差别，不是另一套打分版本。数值与成员范围见[中文说明](数据集评测说明.md)。当前36项核心回归＋3项作者资产回归通过，AIST真实3源与FineDance真实2源迁移控制一致，见[验证记录](VALIDATION.md)。舞蹈FID若有微小复数，完整JSON保留实虚部。分数接近不等于完整协议复现。原生GT打分、转换保真、重建质量、生成能力分别验收。

## 仓库内容与许可

`metadata/`是源ID、划分、格式和校验信息；`results/`是汇总实测与输入身份；没有原始动作、音频、文字、完整特征数组、人体模型或checkpoint。不能因自有包装代码采用MIT而重新授权这些资产。下载原始数据请走作者渠道。

源代码职责：`evaluate.py gt`固定四库GT配方；`score_worker.py`调用作者打分器；`aist_smplx.py`负责SMPL-X30；`lodge_gt.py`核对LODGE202池并调用原指标；`beat2024.py`固定历史调用顺序；`adapters.py`/`contracts.py`处理输入契约；锁文件固定作者版本与资产。未提供SOMA↔SMPL-X转换器。
