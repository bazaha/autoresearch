# `score.py`：固定评分器（`evaluate_bpb` 的化学版）

> 给使用评分器的人看：怎么装、怎么跑、输入长什么样、每条规则怎么判。设计理由见 [`../README.md`](../README.md) 第 5 节；在循环里怎么用见 [`../docs/design.md`](../docs/design.md) 第 3、5 节。

只依赖 RDKit 与 NumPy，CPU 运行，确定性输出；4 核容器上给 3,000 张裁剪图打分约 5–8 秒（视分子大小，主要花在 RDKit 标准化上），层次 bootstrap 不到 1 秒。输出格式模仿上游 `train.py` 的结尾摘要，可以 `grep "^score:"`。它对 agent 是**只读**的。

## 1. 快速上手

```bash
cd research/drug-vlm                       # 以下命令都在本目录运行
uv pip install rdkit==2026.09.1 numpy      # 或 pip；版本不符时评分器直接退出，仅调试时可设 SCORE_ALLOW_RDKIT_MISMATCH=1
python3 -m unittest discover -s harness -p "test_*.py"

# 给一个工具打分（单分子裁剪图显式用 struct_f1；裁剪图和表格请分开打分）
python3 harness/score.py score --gold gold.jsonl --pred molscribe.jsonl --scalar struct_f1 --per-doc-out molscribe_perdoc.jsonl

# 两个工具/配方配对比较（两侧必须来自同一个 gold 文件，评分器会检查）
python3 harness/score.py compare --base molscribe_perdoc.jsonl --new qwen8b_perdoc.jsonl --scalar struct_f1 --alpha 0.05

# 多种子：逗号分隔，种子方差进入区间
python3 harness/score.py compare --base inc_s0.jsonl,inc_s1.jsonl,inc_s2.jsonl --new cand_s0.jsonl --scalar struct_f1 \
    --slice-weights inhouse=0.3,biovista=0.3,uspto=0.2,mrbw=0.2

python3 harness/null_sim.py 20             # 零效应模拟（20 次试验约 2 分钟；120 次约 10 分钟）
```

摸底阶段给现成工具打分时，为每个工具写一个约 30 行的转换脚本，把它的输出转成下面的 JSONL；`--per-doc-out` 导出逐条结果，`score` 会按来源切片（`slice`）和难度标签（`tags`）分别打印分数。

## 2. 输入格式

gold 与 pred 同构，JSONL，一行一个文档/页面/裁剪图。建议金标的 `value` 按原文字符串存储，以保留末位 0（它决定舍入区间）。

```json
{"doc_id": "US1234567B2-p12", "family_id": "US1234567", "slice": "uspto", "doc_type": "patent", "tags": ["chiral"],
 "image": "crops/US1234567B2-p12-1.png", "page": 12, "bbox": [102, 80, 380, 210],
 "compounds":    [{"id": "12a", "smiles": "CC(=O)Nc1ccc(O)cc1"}],
 "measurements": [{"compound_id": "12a", "target": "EGFR", "endpoint": "IC50", "assay_type": "biochemical",
                   "relation": "=", "value": "12.3", "unit": "nM"},
                  {"compound_id": "12a", "target": "EGFR", "endpoint": "% inhibition",
                   "value": 87, "unit": "%", "conc": 1, "conc_unit": "uM"},
                  {"compound_id": "12a", "target": "", "endpoint": "CL", "value": "25", "unit": "mL/min/kg",
                   "species": "rat", "dose": 1, "dose_unit": "mg/kg", "route": "iv"}]}
```

| 字段 | 说明 |
|---|---|
| `doc_id` | 唯一 |
| `family_id` | 专利族 / 文章（论文与其 SI 同族），bootstrap 的聚簇单位；默认等于 `doc_id` |
| `slice` | 只看金标：来源切片（`inhouse`、`biovista`、`uspto`、`mrbw`、`cn`……），用于加权与分组报告 |
| `doc_type` | 只看金标：文献类型（`patent`/`paper`/`si`），S2 用 `--group-field doc_type` 做宏平均 |
| `tags` | 只看金标：难度标签（`chiral`、`abbrev`、`large`、`lowres`……），一个文档可以有多个，单独报告 |
| `image`、`page`、`bbox` | 出处，评分器忽略（页面级联合指标是待建功能） |
| `compounds[].smiles` | 只接受 SMILES/CXSMILES。其他输出格式由 `harness/formats.py`（待建）先转成 CXSMILES，转换失败按无效结构计 |
| `measurements[]` | 见第 4 节 |

单分子裁剪图 = 只有 compound、没有 measurement 的文档，所以同一个评分器覆盖"单分子识别"和"表格/整页抽取"两个阶段。

## 3. 结构身份

| 问题 | 做法 |
|---|---|
| 主键 | RDKit 解析 → `FragmentParent`（去盐/溶剂）→ `Uncharger` → 标准 InChIKey（含立体层）。与输出语言无关——这就是 bpb"与词表无关"在本任务中的对应物 |
| 立体 | 严格：主指标区分立体；`struct_f1_nostereo`（InChIKey 第一段）只做诊断。外消旋/相对构型（CXSMILES `&1`、`o1`）按规范 CXSMILES 成键，不与绝对构型混为一谈 |
| Markush / R 基团 | `[1*]`、`[*:1]`、`$R1$`、`_R1` 四种写法统一为同一 R 编号；位置可变取代（CXSMILES `m:`）的"浮动取代基 + 可连接原子集"被显式编码进键里 |
| 金属 | 金属**与有机部分成键**时（配合物、有机金属）按原样用标准 InChI 成键（RDKit 的金属断键会把顺铂变成氨）；游离金属离子（如阿司匹林锌盐中的 Zn²⁺）按普通盐去掉 |
| 原子映射号、实原子上的显示标签 | 忽略 |
| 体量上限 | 单结构 > 500 重原子、单文档 > 5,000 个化合物或 > 20,000 条记录的部分记为无效，防止对抗性输出拖垮评分 |
| 缺失 | 只含结构的文档（裁剪图）上，**没有输出与答错同价**：精确率的分母至少是金标结构数，否则 F1 = 2c/(c+w+n) 会让"弃答"比"猜错"更划算 |

## 4. 活性/属性记录

**记录键**：(结构键, 终点, 靶点, 关系符, 单位类, 测试浓度) + 数值，全部正确才算一条正确记录；外加"金标定义"的上下文字段。

| 问题 | 做法 |
|---|---|
| 上下文字段（记录键 v2） | `assay_type`、`species`、`matrix`（或 `cell_line`）、`variant`、`ph`、`dose`/`dose_unit`、`route`、`time`。**只有金标填了的字段才计分**：推断出正确的物种有奖励，乱填没有，金标有而预测漏掉算错。常见写法归一：HLM → 肝微粒体 + 人；"SD rats""ICR mice""SD大鼠""ヒト" → 物种；"cellular assay""细胞水平" → cell；"IV bolus""尾静脉注射" → iv；"oral gavage""灌胃" → po；"L858R/T790M" = "T790M, L858R"；"wild type" = "WT" |
| 文字 | NFKC 归一（全角 → 半角）；保留拉丁、希腊（PI3Kα ≠ PI3Kβ；"PI3K alpha" = "PI3Kα"）、中文与日文假名；非空标签归一后变空则判为无法评分（失败即关闭，不会和别的空键相撞） |
| 终点列头 | 括号与列头里**只有被识别的内容才被移走**：单位（取最后一个）、统计量（n=3、± SD）、测试浓度（"(1 µM)" → 条件）、pH（"logD7.4""LogD(7.4)"）、孵育时间（"(72 h)"）、剂量（"(10 mg/kg)"）、物种/基质/实验类型/途径（"CL (rat, iv)"、"HLM CLint"、"Cellular IC50"）、写在靶点位置的靶点名（"EGFR IC50"）。**无法识别的限定词留在终点名里**（"IC50 (HCT116)" ≠ "IC50 (A549)"），所以不同的列永远不会塌缩成同一个键。转运方向（A→B/B→A、AP/BL、"A to B"）和 AUC 区间（0–∞、0–t、0–24 h）规范化后保留在名字里 |
| 终点同义词 | `IC₅₀`→`ic50`；`Kᵢ`→`ki`；`T½`/`half-life`/半衰期/半減期 → `t1/2`；抑制率/阻害率/inhibition rate → `%inhibition`；F%/bioavailability/生物利用度 → `f`；清除率/clearance → `cl` 等 |
| 单位 | 摩尔浓度统一到 nM（含 µmol/L、nmol/mL、微摩尔/升）；时间统一到 h（含 小时、分钟）；复合单位按量纲换算（mL/min/kg = L/h/kg = mL/(min·kg) = L·h⁻¹·kg⁻¹；ng·h/mL = µg·h/mL；µM·h = nmol·h/mL；M⁻¹s⁻¹）；单位里或列头里的十进制倍数（"×10⁻⁶ cm/s""(10⁻⁶ cm/s)""cm/s ×10⁻⁶"）；"protein" 等限定词忽略；% 与分数互换（fu 0.05 = 5%）；**质量浓度（µg/mL）用该记录结构的分子量换算成 nM**；读不懂的单位（含数字或未知词）只和完全相同的写法相等，永远不等于空单位 |
| 单位缺省 | `unit` 为空时依次取数值字符串里的单位（"10 µM"）、列头里的单位（"IC50 (µM)""IC50/nM"）；百分比类终点默认 %，fu 类默认分数 |
| pX | pIC50/pKi 等转为 nM 并翻转关系符；不合理的 pX（绝对值 ≥ 30）判为无效 |
| 数值字符串 | ">10""＜10""≦10"、"10–30""10~30""10 µM–30 µM"、"1,200"（千分位）与 "0,5"（小数逗号）、"12 ± 3""12 (n=3)""12 (10–15)"、"12*""12ᵃ"（脚注记号）、"1.2 × 10⁻⁵""1.2 X 10-8""1.2 E-8""1.2×10<sup>-8</sup>""(1.2 ± 0.3) × 10⁻⁸"、"–0.5"（各种减号）；溢出（"1e400"）判为无效记录，不会让整次评分崩溃 |
| 数值比较 | 同一标度：相对误差 ≤ 0.1%（要求如实转录）；经过非十进制换算（pX、质量浓度、分钟 ↔ 小时、mL/min/kg ↔ L/h/kg）：预测值必须落在**金标按原文印刷精度给出的舍入区间**内——只看金标的精度，否则模型可以靠"少写几位小数"放宽容差 |
| 关系符/区间/定性值 | `≥`→`>`、`~`→`=`；未知关系符各自独立（不会互相匹配）；"NT""n.d.""inactive"、分级 "A"/"+++" 作为类别值比较（连同关系符）——它们计入金标召回，模型不能靠省略非活性数据刷精确率 |
| 测试浓度 | `conc`/`conc_unit`、"% inh @ 1 µM"、"(1 µM)"、或字符串 "1 µM" 都会被解析；解析不了的浓度只与同样写法相等 |
| 匹配 | 同一键内做精确的二分图最大匹配（舍入区间可能嵌套，贪心会漏配）；单个键内金标 × 预测超过 25 万对时改用区间贪心（只会少算、不会多算），所以大量重复记录既拖不慢评分也刷不高分数；重复记录算假阳性；`;` 拼接多个编号不给分（计为孤儿记录） |
| 金标无法表示的条目 | 结构和记录都在两侧对称排除，并计数报告（`n_gold_unscorable`、`n_gold_rec_unscorable`） |

## 5. 复合分数与护栏

- `--scalar auto|composite|struct_f1|record_f1`；`auto` 在有活性金标时 = `0.5·struct_F1 + 0.5·record_F1`，否则 = `struct_F1`。S1 应显式使用 `struct_f1`。
- 护栏/诊断：`invalid_smiles_rate`（无效结构率，分母是原始预测数）、`multifragment_rate`（去除常见盐和溶剂后仍有多个片段）、`orphan_record_rate`（引用了未定义编号的记录）、`id_link_f1`（编号—结构配对）、`record_f1_by_label`（用编号代替结构匹配，区分"结构认错"与"关联错"）、Tanimoto、无立体 F1。比率都在 [0, 1] 内。
- 所有缺失或无法解析的预测按"空"计分；同一文档出现多份预测只取第一份（禁止 best-of-N）。
- RDKit 版本被钉死（`RDKIT_PIN = "2026.09.1"`），版本不符直接退出；`score` 输出金标与评分器的 SHA-256 前缀，并把金标哈希写进每一行逐条结果。

## 6. 统计

- **`score`**：按专利族聚簇的 bootstrap 95% CI，以及按 `slice` 和 `tags` 的分组分数。
- **`--slice-weights inhouse=0.3,...`**：主指标改为各来源分数的加权平均（S1 的 `val_struct`）。数据里出现了没有权重的来源切片会**直接报错**（防止拼写不一致悄悄改变分数），除非加 `--allow-unweighted-slices`；有权重但数据里没有的来源在剩余来源间重新归一化，并打印 `slice_weights_absent`。
- **`--group-field doc_type`**：用金标的另一个字段代替 `slice` 分组，例如 S2 按专利/论文/SI 宏平均：`--group-field doc_type --slice-weights patent=1,paper=1,si=1`。
- **`compare --base a,b,c --new x,y,z [--alpha 0.10]`**：层次配对 bootstrap。每次重采样整族抽取（论文与 SI 同族，跨来源切片的族作为一个整体、在"它跨越的切片组合"内分层），两侧共用（即配对），种子在各自一侧重采样，所以种子间方差会进入区间。输出 Δ、置信区间、`p_not_better`、每个种子的分数，以及每个来源切片、难度标签和护栏的配对增量与标准误。**方向**：护栏比率越低越好（上升超过 k·SE 才否决），分组分数越高越好（下降超过 k·SE 才否决）。各次运行必须针对同一个金标（金标哈希与 doc_id 集合都一致），否则报错，除非加 `--allow-subset`。只给一个种子时只能作为"初筛"。
- **旧做法"把种子合并后只按文档 bootstrap"是反保守的**。`null_sim.py`（3,000 张图、375 个专利族、每侧 3 个种子、每组 120 次零效应试验）的结果：

| 种子间标准差 | 种子合并 + 文档 bootstrap | 层次（专利族 × 种子）bootstrap |
|---|---|---|
| 约 0.49 pp | 2.5% | 0.8% |
| 约 0.56 pp | 7.5% | 2.5% |
| 约 0.70 pp | 10.8% | 1.7% |

（名义误保留率 5%；评委在种子方差更大的设定下测得合并做法误保留 15–22%。新做法偏保守，这对"保留门槛"是合适的方向。）

一个直观的噪声量级：在 60 个文档（120 个结构、120 条记录）的模拟评测上，score ≈ 0.41 的 95% CI 是 [0.34, 0.48]，半宽约 ±0.07。评测集如果只有几十页，0.02–0.03 的"提升"完全淹没在噪声里。

## 7. 修复记录

评分器在交给 agent 之前经过了评委、审稿和红队 agent 的多轮攻击，每个可复现的漏洞都补了测试：

| 版本 | 修复 |
|---|---|
| v1.1–v1.4 | R 基团四种写法统一；外消旋 ≠ 绝对构型；位置可变 Markush 不再被剥成母核；顺铂不再被剥成氨；pX 换算容差不能靠少写小数放宽；种子方差进入配对检验 |
| v1.5 | "12 ± 3""10 µM""1.2 × 10⁻⁵"、列头单位；来源加权与分层 bootstrap；分组与护栏的配对增量与 SE；`--alpha`；记录侧对称排除；游离金属离子去盐；二分图最大匹配 |
| v1.6 | 中文/希腊字符不再被删成空键（"抑制率" ≠ "清除率"、PI3Kα ≠ PI3Kβ）；记录键 v2 的上下文字段；复合单位按量纲换算；单位与列头里的十进制倍数 |
| v1.7 | 裁剪图上缺失与错答同价（弃答不再更划算）；比率不再超过 1；论文与 SI 整族重采样；来源权重拼写检查；`compare` 拒绝不同金标；`--group-field`；`null_sim.py` 修复。红队发现的约 30 处列头/单位/数值问题：括号里的测试浓度与剂量、未识别限定词塌缩、AUC 区间、转运方向、NFKC 与日文、物种/品系/途径/实验类型的写法、多种倍数与指数写法、小数逗号、`~` 区间、全角关系符、脚注记号、字符串浓度、中文单位词、% 与分数、溢出不再崩溃 |

## 8. 仍未覆盖

记录键中的靶点同义词 → UniProt 表（`targets_v1.tsv`）；专利分级活性的图例区间（`value_alt`）；重复测定（n、±SD）作为诊断；`golden_pairs.tsv` 回归集与缩写表一致性测试；Markush 取代基表 F1 与 Sg（频率可变）语义；盐型诊断（ChEMBL Structure Pipeline）；E/Z 与阻转异构体（InChI 不区分阻转异构）；逐条置信度与校准；没有 SMILES 的金标的图级评测；页面级"检测 IoU ≥ 0.5 且结构正确"的联合指标。列头解析是启发式的：训练与评测时，记录应把裸终点名写进 `endpoint`、上下文写进字段（`records_yaml_v1` 与标注指南都这样约定），列头原文另存为 `endpoint_as_printed`。
