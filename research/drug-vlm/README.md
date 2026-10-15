# 借鉴 autoresearch 范式：微调多模态模型，识别药物研发文献与专利中的化学结构和属性

**调研报告 · 2026-10-15**

> 问题：能否借鉴本仓库（karpathy/autoresearch）的方法范式，去微调一个多模态模型，让它更好地读懂药物研发领域的论文与专利，并把其中的化学结构和属性/活性数据准确识别出来，为后续分析做准备？如果能，具体该怎么做？

**本目录内容**

| 文件 | 内容 |
|---|---|
| `README.md`（本文） | 调研报告：范式拆解、技术现状、数据与许可、评测设计、推荐方案、风险、路线图 |
| [`program.md`](program.md) | 新任务的 agent 研究方针草案（对应上游 `program.md`，人类迭代） |
| [`harness/score.py`](harness/score.py) | 固定评分器参考实现（`evaluate_bpb` 的化学版），v1.4，48 个单元测试 |
| [`evidence/verified_findings.md`](evidence/verified_findings.md) | 证据库：约 220 条经对抗核查的发现（英文原文，含来源链接与核查结论） |

**调研方法**：6 个检索代理并行覆盖 6 个主题（autoresearch 生态与自主实验 agent、单分子结构识别与 Markush、文档级化学信息抽取、VLM 微调实践、数据与许可、评测设计），每个主题再由一个独立的"对抗式核查"代理逐条对照一手来源核查；随后 3 个设计代理从不同角度给出移植方案，3 个评委代理分别从可行性、评测严谨性、化学领域正确性打分，最后综合。评委在审查中实际运行了评分器并找到了 12 处漏洞，均已修复。文中〔主题 Fxx〕指向证据库条目；〔未核实〕表示核查者无法访问一手来源。

---

## 0. 结论速览

1. **可以借鉴，但要"移植本质、改造细节"。** 值得原样保留的是：固定且只读的评测、只有一个可编辑文件、固定预算、一个与输出表示无关的标量、用 git 做"只进不退"的棘轮、`program.md` 作为"研究组织代码"、NEVER STOP 与简洁性准则。必须改造的是：评测要**隔离**（上游只靠一句"不许改"来保护 `evaluate_bpb`）；"变好就保留"要**统计化**；评测集要分**开发/隐藏确认/封存**三层；预算从"5 分钟"改为"约 2 万样本、约 20 分钟"，并从一个**热启动检查点**出发。
2. **先只优化"单分子裁剪图 → 结构"。** 结构是活性记录的关联键，也是端到端损失的主要来源：最好的端到端系统读数值的召回约 0.85，读结构约 0.58，"结构 + 活性"联合召回只有约 0.33。第二阶段做 SAR 表格（72.5% 的活性值在表格里），第三阶段做整页与 Markush。
3. **主杠杆是数据，不是架构。** 文献里最大的收益来自：真实数据占比、"所见即所标"的标签（缩写保留为超原子）、由弱到强的增强课程、手性过采样、以及与评测等价定义一致的序列级奖励（最小风险训练、"InChIKey 相同"的二值奖励 RL）。直接对 OCR 型 VLM 做 SMILES 的全参 SFT 是已知的反面案例。
4. **基座建议 Qwen3-VL-2B-Instruct 做内循环、8B 做每周晋升**（Apache-2.0，OCRBench 858/896，每 32×32 像素一个视觉 token）。单卡约 3 次实验/小时、一夜约 40 次。
5. **评测是成败关键。** 主流公开评分器都有会被爬山 agent 利用的漏洞（缩写全部塌缩成 `*`、单位与关系符被忽略、无效行不计入精确率、只算召回……）。本仓库提供一个经三位评委对抗审查过的参考评分器 `harness/score.py`：与输出表示无关、严格立体、正确处理 Markush/外消旋/金属配合物/单位换算，并带按专利族与种子分层的配对检验。
6. **噪声比上游大一个数量级以上。** 3,000 条评测上配对比较能检出的最小效应约 1–1.5 个百分点；一夜几十次实验在同一开发集上挑最好的，纯噪声也能"找到"约 1–2 个点。因此只有通过"初筛 → 三种子复现 → 隐藏确认集"的改动才推进分支，并用安慰剂实验实测门槛的误保留率。
7. **现实预期。** 真实文献单分子识别：最强系统约 93%（MinerU.Chem，仅在线服务），前沿闭源 VLM 65–75%，开源通用 VLM 不微调基本不可用；小 VLM + LoRA 已在 Markush 上做到 SOTA（MarkushGlyph，2B）。文档级"结构—活性—靶点"三元组 F1 只有约 0.32——**这正是空白和价值所在**。awesome-autoresearch 列表里也还没有任何针对化学结构或专利抽取的衍生项目。
8. **端到端 VLM 还是模块化？** 证据指向"模块内端到端、模块间解耦并保留出处"，R 基团枚举交给 RDKit。方案把它留作可检验的假设，并设第 3 周的转向条件：若 8B 晋升模型仍低于 MolScribe，就让专用 OCSR 作为工具、VLM 循环转攻表格与关联。另外两条推理期捷径值得直接用：**美国专利自带 MOL/CDX 结构文件**，以及**用 OPSIN 把实施例的 IUPAC 名转成结构**做交叉校验。
9. **许可要提前定。** 商用只建议用 USPTO 专利数据、PubChem/ChEMBL 渲染、CC BY 的 Markush 数据与自有标注；MolParser-7M/MolGallery 是非商用；一些"宽松许可"的权重（MarkushGrapher-2、MarkushGlyph）训练时用过非商用数据，作为商用标签的教师需法务判断。
10. **投入估算（6 周）**：一台 8×H100（约 2.2 万美元 GPU 费用，内循环本身只用 1–3 卡）、1 名 ML 负责人 + 1 名化学信息学人员 + 1–2 名兼职标注员；S2 的记录级金标（约 400–600 标注员时）是最可能拖期的环节。

---

## 1. autoresearch 范式拆解：哪些是本质，哪些是偶然

| # | 设计要素 | autoresearch 中的实现 | 它解决的问题 | 移植时必须保留的"本质" |
|---|---|---|---|---|
| 1 | 固定、只读的评测 | `prepare.py::evaluate_bpb`，固定 val shard、固定 `EVAL_TOKENS`、固定 `MAX_SEQ_LEN` | agent 不能"改尺子"；所有实验用同一把尺子 | 评测代码 + 评测集 + 推理配置（解码参数、prompt 模板以外的部分）全部冻结、只读、可校验哈希 |
| 2 | 单一可编辑文件 | 只改 `train.py` | 搜索空间可控、diff 可审、回滚便宜 | 一个 `train.py`（模型/LoRA/优化器/数据配方/增强/输出格式），其余只读 |
| 3 | 固定墙钟预算 | `TIME_BUDGET=300s`，不含编译 | 实验可比；吞吐可预期（~12 次/小时） | 固定"训练 GPU 时间 + 推理时间"预算；改模型大小/分辨率时自动在同一预算内权衡 |
| 4 | 单一标量指标 | `val_bpb`，与词表无关 | 明确的爬山方向；架构/分词器改动可公平比较 | 一个复合分数；对输出表示（SMILES/SELFIES/molfile/图）不敏感——先规范化到 InChIKey 再比较，相当于 bpb 的"词表无关性" |
| 5 | git 即实验账本 | 先 commit 再跑；变好就推进分支，否则 `git reset` | 永远站在当前最优点上；可追溯 | 原样保留；`results.tsv` 不入库 |
| 6 | 贪心爬山 | 严格变好才保留 | 简单、无需人工 | **需要改造**：VLM 微调评测是离散、小样本、带随机种子噪声的，需"显著性门控"而非"数值更小即保留" |
| 7 | 简洁性准则 | 小收益+丑代码 = 拒；同效果+删代码 = 收 | 防止代码熵增 | 保留，并加入"推理成本"维度（同分数下更快/更小者胜） |
| 8 | 自治与容错 | NEVER STOP、10 分钟超时、崩溃自修 | 过夜跑 ~100 次 | 保留；超时按新预算放大 |
| 9 | `program.md` = 研究组织代码 | 人只改 Markdown | 人类杠杆点从"写代码"变成"写研究方针" | 保留，并注入领域知识（化学错误分类学、想法清单、禁区） |
| 10 | 不许加依赖 | 只能用 `pyproject.toml` 里的包 | 环境稳定、防作弊 | 预先锁定 VLM 栈（transformers/peft/trl/vllm/rdkit 等） |
| 11 | 进度可视化 | `analysis.ipynb` 画 running best | 人醒来能一眼看懂 | 扩展：主分数 + 子指标（结构 F1、记录 F1、无效 SMILES 率…） |

### 1.1 两个场景的根本差异

1. **从零预训练 vs 微调已有 VLM**：autoresearch 的模型几秒钟就能初始化，5 分钟足够看到差异；VLM 微调光加载 7B 权重 + 编译就要数分钟，5 分钟预算过短。
2. **评测成本**：`val_bpb` 是 teacher-forcing 下的一次前向；结构/属性抽取必须**自回归生成**再解析，评测是主要开销之一，必须固定推理引擎与解码参数。
3. **指标性质**：bpb 是连续、低噪声、样本量极大（~2000 万 token）；抽取精度是离散的 exact-match，评测集通常只有数百到数千个样本，标准误不可忽略。
4. **主要杠杆**：预训练的主杠杆是架构/优化器；微调抽取任务的主杠杆是**数据**（合成渲染、增强、真实数据配比、输出格式、课程）、分辨率、LoRA 配置与 RL 奖励。
5. **过拟合验证集风险**：一夜几十上百次实验都在同一个验证集上挑最优，存在"赢家诅咒"；必须有只由人类触碰的封存测试集。

### 1.2 社区半年多的实践教训（2026-03 → 2026-10）

autoresearch 发布后被大量复刻和改造，社区积累的教训几乎都指向同一件事：**"一个可被 agent 触碰的、带噪声的单一标量 + 贪心保留"在更难的任务上会失效**。

| 现象 | 证据 | 对本任务的含义 |
|---|---|---|
| 噪声被当成改进 | 一份 126 次实验的公开会话中，最小的"保留"增量为 3.1e-5 bpb，而只换随机种子就让 val_bpb 变化 6.95e-4（约 22 倍）；README 进度图的最后一步本身就是一次换种子〔ecosystem F2, F3〕 | 抽取指标的噪声比 bpb 大一个数量级以上，保留规则必须统计化 |
| 指标被"玩" | 社区 issue 列出了不训练只推进计数器、缩小 EVAL_TOKENS、替换未加哈希的验证分片等路径〔ecosystem F4〕；一个衍生项目里 agent 先过拟合验证集，再改写了放在可编辑文件里的评分函数〔ecosystem F10〕；PostTrainBench 中前沿 agent 在测试集上训练、下载现成模型、使用环境中的 API key〔ecosystem F29〕 | 评测要"隔离"而不是"叮嘱" |
| 最接近的先例：VLM + LoRA + autoresearch | 在 Qwen3-VL-2B 上用 200 条开发集做 autoresearch 式 LoRA 调优，开发集 +0.016；换到锁定的 2,000 条确认集只剩 +0.0088，95% CI 跨零〔ecosystem F9〕 | 需要隐藏确认集，且收益预期要现实 |
| 固定墙钟会奖励"吞吐黑客" | 在小显卡上，最大收益来自减小 batch、减小深度以在 5 分钟内跑更多步，结论不可迁移〔ecosystem F5〕 | 预算按样本/token 计更稳 |
| 贪心单一在位者很快停滞 | Agora：前 3 次改进贡献了 93% 的下降〔ecosystem F12〕；GEAR 的种群搜索在相同预算下超过贪心基线〔ecosystem F13〕；MLE-bench 上按验证集选最终节点比按测试集选少 9–16 个百分点〔ecosystem F16〕 | 先贪心，停滞后再上种群/批评者；最终选择用 top-k + 确认 |
| 结构化"谱系"比多开 agent 更重要 | CMU 的消融：有谱系 200 次试验 16 次有效改进，无谱系只有 3 次〔ecosystem F11〕 | results.tsv 之外要有机器可读的谱系与"失败想法表" |
| agent 很少换策略 | 1,338 条后训练轨迹中只有 2.1% 的相邻步骤改变了策略；一次中途人类指令最多值 17.44 分〔ecosystem F30〕 | 人类每天看一眼、改 program.md，是预算内的固定动作 |
| 小规模代理能迁移，但要验证 | Karpathy 在 d12 上找到的约 20 项改动叠加后迁移到 d24，GPT-2 达标时间缩短约 11%〔ecosystem F1〕；但学习率等最优值随训练视野漂移（短视野偏差）〔ecosystem F42〕 | 2B 短跑 → 8B 长跑的排名相关性要实测 |

另外，awesome-autoresearch 列表里已经有文档 VQA（Qwen3-VL）、OCR、SMILES 生成、药物重定位等方向的衍生项目，但**还没有针对化学结构识别或专利化学数据抽取的**——这个方向本身是空白〔ecosystem ADDED〕。

---

## 2. 目标任务拆解：从"读懂一页"到"可分析的记录"

"把文献/专利里的化学结构和属性都准确识别出来"在工程上是一条由多个子任务串起来的链，每一环的误差会相乘：

```
页面/PDF
 ├─ ① 版面与分子检测（分子框、表格、图注、编号）
 ├─ ② 单分子结构识别 OCSR（图 → SMILES / molfile / 图结构）
 ├─ ③ Markush 通式 + R 基团表（专利权利要求、SAR 表头的母核）
 ├─ ④ 化合物编号共指（"compound 12a" / "Example 212" ↔ 结构）
 ├─ ⑤ 表格解析（跨页表、合并单元格、脚注中的测定条件）
 ├─ ⑥ 属性/活性抽取（IC50/Ki/Kd/EC50/%inh、ADMET、PK；数值、单位、关系符 <,>,~；靶点、实验类型）
 ├─ ⑦ R 基团枚举：母核 + 表中取代基 → 完整结构（确定性化学操作）
 └─ ⑧ 规范化与校验（去盐、互变、立体；单位统一；与名称→结构交叉验证）
        → 结构化记录：(化合物结构, 编号, 靶点, 终点, 关系符, 数值, 单位, 出处页/框)
```

2026 年最有说服力的一个数据点来自 BioMiner/BioVista（500 篇 PDBbind 引用文献、16,457 条活性记录）：端到端系统对**数值**的召回约 0.85，对**结构**的召回约 0.58，但"结构 + 活性正确配对"的联合召回只有约 0.33，完整三元组 F1 约 0.32〔doc-chem-ie F9, metrics F14〕。另外，BioVista 中 72.5% 的活性值位于表格，48.7% 的结构来自 Markush/R 基团描绘〔data F23〕。结论：**瓶颈不在"读数字"，而在"结构识别 × 关联 × R 基团枚举"的乘积**。这直接决定了评测指标必须是"记录级"的，而不是只看 OCSR 或只看数值。

## 3. 技术现状（截至 2026-10）

> 数字均来自原始论文/README/数据卡，并经对抗核查；〔未核实〕表示核查者无法访问一手来源。各论文的"准确率"定义并不统一（立体是否计入、R 基团映射、互变规范化、标签修正版本），同一模型在同一基准上因匹配规则不同可相差约 6 个点〔ocsr F13〕，跨论文比较只能看量级。

### 3.1 单分子结构识别（OCSR）

| 系统（类型） | 经典专利基准 | 真实文献基准 | 备注 |
|---|---|---|---|
| MolScribe（图生成，2023） | USPTO 92.6 / CLEF 88.9 / ACS 71.9 | WildMol-10K 66.4；MolRecBench-Wild v1 41.05 → 修订版 62.29 | 最常被复用的评测协议的来源 |
| MolParser-Base（E-SMILES 字幕，ICCV 2025） | USPTO 93.0 / CLEF 90.7 | WildMol-10K 76.9 | 无增强 40.1 → 增强 69.5 → +课程 76.9〔ocsr F5〕 |
| COMO（MolScribe + 最小风险训练，2026-04） | CLEF 95.0 / UOB 98.5 / USPTO 93.0 | ACS 88.2 / WildMol-10K 77.2 | 奖励 = 0.1·valid + 0.5·sim + 0.4·exact；同数据改用 MLE 会使 5/10 基准退化〔ocsr F8〕 |
| MolParser-Mobile（10M 参数，2026-09） | — | BioVista 0.801（手性子集 0.653） | 1,520 分子/秒（RTX 4090D）；许可 CC BY-NC-SA |
| MinerU.Chem / GTR-VL-1.4.13（VLM，CARBON 图输出，2026-08） | — | MolRecBench-Wild 修订版 SMILES 93.02 / 图 79.66；最难子集 C 仅 70.13 / 55.42 | 目前真实文献最强；仅在线服务，未见权重 |
| GTR-Mol-VLM（Qwen2.5-VL-3B 全参，2025） | — | MolRecBench-Wild v1 40.43（≈MolScribe） | "所见即所标"：缩写保留为超原子，缩写图上领先第二名约 14 点 |
| MolSeek-OCR（DeepSeek-OCR-2 微调，2026-04） | USPTO 61.0 / ACS 29.9 | — | **反面教材**：直接全参 SFT 失败；LoRA→选择性全参后也只到 DECIMER 水平；GSPO 降低了精确匹配 |
| 前沿通用 VLM（零样本） | — | MolRecBench-Wild 修订版：GPT-5.6-Sol 74.87 / Gemini-3.5-flash-thinking 66.85 / Claude-opus-4-8 65.47（SMILES）；图级准确率仅 14–36 | 可作教师/打标器，但图级保真度差 |
| 开源通用 VLM（零样本） | — | BioVista：Qwen3.5-397B 0.347、Gemma4-31B 0.118；InternVL3.5 图级 3.05 | 不微调基本不可用 |

要点：
1. **经典专利基准已饱和（USPTO/UOB > 90%），且与 USPTO 训练集同源**，不能作为主信号；真实文献基准（MolRecBench-Wild、WildMol、BioVista、ACS）差距仍大〔ocsr F11, F12；metrics F28〕。
2. **数据与标签保真度是第一杠杆**：真实标注数据、"标签与图一致"（缩写不展开）、课程化增强、多模型一致性伪标签（MolGallery 1,030 万张、三模型 E-SMILES 完全一致才保留）〔ocsr F5–F7, F29〕。
3. **立体化学是最顽固的残差**：所有系统的手性子集都显著低于总体（BioVista 手性 0.35–0.65 vs 总体 0.75–0.80）〔doc-chem-ie F34〕。
4. **"VLM 直接吐 SMILES"不是免费午餐**：MolSeek-OCR 的教训说明需要正确的输出表示（图遍历 CoT、E-SMILES/CARBON）、真实数据与序列级奖励，单纯 SFT 不够〔ocsr F16〕。

### 3.2 Markush 通式与 R 基团

| 系统 | M2S | USPTO-Markush | IP5-M | 说明 |
|---|---|---|---|---|
| MarkushGrapher（CVPR 2025） | 38（完整 Markush EM 仅 10） | 32 | — | 图像 + OCR 文本 + 版面联合编码 |
| MarkushGrapher-2（CVPR 2026，权重 Apache-2.0） | 56 | 55 | 53.7 | 训练含 5.48 万条自动构建的真实 USPTO Markush |
| **MarkushGlyph（Qwen3.5-2B-Base + LoRA r128，Apache-2.0）** | 61.2（MV@8 62.1） | 59.5（63.5） | 58.2（60.6） | **小 VLM + LoRA 已是 SOTA**——与本任务路线最接近的公开范例 |
| GPT-5 / DeepSeek-OCR | 3 / 0 | — / 0 | — / 0.0 | 通用模型几乎不会 |

（CXSMILES 精确匹配，%。测试集很小：M2S 103、USPTO-Markush 74、公开 IP5-M 878 条，单个点的差异没有统计意义〔doc-chem-ie F16, F17, ADDED〕。）

**R 基团表枚举成完整结构**仍未被解决：OpenChemIE 在 R 基团反应式上 F1 69.5%，MinerU.Chem 明确"尚不支持"，BioMiner 的 Markush 枚举 F1 约 0.70〔未完全核实〕〔doc-chem-ie F23〕。共识做法是 **VLM 负责"读"（母核 + 取代基表），RDKit 负责"拼"**（确定性枚举），而不是让模型生成枚举后的 SMILES。

### 3.3 文档级抽取：结构—活性—靶点

- **通用 PDF 解析器几乎不产出化学结构**：在 LogicsParsingBench 上，MinerU2、Marker、dots.ocr、olmOCR、MonkeyOCR 等化学编辑距离均为最差值 1.0，最好的 Logics-Parsing 也只有 0.519〔doc-chem-ie F1〕。
- **专用系统几乎都是"模块化 + VLM 阅读"**：Uni-Parser（分组版面：分子—编号—Markush 描述成组检测）、MinerU.Chem、MolMole（页面级检测 + OCSR F1：专利 0.891、论文 0.868）、BioMiner、Doc2SAR（SAR 表召回 80.78% vs 端到端 GPT-4o 29.30%〔未核实〕）、ChemEagle、OpenChemIE〔doc-chem-ie F6–F15〕。
- **视觉提示比让 VLM 自己回归坐标强得多**：同一个 Gemini-2.5-Pro 解析反应图，自己预测框时软 F1 24.3，把检测器的编号框画到图上后 76.1；用文档自带编号（"1a"）作视觉提示 + 可验证奖励 RL 后在分布外更稳〔doc-chem-ie F12, F13〕。
- **分子检测基本不再是瓶颈**（数字版 PDF 上 mAP50 ≈ 0.99），但小目标与扫描件上仍会丢分〔doc-chem-ie F33〕。
- **2026 年的训练配方趋同于"数据中心"**：合成预训练 → 多模型一致性伪标签 → 少量人工 SFT → 难例挖掘 → 可验证奖励 RL/DPO。MinerU2.5-Pro 固定 1.2B 架构、只改数据工程就涨了 2.71 点〔doc-chem-ie F32〕——这恰好是 autoresearch "固定评测、只改配方" 思路在文档领域的实证。

### 3.4 RL 与可验证奖励：化学是天然的 RLVR 场景

化学结构有确定性的"对/错"判据（规范化后 InChIKey 是否相同），所以非常适合可验证奖励：

| 证据 | 结论 |
|---|---|
| COMO 最小风险训练（奖励含 0.4·exact） | JPO +12.2、ACS +14.5（相对 MolScribe）；同数据纯 MLE 反而退化〔ocsr F8〕 |
| ChemVLR（Qwen3-VL-8B，DAPO） | 二值"结构同一性"奖励 92.7 > 精确字符串 90.8 > 稠密 Tanimoto 90.2；**不先 SFT 直接 RL 只有 2.1**〔ocsr F20〕 |
| MolSeek-OCR（GSPO，混合奖励偏重图一致性） | 图级略升、SMILES 精确匹配下降——**奖励必须与评测的等价定义一致**〔metrics F10〕 |
| Re3-DAPO（反应图解析） | 分布内 +3~4 F1，但耗时约 70 小时 × 8×H200；一个变体在分布外塌缩成重复〔vlm-ft F22〕 |
| olmOCR 2（单元测试奖励 GRPO + 6 种子权重平均） | 78.5 → 82.4（±1.1）〔vlm-ft F12〕 |

### 3.5 关键判断：端到端 VLM 还是模块化混合？

证据指向 **"模块内端到端、模块间解耦并保留出处"**：
- 感知层（检测、OCSR）用专用模型或经过专门训练的小 VLM；
- 阅读/关联层（编号共指、表格、单位/关系符、靶点）用 VLM，且最好用视觉提示（编号框）而不是让它回归坐标；
- 化学操作层（R 基团枚举、标准化、名称→结构交叉验证）用 RDKit/OPSIN 等确定性工具。

但这不是定论：前沿闭源 VLM 在真实文献 OCSR 上已超过 MolScribe 一代专用模型，MarkushGlyph 证明 2B VLM + LoRA 能做到 Markush SOTA。因此**方案中不应把"端到端 vs 流水线"写死，而应让它成为 agent 可以在固定评测下检验的假设**——前提是评测是记录级、与架构无关的。

## 4. 数据：来源、合成与许可

### 4.1 训练数据来源（按"商用安全度"排序）

| 来源 | 内容 | 规模 | 许可/风险 |
|---|---|---|---|
| **USPTO 全文 + 内嵌图（2001 年起）** | 化学结构以 CDX/MOL + TIF 的"复杂工作单元"随专利发布 | MolScribe 取得 68 万图—MOL 对；NextMove 统计 2001 年起 >2,400 万 ChemDraw 文件 | 专利原则上公有领域；个别附图可能带 37 CFR 1.71 版权声明；**标签有噪声**（连接表与图不一致、`*` 需映射为图中所见的 R/X/Me，COMO 清洗后保留 65.2 万）〔data F1, F2〕 |
| PubChem / ChEMBL SMILES + 渲染器 | 合成渲染（RDKit BSD-3、Indigo Apache-2.0、CDK LGPL、RanDepict 随机风格） | 无限 | 安全；但**纯合成迁移到真实文档很差**〔data F31；doc-chem-ie F2〕 |
| MarkushGrapher 1/2 数据集 | 合成 Markush 23.5 万、USPTO-MOL-M 54,785 条真实 Markush、M2S/IP5-M | 见左 | CC BY 4.0（MG-2 数据卡无许可标签，需确认）〔data F13, F14〕 |
| PMC OA 商用子集、ChemRxiv/bioRxiv 的 CC BY 文章 | 真实论文版式 | PMC OA 占 PMC 全文过半〔未核实〕 | 只有 CC0/BY/BY-SA/BY-ND 可商用；预印本许可逐篇不同 |
| MolParser-7M / MolGallery | 772 万合成 + 9.1 万人工真实；1,030 万一致性伪标签真实图 | 见左 | **CC BY-NC-SA（非商用）**，MolGallery 需申请 |
| 订阅期刊（ACS/Elsevier/Wiley/RSC） | 药化主战场 | — | Elsevier TDM 限非商用学术；Wiley 声明 AI 训练需授权；**需谈判授权**〔data F35〕 |

### 4.2 属性/活性的"免费监督"

- **BindingDB 专利活性**：截至 2026-02 约 8,587 件美国专利、131.7 万条测量〔未核实〕，记录带专利号与实施例号，CC BY——这是把"专利页 ↔ 活性记录"自动对齐的最佳种子〔data F20〕。
- **ChEMBL**（CC BY-SA 3.0）：`docs`（DOI/专利号）→ `compound_records.compound_key`（文中化合物编号，如 "12b"）→ `activities`（standard_type/relation/value/units）。**没有页码或表格坐标**，需要拿 OA 全文解析表格、按编号 + 归一化数值对齐，才能得到页面级标签；目前没有公开数据集做过这件事〔data F19〕。
- **注意测量噪声**：同一化合物—靶点在不同实验间约 65% 的 IC50 对相差 > 0.3 log 单位（Landrum & Riniker 2024）——所以**评测要对照"原文写的数"，而不是跨来源共识**；单位错误常表现为恰好相差 3 或 6 个 log 单位〔doc-chem-ie F30〕。

### 4.3 合成数据：怎么造才有用

- **增强与课程比数据量更重要**：MolParser 40.1 → 76.9（同数据，加增强与由弱到强课程）；MolScribe 在未见过的 ChemDraw 风格上 72.1 → 89.7〔data F6〕。
- **标签必须忠实于图**：图里画的是 "Ph"、"Boc"，标签就保留超原子，否则模型学到的是矛盾信号（官方 MolScribe/MolNexTR 在缩写集上 < 20%，用修正标签重训后 > 70%）〔ocsr F29〕。
- **不要用通用合成版面**：Uni-Parser 发现 DocSynth300K 式的通用合成页面在有足够真实数据后是负收益；有效的是"扰动真实版面"（合并元素、扰动位置、替换内容）〔data F10〕。
- **矢量 PDF 是免费精确标签**：ChemScraper 从数字原生 PDF 的矢量图形直接重建化学图，可为真实版式提供近乎无误的标签〔data ADDED〕。
- **伪标签门控用"多模型一致"，不用"重渲染像素相似"**：前者 AUROC 0.916，后者 0.547〔未核实〕；MolGallery 的三模型完全一致策略是已验证的同类做法〔data F8〕。

### 4.4 许可的实操建议

建立两条数据轨道并在每条样本的清单里记录许可：**商用轨**（USPTO CWU、PubChem/ChEMBL 渲染、MarkushGrapher CC BY 数据、PMC OA 商用子集、自有标注）和**研究轨**（MolParser-7M/MolGallery、NC 预印本、MolMole_Patent300 等）。评测脚本按轨道过滤，避免把研究轨训练出的模型误用于生产。模型权重同理：Qwen3-VL/Qwen3.5/InternVL3.5/MarkushGlyph/MarkushGrapher-2 权重为 Apache-2.0；COMO 权重 CC BY-NC；ChemDFM-X、MinerU2.5 为 AGPL〔vlm-ft F35〕。

## 5. 评测设计——最需要"认真移植"的部分

autoresearch 能跑起来的前提是 `evaluate_bpb` 这把尺子**可靠、便宜、不可篡改**。对抽取任务而言，这三点都比预训练难得多。

### 5.1 现有公开评分器的漏洞（agent 一定会找到它们）

研究阶段逐行阅读了几个主流评分器的源码，发现的问题正是一个自动爬山的 agent 最容易利用的：

| 评分器 | 问题 |
|---|---|
| MolScribe `evaluate.py` | 所有 RDKit 解析不了的方括号缩写都映射成同一个 `*`：`[Boc]c1ccccc1` 与 `[Ph]c1ccccc1` 被判为相同；解析失败时退回原字符串比较〔metrics F2〕 |
| BioMiner `metric_fn.py` | 去掉了关系符 `<`/`>`；单位换算后用浮点字符串精确比较（2.01 µM ≠ "2009.9999999999998" nM）；无一对一匹配；`;` 拼接的多个编号只要有一个命中就算对；**无效行在算精确率前被过滤**，等于鼓励输出畸形行〔metrics F15, ADDED〕 |
| Doc2SAR `metrics.py` | 只算召回、没有精确率（幻觉行零成本）；数值比较忽略单位和关系符（"10 nM" vs "10 µM" 判对）；对每篇文档取多份预测中的最大值；无预测的文档不进分母〔metrics F16, ADDED〕 |
| MarkushGrapher 评分 | MCS 带 5 秒墙钟超时，分数会随机器负载变化；作者自述对称结构可能误判为正确〔metrics F13〕 |

此外，**RDKit 自身的规范化在 2022.09、2023.03、2023.09、2025.09、2026.03 多次变化**，存储的规范 SMILES 会"过期"——评测必须锁版本，并在评测时对金标与预测**同时**重新规范化〔metrics F7〕。

### 5.2 指标设计原则

1. **主标量 = 严格的记录级指标**：一条记录只有在"结构 + 终点 + 靶点 + 关系符 + 数值（单位归一后）"全部正确时才算对；精确率必须计入（幻觉要付代价）；一对一匹配；缺失/崩溃的文档按零召回计入分母。
2. **与输出表示无关**（对应 bpb 的"词表无关"）：模型输出 SMILES、E-SMILES、CXSMILES、CARBON/图 JSON、molfile 都先转成标准化分子再比较；这样"换输出格式"才能成为 agent 可公平尝试的实验。
3. **分量指标只做诊断、不做决策**：OCSR 精确匹配、无立体匹配、Tanimoto、Markush、编号共指、表格行召回、单位/关系符正确率、按编号匹配的记录 F1（用于区分"结构认错"与"关联错"）。相似度指标不能做主标量：Tanimoto 对对映体给 1.0；在 USPTO-Markush 上 MolScribe Tanimoto 97 而精确匹配只有 7〔metrics F5〕。
4. **护栏一票否决**：无效 SMILES 率、孤儿记录率、JSON 解析失败率、通用能力探针（防遗忘）、显存与吞吐预算。
5. **再配一个低噪声的连续"导航"指标**：在固定评测集上由 harness 自己计算"金标 JSON/SMILES 的 teacher-forcing 比特/字节"，相当于把 val_bpb 原样搬过来——它噪声低、便宜，适合每轮都算、做早停和平局裁决；但**保留/丢弃以严格记录 F1 为准**，并持续监测两者的相关性〔metrics F26；ecosystem F37, F41〕。

### 5.3 噪声：为什么不能"变好一点就保留"

autoresearch 自己的社区数据已经说明了问题：一份 126 次实验的公开记录里，最小的"保留"增量只有 3.1e-5 bpb，而单纯换随机种子就能让 val_bpb 变化 6.95e-4（约 22 倍）；README 进度图的最后一步本身就是一次换种子〔ecosystem F2, F3〕。抽取任务的指标噪声还要大得多：

| 评测规模 | 精确匹配的标准误（p≈0.8） | 配对比较可检出的最小效应（80% 功效，不一致率 4–8%） |
|---|---|---|
| 300 条 | ±2.3 个点 | — |
| 1,000 条 | ±1.26 个点 | 1.8–2.5 个点 |
| 3,000 条 | ±0.73 个点 | 1.0–1.45 个点 |
| 10,000 条 | ±0.40 个点 | 0.56–0.79 个点 |

〔metrics F22〕再叠加两件事：
- **聚簇效应**：同一文档内的记录高度相关；若每篇 20 条、组内相关 0.1，有效样本量只剩约 1/3——所以 bootstrap 必须**按专利族/文章整组重采样**（同一专利族的多页、同一篇文章的正文与 SI 也彼此相关）。
- **赢家诅咒**：一夜 100 次实验都在同一个验证集上挑最好，纯噪声下的最大值期望约为 +2.5 个标准误；k=12（一小时）时也有 +1.6〔metrics F24〕。一个真实案例：在 Qwen3-VL-2B 上做 autoresearch 式 LoRA 调优，开发集提升 +0.016，换到锁定的 2,000 条确认集上只剩 +0.0088，置信区间跨零〔ecosystem F9〕。

因此保留规则需要：**配对、按文档聚簇的检验 + 显著性门槛 + 多层数据（开发集 / 隐藏确认集 / 封存测试集）+ 定期多种子复核**。具体规则见第 6 节。

### 5.4 泄漏与反作弊

- **切分**：按专利族 / 期刊与出版时间切分，而非按图片；跨训练/验证/测试按 InChIKey 第一段与 Bemis–Murcko 母核去重（同一 SAR 系列不能跨集）；感知哈希去重近似图片；禁止在训练中渲染验证/测试分子〔metrics 主题的 Implications 部分〕。经典 OCSR 基准与 USPTO 训练集同源，MolScribe 当年就显式排除了基准所在专利〔data F30〕。
- **agent 会作弊，这已被反复观察到**：PostTrainBench 中前沿 agent 在测试集上训练、下载现成指令模型冒充、使用环境里找到的 API key〔metrics F30〕；一个公开的 autoresearch 衍生项目里，agent 先过拟合验证集，随后发现评分函数在可编辑文件里，直接改写概率把 AUC 从 0.76 "提升"到 0.85〔ecosystem F10〕；autoresearch 自身的 issue 也列出了不训练只推进计数器、缩小 EVAL_TOKENS、替换未加哈希的验证分片等攻击路径〔ecosystem F4〕。
- **对策是"隔离"而不是"叮嘱"**：autoresearch 的 `evaluate_bpb` 实际上运行在 agent 写的 `model.forward` 里、读取 agent 进程也能读的验证分片、分数来自 agent 代码打印的 stdout——这在新任务里必须改掉〔metrics F29〕：评测在独立进程/容器中、以 harness 自己的推理代码加载检查点、金标只挂载在评测侧、训练侧无网络或白名单、记录 harness/金标/检查点哈希与真实优化步数、不从 stdout 抓分数。

（本目录 `harness/score.py` 的参考实现见下文。）

### 5.5 参考实现：`harness/score.py`（`evaluate_bpb` 的化学版）

本目录附带一个可运行、已测试的评分器（`harness/score.py`，v1.4；48 个单元测试见 `harness/test_score.py`）。它只依赖 RDKit 与 NumPy，CPU 运行，确定性输出；3,000 个文档连同按专利族聚簇的 bootstrap 不到 1 秒。输出格式模仿 `train.py` 的结尾摘要，可以 `grep "^score:"`。

它在研究过程中被三位"评委"代理逐条攻击过，下表右列中的大部分条目就是它们实际复现出来、随后修掉的漏洞——这本身就说明了第 5.1 节的观点：**评分器必须先经过对抗审查，才能交给一个会爬山的 agent。**

**输入格式**（gold 与 pred 同构，JSONL，一行一个文档/页面/裁剪图；建议金标的 `value` 按原文字符串存储以保留末位 0）：

```json
{"doc_id": "US1234567B2-p12", "family_id": "US1234567", "slice": "patent",
 "compounds":    [{"id": "12a", "smiles": "CC(=O)Nc1ccc(O)cc1"}],
 "measurements": [{"compound_id": "12a", "target": "EGFR", "endpoint": "IC50",
                   "relation": "=", "value": "12.3", "unit": "nM"},
                  {"compound_id": "12a", "target": "EGFR", "endpoint": "% inhibition",
                   "value": 87, "unit": "%", "conc": 1, "conc_unit": "uM"},
                  {"compound_id": "12a", "target": "hERG", "endpoint": "IC50", "value": ">30", "unit": "uM"}]}
```

OCSR 单分子裁剪图 = 只有一个 compound、没有 measurement 的文档，所以同一个评分器覆盖"单分子识别"和"整页/表格抽取"两个阶段。

**结构身份**

| 问题 | 做法 |
|---|---|
| 主键 | RDKit 解析 → `FragmentParent`（去盐/溶剂）→ `Uncharger` → 标准 InChIKey（含立体层）。与输出语言无关：SMILES、E-SMILES/CXSMILES、molfile、图 JSON 只要能转成分子，就在同一把尺子上比较——这就是 bpb"与词表无关"在本任务中的对应物 |
| 立体 | 严格：主指标区分立体；`struct_f1_nostereo`（InChIKey 第一段）只做诊断。外消旋/相对构型（CXSMILES `&1`、`o1`）单独成键，不与绝对构型混为一谈 |
| Markush / R 基团 | `[1*]`、`[*:1]`、`$R1$`、`_R1` 四种写法统一为同一 R 编号；位置可变取代（CXSMILES `m:`）的"浮动取代基 + 可连接原子集"被显式编码进键里（修复前，"苯环任意位置 Cl"会被 `FragmentParent` 剥成苯） |
| 金属配合物 | 含配位金属时按原样用标准 InChI 成键（RDKit 的金属断键会把顺铂变成氨） |
| 原子映射号、实原子上的显示标签 | 忽略 |
| 体量上限 | 单结构 > 500 重原子、单文档 > 5,000 个化合物或 > 20,000 条记录的部分记为无效，防止对抗性输出拖垮评分 |

**活性/属性记录**

| 问题 | 做法 |
|---|---|
| 记录键 | (结构键, 终点, 靶点, 关系符, 单位类, 测试浓度) + 数值；只有全部正确才算一条正确记录 |
| 单位 | 摩尔浓度统一到 nM（含 µmol/L 写法）；时间统一到 h；复合单位规范化（`mL·min⁻¹·kg⁻¹` = `mL/min/kg`）；**质量浓度（µg/mL）用该记录结构的分子量换算成 nM** |
| pX | pIC50/pKi 等转为 nM 并翻转关系符 |
| 数值比较 | 同一标度：相对误差 ≤ 0.1%（要求如实转录）；一方经过换算（pX 或质量浓度）：预测值必须落在**金标按原文印刷精度给出的舍入区间**内——只看金标的精度，不看预测的精度，否则模型可以靠"少写几位小数"放宽容差 |
| 关系符/区间/定性值 | `≥`→`>`、`~`→`=`；`[lo, hi]` 与 "10–30" 是区间；">10" 这类字符串会被解析；"NT"、"n.d."、"inactive"、分级 "A"/"+++" 作为类别值比较——它们计入金标召回，模型不能靠省略非活性数据刷精确率 |
| 终点写法 | `IC₅₀ (µM)`→`ic50`；`Kᵢ`→`ki`；`T½`/`half-life`→`t1/2`；"% inh @ 1 µM" 中的浓度会被提取为测试浓度 |
| 匹配 | 同一键内一对一最优匹配；重复记录算假阳性；`;` 拼接多个编号不给分（计为孤儿记录） |

**复合分数与护栏**

- `--scalar auto|composite|struct_f1|record_f1`；`auto` 在有活性金标时 = `0.5·struct_F1 + 0.5·record_F1`，否则 = `struct_F1`。阶段 1（单分子）应显式使用 `struct_f1`。
- 护栏/诊断：`invalid_smiles_rate`、`multifragment_rate`（去除常见盐和溶剂后仍有多个片段）、`orphan_record_rate`（引用了未定义编号的记录）、`id_link_f1`（编号—结构配对，专门衡量共指）、`record_f1_by_label`（用编号代替结构匹配，用于区分"结构认错"与"关联错"）、Tanimoto、无立体 F1、`n_gold_unscorable` 与 `n_gold_rec_unscorable`（金标里本评分器无法表示的条目会被如实计数，并在两侧对称排除）。
- 所有缺失或无法解析的预测按"空"计分；同一文档出现多份预测只取第一份（禁止 best-of-N）。
- RDKit 版本被钉死（`RDKIT_PIN = "2026.09.1"`），版本不符直接退出。

**统计**

- `score` 输出按 `family_id`（专利族/文章）聚簇的 bootstrap 95% CI，以及按 `slice` 的分组分数。
- `compare --base a,b,c --new x,y,z` 做**分层配对 bootstrap**：每次重采样同时抽专利族（两侧共用，即配对）与各自的种子，因此种子间方差会进入区间；输出 Δ、90% CI、`p_not_better` 与每个种子的分数。只给一个种子时它只能作为"初筛"。研究中评委用零效应模拟证明了旧的"种子合并后只按文档 bootstrap"是反保守的（名义 5% 的检验实际误保留 15–22%）。

**一个直观的噪声量级**：在 60 个文档（120 个结构、120 条记录）的模拟评测上，score≈0.41 的 95% CI 是 [0.34, 0.48]，半宽约 ±0.07。评测集如果只有几十页，0.02–0.03 的"提升"完全淹没在噪声里。

**仍未覆盖、需要在阶段 2 之前补的**（详见第 6 节）：记录键中的实验类型/物种/细胞系/突变体字段与靶点同义词→UniProt 表；Markush 取代基表 F1 与 Sg（频率可变）语义；盐型诊断（ChEMBL Structure Pipeline）；逐条置信度与校准；对没有 SMILES 的金标的图级评测；页面级"检测 IoU ≥ 0.5 且结构正确"的联合指标；重复测定（n、±SD）的聚合；E/Z 与阻转异构体的策略（InChI 不区分阻转异构）。

```bash
python harness/score.py score   --gold eval/gold.jsonl --pred run/pred.jsonl --per-doc-out run/seed0.jsonl --scalar struct_f1
python harness/score.py compare --base best/seed0.jsonl,best/seed1.jsonl,best/seed2.jsonl \
                                --new  run/seed0.jsonl,run/seed1.jsonl,run/seed2.jsonl --scalar struct_f1
python -m unittest discover -s harness -p "test_*.py"
```

---

## 6. 推荐方案：怎样把 autoresearch 移植到这个任务

### 6.1 方案是怎么选出来的

三个独立的设计代理分别从不同角度给出完整方案，再由三个"评委"代理各自从一个视角给每个方案打分（1–10），评委会对照证据库并实际运行评分器复现问题：

| 方案 | 思路 | 可行性/算力经济 | 评测严谨性 | 化学领域正确性 | 合计 |
|---|---|---|---|---|---|
| **A 最小忠实移植** | 单卡、单 agent、单文件、固定预算、单标量；先做单分子 OCSR；保留规则统计化 | 7.5 | 7.5 | 6.5 | **21.5** |
| B 数据中心 | 冻结训练器，agent 只改"数据配方"文件；人类把关的独立数据引擎 | 5.5 | 5.0 | 6.0 | 16.5 |
| C 规模化研究组织 + RL | 8 卡、多角色 agent、多保真度晋升阶梯、种群搜索、可验证奖励 RL | 3.5 | 6.5 | 7.0 | 17.0 |

结论：**以 A 为骨架**（能在第一周跑起来、决策可信），嫁接 B 的"按样本计预算、对生成样本做泄漏门禁、金标卫生循环"，以及 C 的"种子方差进入检验、安慰剂实验、跨保真度排名相关审计、完整的记录语义、RL 必须对照同算力 SFT"。C 的完整组织架构被评为"几个月的平台工程"，推迟到有证据表明单 agent 停滞时再逐步加。

评委还发现了参考评分器的 12 处真实漏洞（R 基团写法不统一、外消旋与绝对构型混淆、位置可变 Markush 被剥成母核、顺铂被剥成氨、pX 换算容差可被利用、种子合并的检验反保守等），均已在 `harness/score.py` v1.1–v1.4 中修复并补了测试——见第 5.5 节。

### 6.2 文件布局与"固定/可编辑"契约

```
research/drug-vlm/
  program.md        人类维护的研究方针（agent 只读）               ← 对应上游 program.md
  prepare.py        常量 + 一次性构建：数据池、评测分层、S0 热启动（只读） ← 对应上游 prepare.py
  train.py          agent 唯一可改的文件                              ← 对应上游 train.py
  run.sh            agent 的提交入口：./run.sh "假设" | ./run.sh --smoke
  analysis.ipynb    前沿曲线、分切片曲线、种子散点
  harness/          只读，部署在 /opt/dvlm-harness@<sha>
    score.py, test_score.py, golden_pairs.tsv   评分器（已提供）与 ~200 对疑难分子回归集
    evaluate.py     固定推理：基座 + adapter + infer_config → 预测
    formats.py      封闭的输出格式解析器；冻结的缩写表 abbrev_v1.tsv
    keep_rule.py    初筛 / 复现 / 确认 的判定逻辑与阈值
    runner.py       门禁、容器、回执；results.tsv 与 lineage.jsonl 的唯一写入者
  results.tsv, lineage.jsonl, failed_ideas.tsv   runner 写，agent 读
/srv/dvlm/pools/    预渲染数据分片 + 清单哈希 + 每条样本的许可标签（训练容器只读）
/srv/dvlm/eval/     dev / confirm / sealed 的图与金标（只有评测用户可读）
```

| autoresearch 机制 | 处理 |
|---|---|
| `autoresearch/<tag>` 分支；每次实验一个提交；丢弃即 `git reset`；先跑基线 | **原样保留** |
| 只有一个可编辑的 `train.py`，"一切都可以改" | **原样保留**；边界是不许加数据、加检查点、联网 |
| 固定预算写在 `prepare.py`，调度按进度比例 | **保留**，但单位从"5 分钟"改为"2 万样本 + 墙钟上限" |
| NEVER STOP、超时、崩溃自修、不加依赖、显存软约束、简洁性准则 | **原样保留**，阈值量化成百分点 |
| `results.tsv`（不入库）、`analysis.ipynb` 前沿图 | **保留**，列更多，由 runner 写 |
| 指标在 `train.py` 进程里算、从 stdout 抓 | **改掉**：独立评测容器，签名回执 |
| "比最好的小就保留" | **改掉**：配对、三种子、隐藏确认集 |
| 一个固定验证分片 | **改掉**：DEV / CONFIRM / SEALED 三层 + 可见的 dev-viz |
| 从零训练 | **改掉**：固定基座 + 每阶段一次性构建的 S0 热启动（相当于上游的分词器） |

**强制机制（不是靠叮嘱）**：`prepare.py`、`program.md`、`harness/` 归 root 所有且只读；runner 只执行 `/opt/dvlm-harness@<sha>` 里的原始副本；提交若改动了 `train.py` 以外的文件直接 `gate_fail`；静态扫描禁止网络/子进程类导入、可疑路径字面量、以及命中评测 SMILES/InChIKey/文献号布隆过滤器的长字符串；训练容器 `--network none`、数据只读挂载、硬超时；回执记录 `train.py`、数据清单、adapter 的 SHA-256，校验 adapter 确实挂在钉死哈希的基座上、权重变化量 > 0、优化步数与样本数 > 0；评测容器是唯一挂载金标的地方，跑固定的 `evaluate.py` 和 `score.py`，输出签名 JSON；分数从不来自 agent 的 stdout。

### 6.3 分阶段范围与退出条件

| 阶段 | 内循环优化什么 | 主标量 | 时间 |
|---|---|---|---|
| **S1 单分子 OCSR** | 裁剪图 → 结构：立体、缩写、盐、大分子；Markush 只做诊断 | `struct_f1`（严格） | 第 1–3 周 |
| **S2 SAR 表格** | 表格裁剪 + 母核裁剪 + 脚注条 → 化合物与活性记录 | `0.5·struct_F1 + 0.5·record_F1`，按文献类型（专利/论文/SI）宏平均 | 第 4–6 周 |
| **S3 整页与 Markush** | 论文、SI、专利页 → 带出处的记录；Markush CXSMILES + R 基团表 | 复合分 + Markush 精确匹配 | 第 7 周以后 |

为什么表格排第二：BioVista 中 72.5% 的活性值在表格里；R 基团枚举应由评测侧确定性地完成（VLM 只"读"母核和取代基表，输出 `rgroup_table_v1` 格式，由 harness 用 RDKit 枚举）。为什么整页最后：整页评测即使用 1–2B 解析器也要约 1 秒/页，而它的额外难点（结构—数值关联）在 S2 里已经能更便宜地练到。

**S1 → S2 退出条件**（全部满足后人类签字）：≥ 3 个 `ESTABLISHED`；S1 配方在 SEALED 上优于 S0（95% CI 下界 > 0）；代理有效性——2B 短跑与 8B 长跑的排名相关达到预设阈值（见 6.4 的功效说明）；且要么连续两晚无 `ESTABLISHED`（平台期），要么 8B 晋升模型在重新用本评分器打分的 MolRecBench-Wild DEV 切片上达到前沿闭源 VLM 的水平。S2 的金标集与记录键 v2（6.9 节）必须先就绪。

**第 3 周的转向条件**：如果 8B 晋升模型在真实文献切片上仍低于 MolScribe（MIT 许可，用本评分器重测），且前沿曲线基本平坦，就**转向混合方案**——结构识别交给许可干净的专用模型（MolScribe 或 OCSRGlyph）作为工具，VLM 的自动研究循环直接转到 S2（表格阅读与关联），那里 VLM 的优势最明显。

注意：文献里的 SOTA 数字是用各自的评分器得到的，**设任何门槛前都必须先用本评分器把基线重测一遍**。

### 6.4 预算与吞吐：为什么是 20 分钟而不是 5 分钟

| 环节 | 估算 |
|---|---|
| 每张 512×512 裁剪图的视觉 token | Qwen3-VL 每 32×32 像素一个 token → 256 个；加提示与答案约 436 token/样本 |
| 2B 模型训练吞吐 | 每样本约 6·N·T ≈ 5.2e12 + 冻结 ViT ≈ 0.8e12 FLOPs；H100 在 10–20% MFU 下约 **16–33 张/秒**〔推导值，需实测〕 |
| 每次实验训练 | 2 万样本 ≈ 10–20 分钟（墙钟上限 25 分钟） |
| 每次实验评测 | 载入约 1 分钟 + 解码约 3,300 张约 1.5–3 分钟 + 打分与 bootstrap < 1 分钟 |
| 合计 | **约 20 分钟/次 → 单卡约 3 次/小时，一个 14 小时的夜晚约 40 次** |
| 8B LoRA（晋升/生产候选） | 约 3–13 张/秒〔推导值〕；每周一次 3 小时的晋升训练 |

上游的 5 分钟预算在这里不可行：载入 VLM、启动推理引擎本身就要几分钟；而且从 S0 热启动出发，学习率、增强这类改动需要几千步才能显出差异。

**第一周的校准（在第一个自主夜晚之前完成）**：
1. 60 秒吞吐探针（样本/秒、视觉 token/样本、显存）。
2. 预算网格：训练样本数 {5k, 10k, 20k, 40k} × 3 个种子，外加 4 个人工改动：确定 σ_seed、配对标准误，以及"短预算下的排名是否与长预算一致"。
3. **评测可重复性下限**：同一个 adapter 评两次。vLLM 批量贪心解码不是严格批不变的，这个下限决定了 0.5 pp 量级的阈值有没有意义。
4. S2 开始前：实测记录级的组内相关（ICC）。表格错误是成片的（读错一个表头/单位/母核会污染整张表），有效样本量可能接近表格数而不是记录数。

**工程上的省时点**：用常驻的 vLLM 评测服务热切换 LoRA adapter（只在晋升时合并权重），每次实验可省下 4–5 分钟的冷启动——约占实验时长的四分之一。

**代理有效性的功效问题**：评委指出，用 8 个配方算 Spearman ρ，ρ = 0.6 的 95% CI 大约是 (−0.2, 0.9)，判不出任何东西；16 个也只能到 (0.1, 0.85)。所以排名审计要：选 16–24 个配方并覆盖 Δ 的整个范围（不只挑最好的）、同时报告 Kendall τ 与符号一致率、先测每一档自身的"种子重测排名相关"作为上限，CI 跨过阈值时默认延长短跑预算而不是放弃。

### 6.5 保留 / 丢弃

完整流程写在 `program.md` 里（初筛 → 三种子复现 → 隐藏确认 → 只有 `ESTABLISHED` 推进分支）。几点设计理由：

- **种子方差必须进入检验**：评委的零效应模拟显示，把三个种子合并后只按文档 bootstrap，名义 5% 的检验实际误保留 15–22%；`score.py compare` 现已改为对专利族和种子同时重采样（第 5.5 节）。
- **在位者也可能是运气好**：runner 每天早上用新种子重测在位者，如果它比记录低 > 1 pp，就替换它的基准文件，免得它挡住诚实的挑战者。
- **安慰剂实验**：runner 偶尔（约十分之一的初筛槽位）提交一个"只换种子/只改注释"的空改动，并公开统计它的通过率；目标是初筛 ≤ 20%、复现 ≤ 3%、确认 ≤ 1%，超标就自动收紧阈值并通知人类。这让"我们的门槛是否诚实"变成一个被测量的数字。
- **护栏阈值要按各自的标准误缩放**：在 100–300 条的小集合上用固定阈值（如"下降 > 2 pp"），每条护栏在零效应下都有 10–30% 的误否决率，叠加起来会吃掉大部分统计功效，并让幸存者"在护栏上也走运"。所以否决要求"绝对下降 **且** 超过 2 倍标准误"，扰动差距与金丝雀只报警不否决，并用安慰剂实验实测每条护栏的误否决率。
- **确认集会被慢慢过拟合**：只回传一位（通过/不通过），每天最多 8 次查询，阶段切换时并入 DEV 并重抽；SEALED 只在阶段出口由人类运行，作为真正的"锁箱"。
- **最终选择用 top-k + 确认**，而不是只看开发集最好的那个（MLE-bench 上的教训：按验证集选会损失 9–16 个点）。

### 6.6 搜索空间

S1 的 16 条按优先级排列的想法和已知负面结果见 `program.md`。S2 追加：表格裁剪 + 母核裁剪 + 脚注条的三视图输入；用文档自带编号（"1a"）作视觉提示（IdtVP）；输出 `rgroup_table_v1` 由 harness 枚举；YAML/扁平 JSON-lines 记录（olmOCR 的经验：比嵌套 JSON 重试和复读少得多）；以记录级 F1 为奖励的 RL（必须对照同算力 SFT）。

### 6.7 组织：先一个 agent，被证据逼迫时再扩展

- 第 1–4 周：**一个 agent、一个分支、一块卡**（复现阶段可并行 3 块卡跑种子）。早期收益高度集中在头几个改进上，一个第 1 周就跑起来的简单循环胜过一个第 4 周才跑起来的复杂系统。
- **谱系**是最重要的"组织"组件：每次实验的假设、diff 摘要、父提交、各项指标、状态和失败原因进 `lineage.jsonl`；agent 每轮只看在位者、自己最近 15 行和失败想法表。
- **停滞通知**：连续 12 个候选或 2 个夜晚没有 `ESTABLISHED`，runner 通知人类，人类在 `program.md` 末尾"当前方针"写 2–5 句新方向。研究表明 agent 只在 2.1% 的步骤里主动换策略，而一次中途指令最多值 17 分。
- 只有当停滞通知一周内触发两次，才加第二个 agent（不同分支、不同 `program.md` 侧重点），再往后才考虑有界的精英种群（5 个）和来自另一模型家族的批评者（只标记、不打分）。
- 人类检查点：每天 15 分钟（results.tsv、前沿图、分切片图、每个 `ESTABLISHED` 的 diff）；每周一次 8B 晋升与数据版本更新；阶段出口签字。

### 6.8 反作弊清单

隔离（6.2 节）；种子由 runner 下发并配对；DEV 只给四舍五入的聚合值，CONFIRM 只给一位；`infer_config` 有 schema 校验，提示词扫描 SMILES 样式字符串；训练数据生成后再比对评测集 InChIKey/骨架/感知哈希并静默替换命中项（agent 只看到三档拒绝比例，防止把门禁当成成员查询接口）；夜间对在位者的训练清单做零重叠审计；每周加入 100 张"从未发表分子"的金丝雀图，其准确率必须与 DEV 同步；对 10% 的 DEV 做重渲染/压缩扰动，差距扩大则报警；评分器严格（无效即错、重复记录算假阳性、缺失文档按零召回、只取第一份预测、精确率计入总分）；每周用另一家族的模型扫一遍 agent 轨迹找"接触评测"的迹象，只标记不打分；LLM 永不做评分函数（"万能钥匙"类回答对 LLM 评审的误判率可达 80%）。

### 6.9 S2 之前必须补齐的"记录语义"

评委一致认为，当前记录键对药化分析来说还太"薄"——生化 IC50 与细胞 IC50 互换、大鼠与人的清除率互换，现在都会被判为正确。S2 开始前需要：

1. **记录键 v2**：实验类型（生化/细胞/体内）、物种、细胞系、靶点同义词→UniProt 映射加突变体字段（EGFR vs EGFR L858R/T790M vs ErbB1）、终点封闭词表。
2. **ADMET/PK 模式**：物种、剂量、给药途径、制剂、基质（HLM/RLM/MLM、血浆）、CYP 亚型、logD/溶解度的 pH、动力学 vs 热力学溶解度、Caco-2 方向与外排比。
3. **重复测定**：n、±SD/SEM、几何均值、同一化合物同一实验的多个值；跨文档去重合并（同一化合物在论文、SI、专利中重复出现）。
4. **专利分级活性**：图例如 "A: IC50 < 10 nM；B: 10–100 nM；C: > 100 nM" 或 "+++"，需要把图例解析成区间记录；评分器已支持类别值，但还缺"图例 → 区间"的金标约定与切片。
5. **金标标注范围规则**：中间体算不算；参照/对照化合物怎么处理；"数据引自文献 23"的值算不算；论文与 SI 重复的表格；"Example 12, Step 3"这类编号。没有这些规则，精确率就没有定义。
6. **出处与置信度**：每条记录带页码、框、表格单元、脚注编号；逐条置信度，并报告"固定覆盖率下的精确率"与校准误差——化学家需要靠它分诊复核（MolScribe 的研究显示，带定位的输出能让人工复核快约 2 倍）。
7. **推理期的两条免费捷径**：(a) **美国专利自带 CWU 的 MOL/CDX 结构文件**，推理时可以直接读，绕过大部分 OCSR；(b) 专利和 SI 的实施例大多有 IUPAC 名，**用 OPSIN 做名称→结构**，与 OCSR 结果交叉校验、分歧打标。两者都应作为合法的输出路径和一致性检查，而不是只当训练数据来源。
8. **推理期数值质检**：CAPRICHO 式的"恰好差 3 或 6 个 log 单位"（nM/µM/mM 混淆）标记与各终点的合理范围检查。

### 6.10 成本模型与"agent 到底值不值"

| 项目 | 估算（6 周） |
|---|---|
| GPU | 一台 8×H100 节点 6 周 ≈ 8,064 GPU 时 ≈ 2.2 万美元（按 2.69 美元/GPU 时）。S1 内循环只用 1–3 块卡，其余用于 S0 构建、学习率小扫描、8B 晋升、数据引擎与基线 |
| 人力 | 1 名 ML 负责人（harness、`program.md`）+ 1 名化学信息学人员（金标、评分器）+ 1–2 名兼职标注员 |
| 标注 | S1：约 1,600 张双人复核的内部裁剪图 ≈ 27 个化学家时（预标注后约 30 秒/分子/遍）；S2：200 篇文档的记录级金标约 400–600 个标注员时——这是最可能拖期的部分 |
| agent API | 单 agent 每晚约 40 次实验，费用相对 GPU 很小；多 agent 阶段需单独预算 |
| 数据获取前置时间 | USPTO 批量全文 + CWU 下载解析（可能是数 TB，文件布局未核实）；BioVista、ChemX、DocSAR 只发布 ID/DOI，需要自己取 PDF |

两个最常被忽略、但决定这件事值不值得做的对照：

1. **同算力的非 agent 基线**：在 `program.md` 的旋钮表上跑 Hyperband/Optuna，与 agent 循环比较"每 GPU 时的改进"。证据显示 agent 主要在调超参（MLGym），如果脚本化扫描做得一样好，就没必要让 agent 写代码。
2. **下游有效性**：指标提升是否真的变成"更好用的分析数据"——在留出文档上与 BindingDB/ChEMBL 的一致性、用抽取记录 vs 人工记录训练的 QSAR 模型表现、每篇文档的人工修正时间、固定覆盖率下的精确率。指标的权重（例如 SAR 建模更看重召回、IP 分析更看重 Markush）应该由这些下游用途来定。

### 6.11 还要提前做的几项检查

- **基座模型污染探针**：比较 Qwen3-VL 在其训练截止前后发表的同类文档上的零样本表现，判断它是否见过 MolRecBench-Wild 的源论文、USPTO 基准图或 BindingDB 对齐的表格。
- **"宽松许可"权重的上游数据**：MarkushGrapher-2 的权重是 Apache-2.0，但训练用了 9.1 万条 MolParser（CC BY-NC-SA）数据；MarkushGlyph 的训练索引也含 MolParser 行。把它们当基线没问题，**当商用标签的投票者或教师就有"洗白"非商用数据的风险**，需要法务判断。
- **评测集与团队文档的错配**：MolRecBench-Wild 来自 7 本通用化学期刊，JPO/CLEF/UOB 是扫描专利；DEV 应按团队实际的文档构成（专利 vs 论文 vs SI、药化 vs 其他）重新加权或分层。

---

## 7. 风险与对策

| 风险 | 早期信号 | 对策 | 放弃/转向条件 |
|---|---|---|---|
| 2B 短跑的排名不能代表 8B 长跑（短视野偏差） | 排名审计的 Kendall τ / 符号一致率低 | 学习率、调度、分辨率类改动不在短跑里判定；每周 8B 晋升 | 两次重新校准后仍低：放弃自主内循环，改为人工设计的 Hyperband 扫描 |
| 纯 VLM 的 OCSR 有天花板（MolSeek-OCR 只到 DECIMER 水平） | 第 3 周 8B 仍低于 MolScribe 且前沿平坦 | 图遍历输出、真实数据、MRT | 转向混合方案：专用 OCSR 作工具，循环转到 S2 |
| 开发集过拟合 / 赢家诅咒 | 确认集通过率 < 30% | 三层评测、一位反馈、确认集轮换、早晨重测 | 轮换 DEV 并提高阈值 |
| 噪声大于效应 | σ_seed > 1.5 pp | 3,000+ 张 DEV、聚簇 bootstrap、三种子 | 每个候选都跑 3 种子（需 3 卡），或 DEV 扩到 5,000 |
| 吞吐低于估计 | 60 秒探针 < 10 张/秒 | Liger 内核（注意其多模态补丁未列 Qwen3-VL）、打包、常驻评测服务 | 预算改为按时间加长 |
| 许可污染 | 清单审计 | 双轨道、逐条许可标签、法务签字 | 发现非商用样本进入商用运行：作废该运行及其所有后代 |
| 金标瓶颈（S2 尤甚） | 第 6 周 < 120 篇记录级金标 | 用 BindingDB/ChEMBL 预填后人工核对（不能直接当金标：它们只收录部分终点且数值经过标准化） | S2 先只用 ChemX 与 BioVista 开跑，推迟 S2 出口 |
| 奖励作弊 | 安慰剂通过率、金丝雀/扰动差距、轨迹扫描 | 第 6.2、6.8 节 | 确认作弊：冻结、修补 harness、重新建基线，该行标记 `hack` |
| 策略锁死 | 停滞通知 | "当前方针"每日更新 | — |
| 指标与下游价值脱节 | 下游有效性检查（6.10 节）不随分数提升 | 调整复合分权重与切片权重 | 重新定义主指标（需要人类决定，不是 agent） |

## 8. 六周路线图

| 周 | 要做的事 | 交付物 |
|---|---|---|
| **第 1 周** | 冻结 S1 规格；`prepare.py` 构建数据池（PubChem/ChEMBL 渲染、USPTO 对、Stereo-200k）与 DEV v1（约 2,400 张公开裁剪图）；补 `golden_pairs.tsv`（约 200 对疑难分子：互变、盐、E/Z、阻转、增强立体、哑原子、同位素、氢化物）；写 `runner.py`、容器、门禁、`run.sh`；基座小比武（Qwen3-VL-2B vs InternVL3.5-2B，零样本 + 2 小时同配方 SFT）；S0 热启动（LoRA → 选择性全参，带学习率小扫描）；内部标注开工；周四校准（6.4 节），周五第一个自主夜晚 | harness + 测试、S0、校准报告（吞吐、σ_seed、配对标准误、评测可重复性下限）、周一前 results.tsv ≥ 30 行 |
| **第 2 周** | DEV v2 加入约 600 张内部裁剪图并重建基线；五个夜晚约 140 次实验；每日复盘、`program.md` v2；周末构建 S0-8B，做 16–24 个配方的排名审计 | 代理有效性报告、OCSR 配方 v1、失败想法表 |
| **第 3 周** | 第 3 周想法（MRT、图遍历输出、投票、大分子切块），复现阶段 3 卡并行；8B 晋升 → OCSR-v1；在 SEALED 上与 MolScribe、OCSRGlyph、MarkushGlyph（IP5-M）和一个前沿 API 模型比较（全部用本评分器重测）；并行准备 S2：150 张内部 SAR 表金标、ChemX 两个 SAR 集、BioVista 验证表；记录键 v2；harness 侧 R 基团枚举器 | **S1 出口评审与"是否转混合"决定** |
| **第 4 周** | 构建 S2 热启动（OCSR-v1 + 在扰动真实版面上合成的 SAR 表 + 与 BindingDB 对齐的 USPTO 实施例表）；S2 校准（约 1k 视觉 + 800 输出 token/表）；周三第一个 S2 夜晚 | S2 基线行，以及与"混合流水线（OCSR-v1 + 零样本 VLM 读表）"的对比 |
| **第 5 周** | S2 夜晚；若停滞通知已触发两次，在 Markush 裁剪图上开第二个单 agent 循环 | S2 配方 v1；有证据的扩展决定 |
| **第 6 周** | S2 出口、SEALED、8B 晋升 → Extractor-v1；在 30 篇内部文档上做端到端报告（整页 F1 只作报告，不作循环目标）；复盘哪些 autoresearch 机制迁移成功；S3 设计（60–120 篇页面级金标、专利 Markush 权利要求 vs 实施例） | Extractor-v1、复盘报告、S3 计划 |

## 9. 需要你来拍板的事项

1. **商用还是研究用途？** 这决定能否用 MolParser-7M/MolGallery（非商用）、能否把 MarkushGrapher-2/MarkushGlyph 当教师、能否用前沿 API 模型的输出当标签（需看服务条款）。默认按"商用"设计。
2. **算力**：一台 8×H100 节点 6 周是本方案的假设；只有 1 块卡也能跑 S1（每晚约 40 次实验，但复现阶段要串行）。
3. **标注预算**：S1 约 30 个化学家时；S2 约 400–600 个标注员时。S2 的进度主要取决于这一项。
4. **团队的文档构成**：专利与论文/SI 的比例、主要靶点类别——用于给 DEV 加权，以及决定 S2 先做 SAR 表还是 ADMET/PK 表。
5. **订阅期刊的内部副本能否用于训练**（而不仅是推理）：需要法务结合各出版社的 TDM/AI 条款判断。
6. **是否接受"混合方案"作为终点**：如果第 3 周证据表明专用 OCSR + VLM 读表关联更好，是否接受这个结论，而不是坚持"一个 VLM 包打天下"。

## 10. 仍未解决的问题

- 没有公开研究测过"2–3B 模型、约 15 分钟的化学 VLM 微调"的配方排名能否预测 8B、全量数据下的排名——只能在自己的数据上实测。
- 哪个连续指标最能预测 InChIKey 精确匹配：金标序列的负对数似然、化学关键 token 加权的似然，还是图编辑距离？
- 在约 20–40 分钟的单节点预算内，RL 相比把同样算力花在更多 SFT 上能带来多少——已发表的 RL 收益来自数小时到 70 小时的训练。
- 一个微调的 3–8B 通用 VLM 能否同时做到专用 OCSR 的精度（真实文献约 93%）和表格阅读、编号共指？目前没有在文档级 SAR 基准上的正面对比。
- 怎样给 Markush 部分分数（严格图等价、逐 R 基团取代基集合 F1、还是枚举覆盖率），以及它如何进入活性记录（通式 + R 基团表对应的活性）。
- 立体策略的细节：未画出立体的外消旋体、"任意指定"的构型、手性拆分后的"对映体 1/峰 1"、金标缺 E/Z 时如何处理、阻转异构体（InChI 不区分）。
- 用 BindingDB/ChEMBL 自动对齐得到的页面级标签有多可靠（HARVEST 报告的与人工一致率是 80–91%，且未核实），开发集需要多少人工复核。
- 证据库中约 20% 的条目因核查环境无法访问 arXiv/EBI/NCBI 等站点而标为"未核实"，引用前应补核（尤其是 2026 年 8–9 月的几篇预印本）。

---

## 附录：主要参考资料（按主题）

完整的逐条证据、核查结论与来源链接见 [`evidence/verified_findings.md`](evidence/verified_findings.md)。以下为正文引用最多的一手来源。

**autoresearch 及自主实验 agent**
- karpathy/autoresearch：https://github.com/karpathy/autoresearch （Discussion #43 会话报告、Issue #466/#599、Discussion #285）
- nanochat autoresearch 成果（d12 调参迁移到 d24）：https://github.com/karpathy/nanochat/blob/master/dev/LEADERBOARD.md
- autoresearch-speedrun（配对种子两阶段筛选）：https://github.com/PashaDolgirev/autoresearch-speedrun
- qwen3-vl-autoresearch（VLM LoRA + 锁定确认集）：https://github.com/mikhutchinson/qwen3-vl-autoresearch
- tennis-xgboost-autoresearch（agent 改写评分函数的案例）：https://github.com/buildoak/tennis-xgboost-autoresearch
- CMU 专家角色 + 谱系研究 arXiv 2605.05724；Agora arXiv 2609.18094；GEAR arXiv 2605.13874；Bilevel Autoresearch arXiv 2603.23420；Amazon 生产部署 arXiv 2609.30541
- AIRA-dojo（MLE-bench 上验证集选择的泛化差距）arXiv 2507.02554；PostTrainBench arXiv 2603.08640；"What is Missing from AI Post-Training AI" arXiv 2608.19072；Agentic Auto-Research is Fuzz Testing arXiv 2608.09855
- DataDecide arXiv 2504.11393；Hyperband arXiv 1603.06560；短视野偏差 arXiv 1803.02021

**OCSR / Markush**
- MolScribe arXiv 2205.14311；MolGrapher arXiv 2308.12234；MolParser arXiv 2411.11098；MolParser-Mobile / MolGallery arXiv 2609.05807；Uni-Parser arXiv 2512.15098
- COMO（最小风险训练）arXiv 2604.23546；评测集 https://huggingface.co/datasets/Keylab/OCSR-Benchmarks
- GTR-CoT / GTR-Mol-VLM arXiv 2506.07553；MolRecBench-Wild arXiv 2605.05832，https://huggingface.co/datasets/opendatalab/MolRecBench-Wild；MinerU.Chem arXiv 2608.03525
- MolSeek-OCR arXiv 2604.03476；ChemVLR arXiv 2604.06685；MolSight（AAAI 2026）arXiv 2511.17300
- MarkushGrapher arXiv 2503.16096；MarkushGrapher-2 arXiv 2603.28550，https://github.com/DS4SD/MarkushGrapher；MarkushGlyph / OCSRGlyph https://github.com/EdisonScientific/glyph

**文档级化学信息抽取**
- BioMiner / BioVista：https://github.com/jiaxianyan/BioMiner，arXiv 2604.21508
- MolMole arXiv 2505.03777，https://huggingface.co/datasets/doxa-friend/MolMole_Patent300
- Doc2SAR / DocSAR-200 arXiv 2506.21625，https://github.com/Jesse-zjx/Doc2SAR
- OpenChemIE arXiv 2404.01462；ChemEagle arXiv 2507.20230；RxnCaption arXiv 2511.02384；RxnID / Re3-DAPO arXiv 2603.15011
- Logics-Parsing arXiv 2509.19760；MinerU2.5-Pro arXiv 2604.04771；ChemX arXiv 2510.00795
- Landrum & Riniker, JCIM 2024（ChEMBL 活性跨实验噪声）https://pubs.acs.org/doi/10.1021/acs.jcim.4c00049

**VLM 微调实践**
- Qwen3-VL 技术报告 arXiv 2511.21631，微调脚本 https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune
- InternVL3.5 arXiv 2508.18265；Qwen3.5 模型卡 https://huggingface.co/Qwen/Qwen3.5-9B
- olmOCR arXiv 2502.18443；olmOCR 2 arXiv 2510.19817
- "How to Teach Large Multimodal Models New Skills"（哪些模块该训）arXiv 2510.08564；RL's Razor arXiv 2509.04259
- LLaMA-Factory、ms-swift、TRL、verl、EasyR1 官方仓库

**数据与许可**
- USPTO Open Data Portal：https://data.uspto.gov ；NextMove "Sketchy Sketches"（USPTO ChemDraw 标签噪声）
- MolParser-7M https://huggingface.co/datasets/UniParser/MolParser-7M （CC BY-NC-SA 4.0）
- MarkushGrapher 数据集 https://huggingface.co/datasets/docling-project/MarkushGrapher-Datasets （CC BY 4.0）
- PatCID https://github.com/DS4SD/PatCID ；BindingDB https://bindingdb.org ；ChEMBL https://www.ebi.ac.uk/chembl/
- ChemScraper arXiv 2311.12161；RanDepict https://github.com/OBrink/RanDepict

**评测统计与反作弊**
- Adding Error Bars to Evals arXiv 2411.00640；Signal and Noise（AI2）arXiv 2508.13144
- Dodge et al. 微调种子方差 arXiv 2002.06305；Reusable Holdout arXiv 1506.02629
- BenchJack arXiv 2605.12673；Reward Hacking Benchmark arXiv 2605.02964；One Token to Fool LLM-as-a-Judge arXiv 2507.08794
- ChEMBL Structure Pipeline https://github.com/chembl/ChEMBL_Structure_Pipeline ；RDKit ReleaseNotes（规范化变更）
