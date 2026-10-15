# 工程设计说明：drug-vlm autoresearch 循环

> 本文件是 [`../README.md`](../README.md) 第 6 节的工程细节，给搭建 harness 的工程师和设计评审者看。决策层面的内容（做什么、为什么、什么时候、花多少）以 README 为准；agent 的行为规则以 [`../program.md`](../program.md) 和冻结后的 `harness/keep_rule.yaml` 为准，三者不一致时以后两者为准。英文原始方案见 [`../evidence/design_synthesis.md`](../evidence/design_synthesis.md)。方括号中的数字（如 `[0.5]`）是校准前的默认值。

**目录**：[1 文件布局与契约](#1-文件布局与固定可编辑契约) · [2 构建顺序与最低可用配置](#2-构建顺序最低可用配置与环境) · [3 主指标、护栏与评测层级](#3-主指标护栏与评测层级s1) · [4 预算与吞吐](#4-预算吞吐与-gpu-分配) · [5 保留规则](#5-保留丢弃规则与设计理由) · [6 两个审计](#6-两个审计排名一致性与-hyperband-基线) · [7 搜索空间](#7-搜索空间) · [8 反作弊](#8-反作弊清单) · [9 记录语义](#9-s2-的记录语义) · [10 数据策略](#10-数据策略) · [11 基座模型](#11-基座模型) · [12 下游有效性与金标质量](#12-下游有效性与金标质量)

---

## 1. 文件布局与"固定/可编辑"契约

```
research/drug-vlm/
  README.md, program.md     报告；agent 研究方针（人类维护，agent 只读）                 [已有]
  docs/                     本文件；交付物 schema                                        [已有]
  train.py                  agent 唯一可改的文件（基线约 400 行）                          [待建]
  run.sh                    ./run.sh | --wave | --smoke | --status | --viz | --baseline | --check | --init <tag>   [待建]
  analysis.ipynb            当前最优曲线、分组曲线、种子散点、安慰剂通过率                [待建]
  evidence/                 证据库、三个原始方案、评委打分、最终方案英文原文              [已有]
  harness/                  只读，部署为 /opt/dvlm-harness@<sha>
    score.py, test_score.py 固定评分器与单元测试（说明见 harness/README.md）             [已有]
    null_sim.py             配对检验的零效应模拟（3 对 3 保留规则；1 对 3 初筛）           [已有]
    yaml_to_jsonl.py        records_yaml_v1 → 评分 JSONL；母核 + R 基团用 RDKit molzip 枚举  [已有]
    prepare.py              常量（STAGE、TOKEN_BUDGET、各种上限）；构建数据池、评测层级、W0   [待建]
    dvlm/                   训练侧唯一入口（接口见第 2.3 节）                             [待建]
    formats.py              输出格式注册表 → CXSMILES（第 1.1 节）                       [待建]
    abbrev_v1.tsv, targets_v1.tsv, units_v1.tsv, annotation_guide.md                      [待建]
    enumerate.py            交付用的枚举器（从 yaml_to_jsonl.enumerate_rows 起步，加 Markush 与复核队列） [待建]
    golden_pairs.tsv        约 200 对疑难分子回归集                                       [待建]
    evalserver.py           常驻 vLLM，热切换 LoRA，执行 infer_config                     [待建]
    probes.py               表格转写探针、金丝雀、扰动集的打分；负对数似然               [待建]
    keep_rule.py, keep_rule.yaml   判定流程与冻结阈值                                     [待建]
    runner.py               门禁、容器、运行凭证；results.tsv 与 lineage.jsonl 的唯一写入者  [待建]
  results.tsv, lineage.jsonl, failed_descriptors.tsv    runner 写，agent 只读（.gitignore）
  notes/, requests.md       agent 的笔记与数据请求（.gitignore）
/srv/dvlm/pools（只读）  /srv/dvlm/w0  /srv/dvlm/eval/{viz,dev,confirm,reserve,sealed,probe,canary,perturb}/{images,gold}
/srv/dvlm/private/results_full.tsv   只有人类可读：原始增量、安慰剂标记、全部哈希
/srv/dvlm/policy/current.md          "当前方针"：人类随时改，root 所有、agent 只读、不在 git 里；--status 第一屏显示
/srv/dvlm/run_tag                    ./run.sh --init <tag> 写入；已存在的 tag 被拒绝
```

仓库根目录的 `prepare.py`、`train.py` 是上游的 GPT 预训练示例，与本任务无关；正式落地时建议把本目录拆成独立仓库。

### 1.1 输出格式注册表（唯一来源）

| 格式 id | 阶段 | 说明 |
|---|---|---|
| `smiles`、`cxsmiles` | S1 起 | 直接交给 score.py |
| `molfile` | S1 起 | MDL V2000/V3000，RDKit 读入后转 CXSMILES |
| `graph_json_v1` | S1 起 | 原子（元素、电荷、坐标、R 标签）+ 键（级别、楔形）的 JSON |
| `cot+<格式>` | S1 起 | 先自由推理，只解析最后一个 `<answer>…</answer>` |
| `rgroup_table_v1` | S2 | 母核 + 取代基表，由 `enumerate.py` 用 RDKit 枚举 |
| `records_yaml_v1` | S2 | 活性/属性记录（示例见 [`deliverable_schema.md`](deliverable_schema.md)） |
| `iupac_name` | S2 | 化合物名称，由 OPSIN（MIT）转结构 |

E-SMILES、CARBON 等其他写法如果需要，先加进注册表并补往返测试，才能在 `infer_config` 里使用。转换失败按无效结构计；未知缩写按错，从不映射成 `*`。

### 1.2 与上游机制的对照

| autoresearch 机制 | 处理 |
|---|---|
| `autoresearch/<tag>` 分支；每次实验一个提交；先跑基线 | **保留**；分支只由 runner 推进，agent 在候选分支上提交 |
| 只有一个可编辑的 `train.py`，"一切都可以改" | **保留**；数据只能经 `dvlm` 取，边界是不许联网、加数据、加检查点 |
| 固定预算写在 `prepare.py` | **保留**，单位从"5 分钟"改为"800 万训练 token + 1,500 秒墙钟硬上限" |
| NEVER STOP、超时、崩溃自修、不加依赖、简洁性准则 | **保留**，阈值量化成百分点 |
| `results.tsv`（不入库）、`analysis.ipynb` | **保留**，列更多，由 runner 写 |
| 指标在 `train.py` 进程里算、从 stdout 抓 | **改掉**：独立的评测与评分容器，运行凭证 |
| "比最好的小就保留" | **改掉**：配对、三种子、隐藏确认集 |
| 人类改 `program.md` 来调整方向 | **改为**：`program.md` 只放稳定规则；每天的方向调整写在 git 之外的 `/srv/dvlm/policy/current.md`，因为 agent 的工作区在 runner 管理的分支上，人类在别处提交的修改看不到 |
| 一个固定验证分片 | **改掉**：开发集 / 隐藏确认集 / 封存测试集 + 可见的可视化集 |
| 从零训练 | **改掉**：每个阶段一次性构建、只读的热启动检查点 W0（相当于上游的分词器） |

### 1.3 强制机制（不是靠叮嘱）

依据〔ecosystem F4, F10, F29；metrics F29〕：

- **隔离**：harness、`program.md` 和数据池归 root 所有、只读、不可变（`chattr +i`），runner 只执行 `@sha` 副本；训练容器 `--network none`、没有 API key、数据只读挂载、CPU/内存配额、到时硬杀；评测服务（只看得到图片）解码，评分容器（唯一挂载金标的地方）打分并输出签名 JSON。分数从不来自 agent 的 stdout。
- **门禁**：提交改动 `train.py` 以外的文件 → `gate_fail`；静态扫描（≤ 1,500 行；禁止网络/子进程/ctypes 导入、`from_pretrained`、权重或评测路径、种子字面量、篡改计量器；`INFER_CONFIG` 必须是模块级字面量，用 `ast.literal_eval` 读出后按 schema 校验，训练前就能拒绝越界配置）；长度 ≥ 15 的字符串与评测集 SMILES/InChIKey/文献号的 Bloom 过滤器比对（Bloom 过滤器：一种能快速判断"某字符串是否在集合里"的紧凑数据结构）。
- **数据门禁**：每个训练样本都经 `dvlm.render`；InChIKey 第一段或 SAR 系列骨架在评测黑名单里的分子被静默替换，agent 只看到三档拦截比例（0 / < 0.5% / ≥ 0.5%），防止它借门禁反复试探某个分子是否在评测集里。
- **运行凭证**：adapter 能在 W0 上加载；计量的 token 数与优化器步数一致（±2%）；权重确实变了；runner 记录 `train.py`、数据池、W0、adapter、harness、金标的 SHA-256。

## 2. 构建顺序、最低可用配置与环境

### 2.1 构建顺序（第 1–2 周）

| # | 组件 | 负责人 | 估算 | 验收 |
|---|---|---|---|---|
| 1 | 环境与版本锁定：`uv.lock` 锁定 transformers/peft/trl/vllm/`rdkit==2026.09.1`；支持 `--network none` 的容器运行时；验证 vLLM 对 Qwen3-VL 的 LoRA 热切换 | ML 工程师 | 3 人日 | 同一镜像两次构建哈希一致；LoRA 热切换跑通（否则改用合并权重，每次多 3–4 分钟） |
| 2 | 评测层级与金标：公开集下载、按族划分、InChIKey 第一段跨层去重；可视化集/开发集/确认集/储备集 | 化学信息学 + 化学家 | 5 人日 + 标注 | 重叠审计为 0；20% 双人标注 |
| 3 | `formats.py` + `abbrev_v1.tsv` + `golden_pairs.tsv` | 化学信息学 | 4 人日 | 疑难分子回归集全过；缩写表能复现每个含超原子的金标 |
| 4 | `evalserver.py` + `probes.py` | ML 工程师 | 5 人日 | 同一 adapter 评两次：开发集差 ≤ [0.1] pp、探针差 ≤ [0.5] pp；`INFER_CONFIG` 全字段生效；`probes.py` 输出探针增量的配对 bootstrap SE |
| 5 | `dvlm/` | ML 负责人 | 6 人日 | Meter 与优化器步数差 ≤ 2%；`render` 拦截评测分子；`rewards` 与 score.py 在疑难分子回归集上判定一致 |
| 6 | W0-2B（学习率扫描 → LoRA → 选择性全参） | ML 负责人 | 3 人日 + 约 30 GPU 时 | 开发集上 ≥ 零样本 + [20] pp |
| 7 | `keep_rule.py/yaml` + `runner.py` + `run.sh` | ML 工程师 | 6 人日 | 20 次试运行的运行凭证有效；20 个手动安慰剂中被判 promising 的 ≤ [6] 个（名义率 20%，留余量，见 `harness/README.md` 第 6 节）；描述符键清单与 `program.md` 第 5 节一致 |
| 8 | `train.py` 基线 | ML 负责人 | 1 人日 | `./run.sh --smoke` 60 秒内跑通 |
| 9 | 混合流水线 v0（Docling 文本层与表格集成、分子框检测、CWU/OPSIN/OCSR 交叉核对、实验方法段落链接、读表服务、Parquet/SDF 写出、schema 校验；README 6.4 节） | 化学信息学 + ML 工程师 | 约 [10] 人日（第 2–3 周） | 30 篇摸底文档端到端跑通；分子框检测器若自训，另需约 [1,500] 个框 ≈ [10] 标注小时 |

S1 基础设施（1–8）合计约 33 人日，三名工程人员两周的容量约 30 人日——偏紧，所以有下面的最低可用配置；混合流水线（9）放在第 2–3 周，与档 A 的人手（化学信息学 1 人 + ML 工程师 0.5 人）一致。

### 2.2 最低可用配置（第一个自主夜晚之前必须有）

**必须有**：agent 账户读不到金标和评测路径（文件权限 + 独立的评测进程）；只有 runner 能推进分支；三种子配对的 `score.py compare`；`results.tsv` 与 `lineage.jsonl`；训练容器没有网络和 API key；`train.py` 以外的改动被拒。

**可推迟到第 3–4 周**：Bloom 过滤器、`chattr +i`、签名运行凭证、自动安慰剂注入（校准期的 20 个安慰剂由人手动提交）、经验贝叶斯收缩（之前初筛用原始 Δ）、金丝雀与扰动集、`dvlm.rewards`（RL 通道第 4 周才打开）、Hyperband 基线。`program.md` 第 2 节的"首夜必需"一列与这里一致；`./run.sh --check` 把可推迟的项打印为 DEFERRED，不阻止开跑。

如果第 2 周末最低可用配置还没全绿，第一个自主夜晚推迟到第 3 周初，S1 关口同步顺延。

### 2.3 `dvlm` 接口（草案，以 docstring 为准）

```python
from dvlm import load_base, pools, Meter, render, augment, rewards, save_adapter

model, processor = load_base(lora=dict(r=64, targets="attn"))   # W0 + 可选 PEFT 包装；不接受路径
for sample in pools({"uspto": 0.5, "render": 0.4, "stereo": 0.1}, filter=None):
    ...  # sample: image, smiles, label_views{"faithful","expanded"}, source, license_track,
         #         family_id, is_real, has_stereo, n_heavy, has_abbrev, w0_loss, agreement
meter = Meter()                     # 预算由 prepare.TOKEN_BUDGET 固定；每个带梯度的 batch 都要经过它
if not meter.add(batch): break      # 返回 False 表示预算用完
img = render(smiles, style="indigo_v2")         # 经数据门禁；命中黑名单时静默替换
r = rewards.structure(pred_text, gold_smiles, fmt="smiles")   # 0.1·valid + 0.5·sim·valid + 0.4·exact，exact 用 score.py 的结构键
save_adapter(model, INFER_CONFIG)   # 写 /out/adapter 与 /out/infer_config.json；必须等于 train.py 的 INFER_CONFIG 字面量
```

`./run.sh --check` 逐行运行 `program.md` 第 2 节前置条件表的验收项（第 2.1 节的构建表是它的子集），打印 PASS / FAIL / DEFERRED。

### 2.4 环境清单

一台 8×H100-80GB 节点（约 128 核、≥ 8 TB NVMe）；支持 `--network none`、cgroup 配额的容器运行时；能设置 `chattr +i` 的 root 权限；`uv.lock` 锁定的 Python 栈；`/srv/dvlm` 目录布局（第 1 节）；双人标注的评测金标。只有 1 张 H100 时见 README 第 0 节的"三档投入"。

## 3. 主指标、护栏与评测层级（S1）

**主指标** `val_struct = Σ_s w_s · struct_f1_s`，按来源切片加权：内部文档 `inhouse` 0.30、BioVista `biovista` 0.30、USPTO 2025–26 `uspto` 0.20、MolRecBench-Wild `mrbw` 0.20（内部文档第 3 周加入，之前在已有来源间重新归一化，加入时重测在位者）。权重反映团队自己的文档，而不是 MolRecBench-Wild 的通用化学期刊。生产中美国专利的结构来自 CWU 结构文件，所以 `uspto` 切片代表的是**没有 CWU 的专利**（CN/EP/WO/JP）的绘图风格；它的权重在摸底后按"团队文档中既没有 CWU、也没有英文名的结构"所占比例重定，拿得到 CN/EP 专利金标时用 `cn`/`ep` 专利切片替换或补充。单分子裁剪图上，`struct_f1` 就是精确匹配率：错答、无效、无法转换、未知缩写、**没有输出**都算错（score.py v1.7 起缺失与错答同价）；立体严格，不用"自适应"规则（它会掩盖编造的立体〔ocsr F13〕）；Tanimoto 从不参与判定。

**护栏**（方向不同，否决条件不同；SE 来自 `compare` 输出的配对 bootstrap 标准误）：

| 护栏 | 方向 | 否决条件 |
|---|---|---|
| 无效结构率 `invalid` | 越低越好 | Δ > +max([0.3] pp, 2.5·SE) |
| 多片段率 `multifrag` | 越低越好 | Δ > +max([0.3] pp, 2.5·SE) |
| 每个难度标签（≥ 300 张）：`chiral`、`abbrev`、`large`（> 50 重原子）、`lowres`（扫描/低分辨率） | 越高越好 | Δ < −max([1.5] pp, 2.5·SE) |
| 来源切片 `inhouse` | 越高越好 | Δ < −max([1.5] pp, 2.5·SE) |
| 300 条内部表格单元格转写探针 `probe`（为 S2 保住读表能力〔vlm-ft F25〕） | 越高越好 | Δ < −max([2] pp, 2.5·SE) |

固定否决（不看 SE）：超过 1% 的样本没解码完或解析失败；外推的 8B 部署成本 > [0.5] GPU 秒/张。8 条单侧检验在 z = 2.5 下的族错误率约 5%，安慰剂实测必须 ≤ [10]%。

**只报警**：金丝雀差距（100 个从未发表的分子上的准确率与开发集之差，检测死记硬背）、扰动差距（300 张重新编码的图）。**只诊断**：无立体 F1、Tanimoto、各来源 F1、盐型、Markush（IP5-M/M2S/USPTO-M）、Keylab 基准集、同一输出格式内金标序列的负对数似然（只用于相关性监测，不参与决策）、置信度校准误差。

**错误类型**（`--viz` 与分析用，由 runner 从 score.py 的逐条字段计算）：

| 类型 | 判定 |
|---|---|
| `correct` | `struct_tp = 1` |
| `stereo_only` | `struct_tp = 0` 且 `struct_tp_skeleton = 1`（骨架对、立体错） |
| `invalid` | `n_invalid_struct ≥ 1` |
| `multifrag` | 去盐后仍多片段 |
| `missing` | `n_pred_struct_raw = 0` |
| `wrong_skeleton` | 其余 |

**评测层级**（全部按文章或 DOCDB 专利族划分；DOCDB 专利族 = 欧专局定义的同一发明在各国的专利集合）：

| 来源 | 可视化集 | 开发集 | 隐藏确认集（+储备） | 封存测试集 | 许可 / 说明 |
|---|---|---|---|---|---|
| MolRecBench-Wild（可用 SMILES 评测的 2,392 张；子集 C 只有 298 张〔doc-chem-ie F37〕） | 100 | 1,200 | 700（+300） | — | 无数据卡许可，源论文 CC BY；法务签字后只用于评测 |
| BioVista 非 Markush（约 8,861 张的一半） | 100 | 1,000 | 600（+400） | — | MIT；取自其测试划分，所以不对外报告 BioVista 公开数字（约 886 张的验证划分太小） |
| USPTO 2025–26 CWU 裁剪图（MOL 导出，化学家核对） | 50 | 600 | 400（+200） | — | 公有领域（核对 37 CFR 1.71 版权声明〔data F37，未核实〕） |
| 内部论文/SI/专利（第 3 周加入） | 50 | 600 | 400（+200） | 1,000 | 专有；封存集 = 2026-07-01 之后的文档、EPO/CNIPA/JPO 专利、项目报告 |
| Keylab（CLEF/JPO/UOB/ACS/USPTO-10K）；M2S、USPTO-Markush；IP5-M | 每晚诊断 | | | | 各异；M2S、USPTO-Markush 为 CC BY 4.0；IP5-M 所在的 MG-2 数据卡无许可标签，需法务确认 |

合计：开发集 2,800 张（第 3 周加入内部 600 张后 3,400）；确认集 1,700（之后 2,100）；可视化集 250（之后 300）。如果摸底阶段测得中国专利/中文论文占比 > [30]%，开发集和确认集都要加 `cn` 来源切片，不能只放在封存集。

**功效**：n = 3,400、约 500 个族、不一致率约 6%、设计效应约 1.3 时，配对差的标准误约 0.48 pp；再加约 0.5 pp 的种子标准差，初筛（1 个新种子 vs 3 个已存种子）总标准误约 0.75 pp，复现（3 vs 3）约 0.63 pp。初筛时新候选只有 1 个种子，`score.py compare`（v1.8 起）从在位者 3 个种子的差异借来它缺少的种子方差，否则初筛的标准误偏小、安慰剂过筛偏多（`harness/README.md` 第 6 节）。单侧 5%、80% 功效（乘数约 2.5）下最小可检提升：复现约 1.6 pp、初筛约 1.9 pp〔metrics F22〕。一个 12 分钟的边际改动大概率只动 0.5–2 pp，所以保留会很少。

## 4. 预算、吞吐与 GPU 分配

**吞吐估算**：512² 的裁剪图是 256 个视觉 token〔vlm-ft F32〕，加提示与答案约 436 token；2B 模型 LoRA 每张约 6·N·T + ViT ≈ 6e12 FLOPs，10–20% MFU 下约 16–33 张/秒〔vlm-ft F33，未核实〕，按 25 张/秒（约 1.09 万 token/秒）规划。

**`TOKEN_BUDGET` = 800 万训练 token（约 1.8 万张），由 `dvlm.Meter` 计数，墙钟 1,500 秒硬杀。** 按 token 计，冗长的输出格式和更高的分辨率要自己为成本买单，机器争用只影响墙钟、不影响样本数。预算按预先登记的规则冻结：{400 万, 800 万, 1,600 万} × 3 个种子 + 4 个人工改动，选种子标准差 ≤ [0.7] pp、且排名与 1,600 万一致（Kendall τ ≥ [0.67]）的最小预算；若 800 万的训练中位时间超过 15 分钟，改用 600 万。

| 步骤 | 在哪里 | 秒 |
|---|---|---|
| 启动容器 + 载入合并后的 W0-2B | 训练卡 | 45 |
| 训练 800 万 token | 训练卡 | 735（555–1,140） |
| 保存 adapter + 运行凭证 | 训练卡 | 15 |
| 热切换 LoRA；解码 4,100 张（开发集 + 探针 + 金丝雀 + 扰动；dev-viz 不在每次运行里解码，`--viz` 时按需解码），按 LoRA 推理约慢 1.35 倍估算（需实测） | 评测卡 | ≈ 100 |
| 1,000 张的负对数似然 | 评测卡 | 15 |
| 打分 + 层次配对比较 | CPU | 40 |

约 13.3 训练卡分钟/次，每卡每小时约 4.5 次，约 16 分钟出结论。一个 14 小时的夜晚，3 张训练卡约 190 次运行；每个候选平均约 0.5 次额外运行（约 25% 的候选进入复现，每次补 2 个种子），约 125 个候选（安慰剂与早晨重测在 4 号卡上，不占训练卡）——agent（每波 ≤ 3 个）很可能成为瓶颈，预期 60–110 个想法/晚。只有 1 张 H100 时评测也挤在同一张卡上（每次约 15 分钟 GPU 时间，含解码与负对数似然），约 55 次运行，扣掉安慰剂与复现后约 30–35 个想法/晚。

**评测服务**：vLLM 常驻 W0-2B 并开启 LoRA（4 个槽位，rank ≤ 128）。批量贪心解码不是"批不变"的（同一输入在不同批组合下输出可能不同），所以固定请求顺序和批大小，并把实测的可重复性下限加进总标准误：准备阶段验收要求 ≤ [0.1] pp，之后每周监测，超过 [0.2] pp 时改用确定性解码。`EVAL_DECODE_CAP` = 600 秒/次（约为基线的 6 倍），够用图遍历推理（输出长度在准备阶段实测）或 4 票投票；没解码完的按错。部署目标：8B 每张裁剪图 ≤ 0.5 GPU 秒（按 2.69 美元/GPU 时折算约 0.0004 美元〔vlm-ft F11〕），每页 ≤ 5 GPU 秒。

**GPU 分配（S1）**：0–2 号训练；3 号评测服务；4 号安慰剂（每约 10 次初筛 1 个）、第二个评测服务与早晨重测；5–7 号先建 W0，之后跑 R2、排名一致性审计、Hyperband 和数据引擎的教师模型。

**实验档位**：R0 60 秒冒烟 → R1 初筛（1 个种子）→ R1 复现（再加 2 个种子）→ 隐藏确认集（只评测）→ R2（W0-8B 上的 8B LoRA，3,200 万 token，约 2.6 小时〔吞吐未核实〕；晋升复核用 2 个种子，排名一致性审计用 1 个；每周或按需）→ R3 关口发布。

**W0（相当于上游的分词器，每个阶段固定）**：Qwen3-VL-2B 先 LoRA（r = 64，30 万样本，约 3.4 GPU 时），再冻结 ViT、分模块学习率做选择性全参（120 万样本）——因为 MolSeek-OCR 直接全参 SFT 发散了〔ocsr F16〕。约 18 GPU 时，之前做 4 点学习率扫描（约 12 GPU 时）。训练目标按提示标签支持多种输出格式，超原子如实保留。W0-8B 是 120 万样本上的 r = 128 LoRA，4 卡约 12 小时。每次 S1 关口后用当前最优配方重建一次热启动点，记为 W1。

## 5. 保留/丢弃规则与设计理由

完整步骤由 runner 执行，agent 侧的说明见 `program.md` 第 6 节。

| 步骤 | 条件 | 不满足时 |
|---|---|---|
| 门禁 | 只改了 `train.py`、通过扫描、`INFER_CONFIG` 合法、描述符的键已登记且描述符是新的 | `gate_fail` |
| 运行凭证 | adapter 可加载、计数一致、权重变了、`infer_config.json` 等于 `INFER_CONFIG` | `receipt_invalid` |
| 初筛（种子 s0，与在位者已存的 3 个种子配对） | 收缩后 Δ（收缩上线前用原始 Δ）≥ +[0.5] pp 且 `p_not_better` ≤ [0.20]，护栏不否决 | `discard` / `veto:*` |
| 复现（补跑 s1、s2，三对三） | 平均 Δ ≥ +[0.5] pp、`p_not_better` ≤ [0.05]、≥ 2/3 配对为正 | `replicate_fail` |
| 隐藏确认集（≤ [10] 次/天，只返回通过/不通过） | 平均 Δ > 0、`p_not_better` ≤ [0.10]、没有来源切片或难度标签下降超过 max([2] pp, 2.5·SE) | `confirm_fail` |
| 长程复核（长程敏感改动，或已堆了 2 个未复核的保留） | R2 下界 ≥ 0 | `r2_revert` |
| 推进分支 | — | `established` |

**长程敏感**（短跑结论可能不迁移到长训练，叠加前必须过 R2）：学习率、调度、warmup、分辨率（`max_pixels`）、切块（`tiling`），以及解冻 ViT 等视觉侧改动——2B 与 8B 的视觉塔规模不同（第 11 节）。描述符键的封闭清单（`program.md` 第 5 节，标 H 的键即长程敏感）写在 `keep_rule.yaml` 里，runner 按键自动标记；未登记的键在门禁被拒，防止换个写法绕过失败想法表和 R2。

设计理由：
- **种子方差必须进入检验**：旧做法"把种子合并后只按文档 bootstrap"是反保守的（`harness/README.md` 的零效应模拟）；**公共随机数**：runner 管种子，数据流不变时数据顺序与在位者相同，减小配对差的方差。
- **赢家诅咒**：约 100 个零效应初筛里最好的会高出约 2.5 SE（≈ 1.9 pp）〔metrics F24〕，所以初筛永远不推进分支；种子共享开发集题目，消除不了题目层面的诅咒，因此确认集必须用**全新题目**，每天 ≤ 10 次、100 次查询后或每月从储备集轮换 30%〔metrics F25，未核实〕。
- **在位者也可能是运气好**：每天 07:00 用 3 个新种子重测在位者，若均值低于记录 1 个总标准误以上，替换它的基准文件。**种子约定**：候选总是用在位者当前基准文件的种子编号（初筛用第一个，复现用另外两个）；基准被替换后，runner 对之后的候选改发新的种子编号，公共随机数的配对关系保持不变。
- **经验贝叶斯收缩**：显示和排序用的 Δ 先向总体均值收缩，抑制"小样本上的大增量"。
- **安慰剂实验**：4 号卡每约 10 次初筛跑 1 个不改任何东西的"候选"（不占训练卡），按同样的规则判定；目标（滚动 50 次）：promising ≤ [30]%（名义约 20%，v1.8 的零效应模拟为 13.5–17.5%；30% 的门槛在校准良好时误报警约 3%）、dev_keep ≤ 5%、established ≤ 2%、否决 ≤ 10%；超标就把 z 提高 0.25 并通知人类。误保留率要实测，不能只靠理论推算。
- **层级使用**：可视化集给逐条结果；开发集只给四舍五入的值（0.1 pp，分组 0.5 pp）；确认集只告诉 agent 通过或不通过；封存测试集只在关口和最终发布由人类使用（8 周内 ≤ 4 次）。最终选择：R2 上前 3 名的 `established` 状态进确认集，只有赢家进封存测试集〔ecosystem F16〕。
- **简化（非劣效）**：净删 ≥ 20 行、平均 Δ ≥ −[0.2] pp 走非劣效路径（不要求更好，只要求不明显更差），确认集 90% 下界 > −[0.5] pp 即保留，每天最多 1 次，每周有漂移检查。
- **两个停滞信号**：**停滞通知**（连续 [2] 晚或 [150] 个候选没有 `established`）→ 通知人类改"当前方针"；**关口平台期**（连续 [3] 晚且 ≥ [150] 个候选没有 `established`）→ S1 关口条件之一。

## 6. 两个审计：排名一致性与 Hyperband 基线

**排名一致性审计**（2B 短跑的排名能否代表 8B 长跑）：用 8 个配方算 Spearman ρ 判不出任何东西（ρ = 0.6 的 95% CI 约 (−0.2, 0.9)）。所以取 24 个覆盖整个 R1 增量范围的配方（6 个最好、12 个中间、6 个最差）在 R2 上各跑 1 个种子（约 62 GPU 时），其中 8 个在两个档位都再补 1 个种子（约 22 GPU 时），测出各自的重测信度 r1、r2。**通过**：Fisher-z 变换下 ρ 的 95% 下界 ≥ [0.3]（ρ = 0.6、n = 24 时 CI 约 [0.26, 0.81]）、去衰减 ρ/√(r1·r2)（扣除两档各自测量噪声后的相关）≥ [0.6]、且 ≥ [75]% 的 `established` 在 R2 上为正。CI 跨过 0.3 就再加 12 个配方，期间最多堆 2 个未复核的保留。**失败**（去衰减 ρ < [0.4]）：预算改为 1,600 万 token 或改用 4B 作短跑模型；第二次失败就把搜索交给 R2 上的 Hyperband。视觉侧改动的排名一致性尤其没把握，单独报告。

**同等 GPU 时的非 agent 基线**（第 4 周）：Optuna-TPE + ASHA（先用小预算淘汰、再给幸存者加预算；200 万 → 800 万 → 3,200 万 token），在 `program.md` 的数值旋钮上搜索，同样的 W0 和保留规则，84 GPU 时，等于 agent 前两晚的用量〔ecosystem F36, F40〕。比较指标是"每 GPU 时确认的增益"。如果脚本化扫描做得一样好，旋钮就交给 Hyperband，agent 只负责代码层面的改动。

## 7. 搜索空间

S1 的 22 条想法、长程敏感标记和已知负面结果见 `program.md` 第 9 节。前 4 晚做 1–10（学习率、真实数据比例、增强课程、超原子标签、手性、LoRA 目标、分辨率、EMA、难例加权、回放）；第 4 周起做输出格式、图遍历推理、投票、切块，以及 RL 通道（MRT、按 score.py 结构键给奖励的 GRPO/DAPO）。RL 通道有固定的 1,800 GPU 秒预算（墙钟上限 [2,000] 秒，含加载与保存；常量在 `prepare.py`，普通实验的 1,500 秒上限装不下它），并自动配一个同等 GPU 秒、同样墙钟上限的继续 SFT 对照，因为采样算力逃过了 token 计数。

MRT 奖励写作 0.1·valid + 0.5·sim·valid + 0.4·exact：COMO 的公式本身就让相似度项只在结构有效时计入〔ocsr F8；metrics F5〕；本方案把 exact 改成 score.py 的结构键相同（含立体层与增强立体），而不是 Tanimoto = 1。

S2 追加：表格裁剪 + 母核裁剪 + 脚注条的多视图输入；用文档自带编号（"1a"）作视觉提示（IdtVP）〔doc-chem-ie F13〕；输出 `rgroup_table_v1` 由 harness 枚举〔doc-chem-ie F23〕；YAML 记录（olmOCR 从 JSON 改为 YAML 后重试与复读显著减少〔vlm-ft F13〕），扁平 JSON-lines 作为待检验的替代；在扰动的真实版面上合成 SAR 表〔data F10〕。

## 8. 反作弊清单

- **隔离与门禁**：第 1.3 节。
- **种子**：runner 下发；三种子 + 确认集（Claudini 在停滞后退化成了"搜种子"〔ecosystem ADDED〕）。
- **反馈粒度**：开发集四舍五入、可视化集单独留出、确认集只有通过/不通过且有额度。
- **推理配置**：`INFER_CONFIG` 是 `train.py` 的模块级字面量，门禁在训练前用 `ast.literal_eval` 读出、按 schema 校验，提示词扫描类 SMILES 字符串；越界或不是字面量 → `gate_fail`；写出的 `infer_config.json` 与字面量不一致 → `receipt_invalid`。
- **评分器严格**：无效、重复、缺失都算模型的错；禁止 best-of-N；未知缩写算错〔metrics F15, F16〕。
- **监测**：金丝雀与扰动报警；每晚对训练样本 ID 做重叠审计；运行凭证绑定代码、数据、权重和金标；每周用另一家族的模型扫一遍 agent 轨迹，只标记、不打分〔metrics F35〕。LLM 永不做评分函数：像 "Thought process:"、":" 这样的"万能钥匙"回答可让 LLM 评审的误判率高达 80%〔metrics F35〕。
- **确认作弊时**：冻结、修补 harness、重测在位者，相关行标为 `hack`。

## 9. S2 的记录语义

评委一致认为，v1.5 的记录键对药化分析来说太"薄"——生化 IC50 与细胞 IC50 互换、大鼠与人的清除率互换都会被判为正确。score.py v1.6 已实现第 1–2 条的基本版本，其余待补：

1. **记录键 v2**：结构、终点（封闭词表）、靶点（经 `targets_v1.tsv` 映射到 UniProt 或物种）、突变体、实验类型、物种、细胞系/基质、pH、剂量、途径、条件、关系符、单位类、数值。复合单位（mL/min/kg、ng·h/mL、10⁻⁶ cm/s）按量纲换算。
2. **"金标定义字段"规则**：上下文字段只有金标填了才计分。推断出正确的物种有奖励，乱填没有，金标有而预测漏掉算错。
3. **专利分级活性**：金标同时存印刷的标签（"A"、"+++"）和按图例解析出的区间，两者都算对（待补）。
4. **现在采集、S2b/S3 再评分**：n、±SD、重复序号；`stereo_status`；ADMET/PK 的制剂、动力学 vs 热力学溶解度、Caco-2 外排比、CYP 亚型；出处（页、框、表、行、列、脚注）；实验方法段落与 `assay_uid`（链接正确率先作 S2 诊断）。时间点（`time`/`time_unit`）score.py 已在金标填写时计分。
5. **标注范围规则**（`annotation_guide.md`）：一条记录 = 一个化合物 × 终点 × 体系 × 条件下印刷的一个数值；参照化合物标 `role=reference`；中间体只在有数据时计入；引自他文的值保留并标 `cited=true`；论文与 SI 的重复按文档分别计分；`compound_id` 按原文印刷（"Example 12, Step 3"）；立体和 E/Z 按画出的计分；文中报告的计算值标 `method=calc`。没有这些规则，精确率就没有定义。
6. **规模**：开发集 140 张表（约 70 篇文档：内部 80、BioVista 验证集 40（按 PDB ID 取 PDF）、ChemX 的 MIC-菌株表 20，µg/mL 按分子量换算）；确认集 100；封存集 60（EPO/CNIPA，截止日期之后）。错误按表成片出现，有效样本量接近表格数：标准误约 2–3 pp，最小可检效应约 6 pp，所以 S2 只能找大杠杆。组内相关在第 5 周实测，不借用文献值（metrics F22 的 2.9 是假设值）。S2 的宏平均按文献类型（专利/论文/SI/报告）计算，用金标的 `doc_type` 字段（`--group-field doc_type --slice-weights patent=1,paper=1,si=1,report=1`）；`slice` 仍表示来源。

## 10. 数据策略

- **固定数据池**（每条样本带许可、轨道、来源、`family_id`）：`render`（200 万 PubChem 分子；ChEMBL 分子是 CC BY-SA 3.0，训练商用模型是否触发相同方式共享尚无定论，法务确认前不进商用轨道；剔除评测分子的 InChIKey 第一段和 SAR 系列骨架，用 RDKit、Indigo、CDK 以 RanDepict 风格渲染，两种标签视图）；`uspto`（MolScribe 的 68 万对，清洗：`*` 重映射、CDX 优先于 MOL、多模型分歧样本剔除，并去掉评测专利族和经典基准专利；第 4 周换成自己挖的 2001–2024 版）；`stereo`（Stereo-200k）；`markush`（MarkushGrapher-1 数据，CC BY 4.0；USPTO-MOL-M 待法务）；`inhouse` 与 `replay`；S2 的 `sar`（在扰动真实版面上合成、用 ChEMBL/BindingDB 数值填充的 SAR 表，加 ChemX 与内部训练表；ChEMBL 与来自 ChEMBL 的 BindingDB 行同样待法务）。
- **标签保真**（harness 统一执行）：超原子如实保留；`*` 只用于画出来的通配符；立体只标画出来的；反离子按画出的保留、按母体评分。
- **泄漏控制**：按专利族/文章切分；InChIKey 第一段跨池去重；Murcko 系列整体留在一层并从渲染中拉黑；感知哈希去重；封存集按时间切分；每晚重叠审计。
- **数据引擎**（离线、人工把关、每周更新后重测基线）：收集商用轨道页面；从多模型一致度 0.6–0.9 的区间采样；数字原生 PDF 用 ChemScraper 的矢量标签。化学家修正约 30 秒/分子，双人复核下每 40 小时约 2,400 张。
- **固定的**：数据池、评测层级、生成器、词表、W0、上限、预算。**agent 可调的**：权重、特征筛选、增强代码、渲染样式、课程、标签视图、损失、输出格式。

## 11. 基座模型

- **内循环：Qwen3-VL-2B-Instruct**。Apache-2.0；OCRBench 858（Thinking 版 792）〔vlm-ft F1, F2〕；LLaMA-Factory、ms-swift 已支持〔vlm-ft F29〕，TRL 与 vLLM（含 LoRA 热切换）需第 1 周验证。
- **R2 与生产：Qwen3-VL-8B-Instruct**（OCRBench 896；8B LoRA 单张 H100 放得下〔vlm-ft F30〕）。与 2B 共享分词器、对话模板和视觉分块方式（16 像素 patch、2×2 合并，即每 32×32 像素一个 token），但视觉塔规模不同（Hugging Face 配置：2B 24 层/1024 维，8B 27 层/1152 维），所以视觉侧改动能否迁移必须在 R2 实测。备选短跑模型：Qwen3-VL-4B（881）。
- **第 1 周小比武**（零样本 + 2 小时同配方 SFT）：只有差距 ≥ [3] pp 才换成 InternVL3.5-2B/4B（Apache；836/822）或 Gemma 4 E4B/12B（Apache）。Qwen3-VL 的中文 OCR 能力对中文专利和论文是额外的加分项。
- **排除**：Qwen3.5（默认思考模式、依赖 flash-linear-attention、DeltaNet 状态在打包样本间泄漏〔vlm-ft F4〕）；Gemma 3（固定 896 像素）；AGPL/非商用模型〔vlm-ft F35〕；化学专用 VLM（ChemVLM 8.03、ChemDFM-X 19.06，修订版 MolRecBench-Wild〔ocsr F10〕）。

## 12. 下游有效性与金标质量

**下游有效性**（S2 关口，30 篇有 ChEMBL/BindingDB 覆盖的留出文档）：
1. 数据库记录召回（log10 差 ≤ 0.01 算一致；只看召回，因为数据库只收录子集且做过标准化〔doc-chem-ie F30〕）。
2. 化学家修正时间 vs 预填后的人工录入（目标节省 ≥ [40]%；MolScribe 的实验里，从头画一个约 35 原子的分子要 137 秒，给出预测 SMILES 后 39 秒，叠加预测图结构后 20 秒〔ocsr F2〕）。
3. 至少 4 个模型上，记录 F1 与化学家整理时间的排名一致性。
4. 报告 F0.5（更重视精确率的 F 分数），供 SAR 建模参考。

**金标质量**：20% 的金标双人标注，记录级一致性 F1 就是天花板；每周挖掘分歧、只由人修正、金标版本化后重测基线〔metrics F21〕；缩写表必须能复现每一个含超原子的金标。

**基座污染探针**：比较零样本 2B/8B 在 MolRecBench-Wild 各年份、以及版式匹配的内部文档 2026-07-01 前后的表现，截止日期之前的优势 > 2 SE 就标记该来源。
