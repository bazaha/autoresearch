# drug-vlm autoresearch —— S1：单分子裁剪图 → 化学结构（program.md v1 草案）

> **状态：草案，当前不可运行。** 下面"前置条件"表里标为"首夜必需"的组件全部验收之前，不要启动本循环。
> agent：如果 `./run.sh --check` 不存在，或打印了任何 FAIL，停下来告诉人类，**不要自己搭建任何缺失的组件**。
> DEFERRED（可推迟到第 3–4 周的项）不阻止开跑：DEFERRED 期间，初筛用未收缩的 Δ，`canary_gap_pp`、`perturb_gap_pp` 显示为 `-`。
>
> 这是给 agent 的研究方针，对应上游仓库根目录的 `program.md`。**人类负责迭代这个文件，agent 只读。**
> 方括号里的数字（如 `[0.5]`）是准备阶段校准之前的默认值，校准后由人类写入 `harness/keep_rule.yaml` 并冻结；
> 两者不一致时以 `keep_rule.yaml` 为准。设计依据见 `README.md` 第 6 节与 `docs/design.md`（其中的数字若与本文件或 `keep_rule.yaml` 不一致，以后两者为准）。

这是一个让 LLM 自己做研究的实验：在固定的数据、固定的评测和固定的训练预算下，
把一个视觉语言模型（VLM）微调得更会"读"药物研发文献和专利里画的化学结构。

## 0. 工作目录与术语

- **本项目的所有路径都相对 `research/drug-vlm/`，开始前先 `cd research/drug-vlm`。**
  仓库根目录的 `prepare.py`、`train.py`、`program.md` 是上游的 GPT 预训练示例，与本任务无关：根目录只读 `README.md`（了解原始范式），其余不要读也不要改。
- 你唯一可以修改的文件是 `research/drug-vlm/train.py`。

| 术语 | 含义 |
|---|---|
| 在位者（incumbent） | 当前最好的提交，即分支 `autoresearch/<tag>` 的 HEAD。**只有 runner 能移动这个分支** |
| W0 | 热启动检查点（Qwen3-VL-2B 上一次性训练好的结构识别模型），相当于上游的分词器：只读、每个阶段固定。每次实验都从 W0 开始 |
| DEV / 开发集 | 约 2,800 张裁剪图（第 3 周加入内部 600 张后约 3,400），决定保留/丢弃；你只看到四舍五入的聚合值 |
| CONFIRM / 隐藏确认集 | 约 1,700 张（之后约 2,100）从未用于开发的裁剪图；只告诉你通过或不通过，每天最多 [10] 次 |
| SEALED / 封存测试集 | 只在阶段关口由人类使用，你永远看不到 |
| dev-viz / 可视化集 | 约 250 张（之后约 300）单独留出的裁剪图，**给你看逐条预测与金标**，用于错误分析 |
| 来源切片 / 难度标签 | `slice` 是数据来源（`inhouse`、`biovista`、`uspto`、`mrbw` = MolRecBench-Wild）；`tag` 是难度子集（`chiral`、`abbrev`、`large`、`lowres`），一张图可以有多个标签。两者合称"分组" |
| 层次 bootstrap | 同时重采样专利族和种子的配对检验（`score.py compare`），给出增量的置信区间和 `p_not_better` |
| R0 / R1 / R2 | 实验的"梯级"：R0 = 60 秒冒烟测试；R1 = 2B 模型在固定预算下的正式实验（本循环）；R2 = 8B 模型的长程复核（runner 每周或按需运行） |
| pp | 百分点。本文件和 run.sh 输出里所有增量都以 pp 为单位、带 `_pp` 后缀；比率带 `_pct` 后缀 |
| 准备阶段 | 第 1–2 周的基础设施与噪声校准（README 第 8 节）。不要和 S1/S2 阶段混淆 |
| 收缩后的 Δ | 把带噪声的增量向所有候选的平均值拉回（经验贝叶斯），抑制小样本上的偶然大增量；只用于排序和初筛门槛。收缩上线（第 3–4 周）之前用原始 Δ |
| 运行凭证 | runner 为每次运行记录的核验结果：adapter 能在 W0 上加载、计数与步数一致、权重确实变了、所有输入输出的 SHA-256 |
| 当前方针 | 人类每天写的 2–5 句方向调整，在 `/srv/dvlm/policy/current.md`（root 所有、你只读、不在 git 里）；`./run.sh --status` 第一屏显示它和更新时间。与本文件冲突时以它为准 |

## 1. 当前阶段与目标

**阶段 S1：一张只含一个分子的裁剪图 → 该分子的结构。** 为什么先做结构：结构是后续每一条活性/性质记录的连接键。
BioMiner 在 BioVista 上数值召回约 0.85、结构召回只有 0.58；去掉结构模块后，（化合物, 靶点, 数值）三元组 F1 从 0.32 掉到 0.011。

**主指标 `val_struct`**（越高越好）：

`val_struct = Σ_s w_s · struct_f1_s`，按 DEV 的数据来源加权：内部文档 0.30、BioVista 0.30、USPTO 2025–26 0.20、MolRecBench-Wild 0.20
（内部文档第 3 周才加入，之前按出现的来源重新归一化，加入时 runner 会重测在位者）。

- 每张裁剪图上，`struct_f1` 就是精确匹配：预测结构与金标结构的**标准化母体分子**（去盐、中和）的**带立体信息的标准 InChIKey** 相同才算对；
  外消旋/相对构型（CXSMILES 增强立体 `&1`/`o1`）和带 R 基团的结构按规范 CXSMILES 比较。
- **立体是严格的**：画了楔形键却输出无立体，或者凭空编出立体，都算错。Tanimoto 相似度只做诊断，从不参与判定。
- 无法解析、无法转换、用了缩写表里没有的缩写、超时或没有输出——都算错，而且**没有输出和答错同价**（弃答不会更划算）；多给的结构算假阳性。
- **与输出表示无关**：你的模型用 SMILES、CXSMILES、molfile 还是原子/键 JSON 输出都可以，`harness/formats.py` 先把它转成 RDKit 分子再打分。
  就像 `val_bpb` 与词表大小无关，"换一种输出表示"是可以公平尝试的实验。
- 计算方式（runner 负责填入金标与预测文件）：`python3 harness/score.py score --gold <开发集金标> --pred <预测> --scalar struct_f1 --slice-weights inhouse=0.3,biovista=0.3,uspto=0.2,mrbw=0.2`。
  你可以读 `harness/score.py` 和 `harness/README.md` 来理解指标，但它们是只读的。

S1 的 DEV 和 CONFIRM **不含 Markush 通式**。Markush（IP5-M/M2S/USPTO-M）每晚单独作为诊断报告，不参与保留/丢弃。
SAR 表格与活性记录是 S2（第 5 周起），届时会有新版本的 program.md。

## 2. 前置条件（准备阶段的交付物）

| 组件 | 作用 | 状态 | 首夜必需 | 验收方式 |
|---|---|---|---|---|
| `harness/score.py` + `test_score.py` | 固定评分（相当于 `evaluate_bpb`；说明见 `harness/README.md`） | 已有 v1.8 | 是 | `python3 -m unittest discover -s harness -p "test_*.py"` 全部通过；`score` 输出的 `scorer_version` 行值为 1.8（`grep -E '^scorer_version:[[:space:]]+1\.8$'`） |
| `harness/null_sim.py` | 检查配对比较在零效应下的误保留率 | 已有 | 是 | `python3 harness/null_sim.py`：层次 bootstrap 的误保留率不高于名义 5%；`python3 harness/null_sim.py 400 screen` 与 `400 screen-latent`：每一行初筛的安慰剂 promising 率 ≤ [25]%（名义约 20%；每行 400 次试验时，校准良好的检验超过 25% 的概率 < 1%；v1.8 实测 14–22.5%） |
| `harness/formats.py` + `abbrev_v1.tsv` + `golden_pairs.tsv` | 各种输出格式 → RDKit 分子；超原子缩写展开 | 待建 | 是 | 疑难分子回归集（约 200 对，覆盖互变异构、盐、E/Z、增强立体、R 基团、同位素、金属）全过；缩写表能复现每一个含超原子的金标 |
| `harness/dvlm/`（`load_base`、`pools`、`Meter`、`render`、增强算子、`save_adapter`） | 训练侧的唯一入口 | 待建 | 是 | Meter 计数与优化器步数相差 ≤ 2%；`render` 的数据门禁能拦截评测分子 |
| `harness/dvlm/rewards` | 训练用奖励，复用 score.py 的结构键 | 待建 | 否（RL 通道第 4 周才打开） | 在疑难分子回归集上与 score.py 判定完全一致 |
| `harness/prepare.py` | 常量（STAGE、TOKEN_BUDGET、各种上限）；构建数据池、评测层级和 W0 | 待建 | 是 | `./run.sh --check` 的清单与哈希检查通过 |
| W0-2B 检查点 | 热启动 | 待建 | 是 | DEV 上 ≥ 零样本 + 20 pp |
| `harness/evalserver.py`（vLLM + LoRA 热切换） | 固定推理，支持第 4 节 `INFER_CONFIG` 的全部字段 | 待建 | 是 | 同一 adapter 评两次，DEV 分数差 ≤ [0.1] pp；`INFER_CONFIG` 全字段生效 |
| `harness/probes.py` + 探针集 | 计算 `probe` 及其配对 bootstrap SE、负对数似然 | 待建 | 是 | 探针集就位；同一 adapter 重算差 ≤ [0.5] pp；输出 SE |
| 金丝雀集与扰动集（`probes.py` 的另一半） | `canary_gap`、`perturb_gap`（只报警） | 待建 | 否 | 两个集合就位 |
| `harness/keep_rule.py` + `keep_rule.yaml` | 第 6 节的判定流程；描述符键清单（第 5 节） | 待建 | 是 | 20 个手动安慰剂中被判为 promising 的 ≤ [6] 个（名义率 20%，20 个里超过 4 个的概率约 37%，所以门槛留余量）；阈值冻结；描述符键清单与第 5 节一致 |
| 经验贝叶斯收缩 | 显示和排序用的 Δ | 待建 | 否 | 上线前初筛用原始 Δ |
| `harness/runner.py` + `run.sh` | 门禁、容器、运行凭证，`results.tsv` 与 `lineage.jsonl` 的唯一写入者 | 待建 | 是 | 20 次试运行的运行凭证全部有效 |
| Bloom 过滤器、`chattr +i`、签名运行凭证、自动安慰剂注入 | 强制机制的第二层 | 待建 | 否 | 第 3 周补齐；之前校准用的 20 个安慰剂由人手动提交 |
| 评测层级 `/srv/dvlm/eval/{viz,dev,confirm,reserve,sealed,probe}` | 按文章/专利族划分，InChIKey 第一段跨层去重 | 待建 | 是 | 重叠审计为 0；20% 双人标注 |
| `/srv/dvlm/policy/current.md` | "当前方针"（可以为空） | 待建 | 是 | 文件存在、root 所有、agent 账户只读 |
| 隔离：文件权限、分支保护、无网络训练容器 | 最低可用配置的核心（`docs/design.md` 2.2 节） | 待建 | 是 | 以 agent 账户读取 `/srv/dvlm/eval/*/gold` 与 `/srv/dvlm/private/` 失败；agent 账户推进 `autoresearch/<tag>` 被拒；训练容器内联网失败、环境变量里没有 API key；提交 `train.py` 以外的文件返回 `gate_fail` |
| `train.py` 基线 | W0 + LoRA 的最简单配方（约 400 行） | 待建 | 是 | `./run.sh --smoke` 60 秒内跑通 |
| `.gitignore` 包含 `notes/`、`requests.md`、`results.tsv`、`lineage.jsonl`、`failed_descriptors.tsv`、`runs/` | 这些文件不进入提交 | 待建 | 是 | `git status` 干净 |
| 环境：8×H100 节点、支持 `--network none` 的容器运行时、`uv.lock` 锁定 transformers/peft/trl/vllm/`rdkit==2026.09.1`、验证过 Qwen3-VL LoRA 热切换的 vLLM | 可复现的运行环境 | 待建 | 是 | 同一镜像两次构建哈希一致；LoRA 热切换跑通（否则改用合并权重，每次多 3–4 分钟） |

构建顺序、负责人和人日估算见 `docs/design.md` 第 2 节；"首夜必需"一列就是那里的"最低可用配置"。`./run.sh --check` 逐行运行上表的验收项，打印 PASS / FAIL / DEFERRED：首夜必需的项只有 PASS 或 FAIL；可推迟的项在补齐之前显示 DEFERRED。

## 3. 准备工作（Setup）

与人类一起完成：

1. **约定运行标签**：基于日期，例如 `oct29-s1`。人类给出 tag 后运行 `./run.sh --init oct29-s1`：runner 从当前基线 `train.py` 创建分支 `autoresearch/<tag>`，把 tag 记到 `/srv/dvlm/run_tag`；已存在的 tag 会被拒绝。之后 `./run.sh --status` 会显示它。
2. **通读这些文件**：
   - 本文件，以及"当前方针"（`./run.sh --status` 第一屏）；
   - `README.md`（本项目的调研报告）：读第 3 节（技术现状）和 4.3 节（合成数据），它们是想法先验的来源。第 5、6 节和 `docs/` 是给人类的设计说明，其中的数字若与本文件或 `keep_rule.yaml` 不一致，以后两者为准；
   - `train.py`：你唯一可以修改的文件；
   - `harness/dvlm/` 各函数的 docstring（接口草案见 `docs/design.md` 2.3 节）：数据、预算、渲染、奖励和保存都只能通过它；
   - `harness/score.py`：只为理解指标而读；
   - `results.tsv`、`lineage.jsonl`、`failed_descriptors.tsv`（如果已有）：以前试过什么、为什么失败。
3. **确认环境就绪**：`./run.sh --check`。打印了任何 FAIL 就告诉人类，不要自己修；DEFERRED（可推迟到第 3–4 周的项）不阻止开跑。
4. **基线**：`./run.sh --baseline`。runner 用 3 个种子跑**未修改的 `train.py`**，作为第一个在位者。等它结束。
5. **向人类确认后开始循环。**

## 4. 你能做什么、不能做什么

**可以：只修改 `train.py`，文件内一切都可以改：**
- 数据：各数据池的采样权重、按样本特征筛选和加权、课程（由易到难）、增强代码、渲染样式、标签视图（超原子如实保留 vs 展开）；
- 模型：LoRA 目标模块与 rank（≤ 128）、各模块各自的学习率与调度、EMA/检查点平均、批大小与步数的取舍；
- 损失：包括用 `dvlm.rewards` 做最小风险训练（MRT）或 GRPO 类强化学习（见下文"RL 通道"）；
- 推理配置：`train.py` 里的模块级字面量 `INFER_CONFIG = {...}`（只能写常量；门禁用 `ast.literal_eval` 读取，**在训练前**校验，evalserver 只认这些字段，均有上限）：

| 字段 | 取值 |
|---|---|
| `prompt` | ≤ 2,000 字符，不得含类 SMILES 字符串（会被扫描） |
| `output_format` | `formats.py` 注册表中 S1 可用的一项：`smiles`、`cxsmiles`、`molfile`、`graph_json_v1`；加前缀 `cot+`（如 `cot+smiles`）表示先自由推理、只解析最后一个 `<answer>…</answer>` 里的内容（`rgroup_table_v1`、`records_yaml_v1`、`iupac_name` 只在 S2 开放） |
| `grammar_id` | `none` 或注册表中的语法（如 `smiles_v1`），用于约束解码 |
| `max_pixels` | ≤ 1024²；`pad_to_square`：`true`/`false`（保持长宽比的填充） |
| `tiling` | `none` 或 `auto`（长边 > [1,024] 像素的图额外以 2×2 切块 + 全图的多图形式输入） |
| `decoding` | `greedy`，或 `vote`（k ≤ 4，温度 ≤ 1.0，按 score.py 的结构键多数投票，外消旋与对映体不会被合并） |
| `max_new_tokens` | ≤ [2,048] |

- 用 `dvlm.save_adapter(model, INFER_CONFIG)` 同时写 `/out/adapter` 和 `/out/infer_config.json`（内容必须与字面量一致，否则运行凭证无效）。没有 `INFER_CONFIG` 时 evalserver 用 W0 的默认配置（W0 默认提示、`smiles`、`grammar_id: none`、`max_pixels: 512²`、`pad_to_square: false`、`tiling: none`、`greedy`、`max_new_tokens: 1024`）；字段越界、不是字面量或提示词含类 SMILES 字符串 → 门禁在训练前返回 `gate_fail`（不花训练卡时间），不会静默截断。
- 只在 LLM 线性层上的 LoRA 走快路径（热切换）；改了 ViT 或全参的走慢路径（合并后重新加载，评测多约 4 分钟，计入评测墙钟）。

**不可以：**
- 修改 `train.py` 以外的任何文件。提交里出现其他文件 → `gate_fail`。**提交时只用 `git commit train.py`（或 `git add train.py`），不要用 `git add -A`。**
- 让 `train.py` 超过 1,500 行，或导入 `subprocess`、`ctypes`、`socket`、`multiprocessing.connection` 等进程/网络模块（静态扫描拒绝 → `gate_fail`）。
- 联网、读 API 密钥、新增依赖、加载 W0 以外的任何检查点（训练容器没有网络，`from_pretrained` 会被静态扫描拒绝）。
- 绕过 `dvlm.pools()` / `dvlm.render()` 读写数据，或篡改 `dvlm.Meter`。训练 token 由它计数，数据门禁也在它里面。
- 设置随机种子。种子由 runner 通过 `DVLM_SEED` 下发：候选用在位者当前基准文件的种子编号（初筛 s0，复现 s1、s2），与在位者共享数据顺序，用于配对比较；每天 07:00 的重测用新种子替换基准后，runner 对之后的候选改发新的种子编号。
- 读取评测路径，或在代码里写入评测集的结构、InChIKey、文献号：长度 ≥ 15 的字符串字面量会和评测集的 Bloom 过滤器比对，命中即 `gate_fail`。
  也不要凭记忆往代码里写分子来造训练样本（无法做许可和泄漏检查）；新分子只从 `dvlm.pools()` 来。处理缩写所需的短片段不受影响。
- 试图影响评测：评测在另一个容器里用固定代码加载你的 adapter 和 `infer_config.json`，评分在第三个容器里完成，分数从不来自你的 stdout。

**预算（固定，在 `harness/prepare.py` 里）：**
- **训练预算按 token 计：每次实验 [8M] 个训练 token**（512² 的图约 18k 张），由 `dvlm.Meter` 统计每个带梯度经过模型的 token（图像 token + 提示 + 目标）。
  按 token 而不是按时间计，是为了让"更高分辨率""更长的输出"自己为成本买单，也让机器负载不影响结果。
  `Meter` 用完后保存 adapter 并退出。
- **训练容器墙钟硬上限 [1,500] 秒**（含加载 W0 约 45 秒）。到时还没存出 adapter → `timeout`。
- **评测解码上限 [600] 秒/次**（约为基线的 6 倍），够用推理链或 4 票投票；没解码完的样本按错误计，超过 1% 的样本没解码完 → `veto:decode`。
- **部署成本**：runner 按每张图的解码 token 数外推 8B 模型的成本，超过 [0.5] GPU 秒/张 → `veto:cost`。
- 显存不是否决项；爆显存就是 `crash`。
- **RL 通道**（MRT、GRPO、自蒸馏等任何在训练中采样生成的方法）：采样算力逃过了 token 计数，所以单独计预算。
  当前 **[关闭]**，第 4 周由人类在"当前方针"里打开。打开后，在提交说明里写 `Lane: rl`：预算是 [1,800] GPU 秒（训练 + 采样），
  墙钟上限 [2,000] 秒（含加载与保存，常量在 `harness/prepare.py`）；runner 会自动加一个"在位者配方继续 SFT 同样 GPU 秒"的对照（同样的墙钟上限），
  候选必须同时优于在位者和这个对照，否则 `veto:rl_control`。

## 5. 运行一次实验

**提交说明的格式**（runner 从这里读取描述符等信息）：

```
一句话假设：手性样本 2× 过采样能提升手性标签

Desc: stereo_oversample=2
Expect: +0.8 pp 总体；chiral 标签 +2 pp
```

- `Desc:`（必填）：规范化的"改动描述符"，用 `|` 分隔的 `键=值`，例如 `lr_llm=2e-4|lora_targets=attn|lora_r=64`。**键只能取下表**（同一清单在 `keep_rule.yaml` 里）；未登记的键 → `gate_fail`，需要新键时写进 `requests.md`。

  | 想法（第 9 节） | 键（**H** = 长程敏感，runner 自动标记） |
  |---|---|
  | 1 学习率与调度 | `lr_llm`（H）、`lr_vit`（H）、`lr_merger`（H）、`sched`（H）、`warmup`（H） |
  | 2 数据池权重与筛选 | `real_frac`；`pool_<池名>`（如 `pool_uspto=0.5`）；`filter_<字段>` |
  | 3 增强与课程 | `aug_strength`、`aug_curriculum`、`render_style` |
  | 4 标签视图 | `label_view` |
  | 5 手性 | `stereo_oversample`、`wedge_swap`、`decoy_wedge` |
  | 6 LoRA | `lora_targets`、`lora_r` |
  | 7 分辨率 | `max_pixels`（H）、`pad_to_square` |
  | 8 平均 | `ema`、`ckpt_avg` |
  | 9 难例 | `hard_weight` |
  | 10 回放 | `replay_frac` |
  | 11 输出格式 | `output_format` |
  | 12 推理链 | 用 `output_format` 表示（如 `output_format=cot+smiles`），不另设键 |
  | 13 批大小 | `batch` |
  | 14 语法约束 | `grammar_id` |
  | 15 投票 | `vote_k`、`vote_temp` |
  | 16 切块 | `tiling`（H） |
  | 17 随机化 SMILES | `rand_smiles` |
  | 18–19 RL 通道 | `rl_mrt`、`rl_grpo` |
  | 20–21 视觉侧与全参 | `vit_unfreeze`（H）、`full_ft`（H） |
  | 22 损失细节 | `loss_mask`、`label_smooth` |
  | 提示词、解码长度 | `prompt`、`max_new_tokens` |
- `Expect:`（必填）：预期增量。事后对照预期和结果，是你校准直觉的方式。
- `Differs-From: <commit>` + `Changed: ……`：只在描述符与 `failed_descriptors.tsv` 中已失败的条目重复时需要，写明这次哪里不同；缺了 → `gate_fail`。
- `Fix-Of: <commit>`：修复同一想法的崩溃时使用（最多 2 次），不算重复描述符。
- 上表标 H 的键由 runner 自动标为**长程敏感**（短跑结论可能不迁移到长训练；2B 与 8B 的视觉塔规模也不同），也可以手动写 `Horizon: yes`。

**命令：**
- `./run.sh`：提交当前 HEAD，**阻塞直到这次提交得到终态**（见第 6 节的状态表）。一个想法走完初筛、复现、确认，在 3 卡上大约要 40 分钟，这是正常的。
- `./run.sh --wave cand/1 cand/2 cand/3`：同时提交最多 3 个候选分支（每个都从在位者拉出、只验证一个因素），阻塞直到全部得到终态，逐个打印结果。**推荐的常规用法**：3 张训练卡一直有活干。
- `./run.sh --smoke`：对当前 HEAD 跑 60 秒 R0 冒烟测试（排查崩溃、看吞吐），不计入结果。改了训练循环、损失、模型包装或 `infer_config` 时，runner 也会自动先跑它。
- `./run.sh --status`：第一屏是"当前方针"及其更新时间；然后是运行标签、在位者提交、排队中的候选、今天剩余的 CONFIRM 次数、各卡状态。
- `./run.sh --viz <commit>`：在评测卡上按需解码 dev-viz（约 300 张，十几秒），打印该提交的逐条预测、金标和错误类型。
- `./run.sh --init <tag>`：只在准备工作第 1 步使用（第 3 节）。

**`./run.sh` 的输出**（进度行以 `[runner]` 开头，只是进度，不要据此做任何 git 操作；`---` 之后才是终态）：

```
[runner] cand 7c1e2a9 "手性样本 2× 过采样" desc=stereo_oversample=2
[runner] gate ok · smoke 跳过（训练循环未改）
[runner] screen s0: delta_pp +0.9  p=0.11 → promising，补跑种子 1、2
[runner] replicate: delta_pp +0.7 [+0.1, +1.3]  p=0.03  seeds_pos 3/3 → dev_keep，送确认集
[runner] confirm: PASS
---
status:               established
score_pct:            72.4
delta_pp:             +0.7
ci90_pp:              [+0.1, +1.3]
p_not_better:         0.03
seeds_pos:            3/3
confirm:              PASS
worst_group:          tag/lowres -0.5
invalid_rate_pct:     0.3
multifrag_rate_pct:   0.1
probe_delta_pp:       -0.4
nostereo_f1_pct:      78.1
canary_gap_pp:        1.2
train_tokens_M:       8.0
train_s:              742
eval_s:               104
proj_cost_gpu_s:      0.21
incumbent:            7c1e2a9（autoresearch/oct29-s1 已前移）
```

你看到的数字都经过四舍五入：总分和增量到 0.1 pp，分组增量到 0.5 pp，`p_not_better` 到两位小数，CONFIRM 只有 PASS/FAIL。
`crash` 时打印日志尾部 50 行。

## 6. 结果状态与保留/丢弃

判定由 `harness/keep_rule.py` 执行，你只按结果操作。流程：

1. **门禁**：提交只改了 `train.py`、通过静态扫描和 Bloom 过滤器、描述符是新的（或写了 `Differs-From`/`Changed`）；否则 `gate_fail`。
2. **冒烟**（需要时）：崩溃 → 返回日志尾部，你可以用 `Fix-Of:` 修最多 2 次，之后 `crash`。
3. **初筛 screen**：用种子 s0 训练（与在位者的 s0 共享），和在位者已存的 3 个种子做配对比较（`score.py compare`，族 × 种子层次 bootstrap；新候选只有 1 个种子，它缺少的全局种子方差从在位者 3 个种子的差异借来）。运行凭证无效 → `receipt_invalid`。
4. **否决**（每一步都检查；SE：`invalid`、`multifrag` 和各分组来自 `compare` 的配对 bootstrap，`probe` 来自 `probes.py` 的配对 bootstrap）→ `veto:<护栏名>`。注意方向：

   | 护栏 | 方向 | 否决条件 |
   |---|---|---|
   | 无效结构率 `invalid` | 越低越好 | Δ > +max([0.3] pp, [2.5]·SE) |
   | 去盐后仍为多片段的比例 `multifrag` | 越低越好 | Δ > +max([0.3] pp, [2.5]·SE) |
   | 每个难度标签（≥ 300 张）：`chiral`、`abbrev`、`large`（> 50 重原子）、`lowres`（扫描/低分辨率）；来源切片 `inhouse` | 越高越好 | Δ < −max([1.5] pp, [2.5]·SE) |
   | 300 条内部表格单元格转写探针 `probe`（防遗忘，为 S2 保住读表能力） | 越高越好 | Δ < −max([2] pp, [2.5]·SE) |

   另有固定否决：`veto:decode`、`veto:parse`（> 1% 的样本没解码完或解析失败）、`veto:cost`、`veto:rl_control`。
   金丝雀差距（100 个从未发表的分子）和扰动差距（300 张重新编码的图）只报警、不否决，但人类会看。
5. **promising**：收缩后的 Δ（收缩上线前用原始 Δ）≥ +[0.5] pp 且 `p_not_better` ≤ [0.20]；否则 `discard`。
6. **复现 replicate**：runner 并行补跑种子 s1、s2，三对三比较。平均 Δ ≥ +[0.5] pp、`p_not_better` ≤ [0.05]、三个配对增量中至少 2 个为正、否决仍然干净 → **dev_keep**；否则 `replicate_fail`。
7. **确认 CONFIRM**：在隐藏确认集上评测这三个 adapter。平均 Δ > 0、`p_not_better` ≤ [0.10]、没有来源切片或难度标签下降超过 max([2] pp, 2.5·SE) → PASS；否则 `confirm_fail`。你只看到通过或不通过。
   当天 [10] 次额度用完 → `queued`，第二天由 runner 继续。
8. **长程复核**：长程敏感的候选，或已经有 2 个未经 R2 复核的保留堆在一起时，runner 先在 8B 上跑 R2（约 2.6 小时，吞吐未核实）→ run.sh 返回 `queued`。
   R2 下界 < 0 → `r2_revert`。
9. **established**：runner 把 `autoresearch/<tag>` 前移到这次提交，存下它的 3 个种子作为新基准；同一波里仍为 promising 的候选会被移到新在位者上重新初筛，移不过去的记为 `stale`。

**两条特殊路径（runner 用 `git diff --numstat` 自动判断，你不需要申报）：**
- **简化**：`train.py` 净删除 ≥ 20 行时走非劣效路径：初筛与复现只要求平均 Δ ≥ −[0.2] pp；CONFIRM 的 90% 下界 > −[0.5] pp → `simplify_keep`（每天最多 1 次；每周有漂移检查，防止累计下滑）。同等分数下更简单的代码就是胜利。
- **大改动**：净新增 > 60 行时，第 5、6 步的 Δ 门槛翻倍（+[1.0] pp），p 门槛不变。为了 < 0.5 pp 去写针对某类图的脆弱特判，不值得。

**状态表：**

| 状态 | 出现在 | 你要做什么 |
|---|---|---|
| `promising`、`dev_keep` | 只作为 `[runner]` 进度行 | 什么都不做，等终态 |
| `established`、`simplify_keep` | 终态 | 分支已由 runner 前移；从新的在位者开始下一个想法 |
| `discard`、`replicate_fail`、`confirm_fail`、`veto:*`、`stale` | 终态 | 删掉候选分支，回到在位者。`confirm_fail` 和平均 Δ 为正的 `replicate_fail` 会以"近似成功"留在谱系里，以后可以组合再试 |
| `crash`、`timeout` | 终态 | 读日志尾部。小错误（拼写、少导入）用 `Fix-Of:` 修好重交；想法本身行不通就放弃 |
| `gate_fail` | 终态 | 读拒绝原因（多了文件、命中字面量、描述符重复……），改好再交 |
| `infra`、`receipt_invalid` | 终态（runner 已重试一次） | 通常不是你的问题；原样再交一次或先做别的。`receipt_invalid` 反复出现时检查 `save_adapter` 写出的 `infer_config.json` 是否与 `INFER_CONFIG` 一致 |
| `queued` | 终态（对你而言） | 回到当前在位者继续做别的。runner 之后会把它变成 `established`、`confirm_fail` 或 `r2_revert`；如果在位者变了，你会在循环第 1 步看到 |
| `r2_revert`、`hack` | runner 事后写入 | 在位者可能被回退；从 `./run.sh --status` 显示的在位者继续 |

`results.tsv` 里还有 `kind=base`（基线）和 `kind=rebase`（每天 07:00 用新种子重测在位者；如果它当初只是运气好，基准会被替换）的行。
另外 runner 在另一张卡（4 号）上跑**安慰剂**实验（不改任何东西的"候选"，每约 10 次初筛 1 个，不占你的训练槽位）、重测信度测量，并在 5–7 号卡上跑 Hyperband 基线，用来实测这套门槛的误保留率。这些行只记在仓库之外、只有人类可读的 `results_full.tsv` 里。

**为什么这么麻烦**：这个指标的噪声比 `val_bpb` 大一个数量级以上。DEV 上初筛（1 个新种子 vs 3 个已存种子）的标准误约 0.75 pp，复现（3 vs 3）约 0.63 pp，能可靠检出的最小提升约 1.9 pp 和 1.6 pp；
一夜上百次初筛里挑最好的，纯噪声也能"找到"约 1–2 pp 的提升。上游社区已经多次把"换了随机种子"记成改进。
所以初筛只决定要不要花算力复现，永远不直接推进分支；真正被保留的改动会很少——这是不被噪声骗的代价。

## 7. 记录（runner 写，你只读）

`results.tsv`（制表符分隔，不入库）：

| 列 | 含义 |
|---|---|
| `commit`、`parent` | 短 SHA |
| `stage`、`rung`、`kind` | S1/S2；R0/R1s/R1r/CONF/R2；cand/base/rebase |
| `seeds`、`status` | 用到的种子；第 6 节的状态 |
| `score_pct` | 种子平均的 `val_struct`，0.1 pp |
| `delta_pp`、`ci90_lo_pp`、`ci90_hi_pp`、`p_not_better`、`seeds_pos` | 收缩后的增量（收缩上线前为原始增量）、层次 bootstrap 90% 区间、单侧 p、为正的配对种子数 |
| `confirm` | PASS / FAIL / - |
| `worst_group`、`worst_group_delta_pp` | 增量最差的来源切片或难度标签（0.5 pp） |
| `invalid_rate_pct`、`multifrag_rate_pct`、`nostereo_f1_pct`、`probe_delta_pp`、`canary_gap_pp`、`perturb_gap_pp`、`nll` | 护栏与诊断 |
| `train_tokens_M`、`samples_seen`、`steps`、`train_s`、`eval_s` | 预算 |
| `decode_tok_per_item`、`proj_cost_gpu_s`、`peak_vram_gb`、`gpu_h` | 成本 |
| `real_share`、`stereo_frac`、`abbrev_frac`、`mean_tok` | 这次训练实际看到的数据构成 |
| `horizon_tag`、`descriptor`、`description` | 长程敏感标记、描述符、假设原文 |

与 `score.py` 输出字段的对应：`score_pct` ← `score`（×100）；`invalid_rate_pct` ← `invalid_smiles_rate`；`multifrag_rate_pct` ← `multifragment_rate`；
`nostereo_f1_pct` ← `struct_f1_nostereo`；`ci90_*_pp` ← `compare --alpha 0.10` 的 `delta_ci90`；分组和护栏的增量与 SE ← `compare` 的 `slice/`、`tag/`、`guardrail/` 行；
`probe_delta_pp`、`canary_gap_pp`、`perturb_gap_pp`、`nll` ← `harness/probes.py`。

**错误类型**（`./run.sh --viz` 按可视化集逐条显示，由 score.py 的逐条字段算出）：

| 类型 | 含义 |
|---|---|
| `correct` | 完全正确 |
| `stereo_only` | 骨架对、立体错（`struct_tp = 0` 且 `struct_tp_skeleton = 1`） |
| `invalid` | 解析不出分子 |
| `multifrag` | 去盐后仍有多个片段 |
| `missing` | 没有输出 |
| `wrong_skeleton` | 其余：连接方式就不对 |

`lineage.jsonl` 每次实验一行：假设、描述符、diff 摘要、父提交、种子、各项指标、数据构成、运行凭证、状态与失败原因。
`failed_descriptors.tsv` 以描述符为键记录已失败的想法。

## 8. 实验循环

**永远循环（LOOP FOREVER）：**

1. **运行 `./run.sh --status`，先读最上面的"当前方针"（和更新时间）**：人类可能在你运行期间改过，与本文件冲突时以它为准；再看在位者是哪个提交、有没有排队的候选。
2. 读 `lineage.jsonl` 里在位者那一行、最近 15 行，以及 `failed_descriptors.tsv`；需要时用 `./run.sh --viz` 看 dev-viz 上的错误。
3. 选最多 3 个**彼此独立、各自只改一个因素**的想法。每个想法：`git checkout -B cand/<n> autoresearch/<tag>`，改 `train.py`，`git commit train.py`（按第 5 节的格式写说明）。
4. `./run.sh --wave cand/1 cand/2 cand/3`（或单个 `./run.sh`），等它返回。
5. 按第 6 节的状态表处理每个候选；用完的候选分支删掉（`git branch -D cand/<n>`）。下一轮永远从 `autoresearch/<tag>` 拉新分支。
6. 把学到的东西用两三句话追加到 `notes/`（已在 `.gitignore` 中，不要提交）：预期和结果差在哪、下一步该试什么。
7. 回到第 1 步。

**永不停止（NEVER STOP）：** 循环开始后，不要停下来问人类"要不要继续"。人类可能在睡觉。没有想法了就更努力地想：
用 `--viz` 挖错误类型、重读 `README.md` 第 3 节和它引用的论文、把以前的近似成功组合起来、尝试更大胆的改动。直到人类手动停止你为止。
runner 在连续 [2] 个夜晚或 [150] 个候选没有产生 `established` 时发出**停滞通知**（与 S1 关口的"平台期"不是同一个阈值）；人类会在"当前方针"文件里写一段新方向，你在下一轮第 1 步读到后照做。

需要新数据源或 harness 新功能时，写到 `requests.md`（不入库），由人类决定，然后继续做别的。

## 9. 想法先验

来自文献，详见 `README.md` 第 3、4 节。已发表的增益大多是从零训练时的效果，在 W0 之上要打折扣。
**H** = 长程敏感，需要 R2 复核才能叠加。前 4 个夜晚先做 1–10；第 4 周起做 11–17 和 RL 通道（18–19）。

| # | 在 `train.py` 里改什么 | 预期（边际） | 依据 |
|---|---|---|---|
| 1 | 各模块分别设学习率；调度按预算推导 | ±1–3 pp，H | 已发表配方的学习率跨约 3 个数量级 |
| 2 | 真实裁剪图在混合中的比例 10% / 30% / 50% | +1–4 | 真实数据比例是文献中最大的单一杠杆 |
| 3 | 增强强度 + 由弱到强的课程 | +0.5–2 | MolParser：40.1 → 69.5 → 76.9 |
| 4 | 标签"所见即所标"：图里画 Boc 就输出 Boc，由 harness 的固定缩写表展开 | +1–3 | MolScribe 在缩写集上 < 20%，用修正标签重训后 > 70% |
| 5 | 手性样本 2× 过采样；楔形↔波浪键替换；装饰性楔形键作为负例 | `chiral` 标签 +1–3 | 强系统的手性子集明显低于总体 |
| 6 | LoRA 目标（注意力 / gate+up / 全部线性层），rank 16–128 | ±1–2 | 只训注意力：学到 +24.9、遗忘 −0.6 |
| 7 | `max_pixels` 384²–768²；`pad_to_square` | ±1–3，H | 每 32×32 像素一个视觉 token |
| 8 | EMA / 检查点平均 | +0.3–1 | speedrun 衍生项目中 EMA 贡献约 63% 的收益 |
| 9 | 按池子里预先算好的样本特征加权难例（W0 损失、多模型一致度 0.6–0.9） | +0.5–2 | MolParser 数据引擎 |
| 10 | 5–30% 通用文档/表格回放 | 守住 `probe` | 防遗忘 |
| 11 | 输出 SMILES / CXSMILES / `graph_json_v1`（带坐标） | ±2，可能帮立体 | 输出表示对比研究 |
| 12 | 先图遍历推理、再给答案（`cot+…`） | +1–2，解码更慢 | GTR-CoT 消融 +2.3 至 +3.1 |
| 13 | 批大小 vs 步数 | ±1 | — |
| 14 | 语法约束解码（`grammar_id`） | 降低无效率 | — |
| 15 | 按结构键投票（`decoding: vote`），k ≤ 4 | +0.5–2，评测耗时 × k | MarkushGlyph MV@8 +1–4 |
| 16 | `tiling: auto` 处理大分子 | `large` 标签 +3–8，H | 大分子是公认短板 |
| 17 | 随机化 SMILES 作为训练目标 | ±1 | — |
| 18 | RL 通道：MRT，奖励 = 0.1·valid + 0.5·sim·valid + 0.4·exact（与 COMO 相同；exact 用 score.py 的结构键，由 `dvlm.rewards` 提供） | +1–5 | COMO（EditDistance 版）对比 MolScribe 的精确匹配：JPO +12.9、ACS +16.3 |
| 19 | RL 通道：GRPO/DAPO，按 score.py 结构键的二值奖励 + 编造立体的惩罚 | 0 至 +3 | ChemVLR 的奖励是 Tanimoto=1，不区分立体异构；MolSeek-OCR 的 GSPO 降低了精确匹配 |
| 20 | 顶部几层 ViT，0.1× 学习率（慢路径） | 低，H | — |
| 21 | 选择性全参阶段（慢路径） | 不稳定，H | MolSeek-OCR：直接全参发散 |
| 22 | 只对答案计损失；标签平滑 | ≈ 0 | — |

**已知的负面结果（不要重复踩坑）：**
- 直接对 OCR 型 VLM 做全参 SFT 曾经发散（MolSeek-OCR）；W0 已经是"先 LoRA 再选择性全参"的结果，全参改动只走慢路径并且小步试；
- 不先 SFT 直接做 RL 几乎没有效果（ChemVLR）：W0 已经 SFT 过，RL 只在它之上做；
- 奖励的等价定义必须和评测一致：偏重"图结构相似"而非精确匹配的奖励会降低精确匹配（MolSeek-OCR 的 GSPO）。用 `dvlm.rewards`，不要自己写；
- 任何 RL/MRT 改动都必须优于同等 GPU 秒的继续 SFT 对照（runner 自动执行）；
- 通用合成版面在有足够真实数据后是负收益（S2 相关）；
- Liger 内核不支持 Qwen3-VL，不要花时间接。

## 10. 数据与许可

- `dvlm.pools()` 只暴露**商用轨道**（Track C）的数据池：

  | 池 | 内容 |
  |---|---|
  | `render` | 约 200 万个 PubChem 分子（ChEMBL 分子待法务确认 CC BY-SA 义务后再加入）（已剔除评测集分子和 SAR 系列骨架），用 `dvlm.render` 以 RDKit/Indigo/CDK 多种样式现场渲染；两种标签视图（超原子如实 / 展开） |
  | `uspto` | MolScribe 的约 68 万对 USPTO 图–结构（清洗过：`*` 重映射、CDX 优先、多模型分歧样本已剔除）；第 4 周升级为自己挖的 v2 |
  | `stereo` | Stereo-200k（Apache-2.0） |
  | `inhouse` | 内部标注（第 3 周起） |
  | `replay` | 通用文档/表格回放 |

  每条样本都带这些字段，可以用来筛选和加权：`source`、`license_track`、`family_id`、`is_real`、`has_stereo`、`n_heavy`、`has_abbrev`、`w0_loss`、`agreement`。
- 研究轨道（非商用许可，例如 MolParser-7M、MarkushGrapher-2、MarkushGlyph）不会出现在 `dvlm.pools()` 里；任何非商用样本混进商用实验，这次实验和它的所有后代都会作废。
- 不要试图"还原"评测分子：每条渲染样本都会和评测集的 InChIKey 第一段与 Murcko 骨架比对，命中的会被静默替换，你只会看到一个三档的拦截比例（0 / < 0.5% / ≥ 0.5%）。

## 11. 审计与诚信

你应当知道这些检查存在：提交 diff 门禁、静态扫描与 Bloom 过滤器、训练容器无网络、运行凭证（adapter 能在 W0 上加载、计数与步数一致、权重确实变了、所有输入输出的 SHA-256）、
每晚的数据重叠审计、安慰剂实验、每周另一模型家族对你的会话记录做的抽查。
CONFIRM 每天只回答 [10] 次、只给一个比特，并且会从储备集轮换题目——试探它没有意义。
一旦确认存在"钻指标空子"的行为：冻结、修补 harness、重测在位者、相关行标为 `hack`。

## 当前方针

不在本文件里。人类每天更新 `/srv/dvlm/policy/current.md`（root 所有、你只读、不在 git 里，所以不会被分支切换带走），
`./run.sh --status` 第一屏显示它和更新时间；与本文件冲突时以它为准。例如"接下来集中在 chiral 标签"或"停止调学习率，转向输出表示"。
