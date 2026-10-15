# 交付物 schema v0：下游拿到的数据

> 给下游分析（SAR/QSAR、ADMET 画像、专利布局、数据库合并）和抽取流水线的开发者看。混合流水线 v0（[`../README.md`](../README.md) 第 6.4 节）从第 3 周起按这个 schema 供数；之后 VLM 模型替换其中的环节，schema 不变。字段的评分规则见 [`../harness/README.md`](../harness/README.md)，两者的字段对应见第 2.1 节。本 schema 是 v0 草案，摸底第 9–10 天（10/28–29）与下游用户确认后冻结（README 8.1 节）。

## 1. 四张表（Parquet；化合物另导出带属性的 SDF）

### `documents`（一行 = 一篇文档）

| 字段 | 类型 | 说明 |
|---|---|---|
| `doc_id` | str | 唯一；专利用公开号 + 页范围，论文用 DOI（SI 加后缀 `#si`） |
| `family_id` | str | 专利族（DOCDB）或文章；论文与其 SI 同族 |
| `doc_type` | enum | `patent` / `paper` / `si` / `report`（内部项目报告） |
| `office` | enum | `US` / `CN` / `WO` / `EP` / `JP` / `other`；论文为空 |
| `language` | str | `en` / `zh` / `ja` / … |
| `pub_date` | date | |
| `license_track` | enum | `C`（商用）/ `R`（研究）/ `internal` |
| `pipeline_version`, `model_versions` | str | 抽取流水线与各模型版本 |

### `compounds`（一行 = 一个文档中的一个化合物）

| 字段 | 类型 | 说明 |
|---|---|---|
| `compound_uid` | str | `hash(doc_id, label_norm)`；没有编号的化合物用 `hash(doc_id, page, bbox)` |
| `doc_id` | str | |
| `label_as_printed` | str | "Compound 12a"、"Example 212, Step 3"，按原文 |
| `label_norm` | str | 归一化编号，规则同 `score.py` 的 `_norm_label`："Compound 12a"、"12 a"、"化合物12a" → "12a"；正文、表格和图中的同一编号因此连到同一行 |
| `smiles`, `cxsmiles` | str | 标准化母体（按反离子清单去盐、中和）；`cxsmiles` 保留增强立体与 R 基团 |
| `inchikey`, `inchikey_skeleton` | str | 完整键与第一段 |
| `salt_as_drawn` | str | 原文画出/写出的盐型，如 "HCl"、"Na" |
| `stereo_status` | enum | `absolute` / `relative` / `racemic` / `single_unknown`（任意指定的单一对映体）/ `undefined` |
| `is_markush` | bool | 通式；此时 `cxsmiles` 带 R 标签 |
| `enumerated_from` | str | 由母核 + 取代基表枚举得到的化合物，指向母核行的 `compound_uid`；其余为空 |
| `structure_source` | enum | `vlm_ocsr` / `uspto_cwu` / `opsin_name` / `enumerated` / `manual` |
| `cross_check` | enum | `agree` / `disagree` / `single_source`：多个来源（CWU、OPSIN、识别）是否一致 |
| `role` | enum | `target` / `reference` / `intermediate` |
| `page`, `bbox` | int, [x0,y0,x1,y1] | 出处 |
| `confidence` | float | 0–1，模型平均对数概率经校准 |
| `review_status`, `reviewed_by`, `reviewed_at` | enum, str, datetime | `auto` / `reviewed` / `corrected` / `rejected`；化学家复核记录 |
| `calc_*` | float | RDKit 计算性质：`calc_mw`、`calc_clogp`、`calc_tpsa`、`calc_hbd`、`calc_hba`、`calc_ro5_violations`——**不从文献抽取**，由结构统一计算 |

### `assays`（一行 = 文中定义的一个实验方法）

专利表头常只写"Assay A IC50"或"Activity"，靶点、物种、细胞系和条件写在"生物学实施例/Assay A"段落里。ChEMBL 也以 assay 为组织单位。

| 字段 | 类型 | 说明 |
|---|---|---|
| `assay_uid` | str | `hash(doc_id, assay_label_norm)` |
| `doc_id` | str | |
| `assay_label_as_printed` | str | "Assay A"、"Biological Example 1"、"生物学实施例 1" |
| `target_as_printed`, `target_uniprot`, `variant` | str | 靶点与突变体 |
| `assay_type` | enum | `biochemical` / `cell` / `binding` / `invivo` / … |
| `species`, `cell_line_or_matrix` | str | |
| `conditions` | struct | ATP/底物浓度、孵育时间、检测方法、温度等，按原文 |
| `page`, `bbox`, `protocol_excerpt` | | 方法段落的出处与摘录（≤ 1,000 字） |

### `measurements`（一行 = 一个印刷出来的数值）

| 字段 | 类型 | 说明 |
|---|---|---|
| `record_uid` | str | `hash(doc_id, page, table_id 或 text_block_id, row, col, value_idx)`，重新抽取时保持稳定；`value_idx` 区分同一单元格里的多个数值；模型版本记在 `documents` |
| `doc_id` | str | 化合物没解析出来（`compound_uid` 为空）时记录也不丢文档 |
| `compound_uid` | str | |
| `assay_uid` | str | 所属实验方法；表头或脚注没有指向方法段落时为空 |
| `endpoint` | enum | 封闭词表：`ic50`、`ki`、`kd`、`ec50`、`%inhibition`、`cl`、`clint`、`vss`、`t1/2`、`auc0inf`、`auc0t`、`auc024`、`cmax`、`f`、`ppb`、`papp_ab`、`papp_ba`、`efflux_ratio`、`solubility`、`logd`、`pka`、…（与 `score.py` 的终点名一致） |
| `endpoint_as_printed` | str | 原文列头 |
| `target_as_printed`, `target_uniprot`, `variant` | str | 靶点与突变体（"L858R/T790M"）；PK/理化记录为空；有 `assay_uid` 时默认取自 `assays` |
| `assay_type` | enum | `biochemical` / `cell` / `binding` / `invivo` / … |
| `species`, `cell_line_or_matrix` | str | `human`/`rat`/…；`HCT116`、`liver_microsomes`、`plasma` |
| `condition` | struct | `conc`、`time`、`ph`、`dose`、`route`（`iv`/`po`/…）、`formulation` |
| `relation` | enum | `=` / `<` / `>` / `range` / `cat` |
| `value_as_printed` | str | 原文字符串，如 "12.3 ± 1.1"、">10"、"A" |
| `value_std`, `unit_std` | float, str | 标准单位：摩尔浓度 nM、时间 h、清除率 mL/min/kg、AUC ng·h/mL、Papp 10⁻⁶ cm/s |
| `value_lo`, `value_hi` | float | 区间与删失值的边界（见第 2 节） |
| `pX` | float | 浓度类终点的 −log10(M) |
| `binned_label`, `bin_interval` | str, [lo, hi] | 专利分级活性："A" 与图例给出的区间 |
| `n`, `sd` | int, float | 重复数与标准差（如有） |
| `method` | enum | `measured` / `calc`（文中报告的计算值） |
| `cited` | bool | 引自其他文献 |
| `page`, `table_id`, `text_block_id`, `char_span`, `row`, `col`, `footnote_ids` | | 出处：表格数值有 `table_id`/`row`/`col`；正文数值有 `text_block_id` 与字符区间 |
| `confidence` | float | |
| `review_status`, `reviewed_by`, `reviewed_at` | | 同 `compounds` |
| `qc_flags` | list | `unit_x1000`（恰好差 3 或 6 个 log 单位）、`out_of_range`、`structure_disagree`、`text_mismatch`（读数与 PDF 文本层/专利 XML 不一致）、… |

## 2. 规则

- **删失值**（">10 µM"、"<1 nM"）：保留 `relation`，`value_lo`/`value_hi` 只填一侧。建模时按删失回归处理或丢弃，**不要当成等值**。
- **分级活性**（"A"、"+++"）：`relation = cat`，`binned_label` 与 `bin_interval` 同时给出；没有图例时 `bin_interval` 为空。
- **区间**（"10–30"）：`relation = range`，`value_lo`/`value_hi` 都填。
- **计算值 vs 实测值**：文中报告的 cLogP 等计算值也抽取，但 `method = calc`，下游默认不与实测值混用；分子量、TPSA 等由 `calc_*` 统一计算。
- **数值以文本为准**：数字原生 PDF 与美国专利全文 XML 有文本层时，`value_as_printed` 取文本层；VLM 读数只用于扫描件和核对（不一致标 `text_mismatch`）。
- **跨文档合并**：评分按文档分别计算；交付时再按 InChIKey 合并成跨文档视图（同一化合物在论文、SI、专利中重复出现）。与内部化合物库按 InChIKey + 盐型对齐，冲突进人工队列。
- **稳定 ID**：`compound_uid`、`assay_uid`、`record_uid` 只由文档、归一化编号与位置决定，换模型重新抽取时不变，方便比较版本。
- **人工修正永不被覆盖**：重新抽取时，`review_status = corrected` 或 `rejected` 的行保持原样，新抽取结果与之不一致时只加 `qc_flags`。下游建模可以只取 `reviewed`/`corrected` 行。
- **置信度与复核**：`confidence`、`qc_flags`、`cross_check` 加上 `bbox` 定位，化学家只看可疑条目。

### 2.1 schema → `score.py` JSONL 的字段对应

金标与评测时，按下表把 schema 行写成评分器读的 JSONL（`harness/README.md` 第 2 节）；`records_yaml_v1` 用 `harness/yaml_to_jsonl.py` 转换。

| schema | `score.py` JSONL |
|---|---|
| `compounds.label_as_printed` | `compounds[].id`（评分器内部同样做 `_norm_label`） |
| `compounds.cxsmiles`（没有时 `smiles`） | `compounds[].smiles` |
| `measurements.compound_uid` → 该化合物的编号 | `measurements[].compound_id` |
| `value_as_printed` | `value`（原文字符串，保留末位 0） |
| `endpoint_as_printed` / `endpoint` | `endpoint`（裸终点名更好；列头里的上下文写进下面的字段） |
| `target_as_printed` | `target` |
| `relation`、`unit_std` 前的原文单位 | `relation`、`unit` |
| `assay_type`、`species`、`variant` | 同名字段 |
| `cell_line_or_matrix` | `matrix`（或 `cell_line`） |
| `condition.conc`、`.time`、`.ph`、`.dose`、`.route` | 平铺：`conc`/`conc_unit`、`time`/`time_unit`、`ph`、`dose`/`dose_unit`、`route` |
| `documents.doc_type`、`office`、`language`、`family_id` | 同名字段（只看金标） |

## 3. 不同下游用途的最低质量与最早可用时间（待校准）

| 用途 | 最低质量 | 如何验证 | 最早可用 |
|---|---|---|---|
| SAR/QSAR 建模 | 置信度过滤后记录精确率 ≥ [0.9]，结构精确匹配 ≥ [0.9] | 30 篇留出文档人工核对 | v0（第 3 周）只作预填，化学家复核后可用；自动数据直接建模要等 S2 出口（约第 7 周） |
| ADMET 画像 | 物种/基质正确率 ≥ [0.95]，单位换算错误 = 0 | 抽样 200 条人工核对 | S2b（第 8 周起）；之前只作预填 |
| 专利布局 / FTO | Markush 母核召回 ≥ [0.8] | IP5-M + 内部专利 | S3，第 8 周后；商用模型待定（现有最好的 Markush 模型只在研究轨道） |
| 与 ChEMBL/BindingDB 合并 | 数据库记录召回 ≥ [0.8] | `docs/design.md` 第 12 节 | v0 起，只合并 `cross_check = agree` 且 `confidence` ≥ 阈值的记录 |

## 4. 一个完整例子

原文：某论文（虚构，`doc_id` 用占位符）表 2，列头"Compd | R | EGFR IC50 (nM) | HCT116 IC50 (µM)"，三行：

| Compd | R | EGFR IC50 (nM) | HCT116 IC50 (µM) |
|---|---|---|---|
| 12a | H | 12.3 | >10 |
| 12b | F | 4.1 | 2.5 |
| 12c | Cl | A* | 1.8 |

母核画在表上方（R 位于苯环对位），脚注"*A: IC50 < 10 nM"。

**`records_yaml_v1`（模型或混合流水线读表后的输出）**：

```yaml
table: {doc_id: "EXAMPLE-PAPER-1", page: 5, table_id: "T2"}
scaffold: {cxsmiles: "O=C(Nc1ccc([*:1])cc1)c1ccncc1", bbox: [102, 80, 380, 210]}
rgroups:
  - {compound_id: "12a", R1: "[H]"}
  - {compound_id: "12b", R1: "F"}
  - {compound_id: "12c", R1: "Cl"}
legend: {"A": "<10 nM"}
records:
  - {compound_id: "12a", endpoint: "IC50", target: "EGFR", assay_type: biochemical, value: "12.3", unit: "nM"}
  - {compound_id: "12a", endpoint: "IC50", target: "", cell_line: "HCT116", assay_type: cell, value: ">10", unit: "µM"}
  - {compound_id: "12b", endpoint: "IC50", target: "EGFR", assay_type: biochemical, value: "4.1", unit: "nM"}
  - {compound_id: "12b", endpoint: "IC50", target: "", cell_line: "HCT116", assay_type: cell, value: "2.5", unit: "µM"}
  - {compound_id: "12c", endpoint: "IC50", target: "EGFR", assay_type: biochemical, value: "A", unit: "", footnote: "*"}
  - {compound_id: "12c", endpoint: "IC50", target: "", cell_line: "HCT116", assay_type: cell, value: "1.8", unit: "µM"}
```

**结果**：`harness/yaml_to_jsonl.py` 用 RDKit `molzip` 把母核与三个取代基拼成 12a–12c 的完整结构（交付时是三个 `compounds` 行，`structure_source = enumerated`，`enumerated_from` 指向母核行），并输出评分用 JSONL；六个 `measurements` 行，其中 12a 的 HCT116 值是删失值（`relation = >`、`value_lo = 10000` nM），12c 的 EGFR 值是分级值（`binned_label = A`、`bin_interval = [null, 10]` nM）。键名写成 `compound` 时转换脚本会提示并改正；直接交给 `score.py` 则报 "schema mismatch" 退出，不会悄悄得 0 分。

## 5. 零样本读表提示词（混合流水线 v0 的起点）

```
你是药物化学数据整理员。下面是一张论文/专利里的表格截图（编号框已标在图上）、表格上方的母核结构图，
以及（如有）表头引用的实验方法段落。请按 records_yaml_v1 输出：
1. scaffold：母核的 CXSMILES，可变位置用 [*:n] 并标注 _Rn；看不清时写 null，不要猜。
2. rgroups：每一行一个条目，compound_id 写化合物编号（按原文），R1、R2…写取代基（SMILES 片段或原文缩写）。
3. records：每个单元格一条记录，compound_id 写该行的化合物编号。value 按原文抄写（保留 >、<、±、分级字母），
   unit 写原文单位；列头、脚注或实验方法段落里写明的靶点、细胞系、物种、实验类型、浓度、pH、剂量与途径
   填进对应字段；表头引用的方法编号（如 "Assay A"）写进 assay；没写的留空，不要推测。
4. legend：脚注中分级活性的含义。
只输出 YAML。
```

这个提示词与 YAML 格式在摸底阶段用 20 张内部表核对（记录精确率目标 ≥ [0.8]），之后作为 S2 训练和评测的输出格式基线。
