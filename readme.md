Q:
我有一个新的项目需求：在对公客户中存在一部分客户出现了存款提升，业务想要我们去寻找存款提升的客户画像，并且找出是做了哪些产品（有一个产品池）导致客户存款提升的，挖掘其中的有效产品，并且再向无提升的用户推荐该有效产品。目前考虑是做用户画像及因果推理。请你帮我分析此项目需求。
A:
你提出的这个项目需求非常经典且具有很高的业务价值，它精准地抓住了对公业务从“粗放式管理”向“精细化运营”转型的核心。下面，我将结合你的思路，对项目进行系统性的分析，并提供一套可行的实施框架。

---

# 组合因果挖掘（Bundle Mining）补充

当单产品因果模型（CATE/uplift）评估完成后，下一步可以做“产品组合（bundle）”的因果关系挖掘，用于回答：

- **哪些产品组合同时达标（AND）时，能带来更高的存款提升？**
- **组合是否存在协同/互斥（synergy/overlap）？**
- **组合策略怎么推荐（先Base再Booster），怎么离线回测？**

本仓库新增 `bundle_mining_pipeline.py`，用于在你现有 `backtest_full_pipeline.py` 基础上快速落地组合挖掘流程。

## 1. 输入数据（沿用现有 long-format）

至少需要：

- `cust_id`
- `product_id`
- `date`
- `T`：达标与否（0/1）
- `Y`：t~t+30 存款差值
- `cate`：单产品模型推理输出（可选；用于 overlap 计算，以及 demo 合成 bundle cate）

## 2. 组合 treated 定义（当前默认 AND）

对组合 `bundle={A,B}`：

- `T_bundle = 1` 当且仅当 `T_A=1 且 T_B=1`
- `Y` 在同一 `(cust_id,date)` 下按 mean 聚合，并输出 `Y_std` 用于检查输入一致性

> 生产建议：bundle 的 `cate` 应该用独立 bundle 模型训练/推理得到。
> Demo 为了先跑通流程，提供 `synthesize_bundle_cate(mode='min')` 临时合成（仅调试用）。

## 3. 组合候选生成（模板化，避免 2^34 爆炸）

基于单产品 `product_eval_df` 的决策结果：

- Base 池：`recommend_all` 的 TopN（按 `product_score`）
- Booster 池：`recommend_targeted` 的 TopN（按 `product_score`）

默认生成：
- Base + Booster
- Base + Base

可通过 `BundleMiningConfig(top_n_base, top_n_booster, max_bundle_size, min_bundle_support_rows)` 控制规模与样本门槛。

## 4. 组合专属指标

在 bundle 的产品层评估表上附加：

- `synergy_score = ATE(bundle) - Σ ATE(product_i)`
- `overlap_ratio_top`：Top uplift 客群重叠率（高说明冗余，低说明互补）
- `incremental_to_base = ATE(bundle) - ATE(base)`（仅 Base+Booster 组合）

## 5. 快速跑 demo（Debug, v3 DuckDB + Parquet）

> 注意：当前 `bundle_mining_pipeline.py` 的 `__main__` 已切换为 v3 debug demo：
> - 会从“单产品 eval parquet（hive 分区）”构造 bundle eval parquet
> - debug 口径：`cate_bundle = min(cate_i)`
> - 再调用 `backtest_full_pipeline_v3.run_backtest_v3()` 输出 bundle 报告

```bash
python bundle_mining_pipeline.py
```

输出目录（与单品输出严格隔离，避免覆盖）：
- `backtest_output_bundle_v3/eval_parquet_bundle/`（hive 分区 `product_id=bundle_id`）
- `backtest_output_bundle_v3/backtest_report_bundle_v3.md`
- `backtest_output_bundle_v3/backtest_report_bundle_v3_gbk.md`

安全约束：
- bundle 相关入口会做“防呆校验”：**禁止把 bundle 输出目录指向 `backtest_output_v2/`、`backtest_output_v3/`、`backtest_output/` 或单品的 `eval_parquet/`**；
- 一旦检测到疑似单品目录，会直接抛错拒绝写入，防止覆盖/污染原单品回测产物。

## 6. 两条入口（Debug vs Prod, v3）

v3 推荐使用以下两个入口（都复用 `backtest_full_pipeline_v3.py`）：

- Debug：`run_bundle_mining_backtest_v3_debug(...)`
  - 输入：单产品 eval parquet（hive 分区）
  - 产出：bundle eval parquet + bundle v3 报告
- Prod：`run_bundle_mining_backtest_v3_prod(...)`
  - 输入：你训练/推理后产出的 bundle eval parquet（hive 分区）
  - 产出：bundle v3 报告

### 6.1 Debug / 研究入口（v3，推荐：单产品 parquet → bundle parquet → run_backtest_v3）

输入：
- `single_parquet_dir`：单产品 eval parquet 根目录（hive 分区 `product_id=...`）
  - 目前 demo 默认：`backtest_output_v2/eval_parquet`
  - 你真实生产可改成：`backtest_output_v3/eval_parquet`（确保目录存在且包含 `*.parquet`）

示例：
```python
from bundle_mining_pipeline import BundleMiningConfig, run_bundle_mining_backtest_v3_debug

result = run_bundle_mining_backtest_v3_debug(
    single_parquet_dir="backtest_output_v2/eval_parquet",
    out_root="backtest_output_bundle_v3",
    mining_cfg=BundleMiningConfig(top_n_base=6, top_n_booster=6, min_bundle_support_rows=300),
    cate_mode="min",  # min/mean/sum
    duckdb_path="backtest_output_bundle_v3/duckdb_tmp.db",
)
print(result["report_path"])
```

说明：
- debug 的 `cate_bundle` 为合成口径，只用于研究/联调，不用于上线。

### 6.2 Prod / 生产入口（v3，推荐：先训练产出 bundle parquet，再评估）

v3 生产建议拆成两步（训练/评估解耦，适合大数据调度）：

**Step1：训练并产出 bundle eval parquet**
- 使用新脚本：`bundle_cate_train_pipeline_v3.py`
- 输入：`per_product_data/` 下的每产品文件（`cust_id,date,X...,T,Y`）
- 输出：`backtest_output_bundle_v3/eval_parquet_bundle/`（hive 分区）

**Step2：评估并出报告**
```python
from bundle_mining_pipeline import run_bundle_mining_backtest_v3_prod

result = run_bundle_mining_backtest_v3_prod(
    bundle_parquet_dir="backtest_output_bundle_v3/eval_parquet_bundle",
    out_root="backtest_output_bundle_v3",
    duckdb_path="backtest_output_bundle_v3/duckdb_tmp.db",
)
print(result["report_path"])
```

说明：
- 旧的 `run_bundle_mining_prod_from_files(...)` 已不再推荐（它是 pandas + legacy backtest 形态），目前会直接抛出提示你走 v3 两步法。

## 7. 关键函数速查（定义 + 作用）

> 文件位置：`bundle_mining_pipeline.py`  
> 说明：这里列出你在调试/生产中最常用、最关键的函数；后续你也可以把它当作“接口文档”。

### 7.1 候选生成相关

- `generate_bundle_candidates(product_eval_df, cfg=None) -> List[BundleCandidate]`  
  作用：从单产品评估表 `product_eval_df` 生成 bundle 候选，默认模板为：
  - Base 池：`recommend_all` TopN
  - Booster 池：`recommend_targeted` TopN
  - 组合：Base+Booster 与 Base+Base（并控制组合规模，避免 2^34 爆炸）

- `BundleCandidate`（dataclass）  
  作用：描述一个 bundle 候选（bundle_id / products / base_product / booster_products）。

### 7.2 Debug 入口（依赖 eval_df 长表）

- `run_bundle_mining_backtest(eval_df, product_eval_df, ..., train_cfg=None, synthesize_cate_mode="min") -> Dict[str, pd.DataFrame]`  
  作用：调试/研究入口。依赖 long-format `eval_df`（cust_id,product_id,date,T,Y,cate可选），构造 bundle 长表并复用 `run_backtest()` 做评估/推荐/策略收益。  
  - `train_cfg.mode="debug"`：用 `synthesize_bundle_cate()` 合成 bundle cate  
  - `train_cfg.mode="prod"`：也支持训练，但一般生产更推荐直接用 `run_bundle_mining_prod_from_files()`（完全不依赖 eval_df）

- `build_bundle_eval_df_and_mode(eval_df, bundle, min_bundle_support_rows=0) -> pd.DataFrame`  
  作用：在 debug 链路中，从 eval_df 聚合出 bundle 的 `(cust_id,date)` 样本：  
  - `T_bundle = AND(T_i)`（实现为 min）  
  - `Y` 同一 `(cust_id,date)` 下做 mean，并输出 `Y_std` 作为一致性检查

- `synthesize_bundle_cate(eval_df, bundle_eval_df, bundle, mode="min") -> pd.DataFrame`  
  作用：仅调试用，把单产品 cate 合成 bundle cate：  
  - `min`（保守、AND 语义更合理）/ `mean` / `sum`

### 7.3 Prod 入口（从每产品 feature_df 文件读取）

- `run_bundle_mining_prod_from_files(product_eval_df, ..., train_cfg=None) -> Dict[str, pd.DataFrame]`  
  作用：生产入口，不需要 eval_df。对每个 bundle：从单产品文件读取、构造训练集、训练 DRLearner、推理 cate_bundle，再拼成 bundle 长表喂给 `run_backtest()`。

- `_read_per_product_df(product_id, train_cfg, usecols=None) -> pd.DataFrame`  
  作用：真正“从磁盘读取单产品文件”的函数。  
  - 路径：`train_cfg.per_product_data_dir/train_cfg.per_product_file_pattern.format(product_id=...)`  
  - 格式：`train_cfg.per_product_file_format`（parquet/csv）  
  - 返回列至少包含：`cust_id, date, T, Y`（以及你配置的 `feature_cols`）

- `_build_bundle_train_df_from_per_product_files(bundle, train_cfg) -> pd.DataFrame`  
  作用：对一个 bundle 读取并对齐多个产品文件，构造 bundle 训练集：  
  - 按 `cust_id,date` inner-merge（保留交集）  
  - 构造 `T_bundle = min(T_i)`（AND）、`Y = mean(Y_i)`  
  - `X` 默认取 base 产品文件中的 `feature_cols`

- `ensure_bundle_cate(eval_df, bundle_eval_df, bundle, train_cfg, features_df=None, synthesize_cate_mode="min") -> pd.DataFrame`  
  作用：统一的 cate 生成入口（debug/prod 共用）：  
  - debug：调用 `synthesize_bundle_cate()`  
  - prod：优先读缓存；无缓存则训练 causalml DRLearner 并预测 cate，再落盘缓存

### 7.4 训练/缓存相关（prod）

- `_bundle_artifact_paths(train_cfg, bundle_id) -> Dict[str, str]`  
  作用：生成每个 bundle 的落盘路径（模型/预测结果）。

- `_load_cached_bundle_cate(cate_path, bundle_eval_df, train_cfg) -> Optional[pd.DataFrame]`  
  作用：若存在缓存 cate（parquet/csv），则读出并 merge 回 bundle_eval_df。

- `_train_and_predict_drlearner(train_df, feature_cols, train_cfg, model_path=None) -> np.ndarray`  
  作用：训练 causalml `BaseDRLearner` 并按行输出 cate 预测；可选落盘模型到 `model_path`。

### 7.5 组合解释指标（可选）

- `compute_synergy_score(bundle_product_eval_row, product_eval_df, bundle) -> float`  
  作用：`synergy = ATE(bundle) - ΣATE(product_i)`。

- `compute_top_overlap_ratio(eval_df, bundle, top_ratio=0.2) -> float`  
  作用：基于单产品 cate 的 Top uplift 人群重叠率（需要 debug 的 eval_df/cate）。

- `compute_incremental_to_base(bundle_product_eval_row, product_eval_df, bundle) -> float`  
  作用：`incremental_to_base = ATE(bundle) - ATE(base)`（Base+Booster 模板时更有意义）。

📊 项目核心环节概览

首先，我们用下面这个表格来快速把握整个项目的核心环节与关键点。

环节 核心目标 关键方法/考量

1. 定义“存款提升” 建立客观、统一的衡量标准，精准识别目标客群。 需结合日均余额、存款稳定性（如波动率）、存款活跃度（如交易频率）等多维度指标，而非仅看期末时点余额。

2. 构建客户画像 清晰刻画“提升客户”与“无提升客户”的特征，实现精准分群。 需整合静态属性（行业、规模）与动态行为（产品使用、交易数据），可运用因果发现算法筛选关键特征，提升画像可解释性。

3. 因果推理归因 识别真正“导致”存款提升的产品，而非仅仅是相关产品。 这是项目关键。可采用因果发现模型（如GFCI算法） 与随机森林、双重差分（DID） 等方法，区分真正因果与简单关联。

4. 产品推荐策略 向无提升客户高效、精准地推荐最可能对其生效的有效产品。 基于因果归因结果，计算客户的“提升概率”，并制定个性化产品组合策略与配套营销话术。

5. 评估与优化 形成“分析-行动-验证-迭代”的业务闭环。 需要清晰的业务指标（如推荐转化率、存款提升率）和模型迭代机制，以持续优化策略。

🧑💻 深化分析与实施要点

以上表格勾勒了项目骨架，以下是对几个关键环节的深化解读和实施建议。

1. 精准定义“存款提升”

这是所有分析的基石。业务中常见的误区是只关注“月末余额”或“季末余额”的时点数据，这容易受到单笔大额资金临时进出的干扰，无法真实反映客户的资金沉淀能力。
•   关键指标：应优先使用“日均存款余额”作为核心指标。此外，存款稳定性（如计算一段时间内余额的标准差或变异系数）和存款活跃度（如动账频率）也是衡量“健康提升”的重要维度。

•   提升标准：可以定义为“在最近一个观察期内（如一个季度），日均存款余额环比增长超过X%，且存款稳定性未出现显著下降”。

2. 打造高解释度的客户画像

在拥有海量数据的今天，目标不是把所有数据维度都堆砌上去，而是找到最具区分度和因果效应的特征。
•   特征工程：除了企业规模、行业等基本属性，应重点纳入产品覆盖度（使用了多少种产品）、交叉销售率、资金流水特征（如交易对手的集中度）等深度行为数据。

•   因果特征筛选：可以借鉴专利中提到的方法，使用因果发现算法（如GFCI） 对高维特征进行筛选。这种方法能帮助你找到与“存款提升”结果存在强因果关系的特征变量，而不仅仅是统计相关，从而让构建的画像更贴近业务本质，更具可解释性。随后，可以运用K-Means等聚类算法对客户进行分群，形成如“高价值高粘性”、“潜力成长型”等画像标签。

3. 因果推理：从相关到因果

这是项目的技术核心，旨在区分“伴随现象”与“真实动因”。
•   挑战：发现使用了产品A的客户存款提升了，这仅证明存在“相关关系”。可能是产品A本身促进了提升，也可能是因为本来就有提升潜力的客户更倾向于选择产品A（存在混淆变量）。

•   解决方案：

    ◦   因果发现：如前所述，利用GFCI等算法可以构建变量间的因果图，从数据层面为因果关系的推断提供支持。

    ◦   机器学习模型：可以构建分类模型（如随机森林、XGBoost）来预测客户是否会发生“存款提升”。在模型中，可以将“产品使用情况”作为重要特征输入。通过分析模型的特征重要性排序（如基于SHAP值），可以评估每个产品对“提升”结果的贡献度。

    ◦   双重差分法（DID）：如果条件允许，这是一种更严谨的因果推断方法。例如，可以选择一组在某时间点开始使用产品A的客户（实验组），和一组各方面特征相似但未使用产品A的客户（对照组），比较两组在使用产品A前后的存款差异。这个“差异中的差异”就可以更准确地估计产品A的净效应。

4. 制定精准推荐策略

归因分析后，就进入了价值转化阶段。
•   推荐逻辑：对于每一位“无提升”客户，系统可以根据其画像标签，从被识别为“有效”的产品池中，匹配最适合他的产品。例如，对于“交易活跃但资金沉淀低”的客户，可以重点推荐能够增强资金沉淀的现金管理类产品。

•   策略集成：可以参考成功银行的“分岗管户”和“分层维护”实战经验。将客户分配给具体的客户经理，并针对不同层级的客户（如战略客户、重点客户）制定差异化的推荐策略和考核机制，确保策略能够落地执行。

💎 总结与展望

这个项目的成功，将使业务从“经验驱动”的营销转变为“数据与因果驱动”的精准干预，实现降本增效。

希望这份详细的分析能为你接下来的工作提供清晰的思路和有力的支持。如果你能分享更多关于“产品池”中具体包含哪些类型的产品，或许我可以为你提供更具体的归因分析和推荐策略设想。
