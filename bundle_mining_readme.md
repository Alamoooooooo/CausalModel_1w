# Bundle Mining README

本文档只说明 **bundle（产品组合）因果挖掘** 流程，避免与单产品回测文档混淆。

---

## 1. 项目目标

在已经完成单产品 causalml / DRLearner 训练与评估的基础上，进一步挖掘：

- 哪些产品组合会同时达标
- 哪些组合对存款提升更有效
- 哪些组合存在协同效应或冗余
- 如何把组合结果用于推荐与离线回测

当前 bundle 流程是建立在单产品结果之上的二次挖掘层，核心入口如下：

- `bundle_mining_pipeline.py`
- `bundle_cate_train_pipeline_v3.py`
- `backtest_full_pipeline_v3.py`

---

## 2. 与单产品流程的关系

单产品流程负责：

- 训练单产品 CATE / uplift 模型
- 生成单产品 eval parquet
- 做产品层评估、客户推荐、策略收益、OPE、单日子集评估

bundle 流程负责：

- 从单产品评估结果中生成 bundle 候选
- 为 bundle 独立训练 CATE 模型
- 产出 bundle eval parquet
- 复用 `backtest_full_pipeline_v3.py` 做 bundle 回测与报告

换句话说：

- **单产品** 是基础
- **bundle** 是在基础上做组合挖掘

---

## 3. 输入数据要求

### 3.1 单产品 eval parquet

bundle debug / prod 回测都依赖单产品或 bundle 的长表数据，基本 schema 为：

- `cust_id`
- `product_id`
- `date`
- `cate`
- `T`
- `Y`

其中：

- `T`：达标与否，0/1
- `Y`：`t~t+30` 的活期存款差值
- `cate`：单产品或 bundle 模型输出的 CATE

### 3.2 每产品训练文件

bundle 独立训练时，生产形态推荐使用：

- `per_product_data_dir/{product_id}.parquet`
- 或 `per_product_data_dir/{product_id}.csv`

每个产品文件需至少包含：

- `cust_id`
- `date`
- `X...`
- `T`
- `Y`

说明：

- 当前 bundle 训练默认假设不同产品文件中的 `X` 是同一套客户画像特征
- 如果后续特征不一致，需要扩展交叉特征或拼接逻辑

---

## 4. 代码入口说明

### 4.1 `bundle_mining_pipeline.py`

这是 bundle 挖掘与回测的主入口，包含：

- bundle 候选生成
- bundle debug demo
- bundle prod 回测
- synergy / overlap / incremental 指标
- 与 `backtest_full_pipeline_v3.py` 的报告复用

常用函数：

- `generate_bundle_candidates(...)`
- `build_bundle_eval_df_and_mode(...)`
- `synthesize_bundle_cate(...)`
- `run_bundle_mining_backtest_v3_debug(...)`
- `run_bundle_mining_backtest_v3_prod(...)`

### 4.2 `bundle_cate_train_pipeline_v3.py`

这是 bundle 独立训练脚本，负责：

- 从每产品文件构造 bundle 训练集
- 用 causalml `BaseDRLearner` 训练 bundle CATE
- 输出 bundle eval parquet
- 输出目录为 hive 分区格式

常用函数：

- `train_one_bundle_and_write_eval(...)`
- `build_bundle_train_df_duckdb(...)`
- `train_and_predict_drlearner(...)`
- `write_bundle_eval_parquet(...)`

### 4.3 `backtest_full_pipeline_v3.py`

这是 v3 回测主框架，bundle 侧复用它的能力：

- `evaluate_products_duckdb(...)`
- `generate_recommendations_duckdb(...)`
- `policy_gain_curve_duckdb(...)`
- `temporal_stability_duckdb(...)`
- `run_backtest_v3(...)`

v3 新增支持：

- `eligible_eval_df`
- `eligible_product_eval_df`
- `eligible_customer_reco_df`
- `eligible_reco_empirical_eval_df`
- `eligible_policy_gain_df`
- `single_day_as_of_date`

---

## 5. Bundle 候选生成逻辑

bundle 候选不是穷举全部组合，而是模板化生成，避免组合爆炸。

### 5.1 候选池

基于单产品 `product_eval_df`：

- `recommend_all` 中按 `product_score` 取 TopN 作为 Base 池
- `recommend_targeted` 中按 `product_score` 取 TopN 作为 Booster 池

### 5.2 默认组合模板

- Base + Booster
- Base + Base

### 5.3 关键配置

`BundleMiningConfig` 中常用参数：

- `top_n_base`
- `top_n_booster`
- `max_bundle_size`
- `min_bundle_support_rows`
- `top_ratio_for_overlap`

---

## 6. Bundle treated 定义

当前 bundle 默认采用 **AND** 语义：

- `T_bundle = min(T_i)`
- 即组合内所有产品都达标时，bundle 才算达标

bundle 的结果指标通常按 `(cust_id, date)` 聚合：

- `Y`：同一客户同一天下按 mean 聚合
- `cate`：bundle 训练后推理得到，或 debug 时临时合成

---

## 7. Debug 版流程

debug 版适合先验证逻辑是否正确。

### 7.1 入口

```python
from bundle_mining_pipeline import BundleMiningConfig, run_bundle_mining_backtest_v3_debug

result = run_bundle_mining_backtest_v3_debug(
    single_parquet_dir="backtest_output_v2/eval_parquet",
    out_root="backtest_output_bundle_v3",
    mining_cfg=BundleMiningConfig(top_n_base=6, top_n_booster=6, min_bundle_support_rows=300),
    cate_mode="min",
    duckdb_path="backtest_output_bundle_v3/duckdb_tmp.db",
)
print(result["report_path"])
```

### 7.2 debug 特点

- 从单产品 eval parquet 生成 bundle eval parquet
- `cate_bundle` 使用合成口径
- 适合联调、验证回测链路、快速看报告
- 不用于最终上线

### 7.3 debug 输出

- `backtest_output_bundle_v3/eval_parquet_bundle/`
- `backtest_output_bundle_v3/backtest_report_bundle_v3.md`
- `backtest_output_bundle_v3/backtest_report_bundle_v3_gbk.md`

---

## 8. Prod 版流程

prod 版适合正式训练和正式评估。

### 8.1 Step 1：训练 bundle 并产出 eval parquet

```python
from bundle_cate_train_pipeline_v3 import BundleTrainJobConfig, train_one_bundle_and_write_eval

cfg = BundleTrainJobConfig(
    per_product_data_dir="per_product_data",
    per_product_file_format="parquet",
    feature_cols=["x1", "x2", "x3"],
    out_bundle_parquet_dir="backtest_output_bundle_v3/eval_parquet_bundle",
)

train_one_bundle_and_write_eval(
    bundle_products=["1", "2"],
    base_product="1",
    cfg=cfg,
)
```

### 8.2 Step 2：评估并出报告

```python
from bundle_mining_pipeline import run_bundle_mining_backtest_v3_prod

result = run_bundle_mining_backtest_v3_prod(
    bundle_parquet_dir="backtest_output_bundle_v3/eval_parquet_bundle",
    out_root="backtest_output_bundle_v3",
    duckdb_path="backtest_output_bundle_v3/duckdb_tmp.db",
)
print(result["report_path"])
```

### 8.3 prod 特点

- 每个 bundle 独立训练
- 产出 bundle eval parquet
- 再复用 v3 回测链路出报告
- 更适合大数据与调度任务

---

## 9. 安全防呆机制

bundle 相关代码已经加入目录安全校验，避免误写单品目录。

禁止写入：

- `backtest_output_v2`
- `backtest_output_v3`
- `backtest_output/`

建议始终使用：

- `backtest_output_bundle_v3/`

bundle 训练和回测的输出目录必须与单品目录隔离。

---

## 10. v3 新增单日子集说明

`backtest_full_pipeline_v3.py` 新增了单日可推荐子集链路，bundle 侧已同步启用。

### 10.1 相关输出

- `eligible_eval_df`
- `eligible_product_eval_df`
- `eligible_customer_reco_df`
- `eligible_reco_empirical_eval_df`
- `eligible_policy_gain_df`
- `single_day_as_of_date`

### 10.2 含义

这是为了模拟真实线上投放：

- 只在固定 `as_of_date` 上生成推荐
- 通过 lookback window 过滤近期已经达标的客户
- 更贴近真实推荐场景

### 10.3 bundle 中的对应行为

bundle debug / prod 回测入口都已启用：

- `enable_single_day_reco=True`

因此 bundle 报告也会包含：

- 全量回测结果
- 单日子集结果

---

## 11. 组合指标解释

bundle 评估结果中，通常会附加以下指标：

### 11.1 `synergy_score`

定义：

- `ATE(bundle) - Σ ATE(product_i)`

含义：

- 大于 0：可能存在协同效应
- 小于 0：组合效果不如单品线性叠加

### 11.2 `overlap_ratio_top`

定义：

- bundle 内单品 Top uplift 人群的重叠率

含义：

- 高：组合可能冗余
- 低：组合可能更互补

### 11.3 `incremental_to_base`

定义：

- `ATE(bundle) - ATE(base)`

含义：

- 用于衡量 Base + Booster 结构时，Booster 是否真的带来增量

---

## 12. 性能与大数据建议

### 12.1 候选规模控制

千万级数据下，最重要的是控制组合数。

建议：

- `top_n_base` 不要太大
- `top_n_booster` 不要太大
- `max_bundle_size` 建议先从 2 开始
- `min_bundle_support_rows` 不要太低

### 12.2 训练侧建议

bundle 独立训练时：

- 用 DuckDB 先做样本构造
- 尽量减少 pandas 大表 merge
- 只保留必要特征列
- 训练前先抽样或限定日期窗口

### 12.3 回测侧建议

- 优先用 hive 分区 parquet
- 避免一次性把全部 bundle 结果转成超大 pandas 表
- bundle 数量过多时，建议分批训练、分批落盘、分批回测

---

## 13. 常见问题

### 13.1 为什么 bundle 的 cate 不直接相加？

因为不同产品的 CATE 不具备简单可加性，直接相加通常会失真。  
debug 版的 `min/mean/sum` 只是为了联调，不是生产口径。

### 13.2 为什么 bundle 采用 AND 语义？

因为当前业务语境更接近“组合同时达标才算有效”，AND 更符合组合筛选逻辑。  
如果后续要扩展 OR / 顺序触达 / 先后依赖，需要单独设计。

### 13.3 为什么 bundle 也要跑单日子集？

因为全量历史数据里可能存在重复达标或不可推荐样本，单日子集更接近真实线上投放口径。

---

## 14. 推荐操作顺序

建议按这个顺序执行：

1. 先看单产品评估结果，确认哪些产品进入 `recommend_all` / `recommend_targeted`
2. 用 `bundle_mining_pipeline.py` 生成 bundle 候选
3. 用 `bundle_cate_train_pipeline_v3.py` 独立训练 bundle CATE
4. 产出 bundle eval parquet
5. 用 `run_bundle_mining_backtest_v3_prod()` 做 bundle 回测
6. 看 bundle 报告中的：
   - 产品层评估
   - 客户层推荐
   - policy gain
   - temporal stability
   - eligible 单日子集
   - OPE（如果有 ps / mu0 / mu1）

---

## 15. 文档定位说明

- `bundle_mining_readme.md`：bundle 专用说明
- `readme.md`：建议仅保留索引或兼容说明，避免误解为单产品主文档

如果你在看的是 bundle 挖掘流程，请优先看本文件。
