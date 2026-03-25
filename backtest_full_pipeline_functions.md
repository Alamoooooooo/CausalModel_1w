# backtest_full_pipeline.py 文档（函数说明 + 全流程）

> 目的：在不打断代码的前提下，把 `backtest_full_pipeline.py` 的**每个函数/类的作用、输入输出、关键口径**以及**完整 backtest 流程**记录下来，方便维护与二次开发。
>
> 文件对应：`backtest_full_pipeline.py`（v1/pandas 版完整回测脚本）

---

## 1. 总览：脚本在做什么？

`backtest_full_pipeline.py` 实现了一套“因果推荐”离线回测流水线：

1) **产品层（product level）评估**：对每个产品聚合计算
- ATE（mean(cate)）
- empirical uplift（E[Y|T=1] - E[Y|T=0]）
- Top segment uplift（Top 人群 uplift 相对整体的提升）
- negative uplift 风险（cate < 0 占比）
- Qini/AUUC proxy（排序有效性 proxy，可被外部真实 qini/auuc 覆盖）
- 产出产品门禁 pass_xxx、推荐决策 recommendation_decision、产品打分 product_score、产品标签 product_tag 等

2) **客户层（customer level）推荐生成**：将通过门禁的产品映射到客户侧，生成 Top-K 推荐清单
- adjusted_cate：可选校准后的增量（cate * calibration_factor）
- recommend_score：综合 adjusted_cate + 产品分 + 风险惩罚
- 每客户 Top-K

3) **回测层（backtest）**：对“推荐策略”做离线评估
- 推荐子集 uplift（treated-control 差分 sanity check）
- policy_gain_curve：触达 Top X% 的收益曲线
- temporal_stability：分日期的 ATE vs empirical uplift
- OPE：IPW / DR 离线策略价值估计（需要 ps/mu0/mu1）

---

## 2. 数据要求与字段约定

### 2.1 必需字段（长表 long-format）

`REQUIRED_COLUMNS = ["cust_id", "product_id", "date", "cate", "T", "Y"]`

| 字段 | 含义 |
|---|---|
| cust_id | 客户 ID |
| product_id | 产品 ID |
| date | 日期/周期（可按天/周/月） |
| cate | 模型输出的个体增量（CATE） |
| T | 历史真实是否触达/处理（0/1） |
| Y | 结果变量（金额/0-1 等） |

### 2.2 可选字段（用于 OPE）
| 字段 | 含义 |
|---|---|
| ps | propensity score（倾向得分） |
| mu0, mu1 | 结果模型对 T=0/1 的预测（DR 需要） |

---

## 3. 配置类（dataclass）

### 3.1 `ProductDecisionConfig`
**作用**：产品层门禁阈值与推荐策略开关。  
**关键字段**：
- `min_ate`：ATE 门槛（ATE=mean(cate)）
- `min_empirical_uplift`：经验 uplift 门槛（treated-control）
- `min_qini`, `min_auuc`：排序有效性门槛（可被 external_metrics 覆盖）
- `min_top_uplift_lift`：Top uplift 相对整体 uplift 的提升门槛
- `max_negative_uplift_ratio`：cate<0 比例上限（误推风险控制）
- `min_support_samples`：样本量门槛
- `top_ratio`：Top 人群比例（例如 0.2=Top 20%）
- `enable_calibration`：是否用 empirical_uplift/ate 做产品级 scale 校准
- targeted 推荐相关：
  - `enable_targeted_reco`：是否允许 recommend_targeted
  - `targeted_top_ratio`：对 targeted 产品，仅对产品内 Top 人群开放推荐
  - `min_targeted_top_cate`, `min_targeted_lift`, `allow_targeted_when_ate_negative`：扩展门槛（当前脚本主要用 lift/negative ratio 等）

### 3.2 `CustomerDecisionConfig`
**作用**：客户侧推荐参数。  
- `min_cate`：客户-产品最小可推荐增量（这里用 adjusted_cate）
- `top_k_per_customer`：每客户推荐 K 个产品
- `customer_weight_col`：客户权重（预留）

### 3.3 `SafetyConfig`
**作用**：客户侧安全过滤（第二道门）。  
- `min_customer_expected_gain`：最低预期收益（adjusted_cate）
- `max_customer_negative_share`：预留字段（用于扩展更强的风险控制）
- `enable_product_blacklist_gate`：是否启用产品黑白名单门禁（依赖 recommendation_decision）
- `enable_customer_safe_filter`：是否启用客户层最低收益过滤

### 3.4 `BacktestConfig`
**作用**：回测参数。  
- `policy_bins`：触达比例档位（用于 policy_gain_curve）
- `ps_clip_low/high`：OPE 中 ps clipping
- `random_state`：随机种子

---

## 4. 基础工具函数

### 4.1 `estimate_evaldf_memory(n_rows, cfg=None) -> Dict[str, float]`
**作用**：粗略估算 eval_df 内存占用（MB），用于容量评估。  
**输入**：`n_rows`，以及模拟配置（决定 dtype）。  
**输出**：`{"rows": ..., "estimated_MB_min": ...}`（保守下限估算）。

### 4.2 `_as_category(values, ordered=False) -> pd.Categorical`
**作用**：便捷构造 category（节省内存）。

### 4.3 `simulate_evaldf(cfg=None)`
**作用**：生成一份可用于 pipeline 的模拟 eval_df（支持分块生成）。  
**特点**：
- 客户异质性 `cust_sensitivity`
- 产品画像 `prod_type`（全民收益型/精准收割型/高风险波动型/噪声型）
- 更真实投放机制（ps 与 sensitivity 相关）
- 潜在结果框架（mu0/mu1/tau/Y）
- 支持 `chunk_rows>0` 返回迭代器（yield DataFrame）

### 4.4 `validate_eval_df(eval_df)`
**作用**：校验 eval_df 是否包含必要字段 `REQUIRED_COLUMNS`。

### 4.5 `_safe_divide(a, b)`
**作用**：安全除法（b=0 或 NaN 时返回 0）。

### 4.6 `_normalize_score(series)`
**作用**：min-max 归一化，若序列常数则返回全 0。  
**用途**：产品/客户综合打分时的归一化。

---

## 5. 产品层指标（Model/Metric by Product）

### 5.1 `compute_ate_by_product(eval_df) -> pd.DataFrame`
**作用**：按产品聚合计算 ATE 与分布统计：
- `ate = mean(cate)`
- `cate_std`、`cate_p05/p50/p95`
- `sample_size`, `n_customer`
- `treated_rate=mean(T)`
- `outcome_rate=mean(Y)`

### 5.2 `compute_empirical_uplift_by_product(eval_df) -> pd.DataFrame`
**作用**：按产品计算经验 uplift：
- `empirical_uplift = mean(Y|T=1) - mean(Y|T=0)`
并输出 treated/control 的均值与样本量。

### 5.3 `compute_calibration_factor(product_eval) -> pd.DataFrame`
**作用**：产品级校准因子：
- `calibration_factor = empirical_uplift / ate`（并 clip 到 [0,5]）  
**用途**：客户层推荐时 `adjusted_cate = cate * calibration_factor`。

### 5.4 `compute_top_segment_metrics(eval_df, top_ratio) -> pd.DataFrame`
**作用**：Top 人群 uplift 指标：
- 先按 cate 降序排序
- `top_n = ceil(n*top_ratio)`
- `top_uplift_lift = mean(top) - mean(all)`
- `top_vs_rest_gap = mean(top) - mean(rest)`

### 5.5 `compute_negative_uplift_metrics(eval_df) -> pd.DataFrame`
**作用**：风险指标：
- `negative_uplift_ratio = P(cate<0)`
- `treated_negative_uplift_ratio = P(cate<0 | T=1)`（参考）

### 5.6 `compute_qini_auuc_proxy(eval_df) -> pd.DataFrame`
**作用**：Qini/AUUC 的 proxy（流程一致性用，非严格定义）：
- 按 cate 降序排序
- `cum_gain = cumsum(cate)`
- `auuc = mean(cum_gain)`
- `baseline = mean(cate) * (n+1)/2`
- `qini = auuc - baseline`

---

## 6. 产品层解释性：标签与分布形态

### 6.1 `_infer_distribution_shape(row) -> str`
**作用**：基于分位数/方差，粗判断 cate 分布形态（用于解释/排查）。  
可能输出：`degenerate/right_skew_long_tail/left_skew_long_tail/polarized/noisy_symmetric/normal/...`

### 6.2 `_infer_product_tag(row) -> str`
**作用**：将产品归类为：
- `全民收益型`
- `精准收割型`
- `高风险波动型`
- `噪声型`  
依据 ate/neg_ratio/std/spread/top_lift/gap 等启发式规则。

---

## 7. 产品评估主函数：`evaluate_products`

### 7.1 `evaluate_products(eval_df, product_config=None, external_metrics_df=None) -> pd.DataFrame`
**作用**：构建 `product_eval_df`（产品层评估总表），并完成门禁 + 决策 + 打分 + 标签。  
**主要步骤**：
1. `validate_eval_df`
2. 计算产品聚合：ATE / empirical / top / negative / qini-auuc proxy
3. 如提供 external_metrics_df，则覆盖 qini/auuc proxy
4. 如启用校准：merge `calibration_factor`
5. 计算门禁 `pass_*`：
   - `pass_ate, pass_empirical, pass_qini, pass_auuc, pass_top_lift, pass_negative_risk, pass_support`
6. `pass_rate = mean(pass_* )`
7. targeted 逻辑：`pass_targeted`（允许 ATE<0，只对 Top 人群有价值的产品）
8. 输出决策 `recommendation_decision`：
   - `recommend_all`：全门禁都过
   - `recommend_targeted`：pass_targeted
   - `watchlist`：pass_ate
   - `reject`：其它
9. 输出产品综合分 `product_score`：
   - recommend_all 用 `score_mass`
   - recommend_targeted 用 `score_targeted`
10. 输出 `distribution_shape` / `product_tag`

**输出字段**：包含各类聚合指标 + pass_* + decision + score + tag。

---

## 8. 客户层推荐：`generate_recommendations`

### 8.1 `generate_recommendations(eval_df, product_eval_df, customer_config=None, safety_config=None, product_config=None) -> pd.DataFrame`
**作用**：生成客户-产品推荐明细（Top-K）。  
**步骤**：
1. eval_df join product_eval_df（带入 decision、pass_rate、product_score、neg_ratio、calibration_factor）
2. 若启用产品门禁：只保留 `recommend_all/recommend_targeted`
3. `adjusted_cate = cate * calibration_factor`
4. 过滤：`adjusted_cate > min_cate`
5. targeted 产品：只保留该产品内 Top `targeted_top_ratio` 人群（按 adjusted_cate 降序）
6. `recommend_score = 0.65*norm(adjusted_cate) + 0.25*norm(product_score) + 0.10*(1-norm(neg_ratio))`
7. 客户内排序取 Top-K：`rank_in_customer`
8. 安全过滤（可选）：`adjusted_cate >= min_customer_expected_gain`

**输出**：`customer_reco_df`，每行一条客户-产品推荐记录（含 adjusted_cate、recommend_score、rank_in_customer 等）。

---

## 9. 回测模块

### 9.1 `empirical_uplift_on_recommendations(customer_reco_df) -> pd.DataFrame`
**作用**：在推荐子集上做 treated-control uplift（sanity check）。  
`uplift = mean(Y|T=1) - mean(Y|T=0)`  
注意：不等价严格因果策略评估，可能有 selection bias。

### 9.2 `policy_gain_curve(scored_df, score_col, bins, baseline_mode='global_mean') -> pd.DataFrame`
**作用**：触达 Top X% 的收益曲线。  
- 先按 `score_col` 降序排序
- 对每个 `b in bins`，取 top b 的子集
- baseline：
  - `global_mean`：mean(Y in top) - mean(Y in all)
  - `treated_control_in_top`：mean(Y|T=1, top) - mean(Y|T=0, top)

### 9.3 `temporal_stability(eval_df) -> pd.DataFrame`
**作用**：按 date 输出：
- `model_ate = mean(cate)`
- `empirical_uplift = mean(Y|T=1) - mean(Y|T=0)`
并输出 treated/control 样本量。  
实现方式：避免 groupby.apply，使用 sum/count 聚合后差分。

---

## 10. OPE（Off-policy Evaluation）

### 10.1 `ope_ipw_policy_value(cf_df, policy_flag_col, ps_col='ps', ...) -> float`
**作用**：IPW 估计策略价值（人均）：  
value = mean( I(pi=1)*I(T=1)/ps * Y )

### 10.2 `ope_dr_policy_value(cf_df, policy_flag_col, ps_col='ps', mu1_col='mu1', mu0_col='mu0', ...) -> float`
**作用**：Doubly Robust（DR）策略价值：  
DR term = I(pi=1) * [ (mu1-mu0) + T*(Y-mu1)/ps - (1-T)*(Y-mu0)/(1-ps) ]  
value = mean(DR term)

### 10.3 `build_policy_flag_top_pct(df, score_col, top_pct) -> pd.Series`
**作用**：按 score_col 排序，Top top_pct 置 1，否则 0。  
用于 OPE 或策略曲线定义。

---

## 11. 主流程：`run_backtest`（核心入口）

### 11.1 `run_backtest(eval_df, external_metrics_df=None, product_config=None, ...) -> Dict[str, pd.DataFrame]`
**作用**：一键跑完整回测，返回多个表。  
**步骤**：
1) `product_eval_df = evaluate_products(...)`
2) `customer_reco_df = generate_recommendations(...)`
3) `reco_empirical_eval_df = empirical_uplift_on_recommendations(customer_reco_df)`
4) `policy_gain_df = policy_gain_curve(customer_reco_df, score_col=...)`
5) `temporal_df = temporal_stability(eval_df)`
6) `ope_df`：示例里构造 `policy_top20`（按 cate Top20%），尝试算 IPW/DR；缺列则 NaN+原因

**返回 dict keys**：
- `product_eval_df`
- `customer_reco_df`
- `reco_empirical_eval_df`
- `policy_gain_df`
- `temporal_df`
- `ope_df`

---

## 12. 输出与报告（可选）

### 12.1 `export_backtest_results(result, out_dir='backtest_output')`
**作用**：将 result dict 中的 DataFrame 落盘为 csv。

### 12.2 报告辅助函数
- `_fmt(x, nd=4)`：数值格式化（报告展示）
- `_gate_fail_summary(product_eval_df, gate_cols, ...)`：门禁通过率 + 主要失败原因统计
- `_recommendation_diagnosis_text(product_eval_df, ...)`：当无推荐时输出诊断建议（示例数据负 uplift 比例常接近 0.5）

### 12.3 `render_business_report(result, out_path=..., top_products=20, top_reco_rows=50) -> str`
**作用**：生成业务可读 Markdown 报告，并写文件：
- `backtest_output/backtest_report.md`（UTF-8）
- `backtest_output/backtest_report_gbk.md`（Windows 兼容）

---

## 13. 脚本入口（__main__）示例流程

脚本底部 `if __name__ == "__main__":` 的示例做了：
1. 用 `EvalDFSimConfig` 生成模拟 eval_df（支持 chunk_rows）
2. 可选：`suggest_default_thresholds` 给一套基于分布的阈值建议（自动化调参起点）
3. `run_backtest(...)` 跑完整 pipeline
4. `render_business_report(...)` 产出 markdown 报告 + 关键统计打印

---

## 14. 建议的二次开发点（维护者视角）

1) **external qini/auuc 接入**：将 causalml 或其它工具真实 qini/auuc merge 覆盖 proxy（脚本已支持）。  
2) **policy_flag 用最终推荐清单**：目前 OPE 示例使用 Top20% by cate；更贴近业务应把 `customer_reco_df` 映射回全表生成 policy_flag。  
3) **大数据优化**：当前 v1 是 pandas groupby/sort，2千万行级会慢且吃内存；建议用 v2（parquet + DuckDB）方案。  
4) **门禁阈值自动化**：`suggest_default_thresholds` 只是起点，可扩展为网格搜索/贝叶斯优化，目标函数可用 uplift、风险约束、覆盖率等。
