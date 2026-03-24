# `backtest_full_pipeline.py` 文档（函数说明 + 全流程梳理）

> 因果推荐指标体系：完整回测脚本（整理版）  
> 目标：把 `backtest_full_pipeline.py` 中每个函数的作用、定义（签名）、输入/输出、关键口径与注意事项记录下来，并说明端到端 backtest 流程。

---

## 1. 这个脚本解决什么问题？

该脚本把“**因果推荐（CATE/uplift）**”的评估与回测串成一条可运行 pipeline，主要分为：

1) **产品层评估（Product Level）**  
- 用 ATE / empirical uplift / Qini(AUUC proxy) / Top uplift / 负 uplift 风险等指标，对每个产品做聚合评估与门禁筛选，产出产品标签与是否可推荐的决策。

2) **客户层推荐（Customer Level / Policy Generation）**  
- 基于（可选校准后的）CATE，为每个客户在可推荐产品集合中挑 Top-K，形成客户-产品推荐清单。

3) **回测评估（Backtest / Offline Evaluation）**  
- 在推荐子集上做经验 uplift sanity check  
- policy gain curve（不同触达比例下收益曲线）  
- 时间稳定性（按 date 的 model ATE vs empirical uplift）  
- OPE（IPW/DR）离线策略价值评估（需要 ps/mu0/mu1）

4) **报告输出（Reporting）**  
- 把结果整理成可读的 Markdown 报告（同时输出 UTF-8 与 GBK 版本，便于 Windows 环境打开）。

---

## 2. 数据输入：`eval_df` 规范（最少字段 + 可选字段）

### 2.1 最少必需字段（长表 long format）

| 字段 | 含义 | 类型建议 |
|---|---|---|
| `cust_id` | 客户 ID | int/category |
| `product_id` | 产品 ID | int/category |
| `date` | 时间维度（可为天/周/月等） | datetime/category |
| `cate` | 模型输出的个体增量（CATE/uplift） | float |
| `T` | 历史真实是否触达/达标（0/1） | int8 |
| `Y` | 结果变量（金额或 0/1） | float/int |

脚本会在 `validate_eval_df()` 中检查这些字段。

### 2.2 可选字段（用于 OPE）

| 字段 | 含义 | 用途 |
|---|---|---|
| `ps` | propensity score（倾向得分） | IPW/DR OPE |
| `mu1` | outcome model 对 `T=1` 的预测 | DR OPE |
| `mu0` | outcome model 对 `T=0` 的预测 | DR OPE |

---

## 3. 端到端 backtest 流程（run_backtest 总览）

核心入口函数：`run_backtest(eval_df, ...) -> Dict[str, pd.DataFrame]`

### 3.1 流程步骤（按代码顺序）

1. **输入校验**：`validate_eval_df(eval_df)`
2. **产品评估**：`evaluate_products(eval_df, product_config, external_metrics_df)`  
   - 计算 ATE、经验 uplift、Top uplift 指标、负 uplift 风险、Qini/AUUC proxy  
   - 可选用 empirical/ate 做粗校准 `calibration_factor`  
   - 多门禁筛选，输出 `recommendation_decision`（`recommend_all` / `recommend_targeted` / `watchlist` / `reject`）
3. **生成推荐**：`generate_recommendations(eval_df, product_eval_df, ...)`  
   - 只保留可推荐产品（all + targeted）  
   - 计算 `adjusted_cate = cate * calibration_factor`  
   - 客户内 Top-K（含安全过滤）  
   - targeted 产品只对产品内 Top 人群开放（避免 ATE<0 全量推）
4. **推荐子集经验 uplift**：`empirical_uplift_on_recommendations(customer_reco_df)`
5. **策略收益曲线**：`policy_gain_curve(customer_reco_df, score_col, bins, baseline_mode)`
6. **时间稳定性**：`temporal_stability(eval_df)`
7. **OPE 离线评估**：  
   - 构造策略 flag：`build_policy_flag_top_pct(cf_df, score_col="cate", top_pct=0.2)`  
   - IPW：`ope_ipw_policy_value(...)`（需 `ps`）  
   - DR：`ope_dr_policy_value(...)`（需 `ps, mu1, mu0`）

### 3.2 `run_backtest` 输出表

| key | DataFrame 含义 |
|---|---|
| `product_eval_df` | 产品层聚合指标 + 门禁结果 + 标签 + 评分 |
| `customer_reco_df` | 客户-产品推荐清单（含 `adjusted_cate`, `recommend_score`） |
| `reco_empirical_eval_df` | 推荐子集 treated-control uplift（sanity check） |
| `policy_gain_df` | 不同触达比例下的收益曲线（经验口径） |
| `temporal_df` | 时间稳定性（按 date） |
| `ope_df` | OPE 结果（ipw/dr 是否成功、错误信息等） |

---

## 4. 配置类（dataclasses）

> 这些类主要用于参数管理，提升可读性与可复用性。

### 4.1 `ProductDecisionConfig`

**定义：**
```python
@dataclass
class ProductDecisionConfig:
    ...
```

**作用：**
- 定义产品层门禁阈值与策略开关（是否支持 targeted 推荐、Top 比例、校准等）。

**关键字段：**
- `min_ate`: ATE 门槛（`mean(cate)`）
- `min_empirical_uplift`: empirical uplift 门槛（`E[Y|T=1]-E[Y|T=0]`）
- `min_qini`, `min_auuc`: 排序有效性门槛（proxy 或外部真实值）
- `min_top_uplift_lift`: Top uplift 相对整体 uplift 的优势门槛
- `max_negative_uplift_ratio`: 产品内 `cate<0` 占比上限（风险控制）
- `min_support_samples`: 样本量门槛
- `top_ratio`: Top 人群比例（用于 top 指标）
- `enable_calibration`: 是否使用 `empirical_uplift/ate` 做粗校准

**targeted 推荐相关：**
- `enable_targeted_reco`: 开启 targeted 推荐策略
- `targeted_top_ratio`: targeted 产品只对产品内 Top X% 人群开放推荐
- `min_targeted_top_cate`, `min_targeted_lift`, `allow_targeted_when_ate_negative`: targeted 门禁参数

---

### 4.2 `CustomerDecisionConfig`

**定义：**
```python
@dataclass
class CustomerDecisionConfig:
    ...
```

**作用：**
- 定义客户层推荐策略：`min_cate`、每客 Top-K 等。

**关键字段：**
- `min_cate`: 最小可推荐增量（使用 `adjusted_cate`）
- `top_k_per_customer`: 每客户最多推荐 K 个产品
- `customer_weight_col`: 客户权重（预留，当前未用）

---

### 4.3 `SafetyConfig`

**定义：**
```python
@dataclass
class SafetyConfig:
    ...
```

**作用：**
- 客户层二次安全过滤开关（是否过滤产品黑名单/是否启用客户级最低收益门槛等）。

**关键字段：**
- `min_customer_expected_gain`: 最低预期收益（按 `adjusted_cate`）
- `enable_product_blacklist_gate`: 是否只保留 recommend 产品
- `enable_customer_safe_filter`: 是否启用收益过滤

---

### 4.4 `BacktestConfig`

**定义：**
```python
@dataclass
class BacktestConfig:
    ...
```

**作用：**
- 回测层设置：policy gain curve 档位、OPE ps clip 范围等。

**关键字段：**
- `policy_bins`: 触达比例档位
- `ps_clip_low`, `ps_clip_high`: OPE 中 ps clip 范围（避免极端权重）

---

## 5. 基础工具函数（Utilities）

### 5.1 `validate_eval_df(eval_df: pd.DataFrame) -> None`

**作用：**
- 检查 `eval_df` 是否包含 `REQUIRED_COLUMNS = ["cust_id","product_id","date","cate","T","Y"]`。
- 缺列则抛 `ValueError`。

---

### 5.2 `_safe_divide(a: float, b: float) -> float`

**作用：**
- 安全除法，避免 `b=0` 或 NaN 时异常；返回 0.0。

---

### 5.3 `_normalize_score(series: pd.Series) -> pd.Series`

**作用：**
- 对分数做 min-max 归一化到 [0,1]，用于组合评分。
- 若序列只有 1 个唯一值，则返回全 0（避免除 0）。

---

## 6. 模拟数据模块（EvalDFSimConfig / simulate_evaldf）

> 用于调试 pipeline；真实回测时应替换为你的真实 `eval_df`。

### 6.1 `EvalDFSimConfig`

**定义：**
```python
@dataclass
class EvalDFSimConfig:
    ...
```

**作用：**
- 控制模拟数据规模、处理率、噪声、内存优化开关、是否分块生成等。

---

### 6.2 `estimate_evaldf_memory(n_rows: int, cfg: Optional[EvalDFSimConfig] = None) -> Dict[str, float]`

**作用：**
- 粗略估算 eval_df 的内存占用（MB），用于容量判断。

**输出：**
- `{"rows": ..., "estimated_MB_min": ...}`

---

### 6.3 `_as_category(values: np.ndarray, ordered: bool = False) -> pd.Categorical`

**作用：**
- 将 numpy array 转为 pandas Categorical（节省内存）。

---

### 6.4 `simulate_evaldf(cfg: Optional[EvalDFSimConfig] = None)`

**作用：**
- 生成可用于 `run_backtest()` 的模拟 `eval_df`。
- 特点：引入客户异质性与 4 类产品画像，便于验证门禁与 targeted 推荐逻辑是否按预期工作。

**生成逻辑概要：**
- 客户敏感度 `cust_sensitivity`：控制个体差异
- 产品类型 `prod_type` 四类：`全民收益型 / 精准收割型 / 高风险波动型 / 噪声型`
- `cate`：由产品基础效应 + 客户敏感度 + targeted boost + 噪声生成，并 clip 到 [-6,6]
- `ps`：由客户敏感度等生成（更真实的投放机制），clip 到 [0.01,0.99]
- `T ~ Bernoulli(ps)`
- `tau = true_tau_scale * tanh(cate)`，潜在结果 `mu0, mu1` 与实际 `Y`

**输出字段：**
- 必需字段：`cust_id, product_id, date, cate, T, Y`
- OPE 调试字段：`ps, mu0, mu1`
- 额外真值字段：`product_type_true`（用于验证标签/决策）

**分块生成：**
- 当 `cfg.chunk_rows > 0` 时，返回 iterator（yield DataFrame），适合大规模调试。

---

## 7. 产品层指标（Product Aggregation Metrics）

### 7.1 `compute_ate_by_product(eval_df: pd.DataFrame) -> pd.DataFrame`

**作用：**
- 对每个产品聚合 `cate`，计算：
  - `ate = mean(cate)`
  - `cate_std`、`cate_p05/p50/p95`（用于分布形态与风险解释）
  - `sample_size`、`n_customer`、`treated_rate`、`outcome_rate`

---

### 7.2 `compute_empirical_uplift_by_product(eval_df: pd.DataFrame) -> pd.DataFrame`

**作用：**
- 对每个产品计算经验 uplift：  
  `empirical_uplift = E[Y|T=1] - E[Y|T=0]`

**输出：**
- `treated_mean_outcome / control_mean_outcome / treated_n / control_n`

**注意：**
- 这不是严格因果估计，受历史策略与选择偏差影响（但可作为 sanity check 与业务对齐信号）。

---

### 7.3 `compute_calibration_factor(product_eval: pd.DataFrame) -> pd.DataFrame`

**作用：**
- 粗校准：  
  `calibration_factor = empirical_uplift / ate`（ate=0 时为 1.0）  
- 输出每个产品的校准系数，用于 `adjusted_cate = cate * calibration_factor`。

**注意：**
- 这是按产品整体比例校准；更精细可扩展为分桶校准/Isotonic/Platt。

---

### 7.4 `compute_top_segment_metrics(eval_df: pd.DataFrame, top_ratio: float) -> pd.DataFrame`

**作用：**
- 对每个产品按 `cate` 降序取 Top `top_ratio` 人群，输出：
  - `top_uplift_lift = top_mean_cate - overall_mean_cate`
  - `top_vs_rest_gap = top_mean_cate - rest_mean_cate`

---

### 7.5 `compute_negative_uplift_metrics(eval_df: pd.DataFrame) -> pd.DataFrame`

**作用：**
- 风险指标：
  - `negative_uplift_ratio = P(cate < 0)`
  - `treated_negative_uplift_ratio = P(cate<0 | T=1)`（仅参考）

---

### 7.6 `compute_qini_auuc_proxy(eval_df: pd.DataFrame) -> pd.DataFrame`

**作用：**
- 给出 Qini/AUUC 的 proxy（流程占位）：  
  对 `cate` 降序求累计和构造 `auuc` 与 `qini` 近似指标。
- 若你已有 causalml 等输出的真实 qini/auuc，应在 `evaluate_products()` 中通过 `external_metrics_df` 覆盖。

**注意：**
- proxy 不等价于严格定义的 Qini/AUUC，仅用于保持 pipeline 可跑。

---

## 8. 产品层评估与门禁（Gating & Tagging）

### 8.1 `_infer_distribution_shape(row: pd.Series) -> str`

**作用：**
- 基于分位数与 std 粗略判断产品内 `cate` 分布形态，辅助解释与排查：
  - `right_skew_long_tail / left_skew_long_tail / polarized / noisy_symmetric / normal / degenerate / unknown`

---

### 8.2 `_infer_product_tag(row: pd.Series) -> str`

**作用：**
- 给产品打主标签（4 类）：
  - `全民收益型`
  - `精准收割型`（ATE 不高甚至偏负，但 Top 命中强，适合 targeted）
  - `高风险波动型`（方差/分位差大，负 uplift 占比高）
  - `噪声型`（整体近 0）

---

### 8.3 `evaluate_products(...) -> pd.DataFrame`

**定义（签名）：**
```python
def evaluate_products(
    eval_df: pd.DataFrame,
    product_config: Optional[ProductDecisionConfig] = None,
    external_metrics_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    ...
```

**作用：**
- 聚合产品指标 + 门禁判断 + 决策输出 + 综合评分 + 标签输出。

**主要输出字段：**
- 指标：`ate, empirical_uplift, qini, auuc, top_uplift_lift, top_vs_rest_gap, negative_uplift_ratio, ...`
- 校准：`calibration_factor`
- 门禁：`pass_ate, pass_empirical, pass_qini, pass_auuc, pass_top_lift, pass_negative_risk, pass_support`
- targeted 门禁：`pass_targeted`
- 决策：`recommendation_decision` ∈ {`recommend_all`, `recommend_targeted`, `watchlist`, `reject`}
- 打分：`product_score`
- 标签：`product_tag`, `distribution_shape`

**门禁与决策逻辑（简述）：**
- 全量推荐：所有 gate_cols 都通过 -> `recommend_all`
- 定向推荐：满足 targeted 条件 -> `recommend_targeted`
- 否则：ATE 过线 -> `watchlist`，否则 -> `reject`

---

## 9. 客户层推荐（Policy Generation）

### 9.1 `generate_recommendations(...) -> pd.DataFrame`

**定义（签名）：**
```python
def generate_recommendations(
    eval_df: pd.DataFrame,
    product_eval_df: pd.DataFrame,
    customer_config: Optional[CustomerDecisionConfig] = None,
    safety_config: Optional[SafetyConfig] = None,
    product_config: Optional[ProductDecisionConfig] = None,
) -> pd.DataFrame:
    ...
```

**作用：**
- 生成客户-产品推荐对（最终策略清单）。

**核心步骤：**
1) 将 `eval_df` 与 `product_eval_df` 合并（带上决策、风险、校准因子等）
2) 产品过滤：只保留 `recommend_all` + `recommend_targeted`
3) 校准：`adjusted_cate = cate * calibration_factor`
4) 客户-产品门禁：`adjusted_cate > min_cate`
5) targeted 产品二次过滤：只保留产品内 Top `targeted_top_ratio` 的记录
6) 推荐打分：  
   `recommend_score = 0.65*norm(adjusted_cate) + 0.25*norm(product_score) + 0.10*(1-norm(negative_uplift_ratio))`
7) 客户内 Top-K：按 `recommend_score` 排序取前 K
8) 安全过滤：`adjusted_cate >= min_customer_expected_gain`

**输出：**
- 推荐明细表：包含 `adjusted_cate, recommend_score, rank_in_customer` 等字段。

**性能备注：**
- targeted 的 `groupby.rank/transform` 只对 targeted 子集做，避免全量 2 千万级数据慢到不可用。

---

## 10. 回测模块（Backtest Metrics）

### 10.1 `empirical_uplift_on_recommendations(customer_reco_df: pd.DataFrame) -> pd.DataFrame`

**作用：**
- 在推荐清单子集上计算经验 uplift（treated-control）。
- 主要用于 sanity check（不是严格因果策略评估）。

---

### 10.2 `policy_gain_curve(scored_df, score_col, bins, baseline_mode="global_mean") -> pd.DataFrame`

**作用：**
- 绘制“触达 Top X%”时的收益曲线，回答：只触达高分人群，经验收益是否更高？

**baseline_mode：**
- `global_mean`：Top 子集 Y 均值 - 全局 Y 均值
- `treated_control_in_top`：Top 子集 treated 均值 - control 均值（更像 uplift，但噪声更大）

---

### 10.3 `temporal_stability(eval_df: pd.DataFrame) -> pd.DataFrame`

**作用：**
- 按 `date` 输出：
  - `model_ate = mean(cate)`
  - `empirical_uplift = E[Y|T=1]-E[Y|T=0]`
- 用 groupby 聚合避免 apply 的性能问题与 warning。

---

## 11. OPE（Off-Policy Evaluation）

### 11.1 `ope_ipw_policy_value(...) -> float`

**作用：**
- IPW 估计策略价值（人均）：
  \[
  value = \frac{1}{N}\sum_i I(\pi(x_i)=1)\cdot I(T_i=1)\cdot \frac{Y_i}{ps_i}
  \]
- 需要 `ps`。

**注意：**
- ps 会 clip 到 `[ps_clip_low, ps_clip_high]`，避免极端权重。

---

### 11.2 `ope_dr_policy_value(...) -> float`

**作用：**
- Doubly Robust（DR）估计策略价值（推荐）：
  \[
  DR = I(\pi=1)\cdot[(\mu_1-\mu_0)+T\frac{Y-\mu_1}{ps}-(1-T)\frac{Y-\mu_0}{1-ps}]
  \]
- 需要 `ps, mu1, mu0`。

---

### 11.3 `build_policy_flag_top_pct(df, score_col, top_pct) -> pd.Series`

**作用：**
- 将 `df` 按 `score_col` 排序，取 Top `top_pct` 作为策略推荐标记（1/0）。
- 在 `run_backtest()` 中用于构造示例策略 `policy_top20`。

---

## 12. 输出与报告（Export & Reporting）

### 12.1 `export_backtest_results(result: Dict[str, pd.DataFrame], out_dir="backtest_output") -> None`

**作用：**
- 将 `run_backtest()` 的每个 DataFrame 落盘为 CSV。

---

### 12.2 `_fmt(x: object, nd: int = 4) -> str`

**作用：**
- 报告渲染时的格式化工具：数字千分位、小数位控制、NaN/Inf 输出为 `-`。

---

### 12.3 `_gate_fail_summary(product_eval_df, gate_cols, decision_col="recommendation_decision") -> pd.DataFrame`

**作用：**
- 汇总每个门禁的通过率，并统计失败原因 Top1/Top2。
- 同时把 decision 的分布统计放入 `DataFrame.attrs["decision_summary"]` 供报告渲染使用。

---

### 12.4 `_recommendation_diagnosis_text(product_eval_df, product_config=None) -> str`

**作用：**
- 当推荐池为空/很少时，输出可读诊断文本（重点解释 negative uplift 门禁为何导致无推荐，并给出放宽门禁/改模拟分布的建议）。

---

### 12.5 `render_business_report(result, out_path="backtest_output/backtest_report.md", ...) -> str`

**作用：**
- 将 `run_backtest()` 输出整理成业务可读 Markdown 报告：
  - 概览
  - 产品层：标签分布、门禁通过率、决策分布、Top 产品
  - 客户层：Top 推荐明细
  - policy curve
  - temporal stability
  - OPE
  - 附录：输出表说明
- 同时写入 UTF-8 与 GBK 版本（GBK 用于 Windows 环境更友好）。

---

## 13. 自动阈值建议（可选）

### 13.1 `suggest_default_thresholds(...) -> Dict[str, object]`

**作用：**
- 基于当前批次数据分布（分位数）给出门禁阈值建议，作为调参起点。
- 返回：
  - `product_config`（可用于 `ProductDecisionConfig(**product_config)`）
  - `customer_config`
  - `safety_config`
  - `diagnostics`（分布分位数等）

**注意：**
- 这不是最优阈值，只是避免固定阈值在不同批次下失灵的一种实用起点。

---

## 14. 示例运行（`__main__`）

脚本尾部示例做了：

1) 构造模拟 `eval_df`（`simulate_evaldf(sim_cfg)`）  
2) （可选）自动建议阈值  
3) 跑 `run_backtest(...)`  
4) 生成 Markdown 报告 `render_business_report(...)`  
5) 打印一些关键统计，便于验证推荐/标签是否生效

---

## 15. 维护建议（如何持续更新此文档）

- 只要你新增/修改函数，请同步更新「对应章节」：
  - 函数签名（定义）
  - 输入字段要求（尤其是 DataFrame 列）
  - 输出字段说明
  - 关键口径/坑点（selection bias、proxy 指标、性能）
- 如果后续把 Qini/AUUC 换成严格实现，可在 `compute_qini_auuc_proxy` 小节补充真实定义与实现来源。
