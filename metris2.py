from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ============================================================
# 配置区
# ============================================================

@dataclass
class ProductDecisionConfig:
    min_ate: float = 0.0
    min_qini: float = 0.0
    min_auuc: float = 0.0
    min_top_uplift_lift: float = 0.0
    min_empirical_uplift: float = 0.0
    max_negative_uplift_ratio: float = 0.50
    min_recommendable_customers: int = 100
    min_support_samples: int = 300
    top_ratio: float = 0.20
    use_bootstrap_significance: bool = False
    bootstrap_rounds: int = 200
    random_state: int = 42
    enable_calibration: bool = True


@dataclass
class CustomerDecisionConfig:
    min_cate: float = 0.0
    top_k_per_customer: int = 3
    min_product_pass_rate: float = 0.0
    customer_weight_col: Optional[str] = None


@dataclass
class SafetyConfig:
    max_customer_negative_share: float = 0.5
    min_customer_expected_gain: float = 0.0
    enable_product_blacklist_gate: bool = True
    enable_customer_safe_filter: bool = True


@dataclass
class BusinessConfig:
    value_per_unit_y: Optional[float] = None
    cost_per_recommendation: Optional[float] = None
    fixed_cost_per_product: Optional[float] = None


# ============================================================
# 基础工具函数
# ============================================================

REQUIRED_COLUMNS = ["cust_id", "product_id", "date", "cate", "T", "Y"]


def validate_eval_df(eval_df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in eval_df.columns]
    if missing:
        raise ValueError(f"eval_df 缺少必要字段: {missing}")


def _safe_divide(a: float, b: float) -> float:
    if b == 0 or pd.isna(b):
        return 0.0
    return float(a) / float(b)


def _normalize_score(series: pd.Series) -> pd.Series:
    s = series.astype(float).copy()
    if s.nunique(dropna=True) <= 1:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.min()) / (s.max() - s.min())


# ============================================================
# 模型层指标
# ============================================================


def compute_ate_by_product(eval_df: pd.DataFrame) -> pd.DataFrame:
    out = (
        eval_df.groupby("product_id", as_index=False)
        .agg(
            sample_size=("cust_id", "count"),
            n_customer=("cust_id", "nunique"),
            ate=("cate", "mean"),
            cate_std=("cate", "std"),
            treated_rate=("T", "mean"),
            outcome_rate=("Y", "mean"),
        )
    )
    out["cate_std"] = out["cate_std"].fillna(0.0)
    return out


# ============================================================
# Empirical Uplift（真实增量）
# ============================================================


def compute_empirical_uplift_by_product(eval_df: pd.DataFrame) -> pd.DataFrame:
    frames = []

    for product_id, g in eval_df.groupby("product_id"):
        treated = g[g["T"] == 1]
        control = g[g["T"] == 0]

        treated_mean = treated["Y"].mean()
        control_mean = control["Y"].mean()

        uplift = treated_mean - control_mean

        frames.append({
            "product_id": product_id,
            "empirical_uplift": uplift,
            "treated_mean_outcome": treated_mean,
            "control_mean_outcome": control_mean,
            "treated_n": len(treated),
            "control_n": len(control),
        })

    return pd.DataFrame(frames)


# ============================================================
# Uplift Calibration（增量校准）
# ============================================================


def compute_calibration_factor(product_eval: pd.DataFrame) -> pd.DataFrame:
    df = product_eval.copy()

    df["calibration_factor"] = df.apply(
        lambda r: _safe_divide(r["empirical_uplift"], r["ate"]) if r["ate"] != 0 else 1.0,
        axis=1,
    )

    df["calibration_factor"] = df["calibration_factor"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    df["calibration_factor"] = df["calibration_factor"].clip(0.0, 5.0)

    return df[["product_id", "calibration_factor"]]


# ============================================================
# Top人群指标
# ============================================================


def compute_top_segment_metrics(eval_df: pd.DataFrame, top_ratio: float) -> pd.DataFrame:
    frames = []

    for product_id, g in eval_df.groupby("product_id"):
        g = g.sort_values("cate", ascending=False)
        n = len(g)
        top_n = max(1, int(np.ceil(n * top_ratio)))

        top_g = g.iloc[:top_n]
        rest_g = g.iloc[top_n:]

        overall_mean = g["cate"].mean()
        top_mean = top_g["cate"].mean()
        rest_mean = rest_g["cate"].mean() if len(rest_g) else top_mean

        frames.append({
            "product_id": product_id,
            "top_uplift_lift": top_mean - overall_mean,
            "top_vs_rest_gap": top_mean - rest_mean,
        })

    return pd.DataFrame(frames)


# ============================================================
# 风险指标
# ============================================================


def compute_negative_uplift_metrics(eval_df: pd.DataFrame) -> pd.DataFrame:
    frames = []

    for product_id, g in eval_df.groupby("product_id"):
        neg_mask = g["cate"] < 0
        treated_mask = g["T"] == 1

        frames.append({
            "product_id": product_id,
            "negative_uplift_ratio": neg_mask.mean(),
            "treated_negative_uplift_ratio": (
                (neg_mask & treated_mask).sum() / treated_mask.sum()
                if treated_mask.sum() > 0 else 0.0
            ),
        })

    return pd.DataFrame(frames)


# ============================================================
# Qini / AUUC Proxy
# ============================================================


def compute_qini_auuc_proxy(eval_df: pd.DataFrame) -> pd.DataFrame:
    frames = []

    for product_id, g in eval_df.groupby("product_id"):
        g = g.sort_values("cate", ascending=False)
        n = len(g)
        if n == 0:
            continue

        cum_gain = g["cate"].cumsum()
        auuc = cum_gain.mean()
        baseline = g["cate"].mean() * (n + 1) / 2.0
        qini = auuc - baseline

        frames.append({
            "product_id": product_id,
            "auuc": float(auuc),
            "qini": float(qini),
        })

    return pd.DataFrame(frames)


# ============================================================
# 产品层评估
# ============================================================


def evaluate_products(
    eval_df: pd.DataFrame,
    product_config: Optional[ProductDecisionConfig] = None,
    external_metrics_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:

    validate_eval_df(eval_df)
    config = product_config or ProductDecisionConfig()

    ate_df = compute_ate_by_product(eval_df)
    emp_df = compute_empirical_uplift_by_product(eval_df)
    top_df = compute_top_segment_metrics(eval_df, config.top_ratio)
    neg_df = compute_negative_uplift_metrics(eval_df)
    rank_df = compute_qini_auuc_proxy(eval_df)

    product_eval = (
        ate_df
        .merge(emp_df, on="product_id", how="left")
        .merge(top_df, on="product_id", how="left")
        .merge(neg_df, on="product_id", how="left")
        .merge(rank_df, on="product_id", how="left")
    )

    if external_metrics_df is not None and not external_metrics_df.empty:
        cols = [c for c in ["product_id", "qini", "auuc"] if c in external_metrics_df.columns]
        product_eval = product_eval.drop(columns=["qini", "auuc"], errors="ignore")
        product_eval = product_eval.merge(external_metrics_df[cols], on="product_id", how="left")

    if config.enable_calibration:
        calib_df = compute_calibration_factor(product_eval)
        product_eval = product_eval.merge(calib_df, on="product_id", how="left")
    else:
        product_eval["calibration_factor"] = 1.0

    product_eval["pass_ate"] = product_eval["ate"] > config.min_ate
    product_eval["pass_empirical"] = product_eval["empirical_uplift"] > config.min_empirical_uplift
    product_eval["pass_qini"] = product_eval["qini"] > config.min_qini
    product_eval["pass_auuc"] = product_eval["auuc"] > config.min_auuc
    product_eval["pass_top_lift"] = product_eval["top_uplift_lift"] > config.min_top_uplift_lift
    product_eval["pass_negative_risk"] = product_eval["negative_uplift_ratio"] <= config.max_negative_uplift_ratio
    product_eval["pass_support"] = product_eval["sample_size"] >= config.min_support_samples

    gate_cols = [
        "pass_ate",
        "pass_empirical",
        "pass_qini",
        "pass_auuc",
        "pass_top_lift",
        "pass_negative_risk",
        "pass_support",
    ]

    product_eval["pass_rate"] = product_eval[gate_cols].mean(axis=1)

    product_eval["recommendation_decision"] = np.where(
        product_eval[gate_cols].all(axis=1),
        "recommend",
        np.where(product_eval["pass_ate"], "watchlist", "reject"),
    )

    product_eval["product_score"] = (
        0.25 * _normalize_score(product_eval["ate"]) +
        0.20 * _normalize_score(product_eval["empirical_uplift"]) +
        0.15 * _normalize_score(product_eval["qini"]) +
        0.15 * _normalize_score(product_eval["auuc"]) +
        0.15 * _normalize_score(product_eval["top_uplift_lift"]) +
        0.10 * (1 - _normalize_score(product_eval["negative_uplift_ratio"]))
    )

    return product_eval.sort_values(["recommendation_decision", "product_score"], ascending=[True, False])


# ============================================================
# 客户层推荐
# ============================================================


def evaluate_customers(
    eval_df: pd.DataFrame,
    product_eval_df: pd.DataFrame,
    customer_config: Optional[CustomerDecisionConfig] = None,
    safety_config: Optional[SafetyConfig] = None,
) -> pd.DataFrame:

    validate_eval_df(eval_df)
    customer_config = customer_config or CustomerDecisionConfig()
    safety_config = safety_config or SafetyConfig()

    candidate_df = eval_df.merge(
        product_eval_df[[
            "product_id",
            "recommendation_decision",
            "pass_rate",
            "product_score",
            "negative_uplift_ratio",
            "calibration_factor",
        ]],
        on="product_id",
        how="left",
    )

    candidate_df["product_is_approved"] = candidate_df["recommendation_decision"] == "recommend"
    candidate_df = candidate_df[candidate_df["product_is_approved"]]

    candidate_df["adjusted_cate"] = candidate_df["cate"] * candidate_df["calibration_factor"]
    candidate_df = candidate_df[candidate_df["adjusted_cate"] > customer_config.min_cate]

    candidate_df["recommend_score"] = (
        0.65 * _normalize_score(candidate_df["adjusted_cate"]) +
        0.25 * _normalize_score(candidate_df["product_score"]) +
        0.10 * (1 - _normalize_score(candidate_df["negative_uplift_ratio"]))
    )

    candidate_df["rank_in_customer"] = candidate_df.groupby("cust_id")["recommend_score"].rank(
        method="first", ascending=False
    )

    customer_reco = candidate_df[candidate_df["rank_in_customer"] <= customer_config.top_k_per_customer]

    if safety_config.enable_customer_safe_filter:
        customer_reco = customer_reco[
            (customer_reco["adjusted_cate"] >= safety_config.min_customer_expected_gain)
        ]

    return customer_reco.sort_values(["cust_id", "rank_in_customer"]).reset_index(drop=True)


# ============================================================
# 推荐真实增量验证
# ============================================================


def evaluate_empirical_uplift_on_recommendations(customer_reco_df: pd.DataFrame) -> pd.DataFrame:
    if customer_reco_df.empty:
        return pd.DataFrame([{"empirical_uplift": 0.0}])

    treated = customer_reco_df[customer_reco_df["T"] == 1]
    control = customer_reco_df[customer_reco_df["T"] == 0]

    uplift = treated["Y"].mean() - control["Y"].mean()

    return pd.DataFrame([{
        "empirical_uplift": uplift,
        "treated_mean_outcome": treated["Y"].mean(),
        "control_mean_outcome": control["Y"].mean(),
    }])


# ============================================================
# 主流程
# ============================================================


def run_causal_recommendation_pipeline_v2(
    eval_df: pd.DataFrame,
    external_metrics_df: Optional[pd.DataFrame] = None,
    product_config: Optional[ProductDecisionConfig] = None,
    customer_config: Optional[CustomerDecisionConfig] = None,
    safety_config: Optional[SafetyConfig] = None,
    business_config: Optional[BusinessConfig] = None,
) -> Dict[str, pd.DataFrame]:

    product_eval_df = evaluate_products(
        eval_df=eval_df,
        product_config=product_config,
        external_metrics_df=external_metrics_df,
    )

    customer_reco_df = evaluate_customers(
        eval_df=eval_df,
        product_eval_df=product_eval_df,
        customer_config=customer_config,
        safety_config=safety_config,
    )

    reco_empirical_df = evaluate_empirical_uplift_on_recommendations(customer_reco_df)

    return {
        "product_eval_df": product_eval_df,
        "customer_reco_df": customer_reco_df,
        "reco_empirical_eval_df": reco_empirical_df,
    }


# ---

# # V3 决策增强版（策略收益 + 稳定性 + ROI自动化）

# ## 一、策略收益曲线（Policy Simulation）
# > 目标：回答“如果我只触达Top X%的客户，真实能带来多少增量？”

# ### Step 1：准备评分数据
# 假设已有：
# - `eval_df`：包含 `cust_id, product_id, cate, Y`
# - `recommend_df`：包含 `cust_id, product_id, cate_rank`

# ```python
# import pandas as pd

# # 合并推荐顺序
# policy_df = eval_df.merge(
#     recommend_df[["cust_id","product_id","cate_rank"]],
#     on=["cust_id","product_id"],
#     how="left"
# )

# policy_df = policy_df.sort_values("cate_rank")
# policy_df["cum_pct"] = (
#     policy_df.reset_index().index + 1
# ) / len(policy_df)
# ```

# ### Step 2：分段模拟不同触达比例

# ```python
# bins = [0.01,0.02,0.05,0.1,0.2,0.3,0.5,1.0]
# results = []

# for b in bins:
#     sub = policy_df[policy_df.cum_pct <= b]
#     uplift = sub["Y"].mean() - policy_df["Y"].mean()
#     results.append((b, uplift))

# policy_gain_df = pd.DataFrame(results, columns=["top_pct","uplift_gain"])
# ```

# ### Step 3：策略曲线可视化

# ```python
# import matplotlib.pyplot as plt

# plt.figure()
# plt.plot(policy_gain_df["top_pct"], policy_gain_df["uplift_gain"])
# plt.xlabel("Top Customer %")
# plt.ylabel("Empirical Uplift Gain")
# plt.title("Policy Gain Curve")
# plt.show()
# ```

# 👉 用途：
# - 决定外呼资源分配比例
# - 决定营销预算边界

# ---

# ## 二、时间稳定性（Temporal Stability）
# > 目标：模型是否“当期有效、长期失效”？

# ### Step 1：按时间计算ATE

# ```python
# time_ate = (
#     eval_df
#     .groupby("date")["cate"]
#     .mean()
#     .reset_index(name="ATE")
# )
# ```

# ### Step 2：按时间计算真实增量

# ```python
# time_empirical = (
#     eval_df
#     .groupby("date")
#     .apply(lambda x: x.loc[x.T==1,"Y"].mean() - x.loc[x.T==0,"Y"].mean())
#     .reset_index(name="empirical_uplift")
# )
# ```

# ### Step 3：稳定性可视化

# ```python
# plt.figure()
# plt.plot(time_ate["date"], time_ate["ATE"], label="Model ATE")
# plt.plot(time_empirical["date"], time_empirical["empirical_uplift"], label="Empirical Uplift")
# plt.legend()
# plt.title("Temporal Stability")
# plt.show()
# ```

# 👉 用途：
# - 判断是否需要月度重训
# - 判断节假日/市场波动影响

# ---

# ## 三、推荐ROI自动化（业务决策引擎）
# > 目标：从“模型分数”变成“是否值得推”

# ### Step 1：单产品ROI

# ```python
# product_roi = (
#     eval_df
#     .groupby("product_id")
#     .apply(lambda x: x["cate"].mean() / cost_per_product[x.name])
#     .reset_index(name="roi")
# )
# ```

# ### Step 2：客户级推荐价值

# ```python
# value_df = recommend_df.merge(product_roi, on="product_id")
# value_df["expected_value"] = value_df["cate"] * value_df["roi"]
# ```

# ### Step 3：生成最终推荐清单

# ```python
# final_reco = (
#     value_df
#     .sort_values("expected_value", ascending=False)
#     .groupby("cust_id")
#     .head(3)
# )
# ```

# 👉 输出给业务：
# - 每个客户最值得推荐的TopN产品
# - 对应预期收益
# - 对应营销成本
# - 对应ROI

# ---

# # 最终：完整闭环

# ```
# 因果建模
#    ↓
# CATE估计
#    ↓
# Qini评估排序能力
#    ↓
# Empirical Uplift验证真实性
#    ↓
# Policy Gain决定触达比例
#    ↓
# Temporal Stability保证长期有效
#    ↓
# ROI Engine决定是否值得推
#    ↓
# 业务投放
# ```

# ---

# 如果你还要继续进阶，可以加：
# - 多产品预算约束优化（Knapsack）
# - 多目标优化（增量 vs 成本 vs 客户体验）
# - 实时推荐（流式特征 + 在线推断）
