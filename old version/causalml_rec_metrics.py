from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# 配置区
# ============================================================

@dataclass
class ProductDecisionConfig:
    """
    产品层自动决策阈值配置
    """
    min_ate: float = 0.0
    min_qini: float = 0.0
    min_auuc: float = 0.0
    min_top_uplift_lift: float = 0.0
    max_negative_uplift_ratio: float = 0.50
    min_recommendable_customers: int = 100
    min_support_samples: int = 300
    top_ratio: float = 0.20
    use_bootstrap_significance: bool = False
    bootstrap_rounds: int = 200
    random_state: int = 42


@dataclass
class CustomerDecisionConfig:
    """
    客户层推荐配置
    """
    min_cate: float = 0.0
    top_k_per_customer: int = 3
    min_product_pass_rate: float = 0.0
    customer_weight_col: Optional[str] = None


@dataclass
class SafetyConfig:
    """
    推荐安全阈值
    """
    max_customer_negative_share: float = 0.5
    min_customer_expected_gain: float = 0.0
    enable_product_blacklist_gate: bool = True
    enable_customer_safe_filter: bool = True


@dataclass
class BusinessConfig:
    """
    业务层接口，占位使用。
    当前你说可以先忽略业务层，因此默认不参与最终门禁。
    """
    value_per_unit_y: Optional[float] = None
    cost_per_recommendation: Optional[float] = None
    fixed_cost_per_product: Optional[float] = None


# ============================================================
# 工具函数
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


def _rank_desc(series: pd.Series) -> pd.Series:
    return series.rank(method="first", ascending=False)


# ============================================================
# 评估基础指标
# ============================================================

def compute_ate_by_product(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    在你当前只有 cate/T/Y 的前提下，ATE 使用 cate 的样本均值作为估计。
    """
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
    out["ate_tstat_proxy"] = out["ate"] / np.where(
        out["cate_std"] <= 1e-12,
        np.nan,
        out["cate_std"] / np.sqrt(out["sample_size"].clip(lower=1))
    )
    out["ate_tstat_proxy"] = out["ate_tstat_proxy"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def compute_top_segment_metrics(
    eval_df: pd.DataFrame,
    top_ratio: float = 0.20
) -> pd.DataFrame:
    """
    评估 Top 人群 uplift 是否显著更高：
    - top_mean_cate
    - rest_mean_cate
    - uplift_lift = top_mean - overall_mean
    - top_vs_rest_gap = top_mean - rest_mean
    """
    frames = []
    for product_id, g in eval_df.groupby("product_id"):
        g = g.sort_values("cate", ascending=False).copy()
        n = len(g)
        top_n = max(1, int(np.ceil(n * top_ratio)))
        top_g = g.iloc[:top_n]
        rest_g = g.iloc[top_n:] if top_n < n else g.iloc[:0]

        overall_mean = g["cate"].mean()
        top_mean = top_g["cate"].mean()
        rest_mean = rest_g["cate"].mean() if len(rest_g) > 0 else np.nan

        frames.append(
            {
                "product_id": product_id,
                "top_n": top_n,
                "overall_mean_cate": overall_mean,
                "top_mean_cate": top_mean,
                "rest_mean_cate": rest_mean if pd.notna(rest_mean) else top_mean,
                "top_uplift_lift": top_mean - overall_mean,
                "top_vs_rest_gap": top_mean - (rest_mean if pd.notna(rest_mean) else 0.0),
            }
        )
    return pd.DataFrame(frames)


def compute_negative_uplift_metrics(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    负 uplift 风险控制：
    - overall_negative_uplift_ratio
    - treated_negative_uplift_ratio
    - expected_negative_mass
    """
    frames = []
    for product_id, g in eval_df.groupby("product_id"):
        neg_mask = g["cate"] < 0
        treated_mask = g["T"] == 1

        frames.append(
            {
                "product_id": product_id,
                "negative_uplift_ratio": neg_mask.mean(),
                "treated_negative_uplift_ratio": (
                    (neg_mask & treated_mask).sum() / treated_mask.sum()
                    if treated_mask.sum() > 0 else 0.0
                ),
                "expected_negative_mass": g.loc[neg_mask, "cate"].sum(),
                "expected_positive_mass": g.loc[g["cate"] > 0, "cate"].sum(),
            }
        )
    return pd.DataFrame(frames)


def compute_qini_auuc_proxy(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    由于你当前输入只有 eval_df，而没有单独的 uplift 曲线原始中间输出，
    这里实现一个“可自动化接入”的 proxy 版本，作为排序有效性的统一接口。

    如果你已有 causalml 真实 qini/auuc 结果，建议直接通过外部结果 merge 覆盖。
    """
    frames = []
    for product_id, g in eval_df.groupby("product_id"):
        g = g.sort_values("cate", ascending=False).copy()
        n = len(g)
        if n == 0:
            continue

        g["rank_pct"] = (np.arange(1, n + 1)) / n
        g["cum_gain_proxy"] = g["cate"].cumsum()
        auuc_proxy = g["cum_gain_proxy"].mean()
        random_baseline = g["cate"].mean() * (n + 1) / 2.0
        qini_proxy = auuc_proxy - random_baseline

        frames.append(
            {
                "product_id": product_id,
                "auuc": float(auuc_proxy),
                "qini": float(qini_proxy),
                "auuc_baseline": float(random_baseline),
                "qini_vs_baseline": float(qini_proxy),
            }
        )
    return pd.DataFrame(frames)


def compute_recommendable_population(
    eval_df: pd.DataFrame,
    min_cate: float = 0.0
) -> pd.DataFrame:
    out = (
        eval_df.assign(is_recommendable=(eval_df["cate"] > min_cate).astype(int))
        .groupby("product_id", as_index=False)
        .agg(
            recommendable_customers=("is_recommendable", "sum"),
            recommendable_ratio=("is_recommendable", "mean"),
        )
    )
    return out


# ============================================================
# 可选：Bootstrap 显著性
# ============================================================

def bootstrap_top_uplift_significance(
    eval_df: pd.DataFrame,
    top_ratio: float,
    bootstrap_rounds: int = 200,
    random_state: int = 42
) -> pd.DataFrame:
    rng = np.random.RandomState(random_state)
    results = []

    for product_id, g in eval_df.groupby("product_id"):
        values = g["cate"].dropna().values
        n = len(values)
        if n <= 5:
            results.append(
                {
                    "product_id": product_id,
                    "top_uplift_pvalue": np.nan,
                    "top_uplift_ci_low": np.nan,
                    "top_uplift_ci_high": np.nan,
                }
            )
            continue

        diffs = []
        top_n = max(1, int(np.ceil(n * top_ratio)))

        for _ in range(bootstrap_rounds):
            sample = rng.choice(values, size=n, replace=True)
            sample = np.sort(sample)[::-1]
            top_mean = sample[:top_n].mean()
            overall_mean = sample.mean()
            diffs.append(top_mean - overall_mean)

        diffs = np.array(diffs)
        pvalue = np.mean(diffs <= 0)
        ci_low = np.quantile(diffs, 0.025)
        ci_high = np.quantile(diffs, 0.975)

        results.append(
            {
                "product_id": product_id,
                "top_uplift_pvalue": float(pvalue),
                "top_uplift_ci_low": float(ci_low),
                "top_uplift_ci_high": float(ci_high),
            }
        )

    return pd.DataFrame(results)


# ============================================================
# 产品层自动决策
# ============================================================

def evaluate_products(
    eval_df: pd.DataFrame,
    product_config: Optional[ProductDecisionConfig] = None,
    external_metrics_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    external_metrics_df 可选传入已有的真实指标，至少包含：
    product_id, qini, auuc
    若传入则优先覆盖 proxy 值。
    """
    validate_eval_df(eval_df)
    config = product_config or ProductDecisionConfig()

    ate_df = compute_ate_by_product(eval_df)
    top_df = compute_top_segment_metrics(eval_df, top_ratio=config.top_ratio)
    neg_df = compute_negative_uplift_metrics(eval_df)
    rank_df = compute_qini_auuc_proxy(eval_df)
    pop_df = compute_recommendable_population(eval_df, min_cate=0.0)

    product_eval = (
        ate_df
        .merge(top_df, on="product_id", how="left")
        .merge(neg_df, on="product_id", how="left")
        .merge(rank_df, on="product_id", how="left")
        .merge(pop_df, on="product_id", how="left")
    )

    if external_metrics_df is not None and not external_metrics_df.empty:
        cols = [c for c in ["product_id", "qini", "auuc"] if c in external_metrics_df.columns]
        if "product_id" in cols:
            product_eval = product_eval.drop(columns=[c for c in ["qini", "auuc"] if c in product_eval.columns])
            product_eval = product_eval.merge(external_metrics_df[cols], on="product_id", how="left")

    if config.use_bootstrap_significance:
        sig_df = bootstrap_top_uplift_significance(
            eval_df=eval_df,
            top_ratio=config.top_ratio,
            bootstrap_rounds=config.bootstrap_rounds,
            random_state=config.random_state,
        )
        product_eval = product_eval.merge(sig_df, on="product_id", how="left")
    else:
        product_eval["top_uplift_pvalue"] = np.nan
        product_eval["top_uplift_ci_low"] = np.nan
        product_eval["top_uplift_ci_high"] = np.nan

    product_eval["pass_ate"] = product_eval["ate"] > config.min_ate
    product_eval["pass_qini"] = product_eval["qini"] > config.min_qini
    product_eval["pass_auuc"] = product_eval["auuc"] > config.min_auuc
    product_eval["pass_top_lift"] = product_eval["top_uplift_lift"] > config.min_top_uplift_lift
    product_eval["pass_negative_risk"] = (
        product_eval["negative_uplift_ratio"] <= config.max_negative_uplift_ratio
    )
    product_eval["pass_population"] = (
        product_eval["recommendable_customers"] >= config.min_recommendable_customers
    )
    product_eval["pass_support"] = (
        product_eval["sample_size"] >= config.min_support_samples
    )

    gate_cols = [
        "pass_ate",
        "pass_qini",
        "pass_auuc",
        "pass_top_lift",
        "pass_negative_risk",
        "pass_population",
        "pass_support",
    ]
    product_eval["pass_rate"] = product_eval[gate_cols].mean(axis=1)

    product_eval["recommendation_decision"] = np.where(
        product_eval[gate_cols].all(axis=1),
        "recommend",
        np.where(product_eval["pass_ate"], "watchlist", "reject")
    )

    product_eval["product_score"] = (
        0.25 * _normalize_score(product_eval["ate"]) +
        0.20 * _normalize_score(product_eval["qini"]) +
        0.20 * _normalize_score(product_eval["auuc"]) +
        0.15 * _normalize_score(product_eval["top_uplift_lift"]) +
        0.10 * _normalize_score(product_eval["recommendable_ratio"]) +
        0.10 * (1 - _normalize_score(product_eval["negative_uplift_ratio"]))
    )

    product_eval = product_eval.sort_values(
        ["recommendation_decision", "product_score", "ate"],
        ascending=[True, False, False]
    ).reset_index(drop=True)

    return product_eval


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

    pass_products = product_eval_df.copy()
    if "pass_rate" not in pass_products.columns:
        raise ValueError("product_eval_df 缺少 pass_rate 字段，请先执行 evaluate_products")

    candidate_df = eval_df.merge(
        pass_products[["product_id", "recommendation_decision", "pass_rate", "product_score", "negative_uplift_ratio"]],
        on="product_id",
        how="left"
    )

    candidate_df["product_is_approved"] = candidate_df["recommendation_decision"].eq("recommend")
    candidate_df["pass_product_threshold"] = candidate_df["pass_rate"] >= customer_config.min_product_pass_rate
    candidate_df["pass_cate_threshold"] = candidate_df["cate"] > customer_config.min_cate

    if safety_config.enable_product_blacklist_gate:
        candidate_df = candidate_df[
            candidate_df["product_is_approved"] & candidate_df["pass_product_threshold"]
        ].copy()

    candidate_df["recommend_score"] = (
        0.60 * _normalize_score(candidate_df["cate"]) +
        0.25 * _normalize_score(candidate_df["product_score"]) +
        0.15 * (1 - _normalize_score(candidate_df["negative_uplift_ratio"]))
    )

    if customer_config.customer_weight_col and customer_config.customer_weight_col in candidate_df.columns:
        candidate_df["recommend_score"] = (
            candidate_df["recommend_score"] * candidate_df[customer_config.customer_weight_col].fillna(1.0)
        )

    candidate_df["rank_in_customer"] = candidate_df.groupby("cust_id")["recommend_score"].rank(
        method="first", ascending=False
    )

    customer_reco = candidate_df[
        candidate_df["pass_cate_threshold"] &
        (candidate_df["rank_in_customer"] <= customer_config.top_k_per_customer)
    ].copy()

    if safety_config.enable_customer_safe_filter:
        customer_reco["is_safe_recommendation"] = (
            (customer_reco["cate"] >= safety_config.min_customer_expected_gain) &
            (customer_reco["negative_uplift_ratio"] <= safety_config.max_customer_negative_share)
        )
        customer_reco = customer_reco[customer_reco["is_safe_recommendation"]].copy()
    else:
        customer_reco["is_safe_recommendation"] = True

    return customer_reco.sort_values(["cust_id", "rank_in_customer"]).reset_index(drop=True)


# ============================================================
# 推荐安全评估
# ============================================================

def evaluate_recommendation_safety(customer_reco_df: pd.DataFrame) -> pd.DataFrame:
    if customer_reco_df.empty:
        return pd.DataFrame(
            [{
                "n_customer_recommended": 0,
                "n_recommendation_pairs": 0,
                "avg_expected_uplift": 0.0,
                "median_expected_uplift": 0.0,
                "negative_risk_share": 0.0,
                "safe_recommendation_share": 0.0,
            }]
        )

    out = pd.DataFrame(
        [{
            "n_customer_recommended": customer_reco_df["cust_id"].nunique(),
            "n_recommendation_pairs": len(customer_reco_df),
            "avg_expected_uplift": customer_reco_df["cate"].mean(),
            "median_expected_uplift": customer_reco_df["cate"].median(),
            "negative_risk_share": (customer_reco_df["cate"] < 0).mean(),
            "safe_recommendation_share": customer_reco_df["is_safe_recommendation"].mean()
            if "is_safe_recommendation" in customer_reco_df.columns else np.nan,
        }]
    )
    return out


# ============================================================
# 业务层占位接口
# ============================================================

def evaluate_business_value(
    customer_reco_df: pd.DataFrame,
    business_config: Optional[BusinessConfig] = None
) -> pd.DataFrame:
    """
    当前作为占位接口：
    若未来你补充单位存款价值、单次触达成本、产品固定成本等信息，
    可以直接在这里计算增量收益和 ROI。
    """
    cfg = business_config or BusinessConfig()

    total_expected_uplift = customer_reco_df["cate"].sum() if not customer_reco_df.empty else 0.0
    n_pairs = len(customer_reco_df)

    estimated_revenue = (
        total_expected_uplift * cfg.value_per_unit_y
        if cfg.value_per_unit_y is not None else np.nan
    )

    estimated_cost = (
        n_pairs * cfg.cost_per_recommendation
        if cfg.cost_per_recommendation is not None else np.nan
    )

    if (
        pd.notna(estimated_revenue)
        and pd.notna(estimated_cost)
        and estimated_cost != 0
    ):
        roi = (estimated_revenue - estimated_cost) / estimated_cost
    else:
        roi = np.nan

    return pd.DataFrame(
        [{
            "total_expected_uplift": total_expected_uplift,
            "estimated_revenue": estimated_revenue,
            "estimated_cost": estimated_cost,
            "roi": roi,
        }]
    )


# ============================================================
# 一站式主流程
# ============================================================

def run_causal_recommendation_pipeline(
    eval_df: pd.DataFrame,
    external_metrics_df: Optional[pd.DataFrame] = None,
    product_config: Optional[ProductDecisionConfig] = None,
    customer_config: Optional[CustomerDecisionConfig] = None,
    safety_config: Optional[SafetyConfig] = None,
    business_config: Optional[BusinessConfig] = None,
) -> Dict[str, pd.DataFrame]:
    """
    输入：
        eval_df: 长表，至少包含
            cust_id, product_id, date, cate, T, Y
        external_metrics_df:
            可选，若你已有 causalml 真实 qini/auuc，可传：
            product_id, qini, auuc

    输出：
        product_eval_df: 产品层评估与自动决策
        customer_reco_df: 客户层推荐结果
        safety_summary_df: 推荐安全汇总
        business_summary_df: 业务层汇总（占位）
    """
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

    safety_summary_df = evaluate_recommendation_safety(customer_reco_df)
    business_summary_df = evaluate_business_value(customer_reco_df, business_config)

    return {
        "product_eval_df": product_eval_df,
        "customer_reco_df": customer_reco_df,
        "safety_summary_df": safety_summary_df,
        "business_summary_df": business_summary_df,
    }


# ============================================================
# 结果解释辅助
# ============================================================

def generate_product_reason_tags(product_eval_df: pd.DataFrame) -> pd.DataFrame:
    df = product_eval_df.copy()

    def _reason(row: pd.Series) -> str:
        tags: List[str] = []
        if row.get("pass_ate", False):
            tags.append("ATE正向")
        else:
            tags.append("ATE不足")

        if row.get("pass_qini", False) and row.get("pass_auuc", False):
            tags.append("排序有效")
        else:
            tags.append("排序待验证")

        if row.get("pass_top_lift", False):
            tags.append("Top人群精准")
        else:
            tags.append("Top人群优势不明显")

        if row.get("pass_negative_risk", False):
            tags.append("负uplift风险可控")
        else:
            tags.append("负uplift偏高")

        if row.get("pass_population", False):
            tags.append("可推荐客群充足")
        else:
            tags.append("客群规模偏小")

        return "|".join(tags)

    df["decision_reason_tags"] = df.apply(_reason, axis=1)
    return df


def summarize_framework_definition() -> Dict[str, List[str]]:
    """
    输出体系定义，便于你落文档/汇报。
    """
    return {
        "product_layer": [
            "ATE > 0：产品平均因果效应为正，方向正确",
            "Qini / AUUC > baseline：uplift 排序能力有效",
            "Top人群 uplift 显著更高：精准识别有效客群",
            "负 uplift 比例可控：降低误推风险",
            "样本量与可推荐客群规模达标：保证可执行性",
        ],
        "customer_layer": [
            "客户-产品级 CATE > 0：仅对预测有正向增量的客户推荐",
            "按 recommend_score 做 Top-K 排序：优先推荐预期收益最高的产品",
            "支持客户权重加权：例如按AUM、分层价值、客户等级加权",
        ],
        "safety_layer": [
            "产品级黑名单门禁：拒绝高负uplift风险产品进入推荐池",
            "客户级安全过滤：过滤低预期收益或高风险推荐对",
            "推荐后输出安全汇总：覆盖规模、平均uplift、负风险占比",
        ],
        "business_layer": [
            "总增量收益：sum(cate)",
            "ROI：未来可接入单位收益、触达成本、固定资源成本后计算",
        ],
    }


if __name__ == "__main__":
    # 示例：你后续在工程中读取自己的 eval_df 后可直接调用
    # eval_df = pd.read_parquet("your_eval_df.parquet")
    # metrics_df = pd.read_csv("product_metrics.csv")  # 可选，包含真实 qini/auuc
    #
    # result = run_causal_recommendation_pipeline(
    #     eval_df=eval_df,
    #     external_metrics_df=metrics_df,
    #     product_config=ProductDecisionConfig(
    #         min_ate=0.0,
    #         min_qini=0.0,
    #         min_auuc=0.0,
    #         min_top_uplift_lift=0.0,
    #         max_negative_uplift_ratio=0.4,
    #         min_recommendable_customers=100,
    #         min_support_samples=300,
    #         top_ratio=0.2,
    #     ),
    #     customer_config=CustomerDecisionConfig(
    #         min_cate=0.0,
    #         top_k_per_customer=3,
    #         customer_weight_col=None,
    #     ),
    #     safety_config=SafetyConfig(
    #         max_customer_negative_share=0.4,
    #         min_customer_expected_gain=0.0,
    #     ),
    # )
    #
    # print(result["product_eval_df"].head())
    # print(result["customer_reco_df"].head())
    # print(result["safety_summary_df"])
    # print(result["business_summary_df"])
    pass
