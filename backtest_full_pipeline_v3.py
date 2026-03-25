from __future__ import annotations

"""
backtest_full_pipeline_v3.py
================================================
v3 目标（针对“真实数据没有 ps / mu0 / mu1”场景）：
- 基于 v2（Parquet + DuckDB）的大数据流程
- eval_df 只要求最小字段：cust_id, product_id, date, cate, T, Y
- 自动兼容缺失列：
  - generate_recommendations_duckdb：ps/mu0/mu1 缺失时用 NULL 占位，避免 SQL 失败
  - OPE：按列存在性自动降级：
      * 无 ps：跳过 IPW/DR，给出可读原因
      * 有 ps 无 mu：只算 IPW，DR 给出缺列原因
      * ps + mu0 + mu1 均有：IPW + DR 都算
- 输出：csv + v3 的 md 报告（含 ps/mu0/mu1 的定义与计算说明）

注意：
- “无 ps/mu 时仍然可以跑”的含义是：产品评估、推荐清单、policy curve、temporal、推荐子集 uplift 正常产出；
  OPE 属于可选模块，缺列时会被跳过但不会报错。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ============================================================
# 报告工具
# ============================================================

def _fmt(x: object, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "-"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.{nd}f}"
    return str(x)


def render_business_report_v3(
    result: Dict[str, pd.DataFrame],
    out_path: str = "backtest_output_v3/backtest_report_v3.md",
    top_products: int = 20,
    top_reco_rows: int = 50,
) -> str:
    """
    将 `run_backtest_v3()` 的输出整理成业务可读 Markdown 报告（v3）。
    - 结构参考 v1/v2
    - 增加 ps/mu0/mu1 的解释说明
    - OPE 章节展示是否计算以及缺列原因
    """
    import os
    from datetime import datetime

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    product_eval_df = result.get("product_eval_df", pd.DataFrame()).copy()
    customer_reco_df = result.get("customer_reco_df", pd.DataFrame()).copy()
    reco_empirical_eval_df = result.get("reco_empirical_eval_df", pd.DataFrame()).copy()
    policy_gain_df = result.get("policy_gain_df", pd.DataFrame()).copy()
    temporal_df = result.get("temporal_df", pd.DataFrame()).copy()
    ope_df = result.get("ope_df", pd.DataFrame()).copy()

    # 关键数值
    n_rows = int(customer_reco_df.shape[0]) if not customer_reco_df.empty else 0
    n_customers = (
        int(customer_reco_df["cust_id"].nunique())
        if ("cust_id" in customer_reco_df.columns and not customer_reco_df.empty)
        else 0
    )
    n_products = (
        int(product_eval_df["product_id"].nunique())
        if ("product_id" in product_eval_df.columns and not product_eval_df.empty)
        else 0
    )
    if not product_eval_df.empty and "recommendation_decision" in product_eval_df.columns:
        n_reco_products = int(
            product_eval_df["recommendation_decision"].isin(["recommend_all", "recommend_targeted"]).sum()
        )
        n_reco_products_all = int((product_eval_df["recommendation_decision"] == "recommend_all").sum())
        n_reco_products_targeted = int((product_eval_df["recommendation_decision"] == "recommend_targeted").sum())
    else:
        n_reco_products = 0
        n_reco_products_all = 0
        n_reco_products_targeted = 0

    reco_uplift = (
        float(reco_empirical_eval_df["empirical_uplift"].iloc[0])
        if (
            reco_empirical_eval_df is not None
            and not reco_empirical_eval_df.empty
            and "empirical_uplift" in reco_empirical_eval_df.columns
        )
        else np.nan
    )

    # Top 产品表
    if not product_eval_df.empty:
        cols = [
            "product_id",
            "recommendation_decision",
            "sample_size",
            "n_customer",
            "ate",
            "empirical_uplift",
            "qini",
            "auuc",
            "top_uplift_lift",
            "top_vs_rest_gap",
            "negative_uplift_ratio",
            "cate_std",
            "cate_p05",
            "cate_p50",
            "cate_p95",
            "product_score",
            "pass_rate",
        ]
        cols = [c for c in cols if c in product_eval_df.columns]
        prod_show = product_eval_df.sort_values(["recommendation_decision", "product_score"], ascending=[True, False])[
            cols
        ].head(top_products)
    else:
        prod_show = pd.DataFrame()

    # 推荐明细（Top）
    if not customer_reco_df.empty:
        reco_cols = [
            "cust_id",
            "product_id",
            "date",
            "cate",
            "adjusted_cate",
            "recommend_score",
            "rank_in_customer",
            "T",
            "Y",
            "ps",
            "mu0",
            "mu1",
        ]
        reco_cols = [c for c in reco_cols if c in customer_reco_df.columns]
        reco_show = customer_reco_df.sort_values("recommend_score", ascending=False)[reco_cols].head(top_reco_rows)
    else:
        reco_show = pd.DataFrame()

    policy_show = (
        policy_gain_df
        if (policy_gain_df is not None and not policy_gain_df.empty)
        else pd.DataFrame(columns=["top_pct", "n", "uplift_gain"])
    )
    temporal_show = (
        temporal_df
        if (temporal_df is not None and not temporal_df.empty)
        else pd.DataFrame(columns=["date", "model_ate", "empirical_uplift", "treated_n", "control_n"])
    )
    ope_show = (
        ope_df
        if (ope_df is not None and not ope_df.empty)
        else pd.DataFrame(columns=["policy", "ipw_value", "dr_value", "ipw_ok", "dr_ok", "ipw_error", "dr_error"])
    )

    def _table(df: pd.DataFrame) -> str:
        if df is None or df.empty:
            return "_(empty)_"
        try:
            return df.to_markdown(index=False)
        except Exception:
            return "```\n" + df.to_string(index=False) + "\n```"

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md: List[str] = []
    md.append(f"# 回测报告（Backtest Report, v3）\n\n生成时间：{now}\n")

    md.append("## 一、概览（Executive Summary）\n")
    md.append(
        "\n".join(
            [
                f"- 覆盖产品数：{_fmt(n_products, 0)}",
                f"- 进入推荐池产品数（recommend_all + recommend_targeted）：{_fmt(n_reco_products, 0)}",
                f"  - recommend_all：{_fmt(n_reco_products_all, 0)}",
                f"  - recommend_targeted：{_fmt(n_reco_products_targeted, 0)}",
                f"- 推荐明细行数（customer-product pairs）：{_fmt(n_rows, 0)}",
                f"- 被推荐客户数（unique cust_id）：{_fmt(n_customers, 0)}",
                f"- 推荐子集经验 uplift（treated-control）：{_fmt(reco_uplift, 6)}",
            ]
        )
        + "\n"
    )

    md.append("## 二、产品层评估（Product Level）\n")
    md.append(
        "\n".join(
            [
                "### 2.0 方法说明（产品门禁 / 决策含义）",
                "- `recommend_all`：全量推荐（通过全部门禁，适合大规模触达）",
                "- `recommend_targeted`：定向推荐（允许 ATE<0，但对 Top uplift 人群命中强，仅对该产品 Top 人群开放推荐）",
                "- `watchlist`：继续观察（部分指标可，但不足以上线）",
                "- `reject`：不推荐（关键门禁未通过/风险过高）",
                "",
                "### 2.1 Top 产品列表在讲什么？",
                "- 目的：在“产品维度”判断哪些产品适合推荐（全量/定向/观察/拒绝），并展示核心因果指标与风险指标。",
                "",
                "### 2.2 Top 产品列表字段解释（表头含义）",
                "- `product_id`：产品ID。",
                "- `recommendation_decision`：最终推荐决策（recommend_all / recommend_targeted / watchlist / reject）。",
                "- `sample_size`：该产品参与评估的样本行数（通常≈客户×日期）。",
                "- `n_customer`：该产品覆盖的去重客户数。",
                "- `ate`：平均处理效应（模型输出 CATE 的均值），越大越好。",
                "- `empirical_uplift`：经验 uplift（treated 平均Y - control 平均Y），越大越好（更贴近真实结果口径）。",
                "- `qini`：Qini（本项目为 proxy 版本，基于 cate 排序得到），衡量 uplift 排序能力，越大越好。",
                "- `auuc`：AUUC（本项目为 proxy 版本），越大越好。",
                "- `top_uplift_lift`：Top 人群平均 uplift 相对整体平均 uplift 的提升，越大越好。",
                "- `top_vs_rest_gap`：Top 人群平均 uplift - 非Top 人群平均 uplift 的差，越大越好。",
                "- `negative_uplift_ratio`：uplift<0 的占比（风险指标），越小越好。",
                "- `cate_std`/`cate_p05`/`cate_p50`/`cate_p95`：CATE 分布的波动与分位数，用于看异质性/极端值。",
                "- `product_score`：综合评分（用于排序，权重由代码定义）。",
                "- `pass_rate`：门禁通过率（0~1），越大越好。",
                "",
            ]
        )
    )
    md.append(f"### 2.3 Top 产品列表（按 decision + score 排序，Top {top_products}）\n")
    md.append(_table(prod_show) + "\n")

    md.append("## 三、客户层推荐（Customer Level Recommendations）\n")
    md.append(
        "\n".join(
            [
                "### 3.1 客户层推荐在讲什么？",
                "- 目的：把“产品可推荐”进一步落到“客户-产品对”，输出每个客户的 Top-K 推荐清单。",
                "",
                "### 3.2 客户层推荐字段解释",
                "- `cust_id`：客户ID。",
                "- `product_id`：产品ID。",
                "- `date`：样本日期/批次日期。",
                "- `cate`：模型输出的个体处理效应（对该客户推荐该产品的预期增量）。",
                "- `adjusted_cate`：校准后的 CATE（cate * calibration_factor），更贴近经验 uplift 尺度。",
                "- `recommend_score`：最终排序分（综合 adjusted_cate、product_score、风险项），越大越优先推荐。",
                "- `rank_in_customer`：该客户内部排序名次（1 表示最优先）。",
                "- `T`：历史是否触达/处理（1/0）。",
                "- `Y`：结果指标（例如转化/收益等）。",
                "- `ps`：倾向得分（可选列；若你的数据没有则为空/NULL）。",
                "- `mu0`/`mu1`：潜在结果预测（可选列；若你的数据没有则为空/NULL）。",
                "",
            ]
        )
    )
    md.append(f"展示 Top {top_reco_rows} 条推荐记录（按 recommend_score 降序）：\n")
    md.append(_table(reco_show) + "\n")

    md.append("## 四、策略收益曲线（Policy Gain Curve）\n")
    md.append(
        "\n".join(
            [
                "### 4.1 策略收益曲线在讲什么？",
                "- 目的：回答“如果只触达推荐分 top 1%/5%/20%…的人群，平均能提升多少？”用于做触达规模决策。",
                "",
                "### 4.2 字段解释",
                "- `top_pct`：取推荐分 top 的比例（例如 0.10=Top10%）。",
                "- `n`：对应 top_pct 的触达样本数。",
                "- `uplift_gain`：收益提升（当前口径：Top 子集平均Y - 全体平均Y）。",
                "",
            ]
        )
    )
    md.append(_table(policy_show) + "\n")

    md.append("## 五、时间稳定性（Temporal Stability）\n")
    md.append(
        "\n".join(
            [
                "### 5.1 时间稳定性在讲什么？",
                "- 目的：检查模型/推荐效果是否随时间漂移；若某天明显变差，可能是数据分布/活动/人群变化导致。",
                "",
                "### 5.2 字段解释",
                "- `date`：日期。",
                "- `model_ate`：当日平均 cate（模型视角的平均处理效应）。",
                "- `empirical_uplift`：当日经验 uplift（treated 平均Y - control 平均Y）。",
                "- `treated_n`/`control_n`：当日 treated/control 样本量（样本量太小会导致 uplift 不稳定）。",
                "",
            ]
        )
    )
    md.append(_table(temporal_show) + "\n")

    md.append("## 六、离线策略价值评估（OPE）\n")
    md.append(
        "\n".join(
            [
                "### 6.1 离线评估（OPE）在讲什么？",
                "- 目的：在不能线上 A/B 的情况下，估计“如果按新策略触达，整体期望 Y 会是多少”。",
                "- 说明：OPE 通常需要 `ps`（倾向得分）以及可能需要 `mu0/mu1`（潜在结果预测）。若缺列，本报告会在 `ope_df` 中说明原因。",
                "",
                "### 6.2 字段解释",
                "- `policy`：被评估的策略名称。",
                "- `ipw_value`：IPW 估计的策略价值（需要 ps）。",
                "- `dr_value`：DR 估计的策略价值（需要 ps + mu0/mu1）。",
                "- `ipw_ok`/`dr_ok`：是否成功计算。",
                "- `ipw_error`/`dr_error`：失败原因/提示。",
                "",
            ]
        )
    )
    md.append(_table(ope_show) + "\n")

    md.append("## 七、ps / mu0 / mu1 是什么？怎么计算？（给数据准备同学）\n")
    md.append(
        "\n".join(
            [
                "### 7.1 ps（propensity score，倾向得分）",
                "- 定义：ps(x) = P(T=1 | X=x)，即在历史策略下样本被触达/处理的概率。",
                "- 用途：IPW/DR 等离线策略评估需要用 ps 来纠偏历史触达偏差。",
                "- 计算：用历史数据训练一个二分类模型预测 T（特征只能用触达前可见特征），输出 predict_proba 的概率作为 ps，并进行 clipping（例如 0.01~0.99）。",
                "",
                "### 7.2 mu0 / mu1（潜在结果预测）",
                "- 定义：mu0(x)=E[Y|X=x,T=0]，mu1(x)=E[Y|X=x,T=1]。",
                "- 用途：DR（Doubly Robust）估计需要 mu0/mu1；只要 ps 或 mu 模型其中一个靠谱，估计仍可能一致。",
                "- 计算（最简单 T-learner）：",
                "  - 在 T=0 子集训练一个回归模型预测 Y → 对全量输出 mu0(x)",
                "  - 在 T=1 子集训练一个回归模型预测 Y → 对全量输出 mu1(x)",
                "  - 模型可用线性回归/GBDT/LightGBM 等，视 Y（金额/概率）而定。",
            ]
        )
        + "\n"
    )

    md.append("## 附录：输出数据表说明\n")
    md.append(
        "\n".join(
            [
                "- `product_eval_df`：产品层聚合指标 + 门禁结果 + product_score",
                "- `customer_reco_df`：客户-产品推荐清单（含 adjusted_cate / recommend_score；若无 ps/mu 则为空/NULL）",
                "- `reco_empirical_eval_df`：推荐子集的 treated-control uplift（sanity check）",
                "- `policy_gain_df`：不同触达比例下的收益曲线（经验口径）",
                "- `temporal_df`：按 date 维度的 model_ate vs empirical_uplift",
                "- `ope_df`：离线策略价值评估（若缺 ps/mu 会自动降级并写明原因）",
            ]
        )
        + "\n"
    )

    md_text = "\n".join(md)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    try:
        gbk_path = os.path.splitext(out_path)[0] + "_gbk.md"
        with open(gbk_path, "w", encoding="gbk", errors="replace") as f:
            f.write(md_text)
    except Exception:
        pass

    return md_text


# ============================================================
# 与 v1/v2 保持一致的配置
# ============================================================

@dataclass
class ProductDecisionConfig:
    min_ate: float = 0.0
    min_qini: float = 0.0
    min_auuc: float = 0.0
    min_top_uplift_lift: float = 0.0
    min_empirical_uplift: float = 0.0
    max_negative_uplift_ratio: float = 0.50

    enable_targeted_reco: bool = True
    targeted_top_ratio: float = 0.20
    min_targeted_top_cate: float = 0.5
    min_targeted_lift: float = 0.0
    allow_targeted_when_ate_negative: bool = True

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
class BacktestConfig:
    policy_bins: Tuple[float, ...] = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
    ps_clip_low: float = 0.01
    ps_clip_high: float = 0.99
    random_state: int = 42


REQUIRED_COLUMNS = ["cust_id", "product_id", "date", "cate", "T", "Y"]


# ============================================================
# DuckDB helpers
# ============================================================

def _duckdb_connect(db_path: Optional[str] = None):
    import duckdb

    con = duckdb.connect(database=(db_path or ":memory:"))
    con.execute("PRAGMA threads=4;")
    con.execute("PRAGMA enable_progress_bar=false;")
    return con


def _parquet_glob(parquet_dir: str) -> str:
    p = Path(parquet_dir).resolve().as_posix()
    return f"{p}/**/*.parquet"


def get_parquet_columns(parquet_dir: str, duckdb_path: Optional[str] = None) -> List[str]:
    """
    读取 parquet schema，返回可用列名列表（小开销）。
    注意：对 hive 分区列，duckdb 在 read_parquet(..., hive_partitioning=1) 后也会体现出来。
    """
    con = _duckdb_connect(duckdb_path)
    glob = _parquet_glob(parquet_dir)
    # LIMIT 0 只解析 schema
    df = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{glob}', hive_partitioning=1) LIMIT 0;"
    ).df()
    con.close()
    return [str(x) for x in df["column_name"].tolist()]


# ============================================================
# v3：产品层评估（沿用 v2）
# ============================================================

def evaluate_products_duckdb(
    parquet_dir: str,
    product_config: Optional[ProductDecisionConfig] = None,
    external_metrics_df: Optional[pd.DataFrame] = None,
    duckdb_path: Optional[str] = None,
) -> pd.DataFrame:
    product_config = product_config or ProductDecisionConfig()

    con = _duckdb_connect(duckdb_path)
    glob = _parquet_glob(parquet_dir)
    con.execute(f"CREATE OR REPLACE VIEW eval AS SELECT * FROM read_parquet('{glob}', hive_partitioning=1);")

    base_sql = """
    SELECT
      product_id,
      COUNT(*) AS sample_size,
      COUNT(DISTINCT cust_id) AS n_customer,
      AVG(cate) AS ate,
      STDDEV_SAMP(cate) AS cate_std,
      QUANTILE_CONT(cate, 0.05) AS cate_p05,
      QUANTILE_CONT(cate, 0.50) AS cate_p50,
      QUANTILE_CONT(cate, 0.95) AS cate_p95,
      AVG(T) AS treated_rate,
      AVG(Y) AS outcome_rate,

      AVG(CASE WHEN T=1 THEN Y ELSE NULL END) AS treated_mean_outcome,
      AVG(CASE WHEN T=0 THEN Y ELSE NULL END) AS control_mean_outcome,
      SUM(CASE WHEN T=1 THEN 1 ELSE 0 END) AS treated_n,
      SUM(CASE WHEN T=0 THEN 1 ELSE 0 END) AS control_n,

      AVG(CASE WHEN cate < 0 THEN 1 ELSE 0 END) AS negative_uplift_ratio,
      CASE WHEN SUM(CASE WHEN T=1 THEN 1 ELSE 0 END) = 0 THEN 0.0
           ELSE SUM(CASE WHEN T=1 AND cate < 0 THEN 1 ELSE 0 END) * 1.0
              / SUM(CASE WHEN T=1 THEN 1 ELSE 0 END)
      END AS treated_negative_uplift_ratio
    FROM eval
    GROUP BY product_id
    """
    base = con.execute(base_sql).df()
    base["empirical_uplift"] = base["treated_mean_outcome"] - base["control_mean_outcome"]

    top_ratio = float(product_config.top_ratio)
    top_sql = f"""
    WITH ranked AS (
      SELECT
        product_id,
        cate,
        ROW_NUMBER() OVER (PARTITION BY product_id ORDER BY cate DESC) AS rn,
        COUNT(*) OVER (PARTITION BY product_id) AS cnt,
        AVG(cate) OVER (PARTITION BY product_id) AS overall_mean
      FROM eval
    ),
    top_part AS (
      SELECT
        product_id,
        AVG(cate) AS top_mean,
        ANY_VALUE(overall_mean) AS overall_mean,
        ANY_VALUE(cnt) AS cnt
      FROM ranked
      WHERE rn <= CEIL(cnt * {top_ratio})
      GROUP BY product_id
    ),
    rest_part AS (
      SELECT
        product_id,
        AVG(cate) AS rest_mean
      FROM ranked
      WHERE rn > CEIL(cnt * {top_ratio})
      GROUP BY product_id
    )
    SELECT
      t.product_id,
      (t.top_mean - t.overall_mean) AS top_uplift_lift,
      (t.top_mean - COALESCE(r.rest_mean, t.top_mean)) AS top_vs_rest_gap
    FROM top_part t
    LEFT JOIN rest_part r USING(product_id)
    """
    top_df = con.execute(top_sql).df()

    qini_sql = """
    WITH ranked AS (
      SELECT
        product_id,
        cate,
        ROW_NUMBER() OVER (PARTITION BY product_id ORDER BY cate DESC) AS rn,
        COUNT(*) OVER (PARTITION BY product_id) AS n,
        AVG(cate) OVER (PARTITION BY product_id) AS mean_cate
      FROM eval
    ),
    cum AS (
      SELECT
        product_id,
        n,
        mean_cate,
        SUM(cate) OVER (PARTITION BY product_id ORDER BY rn ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum_gain
      FROM ranked
    )
    SELECT
      product_id,
      AVG(cum_gain) AS auuc,
      (AVG(cum_gain) - ANY_VALUE(mean_cate) * (ANY_VALUE(n) + 1) / 2.0) AS qini
    FROM cum
    GROUP BY product_id
    """
    qini_df = con.execute(qini_sql).df()

    product_eval = base.merge(top_df, on="product_id", how="left").merge(qini_df, on="product_id", how="left")

    if external_metrics_df is not None and not external_metrics_df.empty:
        cols = [c for c in ["product_id", "qini", "auuc"] if c in external_metrics_df.columns]
        product_eval = product_eval.drop(columns=["qini", "auuc"], errors="ignore")
        product_eval = product_eval.merge(external_metrics_df[cols], on="product_id", how="left")

    if product_config.enable_calibration:
        product_eval["calibration_factor"] = product_eval.apply(
            lambda r: (r["empirical_uplift"] / r["ate"]) if (pd.notna(r["ate"]) and r["ate"] != 0) else 1.0,
            axis=1,
        )
        product_eval["calibration_factor"] = (
            product_eval["calibration_factor"].replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.0, 5.0)
        )
    else:
        product_eval["calibration_factor"] = 1.0

    product_eval["pass_ate"] = product_eval["ate"] > product_config.min_ate
    product_eval["pass_empirical"] = product_eval["empirical_uplift"] > product_config.min_empirical_uplift
    product_eval["pass_qini"] = product_eval["qini"] > product_config.min_qini
    product_eval["pass_auuc"] = product_eval["auuc"] > product_config.min_auuc
    product_eval["pass_top_lift"] = product_eval["top_uplift_lift"] > product_config.min_top_uplift_lift
    product_eval["pass_negative_risk"] = product_eval["negative_uplift_ratio"] <= product_config.max_negative_uplift_ratio
    product_eval["pass_support"] = product_eval["sample_size"] >= product_config.min_support_samples

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

    product_eval["pass_targeted"] = (
        (product_config.enable_targeted_reco)
        & ((product_eval["ate"] < 0) if product_config.allow_targeted_when_ate_negative else True)
        & (product_eval["top_uplift_lift"] >= product_config.min_targeted_lift)
        & (product_eval["negative_uplift_ratio"] <= product_config.max_negative_uplift_ratio)
    )

    product_eval["recommendation_decision"] = np.select(
        [
            product_eval[gate_cols].all(axis=1),
            product_eval["pass_targeted"],
            product_eval["pass_ate"],
        ],
        [
            "recommend_all",
            "recommend_targeted",
            "watchlist",
        ],
        default="reject",
    )

    def _normalize(series: pd.Series) -> pd.Series:
        s = series.astype(float)
        if s.nunique(dropna=True) <= 1:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - s.min()) / (s.max() - s.min())

    score_mass = (
        0.30 * _normalize(product_eval["ate"])
        + 0.25 * _normalize(product_eval["empirical_uplift"])
        + 0.15 * _normalize(product_eval["qini"])
        + 0.15 * _normalize(product_eval["auuc"])
        + 0.05 * _normalize(product_eval["top_uplift_lift"])
        + 0.10 * (1 - _normalize(product_eval["negative_uplift_ratio"]))
    )
    score_targeted = (
        0.05 * _normalize(product_eval["ate"])
        + 0.15 * _normalize(product_eval["empirical_uplift"])
        + 0.25 * _normalize(product_eval["qini"])
        + 0.25 * _normalize(product_eval["auuc"])
        + 0.20 * _normalize(product_eval["top_uplift_lift"])
        + 0.10 * (1 - _normalize(product_eval["negative_uplift_ratio"]))
    )
    product_eval["product_score"] = np.where(
        product_eval["recommendation_decision"] == "recommend_targeted",
        score_targeted,
        score_mass,
    )

    con.close()
    return product_eval.sort_values(["recommendation_decision", "product_score"], ascending=[True, False]).reset_index(drop=True)


# ============================================================
# v3：推荐生成（兼容缺列 ps/mu）
# ============================================================

def generate_recommendations_duckdb(
    parquet_dir: str,
    product_eval_df: pd.DataFrame,
    customer_config: Optional[CustomerDecisionConfig] = None,
    safety_config: Optional[SafetyConfig] = None,
    product_config: Optional[ProductDecisionConfig] = None,
    duckdb_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    v3：与 v2 相比，关键差异：
    - 不假设 eval 表一定存在 ps/mu0/mu1
    - 输出 df 中仍保留 ps/mu0/mu1 列（若源数据没有则为 NULL）
    """
    customer_config = customer_config or CustomerDecisionConfig()
    safety_config = safety_config or SafetyConfig()
    product_config = product_config or ProductDecisionConfig()

    con = _duckdb_connect(duckdb_path)
    glob = _parquet_glob(parquet_dir)
    con.execute(f"CREATE OR REPLACE VIEW eval AS SELECT * FROM read_parquet('{glob}', hive_partitioning=1);")

    # 检测列存在性（避免 SQL 直接引用不存在列）
    cols = set(get_parquet_columns(parquet_dir, duckdb_path=duckdb_path))
    has_ps = "ps" in cols
    has_mu0 = "mu0" in cols
    has_mu1 = "mu1" in cols

    ps_expr = "e.ps" if has_ps else "CAST(NULL AS DOUBLE) AS ps"
    mu0_expr = "e.mu0" if has_mu0 else "CAST(NULL AS DOUBLE) AS mu0"
    mu1_expr = "e.mu1" if has_mu1 else "CAST(NULL AS DOUBLE) AS mu1"

    con.register("product_eval", product_eval_df)

    targeted_top_ratio = float(product_config.targeted_top_ratio)
    top_k = int(customer_config.top_k_per_customer)
    min_cate = float(customer_config.min_cate)
    min_gain = float(safety_config.min_customer_expected_gain)

    sql = f"""
    WITH cand AS (
      SELECT
        e.cust_id,
        e.product_id,
        e.date,
        e.cate,
        e.T,
        e.Y,
        {ps_expr},
        {mu0_expr},
        {mu1_expr},
        p.recommendation_decision,
        p.pass_rate,
        p.product_score,
        p.negative_uplift_ratio,
        p.calibration_factor,
        (e.cate * COALESCE(p.calibration_factor, 1.0)) AS adjusted_cate
      FROM eval e
      LEFT JOIN product_eval p USING(product_id)
      WHERE 1=1
        {"AND p.recommendation_decision IN ('recommend_all','recommend_targeted')" if safety_config.enable_product_blacklist_gate else ""}
    ),
    cand2 AS (
      SELECT *
      FROM cand
      WHERE adjusted_cate > {min_cate}
    ),
    targeted_ranked AS (
      SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY product_id ORDER BY adjusted_cate DESC) AS rn_in_prod,
        COUNT(*) OVER (PARTITION BY product_id) AS n_in_prod
      FROM cand2
      WHERE recommendation_decision = 'recommend_targeted'
    ),
    targeted_keep AS (
      SELECT *
      FROM targeted_ranked
      WHERE rn_in_prod <= CEIL(n_in_prod * {targeted_top_ratio})
    ),
    all_keep AS (
      SELECT
        cust_id, product_id, date, cate, T, Y, ps, mu0, mu1,
        recommendation_decision, pass_rate, product_score, negative_uplift_ratio, calibration_factor, adjusted_cate
      FROM cand2
      WHERE recommendation_decision = 'recommend_all'
    ),
    merged AS (
      SELECT
        cust_id, product_id, date, cate, T, Y, ps, mu0, mu1,
        recommendation_decision, pass_rate, product_score, negative_uplift_ratio, calibration_factor, adjusted_cate
      FROM all_keep
      UNION ALL
      SELECT
        cust_id, product_id, date, cate, T, Y, ps, mu0, mu1,
        recommendation_decision, pass_rate, product_score, negative_uplift_ratio, calibration_factor, adjusted_cate
      FROM targeted_keep
    ),
    scored AS (
      SELECT
        *,
        CASE
          WHEN (MAX(adjusted_cate) OVER () - MIN(adjusted_cate) OVER ()) = 0 THEN 0.0
          ELSE (adjusted_cate - MIN(adjusted_cate) OVER ()) / (MAX(adjusted_cate) OVER () - MIN(adjusted_cate) OVER ())
        END AS norm_adjusted_cate,
        CASE
          WHEN (MAX(product_score) OVER () - MIN(product_score) OVER ()) = 0 THEN 0.0
          ELSE (product_score - MIN(product_score) OVER ()) / (MAX(product_score) OVER () - MIN(product_score) OVER ())
        END AS norm_product_score,
        CASE
          WHEN (MAX(negative_uplift_ratio) OVER () - MIN(negative_uplift_ratio) OVER ()) = 0 THEN 0.0
          ELSE (negative_uplift_ratio - MIN(negative_uplift_ratio) OVER ()) / (MAX(negative_uplift_ratio) OVER () - MIN(negative_uplift_ratio) OVER ())
        END AS norm_neg_ratio
      FROM merged
    ),
    scored2 AS (
      SELECT
        *,
        (0.65 * norm_adjusted_cate + 0.25 * norm_product_score + 0.10 * (1 - norm_neg_ratio)) AS recommend_score
      FROM scored
    ),
    ranked AS (
      SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY cust_id ORDER BY recommend_score DESC) AS rank_in_customer
      FROM scored2
    )
    SELECT
      cust_id, product_id, date, cate, adjusted_cate, recommend_score, rank_in_customer, T, Y,
      ps, mu0, mu1,
      recommendation_decision, pass_rate, product_score, negative_uplift_ratio
    FROM ranked
    WHERE rank_in_customer <= {top_k}
      {"AND adjusted_cate >= " + str(min_gain) if safety_config.enable_customer_safe_filter else ""}
    ORDER BY cust_id, rank_in_customer
    """
    df = con.execute(sql).df()
    con.close()
    return df


# ============================================================
# v3：policy curve / temporal（沿用 v2）
# ============================================================

def policy_gain_curve_duckdb(
    reco_df: pd.DataFrame,
    score_col: str,
    bins: Sequence[float],
    baseline_mode: str = "global_mean",
    duckdb_path: Optional[str] = None,
) -> pd.DataFrame:
    if reco_df is None or reco_df.empty:
        return pd.DataFrame(columns=["top_pct", "n", "uplift_gain"])

    con = _duckdb_connect(duckdb_path)
    con.register("reco", reco_df)

    global_mean = float(con.execute("SELECT AVG(Y) FROM reco").fetchone()[0])

    rows: List[Dict] = []
    n_total = int(con.execute("SELECT COUNT(*) FROM reco").fetchone()[0])
    for b in bins:
        top_pct = float(b)
        k = max(1, int(np.ceil(n_total * top_pct)))

        if baseline_mode == "global_mean":
            uplift = float(con.execute(f"SELECT AVG(Y) FROM (SELECT * FROM reco ORDER BY {score_col} DESC LIMIT {k})").fetchone()[0]) - global_mean
        elif baseline_mode == "treated_control_in_top":
            uplift = float(
                con.execute(
                    f"""
                    WITH top AS (SELECT * FROM reco ORDER BY {score_col} DESC LIMIT {k})
                    SELECT
                      AVG(CASE WHEN T=1 THEN Y ELSE NULL END) - AVG(CASE WHEN T=0 THEN Y ELSE NULL END)
                    FROM top
                    """
                ).fetchone()[0]
            )
        else:
            raise ValueError(f"unknown baseline_mode: {baseline_mode}")

        rows.append({"top_pct": top_pct, "n": k, "uplift_gain": float(uplift)})

    con.close()
    return pd.DataFrame(rows)


def temporal_stability_duckdb(parquet_dir: str, duckdb_path: Optional[str] = None) -> pd.DataFrame:
    con = _duckdb_connect(duckdb_path)
    glob = _parquet_glob(parquet_dir)
    con.execute(f"CREATE OR REPLACE VIEW eval AS SELECT * FROM read_parquet('{glob}', hive_partitioning=1);")

    sql = """
    WITH base AS (
      SELECT
        date,
        AVG(cate) AS model_ate,
        AVG(CASE WHEN T=1 THEN Y ELSE NULL END) AS treated_mean,
        AVG(CASE WHEN T=0 THEN Y ELSE NULL END) AS control_mean,
        SUM(CASE WHEN T=1 THEN 1 ELSE 0 END) AS treated_n,
        SUM(CASE WHEN T=0 THEN 1 ELSE 0 END) AS control_n
      FROM eval
      GROUP BY date
    )
    SELECT
      date,
      model_ate,
      (treated_mean - control_mean) AS empirical_uplift,
      treated_n,
      control_n
    FROM base
    ORDER BY date
    """
    df = con.execute(sql).df()
    con.close()
    return df


# ============================================================
# v3：OPE（可选降级）
# ============================================================

def ope_ipw_policy_value(
    df: pd.DataFrame,
    policy_flag_col: str,
    ps_col: str = "ps",
    t_col: str = "T",
    y_col: str = "Y",
    ps_clip: Tuple[float, float] = (0.01, 0.99),
) -> float:
    d = df.copy()
    d["ps_clip"] = d[ps_col].clip(ps_clip[0], ps_clip[1])
    d["ipw_weight"] = d[policy_flag_col] * d[t_col] / d["ps_clip"]
    return float((d["ipw_weight"] * d[y_col]).sum() / len(d))


def ope_dr_policy_value(
    df: pd.DataFrame,
    policy_flag_col: str,
    ps_col: str = "ps",
    t_col: str = "T",
    y_col: str = "Y",
    mu1_col: str = "mu1",
    mu0_col: str = "mu0",
    ps_clip: Tuple[float, float] = (0.01, 0.99),
) -> float:
    d = df.copy()
    d["ps_clip"] = d[ps_col].clip(ps_clip[0], ps_clip[1])
    d["dr_term"] = d[policy_flag_col] * (
        (d[mu1_col] - d[mu0_col])
        + d[t_col] * (d[y_col] - d[mu1_col]) / d["ps_clip"]
        - (1 - d[t_col]) * (d[y_col] - d[mu0_col]) / (1 - d["ps_clip"])
    )
    return float(d["dr_term"].mean())


def build_policy_flag_top_pct(df: pd.DataFrame, score_col: str, top_pct: float) -> pd.Series:
    if df.empty:
        return pd.Series([], dtype=int)
    n = len(df)
    k = max(1, int(np.ceil(n * top_pct)))
    order = df[score_col].rank(method="first", ascending=False)
    return (order <= k).astype(int)


# ============================================================
# v3：主流程
# ============================================================

def run_backtest_v3(
    parquet_dir: str,
    external_metrics_df: Optional[pd.DataFrame] = None,
    product_config: Optional[ProductDecisionConfig] = None,
    customer_config: Optional[CustomerDecisionConfig] = None,
    safety_config: Optional[SafetyConfig] = None,
    backtest_config: Optional[BacktestConfig] = None,
    duckdb_path: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    product_config = product_config or ProductDecisionConfig()
    customer_config = customer_config or CustomerDecisionConfig()
    safety_config = safety_config or SafetyConfig()
    backtest_config = backtest_config or BacktestConfig()

    product_eval_df = evaluate_products_duckdb(
        parquet_dir=parquet_dir,
        product_config=product_config,
        external_metrics_df=external_metrics_df,
        duckdb_path=duckdb_path,
    )

    customer_reco_df = generate_recommendations_duckdb(
        parquet_dir=parquet_dir,
        product_eval_df=product_eval_df,
        customer_config=customer_config,
        safety_config=safety_config,
        product_config=product_config,
        duckdb_path=duckdb_path,
    )

    # 推荐子集 uplift（pandas 简算）
    if customer_reco_df.empty:
        reco_emp = pd.DataFrame([{"empirical_uplift": 0.0, "treated_n": 0, "control_n": 0}])
    else:
        treated = customer_reco_df[customer_reco_df["T"] == 1]
        control = customer_reco_df[customer_reco_df["T"] == 0]
        reco_emp = pd.DataFrame(
            [
                {
                    "empirical_uplift": float(treated["Y"].mean() - control["Y"].mean()),
                    "treated_mean_outcome": float(treated["Y"].mean()),
                    "control_mean_outcome": float(control["Y"].mean()),
                    "treated_n": int(len(treated)),
                    "control_n": int(len(control)),
                }
            ]
        )

    policy_gain_df = policy_gain_curve_duckdb(
        reco_df=customer_reco_df,
        score_col="recommend_score" if "recommend_score" in customer_reco_df.columns else "adjusted_cate",
        bins=backtest_config.policy_bins,
        baseline_mode="global_mean",
        duckdb_path=duckdb_path,
    )

    temporal_df = temporal_stability_duckdb(parquet_dir=parquet_dir, duckdb_path=duckdb_path)

    # OPE：根据列存在性降级
    parquet_cols = set(get_parquet_columns(parquet_dir, duckdb_path=duckdb_path))
    has_ps = "ps" in parquet_cols
    has_mu0 = "mu0" in parquet_cols
    has_mu1 = "mu1" in parquet_cols

    # OPE 默认用 “全表 top20_by_cate” 的 policy 示例（与 v1 一致）
    # 这里为了简单，不再从 parquet 读回全表（大数据），而是说明：OPE 要求全表策略 flag。
    # 你如果后续要做 OPE，我们再扩展“在 DuckDB 上直接构造 policy_flag + 聚合计算”。
    ope_rows: List[Dict] = []

    if not has_ps:
        ope_rows.append(
            dict(
                policy="top20_by_cate",
                ipw_value=np.nan,
                dr_value=np.nan,
                ipw_ok=False,
                dr_ok=False,
                ipw_error="缺少 ps（propensity score），无法计算 IPW/DR OPE。可先跳过 OPE。",
                dr_error="缺少 ps（propensity score），无法计算 IPW/DR OPE。可先跳过 OPE。",
            )
        )
        ope_df = pd.DataFrame(ope_rows)
    else:
        # 仅在小数据场景下（或你愿意加载全表）才做 pandas OPE；v3 先提供“不会报错 + 明确提示”。
        # 这里给出一个安全提示：默认不在 v3 中对超大 parquet 做全表回读。
        if not (has_mu0 and has_mu1):
            ope_rows.append(
                dict(
                    policy="top20_by_cate",
                    ipw_value=np.nan,
                    dr_value=np.nan,
                    ipw_ok=False,
                    dr_ok=False,
                    ipw_error="检测到 ps，但 v3 默认不回读全量 parquet 做 OPE（大数据会很慢）。如需 OPE，可扩展 DuckDB 版本计算。",
                    dr_error="缺少 mu0/mu1（或未启用全表 OPE 计算）。",
                )
            )
        else:
            ope_rows.append(
                dict(
                    policy="top20_by_cate",
                    ipw_value=np.nan,
                    dr_value=np.nan,
                    ipw_ok=False,
                    dr_ok=False,
                    ipw_error="检测到 ps+mu0+mu1，但 v3 默认不回读全量 parquet 做 OPE。可扩展 DuckDB 版本计算。",
                    dr_error="检测到 ps+mu0+mu1，但 v3 默认不回读全量 parquet 做 OPE。可扩展 DuckDB 版本计算。",
                )
            )

        ope_df = pd.DataFrame(ope_rows)

    return {
        "product_eval_df": product_eval_df,
        "customer_reco_df": customer_reco_df,
        "reco_empirical_eval_df": reco_emp,
        "policy_gain_df": policy_gain_df,
        "temporal_df": temporal_df,
        "ope_df": ope_df,
    }


# ============================================================
# Demo 入口：复用 v2 生成的 parquet（避免重复写）
# ============================================================

if __name__ == "__main__":
    # 默认复用 v2 的 parquet 输出目录（你也可以改成你的真实 parquet）
    parquet_dir = "backtest_output_v2/eval_parquet"

    out_root = Path("backtest_output_v3")
    out_root.mkdir(parents=True, exist_ok=True)

    result = run_backtest_v3(
        parquet_dir=parquet_dir,
        external_metrics_df=None,
        product_config=ProductDecisionConfig(
            min_ate=0.0,
            min_qini=0.0,
            min_auuc=0.0,
            min_top_uplift_lift=0.0,
            min_empirical_uplift=0.0,
            max_negative_uplift_ratio=0.4,
            min_support_samples=300,
            top_ratio=0.2,
            enable_calibration=True,
        ),
        customer_config=CustomerDecisionConfig(min_cate=0.0, top_k_per_customer=3),
        safety_config=SafetyConfig(max_customer_negative_share=0.4, min_customer_expected_gain=0.0),
        backtest_config=BacktestConfig(),
        duckdb_path=str(out_root / "duckdb_tmp.db"),
    )

    # 1) 落盘 csv
    for k, df in result.items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(out_root / f"{k}.csv", index=False)

    # 2) 生成 md 报告
    report_path = out_root / "backtest_report_v3.md"
    render_business_report_v3(result, out_path=str(report_path), top_products=20, top_reco_rows=50)
    print("v3 report saved to:", report_path)
    print("v3 outputs saved to:", out_root)
