from __future__ import annotations

"""
backtest_full_pipeline_v2.py
================================================
目标：
- 不修改 backtest_full_pipeline.py（保持原逻辑/原精确度）
- 支持超大 eval_df：通过“按 product_id（可选加 date）分区写 Parquet + DuckDB 外部排序/窗口函数”
  来精确计算：
  - 产品层：ATE / empirical uplift / negative_uplift_ratio / Top ratio 指标 / qini/auuc proxy（均精确）
  - 客户层：per-customer Top-K 推荐（精确，DuckDB window function）
  - policy_gain_curve：全局 Top% 曲线（精确，DuckDB 全局排序）
  - temporal_stability（精确）
  - OPE（IPW/DR）（精确；ps/mu0/mu1 可选）

注意：
- v2 的核心理念：pandas 只做“分块生成/写 parquet + 读取小汇总表 + 报告渲染”
  大规模排序/TopK/全局Top% 交给 DuckDB（可落盘、可外部排序）。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _fmt(x: object, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "-"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.{nd}f}"
    return str(x)


def render_business_report_v2(
    result: Dict[str, pd.DataFrame],
    out_path: str = "backtest_output_v2/backtest_report_v2.md",
    top_products: int = 20,
    top_reco_rows: int = 50,
) -> str:
    """
    将 `run_backtest_v2()` 的输出整理成“可给业务看的” Markdown 报告（v2 版）。

    与 v1 的 render_business_report 风格保持一致，但只依赖 v2 已产出的表。
    输出：
    - 写入 out_path（UTF-8）
    - 同目录写入 *_gbk.md（Windows 更好打开/复制）
    - 返回 markdown 文本
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
    md.append(f"# 回测报告（Backtest Report, v2）\n\n生成时间：{now}\n")

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
            ]
        )
    )
    md.append(f"### 2.1 Top 产品列表（按 decision + score 排序，Top {top_products}）\n")
    md.append(_table(prod_show) + "\n")

    md.append("## 三、客户层推荐（Customer Level Recommendations）\n")
    md.append(f"展示 Top {top_reco_rows} 条推荐记录（按 recommend_score 降序）：\n")
    md.append(_table(reco_show) + "\n")

    md.append("## 四、策略收益曲线（Policy Gain Curve）\n")
    md.append(_table(policy_show) + "\n")

    md.append("## 五、时间稳定性（Temporal Stability）\n")
    md.append(_table(temporal_show) + "\n")

    md.append("## 六、离线策略价值评估（OPE）\n")
    md.append(_table(ope_show) + "\n")

    md.append("## 附录：输出数据表说明\n")
    md.append(
        "\n".join(
            [
                "- `product_eval_df`：产品层聚合指标 + 门禁结果 + product_score",
                "- `customer_reco_df`：客户-产品推荐清单（含 adjusted_cate / recommend_score）",
                "- `reco_empirical_eval_df`：推荐子集的 treated-control uplift（sanity check）",
                "- `policy_gain_df`：不同触达比例下的收益曲线（经验口径）",
                "- `temporal_df`：按 date 维度的 model_ate vs empirical_uplift",
                "- `ope_df`：OPE 输出（v2 当前为占位，后续可补齐 IPW/DR）",
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
# 与 v1 保持一致的配置（复制，避免 import v1 引入额外副作用）
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
# eval_df 模拟数据（沿用 v1 的思路，但这里提供“直接写 parquet”的入口）
# ============================================================

@dataclass
class EvalDFSimConfig:
    n_customers: int = 50_000
    n_products: int = 40
    n_dates: int = 3
    start_date: str = "2026-01-01"
    freq: str = "D"

    chunk_rows: int = 2_000_000

    base_treated_rate: float = 0.15
    ps_noise: float = 0.05

    cate_mean: float = 0.0
    cate_std: float = 1.0
    true_tau_scale: float = 1.0
    y_base: float = 0.0
    y_noise_std: float = 1.0

    use_category: bool = True
    use_float32: bool = True

    random_state: int = 42


def validate_eval_df(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"eval_df 缺少必要字段: {missing}")


def simulate_evaldf(cfg: Optional[EvalDFSimConfig] = None) -> "pd.DataFrame | Iterator[pd.DataFrame]":
    """
    与 v1 simulate_evaldf 口径一致（含产品画像+客户异质性），用于生成 eval_df。
    当 cfg.chunk_rows>0 返回 iterator（分块），否则返回全量 DataFrame（仅适合小数据）。
    """
    import math
    from typing import Iterator

    cfg = cfg or EvalDFSimConfig()
    rng = np.random.default_rng(cfg.random_state)

    cust = np.arange(cfg.n_customers, dtype=np.int32)
    prod = np.arange(cfg.n_products, dtype=np.int16 if cfg.n_products < 32768 else np.int32)
    dates = pd.date_range(cfg.start_date, periods=cfg.n_dates, freq=cfg.freq)

    total_rows = int(cfg.n_customers) * int(cfg.n_products) * int(cfg.n_dates)
    f_dtype = np.float32 if cfg.use_float32 else np.float64

    cust_sensitivity = rng.normal(0, 1, cfg.n_customers).astype(f_dtype)

    prod_type = rng.choice(
        ["全民收益型", "精准收割型", "高风险波动型", "噪声型"],
        size=cfg.n_products,
        p=[0.25, 0.25, 0.25, 0.25],
    )

    base_effect_map = {"全民收益型": 1.2, "精准收割型": -0.3, "高风险波动型": -0.2, "噪声型": 0.0}
    noise_sigma_map = {"全民收益型": 0.35, "精准收割型": 0.50, "高风险波动型": 1.50, "噪声型": 1.00}
    sensitivity_k_map = {"全民收益型": 0.30, "精准收割型": 1.60, "高风险波动型": 0.80, "噪声型": 0.10}

    base_effect_by_prod = np.array([base_effect_map[t] for t in prod_type], dtype=f_dtype)
    noise_sigma_by_prod = np.array([noise_sigma_map[t] for t in prod_type], dtype=f_dtype)
    sensitivity_k_by_prod = np.array([sensitivity_k_map[t] for t in prod_type], dtype=f_dtype)

    def _build_chunk(start: int, size: int) -> pd.DataFrame:
        idx = np.arange(start, start + size, dtype=np.int64)

        cust_idx = (idx % cfg.n_customers).astype(np.int32)
        tmp = idx // cfg.n_customers
        prod_idx = (tmp % cfg.n_products).astype(prod.dtype)
        date_idx = (tmp // cfg.n_products).astype(np.int32)

        cust_id = cust[cust_idx]
        product_id = prod[prod_idx]
        date_vals = dates.values[date_idx]

        sensitivity = cust_sensitivity[cust_idx]
        base_effect = base_effect_by_prod[prod_idx]
        noise_sigma = noise_sigma_by_prod[prod_idx]
        k = sensitivity_k_by_prod[prod_idx]

        gate = (1.0 / (1.0 + np.exp(-2.0 * sensitivity))).astype(f_dtype)
        targeted_boost = (prod_type[prod_idx] == "精准收割型").astype(f_dtype) * (2.0 * gate - 1.0)

        cate = (
            base_effect
            + k * sensitivity
            + 1.2 * targeted_boost
            + rng.normal(0.0, 1.0, size).astype(f_dtype) * noise_sigma
        ).astype(f_dtype)
        cate = np.clip(cate, -6, 6)

        logits = (
            math.log(cfg.base_treated_rate / (1 - cfg.base_treated_rate))
            + 0.6 * sensitivity
            + 0.3 * rng.normal(0, 1, size)
        ).astype(f_dtype)
        ps = (1.0 / (1.0 + np.exp(-logits))).astype(f_dtype)
        ps = np.clip(ps, 0.01, 0.99)
        T = rng.binomial(1, ps).astype(np.int8)

        tau = (cfg.true_tau_scale * np.tanh(cate)).astype(f_dtype)
        mu0 = (cfg.y_base + rng.normal(0.0, cfg.y_noise_std, size=size)).astype(f_dtype)
        mu1 = (mu0 + tau).astype(f_dtype)
        Y = (mu0 + T.astype(f_dtype) * tau + rng.normal(0.0, cfg.y_noise_std, size=size)).astype(f_dtype)

        df = pd.DataFrame(
            {
                "cust_id": cust_id,
                "product_id": product_id,
                "date": date_vals,
                "cate": cate,
                "T": T,
                "Y": Y,
                "ps": ps,
                "mu0": mu0,
                "mu1": mu1,
                "product_type_true": np.array(prod_type, dtype=object)[prod_idx],
            }
        )

        if cfg.use_category:
            df["cust_id"] = pd.Categorical(df["cust_id"])
            df["product_id"] = pd.Categorical(df["product_id"])
            df["date"] = pd.Categorical(df["date"], ordered=True)
        return df

    if cfg.chunk_rows and cfg.chunk_rows > 0:
        def _iter() -> Iterator[pd.DataFrame]:
            for start in range(0, total_rows, int(cfg.chunk_rows)):
                size = min(int(cfg.chunk_rows), total_rows - start)
                yield _build_chunk(start, size)
        return _iter()

    return _build_chunk(0, total_rows)


# ============================================================
# Parquet 分区写入（pyarrow dataset）
# ============================================================

def write_evaldf_parquet_partitioned(
    eval_iter: Iterable[pd.DataFrame],
    out_dir: str,
    partition_cols: Sequence[str] = ("product_id",),
) -> None:
    """
    将 eval_df（可为 iterator）分块写入 Parquet，并按 partition_cols 分区。

    重要说明（pyarrow 分区 vs duckdb hive 分区）：
    - pyarrow.dataset.write_dataset 的 partitioning 参数，默认是“目录名=字段值”的分区方式（hive style），
      但它会把分区列从每个 parquet 文件 schema 里移除（列只存在于目录结构中）。
    - DuckDB 读取这种数据时，必须开启 hive 分区推断：read_parquet(..., hive_partitioning=1)。
    - 因为分区列不在 parquet 文件 schema 里，直接 read_parquet 会“看不到 product_id”，
      从而触发你看到的 BinderException。

    依赖：pyarrow
    输出结构示例（hive partitioning）：
      out_dir/
        product_id=0/part-xxxxx.parquet
        product_id=1/part-xxxxx.parquet
        ...

    注意：如果你看到的目录是 out_dir/0/part.parquet 这种“纯数字目录”，
    DuckDB 的 hive 分区推断不会把它当作 product_id 分区列，因此读不到 product_id。
    需要显式使用 hive 风格目录名（product_id=0）。
    """
    import pyarrow as pa
    import pyarrow.dataset as ds

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    format_ = ds.ParquetFileFormat()
    write_options = format_.make_write_options(compression="zstd")

    part_cols = list(partition_cols)
    # 生成 hive 风格目录：product_id=0/...
    # 这里用显式 Schema 来告诉 pyarrow 分区列的类型
    partitioning = ds.partitioning(
        pa.schema([(c, pa.int32()) for c in part_cols]),
        flavor="hive",
    )

    # 逐块写入
    for i, df in enumerate(eval_iter):
        validate_eval_df(df)

        table = pa.Table.from_pandas(df, preserve_index=False)
        ds.write_dataset(
            table,
            base_dir=str(out_path),
            format="parquet",
            partitioning=partitioning,
            existing_data_behavior="overwrite_or_ignore",
            file_options=write_options,
            basename_template=f"part-{i:05d}-{{i}}.parquet",
        )


# ============================================================
# DuckDB 辅助
# ============================================================

def _duckdb_connect(db_path: Optional[str] = None):
    import duckdb

    con = duckdb.connect(database=(db_path or ":memory:"))
    # 性能向：尽量让临时文件落盘而不是撑爆内存（默认目录在当前工作目录下）
    con.execute("PRAGMA threads=4;")
    con.execute("PRAGMA enable_progress_bar=false;")
    return con


def _parquet_glob(parquet_dir: str) -> str:
    # duckdb 读取分区目录：read_parquet('dir/**/*.parquet')
    # Windows 下务必用绝对路径 + POSIX(/) 风格，否则 glob 可能匹配不到文件，导致分区列无法被识别
    p = Path(parquet_dir).resolve().as_posix()
    return f"{p}/**/*.parquet"


# ============================================================
# v2：产品层评估（精确，DuckDB 负责排序指标）
# ============================================================

def evaluate_products_duckdb(
    parquet_dir: str,
    product_config: Optional[ProductDecisionConfig] = None,
    external_metrics_df: Optional[pd.DataFrame] = None,
    duckdb_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    从 Parquet（按 product_id 分区）精确计算产品层评估表：
    - ATE/treated_rate/outcome_rate/support/empirical_uplift/negative_uplift_ratio（SQL 聚合）
    - Top 指标、qini/auuc proxy（SQL + window/order by，逐 product 精确）
    - 校准、门禁、决策、打分、标签：按 v1 逻辑

    external_metrics_df（可选）：若提供真实 qini/auuc，可覆盖 proxy。
    """
    product_config = product_config or ProductDecisionConfig()

    con = _duckdb_connect(duckdb_path)
    glob = _parquet_glob(parquet_dir)
    con.execute(
        f"CREATE OR REPLACE VIEW eval AS SELECT * FROM read_parquet('{glob}', hive_partitioning=1);"
    )

    # 1) 产品基础聚合（精确）
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

    # 2) Top segment 指标（精确：产品内排序）
    #    top_n = ceil(cnt * top_ratio)
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

    # 3) qini/auuc proxy（精确：产品内排序 + cumsum）
    # v1 proxy:
    #   g sorted by cate desc
    #   cum_gain = cate.cumsum()
    #   auuc = cum_gain.mean()
    #   baseline = mean(cate) * (n+1)/2
    #   qini = auuc - baseline
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

    product_eval = (
        base.merge(top_df, on="product_id", how="left")
        .merge(qini_df, on="product_id", how="left")
    )

    # 覆盖真实 qini/auuc（可选）
    if external_metrics_df is not None and not external_metrics_df.empty:
        cols = [c for c in ["product_id", "qini", "auuc"] if c in external_metrics_df.columns]
        product_eval = product_eval.drop(columns=["qini", "auuc"], errors="ignore")
        product_eval = product_eval.merge(external_metrics_df[cols], on="product_id", how="left")

    # 校准因子（v1 口径）
    if product_config.enable_calibration:
        product_eval["calibration_factor"] = product_eval.apply(
            lambda r: (r["empirical_uplift"] / r["ate"]) if (pd.notna(r["ate"]) and r["ate"] != 0) else 1.0,
            axis=1,
        )
        product_eval["calibration_factor"] = (
            product_eval["calibration_factor"]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(1.0)
            .clip(0.0, 5.0)
        )
    else:
        product_eval["calibration_factor"] = 1.0

    # 门禁（v1 口径）
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
        & (
            (product_eval["ate"] < 0) if product_config.allow_targeted_when_ate_negative else True
        )
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
# v2：DuckDB 精确生成 customer-level Top-K 推荐
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
    精确复刻 v1 generate_recommendations 的关键逻辑：
    - 只保留 recommendation_decision in (recommend_all, recommend_targeted)
    - adjusted_cate = cate * calibration_factor
    - adjusted_cate > min_cate
    - targeted 产品：仅保留产品内 top (targeted_top_ratio) 的记录（按 adjusted_cate desc）
    - recommend_score 组合
    - 每客户 top_k（按 recommend_score desc）
    - 安全过滤：adjusted_cate >= min_customer_expected_gain
    """
    customer_config = customer_config or CustomerDecisionConfig()
    safety_config = safety_config or SafetyConfig()
    product_config = product_config or ProductDecisionConfig()

    import duckdb

    con = _duckdb_connect(duckdb_path)
    glob = _parquet_glob(parquet_dir)
    con.execute(
        f"CREATE OR REPLACE VIEW eval AS SELECT * FROM read_parquet('{glob}', hive_partitioning=1);"
    )

    # product_eval_df 注册成 duckdb 表（小表）
    con.register("product_eval", product_eval_df)

    # 主查询：先 join + 基础过滤
    # targeted: product 内 row_number <= ceil(cnt * targeted_top_ratio)
    targeted_top_ratio = float(product_config.targeted_top_ratio)
    top_k = int(customer_config.top_k_per_customer)
    min_cate = float(customer_config.min_cate)
    min_gain = float(safety_config.min_customer_expected_gain)

    sql = f"""
    WITH cand AS (
      SELECT
        e.*,
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
        cust_id,
        product_id,
        date,
        cate,
        T,
        Y,
        ps,
        mu0,
        mu1,
        product_type_true,
        recommendation_decision,
        pass_rate,
        product_score,
        negative_uplift_ratio,
        calibration_factor,
        adjusted_cate
      FROM cand2
      WHERE recommendation_decision = 'recommend_all'
    ),
    merged AS (
      SELECT
        cust_id,
        product_id,
        date,
        cate,
        T,
        Y,
        ps,
        mu0,
        mu1,
        product_type_true,
        recommendation_decision,
        pass_rate,
        product_score,
        negative_uplift_ratio,
        calibration_factor,
        adjusted_cate
      FROM all_keep
      UNION ALL
      SELECT
        cust_id,
        product_id,
        date,
        cate,
        T,
        Y,
        ps,
        mu0,
        mu1,
        product_type_true,
        recommendation_decision,
        pass_rate,
        product_score,
        negative_uplift_ratio,
        calibration_factor,
        adjusted_cate
      FROM targeted_keep
    ),
    scored AS (
      SELECT
        *,
        -- min-max normalize：duckdb 没有直接的 minmax scaler，这里用窗口整体 min/max
        -- 注意：这里的 normalize 是“全局 normalize”，与 v1 pandas 版本一致（v1 也是全局 normalize）
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
# v2：policy gain curve（DuckDB 精确）
# ============================================================

def policy_gain_curve_duckdb(
    reco_df: pd.DataFrame,
    score_col: str,
    bins: Sequence[float],
    baseline_mode: str = "global_mean",
    duckdb_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    精确计算 policy gain curve：
    - 全局按 score_col desc 排序
    - 取 top_pct 的子集
    - baseline_mode:
        - global_mean: mean(Y in top) - mean(Y in all)
        - treated_control_in_top: mean(Y|T=1, top) - mean(Y|T=0, top)
    """
    if reco_df is None or reco_df.empty:
        return pd.DataFrame(columns=["top_pct", "n", "uplift_gain"])

    con = _duckdb_connect(duckdb_path)
    con.register("reco", reco_df)

    # 全局均值
    global_mean = float(con.execute("SELECT AVG(Y) FROM reco").fetchone()[0])

    rows: List[Dict] = []
    for b in bins:
        top_pct = float(b)
        # top n = ceil(n_total * top_pct)
        n_total = int(con.execute("SELECT COUNT(*) FROM reco").fetchone()[0])
        k = max(1, int(np.ceil(n_total * top_pct)))

        if baseline_mode == "global_mean":
            uplift = float(
                con.execute(
                    f"SELECT AVG(Y) FROM (SELECT * FROM reco ORDER BY {score_col} DESC LIMIT {k})"
                ).fetchone()[0]
            ) - global_mean
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


# ============================================================
# v2：temporal stability（DuckDB 精确）
# ============================================================

def temporal_stability_duckdb(
    parquet_dir: str,
    duckdb_path: Optional[str] = None,
) -> pd.DataFrame:
    con = _duckdb_connect(duckdb_path)
    glob = _parquet_glob(parquet_dir)
    con.execute(
        f"CREATE OR REPLACE VIEW eval AS SELECT * FROM read_parquet('{glob}', hive_partitioning=1);"
    )

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
# v2：主流程
# ============================================================

def run_backtest_v2(
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

    # 推荐子集 uplift（这里用 pandas 简算即可，行数=客户*topk，通常较小）
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

    # OPE：v2 先留接口（需要 policy flag 的定义与 ps/mu0/mu1 列）
    ope_df = pd.DataFrame(
        [
            {
                "policy": "reco_topk",
                "ipw_value": np.nan,
                "dr_value": np.nan,
                "ipw_ok": False,
                "dr_ok": False,
                "ipw_error": "v2: OPE will be added (needs policy flag definition on full eval table).",
                "dr_error": "v2: OPE will be added (needs ps/mu0/mu1 and policy flag).",
            }
        ]
    )

    return {
        "product_eval_df": product_eval_df,
        "customer_reco_df": customer_reco_df,
        "reco_empirical_eval_df": reco_emp,
        "policy_gain_df": policy_gain_df,
        "temporal_df": temporal_df,
        "ope_df": ope_df,
    }


# ============================================================
# 运行示例：先写 parquet，再跑 v2
# ============================================================

if __name__ == "__main__":
    # 输出目录（v2 独立）
    out_root = Path("backtest_output_v2")
    parquet_dir = out_root / "eval_parquet"

    # 1) 生成模拟数据并分区写 parquet（按 product_id）
    sim_cfg = EvalDFSimConfig(
        n_customers=50_000,
        n_products=40,
        n_dates=3,
        chunk_rows=2_000_000,
        use_category=True,
        use_float32=True,
        random_state=42,
    )

    total_rows = sim_cfg.n_customers * sim_cfg.n_products * sim_cfg.n_dates
    print("total_rows:", total_rows)
    parquet_dir.mkdir(parents=True, exist_ok=True)

    eval_iter = simulate_evaldf(sim_cfg)
    if isinstance(eval_iter, pd.DataFrame):
        eval_iter = [eval_iter]

    print("writing parquet to:", parquet_dir)
    write_evaldf_parquet_partitioned(eval_iter, out_dir=str(parquet_dir), partition_cols=("product_id",))

    # 2) 跑 v2 回测（产品层 + duckdb 推荐 + policy curve + temporal）
    result = run_backtest_v2(
        parquet_dir=str(parquet_dir),
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

    # 3) 落盘结果（csv，便于快速查看）
    out_root.mkdir(parents=True, exist_ok=True)
    for k, df in result.items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(out_root / f"{k}.csv", index=False)

    # 4) 生成业务可读报告（Markdown）
    report_path = out_root / "backtest_report_v2.md"
    render_business_report_v2(result, out_path=str(report_path), top_products=20, top_reco_rows=50)
    print("v2 report saved to:", report_path)

    print("v2 outputs saved to:", out_root)
