from __future__ import annotations

"""
因果推荐指标体系：完整回测脚本（整理版）
================================================

本脚本将你目前 `metrics反事实回测.py` 中“已写的代码 + 文件末尾附加的尚未落地的思路”
整合成一个可运行（可按需替换数据读取部分）的回测脚本。

目标：
1) 产品层：ATE / Qini(AUUC) / Top uplift / 负uplift风险 / Empirical uplift(真实增量) 校验
2) 客户层：基于（可选校准后的）CATE 生成 Top-K 推荐
3) 回测层：
   - Empirical uplift on recommendations（在推荐子集上做 treated-control 差分）
   - Policy Simulation（不同触达比例下的策略收益曲线）
   - Temporal Stability（按时间维度的 ATE vs Empirical uplift）
   - Counterfactual Policy Evaluation（IPW / DR OPE：离线策略价值评估）

输入数据最少需要字段（长表 long format）：
- cust_id
- product_id
- date
- cate      : 模型推理输出（个体增量 / CATE）
- T         : 历史真实是否触达/达标（0/1）
- Y         : 结果变量（如活期存款提升，金额或0/1）
- ps        : propensity score（可选，但做 OPE 强烈建议提供）
- mu1, mu0  : 结果模型对 T=1/0 的预测（可选；做 DR OPE 需要）
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ============================================================
# 配置区
# ============================================================

@dataclass
class ProductDecisionConfig:
    """
    产品层决策阈值（用于决定产品是否进推荐池）

    关键字段解释（简版）：
    - min_ate：ATE门槛。ATE=mean(cate)。用于判断产品整体方向是否正向
    - min_empirical_uplift：真实增量门槛。empirical_uplift = E[Y|T=1]-E[Y|T=0]
    - min_qini/min_auuc：排序有效性门槛（可用外部真实qini/auuc覆盖 proxy）
    - min_top_uplift_lift：Top人群 uplift 相对整体 uplift 的优势（top_mean_cate-overall_mean_cate）
    - max_negative_uplift_ratio：产品内 cate<0 占比上限，用于控制误推风险
    - min_support_samples：样本量门槛，保证评估稳定性
    - top_ratio：Top人群比例（如0.2表示看Top20%）
    - enable_calibration：是否用 empirical_uplift/ate 对 cate 做校准（缓解尺度不一致）
    """
    min_ate: float = 0.0
    min_qini: float = 0.0
    min_auuc: float = 0.0
    min_top_uplift_lift: float = 0.0
    min_empirical_uplift: float = 0.0
    max_negative_uplift_ratio: float = 0.50
    min_recommendable_customers: int = 100  # 本脚本暂未做强门禁，但可用于扩展
    min_support_samples: int = 300
    top_ratio: float = 0.20
    use_bootstrap_significance: bool = False  # 本脚本先不实现bootstrap显著性，保留接口
    bootstrap_rounds: int = 200
    random_state: int = 42
    enable_calibration: bool = True


@dataclass
class CustomerDecisionConfig:
    """
    客户层推荐配置

    - min_cate：客户-产品级最小可推荐增量（这里使用 adjusted_cate）
    - top_k_per_customer：每客户最多推荐K个产品
    - customer_weight_col：客户权重（如客户价值/AUM），可用于 recommend_score 加权（这里先预留）
    """
    min_cate: float = 0.0
    top_k_per_customer: int = 3
    min_product_pass_rate: float = 0.0  # 本脚本未用，可扩展为更细产品门禁
    customer_weight_col: Optional[str] = None


@dataclass
class SafetyConfig:
    """
    安全阈值（客户层二次过滤）

    - min_customer_expected_gain：最低预期收益（这里用 adjusted_cate）
    - max_customer_negative_share：保留字段（用于进一步风险过滤扩展）
    """
    max_customer_negative_share: float = 0.5
    min_customer_expected_gain: float = 0.0
    enable_product_blacklist_gate: bool = True
    enable_customer_safe_filter: bool = True


@dataclass
class BacktestConfig:
    """
    回测配置
    """
    # policy simulation 触达比例档位
    policy_bins: Tuple[float, ...] = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
    # OPE ps clipping
    ps_clip_low: float = 0.01
    ps_clip_high: float = 0.99
    random_state: int = 42


# ============================================================
# 基础工具函数
# ============================================================

REQUIRED_COLUMNS = ["cust_id", "product_id", "date", "cate", "T", "Y"]

# ============================================================
# eval_df 模拟数据（用于调试/压测）
# ============================================================

@dataclass
class EvalDFSimConfig:
    """
    用于在本 pipeline 里快速构造一份“长表 long-format”的 eval_df，方便调试与压测。

    设计目标：
    - 可调数据量：通过 n_customers / n_products / n_dates 控制规模（总行数=三者乘积）
    - 可复现：random_state
    - 尽量省内存：可选 category/int32/float32，避免不必要 object
    - 支持分块生成：chunk_rows>0 时返回迭代器（yield DataFrame），便于 2 千万级别数据调试
    """
    n_customers: int = 200_000
    n_products: int = 50
    n_dates: int = 2
    start_date: str = "2026-01-01"
    freq: str = "D"

    # 生成分块：0 表示一次性生成完整 df；>0 表示按 chunk_rows 逐块 yield
    chunk_rows: int = 0

    # 倾向得分/处理率
    base_treated_rate: float = 0.15
    ps_noise: float = 0.05

    # 增量与结果生成参数（可按业务调）
    cate_mean: float = 0.0
    cate_std: float = 1.0
    true_tau_scale: float = 1.0  # 用于把 cate 映射为真实增量 tau
    y_base: float = 0.0          # outcome 基线（金额口径）
    y_noise_std: float = 1.0

    # 内存优化
    use_category: bool = True
    use_float32: bool = True

    random_state: int = 42


def estimate_evaldf_memory(n_rows: int, cfg: Optional[EvalDFSimConfig] = None) -> Dict[str, float]:
    """
    粗略估算 eval_df 的内存占用（MB）。
    说明：这是估算，不考虑 pandas block/索引等额外开销的精确值，但足够做容量判断。
    """
    cfg = cfg or EvalDFSimConfig()
    # 默认采用 category（cust_id/product_id/date） + float32/int8
    bytes_per_row = 0
    # category codes 默认 int32；pandas 可能用 int32/ int16，保守按 4 bytes
    if cfg.use_category:
        bytes_per_row += 4 * 3  # cust_id, product_id, date
    else:
        # 若用 object/string，占用会远高于此处估算；这里给一个保守下限
        bytes_per_row += 32 * 3

    bytes_per_row += (4 if cfg.use_float32 else 8)  # cate
    bytes_per_row += 1  # T int8
    bytes_per_row += (4 if cfg.use_float32 else 8)  # Y
    # ps/mu1/mu0 可选，不算在基础列里

    mb = bytes_per_row * n_rows / (1024 * 1024)
    return {"rows": float(n_rows), "estimated_MB_min": float(mb)}


def _as_category(values: np.ndarray, ordered: bool = False) -> pd.Categorical:
    return pd.Categorical(values, ordered=ordered)


# def simulate_evaldf(
#     cfg: Optional[EvalDFSimConfig] = None,
# ) -> "pd.DataFrame | 'typing.Iterator[pd.DataFrame]'":
#     """
#     构造一份可用于 run_backtest 的 eval_df。

#     字段：
#     - cust_id, product_id, date, cate, T, Y
#     并额外生成（可选用于 OPE 调试）：
#     - ps, mu1, mu0

#     生成逻辑（可解释 & 可控）：
#     - cate：N(cate_mean, cate_std)
#     - ps：sigmoid( a + b*cate + noise )，再 clip 到 (0.01,0.99)
#     - T：Bernoulli(ps)
#     - 真实增量 tau：true_tau_scale * tanh(cate)
#     - mu0：y_base + noise0
#     - mu1：mu0 + tau
#     - Y：mu0 + T*tau + eps
#     """
#     import math
#     from typing import Iterator

#     cfg = cfg or EvalDFSimConfig()
#     rng = np.random.default_rng(cfg.random_state)

#     # 构造维度
#     cust = np.arange(cfg.n_customers, dtype=np.int32)
#     prod = np.arange(cfg.n_products, dtype=np.int16 if cfg.n_products < 32768 else np.int32)
#     dates = pd.date_range(cfg.start_date, periods=cfg.n_dates, freq=cfg.freq)

#     total_rows = int(cfg.n_customers) * int(cfg.n_products) * int(cfg.n_dates)

#     # dtype 选择
#     f_dtype = np.float32 if cfg.use_float32 else np.float64

#     def _build_chunk(start: int, size: int) -> pd.DataFrame:
#         # 把 [0,total_rows) 映射回 (date, product, customer)
#         idx = np.arange(start, start + size, dtype=np.int64)

#         # 展开顺序：date-major -> product -> customer
#         # customer 索引最快变动
#         cust_idx = (idx % cfg.n_customers).astype(np.int32)
#         tmp = idx // cfg.n_customers
#         prod_idx = (tmp % cfg.n_products).astype(prod.dtype)
#         date_idx = (tmp // cfg.n_products).astype(np.int32)

#         cust_id = cust[cust_idx]
#         product_id = prod[prod_idx]
#         date_vals = dates.values[date_idx]

#         cate = rng.normal(cfg.cate_mean, cfg.cate_std, size=size).astype(f_dtype)

#         # propensity score：让 cate 与 ps 有一定相关性（便于 OPE 调试）
#         # 先做一个线性项，再 sigmoid
#         z = (math.log(cfg.base_treated_rate / (1 - cfg.base_treated_rate)) + 0.8 * cate).astype(f_dtype)
#         z = z + rng.normal(0.0, cfg.ps_noise, size=size).astype(f_dtype)
#         ps = (1.0 / (1.0 + np.exp(-z))).astype(f_dtype)
#         ps = np.clip(ps, 0.01, 0.99)

#         T = rng.binomial(1, ps).astype(np.int8)

#         tau = (cfg.true_tau_scale * np.tanh(cate)).astype(f_dtype)

#         mu0 = (cfg.y_base + rng.normal(0.0, cfg.y_noise_std, size=size)).astype(f_dtype)
#         mu1 = (mu0 + tau).astype(f_dtype)
#         Y = (mu0 + T.astype(f_dtype) * tau + rng.normal(0.0, cfg.y_noise_std, size=size)).astype(f_dtype)

#         df = pd.DataFrame(
#             {
#                 "cust_id": cust_id,
#                 "product_id": product_id,
#                 "date": date_vals,
#                 "cate": cate,
#                 "T": T,
#                 "Y": Y,
#                 "ps": ps,
#                 "mu0": mu0,
#                 "mu1": mu1,
#             }
#         )

#         # 内存优化：category
#         if cfg.use_category:
#             df["cust_id"] = _as_category(df["cust_id"].to_numpy())
#             df["product_id"] = _as_category(df["product_id"].to_numpy())
#             df["date"] = _as_category(df["date"].to_numpy(), ordered=True)

#         return df

#     if cfg.chunk_rows and cfg.chunk_rows > 0:
#         def _iter() -> Iterator[pd.DataFrame]:
#             for start in range(0, total_rows, int(cfg.chunk_rows)):
#                 size = min(int(cfg.chunk_rows), total_rows - start)
#                 yield _build_chunk(start, size)
#         return _iter()

#     return _build_chunk(0, total_rows)
def simulate_evaldf(cfg: Optional[EvalDFSimConfig] = None):
    import math
    from typing import Iterator

    cfg = cfg or EvalDFSimConfig()
    rng = np.random.default_rng(cfg.random_state)

    cust = np.arange(cfg.n_customers, dtype=np.int32)
    prod = np.arange(cfg.n_products, dtype=np.int16 if cfg.n_products < 32768 else np.int32)
    dates = pd.date_range(cfg.start_date, periods=cfg.n_dates, freq=cfg.freq)

    total_rows = int(cfg.n_customers) * int(cfg.n_products) * int(cfg.n_dates)
    f_dtype = np.float32 if cfg.use_float32 else np.float64

    # ================================
    # 🆕 客户异质性（真实业务关键）
    # ================================
    cust_sensitivity = rng.normal(0, 1, cfg.n_customers).astype(f_dtype)

    # ================================
    # 🆕 产品效果分层（真实业务关键）
    # ================================
    prod_quality = rng.choice(
        [2.0, 0.8, -0.5],   # 强 / 中 / 弱
        size=cfg.n_products,
        p=[0.2, 0.5, 0.3]
    ).astype(f_dtype)

    def _build_chunk(start: int, size: int) -> pd.DataFrame:
        idx = np.arange(start, start + size, dtype=np.int64)

        cust_idx = (idx % cfg.n_customers).astype(np.int32)
        tmp = idx // cfg.n_customers
        prod_idx = (tmp % cfg.n_products).astype(prod.dtype)
        date_idx = (tmp // cfg.n_products).astype(np.int32)

        cust_id = cust[cust_idx]
        product_id = prod[prod_idx]
        date_vals = dates.values[date_idx]

        # ================================
        # 🆕 真实个体uplift结构
        # ================================
        sensitivity = cust_sensitivity[cust_idx]
        base_effect = prod_quality[prod_idx]

        # 核心：产品强度 × 客户敏感度
        cate = (
            base_effect * (1 + 0.7 * sensitivity)
            + rng.normal(0, 0.5, size)
        ).astype(f_dtype)

        cate = np.clip(cate, -3, 5)

        # ================================
        # 🆕 更真实的投放机制
        # ================================
        logits = (
            math.log(cfg.base_treated_rate / (1 - cfg.base_treated_rate))
            + 0.6 * sensitivity
            + 0.3 * rng.normal(0, 1, size)
        ).astype(f_dtype)

        ps = (1.0 / (1.0 + np.exp(-logits))).astype(f_dtype)
        ps = np.clip(ps, 0.01, 0.99)
        T = rng.binomial(1, ps).astype(np.int8)

        # ================================
        # 潜在结果框架
        # ================================
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
# 模型层指标：产品聚合
# ============================================================

def compute_ate_by_product(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    ATE 口径：对每个产品，取 cate 的均值作为 ATE 估计。
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
    return out


def compute_empirical_uplift_by_product(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    真实增量（经验 uplift）：
    empirical_uplift = E[Y|T=1] - E[Y|T=0]
    """
    frames: List[Dict] = []
    for product_id, g in eval_df.groupby("product_id"):
        treated = g[g["T"] == 1]
        control = g[g["T"] == 0]

        treated_mean = treated["Y"].mean()
        control_mean = control["Y"].mean()
        uplift = treated_mean - control_mean

        frames.append(
            {
                "product_id": product_id,
                "empirical_uplift": uplift,
                "treated_mean_outcome": treated_mean,
                "control_mean_outcome": control_mean,
                "treated_n": len(treated),
                "control_n": len(control),
            }
        )
    return pd.DataFrame(frames)


def compute_calibration_factor(product_eval: pd.DataFrame) -> pd.DataFrame:
    """
    增量校准：用 empirical_uplift / ate 得到 calibration_factor，
    再将每条记录的 cate 乘以 calibration_factor 得到 adjusted_cate。

    直觉：如果模型估计的量纲/尺度偏大或偏小，用真实 uplift 做比例校准。
    注意：这只是粗校准（按产品整体校准），更精细可做分桶校准/Platt/Isotonic。
    """
    df = product_eval.copy()

    df["calibration_factor"] = df.apply(
        lambda r: _safe_divide(r["empirical_uplift"], r["ate"]) if r["ate"] != 0 else 1.0,
        axis=1,
    )
    df["calibration_factor"] = df["calibration_factor"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    df["calibration_factor"] = df["calibration_factor"].clip(0.0, 5.0)

    return df[["product_id", "calibration_factor"]]


def compute_top_segment_metrics(eval_df: pd.DataFrame, top_ratio: float) -> pd.DataFrame:
    """
    Top人群指标：
    - top_uplift_lift = Top均值 - Overall均值
    - top_vs_rest_gap = Top均值 - Rest均值
    """
    frames: List[Dict] = []
    for product_id, g in eval_df.groupby("product_id"):
        g = g.sort_values("cate", ascending=False)
        n = len(g)
        top_n = max(1, int(np.ceil(n * top_ratio)))

        top_g = g.iloc[:top_n]
        rest_g = g.iloc[top_n:]

        overall_mean = g["cate"].mean()
        top_mean = top_g["cate"].mean()
        rest_mean = rest_g["cate"].mean() if len(rest_g) else top_mean

        frames.append(
            {
                "product_id": product_id,
                "top_uplift_lift": top_mean - overall_mean,
                "top_vs_rest_gap": top_mean - rest_mean,
            }
        )
    return pd.DataFrame(frames)


def compute_negative_uplift_metrics(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    风险指标：
    - negative_uplift_ratio：cate<0 的比例
    - treated_negative_uplift_ratio：历史被触达/达标人群中 cate<0 的比例（仅作参考）
    """
    frames: List[Dict] = []
    for product_id, g in eval_df.groupby("product_id"):
        neg_mask = g["cate"] < 0
        treated_mask = g["T"] == 1

        frames.append(
            {
                "product_id": product_id,
                "negative_uplift_ratio": float(neg_mask.mean()),
                "treated_negative_uplift_ratio": (
                    float((neg_mask & treated_mask).sum() / treated_mask.sum())
                    if treated_mask.sum() > 0
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(frames)


def compute_qini_auuc_proxy(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Qini/AUUC proxy：
    若你已有 causalml 输出的真实 qini/auuc，请外部 merge 覆盖。

    注意：这里用 cate 的累计和构造 proxy，只保证流程一致性，不等价于严格定义的 Qini/AUUC。
    """
    frames: List[Dict] = []
    for product_id, g in eval_df.groupby("product_id"):
        g = g.sort_values("cate", ascending=False)
        n = len(g)
        if n == 0:
            continue

        cum_gain = g["cate"].cumsum()
        auuc = cum_gain.mean()
        baseline = g["cate"].mean() * (n + 1) / 2.0
        qini = auuc - baseline

        frames.append({"product_id": product_id, "auuc": float(auuc), "qini": float(qini)})
    return pd.DataFrame(frames)


# ============================================================
# 产品层评估 + 门禁
# ============================================================

def evaluate_products(
    eval_df: pd.DataFrame,
    product_config: Optional[ProductDecisionConfig] = None,
    external_metrics_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    输出 product_eval_df：
    - 各类产品聚合指标
    - 门禁 pass_xxx
    - recommendation_decision (recommend / watchlist / reject)
    - product_score（综合分，用于产品排序）
    """
    validate_eval_df(eval_df)
    config = product_config or ProductDecisionConfig()

    ate_df = compute_ate_by_product(eval_df)
    emp_df = compute_empirical_uplift_by_product(eval_df)
    top_df = compute_top_segment_metrics(eval_df, config.top_ratio)
    neg_df = compute_negative_uplift_metrics(eval_df)
    rank_df = compute_qini_auuc_proxy(eval_df)

    product_eval = (
        ate_df.merge(emp_df, on="product_id", how="left")
        .merge(top_df, on="product_id", how="left")
        .merge(neg_df, on="product_id", how="left")
        .merge(rank_df, on="product_id", how="left")
    )

    # 覆盖真实 qini/auuc（可选）
    if external_metrics_df is not None and not external_metrics_df.empty:
        cols = [c for c in ["product_id", "qini", "auuc"] if c in external_metrics_df.columns]
        product_eval = product_eval.drop(columns=["qini", "auuc"], errors="ignore")
        product_eval = product_eval.merge(external_metrics_df[cols], on="product_id", how="left")

    # 校准因子
    if config.enable_calibration:
        calib_df = compute_calibration_factor(product_eval)
        product_eval = product_eval.merge(calib_df, on="product_id", how="left")
    else:
        product_eval["calibration_factor"] = 1.0

    # 门禁判定
    product_eval["pass_ate"] = product_eval["ate"] > config.min_ate
    product_eval["pass_empirical"] = product_eval["empirical_uplift"] > config.min_empirical_uplift
    product_eval["pass_qini"] = product_eval["qini"] > config.min_qini
    product_eval["pass_auuc"] = product_eval["auuc"] > config.min_auuc
    product_eval["pass_top_lift"] = product_eval["top_uplift_lift"] > config.min_top_uplift_lift
    product_eval["pass_negative_risk"] = product_eval["negative_uplift_ratio"] <= config.max_negative_uplift_ratio
    product_eval["pass_support"] = product_eval["sample_size"] >= config.min_support_samples
    product_eval["pass_targeted"] = (
        (product_eval["top_uplift_lift"] > config.min_top_uplift_lift)
        & (product_eval["negative_uplift_ratio"] <= config.max_negative_uplift_ratio)
        )
    
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

    # product_eval["recommendation_decision"] = np.where(
    #     product_eval[gate_cols].all(axis=1),
    #     "recommend",
    #     np.where(product_eval["pass_ate"], "watchlist", "reject"),
    # )
    product_eval["recommendation_decision"] = np.select(
    [
        product_eval[gate_cols].all(axis=1),                        # 全指标通过
        product_eval["pass_targeted"],                              # 定向有效
        product_eval["pass_ate"],                                   # 平均有效
    ],
    [
        "recommend",        # 全量推荐
        "recommend_targeted",   # 仅高uplift人群
        "watchlist",            # 继续观察
    ],
    default="reject"
)

    # # 综合打分（用于产品排序/展示）
    # product_eval["product_score"] = (
    #     0.25 * _normalize_score(product_eval["ate"])
    #     + 0.20 * _normalize_score(product_eval["empirical_uplift"])
    #     + 0.15 * _normalize_score(product_eval["qini"])
    #     + 0.15 * _normalize_score(product_eval["auuc"])
    #     + 0.15 * _normalize_score(product_eval["top_uplift_lift"])
    #     + 0.10 * (1 - _normalize_score(product_eval["negative_uplift_ratio"]))
    # )

    score_mass = (
        0.30 * _normalize_score(product_eval["ate"])
        + 0.25 * _normalize_score(product_eval["empirical_uplift"])
        + 0.15 * _normalize_score(product_eval["qini"])
        + 0.15 * _normalize_score(product_eval["auuc"])
        + 0.05 * _normalize_score(product_eval["top_uplift_lift"])
        + 0.10 * (1 - _normalize_score(product_eval["negative_uplift_ratio"]))
    )

    score_targeted = (
        0.05 * _normalize_score(product_eval["ate"])
        + 0.15 * _normalize_score(product_eval["empirical_uplift"])
        + 0.25 * _normalize_score(product_eval["qini"])
        + 0.25 * _normalize_score(product_eval["auuc"])
        + 0.20 * _normalize_score(product_eval["top_uplift_lift"])
        + 0.10 * (1 - _normalize_score(product_eval["negative_uplift_ratio"]))
    )

    product_eval["product_score"] = np.where(
        product_eval["recommendation_decision"] == "recommend_targeted",
        score_targeted,
        score_mass
    )

    return product_eval.sort_values(["recommendation_decision", "product_score"], ascending=[True, False])


# ============================================================
# 客户层推荐（策略生成）
# ============================================================

def generate_recommendations(
    eval_df: pd.DataFrame,
    product_eval_df: pd.DataFrame,
    customer_config: Optional[CustomerDecisionConfig] = None,
    safety_config: Optional[SafetyConfig] = None,
) -> pd.DataFrame:
    """
    生成客户-产品推荐对：

    逻辑：
    1) 只保留 recommendation_decision == 'recommend' 的产品
    2) 使用 adjusted_cate = cate * calibration_factor（若启用校准）
    3) adjusted_cate > min_cate 才作为候选
    4) 客户内 Top-K
    5) 可选：按 min_customer_expected_gain 再过滤
    """
    validate_eval_df(eval_df)
    customer_config = customer_config or CustomerDecisionConfig()
    safety_config = safety_config or SafetyConfig()

    candidate_df = eval_df.merge(
        product_eval_df[
            [
                "product_id",
                "recommendation_decision",
                "pass_rate",
                "product_score",
                "negative_uplift_ratio",
                "calibration_factor",
            ]
        ],
        on="product_id",
        how="left",
    )

    # 产品门禁：仅进入推荐池的产品
    if safety_config.enable_product_blacklist_gate:
        candidate_df = candidate_df[candidate_df["recommendation_decision"] == "recommend"].copy()

    # 校准后的增量
    candidate_df["adjusted_cate"] = candidate_df["cate"] * candidate_df["calibration_factor"].fillna(1.0)

    # 客户-产品门禁
    candidate_df = candidate_df[candidate_df["adjusted_cate"] > customer_config.min_cate].copy()

    # 推荐得分：以 adjusted_cate 为主 + 产品综合分 + 风险惩罚
    candidate_df["recommend_score"] = (
        0.65 * _normalize_score(candidate_df["adjusted_cate"])
        + 0.25 * _normalize_score(candidate_df["product_score"])
        + 0.10 * (1 - _normalize_score(candidate_df["negative_uplift_ratio"]))
    )

    # 客户内排序
    candidate_df["rank_in_customer"] = candidate_df.groupby("cust_id")["recommend_score"].rank(
        method="first", ascending=False
    )

    customer_reco = candidate_df[candidate_df["rank_in_customer"] <= customer_config.top_k_per_customer].copy()

    # 安全过滤：最低预期收益
    if safety_config.enable_customer_safe_filter:
        customer_reco = customer_reco[customer_reco["adjusted_cate"] >= safety_config.min_customer_expected_gain].copy()

    return customer_reco.sort_values(["cust_id", "rank_in_customer"]).reset_index(drop=True)


# ============================================================
# 回测一：推荐子集上的经验 uplift（treated-control）
# ============================================================

def empirical_uplift_on_recommendations(customer_reco_df: pd.DataFrame) -> pd.DataFrame:
    """
    在“推荐出来的客户-产品对”子集上，计算经验 uplift：
    uplift = E[Y|T=1] - E[Y|T=0]

    注意：
    - 这是一个简单 sanity check，不等价于严格因果策略评估
    - 若推荐策略与历史触达有偏差，这个估计会受 selection bias 影响
    """
    if customer_reco_df.empty:
        return pd.DataFrame([{"empirical_uplift": 0.0, "treated_n": 0, "control_n": 0}])

    treated = customer_reco_df[customer_reco_df["T"] == 1]
    control = customer_reco_df[customer_reco_df["T"] == 0]

    uplift = treated["Y"].mean() - control["Y"].mean()

    return pd.DataFrame(
        [
            {
                "empirical_uplift": uplift,
                "treated_mean_outcome": treated["Y"].mean(),
                "control_mean_outcome": control["Y"].mean(),
                "treated_n": len(treated),
                "control_n": len(control),
            }
        ]
    )


# ============================================================
# 回测二：Policy Simulation（策略收益曲线）
# ============================================================

def policy_gain_curve(
    scored_df: pd.DataFrame,
    score_col: str,
    bins: Sequence[float],
    baseline_mode: str = "global_mean",
) -> pd.DataFrame:
    """
    目标：回答“只触达 Top X% 的记录，经验收益能有多大？”

    输入：
    - scored_df：一张“可触达记录表”（通常是 customer_reco_df 或 eval_df 的某种子集/合并表）
    - score_col：用于排序的分数列（建议用 adjusted_cate 或 recommend_score）
    - bins：触达比例档位，如 (0.01,0.02,0.05,...,1.0)
    - baseline_mode：
        - global_mean: uplift_gain = mean(Y in top) - mean(Y in all)
        - treated_control_in_top: uplift_gain = mean(Y|T=1, top)-mean(Y|T=0, top)（更接近 uplift 但噪声更大）

    输出：
    - top_pct
    - n
    - uplift_gain
    """
    if scored_df.empty:
        return pd.DataFrame(columns=["top_pct", "n", "uplift_gain"])

    df = scored_df.sort_values(score_col, ascending=False).reset_index(drop=True)
    n_all = len(df)
    df["cum_pct"] = (np.arange(n_all) + 1) / n_all

    global_mean = df["Y"].mean()
    results: List[Dict] = []

    for b in bins:
        sub = df[df["cum_pct"] <= b]
        if sub.empty:
            results.append({"top_pct": b, "n": 0, "uplift_gain": 0.0})
            continue

        if baseline_mode == "global_mean":
            uplift_gain = sub["Y"].mean() - global_mean
        elif baseline_mode == "treated_control_in_top":
            treated = sub[sub["T"] == 1]
            control = sub[sub["T"] == 0]
            uplift_gain = treated["Y"].mean() - control["Y"].mean()
        else:
            raise ValueError(f"unknown baseline_mode: {baseline_mode}")

        results.append({"top_pct": float(b), "n": int(len(sub)), "uplift_gain": float(uplift_gain)})

    return pd.DataFrame(results)


# ============================================================
# 回测三：Temporal Stability（按时间稳定性）
# ============================================================

def temporal_stability(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    输出每个 date 的：
    - model_ate: mean(cate)
    - empirical_uplift: mean(Y|T=1)-mean(Y|T=0)
    """
    if eval_df.empty:
        return pd.DataFrame(columns=["date", "model_ate", "empirical_uplift", "treated_n", "control_n"])

    def _emp_uplift(g: pd.DataFrame) -> pd.Series:
        treated = g[g["T"] == 1]
        control = g[g["T"] == 0]
        return pd.Series(
            {
                "empirical_uplift": treated["Y"].mean() - control["Y"].mean(),
                "treated_n": len(treated),
                "control_n": len(control),
            }
        )

    model_ate = eval_df.groupby("date")["cate"].mean().reset_index(name="model_ate")
    emp = eval_df.groupby("date").apply(_emp_uplift).reset_index()

    return model_ate.merge(emp, on="date", how="left").sort_values("date")


# ============================================================
# 回测四：反事实策略回测（OPE：IPW / DR）
# ============================================================

def ope_ipw_policy_value(
    cf_df: pd.DataFrame,
    policy_flag_col: str,
    ps_col: str = "ps",
    t_col: str = "T",
    y_col: str = "Y",
    ps_clip: Tuple[float, float] = (0.01, 0.99),
) -> float:
    """
    IPW 估计策略价值（人均）：

    value = (1/N) * sum_i [ I(pi(x_i)=1) * I(T_i=1) / ps_i * Y_i ]

    直觉：
    - 只在“策略推荐 且 历史真实触达”的样本上能观测到结果
    - 用 1/ps 校正“更容易被触达的人更可能出现在样本里”的偏差
    """
    df = cf_df.copy()
    if ps_col not in df.columns:
        raise ValueError("IPW 需要 ps propensity score 列")

    df["ps_clip"] = df[ps_col].clip(ps_clip[0], ps_clip[1])
    df["ipw_weight"] = df[policy_flag_col] * df[t_col] / df["ps_clip"]

    return float((df["ipw_weight"] * df[y_col]).sum() / len(df))


def ope_dr_policy_value(
    cf_df: pd.DataFrame,
    policy_flag_col: str,
    ps_col: str = "ps",
    t_col: str = "T",
    y_col: str = "Y",
    mu1_col: str = "mu1",
    mu0_col: str = "mu0",
    ps_clip: Tuple[float, float] = (0.01, 0.99),
) -> float:
    """
    Doubly Robust（DR）策略价值（推荐）：

    DR term = I(pi=1) * [ (mu1-mu0)
                        + T*(Y-mu1)/ps
                        - (1-T)*(Y-mu0)/(1-ps) ]

    value = mean(DR term)

    只要倾向模型准确 或 结果模型准确，估计仍然一致。
    """
    df = cf_df.copy()
    for c in [ps_col, mu1_col, mu0_col]:
        if c not in df.columns:
            raise ValueError(f"DR 需要列: {ps_col}, {mu1_col}, {mu0_col}")

    df["ps_clip"] = df[ps_col].clip(ps_clip[0], ps_clip[1])

    df["dr_term"] = df[policy_flag_col] * (
        (df[mu1_col] - df[mu0_col])
        + df[t_col] * (df[y_col] - df[mu1_col]) / df["ps_clip"]
        - (1 - df[t_col]) * (df[y_col] - df[mu0_col]) / (1 - df["ps_clip"])
    )

    return float(df["dr_term"].mean())


def build_policy_flag_top_pct(
    df: pd.DataFrame,
    score_col: str,
    top_pct: float,
) -> pd.Series:
    """
    根据 score_col 排序，取 Top top_pct 作为策略推荐（1/0）。
    用于 OPE 或 policy simulation。
    """
    if df.empty:
        return pd.Series([], dtype=int)

    n = len(df)
    k = max(1, int(np.ceil(n * top_pct)))
    order = df[score_col].rank(method="first", ascending=False)
    return (order <= k).astype(int)


# ============================================================
# 主流程：一键跑回测
# ============================================================

def run_backtest(
    eval_df: pd.DataFrame,
    external_metrics_df: Optional[pd.DataFrame] = None,
    product_config: Optional[ProductDecisionConfig] = None,
    customer_config: Optional[CustomerDecisionConfig] = None,
    safety_config: Optional[SafetyConfig] = None,
    backtest_config: Optional[BacktestConfig] = None,
) -> Dict[str, pd.DataFrame]:
    """
    返回一个 dict，包含：
    - product_eval_df
    - customer_reco_df
    - reco_empirical_eval_df
    - policy_gain_df
    - temporal_df
    - ope_df（若 ps/mu1/mu0 等列缺失，会给 NaN 并附加原因）
    """
    validate_eval_df(eval_df)
    product_config = product_config or ProductDecisionConfig()
    customer_config = customer_config or CustomerDecisionConfig()
    safety_config = safety_config or SafetyConfig()
    backtest_config = backtest_config or BacktestConfig()

    # 1) 产品评估
    product_eval_df = evaluate_products(
        eval_df=eval_df,
        product_config=product_config,
        external_metrics_df=external_metrics_df,
    )

    # 2) 生成推荐
    customer_reco_df = generate_recommendations(
        eval_df=eval_df,
        product_eval_df=product_eval_df,
        customer_config=customer_config,
        safety_config=safety_config,
    )

    # 3) 推荐子集上的经验 uplift
    reco_empirical_eval_df = empirical_uplift_on_recommendations(customer_reco_df)

    # 4) 策略收益曲线（用 recommend_score）
    policy_gain_df = policy_gain_curve(
        scored_df=customer_reco_df,
        score_col="recommend_score" if "recommend_score" in customer_reco_df.columns else "adjusted_cate",
        bins=backtest_config.policy_bins,
        baseline_mode="global_mean",
    )

    # 5) 时间稳定性
    temporal_df = temporal_stability(eval_df)

    # 6) 反事实策略价值（OPE）
    ope_rows: List[Dict] = []
    cf_df = eval_df.copy()

    # 用“推荐策略”构造 policy flag：这里示例采用 Top 20% adjusted_cate
    # 注意：更严格的做法是直接用 customer_reco_df 的输出作为策略（即 policy_flag=1 for recommended pairs）
    if "ps" in cf_df.columns:
        # 若你想用“推荐结果”做策略flag，则需将 customer_reco_df 映射回 eval_df
        # 这里提供两种方式：
        # A) top_pct 方式：更像通用策略曲线
        # B) reco_df 方式：更贴近“最终策略清单”
        if "mu1" in cf_df.columns and "mu0" in cf_df.columns:
            pass

    # A) Top pct 策略（基于 cate 或 adjusted_cate）
    score_for_policy = "cate"
    cf_df["policy_top20"] = build_policy_flag_top_pct(cf_df, score_col=score_for_policy, top_pct=0.2)

    try:
        ipw_v = ope_ipw_policy_value(
            cf_df=cf_df,
            policy_flag_col="policy_top20",
            ps_col="ps",
            ps_clip=(backtest_config.ps_clip_low, backtest_config.ps_clip_high),
        )
        ipw_ok = True
        ipw_err = ""
    except Exception as e:
        ipw_v = np.nan
        ipw_ok = False
        ipw_err = str(e)

    try:
        dr_v = ope_dr_policy_value(
            cf_df=cf_df,
            policy_flag_col="policy_top20",
            ps_col="ps",
            mu1_col="mu1",
            mu0_col="mu0",
            ps_clip=(backtest_config.ps_clip_low, backtest_config.ps_clip_high),
        )
        dr_ok = True
        dr_err = ""
    except Exception as e:
        dr_v = np.nan
        dr_ok = False
        dr_err = str(e)

    ope_rows.append(
        {
            "policy": "top20_by_cate",
            "ipw_value": ipw_v,
            "dr_value": dr_v,
            "ipw_ok": ipw_ok,
            "dr_ok": dr_ok,
            "ipw_error": ipw_err,
            "dr_error": dr_err,
        }
    )

    ope_df = pd.DataFrame(ope_rows)

    return {
        "product_eval_df": product_eval_df,
        "customer_reco_df": customer_reco_df,
        "reco_empirical_eval_df": reco_empirical_eval_df,
        "policy_gain_df": policy_gain_df,
        "temporal_df": temporal_df,
        "ope_df": ope_df,
    }


# ============================================================
# 可选：输出/落盘辅助
# ============================================================

def export_backtest_results(
    result: Dict[str, pd.DataFrame],
    out_dir: str = "backtest_output",
) -> None:
    """
    简单落盘：将各 DataFrame 输出为 csv。
    """
    import os

    os.makedirs(out_dir, exist_ok=True)
    for k, df in result.items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(os.path.join(out_dir, f"{k}.csv"), index=False)


def _fmt(x: object, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "-"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.{nd}f}"
    return str(x)


def _gate_fail_summary(
    product_eval_df: pd.DataFrame,
    gate_cols: Sequence[str],
    decision_col: str = "recommendation_decision",
) -> pd.DataFrame:
    """
    统计门禁通过率 + 主要失败原因（Top1/Top2）。
    """
    if product_eval_df.empty:
        return pd.DataFrame()

    rows: List[Dict] = []
    for c in gate_cols:
        if c in product_eval_df.columns:
            rows.append({"gate": c, "pass_rate": float(product_eval_df[c].mean())})

    # 每个产品的失败门禁列表
    fail_reasons: List[str] = []
    for _, r in product_eval_df.iterrows():
        fails = [c for c in gate_cols if (c in product_eval_df.columns and (not bool(r[c])))]
        if not fails:
            fail_reasons.append("(all_pass)")
        else:
            fail_reasons.extend(fails)

    if len(fail_reasons) == 0:
        top1, top2 = "-", "-"
    else:
        vc = pd.Series(fail_reasons).value_counts()
        top1 = f"{vc.index[0]} ({int(vc.iloc[0])})" if len(vc) >= 1 else "-"
        top2 = f"{vc.index[1]} ({int(vc.iloc[1])})" if len(vc) >= 2 else "-"

    out = pd.DataFrame(rows).sort_values("gate")
    out["top_fail_reason_1"] = top1
    out["top_fail_reason_2"] = top2

    # decision 分布
    if decision_col in product_eval_df.columns:
        dec = product_eval_df[decision_col].value_counts().rename_axis("decision").reset_index(name="n_products")
        dec["share"] = dec["n_products"] / dec["n_products"].sum()
    else:
        dec = pd.DataFrame()

    # 作为一个 dict-like 返回：这里用 DataFrame 的 attrs 传递 decision summary（报告里会单独输出）
    out.attrs["decision_summary"] = dec
    return out


def _recommendation_diagnosis_text(
    product_eval_df: pd.DataFrame,
    product_config: "ProductDecisionConfig | None" = None,
) -> str:
    """
    当 recommend=0 或推荐很少时，给出可读的诊断说明（重点解释示例数据为何无推荐）。
    """
    if product_eval_df.empty:
        return "当前 product_eval_df 为空，无法做推荐诊断。"

    product_config = product_config or ProductDecisionConfig()

    n_reco = int((product_eval_df.get("recommendation_decision") == "recommend").sum())
    if n_reco > 0:
        return ""

    # 重点解释 negative uplift 门禁
    if "negative_uplift_ratio" in product_eval_df.columns:
        neg_mean = float(product_eval_df["negative_uplift_ratio"].mean())
        neg_q50 = float(product_eval_df["negative_uplift_ratio"].quantile(0.5))
    else:
        neg_mean, neg_q50 = np.nan, np.nan

    msg = []
    msg.append("### 诊断：为什么本次没有任何产品进入 recommend？")
    msg.append("")
    msg.append(
        "本 pipeline 的产品层采用“多门禁同时满足才进入 recommend”的机制。"
        "在示例模拟数据中，`cate` 近似对称分布（默认 N(0,1)），因此 `cate < 0` 的比例通常接近 50%。"
    )
    msg.append("")
    msg.append(
        f"- 当前配置 `max_negative_uplift_ratio={product_config.max_negative_uplift_ratio}`"
        f"，而本次数据 `negative_uplift_ratio` 中位数≈{_fmt(neg_q50, 4)}，均值≈{_fmt(neg_mean, 4)}"
        "，因此大部分/全部产品会在 `pass_negative_risk` 上失败，最终导致 recommend=0。"
    )
    msg.append("")
    msg.append("可选解决方案（用于让示例报告更像业务结果）：")
    msg.append(
        "- 方案A（改门禁）：把 `max_negative_uplift_ratio` 放宽到 0.55~0.60（先验证流程/报告展示）。"
    )
    msg.append(
        "- 方案B（改模拟分布）：提高 `EvalDFSimConfig.cate_mean`（例如 0.2~0.4）或改变 cate 生成逻辑，让负 uplift 比例明显低于 0.5。"
    )
    return "\n".join(msg)


def render_business_report(
    result: Dict[str, pd.DataFrame],
    out_path: str = "backtest_output/backtest_report.md",
    top_products: int = 20,
    top_reco_rows: int = 50,
) -> str:
    """
    将 `run_backtest()` 的输出整理成“可给业务看的” Markdown 报告。

    输出：
    - 写入 out_path
    - 返回 markdown 文本（便于你在 notebook/console 直接 print）
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
    n_customers = int(customer_reco_df["cust_id"].nunique()) if ("cust_id" in customer_reco_df.columns and not customer_reco_df.empty) else 0
    n_products = int(product_eval_df["product_id"].nunique()) if ("product_id" in product_eval_df.columns and not product_eval_df.empty) else 0
    n_reco_products = int((product_eval_df.get("recommendation_decision") == "recommend").sum()) if not product_eval_df.empty else 0

    reco_uplift = float(reco_empirical_eval_df["empirical_uplift"].iloc[0]) if not reco_empirical_eval_df.empty else np.nan

    # 产品表（取 Top）
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
            "negative_uplift_ratio",
            "product_score",
        ]
        cols = [c for c in cols if c in product_eval_df.columns]
        prod_show = product_eval_df.sort_values(["recommendation_decision", "product_score"], ascending=[True, False])[cols].head(top_products)
    else:
        prod_show = pd.DataFrame()

    # 推荐明细（取 Top）
    if not customer_reco_df.empty:
        reco_cols = [
            "cust_id",
            "product_id",
            "date",
            "cate",
            "adjusted_cate",
            "recommend_score",
            "T",
            "Y",
        ]
        reco_cols = [c for c in reco_cols if c in customer_reco_df.columns]
        reco_show = customer_reco_df.sort_values("recommend_score", ascending=False)[reco_cols].head(top_reco_rows)
    else:
        reco_show = pd.DataFrame()

    # policy curve 简化显示
    if not policy_gain_df.empty:
        policy_show = policy_gain_df.copy()
    else:
        policy_show = pd.DataFrame(columns=["top_pct", "n", "uplift_gain"])

    # temporal 简化显示
    if not temporal_df.empty:
        temporal_show = temporal_df.copy()
    else:
        temporal_show = pd.DataFrame(columns=["date", "model_ate", "empirical_uplift", "treated_n", "control_n"])

    # ope
    ope_show = ope_df.copy() if not ope_df.empty else pd.DataFrame(columns=["policy", "ipw_value", "dr_value", "ipw_ok", "dr_ok", "ipw_error", "dr_error"])

    # 生成 markdown（表格用 pandas.to_markdown，若环境无 tabulate 则 fallback 到 to_string）
    def _table(df: pd.DataFrame) -> str:
        if df is None or df.empty:
            return "_(empty)_"
        try:
            return df.to_markdown(index=False)
        except Exception:
            return "```\n" + df.to_string(index=False) + "\n```"

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md = []
    md.append(f"# 回测报告（Backtest Report）\n\n生成时间：{now}\n")
    md.append("## 一、概览（Executive Summary）\n")
    md.append(
        "\n".join(
            [
                f"- 覆盖产品数：{_fmt(n_products, 0)}",
                f"- 进入推荐池产品数（decision=recommend）：{_fmt(n_reco_products, 0)}",
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
                "- `recommend`：通过所有产品门禁（可进入推荐池）",
                "- `watchlist`：仅满足部分门禁（例如 ATE 为正但风险或排序能力不足）",
                "- `reject`：关键门禁未通过（整体方向/风险等不满足）",
                "",
                "产品进入 recommend 需要同时满足的门禁包括：ATE、empirical uplift、qini/auuc proxy、top lift、negative uplift 风险、样本量等。",
                "",
            ]
        )
    )

    gate_cols = [
        "pass_ate",
        "pass_empirical",
        "pass_qini",
        "pass_auuc",
        "pass_top_lift",
        "pass_negative_risk",
        "pass_support",
    ]
    gate_summary = _gate_fail_summary(product_eval_df, gate_cols=gate_cols)
    dec_summary = gate_summary.attrs.get("decision_summary") if hasattr(gate_summary, "attrs") else None

    md.append("### 2.1 门禁通过率与主要失败原因汇总\n")
    md.append(_table(gate_summary) + "\n")
    if isinstance(dec_summary, pd.DataFrame) and not dec_summary.empty:
        md.append("### 2.2 产品决策分布（recommend/watchlist/reject）\n")
        md.append(_table(dec_summary) + "\n")

    diag_txt = _recommendation_diagnosis_text(product_eval_df, product_config=None)
    if diag_txt:
        md.append(diag_txt + "\n")

    md.append("### 2.3 Top 产品列表（按 decision + score 排序）\n")
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
                "- `ope_df`：IPW/DR 离线策略价值（需 ps/mu0/mu1）",
            ]
        )
        + "\n"
    )

    md_text = "\n".join(md)
    # Windows 终端 `type` 对 UTF-8 支持不稳定，额外输出一个 GBK 版本便于业务同事直接打开/复制
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
# 可选：基于分位数的默认阈值建议（自动化调参起点）
# ============================================================

def suggest_default_thresholds(
    eval_df: pd.DataFrame,
    top_ratio: float = 0.2,
    negative_ratio_cap_quantile: float = 0.7,
    min_support_samples: int = 300,
    min_recommendable_customers_quantile: float = 0.5,
) -> Dict[str, object]:
    """
    基于“当前批次数据分布”的阈值建议（一个自动化调参起点）

    思路：
    1) 先按产品聚合出核心评估表（ate、empirical_uplift、qini/auuc proxy、top lift、negative ratio、sample_size）
    2) 对部分指标采用分位数做门槛（相对阈值），避免固定阈值在不同批次下失灵
    3) 返回可直接用于 ProductDecisionConfig / CustomerDecisionConfig / SafetyConfig 的建议参数 dict

    返回字段：
    - product_config: ProductDecisionConfig(...) 的建议参数 dict
    - customer_config: CustomerDecisionConfig(...) 的建议参数 dict
    - safety_config: SafetyConfig(...) 的建议参数 dict
    - diagnostics: 一些分布统计，便于你检查阈值合理性

    注意：
    - 这里只给“建议起点”，不等于最终最优阈值
    - 若你有外部 qini/auuc，建议将 external_metrics_df merge 后再做阈值建议
    """
    validate_eval_df(eval_df)

    tmp_prod = evaluate_products(
        eval_df=eval_df,
        product_config=ProductDecisionConfig(
            top_ratio=top_ratio,
            enable_calibration=False,  # 阈值建议阶段先不校准，避免校准对分布造成二次扰动
            min_support_samples=min_support_samples,
        ),
        external_metrics_df=None,
    )

    # 分位数门槛：qini/auuc/top lift 取中位数（或更高）
    qini_thr = float(tmp_prod["qini"].quantile(0.5)) if "qini" in tmp_prod.columns else 0.0
    auuc_thr = float(tmp_prod["auuc"].quantile(0.5)) if "auuc" in tmp_prod.columns else 0.0
    top_lift_thr = float(tmp_prod["top_uplift_lift"].quantile(0.5)) if "top_uplift_lift" in tmp_prod.columns else 0.0

    # ATE：通常保留正向，同时可提升到某个分位数（这里用 max(0, 0.3分位)）
    ate_q30 = float(tmp_prod["ate"].quantile(0.3))
    min_ate = max(0.0, ate_q30)

    # Empirical uplift：同理（如果Y是金额，门槛需要业务解释；这里给一个相对阈值）
    emp_q30 = float(tmp_prod["empirical_uplift"].quantile(0.3))
    min_empirical = max(0.0, emp_q30)

    # 负uplift比例：希望越小越好，因此取较低的分位数上限（这里用 70%分位作为 cap）
    neg_cap = float(tmp_prod["negative_uplift_ratio"].quantile(negative_ratio_cap_quantile))

    # 可推荐客群规模：按产品内 cate>0 的人数统计（这里用中位数）
    recommendable_count = (
        eval_df.assign(pos=(eval_df["cate"] > 0).astype(int))
        .groupby("product_id")["pos"]
        .sum()
        .rename("recommendable_customers")
        .reset_index()
    )
    min_reco_customers = int(recommendable_count["recommendable_customers"].quantile(min_recommendable_customers_quantile))

    product_cfg = dict(
        min_ate=float(min_ate),
        min_qini=float(qini_thr),
        min_auuc=float(auuc_thr),
        min_top_uplift_lift=float(top_lift_thr),
        min_empirical_uplift=float(min_empirical),
        max_negative_uplift_ratio=float(neg_cap),
        min_recommendable_customers=int(max(1, min_reco_customers)),
        min_support_samples=int(min_support_samples),
        top_ratio=float(top_ratio),
        enable_calibration=True,
    )

    # 客户层：默认只要求 >0
    customer_cfg = dict(
        min_cate=0.0,
        top_k_per_customer=2,
        customer_weight_col=None,
    )

    # 安全层：默认与产品层负uplift门槛一致 + 最低收益为0
    safety_cfg = dict(
        max_customer_negative_share=float(neg_cap),
        min_customer_expected_gain=0.0,
        enable_product_blacklist_gate=True,
        enable_customer_safe_filter=True,
    )

    diagnostics = {
        "ate_quantiles": tmp_prod["ate"].quantile([0.1, 0.3, 0.5, 0.7, 0.9]).to_dict(),
        "empirical_uplift_quantiles": tmp_prod["empirical_uplift"].quantile([0.1, 0.3, 0.5, 0.7, 0.9]).to_dict(),
        "qini_quantiles": tmp_prod["qini"].quantile([0.1, 0.3, 0.5, 0.7, 0.9]).to_dict(),
        "auuc_quantiles": tmp_prod["auuc"].quantile([0.1, 0.3, 0.5, 0.7, 0.9]).to_dict(),
        "top_uplift_lift_quantiles": tmp_prod["top_uplift_lift"].quantile([0.1, 0.3, 0.5, 0.7, 0.9]).to_dict(),
        "negative_uplift_ratio_quantiles": tmp_prod["negative_uplift_ratio"].quantile([0.1, 0.3, 0.5, 0.7, 0.9]).to_dict(),
        "recommendable_customers_quantiles": recommendable_count["recommendable_customers"].quantile([0.1, 0.3, 0.5, 0.7, 0.9]).to_dict(),
        "n_products": int(tmp_prod["product_id"].nunique()),
        "n_rows": int(len(eval_df)),
    }

    return {
        "product_config": product_cfg,
        "customer_config": customer_cfg,
        "safety_config": safety_cfg,
        "diagnostics": diagnostics,
    }


# ============================================================
# 运行示例（按需修改数据读取部分）
# ============================================================

if __name__ == "__main__":
    # 你需要把这里替换成自己的 eval_df 读取方式
    # 例：
    # eval_df = pd.read_parquet("eval_df.parquet")
    # 或者 eval_df = pd.read_csv("eval_df.csv")
    #
    # 注意：eval_df 至少需要 cust_id, product_id, date, cate, T, Y
    # 如果要跑 OPE，请补充 ps；如果要跑 DR OPE，请补充 mu1, mu0

    # --------------------------------------------------------
    # 调试用：模拟一份 eval_df（按需调参）
    sim_cfg = EvalDFSimConfig(
        n_customers=50_000,
        n_products=40,
        n_dates=3,
        chunk_rows=0,        # 大数据建议设为 2_000_000 之类，做分块调试
        use_category=True,
        use_float32=True,
        random_state=42,
    )
    n_rows = sim_cfg.n_customers * sim_cfg.n_products * sim_cfg.n_dates
    print("eval_df rows:", n_rows)
    print("estimated memory:", estimate_evaldf_memory(n_rows, sim_cfg))
    eval_df = simulate_evaldf(sim_cfg)
    # --------------------------------------------------------

    # 1) 自动建议阈值（可选）
    # suggestion = suggest_default_thresholds(eval_df)
    # print("Suggested thresholds:", suggestion)

    # 2) 正式回测
    result = run_backtest(
        eval_df=eval_df,
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
        customer_config=CustomerDecisionConfig(
            min_cate=0.0,
            top_k_per_customer=3,
            customer_weight_col=None,
        ),
        safety_config=SafetyConfig(
            max_customer_negative_share=0.4,
            min_customer_expected_gain=0.0,
        ),
        backtest_config=BacktestConfig(),
    )

    # 3) 输出/落盘（可选）
    # export_backtest_results(result, out_dir="backtest_output")

    # 4) 生成业务可读报告（Markdown）
    report_path = "backtest_output/backtest_report.md"
    render_business_report(result, out_path=report_path, top_products=20, top_reco_rows=50)
    print(f"report saved: {report_path}")
    print("note: also saved GBK version for Windows console:", "backtest_output/backtest_report_gbk.md")
