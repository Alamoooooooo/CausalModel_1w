from __future__ import annotations

"""
组合因果挖掘（Bundle Mining）- 在现有单产品 pipeline 基础上的扩展
========================================================

目标
----
你当前已有单产品 long-format eval_df:
- cust_id, product_id, date, cate, T, Y
并已实现产品层门禁 + 客户层Top-K推荐 + 回测指标（ATE/empirical uplift/policy gain/OPE 等）。

本文件新增：
1) 组合候选生成（基于产品标签/评分/决策结果，模板化，控制组合规模）
2) Bundle treated 定义（AND为主）：T_bundle = AND(T_p==1 for p in bundle)
3) Bundle 评估表构造：把 bundle 当作“新产品”，复用 backtest_full_pipeline.run_backtest()
4) 组合专属指标：
   - synergy_score：ATE(bundle) - sum(ATE(products))
   - overlap_penalty：Top uplift 人群重叠率（推荐风险/冗余）
   - incremental_to_base：ATE(base+booster) - ATE(base)

注意
----
- 本文件默认“组合的 cate_bundle 需要独立训练推理获得”，因为单产品 cate 不可直接相加。
- 为了先把流程跑通：提供了一个可选的临时合成器 synthesize_bundle_cate()（仅调试用，不用于上线）。
- 输入 eval_df 至少需要 cust_id/product_id/date/T/Y，若想复用你已有单产品模型推理输出，则还需要 cate。

推荐落地方式（生产）
--------------------
- 第一步：用 generate_bundle_candidates() 生成 bundle 列表
- 第二步：为每个 bundle 构造训练样本（cust_id,date,X,T_bundle,Y），训练一个 CATE 模型并推理得到 cate_bundle
- 第三步：把 bundle 结果整理成 bundle_eval_df（长表），丢进 run_backtest() 出报告/策略收益

"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# 复用你现有的“整理版完整回测 pipeline”
# 该文件在仓库根目录：backtest_full_pipeline.py
from backtest_full_pipeline import (  # noqa: F401
    BacktestConfig,
    CustomerDecisionConfig,
    ProductDecisionConfig,
    SafetyConfig,
    run_backtest,
    validate_eval_df,
)


# =========================
# 配置：组合候选生成
# =========================

@dataclass(frozen=True)
class BundleCandidate:
    bundle_id: str
    products: Tuple[str, ...]
    base_product: Optional[str] = None
    booster_products: Tuple[str, ...] = ()


@dataclass
class BundleMiningConfig:
    """
    - and_mode: True => treated 当且仅当 bundle 内所有产品T都为1（AND，默认）
    - max_bundle_size: 最大组合大小（建议 2~3）
    - top_n_base: Base 产品候选数量（从 recommend_all 里按 product_score 取TopN）
    - top_n_booster: Booster 产品候选数量（从 recommend_targeted 里按 product_score 取TopN）
    - allow_base_plus_booster: 是否生成 Base + Booster 模板组合
    - allow_base_plus_base: 是否生成 Base + Base 模板组合
    - min_bundle_support_rows: bundle treated 记录数门槛（AND时很重要）
    - top_ratio_for_overlap: 用于计算 Top uplift 人群（按 cate）的人群比例
    """
    and_mode: bool = True
    max_bundle_size: int = 3

    top_n_base: int = 8
    top_n_booster: int = 8

    allow_base_plus_booster: bool = True
    allow_base_plus_base: bool = True

    min_bundle_support_rows: int = 500
    top_ratio_for_overlap: float = 0.2

    random_state: int = 42


# =========================
# 工具函数：bundle candidates
# =========================

def _as_str_product_id(series: pd.Series) -> pd.Series:
    # pandas category/number 都统一成 string，保证 bundle_id 稳定
    return series.astype(str)


def generate_bundle_candidates(
    product_eval_df: pd.DataFrame,
    cfg: Optional[BundleMiningConfig] = None,
) -> List[BundleCandidate]:
    """
    从 product_eval_df 生成 bundle 候选（控制规模，避免组合爆炸）

    product_eval_df 需至少包含：
    - product_id
    - recommendation_decision in {recommend_all, recommend_targeted, watchlist, reject}
    - product_score
    - product_tag (可选，但建议有：全民收益型/精准收割型等)

    生成策略（默认）：
    - Base 池：recommend_all 的 TopN
    - Booster 池：recommend_targeted 的 TopN
    - 组合模板：
        1) Base + Booster （优先）
        2) Base + Base （互补型，备选）
    """
    cfg = cfg or BundleMiningConfig()

    required_cols = {"product_id", "recommendation_decision", "product_score"}
    missing = [c for c in required_cols if c not in product_eval_df.columns]
    if missing:
        raise ValueError(f"product_eval_df missing columns: {missing}")

    df = product_eval_df.copy()
    df["product_id"] = _as_str_product_id(df["product_id"])

    base_pool = (
        df[df["recommendation_decision"] == "recommend_all"]
        .sort_values("product_score", ascending=False)
        .head(cfg.top_n_base)
    )
    booster_pool = (
        df[df["recommendation_decision"] == "recommend_targeted"]
        .sort_values("product_score", ascending=False)
        .head(cfg.top_n_booster)
    )

    base_ids = base_pool["product_id"].tolist()
    booster_ids = booster_pool["product_id"].tolist()

    bundles: List[BundleCandidate] = []

    # Base + Booster
    if cfg.allow_base_plus_booster:
        for b in base_ids:
            for u in booster_ids:
                if b == u:
                    continue
                prods = tuple(sorted([b, u]))
                bundle_id = f"bundle_and__{prods[0]}__{prods[1]}"
                bundles.append(
                    BundleCandidate(
                        bundle_id=bundle_id,
                        products=prods,
                        base_product=b,
                        booster_products=(u,),
                    )
                )

    # Base + Base
    if cfg.allow_base_plus_base:
        for i in range(len(base_ids)):
            for j in range(i + 1, len(base_ids)):
                p1, p2 = base_ids[i], base_ids[j]
                prods = tuple(sorted([p1, p2]))
                bundle_id = f"bundle_and__{prods[0]}__{prods[1]}"
                bundles.append(BundleCandidate(bundle_id=bundle_id, products=prods, base_product=None))

    # 控制 max_bundle_size（当前仅生成 size=2，留接口给未来 size=3）
    bundles = [b for b in bundles if len(b.products) <= cfg.max_bundle_size]

    # 去重（保险）
    uniq: Dict[str, BundleCandidate] = {}
    for b in bundles:
        uniq[b.bundle_id] = b

    return list(uniq.values())


# =========================
# bundle eval_df 构造（把 bundle 当新“产品”）
# =========================

def build_bundle_eval_df_and_mode(
    eval_df: pd.DataFrame,
    bundle: BundleCandidate,
    min_bundle_support_rows: int = 0,
) -> pd.DataFrame:
    """
    AND 模式：T_bundle=1 当且仅当 bundle 内所有产品 T==1

    输入 eval_df 是产品长表（每行一个 cust_id-date-product 记录），至少包含：
    - cust_id, date, product_id, T, Y
    若已有 cate（来自单产品推理）也会带上，后续可以用 synthesize_bundle_cate() 临时合成 bundle 的 cate。

    输出：bundle_eval_df（长表），字段：
    - cust_id, date, product_id=bundle_id, T (bundle), Y (按 cust_id-date 的 Y 聚合/一致性校验), cate(可选)
    """
    validate_eval_df(eval_df)

    df = eval_df.copy()
    df["product_id"] = _as_str_product_id(df["product_id"])

    prods = set(bundle.products)
    sub = df[df["product_id"].isin(prods)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["cust_id", "product_id", "date", "cate", "T", "Y"])

    # 以 cust_id-date 聚合，计算 AND
    gkey = ["cust_id", "date"]

    # AND: all T==1
    t_and = sub.groupby(gkey, observed=False)["T"].min().rename("T_bundle").reset_index()

    # Y: 你的定义是 t~t+30 存款差值，通常在同一 cust_id-date 下对所有产品行应一致。
    # 这里取 mean 并做一个一致性检查（方差过大说明输入表需要清洗）
    y_agg = sub.groupby(gkey, observed=False)["Y"].mean().rename("Y").reset_index()
    y_std = sub.groupby(gkey, observed=False)["Y"].std().rename("Y_std").reset_index()

    out = t_and.merge(y_agg, on=gkey, how="left").merge(y_std, on=gkey, how="left")
    out["product_id"] = bundle.bundle_id
    out.rename(columns={"T_bundle": "T"}, inplace=True)

    # cate：默认不提供（应来自 bundle 模型推理）。若输入有单产品 cate，可后续用 synthesize_bundle_cate()
    if "cate" in sub.columns:
        out["cate"] = np.nan
    else:
        out["cate"] = np.nan

    out = out[["cust_id", "product_id", "date", "cate", "T", "Y", "Y_std"]]

    if min_bundle_support_rows > 0:
        treated_n = int((out["T"] == 1).sum())
        if treated_n < min_bundle_support_rows:
            # 返回空表示样本不足
            return pd.DataFrame(columns=out.columns)

    return out


# =========================
# 临时：合成 bundle cate（仅用于流程跑通/调试）
# =========================

def synthesize_bundle_cate(
    eval_df: pd.DataFrame,
    bundle_eval_df: pd.DataFrame,
    bundle: BundleCandidate,
    mode: str = "min",
) -> pd.DataFrame:
    """
    把单产品 cate 合成 bundle 的 cate（仅调试用！生产应独立训练 bundle 模型）

    mode:
    - min: 取 bundle 内产品 cate 的 min（AND语义下较保守）
    - mean: 取 mean
    - sum: 取 sum（通常会偏大，不推荐）
    """
    df = eval_df.copy()
    df["product_id"] = _as_str_product_id(df["product_id"])

    sub = df[df["product_id"].isin(set(bundle.products))].copy()
    if sub.empty:
        return bundle_eval_df

    if "cate" not in sub.columns:
        return bundle_eval_df

    gkey = ["cust_id", "date"]

    if mode == "min":
        cate = sub.groupby(gkey, observed=False)["cate"].min().rename("cate_bundle").reset_index()
    elif mode == "mean":
        cate = sub.groupby(gkey, observed=False)["cate"].mean().rename("cate_bundle").reset_index()
    elif mode == "sum":
        cate = sub.groupby(gkey, observed=False)["cate"].sum().rename("cate_bundle").reset_index()
    else:
        raise ValueError(f"unknown mode: {mode}")

    out = bundle_eval_df.merge(cate, on=gkey, how="left")
    out["cate"] = out["cate_bundle"]
    out.drop(columns=["cate_bundle"], inplace=True)
    return out


# =========================
# 组合专属指标：synergy / overlap / incremental-to-base
# =========================

def compute_synergy_score(
    bundle_product_eval_row: pd.Series,
    product_eval_df: pd.DataFrame,
    bundle: BundleCandidate,
) -> float:
    """
    synergy = ATE(bundle) - sum(ATE(products))
    解释：组合整体效应是否超过单品线性叠加（>0 则可能存在协同）。
    """
    ate_bundle = float(bundle_product_eval_row.get("ate", 0.0))
    ate_map = (
        product_eval_df.assign(product_id=_as_str_product_id(product_eval_df["product_id"]))
        .set_index("product_id")["ate"]
        .to_dict()
        if "ate" in product_eval_df.columns
        else {}
    )
    ate_sum = 0.0
    for p in bundle.products:
        ate_sum += float(ate_map.get(str(p), 0.0))
    return float(ate_bundle - ate_sum)


def compute_top_overlap_ratio(
    eval_df: pd.DataFrame,
    bundle: BundleCandidate,
    top_ratio: float = 0.2,
) -> float:
    """
    Top uplift 人群重叠率（基于单产品 cate）
    overlap = |Top(A) ∩ Top(B) ∩ ...| / |Top(A) ∪ Top(B) ∪ ...|
    - overlap 高：组合可能冗余；overlap 低：更互补（但不保证协同）
    """
    df = eval_df.copy()
    df["product_id"] = _as_str_product_id(df["product_id"])
    if "cate" not in df.columns:
        return np.nan

    sets: List[set] = []
    for p in bundle.products:
        g = df[df["product_id"] == str(p)].copy()
        if g.empty:
            continue
        g = g.sort_values("cate", ascending=False)
        k = max(1, int(np.ceil(len(g) * float(top_ratio))))
        top = set(g.head(k)["cust_id"].astype(str).tolist())
        sets.append(top)

    if len(sets) <= 1:
        return 0.0

    inter = set.intersection(*sets)
    union = set.union(*sets)
    if len(union) == 0:
        return 0.0
    return float(len(inter) / len(union))


def compute_incremental_to_base(
    bundle_product_eval_row: pd.Series,
    product_eval_df: pd.DataFrame,
    bundle: BundleCandidate,
) -> float:
    """
    incremental_to_base = ATE(bundle) - ATE(base_product)
    仅在 bundle.base_product 存在时计算，否则 NaN。
    """
    if not bundle.base_product:
        return np.nan
    ate_bundle = float(bundle_product_eval_row.get("ate", 0.0))
    pe = product_eval_df.copy()
    pe["product_id"] = _as_str_product_id(pe["product_id"])
    row = pe[pe["product_id"] == str(bundle.base_product)]
    if row.empty:
        return np.nan
    ate_base = float(row.iloc[0].get("ate", 0.0))
    return float(ate_bundle - ate_base)


# =========================
# 主流程：bundle mining（离线）
# =========================

def run_bundle_mining_backtest(
    eval_df: pd.DataFrame,
    product_eval_df: pd.DataFrame,
    mining_cfg: Optional[BundleMiningConfig] = None,
    product_cfg_for_bundle: Optional[ProductDecisionConfig] = None,
    customer_cfg: Optional[CustomerDecisionConfig] = None,
    safety_cfg: Optional[SafetyConfig] = None,
    backtest_cfg: Optional[BacktestConfig] = None,
    synthesize_cate_mode: Optional[str] = "min",
) -> Dict[str, pd.DataFrame]:
    """
    输出：
    - bundle_candidates_df
    - bundle_product_eval_df（每个 bundle 的产品层评估结果）
    - bundle_metrics_df（synergy/overlap/incremental）
    - bundle_backtest_result_map（每个 bundle 一个 backtest dict；体积大，默认不落盘）
    """
    mining_cfg = mining_cfg or BundleMiningConfig()
    validate_eval_df(eval_df)

    # 1) 候选
    bundles = generate_bundle_candidates(product_eval_df, cfg=mining_cfg)
    bundle_candidates_df = pd.DataFrame(
        [
            {
                "bundle_id": b.bundle_id,
                "products": ",".join(map(str, b.products)),
                "base_product": b.base_product,
                "boosters": ",".join(map(str, b.booster_products)) if b.booster_products else "",
                "bundle_size": len(b.products),
            }
            for b in bundles
        ]
    )

    # 2) 构造 bundle eval_df 并拼成一个“大 eval_df”（bundle 维度）
    bundle_eval_frames: List[pd.DataFrame] = []
    kept: List[BundleCandidate] = []

    for b in bundles:
        if not mining_cfg.and_mode:
            raise NotImplementedError("当前版本只实现 AND bundle；如需 OR/序列，可继续扩展。")

        bdf = build_bundle_eval_df_and_mode(
            eval_df=eval_df,
            bundle=b,
            min_bundle_support_rows=mining_cfg.min_bundle_support_rows,
        )
        if bdf.empty:
            continue

        # 可选：用单产品 cate 合成，先把流程跑通
        if synthesize_cate_mode is not None:
            bdf = synthesize_bundle_cate(eval_df=eval_df, bundle_eval_df=bdf, bundle=b, mode=synthesize_cate_mode)

        # backtest_full_pipeline 需要 REQUIRED_COLUMNS；我们多了 Y_std 不影响
        bundle_eval_frames.append(bdf)
        kept.append(b)

    if not bundle_eval_frames:
        return {
            "bundle_candidates_df": bundle_candidates_df,
            "bundle_product_eval_df": pd.DataFrame(),
            "bundle_metrics_df": pd.DataFrame(),
        }

    bundle_eval_all = pd.concat(bundle_eval_frames, axis=0, ignore_index=True)

    # 3) 对所有 bundle 一次性跑 backtest（把 bundle 当 product_id）
    bundle_result = run_backtest(
        eval_df=bundle_eval_all.drop(columns=["Y_std"], errors="ignore"),
        external_metrics_df=None,
        product_config=product_cfg_for_bundle or ProductDecisionConfig(
            # bundle 的样本更少，默认把支持门槛放低一点，避免全 reject
            min_support_samples=max(50, int(mining_cfg.min_bundle_support_rows)),
            max_negative_uplift_ratio=0.55,
            top_ratio=0.2,
            enable_calibration=True,
        ),
        customer_config=customer_cfg or CustomerDecisionConfig(min_cate=0.0, top_k_per_customer=1),
        safety_config=safety_cfg or SafetyConfig(min_customer_expected_gain=0.0),
        backtest_config=backtest_cfg or BacktestConfig(),
    )

    bundle_product_eval_df = bundle_result["product_eval_df"].copy()
    bundle_product_eval_df["product_id"] = _as_str_product_id(bundle_product_eval_df["product_id"])

    # 4) 组合专属指标（synergy/overlap/incremental）
    # 把 kept bundle 映射为 dict
    bundle_map: Dict[str, BundleCandidate] = {b.bundle_id: b for b in kept}

    rows: List[Dict] = []
    for _, r in bundle_product_eval_df.iterrows():
        bid = str(r["product_id"])
        b = bundle_map.get(bid)
        if b is None:
            continue
        rows.append(
            {
                "bundle_id": bid,
                "products": ",".join(map(str, b.products)),
                "synergy_score": compute_synergy_score(r, product_eval_df=product_eval_df, bundle=b),
                "overlap_ratio_top": compute_top_overlap_ratio(eval_df, bundle=b, top_ratio=mining_cfg.top_ratio_for_overlap),
                "incremental_to_base": compute_incremental_to_base(r, product_eval_df=product_eval_df, bundle=b),
            }
        )

    bundle_metrics_df = pd.DataFrame(rows)

    # 5) 合并回 product_eval 方便排序
    bundle_product_eval_df = bundle_product_eval_df.merge(
        bundle_metrics_df, left_on="product_id", right_on="bundle_id", how="left"
    ).drop(columns=["bundle_id"], errors="ignore")

    return {
        "bundle_candidates_df": bundle_candidates_df,
        "bundle_product_eval_df": bundle_product_eval_df,
        "bundle_metrics_df": bundle_metrics_df,
        "bundle_backtest_result": bundle_result,
        "bundle_eval_all": bundle_eval_all,
    }


if __name__ == "__main__":
    # Demo：用 backtest_full_pipeline 的模拟数据做一次 bundle mining 演示
    from backtest_full_pipeline import EvalDFSimConfig, simulate_evaldf, evaluate_products

    sim_cfg = EvalDFSimConfig(
        n_customers=30_000,
        n_products=34,
        n_dates=2,
        use_category=True,
        use_float32=True,
        random_state=42,
    )
    eval_df = simulate_evaldf(sim_cfg)

    # 先做单产品评估，得到 recommend_all / recommend_targeted
    single_prod_eval = evaluate_products(eval_df)

    result = run_bundle_mining_backtest(
        eval_df=eval_df,
        product_eval_df=single_prod_eval,
        mining_cfg=BundleMiningConfig(
            and_mode=True,
            max_bundle_size=3,
            top_n_base=6,
            top_n_booster=6,
            min_bundle_support_rows=300,
        ),
        synthesize_cate_mode="min",  # 仅demo
    )

    print("bundle_candidates:", result["bundle_candidates_df"].shape)
    print("bundle_product_eval:", result["bundle_product_eval_df"].shape)
    print(result["bundle_product_eval_df"][["product_id", "recommendation_decision", "ate", "synergy_score", "overlap_ratio_top"]].head(10))
