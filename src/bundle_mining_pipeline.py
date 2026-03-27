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

# 复用 v3：Parquet + DuckDB 大数据回测 pipeline
# 该文件在仓库根目录：backtest_full_pipeline_v3.py
from backtest_full_pipeline_v3 import (  # noqa: F401
    BacktestConfig,
    CustomerDecisionConfig,
    ProductDecisionConfig,
    SafetyConfig,
    evaluate_products_duckdb,
    render_business_report_v3,
    run_backtest_v3,
)

# 可选：生产训练依赖（causalml）
# - debug 模式不需要 causalml
# - prod 模式如果缺少 causalml，会在运行时抛出清晰错误
try:
    from causalml.inference.meta import BaseDRLearner  # type: ignore
except Exception:  # pragma: no cover
    BaseDRLearner = None  # type: ignore


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


@dataclass
class BundleTrainConfig:
    """
    组合模型训练/推理配置（生产用）

    mode:
    - debug: 允许用单产品 cate 合成 bundle cate（仅用于流程调试）
    - prod : 独立训练 bundle 的 DRLearner 并推理得到 cate_bundle

    artifacts_dir:
    - 存放每个 bundle 的模型与 cate 结果（parquet）
      推荐结构：artifacts_dir/{bundle_id}/model.pkl + cate.parquet

    特别说明（与你当前 repo 的现状对齐）:
    - 生产训练需要 X 特征列；但当前 bundle_mining_pipeline.py 的输入 eval_df 只有 cust_id/product_id/date/T/Y/cate。
      因此这里做成“feature_cols 可传入 + features_df 可选 merge”的形式：
        - 若 eval_df 已包含 X 列：直接用
        - 若 X 在独立 features_df（按 cust_id,date 或 cust_id,date,product_id 对齐）：由你在调用时传入并 merge

    性能建议（千万级）:
    - prefer_parquet: 使用 parquet 进行 cate 落盘，速度/体积更好
    - float32: cate 用 float32
    - chunk_rows: 仅用于推理阶段分块写出（pandas+sklearn/causalml 并不真正支持 out-of-core 训练）
    """
    mode: str = "debug"  # debug|prod
    artifacts_dir: str = "bundle_artifacts"
    force_retrain: bool = False

    # 训练特征列（必须）
    feature_cols: Optional[List[str]] = None

    # 特征表 merge 方式（可选）
    # - None: 不 merge，期望 eval_df 已含特征列
    # - ["cust_id","date"]: 最常见
    feature_merge_keys: Optional[List[str]] = None

    # 生产：单产品训练数据读取
    # 你的实际情况：每个产品一个文件，列为 [cust_id, date, X..., T, Y]
    # 组合训练时需要把 bundle 内多个产品文件对齐到 cust_id-date 粒度，并构造 T_bundle/Y。
    per_product_data_dir: str = "per_product_data"
    per_product_file_pattern: str = "{product_id}.parquet"  # 也可以是 {product_id}.csv
    per_product_file_format: str = "parquet"  # parquet|csv
    # 读取时只保留必要列：keys + X + T + Y
    t_col: str = "T"
    y_col: str = "Y"

    # 模型训练参数（给 DRLearner 的 outcome/treatment/model 等留接口）
    # 这里先保留 dict，便于你接入你单产品训练时的配置
    learner_params: Optional[Dict[str, object]] = None

    prefer_parquet: bool = True
    cate_float32: bool = True
    chunk_rows: int = 2_000_000


# =========================
# 工具函数：bundle candidates
# =========================

def _as_str_product_id(series: pd.Series) -> pd.Series:
    # pandas category/number 都统一成 string，保证 bundle_id 稳定
    return series.astype(str).str.strip()


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
                bundle_id = "bundle_and__" + "__".join(str(x).strip() for x in prods)
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
                bundle_id = "bundle_and__" + "__".join(str(x).strip() for x in prods)
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
    保留旧版 pandas 构造（仅用于小数据 debug）。
    v3 大数据推荐使用 DuckDB 版本：build_bundle_eval_parquet_duckdb_debug()。
    """
    df = eval_df.copy()
    df["product_id"] = _as_str_product_id(df["product_id"])

    prods = set(bundle.products)
    sub = df[df["product_id"].isin(prods)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["cust_id", "product_id", "date", "cate", "T", "Y"])

    gkey = ["cust_id", "date"]
    t_and = sub.groupby(gkey, observed=False)["T"].min().rename("T_bundle").reset_index()
    y_agg = sub.groupby(gkey, observed=False)["Y"].mean().rename("Y").reset_index()

    out = t_and.merge(y_agg, on=gkey, how="left")
    out["product_id"] = bundle.bundle_id
    out.rename(columns={"T_bundle": "T"}, inplace=True)
    out["cate"] = np.nan
    out = out[["cust_id", "product_id", "date", "cate", "T", "Y"]]

    if min_bundle_support_rows > 0:
        treated_n = int((out["T"] == 1).sum())
        if treated_n < min_bundle_support_rows:
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
# 生产：bundle 独立训练 / 推理 / 落盘
# =========================

def _ensure_dir(path: str) -> None:
    import os
    os.makedirs(path, exist_ok=True)


def _assert_bundle_out_dir_safe(path: str, *, arg_name: str) -> None:
    """
    防呆：禁止 bundle 输出写入单品 backtest 目录，避免覆盖/污染单品结果。

    规则（简单强约束）：
    - path 必须落在 backtest_output_bundle_v3 下（或其子路径）
    - 且不能包含单品关键目录片段：backtest_output_v2/v3/backtest_output/eval_parquet

    说明：
    - Windows 下路径分隔符复杂，这里统一用 lower + replace("\","/") 做判断。
    """
    p = (path or "").replace("\\", "/").lower()

    # 强制要求：bundle 输出必须在 bundle 根目录下
    if "backtest_output_bundle_v3" not in p:
        raise ValueError(
            f"[bundle-output-safety] {arg_name} must be under 'backtest_output_bundle_v3/'. "
            f"Got: {path}"
        )

    # 注意：bundle 自己的目录名就叫 eval_parquet_bundle，所以这里不能用 "/eval_parquet" 做简单包含判断
    forbidden = ["backtest_output_v2", "output/backtest_output_v3", "backtest_output/"]
    hit = [x for x in forbidden if x in p]
    if hit:
        raise ValueError(
            f"[bundle-output-safety] {arg_name} points to a single-product output directory (hits={hit}). "
            f"Refuse to write. Got: {path}"
        )


def _bundle_artifact_paths(train_cfg: BundleTrainConfig, bundle_id: str) -> Dict[str, str]:
    import os
    bdir = os.path.join(train_cfg.artifacts_dir, bundle_id)
    _ensure_dir(bdir)
    return {
        "bundle_dir": bdir,
        "model_path": os.path.join(bdir, "drlearner_model.pkl"),
        "cate_path": os.path.join(bdir, "cate.parquet" if train_cfg.prefer_parquet else "cate.csv"),
    }


def _merge_features_if_needed(
    bundle_eval_df: pd.DataFrame,
    features_df: Optional[pd.DataFrame],
    train_cfg: BundleTrainConfig,
) -> pd.DataFrame:
    """
    旧接口保留：如果你已经在外部把特征整理成一个 features_df（cust_id-date 粒度），可用此 merge。
    你的现状是“每个产品独立落盘一个文件”，生产推荐走 _build_bundle_train_df_from_per_product_files()。
    """
    if features_df is None:
        return bundle_eval_df
    if not train_cfg.feature_merge_keys:
        raise ValueError("features_df provided but BundleTrainConfig.feature_merge_keys is None")
    keys = train_cfg.feature_merge_keys
    missing = [k for k in keys if k not in bundle_eval_df.columns]
    if missing:
        raise ValueError(f"bundle_eval_df missing merge keys: {missing}")
    missing2 = [k for k in keys if k not in features_df.columns]
    if missing2:
        raise ValueError(f"features_df missing merge keys: {missing2}")
    feat_cols = train_cfg.feature_cols
    if not feat_cols:
        raise ValueError("BundleTrainConfig.feature_cols must be set for prod mode")
    keep_cols = list(dict.fromkeys(keys + feat_cols))
    return bundle_eval_df.merge(features_df[keep_cols], on=keys, how="left")


def _load_cached_bundle_cate(
    cate_path: str,
    bundle_eval_df: pd.DataFrame,
    train_cfg: BundleTrainConfig,
) -> Optional[pd.DataFrame]:
    import os
    if not os.path.exists(cate_path):
        return None

    gkey = ["cust_id", "date"]
    if train_cfg.prefer_parquet and cate_path.endswith(".parquet"):
        cached = pd.read_parquet(cate_path)
    else:
        cached = pd.read_csv(cate_path)

    # 要求至少包含 cust_id/date/cate
    for c in ["cust_id", "date", "cate"]:
        if c not in cached.columns:
            return None

    out = bundle_eval_df.merge(cached[gkey + ["cate"]], on=gkey, how="left", suffixes=("", "_cached"))
    # 若 bundle_eval_df 原本 cate 是 NaN，则直接覆盖；否则保留原值
    out["cate"] = out["cate"].where(out["cate"].notna(), out["cate_cached"])
    out.drop(columns=["cate_cached"], inplace=True, errors="ignore")
    return out


def _train_and_predict_drlearner(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    train_cfg: BundleTrainConfig,
    model_path: Optional[str] = None,
) -> np.ndarray:
    """
    训练 causalml DRLearner 并返回 cate 预测（按 train_df 行顺序）。
    说明：
    - 这里提供一个最小可运行实现：BaseDRLearner + sklearn 默认模型（若你已有单品训练的 base learner，请在 learner_params 里传入）
    - 训练/预测均在内存内完成；千万级需要严格控制候选 bundle 数量 + 特征列数
    """
    if BaseDRLearner is None:
        raise ImportError("prod mode requires causalml. Please install causalml to use DRLearner training.")

    from sklearn.ensemble import RandomForestRegressor
    from joblib import dump

    params = train_cfg.learner_params or {}

    model = params.get("model")
    if model is None:
        model = RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 100)),
            max_depth=params.get("max_depth", None),
            random_state=int(params.get("random_state", 42)),
            n_jobs=-1,
        )

    outcome_model = params.get("outcome_model", model)
    treatment_model = params.get("treatment_model", model)

    learner = BaseDRLearner(
        outcome_learner=outcome_model,
        treatment_learner=treatment_model,
        control_name=0,
    )

    X = train_df[feature_cols]
    T = train_df["T"].astype(int)
    y = train_df["Y"].astype(float)

    learner.fit(X=X, treatment=T, y=y)
    cate = learner.predict(X)

    if model_path:
        dump(learner, model_path)

    return np.asarray(cate).reshape(-1)


def _load_product_file(path: str, file_format: str) -> pd.DataFrame:
    if file_format == "parquet":
        return pd.read_parquet(path)
    if file_format == "csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported file_format: {file_format}")


def _build_bundle_train_df_from_per_product_files(
    bundle: BundleCandidate,
    train_cfg: BundleTrainConfig,
) -> pd.DataFrame:
    """
    从 per_product_data_dir 读取 bundle 内每个产品的样本，并在 cust_id/date 粒度上构造 bundle 训练表。
    约定：
    - 每个产品文件至少包含 cust_id/date/T/Y 以及 train_cfg.feature_cols 中的特征列
    - 同一 cust_id/date 在不同产品文件中可对齐
    """
    import os

    if not train_cfg.feature_cols:
        raise ValueError("BundleTrainConfig.feature_cols must be set for prod mode")

    frames = []
    needed_cols = ["cust_id", "date", train_cfg.t_col, train_cfg.y_col] + list(train_cfg.feature_cols)

    for product_id in bundle.products:
        file_name = train_cfg.per_product_file_pattern.format(product_id=product_id)
        file_path = os.path.join(train_cfg.per_product_data_dir, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"missing per-product file for bundle product {product_id}: {file_path}")
        df = _load_product_file(file_path, train_cfg.per_product_file_format)
        missing = [c for c in needed_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{file_path} missing columns: {missing}")
        df = df[needed_cols].copy()
        df["product_id"] = str(product_id)
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["cust_id", "date", "product_id", "T", "Y"] + list(train_cfg.feature_cols))

    all_df = pd.concat(frames, ignore_index=True)
    all_df["cust_id"] = _as_str_product_id(all_df["cust_id"])
    all_df["product_id"] = _as_str_product_id(all_df["product_id"])

    gkey = ["cust_id", "date"]
    feat_agg = all_df.groupby(gkey, observed=False)[train_cfg.feature_cols].mean().reset_index()
    t_and = all_df.groupby(gkey, observed=False)[train_cfg.t_col].min().rename("T").reset_index()
    y_agg = all_df.groupby(gkey, observed=False)[train_cfg.y_col].mean().rename("Y").reset_index()

    out = feat_agg.merge(t_and, on=gkey, how="left").merge(y_agg, on=gkey, how="left")
    out["product_id"] = bundle.bundle_id
    return out[["cust_id", "product_id", "date", "T", "Y"] + list(train_cfg.feature_cols)]


def run_bundle_mining_backtest_v3_debug(
    eval_df: pd.DataFrame,
    product_eval_df: pd.DataFrame,
    output_dir: str = "output/backtest_output_v3",
    eval_parquet_dir: str = "output/backtest_output_v3/eval_parquet",
    bundle_cfg: Optional[BundleMiningConfig] = None,
) -> pd.DataFrame:
    """
    Bundle mining 的可直接运行 debug 入口。

    说明：
    - 输入 eval_df 应为单产品长表，至少包含 cust_id/product_id/date/T/Y，最好包含 cate
    - product_eval_df 需要有 product_id/recommendation_decision/product_score
    - 产物默认落在 output/backtest_output_v3 / output/backtest_output_v3/eval_parquet
    """
    bundle_cfg = bundle_cfg or BundleMiningConfig()
    _ensure_dir(output_dir)
    _ensure_dir(eval_parquet_dir)

    bundles = generate_bundle_candidates(product_eval_df, bundle_cfg)
    results: List[pd.DataFrame] = []

    for bundle in bundles:
        bundle_eval_df = build_bundle_eval_df_and_mode(
            eval_df=eval_df,
            bundle=bundle,
            min_bundle_support_rows=bundle_cfg.min_bundle_support_rows,
        )
        if bundle_eval_df.empty:
            continue

        if "cate" in eval_df.columns:
            bundle_eval_df = synthesize_bundle_cate(eval_df, bundle_eval_df, bundle, mode="min")

        bundle_eval_df = bundle_eval_df.copy()
        bundle_eval_df["product_id"] = bundle_eval_df["product_id"].astype(str)
        results.append(bundle_eval_df)

    if not results:
        return pd.DataFrame(columns=["cust_id", "product_id", "date", "cate", "T", "Y"])

    out_df = pd.concat(results, ignore_index=True)
    out_path = f"{eval_parquet_dir}/bundle_eval_debug.parquet"
    try:
        out_df.to_parquet(out_path, index=False)
    except Exception:
        pass

    return out_df


def run_bundle_mining_backtest_v3_prod(
    eval_df: pd.DataFrame,
    product_eval_df: pd.DataFrame,
    train_cfg: BundleTrainConfig,
    bundle_cfg: Optional[BundleMiningConfig] = None,
) -> pd.DataFrame:
    """
    Bundle mining 生产入口：
    - 先从 per_product_data_dir 构造 bundle 训练表
    - 训练 DRLearner 得到 cate_bundle
    - 输出 bundle 长表，后续可接 run_backtest_v3()
    """
    bundle_cfg = bundle_cfg or BundleMiningConfig()
    if not train_cfg.feature_cols:
        raise ValueError("BundleTrainConfig.feature_cols must be set for prod mode")

    bundles = generate_bundle_candidates(product_eval_df, bundle_cfg)
    results: List[pd.DataFrame] = []

    for bundle in bundles:
        paths = _bundle_artifact_paths(train_cfg, bundle.bundle_id)
        if not train_cfg.force_retrain:
            cached = _load_cached_bundle_cate(paths["cate_path"], pd.DataFrame(columns=["cust_id", "date", "cate"]), train_cfg)
            if cached is not None and not cached.empty:
                results.append(cached)
                continue

        train_df = _build_bundle_train_df_from_per_product_files(bundle, train_cfg)
        if train_df.empty:
            continue

        cate = _train_and_predict_drlearner(train_df, train_cfg.feature_cols, train_cfg, paths["model_path"])
        bundle_eval_df = train_df[["cust_id", "product_id", "date", "T", "Y"]].copy()
        bundle_eval_df["cate"] = cate
        results.append(bundle_eval_df)

        try:
            if train_cfg.prefer_parquet:
                bundle_eval_df.to_parquet(paths["cate_path"], index=False)
            else:
                bundle_eval_df.to_csv(paths["cate_path"], index=False)
        except Exception:
            pass

    if not results:
        return pd.DataFrame(columns=["cust_id", "product_id", "date", "cate", "T", "Y"])

    out_df = pd.concat(results, ignore_index=True)
    return out_df


def _demo_bundle_mining_entry() -> None:
    """
    演示入口（直接运行本文件时使用）。
    优先读取 output/backtest_output_v3/eval_parquet 下的示例数据；若不存在，则生成一个最小可跑的示例。
    """
    import os

    eval_parquet_dir = os.path.join("output/backtest_output_v3", "eval_parquet")
    output_dir = "output/backtest_output_v3"

    demo_eval_path = os.path.join(eval_parquet_dir, "eval.parquet")
    demo_product_eval_path = os.path.join(eval_parquet_dir, "product_eval.parquet")

    if os.path.exists(demo_eval_path) and os.path.exists(demo_product_eval_path):
        eval_df = pd.read_parquet(demo_eval_path)
        product_eval_df = pd.read_parquet(demo_product_eval_path)
    else:
        eval_df = pd.DataFrame(
            {
                "cust_id": ["客户A", "客户A", "客户B", "客户B"],
                "product_id": ["产品甲", "产品乙", "产品甲", "产品乙"],
                "date": ["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-01"],
                "cate": [0.12, 0.08, 0.15, 0.10],
                "T": [1, 0, 1, 1],
                "Y": [1.0, 0.0, 0.5, 1.0],
            }
        )
        product_eval_df = pd.DataFrame(
            {
                "product_id": ["产品甲", "产品乙"],
                "recommendation_decision": ["recommend_all", "recommend_targeted"],
                "product_score": [0.92, 0.88],
            }
        )

    print("[bundle-debug] 使用 eval 目录：output/backtest_output_v3/eval_parquet")
    print("[bundle-debug] 使用输出目录：output/backtest_output_v3")
    print("[bundle-debug] 示例产品ID：产品甲、产品乙（非数字ID 也可正常运行）")

    bundle_eval_df = run_bundle_mining_backtest_v3_debug(
        eval_df=eval_df,
        product_eval_df=product_eval_df,
        output_dir=output_dir,
        eval_parquet_dir=eval_parquet_dir,
    )

    print(f"[bundle-debug] bundle_eval_df rows={len(bundle_eval_df)}")

    # 可选：如果仓库中的 backtest_full_pipeline_v3 已提供可用的回测函数，可继续接回测。
    try:
        _ = run_backtest_v3
        print("[bundle-debug] 已加载 run_backtest_v3，可在此基础上继续接主回测流程。")
    except Exception:
        pass


if __name__ == "__main__":
    # 运行示例：
    #   python src/bundle_mining_pipeline.py
    #
    # 中文示例（支持中文 product_id，不依赖纯数字ID）：
    #   python src/bundle_mining_pipeline.py
    #   产品示例：产品甲、产品乙、产品丙
    #
    # 说明：
    # - 默认会优先读取 output/backtest_output_v3/eval_parquet/eval.parquet 和 product_eval.parquet
    # - 若文件不存在，会自动生成一个最小可跑的内置 demo
    _demo_bundle_mining_entry()
