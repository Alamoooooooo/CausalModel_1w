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

    特别说明（与你当前 repo 的现状对齐）：
    - 生产训练需要 X 特征列；但当前 bundle_mining_pipeline.py 的输入 eval_df 只有 cust_id/product_id/date/T/Y/cate。
      因此这里做成“feature_cols 可传入 + features_df 可选 merge”的形式：
        - 若 eval_df 已包含 X 列：直接用
        - 若 X 在独立 features_df（按 cust_id,date 或 cust_id,date,product_id 对齐）：由你在调用时传入并 merge

    性能建议（千万级）：
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
    - Windows 下路径分隔符复杂，这里统一用 lower + replace("\\","/") 做判断。
    """
    p = (path or "").replace("\\", "/").lower()

    # 强制要求：bundle 输出必须在 bundle 根目录下
    if "backtest_output_bundle_v3" not in p:
        raise ValueError(
            f"[bundle-output-safety] {arg_name} must be under 'backtest_output_bundle_v3/'. "
            f"Got: {path}"
        )

    # 注意：bundle 自己的目录名就叫 eval_parquet_bundle，所以这里不能用 "/eval_parquet" 做简单包含判断
    forbidden = ["backtest_output_v2", "backtest_output_v3", "backtest_output/"]
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

    # 默认基学习器（可按你单品训练替换）
    # DRLearner 需要 outcome_model / treatment_model（或是 BaseDRLearner 的默认配置）
    # causalml 的 BaseDRLearner 接口可能随版本不同，这里尽量用最常见形态：
    # BaseDRLearner(learner=..., control_outcome_learner=..., treatment_outcome_learner=...)
    base_learner = params.get(
        "base_learner",
        RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 200)),
            min_samples_leaf=int(params.get("min_samples_leaf", 50)),
            max_depth=params.get("max_depth", None),
            random_state=int(params.get("random_state", 42)),
            n_jobs=int(params.get("n_jobs", -1)),
        ),
    )

    X = train_df[feature_cols].to_numpy()
    T = train_df["T"].to_numpy()
    y = train_df["Y"].to_numpy()

    learner = BaseDRLearner(learner=base_learner)
    learner.fit(X=X, treatment=T, y=y)

    cate = learner.predict(X=X)

    # 落盘模型（可选）
    if model_path:
        try:
            dump(learner, model_path)
        except Exception:
            # 模型落盘失败不影响 cate 生成
            pass

    return np.asarray(cate).reshape(-1)


def _read_per_product_df(
    product_id: str,
    train_cfg: BundleTrainConfig,
    usecols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    读取单产品训练数据文件：每个产品一个文件，粒度为 cust_id-date。
    文件路径：{per_product_data_dir}/{per_product_file_pattern.format(product_id=...)}
    """
    import os

    fname = train_cfg.per_product_file_pattern.format(product_id=str(product_id))
    path = os.path.join(train_cfg.per_product_data_dir, fname)

    if train_cfg.per_product_file_format == "parquet":
        df = pd.read_parquet(path, columns=usecols)
    elif train_cfg.per_product_file_format == "csv":
        df = pd.read_csv(path, usecols=usecols)
    else:
        raise ValueError("per_product_file_format must be parquet or csv")

    # 基础校验
    for c in ["cust_id", "date", train_cfg.t_col, train_cfg.y_col]:
        if c not in df.columns:
            raise ValueError(f"per-product df for product={product_id} missing column: {c}")
    return df


def _build_bundle_train_df_from_per_product_files(
    bundle: BundleCandidate,
    train_cfg: BundleTrainConfig,
) -> pd.DataFrame:
    """
    生产：从 bundle 内各单产品文件构造组合训练集（cust_id-date 粒度）

    输入（每个产品文件）列为：
    - cust_id, date, X..., T, Y

    输出列为：
    - cust_id, date, X..., T, Y
    其中：
    - T 为 T_bundle = AND(T_i)
    - Y 为按 cust_id-date 聚合的 mean(Y_i)（通常应一致）
    - X：默认直接取 bundle.base_product 的 X（因为不同产品的特征表通常相同；若你实际是“产品特征不同”，需要扩展为 concat/交叉特征）
    """
    if not train_cfg.feature_cols:
        raise ValueError("prod mode requires BundleTrainConfig.feature_cols (X columns list)")

    gkey = ["cust_id", "date"]
    usecols = list(dict.fromkeys(gkey + train_cfg.feature_cols + [train_cfg.t_col, train_cfg.y_col]))

    # 1) 先读取 base 产品作为特征来源（默认）
    base_pid = bundle.base_product or bundle.products[0]
    base_df = _read_per_product_df(product_id=str(base_pid), train_cfg=train_cfg, usecols=usecols).copy()
    base_df.rename(columns={train_cfg.t_col: f"{train_cfg.t_col}__{base_pid}", train_cfg.y_col: f"{train_cfg.y_col}__{base_pid}"}, inplace=True)

    merged = base_df

    # 2) 依次 merge 其它产品的 T/Y（只 merge 两列，节省内存）
    for pid in bundle.products:
        pid = str(pid)
        if pid == str(base_pid):
            continue
        df2 = _read_per_product_df(product_id=pid, train_cfg=train_cfg, usecols=gkey + [train_cfg.t_col, train_cfg.y_col]).copy()
        df2.rename(columns={train_cfg.t_col: f"{train_cfg.t_col}__{pid}", train_cfg.y_col: f"{train_cfg.y_col}__{pid}"}, inplace=True)
        merged = merged.merge(df2, on=gkey, how="inner")

    # 3) 构造 T_bundle / Y
    t_cols = [c for c in merged.columns if c.startswith(f"{train_cfg.t_col}__")]
    y_cols = [c for c in merged.columns if c.startswith(f"{train_cfg.y_col}__")]

    merged["T"] = merged[t_cols].min(axis=1).astype(np.int8)
    merged["Y"] = merged[y_cols].mean(axis=1)

    # 4) 列裁剪：保留 key + X + T + Y
    out_cols = gkey + train_cfg.feature_cols + ["T", "Y"]
    out = merged[out_cols].copy()

    # 缺失处理
    out[train_cfg.feature_cols] = out[train_cfg.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return out


def ensure_bundle_cate(
    eval_df: pd.DataFrame,
    bundle_eval_df: pd.DataFrame,
    bundle: BundleCandidate,
    train_cfg: BundleTrainConfig,
    features_df: Optional[pd.DataFrame] = None,
    synthesize_cate_mode: str = "min",
) -> pd.DataFrame:
    """
    给 bundle_eval_df 补齐 cate：

    - debug：用 synthesize_bundle_cate() 合成（仅调试）
    - prod ：读取缓存 cate；没有则训练 DRLearner -> 推理 -> 落盘 -> merge 回 bundle_eval_df

    约束：
    - prod 模式必须提供 train_cfg.feature_cols（以及必要时的 features_df+merge_keys），否则无法训练
    """
    if train_cfg.mode not in {"debug", "prod"}:
        raise ValueError("BundleTrainConfig.mode must be one of: debug, prod")

    if train_cfg.mode == "debug":
        return synthesize_bundle_cate(eval_df=eval_df, bundle_eval_df=bundle_eval_df, bundle=bundle, mode=synthesize_cate_mode)

    # prod
    paths = _bundle_artifact_paths(train_cfg, bundle.bundle_id)
    cached = None if train_cfg.force_retrain else _load_cached_bundle_cate(paths["cate_path"], bundle_eval_df, train_cfg)
    if cached is not None:
        return cached

    # 训练数据准备（两种来源）：
    # A) 推荐：从“每个产品独立落盘文件”构造组合训练集（你的真实生产形态）
    # B) 兼容：外部传入 features_df，merge 到 bundle_eval_df（适用于你以后把特征做成统一表的情况）
    if features_df is None:
        train_df = _build_bundle_train_df_from_per_product_files(bundle=bundle, train_cfg=train_cfg)
        feature_cols = list(train_cfg.feature_cols or [])
    else:
        df = bundle_eval_df.copy()
        df = _merge_features_if_needed(df, features_df=features_df, train_cfg=train_cfg)

        feature_cols = train_cfg.feature_cols
        if not feature_cols:
            raise ValueError("prod mode requires BundleTrainConfig.feature_cols")
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"prod mode missing feature cols in training df: {missing}")

        used_cols = ["cust_id", "date", "T", "Y"] + feature_cols
        train_df = df[used_cols].copy()
        train_df[feature_cols] = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    cate = _train_and_predict_drlearner(
        train_df=train_df,
        feature_cols=feature_cols,
        train_cfg=train_cfg,
        model_path=paths["model_path"],
    )
    cate = cate.astype(np.float32) if train_cfg.cate_float32 else cate.astype(np.float64)

    out = bundle_eval_df.copy()
    out["cate"] = cate

    # 落盘 cate（只存 join keys + cate，避免重复存 T/Y）
    gkey = ["cust_id", "date"]
    cate_out = out[gkey + ["cate"]].copy()

    if train_cfg.prefer_parquet:
        cate_out.to_parquet(paths["cate_path"], index=False)
    else:
        cate_out.to_csv(paths["cate_path"], index=False)

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
# v3（DuckDB）: 从单产品 parquet 生成 bundle parquet（debug：cate=min/mean/sum）
# =========================

def _duckdb_connect(db_path: Optional[str] = None):
    import duckdb

    con = duckdb.connect(database=(db_path or ":memory:"))
    con.execute("PRAGMA threads=4;")
    con.execute("PRAGMA enable_progress_bar=false;")
    return con


def _parquet_glob(parquet_dir: str) -> str:
    from pathlib import Path

    p = Path(parquet_dir).resolve().as_posix()
    return f"{p}/**/*.parquet"


def build_bundle_eval_parquet_duckdb_debug(
    *,
    single_parquet_dir: str,
    out_bundle_parquet_dir: str,
    bundle: BundleCandidate,
    cate_mode: str = "min",
    min_bundle_support_rows: int = 0,
    duckdb_path: Optional[str] = None,
) -> str:
    """
    直接在 DuckDB 里从单产品 eval parquet 生成该 bundle 的 eval parquet（hive 分区，product_id=bundle_id）。

    输出 parquet schema（满足 v3 REQUIRED_COLUMNS）：
    - cust_id, product_id(=bundle_id), date, cate, T, Y

    debug cate 口径（只为跑通/研究，非生产）：
    - min/mean/sum : 对 bundle 内单品 cate 聚合
    """
    import os
    from pathlib import Path

    os.makedirs(out_bundle_parquet_dir, exist_ok=True)
    out_part_dir = os.path.join(out_bundle_parquet_dir, f"product_id={bundle.bundle_id}")
    os.makedirs(out_part_dir, exist_ok=True)
    out_path = os.path.join(out_part_dir, "part-00000.parquet")

    prods = [str(p) for p in bundle.products]
    prod_list_sql = ",".join([f"'{p}'" for p in prods])

    if cate_mode == "min":
        cate_expr = "MIN(cate)"
    elif cate_mode == "mean":
        cate_expr = "AVG(cate)"
    elif cate_mode == "sum":
        cate_expr = "SUM(cate)"
    else:
        raise ValueError(f"unknown cate_mode: {cate_mode}")

    con = _duckdb_connect(duckdb_path)
    glob = _parquet_glob(single_parquet_dir)

    # 1) 聚合成 bundle 粒度（cust_id,date）
    # 2) min_support：对 treated(T=1) 的样本量做门槛（用 HAVING）
    sql = f"""
    WITH sub AS (
      SELECT
        cust_id,
        date,
        MIN(T) AS T_bundle,
        AVG(Y) AS Y_bundle,
        {cate_expr} AS cate_bundle
      FROM read_parquet('{glob}', hive_partitioning=1)
      WHERE CAST(product_id AS VARCHAR) IN ({prod_list_sql})
      GROUP BY cust_id, date
    ),
    filtered AS (
      SELECT *
      FROM sub
      {"WHERE 1=1" if min_bundle_support_rows <= 0 else f"WHERE 1=1"}
    )
    SELECT
      cust_id,
      '{bundle.bundle_id}' AS product_id,
      date,
      cate_bundle AS cate,
      T_bundle AS T,
      Y_bundle AS Y
    FROM filtered
    """

    # treated 支持度门槛：需要二次聚合得到 treated_n（避免全表回读）
    if min_bundle_support_rows > 0:
        sql = f"""
        WITH sub AS (
          SELECT
            cust_id,
            date,
            MIN(T) AS T_bundle,
            AVG(Y) AS Y_bundle,
            {cate_expr} AS cate_bundle
          FROM read_parquet('{glob}', hive_partitioning=1)
          WHERE CAST(product_id AS VARCHAR) IN ({prod_list_sql})
          GROUP BY cust_id, date
        ),
        stats AS (
          SELECT SUM(CASE WHEN T_bundle=1 THEN 1 ELSE 0 END) AS treated_n
          FROM sub
        )
        SELECT
          s.cust_id,
          '{bundle.bundle_id}' AS product_id,
          s.date,
          s.cate_bundle AS cate,
          s.T_bundle AS T,
          s.Y_bundle AS Y
        FROM sub s
        CROSS JOIN stats t
        WHERE t.treated_n >= {int(min_bundle_support_rows)}
        """

    # 直接 COPY 到 hive 分区目录
    out_path_posix = Path(out_path).resolve().as_posix()
    con.execute(f"COPY ({sql}) TO '{out_path_posix}' (FORMAT PARQUET);")
    con.close()

    return out_path


def run_bundle_mining_backtest_v3_debug(
    *,
    single_parquet_dir: str = "output/backtest_output_v2/eval_parquet",
    out_root: str = "backtest_output_bundle_v3",
    mining_cfg: Optional[BundleMiningConfig] = None,
    product_cfg_for_bundle: Optional[ProductDecisionConfig] = None,
    customer_cfg: Optional[CustomerDecisionConfig] = None,
    safety_cfg: Optional[SafetyConfig] = None,
    backtest_cfg: Optional[BacktestConfig] = None,
    cate_mode: str = "min",
    duckdb_path: Optional[str] = None,
) -> Dict[str, object]:
    """
    v3 debug：从单产品 eval parquet 生成 bundle parquet（cate 合成），然后跑 run_backtest_v3 并输出 v3 报告。
    """
    import os

    mining_cfg = mining_cfg or BundleMiningConfig()

    _assert_bundle_out_dir_safe(out_root, arg_name="out_root")
    os.makedirs(out_root, exist_ok=True)
    bundle_parquet_dir = os.path.join(out_root, "eval_parquet_bundle")
    _assert_bundle_out_dir_safe(bundle_parquet_dir, arg_name="bundle_parquet_dir")
    os.makedirs(bundle_parquet_dir, exist_ok=True)

    # 1) 单品评估（用于候选生成）
    single_product_eval_df = evaluate_products_duckdb(
        parquet_dir=single_parquet_dir,
        product_config=ProductDecisionConfig(),
        external_metrics_df=None,
        duckdb_path=duckdb_path,
    )

    # 2) 候选
    bundles = generate_bundle_candidates(single_product_eval_df, cfg=mining_cfg)
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

    kept: List[BundleCandidate] = []
    for b in bundles:
        build_bundle_eval_parquet_duckdb_debug(
            single_parquet_dir=single_parquet_dir,
            out_bundle_parquet_dir=bundle_parquet_dir,
            bundle=b,
            cate_mode=cate_mode,
            min_bundle_support_rows=mining_cfg.min_bundle_support_rows,
            duckdb_path=duckdb_path,
        )
        kept.append(b)

    # 3) bundle 回测（v3）
    bundle_result = run_backtest_v3(
        parquet_dir=bundle_parquet_dir,
        external_metrics_df=None,
        product_config=product_cfg_for_bundle or ProductDecisionConfig(
            min_support_samples=max(50, int(mining_cfg.min_bundle_support_rows)),
            max_negative_uplift_ratio=0.55,
            top_ratio=0.2,
            enable_calibration=True,
        ),
        customer_config=customer_cfg or CustomerDecisionConfig(min_cate=0.0, top_k_per_customer=1),
        safety_config=safety_cfg or SafetyConfig(min_customer_expected_gain=0.0),
        backtest_config=backtest_cfg or BacktestConfig(),
        duckdb_path=duckdb_path,
        enable_single_day_reco=True,
    )

    # 4) synergy/incremental（小表 pandas merge）
    bundle_product_eval_df = bundle_result["product_eval_df"].copy()
    bundle_product_eval_df["product_id"] = _as_str_product_id(bundle_product_eval_df["product_id"])

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
                "synergy_score": compute_synergy_score(r, product_eval_df=single_product_eval_df, bundle=b),
                "incremental_to_base": compute_incremental_to_base(r, product_eval_df=single_product_eval_df, bundle=b),
            }
        )
    bundle_metrics_df = pd.DataFrame(rows)
    bundle_product_eval_df = bundle_product_eval_df.merge(
        bundle_metrics_df, left_on="product_id", right_on="bundle_id", how="left"
    ).drop(columns=["bundle_id"], errors="ignore")

    # 5) 报告
    bundle_result_with_metrics = dict(bundle_result)
    bundle_result_with_metrics["product_eval_df"] = bundle_product_eval_df
    report_path = os.path.join(out_root, "backtest_report_bundle_v3.md")
    render_business_report_v3(bundle_result_with_metrics, out_path=report_path)

    return {
        "single_product_eval_df": single_product_eval_df,
        "bundle_candidates_df": bundle_candidates_df,
        "bundle_metrics_df": bundle_metrics_df,
        "bundle_result": bundle_result_with_metrics,
        "bundle_parquet_dir": bundle_parquet_dir,
        "report_path": report_path,
    }


def run_bundle_mining_backtest_v3_prod(
    *,
    bundle_parquet_dir: str = "backtest_output_bundle_v3/eval_parquet_bundle",
    out_root: str = "backtest_output_bundle_v3",
    product_cfg_for_bundle: Optional[ProductDecisionConfig] = None,
    customer_cfg: Optional[CustomerDecisionConfig] = None,
    safety_cfg: Optional[SafetyConfig] = None,
    backtest_cfg: Optional[BacktestConfig] = None,
    duckdb_path: Optional[str] = None,
) -> Dict[str, object]:
    """
    v3 prod：直接对“已经训练/推理好的 bundle eval parquet”做回测评估并输出报告。
    """
    import os

    _assert_bundle_out_dir_safe(out_root, arg_name="out_root")
    _assert_bundle_out_dir_safe(bundle_parquet_dir, arg_name="bundle_parquet_dir")

    os.makedirs(out_root, exist_ok=True)
    bundle_result = run_backtest_v3(
        parquet_dir=bundle_parquet_dir,
        external_metrics_df=None,
        product_config=product_cfg_for_bundle or ProductDecisionConfig(),
        customer_config=customer_cfg or CustomerDecisionConfig(),
        safety_config=safety_cfg or SafetyConfig(),
        backtest_config=backtest_cfg or BacktestConfig(),
        duckdb_path=duckdb_path,
        enable_single_day_reco=True,
    )

    report_path = os.path.join(out_root, "backtest_report_bundle_v3.md")
    render_business_report_v3(bundle_result, out_path=report_path)

    return {"bundle_result": bundle_result, "bundle_parquet_dir": bundle_parquet_dir, "report_path": report_path}


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
    train_cfg: Optional[BundleTrainConfig] = None,
    features_df: Optional[pd.DataFrame] = None,
    synthesize_cate_mode: str = "min",
) -> Dict[str, pd.DataFrame]:
    """
    旧版（pandas in-memory）debug/研究入口：用于小数据快速跑通逻辑。
    大数据 v3 推荐使用：run_bundle_mining_backtest_v3_debug() / run_bundle_mining_backtest_v3_prod()。
    """
    mining_cfg = mining_cfg or BundleMiningConfig()
    train_cfg = train_cfg or BundleTrainConfig(mode="debug")
    raise RuntimeError(
        "run_bundle_mining_backtest() is pandas-based (small data) and is deprecated in v3 migration. "
        "Use run_bundle_mining_backtest_v3_debug/prod."
    )

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
    # 性能优化：先把 eval_df 裁剪到本次 bundles 涉及的产品集合，避免每个 bundle 都扫全量
    needed_products = set()
    for b in bundles:
        needed_products.update(map(str, b.products))

    # 仅保留必要列（避免中间表过大）
    base_cols = ["cust_id", "product_id", "date", "T", "Y"]
    if "cate" in eval_df.columns:
        base_cols.append("cate")
    base_cols = [c for c in base_cols if c in eval_df.columns]

    df_small = eval_df[base_cols].copy()
    df_small["product_id"] = _as_str_product_id(df_small["product_id"])
    df_small = df_small[df_small["product_id"].isin(needed_products)].copy()

    bundle_eval_frames: List[pd.DataFrame] = []
    kept: List[BundleCandidate] = []

    for b in bundles:
        if not mining_cfg.and_mode:
            raise NotImplementedError("当前版本只实现 AND bundle；如需 OR/序列，可继续扩展。")

        bdf = build_bundle_eval_df_and_mode(
            eval_df=df_small,
            bundle=b,
            min_bundle_support_rows=mining_cfg.min_bundle_support_rows,
        )
        if bdf.empty:
            continue

        # 关键：按 mode 决定 cate 来源
        bdf = ensure_bundle_cate(
            eval_df=df_small,
            bundle_eval_df=bdf,
            bundle=b,
            train_cfg=train_cfg,
            features_df=features_df,
            synthesize_cate_mode=synthesize_cate_mode,
        )

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

    raise RuntimeError(
        "run_bundle_mining_backtest() is pandas-based (small data). "
        "For v3 large-scale evaluation use run_bundle_mining_backtest_v3_debug/prod."
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


def run_bundle_mining_prod_from_files(
    product_eval_df: pd.DataFrame,
    mining_cfg: Optional[BundleMiningConfig] = None,
    product_cfg_for_bundle: Optional[ProductDecisionConfig] = None,
    customer_cfg: Optional[CustomerDecisionConfig] = None,
    safety_cfg: Optional[SafetyConfig] = None,
    backtest_cfg: Optional[BacktestConfig] = None,
    train_cfg: Optional[BundleTrainConfig] = None,
) -> Dict[str, pd.DataFrame]:
    """
    生产入口：完全不依赖 eval_df。

    输入：
    - product_eval_df：单产品评估结果（用于候选生成：recommend_all / recommend_targeted / product_score / tag）
    - train_cfg：必须为 mode="prod"，并配置 per_product_data_dir / feature_cols / 文件格式等
      每个产品一个文件，列为 [cust_id, date, X..., T, Y]

    流程：
    1) 生成 bundle candidates
    2) 对每个 bundle：
       - 从 per-product 文件构造 bundle 训练集（cust_id,date,X,T_bundle,Y）
       - 训练/预测 cate_bundle（带缓存：bundle_artifacts/{bundle_id}/cate.parquet）
       - 构造 bundle_eval_long：cust_id,date,product_id=bundle_id,cate,T,Y
    3) concat 所有 bundle_eval_long，复用 run_backtest() 输出评估与策略结果
    """
    mining_cfg = mining_cfg or BundleMiningConfig()
    train_cfg = train_cfg or BundleTrainConfig(mode="prod")
    if train_cfg.mode != "prod":
        raise ValueError("run_bundle_mining_prod_from_files requires train_cfg.mode='prod'")

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

    bundle_eval_frames: List[pd.DataFrame] = []
    kept: List[BundleCandidate] = []

    for b in bundles:
        # 构造训练集（从文件读取），并训练/预测 cate
        train_df = _build_bundle_train_df_from_per_product_files(bundle=b, train_cfg=train_cfg)

        # 支持度门槛（AND 的 treated 样本不足时不训练）
        if mining_cfg.min_bundle_support_rows > 0:
            treated_n = int((train_df["T"] == 1).sum())
            if treated_n < int(mining_cfg.min_bundle_support_rows):
                continue

        # 用 bundle_eval_df 的形态承载 cate（与 backtest 输入对齐）
        bundle_eval_df = train_df[["cust_id", "date", "T", "Y"]].copy()
        bundle_eval_df["product_id"] = b.bundle_id
        bundle_eval_df["cate"] = np.nan
        bundle_eval_df = bundle_eval_df[["cust_id", "product_id", "date", "cate", "T", "Y"]]

        bundle_eval_df = ensure_bundle_cate(
            eval_df=bundle_eval_df,  # prod 下不会用到 eval_df 的内容（只用于函数签名兼容）
            bundle_eval_df=bundle_eval_df,
            bundle=b,
            train_cfg=train_cfg,
            features_df=None,
            synthesize_cate_mode="min",
        )

        bundle_eval_frames.append(bundle_eval_df)
        kept.append(b)

    if not bundle_eval_frames:
        return {
            "bundle_candidates_df": bundle_candidates_df,
            "bundle_product_eval_df": pd.DataFrame(),
            "bundle_metrics_df": pd.DataFrame(),
        }

    bundle_eval_all = pd.concat(bundle_eval_frames, axis=0, ignore_index=True)

    # 复用回测（已迁移到 v3 的 parquet + duckdb 版本）
    raise RuntimeError(
        "run_bundle_mining_prod_from_files still uses legacy run_backtest(). "
        "Please use the new v3 flow: (1) run bundle_cate_train_pipeline_v3.py to produce bundle parquet "
        "(2) run run_bundle_mining_backtest_v3_prod(bundle_parquet_dir=...)."
    )

    bundle_product_eval_df = bundle_result["product_eval_df"].copy()
    bundle_product_eval_df["product_id"] = _as_str_product_id(bundle_product_eval_df["product_id"])

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
                "overlap_ratio_top": np.nan,  # 生产不一定有单产品 cate 长表，这里先留空
                "incremental_to_base": compute_incremental_to_base(r, product_eval_df=product_eval_df, bundle=b),
            }
        )

    bundle_metrics_df = pd.DataFrame(rows)

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
    # v3 Debug demo：从单产品 v3 eval parquet 生成 bundle parquet（cate=min 合成）并跑 v3 回测报告
    result = run_bundle_mining_backtest_v3_debug(
        single_parquet_dir="output/backtest_output_v2/eval_parquet",
        out_root="backtest_output_bundle_v3",
        mining_cfg=BundleMiningConfig(
            and_mode=True,
            max_bundle_size=3,
            top_n_base=6,
            top_n_booster=6,
            min_bundle_support_rows=300,
        ),
        cate_mode="min",
        duckdb_path="backtest_output_bundle_v3/duckdb_tmp.db",
    )

    print("bundle_candidates:", result["bundle_candidates_df"].shape)
    print("bundle_metrics:", result["bundle_metrics_df"].shape)
    print("bundle_parquet_dir:", result["bundle_parquet_dir"])
    print("report_path:", result["report_path"])


