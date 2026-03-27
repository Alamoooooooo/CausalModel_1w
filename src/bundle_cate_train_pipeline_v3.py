from __future__ import annotations

"""
bundle_cate_train_pipeline_v3.py

目标：为 bundle（产品组合）独立训练 CATE 模型，并产出符合 backtest_full_pipeline_v3 的 eval parquet。

输入（生产形态）：
- per_product_data_dir 下每个产品一个文件（parquet/csv），粒度为 (cust_id, date)
- 列至少包含：cust_id, date, X..., T, Y
- product_id 可以是中文字符串，文件名/目录名会直接使用原始字符串，DuckDB/Parquet 兼容中文路径。

输出：
- bundle_parquet_dir（hive partition）：{out_dir}/product_id={bundle_id}/part-*.parquet
- schema：cust_id, product_id(=bundle_id), date, cate, T, Y

说明（大数据友好设计）：
- 组合样本构造尽量使用 DuckDB，避免 pandas 大表 merge 爆内存
- 模型训练（causalml DRLearner）依然是内存训练：
  - E 级数据必须“抽样/分期/分桶”训练，这里提供 sample/日期窗口参数
- 该脚本只负责“训练+产出 bundle eval parquet”，评估/报告请用 bundle_mining_pipeline.py 的 v3 prod 入口：
  run_bundle_mining_backtest_v3_prod(bundle_parquet_dir=...)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import argparse
import numpy as np
import pandas as pd

try:
    from causalml.inference.meta import BaseDRLearner  # type: ignore
except Exception:  # pragma: no cover
    BaseDRLearner = None  # type: ignore


# -------------------------
# DuckDB helpers
# -------------------------

def _duckdb_connect(db_path: Optional[str] = None):
    import duckdb

    con = duckdb.connect(database=(db_path or ":memory:"))
    con.execute("PRAGMA threads=4;")
    con.execute("PRAGMA enable_progress_bar=false;")
    return con


# -------------------------
# Config
# -------------------------

@dataclass
class BundleTrainJobConfig:
    per_product_data_dir: str = "per_product_data"
    per_product_file_pattern: str = "{product_id}.parquet"  # or "{product_id}.csv"
    per_product_file_format: str = "parquet"  # parquet|csv

    # 必须列名（与单品一致）
    cust_id_col: str = "cust_id"
    date_col: str = "date"
    t_col: str = "T"
    y_col: str = "Y"

    # 特征列（X）
    feature_cols: Optional[List[str]] = None

    # 输出 bundle eval parquet（hive partition）
    out_bundle_parquet_dir: str = "output/backtest_output_v3/eval_parquet_bundle"

    # DRLearner 模型与 cate 缓存目录（可选）
    artifacts_dir: str = "bundle_artifacts_v3"
    force_retrain: bool = False

    # 大数据训练控制
    # - date_window: 只用某些 date 训练（如最近 N 天），None 表示不过滤
    # - sample_frac: 采样比例（0~1），None 表示不采样
    # - sample_limit: 采样行数上限（优先级高于 sample_frac），None 表示不限制
    date_window: Optional[Tuple[str, str]] = None  # ("2026-01-01","2026-02-01")
    sample_frac: Optional[float] = None
    sample_limit: Optional[int] = 3_000_000

    # bundle treated/outcome 口径（AND）
    # T_bundle = MIN(T_i)
    # Y_bundle = AVG(Y_i)
    # X_bundle = base 产品的 X（假设跨产品一致）
    # 注意：如果将来你要“顺序处理/OR/强依赖”，这里需要扩展
    and_mode: bool = True

    # learner 参数（可按你单品训练的参数替换）
    learner_params: Optional[Dict[str, object]] = None


# -------------------------
# I/O helpers
# -------------------------

def _per_product_path(cfg: BundleTrainJobConfig, product_id: str) -> str:
    return str(Path(cfg.per_product_data_dir) / cfg.per_product_file_pattern.format(product_id=str(product_id)))


def _ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def _assert_bundle_out_dir_safe(path: str, *, arg_name: str) -> None:
    """
    防呆：禁止 bundle 训练/落盘写入单品 backtest 输出目录，避免覆盖/污染单品结果。
    固定约束：bundle 产物必须落在 output/backtest_output_v3/ 或 backtest_output_bundle_v3/ 下。

    - 允许：output/backtest_output_v3、backtest_output_bundle_v3 及其子目录
    - 禁止：backtest_output_v2 / backtest_output/
    """
    p = (path or "").replace("\\", "/").lower()
    allowed = ("output/backtest_output_v3", "backtest_output_bundle_v3")
    if not any(token in p for token in allowed):
        raise ValueError(
            f"[bundle-output-safety] {arg_name} must be under 'output/backtest_output_v3/' or 'backtest_output_bundle_v3/'. Got: {path}"
        )
    forbidden = ["backtest_output_v2", "backtest_output/"]
    hit = [x for x in forbidden if x in p]
    if hit:
        raise ValueError(
            f"[bundle-output-safety] {arg_name} points to a single-product output directory (hits={hit}). "
            f"Refuse to write. Got: {path}"
        )


def _bundle_id_from_products(products: Sequence[str]) -> str:
    ps = tuple(sorted(str(p).strip() for p in products))
    if len(ps) == 2:
        return f"bundle_and__{ps[0]}__{ps[1]}"
    return "bundle_and__" + "__".join(ps)


def _default_demo_cfg() -> BundleTrainJobConfig:
    return BundleTrainJobConfig(
        per_product_data_dir="output/backtest_output_v3/eval_parquet",
        per_product_file_pattern="product_id={product_id}/part-*.parquet",
        per_product_file_format="parquet",
        feature_cols=None,
        out_bundle_parquet_dir="output/backtest_output_v3/eval_parquet_bundle",
        sample_limit=200_000,
    )


def _print_demo_usage() -> None:
    print("示例命令：python src/bundle_cate_train_pipeline_v3.py")
    print("示例命令：python src/bundle_cate_train_pipeline_v3.py --demo")
    print("示例命令：python src/bundle_cate_train_pipeline_v3.py --bundle_products 活动A 活动B --base_product 活动A --feature_cols x1 x2 x3")
    print("示例命令：python src/bundle_cate_train_pipeline_v3.py --bundle_products 活动A 活动B --base_product 活动A --per_product_data_dir output/backtest_output_v3/eval_parquet --out_bundle_parquet_dir output/backtest_output_v3/eval_parquet_bundle")
    print("中文 product_id 示例：活动A、活动B、会员权益包、优惠券包")


def _run_demo() -> None:
    cfg = _default_demo_cfg()
    _assert_bundle_out_dir_safe(cfg.out_bundle_parquet_dir, arg_name="cfg.out_bundle_parquet_dir")

    print("bundle_cate_train_pipeline_v3 demo 已启动。")
    print(f"输入目录: {cfg.per_product_data_dir}")
    print(f"输出目录: {cfg.out_bundle_parquet_dir}")
    print("请先确认 per_product_data_dir 下存在按 product_id 分区的 parquet 文件。")
    print("例如：output/backtest_output_v3/eval_parquet/product_id=活动A/part-00000.parquet")

    demo_bundle_products = ["活动A", "活动B"]
    demo_base_product = "活动A"

    try:
        out_path = train_one_bundle_and_write_eval(
            bundle_products=demo_bundle_products,
            base_product=demo_base_product,
            cfg=cfg,
            duckdb_path="bundle_train_tmp.duckdb",
        )
        print(f"已写出 bundle eval parquet: {out_path}")
    except Exception as exc:
        print(f"demo 执行失败（通常是因为本地没有对应输入数据或缺少 feature_cols）：{exc}")
        print("如需正式运行，请设置 cfg.feature_cols，并确保输入 parquet 已存在。")
        _print_demo_usage()


# -------------------------
# Training set construction (DuckDB)
# -------------------------

def build_bundle_train_df_duckdb(
    *,
    bundle_products: Sequence[str],
    base_product: Optional[str],
    cfg: BundleTrainJobConfig,
    duckdb_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    用 DuckDB 读取 per-product 文件并 join 到 (cust_id,date) 粒度，生成训练数据。
    默认：X 取 base_product 的 feature_cols；其它产品只 join T/Y。
    """
    if not cfg.feature_cols:
        raise ValueError("cfg.feature_cols must be provided for training")

    if not cfg.and_mode:
        raise NotImplementedError("only AND mode supported for now")

    base_pid = str(base_product or bundle_products[0])

    con = _duckdb_connect(duckdb_path)

    # base 表：cust_id,date,X...,T,Y
    base_path = _per_product_path(cfg, base_pid).replace("\\", "/")
    if cfg.per_product_file_format == "parquet":
        base_rel = f"read_parquet('{base_path}')"
    elif cfg.per_product_file_format == "csv":
        base_rel = f"read_csv_auto('{base_path}')"
    else:
        raise ValueError("per_product_file_format must be parquet or csv")

    # 只选择必要列，减少 IO
    x_cols = ", ".join([f"b.{c}" for c in cfg.feature_cols])
    sql = f"""
    WITH base AS (
      SELECT
        b.{cfg.cust_id_col} AS cust_id,
        b.{cfg.date_col} AS date,
        {x_cols},
        b.{cfg.t_col} AS T__{base_pid},
        b.{cfg.y_col} AS Y__{base_pid}
      FROM {base_rel} b
    )
    SELECT * FROM base
    """
    con.execute("CREATE OR REPLACE TEMP VIEW merged AS " + sql)
    con.execute("CREATE OR REPLACE TEMP VIEW merged_src AS SELECT * FROM merged")

    # join 其它产品的 T/Y
    for pid in bundle_products:
        pid = str(pid)
        if pid == base_pid:
            continue
        p_path = _per_product_path(cfg, pid).replace("\\", "/")
        if cfg.per_product_file_format == "parquet":
            rel = f"read_parquet('{p_path}')"
        else:
            rel = f"read_csv_auto('{p_path}')"

        join_sql = f"""
        CREATE OR REPLACE TEMP VIEW merged AS
        SELECT
          m.*,
          p.{cfg.t_col} AS T__{pid},
          p.{cfg.y_col} AS Y__{pid}
        FROM merged m
        INNER JOIN {rel} p
          ON m.cust_id = p.{cfg.cust_id_col}
         AND m.date    = p.{cfg.date_col}
        """
        con.execute(join_sql)

    # 训练过滤（日期窗口）
    where_clauses: List[str] = []
    if cfg.date_window is not None:
        start, end = cfg.date_window
        where_clauses.append(f"date >= '{start}' AND date <= '{end}'")

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    # 生成 T/Y
    t_cols = [f"T__{str(pid)}" for pid in [base_pid] + [str(p) for p in bundle_products if str(p) != base_pid]]
    y_cols = [f"Y__{str(pid)}" for pid in [base_pid] + [str(p) for p in bundle_products if str(p) != base_pid]]

    t_expr = "LEAST(" + ", ".join(t_cols) + ")"
    y_expr = "(" + " + ".join(y_cols) + ") / " + str(len(y_cols))

    final_sql = f"""
    SELECT
      cust_id,
      date,
      {", ".join(cfg.feature_cols)},
      CAST({t_expr} AS INTEGER) AS T,
      CAST({y_expr} AS DOUBLE) AS Y
    FROM merged
    {where_sql}
    """

    # 采样：优先 limit，其次 frac
    if cfg.sample_limit is not None:
        final_sql += f"\nLIMIT {int(cfg.sample_limit)}"
    elif cfg.sample_frac is not None:
        final_sql = f"SELECT * FROM ({final_sql}) USING SAMPLE BERNOULLI({float(cfg.sample_frac) * 100.0});"

    _assert_bundle_out_dir_safe(cfg.out_bundle_parquet_dir, arg_name="cfg.out_bundle_parquet_dir")

    df = con.execute(final_sql).df()
    con.close()

    df[cfg.feature_cols] = df[cfg.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


# -------------------------
# Model train + predict
# -------------------------

def train_and_predict_drlearner(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    cfg: BundleTrainJobConfig,
) -> np.ndarray:
    if BaseDRLearner is None:
        raise ImportError("This script requires causalml. Please install causalml to run bundle training.")

    from sklearn.ensemble import RandomForestRegressor

    params = cfg.learner_params or {}
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
    cate = np.asarray(learner.predict(X=X)).reshape(-1)
    return cate


# -------------------------
# Write bundle eval parquet (hive)
# -------------------------

def write_bundle_eval_parquet(
    *,
    train_df: pd.DataFrame,
    cate: np.ndarray,
    bundle_id: str,
    cfg: BundleTrainJobConfig,
) -> str:
    """
    写出 v3 所需 eval parquet（hive partition）。
    """
    _assert_bundle_out_dir_safe(cfg.out_bundle_parquet_dir, arg_name="cfg.out_bundle_parquet_dir")
    _ensure_dir(cfg.out_bundle_parquet_dir)
    out_dir = Path(cfg.out_bundle_parquet_dir) / f"product_id={bundle_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out = train_df[["cust_id", "date", "T", "Y"]].copy()
    out["product_id"] = bundle_id
    out["cate"] = cate.astype(np.float32)
    out = out[["cust_id", "product_id", "date", "cate", "T", "Y"]]

    path = out_dir / "part-00000.parquet"
    out.to_parquet(path, index=False)
    return str(path)


# -------------------------
# Main API
# -------------------------

def train_one_bundle_and_write_eval(
    *,
    bundle_products: Sequence[str],
    base_product: Optional[str],
    cfg: BundleTrainJobConfig,
    duckdb_path: Optional[str] = None,
) -> str:
    """
    训练单个 bundle，并写出 bundle eval parquet（hive partition）。
    返回写出的 parquet 文件路径。
    """
    bundle_id = _bundle_id_from_products(bundle_products)
    train_df = build_bundle_train_df_duckdb(
        bundle_products=bundle_products,
        base_product=base_product,
        cfg=cfg,
        duckdb_path=duckdb_path,
    )

    cate = train_and_predict_drlearner(train_df, feature_cols=list(cfg.feature_cols or []), cfg=cfg)

    return write_bundle_eval_parquet(train_df=train_df, cate=cate, bundle_id=bundle_id, cfg=cfg)


if __name__ == "__main__":
    # 运行方式示例：
    # 中文单 bundle 示例（请替换成真实 product_id）：
    #   python src/bundle_cate_train_pipeline_v3.py --demo
    #   python src/bundle_cate_train_pipeline_v3.py --bundle_products 活动A 活动B --base_product 活动A --feature_cols x1 x2 x3
    #   python src/bundle_cate_train_pipeline_v3.py --bundle_products 会员权益包 优惠券包 --base_product 会员权益包 --per_product_data_dir output/backtest_output_v3/eval_parquet --out_bundle_parquet_dir output/backtest_output_v3/eval_parquet_bundle
    #
    # 说明：
    # - demo 默认读取 output/backtest_output_v3/eval_parquet 下的单品评估 parquet
    # - demo 默认写入 output/backtest_output_v3/eval_parquet_bundle
    # - 若本地没有输入数据，脚本会给出提示，不会破坏已有逻辑

    _print_demo_usage()
    _run_demo()
