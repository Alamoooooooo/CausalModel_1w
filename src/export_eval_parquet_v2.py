from __future__ import annotations

"""
export_eval_parquet_v2.py

从 backtest_full_pipeline_v2 中抽取“生成 eval parquet”的最小可测试脚本。
用途：
- 直接替换原有 `backtest_output_v2/eval_parquet` 目录内容，避免改动其他代码引用路径
- 验证中文 product_id 作为原始字符串时，hive 分区 parquet 是否可正常写出/读取
- 保持与 v2 的数据结构一致：cust_id, product_id, date, cate, T, Y

v3 默认 demo 目录已切换到：
- `backtest_output_v3/eval_parquet`

默认 demo 生成模拟数据；也支持直接把已有 DataFrame 分块写成分区 parquet。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

import argparse
import shutil

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["cust_id", "product_id", "date", "cate", "T", "Y"]


@dataclass
class EvalDFSimConfig:
    n_customers: int = 1000
    n_products: int = 4
    n_dates: int = 2
    start_date: str = "2026-01-01"
    freq: str = "D"
    chunk_rows: int = 0
    use_category: bool = False
    use_float32: bool = True
    random_state: int = 42


def validate_eval_df(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"eval_df 缺少必要字段: {missing}")


def simulate_evaldf(cfg: Optional[EvalDFSimConfig] = None) -> "pd.DataFrame | Iterator[pd.DataFrame]":
    cfg = cfg or EvalDFSimConfig()
    rng = np.random.default_rng(cfg.random_state)

    cust = np.arange(cfg.n_customers, dtype=np.int32)
    prod = np.array([f"活动{i+1}" for i in range(cfg.n_products)], dtype=object)
    dates = pd.date_range(cfg.start_date, periods=cfg.n_dates, freq=cfg.freq)
    total_rows = int(cfg.n_customers) * int(cfg.n_products) * int(cfg.n_dates)
    f_dtype = np.float32 if cfg.use_float32 else np.float64

    def _build_chunk(start: int, size: int) -> pd.DataFrame:
        idx = np.arange(start, start + size, dtype=np.int64)
        cust_idx = (idx % cfg.n_customers).astype(np.int32)
        tmp = idx // cfg.n_customers
        prod_idx = (tmp % cfg.n_products).astype(np.int32)
        date_idx = (tmp // cfg.n_products).astype(np.int32)

        product_id = prod[prod_idx]
        cate = rng.normal(0, 1, size=size).astype(f_dtype)
        ps = np.clip(rng.uniform(0.05, 0.95, size=size).astype(f_dtype), 0.01, 0.99)
        T = rng.binomial(1, ps).astype(np.int8)
        tau = np.tanh(cate).astype(f_dtype)
        Y = (tau + rng.normal(0, 1, size=size).astype(f_dtype) + T.astype(f_dtype) * tau).astype(f_dtype)

        df = pd.DataFrame(
            {
                "cust_id": cust[cust_idx],
                "product_id": product_id,
                "date": dates.values[date_idx],
                "cate": cate,
                "T": T,
                "Y": Y,
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


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def write_evaldf_parquet_partitioned(
    eval_iter: Iterable[pd.DataFrame],
    out_dir: str,
    partition_cols: Sequence[str] = ("product_id",),
    replace_existing: bool = True,
) -> None:
    import pyarrow as pa
    import pyarrow.dataset as ds

    out_path = Path(out_dir)
    if replace_existing and out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    format_ = ds.ParquetFileFormat()
    write_options = format_.make_write_options(compression="zstd")
    partitioning = ds.partitioning(
        pa.schema([(c, pa.string()) for c in partition_cols]),
        flavor="hive",
    )

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


def _default_out_dir() -> str:
    return "output/backtest_output_v3/eval_parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export eval parquet for v3 demo/testing and replace existing directory contents.")
    parser.add_argument("--out-dir", default=_default_out_dir(), help="target eval parquet dir")
    parser.add_argument("--demo", action="store_true", help="generate demo eval df with Chinese product_id")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    # 示例命令：
    # 1) 生成 v3 demo parquet 到默认目录：
    #    python src/export_eval_parquet_v2.py --demo
    # 2) 指定输出目录：
    #    python src/export_eval_parquet_v2.py --demo --out-dir backtest_output_v3/eval_parquet
    # 中文 product_id 示例会使用“活动一 / 活动二 / 活动三 / 活动四”，用于验证分区目录与读取兼容性。
    cfg = EvalDFSimConfig(n_customers=2000, n_products=20, n_dates=20, chunk_rows=0, use_category=False)
    eval_iter = simulate_evaldf(cfg)

    if isinstance(eval_iter, pd.DataFrame):
        eval_iter = [eval_iter]

    write_evaldf_parquet_partitioned(eval_iter, out_dir=str(out_dir), partition_cols=("product_id",), replace_existing=True)
    print("wrote parquet to:", out_dir)


if __name__ == "__main__":
    main()