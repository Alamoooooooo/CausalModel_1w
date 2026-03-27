from __future__ import annotations

"""
debug_full_pipeline_v3.py
=========================

总调试入口：按顺序串联 v3 全流程的“数据生成 -> 主回测 -> bundle mining -> bundle 训练/导出”。

用途
----
- 作为本仓库的“一键总调试入口”
- 方便在本地快速验证 v3 链路是否完整可跑
- 保留现有核心脚本不变，仅通过编排层统一调试流程

默认流程
--------
1. 生成/刷新中文 product_id 的 v3 demo eval parquet 到 `output/backtest_output_v3/eval_parquet`
2. 运行 `src/backtest_full_pipeline_v3.py` 的全流程回测
3. 运行 `src/bundle_mining_pipeline.py` 的 bundle 调试入口
4. 运行 `src/bundle_cate_train_pipeline_v3.py` 的 bundle 训练/导出入口
5. 汇总每一步输出路径与状态

运行示例
--------
- `python src/debug_full_pipeline_v3.py`
- `python src/debug_full_pipeline_v3.py --skip_bundle_train`
- `python src/debug_full_pipeline_v3.py --skip_bundle_mining`
- `python src/debug_full_pipeline_v3.py --out_dir output/backtest_output_v3_test/debug_full`

说明
----
- 本文件不重写核心算法，只负责顺序调用现有脚本。
- 所有 `product_id` 按字符串处理，支持中文，如：活动A、会员权益包、优惠券包。
- 所有结果默认写入 `output/backtest_output_v3/...` 或 `output/backtest_output_v3_test/...`，保持目录结构一致，仅变更根路径前缀。
"""

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DebugFullPipelineConfig:
    """总调试入口配置。

    字段说明
    --------
    - out_dir: 总调试输出目录。默认写入 `output/backtest_output_v3_test/debug_full`。
    - parquet_dir: v3 单品 eval parquet 根目录。默认 `output/backtest_output_v3/eval_parquet`。
    - bundle_output_dir: bundle 输出目录。默认 `output/backtest_output_v3/eval_parquet_bundle`。
    - run_export: 是否执行 v3 demo 数据生成。
    - run_backtest: 是否执行主回测。
    - run_bundle_mining: 是否执行 bundle mining 调试入口（默认关闭）。
    - run_bundle_train: 是否执行 bundle 训练/导出入口（默认关闭）。
    - skip_on_failure: 某一步失败后是否继续后续步骤。
    """

    out_dir: str = "output/backtest_output_v3_test/debug_full"
    parquet_dir: str = "output/backtest_output_v3/eval_parquet"
    bundle_output_dir: str = "output/backtest_output_v3/eval_parquet_bundle"
    run_export: bool = True
    run_backtest: bool = True
    run_bundle_mining: bool = False
    run_bundle_train: bool = False
    skip_on_failure: bool = True


def _run_step(name: str, command: list[str], *, cwd: Optional[str] = None, skip_on_failure: bool = True) -> int:
    """执行单个步骤并返回退出码。

    参数
    ----
    name:
        步骤名，用于日志输出，如 `export_demo`、`backtest_full`。
    command:
        要执行的命令列表，直接传给 `subprocess.run()`。
    cwd:
        子进程工作目录，默认使用当前仓库根目录。
    skip_on_failure:
        若为 True，命令失败只打印错误并继续；否则抛出异常终止总流程。

    返回
    ----
    int
        子进程退出码。0 表示成功。
    """
    print(f"[debug-full] START {name}: {' '.join(command)}")
    try:
        completed = subprocess.run(command, cwd=cwd, check=False)
        rc = int(completed.returncode)
    except Exception as exc:
        print(f"[debug-full] ERROR {name}: {exc}")
        if skip_on_failure:
            return 1
        raise

    if rc == 0:
        print(f"[debug-full] DONE {name}")
    else:
        print(f"[debug-full] FAIL {name} (rc={rc})")
        if not skip_on_failure:
            raise RuntimeError(f"step {name} failed with rc={rc}")
    return rc


def build_demo_data(cfg: DebugFullPipelineConfig) -> int:
    """生成/刷新 v3 demo eval parquet。

    说明
    ----
    - 实际调用 `src/export_eval_parquet_v2.py --demo`，默认写入 `output/backtest_output_v3/eval_parquet`
    - 这是总流程的第一步，负责准备后续主回测与 bundle 调试所需的单品长表数据
    - demo 数据会包含中文 `product_id`，用于验证中文路径与 Hive 分区兼容性
    """
    return _run_step(
        "build_demo_data",
        [sys.executable, "src/export_eval_parquet_v2.py", "--demo", "--out-dir", cfg.parquet_dir],
        skip_on_failure=cfg.skip_on_failure,
    )


def run_full_backtest(cfg: DebugFullPipelineConfig) -> int:
    """运行 v3 主回测全流程。

    说明
    ----
    - 通过 `src/backtest_full_pipeline_v3.py` 的命令行入口执行
    - 默认读取 `cfg.parquet_dir`，输出写入 `cfg.out_dir/full_backtest`
    - 若仓库中 `backtest_full_pipeline_v3.py` 的默认行为与当前路径不同，可通过命令行参数覆盖
    - 这是验证单品回测链路是否完整的核心步骤
    """
    out_dir = str(Path(cfg.out_dir) / "full_backtest")
    return _run_step(
        "run_full_backtest",
        [sys.executable, "src/backtest_full_pipeline_v3.py", "--mode", "full", "--parquet_dir", cfg.parquet_dir, "--out_dir", out_dir, "--run_tests"],
        skip_on_failure=cfg.skip_on_failure,
    )


def run_bundle_mining(cfg: DebugFullPipelineConfig) -> int:
    """运行 bundle mining 调试入口。

    说明
    ----
    - 调用 `src/bundle_mining_pipeline.py`
    - 入口内部会优先从 `output/backtest_output_v3/eval_parquet` 读取单品长表，并使用中文 `product_id` 示例
    - 该步骤用于验证组合候选生成、bundle eval 构造、以及与主回测函数的衔接
    - 当前 demo 下若候选不足，可能会出现 bundle 结果为 0 行，但只要流程不报错，就说明入口可运行
    """
    return _run_step(
        "run_bundle_mining",
        [sys.executable, "src/bundle_mining_pipeline.py"],
        skip_on_failure=cfg.skip_on_failure,
    )


def run_bundle_train(cfg: DebugFullPipelineConfig) -> int:
    """运行 bundle 训练/导出入口。

    说明
    ----
    - 调用 `src/bundle_cate_train_pipeline_v3.py --demo`
    - 默认示例会读取 `output/backtest_output_v3/eval_parquet`，输出写入 `output/backtest_output_v3/eval_parquet_bundle`
    - 该步骤用于验证 bundle 训练、CATE 推理、以及 Hive 分区 bundle parquet 导出链路
    - 若本地缺少真实 `feature_cols`，脚本会打印提示并退出；这属于预期行为
    """
    return _run_step(
        "run_bundle_train",
        [sys.executable, "src/bundle_cate_train_pipeline_v3.py", "--demo"],
        skip_on_failure=cfg.skip_on_failure,
    )


def main() -> None:
    """总调试入口。

    运行顺序
    --------
    1. 生成 demo 数据
    2. 跑主回测
    3. 跑 bundle mining
    4. 跑 bundle 训练/导出

    参数
    ----
    - --out_dir: 总调试结果目录
    - --parquet_dir: 单品 v3 eval parquet 根目录
    - --run_bundle_mining: 显式开启 bundle mining；默认不执行。
    - --run_bundle_train: 显式开启 bundle 训练/导出；默认不执行。
    - --skip_export: 跳过 demo 数据生成。
    - --skip_backtest: 跳过主回测。
    - --fail_fast: 任一步失败后立即停止。
    """
    parser = argparse.ArgumentParser(description="Full debug pipeline v3 orchestrator")
    parser.add_argument("--out_dir", default="output/backtest_output_v3_test/debug_full")
    parser.add_argument("--parquet_dir", default="output/backtest_output_v3/eval_parquet")
    parser.add_argument("--run_bundle_mining", action="store_true")
    parser.add_argument("--run_bundle_train", action="store_true")
    parser.add_argument("--skip_export", action="store_true")
    parser.add_argument("--skip_backtest", action="store_true")
    parser.add_argument("--fail_fast", action="store_true")
    args = parser.parse_args()

    cfg = DebugFullPipelineConfig(
        out_dir=args.out_dir,
        parquet_dir=args.parquet_dir,
        run_export=not args.skip_export,
        run_backtest=not args.skip_backtest,
        run_bundle_mining=args.run_bundle_mining,
        run_bundle_train=args.run_bundle_train,
        skip_on_failure=not args.fail_fast,
    )

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    results = {}
    if cfg.run_export:
        results["export"] = build_demo_data(cfg)
    if cfg.run_backtest:
        results["backtest"] = run_full_backtest(cfg)
    if cfg.run_bundle_mining:
        results["bundle_mining"] = run_bundle_mining(cfg)
    if cfg.run_bundle_train:
        results["bundle_train"] = run_bundle_train(cfg)

    print("[debug-full] finished")
    for k, v in results.items():
        print(f"[debug-full] {k} => rc={v}")


if __name__ == "__main__":
    # 运行示例对应关系：
    # 1) 默认主线（数据生成 + 主回测）：
    #    python src/debug_full_pipeline_v3.py --out_dir output/backtest_output_v3_test/debug_full --parquet_dir output/backtest_output_v3/eval_parquet
    # 2) 全流程（数据生成 + 主回测 + bundle mining + bundle train）：
    #    python src/debug_full_pipeline_v3.py --out_dir output/backtest_output_v3_test/debug_full --parquet_dir output/backtest_output_v3/eval_parquet --run_bundle_mining --run_bundle_train
    # 3) 只跑数据生成：
    #    python src/debug_full_pipeline_v3.py --out_dir output/backtest_output_v3_test/debug_full --parquet_dir output/backtest_output_v3/eval_parquet --skip_backtest
    # 4) 只跑主回测：
    #    python src/debug_full_pipeline_v3.py --out_dir output/backtest_output_v3_test/debug_full --parquet_dir output/backtest_output_v3/eval_parquet --skip_export
    # 5) 只跑 bundle mining：
    #    python src/debug_full_pipeline_v3.py --out_dir output/backtest_output_v3_test/debug_full --parquet_dir output/backtest_output_v3/eval_parquet --skip_export --skip_backtest --run_bundle_mining
    # 6) 只跑 bundle train：
    #    python src/debug_full_pipeline_v3.py --out_dir output/backtest_output_v3_test/debug_full --parquet_dir output/backtest_output_v3/eval_parquet --skip_export --skip_backtest --run_bundle_train
    # 中文产品示例：活动A、活动B、会员权益包、优惠券包
    # 说明：该入口只做编排，不重写业务逻辑。
    main()
