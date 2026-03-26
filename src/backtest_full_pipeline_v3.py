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

新增：
- 单日可推荐子集链路（as_of_date + lookback_days）
- 仅对固定日期做线上投放口径的推荐与评估
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


def render_business_report_v3(
    result: Dict[str, pd.DataFrame],
    out_path: str = "backtest_output_v3/backtest_report_v3.md",
    top_products: int = 20,
    top_reco_rows: int = 50,
) -> str:
    import os
    from datetime import datetime

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    product_eval_df = result.get("product_eval_df", pd.DataFrame()).copy()
    customer_reco_df = result.get("customer_reco_df", pd.DataFrame()).copy()
    reco_empirical_eval_df = result.get("reco_empirical_eval_df", pd.DataFrame()).copy()
    policy_gain_df = result.get("policy_gain_df", pd.DataFrame()).copy()
    temporal_df = result.get("temporal_df", pd.DataFrame()).copy()
    temporal_reco_df = result.get("temporal_reco_df", pd.DataFrame()).copy()
    ope_df = result.get("ope_df", pd.DataFrame()).copy()

    eligible_eval_df = result.get("eligible_eval_df", pd.DataFrame()).copy()
    eligible_product_eval_df = result.get("eligible_product_eval_df", pd.DataFrame()).copy()
    eligible_customer_reco_df = result.get("eligible_customer_reco_df", pd.DataFrame()).copy()
    eligible_reco_empirical_eval_df = result.get("eligible_reco_empirical_eval_df", pd.DataFrame()).copy()
    eligible_policy_gain_df = result.get("eligible_policy_gain_df", pd.DataFrame()).copy()
    single_day_as_of_date_df = result.get("single_day_as_of_date", pd.DataFrame()).copy()

    n_rows = int(customer_reco_df.shape[0]) if not customer_reco_df.empty else 0
    n_customers = int(customer_reco_df["cust_id"].nunique()) if ("cust_id" in customer_reco_df.columns and not customer_reco_df.empty) else 0
    n_products = int(product_eval_df["product_id"].nunique()) if ("product_id" in product_eval_df.columns and not product_eval_df.empty) else 0

    if not product_eval_df.empty and "recommendation_decision" in product_eval_df.columns:
        n_reco_products = int(product_eval_df["recommendation_decision"].isin(["recommend_all", "recommend_targeted"]).sum())
        n_reco_products_all = int((product_eval_df["recommendation_decision"] == "recommend_all").sum())
        n_reco_products_targeted = int((product_eval_df["recommendation_decision"] == "recommend_targeted").sum())
    else:
        n_reco_products = 0
        n_reco_products_all = 0
        n_reco_products_targeted = 0

    reco_uplift = (
        float(reco_empirical_eval_df["empirical_uplift"].iloc[0])
        if (reco_empirical_eval_df is not None and not reco_empirical_eval_df.empty and "empirical_uplift" in reco_empirical_eval_df.columns)
        else np.nan
    )

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
        prod_show = product_eval_df.sort_values(["recommendation_decision", "product_score"], ascending=[True, False])[cols].head(top_products)
    else:
        prod_show = pd.DataFrame()

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

    policy_show = policy_gain_df if (policy_gain_df is not None and not policy_gain_df.empty) else pd.DataFrame(columns=["top_pct", "n", "uplift_gain"])
    temporal_show = temporal_df if (temporal_df is not None and not temporal_df.empty) else pd.DataFrame(columns=["date", "model_ate", "empirical_uplift", "treated_n", "control_n"])
    temporal_reco_show = temporal_reco_df if (temporal_reco_df is not None and not temporal_reco_df.empty) else pd.DataFrame(columns=["date", "reco_model_ate", "reco_empirical_uplift", "treated_n", "control_n"])
    ope_show = ope_df if (ope_df is not None and not ope_df.empty) else pd.DataFrame(columns=["policy", "ipw_value", "dr_value", "ipw_ok", "dr_ok", "ipw_error", "dr_error"])

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

    md.append("### 2.4 如何解读产品层结果（好/坏信号）\n")
    md.append(
        "\n".join(
            [
                "**好信号（更可能可上线）**：",
                "- `ate>0` 且 `empirical_uplift>0`（方向一致，且真实结果口径为正）",
                "- `qini/auuc` 较高（说明排序“会挑人”，定向价值更大）",
                "- `top_uplift_lift`、`top_vs_rest_gap` 明显 > 0（Top 人群显著更好）",
                "- `negative_uplift_ratio` 低（风险小）",
                "- `sample_size`/`n_customer` 充足（估计更稳健）",
                "- `pass_rate` 高，且 `recommendation_decision` 为 `recommend_all` 或 `recommend_targeted`",
                "",
                "**坏信号（需要观察/调参/重训）**：",
                "- `ate` 与 `empirical_uplift` 长期反向（优先排查口径/漂移/校准/混杂）",
                "- `negative_uplift_ratio` 高（容易误伤用户）",
                "- 样本量过小仍靠前（可能是噪声，建议提高 `min_support_samples`）",
                "- `top_uplift_lift` 很低/为负（说明挑人能力不足）",
                "",
            ]
        )
        + "\n"
    )

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

    md.append("### 3.3 如何解读客户层推荐（好/坏信号）\n")
    md.append(
        "\n".join(
            [
                "**好信号**：",
                "- 推荐清单里 `adjusted_cate` 大多为正，且头部记录（rank=1）明显更高",
                "- 同一客户 Top1 的 `recommend_score` 明显高于 Top3（排序分有区分度）",
                "- 推荐主要来自产品层 `recommend_all/recommend_targeted` 池（策略一致、可控）",
                "",
                "**坏信号**：",
                "- 大量推荐记录 `adjusted_cate<=0` 仍被输出（通常是 `min_cate` 太低、校准异常或数据噪声）",
                "- 推荐过度集中在少数产品，但这些产品 `empirical_uplift`/风险指标一般（可能权重偏“强产品”或门禁太松）",
                "- 客户内 rank 的分数差异很小（说明 score 信号弱，需调权重或加更强的门禁）",
                "",
            ]
        )
        + "\n"
    )

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

    md.append("### 4.3 如何解读策略收益曲线（好/坏信号）\n")
    md.append(
        "\n".join(
            [
                "**好信号**：",
                "- `uplift_gain` 随 `top_pct` 增大应逐步下降并最终趋近 0（top 1% > top 2% > ...）",
                "- top 小比例（如 1%/2%/5%）`uplift_gain` 明显 > 0（说明排序有用）",
                "- 可用“曲线拐点”确定触达规模：从 1% 增到 5% 收益仍高，但到 30% 变平，说明扩大触达会稀释效果",
                "",
                "**坏信号**：",
                "- 曲线不单调或 top 小比例≈0/为负（说明推荐分与真实 Y 关系弱，需调权重/重训/排查口径）",
                "",
            ]
        )
        + "\n"
    )

    md.append("## 五、时间稳定性（Temporal Stability）\n")
    md.append(
        "\n".join(
            [
                "### 5.1 时间稳定性在讲什么？",
                "- 目的：检查模型/推荐效果是否随时间漂移；若某天明显变差，可能是数据分布/活动/人群变化导致。",
                "",
                "### 5.2 全量时间稳定性（全量产品/全量样本）字段解释",
                "- `date`：日期。",
                "- `model_ate`：当日平均 cate（全量样本的模型视角平均处理效应）。",
                "- `empirical_uplift`：当日经验 uplift（全量样本 treated 平均Y - control 平均Y）。",
                "- `treated_n`/`control_n`：当日 treated/control 样本量。",
                "- 说明：该口径适合做“全局健康度/漂移监控”，可能会被大量无效产品稀释。",
                "",
            ]
        )
    )
    md.append(_table(temporal_show) + "\n")

    md.append("### 5.3 推荐子集时间稳定性（仅最终推荐清单）\n")
    md.append(
        "\n".join(
            [
                "- 目的：只看最终输出的 `customer_reco_df`（经过产品门禁/客户门禁后的推荐清单），避免全量无因果产品对总体的稀释。",
                "- `reco_model_ate`：推荐清单内（默认用 `adjusted_cate`）的当日均值。",
                "- `reco_empirical_uplift`：推荐清单内 treated 平均Y - control 平均Y。",
                "- `treated_n`/`control_n`：推荐清单内 treated/control 样本量。",
                "",
            ]
        )
    )
    md.append(_table(temporal_reco_show) + "\n")

    md.append("### 5.4 如何解读时间稳定性（好/坏信号）\n")
    md.append(
        "\n".join(
            [
                "**好信号**：",
                "- `empirical_uplift` 比较平稳，且与 `model_ate` 大体同向变化",
                "- `treated_n/control_n` 稳定且不太小（样本量充足时结论更可信）",
                "",
                "**坏信号**：",
                "- `empirical_uplift` 断崖式波动且样本量并不小（更可能是分布漂移/活动变化/数据口径变化）",
                "- 某些日期 treated/control 极不平衡（经验 uplift 会变得不稳定）",
                "",
            ]
        )
        + "\n"
    )

    if not eligible_eval_df.empty or not eligible_customer_reco_df.empty or not eligible_policy_gain_df.empty:
        md.append("## 六、单日可推荐子集（线上投放口径）\n")
        md.append(
            "\n".join(
                [
                    "### 6.1 口径说明",
                    "- 该链路只在固定 `as_of_date` 上生成推荐，不对全量历史每天重复计算。",
                    "- eligible 规则：在 `[as_of_date - lookback_days, as_of_date - 1]` 内，`cust_id + product_id` 从未出现过 `T=1`，则视为可推荐。",
                    "- 该子集更贴近真实线上投放：只给近期未达标的客户推荐产品。",
                    "- 注意：单日子集可能没有真实 `T/Y`；若缺失，则只输出推荐清单，不做 treated/control uplift sanity check。",
                    "- `T` 在这里表示是否达标，`Y` 表示 `t~t+30` 的活期存款差额；不要把它误解成曝光/点击回流。",
                    "",
                    "### 6.2 当前单日配置",
                    f"- `as_of_date`：{_fmt(single_day_as_of_date_df.iloc[0]['as_of_date']) if not single_day_as_of_date_df.empty and 'as_of_date' in single_day_as_of_date_df.columns else '-'}",
                    f"- `lookback_days`：{_fmt(single_day_as_of_date_df.iloc[0]['lookback_days'], 0) if not single_day_as_of_date_df.empty and 'lookback_days' in single_day_as_of_date_df.columns else '-'}",
                    "",
                    "### 6.3 单日子集输出表",
                    "- `eligible_eval_df`：单日可推荐 eval 子集",
                    "- `eligible_product_eval_df`：单日子集上的产品评估",
                    "- `eligible_customer_reco_df`：单日子集上的客户推荐清单",
                    "- `eligible_reco_empirical_eval_df`：单日子集 treated-control uplift sanity check（若无 T/Y 则为空）",
                    "- `eligible_policy_gain_df`：单日子集策略收益曲线",
                    "",
                    "### 6.4 如何解读单日子集结果",
                    "- 若 eligible 样本数过少，单日结果波动会增大，需要适当调大 `lookback_days` 或检查数据覆盖。",
                    "- 若单日 `eligible_policy_gain_df` 在 top 小比例上仍明显为正，说明当天线上推荐的优先排序是有效的。",
                    "- 若单日子集与全量回测差异很大，通常说明全量中存在较多重复达标/无效样本，线上投放应优先参考单日子集。",
                    "",
                ]
            )
            + "\n"
        )

    md.append("## 七、离线策略价值评估（OPE）\n")
    md.append(
        "\n".join(
            [
                "### 7.1 离线评估（OPE）在讲什么？",
                "- 目的：在不能线上 A/B 的情况下，估计“如果按新策略触达，整体期望 Y 会是多少”。",
                "- 说明：OPE 通常需要 `ps`（倾向得分）以及可能需要 `mu0/mu1`（潜在结果预测）。若缺列，本报告会在 `ope_df` 中说明原因。",
                "",
                "### 7.2 字段解释",
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

    md.append("### 7.3 如何解读 OPE（若未计算可忽略）\n")
    md.append(
        "\n".join(
            [
                "- 只有当 `ipw_ok/dr_ok=True` 时，才建议把 `ipw_value/dr_value` 纳入判断。",
                "- 若缺少 `ps` 或 `mu0/mu1` 导致无法计算（报告会写明原因），请忽略该部分，不影响其它章节对“好/坏”的判断。",
                "",
            ]
        )
        + "\n"
    )

    md.append("## 八、ps / mu0 / mu1 是什么？怎么计算？（给数据准备同学）\n")
    md.append(
        "\n".join(
            [
                "### 8.1 ps（propensity score，倾向得分）",
                "- 定义：ps(x) = P(T=1 | X=x)，即在历史策略下样本被触达/处理的概率。",
                "- 用途：IPW/DR 等离线策略评估需要用 ps 来纠偏历史触达偏差。",
                "- 计算：用历史数据训练一个二分类模型预测 T（特征只能用触达前可见特征），输出 predict_proba 的概率作为 ps，并进行 clipping（例如 0.01~0.99）。",
                "",
                "### 8.2 mu0 / mu1（潜在结果预测）",
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

    md.append("## 九、配置参数与调参指南（Config & Tuning Guide）\n")
    md.append(
        "\n".join(
            [
                "本章节解释：产品门禁（哪些产品能推荐）、客户层输出（每个客户推荐什么）、安全控制（更保守/更激进）、以及排序权重（更偏个性化/更偏强产品/更偏安全）。",
                "",
                "### 9.0 本次运行参数快照（便于复现）",
                "- 说明：以下参数来自本次运行时传入 `run_backtest_v3()` 的 config。",
                "",
                "#### ProductDecisionConfig",
                f"- min_ate={_fmt(getattr(result.get('product_config', None), 'min_ate', None))}",
                f"- min_empirical_uplift={_fmt(getattr(result.get('product_config', None), 'min_empirical_uplift', None))}",
                f"- min_qini={_fmt(getattr(result.get('product_config', None), 'min_qini', None))}",
                f"- min_auuc={_fmt(getattr(result.get('product_config', None), 'min_auuc', None))}",
                f"- min_top_uplift_lift={_fmt(getattr(result.get('product_config', None), 'min_top_uplift_lift', None))}",
                f"- max_negative_uplift_ratio={_fmt(getattr(result.get('product_config', None), 'max_negative_uplift_ratio', None))}",
                f"- min_support_samples={_fmt(getattr(result.get('product_config', None), 'min_support_samples', None), 0)}",
                f"- top_ratio={_fmt(getattr(result.get('product_config', None), 'top_ratio', None))}",
                f"- enable_targeted_reco={getattr(result.get('product_config', None), 'enable_targeted_reco', None)}",
                f"- targeted_top_ratio={_fmt(getattr(result.get('product_config', None), 'targeted_top_ratio', None))}",
                f"- min_targeted_lift={_fmt(getattr(result.get('product_config', None), 'min_targeted_lift', None))}",
                f"- allow_targeted_when_ate_negative={getattr(result.get('product_config', None), 'allow_targeted_when_ate_negative', None)}",
                f"- enable_calibration={getattr(result.get('product_config', None), 'enable_calibration', None)}",
                "",
                "#### CustomerDecisionConfig",
                f"- min_cate={_fmt(getattr(result.get('customer_config', None), 'min_cate', None))}",
                f"- top_k_per_customer={_fmt(getattr(result.get('customer_config', None), 'top_k_per_customer', None), 0)}",
                f"- min_product_pass_rate={_fmt(getattr(result.get('customer_config', None), 'min_product_pass_rate', None))}",
                f"- customer_weight_col={getattr(result.get('customer_config', None), 'customer_weight_col', None)}",
                "",
                "#### SafetyConfig",
                f"- enable_product_blacklist_gate={getattr(result.get('safety_config', None), 'enable_product_blacklist_gate', None)}",
                f"- enable_customer_safe_filter={getattr(result.get('safety_config', None), 'enable_customer_safe_filter', None)}",
                f"- min_customer_expected_gain={_fmt(getattr(result.get('safety_config', None), 'min_customer_expected_gain', None))}",
                f"- max_customer_negative_share={_fmt(getattr(result.get('safety_config', None), 'max_customer_negative_share', None))}",
                "",
                "#### BacktestConfig",
                f"- policy_bins={getattr(result.get('backtest_config', None), 'policy_bins', None)}",
                f"- ps_clip_low={_fmt(getattr(result.get('backtest_config', None), 'ps_clip_low', None))}",
                f"- ps_clip_high={_fmt(getattr(result.get('backtest_config', None), 'ps_clip_high', None))}",
                "",
                "### 9.1 产品门禁与决策参数（ProductDecisionConfig）怎么理解/怎么调？",
                "- `min_ate`：ATE 下限。调大→更保守（产品池更小、平均效果更稳）；调小→更激进（覆盖更大、风险更高）。",
                "- `min_empirical_uplift`：经验 uplift 下限。调大→更贴近历史真实结果、减少“模型幻觉”；调小→更多依赖模型 cate/排序能力。",
                "- `min_qini` / `min_auuc`：排序能力门槛。调大→更强调“挑对人”（异质性强、定向价值高）；调小→更强调“平均有效”。",
                "- `min_top_uplift_lift`：Top 人群比整体更强的门槛。调大→更偏强异质性产品；调小→更偏均匀有效产品。",
                "- `max_negative_uplift_ratio`：负 uplift 占比上限（风险闸门）。调小→更安全但覆盖变小；调大→更激进但更可能误伤用户。",
                "- `min_support_samples`：最小样本量。调大→更稳健但冷门产品容易被拒；调小→覆盖更多但估计更抖。",
                "- `top_ratio`：Top uplift 相关指标（top_uplift_lift/top_vs_rest_gap）的 Top 比例。调小→更关注极头部；调大→更关注更大范围的好人群。",
                "- `enable_targeted_reco`：是否启用定向推荐通道。关掉→只有 recommend_all/watchlist/reject。",
                "- `targeted_top_ratio`：定向推荐开放的人群比例。调小→更保守（更少人被定向触达）；调大→更激进。",
                "- `min_targeted_lift`：定向推荐的 lift 门槛。调大→更尖、更少产品进入 targeted；调小→更多产品可 targeted。",
                "- `allow_targeted_when_ate_negative`：允许 ATE<0 但 Top 很强的产品走 targeted。开→偏增长型；关→偏安全型。",
                "- `enable_calibration`：是否用 empirical_uplift/ate 做尺度校准。开→更贴近历史结果口径；关→保持模型尺度（更稳定但可能与业务指标尺度不一致）。",
                "",
                "### 9.2 客户层输出参数（CustomerDecisionConfig）怎么理解/怎么调？",
                "- `min_cate`：客户-产品对的 uplift 门槛。调大→只推更确定增益的组合；调小→推荐更多但平均增益可能下降。",
                "- `top_k_per_customer`：每个客户最多推荐多少条。调大→覆盖更广但可能稀释；调小→更聚焦。",
                "- `min_product_pass_rate`：产品门禁通过率阈值（当前 v3 未用于强过滤，属于预留；若要生效可加到 SQL WHERE）。",
                "- `customer_weight_col`：客户权重列（当前 v3 未使用，预留）。",
                "",
                "### 9.3 安全与回测参数（SafetyConfig / BacktestConfig）怎么理解/怎么调？",
                "- `enable_product_blacklist_gate`：是否仅从 recommend_all/targeted 产品池中做候选。一般建议开启（更可控）。",
                "- `enable_customer_safe_filter` + `min_customer_expected_gain`：客户侧最低期望增益过滤。调大→更保守；调小→更激进。",
                "- `max_customer_negative_share`：客户层负收益容忍度（当前 v3 未显式使用，预留）。",
                "- `policy_bins`：policy curve 的切分点。更密→更细但更复杂；更稀→更易解释但粗糙。",
                "- `ps_clip_low/high`：ps 裁剪范围（仅 OPE）。clip 更紧→权重更稳定但偏差可能增大；clip 更松→方差更大（易出现极端权重）。",
                "",
                "### 9.4 排序权重（最常用调参旋钮）怎么调？",
                "- 产品排序 `product_score` 有两套权重（mass vs targeted）：",
                "  - 提高 `ate/empirical_uplift` 权重 → 更偏“平均收益更大”的产品",
                "  - 提高 `qini/auuc/top_uplift_lift` 权重 → 更偏“更会挑人/异质性强”的产品",
                "  - 提高 `(1 - negative_uplift_ratio)` 权重 → 更偏“更安全”的产品",
                "- 客户层排序 `recommend_score = 0.65*norm_adjusted_cate + 0.25*norm_product_score + 0.10*(1-norm_neg_ratio)`：",
                "  - 提高 adjusted_cate 权重 → 更个性化（更强调人-货匹配 uplift）",
                "  - 提高 product_score 权重 → 更偏强产品（少数强产品更容易被推给更多人）",
                "  - 提高安全项权重 → 更保守（减少可能负收益的人群）",
                "",
            ]
        )
        + "\n"
    )

    md.append("## 十、如何解读回测好坏（Interpretation Checklist）\n")
    md.append(
        "\n".join(
            [
                "本项目的“回测”是离线验证：用模型的 uplift（`cate`）决定“推给谁”，再用历史数据中的 `T/Y` 做 sanity check 与策略收益模拟。",
                "建议按 **必须过线 / 加分项 / 风险项** 三类信号来判断效果好坏。",
                "",
                "### 10.1 这套回测到底怎么测的？（口径说明）",
                "- **产品层（Product Level）**：对每个产品汇总 `cate` 得到 `ate`，并用历史 `T/Y` 计算 `empirical_uplift=mean(Y|T=1)-mean(Y|T=0)` 作为经验对照；再用 `cate` 排序构造 proxy 的 `qini/auuc/top_uplift_lift` 衡量“会不会挑人”。",
                "- **客户层（Customer Level）**：在可推荐产品池中，对每个客户按 `recommend_score` 排序取 Top-K，形成推荐清单 `customer_reco_df`。",
                "- **策略层（Policy Level）**：",
                "  - `policy_gain_df`：把推荐分从高到低取 top 1%/5%/…，看这些样本平均 `Y` 比全体平均 `Y` 高多少（当前实现口径）。",
                "  - `temporal_df`：按日期观察 `model_ate` 与 `empirical_uplift` 是否稳定，排查漂移。",
                "  - `ope_df`：当你提供 `ps/mu0/mu1` 时可做 IPW/DR 的离线策略评估（v3 默认不回读全量 parquet 做 OPE，因此会提示如何扩展）。",
                "",
                "### 10.2 必须过线（不然说明策略/模型基本不可用）",
                "1) **方向一致性**：进入推荐池（recommend_all/targeted）的产品，`ate` 与 `empirical_uplift` 不应长期大量反向。",
                "   - 若大量出现 `ate>0` 但 `empirical_uplift<0`：优先排查数据泄露、标签口径、分布漂移、校准过强、或 treated/control 结构性差异。",
                "2) **收益曲线形状**：`policy_gain_df.uplift_gain` 随 `top_pct` 增大应逐步下降并趋近 0；top 1%/2% 通常应明显高于 top 50%。",
                "   - 若 top 很小比例仍不如整体：排序几乎无效（或推荐分与真实收益无关）。",
                "3) **时间稳定性**：`temporal_df` 中 `empirical_uplift` 不应在样本量足够（treated_n/control_n 大）时出现断崖式变动。",
                "",
                "### 10.3 加分项（越多越好，说明更可上线）",
                "- 推荐池产品数量合理（不为 0，也不是几乎全量）。",
                "- `negative_uplift_ratio` 低（更安全）。",
                "- `top_uplift_lift`、`top_vs_rest_gap` 明显 > 0（说明模型“会挑人”，适合定向）。",
                "- `reco_empirical_eval_df.empirical_uplift` 为正且量级符合业务预期（推荐清单子集的经验 uplift sanity check）。",
                "",
                "### 10.4 风险项（看到要警惕，通常需要调参/加门禁/重训）",
                "- `negative_uplift_ratio` 高但仍被推荐：门禁太松/安全权重太低。",
                "- 小样本产品（sample_size/n_customer 很小）排到很前：可能噪声大，建议提高 `min_support_samples` 或加置信度判断。",
                "- `empirical_uplift` 与 `ate` 偏离很大：考虑关闭/调整校准（enable_calibration）、或按业务口径重新训练模型。",
                "- policy curve 不单调：考虑调整 `recommend_score` 权重、提高 `min_cate`、或把风险项权重调高。",
                "",
                "### 10.5 常用排查与调参建议（快速指引）",
                "- **想更保守、更安全**：降低 `max_negative_uplift_ratio`、提高 `min_support_samples`、提高客户侧 `min_cate` / `min_customer_expected_gain`、提高推荐分里安全项权重。",
                "- **想更激进、更覆盖**：放宽 `min_ate/min_empirical_uplift/min_qini`、降低 `min_cate`、提高 `top_k_per_customer`。",
                "- **想更强调“挑人”能力（定向更尖）**：提高 `min_qini/min_auuc/min_top_uplift_lift`，并调小 `targeted_top_ratio`（只给最头部人群）。",
                "",
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
                "- `temporal_df`：按 date 维度的 model_ate vs empirical_uplift（全量产品/全量样本）",
                "- `temporal_reco_df`：按 date 维度的 reco_model_ate vs reco_empirical_uplift（仅推荐清单子集）",
                "- `eligible_eval_df`：单日可推荐 eval 子集",
                "- `eligible_product_eval_df`：单日子集上的产品评估",
                "- `eligible_customer_reco_df`：单日子集上的客户推荐清单",
                "- `eligible_reco_empirical_eval_df`：单日子集 treated-control uplift（sanity check）",
                "- `eligible_policy_gain_df`：单日子集策略收益曲线",
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
    con = _duckdb_connect(duckdb_path)
    glob = _parquet_glob(parquet_dir)
    df = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{glob}', hive_partitioning=1) LIMIT 0;").df()
    con.close()
    return [str(x) for x in df["column_name"].tolist()]


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
      AVG(CASE WHEN cate < 0 THEN 1 ELSE 0 END) AS negative_uplift_ratio
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
        if cols:
            product_eval = product_eval.drop(columns=["qini", "auuc"], errors="ignore")
            product_eval = product_eval.merge(external_metrics_df[cols], on="product_id", how="left")

    if product_config.enable_calibration:
        product_eval["calibration_factor"] = product_eval.apply(
            lambda r: (r["empirical_uplift"] / r["ate"]) if (pd.notna(r["ate"]) and r["ate"] != 0) else 1.0,
            axis=1,
        )
        product_eval["calibration_factor"] = product_eval["calibration_factor"].replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.0, 5.0)
    else:
        product_eval["calibration_factor"] = 1.0

    product_eval["pass_ate"] = product_eval["ate"] > product_config.min_ate
    product_eval["pass_empirical"] = product_eval["empirical_uplift"] > product_config.min_empirical_uplift
    product_eval["pass_qini"] = product_eval["qini"] > product_config.min_qini
    product_eval["pass_auuc"] = product_eval["auuc"] > product_config.min_auuc
    product_eval["pass_top_lift"] = product_eval["top_uplift_lift"] > product_config.min_top_uplift_lift
    product_eval["pass_negative_risk"] = product_eval["negative_uplift_ratio"] <= product_config.max_negative_uplift_ratio
    product_eval["pass_support"] = product_eval["sample_size"] >= product_config.min_support_samples
    gate_cols = ["pass_ate", "pass_empirical", "pass_qini", "pass_auuc", "pass_top_lift", "pass_negative_risk", "pass_support"]
    product_eval["pass_rate"] = product_eval[gate_cols].mean(axis=1)

    product_eval["pass_targeted"] = (
        (product_config.enable_targeted_reco)
        & ((product_eval["ate"] < 0) if product_config.allow_targeted_when_ate_negative else True)
        & (product_eval["top_uplift_lift"] >= product_config.min_targeted_lift)
        & (product_eval["negative_uplift_ratio"] <= product_config.max_negative_uplift_ratio)
    )

    product_eval["recommendation_decision"] = np.select(
        [product_eval[gate_cols].all(axis=1), product_eval["pass_targeted"], product_eval["pass_ate"]],
        ["recommend_all", "recommend_targeted", "watchlist"],
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
    product_eval["product_score"] = np.where(product_eval["recommendation_decision"] == "recommend_targeted", score_targeted, score_mass)

    con.close()
    return product_eval.sort_values(["recommendation_decision", "product_score"], ascending=[True, False]).reset_index(drop=True)


def generate_recommendations_duckdb(
    parquet_dir: str,
    product_eval_df: pd.DataFrame,
    customer_config: Optional[CustomerDecisionConfig] = None,
    safety_config: Optional[SafetyConfig] = None,
    product_config: Optional[ProductDecisionConfig] = None,
    duckdb_path: Optional[str] = None,
) -> pd.DataFrame:
    customer_config = customer_config or CustomerDecisionConfig()
    safety_config = safety_config or SafetyConfig()
    product_config = product_config or ProductDecisionConfig()

    con = _duckdb_connect(duckdb_path)
    glob = _parquet_glob(parquet_dir)
    con.execute(f"CREATE OR REPLACE VIEW eval AS SELECT * FROM read_parquet('{glob}', hive_partitioning=1);")

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
                    SELECT AVG(CASE WHEN T=1 THEN Y ELSE NULL END) - AVG(CASE WHEN T=0 THEN Y ELSE NULL END)
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


def temporal_stability_reco_df(customer_reco_df: pd.DataFrame, score_col_for_model: str = "adjusted_cate") -> pd.DataFrame:
    if customer_reco_df is None or customer_reco_df.empty:
        return pd.DataFrame(columns=["date", "reco_model_ate", "reco_empirical_uplift", "treated_n", "control_n"])

    d = customer_reco_df.copy()
    if score_col_for_model not in d.columns:
        score_col_for_model = "cate" if "cate" in d.columns else d.columns[0]

    grouped = d.groupby("date", dropna=False)
    treated = d.loc[d["T"] == 1].groupby("date", dropna=False)["Y"].mean().rename("treated_mean")
    control = d.loc[d["T"] == 0].groupby("date", dropna=False)["Y"].mean().rename("control_mean")
    treated_n = d.assign(_is_treated=(d["T"] == 1).astype(int)).groupby("date", dropna=False)["_is_treated"].sum().rename("treated_n").astype(int)
    control_n = d.assign(_is_control=(d["T"] == 0).astype(int)).groupby("date", dropna=False)["_is_control"].sum().rename("control_n").astype(int)

    out = pd.DataFrame(
        {
            "date": grouped.size().index,
            "reco_model_ate": grouped[score_col_for_model].mean().values,
        }
    ).set_index("date")
    out = out.join(treated, how="left").join(control, how="left").join(treated_n, how="left").join(control_n, how="left")
    out["reco_empirical_uplift"] = out["treated_mean"] - out["control_mean"]
    out = out.drop(columns=["treated_mean", "control_mean"])
    return out.reset_index().sort_values("date").reset_index(drop=True)


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


def _infer_latest_date(parquet_dir: str, duckdb_path: Optional[str] = None) -> Optional[str]:
    con = _duckdb_connect(duckdb_path)
    glob = _parquet_glob(parquet_dir)
    con.execute(f"CREATE OR REPLACE VIEW eval AS SELECT * FROM read_parquet('{glob}', hive_partitioning=1);")
    row = con.execute("SELECT CAST(MAX(date) AS VARCHAR) AS latest_date FROM eval").fetchone()
    con.close()
    return row[0] if row and row[0] is not None else None


def build_eligible_eval_df(
    parquet_dir: str,
    as_of_date: str,
    lookback_days: int = 30,
    duckdb_path: Optional[str] = None,
) -> pd.DataFrame:
    con = _duckdb_connect(duckdb_path)
    glob = _parquet_glob(parquet_dir)
    con.execute(f"CREATE OR REPLACE VIEW eval AS SELECT * FROM read_parquet('{glob}', hive_partitioning=1);")
    cols = set(get_parquet_columns(parquet_dir, duckdb_path=duckdb_path))
    has_t = "T" in cols
    sql = f"""
    WITH history AS (
      SELECT *
      FROM eval
      WHERE date < DATE '{as_of_date}'
        AND date >= DATE '{as_of_date}' - INTERVAL {int(lookback_days)} DAY
    ),
    eligible_pairs AS (
      SELECT cust_id, product_id
      FROM history
      GROUP BY cust_id, product_id
      {"HAVING MAX(CASE WHEN T = 1 THEN 1 ELSE 0 END) = 0" if has_t else ""}
    )
    SELECT e.*
    FROM eval e
    INNER JOIN eligible_pairs p USING(cust_id, product_id)
    WHERE e.date = DATE '{as_of_date}'
    """
    df = con.execute(sql).df()
    con.close()
    return df


def run_backtest_v3(
    parquet_dir: str,
    external_metrics_df: Optional[pd.DataFrame] = None,
    product_config: Optional[ProductDecisionConfig] = None,
    customer_config: Optional[CustomerDecisionConfig] = None,
    safety_config: Optional[SafetyConfig] = None,
    backtest_config: Optional[BacktestConfig] = None,
    duckdb_path: Optional[str] = None,
    as_of_date: Optional[str] = None,
    lookback_days: int = 30,
    enable_single_day_reco: bool = False,
    mode: str = "both",
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

    single_day_result: Dict[str, pd.DataFrame] = {}
    run_full = mode in {"full", "both"}
    run_single_day = mode in {"single_day", "both"} and enable_single_day_reco
    if run_single_day:
        inferred_as_of_date = as_of_date or _infer_latest_date(parquet_dir, duckdb_path=duckdb_path)
        if inferred_as_of_date is None:
            eligible_eval_df = pd.DataFrame()
            eligible_product_eval_df = pd.DataFrame()
            eligible_customer_reco_df = pd.DataFrame()
            eligible_reco_emp = pd.DataFrame()
            eligible_policy_gain_df = pd.DataFrame()
        else:
            eligible_eval_df = build_eligible_eval_df(
                parquet_dir=parquet_dir,
                as_of_date=inferred_as_of_date,
                lookback_days=lookback_days,
                duckdb_path=duckdb_path,
            )
            if not eligible_eval_df.empty:
                eligible_product_eval_df = evaluate_products_duckdb(
                    parquet_dir=parquet_dir,
                    product_config=product_config,
                    external_metrics_df=external_metrics_df,
                    duckdb_path=duckdb_path,
                )
                eligible_product_eval_df = eligible_product_eval_df[
                    eligible_product_eval_df["product_id"].isin(eligible_eval_df["product_id"].unique())
                ].reset_index(drop=True)
            else:
                eligible_product_eval_df = pd.DataFrame()

            if not eligible_eval_df.empty and {"T", "Y"}.issubset(eligible_eval_df.columns):
                treated_eligible = eligible_eval_df[eligible_eval_df["T"] == 1]
                control_eligible = eligible_eval_df[eligible_eval_df["T"] == 0]
                eligible_reco_emp = pd.DataFrame(
                    [
                        {
                            "empirical_uplift": float(treated_eligible["Y"].mean() - control_eligible["Y"].mean()),
                            "treated_n": int(len(treated_eligible)),
                            "control_n": int(len(control_eligible)),
                            "eval_available": True,
                            "eval_reason": "T/Y available",
                        }
                    ]
                )
            else:
                eligible_reco_emp = pd.DataFrame(
                    [
                        {
                            "empirical_uplift": np.nan,
                            "treated_n": np.nan,
                            "control_n": np.nan,
                            "eval_available": False,
                            "eval_reason": "single-day eval skipped: missing T/Y",
                        }
                    ]
                )

            eligible_customer_reco_df = generate_recommendations_duckdb(
                parquet_dir=parquet_dir,
                product_eval_df=eligible_product_eval_df,
                customer_config=customer_config,
                safety_config=safety_config,
                product_config=product_config,
                duckdb_path=duckdb_path,
            )
            eligible_policy_gain_df = (
                policy_gain_curve_duckdb(
                    reco_df=eligible_customer_reco_df,
                    score_col="recommend_score" if "recommend_score" in eligible_customer_reco_df.columns else "adjusted_cate",
                    bins=backtest_config.policy_bins,
                    baseline_mode="global_mean",
                    duckdb_path=duckdb_path,
                )
                if not eligible_customer_reco_df.empty
                else pd.DataFrame(columns=["top_pct", "n", "uplift_gain"])
            )
            single_day_result = {
                "eligible_eval_df": eligible_eval_df,
                "eligible_product_eval_df": eligible_product_eval_df,
                "eligible_customer_reco_df": eligible_customer_reco_df,
                "eligible_reco_empirical_eval_df": eligible_reco_emp,
                "eligible_policy_gain_df": eligible_policy_gain_df,
                "single_day_as_of_date": pd.DataFrame([{"as_of_date": inferred_as_of_date, "lookback_days": lookback_days}]),
            }

    customer_reco_df = (
        generate_recommendations_duckdb(
            parquet_dir=parquet_dir,
            product_eval_df=product_eval_df,
            customer_config=customer_config,
            safety_config=safety_config,
            product_config=product_config,
            duckdb_path=duckdb_path,
        )
        if run_full
        else pd.DataFrame()
    )

    if not run_full:
        reco_emp = pd.DataFrame(columns=["empirical_uplift", "treated_n", "control_n"])
        policy_gain_df = pd.DataFrame(columns=["top_pct", "n", "uplift_gain"])
        temporal_df = pd.DataFrame(columns=["date", "model_ate", "empirical_uplift", "treated_n", "control_n"])
        temporal_reco_df = pd.DataFrame(columns=["date", "reco_model_ate", "reco_empirical_uplift", "treated_n", "control_n"])
        ope_df = pd.DataFrame(columns=["policy", "ipw_value", "dr_value", "ipw_ok", "dr_ok", "ipw_error", "dr_error"])
    elif customer_reco_df.empty:
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

    if run_full:
        policy_gain_df = policy_gain_curve_duckdb(
            reco_df=customer_reco_df,
            score_col="recommend_score" if "recommend_score" in customer_reco_df.columns else "adjusted_cate",
            bins=backtest_config.policy_bins,
            baseline_mode="global_mean",
            duckdb_path=duckdb_path,
        )
        temporal_df = temporal_stability_duckdb(parquet_dir=parquet_dir, duckdb_path=duckdb_path)
        temporal_reco_df = temporal_stability_reco_df(customer_reco_df, score_col_for_model="adjusted_cate")
        ope_df = pd.DataFrame(columns=["policy", "ipw_value", "dr_value", "ipw_ok", "dr_ok", "ipw_error", "dr_error"])
    else:
        policy_gain_df = pd.DataFrame(columns=["top_pct", "n", "uplift_gain"])
        temporal_df = pd.DataFrame(columns=["date", "model_ate", "empirical_uplift", "treated_n", "control_n"])
        temporal_reco_df = pd.DataFrame(columns=["date", "reco_model_ate", "reco_empirical_uplift", "treated_n", "control_n"])
        ope_df = pd.DataFrame(columns=["policy", "ipw_value", "dr_value", "ipw_ok", "dr_ok", "ipw_error", "dr_error"])

    result = {
        "product_eval_df": product_eval_df if run_full else pd.DataFrame(),
        "customer_reco_df": customer_reco_df,
        "reco_empirical_eval_df": reco_emp,
        "policy_gain_df": policy_gain_df,
        "temporal_df": temporal_df,
        "temporal_reco_df": temporal_reco_df,
        "ope_df": ope_df,
    }
    result.update(single_day_result)
    result["product_config"] = product_config
    result["customer_config"] = customer_config
    result["safety_config"] = safety_config
    result["backtest_config"] = backtest_config
    return result


def main() -> None:
    import argparse
    import shutil

    # CLI 入口说明：
    # - 这是 backtest_full_pipeline_v3.py 的脚本入口，用于直接执行回测流程。
    # - --mode 控制执行场景：
    #   * full：仅跑全量回测，生成产品层/客户层/策略层输出。
    #   * single_day：仅跑单日可推荐子集，适合检查某个 as_of_date 的线上投放口径。
    #   * both：同时跑 full + single_day。
    # - --out_dir 指定输出根目录；每个 case 会在其子目录下输出 csv、DuckDB 临时库和 md 报告。
    # - --run_tests 不是单元测试，而是按顺序执行多个可执行场景，便于一键回归。
    parser = argparse.ArgumentParser(description="Backtest full pipeline v3")
    parser.add_argument("--mode", choices=["full", "single_day", "both"], default="both")
    parser.add_argument("--parquet_dir", default="output/backtest_output_v2/eval_parquet")
    parser.add_argument("--out_dir", default="output/backtest_output_v3")
    parser.add_argument("--as_of_date", default=None)
    parser.add_argument("--lookback_days", type=int, default=30)
    parser.add_argument("--enable_single_day_reco", action="store_true")
    parser.add_argument("--run_tests", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    def _save_result(result: Dict[str, pd.DataFrame], target_dir: Path) -> None:
        # 将回测结果落盘到 case 目录：
        # - 每个 DataFrame 单独保存为 CSV
        # - 同时生成 v3 Markdown 报告，便于人工查看回测结论
        target_dir.mkdir(parents=True, exist_ok=True)
        for k, df in result.items():
            if isinstance(df, pd.DataFrame):
                df.to_csv(target_dir / f"{k}.csv", index=False)
        report_path = target_dir / "backtest_report_v3.md"
        render_business_report_v3(result, out_path=str(report_path), top_products=20, top_reco_rows=50)
        print("v3 report saved to:", report_path)
        print("v3 outputs saved to:", target_dir)

    def _run_case(case_name: str, **kwargs) -> Dict[str, pd.DataFrame]:
        # 执行一个独立场景（例如 test_single_day / test_full / test_both）。
        # 每个场景都使用独立输出目录和独立 DuckDB 临时库，互不干扰。
        case_dir = out_root / case_name
        if case_dir.exists():
            try:
                shutil.rmtree(case_dir)
            except PermissionError:
                for p in case_dir.glob("**/*"):
                    if p.is_file():
                        try:
                            p.unlink()
                        except PermissionError:
                            pass
                for p in sorted([p for p in case_dir.glob("**/*") if p.is_dir()], reverse=True):
                    try:
                        p.rmdir()
                    except OSError:
                        pass
                try:
                    case_dir.rmdir()
                except OSError:
                    pass
        case_dir.mkdir(parents=True, exist_ok=True)
        result = run_backtest_v3(
            parquet_dir=args.parquet_dir,
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
            duckdb_path=str(case_dir / "duckdb_tmp.db"),
            **kwargs,
        )
        _save_result(result, case_dir)
        return result

    if args.run_tests:
        # 一键回归入口：顺序执行多个可执行场景，不做单元测试式断言。
        print("[RUN] Executing single_day case...")
        _run_case(
            "test_single_day",
            as_of_date=args.as_of_date,
            lookback_days=args.lookback_days,
            enable_single_day_reco=True,
            mode="single_day",
        )

        print("[RUN] Executing full case...")
        _run_case(
            "test_full",
            as_of_date=args.as_of_date,
            lookback_days=args.lookback_days,
            enable_single_day_reco=False,
            mode="full",
        )

        print("[RUN] Executing both case...")
        _run_case(
            "test_both",
            as_of_date=args.as_of_date,
            lookback_days=args.lookback_days,
            enable_single_day_reco=True,
            mode="both",
        )

        print("[RUN] All cases finished.")
        return

    # 单场景直接执行入口：根据 --mode / --enable_single_day_reco 运行一次回测并输出到 out_dir。
    result = run_backtest_v3(
        parquet_dir=args.parquet_dir,
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
        as_of_date=args.as_of_date,
        lookback_days=args.lookback_days,
        enable_single_day_reco=(args.enable_single_day_reco or args.mode in {"single_day", "both"}),
        mode=args.mode,
    )

    _save_result(result, out_root)


if __name__ == "__main__":
    main()
