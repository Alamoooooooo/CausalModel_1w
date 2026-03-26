# 回测报告（Backtest Report, v3）

生成时间：2026-03-26 09:34:01

## 一、概览（Executive Summary）

- 覆盖产品数：15
- 进入推荐池产品数（recommend_all + recommend_targeted）：15
  - recommend_all：15
  - recommend_targeted：0
- 推荐明细行数（customer-product pairs）：50,000
- 被推荐客户数（unique cust_id）：50,000
- 推荐子集经验 uplift（treated-control）：0.810257

## 二、产品层评估（Product Level）

### 2.0 方法说明（产品门禁 / 决策含义）
- `recommend_all`：全量推荐（通过全部门禁，适合大规模触达）
- `recommend_targeted`：定向推荐（允许 ATE<0，但对 Top uplift 人群命中强，仅对该产品 Top 人群开放推荐）
- `watchlist`：继续观察（部分指标可，但不足以上线）
- `reject`：不推荐（关键门禁未通过/风险过高）

### 2.1 Top 产品列表在讲什么？
- 目的：在“产品维度”判断哪些产品适合推荐（全量/定向/观察/拒绝），并展示核心因果指标与风险指标。

### 2.2 Top 产品列表字段解释（表头含义）
- `product_id`：产品ID。
- `recommendation_decision`：最终推荐决策（recommend_all / recommend_targeted / watchlist / reject）。
- `sample_size`：该产品参与评估的样本行数（通常≈客户×日期）。
- `n_customer`：该产品覆盖的去重客户数。
- `ate`：平均处理效应（模型输出 CATE 的均值），越大越好。
- `empirical_uplift`：经验 uplift（treated 平均Y - control 平均Y），越大越好（更贴近真实结果口径）。
- `qini`：Qini（本项目为 proxy 版本，基于 cate 排序得到），衡量 uplift 排序能力，越大越好。
- `auuc`：AUUC（本项目为 proxy 版本），越大越好。
- `top_uplift_lift`：Top 人群平均 uplift 相对整体平均 uplift 的提升，越大越好。
- `top_vs_rest_gap`：Top 人群平均 uplift - 非Top 人群平均 uplift 的差，越大越好。
- `negative_uplift_ratio`：uplift<0 的占比（风险指标），越小越好。
- `cate_std`/`cate_p05`/`cate_p50`/`cate_p95`：CATE 分布的波动与分位数，用于看异质性/极端值。
- `product_score`：综合评分（用于排序，权重由代码定义）。
- `pass_rate`：门禁通过率（0~1），越大越好。

### 2.3 Top 产品列表（按 decision + score 排序，Top 20）

```
        product_id recommendation_decision  sample_size  n_customer      ate  empirical_uplift         qini         auuc  top_uplift_lift  top_vs_rest_gap  negative_uplift_ratio  cate_std  cate_p05  cate_p50  cate_p95  product_score  pass_rate
 bundle_and__33__9           recommend_all       150000       50000 1.003250          0.755182 17682.353256 92926.627352         0.581463         0.726828               0.009153  0.417987  0.311321  1.006992  1.687514       0.647940        1.0
bundle_and__20__25           recommend_all       150000       50000 1.003980          0.741923 17629.933963 92928.951190         0.579582         0.724477               0.008800  0.416863  0.312052  1.007752  1.680492       0.614371        1.0
 bundle_and__25__4           recommend_all       150000       50000 1.003472          0.756428 17631.005047 92891.885807         0.579803         0.724754               0.008660  0.416799  0.315197  1.006661  1.686197       0.599377        1.0
bundle_and__18__33           recommend_all       150000       50000 1.003084          0.740458 17689.284596 92921.096080         0.582174         0.727718               0.008913  0.418221  0.310469  1.005954  1.686659       0.587797        1.0
bundle_and__25__33           recommend_all       150000       50000 1.003571          0.750371 17647.999082 92916.340715         0.579902         0.724878               0.009113  0.417312  0.312949  1.007922  1.685075       0.568426        1.0
bundle_and__18__20           recommend_all       150000       50000 1.003723          0.731997 17635.661062 92915.403394         0.580020         0.725026               0.008767  0.416934  0.310974  1.007408  1.684397       0.514694        1.0
 bundle_and__18__9           recommend_all       150000       50000 1.002486          0.753492 17671.110763 92858.070343         0.582392         0.727990               0.008933  0.417798  0.312535  1.004369  1.688864       0.444617        1.0
bundle_and__18__25           recommend_all       150000       50000 1.003082          0.737235 17649.733298 92881.388321         0.581170         0.726463               0.008780  0.417312  0.312495  1.005567  1.686990       0.438019        1.0
bundle_and__20__33           recommend_all       150000       50000 1.002862          0.759574 17635.119637 92850.258931         0.579478         0.724347               0.008947  0.416963  0.309442  1.006709  1.682874       0.411646        1.0
 bundle_and__20__4           recommend_all       150000       50000 1.002426          0.768894 17622.220042 92804.703591         0.579301         0.724126               0.008633  0.416588  0.310377  1.006792  1.682522       0.359912        1.0
 bundle_and__18__4           recommend_all       150000       50000 1.002365          0.747210 17642.276341 92820.126475         0.580769         0.725961               0.008587  0.416972  0.311998  1.005452  1.685987       0.307662        1.0
  bundle_and__4__9           recommend_all       150000       50000 1.002326          0.754513 17637.627859 92812.608422         0.580462         0.725577               0.008827  0.416769  0.313250  1.005644  1.686084       0.279906        1.0
 bundle_and__20__9           recommend_all       150000       50000 1.002530          0.749581 17637.826581 92828.072448         0.580418         0.725522               0.008993  0.417002  0.311563  1.006096  1.683637       0.274763        1.0
 bundle_and__25__9           recommend_all       150000       50000 1.003060          0.729181 17636.293469 92866.286002         0.579764         0.724705               0.009080  0.416909  0.313956  1.007018  1.683701       0.259312        1.0
 bundle_and__33__4           recommend_all       150000       50000 1.002479          0.739112 17649.521091 92835.979807         0.579361         0.724201               0.008800  0.417119  0.311174  1.006644  1.684042       0.252426        1.0
```

### 2.4 如何解读产品层结果（好/坏信号）

**好信号（更可能可上线）**：
- `ate>0` 且 `empirical_uplift>0`（方向一致，且真实结果口径为正）
- `qini/auuc` 较高（说明排序“会挑人”，定向价值更大）
- `top_uplift_lift`、`top_vs_rest_gap` 明显 > 0（Top 人群显著更好）
- `negative_uplift_ratio` 低（风险小）
- `sample_size`/`n_customer` 充足（估计更稳健）
- `pass_rate` 高，且 `recommendation_decision` 为 `recommend_all` 或 `recommend_targeted`

**坏信号（需要观察/调参/重训）**：
- `ate` 与 `empirical_uplift` 长期反向（优先排查口径/漂移/校准/混杂）
- `negative_uplift_ratio` 高（容易误伤用户）
- 样本量过小仍靠前（可能是噪声，建议提高 `min_support_samples`）
- `top_uplift_lift` 很低/为负（说明挑人能力不足）


## 三、客户层推荐（Customer Level Recommendations）

### 3.1 客户层推荐在讲什么？
- 目的：把“产品可推荐”进一步落到“客户-产品对”，输出每个客户的 Top-K 推荐清单。

### 3.2 客户层推荐字段解释
- `cust_id`：客户ID。
- `product_id`：产品ID。
- `date`：样本日期/批次日期。
- `cate`：模型输出的个体处理效应（对该客户推荐该产品的预期增量）。
- `adjusted_cate`：校准后的 CATE（cate * calibration_factor），更贴近经验 uplift 尺度。
- `recommend_score`：最终排序分（综合 adjusted_cate、product_score、风险项），越大越优先推荐。
- `rank_in_customer`：该客户内部排序名次（1 表示最优先）。
- `T`：历史是否触达/处理（1/0）。
- `Y`：结果指标（例如转化/收益等）。
- `ps`：倾向得分（可选列；若你的数据没有则为空/NULL）。
- `mu0`/`mu1`：潜在结果预测（可选列；若你的数据没有则为空/NULL）。

展示 Top 50 条推荐记录（按 recommend_score 降序）：

```
 cust_id         product_id       date     cate  adjusted_cate  recommend_score  rank_in_customer  T         Y  ps  mu0  mu1
    6273  bundle_and__25__4 2026-01-01 3.091275       2.330238         0.956363                 1  0 -0.965616 NaN  NaN  NaN
    8091  bundle_and__25__4 2026-01-01 2.945103       2.220052         0.925627                 1  1  1.892095 NaN  NaN  NaN
    7138  bundle_and__25__4 2026-01-01 2.668949       2.011884         0.867559                 1  0  0.942011 NaN  NaN  NaN
   42278  bundle_and__25__4 2026-01-03 2.668679       2.011681         0.867503                 1  1 -0.113373 NaN  NaN  NaN
   42665  bundle_and__25__4 2026-01-03 2.661920       2.006585         0.866081                 1  1  1.124723 NaN  NaN  NaN
   31654 bundle_and__20__25 2026-01-03 2.785853       2.058695         0.865389                 1  0  0.152932 NaN  NaN  NaN
   26192  bundle_and__25__4 2026-01-01 2.645212       1.993991         0.862568                 1  0 -1.798136 NaN  NaN  NaN
   46119  bundle_and__25__4 2026-01-02 2.641628       1.991289         0.861815                 1  0  1.131329 NaN  NaN  NaN
    9362  bundle_and__25__4 2026-01-03 2.630749       1.983089         0.859527                 1  0  0.086667 NaN  NaN  NaN
   43634  bundle_and__25__4 2026-01-02 2.617239       1.972905         0.856686                 1  0 -0.708252 NaN  NaN  NaN
   11589 bundle_and__20__25 2026-01-03 2.738149       2.023443         0.855555                 1  0  0.873319 NaN  NaN  NaN
   44383 bundle_and__18__20 2026-01-03 3.034763       2.213197         0.851364                 1  0 -0.322401 NaN  NaN  NaN
   33806  bundle_and__25__4 2026-01-01 2.591891       1.953797         0.851356                 1  0  1.487151 NaN  NaN  NaN
   38393  bundle_and__25__4 2026-01-02 2.584620       1.948316         0.849827                 1  0  1.167278 NaN  NaN  NaN
   35201  bundle_and__25__4 2026-01-02 2.580009       1.944840         0.848858                 1  0  1.103890 NaN  NaN  NaN
   21251 bundle_and__20__25 2026-01-03 2.696607       1.992743         0.846992                 1  0 -0.178150 NaN  NaN  NaN
   30343 bundle_and__18__33 2026-01-02 2.877223       2.123912         0.846783                 1  0 -0.229572 NaN  NaN  NaN
   49460  bundle_and__25__4 2026-01-03 2.567102       1.935111         0.846144                 1  0  0.911043 NaN  NaN  NaN
   27501  bundle_and__25__4 2026-01-01 2.566911       1.934967         0.846104                 1  0 -0.890632 NaN  NaN  NaN
   38587  bundle_and__25__4 2026-01-02 2.559937       1.929710         0.844637                 1  0  1.567543 NaN  NaN  NaN
   40790  bundle_and__25__4 2026-01-02 2.552998       1.924479         0.843178                 1  0  2.202050 NaN  NaN  NaN
   36256 bundle_and__20__25 2026-01-03 2.672648       1.975038         0.842053                 1  0  1.956321 NaN  NaN  NaN
   20101  bundle_and__25__4 2026-01-03 2.537440       1.912751         0.839907                 1  0  0.661553 NaN  NaN  NaN
    9396  bundle_and__25__4 2026-01-01 2.532908       1.909335         0.838954                 1  0 -0.031938 NaN  NaN  NaN
   28610  bundle_and__25__4 2026-01-02 2.532786       1.909243         0.838928                 1  0  1.867577 NaN  NaN  NaN
   36341  bundle_and__25__4 2026-01-02 2.506392       1.889346         0.833378                 1  0  0.045601 NaN  NaN  NaN
   10668  bundle_and__25__4 2026-01-01 2.503043       1.886822         0.832674                 1  0  0.686833 NaN  NaN  NaN
    8864  bundle_and__25__4 2026-01-03 2.493010       1.879259         0.830564                 1  0 -0.307880 NaN  NaN  NaN
   28201  bundle_and__33__9 2026-01-01 2.761494       2.078674         0.829827                 1  0  0.401296 NaN  NaN  NaN
   10969  bundle_and__25__4 2026-01-02 2.489341       1.876493         0.829793                 1  0  1.381576 NaN  NaN  NaN
   19337  bundle_and__25__4 2026-01-03 2.489059       1.876281         0.829734                 1  0  2.434182 NaN  NaN  NaN
   26430  bundle_and__25__4 2026-01-02 2.482204       1.871114         0.828292                 1  0 -0.010601 NaN  NaN  NaN
     452  bundle_and__25__4 2026-01-02 2.474337       1.865183         0.826638                 1  0  0.612507 NaN  NaN  NaN
   27130  bundle_and__25__4 2026-01-03 2.464099       1.857466         0.824485                 1  1  1.208208 NaN  NaN  NaN
   46121  bundle_and__25__4 2026-01-03 2.463242       1.856820         0.824305                 1  0 -0.602007 NaN  NaN  NaN
   47896  bundle_and__25__4 2026-01-01 2.456084       1.851424         0.822800                 1  0 -0.389554 NaN  NaN  NaN
   35399 bundle_and__20__25 2026-01-03 2.577488       1.904717         0.822437                 1  0 -0.565329 NaN  NaN  NaN
   42407  bundle_and__25__4 2026-01-01 2.451837       1.848223         0.821907                 1  0 -1.180444 NaN  NaN  NaN
   35518  bundle_and__25__4 2026-01-03 2.451336       1.847845         0.821801                 1  0  1.951133 NaN  NaN  NaN
    9198  bundle_and__25__4 2026-01-03 2.446587       1.844265         0.820803                 1  0  0.956244 NaN  NaN  NaN
   42493 bundle_and__20__25 2026-01-03 2.564780       1.895326         0.819818                 1  0  0.160908 NaN  NaN  NaN
   22502  bundle_and__25__4 2026-01-03 2.439562       1.838970         0.819326                 1  0 -0.478621 NaN  NaN  NaN
   20841  bundle_and__25__4 2026-01-02 2.436347       1.836546         0.818650                 1  0 -1.317434 NaN  NaN  NaN
   42488  bundle_and__25__4 2026-01-03 2.432129       1.833367         0.817763                 1  0 -0.347543 NaN  NaN  NaN
   22194  bundle_and__25__4 2026-01-01 2.431301       1.832742         0.817589                 1  0  0.904596 NaN  NaN  NaN
   23923  bundle_and__25__4 2026-01-03 2.426425       1.829067         0.816563                 1  0  0.285625 NaN  NaN  NaN
   26868  bundle_and__25__4 2026-01-03 2.426084       1.828810         0.816492                 1  1 -1.172950 NaN  NaN  NaN
   10633  bundle_and__25__4 2026-01-01 2.424277       1.827447         0.816112                 1  0  0.936151 NaN  NaN  NaN
   13444  bundle_and__25__4 2026-01-02 2.420883       1.824889         0.815398                 1  0  0.321824 NaN  NaN  NaN
   45934  bundle_and__25__4 2026-01-01 2.418839       1.823348         0.814968                 1  0  0.428011 NaN  NaN  NaN
```

### 3.3 如何解读客户层推荐（好/坏信号）

**好信号**：
- 推荐清单里 `adjusted_cate` 大多为正，且头部记录（rank=1）明显更高
- 同一客户 Top1 的 `recommend_score` 明显高于 Top3（排序分有区分度）
- 推荐主要来自产品层 `recommend_all/recommend_targeted` 池（策略一致、可控）

**坏信号**：
- 大量推荐记录 `adjusted_cate<=0` 仍被输出（通常是 `min_cate` 太低、校准异常或数据噪声）
- 推荐过度集中在少数产品，但这些产品 `empirical_uplift`/风险指标一般（可能权重偏“强产品”或门禁太松）
- 客户内 rank 的分数差异很小（说明 score 信号弱，需调权重或加更强的门禁）


## 四、策略收益曲线（Policy Gain Curve）

### 4.1 策略收益曲线在讲什么？
- 目的：回答“如果只触达推荐分 top 1%/5%/20%…的人群，平均能提升多少？”用于做触达规模决策。

### 4.2 字段解释
- `top_pct`：取推荐分 top 的比例（例如 0.10=Top10%）。
- `n`：对应 top_pct 的触达样本数。
- `uplift_gain`：收益提升（当前口径：Top 子集平均Y - 全体平均Y）。

```
 top_pct     n  uplift_gain
    0.01   500     0.191954
    0.02  1000     0.152755
    0.05  2500     0.151377
    0.10  5000     0.109331
    0.20 10000     0.099142
    0.30 15000     0.084469
    0.50 25000     0.054086
    1.00 50000     0.000000
```

### 4.3 如何解读策略收益曲线（好/坏信号）

**好信号**：
- `uplift_gain` 随 `top_pct` 增大应逐步下降并最终趋近 0（top 1% > top 2% > ...）
- top 小比例（如 1%/2%/5%）`uplift_gain` 明显 > 0（说明排序有用）
- 可用“曲线拐点”确定触达规模：从 1% 增到 5% 收益仍高，但到 30% 变平，说明扩大触达会稀释效果

**坏信号**：
- 曲线不单调或 top 小比例≈0/为负（说明推荐分与真实 Y 关系弱，需调权重/重训/排查口径）


## 五、时间稳定性（Temporal Stability）

### 5.1 时间稳定性在讲什么？
- 目的：检查模型/推荐效果是否随时间漂移；若某天明显变差，可能是数据分布/活动/人群变化导致。

### 5.2 全量时间稳定性（全量产品/全量样本）字段解释
- `date`：日期。
- `model_ate`：当日平均 cate（全量样本的模型视角平均处理效应）。
- `empirical_uplift`：当日经验 uplift（全量样本 treated 平均Y - control 平均Y）。
- `treated_n`/`control_n`：当日 treated/control 样本量。
- 说明：该口径适合做“全局健康度/漂移监控”，可能会被大量无效产品稀释。

```
      date  model_ate  empirical_uplift  treated_n  control_n
2026-01-01   1.003402          0.742296    26800.0   723200.0
2026-01-02   1.003075          0.757875    26361.0   723639.0
2026-01-03   1.002462          0.742659    26471.0   723529.0
```

### 5.3 推荐子集时间稳定性（仅最终推荐清单）

- 目的：只看最终输出的 `customer_reco_df`（经过产品门禁/客户门禁后的推荐清单），避免全量无因果产品对总体的稀释。
- `reco_model_ate`：推荐清单内（默认用 `adjusted_cate`）的当日均值。
- `reco_empirical_uplift`：推荐清单内 treated 平均Y - control 平均Y。
- `treated_n`/`control_n`：推荐清单内 treated/control 样本量。

```
      date  reco_model_ate  reco_empirical_uplift  treated_n  control_n
2026-01-01        1.062654               0.846708        627      16041
2026-01-02        1.065789               0.842848        547      16142
2026-01-03        1.064298               0.738701        568      16075
```

### 5.4 如何解读时间稳定性（好/坏信号）

**好信号**：
- `empirical_uplift` 比较平稳，且与 `model_ate` 大体同向变化
- `treated_n/control_n` 稳定且不太小（样本量充足时结论更可信）

**坏信号**：
- `empirical_uplift` 断崖式波动且样本量并不小（更可能是分布漂移/活动变化/数据口径变化）
- 某些日期 treated/control 极不平衡（经验 uplift 会变得不稳定）


## 六、离线策略价值评估（OPE）

### 6.1 离线评估（OPE）在讲什么？
- 目的：在不能线上 A/B 的情况下，估计“如果按新策略触达，整体期望 Y 会是多少”。
- 说明：OPE 通常需要 `ps`（倾向得分）以及可能需要 `mu0/mu1`（潜在结果预测）。若缺列，本报告会在 `ope_df` 中说明原因。

### 6.2 字段解释
- `policy`：被评估的策略名称。
- `ipw_value`：IPW 估计的策略价值（需要 ps）。
- `dr_value`：DR 估计的策略价值（需要 ps + mu0/mu1）。
- `ipw_ok`/`dr_ok`：是否成功计算。
- `ipw_error`/`dr_error`：失败原因/提示。

```
       policy  ipw_value  dr_value  ipw_ok  dr_ok                                         ipw_error                                          dr_error
top20_by_cate        NaN       NaN   False  False 缺少 ps（propensity score），无法计算 IPW/DR OPE。可先跳过 OPE。 缺少 ps（propensity score），无法计算 IPW/DR OPE。可先跳过 OPE。
```

### 6.3 如何解读 OPE（若未计算可忽略）

- 只有当 `ipw_ok/dr_ok=True` 时，才建议把 `ipw_value/dr_value` 纳入判断。
- 若缺少 `ps` 或 `mu0/mu1` 导致无法计算（报告会写明原因），请忽略该部分，不影响其它章节对“好/坏”的判断。


## 七、ps / mu0 / mu1 是什么？怎么计算？（给数据准备同学）

### 7.1 ps（propensity score，倾向得分）
- 定义：ps(x) = P(T=1 | X=x)，即在历史策略下样本被触达/处理的概率。
- 用途：IPW/DR 等离线策略评估需要用 ps 来纠偏历史触达偏差。
- 计算：用历史数据训练一个二分类模型预测 T（特征只能用触达前可见特征），输出 predict_proba 的概率作为 ps，并进行 clipping（例如 0.01~0.99）。

### 7.2 mu0 / mu1（潜在结果预测）
- 定义：mu0(x)=E[Y|X=x,T=0]，mu1(x)=E[Y|X=x,T=1]。
- 用途：DR（Doubly Robust）估计需要 mu0/mu1；只要 ps 或 mu 模型其中一个靠谱，估计仍可能一致。
- 计算（最简单 T-learner）：
  - 在 T=0 子集训练一个回归模型预测 Y → 对全量输出 mu0(x)
  - 在 T=1 子集训练一个回归模型预测 Y → 对全量输出 mu1(x)
  - 模型可用线性回归/GBDT/LightGBM 等，视 Y（金额/概率）而定。

## 八、配置参数与调参指南（Config & Tuning Guide）

本章节解释：产品门禁（哪些产品能推荐）、客户层输出（每个客户推荐什么）、安全控制（更保守/更激进）、以及排序权重（更偏个性化/更偏强产品/更偏安全）。

### 8.0 本次运行参数快照（便于复现）
- 说明：以下参数来自本次运行时传入 `run_backtest_v3()` 的 config。不同业务可按目标调参。

#### ProductDecisionConfig
- min_ate=0.0000
- min_empirical_uplift=0.0000
- min_qini=0.0000
- min_auuc=0.0000
- min_top_uplift_lift=0.0000
- max_negative_uplift_ratio=0.5500
- min_support_samples=300
- top_ratio=0.2000
- enable_targeted_reco=True
- targeted_top_ratio=0.2000
- min_targeted_lift=0.0000
- allow_targeted_when_ate_negative=True
- enable_calibration=True

#### CustomerDecisionConfig
- min_cate=0.0000
- top_k_per_customer=1
- min_product_pass_rate=0.0000
- customer_weight_col=None

#### SafetyConfig
- enable_product_blacklist_gate=True
- enable_customer_safe_filter=True
- min_customer_expected_gain=0.0000
- max_customer_negative_share=0.5000

#### BacktestConfig
- policy_bins=(0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
- ps_clip_low=0.0100
- ps_clip_high=0.9900

说明：当前 v3 的 demo 入口里没有把 config 注入到 result 中；若你希望报告自动打印本次 config，我会在 v3 里把 config 对象一并放进 result（下方会实现）。

### 8.1 产品门禁与决策参数（ProductDecisionConfig）怎么理解/怎么调？
- `min_ate`：ATE 下限。调大→更保守（产品池更小、平均效果更稳）；调小→更激进（覆盖更大、风险更高）。
- `min_empirical_uplift`：经验 uplift 下限。调大→更贴近历史真实结果、减少“模型幻觉”；调小→更多依赖模型 cate/排序能力。
- `min_qini` / `min_auuc`：排序能力门槛。调大→更强调“挑对人”（异质性强、定向价值高）；调小→更强调“平均有效”。
- `min_top_uplift_lift`：Top 人群比整体更强的门槛。调大→更偏强异质性产品；调小→更偏均匀有效产品。
- `max_negative_uplift_ratio`：负 uplift 占比上限（风险闸门）。调小→更安全但覆盖变小；调大→更激进但更可能误伤用户。
- `min_support_samples`：最小样本量。调大→更稳健但冷门产品容易被拒；调小→覆盖更多但估计更抖。
- `top_ratio`：Top uplift 相关指标（top_uplift_lift/top_vs_rest_gap）的 Top 比例。调小→更关注极头部；调大→更关注更大范围的好人群。
- `enable_targeted_reco`：是否启用定向推荐通道。关掉→只有 recommend_all/watchlist/reject。
- `targeted_top_ratio`：定向推荐开放的人群比例。调小→更保守（更少人被定向触达）；调大→更激进。
- `min_targeted_lift`：定向推荐的 lift 门槛。调大→更尖、更少产品进入 targeted；调小→更多产品可 targeted。
- `allow_targeted_when_ate_negative`：允许 ATE<0 但 Top 很强的产品走 targeted。开→偏增长型；关→偏安全型。
- `enable_calibration`：是否用 empirical_uplift/ate 做尺度校准。开→更贴近历史结果口径；关→保持模型尺度（更稳定但可能与业务指标尺度不一致）。

### 8.2 客户层输出参数（CustomerDecisionConfig）怎么理解/怎么调？
- `min_cate`：客户-产品对的 uplift 门槛。调大→只推更确定增益的组合；调小→推荐更多但平均增益可能下降。
- `top_k_per_customer`：每个客户最多推荐多少条。调大→覆盖更广但可能稀释；调小→更聚焦。
- `min_product_pass_rate`：产品门禁通过率阈值（当前 v3 未用于强过滤，属于预留；若要生效可加到 SQL WHERE）。
- `customer_weight_col`：客户权重列（当前 v3 未使用，预留）。

### 8.3 安全与回测参数（SafetyConfig / BacktestConfig）怎么理解/怎么调？
- `enable_product_blacklist_gate`：是否仅从 recommend_all/targeted 产品池中做候选。一般建议开启（更可控）。
- `enable_customer_safe_filter` + `min_customer_expected_gain`：客户侧最低期望增益过滤。调大→更保守；调小→更激进。
- `max_customer_negative_share`：客户层负收益容忍度（当前 v3 未显式使用，预留）。
- `policy_bins`：policy curve 的切分点。更密→更细但更复杂；更稀→更易解释但粗糙。
- `ps_clip_low/high`：ps 裁剪范围（仅 OPE）。clip 更紧→权重更稳定但偏差可能增大；clip 更松→方差更大（易出现极端权重）。

### 8.4 排序权重（最常用调参旋钮）怎么调？
- 产品排序 `product_score` 有两套权重（mass vs targeted）：
  - 提高 `ate/empirical_uplift` 权重 → 更偏“平均收益更大”的产品
  - 提高 `qini/auuc/top_uplift_lift` 权重 → 更偏“更会挑人/异质性强”的产品
  - 提高 `(1 - negative_uplift_ratio)` 权重 → 更偏“更安全”的产品
- 客户层排序 `recommend_score = 0.65*norm_adjusted_cate + 0.25*norm_product_score + 0.10*(1-norm_neg_ratio)`：
  - 提高 adjusted_cate 权重 → 更个性化（更强调人-货匹配 uplift）
  - 提高 product_score 权重 → 更偏强产品（少数强产品更容易被推给更多人）
  - 提高安全项权重 → 更保守（减少可能负收益的人群）


## 九、如何解读回测好坏（Interpretation Checklist）

本项目的“回测”是离线验证：用模型的 uplift（`cate`）决定“推给谁”，再用历史数据中的 `T/Y` 做 sanity check 与策略收益模拟。
建议按 **必须过线 / 加分项 / 风险项** 三类信号来判断效果好坏。

### 9.1 这套回测到底怎么测的？（口径说明）
- **产品层（Product Level）**：对每个产品汇总 `cate` 得到 `ate`，并用历史 `T/Y` 计算 `empirical_uplift=mean(Y|T=1)-mean(Y|T=0)` 作为经验对照；再用 `cate` 排序构造 proxy 的 `qini/auuc/top_uplift_lift` 衡量“会不会挑人”。
- **客户层（Customer Level）**：在可推荐产品池中，对每个客户按 `recommend_score` 排序取 Top-K，形成推荐清单 `customer_reco_df`。
- **策略层（Policy Level）**：
  - `policy_gain_df`：把推荐分从高到低取 top 1%/5%/…，看这些样本平均 `Y` 比全体平均 `Y` 高多少（当前实现口径）。
  - `temporal_df`：按日期观察 `model_ate` 与 `empirical_uplift` 是否稳定，排查漂移。
  - `ope_df`：当你提供 `ps/mu0/mu1` 时可做 IPW/DR 的离线策略评估（v3 默认不回读全量 parquet 做 OPE，因此会提示如何扩展）。

### 9.2 必须过线（不然说明策略/模型基本不可用）
1) **方向一致性**：进入推荐池（recommend_all/targeted）的产品，`ate` 与 `empirical_uplift` 不应长期大量反向。
   - 若大量出现 `ate>0` 但 `empirical_uplift<0`：优先排查数据泄露、标签口径、分布漂移、校准过强、或 treated/control 结构性差异。
2) **收益曲线形状**：`policy_gain_df.uplift_gain` 随 `top_pct` 增大应逐步下降并趋近 0；top 1%/2% 通常应明显高于 top 50%。
   - 若 top 很小比例仍不如整体：排序几乎无效（或推荐分与真实收益无关）。
3) **时间稳定性**：`temporal_df` 中 `empirical_uplift` 不应在样本量足够（treated_n/control_n 大）时出现断崖式变动。

### 9.3 加分项（越多越好，说明更可上线）
- 推荐池产品数量合理（不为 0，也不是几乎全量）。
- `negative_uplift_ratio` 低（更安全）。
- `top_uplift_lift`、`top_vs_rest_gap` 明显 > 0（说明模型“会挑人”，适合定向）。
- `reco_empirical_eval_df.empirical_uplift` 为正且量级符合业务预期（推荐清单子集的经验 uplift sanity check）。

### 9.4 风险项（看到要警惕，通常需要调参/加门禁/重训）
- `negative_uplift_ratio` 高但仍被推荐：门禁太松/安全权重太低。
- 小样本产品（sample_size/n_customer 很小）排到很前：可能噪声大，建议提高 `min_support_samples` 或加置信度判断。
- `empirical_uplift` 与 `ate` 偏离很大：考虑关闭/调整校准（enable_calibration）、或按业务口径重新训练模型。
- policy curve 不单调：考虑调整 `recommend_score` 权重、提高 `min_cate`、或把风险项权重调高。

### 9.5 常用排查与调参建议（快速指引）
- **想更保守、更安全**：降低 `max_negative_uplift_ratio`、提高 `min_support_samples`、提高客户侧 `min_cate` / `min_customer_expected_gain`、提高推荐分里安全项权重。
- **想更激进、更覆盖**：放宽 `min_ate/min_empirical_uplift/min_qini`、降低 `min_cate`、提高 `top_k_per_customer`。
- **想更强调“挑人”能力（定向更尖）**：提高 `min_qini/min_auuc/min_top_uplift_lift`，并调小 `targeted_top_ratio`（只给最头部人群）。


## 附录：输出数据表说明

- `product_eval_df`：产品层聚合指标 + 门禁结果 + product_score
- `customer_reco_df`：客户-产品推荐清单（含 adjusted_cate / recommend_score；若无 ps/mu 则为空/NULL）
- `reco_empirical_eval_df`：推荐子集的 treated-control uplift（sanity check）
- `policy_gain_df`：不同触达比例下的收益曲线（经验口径）
- `temporal_df`：按 date 维度的 model_ate vs empirical_uplift（全量产品/全量样本）
- `temporal_reco_df`：按 date 维度的 reco_model_ate vs reco_empirical_uplift（仅推荐清单子集）
- `ope_df`：离线策略价值评估（若缺 ps/mu 会自动降级并写明原因）
