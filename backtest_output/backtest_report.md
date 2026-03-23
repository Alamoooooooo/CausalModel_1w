# 回测报告（Backtest Report）

生成时间：2026-03-23 11:14:57

## 一、概览（Executive Summary）

- 覆盖产品数：40
- 进入推荐池产品数（decision=recommend）：0
- 推荐明细行数（customer-product pairs）：0
- 被推荐客户数（unique cust_id）：0
- 推荐子集经验 uplift（treated-control）：0.000000

## 二、产品层评估（Product Level）

### 2.0 方法说明（产品门禁 / 决策含义）
- `recommend`：通过所有产品门禁（可进入推荐池）
- `watchlist`：仅满足部分门禁（例如 ATE 为正但风险或排序能力不足）
- `reject`：关键门禁未通过（整体方向/风险等不满足）

产品进入 recommend 需要同时满足的门禁包括：ATE、empirical uplift、qini/auuc proxy、top lift、negative uplift 风险、样本量等。

### 2.1 门禁通过率与主要失败原因汇总

```
              gate  pass_rate       top_fail_reason_1 top_fail_reason_2
          pass_ate        0.6 pass_negative_risk (40)     pass_ate (16)
         pass_auuc        1.0 pass_negative_risk (40)     pass_ate (16)
    pass_empirical        1.0 pass_negative_risk (40)     pass_ate (16)
pass_negative_risk        0.0 pass_negative_risk (40)     pass_ate (16)
         pass_qini        1.0 pass_negative_risk (40)     pass_ate (16)
      pass_support        1.0 pass_negative_risk (40)     pass_ate (16)
     pass_top_lift        1.0 pass_negative_risk (40)     pass_ate (16)
```

### 2.2 产品决策分布（recommend/watchlist/reject）

```
 decision  n_products  share
watchlist          24    0.6
   reject          16    0.4
```

### 诊断：为什么本次没有任何产品进入 recommend？

本 pipeline 的产品层采用“多门禁同时满足才进入 recommend”的机制。在示例模拟数据中，`cate` 近似对称分布（默认 N(0,1)），因此 `cate < 0` 的比例通常接近 50%。

- 当前配置 `max_negative_uplift_ratio=0.5`，而本次数据 `negative_uplift_ratio` 中位数≈0.4997，均值≈0.4997，因此大部分/全部产品会在 `pass_negative_risk` 上失败，最终导致 recommend=0。

可选解决方案（用于让示例报告更像业务结果）：
- 方案A（改门禁）：把 `max_negative_uplift_ratio` 放宽到 0.55~0.60（先验证流程/报告展示）。
- 方案B（改模拟分布）：提高 `EvalDFSimConfig.cate_mean`（例如 0.2~0.4）或改变 cate 生成逻辑，让负 uplift 比例明显低于 0.5。

### 2.3 Top 产品列表（按 decision + score 排序）

```
 product_id recommendation_decision  sample_size  n_customer       ate  empirical_uplift         qini         auuc  top_uplift_lift  negative_uplift_ratio  product_score
         36                  reject       150000       50000 -0.000718          0.367529 42456.554557 42402.687500         1.404633               0.500900       0.589754
          2                  reject       150000       50000 -0.001179          0.372655 42286.650979 42198.242188         1.403349               0.501407       0.463173
         29                  reject       150000       50000 -0.000083          0.359869 42348.313030 42342.054688         1.398505               0.499520       0.463071
         12                  reject       150000       50000 -0.001234          0.362877 42372.518924 42279.984375         1.400176               0.500207       0.458174
         38                  reject       150000       50000 -0.001343          0.358233 42316.222839 42215.507812         1.396653               0.498087       0.399460
         27                  reject       150000       50000 -0.001482          0.358208 42335.665290 42224.523438         1.400361               0.500573       0.395646
          1                  reject       150000       50000 -0.002079          0.350877 42403.493718 42247.593750         1.402767               0.501340       0.392323
         34                  reject       150000       50000 -0.000410          0.346664 42360.092490 42329.312500         1.399316               0.500173       0.387019
         16                  reject       150000       50000 -0.004737          0.363534 42458.312680 42103.074219         1.402022               0.501747       0.383642
          5                  reject       150000       50000 -0.001339          0.372896 42208.159847 42107.722656         1.396957               0.500927       0.346857
          7                  reject       150000       50000 -0.001438          0.360021 42255.809799 42147.972656         1.397166               0.499647       0.338243
         39                  reject       150000       50000 -0.003103          0.374673 42258.933185 42026.171875         1.396277               0.500767       0.323703
         28                  reject       150000       50000 -0.003067          0.363128 42282.696114 42052.667969         1.399741               0.500867       0.317612
         11                  reject       150000       50000 -0.001498          0.354907 42251.602312 42139.230469         1.397524               0.500527       0.292570
         35                  reject       150000       50000 -0.004755          0.356142 42320.375675 41963.750000         1.396276               0.500247       0.222707
          8                  reject       150000       50000 -0.003787          0.350660 42199.490170 41915.472656         1.391607               0.500340       0.095445
         26               watchlist       150000       50000  0.004304          0.385761 42384.120177 42706.886719         1.401642               0.498220       0.834155
         19               watchlist       150000       50000  0.006247          0.372307 42317.813756 42786.367188         1.400238               0.498173       0.775013
         17               watchlist       150000       50000  0.002331          0.375276 42425.723689 42600.574219         1.405230               0.499393       0.753848
         13               watchlist       150000       50000  0.004531          0.375657 42323.212107 42663.027344         1.401352               0.498213       0.747225
```

## 三、客户层推荐（Customer Level Recommendations）

展示 Top 50 条推荐记录（按 recommend_score 降序）：

_(empty)_

## 四、策略收益曲线（Policy Gain Curve）

_(empty)_

## 五、时间稳定性（Temporal Stability）

```
      date  model_ate  empirical_uplift  treated_n  control_n
2026-01-01   0.000560          0.360715   353423.0  1646577.0
2026-01-02   0.000819          0.367192   352057.0  1647943.0
2026-01-03   0.000638          0.360743   352491.0  1647509.0
```

## 六、离线策略价值评估（OPE）

```
       policy  ipw_value  dr_value  ipw_ok  dr_ok ipw_error dr_error
top20_by_cate   0.170056  0.169592    True   True                   
```

## 附录：输出数据表说明

- `product_eval_df`：产品层聚合指标 + 门禁结果 + product_score
- `customer_reco_df`：客户-产品推荐清单（含 adjusted_cate / recommend_score）
- `reco_empirical_eval_df`：推荐子集的 treated-control uplift（sanity check）
- `policy_gain_df`：不同触达比例下的收益曲线（经验口径）
- `temporal_df`：按 date 维度的 model_ate vs empirical_uplift
- `ope_df`：IPW/DR 离线策略价值（需 ps/mu0/mu1）
