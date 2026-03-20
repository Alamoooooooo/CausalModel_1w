这些阈值本质上是把“模型输出”翻译成“是否投放”的业务门槛。  
可以把它们分成 3 层理解：

- 产品层：这个产品整体值不值得进入推荐池
- 客户层：这个客户是否值得推荐某个产品
- 安全层：推荐出去会不会有较大误伤风险

下面按参数逐个解释，并给你一个调参思路。

---

## 1. 产品层参数：`ProductDecisionConfig`

### `min_ate=0.0`
含义：
- 产品级平均因果效应下限
- 代码里 `ate = mean(cate)`，所以这里是在要求：
  - 该产品对全体客户平均来看，增量效应至少为正

业务解释：
- `ate > 0` 说明“整体方向正确”
- 如果 `ate <= 0`，说明这个产品整体推下去未必提升活期存款，不适合进入推荐池

怎么调：
- **探索期**：设成 `0.0`
  - 只要方向为正就保留
- **保守投放期**：设成略大于 0，比如 `0.001`、`0.005`
  - 只有平均提升达到一定强度才放行
- 如果你的 `Y` 是金额增量，阈值应按业务单位来定，比如：
  - `min_ate = 1000` 表示平均每客户预期至少提升 1000 元
- 如果 `Y` 做过标准化，则这个值要按标准化后的尺度理解

建议：
- 初期先 `0.0`
- 后续看 34 个产品的 `ate` 分布，再定分位数阈值，比如保留前 50% 正向产品

---

### `min_qini=0.0`
含义：
- 产品 uplift 排序能力下限
- 要求产品的 `qini` 至少大于该值

业务解释：
- `Qini > 0` 表示模型对“谁更值得推”有一定排序能力
- 如果 `Qini <= 0`，说明虽然平均可能有点正效应，但很难找准“最该推的人”

怎么调：
- **默认建议**：`0.0`
- 如果你的 qini 数值量纲差异较大，建议不要直接用固定值，而是：
  - 用相对门槛，如大于全产品中位数
  - 或至少大于 baseline / 统计波动范围
- 如果你现在 34 个产品里 qini 普遍偏小，可以先只要求 `>0`
- 如果后面产品多了，可以提高到：
  - `min_qini = 所有产品 qini 的 30%分位数`
  - 或者 `min_qini = 中位数`

建议：
- 你当前阶段不要设太高，先用 `0.0`
- 因为你的目标先是形成稳定框架，不是过早收紧

---

### `min_auuc=0.0`
含义：
- uplift 曲线面积下限
- 衡量整体排序收益

业务解释：
- `AUUC > 0` 说明按模型排序投放，比随机投放更有累计收益
- 和 qini 类似，但更偏向全局累计收益视角

怎么调：
- 跟 `min_qini` 类似
- 初期建议 `0.0`
- 后续建议做相对阈值而非绝对阈值

建议：
- `qini` 和 `auuc` 最好不要只看一个
- 一般：
  - `qini` 更强调比随机好多少
  - `auuc` 更强调整体累计 uplift 表现

---

### `min_top_uplift_lift=0.0`
含义：
- Top 人群 uplift 相对整体 uplift 至少提升多少
- 代码里是：
  - `top_uplift_lift = top_mean_cate - overall_mean_cate`

业务解释：
- 这是“精准有效”的关键指标
- 如果这个值 > 0，说明模型找出的 Top 人群确实比平均客户更值得推
- 如果接近 0，说明模型虽然能给 cate，但筛选价值不明显

怎么调：
- **探索期**：`0.0`
- **强调精准营销时**：调大一些
  - 比如要求 Top 20% 客户的平均 uplift 比整体高 10% 或 20%
- 如果你想用相对比例而不是绝对差值，后面可改成：
  - `(top_mean_cate / overall_mean_cate) - 1`

建议：
- 当前先设 `0.0`
- 后续可结合产品分布，调成分位数阈值

---

### `max_negative_uplift_ratio=0.4`
含义：
- 允许负 uplift 客户占比上限
- 即一个产品下，`cate < 0` 的客户比例不能太高

业务解释：
- 这是风险控制核心
- 如果一个产品虽然平均 `ate > 0`，但有大量客户 `cate < 0`，就说明它“平均有效，但误伤很多人”
- 比如：
  - 产品 A：`ate=0.01`，但 60% 客户 `cate<0`
  - 这种产品不适合大范围推荐

怎么调：
- **保守策略**：`0.3` ~ `0.4`
- **平衡策略**：`0.4` ~ `0.5`
- **探索策略**：`0.5` ~ `0.6`

建议：
- 你的场景是银行产品推荐，误推成本通常不低，建议先偏保守
- 推荐初始值：
  - `0.35` 或 `0.4`

---

### `min_recommendable_customers=100`
含义：
- 至少要有多少客户 `cate > 0`，这个产品才值得进入推荐池

业务解释：
- 防止出现“模型说有效，但其实只有很小一撮客户有效”
- 这种产品即使因果上成立，业务上可能不值得投入资源

怎么调：
- 和你的客户总量、投放成本有关
- 你现在约 6 万用户，34 个产品，平均每产品覆盖潜在人群可能不少
- 初始可按绝对值或相对值设

推荐两种设法：

#### 绝对值法
- `100`：适合探索期
- `300`：适合正式投放前
- `500+`：适合要规模收益的产品

#### 相对值法
更推荐：
- 设为总客户数的某个比例
- 比如：
  - `0.5% * 60000 = 300`
  - `1% * 60000 = 600`

建议：
- 如果产品很多、长尾产品多：先 `100`
- 如果你要控制资源集中：可提到 `300`

---

### `min_support_samples=300`
含义：
- 产品样本量至少多少，才认为评估稳定

业务解释：
- 样本太小，即使 `ate/qini` 好看，也可能不稳
- 这是“统计可靠性门槛”

怎么调：
- 如果某产品只有几十个样本，最好不要自动推荐
- 常见建议：
  - 探索期：`100`
  - 常规：`300`
  - 严格：`500`

建议：
- 你当前 6 万客户、34 产品，设 `300` 是合理的
- 如果个别产品覆盖率低，可以先降到 `200`，但要配合 watchlist 机制

---

### `top_ratio=0.2`
含义：
- Top 人群比例
- 用前 20% 客户来衡量“Top uplift 是否显著更高”

业务解释：
- 反映你想把营销资源集中在多大的人群上
- 比例越小，越偏“精准打击”
- 比例越大，越偏“规模投放”

怎么调：
- `0.1`：只看最优 10%，偏精准
- `0.2`：常用默认值，平衡
- `0.3`：偏向更大投放覆盖

建议：
- 你现在适合 `0.2`
- 如果未来客户经理资源很紧，可降到 `0.1`
- 如果要做批量外呼、短信投放，可提到 `0.3`

---

## 2. 客户层参数：`CustomerDecisionConfig`

### `min_cate=0.0`
含义：
- 客户-产品级推荐下限
- 只有当某客户对某产品的 `cate > min_cate`，才考虑推荐

业务解释：
- 这是客户级“是否值得推”的最直接门槛
- `cate > 0` 表示模型预测推荐该产品对该客户有正向增量

怎么调：
- **探索期**：`0.0`
- **保守投放期**：设一个更高阈值
  - 例如只保留前景更强的客户
- 如果 `Y` 是金额提升，建议按业务价值设阈值
  - 例如 `min_cate = 500`
  - 表示预期提升不足 500 元的不推

建议：
- 初期用 `0.0`
- 后面结合客户触达成本调整：
  - 若一次触达成本较高，就提高 `min_cate`

---

### `top_k_per_customer=3`
含义：
- 每个客户最多推荐几个产品

业务解释：
- 防止一个客户被同时推太多产品
- 也符合实际业务逻辑：客户经理不可能一次讲 8 个产品

怎么调：
- `1`：最严格，只推荐最优产品
- `2~3`：常见设置
- `5`：适合后续再由人工筛选

建议：
- 银行推荐一般建议从 `1` 或 `2` 开始
- `3` 偏宽松
- 如果你的产品之间可能互斥，最好先 `1`

---

### `customer_weight_col=None`
含义：
- 客户权重字段
- 用于把客户价值纳入推荐排序

业务解释：
- 不是所有客户同样重要
- 如果你有字段比如：
  - 客户 AUM
  - 客户等级
  - 存款规模
  - 战略客户标识
- 可以把这些作为权重，优先推荐高价值客户

怎么调：
- 没有客户价值字段时：`None`
- 有字段时：传列名，比如：
  - `"cust_value_score"`
  - `"aum_weight"`

建议：
- 如果后续要贴近业务落地，这个非常值得接入
- 因为业务上常常不是“谁 uplift 最大就先推谁”，而是“谁 uplift 大且客户价值高就先推谁”

---

## 3. 安全层参数：`SafetyConfig`

### `max_customer_negative_share=0.4`
注意：
- 这个名字有一点歧义，代码里实际用的是客户推荐记录合并后的 `negative_uplift_ratio`
- 本质上还是在控制产品层风险向客户层传导

含义：
- 对进入客户推荐名单的记录，再做一次风险约束
- 如果该产品整体负 uplift 风险过高，则即使某客户 `cate > 0`，也可能被过滤

业务解释：
- 有些产品个体看着正效应，但整体风险结构不好
- 这相当于客户层再次复核“这个产品整体上是不是太冒险”

怎么调：
- 若想强风控：`0.3`
- 若平衡：`0.4`
- 若探索：`0.5`

建议：
- 和 `max_negative_uplift_ratio` 保持一致或略更严格

---

### `min_customer_expected_gain=0.0`
含义：
- 客户推荐记录的最低预期增益阈值
- 推荐前再过滤一遍小于此值的客户-产品对

业务解释：
- 有些客户虽然 `cate > 0`，但只有一点点正效应，业务上不一定值得推
- 特别当触达成本、客户经理精力有限时，这个阈值很重要

怎么调：
- 如果没有清晰成本口径：先 `0.0`
- 如果有触达成本，可设为“保本线”
  - 比如单次触达成本折合成需要至少提升多少存款收益

建议：
- 未来最应该从这里接业务价值
- 例如：
  - 客户经理触达一次成本 = 20 元
  - 你换算出至少需要 300 元的存款增量才值得推
  - 那就设 `min_customer_expected_gain = 300`

---

## 4. 你应该如何调整：推荐调参路线

我建议你不要一上来就“拍脑袋定死”，而是分 3 个阶段。

---

### 第一阶段：框架跑通
目标：
- 不错杀太多产品
- 先看整体分布

建议参数：

```python
ProductDecisionConfig(
    min_ate=0.0,
    min_qini=0.0,
    min_auuc=0.0,
    min_top_uplift_lift=0.0,
    max_negative_uplift_ratio=0.5,
    min_recommendable_customers=100,
    min_support_samples=200,
    top_ratio=0.2,
)

CustomerDecisionConfig(
    min_cate=0.0,
    top_k_per_customer=3,
    customer_weight_col=None,
)

SafetyConfig(
    max_customer_negative_share=0.5,
    min_customer_expected_gain=0.0,
)
```

适用场景：
- 先把 34 个产品跑一遍
- 看每个产品被分到 `recommend / watchlist / reject` 的分布

---

### 第二阶段：形成可用推荐池
目标：
- 开始提高推荐质量
- 控制误推风险

建议参数：

```python
ProductDecisionConfig(
    min_ate=0.0,
    min_qini=0.0,
    min_auuc=0.0,
    min_top_uplift_lift=0.0,
    max_negative_uplift_ratio=0.4,
    min_recommendable_customers=300,
    min_support_samples=300,
    top_ratio=0.2,
)

CustomerDecisionConfig(
    min_cate=0.0,
    top_k_per_customer=2,
    customer_weight_col=None,
)

SafetyConfig(
    max_customer_negative_share=0.4,
    min_customer_expected_gain=0.0,
)
```

适用场景：
- 准备给业务一版候选推荐名单
- 希望产品池不要太杂

---

### 第三阶段：真实投放前
目标：
- 尽量保证命中率
- 限制资源浪费

建议参数：

```python
ProductDecisionConfig(
    min_ate=你们定义的业务最低有效值,
    min_qini=全产品qini中位数,
    min_auuc=全产品auuc中位数,
    min_top_uplift_lift=全产品该指标中位数,
    max_negative_uplift_ratio=0.3,
    min_recommendable_customers=300或500,
    min_support_samples=500,
    top_ratio=0.1,
)

CustomerDecisionConfig(
    min_cate=业务保本线,
    top_k_per_customer=1,
    customer_weight_col="客户价值权重列",
)

SafetyConfig(
    max_customer_negative_share=0.3,
    min_customer_expected_gain=业务保本线,
)
```

---

## 5. 对你当前场景的具体建议
结合你现在：
- 34 个产品
- 6 万用户
- 单产品单独建模
- 已有 `ATE/CATE/Qini/AUUC`

我建议你当前先用这版：

```python
ProductDecisionConfig(
    min_ate=0.0,
    min_qini=0.0,
    min_auuc=0.0,
    min_top_uplift_lift=0.0,
    max_negative_uplift_ratio=0.4,
    min_recommendable_customers=300,
    min_support_samples=300,
    top_ratio=0.2,
)

CustomerDecisionConfig(
    min_cate=0.0,
    top_k_per_customer=2,
    customer_weight_col=None,
)

SafetyConfig(
    max_customer_negative_share=0.4,
    min_customer_expected_gain=0.0,
)
```

原因：
- 不会过于宽松
- 也不会因为阈值太严把产品全筛没
- 比较适合先产出第一版“可解释的推荐名单”

---

## 6. 最关键的一条：优先用“相对阈值”替代“固定阈值”
因为不同产品的 `cate/qini/auuc` 量纲和分布常常不稳定，所以后面最好改成：

- `min_qini = qini 的中位数`
- `min_auuc = auuc 的中位数`
- `min_top_uplift_lift = top_uplift_lift 的中位数`
- `min_ate = max(0, ate 的 30%分位数)`

也就是不要永远写死 `0.0`，而是让阈值跟着当期产品分布走。

这个思路更适合自动化决策体系。

---

## 7. 一句话总结每个参数
- `min_ate`：产品整体有没有正向价值
- `min_qini`：模型能不能把值得推荐的人排前面
- `min_auuc`：按模型排序投放有没有累计收益
- `min_top_uplift_lift`：Top 人群是否明显更值得投
- `max_negative_uplift_ratio`：误伤风险是否过高
- `min_recommendable_customers`：产品有没有足够大的可投客户群
- `min_support_samples`：评估结果是否稳
- `top_ratio`：你要把营销资源集中在前多少人
- `min_cate`：单客户是否值得推
- `top_k_per_customer`：一个客户最多推几个产品
- `customer_weight_col`：是否优先高价值客户
- `max_customer_negative_share`：客户层再做一次风险过滤
- `min_customer_expected_gain`：单次推荐至少要有多少预期收益

如果你愿意，我下一步可以直接把这套参数解释写进你当前的 `causalml_rec_metrics.py` 文件注释里，并顺手帮你增加一套“基于分位数自动生成阈值”的 `suggest_default_thresholds(eval_df)` 函数。









=================================================================

这些阈值本质上是在回答三件事：

- 这个产品整体值不值得进推荐池
- 这个客户是否值得被推荐这个产品
- 推荐结果是否足够安全、值得资源投入

可按 3 层理解。

---

## 1. 产品层参数：`ProductDecisionConfig`

### `min_ate`
含义：
- 产品级平均因果效应下限
- 代码里 `ate = mean(cate)`，表示这个产品对全体客户平均是否是正向增量

怎么理解：
- `ate > 0`：整体方向正确
- `ate <= 0`：整体不建议投放

怎么调：
- 探索期：`0.0`
- 保守期：略大于 0
- 如果 `Y` 是金额，可以设成业务最小有效值，比如 `500`、`1000`

建议：
- 你当前先用 `0.0`

---

### `min_qini`
含义：
- 产品 uplift 排序能力下限

怎么理解：
- `qini > 0` 说明模型排序优于随机
- 如果 `qini <= 0`，说明即使平均有效，也未必能找准“该推给谁”

怎么调：
- 初期：`0.0`
- 后期：建议用相对阈值，如产品间 `qini` 中位数或 30% 分位数

建议：
- 当前先用 `0.0`

---

### `min_auuc`
含义：
- uplift 累计收益面积下限

怎么理解：
- `auuc > 0`：按模型排序投放比随机更有累计收益

怎么调：
- 初期：`0.0`
- 后期：建议改成相对阈值，例如全产品 `auuc` 中位数

建议：
- 当前先用 `0.0`

---

### `min_top_uplift_lift`
含义：
- Top 人群 uplift 相对整体 uplift 至少高多少
- 代码里是：
  - `top_uplift_lift = top_mean_cate - overall_mean_cate`

怎么理解：
- 用来判断“Top 人群是否真的更值得推”
- > 0 说明模型对高潜客户有识别价值

怎么调：
- 探索期：`0.0`
- 精准投放期：逐步提高
- 后期可用分位数阈值代替固定值

建议：
- 当前先 `0.0`

---

### `max_negative_uplift_ratio`
含义：
- 允许负 uplift 客户占比上限
- 即某产品下 `cate < 0` 的客户比例不能太高

怎么理解：
- 控制误伤风险
- 产品平均可能是正的，但如果很多客户是负 uplift，说明不适合广泛推荐

怎么调：
- 保守：`0.3`
- 平衡：`0.4`
- 宽松探索：`0.5`

建议：
- 银行场景建议先 `0.4`

---

### `min_recommendable_customers`
含义：
- 至少有多少客户 `cate > 0`，这个产品才值得进入推荐池

怎么理解：
- 避免某产品虽然有效，但只对极少数客户有效，业务上不划算

怎么调：
- 绝对值法：`100 / 300 / 500`
- 相对值法：总客户数的 `0.5%~1%`

你这里 6 万客户，建议参考：
- `100`：探索期
- `300`：较合理
- `600`：更重视规模收益

建议：
- 当前先 `300`

---

### `min_support_samples`
含义：
- 产品最少样本量要求

怎么理解：
- 样本太少，ATE/Qini/AUUC 看起来再好也可能不稳

怎么调：
- 探索期：`100~200`
- 常规：`300`
- 严格：`500`

建议：
- 当前 `300` 合理

---

### `top_ratio`
含义：
- Top 人群比例
- 例如 `0.2` 表示看前 20% 客户的 uplift 表现

怎么理解：
- 比例越小，越偏精准营销
- 比例越大，越偏规模投放

怎么调：
- `0.1`：精准
- `0.2`：平衡
- `0.3`：更偏规模

建议：
- 当前 `0.2` 最合适

---

## 2. 客户层参数：`CustomerDecisionConfig`

### `min_cate`
含义：
- 客户-产品级推荐门槛
- 只有 `cate > min_cate` 才推荐

怎么理解：
- 控制单客户是否值得推
- `0.0` 表示只要预测正向就可以推

怎么调：
- 初期：`0.0`
- 如果触达成本高：提高这个阈值
- 如果 `Y` 是金额增量，可直接设业务保本线

建议：
- 当前先 `0.0`

---

### `top_k_per_customer`
含义：
- 每个客户最多推荐几个产品

怎么理解：
- 防止一个客户被推荐过多产品
- 也更贴近客户经理实际执行

怎么调：
- `1`：最严格
- `2`：较实用
- `3`：偏宽松

建议：
- 你当前更建议 `2`
- 若产品有互斥关系，建议直接 `1`

---

### `customer_weight_col`
含义：
- 客户权重列名
- 用于把客户价值纳入排序

怎么理解：
- 如果客户价值不同，不应该只按 `cate` 排序
- 可以优先高价值客户

可接的字段示例：
- AUM
- 客户等级
- 存款规模
- 战略客户标识
- 客户经营价值分层

建议：
- 暂时没有就 `None`
- 后续非常建议接入

---

## 3. 安全层参数：`SafetyConfig`

### `max_customer_negative_share`
含义：
- 客户推荐时再次做风险过滤
- 当前代码里本质上还是利用产品层 `negative_uplift_ratio` 来过滤高风险产品

怎么理解：
- 防止产品整体负向风险过高，却因为个别客户 `cate > 0` 仍然被放进推荐名单

怎么调：
- 保守：`0.3`
- 平衡：`0.4`
- 宽松：`0.5`

建议：
- 跟 `max_negative_uplift_ratio` 保持一致即可
- 当前建议 `0.4`

---

### `min_customer_expected_gain`
含义：
- 推荐出去之前，再过滤一次“预期收益太小”的客户-产品对

怎么理解：
- 有些 `cate` 虽然大于 0，但太小，不值得客户经理投入精力

怎么调：
- 还没建立业务价值换算前：`0.0`
- 建立保本线后：设为最小可接受收益

建议：
- 当前先 `0.0`
- 后面接 ROI 时重点用这个参数

---

## 4. 你当前最适合的初始参数

结合你现在：
- 34 个产品
- 6 万用户
- 单产品独立建模
- 已经有 `cate / ate / qini / auuc`

推荐先用这版：

```python
ProductDecisionConfig(
    min_ate=0.0,
    min_qini=0.0,
    min_auuc=0.0,
    min_top_uplift_lift=0.0,
    max_negative_uplift_ratio=0.4,
    min_recommendable_customers=300,
    min_support_samples=300,
    top_ratio=0.2,
)

CustomerDecisionConfig(
    min_cate=0.0,
    top_k_per_customer=2,
    customer_weight_col=None,
)

SafetyConfig(
    max_customer_negative_share=0.4,
    min_customer_expected_gain=0.0,
)
```

原因：
- 不会太松，能控住风险
- 也不会太严，避免 34 个产品被筛掉太多
- 适合先产出第一版可解释的推荐结果

---

## 5. 更推荐的调参原则：从固定阈值改成相对阈值
长期看，不建议一直写死 `0.0` 或 `0.4`，因为不同批次、不同产品分布会变。

更好的自动化方式是：

- `min_qini = 全产品 qini 中位数`
- `min_auuc = 全产品 auuc 中位数`
- `min_top_uplift_lift = 全产品该指标中位数`
- `min_ate = max(0, ate 的 30%分位数)`

这样阈值会跟着当期结果分布自动变化，更适合自动化决策体系。

---

## 6. 一句话记忆每个参数

- `min_ate`：产品整体是否正向
- `min_qini`：排序是否有效
- `min_auuc`：累计收益是否优于随机
- `min_top_uplift_lift`：Top 人群是否明显更值推
- `max_negative_uplift_ratio`：误伤风险是否过高
- `min_recommendable_customers`：产品有没有足够可投客群
- `min_support_samples`：样本量够不够稳
- `top_ratio`：营销资源聚焦前多少人
- `min_cate`：单客户是否值得推荐
- `top_k_per_customer`：每个客户最多推几个产品
- `customer_weight_col`：是否优先高价值客户
- `max_customer_negative_share`：客户层再做一遍风控
- `min_customer_expected_gain`：单次推荐至少要有多少收益

如果后续你要把这套体系真正做成自动生产，我最建议的下一步不是继续手调，而是加一个“根据当前 34 个产品分布自动给出建议阈值”的函数。这样每次跑完模型后，系统可以自动产出一版推荐阈值。