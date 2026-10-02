# 公共接口与全盘扫描契约

## 一条主线流程

`crypto-boom scan` → 读取已激活模型 → 获取交易所范围与时钟 → 全盘批量取数 →
规范校验 → 因果特征 → 数值预测 → 完整覆盖台账及报告。

安装 `prediction` extra，先发布并激活可信模型（见[当前研究配方](hourly-workflow.md)）。
随后每次只需 `uv run --extra prediction crypto-boom scan`，不传标的、hourly 或模型路径。

默认全盘范围是 **Binance 现货、USDT 报价、快照时 TRADING 的现货成员**，不是全球市场，
也不以训练中出现过的币种过滤最新范围。保留不支持的元数据、历史不足、断档和请求失败；
这些成员没有分数，不是假阴性。不能把自然无成交直接作为坏数据，也不能填造缺失分钟。

## 边界与可复用能力

| 所有者 | 接口 | 契约 |
|---|---|---|
| 历史获取 | `history`、`storage`；原 archive 系列 CLI | 已声明范围的历史下载与校验，不含模型逻辑 |
| 最新获取 | `market_data.SpotSnapshotClient.universe/bars` | 范围快照、统一时间、请求预算、精确窗口缓存 |
| 数据清理/准入 | `bars.decode_minute_page/admit_bars` | 统一分钟表、排序、价格/成交/时间校验；拒绝而非猜测修复 |
| 公共特征 | `features.past_sequence/SequenceRecipe/sequence_matrix` | 只看过去；由配方给出历史、聚合及窗口；不含特定模型名 |
| 模型运行 | `model_runtime.publish_model/activate_model/load_active_model/predict_bars` | 内容寻址、显式信任、兼容检查、配方和输出语义绑定 |
| 扫描/报告 | `config.ProjectSettings`、`market_scan.scan_market` | 全部成员、共同起点、成功/失败台账、原子发布 |
| 研究 | `research/*` 和保留的实验 CLI | 训练、比较、选型、导出、特定模型回放，不是公共部署契约 |

不是任意模型插件系统。当前 `numeric-sequence-v1` 支持 1—8 个数值输出和有限历史
序列配方（最多1440分钟），每个估计器须匹配特征数及 predict 协议。不同输入形态需要
显式新协议与验证，不能只替换权重。公共运行层不导入研究层。

## 配置和模型指派

项目统一配置见 [组合式数据接口](data-access.md)。复制 `crypto-boom.example.toml` 为
`crypto-boom.toml`，以 `[data].root` 指定唯一数据根；相对路径按配置文件位置解析。
扫描参数仍是 `[scan].workers`、`timeout_seconds`。命令可用 `--config FILE` 显式选择，
否则向上查找项目配置；没有配置时仅在识别出的本项目根使用 `data/`，不按子目录另建数据根。
旧 `[scan].data_dir` 已移除，请改为 `[data].root`；只保留一套项目配置。

`data.root/models/<内容ID>/` 保存不可变 manifest 和权重；`active.json` 是本地选择：

```sh
uv run --extra prediction crypto-boom model list
uv run --extra prediction crypto-boom model activate --id MODEL_ID --trust-model
uv run --extra prediction crypto-boom scan
```

模型 ID 是配置/权重身份，不是研究目录路径。激活时校验来源可信性、内容哈希和环境；
每次扫描再次校验。joblib 可以执行代码，哈希不能证明来源安全，绝不激活来历不明的模型。
权重不随 Git 发布；换机器需搬运可信注册模型并重新显式激活。无需修改扫描源码。
当前选择的模型及表现记录在[研究报告](path-prediction-study.md)，不写死在公共调用中。

## 最新、缓存和预算

先取交易所时钟，再按模型声明的决策步长取最近完成边界；所有币种使用相同边界。
扫描途中不混入新分钟。报告列出起点、估计年龄，以及完成时是否可能已有更新起点。
“最新”指扫描开始的共同快照，不承诺完成瞬间实时。缓存仅复用同一币种、同一起点、
同一历史长度且哈希正确的数据；不以旧起点缓存冒充新行情。

最多2000预测成员，超限整体拒绝而不截断；最多4并发、4请求/秒、5000行情请求、900秒预算，
另有最多3次产品元数据请求（每次最长10秒，仍受同一整轮截止时间约束），
单请求12秒超时，已发出的请求可能在预算后才返回。使用交易所分钟权重预算的一半；
HTTP 418/429 停止继续获取，不重试其他域名。参见
[Binance REST 限流规范](https://developers.binance.com/en/docs/products/spot/rest-api)。
共享IP上其他进程也消耗额度，本工具不保证不会被限流。同一数据目录建议只运行一个扫描进程。
快照可持续增长；不自动删除审计数据，保留/清理由部署方按其审计期限管理。

## 报告与退出码

每次新建 `data.root/reports/<UTC时间-运行ID>/`：

- `report.md`：可读摘要、排序前30预览、范围、时效和风险边界。
- `report.json`：完整模型契约、所有成员状态、源哈希、缓存身份和预测数值。
- `predictions.csv`：所有成功和失败成员、数据来源及产品候选，可排序；JSON/CSV 不受前30预览限制。
- `products.csv`：四类产品的独立覆盖表，包含同名现货候选、映射状态、对应现货预测状态。
- `exchange-info.json`：原始范围快照，可审核成员来源。

成功状态是 `success`；失败区分 `metadata_error`、`fetch_error`、
`data_or_prediction_error`、`not_completed`、`not_attempted`。
退出码 0 = 范围内全员成功，2 = 部分完成（报告仍发布），1 = 启动/范围/发布失败。
范围无法确定时不发布虚假覆盖报告。报告原子发布，原报告不覆盖。

P90 是条件分位数而非90%上涨概率；数值排序不等于交易建议。新上市人口可能超出训练
覆盖，数据完整也不证明预测有效。未部署的下跌/质量头不应被读作风险为零。

## 2026-10-01 数据处理约化

- 公共扫描仍为 `crypto-boom scan`，不新增管理器、插件层或模块。
- `crypto-boom-predict` / `crypto-boom-study` 的既有命令用法保留，但实现归入
  `research.predict_cli` / `research.study_cli`；脚本模块路径相应迁移，不保留转发壳。
- Binance 范围选择归 `binance_source.select_spot_usdt_universe`，数据获取层不再
  为调用这一函数导入 qualification/live 验证链。旧 qualification 函数导入点已移除。
- `load_bar_files` 仍校验输入预算、UTC 类型和全文件哈希，但在 Parquet 扫描时筛选
  时间窗口，不再先展开整个文件再截取。保留文件顺序、重复行和原有严格时间上界；
  数据准入仍负责拒绝重复，读取器不会静默去重或填补。
- 删除无调用者的固定五分钟/百万成交额选择器及未使用的 ALL_FEATURES 合集。
  可复用因果特征仍留在公共层；模型、训练人口及阈值仍由研究拥有。

本轮不改预测算法、权重、目标或缓存格式。特征构建的代码身份变化会产生新缓存身份，
旧缓存和历史研究证据不删除。保留单标的研究取数，因为其完成分钟/超时契约与全盘共同
快照不同，不能仅为减少文件数强行合并。庞大的历史验证子系统尚未整体重构。

## 2026-10-02：交易产品池与预测数据源分离

当前决议：接受 Binance、OKX 的 CEX 现货和永续产品；不建设 OKX K 线、成交或
特征数据接入。OKX 仅查询产品元数据。默认延续 USDT 范围，永续与交割合约分开，
不自动扩展到币本位合约、DEX、股票或商品类模型。

- 行情/特征/现有模型：仍为 Binance 现货，不把现货预测描述成合约收益预测。
- 产品池：分别保存交易所、spot/perpetual、原生产品 ID、报价/结算、合约规格及快照时间。
- 跨市场映射：同名 ticker 仅是候选，不能自动确认资产身份；不自动去掉 `1000` 等倍率。
  非现货产品必须先确认映射；缺少可用 Binance 现货输入时报告未覆盖，而不是零分。
- 产品状态：公共 `live/TRADING` 不等于账户权限、充分流动性或可成交收益。
  永续还涉及资金费、基差和清算风险，现有上行分位数模型没有预测这些风险。

本机已生成四类元数据池，位于 `data/runs/cex-products-20261002/20261002T005141/`：
Binance 现货 503，USDT 永续 528（526 COIN、2 INDEX）；OKX 加密现货 305，
加密 USDT 线性永续 286。OKX 使用 `instCategory=1`；Binance 分类原样保留，
不假称已经完成全部资产类型审查。数量是产品数，跨交易所、产品类型可能重叠。

该阶段只有研究脚本导出的 `products.csv`、元数据原文和来源哈希。
生产 `crypto-boom scan` 现已接入元数据池，命令不变；预测覆盖分母仍是 Binance 现货。
报告采用 `market-scan-v2`：`rows` 保留现货预测，`product_pool.products` 独立列出产品。
`native` 只表示 Binance 现货原生对应；其他同名产品一律为 `ticker_match_unverified`，
不据此自动执行、过滤模型人口或将分数作为合约收益预测。`no_exact_ticker_match` 不等于
资产不存在（可能是别名/倍率），`source_unavailable` 也不等于未上市。不自动去掉倍率。

`product_metadata_status` 和预测 `status` 分开，CLI 同时输出两者；退出码仍按现货
预测完成情况决定。产品来源失败时可以输出有效现货预测，使用产品池时必须检查前者。
每个来源记录时间、错误或筛选数量、接收字节哈希及 base64 原文；池快照时间不冒称
等于模型共同起点。没有缓存兜底、重试或跳域；418/429 后跳过该主机的后续产品请求。
本轮只有产品元数据，不包含 OKX K 线/成交接口、账户权限查询、下单或资产同一性认证。
