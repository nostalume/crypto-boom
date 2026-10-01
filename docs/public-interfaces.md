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
| 扫描/报告 | `market_scan.ScanSettings/scan_market` | 全部成员、共同起点、成功/失败台账、原子发布 |
| 研究 | `research/*` 和保留的实验 CLI | 训练、比较、选型、导出、特定模型回放，不是公共部署契约 |

不是任意模型插件系统。当前 `numeric-sequence-v1` 支持 1—8 个数值输出和有限历史
序列配方（最多1440分钟），每个估计器须匹配特征数及 predict 协议。不同输入形态需要
显式新协议与验证，不能只替换权重。公共运行层不导入研究层。

## 配置和模型指派

可复制 `scan.example.toml` 为 `scan.toml`；也可传 `scan --config FILE`。
只配置 `data_dir`、`workers`（1—4）、`timeout_seconds`（1—900）。相对数据目录
按配置文件目录解析；无配置时使用当前目录的 `data/`。显式不存在的配置会拒绝。
未知键（包括 symbol）拒绝，避免误以为只扫描一个标的。

`data_dir/models/<内容ID>/` 保存不可变 manifest 和权重；`active.json` 是本地选择：

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

最多2000成员，超限整体拒绝而不截断；最多4并发、4请求/秒、5000请求、900秒预算，
单请求12秒超时，已发出的请求可能在预算后才返回。使用交易所分钟权重预算的一半；
HTTP 418/429 停止继续获取，不重试其他域名。参见
[Binance REST 限流规范](https://developers.binance.com/en/docs/products/spot/rest-api)。
共享IP上其他进程也消耗额度，本工具不保证不会被限流。同一数据目录建议只运行一个扫描进程。
快照可持续增长；不自动删除审计数据，保留/清理由部署方按其审计期限管理。

## 报告与退出码

每次新建 `data_dir/reports/<UTC时间-运行ID>/`：

- `report.md`：可读摘要、排序前30预览、范围、时效和风险边界。
- `report.json`：完整模型契约、所有成员状态、源哈希、缓存身份和预测数值。
- `predictions.csv`：所有成功和失败成员，可排序；JSON/CSV 不受前30预览限制。
- `exchange-info.json`：原始范围快照，可审核成员来源。

成功状态是 `success`；失败区分 `metadata_error`、`fetch_error`、
`data_or_prediction_error`、`not_completed`、`not_attempted`。
退出码 0 = 范围内全员成功，2 = 部分完成（报告仍发布），1 = 启动/范围/发布失败。
范围无法确定时不发布虚假覆盖报告。报告原子发布，原报告不覆盖。

P90 是条件分位数而非90%上涨概率；数值排序不等于交易建议。新上市人口可能超出训练
覆盖，数据完整也不证明预测有效。未部署的下跌/质量头不应被读作风险为零。
