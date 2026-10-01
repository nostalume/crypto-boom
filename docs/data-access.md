# 组合式数据能力与统一数据根

## 不是一个万能 read(request)

不同维度拥有不同职责，调用方显式组合；不按数据类型写一个大型分派函数：

| 维度 | 现有实现 | 所有权 |
|---|---|---|
| 来源与标的 | `market.InstrumentId(VenueId(...), environment, symbol)` | 明确市场及原生标的，不绑定模型 |
| 时间范围 | `market.TimeWindow` | UTC 微秒、半开区间 `[start,end)`；有时区 datetime 可转换 |
| 原生历史来源 | `storage.source.select_minute_partitions` | 查找 Binance 现货分钟归档、来源校验、缺失月份、冲突版本拒绝 |
| 数据类型读取 | `bars.read_bar_window` | 分钟 Parquet 的有界读取与列投影，不联网，不自动换周期 |
| 时间尺度 | `bars.BarPeriod` / `aggregate_minute_bars` | 真实分钟 OHLC 聚合，无填充、插值、未来数据 |
| 最新来源 | `market_data.SpotSnapshotClient` | 市场范围、交易所时钟、网络预算与完成窗口 |
| 因果特征 | `features` | 用已选数据计算特征，不决定存储位置或训练目标 |

资金费率、逐笔成交、盘口应拥有各自的来源适配器、读取与变换函数，可以复用来源标识、
时间范围和项目根；不应伪装为 K线。**本轮没有实现这些类型的存储读取能力。**
当前历史归档适配器只承诺 Binance Spot production / 1m；其他来源明确拒绝。
`TimeWindow` 表达事件时间，不证明数据在过去该时点已可获取；历史可知性仍需额外证据。

## 组合示例（离线）

```python
from datetime import UTC, datetime
from crypto_boom.config import project_settings
from crypto_boom.market import Environment, InstrumentId, TimeWindow, VenueId
from crypto_boom.storage.source import select_minute_partitions
from crypto_boom.bars import (
    SOURCE_COLUMNS, BarPeriod, read_bar_window, aggregate_minute_bars,
)

settings = project_settings()
instrument = InstrumentId(VenueId("binance", "spot"), Environment.PRODUCTION, "BTCUSDT")
window = TimeWindow.from_datetimes(
    datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 2, tzinfo=UTC),
)
selection = select_minute_partitions(
    (settings.data_root / "canonical/archives", *settings.corpus_roots),
    instrument, window,
)
if selection.missing_months:
    raise ValueError(f"Missing archive months: {selection.missing_months}")
minutes, receipts = read_bar_window(
    [part.path / "klines.parquet" for part in selection.partitions], window,
    columns=(*SOURCE_COLUMNS, "open_price"),
)
hours = aggregate_minute_bars(minutes, BarPeriod(60))
```

示例需要本地实际存在的数据，不会偷偷联网补齐。多标的是明确的标的集合逐一组合，
全盘标的集合仍由交易所范围快照产生，而非研究样本池。
选到月份不等于分钟完整：缺失分钟、坏质量、边缘不完整周期由数据准入/聚合继续拒绝。
同一月份多个不同 manifest 会报版本歧义；已有研究可继续读取已固定版本的文件。
不按文件修改时间或挂载顺序决定数据真相。

精确分钟时点可用 `[t,t+1分钟)`；“截至某时点之前的历史”用明确的回看窗口表达，
目前没有隐式 nearest/asof 填补。全盘 latest 在扫描开始解析为共同完成边界。
数据周期、历史回看长度和未来目标长度是不同参数，后两者由模型/研究拥有。

聚合以 UTC Unix epoch 对齐，周期为1—1440整数分钟；要求连续、有效、完整的桶。
open/close 取首末，high/low 取最大最小，报价成交额、主动买入报价额、笔数求和。
聚合结果是 Float64 分析视图，不替代原始归档的 decimal 精度。
必须有真实 `open_price`；旧模型投影缺少该列时不能用前收盘价伪造开盘价。
聚合后不是分钟数据，不能直接喂给声明分钟输入的模型或再次冒充分钟聚合。

## 一个项目配置

复制 `crypto-boom.example.toml` 为 `crypto-boom.toml`：

```toml
[data]
root = "data"
legacy_corpora = []

[scan]
workers = 4
timeout_seconds = 900
```

路径相对配置文件；legacy_corpora 相对数据根，也可明确给出绝对路径。
配置自动向上查找；命令行 `--config` 优先，不悄悄混合多个配置。
无项目上下文则要求显式配置，避免在陌生目录建立另一个数据根。
本机 `.publish-work/crypto-boom.toml` 使用 `root="../data"`，故实际根为
`E:\Proj\script\crypto-boom\data`；该本机配置不入 Git。

默认落点：

```text
data/
├─ raw/monthly/          # 新获取的原始归档
├─ canonical/archives/  # 新规范历史分区，沿用其版本化原生布局
├─ snapshots/           # 全盘扫描的精确完成窗口与校验记录
├─ derived/features/    # 跨研究运行共享的特征缓存
├─ derived/targets/     # 跨研究运行共享的目标缓存
├─ models/              # 注册模型与激活记录
├─ reports/             # 扫描报告
└─ runs/                # 样本选择、数据集清单等实验运行记录
```

`crypto-boom scan/model`、研究 `acquire/build` 和当前研究模型 `publish` 的默认存储
均接项目配置。`acquire/build --output` 只覆盖实验记录位置，不再将新行情/缓存复制进实验。
底层函数仍允许调用方显式指定存储根；旧研究调用不传新参数时保留原有隔离契约。
旧 archive 命令的显式 `--output`、其他模型研究脚本的显式产物路径暂未统一改写，
它们不属于新的默认流程。不要把这一阶段称为全部旧入口迁移完成。

## 当前迁移与边界

已将本机在用 models/snapshots/reports 从 `.publish-work/data` 移至统一根，迁移逐文件
核对内容哈希；历史报告中的旧 report_directory 是生成时的位置，未改写历史证据。
历史 corpus 通过 legacy_corpora 挂载复用，读取时验证分区，不复制129币种大样本库。
没有删除旧研究目录、临时文件或任何历史数据。

仍待后续迁移：

- 冻结 pool/dataset 清单中的绝对路径引用；
- 老特征缓存身份对源路径的依赖（目前仅同路径、同配方跨运行共享）；
- 基于引用关系的安全去重/清理，以及最新窗口与历史分区的更细粒度共享。

本轮不引入万能 DataManager、类型插件注册器或数据库；现有版本化归档布局已经能完成
当前分区查找。若后续多来源查询确实需要索引，再以真实消费者确定最小索引契约。
