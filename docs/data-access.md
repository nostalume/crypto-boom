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
    (settings.data_root / "canonical/archives",),
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

[scan]
workers = 4
timeout_seconds = 900
```

数据根路径相对配置文件。旧 legacy_corpora 配置已移除；额外导入语料用研究命令
显式 `--reuse-corpus`，不长期挂载实验目录。
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
旧 `[scan].data_dir` 已删除；扫描直接使用 `ProjectSettings`，不再维护第二套设置对象。
旧 archive 命令的显式 `--output`、其他模型研究脚本的显式产物路径暂未统一改写，
它们不属于新的默认流程。不要把这一阶段称为全部旧入口迁移完成。

## 当前迁移与边界

已将本机在用 models/snapshots/reports 从 `.publish-work/data` 移至统一根，迁移逐文件
核对内容哈希；历史报告中的旧 report_directory 是生成时的位置，未改写历史证据。
四批历史 corpus 已实际迁入 `canonical/archives`：148 个不同币种、2,035 个原生分区，
4,070 个文件共 8,545,414,419 字节，逐文件核对迁移前后哈希。旧 corpus 路径已移除，
没有目录跳转或后台重复下载。
研究模型导出也已移至 `runs/selected-hourly-export`，逐文件核验哈希后移除空的
`.publish-work/data`；没有删除历史原始行情。

当前研究入口分别在：

- `runs/expanded-path-current-20261001/`：129 币种扩展研究池，705,769 条样本。
- `runs/acquired-path-current-20261001/`：8 币种获取池，43,008 条样本；与扩展池有一个币种重叠。
- `runs/universe-expansion-confirm-20260928-v1-migrated/pool.json`：2026 年 5–8 月，141 币种、564 个分区。
- `runs/universe-expansion-earlyblock-20260929-v1-migrated/pool.json`：2025 年 1–4 月，113 币种、434 个分区。

后两者只迁移原生语料和本地库存池（`local-native-pool-v1`），未重新生成标签或并入
训练集；币种、分区身份、原时间边界和本地缺月台账保留。可从池记录取相对分区路径，
或使用公共 `select_minute_partitions` 按标的/时间选择，再独立调用分钟读取和周期聚合。
这是研究库存记录，不冒充 `sample_pool` 的交易所可用性证据。


前两者的特征和目标均在共享 `derived/features`、`derived/targets`，采用 60 分钟起点、
360 分钟目标、`minimum_turnover=0`。内容寻址的 dataset JSON 由
`research.path_dataset.load_path_dataset` 读取；不能将 `pool.json` 当作 dataset。
扩展池的 pool 是研究专用本地审计记录，不是 archive 随机抽样池，不能传给
`load_sample_pool`；本地缺月仍标记为 locally_absent，不伪称交易所 not_found。
原 heldout_symbol 分组保留。129 币种的新旧特征/目标逐表完全一致，8 币种新旧
数据集也完全一致；这是迁移一致性证据，不是模型性能提升。两种 pool 的不同采样
含义不以统一读取函数掩盖。

### 可迁移的缓存与清单

新写入使用 `feature-cache-v2`、`sample-pool-v2`、`path-dataset-v2`；旧 v1 仍可读取，
不覆盖旧证据。目标缓存继续使用已与路径无关的 v1。

- 特征身份包含源文件内容哈希/大小及代码、采样配方，不含文件名、绝对路径或输入顺序。
  这是**文件字节身份**：重新编码 Parquet 即使逻辑行相同，也可能产生新身份。
- 样本池身份包含选择、可用性及分区事实；数据集身份包含池身份与特征/目标身份。
- 新清单在磁盘存相对引用，加载后返回绝对路径运行视图；不会改写磁盘清单。
  整体迁移被引用目录树可保留引用；只复制 manifest 不够。生成引用要求同卷，外部挂载
  不会自动随数据根迁移，也不会全盘搜索失效引用。
- 单独移动原始文件时，可通过 `feature_source_paths(..., paths=[...])` 或研究
  `build_target_cache(..., source_paths=[...])` 显式重新绑定，并验证字节身份。
  数据集构建保存当前原始文件绑定，供序列研究使用；绑定变化须写入新的运行目录。
  读取已缓存特征不要求原始文件在线，重新使用原始数据时才核验源身份。

对可读取的获取池，可显式导出新清单，不搬运行情、不覆盖历史：

```python
from crypto_boom.sample_pool import export_sample_pool

export_sample_pool(
    settings.data_root / "runs/acquired-path-current-20261001/pool.json",
    settings.data_root / "runs/portable-pool/pool.json",
)
```

其中 `settings` 来自前述 `project_settings()`。导出前校验旧池及分区；目标已存在则拒绝。
旧特征缓存不会自动转换成 v2；新构建按新身份写入，旧缓存保留以供历史回放。
仍未实现基于引用关系的安全去重/清理，以及最新窗口与历史分区的细粒度共享。

本轮不引入万能 DataManager、类型插件注册器或数据库；现有版本化归档布局已经能完成
当前分区查找。若后续多来源查询确实需要索引，再以真实消费者确定最小索引契约。

### 历史证据与恢复边界

历史研究报告、v1 缓存和稀疏序列产物仍保留，不将旧评估改写为新实验。
旧缓存可读取已有特征/目标，但旧清单中的绝对原始路径已退役，不能直接重跑原脚本；
当前开发使用上述新 dataset。需要复现旧输入时，用迁移映射重新绑定原始数据，
再由 `feature_source_paths(..., paths=...)` 核验内容，不自动猜测路径。

本机映射、逐文件哈希、迁移和重建脚本、数值一致性结果位于
`data/path-quality-20261001/broad-cutover/`。这些是本地操作证据，不是公共接口。
旧 `runs/path-pool-current-20261001` 是上一轮的历史清单，以本轮 acquired 路径为准。
未删除历史衍生缓存，也未重新训练模型；目录整理不改变模型权重和原评估结论。
