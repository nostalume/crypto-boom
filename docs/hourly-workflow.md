# 一键小时预测报告与研究流程

在仓库根目录运行，Python 3.12。安装/同步可选依赖：

```powershell
uv sync --extra prediction
```

依赖同步可能联网。报告仅访问公开行情，不需要交易账户，不下单。

## 1. 准备模型（一次性）

拥有本项目已完成的本地尺度研究目录时，导出验证胜出的单个上涨空间头：

```powershell
uv run --extra prediction crypto-boom-study export-hourly --study ../data/path-quality-20261001/expanded-path-v1/scale --model data/hourly-upside-v1 --trust-model
```

示例适用于本机 `.publish-work` 布局；其他机器替换 `--study` 路径。
研究目录和模型不包含在git中。`--trust-model` 表示文件确为自己可信的joblib，
joblib可执行代码，不要用它加载网络来源不明的文件。导出不改变参数，不导出失败的质量头。
模型目录必须不存在；已有目录不会覆盖。

## 2. 一条命令生成最新小时报告

```powershell
uv run --extra prediction crypto-boom-study report --model data/hourly-upside-v1 --symbol STXUSDT --trust-model
```

自动读取最新已完成分钟数据，计算24小时历史特征，从**最新完成UTC整点**预测未来6小时。
默认写入唯一的 `data/reports/<UTC时间>/`，可重复运行，无需手动改文件名：

- `report.md`：中文可读解释；
- `prediction.json`：结构化结果、模型/来源身份及训练来源；
- `source.parquet`：这次读取的行情快照，可供回放。

可用 `--output data/reports/my-run` 指定尚不存在的目录。每次只处理一个币种，
避免默默发起几百币种的网络请求。未提供多币种并发或持续运行服务。

**解释边界：** P90是未来最大上涨幅度的条件分位数估计，不是90%上涨概率。
预测起点可能比最新完成分钟早0—59分钟，报告会明确注明。保持率、下跌和终点预测
当前不可用，JSON为null，不应理解为风险为零。本工具不决定买入、仓位或退出。

## 3. 离线回放（不联网）

```powershell
uv run --extra prediction crypto-boom-study report --model data/hourly-upside-v1 --bars data/reports/my-run/source.parquet --output data/reports/replay --trust-model
```

`--bars`与`--symbol`互斥，可重复传入多份同币种规范分钟parquet。回放报告明确标为
historical_replay。需至少有截至最后完成整点的1441根连续有效分钟；断档不能填造。

## 4. 从样本池研究到新训练

已有公共子命令可查看具体限制：

```powershell
uv run --extra prediction crypto-boom-study acquire --help
uv run --extra prediction crypto-boom-study audit --help
uv run --extra prediction crypto-boom-study build --help
uv run --extra prediction crypto-boom-study screen --help
uv run --extra prediction crypto-boom-study rolling --help
```

`acquire`获取/复用明确范围样本并输出池台账；`audit`核查本地源；`build`独立生成
因果特征和未来路径目标；`screen`/`rolling`为已有31特征路径研究的比较入口。
缓存由源、策略和代码身份绑定，目标不会进入特征。示例中的路径为自己已生成的文件：

```powershell
uv run --extra prediction crypto-boom-study build --pool data/pool/pool.json --output data/dataset --step-minutes 60 --minimum-turnover 0 --horizons 360
uv run --extra prediction crypto-boom-study train-hourly --dataset data/dataset/DATASET_ID.json --train-end 2026-01-01 --model data/hourly-new-fit
```

将 `DATASET_ID.json` 替换为build输出的manifest路径。训练使用固定152维小时配方、
六小时训练网格，清除跨训练截止时间的标签，保留零活动但拒绝不足历史。
限定100—200000候选训练起点；只训练上行P90，不自动选型或宣布有效。
它使用传入数据集中截止日前的样本，**不会自动保留论文中的30币种**；需要验证时，
必须在调用方建立独立训练/评价人口。新产物标记 `new_fit_not_evaluated`。

如要使用已经评价的研究模型，应执行export-hourly，而不是重新训练后引用旧成绩。
完整多尺度实验矩阵仍为本地研究，不宣称screen/rolling已经自动评价新的小时配方。

## 5. 常见问题

| 提示/现象 | 处理 |
|---|---|
| 需要trust-model | 确认来源可信后显式加参数；不要盲目信任文件 |
| snapshot became stale | 获取时跨分钟，重新运行；不偷偷使用旧行情 |
| 缺少历史或断档 | 检查币种历史长度与数据完整性，不以填充规避 |
| sklearn版本不兼容 | 使用锁定环境或重新导出/验证；不忽略pickle版本警告 |
| 输出目录已存在 | 使用默认时间目录或新目录，原报告不会覆盖 |
| 最新分钟与预测起点不同 | 这是训练一致的UTC整点策略，滞后明确展示 |

网络获取失败不会回退到缓存并冒充最新；生成失败不发布半份报告。CLI可能显示原始异常
以便诊断。报告快照和模型是本地数据，不会自动上传或进入git。
