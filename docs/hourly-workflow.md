# 当前模型研究配方（不是公共扫描接口）

公共操作统一为 [crypto-boom scan](public-interfaces.md)。本页仅记录当前选定的
`hour_scaled_context_up_360` 研究模型：24小时连续历史、60分钟聚合、152维特征、
未来6小时最大分钟收盘上涨空间 P90。模型比较、失败结果与边界见[论文式报告](path-prediction-study.md)。
尺度和目标写入模型元数据；不是框架名称，也不是所有未来模型的固定配置。

## 已评价模型迁移（一次性、不重新训练）

以下路径均是研究者自己的输入/输出示例，不是产品预设：

```sh
uv sync --extra prediction
uv run --extra prediction python -m crypto_boom.research.hourly_cli export-hourly --study PATH_TO_SCALE_STUDY --model data/research-export --trust-model
uv run --extra prediction python -m crypto_boom.research.hourly_cli publish --model data/research-export --registry data/models --trust-model --activate
uv run --extra prediction crypto-boom scan
```

已有旧 `hourly-upside-v1` 导出模型可直接执行 publish，省略 export。
`--registry` 应对应公共配置 data_dir 下的 models。发布返回内容 ID；可不加 --activate，
之后使用公共 `model activate --id ID --trust-model` 选择模型。迁移不重新拟合权重。
模型与研究数据不随 Git 提供；joblib 能执行代码，只允许使用自己可信的产物。

## 训练与研究回放

```sh
uv run --extra prediction crypto-boom-study build --pool data/pool/pool.json --output data/dataset --step-minutes 60 --minimum-turnover 0 --horizons 360
uv run --extra prediction python -m crypto_boom.research.hourly_cli train-hourly --dataset data/dataset/DATASET_ID.json --train-end 2026-01-01 --model data/new-fit
uv run --extra prediction python -m crypto_boom.research.hourly_cli report --model data/research-export --bars PATH_TO_CANONICAL_BARS --output data/replay --trust-model
```

build 输出 manifest 替换 DATASET_ID。训练采用固定当前配方和六小时训练网格，清除
跨训练截止时间标签，保留零活动而拒绝历史不足；100—200000候选起点。
只拟合上涨 P90，新模型标记 `new_fit_not_evaluated`，不自动保留论文的30币种，
不自动继承旧测试成绩。研究者负责独立训练/测试人口、时间切分和评价。
单标的在线报告仍可用 research 脚本的 --symbol 作诊断，不是部署推荐操作。

旧 `crypto-boom-study train-hourly/export-hourly/report` 已移除，改为上面的 research 模块命令。
acquire/audit/build/screen/rolling 保留为既有研究工具；通用数据/特征实现可复用，
不为每个候选研究模型继续增加公共子命令。公共扫描无需研究目录或某个实验的固定路径。

相对异常独立输出、时间/持续性目标、困难对照与事件权重、存续偏差等研究问题仍未全部
解决。本次接口重构不改变研究结果，也不代表盈利能力验证。
