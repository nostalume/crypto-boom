"""One report operation: trusted model + live/replayed bars -> JSON and Markdown."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from crypto_boom import _artifacts
from crypto_boom.bars import MINUTE_US, load_bar_files
from crypto_boom.research.hourly import forecast_hourly, load_hourly_model


def generate_prediction_report(
    model_directory: Path,
    output: Path,
    *,
    symbol: str | None = None,
    bar_paths: list[Path] | None = None,
    trusted: bool = False,
) -> dict:
    """Fresh directory only; single symbol, no silent network fallback in replay."""
    if (symbol is None) == (not bar_paths):
        raise ValueError("choose exactly one of live symbol or replay bar paths")
    if output.exists():
        raise FileExistsError(output)
    model, metadata = load_hourly_model(model_directory, trusted=trusted)
    server_ms = None
    if symbol is not None:
        from crypto_boom.latest_market import fetch_latest

        source, server_ms = fetch_latest(symbol, minutes=1501)
        if (
            source["open_time"].dt.epoch("us")[-1] + MINUTE_US
            != server_ms // 60000 * MINUTE_US
        ):
            raise ValueError("latest source is stale")
        receipts = [{"source": "Binance public Spot minute endpoint"}]
    else:
        source, receipts = load_bar_files(bar_paths or [], start_us=0, end_us=2**63 - 1)
    result = forecast_hourly(model, metadata, source)
    if not re.fullmatch(r"[A-Z0-9]{3,32}", result["symbol"]):
        raise ValueError("unsupported report symbol")
    result.update(
        mode="live_hourly" if symbol is not None else "historical_replay",
        generated_at_utc=datetime.now(UTC).isoformat(),
        server_ms=server_ms,
        provenance=metadata["provenance"],
        sources=receipts,
    )
    decision = datetime.fromtimestamp(
        result["decision_us"] / 1_000_000, UTC
    ).isoformat()
    text = (
        f"# {result['symbol']} 实验性预测报告\n\n"
        f"- 模式:{result['mode']}\n- 预测起点(UTC):{decision}\n"
        f"- 相对最新已完成分钟滞后:{result['hour_lag_minutes']} 分钟(保持训练时的整点规则)\n"
        f"- 未来六小时最大上涨空间 P90:**{result['upside_p90']:.2%}**\n"
        f"- 过去24小时有成交分钟比例:{result['observed_fraction_24h']:.2%}\n"
        f"- 模型身份:`{result['model_sha256']}`\n\n"
        "## 如何阅读\n\n"
        f"{result['interpretation']}\n\n"
        "起点价格取该整点前最后一根分钟收盘价。每个小时桶均已完成;"
        "报告不是当前分钟的新预测。跨币种/时期有效性未获保证。\n\n"
        "保持率、下跌及终点收益在此模型中未提供,不应被当作零风险。"
        "已有实验未支持可靠的持续上涨识别;本报告不是买卖指令或扣费后收益预测。\n\n"
        f"训练来源:`{metadata['provenance'].get('kind', 'unknown')}`;"
        "new_fit_not_evaluated 表示新训练尚未评价。历史回放不代表实时行情。\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with _artifacts.publication_staging_directory(
        output.parent, prefix="report-"
    ) as staging:
        source.sort("open_time").tail(1501).write_parquet(staging / "source.parquet")
        result["source_snapshot_sha256"] = _artifacts.file_identity(
            staging / "source.parquet"
        )[0]
        (staging / "prediction.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        (staging / "report.md").write_text(text, encoding="utf-8")
        staging.rename(output)
    return result
