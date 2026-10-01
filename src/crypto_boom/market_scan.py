"""Full observed-universe scan with one model/time snapshot and explicit coverage."""

from __future__ import annotations

import asyncio
import csv
import json
import time
import tomllib
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import aiohttp

from crypto_boom import _artifacts
from crypto_boom.bars import MINUTE_US
from crypto_boom.market_data import ScanStopped, SpotSnapshotClient
from crypto_boom.model_runtime import load_active_model, predict_bars


@dataclass(frozen=True)
class ScanSettings:
    data_dir: Path = Path("data")
    workers: int = 4
    timeout_seconds: int = 900

    def __post_init__(self) -> None:
        if type(self.workers) is not int or not 1 <= self.workers <= 4:
            raise ValueError("workers must be 1..4")
        if (
            type(self.timeout_seconds) is not int
            or not 1 <= self.timeout_seconds <= 900
        ):
            raise ValueError("timeout_seconds must be 1..900")


def scan_settings(path: Path | None = None) -> ScanSettings:
    if path is not None and not path.is_file():
        raise ValueError("explicit scan configuration does not exist")
    path = path or Path("scan.toml")
    if not path.exists():
        return ScanSettings()
    if path.stat().st_size > 64_000:
        raise ValueError("scan configuration exceeds size limit")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    if set(data) != {"scan"} or set(data["scan"]) - {
        "data_dir",
        "workers",
        "timeout_seconds",
    }:
        raise ValueError("unknown scan configuration keys")
    values = data["scan"]
    directory = Path(values.get("data_dir", "data"))
    if not directory.is_absolute():
        directory = path.resolve().parent / directory
    return ScanSettings(
        directory, values.get("workers", 4), values.get("timeout_seconds", 900)
    )


async def scan_market(settings: ScanSettings) -> dict:
    """No caller symbol list or model path: resolve activation, enumerate full scope."""
    models, manifest, recipe = load_active_model(settings.data_dir / "models")
    identity = manifest["identity"]
    step_us = identity["decision_step_minutes"] * MINUTE_US
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    output = settings.data_dir / "reports" / run_id
    results: dict[str, dict] = {}
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=12)
    ) as session:
        client = SpotSnapshotClient(session, seconds=settings.timeout_seconds)
        symbols, scope, raw, server_ms = await client.universe()
        decision = server_ms * 1000 // step_us * step_us
        refused = set(scope["metadata_refusals"])
        pending = iter(symbols)
        stop_reason = None

        async def worker() -> None:
            nonlocal stop_reason
            for symbol in pending:
                if symbol in refused:
                    results[symbol] = {
                        "symbol": symbol,
                        "state": "metadata_error",
                        "reason": "unsupported instrument metadata",
                    }
                    continue
                if client.stopped or time.monotonic() >= client.deadline:
                    results[symbol] = {
                        "symbol": symbol,
                        "state": "not_attempted",
                        "reason": stop_reason or "run budget exhausted",
                    }
                    continue
                try:
                    bars, receipt = await client.bars(
                        symbol,
                        decision_us=decision,
                        history_minutes=recipe.history_minutes,
                        cache=settings.data_dir / "snapshots",
                    )
                    values = predict_bars(models, manifest, bars, decision_us=decision)
                    results[symbol] = {
                        "symbol": symbol,
                        "state": "success",
                        "values": values,
                        **receipt,
                    }
                except ScanStopped as exc:
                    client.stopped = True
                    stop_reason = str(exc)
                    results[symbol] = {
                        "symbol": symbol,
                        "state": "not_completed",
                        "reason": str(exc) or type(exc).__name__,
                    }
                except (aiohttp.ClientError, TimeoutError, OSError) as exc:
                    results[symbol] = {
                        "symbol": symbol,
                        "state": "fetch_error",
                        "reason": str(exc) or type(exc).__name__,
                    }
                except (ValueError, TypeError, KeyError, OverflowError) as exc:
                    results[symbol] = {
                        "symbol": symbol,
                        "state": "data_or_prediction_error",
                        "reason": str(exc) or type(exc).__name__,
                    }

        await asyncio.gather(*(worker() for _ in range(settings.workers)))
        elapsed = time.monotonic() - client.started
        requests = client.requests
    if set(results) != set(symbols):
        raise AssertionError("universe coverage ledger incomplete")
    rows = [results[s] for s in symbols]
    counts = dict(Counter(r["state"] for r in rows))
    status = "complete" if counts.get("success", 0) == len(symbols) else "partial"
    rank_by = identity["rank_by"]
    ranking = sorted(
        (r for r in rows if r["state"] == "success"),
        key=lambda r: (-r["values"][rank_by], r["symbol"]),
    )
    result = {
        "schema": "market-scan-v1",
        "run_id": run_id,
        "status": status,
        "scope": scope,
        "model_id": manifest["model_id"],
        "model_contract": identity,
        "decision_us": decision,
        "exchange_snapshot_ms": server_ms,
        "elapsed_seconds": elapsed,
        "http_requests": requests,
        "estimated_decision_age_seconds_at_finish": (server_ms * 1000 - decision) / 1e6
        + elapsed,
        "newer_decision_may_be_available": server_ms * 1000 + elapsed * 1e6
        >= decision + step_us,
        "eligible_symbols": len(symbols),
        "counts": counts,
        "stop_reason": stop_reason,
        "rank_by": rank_by,
        "rows": rows,
        "report_directory": str(output.resolve()),
    }
    when = datetime.fromtimestamp(decision / 1e6, UTC).isoformat()
    description = next(o for o in identity["outputs"] if o["name"] == rank_by)
    lines = [
        "# 全盘预测报告",
        "",
        f"状态: **{status}**",
        f"范围: {scope['scope']}",
        f"共同预测起点 UTC: {when}",
        f"模型 ID: `{manifest['model_id']}`",
        f"模型步长: {identity['decision_step_minutes']} 分钟; 历史: {recipe.history_minutes} 分钟",
        f"范围成员: {len(symbols)}; 成功: {counts.get('success', 0)}; 其余均列入状态台账",
        f"耗时: {elapsed:.1f} 秒; 请求数: {requests}",
        f"完成时可能已有更新起点: {result['newer_decision_may_be_available']}",
        "",
        "## 排序预览(完整结果见 predictions.csv / report.json)",
        f"排序字段: {rank_by}; 目标: {description['label']}; 未来窗口: {description['horizon_minutes']} 分钟; 单位: {description['unit']}",
        "",
        "| 标的 | 数值 |",
        "|---|---:|",
    ]
    for row in ranking[:30]:
        value = row["values"][rank_by]
        formatted = (
            f"{value:.2%}"
            if description["unit"] in ("fraction", "return_fraction")
            else f"{value:.4f}"
        )
        lines.append(f"| {row['symbol']} | {formatted} |")
    lines.extend(
        [
            "",
            "## 边界",
            "全盘指本次观测到的指定交易所/市场/报价币范围,不是所有交易所。",
            "所有成员采用同一完成时间; 不混入扫描途中更新的分钟。跨越新起点会明确标记,不伪称完成时最新。",
            "排序不是买卖建议。分位数不是上涨概率,缺失输出不是零风险。模型仍为实验状态。",
            "当前上市范围可能超出训练人口; 新资产/历史不足、接口失败、限流和未尝试均不当负例。",
            "",
            "## 状态计数",
            json.dumps(counts, ensure_ascii=False),
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with _artifacts.publication_staging_directory(
        output.parent, prefix="scan-"
    ) as staging:
        (staging / "exchange-info.json").write_bytes(raw)
        result["universe_sha256"] = _artifacts.file_identity(
            staging / "exchange-info.json"
        )[0]
        (staging / "report.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        (staging / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        with (staging / "predictions.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["symbol", "state", "reason", *models]
            )
            writer.writeheader()
            for row in [*ranking, *(r for r in rows if r["state"] != "success")]:
                writer.writerow(
                    {
                        "symbol": row["symbol"],
                        "state": row["state"],
                        "reason": row.get("reason", ""),
                        **row.get("values", {}),
                    }
                )
        staging.rename(output)
    return result
