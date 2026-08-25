from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .aggressive_decision_agent import AggressiveDecisionAgent
from .conservative_decision_agent import ConservativeDecisionAgent
from .judge_decision_agent import JudgeDecisionAgent, format_judge_block, market_sentiment_from_vix, weighted_core_score
from .market_data import YahooMarketData
from .deepseek_helpers import make_client, parse_json_text, response_text


PORTFOLIO_SCHEMA = {
    "type": "object",
    "properties": {
        "conclusion": {"type": "string"},
        "confidence_pct": {"type": "integer", "minimum": 60, "maximum": 90},
        "risks": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "scores": {
            "type": "object",
            "properties": {
                "aggressiveness": {"type": "number", "minimum": 0, "maximum": 10},
                "defensiveness": {"type": "number", "minimum": 0, "maximum": 10},
                "concentration_risk": {"type": "number", "minimum": 0, "maximum": 10},
                "liquidity": {"type": "number", "minimum": 0, "maximum": 10},
                "inflation_resistance": {"type": "number", "minimum": 0, "maximum": 10},
                "long_term_compounding": {"type": "number", "minimum": 0, "maximum": 10},
            },
            "required": [
                "aggressiveness", "defensiveness", "concentration_risk", "liquidity",
                "inflation_resistance", "long_term_compounding",
            ],
            "additionalProperties": False,
        },
        "plans": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
    },
    "required": ["conclusion", "confidence_pct", "risks", "scores", "plans"],
    "additionalProperties": False,
}


class PortfolioStore:
    """读取并安全更新用户维护的 portfolio.yaml。"""

    def __init__(self, path: Path, trade_log_path: Path):
        self.path = path
        self.trade_log_path = trade_log_path

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"base_currency": "USD", "cash": 0.0, "holdings": [], "other_assets": []}
        data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        data.setdefault("base_currency", "USD")
        data.setdefault("cash", 0.0)
        data.setdefault("holdings", [])
        data.setdefault("other_assets", [])
        return data

    def held_tickers(self) -> list[str]:
        return sorted({str(row.get("symbol", "")).upper() for row in self.load()["holdings"] if row.get("symbol")})

    def apply_trades(self, trades: list[dict[str, Any]]) -> dict[str, Any]:
        portfolio = copy.deepcopy(self.load())
        holdings = [dict(row) for row in portfolio["holdings"]]
        cash = float(portfolio.get("cash") or 0)

        # 先在内存中校验整批交易，全部合法后才写文件，避免只更新一半。
        for trade in trades:
            symbol = str(trade["symbol"]).upper()
            action = trade["action"]
            quantity = float(trade["quantity"])
            price = float(trade["price"])
            fees = float(trade.get("fees") or 0)
            if quantity <= 0 or price <= 0 or fees < 0:
                raise ValueError(f"{symbol} 的数量、价格或手续费不合法")
            matches = [idx for idx, row in enumerate(holdings) if str(row.get("symbol", "")).upper() == symbol]
            if len(matches) > 1:
                asset_types = ", ".join(
                    f"{row.get('asset_type', 'unknown')}:{row.get('name') or symbol}"
                    for row in holdings
                    if str(row.get("symbol", "")).upper() == symbol
                )
                raise ValueError(f"{symbol} 有多个持仓行（{asset_types}），请说明交易的是股票还是期权")
            row = holdings[matches[0]] if matches else {"symbol": symbol, "asset_type": "stock", "quantity": 0.0, "average_cost": 0.0}
            old_quantity = float(row.get("quantity") or 0)
            old_cost = float(row.get("average_cost") or 0)
            if action == "buy":
                new_quantity = old_quantity + quantity
                row["average_cost"] = (old_quantity * old_cost + quantity * price + fees) / new_quantity
                row["quantity"] = new_quantity
                cash -= quantity * price + fees
                if matches:
                    holdings[matches[0]] = row
                else:
                    holdings.append(row)
            elif action == "sell":
                if not matches:
                    raise ValueError(f"卖出 {symbol}，但配置中没有这个持仓")
                if quantity > old_quantity:
                    raise ValueError(f"卖出 {symbol} {quantity:g} 股，但配置中仅持有 {old_quantity:g} 股")
                row["quantity"] = old_quantity - quantity
                cash += quantity * price - fees
                if row["quantity"] > 0:
                    holdings[matches[0]] = row
                else:
                    holdings.pop(matches[0])
            else:
                raise ValueError(f"不支持的交易方向：{action}")

        portfolio["cash"] = round(cash, 4)
        portfolio["holdings"] = sorted(
            holdings,
            key=lambda row: (
                str(row["symbol"]).upper(),
                str(row.get("asset_type", "")),
                str(row.get("name", "")),
            ),
        )
        self._save(portfolio)
        self._append_trade_log(trades)
        return portfolio

    def position_quantity(self, symbol: str) -> float:
        matches = [
            row for row in self.load().get("holdings", [])
            if str(row.get("symbol", "")).upper() == symbol.upper()
        ]
        if not matches:
            raise ValueError(f"{symbol.upper()} 当前不在 portfolio.yaml 持仓里")
        if len(matches) > 1:
            asset_types = ", ".join(f"{row.get('asset_type', 'unknown')}:{row.get('name') or symbol}" for row in matches)
            raise ValueError(f"{symbol.upper()} 有多个持仓行（{asset_types}），请说明交易的是股票还是期权")
        return float(matches[0].get("quantity") or 0)

    def write_text_snapshot(self, path: Path) -> None:
        portfolio = self.load()
        lines = [
            "# This file is auto-synced from portfolio.yaml after a recorded trade.",
            "# You can still edit it manually; the program will import it again after the content changes.",
            f"base_currency: {portfolio.get('base_currency', 'USD')}",
            f"cash: {float(portfolio.get('cash') or 0):.4f}",
            "",
            "holdings:",
        ]
        for row in portfolio.get("holdings", []):
            name = row.get("name") or row.get("symbol", "")
            lines.append(
                "- "
                f"{row.get('symbol')} | {name} | {row.get('asset_type', 'stock')} | "
                f"quantity={float(row.get('quantity') or 0):g} | "
                f"average_cost={float(row.get('average_cost') or 0):g}"
            )
        lines.append("")
        lines.append("other_assets:")
        for row in portfolio.get("other_assets", []):
            lines.append(
                "- "
                f"{row.get('name')} | {row.get('asset_type', 'other')} | "
                f"value={float(row.get('value') or 0):g} | liquidity={row.get('liquidity', 'unknown')}"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def replace(self, portfolio: dict[str, Any], source_text: str) -> dict[str, Any]:
        """用用户明确提供的完整快照替换配置，并保留一份变更记录。"""
        normalized = {
            "base_currency": str(portfolio.get("base_currency") or "USD").upper(),
            "cash": round(float(portfolio.get("cash") or 0), 4),
            "holdings": [],
            "other_assets": portfolio.get("other_assets") or [],
        }
        for holding in portfolio.get("holdings") or []:
            quantity = float(holding.get("quantity") or 0)
            average_cost = float(holding.get("average_cost") or 0)
            manual_value = float(holding.get("market_value") or 0)
            if (quantity <= 0 and manual_value <= 0) or average_cost < 0:
                raise ValueError(f"{holding.get('symbol', '未知标的')} 至少需要有效数量或当前市值")
            normalized["holdings"].append(
                {
                    "symbol": str(holding["symbol"]).upper(),
                    "name": str(holding.get("name") or ""),
                    "asset_type": str(holding.get("asset_type") or "equity"),
                    "quantity": quantity,
                    "average_cost": average_cost,
                    "manual_market_value": manual_value,
                }
            )
        normalized["holdings"].sort(key=lambda row: row["symbol"])
        self._save(normalized)
        self._append_event({"event": "portfolio_snapshot", "source_text": source_text, "portfolio": normalized})
        return normalized

    def _save(self, portfolio: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(yaml.safe_dump(portfolio, allow_unicode=True, sort_keys=False), encoding="utf-8")
        temporary.replace(self.path)

    def _append_trade_log(self, trades: list[dict[str, Any]]) -> None:
        self._append_event({"event": "trades", "trades": trades})

    def _append_event(self, record: dict[str, Any]) -> None:
        self.trade_log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), **record}
        with self.trade_log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


class PortfolioAdvisor:
    def __init__(self, store: PortfolioStore, api_key: str, model: str):
        self.store = store
        self.market_data = YahooMarketData()
        self.api_key = api_key
        self.client = make_client(api_key)
        self.model = model

    def analyze(self) -> str:
        snapshot = self._valuation_snapshot()
        if snapshot["total_value"] <= 0:
            return "资产配置尚为空。请先编辑 portfolio.yaml，填写现金、持仓或其他资产。"
        if not self.client:
            return "资产配置分析生成失败：未配置 DEEPSEEK_API_KEY 或 DeepSeek 客户端不可用。"

        try:
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "你是私人高级美股分析师。基于用户真实资产配置给出决策导向但非交易指令的组合诊断。"
                            "先总结组合特征和置信度，再识别风险；现金流质量优先于会计利润，并对每个集中持仓做空头检查。"
                            "集中度风险分越高代表风险越高，其他分数越高代表该能力越强。"
                            "改进方向交给后续激进派/保守派/裁判总结输出；本段只负责组合诊断，不重复写方案。"
                            "不要假设未提供的负债、收入、年龄或风险承受能力；缺少这些信息要列为不确定性。"
                        ),
                    },
                    {"role": "user", "content": json.dumps(snapshot, ensure_ascii=False)},
                ],
                text={"format": {"type": "json_schema", "name": "portfolio_review", "schema": PORTFOLIO_SCHEMA, "strict": True}},
            )
            data = parse_json_text(response_text(response))
            return self._format_report(data, snapshot)
        except Exception as exc:
            return f"资产配置分析生成失败：{type(exc).__name__}: {exc}"

    def _valuation_snapshot(self) -> dict[str, Any]:
        portfolio = self.store.load()
        valued_holdings = []
        for holding in portfolio["holdings"]:
            row = dict(holding)
            quantity = float(row.get("quantity") or 0)
            manual_value = float(row.get("manual_market_value") or 0)
            if quantity <= 0 and manual_value > 0:
                price = 0.0
                market_value = manual_value
                price_source = "user_provided_market_value"
            else:
                price_source = "latest_close"
                try:
                    history = self.market_data.get_history(str(row["symbol"]), period="5d", interval="1d")
                    price = float(history.iloc[-1]["close"])
                except Exception:
                    price = float(row.get("average_cost") or 0)
                    price_source = "average_cost_fallback"
                market_value = quantity * price
            row["market_price"] = round(price, 4)
            row["market_value"] = round(market_value, 4)
            row["price_source"] = price_source
            valued_holdings.append(row)

        cash = float(portfolio.get("cash") or 0)
        other_assets = [dict(row) for row in portfolio["other_assets"]]
        total = cash + sum(row["market_value"] for row in valued_holdings) + sum(float(row.get("value") or 0) for row in other_assets)
        for row in valued_holdings:
            row["weight_pct"] = round(row["market_value"] / total * 100, 2) if total else 0
        for row in other_assets:
            row["weight_pct"] = round(float(row.get("value") or 0) / total * 100, 2) if total else 0
        return {
            "base_currency": portfolio["base_currency"],
            "total_value": round(total, 2),
            "cash": cash,
            "cash_weight_pct": round(cash / total * 100, 2) if total else 0,
            "holdings": valued_holdings,
            "other_assets": other_assets,
            "data_limits": "未提供收入、负债、期限和风险承受能力；latest_close不是盘中实时成交价。",
        }

    def _format_report(self, data: dict[str, Any], snapshot: dict[str, Any]) -> str:
        scores = data["scores"]
        decision_block = self._portfolio_decision_debate(data, snapshot)
        lines = [
            "# 资产配置诊断",
            "",
            f"**结论：** {data['conclusion']}（置信度：{data['confidence_pct']}%）",
            f"**组合估值：** {snapshot['total_value']:,.2f} {snapshot['base_currency']}；现金占比 {snapshot['cash_weight_pct']}%",
            "",
            "## 风险",
            *[f"- {risk}" for risk in data["risks"]],
            "",
            "## 配置评分",
            f"- **攻击性：{scores['aggressiveness']:g}/10**",
            f"- **防守性：{scores['defensiveness']:g}/10**",
            f"- **集中度风险：{scores['concentration_risk']:g}/10**",
            f"- **流动性：{scores['liquidity']:g}/10**",
            f"- **抗通胀能力：{scores['inflation_resistance']:g}/10**",
            f"- **长期复利质量：{scores['long_term_compounding']:g}/10**",
            "",
            "## 激进派 / 保守派 / 裁判总结",
            decision_block,
            "",
            "这不构成投资建议。",
        ]
        return "\n".join(lines)

    def _portfolio_decision_debate(self, data: dict[str, Any], snapshot: dict[str, Any]) -> str:
        weighted_score = weighted_core_score(
            fundamental_score={
                "total_score": round(
                    (
                        float(data["scores"].get("long_term_compounding", 0))
                        + float(data["scores"].get("inflation_resistance", 0))
                    )
                    / 2
                    * 10
                ),
                "direction": "组合基本面质量代理分",
            },
            technical_score={"total_score": 50, "direction": "资产配置分析不含技术面"},
            news_score={"total_score": 50, "direction": "资产配置分析不含当日消息面"},
            market_sentiment_score=market_sentiment_from_vix(self._vix_indicator()),
        )
        context = {
            "scope": "portfolio_allocation",
            "weights": "基本面40%，技术面30%，消息面20%，市场情绪10%",
            "weighted_score": weighted_score,
            "portfolio_snapshot": snapshot,
            "portfolio_review": data,
        }
        try:
            aggressive = AggressiveDecisionAgent(self.api_key).analyze(context)
            conservative = ConservativeDecisionAgent(self.api_key).analyze(context)
            judge = JudgeDecisionAgent(self.api_key).judge(context, aggressive, conservative)
            return format_judge_block(judge)
        except Exception as exc:
            return f"三方决策总结：生成失败（{type(exc).__name__}: {exc}）。这不构成投资建议。"

    def _vix_indicator(self) -> dict[str, Any]:
        try:
            frame = self.market_data.get_history("^VIX", period="5d", interval="1d")
            latest = frame.iloc[-1]
            previous = frame.iloc[-2] if len(frame) >= 2 else latest
            last = float(latest["close"])
            previous_close = float(previous["close"])
            return {
                "last": round(last, 4),
                "previous_close": round(previous_close, 4),
                "change_pct": round(last / previous_close - 1, 6) if previous_close else None,
            }
        except Exception as exc:
            return {"available": False, "reason": str(exc)}
