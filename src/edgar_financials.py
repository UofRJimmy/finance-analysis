from __future__ import annotations

import json
import os
from dataclasses import dataclass
from statistics import median
from typing import Any

from .config import Settings
from .news_sources.finnhub_client import FinnhubClient
from .deepseek_helpers import make_client, parse_json_text, response_text


CONCEPTS = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
    "gross_profit": ["GrossProfit"],
    "net_income": ["NetIncomeLoss"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "receivables": ["AccountsReceivableNetCurrent", "AccountsReceivableNet"],
    "inventory": ["InventoryNet"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "restructuring": ["RestructuringCharges"],
    "impairment": ["AssetImpairmentCharges"],
    "asset_sale_gain": ["GainLossOnSaleOfPropertyPlantEquipment", "GainsLossesOnSalesOfAssets"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "debt_current": ["LongTermDebtCurrent", "ShortTermBorrowings"],
    "debt_noncurrent": ["LongTermDebtNoncurrent", "LongTermDebtAndFinanceLeaseObligationsNoncurrent"],
    "stockholders_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "diluted_shares": ["WeightedAverageNumberOfDilutedSharesOutstanding", "CommonStockSharesOutstanding"],
}


REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "macro_cycle": {"type": "string"},
        "industry_lifecycle": {"type": "string"},
        "rate_impact": {"type": "string"},
        "income_comment": {"type": "string"},
        "balance_sheet_comment": {"type": "string"},
        "cash_flow_comment": {"type": "string"},
        "brand_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "switching_cost_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "network_effect_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "cost_advantage_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "patent_barrier_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "moat_erosion": {"type": "string"},
        "rating": {"type": "string", "enum": ["强烈买入", "买入", "持有", "卖出", "强烈卖出"]},
        "confidence_pct": {"type": "integer", "minimum": 60, "maximum": 90},
        "risks": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 3},
        "leading_indicators": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
    },
    "required": [
        "macro_cycle",
        "industry_lifecycle",
        "rate_impact",
        "income_comment",
        "balance_sheet_comment",
        "cash_flow_comment",
        "brand_score",
        "switching_cost_score",
        "network_effect_score",
        "cost_advantage_score",
        "patent_barrier_score",
        "moat_erosion",
        "rating",
        "confidence_pct",
        "risks",
        "leading_indicators",
    ],
    "additionalProperties": False,
}


@dataclass
class EdgarFinancialAnalyzer:
    settings: Settings

    def __post_init__(self) -> None:
        self.client = make_client(self.settings.deepseek_api_key)
        self.finnhub = FinnhubClient(self.settings.finnhub_api_key)

    def analyze(self, ticker: str, user_question: str) -> str:
        if not self.settings.edgar_identity:
            return "无法分析财报：请先在 .env 配置 EDGAR_IDENTITY=姓名 邮箱。"
        if not self.client:
            return "无法分析财报：未配置 DeepSeek API。"

        company_name, source, raw_series = self._load_sec_facts(ticker)
        computed = _compute_financial_metrics(raw_series, self._current_price(ticker), self.settings.dcf_discount_rate)
        macro_research = self._macro_and_industry_research(ticker, company_name)
        narrative = self._build_narrative(ticker, company_name, user_question, source, computed, macro_research)
        return _render_report(ticker, company_name, source, computed, narrative)

    def fundamentals(self, ticker: str) -> dict[str, Any]:
        """Return compact SEC fundamentals in a Finnhub-like shape for scoring."""
        if not self.settings.edgar_identity:
            raise RuntimeError("请先在 .env 配置 EDGAR_IDENTITY=姓名 邮箱，SEC EDGAR 要求声明访问身份。")
        company_name, source, raw_series = self._load_sec_facts(ticker)
        computed = _compute_financial_metrics(raw_series, self._current_price(ticker), self.settings.dcf_discount_rate)
        latest = (computed.get("annual_rows") or [{}])[0]
        shares = _latest(raw_series.get("diluted_shares", {}))
        debt = (_latest(raw_series.get("debt_current", {})) or 0.0) + (
            _latest(raw_series.get("debt_noncurrent", {})) or 0.0
        )
        equity = _latest(raw_series.get("stockholders_equity", {}))
        metric = {
            "freeCashFlowPerShareTTM": _per_share(latest.get("free_cash_flow"), shares),
            "cashFlowPerShareTTM": _per_share(latest.get("operating_cash_flow"), shares),
            "netIncomePerShareTTM": _per_share(latest.get("net_income"), shares),
            "currentRatioAnnual": _ratio(latest.get("current_assets"), latest.get("current_liabilities")),
            "totalDebt/totalEquityAnnual": _ratio(debt, equity),
        }
        return {
            "metric": metric,
            "source": {
                **source,
                "company_name": company_name,
                "fundamental_source": "SEC EDGAR XBRL fundamentals",
                "latest_fiscal_year": latest.get("year"),
            },
            "computed_financials": computed,
        }

    def _load_sec_facts(self, ticker: str) -> tuple[str, dict, dict]:
        data_dir = self.settings.edgar_data_dir
        cache_dir = data_dir.parent / "edgar_cache"
        data_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["EDGAR_LOCAL_DATA_DIR"] = str(data_dir)
        os.environ["EDGAR_CACHE_DIR"] = str(cache_dir)

        # Import after setting paths; edgartools creates its cache during import.
        from edgar import Company, set_identity

        set_identity(self.settings.edgar_identity)
        company = Company(ticker)
        facts = company.get_facts()
        if facts is None:
            raise RuntimeError(f"SEC 未返回 {ticker} 的 XBRL facts")

        series = {name: _first_available_series(facts, concepts) for name, concepts in CONCEPTS.items()}
        source = {"provider": "SEC EDGAR XBRL", "ticker": ticker}
        try:
            latest = company.get_filings(form=["10-K", "10-Q"], amendments=False).latest()
            source.update(
                {
                    "form": str(getattr(latest, "form", "")),
                    "filing_date": str(getattr(latest, "filing_date", "")),
                    "accession_number": str(getattr(latest, "accession_no", "")),
                }
            )
        except Exception:
            pass
        return str(getattr(company, "name", ticker)), source, series

    def _current_price(self, ticker: str) -> float | None:
        try:
            return self.finnhub.get_quote(ticker).current if self.finnhub.enabled else None
        except Exception:
            return None

    def _macro_and_industry_research(self, ticker: str, company_name: str) -> str:
        return (
            "未启用外部网页检索。宏观和行业定位仅依据本次 EDGAR 财务数据与模型的通用金融框架，"
            "不将其表述为实时市场事实。"
        )

    def _build_narrative(
        self,
        ticker: str,
        company_name: str,
        question: str,
        source: dict,
        computed: dict,
        macro_research: str,
    ) -> dict:
        prompt = {
            "question": question,
            "ticker": ticker,
            "company": company_name,
            "sec_source": source,
            "computed_financials": computed,
            "macro_and_industry_research": macro_research,
        }
        response = self.client.responses.create(
            model=self.settings.deepseek_model_analyze,
            input=[
                {
                    "role": "system",
                    "content": (
                        "你是高级美股财报分析师。严格基于提供的 SEC 计算结果和宏观研究输出。"
                        "现金流优先于利润；没有数据就明确说数据不足。"
                        "五档评级是 research 标签，不是直接交易指令。"
                        "若 DCF 安全边际低于30%，必须明确标记高估、建议等待，评级不得给买入或强烈买入。"
                        "每项风险一句话，专业术语首次出现时用大白话解释。"
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "edgar_financial_report",
                    "schema": REPORT_SCHEMA,
                    "strict": True,
                }
            },
        )
        return parse_json_text(response_text(response))


def _first_available_series(facts, concepts: list[str]) -> dict[int, float]:
    for concept in concepts:
        try:
            frame = facts.time_series(concept, periods=80)
        except Exception:
            continue
        if frame is None or frame.empty:
            continue
        result: dict[int, float] = {}
        annual = frame[frame["fiscal_period"] == "FY"]
        for _, row in annual.iterrows():
            year = row.get("fiscal_year")
            value = row.get("numeric_value")
            if year is None or value is None or year != year or value != value:
                continue
            result.setdefault(int(year), float(value))
        if result:
            return dict(sorted(result.items(), reverse=True)[:5])
    return {}


def _compute_financial_metrics(series: dict[str, dict[int, float]], current_price: float | None, discount_rate: float) -> dict:
    years = sorted({year for values in series.values() for year in values}, reverse=True)[:5]
    rows = []
    for year in years:
        revenue = _at(series, "revenue", year)
        gross_profit = _at(series, "gross_profit", year)
        net_income = _at(series, "net_income", year)
        restructuring = _at(series, "restructuring", year) or 0.0
        impairment = _at(series, "impairment", year) or 0.0
        gains = _at(series, "asset_sale_gain", year) or 0.0
        cfo = _at(series, "operating_cash_flow", year)
        capex = _at(series, "capex", year)
        current_assets = _at(series, "current_assets", year)
        current_liabilities = _at(series, "current_liabilities", year)
        fcf = cfo - abs(capex) if cfo is not None and capex is not None else None
        adjusted_net_income = net_income + abs(restructuring) + abs(impairment) - gains if net_income is not None else None
        rows.append(
            {
                "year": year,
                "revenue": revenue,
                "gross_profit": gross_profit,
                "gross_margin": _ratio(gross_profit, revenue),
                "net_income": net_income,
                "adjusted_net_income_proxy": adjusted_net_income,
                "current_assets": current_assets,
                "current_liabilities": current_liabilities,
                "net_working_capital": _subtract(current_assets, current_liabilities),
                "receivables": _at(series, "receivables", year),
                "inventory": _at(series, "inventory", year),
                "operating_cash_flow": cfo,
                "capex": capex,
                "free_cash_flow": fcf,
                "fcf_to_net_income": _ratio(fcf, net_income),
            }
        )

    for index, row in enumerate(rows):
        older = rows[index + 1] if index + 1 < len(rows) else None
        row["revenue_growth"] = _growth(row["revenue"], older["revenue"] if older else None)
        row["adjusted_net_income_growth"] = _growth(
            row["adjusted_net_income_proxy"], older["adjusted_net_income_proxy"] if older else None
        )
        row["net_working_capital_change"] = _subtract(
            row["net_working_capital"], older["net_working_capital"] if older else None
        )
        row["receivables_growth"] = _growth(row["receivables"], older["receivables"] if older else None)
        row["inventory_growth"] = _growth(row["inventory"], older["inventory"] if older else None)
        row["receivables_alert"] = _faster_than_revenue(row["receivables_growth"], row["revenue_growth"])
        row["inventory_alert"] = _faster_than_revenue(row["inventory_growth"], row["revenue_growth"])

    margins = [row["gross_margin"] for row in reversed(rows) if row["gross_margin"] is not None]
    margin_trend = "数据不足"
    if len(margins) >= 2:
        change = margins[-1] - margins[0]
        margin_trend = "走阔" if change > 0.01 else "收窄" if change < -0.01 else "基本稳定"

    latest_three_ratios = [row["fcf_to_net_income"] for row in rows[:3]]
    fcf_quality_warning = len(latest_three_ratios) == 3 and all(ratio is not None and ratio < 0.8 for ratio in latest_three_ratios)
    dcf = _dcf(series, rows, current_price, discount_rate)
    return {
        "annual_rows": rows,
        "gross_margin_five_year_trend": margin_trend,
        "fcf_to_net_income_three_year_warning": fcf_quality_warning,
        "dcf": dcf,
        "calculation_notes": [
            "扣非净利润为 SEC GAAP 净利润加回重组/减值、扣除资产出售收益的税前代理值，不等同公司披露的 non-GAAP 指标。",
            "自由现金流=经营现金流-资本开支。",
            "DCF 固定预测5年、永续增长率2.5%，增长率和折现率做上下浮动形成区间。",
        ],
    }


def _dcf(series: dict[str, dict[int, float]], rows: list[dict], current_price: float | None, discount_rate: float) -> dict:
    latest_fcf = next((row["free_cash_flow"] for row in rows if row["free_cash_flow"] is not None), None)
    if latest_fcf is None or latest_fcf <= 0:
        return {"available": False, "reason": "最新自由现金流缺失或非正数", "current_price": current_price}
    growth_values = [row["revenue_growth"] for row in rows[:4] if row.get("revenue_growth") is not None]
    growth = max(-0.05, min(0.15, median(growth_values) if growth_values else 0.05))
    next_year_fcf = latest_fcf * (1 + growth)
    cash = _latest(series.get("cash", {})) or 0.0
    debt = (_latest(series.get("debt_current", {})) or 0.0) + (_latest(series.get("debt_noncurrent", {})) or 0.0)
    shares = _latest(series.get("diluted_shares", {}))
    low = _dcf_value(next_year_fcf, growth - 0.02, discount_rate + 0.01, cash, debt, shares)
    high = _dcf_value(next_year_fcf, growth + 0.02, max(0.06, discount_rate - 0.01), cash, debt, shares)
    values = sorted([value for value in [low, high] if value is not None])
    if len(values) != 2:
        return {"available": False, "reason": "缺少稀释股数或 DCF 参数无效", "current_price": current_price}
    midpoint = sum(values) / 2
    safety_margin = ((midpoint - current_price) / midpoint * 100) if current_price is not None and midpoint else None
    return {
        "available": True,
        "next_year_fcf_estimate": next_year_fcf,
        "growth_rate": growth,
        "discount_rate": discount_rate,
        "intrinsic_value_per_share_range": values,
        "current_price": current_price,
        "safety_margin_pct": safety_margin,
        "valuation_flag": "高估，建议等待" if safety_margin is not None and safety_margin < 30 else "安全边际达到30%门槛",
    }


def _dcf_value(next_fcf: float, growth: float, discount: float, cash: float, debt: float, shares: float | None) -> float | None:
    terminal_growth = 0.025
    if shares is None or shares <= 0 or discount <= terminal_growth:
        return None
    projected = []
    fcf = next_fcf
    for _ in range(5):
        projected.append(fcf)
        fcf *= 1 + growth
    pv = sum(value / ((1 + discount) ** (index + 1)) for index, value in enumerate(projected))
    terminal = projected[-1] * (1 + terminal_growth) / (discount - terminal_growth)
    equity_value = pv + terminal / ((1 + discount) ** 5) + cash - debt
    return equity_value / shares


def _render_report(ticker: str, company_name: str, source: dict, computed: dict, narrative: dict) -> str:
    rows = computed.get("annual_rows") or []
    latest = rows[0] if rows else {}
    moat_scores = [
        narrative["brand_score"],
        narrative["switching_cost_score"],
        narrative["network_effect_score"],
        narrative["cost_advantage_score"],
        narrative["patent_barrier_score"],
    ]
    moat_average = sum(moat_scores) / len(moat_scores)
    moat_label = "宽护城河" if moat_average >= 4 else "一般/窄护城河"
    dcf = computed.get("dcf") or {}
    if dcf.get("available"):
        value_range = dcf["intrinsic_value_per_share_range"]
        dcf_text = (
            f"明年预估自由现金流 {_fmt(dcf.get('next_year_fcf_estimate'))}；"
            f"增长率 {dcf.get('growth_rate', 0):.1%}；折现率 {dcf.get('discount_rate', 0):.1%}；"
            f"每股内在价值区间 {value_range[0]:.2f}-{value_range[1]:.2f}；"
            f"当前价 {_fmt_price(dcf.get('current_price'))}；安全边际 {_fmt_pct(dcf.get('safety_margin_pct'), already_percent=True)}；"
            f"标记：{dcf.get('valuation_flag')}。"
        )
    else:
        dcf_text = f"无法完成 DCF：{dcf.get('reason', '数据不足')}。"
    risks = "\n".join(f"- {risk}" for risk in narrative["risks"])
    indicators = "\n".join(f"- {item}" for item in narrative["leading_indicators"])
    source_text = f"{source.get('provider')} {source.get('form', '')} {source.get('filing_date', '')}".strip()
    return f"""# {company_name} ({ticker}) 财报分析

数据来源：{source_text}

## Step 1: 宏观定位（3句话内）
1. {narrative['macro_cycle']}
2. {narrative['industry_lifecycle']}
3. {narrative['rate_impact']}

## Step 2: 财报三维度穿透
- 利润表：{narrative['income_comment']} 最新扣非净利润代理值：{_fmt(latest.get('adjusted_net_income_proxy'))}；同比增速：{_fmt_pct(latest.get('adjusted_net_income_growth'))}；毛利率5年趋势：{computed.get('gross_margin_five_year_trend')}。
- 资产负债表：{narrative['balance_sheet_comment']} 最新净营运资本：{_fmt(latest.get('net_working_capital'))}；同比变化：{_fmt(latest.get('net_working_capital_change'))}；应收警报：{latest.get('receivables_alert')}；存货警报：{latest.get('inventory_alert')}。
- 现金流量表：{narrative['cash_flow_comment']} 最新 FCF/净利润：{_fmt_pct(latest.get('fcf_to_net_income'))}；连续3年低于80%风险：{computed.get('fcf_to_net_income_three_year_warning')}。

## Step 3: 护城河量化评估
- 品牌溢价：{narrative['brand_score']}/5
- 转换成本：{narrative['switching_cost_score']}/5
- 网络效应：{narrative['network_effect_score']}/5
- 成本优势：{narrative['cost_advantage_score']}/5
- 专利壁垒：{narrative['patent_barrier_score']}/5
- 综合：{moat_average:.1f}/5，{moat_label}。未来3年侵蚀判断：{narrative['moat_erosion']}

## Step 4: 估值与安全边际
- 简化 DCF：{dcf_text}

## Step 5: 决策结论
- 最终评级（research标签）：{narrative['rating']}
- 置信度：{narrative['confidence_pct']}%
- 最关键的三个风险点：
{risks}
- 需要跟踪的先行指标：
{indicators}

这不构成投资建议，评级是研究标签而非直接交易指令。
"""


def _at(series: dict[str, dict[int, float]], name: str, year: int) -> float | None:
    return series.get(name, {}).get(year)


def _latest(values: dict[int, float]) -> float | None:
    return values[max(values)] if values else None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def _per_share(value: float | None, shares: float | None) -> float | None:
    if value is None or shares in (None, 0):
        return None
    return value / shares


def _subtract(left: float | None, right: float | None) -> float | None:
    return left - right if left is not None and right is not None else None


def _growth(current: float | None, previous: float | None) -> float | None:
    if current is None or previous in (None, 0):
        return None
    return current / previous - 1


def _faster_than_revenue(metric_growth: float | None, revenue_growth: float | None) -> bool | None:
    if metric_growth is None or revenue_growth is None:
        return None
    return metric_growth > revenue_growth + 0.05


def _fmt(value: float | None) -> str:
    if value is None:
        return "数据不足"
    return f"{value:,.0f}"


def _fmt_pct(value: float | None, already_percent: bool = False) -> str:
    if value is None:
        return "数据不足"
    return f"{value:.1f}%" if already_percent else f"{value:.1%}"


def _fmt_price(value: float | None) -> str:
    return "数据不足" if value is None else f"{value:.2f}"
