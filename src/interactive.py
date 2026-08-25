from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from .config import Settings
from .dedup_store import DedupStore
from .fundamental_agent import FundamentalAgent
from .market_calendar import is_us_market_session, market_closed_reason
from .market_data import YahooMarketData
from .news_sources.finnhub_client import FinnhubClient
from .deepseek_helpers import make_client, parse_json_text, response_text
from .portfolio import PortfolioAdvisor, PortfolioStore
from .premarket_summary import generate_summary as generate_premarket_summary
from .reporter import Reporter
from .technical_agent import TechnicalAgent
from .tickertick_question_agent import TickerTickQuestionAgent
from .watchlist import Watchlist


COMMON_ALIASES = {
    "苹果": "AAPL",
    "苹果公司": "AAPL",
    "英伟达": "NVDA",
    "微软": "MSFT",
    "特斯拉": "TSLA",
    "亚马逊": "AMZN",
    "谷歌": "GOOGL",
    "alphabet": "GOOGL",
    "meta": "META",
    "脸书": "META",
    "amd": "AMD",
    "超微半导体": "AMD",
    "台积电": "TSM",
    "伯克希尔": "BRK.B",
}


RESEARCH_ROLE = """你是本项目里的高级美股分析师，只服务于用户的个人 watchlist。
你基于项目实时抓取和存储的新闻、历史分析、盘前总结和行情，给出机构 research 风格但通俗易懂的解读。
你的核心价值观：
1. 永远基于概率而非确定性思考；市场判断的置信度必须在 60%-90% 之间。
2. 现金流比利润更重要：利润包含会计判断，现金流更接近企业真实收支。没有现金流数据时必须明确说数据不足，不能编造。
3. 没有永远的好公司；每只股票都要接受空头检查，包括估值、竞争、需求、监管、资产负债表、现金流和执行风险。
4. 回答必须服务于投资决策：指出哪些事实会增强或削弱当前判断，以及后续应观察什么；但禁止给出买入、卖出、加仓、减仓等直接操作指令。
专业名词第一次出现时必须用一句大白话解释。"""


MARKET_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "conclusion": {"type": "string"},
        "confidence_pct": {"type": "integer", "minimum": 60, "maximum": 90},
        "evidence": {"type": "string"},
        "cash_flow_view": {"type": "string"},
        "bear_checklist": {"type": "string"},
        "uncertainty": {"type": "string"},
        "decision_implication": {"type": "string"},
        "background": {"type": "string"},
    },
    "required": [
        "conclusion",
        "confidence_pct",
        "evidence",
        "cash_flow_view",
        "bear_checklist",
        "uncertainty",
        "decision_implication",
        "background",
    ],
    "additionalProperties": False,
}


TRADE_REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "complete": {"type": "boolean"},
        "missing_information": {"type": "string"},
        "trades": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["buy", "sell"]},
                    "position_effect": {"type": "string", "enum": ["normal", "partial", "full"]},
                    "symbol_or_company": {"type": "string"},
                    "quantity": {"type": "number"},
                    "price": {"type": "number"},
                    "fees": {"type": "number"},
                },
                "required": ["action", "position_effect", "symbol_or_company", "quantity", "price", "fees"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["complete", "missing_information", "trades"],
    "additionalProperties": False,
}


PORTFOLIO_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "complete": {"type": "boolean"},
        "missing_information": {"type": "string"},
        "base_currency": {"type": "string"},
        "cash": {"type": "number"},
        "holdings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "symbol_or_company": {"type": "string"},
                    "name": {"type": "string"},
                    "asset_type": {"type": "string"},
                    "quantity": {"type": "number"},
                    "average_cost": {"type": "number"},
                    "market_value": {"type": "number"},
                },
                "required": ["symbol_or_company", "name", "asset_type", "quantity", "average_cost", "market_value"],
                "additionalProperties": False,
            },
        },
        "other_assets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "asset_type": {"type": "string"},
                    "value": {"type": "number"},
                    "liquidity": {"type": "string"},
                },
                "required": ["name", "asset_type", "value", "liquidity"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["complete", "missing_information", "base_currency", "cash", "holdings", "other_assets"],
    "additionalProperties": False,
}


class InteractiveAssistant:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.watchlist = Watchlist(settings.watchlist_file)
        self.finnhub = FinnhubClient(settings.finnhub_api_key)
        self.client = make_client(settings.deepseek_api_key)
        self.model = settings.deepseek_model_analyze
        self.fundamental_agent = FundamentalAgent(settings, self.finnhub)
        self.portfolio_store = PortfolioStore(settings.portfolio_file, settings.trade_log_path)
        self.portfolio_advisor = PortfolioAdvisor(
            self.portfolio_store,
            settings.deepseek_api_key,
            settings.deepseek_model_summary,
        )
        self.reporter = Reporter(
            settings.report_dir,
            settings.display_timezone,
            settings.dingtalk_webhook_url,
            settings.dingtalk_keyword,
            settings.dingtalk_secret,
        )
        self.technical_agent = TechnicalAgent(
            YahooMarketData(),
            settings.deepseek_api_key,
            settings.deepseek_model_analyze,
        )
        self.tickertick_question_agent = TickerTickQuestionAgent(
            settings.deepseek_api_key,
            settings.deepseek_model_analyze,
        )
        self._conversation_history: list[dict[str, str]] = []

    def run(self, stop_event) -> None:
        self._print_menu()
        while not stop_event.is_set():
            try:
                text = input("史上最强金融分析师> ").strip()
            except (EOFError, KeyboardInterrupt):
                stop_event.set()
                break
            if not text:
                continue
            if text.lower() in {"exit", "quit", "退出"}:
                stop_event.set()
                break
            try:
                self.handle(text)
            except Exception as exc:
                print(f"指令处理失败: {type(exc).__name__}: {exc}", flush=True)

    def prepare_portfolio(self) -> None:
        """启动时只导入自由文本，不主动输出分析、历史记录或改写 watchlist。"""
        self._import_portfolio_text(display=False)

    def _print_menu(self) -> None:
        print(
            "\n===== 功能菜单 =====\n"
            "请输入文字指令，不用输入数字：\n"
            "资产配置分析\n"
            "盘前报告\n"
            "盘后报告\n"
            "添加“股票名称”\n"
            "移除“股票名称”\n"
            "也可以直接输入金融问题或交易记录（如分析Nvda）；输入 exit 退出。\n",
            flush=True,
        )

    def _import_portfolio_text(self, display: bool = True) -> None:
        path = self.settings.portfolio_text_file
        if not path.exists():
            return
        text = path.read_text(encoding="utf-8").strip()
        meaningful = "\n".join(line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#"))
        if not meaningful:
            return
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        state_path = self.settings.portfolio_text_state_path
        try:
            state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
        except (OSError, json.JSONDecodeError):
            state = {}
        if state.get("sha256") == digest:
            return

        if display:
            print("检测到新的 portfolio.txt，正在交给 ChatGPT 整理资产配置...", flush=True)
        if self._parse_and_save_portfolio(meaningful, display=display):
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps({"sha256": digest}, ensure_ascii=False), encoding="utf-8")

    def handle(self, text: str) -> None:
        report_kind = _detect_report_request(text)
        if report_kind:
            self._show_saved_report(report_kind)
            return
        if _looks_like_trade_report(text):
            self._record_trade_report(text)
            return
        if _looks_like_portfolio_input(text):
            self._record_portfolio_snapshot(text)
            return
        if _looks_like_portfolio_review(text):
            print(self.portfolio_advisor.analyze(), flush=True)
            return
        action = _detect_watchlist_action(text)
        if action:
            query = _extract_stock_query(text)
            self._update_watchlist(action, query)
            return
        if _looks_like_bare_stock(text):
            ticker = self.resolve_ticker(text)
            if ticker:
                self._apply_watchlist_change("add", ticker, text)
                return
        self.answer_question(text)

    def _show_saved_report(self, report_kind: str) -> None:
        is_premarket = report_kind == "premarket"
        report_dir = self.settings.report_dir / ("premarket" if is_premarket else "close")
        files = sorted(report_dir.glob("*.md"), key=lambda path: path.stem, reverse=True)
        if is_premarket:
            today_path = report_dir / f"{datetime.now().date().isoformat()}.md"
            market_session = is_us_market_session()
            if not market_session:
                files = [path for path in files if path.stem < today_path.stem]
                print(f"今天{market_closed_reason()}，不生成新的盘前报告，改为显示最近一份已整理报告。", flush=True)
            elif today_path.exists() and self._premarket_report_is_failed(today_path):
                print("今天的盘前报告生成失败，将重新调用 deepseek-v4-flash 生成...", flush=True)
                try:
                    generate_premarket_summary(display=False)
                except Exception as exc:
                    print(f"盘前报告重新生成失败: {type(exc).__name__}: {exc}", flush=True)
                files = sorted(report_dir.glob("*.md"), key=lambda path: path.stem, reverse=True)
            elif not today_path.exists():
                if self._premarket_generation_time_passed():
                    print("今天还没有盘前报告，且已过开盘前30分钟，正在即时整理并保存...", flush=True)
                    try:
                        generate_premarket_summary(display=False)
                    except Exception as exc:
                        print(f"盘前报告即时整理失败: {type(exc).__name__}: {exc}", flush=True)
                    files = sorted(report_dir.glob("*.md"), key=lambda path: path.stem, reverse=True)
                else:
                    files = [path for path in files if path.stem < today_path.stem]
                    print("今天还没到盘前报告整理时间，改为显示最近一个交易日已整理的盘前报告。", flush=True)
        if not files:
            name = "盘前" if is_premarket else "盘后"
            print(f"目前还没有已整理的{name}报告。", flush=True)
            return
        if not is_premarket:
            files = self._filter_reports_after_watchlist_change(files)
            if not files:
                print("watchlist 已变更，目前还没有基于当前 watchlist 生成的新盘后报告。", flush=True)
                return
        latest = files[0]
        title = f"{'盘前' if is_premarket else '盘后'}报告 · {latest.stem}"
        content = latest.read_text(encoding="utf-8")
        self.reporter.display_summary(content, title)
        self.reporter.notifier.send_markdown(title, content)
        self._remember_turn(f"查看{title}", content)

    def _premarket_report_is_failed(self, path) -> bool:
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            return False
        markers = [
            "无法获取模型总结",
            "盘前总结没有生成",
            "模型调用失败",
            "deepseek-v4-flash 盘前总结暂时不可用",
        ]
        return any(marker in content for marker in markers)

    def _premarket_generation_time_passed(self) -> bool:
        zone = ZoneInfo(self.settings.premarket_summary_timezone)
        now = datetime.now(zone)
        scheduled = now.replace(
            hour=self.settings.premarket_summary_hour,
            minute=0,
            second=0,
            microsecond=0,
        )
        return now >= scheduled

    def _record_portfolio_snapshot(self, text: str) -> None:
        if self._parse_and_save_portfolio(text):
            print(self.portfolio_advisor.analyze(), flush=True)

    def _parse_and_save_portfolio(self, text: str, display: bool = True) -> bool:
        if not self.client:
            if display:
                print("无法记录配置：未配置 DeepSeek API。", flush=True)
            return False
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "把自由格式文本中的当前个人资产配置提取为结构化快照。文本可能来自券商复制、表格或随手记录。"
                        "金额单位按原文换算，例如2万=20000；公司能识别时 symbol_or_company 优先返回美股ticker。"
                        "证券有持股数量时填quantity；只有当前市值时quantity填0并填market_value。"
                        "平均成本或当前市值未提供时对应字段填0，不要因此拒绝整个配置。"
                        "现金未提及填0；其他资产必须有当前估值，不能把目标配置或计划当作现有资产。"
                        "只要至少识别到一项现金、证券或其他资产即可complete=true；完全没有可用资产数据时才false。禁止编造数字。"
                    ),
                },
                {"role": "user", "content": text},
            ],
            text={"format": {"type": "json_schema", "name": "portfolio_input", "schema": PORTFOLIO_INPUT_SCHEMA, "strict": True}},
        )
        parsed = parse_json_text(response_text(response))
        if not parsed["complete"]:
            if display:
                print(f"暂未记录配置：{parsed['missing_information']}。", flush=True)
            return False

        holdings = []
        for row in parsed["holdings"]:
            ticker = self.resolve_ticker(str(row["symbol_or_company"]))
            if not ticker:
                if display:
                    print(f"暂未记录配置：无法确认“{row['symbol_or_company']}”对应的 ticker。", flush=True)
                return False
            holdings.append(
                {
                    "symbol": ticker,
                    "name": row["name"],
                    "asset_type": row["asset_type"],
                    "quantity": row["quantity"],
                    "average_cost": row["average_cost"],
                    "market_value": row["market_value"],
                }
            )
        portfolio = {
            "base_currency": parsed["base_currency"] or "USD",
            "cash": parsed["cash"],
            "holdings": holdings,
            "other_assets": parsed["other_assets"],
        }
        self.portfolio_store.replace(portfolio, text)
        if display:
            print(f"资产配置已整理：{len(holdings)} 个证券持仓，{len(parsed['other_assets'])} 项其他资产。", flush=True)
        return True

    def _record_trade_report(self, text: str) -> None:
        if not self.client:
            print("无法记录交易：未配置 DeepSeek API，不能可靠解析自然语言交易。", flush=True)
            return
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "从用户对已完成交易的陈述中提取交易。每笔必须有方向、公司或ticker、数量、每股成交价；"
                        "手续费未提及填0。缺少任一必填信息时 complete=false，说明缺什么，不要猜测。"
                        "不要把咨询是否应该交易的问题识别为已完成交易。"
                        "如果用户说清仓、全卖、liquidated、closed 或 sold all，设置 action=sell、position_effect=full、quantity=0，并提取成交价。"
                        "如果用户说减仓或 partial sell，设置 position_effect=partial，并要求明确数量。"
                        "普通已完成买入或卖出设置 position_effect=normal。"
                    ),
                },
                {"role": "user", "content": text},
            ],
            text={"format": {"type": "json_schema", "name": "completed_trade_report", "schema": TRADE_REPORT_SCHEMA, "strict": True}},
        )
        parsed = parse_json_text(response_text(response))
        if not parsed["complete"] or not parsed["trades"]:
            detail = parsed["missing_information"] or "没有识别到完整的已完成交易"
            print(f"暂未更新资产配置：{detail}。请提供标的、买卖方向、数量和每股成交价。", flush=True)
            return

        normalized = []
        for trade in parsed["trades"]:
            ticker = self.resolve_ticker(str(trade["symbol_or_company"]))
            if not ticker:
                print(f"暂未更新资产配置：无法确认“{trade['symbol_or_company']}”对应的 ticker。", flush=True)
                return
            position_effect = trade.get("position_effect", "normal")
            quantity = float(trade["quantity"] or 0)
            if trade["action"] == "sell" and position_effect == "full":
                try:
                    quantity = self.portfolio_store.position_quantity(ticker)
                except ValueError as exc:
                    print(f"暂未更新资产配置：{exc}", flush=True)
                    return
            elif quantity <= 0:
                print(f"暂未更新资产配置：{ticker} 需要明确卖出/买入数量；如果是全卖，请说“清仓 {ticker} 成交价”。", flush=True)
                return
            normalized.append(
                {
                    "action": trade["action"],
                    "symbol": ticker,
                    "quantity": quantity,
                    "price": trade["price"],
                    "fees": trade["fees"],
                    "position_effect": position_effect,
                    "source_text": text,
                }
            )

        portfolio = self.portfolio_store.apply_trades(normalized)
        self._sync_portfolio_text_state()
        for trade in normalized:
            if trade["action"] == "buy":
                self.watchlist.add(trade["symbol"])
            elif trade["action"] == "sell" and not any(
                str(row.get("symbol", "")).upper() == trade["symbol"] for row in portfolio.get("holdings", [])
            ):
                self.watchlist.remove(trade["symbol"])
        summary = "；".join(
            f"{'买入' if row['action'] == 'buy' else '卖出'} {row['symbol']} {row['quantity']:g}股 @ {row['price']:g}"
            for row in normalized
        )
        print(f"资产配置已更新：{summary}", flush=True)
        print(self.portfolio_advisor.analyze(), flush=True)

    def _sync_portfolio_text_state(self) -> None:
        self.portfolio_store.write_text_snapshot(self.settings.portfolio_text_file)
        text = self.settings.portfolio_text_file.read_text(encoding="utf-8")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        self.settings.portfolio_text_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings.portfolio_text_state_path.write_text(
            json.dumps({"sha256": digest}, ensure_ascii=False),
            encoding="utf-8",
        )

    def resolve_ticker(self, query: str) -> str | None:
        normalized = query.strip().lower()
        if normalized in COMMON_ALIASES:
            return COMMON_ALIASES[normalized]
        if re.fullmatch(r"[A-Za-z]{1,5}(?:\.[A-Za-z])?", query.strip()):
            return query.strip().upper()

        # Finnhub is authoritative for English company names.
        if self.finnhub.enabled:
            try:
                matches = self.finnhub.search_symbol(query)
                us_equities = [row for row in matches if row.get("type") == "Common Stock" and "." not in str(row.get("symbol", ""))]
                candidate = (us_equities or matches)[0] if (us_equities or matches) else None
                if candidate and candidate.get("symbol"):
                    return str(candidate["symbol"]).upper()
            except Exception:
                pass

        # Chinese or informal company names are resolved by the model, then verified through Finnhub when possible.
        if self.client:
            ticker = self._resolve_with_model(query)
            # Finnhub 的免费权限可能无法验证标的；模型返回规范 ticker 时仍允许录入，
            # 后续 Yahoo 行情请求会再次提供数据层校验。
            if ticker and (re.fullmatch(r"[A-Z]{1,5}(?:\.[A-Z])?", ticker) or self._verify_ticker(ticker)):
                return ticker
        return None

    def _resolve_with_model(self, query: str) -> str | None:
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": "把用户输入的美股公司名称转换为其主要美国交易 ticker。无法确定时 ticker 返回空字符串。",
                },
                {"role": "user", "content": query},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "ticker_resolution",
                    "schema": {
                        "type": "object",
                        "properties": {"ticker": {"type": "string"}},
                        "required": ["ticker"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                }
            },
        )
        data = parse_json_text(response_text(response))
        ticker = str(data.get("ticker") or "").strip().upper()
        return ticker or None

    def _verify_ticker(self, ticker: str) -> bool:
        if not self.finnhub.enabled:
            return bool(re.fullmatch(r"[A-Z]{1,5}(?:\.[A-Z])?", ticker))
        try:
            profile = self.finnhub.get_profile(ticker)
            return bool(profile.get("ticker") or profile.get("name"))
        except Exception:
            return False

    def _update_watchlist(self, action: str, query: str) -> None:
        if not query:
            print("请提供股票名称或 ticker。", flush=True)
            return
        ticker = self.resolve_ticker(query)
        if not ticker:
            print(f"无法确认“{query}”对应的美股 ticker。", flush=True)
            return
        self._apply_watchlist_change(action, ticker, query)

    def _apply_watchlist_change(self, action: str, ticker: str, original: str) -> None:
        if action == "remove":
            changed = self.watchlist.remove(ticker)
            message = f"已从 watchlist 移除 {ticker}（{original}）。" if changed else f"{ticker} 不在 watchlist 中。"
        else:
            changed = self.watchlist.add(ticker)
            message = f"已加入 watchlist：{ticker}（{original}）。" if changed else f"{ticker} 已在 watchlist 中。"
        if changed:
            self._mark_watchlist_changed()
            self._invalidate_today_close_report()
        print(message, flush=True)
        print(f"当前 watchlist：{', '.join(self.watchlist.get()) or '空'}", flush=True)

    def _mark_watchlist_changed(self) -> None:
        path = self.settings.data_dir / "watchlist_changed_at.txt"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
        except OSError:
            pass

    def _filter_reports_after_watchlist_change(self, files: list) -> list:
        marker = self.settings.data_dir / "watchlist_changed_at.txt"
        if not marker.exists():
            return files
        try:
            marker_mtime = marker.stat().st_mtime
            return [path for path in files if path.stat().st_mtime >= marker_mtime]
        except OSError:
            return files

    def _invalidate_today_close_report(self) -> None:
        path = self.settings.report_dir / "close" / f"{datetime.now().date().isoformat()}.md"
        if not path.exists():
            return
        archived = path.with_suffix(path.suffix + ".stale")
        try:
            path.replace(archived)
        except OSError:
            pass

    def answer_question(self, question: str) -> None:
        print("正在整理金融数据，请稍候...", flush=True)
        try:
            self._answer_question(question)
        except Exception as exc:
            print(f"金融问题回答失败：{type(exc).__name__}: {exc}", flush=True)

    def _answer_question(self, question: str) -> None:
        if not self.client:
            print("无法回答：未配置 DeepSeek API。", flush=True)
            return
        classification = self._classify_question(question)
        category = classification["category"]
        tickers = [str(ticker).upper() for ticker in classification.get("tickers", []) if ticker]
        if not tickers and _looks_like_follow_up(question):
            inherited = self._last_discussed_tickers()
            if inherited:
                tickers = inherited
        if category == "non_financial":
            print("无法回答：仅支持金融类问题。", flush=True)
            return
        if category == "financial_report":
            ticker = tickers[0] if tickers else self._resolve_report_ticker(question)
            if not ticker:
                print("无法确认需要分析的公司，请提供股票代码或公司名称。", flush=True)
                return
            answer = self.fundamental_agent.analyze_report(ticker, question)
            print(answer, flush=True)
            self._remember_turn(question, answer)
            return
        if category == "general_knowledge":
            answer = self._answer_general_knowledge(question)
            print(answer, flush=True)
            self._remember_turn(question, answer)
            return

        if category == "current_market" and self._is_outside_local_scope(tickers):
            answer = self.tickertick_question_agent.answer(question, tickers)
            print(answer, flush=True)
            self._remember_turn(question, answer)
            return

        context = self._project_market_context(tickers, question)
        has_news = bool(context["news_and_analyses"] or context["premarket_summary"])
        if classification.get("needs_news", True) and not has_news:
            notice = "我的新闻库里目前没有抓到相关消息。"
            print(notice, flush=True)
            self._remember_turn(question, notice)
            return

        answer = self._format_market_answer(question, context, tickers)
        print(answer, flush=True)
        self._remember_turn(question, answer)

    def _is_outside_local_scope(self, tickers: list[str]) -> bool:
        """Only use the external news fallback for a clearly non-watchlist ticker."""
        if not tickers:
            return False
        watchlist = set(self.watchlist.get())
        return all(ticker not in watchlist for ticker in tickers)

    def _project_market_context(self, tickers: list[str], question: str) -> dict:
        records = self._recent_records(tickers)
        quote_symbols = tickers[:3]
        include_technical = _question_needs_technical_context(question)
        include_vix = _question_needs_market_sentiment(question)
        return {
            "news_and_analyses": records,
            "premarket_summary": self._latest_premarket_summary(),
            "real_time_quotes": self._quotes(quote_symbols),
            "cash_flow_and_balance_sheet": [],
            "vix_indicator": self._vix_indicator() if include_vix else {"available": False, "reason": "当前问题未请求市场情绪数据"},
            "technical_snapshots": self._technical_snapshots(quote_symbols) if include_technical else {},
            "watchlist": self.watchlist.get(),
            "conversation_context": self._conversation_context(),
        }

    def _remember_turn(self, question: str, answer: str) -> None:
        self._conversation_history.append(
            {
                "user": question.strip(),
                "assistant": answer.strip()[:3000],
            }
        )
        self._conversation_history = self._conversation_history[-8:]

    def _conversation_context(self) -> list[dict[str, str]]:
        return self._conversation_history[-6:]

    def _last_discussed_tickers(self) -> list[str]:
        known = set(self.watchlist.get())
        found: list[str] = []
        for turn in reversed(self._conversation_history):
            text = f"{turn.get('user', '')}\n{turn.get('assistant', '')}"
            for ticker in re.findall(r"\b[A-Z]{1,5}(?:\.[A-Z])?\b", text):
                if ticker in known and ticker not in found:
                    found.append(ticker)
            if found:
                return found[:3]
        return []

    def _recent_records(self, tickers: list[str]) -> list[dict]:
        store = DedupStore(self.settings.sqlite_path)
        try:
            records = store.recent_records(datetime.now(timezone.utc) - timedelta(days=7))[:30]
        finally:
            store.close()
        compact = [
            {
                "headline": row["news"].get("headline"),
                "summary": row["news"].get("summary"),
                "url": row["news"].get("url"),
                "published_at": row["news"].get("published_at"),
                "ticker": row["analysis"].get("ticker"),
                "sentiment": row["analysis"].get("sentiment"),
                "reasoning": row["analysis"].get("reasoning_zh"),
                "priced_in": row["analysis"].get("priced_in"),
            }
            for row in records
        ]
        if tickers:
            wanted = set(tickers)
            compact = [row for row in compact if row["ticker"] in wanted]
        return compact

    def _latest_premarket_summary(self) -> str:
        report_dir = self.settings.report_dir / "premarket"
        files = sorted(report_dir.glob("*.md"), reverse=True) if report_dir.exists() else []
        if not files:
            return ""
        return files[0].read_text(encoding="utf-8")[:8000]

    def _quotes(self, tickers: list[str]) -> list[dict]:
        if not self.finnhub.enabled:
            return []
        quotes = []
        for ticker in tickers[:20]:
            try:
                quote = self.finnhub.get_quote(ticker)
                quotes.append(
                    {
                        "ticker": ticker,
                        "current": quote.current,
                        "previous_close": quote.previous_close,
                        "change_pct": quote.change_pct,
                    }
                )
            except Exception:
                continue
        return quotes

    def _financial_metrics(self, tickers: list[str]) -> list[dict]:
        if not self.finnhub.enabled:
            return []
        selected_keys = [
            "freeCashFlowPerShareTTM",
            "cashFlowPerShareTTM",
            "netIncomePerShareTTM",
            "currentRatioAnnual",
            "totalDebt/totalEquityAnnual",
        ]
        results = []
        for ticker in tickers[:10]:
            try:
                payload = self.fundamental_agent.basic_financials(ticker)
                metrics = payload.get("metric") or {}
                results.append(
                    {
                        "ticker": ticker,
                        "metrics": {key: metrics.get(key) for key in selected_keys if metrics.get(key) is not None},
                        "fundamental_agent_score": payload.get("agent_score", {}),
                    }
                )
            except Exception:
                continue
        return results

    def _vix_indicator(self) -> dict:
        try:
            return self.fundamental_agent.get_vix()
        except Exception as exc:
            return {"available": False, "reason": str(exc)}

    def _technical_snapshots(self, tickers: list[str]) -> dict:
        snapshots = {}
        for ticker in tickers[:10]:
            try:
                snapshots[ticker] = self.technical_agent.get_snapshot(ticker)
            except Exception as exc:
                snapshots[ticker] = {"available": False, "reason": str(exc)}
        return snapshots

    def _classify_question(self, question: str) -> dict:
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "Use conversation_context to resolve follow-up words such as 它, 这个, 刚才那个, 继续, 为什么, 那. "
                        "If the previous turn discussed a ticker, company, sector, or market theme and the current question omits it, inherit that subject. "
                        "把问题分为 financial_report（财报、年报、季报、10-K、10-Q、三表或公司基本面分析）、"
                        "current_market（近期新闻、某股最近影响、今天情绪、实时行情）、"
                        "general_knowledge（不涉及近期状况的金融知识科普）或 non_financial。"
                        "同时提取问题涉及的美股 ticker；needs_news 表示回答是否必须依赖近期新闻。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"question": question, "conversation_context": self._conversation_context()},
                        ensure_ascii=False,
                    ),
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "question_classification",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "enum": ["financial_report", "current_market", "general_knowledge", "non_financial"],
                            },
                            "tickers": {"type": "array", "items": {"type": "string"}},
                            "needs_news": {"type": "boolean"},
                        },
                        "required": ["category", "tickers", "needs_news"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                }
            },
        )
        return parse_json_text(response_text(response))

    def _resolve_report_ticker(self, question: str) -> str | None:
        direct = re.search(r"\b[A-Z]{1,5}(?:\.[A-Z])?\b", question)
        if direct:
            return direct.group(0).upper()
        for alias, ticker in COMMON_ALIASES.items():
            if alias in question.lower():
                return ticker
        return self._resolve_with_model(question) if self.client else None

    def _format_market_answer(self, question: str, context: dict, tickers: list[str]) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        RESEARCH_ROLE
                        + "\n只使用提供的数据回答，不得用训练记忆补充近期事实。"
                        "如果数据里有conversation_context，必须用它承接用户上文；遇到“它、这个、刚才那个、继续、为什么、那”时，先判断指代对象再回答。"
                        "结论必须一句话说明利好/利空/中性及力度，并给出 60%-90% 置信度。"
                        "依据必须对应新闻或行情；现金流视角要优先于利润，缺数据时明确写数据不足。"
                        "如果数据里包含 news_agent、technical_agent 或 fundamental_agent 的评分，必须在依据或风险中写出对应总分和方向。"
                        "空头检查必须主动寻找估值、竞争、需求、监管、资产负债表、现金流和执行风险。"
                        "不确定性必须说明 price in（市场是否已经提前反映）及待验证点。"
                        "决策含义只说明应关注哪些验证信号，不能给直接交易指令。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"问题：{question}\n"
                        f"最近对话：{json.dumps(self._conversation_context(), ensure_ascii=False)}\n"
                        f"数据：{json.dumps(context, ensure_ascii=False)}"
                    ),
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "market_research_answer",
                    "schema": MARKET_ANSWER_SCHEMA,
                    "strict": True,
                }
            },
        )
        data = parse_json_text(response_text(response))
        confidence = max(60, min(90, int(data["confidence_pct"])))
        parts = [
            f"1. 结论：{data['conclusion']}（置信度：{confidence}%）",
            f"2. 依据：{data['evidence']}\n现金流视角：{data['cash_flow_view']}",
            f"3. 不确定性与风险：{data['uncertainty']}\n空头检查清单：{data['bear_checklist']}",
            f"4. 决策含义：{data['decision_implication']}",
        ]
        if data.get("background"):
            parts[-1] += f"\n背景延伸：{data['background']}"
        if tickers:
            parts.append("这不构成投资建议。")
        return "\n\n".join(parts)

    def _answer_general_knowledge(self, question: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        RESEARCH_ROLE
                        + "\n回答纯金融知识问题时必须说明这个知识如何服务于投资决策。"
                        "如果给出判断性结论，标注 60%-90% 的置信度；不涉及当前市场事实。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"question": question, "conversation_context": self._conversation_context()},
                        ensure_ascii=False,
                    ),
                },
            ],
        )
        answer = response_text(response).strip()
        return "这部分是基于通用金融知识，不是基于今天抓到的新闻数据。\n\n" + answer

def _detect_watchlist_action(text: str) -> str | None:
    lowered = text.lower()
    if any(word in lowered for word in ["移除", "删除", "去掉", "remove"]):
        return "remove"
    if any(word in lowered for word in ["添加", "加入", "新增", "加到", "add"]):
        return "add"
    return None


def _detect_report_request(text: str) -> str | None:
    normalized = re.sub(r"[，。！？,.!?\s]", "", text.lower())
    premarket_terms = ["盘前报告", "盘前总结"]
    close_terms = ["盘后报告", "盘后总结", "收盘报告", "收盘总结"]
    request_words = ["", "查看", "显示", "打开", "输出", "给我", "我要看", "看看"]
    if normalized in {prefix + term for prefix in request_words for term in premarket_terms}:
        return "premarket"
    if normalized in {prefix + term for prefix in request_words for term in close_terms}:
        return "close"
    return None


def _extract_stock_query(text: str) -> str:
    cleaned = re.sub(r"(我想|请|帮我|把|从|到|在|watchlist|观察列表|股票|标的|里面|里|中)", " ", text, flags=re.I)
    cleaned = re.sub(r"(添加|加入|新增|加到|移除|删除|去掉|add|remove)", " ", cleaned, flags=re.I)
    return " ".join(cleaned.split()).strip("，。,. ")


def _looks_like_bare_stock(text: str) -> bool:
    if len(text) > 30 or any(mark in text for mark in ["?", "？"]):
        return False
    question_words = ["为什么", "怎么样", "如何", "多少", "走势", "新闻", "影响", "分析", "是什么"]
    return not any(word in text for word in question_words)


def _looks_like_follow_up(text: str) -> bool:
    lowered = text.lower()
    markers = [
        "它", "这个", "这件事", "刚才", "刚刚", "上面", "前面", "继续", "那", "所以",
        "为什么", "风险呢", "影响呢", "怎么看", "然后呢", "what about", "why", "continue",
    ]
    return any(marker in lowered for marker in markers)


def _question_needs_technical_context(text: str) -> bool:
    terms = ["技术", "均线", "macd", "rsi", "kdj", "k线", "蜡烛", "压力位", "支撑位", "背离", "放量", "走势"]
    lowered = text.lower()
    return any(term in lowered for term in terms)


def _question_needs_market_sentiment(text: str) -> bool:
    terms = ["大盘", "市场情绪", "vix", "恐慌", "波动率", "宏观", "美股市场"]
    lowered = text.lower()
    return any(term in lowered for term in terms)


def _looks_like_trade_report(text: str) -> bool:
    """识别用户在汇报已完成交易时常用的自然表达。"""
    explicit_record_words = ["记录交易", "记一笔", "录入交易", "更新持仓"]
    completed_words = [
        "买了", "卖了", "新买", "新购", "购入", "售出", "成交", "交易了",
        "建仓", "清仓", "减仓", "全卖", "补仓", "新增持仓", "增加持仓", "新的买入", "新增买入",
        "已经买入", "已经卖出", "完成买入", "完成卖出",
    ]
    past_context = ["今天", "刚刚", "刚才", "昨日", "昨天", "已", "成交"]
    trade_direction = any(word in text for word in ["买入", "卖出", "买", "卖"])
    record_context = any(word in text for word in ["记录", "录入", "更新", "记下"])
    has_past_context = any(word in text for word in past_context)
    has_quantity = bool(re.search(r"\d|[零一二三四五六七八九十百千万两]+\s*(股|份|张)", text))
    return (
        any(word in text for word in explicit_record_words)
        or any(word in text for word in completed_words)
        or (trade_direction and record_context)
        or (trade_direction and has_past_context and has_quantity)
    )


def _looks_like_portfolio_review(text: str) -> bool:
    review_words = ["分析", "评价", "评估", "查看", "重新分析"]
    portfolio_words = ["资产配置", "投资组合", "组合配置", "我的配置"]
    return any(word in text for word in review_words) and any(word in text for word in portfolio_words)


def _looks_like_portfolio_input(text: str) -> bool:
    action_words = ["记住", "记录", "录入", "设置", "保存"]
    portfolio_words = ["资产配置", "投资组合", "组合配置", "我的配置", "我的资产"]
    return any(word in text for word in action_words) and any(word in text for word in portfolio_words)
