# 美股新闻分析 Agent

一个本地运行的 Python 后端 Agent，用于监控美股 watchlist 新闻，判断新闻是否会影响持仓标的，并输出简短的利多/利空分析。

## 配置

1. 安装 Python 3.11+。
2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 在 `.env` 中配置：

```env
FINNHUB_API_KEY=your_finnhub_key
DEEPSEEK_API_KEY=your_deepseek_key
DEEPSEEK_MODEL_ANALYZE=deepseek-v4-flash
DEEPSEEK_MODEL_SUMMARY=deepseek-v4-flash
POLL_INTERVAL_SECONDS=90
TICKERTICK_COMPANY_NEWS_INTERVAL_SECONDS=300
FINNHUB_MIN_REQUEST_INTERVAL_SECONDS=1.1
NEWS_MAX_AGE_HOURS=2
NEWS_RETENTION_HOURS=720
NEUTRAL_NEWS_RETENTION_HOURS=168
DB_CLEANUP_INTERVAL_HOURS=24
NEWS_DB_MAX_ITEMS=0
DISPLAY_TIMEZONE=Asia/Shanghai
PREMARKET_SUMMARY_HOUR=9
PREMARKET_SUMMARY_TIMEZONE=America/New_York
CLOSE_SUMMARY_HOUR=16
CLOSE_SUMMARY_MINUTE=10
EDGAR_IDENTITY=Your Name your@email.com
DCF_DISCOUNT_RATE=0.10
```

TickerTick 新闻无需额外 API key。持续轮询会按当前 watchlist 分别拉取 `curated`、`market`、`earning` 三类最新新闻；盘前报告只使用 `industry` 行业新闻。

`EDGAR_IDENTITY` 是 SEC 要求的访问身份，格式建议为真实姓名加邮箱。

4. 修改 `watchlist.txt`，一行一个 ticker。保存后下一轮轮询自动生效。
5. 把资产配置以任意格式写入 `portfolio.txt`。程序启动时先让 ChatGPT 整理为内部配置，再按最新可得价格分析，并自动把持仓 ticker 加入 watchlist。`portfolio.yaml` 由程序维护，无需手工编辑。

## 运行

```bash
python main.py
```

启动后终端会出现 `finance>` 提示符。示例：

```text
finance> 英伟达
finance> 添加 Apple Inc.
finance> 我想移除特斯拉
finance> NVDA 最近有什么重要新闻？
finance> 美联储最新利率信号对科技股有什么影响？
finance> 分析 AAPL 最新财报
finance> 英伟达最近一季财报怎么样？
finance> 今天以190美元买入5股AAPL，手续费1美元
finance> 今天以210美元卖出2股AAPL
finance> 重新分析我的资产配置
finance> 记住我的资产配置：现金2万美元，AAPL 10股成本180美元，另有1万美元国债
finance> 盘前报告
finance> 盘后报告
```

- 单独输入股票名称、全名或 ticker：自动解析并加入 `watchlist.txt`。
- 输入“添加/加入”：加入 watchlist；输入“移除/删除”：从 watchlist 删除。
- 其他输入作为金融问题处理。系统优先使用已监控新闻与结论，资料不足时再进行网页搜索。
- 每次启动以启动前的数据库为固定快照，只显示最后10条已有“消息+技术”完整结论的非中性告警；本页按时间正序排列，最新记录显示为第10条。输入“上翻”才加载紧接着更早的10条，后续页面采用相同顺序。中性或缺少技术综合结论的内部记录不会显示。
- 非金融问题会直接拒绝回答。输入 `exit` 可退出程序。
- 近期市场判断固定按“结论、依据、不确定性与风险、背景延伸”输出；具体标的分析会附带“不构成投资建议”。
- 纯金融知识问题会明确标注为通用知识，不会伪装成当天新闻结论。
- 财报问题会通过 edgartools 查询 SEC XBRL 数据，严格按宏观定位、三表穿透、护城河、DCF、安全边际和决策结论五步输出。
- 已完成交易需要提供标的、方向、数量和每股成交价；agent 会更新 `portfolio.yaml`、记录到 `data/trades.jsonl`，随后重新生成资产配置诊断。交易记录只修改本地配置，不会连接券商或执行真实交易。
- `portfolio.txt` 内容变化后，下次启动会自动重新整理；内容未变化时直接使用现有内部配置，避免重复调用模型或覆盖终端中已记录的交易。
- 配置诊断固定输出攻击性、防守性、集中度风险、流动性、抗通胀能力和长期复利质量六项评分，并给出不超过三套改进方案。
- 新闻只使用 TickerTick：`curated` 用于各大网站的标的新闻，`market` 用于直接关联的市场消息面，`earning` 用于公司经营/财报新闻；盘前报告使用 `industry` 行业新闻。
- Yahoo Finance 数据适合个人研究，但不是交易所官方实时行情；程序设置了超时和数据校验，数据源暂时不可用时会跳过技术面，不会阻塞新闻监控。

## 输出

- 实时新闻分析：`reports/intraday_alerts/YYYY-MM-DD.md`
- 盘前报告：`reports/premarket/YYYY-MM-DD.md`
- 盘后报告：`reports/close/YYYY-MM-DD.md`
- SQLite 运行状态（去重、问答与盘后消息面）：`data/news_history.sqlite3`

不相关新闻会被静默跳过；模型最终判断为 `neutral` 的新闻也不会输出正式报告。
默认只分析最近 2 小时内发布、且数据库里没见过的新闻；没有新新闻时不会输出结果。新闻时间统一显示为上海时区。
完整新闻告警保留30天；中性分析保留7天。每24小时执行一次分组清理，仅含过期中性分析的新闻正文会一起删除，同一新闻若还有非中性结论则继续保留。`NEWS_DB_MAX_ITEMS=0` 表示不再按条数提前删除。
盘前报告在美东开盘前半小时（09:00 ET）自动整理、显示并推送，盘后报告在收盘后（默认16:10 ET）自动整理、显示并推送；输入“盘前报告”或“盘后报告”仍可再次查看最近一份正文。
盘后报告已替换为 watchlist 逐标的风险报告：只使用日线和周线计算压力区、支撑区、确认背离、趋势、波动与量能，再融合过去24小时消息面和持仓集中度生成风险系数。报告不使用15分钟线；在下一份盘后报告生成前，输入“盘后报告”始终读取最近一次已整理文件。
新闻轮询、盘前总结和盘后总结使用三个独立执行通道；定时总结会暂停新一轮新闻分析，避免并发争抢模型额度。模型服务失败时仍会写出本地规则版总结，定时任务允许在错过触发后的一小时内补跑。

## 声明

本工具仅作个人新闻聚合与辅助分析用途，不构成投资建议。模型判断可能存在错误、滞后或误判，请勿作为唯一交易依据。
