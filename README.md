# 九转神奇（TD Sequential）量化研究框架

## 项目结构

```
td_sequential/
├── main.py              # 运行入口
├── data_loader.py       # 数据获取（A股/美股/港股）
├── signal_detector.py   # 信号识别（Setup 9 / Countdown 13）
├── reliability.py       # 可靠性验证（胜率/盈亏比/统计检验）
├── combo_strategy.py    # 组合策略（趋势/RSI/成交量/布林带）
├── backtester.py        # 回测引擎（vectorbt + 内置兜底）
├── requirements.txt     # 依赖清单
└── output/              # 报告输出目录（自动创建）
    ├── reliability_report.csv
    ├── combo_report.csv
    └── backtest_chart.html
```

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

```bash
python main.py
```

默认运行：A股沪深300（000300）日线，2015~2024。

## 自定义研究

```python
from main import run_full_research

# A股个股（贵州茅台）
run_full_research(market="a_share", symbol="600519", freq="daily")

# 美股 ETF（标普500）
run_full_research(market="us", symbol="SPY", freq="daily")

# 港股（恒生指数）
run_full_research(market="hk", symbol="^HSI", freq="weekly")

# A股周线研究
run_full_research(market="a_share", symbol="000300", freq="weekly")
```

## 单模块使用

### 只获取信号

```python
from data_loader import DataLoader
from signal_detector import TDSequential

loader = DataLoader()
df = loader.load(market="a_share", symbol="000300", freq="daily",
                 start="2020-01-01", end="2024-12-31")

td = TDSequential(df)
signals = td.detect_all()
print(signals)

# 可视化最近200根K线的信号
td.plot(signals, last_n=200, save_path="output/signal_chart.png")
```

### 只做可靠性分析

```python
from reliability import ReliabilityAnalyzer

analyzer = ReliabilityAnalyzer(df, signals)
report = analyzer.full_report()
analyzer.print_summary(report)
```

### 只测组合策略

```python
from combo_strategy import ComboStrategy

combo = ComboStrategy(df, signals)
results = combo.run_all_combos(hold_days=10)
combo.print_summary(results)
best = combo.get_best_combo(results)
print("最优组合：", best["name"])
```

## 研究扩展建议

1. **多标的批量验证**：对沪深300成分股逐一运行，统计整体胜率分布
2. **参数敏感性测试**：修改 `sl_stop` / `tp_stop` / `hold_days`，观察指标变化
3. **周期对比**：同一标的日线 vs 周线结果对比
4. **机器学习增强**：以各指标为特征，训练分类器预测信号成功率

## 核心指标解读

| 指标 | 说明 | 参考标准 |
|------|------|---------|
| 胜率 | 信号后 N 日盈利概率 | >55% 有参考价值 |
| 盈亏比 | 平均盈利/平均亏损 | >1.5 才值得交易 |
| p值（二项检验） | 胜率是否显著偏离 50% | <0.05 为统计显著 |
| 夏普比率 | 风险调整后收益 | >1.0 为及格，>1.5 优秀 |
| 最大回撤 | 净值最大跌幅 | <20% 为可接受 |
| 卡玛比率 | 年化收益/最大回撤 | >1.0 较好 |

## Countdown Modes

`TDSequential` now supports two countdown implementations via `countdown_mode`:

- `simplified` (default): uses only the base Countdown counting condition.
- `strict`: adds qualifier + cancellation checks, which typically reduces signal count.

Example:

```python
from signal_detector import TDSequential

td_simple = TDSequential(df, countdown_mode="simplified")
td_strict = TDSequential(df, countdown_mode="strict")

sig_simple = td_simple.detect_all()
sig_strict = td_strict.detect_all()
```

In normal datasets, `strict` should produce fewer or equal Countdown signals than `simplified`.
