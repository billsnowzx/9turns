"""
九转神奇（TD Sequential）量化研究框架
运行入口：python main.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd

from data_loader import DataLoader
from signal_detector import TDSequential
from reliability import ReliabilityAnalyzer
from combo_strategy import ComboStrategy
from backtester import Backtester
from splitter import WalkForwardSplitter


def _gap_ratio(a: float, b: float) -> float:
    denom = max(abs(a), 1e-12)
    return abs(a - b) / denom


def run_full_research(
    market="a_share",       # "a_share" | "us" | "hk"
    symbol="000300",        # A股指数代码 / 美股ticker
    freq="daily",           # "daily" | "weekly"
    start="2015-01-01",
    end="2024-12-31",
):
    print("=" * 60)
    print(f"  九转神奇研究框架 | 市场:{market} | 标的:{symbol} | 周期:{freq}")
    print("=" * 60)

    print("\n[1/5] 获取历史行情数据...")
    loader = DataLoader()
    price_df = loader.load(market=market, symbol=symbol, freq=freq, start=start, end=end)
    print(f"  数据量：{len(price_df)} 根K线 | 时间范围：{price_df.index[0].date()} ~ {price_df.index[-1].date()}")

    print("\n[2/5] 全样本信号识别...")
    td = TDSequential(price_df)
    signals_df = td.detect_all()
    buy9 = signals_df[signals_df["signal"] == "buy9"]
    sell9 = signals_df[signals_df["signal"] == "sell9"]
    print(f"  买9信号：{len(buy9)} 个 | 卖9信号：{len(sell9)} 个")

    splitter = WalkForwardSplitter(train_ratio=0.7, n_splits=1)
    train_df, test_df = splitter.split(price_df)[0]
    print(
        f"  切分完成：train={len(train_df)} ({train_df.index[0].date()}~{train_df.index[-1].date()}) | "
        f"test={len(test_df)} ({test_df.index[0].date()}~{test_df.index[-1].date()})"
    )

    print("\n[3/5] 可靠性验证...")
    analyzer = ReliabilityAnalyzer(price_df, signals_df)
    rel_report = analyzer.full_report()
    analyzer.print_summary(rel_report)
    analyzer.save_report(rel_report, path="output/reliability_report.csv")

    print("\n[4/5] 样本内选组合（train）...")
    td_train = TDSequential(train_df)
    train_signals_df = td_train.detect_all()
    combo_train = ComboStrategy(train_df, train_signals_df)
    train_combo_results = combo_train.run_all_combos()
    combo_train.print_summary(train_combo_results)
    combo_train.save_report(train_combo_results, path="output/combo_report_train.csv")

    best_train_combo = combo_train.get_best_combo(train_combo_results)
    print(f"  样本内最优组合：{best_train_combo['name']}")

    print("\n[5/5] 回测：train / test...")
    bt_train = Backtester(train_df)
    bt_train_result = bt_train.run(
        entry_signals=best_train_combo["entries"],
        exit_signals=best_train_combo["exits"],
        sl_stop=0.05,
        tp_stop=0.15,
        strategy_name=f"{best_train_combo['name']} [TRAIN]",
    )
    bt_train.print_stats(bt_train_result)

    td_test = TDSequential(test_df)
    test_signals_df = td_test.detect_all()
    combo_test = ComboStrategy(test_df, test_signals_df)
    test_combo_results = combo_test.run_all_combos()

    matched_test_combo = next(
        (item for item in test_combo_results if item["name"] == best_train_combo["name"]),
        test_combo_results[0],
    )

    bt_test = Backtester(test_df)
    bt_test_result = bt_test.run(
        entry_signals=matched_test_combo["entries"],
        exit_signals=matched_test_combo["exits"],
        sl_stop=0.05,
        tp_stop=0.15,
        strategy_name=f"{best_train_combo['name']} [TEST]",
    )
    bt_test.print_stats(bt_test_result)
    bt_test.plot(bt_test_result, save_path="output/backtest_chart.html")

    train_ann = float(bt_train_result.get("annual_return", 0.0))
    test_ann = float(bt_test_result.get("annual_return", 0.0))
    gap = _gap_ratio(train_ann, test_ann)

    print("\n  ===== In-sample vs Out-of-sample =====")
    print(f"  in-sample annual_return:   {train_ann:.2%}")
    print(f"  out-of-sample annual_return: {test_ann:.2%}")
    print(f"  relative gap: {gap:.2%}")
    if gap > 0.5:
        print("  WARNING: train/test 差距 > 50%，可能存在过拟合。")

    summary = {
        "train": {
            "dataset": "train",
            "combo_name": best_train_combo["name"],
            "annual_return": train_ann,
            "sharpe": bt_train_result.get("sharpe"),
            "max_drawdown": bt_train_result.get("max_drawdown"),
            "calmar": bt_train_result.get("calmar"),
        },
        "test": {
            "dataset": "test",
            "combo_name": best_train_combo["name"],
            "annual_return": test_ann,
            "sharpe": bt_test_result.get("sharpe"),
            "max_drawdown": bt_test_result.get("max_drawdown"),
            "calmar": bt_test_result.get("calmar"),
        },
        "annual_return_gap": gap,
    }
    pd.DataFrame([summary["train"], summary["test"]]).to_csv(
        "output/walkforward_summary.csv", index=False, encoding="utf-8-sig"
    )

    print("\n" + "=" * 60)
    print("  研究完成！报告已保存到 output/ 目录")
    print("=" * 60)

    return rel_report, train_combo_results, {
        "train": bt_train_result,
        "test": bt_test_result,
        "summary": summary,
    }


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    run_full_research(
        market="a_share",
        symbol="000300",
        freq="daily",
        start="2015-01-01",
        end="2024-12-31",
    )
