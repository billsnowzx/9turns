from pathlib import Path

from hstech_aggregation import aggregate_results


ROOT = Path("output/hstech30")
SUMMARY_OUT = Path("output/hstech30_summary.csv")
DETAIL_OUT = Path("output/hstech30_detail.csv")
REGIME_OUT = Path("output/hstech30_regime_summary.csv")


def main():
    verdict = aggregate_results(ROOT, DETAIL_OUT, SUMMARY_OUT, REGIME_OUT)
    print("Aggregation complete.")
    print(f"- detail: {DETAIL_OUT}")
    print(f"- summary: {SUMMARY_OUT}")
    print(f"- regime summary: {REGIME_OUT}")
    print(f"- verdict: {verdict}")


if __name__ == "__main__":
    main()
