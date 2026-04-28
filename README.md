# 9turns Research Framework

A TD Sequential research framework focused on reducing lookahead bias, adding train/test separation, and improving statistical reliability.

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

Outputs are saved under `output/`.

## Project Structure

- `main.py`: end-to-end research pipeline
- `data_loader.py`: market data loading + cache + frequency validation
- `signal_detector.py`: setup/countdown signal detection
- `combo_strategy.py`: filter combinations + train-set ranking
- `backtester.py`: backtest engine (simple, vectorbt fallback)
- `reliability.py`: forward-return and significance analysis
- `splitter.py`: walk-forward split utility
- `config.yaml`: central configuration

## Known Limitations

- Countdown has two implementations (`simplified` and `strict`) but both are still practical approximations of full DeMark rules.
- Single-instrument workflow: multi-symbol portfolio optimization is not implemented.
- Multi-market studies require manual orchestration (looping symbols/markets externally).
- Some validations still rely on fixed sample-size heuristics.

## How To Read Results

- Prefer out-of-sample metrics over in-sample metrics.
- If train/test annual return gap is >50%, treat the strategy as likely overfit.
- Significance decisions are based on Bonferroni-corrected p-values (`p_bonferroni`), not raw p-values.
- Validate consistency across windows (`3d/5d/10d/20d`) instead of one window only.

## Countdown Modes

`TDSequential` supports:

- `simplified`: base countdown condition only.
- `strict`: adds qualifier + cancellation logic, typically generating fewer signals.

## Extension Guide

### Add a new filter

1. Add a mask method in `ComboStrategy`.
2. Add it to `combo_defs` in `run_all_combos`.
3. Add tests under `tests/test_combo_strategy.py`.

### Add a new market source

1. Implement a new loader branch in `DataLoader.load`.
2. Normalize columns to `open/high/low/close/volume`.
3. Ensure `_validate_and_align_frequency` and `_validate` still pass.

### Tune parameters

Edit `config.yaml`:

- `FORWARD_DAYS`
- `countdown_mode`
- `hold_days`
- `data.*`
- `backtest.*`

No code changes required for common parameter tuning.
