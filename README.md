# Order Flow Imbalance Prediction

An out-of-sample experiment on whether changes in best-quote depth predict the next mid-price move.

## Overview

Rather than assume that order flow is predictive, this project compares a simple rolling OFI signal against two elementary forecasts. The target is the direction of the next nonzero mid-price change, not a fixed clock-time return.

## Method

- Construct signed bid and ask flow from changes in best prices and displayed quantities. Subtract ask flow from bid flow.
- Sum the last 20 quote-update imbalances by default and normalize by current average displayed depth.
- At each nonzero mid-price change, record the signal and resolve its label only when the subsequent move occurs.
- Split the observations chronologically (70% training, 30% testing). Fit five signal bins from training data only, smoothing estimated up probabilities toward the training up rate.
- Compare accuracy and Brier score with the training up-rate forecast and the previous-move direction.

## Run

```bash
python order_flow_imbalance.py
python order_flow_imbalance.py quotes.csv --window 20
```

The no-argument command uses a synthetic queue process with seed 7. For a real dataset, supply one instrument and one continuous session in feed order, with consistent timestamps and positive sizes.

```csv
timestamp,bid,bid_size,ask,ask_size
09:15:00.001,100.00,80,100.05,65
09:15:00.012,100.00,72,100.05,65
```

The two rows illustrate the format, not a sufficient training dataset.

## Reproduced results

1,356 training moves and 582 held-out synthetic moves.

| Forecast | Accuracy | Brier score |
|---|---:|---:|
| Rolling OFI | 51.9% | 0.2496 |
| Training up rate | 52.1% | 0.2496 |
| Repeat last move | 51.4% | 0.4863 |

OFI did not beat the training-up-rate baseline on accuracy and tied its displayed Brier score. The experiment therefore does not establish a predictive edge. This negative result is useful: it makes the distinction between a plausible signal and a validated signal explicit.

## Limitations and next steps

A toy queue model cannot establish live-market predictability. Real multi-session quotes, walk-forward evaluation, uncertainty intervals and transaction-cost-aware execution are needed before drawing trading conclusions.

## Requirements and files

Python 3.9+; no third-party dependencies are required.

- `order_flow_imbalance.py` — original project implementation.
- `requirements.txt` — dependency notes.
- `results/` — outputs from the example run, including [terminal output](results/example_run.txt).

The example was rerun using the bundled source. This is a research/educational project; it is not a production system.
