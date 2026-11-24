"""Does recent order flow predict the direction of the next mid-price change?

Run with Python 3.9+:
    python order_flow_imbalance.py
    python order_flow_imbalance.py quotes.csv --window 20

CSV format (one instrument, one continuous trading session):
timestamp,bid,bid_size,ask,ask_size
09:15:00.001,100.00,80,100.05,65
09:15:00.012,100.00,72,100.05,65

Use ISO timestamps in one timezone, sorted in feed order. Equal timestamps
are allowed when the feed gives the ordering. Sizes are displayed quantities.
"""
import argparse
import csv
import random
from bisect import bisect_right
from collections import deque
from datetime import datetime, time
from decimal import Decimal, InvalidOperation
from typing import NamedTuple


class Quote(NamedTuple):
    bid: Decimal
    bid_size: int
    ask: Decimal
    ask_size: int


def read_quotes(path):
    with open(path, newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {"timestamp", "bid", "ask", "bid_size", "ask_size"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("CSV needs " + ", ".join(sorted(required)))
        previous = None
        for row in reader:
            try:
                stamp = row["timestamp"].replace("Z", "+00:00")
                try:
                    current = datetime.fromisoformat(stamp)
                except ValueError:
                    current = time.fromisoformat(stamp)
                if previous is not None and current < previous:
                    raise ValueError("timestamps went backwards")
                if isinstance(current, datetime) and previous is not None:
                    if current.date() != previous.date():
                        raise ValueError("supply one session at a time")
                q = Quote(Decimal(row["bid"]), int(row["bid_size"]),
                          Decimal(row["ask"]), int(row["ask_size"]))
                if not (q.bid.is_finite() and q.ask.is_finite() and 0 < q.bid < q.ask):
                    raise ValueError("expected positive bid below ask")
                if min(q.bid_size, q.ask_size) <= 0:
                    raise ValueError("expected positive displayed quantities")
            except (ValueError, TypeError, AttributeError, InvalidOperation) as error:
                raise ValueError(f"CSV line {reader.line_num}: {error}") from error
            yield q
            previous = current


def demo_quotes(count=30000, seed=7):
    """Toy book: queues gain or lose lots. An exhausted queue moves the quote."""
    rng = random.Random(seed)
    bid = Decimal("100.00")
    tick = Decimal("0.05")
    sizes = [70, 70]
    for _ in range(count):
        yield Quote(bid, sizes[0], bid + tick, sizes[1])
        side = rng.randrange(2)
        sizes[side] += rng.randint(-35, 24)
        if sizes[side] <= 0:
            bid += tick if side == 1 else -tick
            sizes = [rng.randint(35, 100), rng.randint(35, 100)]


def order_flow(old, new):
    # When a best price changes, the queues belong to different price levels.
    bid_flow = ask_flow = 0
    if new.bid >= old.bid:
        bid_flow += new.bid_size
    if new.bid <= old.bid:
        bid_flow -= old.bid_size
    if new.ask <= old.ask:
        ask_flow += new.ask_size
    if new.ask >= old.ask:
        ask_flow -= old.ask_size
    return bid_flow - ask_flow


def observations(quotes, window):
    """One forecast just after each price move, resolved at the following move."""
    history = deque(maxlen=window)
    old = None
    pending = None
    for quote in quotes:
        if old is not None:
            history.append(order_flow(old, quote))
            # Twice the mid is enough to determine direction. Decimal keeps ties exact.
            change = (quote.bid + quote.ask) - (old.bid + old.ask)
            if change:
                if pending is not None:
                    yield (*pending, int(change > 0))
                if len(history) == window:
                    depth = (quote.bid_size + quote.ask_size) / 2
                    pending = (sum(history) / depth, int(change > 0))
        old = quote


def evaluate(rows):
    # Keep equal signals in the same bin. A tied cutoff can leave a bin empty.
    split = len(rows) * 7 // 10
    train, test = rows[:split], rows[split:]
    if len(train) < 25 or len(test) < 10:
        raise ValueError("need more quote updates (at least 36 labeled price moves)")
    signals = sorted(signal for signal, _, _ in train)
    cuts = [signals[len(train) * i // 5] for i in range(1, 5)]
    counts = [0] * 5
    ups = [0] * 5
    for signal, _, up in train:
        group = bisect_right(cuts, signal)
        counts[group] += 1
        ups[group] += up
    base = (sum(ups) + 1) / (len(train) + 2)
    # Give sparse bins two prior observations at the overall training up rate.
    probabilities = [(u + 2 * base) / (n + 2) for u, n in zip(ups, counts)]
    print(f"Chronological split: {len(train)} training moves, {len(test)} test moves")
    print("OFI bin (low to high)    Fitted P(up)    Test moves    Actual up")
    groups = [[] for _ in probabilities]
    forecasts = []
    for signal, last_up, up in test:
        group = bisect_right(cuts, signal)
        groups[group].append(up)
        forecasts.append((probabilities[group], base, last_up, up))
    for i, outcomes in enumerate(groups):
        actual = f"{sum(outcomes) / len(outcomes):.1%}" if outcomes else "—"
        print(f"{i + 1:>5} {probabilities[i]:>30.1%} {len(outcomes):>13} {actual:>12}")
    print("\nForecast                 Accuracy    Brier score (lower is better)")
    for col, name in enumerate(["Rolling OFI", "Training up rate", "Repeat last move"]):
        accuracy = sum((r[col] >= 0.5) == r[-1] for r in forecasts) / len(test)
        brier = sum((r[col] - r[-1]) ** 2 for r in forecasts) / len(test)
        print(f"{name:<24} {accuracy:>7.1%} {brier:>14.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test whether order flow predicts the next mid-price move.")
    parser.add_argument("csv", nargs="?", help="ordered best quotes for one session")
    parser.add_argument("--window", type=int, default=20, help="quote updates per signal")
    args = parser.parse_args()
    if args.window < 1:
        parser.error("--window must be positive")
    print(args.csv or "Synthetic queue demo (seed 7)")
    quotes = read_quotes(args.csv) if args.csv else demo_quotes()
    try:
        evaluate(list(observations(quotes, args.window)))
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(2, f"{error}\n")
