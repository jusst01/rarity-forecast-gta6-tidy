"""Wide-Spread Rarity Forecaster — gta6-trophy-rarity-forecast-01 (standard library only).

A saturating log-time forecaster for GTA VI trophy earn rates. Histories are
collapsed into events; a count-consistent dilution branch handles flat
histories (one disclosed launch refresh, earners held fixed), while the live
branch fits Theil-Sen lines against ln(1 + hours) for both players and
earners, damps the slopes, shrinks them toward their (tier, rate-band) group
medians, caps the median's logit movement, blends toward the last read with a
horizon-dependent persistence weight, and widens the spread by a factor 1.2.

Every failure path falls back to a persistence forecast so the program never
raises and never emits an invalid row.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader runs `python3 -P -s`: -P leaves this file's directory off sys.path,
# so put it back first in case a sibling module is ever shipped beside main.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at those five levels.
Z5 = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]

# --- This entry's constants (do not retune toward sibling entries). ---------
SLOPE_DAMPING = 0.6          # player extrapolation and earner slope damping
LAMBDA_BASE = 0.80           # persistence shrinkage base
LAMBDA_GAIN = 0.35           # persistence shrinkage gain
LAMBDA_CAP = 0.97            # clamp: unclamped lambda > 1 crashes the spread
MOVE_LO, MOVE_HI = -1.9, 1.4  # tighter median movement cap, logit units
SPREAD_MULT = 1.2            # widened spread factor
# ---------------------------------------------------------------------------

FLAT_GROWTH = 3868.0 / 638.0  # the disclosed launch refresh's player growth
FLAT_MIX_W = 0.95            # mixture weight on the diluted component
FLAT_SIG_DILUTED = 0.10
FLAT_SIG_TAIL = 0.15
PINNED = [99.0, 99.9, 100.0, 100.0, 100.0]

BANDS = ((0.0, 1.0), (1.0, 10.0), (10.0, 50.0), (50.0, 100.0))


# --------------------------------------------------------------------------- #
# Numeric helpers
# --------------------------------------------------------------------------- #

def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def logit_pct(p):
    """Logit of a percentage, clamped to [0.01, 99.99] first."""
    p = clamp(float(p), 0.01, 99.99)
    return math.log((p / 100.0) / (1.0 - p / 100.0))


def inv_logit(l):
    """Inverse logit returning a percentage, guarded for large |l|."""
    l = float(l)
    if l >= 0.0:
        if l > 700.0:
            return 100.0
        return 100.0 / (1.0 + math.exp(-l))
    if l < -700.0:
        return 0.0
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return float(s[mid])
    return (s[mid - 1] + s[mid]) / 2.0


def round_half_up(x):
    """Half-up rounding to an int, the house convention (JS Math.round)."""
    scaled = float(x)
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    return int(whole)


def theil_sen(xs, ys):
    """Median pairwise slope and the median residual intercept."""
    slopes = []
    n = len(xs)
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[j] - xs[i]
            if dx != 0.0:
                slopes.append((ys[j] - ys[i]) / dx)
    if not slopes:
        return 0.0, median(ys) if ys else 0.0
    slope = median(slopes)
    return slope, median([y - slope * x for x, y in zip(xs, ys)])


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def mixture_quantile(level, comps):
    """Bisection for the level-quantile of a two-component normal mixture."""
    lo, hi = -16.0, 16.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        cdf = 0.0
        for w, mu, sigma in comps:
            cdf += w * norm_cdf((mid - mu) / sigma)
        if cdf < level:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def round6(x):
    """Deterministic 6-decimal rounding that keeps whole numbers as ints."""
    v = round(float(x), 6)
    if v != v:  # NaN guard
        return 0.0
    if v.is_integer():
        return int(v)
    return v


# --------------------------------------------------------------------------- #
# Case parsing
# --------------------------------------------------------------------------- #

def parse_ts(value):
    """Epoch seconds from an ISO-8601 timestamp; raises on anything unusable."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("no timestamp")
    t = value.strip()
    if t.endswith("Z") or t.endswith("z"):
        t = t[:-1] + "+00:00"
    dt = datetime.fromisoformat(t)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def as_number(value, default=0.0):
    try:
        if isinstance(value, bool) or value is None:
            return default
        f = float(value)
        if not math.isfinite(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _hashable(value):
    """A stable, hashable marker for a raw rate, preserving nullness."""
    if value is None:
        return None
    try:
        f = float(value)
        return f if math.isfinite(f) else repr(value)
    except (TypeError, ValueError):
        return repr(value)


def eventize(history):
    """Collapse reads into distinct states, oldest first.

    Each event: hours since the first read, player count N, rate values.
    A read joins the previous event when both its player count and its raw
    rate tuple match; a null rate is its own marker but reads as 0.0.
    """
    if not isinstance(history, list):
        return []
    reads = [h for h in history if isinstance(h, dict)]
    if not reads:
        return []

    try:
        t0 = parse_ts(reads[0].get("at"))
        have_clock = True
    except (ValueError, TypeError, OverflowError):
        t0 = 0.0
        have_clock = False

    events = []
    prev_key = None
    for index, read in enumerate(reads):
        raw = read.get("rates")
        raw_tuple = tuple(raw) if isinstance(raw, list) else ()
        key = (_hashable(read.get("players_tracked")),
               tuple(_hashable(r) for r in raw_tuple))
        if key == prev_key:
            continue
        prev_key = key

        if have_clock:
            try:
                hours = (parse_ts(read.get("at")) - t0) / 3600.0
            except (ValueError, TypeError, OverflowError):
                hours = float(index) * 0.5
        else:
            hours = float(index) * 0.5
        hours = max(0.0, as_number(hours, float(index) * 0.5))

        n = max(1.0, as_number(read.get("players_tracked"), 1.0))
        rates = [0.0 if r is None else clamp(as_number(r, 0.0), 0.0, 100.0)
                 for r in raw_tuple]
        events.append({"hours": hours, "n": n, "rates": rates})

    # Oldest first by hours, stable for ties.
    events.sort(key=lambda e: e["hours"])
    return events


def last_rate_of(event, index):
    rates = event["rates"]
    if index < len(rates):
        return rates[index]
    return 0.0


def band_key(rate):
    r = clamp(rate, 0.0, 100.0)
    for lo, hi in BANDS:
        if lo <= r < hi:
            return (lo, hi)
    return BANDS[-1]


def horizon_days_of(case):
    return max(0.0, as_number(case.get("horizon_minutes"), 0.0) / 1440.0)


# --------------------------------------------------------------------------- #
# Branches
# --------------------------------------------------------------------------- #

def flat_row(rate, n_last):
    """Count-consistent dilution: earners fixed across the one refresh."""
    if rate >= 99.99:
        return list(PINNED)
    earners = round_half_up(rate / 100.0 * n_last)
    if earners > 0:
        diluted = 100.0 * earners / (n_last * FLAT_GROWTH)
    else:
        diluted = 0.02
    comps = [
        (FLAT_MIX_W, logit_pct(diluted), FLAT_SIG_DILUTED),
        (1.0 - FLAT_MIX_W, logit_pct(max(rate, 0.05)), FLAT_SIG_TAIL),
    ]
    qs = [inv_logit(mixture_quantile(level, comps)) for level in LEVELS]
    return [round6(clamp(v, 0.0, 100.0)) for v in sorted(qs)]


def live_rows(events, trophies, horizon_days, x_tgt):
    """Saturating log-time fits for the live branch."""
    xs = [math.log1p(e["hours"]) for e in events]
    x_last = xs[-1]
    n_last = events[-1]["n"]
    dx = x_tgt - x_last

    # Players: Theil-Sen on ln(max(N, 10)), then two anchored extrapolations.
    ln_ns = [math.log(max(e["n"], 10.0)) for e in events]
    b_n, _ = theil_sen(xs, ln_ns)
    n_tgt = median([n_last * math.exp(f * SLOPE_DAMPING * b_n * dx)
                    for f in (0.85, 1.0)])
    n_tgt = max(n_tgt, 1e-9)

    lam = min(LAMBDA_CAP, LAMBDA_BASE + LAMBDA_GAIN * math.exp(-horizon_days / 2.0))
    n_events = len(events)
    w = n_events / float(n_events + 6)

    # Per-trophy earner slopes and event logit-rate residual spreads.  A pinned
    # trophy skips steps 1-7 of the spec, so it contributes no slope to any
    # group (its capped rate would otherwise lend the [50,100) band the
    # player-growth slope itself and skew every trophy grouped with it).
    last_rates = [last_rate_of(events[-1], i) for i in range(len(trophies))]
    raw_slopes = [None] * len(trophies)
    rsds = [0.12] * len(trophies)
    for i, _trophy in enumerate(trophies):
        if last_rates[i] >= 99.99:
            continue
        points_y = []
        y_logits = []
        for e in events:
            rate = last_rate_of(e, i)
            earners = max(0.5, float(round_half_up(rate / 100.0 * e["n"])))
            points_y.append(math.log(earners))
            y_logits.append(logit_pct(max(rate, 0.01)))
        raw_slopes[i] = theil_sen(xs, points_y)[0]

        b_r, a_r = theil_sen(xs, y_logits)
        resid = [y - (a_r + b_r * x) for x, y in zip(xs, y_logits)]
        rsd = math.sqrt(sum(r * r for r in resid) / len(resid)) if resid else 0.0
        rsds[i] = clamp(rsd, 0.12, 0.6)

    fitted = [s for s in raw_slopes if s is not None]
    global_median = median(fitted) if fitted else 0.0

    groups = {}
    for i, trophy in enumerate(trophies):
        if raw_slopes[i] is None:
            continue
        key = (str(trophy.get("tier", "")), band_key(last_rates[i]))
        groups.setdefault(key, []).append(raw_slopes[i])
    group_medians = {k: median(v) for k, v in groups.items()}

    rows = []
    for i, trophy in enumerate(trophies):
        # A trophy at or above 99.99 skips the fits entirely.
        if last_rates[i] >= 99.99:
            rows.append({"id": trophy["id"], "q": list(PINNED)})
            continue
        try:
            key = (str(trophy.get("tier", "")), band_key(last_rates[i]))
            g_med = group_medians.get(key, global_median)
            b_shrunk = w * raw_slopes[i] + (1.0 - w) * g_med
            q = live_row(last_rates[i], n_last, n_tgt, dx, lam, b_shrunk,
                         rsds[i], horizon_days)
            rows.append({"id": trophy["id"], "q": q})
        except Exception:
            rows.append({"id": trophy.get("id"),
                         "q": persistence_row(last_rates[i], horizon_days)})
    return rows


def live_row(rate, n_last, n_tgt, dx, lam, b_shrunk, rsd, horizon_days):
    """One LIVE trophy row: capped trend, persistence blend, widened spread."""
    if rate >= 99.99:
        return list(PINNED)

    last_logit = logit_pct(max(rate, 0.05))

    # Trend median on the earner scale, then back to a rate.
    e_last = round_half_up(rate / 100.0 * n_last)
    ln_e_tgt = math.log(max(e_last, 0.5)) + SLOPE_DAMPING * b_shrunk * dx
    e_tgt = min(math.exp(min(ln_e_tgt, 700.0)), 0.999 * n_tgt)
    trend_rate = 100.0 * e_tgt / n_tgt
    trend_logit = logit_pct(trend_rate)

    # Tighter movement cap, then the persistence blend.
    capped = clamp(trend_logit, last_logit + MOVE_LO, last_logit + MOVE_HI)
    m = (1.0 - lam) * capped + lam * last_logit

    # Widened spread, with the disagreement term guarded non-negative.
    disagree = lam * (1.0 - lam) * (trend_logit - last_logit) ** 2
    if not math.isfinite(disagree) or disagree < 0.0:
        disagree = 0.0
    sigma = SPREAD_MULT * math.sqrt(
        0.10 ** 2 + 0.16 ** 2 * horizon_days + 0.25 * rsd ** 2 + disagree)
    if e_last < 1:
        sigma = max(sigma, SPREAD_MULT)

    qs = [inv_logit(m + z * sigma) for z in Z5]
    return [round6(clamp(v, 0.0, 100.0)) for v in sorted(qs)]


# --------------------------------------------------------------------------- #
# Forecast entry points
# --------------------------------------------------------------------------- #

def persistence_row(rate, horizon_days):
    """Never-crash fallback: last rate, random-walk logit spread."""
    try:
        r = clamp(as_number(rate, 0.0), 0.0, 100.0)
        spread = max(0.35, 0.5 * math.sqrt(max(0.0, horizon_days)))
        lg = logit_pct(max(r, 0.01))
        qs = [inv_logit(lg + z * spread) for z in Z5]
        return [round6(clamp(v, 0.0, 100.0)) for v in sorted(qs)]
    except Exception:
        return [0.0, 0.0, 0.0, 0.0, 0.0]


def valid_rows(rows, trophies):
    if not isinstance(rows, list) or len(rows) != len(trophies):
        return False
    seen = set()
    for row, trophy in zip(rows, trophies):
        if not isinstance(row, dict):
            return False
        if row.get("id") != trophy.get("id"):
            return False
        if row["id"] in seen:
            return False
        seen.add(row["id"])
        q = row.get("q")
        if not isinstance(q, list) or len(q) != 5:
            return False
        prev = -math.inf
        for v in q:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return False
            if not math.isfinite(float(v)) or not (0.0 <= float(v) <= 100.0):
                return False
            if float(v) < prev:
                return False
            prev = float(v)
    return True


def persistence_forecast(case):
    """Persistence rows for every trophy the case exposes."""
    trophies = case.get("trophies") if isinstance(case, dict) else None
    if not isinstance(trophies, list):
        return {"trophies": []}
    trophies = [t for t in trophies if isinstance(t, dict) and "id" in t]
    hd = horizon_days_of(case if isinstance(case, dict) else {})
    history = case.get("history")
    last_rates = []
    if isinstance(history, list):
        for h in reversed(history):
            if isinstance(h, dict) and isinstance(h.get("rates"), list):
                last_rates = h["rates"]
                break
    rows = []
    for i, trophy in enumerate(trophies):
        rate = 0.0
        if i < len(last_rates) and last_rates[i] is not None:
            rate = as_number(last_rates[i], 0.0)
        rows.append({"id": trophy["id"], "q": persistence_row(rate, hd)})
    return {"trophies": rows}


def forecast_case(case, name):
    """Forecast one case; never raises, never returns an invalid row set."""
    trophies = []
    try:
        trophies = case["trophies"]
        if not isinstance(trophies, list):
            raise ValueError("trophies is not a list")
        trophies = [t for t in trophies if isinstance(t, dict) and "id" in t]
        if not trophies:
            return {"trophies": []}

        hd = horizon_days_of(case)
        events = eventize(case.get("history"))

        flat = (len(events) < 2
                or len({e["n"] for e in events}) <= 1)

        if flat:
            n_last = events[-1]["n"] if events else 1.0
            rows = []
            for i, trophy in enumerate(trophies):
                rate = last_rate_of(events[-1], i) if events else 0.0
                try:
                    rows.append({"id": trophy["id"], "q": flat_row(rate, n_last)})
                except Exception:
                    rows.append({"id": trophy["id"],
                                 "q": persistence_row(rate, hd)})
        else:
            horizon_min = as_number(case.get("horizon_minutes"), 0.0)
            x_tgt = math.log1p(events[-1]["hours"] + max(0.0, horizon_min) / 60.0)
            rows = live_rows(events, trophies, hd, x_tgt)

        if not valid_rows(rows, trophies):
            return persistence_forecast(case)
        return {"trophies": rows}
    except Exception:
        try:
            return persistence_forecast(case)
        except Exception:
            return {"trophies": [{"id": t.get("id"),
                                  "q": [0.0, 0.0, 0.0, 0.0, 0.0]}
                                 for t in trophies]}


def salvage_last_rates(text, n_trophies):
    """Last readable rates from a malformed case file, or None.

    History is decoded read by read from the first object that parses, so a
    file truncated mid-history still yields its most recent complete read.
    """
    marker = '"history"'
    pos = text.find(marker)
    if pos < 0:
        return None
    brace = text.find("[", pos + len(marker))
    if brace < 0:
        return None
    decoder = json.JSONDecoder()
    tail = text[brace + 1:]
    last = None
    i = 0
    while i < len(tail):
        while i < len(tail) and tail[i] in " \t\r\n,":
            i += 1
        if i >= len(tail) or tail[i] == "]":
            break
        try:
            value, end = decoder.raw_decode(tail, i)
        except ValueError:
            break
        if isinstance(value, dict) and isinstance(value.get("rates"), list):
            last = value["rates"]
        i = end
    if last is None:
        return None
    return [as_number(r, 0.0) if r is not None else 0.0
            for r in last[:n_trophies]]


def salvage_trophies(text):
    """Best-effort trophy list from a malformed case file."""
    marker = '"trophies"'
    pos = text.find(marker)
    if pos < 0:
        return None
    brace = text.find("[", pos + len(marker))
    if brace < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(text[brace:])
    except ValueError:
        return None
    if not isinstance(value, list):
        return None
    return [t for t in value if isinstance(t, dict) and "id" in t]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        out_path = OUTPUT_DIR / (path.stem + ".out")
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            text = ""
        case = None
        try:
            case = json.loads(text)
        except Exception:
            case = None

        if isinstance(case, dict):
            forecast = forecast_case(case, path.name)
        else:
            trophies = salvage_trophies(text) if text else None
            if trophies is None:
                forecast = {"trophies": []}
            else:
                rates = salvage_last_rates(text, len(trophies)) or []
                forecast = {"trophies": [
                    {"id": t["id"],
                     "q": persistence_row(rates[i] if i < len(rates) else 0.0, 0.0)}
                    for i, t in enumerate(trophies)]}

        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
