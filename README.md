# Wide-Spread Rarity Forecaster

[![GTA VI Trophy Rarity Forecast](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![Rank](https://make.fun/badge/jusst01/rarity-forecast-gta6-tidy/rank.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![Reward](https://make.fun/badge/jusst01/rarity-forecast-gta6-tidy/reward.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)

[Competition Page](https://make.fun/c/gta6-trophy-rarity-forecast-01)

## Model Overview

A pure-standard-library Python forecaster for GTA VI trophy earn rates, one week out.

**Eventizing.** The refresh history is collapsed into distinct states, oldest first: a new
event begins whenever the players-tracked count or the raw rate tuple changes (a null rate is
its own marker but reads as 0.0). Each event carries the hours since the first read and the
player count. Every fit runs against `x = ln(1 + hours since first read)`, a saturating time
axis, so later refreshes contribute ever less leverage.

**Flat branch.** When there are fewer than two events, or every event shares one player count,
the disclosed launch refresh is modelled as pure dilution: earners are integers, held fixed
across the move from 638 to 3,868 tracked players (`G = 3868/638`). The diluted rate
`100*E/(N_last*G)` (or `0.02` when `E = 0`) and the untouched `max(last_rate, 0.05)` form a
two-component logit-normal mixture — weight 0.95 at sigma 0.10 and weight 0.05 at sigma 0.15 —
whose quantiles are found by 60 rounds of bisection on [-16, 16] over the normal CDF.

**Live branch.** Players are fitted with a Theil-Sen line of `ln(max(N,10))` against `x`, and
the target count is the median of two anchored extrapolations (factors 0.85 and 1.0) with
**slope damping 0.6**. Earners get their own Theil-Sen fit per trophy, then shrink toward the
median slope of their (tier, rate-band) group with weight `n/(n+6)`. The trend median is
`ln(max(E_last,0.5)) + 0.6*b_shrunk*(x_tgt - x_last)`, capped at `0.999*N_tgt`.

**Persistence and spread.** The median blends the trend with the last read using
`lambda = min(0.97, 0.80 + 0.35*exp(-horizon_days/2))` (about 0.84 at the 4.5-day horizon; the
0.97 clamp keeps `lambda*(1-lambda)` positive). Median movement is capped to
**[-1.9, +1.4]** logit, tighter than sibling entries. The spread is
`1.2 * sqrt(0.10^2 + 0.16^2*h + 0.25*rsd^2 + lambda*(1-lambda)*disagreement^2)` with the
residual spread clamped to [0.12, 0.6], a floor of 1.2 when fewer than one earner is implied,
and the whole spread **widened by 1.2** — trading sharpness for interval honesty.

Trophies at or above 99.99% get the pinned row `[99.0, 99.9, 100.0, 100.0, 100.0]`. Every
failure path — case, branch, row — falls back to a persistence forecast with spread
`max(0.35, 0.5*sqrt(h))`, and finished rows are validated before they are returned, so the
program never raises and never emits an invalid row.

## Run

```bash
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
