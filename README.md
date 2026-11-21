# Wide-Spread Rarity Forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/jusst01/rarity-forecast-gta6-tidy/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/jusst01/rarity-forecast-gta6-tidy/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

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
two-component logit-normal mixture — weight **0.95 at sigma 0.10** and weight **0.05 at sigma
0.15** — whose quantiles are found by 60 rounds of bisection on [-16, 16] over the normal CDF.

**Live branch.** Players are fitted with a Theil-Sen line of `ln(max(N,10))` against `x`, and
the target count is the median of two anchored extrapolations (factors 0.85 and 1.0) with
**slope damping 0.6**. Earners get their own Theil-Sen fit per trophy, then shrink toward the
median slope of their `(tier, rate-band)` group with weight `n/(n+6)` (bands `[0,1)`, `[1,10)`,
`[10,50)`, `[50,100)`; the global median stands in for a missing group). The trend median is
`ln(max(E_last,0.5)) + 0.6*b_shrunk*(x_tgt - x_last)` — the same **0.6** earner-slope damping —
capped at `0.999*N_tgt`. Trophies already at 99.99% take the pinned row before any fit, so they
contribute no slope to a group median.

**Persistence and spread.** The median blends the trend with the last read using
`lambda = min(0.97, 0.80 + 0.35*exp(-horizon_days/2))` (about 0.84 at the 4.5-day horizon; the
0.97 clamp keeps `lambda*(1-lambda)` positive). Median movement is capped to
**[-1.9, +1.4]** logit, tighter than sibling entries. The spread is

```
sigma = 1.2 * sqrt(0.10^2 + 0.16^2*h + 0.25*rsd^2 + lambda*(1-lambda)*disagreement^2)
```

with the residual spread clamped to **[0.12, 0.6]**, the disagreement term guarded non-negative,
and the **1.2 multiplier applied last, outside the square root**. When `E_last < 1`, `sigma` is
raised to at least **1.2 after** that multiplier is applied. Quantiles are `inv_logit(m + z*sigma)`
for the five standard-normal z values, sorted and clamped to [0, 100].

Trophies at or above 99.99% get exactly `[99.0, 99.9, 100.0, 100.0, 100.0]` in both branches.
Every failure path — case, branch, or single row — falls back to a persistence forecast: the
last rate carried forward unchanged, with logit spread `max(0.35, 0.5*sqrt(horizon_days))`.
Finished rows are validated (ids match the case, five finite monotone quantiles in [0, 100])
before they are returned, and a malformed case file is decoded trophy by trophy and read by
read so it still yields a persistence output whenever any trophy is readable. The program never
raises and never emits an invalid row.

## Run

```bash
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
python3 tools/edge_check.py        # self-check: practice + degenerate cases
```
