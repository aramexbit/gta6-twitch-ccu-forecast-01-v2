#!/usr/bin/env python3
"""Forecast GTA VI's launch-day Twitch viewers hour by hour (make.fun v2).

Method (fitted offline on the kit's comparables, validated by leave-one-out):
every past launch gives a release-day curve of hourly viewers. For a target
launch we only know its hour-1 print, so for each hour h = 2..24 we predict
the log-ratio ln(viewers_h / viewers_1) with a LOCAL-LINEAR regression on
ln(viewers_1): a Gaussian kernel (bandwidth 1.2 in log space) weights past
launches by how close their hour-1 print is, then a weighted least-squares
line in ln(viewers_1) is fitted per hour. Big launches borrow their shape
from other big launches; small launches borrow from small ones -- the shape
of a release day depends strongly on how much of the world was already
unlocked at 00:00 UTC, which the hour-1 print encodes.

The five quantiles come from the predicted median plus a spread: the spread
for hour h is the standard deviation of the other launches' leave-one-out
residuals at that hour, taken from launches whose hour-1 print is near the
target's (within 1.2 log units), smoothed over neighbouring hours and
scaled by 0.9. No channels, no pre-release ramp and no early-access filter:
all three were tested in the bake-off and hurt the big-launch regime that
GTA VI will be in.

Pure Python 3 standard library, deterministic, no network, no model calls.
Reads every *.in case in $MAKEFUN_INPUT_DIR (else /input) and writes one
*.out per case to $MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
from datetime import datetime, timezone

Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)

BANDWIDTH = 1.2        # Gaussian kernel width on ln(hour-1 viewers)
SIGMA_SCALE = 0.9      # shrink on the residual spread (pinball-optimal < 1)
SIGMA_WINDOW = 1.2     # local residual window, in ln(hour-1) units
SIGMA_MIN = 0.10
SIGMA_MAX = 1.2
EVAL_HEADROOM = 0.5    # how far past the pool's ln(hour-1) range we extrapolate
MIN_POOL = 8           # curves needed per hour before we trust the fit
MIN_DAY_POINTS = 15    # of hours 2..24 a comparable must have to join the pool


def _parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _hour_index(day_start, when):
    """Hour 1..24 of the release day that `when` falls in, else None."""
    delta = (when - day_start).total_seconds()
    if 0 <= delta < 24 * 3600:
        return int(delta // 3600) + 1
    return None


def extract_curves(case):
    """Release-day curves of the case's comparables.

    Returns a list of dicts: {'lnv1': float, 'day': {hour: viewers}}.
    A curve joins only if hour 1 exists and at least MIN_DAY_POINTS of
    hours 2..24 are present with positive viewers.
    """
    curves = []
    for comp in case.get("comparables") or []:
        try:
            day_start = _parse_time(comp["day_start"])
        except (KeyError, TypeError, ValueError):
            continue
        day = {}
        for point in comp.get("hourly") or []:
            try:
                hour = _hour_index(day_start, _parse_time(point["from"]))
                viewers = float(point["viewers"])
            except (KeyError, TypeError, ValueError):
                continue
            if hour is not None and viewers > 0:
                day[hour] = viewers
        if 1 not in day:
            continue
        if sum(1 for h in range(2, 25) if h in day) < MIN_DAY_POINTS:
            continue
        curves.append({"lnv1": math.log(day[1]), "day": day})
    return curves


def _wls_line(xs, ys, ws):
    """Weighted least squares for y ~ a + b*x. Returns (a, b) or None."""
    sw = sum(ws)
    if sw <= 0:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    ridge = 1e-9 * (sxx + 1.0)
    if sxx + ridge <= 0:
        return None
    b = sxy / (sxx + ridge)
    return (my - b * mx, b)


def _weighted_median(pairs):
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    if total <= 0:
        return 0.0
    acc = 0.0
    for value, weight in pairs:
        acc += weight
        if acc >= total / 2.0:
            return value
    return pairs[-1][0]


def local_linear_median(lnv1_target, pool, hour):
    """Predicted median log-ratio ln(viewers_h / viewers_1) for one hour.

    Local-linear (kernel-weighted) regression on ln(hour-1 viewers).
    Falls back to the kernel-weighted median when the fit is degenerate.
    """
    xs, ys, ws = [], [], []
    for curve in pool:
        day = curve["day"]
        if hour not in day:
            continue
        xs.append(curve["lnv1"])
        ys.append(math.log(day[hour] / day[1]))
        ws.append(math.exp(-0.5 * ((curve["lnv1"] - lnv1_target) / BANDWIDTH) ** 2))
    if len(xs) < MIN_POOL:
        return 0.0
    lo = min(xs) - EVAL_HEADROOM
    hi = max(xs) + EVAL_HEADROOM
    x_eval = min(max(lnv1_target, lo), hi)
    fit = _wls_line(xs, ys, ws)
    if fit is None:
        return _weighted_median(list(zip(ys, ws)))
    a, b = fit
    return a + b * x_eval


def _pstdev(values):
    n = len(values)
    if n < 2:
        return 0.3
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / n)


def forecast(case):
    """Return the {"hours": [...]} forecast object for one case."""
    pool = extract_curves(case)
    known = (case.get("known") or [{}])[0]
    try:
        hour1 = max(float(known["viewers"]), 1.0)
    except (KeyError, TypeError, ValueError):
        hour1 = 1.0
    lnv1 = math.log(hour1)

    medians = {}
    for hour in range(2, 25):
        medians[hour] = local_linear_median(lnv1, pool, hour)

    # Pass 1: leave-one-out residuals of every pool curve, per hour.
    residuals = {hour: [] for hour in range(2, 25)}
    residual_lnv1 = {hour: [] for hour in range(2, 25)}
    for i, curve in enumerate(pool):
        rest = pool[:i] + pool[i + 1:]
        for hour in range(2, 25):
            day = curve["day"]
            if hour not in day:
                continue
            pred = local_linear_median(curve["lnv1"], rest, hour)
            residuals[hour].append(math.log(day[hour] / day[1]) - pred)
            residual_lnv1[hour].append(curve["lnv1"])

    # Pass 2: spread per hour from residuals near the target's hour-1 level.
    sigmas = {}
    for hour in range(2, 25):
        if len(pool) < MIN_POOL:
            sigmas[hour] = 0.15 * math.sqrt(hour - 1)
            continue
        local = [r for r, lv in zip(residuals[hour], residual_lnv1[hour])
                 if abs(lv - lnv1) <= SIGMA_WINDOW]
        if len(local) < 5:
            local = residuals[hour]
        raw = _pstdev(local) if local else 0.3
        sigmas[hour] = raw
    smoothed = {}
    for hour in range(2, 25):
        neighbours = [sigmas[h] for h in (hour - 1, hour, hour + 1)
                      if 2 <= h <= 24]
        smoothed[hour] = sum(neighbours) / len(neighbours)

    rows = []
    for hour in range(2, 25):
        if len(pool) < MIN_POOL:
            median = 0.0
            sigma = 0.15 * math.sqrt(hour - 1)
        else:
            median = medians[hour]
            sigma = min(max(smoothed[hour] * SIGMA_SCALE, SIGMA_MIN), SIGMA_MAX)
        quantiles = []
        for z in Z:
            q = hour1 * math.exp(median + z * sigma)
            quantiles.append(min(max(q, 1.0), 1e9))
        for i in range(1, len(quantiles)):
            quantiles[i] = max(quantiles[i], quantiles[i - 1])
        rows.append({"hour": hour, "q": quantiles})
    return {"hours": rows}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(input_dir) if n.endswith(".in"))
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(input_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        result = forecast(case)
        out_path = os.path.join(output_dir, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f)


if __name__ == "__main__":
    main()
