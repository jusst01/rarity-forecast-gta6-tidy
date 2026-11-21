#!/usr/bin/env python3
"""Self-check for main.py: practice cases plus mutated edge cases.

Loads main.py, runs forecast_case (and, for the malformed files, main() end to
end) over the kit's practice cases and a battery of degenerate inputs, then
asserts every output row is valid: correct ids in order, five finite
non-decreasing quantiles in [0, 100], and a whole-file size under the 65,536
byte cap.  Prints one line per case and exits non-zero on any failure.

    python3 tools/edge_check.py [--kit ../kit/input] [--main main.py]
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

SIZE_CAP = 65536  # the I/O contract's per-output ceiling, in bytes


# --------------------------------------------------------------------------- #
# Output validation
# --------------------------------------------------------------------------- #

def validate(forecast, expected_ids, label):
    """Return a list of problems with a forecast dict (empty when it is valid)."""
    problems = []
    if not isinstance(forecast, dict):
        return ["%s: not a dict" % label]
    rows = forecast.get("trophies")
    if not isinstance(rows, list):
        return ["%s: 'trophies' is not a list" % label]
    if len(rows) != len(expected_ids):
        problems.append("%s: %d rows, expected %d" % (label, len(rows), len(expected_ids)))
    got_ids = [r.get("id") if isinstance(r, dict) else None for r in rows]
    if got_ids != list(expected_ids):
        problems.append("%s: ids %r != %r" % (label, got_ids[:8], list(expected_ids)[:8]))
    if len(set(map(repr, got_ids))) != len(got_ids):
        problems.append("%s: duplicate ids" % label)
    for row in rows:
        if not isinstance(row, dict):
            problems.append("%s: a row is not a dict" % label)
            continue
        if set(row) != {"id", "q"}:
            problems.append("%s: id %r has keys %r" % (label, row.get("id"), sorted(row)))
        q = row.get("q")
        if not isinstance(q, list) or len(q) != 5:
            problems.append("%s: id %r q is not five values" % (label, row.get("id")))
            continue
        prev = -math.inf
        for v in q:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                problems.append("%s: id %r has non-number %r" % (label, row.get("id"), v))
                break
            if not math.isfinite(float(v)):
                problems.append("%s: id %r has non-finite %r" % (label, row.get("id"), v))
                break
            if not (0.0 <= float(v) <= 100.0):
                problems.append("%s: id %r out of [0,100]: %r" % (label, row.get("id"), v))
                break
            if float(v) < prev:
                problems.append("%s: id %r not monotone: %r" % (label, row.get("id"), q))
                break
            prev = float(v)
    encoded = json.dumps(forecast, separators=(",", ":")).encode("utf-8")
    if len(encoded) > SIZE_CAP:
        problems.append("%s: %d bytes over the %d cap" % (label, len(encoded), SIZE_CAP))
    return problems


def ids_of(case):
    return [t["id"] for t in case.get("trophies", [])
            if isinstance(t, dict) and "id" in t]


# --------------------------------------------------------------------------- #
# Mutations of a well-formed case
# --------------------------------------------------------------------------- #

def with_history(base, history):
    case = json.loads(json.dumps(base))
    case["history"] = history
    return case


def mutated_cases(base):
    """(label, case_dict_or_None, raw_text_or_None) triples."""
    hist = base.get("history", [])
    last = hist[-1] if hist else {"rates": []}
    n_rates = len(last.get("rates", []))
    return [
        ("kit:empty_history", with_history(base, []), None),
        ("kit:single_read", with_history(base, [last]), None),
        ("kit:all_null_rates",
         with_history(base, [dict(h, rates=[None] * len(h.get("rates", [])))
                             for h in hist]), None),
        ("kit:all_zero_rates",
         with_history(base, [dict(h, rates=[0.0] * len(h.get("rates", [])))
                             for h in hist]), None),
        ("kit:one_player_count",
         with_history(base, [dict(h, players_tracked=638) for h in hist]), None),
        ("kit:two_events_rising",
         with_history(base, [
             dict(last, players_tracked=600, rates=[1.0] * n_rates),
             dict(last, players_tracked=1200, rates=[2.0] * n_rates)]), None),
        ("kit:nulls_and_growth",
         with_history(base, [
             dict(h, players_tracked=600 + 300 * k,
                  rates=[None if i % 3 == 0 else (i % 7) * 1.5
                         for i in range(len(h.get("rates", [])))])
             for k, h in enumerate(hist)]), None),
    ]


def malformed_cases(p00_text):
    """Raw texts that are not valid JSON but still expose trophies and history."""
    hist_at = p00_text.find('],"history"')
    cut = (hist_at + 2 + 400) if hist_at >= 0 else len(p00_text) // 2
    return [
        ("file:truncated", p00_text[:cut]),
        ("file:truncated_mid_trophies",
         p00_text[:p00_text.find('"name":"Vice City')]),
        ("file:garbage_after_prefix",
         (p00_text[:hist_at + 2] if hist_at >= 0 else p00_text[:200])
         + ', "history": [{"slot": "x", "at": '),
        ("file:empty", ""),
    ]


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

def run_main(main_path, case_text, workdir):
    """Run main.py as the grader does; return (rc, stderr, parsed p00.out or None)."""
    indir = Path(workdir) / "in"
    outdir = Path(workdir) / "out"
    shutil.rmtree(indir, ignore_errors=True)
    shutil.rmtree(outdir, ignore_errors=True)
    indir.mkdir(parents=True)
    outdir.mkdir(parents=True)
    (indir / "p00.in").write_text(case_text, encoding="utf-8")
    env = dict(os.environ, MAKEFUN_INPUT_DIR=str(indir), MAKEFUN_OUTPUT_DIR=str(outdir))
    proc = subprocess.run([sys.executable, "-P", "-s", str(main_path)],
                          env=env, capture_output=True, text=True)
    out_path = outdir / "p00.out"
    parsed = None
    if out_path.exists():
        try:
            parsed = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            parsed = None
    return proc.returncode, proc.stderr, parsed


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kit", default=str(here.parent.parent / "kit" / "input"),
                    help="directory holding p00.in / p01.in")
    ap.add_argument("--main", default=str(here.parent / "main.py"),
                    help="path to main.py")
    args = ap.parse_args()

    main_path = Path(args.main).resolve()
    kit = Path(args.kit).resolve()
    sys.path.insert(0, str(main_path.parent))
    sys.path.insert(0, str(main_path.parent / "tools"))
    import main as model  # noqa: E402  (path set above)

    failures = []
    checked = 0

    def report(label, problems):
        nonlocal checked
        checked += 1
        if problems:
            failures.extend(problems)
            print("FAIL %-28s %s" % (label, problems[0]))
            for extra in problems[1:4]:
                print("       %s" % extra)
        else:
            print("ok   %-28s" % label)

    # 1. Kit practice cases, through forecast_case.
    practice = sorted(kit.glob("*.in")) if kit.is_dir() else []
    if not practice:
        failures.append("no practice cases found under %s" % kit)
        print("FAIL no practice cases under %s" % kit)
    for path in practice:
        case = json.loads(path.read_text(encoding="utf-8"))
        label = "practice:" + path.stem
        try:
            forecast = model.forecast_case(case, path.name)
        except Exception:
            report(label, ["%s raised: %s" % (label, traceback.format_exc(limit=2))])
            continue
        report(label, validate(forecast, ids_of(case), label))

    # 2. Mutated edge cases, through forecast_case.
    if practice:
        base = json.loads(practice[0].read_text(encoding="utf-8"))
        p00_text = practice[0].read_text(encoding="utf-8")
    else:
        base, p00_text = {"trophies": [], "history": []}, ""

    for label, case, _raw in mutated_cases(base):
        try:
            forecast = model.forecast_case(case, label)
        except Exception:
            report(label, ["%s raised: %s" % (label, traceback.format_exc(limit=2))])
            continue
        report(label, validate(forecast, ids_of(case), label))

    # 3. Malformed files, end to end through main(): no traceback, valid rows.
    with tempfile.TemporaryDirectory() as tmp:
        for label, text in malformed_cases(p00_text):
            rc, err, parsed = run_main(main_path, text, tmp)
            problems = []
            if rc != 0:
                problems.append("%s: exit code %d" % (label, rc))
            if "Traceback" in err:
                problems.append("%s: traceback in stderr" % label)
            if parsed is None:
                problems.append("%s: no parseable p00.out" % label)
            else:
                # A malformed file yields rows only for the trophies that could
                # still be read; the spec allows an empty set when none can.
                salvaged = model.salvage_trophies(text) if text else None
                expected = [t["id"] for t in salvaged] if salvaged else []
                problems.extend(validate(parsed, expected, label))
            report(label, problems)

    # 4. Determinism: two runs of the practice case must be byte-identical.
    if practice:
        with tempfile.TemporaryDirectory() as tmp:
            digests = []
            for _ in range(2):
                rc, err, _parsed = run_main(
                    main_path, p00_text, tmp)
                out = Path(tmp) / "out" / "p00.out"
                digests.append(out.read_bytes() if out.exists() else b"")
            if digests[0] != digests[1] or not digests[0]:
                report("determinism", ["determinism: two runs differ"])
            else:
                report("determinism", [])

    print("\n%d checks, %d failure(s)" % (checked, len(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
