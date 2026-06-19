#!/usr/bin/env python3
"""Locate the B200 tp2x4 e2e-p90 SLO crossover (9,500 ms) from measured sweep points,
and summarize DCGM utilization per concurrency to settle compute-vs-bandwidth bound.

Usage: knee-analysis.py --results-dir DIR --dcgm-dir DIR
  results-dir: raw bench JSONs (fin-support_b200-tp2x4-*_cNNN_*.json)
  dcgm-dir:    per-concurrency DCGM csv (dcgm_cNNN.csv) from dcgm-sampler.sh
"""
import argparse, glob, json, os, re, statistics

GATE_P90 = 9500.0
GATE_P50 = 6500.0

def load_points(results_dir):
    pts = []
    for f in glob.glob(os.path.join(results_dir, "*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        c = d.get("concurrency")
        e2e = d.get("e2e_ms", {})
        if c is None or "p90" not in e2e:
            continue
        # total tok/s: prefer explicit; else derive
        pts.append({
            "conc": c,
            "p50": e2e.get("p50"),
            "p90": e2e.get("p90"),
            "tps": d.get("total_toks_per_s") or d.get("throughput_total_toks_per_s"),
            "err": d.get("error_rate", 0.0),
            "file": os.path.basename(f),
        })
    pts.sort(key=lambda p: p["conc"])
    return pts

def interp_crossover(pts, key, gate):
    """Piecewise-linear crossover where `key` first exceeds `gate`."""
    below = [p for p in pts if p[key] is not None and p[key] <= gate]
    above = [p for p in pts if p[key] is not None and p[key] > gate]
    if not below or not above:
        return None
    lo = max(below, key=lambda p: p["conc"])
    hi = min((p for p in above if p["conc"] > lo["conc"]), key=lambda p: p["conc"], default=None)
    if hi is None:
        return None
    frac = (gate - lo[key]) / (hi[key] - lo[key])
    return lo["conc"] + frac * (hi["conc"] - lo["conc"]), lo, hi

def summarize_dcgm(path):
    if not os.path.exists(path):
        return None
    rows = [l.strip().split(",") for l in open(path) if l.strip() and not l.startswith("ts")]
    def col(i):
        vals = []
        for r in rows:
            try:
                vals.append(float(r[i]))
            except (ValueError, IndexError):
                pass
        return vals
    # cols: ts,gpu,gr_engine,dram,tensor,power,fb
    gr, dr, te, pw = col(2), col(3), col(4), col(5)
    # drop near-idle samples (gr<0.05) so we measure under-load utilization
    busy = [i for i in range(len(rows)) if len(rows[i]) > 2 and _f(rows[i][2]) > 0.05]
    def busy_mean(c):
        vals = [_f(rows[i][c]) for i in busy if _f(rows[i][c]) is not None]
        return statistics.mean(vals) if vals else None
    return {
        "samples": len(rows), "busy_samples": len(busy),
        "gr_engine_mean": busy_mean(2), "dram_mean": busy_mean(3),
        "tensor_mean": busy_mean(4), "power_mean": busy_mean(5),
    }

def _f(x):
    try: return float(x)
    except: return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--dcgm-dir", default=None)
    args = ap.parse_args()

    pts = load_points(args.results_dir)
    print("=== B200 tp2x4 measured points ===")
    print(f"{'conc':>5} {'p50':>8} {'p90':>8} {'tok/s':>9} {'err':>6}")
    for p in pts:
        print(f"{p['conc']:>5} {p['p50'] or 0:>8.0f} {p['p90'] or 0:>8.0f} {(p['tps'] or 0):>9.0f} {p['err']:>6.4f}")

    print("\n=== SLO crossover (piecewise-linear interpolation) ===")
    for key, gate, label in [("p90", GATE_P90, "e2e p90"), ("p50", GATE_P50, "e2e p50")]:
        r = interp_crossover(pts, key, gate)
        if r:
            c, lo, hi = r
            print(f"{label} crosses {gate:.0f} ms at ~c{c:.0f}  (between c{lo['conc']} {lo[key]:.0f}ms and c{hi['conc']} {hi[key]:.0f}ms)")
        else:
            print(f"{label}: no bracketing pair (need points on both sides of {gate:.0f} ms)")

    if args.dcgm_dir:
        print("\n=== DCGM utilization per concurrency (busy-sample mean) ===")
        print(f"{'conc':>5} {'GR_ENG':>8} {'DRAM':>8} {'TENSOR':>8} {'power_W':>8}  bound-hint")
        for f in sorted(glob.glob(os.path.join(args.dcgm_dir, "dcgm_c*.csv"))):
            c = re.search(r"dcgm_c(\d+)", f)
            s = summarize_dcgm(f)
            if not s or s["gr_engine_mean"] is None:
                continue
            dram, tens = s["dram_mean"], s["tensor_mean"]
            hint = "BW-bound" if (dram and tens and dram > tens and dram > 0.5) else \
                   "compute-bound" if (tens and dram and tens >= dram) else "mixed"
            print(f"{c.group(1):>5} {s['gr_engine_mean']:>8.3f} {dram:>8.3f} {tens:>8.3f} {s['power_mean'] or 0:>8.0f}  {hint}")

if __name__ == "__main__":
    main()
