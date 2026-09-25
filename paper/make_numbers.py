#!/usr/bin/env python3
"""Turn results/summary.json (+ results/leakage.json) into LaTeX macros
(numbers.tex) and the CSL table (table_csl.tex), so every number in the
manuscript is generated, not typed."""

import json
import os
import sys

import numpy as np

root = os.path.dirname(os.path.abspath(__file__))
res_dir = os.path.join(root, "..", "results")
summary = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(res_dir, "summary.json")))
leak = json.load(open(os.path.join(res_dir, "leakage.json")))
M = {}


def f2(x):
    return f"{x:.2f}"


def pct(x):
    return f"{100 * x:.1f}"


# ---- delay realisation ------------------------------------------------------
d = summary["delay"]
below = [r for r in d if r["S"] < r["k"]]
excess = [r["rel_l2_error"] ** 2 - (1 - r["S"] / r["k"]) for r in below]
assert min(excess) >= -1e-6, "a delay fit violates the Theorem 1 bound"
M["delayBoundN"] = str(len(below))
M["delayExcess"] = f2(np.median(excess))
over = []
for k in sorted({r["k"] for r in d}):
    rr = sorted((r for r in d if r["k"] == k), key=lambda r: r["S"])
    s10 = next((r["S"] for r in rr if r["rel_l2_error"] <= 0.105), None)
    if s10:
        over.append(s10 / k)
M["delayOverhead"] = f"{np.median(over):.1f}"

# ---- token-level ------------------------------------------------------------
ps = summary["phase_stats"]
M["nSeeds"] = str(ps["n_seeds"])
M["aucAtKmin"], M["aucAtKmax"] = f2(ps["lti_auc_at_k"][0]), f2(ps["lti_auc_at_k"][1])
for tag, thr in zip("abcd", ("0.8", "0.9", "0.95", "0.99")):
    M[f"ratioLTI{tag}"] = f"{ps['threshold_ratios'][f'lti@{thr}']['median']:.1f}"
    M[f"ratioSel{tag}"] = f"{ps['threshold_ratios'][f'selective@{thr}']['median']:.1f}"
M["belowN"] = str(ps["lti_below_cells"])
M["belowUnderLTI"] = str(ps["lti_below_cells_under_surrogate"])
M["belowUnderSel"] = str(ps["selective_below_cells_under_surrogate"])
M["leakLo"], M["leakHi"] = f2(leak["min"]), f2(leak["max"])
M["pairLeLo"], M["pairLeHi"] = f2(leak["pairwise_le_km3"][0]), f2(leak["pairwise_le_km3"][1])
M["pairKmTwoHi"] = f2(leak["pairwise_km2"][1])
M["pairKmOneHi"] = f2(leak["pairwise_km1"][1])
M["pairKHi"] = f2(leak["pairwise_k"][1])
M["selDiff"] = f"${ps['sel_minus_lti_mean']:+.3f}$"
M["selCI"] = f"{ps['sel_minus_lti_ci95']:.3f}"
p = ps["sel_minus_lti_wilcoxon_p"]
M["selP"] = f"{p:.2f}" if p >= 0.01 else (f"{p:.3f}" if p >= 0.001 else "<0.001")
M["selPt"] = f"{ps['sel_minus_lti_ttest_p']:.2f}"
M["selMaxAbs"] = f2(max(abs(x) for x in ps["sel_minus_lti_range"]))
M["nReachLTId"] = str(ps["threshold_ratios"]["lti@0.99"]["n"])
M["dtCV"] = f"{100 * ps['dt_cv_median']:.0f}"


def mean_over_k(tab, rel):
    """Mean seed-averaged AUROC over k in {4, 8, 12, 16} at S = rel(k)."""
    v = [tab[f"k{k}_S{rel(k)}"][0] for k in (4, 8, 12, 16) if f"k{k}_S{rel(k)}" in tab]
    return f2(np.mean(v))


M["shiftAtK"] = mean_over_k(ps["shift"], lambda k: k)
M["shiftAtKp"] = mean_over_k(ps["shift"], lambda k: k + 1)
M["frozenAtK"] = mean_over_k(ps["frozen"], lambda k: k)
M["frozenAtTwoK"] = mean_over_k(ps["frozen"], lambda k: 2 * k)
M["ltiAtK"] = mean_over_k(ps["lti_cells"], lambda k: k)
M["ltiAtTwoK"] = mean_over_k(ps["lti_cells"], lambda k: 2 * k)
M["ltiEightThirtyTwo"] = f2(ps["lti_cells"]["k8_S32"][0])
M["nocmpEightThirtyTwo"] = f2(ps["lti_nocmp"]["k8_S32"][0])
M["nocmpMaxDiff"] = f2(max(abs(ps["lti_nocmp"][c][0] - ps["lti_cells"][c][0]) for c in ps["lti_nocmp"]))
fz = [ps["frozen"][f"k{k}_S{k}"][0] for k in (4, 8, 12)]
M["frozenKlo"], M["frozenKhi"] = f2(min(fz)), f2(max(fz))
M["frozenSixteenTwoK"] = f2(ps["frozen"]["k16_S32"][0])
M["frozenSixteen"], M["frozenSixteenCI"] = f2(ps["frozen"]["k16_S16"][0]), f2(ps["frozen"]["k16_S16"][1])
sk = [ps["shift"][f"k{k}_S{k}"][0] for k in (4, 8, 12, 16)]
M["shiftAtKlo"], M["shiftAtKhi"] = f2(min(sk)), f2(max(sk))

# ---- CSL ----------------------------------------------------------------------
cs = summary["csl_stats"]
acc = cs["acc"]
Smax = max(int(k.split("_S")[1]) for k in acc if k.startswith("lti_S"))
M["cslSmax"] = str(Smax)
M["cslLTIatTwo"] = pct(acc["lti_S2"][0])
M["cslOrBest"] = pct(max(v[0] for k, v in acc.items() if k.startswith("oracle_")))
M["cslLTIBest"], M["cslLTIBestCI"] = pct(acc[f"lti_S{Smax}"][0]), pct(acc[f"lti_S{Smax}"][1])
M["cslSelBest"], M["cslSelBestCI"] = pct(acc[f"selective_S{Smax}"][0]), pct(acc[f"selective_S{Smax}"][1])
aux = {int(k.split("_S")[1]): v for k, v in acc.items() if k.startswith("lti_aux_")}
if aux:
    Sa = max(aux, key=lambda S: aux[S][0])
    M["cslAuxBest"], M["cslAuxBestCI"], M["cslAuxBestS"] = pct(aux[Sa][0]), pct(aux[Sa][1]), str(Sa)
    M["cslLTIatAuxS"] = pct(acc[f"lti_S{Sa}"][0])
M["cslSelMinusLTI"] = f"${100 * cs['sel_minus_lti_mean']:+.1f}$"
M["cslSelMinusLTICI"] = f"{100 * cs['sel_minus_lti_ci95']:.1f}"
if "acc_eval256_minus_eval128_mean" in cs:
    M["cslEvalLenGain"] = f"{100 * cs['acc_eval256_minus_eval128_mean']:.1f}"
sh = {int(k.split("_S")[1]): v for k, v in acc.items() if k.startswith("shift_")}
if sh:
    Sb = max(sh, key=lambda S: sh[S][0])
    M["cslShiftBest"], M["cslShiftBestCI"], M["cslShiftBestS"] = pct(sh[Sb][0]), pct(sh[Sb][1]), str(Sb)
    M["cslShiftAtNine"] = pct(sh[9][0]) if 9 in sh else "--"
    M["cslShiftAtFour"] = pct(sh[4][0]) if 4 in sh else "--"
    M["cslOrAtEight"] = pct(acc["oracle_S8"][0])
pool = cs["pool_class_r2"]
M["poolTwoOne"], M["poolTwoSixteen"] = pct(pool["S2_w1"]), pct(pool["S2_w16"])

W = summary["csl_horizon_W"]
M["cslWmin"], M["cslWmax"] = str(min(W.values())), str(max(W.values()))
ons = summary["csl_onsets"]
S_grid = sorted({int(k.split("_S")[1]) for k in acc if k.startswith("oracle_")})
grid_up = lambda w: min((S for S in S_grid if S >= w), default=None)
M["cslGridOr"] = str(sum(o["onset_S"] == grid_up(o["horizon"]) for o in ons if o["model"] == "oracle"))

with open(os.path.join(root, "numbers.tex"), "w") as fh:
    fh.write("% auto-generated by make_numbers.py -- do not edit\n")
    for k, v in M.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# ---- Table: horizons and onsets ----------------------------------------------
by = {(o["model"], o["skip"]): o["onset_S"] for o in ons}
fmt = lambda v: "---" if v is None else str(v)
skips = sorted({o["skip"] for o in ons})
rows = [
    r"$r$ & " + " & ".join(map(str, skips)) + r" \\ \midrule",
    r"$W^*(r)$ & " + " & ".join(str(W[str(s)]) for s in skips) + r" \\",
    r"Exact counts & " + " & ".join(fmt(by[("oracle", s)]) for s in skips) + r" \\",
    r"LTI & " + " & ".join(fmt(by[("lti", s)]) for s in skips) + r" \\",
    r"Input-dep.\ $\Delta_t$ & " + " & ".join(fmt(by[("selective", s)]) for s in skips) + r" \\",
    r"Shift reg.\ & " + " & ".join(fmt(by.get(("shift", s))) for s in skips) + r" \\",
]
with open(os.path.join(root, "table_csl.tex"), "w") as fh:
    fh.write("% auto-generated by make_numbers.py -- do not edit\n"
             "\\begin{tabular}{lcccccccccc}\n\\toprule\n" + "\n".join(rows)
             + "\n\\bottomrule\n\\end{tabular}\n")

for k, v in M.items():
    print(f"{k:24s} {v}")
