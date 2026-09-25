# Limits on delay detection by low-order state-space models

`walk_ssm_barrier.py` tests how much state an SSM needs to detect that the
current token repeats the one `k` steps earlier, the primitive behind cycle
detection by random-walk graph learners. Theory: for Gaussian inputs, exact
detection needs `S >= k+1` and a singular (FIR) state matrix, and is
impossible for invertible dynamics at any `S`; for `S < k` any linear probe
explains at most `S/k` of the lagged token's variance (Hankel rank), which
bounds every readout for Gaussian inputs. The code uses plain PyTorch and
runs on CPU.

## Install & run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python walk_ssm_barrier.py --quick --workers 8     # smoke test, a few minutes
python walk_ssm_barrier.py --workers 8             # full run (resumable)
python walk_ssm_barrier.py --plots-only            # redraw figures from results/
```

Each training run happens in its own process with 1 torch thread
(`--workers N --threads 1`). Finished runs are cached under
`results/runs/`, so an interrupted job picks up where it stopped. If
other jobs are using the CPU, reduce `--workers`.

## Experiments

| id | what | figure |
|----|------|--------|
| `delay` | gradient fit of an `S`-state diagonal LTI kernel to the delay `z^-k`, against the bound `sqrt(1-S/k)` | `figures/fig1_delay_realization.pdf` |
| `phase` | token-level lag-`k` revisit detection on non-backtracking walks over random 4-regular graphs, `(S, k)` grid, LTI vs input-dependent step, 3 seeds | `figures/fig2_phase_diagram.pdf` (double column) |
| `phase` controls | frozen fitted poles, nilpotent shift register, no comparison features | `figures/fig3_auroc_vs_ratio.pdf` |
| `csl` | 10-class CSL accuracy vs `S`: LTI, input-dependent step, per-lag revisit loss, trained shift register, exact-count reference; pooled-walk curves | `figures/fig4_csl_accuracy.pdf`, Table I |

`leakage_check.py` tests whether short-lag walk structure predicts lag-`k`
revisits. Pairwise coincidences among the `S` most recent tokens do not for
`S <= k-3` (AUROC about 0.5), so the i.i.d. Gaussian surrogate of Corollary 1
is fair there; within two lags of `k` they reach AUROC 0.53-0.67. It writes
`results/leakage.json`.

The figures are sized for IEEE layouts: 3.5 in for a single column,
7.16 in for the full text width. Text is set in 8 pt serif with
embedded TrueType fonts, and the PDFs are saved at 600 dpi with tight
bounding boxes.

## Results files

* `results/summary.json`: all metrics, horizons and onsets
* `results/*.csv`
* `figures/*.pdf`: the four IEEE-formatted figures

## Manuscript (IEEE Signal Processing Letters)

`paper/main.tex` (IEEEtran `journal`), `paper/refs.bib`. All numbers and
Table I come from `results/summary.json` via `paper/make_numbers.py`, and
figures are read from `../figures/`.

```bash
python walk_ssm_barrier.py --plots-only   # (re)draw figures from results/
python leakage_check.py                   # -> results/leakage.json
cd paper
python make_numbers.py                    # -> numbers.tex, table_csl.tex
pdflatex -interaction=nonstopmode main
bibtex main
pdflatex -interaction=nonstopmode main
pdflatex -interaction=nonstopmode main    # -> main.pdf (4 pages)
python make_txt.py                        # -> manuscript.txt (plain text)
```

The review that motivated the current version is in `review/`.

## Submitting to IEEE SPL

EDICS (SPS Unified EDICS, April 2026 list):

| role | code | title |
|------|------|-------|
| primary | `TM-DSP-SMOD` | Signal and system modeling |
| secondary | `TM-SSP-DETC` | Detection and classification |
| secondary | `ML-CON-PERF` | Performance analysis and bounds |
| secondary | `ML-DLR-GNN` | Graph neural networks |

AI use is disclosed in the Acknowledgment section, as IEEE policy requires
for AI-generated text, figures and code.
