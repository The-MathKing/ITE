# Memory Is Expressivity — walk-based SSMs on graphs (Phase 2 rig)

`walk_ssm_barrier.py` benchmarks the claim that a walk-based state-space model
with real state dimension `S` cannot detect closed walks longer than `S - 1`.
LTI models are covered by a Hankel-rank argument; selective models are
measured empirically. The code uses plain PyTorch and runs on CPU.

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
| `delay` | gradient fit of a diagonal LTI kernel to the pure delay `z^-k` vs `S` | `figures/fig1_delay_realization.pdf` |
| `phase` | token-level lag-`k` revisit detection on non-backtracking walks over random 4-regular graphs, `(S, k)` grid, LTI vs selective | `figures/fig2_phase_diagram.pdf` (double column) |
| `csl` | 10-class CSL (1-WL: 10 %), accuracy vs `S`; LTI, selective, and an exact-count oracle with window `S` | `figures/fig3_csl_accuracy.pdf` |
| `csl` | per-class onset `S` vs predicted `W*(s)` from the exact NB-return horizon | `figures/fig4_csl_threshold.pdf` |

The figures are sized for IEEE layouts: 3.5 in for a single column,
7.16 in for the full text width. Text is set in 8 pt serif with
embedded TrueType fonts, and the PDFs are saved at 600 dpi with tight
bounding boxes.

## Results files

* `results/summary.json`: all metrics, horizons and onsets
* `results/*.csv`
* the four PDFs, or just confirm they rendered

## Manuscript (IEEE Signal Processing Letters)

`paper/main.tex` (IEEEtran `journal`), `paper/refs.bib`. All numbers and
Table I come from `results/summary.json` via `paper/make_numbers.py`, and
figures are read from `../figures/`.

```bash
python walk_ssm_barrier.py --plots-only   # (re)draw figures from results/
cd paper
python make_numbers.py                    # -> numbers.tex, table_csl.tex
pdflatex -interaction=nonstopmode main
bibtex main
pdflatex -interaction=nonstopmode main
pdflatex -interaction=nonstopmode main    # -> main.pdf (4 pages)
```

Before submitting: fill in the author, affiliation and footnote
placeholders, and check the `refs.bib` fields against the publishers.
