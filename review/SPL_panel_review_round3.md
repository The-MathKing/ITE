# SPL panel — round 3 (main_5.pdf, commit ee6fee2)

**Verdict: ready to submit after about 30 minutes of text edits.** There are no correctness errors left. The predicted outcome is *Accept with minor revisions*.

**Page budget.** The body ends on p.4, column 1. The references fill p.4 column 2 and continue onto p.5 (references only), which complies. The references take about 1.1 columns, so the body could grow by roughly 0.8 column before p.5 overflows.

## Proofs (line by line)

**Proposition 1 (a.s. version).** Correct.
- **Gaussian dichotomy.** In finite dimensions, two zero-mean Gaussians are either equivalent (same support subspace) or mutually singular. The claim holds.
- **Direction of the witness functional.** It is right, but not justified in the text. Under H1, (A s_{t−1}, x_t) has the same marginals as under H0 but is correlated. So supp P1 ⊆ supp P0, and a separating functional can only vanish under H1. **Add one clause.**
- **Witness to FIR realisation.** The step to a⊤A^mB = δ_{m,k}v⊤ (m ≥ 1) with v ≠ 0 holds. If v = 0, the functional vanishes under H0 too, because s_{t−1} has the same law under both hypotheses.
- **(a)** Holds. The Hankel rank is k, so S ≥ k. At S = k, a minimal FIR realisation is nilpotent and (A, AB) is controllable. But rank A[B,…,A^{k−1}B] ≤ rank A < k. Contradiction.
- **(b)** Holds. a ⊥ A^{k+1}R = R, so v⊤ = a⊤A^kB = 0.
- **(c)** Holds.
- **Worst-case (k+1)d remark.** Correct.
- **Missing hypothesis.** A "stationary past" requires A to be stable, at least on the reachable space. **Add "stable".**
- **Non-uniformity.** (a) gives error > 0, but that error is not bounded away from zero. At S = k it can be made arbitrarily small (Theorem 1). **Add a clause**, so "exact detection requires S ≥ k+1" is not read as a practical barrier.

**Theorem 1.** Correct.
- The "not attained" argument via Prop. 1(a) is valid; the realisation has the same form.
- The Nehari wording is now right, and Mirsky is cited.

**Corollary 1.** Correct.
- The lag-(k−1) / m = 0 tap count min(m+1, 2k−1−m) ≤ k checks.
- So do KL = I(s_{t−1}; x_{t−k}), convexity, data processing and Pinsker.
- **Overreach.** "Each token carries divergence **of order** S/(2ℓ)" should read "**at most about** S/(2ℓ)". The corollary gives only an upper bound.

## Scope overreach (fix)

The abstract says "invertible dynamics such as S4D **and Mamba** cannot achieve it". The body says "Every S4D or Mamba discretisation … is invertible, so exact detection is never available to them".
- Prop. 1 covers **LTI** (A, B) only.
- Mamba's A_t = exp(Δ_tΛ) is input-dependent, and Remark 1 itself says input-dependent dynamics break the argument.
- A systems referee will flag this.

**Fix:** "LTI invertible dynamics such as S4D (or Mamba with a fixed step)".

## Claims vs. data (checked against ee6fee2)

**Correct as written:**
- the leakage numbers;
- shift register 0.51–0.62 at S = k and 1.00 at S = k+1;
- frozen poles 0.87 / 0.97 vs learned 0.70 / 0.85;
- frozen k = 16 reaching 0.95 only at 2k;
- the threshold ratios, including "0.99 reached for only 7 of 9";
- Wilcoxon p = 0.005 and t-test p = 0.14;
- CSL shift register 47.2% at S = 9 and 53.6 ± 2.0 at S = 32;
- aux loss 56.2 ± 13.7;
- evaluation length −2.5 points;
- the r = 2 pooling result 53.7 → 86.3;
- Table I.

**Issues:**

1. **Leakage values are maxima but read as point values.** The data ranges are:

   | S | range | reported |
   |---|---|---|
   | k−2 | 0.51–0.53 | 0.53 |
   | k−1 | 0.52–0.58 | 0.58 |
   | k | 0.54–0.67 | 0.67 |

   Write "up to 0.53 / 0.58 / 0.67".
2. **Abstract "need 4.0× the minimal state"** gives no threshold. Write "a median 4.0× … to reach AUROC 0.95".
3. **The CSL diagnosis is still one step ahead of the evidence.**
   - The abstract says "the remaining gap lies in the readout". The body and conclusion say "learning revisit statistics … **from graph-level labels**".
   - But LTI + per-lag revisit loss, i.e. token-level labels, reaches only 56.2%. Graph-level labels alone therefore do not explain the gap.
   - The run that would settle it, **shift register + per-lag revisit loss**, is missing. It needs about 21 runs, roughly 3 core-h.
     - If it reaches about 96%, the claim "readout learned from graph-level labels" is proven.
     - If not, the bottleneck is the pooled classifier head.
   - **Without the run:** make the abstract, body and conclusion agree on the weaker "lies downstream of the dynamics (readout and pooling)".
4. **Unused evidence.** In Table I, the shift register's onsets are 4, 6 and 6 for r = 2, 3, 13. That is exactly the first tested S ≥ W*(r)+1, which Prop. 1(c) predicts, because an S-register sees lags ≤ S−1. **Add one clause.** It is the only graph-level test of the theory with a trained model.

## Presentation

- **Abstract.** 151 words; within the 150–250 range of IEEE's general author guidance. SPL states no separate limit that I know of, so check the current SPL author instructions.
- **Figures and table.** Readable at print size. Fig. 1's shaded bound curves work.
- **References (28).** All plausible. Mirsky 1960 (Quart. J. Math. 11:50–59) is correct. Kim et al., ICLR 2025: verify the author list against OpenReview before submitting.

## Predicted decision

| Referee | Recommendation | Main reason |
|---|---|---|
| R1 (systems) | Minor | Mamba scope sentence; stability hypothesis; non-uniformity clause |
| R2 (GNN) | Minor | The CSL "graph-level labels" claim vs the aux-loss result; would like shift + aux |
| R3 (empiricist) | Minor / Accept | Leakage numbers as ranges; threshold in the abstract |
| AE | **Accept with minor revisions** | Assuming the Mamba overreach is fixed before submission |

## Checklist (all text; one optional run)

1. Limit the invertibility claims to LTI: "S4D (or Mamba with a fixed step)", in the abstract and after Prop. 1.
2. Prop. 1: add "stable A" and the supp P1 ⊆ supp P0 clause. Note that the error at S = k can be made arbitrarily small.
3. "Of order S/(2ℓ)" becomes "at most about S/(2ℓ)".
4. Leakage values as "up to"; add the AUROC 0.95 threshold to the abstract's 4.0×.
5. Make the CSL diagnosis consistent in the abstract, body and conclusion, or run shift + aux (about 3 core-h).
6. Add the shift-register onsets = W*+1 sentence.
7. Check the Kim et al. author list.
