# SPL panel — round 2 (revised manuscript, commit f2102a5)

**What was checked.** The revised PDF, and `results/summary.json` / `numbers.tex` at f2102a5 (branch `claude/ieee-gnn-statespace-pipeline-1r7s1t`). There is one new check, `review/check_pairwise_leakage.py`.

**Overall.** The revision fixes most round-1 issues:
- title and signal-processing framing;
- the S/k rate;
- the Gaussian corollary;
- W* reframed;
- the oracle is no longer "best";
- 3 seeds and confidence intervals;
- the shift-register, frozen-pole and comparison-feature controls;
- related work.

Two correctness problems remain, and several sentences overclaim. None of them need a large rewrite.

## A. Correctness

### A1. MAJOR: Proposition 1 mixes two notions of "exact"

Parts (a) and (b) are worst-case statements: "there are two token sequences…". Part (c) holds only almost surely: "for almost every w", "almost surely".

Under the worst-case notion, (c) is **false** for d > 1:
- Take the S = k+1 register that stores wᵀx.
- Pick η ⊥ w and set x'_{t−k} = x_t + η.
- Then s_t and x_t are unchanged, but y'_t = 0.

The correct worst-case threshold is **S ≥ (k+1)d**. Proof: let V_j = A^jB·ℝ^d. If Σ_{j=0}^{k} A^jBη_j = 0 with η_{j0} the first nonzero term, multiply by A^{k−j0}. This gives a confusable pair. So exactness forces V_0, …, V_k to be independent and each of dimension d.

**Fix (recommended).** State all three parts almost surely, for i.i.d. Gaussian tokens with a stationary (infinite) past:
- **(a′)** If S ≤ k, every g has Bayes error > 0.
  - For fixed (A, B), the Gaussian MMSE is a minimum over linear probes. It is > 0 because exact realisation of δ_{m,k} (m ≥ 1) is impossible at S ≤ k.
  - So all λ_i < 1, KL < ∞, TV < 1, and the error is > 0.
- **(b′)** The same holds at every S when A is invertible. Exact realisation forces the reachable and observable part of A to be nilpotent.
- **(c)** Unchanged.

This matches the Corollary's assumptions and costs about 3 lines.

### A2. MINOR: hypotheses and wording in Proposition 1

- **Sequence length.** The (b) construction uses lags up to k+q ≤ k+S. With a finite past and S ≥ (t+1)d, a generic invertible A makes s_t injective, so detection *is* exact. Your own walks have T = 64 while S reaches 64, so state "t ≥ k+S" or "stationary past".
- **"Nilpotent" is stronger than proved.** The abstract says exact detection "requires … a nilpotent (FIR) state matrix", but Prop. 1 only proves "invertible ⇒ impossible". A = N_{k+1} ⊕ D with a readout that ignores D is exact (a.s.). Say "a singular A whose nilpotent part has index ≥ k+1".

### A3. MINOR: Theorem 1 and Corollary 1 details

- **"Not attained" is unproved.** Add one line: an exact realisation makes (A, AB, c) a minimal realisation of an FIR filter, so A is nilpotent on it; but the realisation must also be controllable through AB, which forces A to be invertible there. Contradiction.
- **"Equality in the limit is Nehari's theorem" is imprecise.** Nehari: ‖Γ_E‖ = dist_{L∞}(E, H∞⁻).
- **Spectral-norm EYM is Mirsky (1960).** Eckart–Young 1936 is the Frobenius case; cite Mirsky too.
- **"i.i.d. unit-variance tokens"** should read "i.i.d. with E[xxᵀ] = I_d".
- **Horizon for the Hankel section.** The bound needs t ≥ 2k−1, or a stationary past.
- **The Corollary's KL step is correct but terse.** The link is lag k−1 relative to s_{t−1} with the m = 0 tap included. That again gives a k×k anti-diagonal with multiplicity min(m+1, 2k−1−m) ≤ k. Write that out; "the same k × k Hankel count" is not enough for a referee.
- **Everything else checks line by line:** Prop. 1(a)–(b) worst-case algebra, the tightness construction, the Frobenius bound, the Gaussian MMSE, KL = I(s_{t−1}; x_{t−k}), convexity, data processing and Pinsker.

### A4. MAJOR, but easy: the leakage statement is wrong

The manuscript says: *"this short-lag walk structure carries no information about lag-k revisits"*. My first check, which you adopted, only used 1[v_t = v_{t−l}]. A register also sees coincidences **among stored tokens**. A closed walk that leaves and returns along the same path raises 1[v_{t−j} = v_{t−k+j}].

Logistic regression on all pairwise window coincidences (`check_pairwise_leakage.py`):

| k | S ≤ k−3 | S = k−2 | S = k−1 | S = k |
|---|---|---|---|---|
| 8 | 0.49–0.51 | 0.506 | 0.567 | 0.671 |
| 12 | 0.50 | 0.527 | 0.571 | 0.661 |
| 16 | 0.49–0.50 | 0.528 | 0.577 | 0.661 |

This explains the shift register's 0.53 at S = k−1 and 0.61 at S = k. It also removes the current internal contradiction with the sentence "comes from walk structure among the stored lags".

**Fix.** "…carries no information for S ≤ k−3; within two lags of k, pairwise coincidences among stored tokens give AUROC up to 0.53–0.66, which also accounts for the shift register at S = k." The Gaussian surrogate then remains a fair model for S ≤ k−3. All 22 cells stay below it either way.

## B. Claims vs. data

| Claim | Data | Verdict |
|---|---|---|
| "median ratio … 4.0 at 0.99" | Only 7 of 9 values of k reach 0.99 (k = 14, 16 never do); selective has 6 of 9. The median over the lags that reached it is biased low. | Add "(k ≤ 12; k = 14, 16 not reached)" or drop the 0.99 column. |
| "Wilcoxon p = 0.005" | The paired t-test on the same 81 cells gives p = 0.14. Reporting only the significant test looks selective. | Report both, or say "negligible (|Δ| ≤ 0.03, mean −0.001)". |
| "The information limit is therefore nearly attainable at S ≈ k"; abstract "most of this overhead lies in learning the dynamics" | Frozen poles at S = k: 0.956 / 0.966 / 0.92 for k = 4 / 8 / 12, but **0.64 ± 0.26** at k = 16. They reach 0.95 only near S ≈ 2k for k = 16. The frozen share of the overhead grows with k, as γ^{−k} conditioning predicts. | Qualify: "for k ≤ 12; at k = 16 conditioning of the optimal realisation accounts for about half of the overhead". |
| "Trained diagonal SSMs stay below the Gaussian ceiling" | True for all 22 cells with S < k. On walks the surrogate is only a proven ceiling for i.i.d. tokens (see A4). | Say "Gaussian-surrogate ceiling". |
| "an input-dependent step does not change this" (conclusion) | The gate varies by only 1%. The null says nothing about selectivity where inputs carry content. | Add "on random-feature inputs, where the gate stays inactive". |
| "pooled estimators can succeed below W* at a sample cost that grows as S/W* shrinks" | Heuristic: the per-token divergence is ≤ ½ln(ℓ/(ℓ−S)) ≈ S/2ℓ, but tokens are correlated. | "We expect…", or cite the divergence as the heuristic. |
| Exact-count reference at "S" (Fig. 4, Table I) | Uses lags 1..S. By Prop. 1(c), an S-state register gives lags ≤ S−1, so the comparison is off by one in the reference's favour. | One clause in the caption. |
| Numbers checked and correct | AUROC at S = k 0.69–0.72; 22/22 cells below the surrogate; shift register 0.51–0.62 at S = k and 1.00 at S = k+1; frozen 0.87/0.97 vs learned 0.70/0.85; no-comparison-features Δ ≤ 0.02; CSL 60.4 ± 7.9, 55.2 ± 8.5, aux 56.2 ± 13.7; paired difference −1.7 ± 1.1; r = 2 at 53.7% → 86.3%; τ invariance. | Fine. |

**Still missing (optional, about 1 core-h).** A frozen-pole CSL run, e.g. γP poles covering lags 1..8 at S = 8–16, with only the readout trained. This is the direct test of your stated "likely bottleneck"; right now it is hedged with "we do not isolate it". Without it, referees accept the hedge. With it, the CSL section gets a conclusion.

## C. Presentation

- **Abstract length.** About 200 words (203 by extraction). Check the current SPL limit and cut if needed. The Fig. 3 and Fig. 4 details can go.
- **Fig. 2(b).** Mostly ".00" cells. It could be one sentence, which would free about 40% of a double-column figure.
- **Fig. 1.** The √(1−S/k) dotted curves are hidden under the markers. Draw them thicker or in grey, behind the data.
- **"Theorem 1(2)"** means Eq. (2). Write "Eq. (2)".
- **References are plausible.** Add Mirsky 1960. Mark Kim et al. ICLR 2025 [verify author list] before submitting.

## D. Predicted decision (round 2)

- **R1 (systems):** Minor if A1 is fixed; Major if a referee finds the (c) vs (a) quantifier clash first.
- **R2 (GNN):** Minor. Would ask for the CSL frozen-pole control or accept the hedge.
- **R3 (empiricist):** Minor. Would flag the 0.99 ratio bias, the Wilcoxon-only reporting, and "nearly attainable at S ≈ k" against k = 16.
- **AE:** "Accept with minor revisions" is achievable **only after** A1 and A4 are fixed before submission. A wrong proposition in a letter is a common reason for Reject-and-resubmit.

## E. Pre-submission checklist, ranked

1. **A1.** Restate Prop. 1 almost surely, with a stationary past. **[FIX; no experiments]**
2. **A4.** Correct the leakage sentence with the pairwise numbers. **[FIX; check already run]**
3. **B.** Qualify "nearly attainable at S ≈ k" and "most of this overhead" (k = 16). **[text]**
4. **B.** Fix the 0.99 ratio median and report the t-test alongside Wilcoxon. **[text]**
5. **A2, A3.** Add the length hypothesis, "singular A" instead of "nilpotent", a one-line proof of "not attained", Mirsky, E[xxᵀ] = I. **[text]**
6. Optional: the CSL frozen-pole control (about 1 core-h).
7. Fig. 1 bound visibility; shrink Fig. 2(b); abstract length.
