# SPL pre-submission panel review — "Memory Is Expressivity"

What was checked: `main.tex`, `main.pdf`, `refs.bib`, `numbers.tex`, `table_csl.tex`, `make_numbers.py`, `walk_ssm_barrier.py` and `results/summary.json`. I also ran three new checks (the scripts are in `review/`):

- **(C1)** H2-optimal S-state dense LTI approximations of z^-k, for k = 4 and 6.
- **(C2)** How sensitive W*(s) is to τ.
- **(C3)** A token-level "structural leakage" oracle, plus the optimal AUROC of a Gaussian surrogate.

Page status: the technical content ends about 70% of the way down p.3, and the references run onto p.4. **You have about 0.6–0.7 column-pages of unused budget.** Every fix below fits.

---

## 1. Proof audit

### Proposition 1 (Cayley–Hamilton)

**Algebra: correct.**
- The claim is A^S = Σ c_i A^i, so A^k = Σ c_i A^{i+k-S}. The exponents lie in {k−S,…,k−1}, and that set is inside {1,…,k−1} exactly when S < k.
- The perturbation cancels: s_t − s'_t = −A^k B e + A^{k−S}(Σ c_i A^i) B e = 0.
- x_t is untouched and x'_{t−k} ≠ x_t.
- The notation clashes: c_i is used here and c is used for the probe vector in Thm 1.

**Flaw P1-a (weak / off by one).** The statement also holds at S = k.

Proof sketch. Let p be the minimal polynomial of A on the Krylov space of Be, with degree q ≤ S.
- If q ≤ k−1, use λ^{k−q} p(λ). This is the paper's construction.
- If q = k and p = λ^k, then A^k B e = 0. Perturbing x_{t−k} alone then leaves s_t unchanged.
- Otherwise, let i < k be the largest index with p_i ≠ 0. The polynomial λ^{k−i} p(λ) has zero constant term and coefficient p_i ≠ 0 on λ^k. Perturb along its coefficients; this touches lags up to 2k.

Exact detection first becomes possible at **S = k+1**. The FIR shift register s_t = (wᵀx_t,…,wᵀx_{t−k}) with g = 1[wᵀx_t = s_{t,k+1}] does it almost surely. The correct exact threshold is therefore S ≥ k+1, not S ≥ k.

**Flaw P1-b (wrong as a threshold for the models studied).** For **any invertible A and any S**, confusable pairs exist once t ≥ k+S.
- Counterexample for every S: take A = aI. Set x'_{t−k} = x_{t−k} + e and x'_{t−k−1} = x_{t−k−1} − a^{-1} e. Then s_t − s'_t = a^k B e − a^{k+1} a^{-1} B e = 0.
- In general, λ^k p(λ) works because p_0 = ±det A ≠ 0.
- Every S4D or Mamba discretisation A = exp(ΔΛ) is invertible. Prop. 1 therefore never separates S < k from S ≥ k in the model class you test.
- The sentence *"It is a pointwise map of (s_t,x_t) and is therefore covered by Proposition 1"* is true, but it is vacuous for that reason.

**Flaw P1-c (realisability).** The perturbed sequence changes the tokens at t−j.
- If node v_{t−j} is visited again elsewhere in the window (on n = 24 with T = 64 it always is), x' is no longer a walk-feature sequence of the same walk.
- The confusable set also has probability zero under the Gaussian feature law.
- So Prop. 1 says nothing about error probability on your data.
- The strongest honest probabilistic version is: the class-conditional laws of (s_t, x_t) are mutually absolutely continuous, so the Bayes error is > 0. That holds at every S for invertible A, so it has no rate.

**Corrected statement (3 lines).**
> *Prop. 1′.* An LTI SSM with any readout detects lag-k revisits exactly on all sequences only if S ≥ k+1, and this is attained by an FIR shift register. If A is invertible, exact detection is impossible for every S once t ≥ k+S.

Otherwise, demote Prop. 1 to one sentence inside the Remark.

### Theorem 1 (Hankel / Eckart–Young / Nehari)

**Correct as stated:**
- **Factorisation.** s_t = A s_{t−1} + B x_t gives h_m = cᵀA^m B for m ≥ 0. Block (i,j) of H_k is h_{i+j+1} = (cᵀA^i)(A^{j+1}B), a k×S times S×kd product, so the rank is ≤ S.
- **Target.** The block anti-diagonal with blocks wᵀ satisfies HHᵀ = I_k, which gives k unit singular values.
- **Spectral Eckart–Young (Weyl).** ‖H_k(e)‖₂ ≥ σ_{S+1} = 1.
- **e₀.** The Hankel section contains only m ≥ 1, so (2), (3) and the MSE bound do not need e₀ cancelled. e₀ can only add to the MSE. *"(which can be cancelled by b)"* is harmless but irrelevant.
- **Multiplicity.** The count of (i,j) with i+j = m−1 is min(m, 2k−m).
- **1/k.** It follows from min(·) ≤ k and white isotropic tokens.
- **Tightness.** With A = ρP, B = e₁wᵀ, c = ρ^{−k}e₁ and b = −ρ^{−k}w:
  - cᵀA^mB = ρ^{m−k} wᵀ·1[k | m];
  - e_k = 0;
  - e_{jk} = ρ^{(j−1)k} wᵀ for j ≥ 2;
  - MSE = Σ_{j≥1} ρ^{2jk} = ρ^{2k}/(1−ρ^{2k}).
- **Exact realisation at S = k is impossible in this convention.** Minimal realisations of z^{−k} are nilpotent, and h_m = cᵀA^{m−1}(AB) would need B̂ ∈ range(Â), which contradicts controllability. The infimum is therefore not attained, and the "for every ε" phrasing is right.

**T1-a (unclear / minor misattribution).**
- The sentence *"A finite section is dominated by the Hankel operator norm, which by Nehari's theorem is at most ‖E‖∞"* uses an elementary inequality. The Hankel operator is a compression of the Laurent operator.
- Nehari's theorem is the **equality** ‖Γ_E‖ = dist(E, H∞⁻).
- For 1×d symbols, define ‖E(e^{jω})‖ as the Euclidean norm of the row. It is correct, but unstated.

**T1-b (weak: (2) is attained by the trivial probe).**
- With c = 0 and b = 0, E = −wᵀz^{−k}, so ‖E‖∞ = 1.
- Eq. (2) therefore says *no probe with S < k beats outputting zero in H∞ / Hankel norm*. This is AAK: all Hankel singular values of z^{−k} equal 1, so the optimal Hankel-norm approximant of degree < k is 0.
- Say this. It is sharper and more honest than "H∞ norm at least one".

**T1-c (weak, and the most important fix: the 1/k bound does not depend on S).**
- Use Eckart–Young–Mirsky in **Frobenius** form: Σ_m min(m, 2k−m)‖e_m‖² = ‖H_k(e)‖_F² ≥ Σ_{i>S} σ_i² = k − S. This gives
  **MSE ≥ 1 − S/k**, i.e. *explained variance ≤ S/k*.
- Stacking d probes gives: total MSE for the vector x_{t−k} ≥ d − S/k.
- **C1 (numerical optimum, dense A):**

  | k, S | best MSE | 1 − S/k | paper's 1/k |
  |---|---|---|---|
  | 4, 1 | .894 | .75 | .25 |
  | 4, 2 | .699 | .50 | .25 |
  | 4, 3 | .446 | .25 | .25 |
  | 6, 2 | .849 | .667 | .167 |
  | 6, 4 | .503 | .333 | .167 |

  At S = k: .0016 (k = 4) and .0019 (k = 6).
- **Your own Fig. 1 data** satisfy rel_err² ≥ 1 − S/k at all 16 points with S < k, with a gap of only 0.08–0.2 (e.g. k = 16, S = 14: .204 vs .125; k = 8, S = 6: .39 vs .25). Overlaying √(1 − S/k) on Fig. 1 turns it from a sanity check into a validation of the theory.

**T1-d (wrong as applied to the data).**
- *"quantifies how badly any linear read-out of the state must fail on typical inputs"* — the MSE corollary assumes **i.i.d. white** tokens.
- Walk tokens are not white: E[x_t x_{t−m}ᵀ] = ρ_G(m) I_d, and on your n = 24 graphs this is about 0.03–0.04 and does not decay to 0.
- Say "for i.i.d. tokens (no other revisits)", or state the version MSE ≥ λ_min(Σ_x)(1 − S/k).
- The Hankel and H∞ statements, (2) and (3), are data-free and fine.

**T1-e (minor).** Tightness within the **tested** class is unproven.
- For even k, ρP has two real eigenvalues ±ρ.
- The diagonal-complex parameterisation with S = k (k/2 complex modes, no real mode) cannot realise both.
- This is consistent with the Fig. 1 plateau of 0.05–0.10 at S = k rather than roughly 10⁻³.

**Nonlinear-readout gap (the paper says so, but only implicitly).**
- Thm 1 covers linear probes.
- The text's claim that *"the requirement grows linearly in k as Theorem 1 predicts"* is about a nonlinear-readout AUROC that Thm 1 does not cover.

**Closing the gap in about 8 lines (i.i.d. Gaussian surrogate):**
1. For jointly Gaussian (s_t, x_t, x_{t−k}), E[x_{t−k} | s_t, x_t] is **linear**. Hence **every** g has MSE ≥ d − S/k.
2. For detection, compare H1: x_{t−k} = x_t against H0: independent. Condition on x_t and whiten. Then KL(P1‖P0) = −½ Σ log(1 − λ_i), where λ_i are the squared canonical correlations and Σλ_i ≤ S/k by step 1. By convexity, **KL ≤ ½ log(k/(k−S))**. Pinsker or Bretagnolle–Huber then bounds the balanced accuracy of **any** readout.
3. **C3.** The optimal AUROC with one canonical correlation λ = S/k is:

   | λ = S/k | .125 | .25 | .5 | .667 | .75 |
   |---|---|---|---|---|---|
   | optimal AUROC | .59 | .64 | .71 | .77 | .80 |

   **All 44 below-bound cells (both models) sit under the surrogate curve.** Examples from the LTI panel:

   | cell | S/k | observed | surrogate |
   |---|---|---|---|
   | k=16, S=2 | .125 | .53 | .59 |
   | k=4, S=2 | .5 | .61 | .71 |
   | k=3, S=2 | .67 | .64 | .77 |
   | k=16, S=12 | .75 | .66 | .80 |

4. **Surrogate validity on your graphs (C3).** A Bayes-table predictor of y^{(k)}_t from the exact revisit indicators at lags 1..S−1 has AUROC 0.49–0.52 for every (k, S < k). Short-lag structure therefore carries **no** information about lag-k revisits on your graph pool. The i.i.d. surrogate is a fair model, and the "partial correlation" explanation for AUROC > 0.5 is supported. Say both.

---

## 2. Claim-vs-data audit

| # | Quoted claim | Data | Verdict |
|---|---|---|---|
| D1 | "median of 4.0× the Hankel minimum" | Depends on the threshold. Median ratio: 2.0× at AUROC 0.8; 3.4× at 0.9; 4.0× at 0.95; 5.3× at 0.99. Per-k at 0.95: 3.0–4.8. The S grid {…,24,32,48,64} quantises ratios in 25–50% steps. | **Overstated.** Report the ratio as a curve against the threshold, or report S at a fixed AUROC with the grid resolution stated. |
| D2 | "the requirement grows linearly in k as Theorem 1 predicts" | Thm 1 gives only a lower bound, for linear probes. At S = k, AUROC is .63–.73 (k = 4, 8, 12, 16). Fig. 2 shows **no transition at S = k**; e.g. k = 16 rises smoothly .53, .57, .62, .66, .73, .80, .85, .91, .96. | **Wrong / overstated.** "Consistent with", not "predicts". "Phase diagram" is a misnomer. |
| D3 | "The overhead is thus a property of end-to-end training … rather than of realisability" | This compares an L2-error-0.1 threshold on a scalar delay fit with an AUROC-0.95 threshold on an 8-dim detection task. No experiment isolates training. | **Unsupported.** Needs the frozen-dynamics control (A5). |
| D4 | "An error of 0.1 is first reached at a median of 1.0k" | k = 4, S = 4 error = 0.1004 > 0.1, so the ratio for k = 4 is 1.5. The median is still 1.0. | Fine, but say "≈0.1". |
| D5 | "Selectivity changes AUROC by +0.000 … do not escape the memory requirement"; abstract: "do not reduce this" | **One seed** (162 runs, seed 0 only). The per-cell difference has SD 0.016 and range [−0.082, +0.085]. The same paragraph reports 3.4× (selective) vs 4.0× (LTI). | **Internally inconsistent and statistically unsupported.** No equivalence claim is possible without seeds. |
| D6 | "selective (Mamba-type)" | Only Δ_t is input-dependent. B_t = Δ_t B has no separate x-dependent B or C. W_Δ is initialised with std 0.02. Random i.i.d. features give the gate nothing to select on, so the null is expected by design. | **Overstated.** Rename it "input-dependent step size". Report ‖W_Δ‖ or Var(Δ_t) after training. |
| D7 | "Below the bound … mean AUROC is 0.61" | Averages cells with S/k from .125 to .86. | **Weak.** Plot against S/k with the Gaussian surrogate curve (§1). |
| D8 | "For all 10 classes, its onset is exactly the first tested S ≥ W*(s)" | True on the even grid. Exact equality holds for only 4/10 (cslExactOr = 4). Near-tautological: the oracle's features **are** ρ̂(ℓ), ℓ ≤ S, the same quantities that define W*. It only shows that 16×256 tokens resolve τ. | **Overstated.** C2: W* is unchanged for τ ∈ [10⁻⁴, 3·10⁻³] and changes at 5·10⁻³. That is good news, so report it. |
| D9 | "This is the best a window-S reader can do" | **Wrong.** (a) Linear logistic regression on marginal lag frequencies is not optimal; joint within-window patterns (CRaWl) carry more. (b) Your data contradict it: at S = 2 the oracle scores **10.0%**, because NB walks cannot return at lags 1–2 and the features are identically 0. The trained LTI scores **24.0 ± 0.8%**, with class s = 2 at 86%. | **FATAL to the framing.** Rename it "exact-count reference". |
| D10 | "The prediction is that class s becomes recognisable once S ≥ W*(s)"; Fig. 4 legend "onset = W* (bound)" | s = 2 reaches 80% at S = 2 < W* = 3 for both trained models, which is below the "bound" in your own figure. With pooling over N tokens, any explained variance > 0 gives consistent classification as N → ∞. | **Wrong as a consequence of Thm 1.** W* is a horizon for *exact / token-level* access, not a barrier for pooled classification. |
| D11 | "(approximate) linear access to revisits only at lags ℓ ≤ S" | Thm 1 (corrected) allows explained variance up to S/ℓ at ℓ > S. | **Wrong ("only").** |
| D12 | "binding constraint … is the optimisation of precise, ill-conditioned delays" | At S = 64 the token-level AUROC for k ≤ 8 is ≥ 0.998 with per-token supervision. CSL models get only a graph label through mean pooling, with T_train = 128 vs T_test = 256 and 3000 steps. The difference is the **supervision signal**, not delay conditioning. The unused `aux_weight` revisit loss is already in the code. | **Unsupported.** Run aux-loss and frozen-encoder controls (A6). |
| D13 | "y_t = 1 exactly when the last k steps close an NB cycle, so revisit statistics are cycle statistics"; abstract "recognising a cycle of length k reduces to…" | Closed NB walks include repeated cycles (a triangle twice gives lag 6) and tailed "lollipops" (lag 2|P| + |C|). These are traces of the Hashimoto operator, not cycle counts. | **Wrong wording.** Use "closed NB walks". |
| D14 | "test on 64 unseen graphs" (phase); CSL "test accuracy" | Phase: correct. CSL: the same 10 graphs; only walks and features are fresh. | **Unclear.** State it. |
| D15 | "the code regenerates every figure and number" | README full run uses the default `--phase-seeds 2`, but summary.json holds seed 0 only. Rerunning as documented changes the numbers. | **Wrong (reproducibility).** Fix the default or the README. |
| D16 | "best LTI … 60.4 ± 2.6% at S = 64" | Correct (population SD, n = 3). The curve is still rising (58.7 → 60.4), so "best" is limited by the grid. | Fine; say "at the largest S tested". |
| D17 | "exceeds 95% from S = 8"; oracle 98.3% | Correct. | Fine. |
| D18 | "Only 3 (LTI) and 3 (selective) classes reach 80%" | Correct (s = 2, 3, 13). | Fine. |

### Design attacks and the cheapest fix for each

- **One seed on the phase grid.**
  - Fix: 2 more seeds and CIs. Cost: 4.4 core-h per seed, so ~9 core-h (~1.2 h on 8 workers).
- **Three seeds on CSL with no test.**
  - Fix: 5 seeds, report 95% t-intervals, and run a paired test for LTI vs selective. About 3 core-h extra.
- **The (Qx−Ks)^{⊙2} comparison features** are a task-specific inductive bias.
  - Fix: ablate them in one phase column (k = 8). About 0.5 core-h.
  - Covered by the theory either way.
- **τ = 2·10⁻³.** Not tuned (C2); one sentence settles it. No cost.
- **Walk lengths.**
  - T = 64 on n = 24 revisits every node about 2.7 times. That is fine for token labels.
  - For CSL, the 128 (train) vs 256 (test) mismatch is untested.
  - Fix: evaluate at T = 128 as well. No retraining needed.
- **Pooling 16 walks.** This is what lets s = 2 pass below its horizon.
  - Fix: report accuracy against the number of pooled tokens N for one S < W* and one S ≥ W*. This directly tests the "information vs pooled SNR" story, and it is evaluation only (cheap).
- **Union-of-two-Hamiltonian-cycles 4-regular graphs.**
  - This is not uniform on 4-regular graphs (contiguity results make it asymptotically fine, but n = 24 is small).
  - Harmless; one sentence. Optional: a configuration-model check on one column.
- **Is the oracle a fair baseline?** No, as either an "upper bound" or a "best window reader" (D9). As an *exact-feature reference* it is fine.

### What a referee will demand

1. **A frozen-dynamics control**: A and B set to the S = k delay fit (Fig. 1), with only the readout trained. This is the only way to support D3.
2. **An explicit-window baseline** (1-D conv / FIR window of length S with the same readout), i.e. CRaWl-lite. Cheap.
3. **A small causal transformer or attention baseline** with window ≥ k. Optional in a letter; one column of the phase grid is enough.
4. **GIN + random node features** on CSL. RNI is known to break CSL, so this is the fair non-walk comparator given that the paper uses random features. The 1-WL 10% line needs no run: CSL graphs are 4-regular with no features, so the 10% is provable. Say that in one clause.
5. **CIs and a paired test.**

---

## 3. Novelty and positioning

Closest prior work. Items marked [verify] need a bibliographic check before citing.

- **Realisation / model reduction.**
  - Ho & Kalman 1966 (cited).
  - Kung, "A new identification and model reduction algorithm via singular value decomposition", 1978 [verify venue: Asilomar Conf.].
  - Glover 1984 (cited) and AAK 1971 (cited).
  - Antoulas, *Approximation of Large-Scale Dynamical Systems*, SIAM 2005.
  - Partington, *An Introduction to Hankel Operators*, LMS Student Texts, 1988.
  - That z^{−k} has McMillan degree k and k unit Hankel singular values, and that its optimal Hankel-norm approximant of degree < k is zero, is **textbook**.
- **SSM memory as delay approximation.** **Not cited, and a referee will catch it.**
  - Gu, Dao, Ermon, Rudra, Ré, "HiPPO: Recurrent Memory with Optimal Polynomial Projections", NeurIPS 2020. HiPPO-LegT is a sliding-window / delay memory with N states.
  - Voelker, Kajić, Eliasmith, "Legendre Memory Units", NeurIPS 2019. The design objective is N-state Padé-type approximation of a delay.
- **Linear-RNN memory: approximation and optimisation.**
  - Li, Han, E, Li, "On the Curse of Memory in Recurrent Neural Networks: Approximation and Optimization Analysis", ICLR 2021. This directly anticipates the "optimisation overhead" story; must cite.
  - Wang & Xue, "State-space models with layer-wise nonlinearity are universal approximators with exponential decaying memory", NeurIPS 2023 [verify].
  - Orvieto et al., "Resurrecting Recurrent Neural Networks for Long Sequences", ICML 2023.
  - Orvieto et al., "Universality of Linear Recurrences Followed by Non-linear Projections…", ICML 2024 [verify].
  - Zucchet & Orvieto, "Recurrent neural networks: vanishing and exploding gradients are not the end of the story", NeurIPS 2024 [verify].
- **Recall / copying lower bounds.**
  - Jelassi et al., ICML 2024 (cited).
  - Arora et al., "Zoology: Measuring and Improving Recall in Efficient Language Models", ICLR 2024 [verify venue].
  - Arora et al., "Simple linear attention language models balance the recall-throughput tradeoff" (Based), ICML 2024.
  - Wen, Dang, Lyu, "RNNs are not Transformers (Yet)…" [verify venue/year].
- **State tracking.**
  - Merrill, Petty, Sabharwal, "The Illusion of State in State-Space Models", ICML 2024.
  - Sarrof, Veitsman, Hahn, "The Expressive Capacity of State Space Models: A Formal Language Perspective", NeurIPS 2024 [verify].
  - Grazzi et al., "Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues", ICLR 2025 [verify].
- **Walk-based expressivity.**
  - CRaWl (cited) and NeuralWalker (cited).
  - Kim et al., "Revisiting Random Walks for Learning on Graphs", ICLR 2025 [verify].
  - Kriege, "Weisfeiler and Leman Go Walking: Random Walk Kernels Revisited", NeurIPS 2022 [verify].
  - Hashimoto 1989 for the NB operator [verify: Adv. Stud. Pure Math. 15]. It is used by name but not cited.

**What is new:**
1. Framing lag-k revisit detection as a z^{−k} realisation problem for walk-SSMs.
2. With the fixes: the **S-dependent explained-variance bound (S/k) and KL bound (½log(k/(k−S))) holding for all readouts** under the i.i.d.-Gaussian-feature model. I have not seen this stated for SSMs; it is modest but citable.
3. Prop. 1′: the FIR-vs-invertible dichotomy. This is useful because every S4D or Mamba A is invertible.
4. The NB return-profile horizon W*(s) for CSL, and the measured gap between exact-count, direct-fit and trained models.

**What is not new:** "S ≥ k for a lag-k delay" as stated. That is Ho–Kalman / McMillan degree, and a systems referee will say so in one line.

**A defensible framing:** *"Delay-and-compare detection with order-S LTI filters: exact Hankel/AAK and information-theoretic limits, and what they imply for random-walk graph learners."* The contribution is the **quantitative, readout-agnostic detection bound as a function of S/k**, plus its validation against optimal fits and trained models. The contribution is not the threshold itself.

---

## 4. SPL fit and presentation

**Desk-reject risk: moderate to high as written.**
- The title ("Memory Is Expressivity") and the first paragraph (1-WL, GNNs, CSL) read as graph ML.
- The AE will ask why it is not in LoG, TMLR or NeurIPS.

**Fixes:**
- **Title:** "Hankel-Norm and Information Limits on Delay Detection by Low-Order State-Space Models, with Application to Graph Random Walks".
- **Abstract, sentence 1:** start from an order-S LTI filter detecting whether an i.i.d. input repeats after k samples.
- **Intro, paragraph 1:** echo / repetition detection, realisation theory, then SSM backbones. Move GNNs to paragraph 2.
- **Keywords:** add "model order reduction" and "detection"; drop "expressivity".
- Graph signal processing is in scope. Cite one GSP venue paper only if you actually use GSP.

**Page budget (about 0.65 column free now):**
- **Cut or compress:**
  - Prop. 1 proof becomes Prop. 1′ with a 3-line proof; saves about 3 lines.
  - Fig. 4 is the weakest figure. The y-axis to 70 squashes the only informative points, 7/10 points per model are "missing" markers, the oracle markers sit on the diagonal (tautology, D8), and the legend calls W* a "bound" that the data violate. **Delete it.** Table I already carries the information.
  - Merge the conclusion into 4 lines.
- **Add:**
  - Thm 1 with (1 − S/k) and the Gaussian/KL corollary: about 10 lines.
  - Overlay √(1−S/k) on Fig. 1: 0 lines.
  - Replace Fig. 4 with a single-column figure of AUROC against S/k (all k collapsed), with the surrogate optimal-AUROC curve and the frozen-dynamics control. This one figure carries the paper.
  - Controls text: about 8 lines.
  - Related-work sentence with 5–6 new refs: about 6 lines, plus references on p.4 (there is room).

**Notation (collisions):**
- **s** is the skip, **s_t** the state and **S** the state dimension. Rename the skip to *r* or *q*.
- **c** is both the Cayley–Hamilton coefficients c_i and the probe vector c.
- **b** is both the probe term and the bias in softplus(W_Δ x_t + b).
- **ρ** is both the pole radius and the return probability ρ_G(ℓ).
- **e** is used for the perturbation, the error e_m and the basis vector e₁.
- **K** in (Qx − Ks) sits next to lag k.
- **δ_{mk}** vs δ_{m,k}: inconsistent.
- H_k(e) is used before e_m is defined for vector outputs.

**Other presentation issues:**
- **Undefined or ambiguous:**
  - The norm in "sup_ω ‖E(e^{jω})‖".
  - "real state dimension" for complex diagonal modes (define it once).
  - "stationary NB walk" (the uniform distribution on directed edges; say so).
  - "Above it, the median is 0.01" (S ≥ k or S > k?).
  - "a median of 1.0 k states" (write "S = k").
  - "Remark" in the text vs "Remark 1".
- **Fig. 2:** 5-pt cell labels are unreadable at print size; drop the numbers or use 7 pt. The two panels are nearly identical, so plot the difference or one panel plus a Δ strip.
- **Fig. 3:** the caption must say what the oracle is and that the chance line is provable, not run.
- **Table I:** replace "--" with a proper dash and add seeds / threshold (80%) to the caption.
- **Captions are incomplete:** Fig. 1 does not say the number of restarts or J = 64 truncation; Fig. 2 does not give seeds, T, n or d.
- **Code docstring** says "S = 2M", but the paper says odd S gets a real mode. Harmless, but fix it.

**refs.bib: all 19 entries are plausible.** Authors, venues and years match my knowledge:
- Mamba: COLM 2024. Correct.
- Graph-Mamba (Wang et al.): arXiv 2402.00789. Correct.
- Behrouz & Hashemi: KDD 2024. Correct.
- NeuralWalker: ICLR 2025. Correct.
- CRaWl: TMLR 2023. Correct.
- Dwivedi et al.: JMLR vol. 24, 2023. Correct.
- Classical refs (Ho–Kalman 1966, Nehari 1957, AAK 1971, Glover 1984, Alon et al. 2007): volumes and pages look right.

**Issues:**
- Missing pages / article numbers for JMLR (24(43):1–48) [verify].
- TMLR has no volume field; add the OpenReview ID or the ISSN convention.
- Missing: Hashimoto, Eckart–Young (1936, Psychometrika), HiPPO, LMU, Li et al. 2021.
- Remove the NOTE comment from the .bib.

---

## 5. Simulated decision

**Referee 1 (control / systems theorist): Major revision.**
1. "S ≥ k for z^{−k}" is the McMillan degree, and the H∞ ≥ 1 statement is attained by the zero probe. Both are textbook (AAK/Glover). As written, the contribution is a restatement.
2. Prop. 1 is off by one (it holds at S = k), and it is not a threshold for invertible A, which covers all tested models. The exact threshold is S = k+1, via an FIR register.
3. The MSE bound 1/k ignores S. The Frobenius EYM gives 1 − S/k, and the data (Fig. 1) track it. The "white tokens" assumption is not the data law.

**Referee 2 (GNN expressivity): Reject (resubmit).**
1. The graph-level "prediction" S ≥ W*(s) does not follow from Thm 1 and is violated by the authors' own s = 2 result. Pooling makes any nonzero correlation sufficient.
2. The "best window-S reader" oracle is beaten by the trained SSM at S = 2 (24% vs 10%). It is neither optimal nor an upper bound, and its match to W* is tautological.
3. "Revisits = cycles" is incorrect (closed NB walks). There is no comparison to CRaWl-style window models or GIN + RNI on CSL. The related work misses walk-expressivity and SSM-memory literature (HiPPO, LMU, Merrill et al.).

**Referee 3 (skeptical empiricist): Major revision.**
1. The phase grid uses a single seed. "Selectivity does not help" rests on a mean difference of 0.000 with no variance estimate, and it contradicts the 3.4× vs 4.0× numbers in the same paragraph. The "selective" model is only a Δ-gated SSM on inputs with nothing to select.
2. "4× overhead due to optimisation" is an uncontrolled inference: the thresholds are incommensurate and there is no frozen-dynamics control. The ratio ranges from 2× to 5.3× depending on the AUROC threshold.
3. The CSL shortfall is attributed to delay conditioning without ruling out weak graph-level supervision, train/test length mismatch or the training budget. The aux-loss path exists in the code but was never run.

**AE decision letter.**
> Dear Authors, the three referees agree that the question is interesting and the code release is commendable. However, they identify (i) a mis-specified central proposition, (ii) a theorem whose stated form is classical and whose quantitative corollary is loose and assumes a data law not used in the experiments, (iii) a graph-level "prediction" that does not follow from the theory and is contradicted by the reported data, and (iv) empirical conclusions (selectivity, optimisation overhead) drawn from a single seed without controls. Several of these go to the core claims, so the manuscript cannot be accepted in its current form. SPL does not normally allow multi-round major revisions. **Decision: Reject and encourage resubmission.** A resubmission should reframe the work as a signal-processing detection result, state and prove the S-dependent bounds, correct Proposition 1, remove or reframe the CSL "prediction", and add seeds, confidence intervals, and the controls requested by Referee 3.

---

## 6. Action plan

### Ranked issues

In the table, **Exp?** means "needs new experiments". CPU costs are estimates from the recorded run times.

| Sev | Location | Problem | Fix | Exp? (CPU) | Pages |
|---|---|---|---|---|---|
| **FATAL** | §III-A, Table I, Fig. 4, abstract, conclusion | "Class s recognisable once S ≥ W*(s)" does not follow from Thm 1 and is violated (s = 2 at S = 2) | Reframe W* as the horizon for **exact/token-level** access. For pooled classification, state explicitly that the bound governs per-token SNR (explained variance ≤ S/W*), so the required pooled sample size grows as the explained variance shrinks. Add an accuracy-vs-pooled-tokens plot for one S < W* and one S ≥ W* | Y (eval only, <0.2 core-h) | +3 lines |
| **FATAL** | §IV CSL | "best a window-S reader can do"; the oracle is beaten at S = 2 | Rename "exact-count reference"; delete "best"; report 10% vs 24% honestly | N | 0 |
| **FATAL** | Prop. 1 | Off by one; not a threshold for invertible A (all tested models) | Replace with Prop. 1′ (S ≥ k+1 via FIR; impossible ∀S for invertible A) | N | −2 lines |
| **MAJOR** | Thm 1, abstract | 1/k bound independent of S; H∞ ≥ 1 is attained by the zero probe | MSE ≥ 1 − S/k (vector: d − S/k); state (2) as "no better than zero probe" (AAK) | N | +3 lines |
| **MAJOR** | Thm 1 / Remark | Nonlinear readout not covered; "white tokens" is not the data law | Gaussian corollary: MMSE ≥ 1 − S/k for every g; KL ≤ ½log(k/(k−S)); state the i.i.d. surrogate and cite the C3 leakage check (AUROC ≈ 0.5) as justification | N (C3 already run) | +7 lines |
| **MAJOR** | §IV phase | "4× … property of training, not realisability" is uncontrolled | Frozen-dynamics control: A, B from the Fig. 1 fit at S ∈ {k, 1.5k, 2k}, train the readout only, k ∈ {4, 8, 12, 16} | Y (~1.5 core-h) | +4 lines, new Fig. |
| **MAJOR** | §IV phase | Single seed; selectivity null claim; 3.4 vs 4.0 contradiction | 3 seeds and CIs; paired test; call the difference "not significant (p = …)"; drop "do not reduce" | Y (~9 core-h) | 0 |
| **MAJOR** | §IV phase | "4.0×" depends on the threshold; "linearly as Thm 1 predicts" | Report ratio at 0.8/0.9/0.95/0.99, or S_emp vs k with a fitted slope; say "consistent with" | N | +1 line |
| **MAJOR** | Positioning | Missing HiPPO, LMU, Li et al. 2021, Merrill et al., Zoology/Based, Hashimoto | Add a 4–5-line related-work paragraph; state plainly that S ≥ k is McMillan degree | N | +5 lines, +0.3 col refs |
| **MAJOR** | Title / abstract / intro | Reads as graph ML (SPL scope) | New title and SP-first abstract (below) | N | 0 |
| **MAJOR** | §IV CSL, conclusion | "binding constraint = optimisation of delays" is unsupported | (a) aux revisit loss already coded (`aux_weight=1`); (b) freeze the phase-trained encoder and fit a linear pooled head; (c) evaluate at T = 128 | Y (~3 core-h) | +3 lines |
| **MAJOR** | §IV models | "Mamba-type" overstated; nothing to select on | Rename "input-dependent step (Δ_t)"; report Var(Δ_t); one sentence on why a random-feature task gives the gate no signal | N (log only) | +1 line |
| MINOR | §II, abstract | "revisits = cycles" | "closed NB walks (traces of the Hashimoto operator)" | N | 0 |
| MINOR | §III-A | "access … only at lags ℓ ≤ S" | "explained variance at lag ℓ > S is at most S/ℓ" | N | 0 |
| MINOR | Thm 1 proof | Nehari misattribution; norm on 1×d symbols | "elementary compression bound; equality is Nehari" | N | 0 |
| MINOR | Thm 1 tightness | Not realisable in the diagonal-complex class at even S = k | One sentence | N | +1 line |
| MINOR | §IV CSL | Test graphs = train graphs; 3-seed SD with ddof = 0 | State it; use ddof = 1 or CIs | N | 0 |
| MINOR | §IV | τ choice | "W* is invariant for τ ∈ [10⁻⁴, 3·10⁻³]" | N (done: C2) | +1 line |
| MINOR | §IV | (Qx−Ks)² features | Ablate at k = 8 | Y (~0.5 core-h) | +1 line |
| MINOR | Baselines | No explicit-window / attention / GIN+RNI | FIR-window readout on one phase column; GIN+RNI on CSL (5 seeds) | Y (~2 core-h) | +2 lines |
| MINOR | Fig. 2 | 5-pt labels; redundant panels | Drop the numbers, show LTI plus a Δ(sel − LTI) strip | N | 0 |
| MINOR | Fig. 4 | Weakest figure; calls W* a "bound" | Replace with AUROC vs S/k plus the surrogate bound and frozen control | Y (uses above) | 0 |
| MINOR | Fig. 1 | No theory overlay | Overlay √(1−S/k) | N | 0 |
| MINOR | Notation | s/S/s_t, c, b, ρ, e, K collisions | Rename the skip to r and C-H coefficients to α_i; pole radius r₀ → γ | N | 0 |
| MINOR | README / code | Default 2 phase seeds vs published 1 | Align after rerunning with 3 | N | 0 |
| MINOR | refs | Missing JMLR pages; TMLR format; .bib NOTE | Fix | N | 0 |

**Total new CPU: about 16 core-hours, roughly 2–3 wall-hours on 8 workers.**

### The 5 changes that most increase acceptance probability

1. **Replace the theory block** with Prop. 1′ (the FIR / invertible dichotomy) and Thm 1′ (Hankel = zero-probe optimality; **MSE ≥ 1 − S/k**; Gaussian corollary **for every readout**: MMSE ≥ 1 − S/k, KL ≤ ½log(k/(k−S))). This turns a textbook restatement into a quantitative, readout-agnostic result.
2. **Retract the CSL "prediction" and the "best window reader" claim.** Reframe W* as a token-level horizon, and show pooled accuracy against the number of tokens to explain s = 2.
3. **Frozen-dynamics control plus the Gaussian-surrogate AUROC curve** in one new figure replacing Fig. 4. This is the only thing that can support "information vs optimisation".
4. **Seeds, CIs and a paired test** on the phase grid. Rename "selective" and state honestly that the random-feature task gives a gate nothing to use.
5. **Signal-processing reframing and positioning**: title, abstract, intro paragraph 1, and HiPPO / LMU / Li et al. / Merrill et al. / Zoology in related work, with an explicit "S ≥ k is the McMillan degree; our contribution is the S/k rate and its consequences".

### Rewritten abstract (≤150 words)

> An order-S linear time-invariant (LTI) state-space model reads an i.i.d. Gaussian sequence and must decide whether the current sample repeats the one k steps earlier. This delay-and-compare primitive underlies cycle detection by random-walk graph learners. The lag-k delay has k unit Hankel singular values. Hence for S<k no LTI probe improves on the zero probe in Hankel norm, any probe explains at most a fraction S/k of the delayed sample's variance, and, for any nonlinear readout, the detection divergence is at most ½log(k/(k−S)). Exact detection requires S≥k+1 and a nilpotent (FIR) state matrix; it is impossible at every S for invertible dynamics such as S4D and Mamba. Optimal fits follow the S/k law, and trained diagonal SSMs stay below the implied AUROC ceiling. [With frozen optimal dynamics, AUROC reaches X at S=k, so the remaining overhead is due to optimisation.] On circular skip-link graphs, non-backtracking return horizons predict when exact-count features separate classes.

(Fill in X only if the control confirms it; otherwise drop the bracketed sentence.)

### Rewritten contributions paragraph

> Our contributions are threefold. (i) *Limits.* The pure delay z^{-k} has k unit Hankel singular values (classically, McMillan degree k [Ho–Kalman, Glover]). We turn this into bounds for delay-and-compare detection: for an order-S LTI state, no probe beats the zero probe in Hankel norm, the explained variance of the lag-k sample is at most S/k, and under Gaussian inputs this bounds every nonlinear readout (MMSE ≥ 1−S/k; KL ≤ ½log(k/(k−S))). Exact detection needs S≥k+1 with nilpotent dynamics and is impossible for invertible (S4D/Mamba) dynamics. (ii) *Graph consequence.* From the non-backtracking (Hashimoto) operator we compute the return horizon W*(r) at which exact-count features separate each CSL class. We show that pooled estimators can succeed below W* at a sample cost governed by S/W*. (iii) *Validation.* Optimal fits track the S/k law. Controlled experiments (frozen optimal dynamics, three seeds, input-dependent step sizes) separate the information limit from the optimisation overhead of trained SSMs.

---

### Checks run for this review (`review/`)

- `check_bounds_and_tau.py`
  - W*(s) against τ: unchanged for τ ∈ [10⁻⁴, 3·10⁻³].
  - Dense-A H2-optimal fits confirm MSE ≥ 1 − S/k and show that 1/k is loose.
- `check_leakage_and_gaussian_auroc.py`
  - Exact short-lag revisit indicators give AUROC ≈ 0.50 for lag-k revisits on your graph pool.
  - Gaussian-surrogate optimal AUROC against S/k.
