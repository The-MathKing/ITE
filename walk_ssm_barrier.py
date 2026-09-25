#!/usr/bin/env python3
"""
Memory Is Expressivity: a state-dimension barrier for walk-based SSMs on graphs.

Phase 2 experimental rig (pure PyTorch + NumPy + Matplotlib, CPU-friendly).

Claim under test
----------------
A walk-based graph model reads a (non-backtracking) random walk v_0, v_1, ...
with fresh random node features x_t = r_{v_t}.  Detecting a closed walk of
length k means deciding, at every step t, whether v_t == v_{t-k}.

  * Theorem (LTI barrier).  For an LTI state-space model with real state
    dimension S, any linear probe z_t = c^T s_t + d x_t of the lag-k token
    has an error transfer function E with ||E||_inf >= 1 and
    sum_m m e_m^2 >= 1 whenever S < k: the target impulse delta_{m,k}
    (m >= 1) has a rank-k Hankel matrix with unit singular values, while an
    order-S model has Hankel rank <= S (Eckart-Young + Nehari).  The bound
    is tight: S = k achieves arbitrarily small error.
  * H1: trained LTI (S4D-style) walk models show a phase transition along
    S = c k; we report the empirical overhead c >= 1.
  * H2: input-dependent (selective, Mamba-style) SSMs are not covered by the
    Hankel argument; we measure whether they empirically escape it.
  * Graph consequence: on CSL graphs (1-WL fails, 10% accuracy), class s can
    only be recognised once the model can see closed walks up to the exact
    "distinguishability horizon" W*(s), computed here from the non-backtracking
    operator.  Prediction: per-class onset at S ~ W*(s).

Experiments
-----------
  delay : gradient fit of a diagonal LTI kernel to the pure delay z^{-k}
  phase : token-level lag-k revisit detection on random 4-regular graphs,
          (S, k) grid, LTI vs selective               -> phase diagram
  csl   : 10-class CSL classification vs S, LTI vs selective, several seeds

Outputs
-------
  figures/fig1_delay_realization.pdf
  figures/fig2_phase_diagram.pdf
  figures/fig3_csl_accuracy.pdf
  figures/fig4_csl_threshold.pdf
  results/runs/*.json        (one file per training run; reruns resume)
  results/summary.json       (everything needed for Phase 3)
  results/*.csv              (tables)

Usage
-----
  python walk_ssm_barrier.py                  # full run, 8 worker processes
  python walk_ssm_barrier.py --workers 6      # be nicer to other jobs
  python walk_ssm_barrier.py --quick          # ~2-5 min smoke test
  python walk_ssm_barrier.py --plots-only     # redraw figures from results/
  python walk_ssm_barrier.py --exp phase csl  # subset of experiments
"""

import argparse
import csv
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CSL_N = 41
CSL_SKIPS = [2, 3, 4, 5, 6, 9, 11, 12, 13, 16]

# --------------------------------------------------------------------------
# Graphs and walks
# --------------------------------------------------------------------------


def random_4regular(n, rng):
    """Union of two random Hamiltonian cycles (simple, connected, 4-regular)."""
    while True:
        edges = set()
        ok = True
        for _ in range(2):
            p = rng.permutation(n)
            for i in range(n):
                u, v = int(p[i]), int(p[(i + 1) % n])
                e = (min(u, v), max(u, v))
                if e in edges:
                    ok = False
                    break
                edges.add(e)
            if not ok:
                break
        if ok:
            nbrs = [[] for _ in range(n)]
            for u, v in edges:
                nbrs[u].append(v)
                nbrs[v].append(u)
            return np.array(nbrs, dtype=np.int64)


def csl_graph(s, n=CSL_N):
    """Circular skip-link graph C_{n,s}: i ~ i +- 1, i ~ i +- s (4-regular)."""
    i = np.arange(n)
    return np.stack([(i - 1) % n, (i + 1) % n, (i - s) % n, (i + s) % n], 1)


def nb_walks(nbrs, gidx, T, rng):
    """Non-backtracking walks.

    nbrs : (G, n, d) neighbour table of a graph pool
    gidx : (B,) graph index of each walk
    returns (B, T) node indices
    """
    G, n, d = nbrs.shape
    B = len(gidx)
    W = np.empty((B, T), dtype=np.int64)
    W[:, 0] = rng.integers(0, n, B)
    W[:, 1] = nbrs[gidx, W[:, 0], rng.integers(0, d, B)]
    for t in range(2, T):
        rows = nbrs[gidx, W[:, t - 1]]                       # (B, d)
        prev_pos = np.argmax(rows == W[:, t - 2, None], axis=1)
        r = rng.integers(0, d - 1, B)
        r = r + (r >= prev_pos)
        W[:, t] = rows[np.arange(B), r]
    return W


def walk_features(W, n, d_feat, rng):
    """Fresh i.i.d. Gaussian node features per walk (random node identities)."""
    B, T = W.shape
    feats = rng.standard_normal((B, n, d_feat)).astype(np.float32)
    return feats[np.arange(B)[:, None], W]                   # (B, T, d_feat)


def nb_return_rates(nbrs, L):
    """Exact P(v_l == v_0), l = 1..L, for a stationary NB walk."""
    n, d = nbrs.shape
    E = [(u, int(v)) for u in range(n) for v in nbrs[u]]
    idx = {e: j for j, e in enumerate(E)}
    m = len(E)
    Bm = np.zeros((m, m))
    for (u, v), j in idx.items():
        for w in nbrs[v]:
            if w != u:
                Bm[j, idx[(v, int(w))]] = 1.0 / (d - 1)
    tail = np.array([u for u, _ in E])
    head = np.array([v for _, v in E])
    close = (head[None, :] == tail[:, None]).astype(float)
    P = np.eye(m)
    out = []
    for _ in range(L):
        out.append(float((P * close).sum() / m))
        P = P @ Bm
    return np.array(out)


def csl_horizons(L=30, tol=2e-3):
    """W*(s): smallest window L such that the lag-1..L NB return profile of
    class s differs from every other class by more than tol."""
    R = np.array([nb_return_rates(csl_graph(s), L) for s in CSL_SKIPS])
    W = {}
    for a, s in enumerate(CSL_SKIPS):
        W[s] = None
        for Lw in range(1, L + 1):
            if all(np.abs(R[a, :Lw] - R[b, :Lw]).max() > tol
                   for b in range(len(CSL_SKIPS)) if b != a):
                W[s] = Lw
                break
    return W, R


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


def inv_softplus(x):
    return x + torch.log(-torch.expm1(-x))


class WalkSSM(nn.Module):
    """Single diagonal complex SSM layer + pointwise MLP readout.

    Real state dimension S = 2M (M complex modes; real and imaginary parts
    are both fed to the readout).  The readout sees the full state and the
    current token, matching the setting of the theorem.

      LTI       : a = exp(dt * lam),   dt a learned per-mode constant
      selective : a_t = exp(dt_t * lam), dt_t = softplus(W x_t + b)
      both      : s_t = a_t * s_{t-1} + dt_t * (B x_t)
    """

    def __init__(self, d_in, S, selective, d_hidden=128, n_out=1, pool=False, d_cmp=16,
                 aux=False, kind="diag", poles=None, use_cmp=True):
        super().__init__()
        assert S >= 1
        # kind: "diag"   learned diagonal complex SSM (LTI or selective)
        #       "frozen" diagonal SSM with poles fixed to `poles` [(|a|, arg a)]
        #       "shift"  nilpotent shift register s_t = (b^T x_t, ..., b^T x_{t-S+1})
        self.kind, self.use_cmp, self.S = kind, use_cmp, S
        # M complex modes (2 real dims each) + one real mode if S is odd
        M, R = S // 2, S % 2
        self.Mc, self.M = M, M + R
        self.selective, self.pool = selective, pool
        im_mask = torch.ones(M + R)
        if R:
            im_mask[-1] = 0.0
        self.register_buffer("im_mask", im_mask)
        M = M + R
        # S4D-Lin initialisation
        self.log_neg_re = nn.Parameter(torch.full((M,), math.log(0.5)))
        self.im = nn.Parameter(math.pi * torch.arange(M, dtype=torch.float32))
        dt0 = torch.exp(torch.empty(M).uniform_(math.log(1e-2), math.log(0.5)))
        if selective:
            self.dt_proj = nn.Linear(d_in, M)
            nn.init.normal_(self.dt_proj.weight, std=0.02)
            with torch.no_grad():
                self.dt_proj.bias.copy_(inv_softplus(dt0))
        else:
            self.log_dt = nn.Parameter(torch.log(dt0))
        self.B_re = nn.Parameter(torch.randn(M, d_in) / math.sqrt(d_in))
        self.B_im = nn.Parameter(torch.randn(M, d_in) / math.sqrt(d_in))
        if kind == "frozen":
            assert R == 0 and len(poles) == M
            pa = torch.tensor(poles, dtype=torch.float32)
            self.register_buffer("a_fixed", torch.polar(pa[:, 0], pa[:, 1]))
        if kind == "shift":
            self.b_shift = nn.Parameter(torch.randn(d_in) / math.sqrt(d_in))
        # comparison features (q(x_t) - k(s_t))^2: lets the readout test token
        # equality easily; still a pointwise function g(s_t, x_t)
        self.q = nn.Linear(d_in, d_cmp)
        self.k = nn.Linear(S, d_cmp)
        f = S + d_in + (d_cmp if use_cmp else 0)
        self.mlp = nn.Sequential(
            nn.LayerNorm(f), nn.Linear(f, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden), nn.GELU())
        self.head = nn.Linear(d_hidden, n_out)
        # optional per-token self-supervised head (graph-agnostic revisit label)
        self.aux_head = nn.Linear(d_hidden, aux) if aux else None

    def states(self, x):
        """Real state sequence (B, T, S)."""
        Bsz, T, _ = x.shape
        if self.kind == "shift":
            u = F.pad(x @ self.b_shift, (self.S - 1, 0))      # (B, T + S - 1)
            return u.unfold(1, self.S, 1).flip(-1)            # lags 0..S-1
        lam = torch.complex(-torch.exp(self.log_neg_re), self.im * self.im_mask)
        Bc = torch.complex(self.B_re, self.B_im * self.im_mask[:, None])
        u = torch.complex(x, torch.zeros_like(x)) @ Bc.T      # (B, T, M)
        if self.kind == "frozen":
            a = self.a_fixed.expand(Bsz, T, self.M)
            b = u
        else:
            if self.selective:
                dt = F.softplus(self.dt_proj(x))             # (B, T, M)
            else:
                dt = torch.exp(self.log_dt).expand(Bsz, T, self.M)
            a = torch.exp(dt * lam)
            b = dt * u
        h = torch.zeros(Bsz, self.M, dtype=b.dtype)
        hs = []
        for t in range(T):
            h = a[:, t] * h + b[:, t]
            hs.append(h)
        H = torch.stack(hs, 1)
        return torch.cat([H.real, H.imag[..., :self.Mc]], -1)

    def forward(self, x):                                    # x: (B, T, d)
        Hr = self.states(x)
        feats = [Hr, x]
        if self.use_cmp:
            feats.append((self.q(x) - self.k(Hr)) ** 2)
        z = self.mlp(torch.cat(feats, -1))
        if not self.pool:
            return self.head(z)
        out = self.head(z.mean(1))
        if self.aux_head is not None:
            return out, self.aux_head(z)
        return out


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def auroc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    npos, nneg = labels.sum(), (~labels).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def cosine_lr(opt, step, total, base, warm=100):
    lr = base * min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * step / total))
    for g in opt.param_groups:
        g["lr"] = lr


# --------------------------------------------------------------------------
# Jobs (each runs in its own worker process)
# --------------------------------------------------------------------------


def _setup(seed, threads):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def job_delay(cfg):
    """Fit Re(sum_m c_m a_m^j) to delta_{j,k} for j = 1..J-1 (the j = 0 tap is
    free, as the readout sees the current token); best of restarts."""
    rng = _setup(cfg["seed"], cfg["threads"])
    k, S, J = cfg["k"], cfg["S"], cfg["J"]
    M = S // 2
    target = torch.zeros(J)
    target[k] = 1.0
    j = torch.arange(J, dtype=torch.float32)
    best, best_poles = float("inf"), None
    for r in range(cfg["restarts"]):
        torch.manual_seed(cfg["seed"] * 100 + r)
        rho = nn.Parameter(torch.randn(M) * 0.5 + 1.0)
        th = nn.Parameter(2 * math.pi * torch.rand(M))
        c = nn.Parameter(torch.randn(M, 2) * 0.1)
        opt = torch.optim.Adam([rho, th, c], lr=cfg["lr"])
        for step in range(cfg["steps"]):
            cosine_lr(opt, step, cfg["steps"], cfg["lr"])
            mag = torch.sigmoid(rho)[:, None] ** j[None]      # (M, J)
            ang = th[:, None] * j[None]
            h = (c[:, :1] * mag * torch.cos(ang) - c[:, 1:] * mag * torch.sin(ang)).sum(0)
            loss = ((h[1:] - target[1:]) ** 2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if math.sqrt(loss.item()) < best:
            best = math.sqrt(loss.item())
            best_poles = [[float(m), float(t)] for m, t in
                          zip(torch.sigmoid(rho).detach(), th.detach())]
    return {**cfg, "rel_l2_error": best, "poles": best_poles}


def job_phase(cfg):
    rng = _setup(cfg["seed"], cfg["threads"])
    n, T, k, df = cfg["n"], cfg["T"], cfg["k"], cfg["d_feat"]
    g_rng = np.random.default_rng(12345)          # same graph pools for every run
    train_pool = np.stack([random_4regular(n, g_rng) for _ in range(cfg["n_train_graphs"])])
    test_pool = np.stack([random_4regular(n, g_rng) for _ in range(cfg["n_test_graphs"])])
    poles = None
    if cfg.get("kind") == "frozen":
        with open(cfg["poles_from"]) as fh:
            poles = json.load(fh)["poles"]
    model = WalkSSM(df, cfg["S"], cfg["selective"], d_hidden=cfg["d_hidden"],
                    kind=cfg.get("kind", "diag"), poles=poles, use_cmp=cfg.get("use_cmp", True))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)

    def batch(pool, B, r):
        g = r.integers(0, len(pool), B)
        W = nb_walks(pool, g, T, r)
        X = torch.from_numpy(walk_features(W, n, df, r))
        Y = torch.from_numpy((W[:, k:] == W[:, :-k]).astype(np.float32))
        return X, Y

    t0 = time.time()
    for step in range(cfg["steps"]):
        cosine_lr(opt, step, cfg["steps"], cfg["lr"])
        X, Y = batch(train_pool, cfg["batch"], rng)
        logits = model(X)[:, k:, 0]
        p = Y.mean().clamp(1e-3, 0.5)
        loss = F.binary_cross_entropy_with_logits(logits, Y, pos_weight=(1 - p) / p)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    ev = np.random.default_rng(cfg["seed"] + 999)
    S_, L_ = [], []
    with torch.no_grad():
        for _ in range(cfg["eval_batches"]):
            X, Y = batch(test_pool, cfg["batch"], ev)
            S_.append(model(X)[:, k:, 0].flatten().numpy())
            L_.append(Y.flatten().numpy())
        extra = {}
        if cfg["selective"]:
            # how input-dependent the learned step size is: std over tokens / mean, per mode
            dt = F.softplus(model.dt_proj(X))
            extra["dt_cv"] = float((dt.std((0, 1)) / dt.mean((0, 1))).mean())
    sc, lb = np.concatenate(S_), np.concatenate(L_)
    pred = sc > 0
    bal = 0.5 * (pred[lb == 1].mean() + (~pred[lb == 0]).mean())
    return {**cfg, "auroc": auroc(sc, lb), "bal_acc": float(bal),
            "pos_rate": float(lb.mean()), "seconds": time.time() - t0, **extra}


def job_csl(cfg):
    rng = _setup(cfg["seed"], cfg["threads"])
    T, df = cfg["T"], cfg["d_feat"]
    pool = np.stack([csl_graph(s) for s in CSL_SKIPS])
    C = len(CSL_SKIPS)
    aux_w, aux_win = cfg.get("aux_weight", 0.0), cfg.get("aux_window", 16)
    model = WalkSSM(df, cfg["S"], cfg["selective"], d_hidden=cfg["d_hidden"],
                    n_out=C, pool=True, aux=aux_win if aux_w > 0 else 0)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)

    def batch(y, r, walks=False, T=T):
        W = nb_walks(pool, y, T, r)
        X = torch.from_numpy(walk_features(W, CSL_N, df, r))
        return (X, W) if walks else X

    def revisit_lags(W):
        # multi-label 1[v_t == v_{t-l}], l = 1..aux_win: label-free and
        # graph-agnostic self-supervision, identical for every S
        lab = np.zeros(W.shape + (aux_win,), dtype=np.float32)
        mask = np.zeros_like(lab)
        for l in range(1, aux_win + 1):
            lab[:, l:, l - 1] = W[:, l:] == W[:, :-l]
            mask[:, l:, l - 1] = 1.0
        return torch.from_numpy(lab), torch.from_numpy(mask)

    t0 = time.time()
    for step in range(cfg["steps"]):
        cosine_lr(opt, step, cfg["steps"], cfg["lr"])
        y = rng.integers(0, C, cfg["batch"])
        X, Wk = batch(y, rng, walks=True)
        out = model(X)
        if aux_w > 0:
            out, aux_logits = out
            lab, mask = revisit_lags(Wk)
            aux = F.binary_cross_entropy_with_logits(aux_logits, lab, weight=mask, reduction="sum")
            loss = F.cross_entropy(out, torch.from_numpy(y)) + aux_w * aux / mask.sum()
        else:
            loss = F.cross_entropy(out, torch.from_numpy(y))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    # test: each test graph is judged from `eval_walks` independent walks
    model.eval()
    y = np.repeat(np.arange(C), cfg["eval_per_class"])

    def evaluate(T_eval, seed_off):
        ev = np.random.default_rng(cfg["seed"] + seed_off)
        logit_sum = torch.zeros(len(y), C)
        by_walks = {}
        with torch.no_grad():
            for w in range(1, cfg["eval_walks"] + 1):
                for i in range(0, len(y), 250):
                    out = model(batch(y[i:i + 250], ev, T=T_eval))
                    logit_sum[i:i + 250] += out[0] if isinstance(out, tuple) else out
                if w & (w - 1) == 0:                         # 1, 2, 4, 8, 16 walks
                    p = logit_sum.argmax(1).numpy()
                    by_walks[w] = [float((p[y == c] == c).mean()) for c in range(C)]
        pred = logit_sum.argmax(1).numpy()
        return pred, by_walks

    pred, by_walks = evaluate(cfg.get("eval_T", T), 999)
    per_class = [float((pred[y == c] == c).mean()) for c in range(C)]
    res = {**cfg, "acc": float((pred == y).mean()), "per_class_acc": per_class,
           "per_class_acc_by_walks": {str(w): v for w, v in by_walks.items()},
           "seconds": time.time() - t0}
    if cfg.get("eval_T", T) != T:
        p2, _ = evaluate(T, 1999)
        res["acc_at_train_T"] = float((p2 == y).mean())
    return res


def job_csl_oracle(cfg):
    """Explicit-window reference (CRaWl-style): multinomial logistic regression
    on the exact revisit counts at lags 1..W, W = S.  By the theorem this
    is what an S-state LTI walk model can at best observe."""
    rng = _setup(cfg["seed"], cfg["threads"])
    T, W = cfg["T"], cfg["S"]
    pool = np.stack([csl_graph(s) for s in CSL_SKIPS])
    C = len(CSL_SKIPS)

    def feats(y, r):
        Wk = nb_walks(pool, y, T, r)
        return torch.tensor(np.stack([(Wk[:, l:] == Wk[:, :-l]).mean(1)
                                      for l in range(1, W + 1)], 1), dtype=torch.float32)

    y = rng.integers(0, C, cfg["n_train_walks"])
    X = feats(y, rng)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    lin = nn.Linear(W, C)
    opt = torch.optim.LBFGS(lin.parameters(), max_iter=300, line_search_fn="strong_wolfe")
    Xn, yt = (X - mu) / sd, torch.from_numpy(y)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lin(Xn), yt) + 1e-4 * lin.weight.pow(2).sum()
        loss.backward()
        return loss
    opt.step(closure)

    ev = np.random.default_rng(cfg["seed"] + 999)
    yv = np.repeat(np.arange(C), cfg["eval_per_class"])
    lp = torch.zeros(len(yv), C)
    with torch.no_grad():
        for _ in range(cfg["eval_walks"]):
            lp += F.log_softmax(lin((feats(yv, ev) - mu) / sd), -1)
    pred = lp.argmax(1).numpy()
    per_class = [float((pred[yv == c] == c).mean()) for c in range(C)]
    return {**cfg, "acc": float((pred == yv).mean()), "per_class_acc": per_class, "seconds": 0.0}


def job_csl_dispatch(cfg):
    return job_csl_oracle(cfg) if cfg["model"] == "oracle" else job_csl(cfg)


JOBS = {"delay": job_delay, "phase": job_phase, "csl": job_csl_dispatch}


def run_name(cfg):
    keys = [k for k in ("exp", "model", "S", "k", "seed") if k in cfg]
    return "_".join(f"{k}{cfg[k]}" if k != "exp" else str(cfg[k]) for k in keys)


def _dispatch(cfg):
    return JOBS[cfg["exp"]](cfg)


def run_all(cfgs, workers, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    todo = [c for c in cfgs if not os.path.exists(os.path.join(run_dir, run_name(c) + ".json"))]
    print(f"{len(cfgs) - len(todo)} runs cached, {len(todo)} to go, {workers} workers")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_dispatch, c): c for c in todo}
        for i, f in enumerate(as_completed(futs), 1):
            res = f.result()
            with open(os.path.join(run_dir, run_name(res) + ".json"), "w") as fh:
                json.dump(res, fh)
            metric = {k: res[k] for k in ("auroc", "acc", "rel_l2_error") if k in res}
            print(f"[{i}/{len(todo)} {time.time() - t0:7.0f}s] {run_name(res)} "
                  + " ".join(f"{k}={v:.4f}" for k, v in metric.items()), flush=True)
    out = []
    for c in cfgs:
        with open(os.path.join(run_dir, run_name(c) + ".json")) as fh:
            out.append(json.load(fh))
    return out


# --------------------------------------------------------------------------
# Configurations
# --------------------------------------------------------------------------


def make_configs(args):
    q = args.quick
    th = args.threads
    cfgs = {"delay": [], "phase": [], "csl": []}

    ks_delay = [4, 8, 12] if q else [4, 8, 12, 16]
    Ss_delay = [2, 4, 8, 12, 16, 20, 24, 32] if q else list(range(2, 34, 2)) + [40, 48, 56, 64]
    for k in ks_delay:
        for S in Ss_delay:
            cfgs["delay"].append(dict(exp="delay", model="lti", S=S, k=k, seed=0, J=64,
                                      steps=800 if q else 3000,
                                      restarts=2 if q else 4, lr=2e-2, threads=th))

    ks = [3, 4, 8] if q else [3, 4, 5, 6, 8, 10, 12, 14, 16]
    Ss = [2, 4, 8, 16] if q else [2, 4, 8, 12, 16, 24, 32, 48, 64]
    for model in ("lti", "selective"):
        for k in ks:
            for S in Ss:
                for seed in range(1 if q else args.phase_seeds):
                    cfgs["phase"].append(dict(
                        exp="phase", model=model, selective=model == "selective",
                        S=S, k=k, seed=seed, n=24, T=48 if q else 64, d_feat=8,
                        d_hidden=64 if q else 128, batch=64,
                        steps=200 if q else 2000, lr=3e-3, eval_batches=4 if q else 32,
                        n_train_graphs=256, n_test_graphs=64, threads=th))

    # controls (review): frozen optimal poles, nilpotent shift register (FIR),
    # and the readout without comparison features
    run_dir = os.path.join(args.out, "results_quick" if q else "results", "runs")
    base = dict(exp="phase", selective=False, n=24, T=48 if q else 64, d_feat=8,
                d_hidden=64 if q else 128, batch=64, steps=200 if q else 2000, lr=3e-3,
                eval_batches=4 if q else 32, n_train_graphs=256, n_test_graphs=64, threads=th)
    ks_ctl = [4, 8] if q else [4, 8, 12, 16]
    seeds_ctl = range(1 if q else args.phase_seeds)
    for k in ks_ctl:
        for S in sorted({k, 3 * k // 2, 2 * k}):
            if k not in ks_delay or S not in Ss_delay:
                continue
            dname = run_name(dict(exp="delay", model="lti", S=S, k=k, seed=0))
            for seed in seeds_ctl:
                cfgs["phase"].append(dict(base, model="frozen", kind="frozen", S=S, k=k,
                                          seed=seed, poles_from=os.path.join(run_dir, dname + ".json")))
        for S in sorted(set(Ss[:7]) | {k - 1, k, k + 1, k + 2}):
            for seed in seeds_ctl:
                cfgs["phase"].append(dict(base, model="shift", kind="shift", S=S, k=k, seed=seed))
    for S in Ss:
        for seed in seeds_ctl:
            cfgs["phase"].append(dict(base, model="lti_nocmp", use_cmp=False, S=S, k=8, seed=seed))

    Ss_csl = [2, 4, 6, 8, 12, 16] if q else [2, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64]
    for model in ("lti", "selective"):
        for S in Ss_csl:
            for seed in range(1 if q else args.csl_seeds):
                cfgs["csl"].append(dict(
                    exp="csl", model=model, selective=model == "selective", S=S,
                    seed=seed, T=64 if q else 128, eval_T=64 if q else 256,
                    d_feat=8, d_hidden=64 if q else 128,
                    batch=64, steps=200 if q else 3000, lr=3e-3,
                    eval_per_class=20 if q else 100, eval_walks=2 if q else 16,
                    threads=th))
    # CSL controls: per-lag self-supervised revisit loss, and pooled-walk curves
    csl_base = dict(exp="csl", selective=False, T=64 if q else 128, eval_T=64 if q else 256,
                    d_feat=8, d_hidden=64 if q else 128, batch=64, steps=200 if q else 3000,
                    lr=3e-3, eval_per_class=20 if q else 100, eval_walks=2 if q else 16,
                    threads=th)
    for S in ([8] if q else [8, 16, 32, 64]):
        for seed in range(1 if q else args.csl_seeds):
            cfgs["csl"].append(dict(csl_base, model="lti_aux", S=S, seed=seed,
                                    aux_weight=1.0, aux_window=16))
    for S in ([2] if q else [2, 4, 8]):
        for seed in range(1 if q else args.csl_seeds):
            cfgs["csl"].append(dict(csl_base, model="lti_pool", S=S, seed=seed))
    for S in Ss_csl:                           # explicit-window reference, W = S
        for seed in range(1 if q else args.csl_seeds):
            cfgs["csl"].append(dict(
                exp="csl", model="oracle", S=S, seed=seed, T=64 if q else 256,
                n_train_walks=2000 if q else 20000,
                eval_per_class=20 if q else 100, eval_walks=2 if q else 16, threads=th))
    return cfgs


# --------------------------------------------------------------------------
# Figures (IEEE two-column: column 3.5 in, text width 7.16 in, 8 pt text)
# --------------------------------------------------------------------------

COL_W, TEXT_W = 3.5, 7.16
PHASE_THR = 0.95
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, VIOLET, INK2 = "#e87ba4", "#4a3aa7", "#52514e"
MODEL_STYLE = {
    "lti": dict(color=BLUE, marker="o", label="LTI, learned poles"),
    "selective": dict(color=ORANGE, marker="s", label=r"Input-dependent step $\Delta_t$"),
    "oracle": dict(color=INK2, marker="^", ls="--", label=r"Exact-count reference, lags $\leq S$"),
    "frozen": dict(color=AQUA, marker="D", label="LTI, frozen fitted poles"),
    "shift": dict(color=VIOLET, marker="v", label=r"Shift register (nilpotent $A$)"),
    "lti_aux": dict(color=MAGENTA, marker="P", ls="-.", label="LTI + per-lag revisit loss"),
}


def surrogate_auroc(lam, n=400000, seed=1):
    """Optimal AUROC for H1: (u, v) correlated with squared correlation lam vs
    H0: independent, u, v ~ N(0, 1): the i.i.d.-Gaussian surrogate in which a
    state explains a fraction lam of the variance of the lagged token."""
    if lam <= 0:
        return 0.5
    if lam >= 1:
        return 1.0
    rng = np.random.default_rng(seed)
    r = math.sqrt(lam)
    u0, v0, z = rng.standard_normal((3, n))
    u1 = r * v0 + math.sqrt(1 - lam) * z

    def llr(u, v):
        return -(u * u - 2 * r * u * v + v * v) / (2 * (1 - lam)) + (u * u + v * v) / 2
    sc = np.concatenate([llr(u0, rng.standard_normal(n)), llr(u1, v0)])
    return auroc(sc, np.r_[np.zeros(n), np.ones(n)])


def ieee_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Nimbus Roman No9 L",
                       "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.6, "lines.linewidth": 1.2, "lines.markersize": 3.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.direction": "in", "ytick.direction": "in",
        "axes.grid": True, "grid.linewidth": 0.3, "grid.color": "#d0d0d0",
        "legend.frameon": False, "legend.handlelength": 1.8,
        "pdf.fonttype": 42, "ps.fonttype": 42,          # embed TrueType (PDF eXpress)
        "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.015,
    })
    return plt


def cell_means(res, model, key="auroc"):
    """{(k, S): [values over seeds]} for one model."""
    out = {}
    for r in res:
        if r["model"] == model:
            out.setdefault((r["k"], r["S"]), []).append(r[key])
    return out


def ci95(v):
    from scipy import stats
    v = np.asarray(v, float)
    if len(v) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v)))


def fig_delay(res, out):
    plt = ieee_style()
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    ks = sorted({r["k"] for r in res})
    colors = [BLUE, ORANGE, AQUA, VIOLET]
    markers = ["o", "s", "^", "D"]
    for i, k in enumerate(ks):
        rr = sorted((r for r in res if r["k"] == k), key=lambda r: r["S"])
        ax.semilogy([r["S"] for r in rr], [max(r["rel_l2_error"], 1e-6) for r in rr],
                    color=colors[i % 4], marker=markers[i % 4], label=f"$k={k}$")
        Sb = np.arange(1, k)
        ax.semilogy(Sb, np.sqrt(1 - Sb / k), color=colors[i % 4], ls=":", lw=1.0)
    ax.plot([], [], color=INK2, ls=":", lw=1.0, label=r"bound $\sqrt{1-S/k}$")
    ax.set_xlabel("Real state dimension $S$")
    ax.set_ylabel(r"$\|h-\delta_k\|_2$ (best fit)")
    ax.set_ylim(8e-4, 1.5)
    ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=0.7,
              handlelength=1.4, fontsize=6.5)
    fig.savefig(out)
    plt.close(fig)


def fig_phase(res, out):
    plt = ieee_style()
    from matplotlib.colors import LinearSegmentedColormap
    seq = LinearSegmentedColormap.from_list("blue_seq", ["#f4f8fd", "#9cc3ee", BLUE, "#0d3b73"])
    div = LinearSegmentedColormap.from_list("div", [BLUE, "#e9e8e4", ORANGE])
    lti, sel = cell_means(res, "lti"), cell_means(res, "selective")
    ks = sorted({k for k, _ in lti})
    Ss = sorted({S for _, S in lti})
    Z = np.array([[np.mean(lti[(k, S)]) for k in ks] for S in Ss])
    D = np.array([[np.mean(sel[(k, S)]) - np.mean(lti[(k, S)]) for k in ks] for S in Ss])
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.2), sharey=True,
                             gridspec_kw=dict(wspace=0.32))
    im0 = axes[0].imshow(Z, origin="lower", aspect="auto", cmap=seq, vmin=0.5, vmax=1.0)
    im1 = axes[1].imshow(D, origin="lower", aspect="auto", cmap=div, vmin=-0.1, vmax=0.1)
    for i in range(len(Ss)):
        for j in range(len(ks)):
            axes[0].text(j, i, f"{Z[i, j]:.2f}".lstrip("0"), ha="center", va="center",
                         fontsize=6, color="white" if Z[i, j] > 0.8 else "#0b0b0b")
            dtxt = ".00" if abs(D[i, j]) < 0.005 else f"{D[i, j]:+.2f}".replace("0.", ".")
            axes[1].text(j, i, dtxt, ha="center", va="center",
                         fontsize=6, color="#0b0b0b")
    for ax in axes:
        xs, ys = [], []
        for j, k in enumerate(ks):
            i0 = next((i for i, S in enumerate(Ss) if S >= k), len(Ss))
            xs += [j - 0.5, j + 0.5]
            ys += [i0 - 0.5, i0 - 0.5]
        ax.plot(xs, ys, color="#e34948", lw=1.4, ls="--", label="$S=k$ (Hankel threshold)")
        ax.set_xticks(range(len(ks)), [str(k) for k in ks])
        ax.set_yticks(range(len(Ss)), [str(S) for S in Ss])
        ax.set_xlabel("Revisit lag $k$")
        ax.grid(False)
    xe, ye = [], []
    for j in range(len(ks)):
        i0 = next((i for i in range(len(Ss)) if Z[i, j] >= PHASE_THR), len(Ss))
        xe += [j - 0.5, j + 0.5]
        ye += [i0 - 0.5, i0 - 0.5]
    axes[0].plot(xe, ye, color="#0b0b0b", lw=1.0, ls=":", label=f"AUROC $\\geq$ {PHASE_THR}")
    axes[0].set_title("(a) LTI, learned poles: test AUROC")
    axes[1].set_title(r"(b) Input-dependent $\Delta_t$ minus LTI")
    axes[0].set_ylabel("Real state dimension $S$")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.46, 0.99), ncol=2, fontsize=7)
    for im, ax, lab in ((im0, axes[0], "AUROC"), (im1, axes[1], r"$\Delta$AUROC")):
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        cb.set_label(lab)
        cb.outline.set_linewidth(0.4)
    fig.savefig(out)
    plt.close(fig)


def fig_ratio(res, out):
    """AUROC against S/k for all variants, with the Gaussian-surrogate curve."""
    plt = ieee_style()
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    lam = np.r_[np.linspace(0.02, 0.98, 25)]
    ax.plot(lam, [surrogate_auroc(x, n=100000) for x in lam], color="#0b0b0b", lw=1.2,
            label=r"Gaussian surrogate, $\lambda=S/k$")
    ax.axvline(1.0, color="#e34948", ls="--", lw=1.0)
    for model, size, alpha in (("lti", 9, 0.55), ("selective", 9, 0.55), ("frozen", 16, 0.9),
                               ("shift", 14, 0.9)):
        cm = cell_means(res, model)
        if not cm:
            continue
        st = MODEL_STYLE[model]
        x = np.array([S / k for (k, S) in cm])
        y = np.array([np.mean(v) for v in cm.values()])
        ax.scatter(x, y, s=size, color=st["color"], marker=st["marker"], alpha=alpha,
                   edgecolors="white", linewidths=0.3, label=st["label"], zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16],
                  ["1/8", "1/4", "1/2", "1", "2", "4", "8", "16"])
    ax.set_xlabel("State per lag, $S/k$")
    ax.set_ylabel("Test AUROC (lag-$k$ revisit)")
    ax.set_ylim(0.45, 1.02)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6,
              handletextpad=0.3, columnspacing=0.8)
    fig.savefig(out)
    plt.close(fig)


def phase_thresholds(res, thr=PHASE_THR, models=("lti", "selective")):
    rows = []
    for model in models:
        cm = cell_means(res, model)
        for k in sorted({k for k, _ in cm}):
            Ss = sorted(S for kk, S in cm if kk == k)
            S_emp = next((S for S in Ss if np.mean(cm[(k, S)]) >= thr), None)
            rows.append(dict(model=model, k=k, thr=thr, hankel_S=k, empirical_S=S_emp,
                             ratio=None if S_emp is None else S_emp / k))
    return rows


def phase_stats(res):
    """Everything the manuscript reports about the token-level experiments."""
    from scipy import stats
    out = {}
    lti, sel = cell_means(res, "lti"), cell_means(res, "selective")
    cells = sorted(lti)
    d_cell = np.array([np.mean(sel[c]) - np.mean(lti[c]) for c in cells])
    out["n_seeds"] = int(min(len(v) for v in lti.values()))
    out["sel_minus_lti_mean"] = float(d_cell.mean())
    out["sel_minus_lti_ci95"] = ci95(d_cell)
    out["sel_minus_lti_absmean"] = float(np.abs(d_cell).mean())
    out["sel_minus_lti_range"] = [float(d_cell.min()), float(d_cell.max())]
    out["sel_minus_lti_wilcoxon_p"] = float(stats.wilcoxon(d_cell).pvalue)
    out["sel_minus_lti_ttest_p"] = float(stats.ttest_1samp(d_cell, 0).pvalue)
    seed_sd = [np.std(v, ddof=1) for v in lti.values() if len(v) > 1]
    out["lti_seed_sd_median"] = float(np.median(seed_sd)) if seed_sd else None
    ratios = {}
    for thr in (0.8, 0.9, 0.95, 0.99):
        for model in ("lti", "selective"):
            rr = [r["ratio"] for r in phase_thresholds(res, thr, (model,)) if r["ratio"]]
            ratios[f"{model}@{thr}"] = dict(median=float(np.median(rr)) if rr else None,
                                            min=float(min(rr)) if rr else None,
                                            max=float(max(rr)) if rr else None, n=len(rr))
    out["threshold_ratios"] = ratios
    for model in ("lti", "selective"):
        cm = cell_means(res, model)
        below = [np.mean(v) for (k, S), v in cm.items() if S < k]
        at = [np.mean(v) for (k, S), v in cm.items() if S == k]
        above = [np.mean(v) for (k, S), v in cm.items() if S >= 4 * k]
        out[f"{model}_auc_below"] = float(np.mean(below))
        out[f"{model}_auc_at_k"] = [float(min(at)), float(max(at))] if at else None
        out[f"{model}_auc_above4k"] = float(np.mean(above))
        # surrogate check: below-bound cells vs optimal Gaussian-surrogate AUROC
        viol = [(k, S, float(np.mean(v)), surrogate_auroc(S / k, n=100000))
                for (k, S), v in cm.items() if S < k]
        out[f"{model}_below_cells"] = len(viol)
        out[f"{model}_below_cells_under_surrogate"] = int(sum(o <= s + 0.01 for _, _, o, s in viol))
        out[f"{model}_below_max_excess"] = float(max(o - s for _, _, o, s in viol))
    dtcv = [r["dt_cv"] for r in res if r["model"] == "selective" and "dt_cv" in r]
    out["dt_cv_median"] = float(np.median(dtcv)) if dtcv else None
    out["dt_cv_max"] = float(np.max(dtcv)) if dtcv else None
    for model in ("frozen", "shift", "lti_nocmp"):
        cm = cell_means(res, model)
        out[model] = {f"k{k}_S{S}": [float(np.mean(v)), ci95(v)] for (k, S), v in sorted(cm.items())}
    # matching LTI cells for the controls
    out["lti_cells"] = {f"k{k}_S{S}": [float(np.mean(v)), ci95(v)] for (k, S), v in sorted(lti.items())}
    return out


def _csl_table(res, models=("lti", "selective", "oracle", "lti_aux")):
    tab = {}
    for model in models:
        for S in sorted({r["S"] for r in res}):
            rr = [r for r in res if r["model"] == model and r["S"] == S]
            if rr:
                tab[(model, S)] = dict(acc=[r["acc"] for r in rr],
                                       per_class=np.mean([r["per_class_acc"] for r in rr], 0))
    return tab


def fig_csl_acc(res, out):
    plt = ieee_style()
    tab = _csl_table(res)
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    for model in ("lti", "selective", "lti_aux", "oracle"):
        Ss = sorted(S for (m, S) in tab if m == model)
        if not Ss:
            continue
        mu = np.array([np.mean(tab[(model, S)]["acc"]) for S in Ss])
        ci = np.array([ci95(tab[(model, S)]["acc"]) for S in Ss])
        st = MODEL_STYLE[model]
        ax.plot(Ss, 100 * mu, color=st["color"], marker=st["marker"], ls=st.get("ls", "-"),
                label=st["label"])
        ax.fill_between(Ss, 100 * (mu - ci), 100 * (mu + ci), color=st["color"], alpha=0.15, lw=0)
    ax.axhline(10, color=INK2, ls=":", lw=0.8)
    ax.text(62, 11, "1-WL (provably chance)", ha="right", va="bottom", fontsize=6, color=INK2)
    ax.set_xlabel("Real state dimension $S$")
    ax.set_ylabel("CSL test accuracy (%)")
    ax.set_ylim(0, 102)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6)
    fig.savefig(out)
    plt.close(fig)


def csl_onsets(res, horizons, thr=0.8):
    tab = _csl_table(res, ("lti", "selective", "oracle"))
    rows = []
    for model in ("lti", "selective", "oracle"):
        Ss = sorted(S for (m, S) in tab if m == model)
        for c, s in enumerate(CSL_SKIPS):
            onset = next((S for S in Ss if tab[(model, S)]["per_class"][c] >= thr), None)
            rows.append(dict(model=model, skip=s, horizon=horizons[s], onset_S=onset))
    return rows


def csl_stats(res):
    from scipy import stats
    out = {}
    tab = _csl_table(res)
    out["acc"] = {f"{m}_S{S}": [float(np.mean(v["acc"])), ci95(v["acc"]), len(v["acc"])]
                  for (m, S), v in sorted(tab.items())}
    # paired LTI vs input-dependent step over all (S, seed)
    pairs = []
    for r in res:
        if r["model"] == "lti":
            q = [x for x in res if x["model"] == "selective" and x["S"] == r["S"]
                 and x["seed"] == r["seed"]]
            if q:
                pairs.append(q[0]["acc"] - r["acc"])
    if pairs:
        out["sel_minus_lti_mean"] = float(np.mean(pairs))
        out["sel_minus_lti_ci95"] = ci95(pairs)
        out["sel_minus_lti_wilcoxon_p"] = float(stats.wilcoxon(pairs).pvalue)
    at_train = [(r["model"], r["S"], r["acc"], r["acc_at_train_T"]) for r in res
                if "acc_at_train_T" in r]
    if at_train:
        out["acc_eval256_minus_eval128_mean"] = float(np.mean([a - b for _, _, a, b in at_train]))
    pool = {}
    for r in res:
        if r["model"] == "lti_pool":
            for w, pc in r["per_class_acc_by_walks"].items():
                pool.setdefault((r["S"], int(w)), []).append(pc)
    out["pool_class_r2"] = {f"S{S}_w{w}": float(np.mean([pc[0] for pc in v]))
                            for (S, w), v in sorted(pool.items())}
    out["pool_overall"] = {f"S{S}_w{w}": float(np.mean(v))
                           for (S, w), v in sorted(pool.items())}
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def write_csv(path, rows, cols):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", nargs="+", default=["delay", "phase", "csl"], choices=list(JOBS))
    ap.add_argument("--workers", type=int, default=8, help="parallel worker processes")
    ap.add_argument("--threads", type=int, default=1, help="torch threads per worker")
    ap.add_argument("--phase-seeds", type=int, default=3)
    ap.add_argument("--csl-seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="tiny smoke-test configuration")
    ap.add_argument("--plots-only", action="store_true")
    ap.add_argument("--out", default=".", help="root for results/ and figures/")
    args = ap.parse_args()

    res_dir = os.path.join(args.out, "results_quick" if args.quick else "results")
    fig_dir = os.path.join(args.out, "figures_quick" if args.quick else "figures")
    run_dir = os.path.join(res_dir, "runs")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    cfgs = make_configs(args)
    summary = {"env": dict(torch=torch.__version__, numpy=np.__version__,
                           workers=args.workers, threads=args.threads, quick=args.quick)}

    horizons, R = csl_horizons()
    summary["csl_horizon_W"] = {str(s): w for s, w in horizons.items()}
    summary["csl_nb_return_rates_lag1_12"] = {str(s): R[i, :12].round(5).tolist()
                                              for i, s in enumerate(CSL_SKIPS)}
    print("CSL distinguishability horizons W*(s):", horizons)
    summary["csl_horizon_tau_invariant"] = {
        str(t): [csl_horizons(tol=t)[0][s] for s in CSL_SKIPS] for t in (1e-4, 1e-3, 3e-3, 5e-3)}

    results = {}
    for exp in args.exp:
        print(f"\n=== {exp}: {len(cfgs[exp])} runs ===")
        if args.plots_only:
            results[exp] = [json.load(open(os.path.join(run_dir, run_name(c) + ".json")))
                            for c in cfgs[exp]
                            if os.path.exists(os.path.join(run_dir, run_name(c) + ".json"))]
        else:
            results[exp] = run_all(cfgs[exp], args.workers, run_dir)

    if results.get("delay"):
        fig_delay(results["delay"], os.path.join(fig_dir, "fig1_delay_realization.pdf"))
        write_csv(os.path.join(res_dir, "delay.csv"), results["delay"], ["k", "S", "rel_l2_error"])
        summary["delay"] = [{k: r[k] for k in ("k", "S", "rel_l2_error")} for r in results["delay"]]
    if results.get("phase"):
        ph = results["phase"]
        fig_phase(ph, os.path.join(fig_dir, "fig2_phase_diagram.pdf"))
        fig_ratio(ph, os.path.join(fig_dir, "fig4_auroc_vs_ratio.pdf"))
        cols = ["model", "k", "S", "seed", "auroc", "bal_acc", "pos_rate", "dt_cv", "seconds"]
        write_csv(os.path.join(res_dir, "phase.csv"), ph, cols)
        summary["phase"] = [{k: r.get(k) for k in cols} for r in ph]
        summary["phase_thresholds"] = phase_thresholds(ph)
        write_csv(os.path.join(res_dir, "phase_thresholds.csv"), summary["phase_thresholds"],
                  ["model", "k", "thr", "hankel_S", "empirical_S", "ratio"])
        summary["phase_stats"] = phase_stats(ph)
    if results.get("csl"):
        cs = results["csl"]
        fig_csl_acc(cs, os.path.join(fig_dir, "fig3_csl_accuracy.pdf"))
        rows = csl_onsets(cs, horizons)
        write_csv(os.path.join(res_dir, "csl_onsets.csv"), rows,
                  ["model", "skip", "horizon", "onset_S"])
        cols = ["model", "S", "seed", "acc", "acc_at_train_T", "seconds"]
        write_csv(os.path.join(res_dir, "csl.csv"), cs, cols)
        summary["csl"] = [{**{k: r.get(k) for k in cols}, "per_class_acc": r["per_class_acc"]}
                          for r in cs]
        summary["csl_onsets"] = rows
        summary["csl_stats"] = csl_stats(cs)

    with open(os.path.join(res_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"\nFigures -> {fig_dir}/   Results -> {res_dir}/summary.json")


if __name__ == "__main__":
    main()
