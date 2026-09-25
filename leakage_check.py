#!/usr/bin/env python3
"""Does short-lag walk structure predict lag-k revisits?

Two checks on the token-level graph pools, both scored by test AUROC on unseen
graphs for the label y_t = 1[v_t = v_{t-k}]:

  current : Bayes-optimal lookup table on the revisit indicators between the
            current token and lags 1..S-1, for every (k, S < k) cell.
  pairwise: logistic regression on ALL pairwise coincidences among the S
            window positions t, ..., t-S+1 (what an S-state shift register
            could compare), for k in {4, 8, 12, 16} and S <= k.  A closed walk
            that leaves and returns along the same path shows up here
            (check contributed by the pre-submission review).

Values near 0.5 mean the i.i.d. Gaussian surrogate of Corollary 1 is a fair
model of the walk tokens. Writes results/leakage.json.
"""

import json
import os

import numpy as np
import torch

from walk_ssm_barrier import auroc, nb_walks, random_4regular


def main(n=24, T=64, n_train=20000, n_test=4000):
    g = np.random.default_rng(12345)                      # same pools as the rig
    train = np.stack([random_4regular(n, g) for _ in range(256)])
    test = np.stack([random_4regular(n, g) for _ in range(64)])
    r = np.random.default_rng(0)
    Wtr = nb_walks(train, r.integers(0, len(train), n_train), T, r)
    Wte = nb_walks(test, r.integers(0, len(test), n_test), T, r)

    def feats(W, k, L):
        X = np.stack([W[:, k:] == W[:, k - l:T - l] for l in range(1, L + 1)], -1).reshape(-1, L)
        y = (W[:, k:] == W[:, :T - k]).reshape(-1)
        return X, y

    out = {}
    for k in [3, 4, 5, 6, 8, 10, 12, 14, 16]:
        for S in [2, 4, 8, 12, 16]:
            if S >= k:
                continue
            L = S - 1
            Xtr, ytr = feats(Wtr, k, L)
            Xte, yte = feats(Wte, k, L)
            key_tr, key_te = Xtr.dot(1 << np.arange(L)), Xte.dot(1 << np.arange(L))
            num = np.bincount(key_tr, weights=ytr, minlength=1 << L)
            den = np.bincount(key_tr, minlength=1 << L)
            p = (num + 0.5 * ytr.mean()) / (den + 0.5)
            out[f"k{k}_S{S}"] = auroc(p[key_te], yte)
    vals = list(out.values())
    res = dict(cells=out, min=float(min(vals)), max=float(max(vals)))

    # pairwise coincidences inside the window (subsampled walks keep it fast)
    Ptr, Pte = Wtr[:6000], Wte[:2000]

    def pair_feats(W, k, S):
        t = np.arange(max(k, S - 1), T)
        cols = [W[:, t - a] == W[:, t - b] for a in range(S) for b in range(a + 1, S)]
        y = (W[:, t] == W[:, t - k]).reshape(-1)
        X = np.stack(cols, -1).reshape(-1, len(cols)).astype(np.float32) if cols else None
        return X, y

    def pair_auc(k, S):
        Xtr, ytr = pair_feats(Ptr, k, S)
        Xte, yte = pair_feats(Pte, k, S)
        if Xtr is None:
            return 0.5
        Xtr, Xte, yt = torch.tensor(Xtr), torch.tensor(Xte), torch.tensor(ytr, dtype=torch.float32)
        lin = torch.nn.Linear(Xtr.shape[1], 1)
        opt = torch.optim.LBFGS(lin.parameters(), max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(lin(Xtr)[:, 0], yt) \
                + 1e-4 * lin.weight.pow(2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            return auroc(lin(Xte)[:, 0].numpy(), yte)

    torch.manual_seed(0)
    pair = {}
    for k in [4, 8, 12, 16]:
        for S in sorted({2, 4, 8, 12, k - 3, k - 2, k - 1, k}):
            if 2 <= S <= k:
                pair[f"k{k}_S{S}"] = pair_auc(k, S)
    res["pairwise"] = pair
    for off, name in ((3, "le_km3"), (2, "km2"), (1, "km1"), (0, "k")):
        v = [a for key, a in pair.items()
             if (lambda k, S: (S <= k - 3) if off == 3 else (S == k - off))(
                 *map(int, key[1:].split("_S")))]
        res[f"pairwise_{name}"] = [float(min(v)), float(max(v))]
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "leakage.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
