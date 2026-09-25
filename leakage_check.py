#!/usr/bin/env python3
"""Does short-lag walk structure predict lag-k revisits?

For every (k, S < k) cell of the token-level grid, fit a Bayes-optimal lookup
table from the exact revisit indicators at lags 1..S-1 (what a shift register
with S states could see besides the current token) to the lag-k revisit label,
and report its test AUROC on unseen graphs. Values near 0.5 mean the i.i.d.
Gaussian surrogate of Corollary 1 is a fair model of the walk tokens.
Writes results/leakage.json.
"""

import json
import os

import numpy as np

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
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "leakage.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
