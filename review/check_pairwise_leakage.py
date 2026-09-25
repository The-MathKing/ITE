"""Structural leakage with ALL pairwise coincidences inside the window.

The first leakage check only used 1[v_t = v_{t-l}], l < S. A register holding
lags 0..S-1 also sees coincidences among past tokens, e.g. 1[v_{t-j} = v_{t-k+j}]
(a closed walk that leaves and returns along the same edge path). Here a logistic
regression on every pairwise indicator among window positions t, ..., t-S+1
predicts y_t = 1[v_t = v_{t-k}].
"""
import sys, numpy as np, torch
sys.path.insert(0, '.')
from walk_ssm_barrier import random_4regular, nb_walks, auroc
g = np.random.default_rng(12345); n, T = 24, 64
train = np.stack([random_4regular(n, g) for _ in range(256)])
test = np.stack([random_4regular(n, g) for _ in range(64)])
r = np.random.default_rng(0)
Wtr = nb_walks(train, r.integers(0, 256, 6000), T, r)
Wte = nb_walks(test, r.integers(0, 64, 2000), T, r)

def feats(W, k, S):
    t = np.arange(max(k, S - 1), T)
    cols = [(W[:, t - a] == W[:, t - b]) for a in range(S) for b in range(a + 1, S)]
    X = np.stack(cols, -1).reshape(-1, len(cols)).astype(np.float32) if cols else None
    y = (W[:, t] == W[:, t - k]).reshape(-1)
    return X, y

def fit_auc(k, S):
    Xtr, ytr = feats(Wtr, k, S); Xte, yte = feats(Wte, k, S)
    if Xtr is None: return 0.5
    Xtr, Xte = torch.tensor(Xtr), torch.tensor(Xte)
    lin = torch.nn.Linear(Xtr.shape[1], 1)
    opt = torch.optim.LBFGS(lin.parameters(), max_iter=200, line_search_fn="strong_wolfe")
    yt = torch.tensor(ytr, dtype=torch.float32)
    def cl():
        opt.zero_grad()
        l = torch.nn.functional.binary_cross_entropy_with_logits(lin(Xtr)[:, 0], yt) + 1e-4 * lin.weight.pow(2).sum()
        l.backward(); return l
    opt.step(cl)
    with torch.no_grad():
        return auroc(lin(Xte)[:, 0].numpy(), yte)

for k in [4, 8, 12, 16]:
    Ss = sorted({2, 4, 8, 12, k // 2, k // 2 + 1, k // 2 + 2, k - 2, k - 1, k} - {0, 1})
    print(f"k={k}", " ".join(f"S={S}:{fit_auc(k, S):.3f}" for S in Ss if S <= k), flush=True)
