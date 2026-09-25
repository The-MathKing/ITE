import numpy as np, torch, sys, json
sys.path.insert(0,'.')
from walk_ssm_barrier import csl_graph, nb_return_rates, CSL_SKIPS
# 1) tau sensitivity of W*
R=np.array([nb_return_rates(csl_graph(s),30) for s in CSL_SKIPS])
def Wstar(tol):
    out=[]
    for a in range(10):
        w=None
        for L in range(1,31):
            if all(np.abs(R[a,:L]-R[b,:L]).max()>tol for b in range(10) if b!=a): w=L;break
        out.append(w)
    return out
for tol in [0,1e-4,5e-4,1e-3,2e-3,3e-3,5e-3,1e-2,2e-2]:
    print('tau',tol,Wstar(tol))
# 2) H2-optimal S-state LTI approx of z^-k (m>=1), dense A, scalar in/out
torch.manual_seed(0)
def best(S,k,L=160,restarts=6,steps=3000):
    b=1e9
    m=torch.arange(1,L+1)
    tgt=(m==k).float()
    for r in range(restarts):
        A=torch.nn.Parameter(torch.randn(S,S)*0.3/np.sqrt(S)); B=torch.nn.Parameter(torch.randn(S)); C=torch.nn.Parameter(torch.randn(S)*0.1)
        opt=torch.optim.Adam([A,B,C],lr=1e-2)
        for it in range(steps):
            # spectral normalisation for stability
            An=A/ torch.clamp(torch.linalg.matrix_norm(A,2),min=1.0)*0.999
            x=B; hs=[]
            for _ in range(L):
                x=An@x; hs.append(C@x)
            h=torch.stack(hs)
            loss=((h-tgt)**2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        b=min(b,loss.item())
    return b
for k in [4,6]:
    for S in range(1,k+1):
        print(f'k={k} S={S} bestMSE={best(S,k):.4f}  bound 1-S/k={1-S/k:.4f}  paper 1/k={1/k:.3f}',flush=True)
