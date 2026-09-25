import numpy as np, sys
sys.path.insert(0,'.')
from walk_ssm_barrier import random_4regular, nb_walks, auroc
g=np.random.default_rng(12345)
n=24
train=np.stack([random_4regular(n,g) for _ in range(256)]); test=np.stack([random_4regular(n,g) for _ in range(64)])
r=np.random.default_rng(0)
T=64
def walks(pool,B):
    return nb_walks(pool, r.integers(0,len(pool),B), T, r)
Wtr=walks(train,20000); Wte=walks(test,4000)
def feats(W,k,L):
    # indicators at lags 1..L at position t (t>=k), label at lag k
    t0=k
    X=np.stack([(W[:,t0:]==W[:,t0-l:T-l]) for l in range(1,L+1)],-1).reshape(-1,L)
    y=(W[:,t0:]==W[:,:T-t0]).reshape(-1)
    return X,y
print("window-oracle AUROC (Bayes table on exact revisit indicators at lags 1..L), L=S-1 (shift register w/ S states)")
for k in [3,4,5,6,8,10,12,14,16]:
    row=[]
    for S in [2,4,8,12,16]:
        if S>=k: row.append('   - '); continue
        L=S-1 if S>1 else 0
        if L==0: row.append(' 0.50'); continue
        Xtr,ytr=feats(Wtr,k,L); Xte,yte=feats(Wte,k,L)
        key_tr=Xtr.dot(1<<np.arange(L)); key_te=Xte.dot(1<<np.arange(L))
        num=np.bincount(key_tr,weights=ytr,minlength=1<<L); den=np.bincount(key_tr,minlength=1<<L)
        p=(num+0.5*ytr.mean())/(den+0.5)
        row.append(f'{auroc(p[key_te],yte):5.2f}')
    print('k',k,row)
# Gaussian surrogate: optimal AUROC when one canonical correlation^2 = lam=S/k
rng=np.random.default_rng(1); N=400000
print("Gaussian-surrogate AUROC upper value at explained variance lam=S/k")
for lam in [1/8,1/4,1/3,1/2,2/3,3/4]:
    rr=np.sqrt(lam)
    u0,v0=rng.standard_normal(N),rng.standard_normal(N)
    z=rng.standard_normal(N); u1=v0*rr+np.sqrt(1-lam)*z; v1=v0
    llr=lambda u,v: -(u*u-2*rr*u*v+v*v)/(2*(1-lam))+(u*u+v*v)/2
    sc=np.concatenate([llr(u0,v0),llr(u1,v1)]); lb=np.r_[np.zeros(N),np.ones(N)]
    print(f'lam={lam:.3f} AUROC*={auroc(sc,lb):.3f}')
