import sys, math, itertools, numpy as np
sys.path.insert(0, "/home/user/world_models")
from cwk import kernels as KR
pi, noise, seed = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]); rng = np.random.default_rng(seed)
d, R, n, nt = 2, 3, 80000, 4000
J = np.array([[1.0, 0.3], [-0.2, 0.8]])
def Gtrue(x):
    P = np.zeros((len(x), 2, 3)); P[:, 0, 0] = 1; P[:, 1, 0] = -1; P[:, 0, 1] = x[:, 0]; P[:, 1, 2] = x[:, 1]
    return np.einsum("de,ber->bdr", J, P)
A = 1/math.sqrt(pi/3)
def draw(m): return (rng.random((m, R)) < pi) * rng.uniform(-1, 1, (m, R)) * A
x = rng.uniform(-1, 1, (n, 2)); xi = draw(n); r = np.einsum("bdr,br->bd", Gtrue(x), xi) + noise * rng.standard_normal((n, d))
xt, xt2, xit = rng.uniform(-1, 1, (nt, 2)), rng.uniform(-1, 1, (nt, 2)), draw(nt)
rt = np.einsum("bdr,br->bd", Gtrue(xt), xit) + noise * rng.standard_normal((nt, d)); tgt = np.einsum("bdr,br->bd", Gtrue(xt2), xit)
def R2(p): return round(100 * (1 - ((p - tgt) ** 2).sum() / ((tgt - tgt.mean(0)) ** 2).sum()), 2)
def blup(Gs, Gt, nug):
    M = Gs @ Gs.transpose(0, 2, 1) + nug * np.eye(d)
    return np.einsum("bdr,br->bd", Gt, np.einsum("bdr,bd->br", Gs, np.linalg.solve(M, rt[..., None])[..., 0]))
Gs, Gt = Gtrue(xt), Gtrue(xt2)
print("linear BLUP", R2(blup(Gs, Gt, noise**2)))
# exact Bayes via per-pattern grid quadrature
ngrid = {1: 600, 2: 220, 3: 36}
num = np.zeros((nt, R)); den = np.zeros(nt); logs=[]
pats = [S for k in range(R+1) for S in itertools.combinations(range(R), k)]
# compute log evidence per pattern then combine
lev = []; post = []
for S in pats:
    k = len(S); lp = k*math.log(pi) + (R-k)*math.log(1-pi)
    if k == 0:
        ll = -0.5*(rt**2).sum(1)/noise**2; lev.append(lp+ll); post.append(np.zeros((nt,R))); continue
    g1 = (np.arange(ngrid[k])+0.5)/ngrid[k]*2*A - A
    U = np.array(list(itertools.product(g1, repeat=k)))  # (M,k)
    L = np.empty(nt); E = np.zeros((nt,R))
    for i in range(0, nt, 200):
        pred = np.einsum("bdk,mk->bmd", Gs[i:i+200][:,:,list(S)], U)
        ll = -0.5*((rt[i:i+200,None]-pred)**2).sum(-1)/noise**2
        mx = ll.max(1, keepdims=True); w = np.exp(ll-mx)
        L[i:i+200] = mx[:,0] + np.log(w.mean(1)); E[i:i+200][:, list(S)] = (w @ U)/w.sum(1, keepdims=True)
    lev.append(lp+L); post.append(E)
lev = np.stack(lev,1); W = np.exp(lev-lev.max(1,keepdims=True)); W/=W.sum(1,keepdims=True)
xb = np.einsum("bs,bsr->br", W, np.stack(post,1))
print("exact Bayes (true G, true prior)", R2(np.einsum("bdr,br->bd", Gt, xb)))
# GLR/best-subset fault isolation (true G, then HH G)
def glr(Gs, Gt, lam):
    best = np.full(nt, np.inf); xh = np.zeros((nt,R))
    for S in pats[:1+R+3]:
        if len(S)==0: rss = (rt**2).sum(1); u = np.zeros((nt,R))
        else:
            Gk = Gs[:,:,list(S)]; sol = np.linalg.lstsq if False else None
            M = Gk.transpose(0,2,1)@Gk + 1e-9*np.eye(len(S)); uk = np.linalg.solve(M, np.einsum("bdk,bd->bk", Gk, rt)[...,None])[...,0]
            uk = np.clip(uk, -A, A); rss = ((rt-np.einsum("bdk,bk->bd",Gk,uk))**2).sum(1); u = np.zeros((nt,R)); u[:,list(S)] = uk
        c = rss/noise**2 + lam*len(S); m = c<best; best[m]=c[m]; xh[m]=u[m]
    return np.einsum("bdr,br->bd", Gt, xh)
for lam in [2,4,6,8,12]: print("GLR best-subset lam", lam, "true G", R2(glr(Gs,Gt,lam)), end="; ")
print()
psi = lambda z: np.c_[np.ones(len(z)), z]
S_ = KR.hh_moments(psi(x), r); G0, _ = KR.bm_lift(S_, 3, d, R, restarts=8, seed=seed)
Gh = lambda z: np.einsum("bq,qdr->bdr", psi(z), G0)
print("HH BLUP", R2(blup(Gh(xt), Gh(xt2), noise**2)))
for lam in [2,4,6,8,12]: print("GLR lam", lam, "HH G (gauge-arbitrary)", R2(glr(Gh(xt),Gh(xt2),lam)), end="; ")
print()
