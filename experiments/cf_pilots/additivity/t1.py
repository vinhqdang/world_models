import numpy as np
rng=np.random.default_rng(0)
def feats(a):  # a: (n,p) -> (n,p+1)
    return np.c_[np.ones(len(a)),a]
def quadfeats(a):
    f=feats(a); p1=f.shape[1]; iu=np.triu_indices(p1)
    return f[:,iu[0]]*f[:,iu[1]], iu, p1
def fit_mean(a,y,deg=1):
    X=feats(a) if deg==1 else np.c_[feats(a),a**2,a**3]
    return np.linalg.lstsq(X,y,rcond=None)[0], (lambda b,B: (feats(b) if deg==1 else np.c_[feats(b),b**2,b**3])@B)
def fit_M(a,r):
    """scalar outcome: r^2 ~ f(a)^T M f(a); returns symmetric M"""
    Q,iu,p1=quadfeats(a)
    c=np.linalg.lstsq(Q,r**2,rcond=None)[0]
    M=np.zeros((p1,p1)); M[iu]=c; M=(M+M.T)/2+np.diag(np.diag(M))/2  # off-diag coef = 2 M_jk
    # project to PSD
    w,V=np.linalg.eigh(M); M=(V*np.clip(w,0,None))@V.T
    return M
def r2(pred,truth): return 1-((pred-truth)**2).sum()/((truth-truth.mean())**2).sum()
def dial(lam,n=20000,ntest=4000,noise="tanh"):
    # Y = c( g a + s T[(1-lam)+lam u] ), u=a/amax, a uniform [-1,1]
    m=(1-lam)**2+lam**2/3; v=0.394
    s=1/(1+np.sqrt(3*v*m/0.98)); g=0.98*(1-s); amax=1.0
    def F(a,T): return g*a+s*T*((1-lam)+lam*a/amax)
    a=rng.uniform(-1,1,(n,1)); T=np.tanh(rng.standard_normal(n)); y=F(a[:,0],T)
    B,pm=fit_mean(a,y); r=y-pm(a,B); M=fit_M(a,r)
    at=rng.uniform(-1,1,(ntest,1)); ap=rng.uniform(-1,1,(ntest,1)); Tt=np.tanh(rng.standard_normal(ntest))
    yt=F(at[:,0],Tt); yc=F(ap[:,0],Tt)
    mu_a,mu_ap=pm(at,B),pm(ap,B); rt=yt-mu_a
    fa,fp=feats(at),feats(ap)
    Kaa=np.einsum('ni,ij,nj->n',fa,M,fa); Kpa=np.einsum('ni,ij,nj->n',fp,M,fa); Kpp=np.einsum('ni,ij,nj->n',fp,M,fp)
    tau=1e-3*np.mean(Kaa)
    out={}
    out['oracle_no_outcome']=r2(g*ap[:,0],yc)
    out['residual_reuse']=r2(yt+mu_ap-mu_a,yc)
    out['sd_ratio_unsigned']=r2(mu_ap+np.sqrt(Kpp/(Kaa+tau))*rt,yc)
    out['polarized_kriging']=r2(mu_ap+Kpa/(Kaa+tau)*rt,yc)
    # CiC quantile transport using a kernel-free exact-law version: rank of T under |s(a)|: equals unsigned sd ratio for symmetric T
    out['rho_RR']=np.var(yc-yt-(mu_ap-mu_a))/np.var(yc-g*ap[:,0])
    return out
for lam in [0,0.5,1.0]:
    print(lam,{k:round(v,3) for k,v in dial(lam).items()})
