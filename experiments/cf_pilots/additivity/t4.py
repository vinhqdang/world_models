import numpy as np
from t2 import feats,r2,krig
from t3 import coef_fit,lift
rng=np.random.default_rng(3)
def polymean(a,y,deg=5):
    X=np.concatenate([a**k for k in range(deg+1)],1); B=np.linalg.lstsq(X,y,rcond=None)[0]
    return lambda b: np.concatenate([b**k for k in range(deg+1)],1)@B
def run(name,F,sampleT,phi=None,rank=1,n=60000,nt=6000,coords=None):
    a=rng.uniform(-1,1,(n,1)); T=sampleT(n); y=F(a,T)
    mu=polymean(a,y); r=y-mu(a)
    at=rng.uniform(-1,1,(nt,1)); ap=rng.uniform(-1,1,(nt,1)); Tt=sampleT(nt); yt,yc=F(at,Tt),F(ap,Tt); rt=yt-mu(at)
    out=dict(oracle=r2(mu(ap),yc),RR=r2(yt+mu(ap)-mu(at),yc),rho=((yc-yt-mu(ap)+mu(at))**2).sum()/((yc-mu(ap))**2).sum())
    for nm,ph in phi.items():
        fa=ph(a,mu); C,iu,p1=coef_fit(fa[:,1:],r); l,G=lift(C,iu,p1,rank)
        tau=1e-3*np.mean(r**2)
        pk=mu(ap)+krig(G,np.zeros(y.shape[1]),ph(at,mu)[:,1:],ph(ap,mu)[:,1:],rt,tau)
        out['PK_'+nm]=r2(pk,yc)
        if y.shape[1]>1: out['PK_'+nm+'_c0']=r2(pk[:,:1],yc[:,:1])
    if y.shape[1]>1: out['RR_c0']=r2((yt+mu(ap)-mu(at))[:,:1],yc[:,:1]); out['oracle_c0']=r2(mu(ap)[:,:1],yc[:,:1])
    print(name,{k:round(float(v),3) for k,v in out.items()})
phis={'affine_a':lambda a,mu: np.c_[np.ones(len(a)),a], 'mean_feat':lambda a,mu: np.c_[np.ones(len(a)),mu(a)-mu(0*a)],'a_and_mean':lambda a,mu: np.c_[np.ones(len(a)),a,mu(a)-mu(0*a)]}
for th,sig in [(0.3,0.1),(0.6,0.2),(1.2,0.3)]:
    N=lambda n,sig=sig: sig*np.clip(rng.standard_normal((n,1)),-2.5,2.5)
    run(f"cos pitch th={th} sig={sig}",lambda a,T,th=th: np.cos(th*a+T),N,phis)
    run(f"sin pitch th={th} sig={sig}",lambda a,T,th=th: np.sin(th*a+T),N,phis)
    run(f"(cos,sin) th={th} sig={sig}",lambda a,T,th=th: np.c_[np.cos(th*a+T),np.sin(th*a+T)],N,phis,rank=2)
