import numpy as np
from t2 import feats,r2,krig
from t3 import coef_fit,lift
from t4 import polymean
rng=np.random.default_rng(4)
for th,sig in [(0.3,0.1),(0.6,0.2),(1.2,0.3)]:
    N=lambda n: sig*np.clip(rng.standard_normal((n,1)),-2.5,2.5)
    F=lambda a,T: np.cos(th*a+T)
    n=80000; a=rng.uniform(-1,1,(n,1)); y=F(a,N(n)); mu=polymean(a,y); r=y-mu(a)
    nt=6000; at=rng.uniform(-1,1,(nt,1)); ap=rng.uniform(-1,1,(nt,1)); Tt=N(nt); yt,yc=F(at,Tt),F(ap,Tt); rt=yt-mu(at)
    out=dict(oracle=r2(mu(ap),yc),RR=r2(yt+mu(ap)-mu(at),yc))
    # true BLUP per pair by MC
    E=N(4000)[:,0]; Ya=np.cos(th*at+E[None]); Yp=np.cos(th*ap+E[None])
    ca=Ya-Ya.mean(1,keepdims=True); cp=Yp-Yp.mean(1,keepdims=True)
    L=(ca*cp).mean(1)/(ca**2).mean(1); out['trueBLUP']=r2(mu(ap)+L[:,None]*rt,yc)
    for rank in [1,2]:
        C,iu,p1=coef_fit(a,r); l,G=lift(C,iu,p1,rank)
        for t in [1e-3,1e-1]:
            out[f'PK_r{rank}_tau{t}']=r2(mu(ap)+krig(G,np.zeros(1),at,ap,rt,t*np.mean(r**2)),yc)
    # heteroscedastic nugget version: features (1,a) for gain + free nugget estimated = M00 treated as non-shared
    # two-atom posterior (oracle structural knowledge of fold)
    print(th,sig,{k:round(float(v),3) for k,v in out.items()})
