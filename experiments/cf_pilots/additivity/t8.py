import numpy as np
rng=np.random.default_rng(7)
def feats(a): return np.c_[np.ones(len(a)),a]
for lam in [0,0.25,0.5,0.75,1.0]:
    m=(1-lam)**2+lam**2/3; s=1/(1+np.sqrt(3*0.394*m/0.98)); g=0.98*(1-s)
    n=8000; a=rng.uniform(-1,1,(n,1)); T=np.tanh(rng.standard_normal(n)); y=g*a[:,0]+s*T*((1-lam)+lam*a[:,0])
    X=feats(a); r=y-X@np.linalg.lstsq(X,y,rcond=None)[0]
    Q=np.c_[np.ones(n),a,a**2]; c=np.linalg.lstsq(Q,r**2,rcond=None)[0]; M=np.array([[c[0],c[1]/2],[c[1]/2,c[2]]])
    w,V=np.linalg.eigh(M); M=(V*np.clip(w,0,None))@V.T
    a1=rng.uniform(-1,1,(20000,1)); a2=rng.uniform(-1,1,(20000,1)); f1,f2=feats(a1),feats(a2)
    num=np.einsum('ni,ij,nj->n',f2-f1,M,f2-f1).mean(); den=np.einsum('ni,ij,nj->n',f2,M,f2).mean()
    print(f"lam={lam}: factual-only predicted rho={num/den:.3f}  theory={2*lam**2/(3*m):.3f}")
