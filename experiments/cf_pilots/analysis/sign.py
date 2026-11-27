import sys; sys.path.insert(0,"/home/user/world_models")
import numpy as np
from cwk import kernels as KR
rng=np.random.default_rng(0); n=200000; nt=40000
def draw(m): return rng.uniform(-np.sqrt(3),np.sqrt(3),(m,2))
def world(a,xi,flip): s=np.abs(a) if flip else a; return xi[:,0]+s*xi[:,1]
for flip in (0,1):
  a=rng.uniform(-1,1,n); xi=draw(n); y=world(a,xi,flip)+0.05*rng.standard_normal(n)
  at=rng.uniform(-1,1,nt); at2=rng.uniform(-1,1,nt); xit=draw(nt)
  yt=world(at,xit,flip)+0.05*rng.standard_normal(nt); tgt=world(at2,xit,flip)
  cross=np.sign(at)!=np.sign(at2)
  for name,bas in [("(1,a)",lambda z:np.c_[np.ones_like(z),z]),("(1,|a|)",lambda z:np.c_[np.ones_like(z),np.abs(z)]),("(1,a,|a|)",lambda z:np.c_[np.ones_like(z),z,np.abs(z)]),("(1,a,a2,a3)",lambda z:np.c_[np.ones_like(z),z,z**2,z**3])]:
    P=bas(a); q=P.shape[1]
    Phi,_=KR.product_basis(P); cond=np.linalg.cond(Phi.T@Phi)
    Sm=KR.hh_moments(P,y[:,None]); G,l=KR.bm_lift(Sm,q,1,2,restarts=8,seed=0)
    Gs=np.einsum("bq,qdr->bdr",bas(at),G); Gt=np.einsum("bq,qdr->bdr",bas(at2),G)
    A=Gs.transpose(0,2,1)@Gs+2.5e-3*np.eye(2); w=np.linalg.solve(A,np.einsum("bdr,bd->br",Gs,yt[:,None])[...,None])[...,0]
    p=np.einsum("bdr,br->bd",Gt,w)[:,0]
    R2=lambda m:100*(1-((p[m]-tgt[m])**2).sum()/((tgt[m]-tgt[m].mean())**2).sum())
    rr=yt  # residual reuse (means zero)
    print(f"truth={'|a|' if flip else 'a'} basis={name:12s} cond(PhiTPhi)={cond:9.2e} R2 all={R2(np.ones(nt,bool)):6.1f} cross-zero={R2(cross):6.1f} same-side={R2(~cross):6.1f}")
