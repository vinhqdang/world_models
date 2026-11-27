import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,"/home/user/world_models")
import numpy as np
from cwk import kernels as KR
rng=np.random.default_rng(1); n=200000; nt=40000; d=2; noise=0.05
J=np.array([[1.0,0.3],[-0.2,0.8]]); Jr=np.array([[0,-1.],[1,0]])
def G_syn(x):
    P=np.zeros((len(x),2,3)); P[:,0,0]=1;P[:,1,0]=-1;P[:,0,1]=x[:,0];P[:,1,2]=x[:,1]; return np.einsum("de,ber->bdr",J,P)
def G_rot(x,sgn): return np.eye(2)[None]+sgn*x[:,:1,None]*Jr[None]   # (I + s a1 J) xi, r=2
psi=lambda z:np.c_[np.ones(len(z)),z]
def run(Gf,R,name):
    x=rng.uniform(-1,1,(n,2)); xi=rng.uniform(-np.sqrt(3),np.sqrt(3),(n,Gf(x[:1]).shape[-1]))
    r=np.einsum("bdr,br->bd",Gf(x),xi)+noise*rng.standard_normal((n,d))
    xt,xt2=rng.uniform(-1,1,(nt,2)),rng.uniform(-1,1,(nt,2)); xit=rng.uniform(-np.sqrt(3),np.sqrt(3),(nt,xi.shape[1]))
    rt=np.einsum("bdr,br->bd",Gf(xt),xit)+noise*rng.standard_normal((nt,d)); tgt=np.einsum("bdr,br->bd",Gf(xt2),xit)
    R2=lambda p:round(100*(1-((p-tgt)**2).sum()/((tgt-tgt.mean(0))**2).sum()),2)
    def blupK(Kss,Kts,nug): return np.einsum("bij,bj->bi",Kts,np.linalg.solve(Kss+nug*np.eye(d),rt[...,None])[...,0])
    Go,Go2=Gf(xt),Gf(xt2); out={"oracle":R2(blupK(Go@Go.transpose(0,2,1),Go2@Go.transpose(0,2,1),noise**2))}
    S=KR.hh_moments(psi(x),r); q=3
    def Ksym(p1,p2):
        K=np.zeros((len(p1),d,d))
        for (j,k),M in S.items():
            K+= (p1[:,j]*p2[:,k])[:,None,None]*M if j==k else 0.5*(p1[:,j]*p2[:,k]+p1[:,k]*p2[:,j])[:,None,None]*M
        return K
    Css=Ksym(psi(xt),psi(xt)); w,V=np.linalg.eigh(Css); Css=np.einsum("bij,bj,bkj->bik",V,np.maximum(w,0),V)
    out["hh_sym"]=R2(blupK(Css,Ksym(psi(xt2),psi(xt)),noise**2))
    G,_=KR.bm_lift(S,q,d,R,restarts=8,seed=0); Gs=np.einsum("bq,qdr->bdr",psi(xt),G); Gt=np.einsum("bq,qdr->bdr",psi(xt2),G)
    out["hh_lift"]=R2(blupK(Gs@Gs.transpose(0,2,1),Gt@Gs.transpose(0,2,1),noise**2))
    out["rr"]=R2(rt)
    print(name,out)
#run(G_syn,3,"syn2-channel (dense uniform)")
#run(lambda x:G_rot(x,1),2,"rotation +J")
#run(lambda x:G_rot(x,-1),2,"rotation -J")
# thrust-misalignment fault: r = theta J a (+ isotropic offset driver xi0, xi1 along two fixed axes)
def G_mis(x, off):
    g=np.einsum("ij,bj->bi",Jr,x)[:,:,None]
    if off: g=np.concatenate([g,np.repeat(np.eye(2)[None],len(x),0)*0.5],-1)
    return g
for R in (1,2,3):
    run(lambda x:G_mis(x,0),R,f"misalign only R={R}")
for R in (3,4):
    run(lambda x:G_mis(x,1),R,f"misalign+offset R={R}")
