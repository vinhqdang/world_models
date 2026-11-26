# Hindsight planning on the multi-step boat: after one factual trajectory (T transitions), find the heading sequence that
# would have reached a goal attained by another sequence under the same field and gusts. Scored by true replay.
import numpy as np, sys, time
from boatT import *
import pilot as P
def cf_roll(method,p0,psiC,XF,TF,R,alpha,inn,kpar):
    # p0 (B,2), psiC (B,T); factual quantities already tiled to B
    B,T=psiC.shape; uC=U(psiC); p=p0
    for t in range(T):
        if method=="model": d=0
        elif method=="replay": d=R[:,t]
        else:
            w,K,OM,Dv=kpar
            kq=P.Kmat((p+0.5*uC[:,t])[:,None],np.full((B,1),t+0.5),XF,TF,w,K,OM,Dv).reshape(B,2,2*T)
            d=np.einsum('eaj,ej->ea',kq,alpha)+(inn[:,t] if method=="twin" else 0)
        p=p+uC[:,t]+d
    return p
def plan(method,p0,goal,XF,TF,R,alpha,inn,kpar,r,psiF,C=256,iters=8,elite=32):
    E,T=psiF.shape
    mu=np.zeros((E,T)); sd=np.full((E,T),np.pi)          # same cold-started CEM for every planner
    rep=lambda a: np.repeat(a,C,0)
    for it in range(iters):
        cand=mu[:,None]+sd[:,None]*r.standard_normal((E,C,T)); cand[:,0]=mu
        end=cf_roll(method,rep(p0),cand.reshape(E*C,T),rep(XF),rep(TF),rep(R),None if alpha is None else rep(alpha),None if inn is None else rep(inn),kpar)
        cost=((end-rep(goal))**2).sum(-1).reshape(E,C); idx=np.argsort(cost,1)[:,:elite]
        el=np.take_along_axis(cand,idx[...,None],1); mu=el.mean(1); sd=el.std(1)+1e-3
        best=cand[np.arange(E),idx[:,0]]
    return best,np.sqrt(np.sort(cost,1)[:,0])
def run(S,G,L,D,sf,T=4,Etr=2500,E=100,seed=0,tol=0.3):
    r=np.random.default_rng(seed)
    wtr=world(r,Etr,T,S,G,L,D,sf); p0=r.uniform(-1,1,(Etr,2)); psi=behaviour(r,Etr,T)
    Pt=rollout(wtr,p0,psi); u=U(psi); R=Pt[:,1:]-Pt[:,:-1]-u; X=Pt[:,:-1]+0.5*u; Tm=np.tile(np.arange(T)+0.5,(Etr,1))
    K,OM,Dv=P.bank(r); nv=Etr//5; lams=[1e-3,1e-2]; ws=P.fit_kernel(X[nv:],Tm[nv:],R[nv:],K,OM,Dv,lams); w=ws[1]
    kd=np.einsum('j,ja,ja->a',w,Dv,Dv); nug=max(1e-4,float(np.mean(R**2)-kd.mean()))
    wte=world(r,E,T,S,G,L,D,sf); p0=r.uniform(-1,1,(E,2)); psiF=behaviour(r,E,T); psiG=behaviour(r,E,T)
    PF=rollout(wte,p0,psiF); goal=rollout(wte,p0,psiG)[:,-1]
    uF=U(psiF); R=PF[:,1:]-PF[:,:-1]-uF; XF=PF[:,:-1]+0.5*uF; TF=np.tile(np.arange(T)+0.5,(E,1))
    ko=P.oracle_kernel(r,S,G,L,D); res={}
    for name,meth,kp,ng in[("model-only","model",None,0),("replay (chained residual reuse)","replay",None,0),("post(learned K)","post",(w,K,OM,Dv),nug),
                           ("twin(learned K)","twin",(w,K,OM,Dv),nug),("twin(oracle K)","twin",ko,max(sf*sf,1e-4))]:
        alpha=inn=None
        if meth in("post","twin"):
            Kff=P.Kmat(XF,TF,XF,TF,*kp).reshape(E,2*T,2*T)+ng*np.eye(2*T)
            alpha=np.linalg.solve(Kff,R.reshape(E,2*T,1))[...,0]; inn=R-np.einsum('eij,ej->ei',Kff-ng*np.eye(2*T),alpha).reshape(E,T,2)
        rr=np.random.default_rng(123)
        a,pc=plan(meth,p0,goal,XF,TF,R,alpha,inn,kp,rr,psiF)
        d=np.linalg.norm(rollout(wte,p0,a)[:,-1]-goal,axis=-1); res[name]=(np.mean(d<tol),np.median(d),np.median(pc),np.mean(d<0.5))
    print(f"hindsight planning T={T} L={L} sf={sf} E={E} tol={tol}: "+" | ".join(f"{k}: {v[0]:.2f}/{v[3]:.2f} (med {v[1]:.2f}, planned {v[2]:.2f})" for k,v in res.items()),flush=True)
if __name__=="__main__":
    for L,sf in ((5,0.3),(5,0.0)): run(0.5,0.4,L,0.3,sf)
