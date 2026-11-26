import numpy as np, sys, time
from scipy.optimize import nnls
from boatT import *
def bank(r,J=200):
    lam=np.exp(r.uniform(np.log(1.5),np.log(80),J)); th=r.uniform(0,2*np.pi,J)
    k=(2*np.pi/lam)[:,None]*np.stack([np.cos(th),np.sin(th)],-1); om=r.uniform(-0.5,0.5,J)
    kh=k/np.linalg.norm(k,axis=1,keepdims=True); kp=np.stack([kh[:,1],-kh[:,0]],-1)
    K=np.concatenate([k,k]); OM=np.concatenate([om,om]); Dv=np.concatenate([kh,kp])
    # uniform atoms
    ua=np.array([[1,0],[0,1],[np.sqrt(.5),np.sqrt(.5)],[np.sqrt(.5),-np.sqrt(.5)]])
    K=np.concatenate([K,np.zeros((4,2))]); OM=np.concatenate([OM,np.zeros(4)]); Dv=np.concatenate([Dv,ua])
    return K,OM,Dv
def kern_atoms(dp,dt,K,OM):  # (...,2),(...) -> (...,J)
    return np.cos(dp@K.T+dt[...,None]*OM)
def n_rows(E,T): return E*T*(T-1)/2
def fit_kernel(X,Tm,R,K,OM,Dv,lams,chunk=200):
    # off-diagonal (s != t) second-moment matching: r_s r_t^T ~ sum_j w_j d_j d_j^T cos(k_j.(x_s-x_t)+om_j (s-t)), w>=0
    E,T,_=R.shape; J=len(K); iu=np.triu_indices(T,1)
    DD=np.einsum('ja,jb->jab',Dv,Dv).reshape(J,4)
    AtA=np.zeros((J,J)); Atb=np.zeros(J)
    for c in range(0,E,chunk):
        x=X[c:c+chunk]; r=R[c:c+chunk]; tm=Tm[c:c+chunk]
        dp=x[:,iu[0]]-x[:,iu[1]]; dt=tm[:,iu[0]]-tm[:,iu[1]]
        C=kern_atoms(dp,dt,K,OM).reshape(-1,J)           # (n,J)
        y=np.einsum('npa,npb->npab',r[:,iu[0]],r[:,iu[1]]).reshape(-1,4)
        # design row for entry (ab): C[n,j]*DD[j,ab]
        G4=DD@DD.T                                         # (J,J) sum over ab
        AtA+=(C.T@C)*G4; Atb+=((C.T@y)*DD).sum(1)
    lamr=1e-6*np.trace(AtA)/J
    Lc=np.linalg.cholesky(AtA+lamr*np.eye(J))
    c=(K**2).sum(1)+OM**2
    ws=[]
    for lam in lams:
        w,_=nnls(Lc.T,np.linalg.solve(Lc,Atb-0.5*lam*n_rows(E,T)*c),maxiter=20*J); ws.append(w)
    return ws
def Kmat(XA,TA,XB,TB,w,K,OM,Dv):
    # XA (E,n,2) XB (E,m,2) -> (E,n,2,m,2)
    dp=XA[:,:,None]-XB[:,None]; dt=TA[:,:,None]-TB[:,None]
    C=kern_atoms(dp,dt,K,OM)
    DD=(w[:,None]*np.einsum('ja,jb->jab',Dv,Dv).reshape(len(w),4))
    E,n,m=C.shape[:3]
    return (C.reshape(-1,C.shape[-1])@DD).reshape(E,n,m,2,2).transpose(0,1,3,2,4)
def predict(method,p0,psiF,PF,psiC,kpar,nug):
    """returns counterfactual positions (E,T+1,2). psiF factual headings, PF factual positions, psiC alternative."""
    E,T=psiF.shape; uF=U(psiF); uC=U(psiC)
    R=PF[:,1:]-PF[:,:-1]-uF                       # residuals with exact means mu(p,a)=p+u(a)
    XF=PF[:,:-1]+0.5*uF; TF=np.tile(np.arange(T)+0.5,(E,1))
    P=[p0]; p=p0
    if method in("twin","post"):
        w,K,OM,Dv=kpar
        Kff=Kmat(XF,TF,XF,TF,w,K,OM,Dv).reshape(E,2*T,2*T)+nug*np.eye(2*T)
        alpha=np.linalg.solve(Kff,R.reshape(E,2*T,1))[...,0]
        gF=np.einsum('eij,ej->ei',Kff-nug*np.eye(2*T),alpha).reshape(E,T,2)   # smoothed persistent part at factual points
        inn=R-gF                                                              # smoothed innovation (fresh estimate)
    for t in range(T):
        if method=="model": d=0
        elif method=="replay": d=R[:,t]
        elif method=="disc": d=R[:,t]*np.exp(-(((p+0.5*uC[:,t])-XF[:,t])**2).sum(-1,keepdims=True)/(2*kpar**2))
        else:
            xq=(p+0.5*uC[:,t])[:,None]; tq=np.full((E,1),t+0.5)
            kq=Kmat(xq,tq,XF,TF,w,K,OM,Dv).reshape(E,2,2*T)
            g=np.einsum('eaj,ej->ea',kq,alpha)
            d=g+(inn[:,t] if method=="twin" else 0)
        p=p+uC[:,t]+d; P.append(p)
    return np.stack(P,1)
def oracle_kernel(r,S,G,L,D,M=500):
    lam=r.uniform(L,2*L,M); th=r.uniform(0,2*np.pi,M)
    k=(2*np.pi/lam)[:,None]*np.stack([np.cos(th),np.sin(th)],-1); kh=k/np.linalg.norm(k,axis=1,keepdims=True)
    K=np.concatenate([k,np.zeros((1,2))]); OM=np.concatenate([r.uniform(-D,D,M),[0]])
    Dv=np.concatenate([np.stack([kh[:,1],-kh[:,0]],-1),[[1,0]]])
    w=np.concatenate([np.full(M,G*G/2/M),[S*S*0.3943]])
    return w,K,OM,Dv
def run(S,G,L,D,sf,T=8,Etr=3000,Ete=1500,query="indep",seed=0,verbose=True):
    r=np.random.default_rng(seed)
    # factual training episodes
    wtr=world(r,Etr,T,S,G,L,D,sf); p0=r.uniform(-1,1,(Etr,2)); psi=behaviour(r,Etr,T)
    P=rollout(wtr,p0,psi); u=U(psi); R=P[:,1:]-P[:,:-1]-u; X=P[:,:-1]+0.5*u; Tm=np.tile(np.arange(T)+0.5,(Etr,1))
    K,OM,Dv=bank(r); t0=time.time(); nv=Etr//5
    lams=[0,1e-4,1e-3,1e-2,1e-1,1]
    ws=fit_kernel(X[nv:],Tm[nv:],R[nv:],K,OM,Dv,lams); best=None
    for lam,w in zip(lams,ws):   # factual selection: leave-one-time-out prediction of r_t from the other residuals of the episode
        kd=np.einsum('j,ja,ja->a',w,Dv,Dv); ng=max(1e-4,float(np.mean(R[nv:]**2)-kd.mean()))
        Kff=Kmat(X[:nv],Tm[:nv],X[:nv],Tm[:nv],w,K,OM,Dv).reshape(nv,2*T,2*T)+ng*np.eye(2*T)
        Ki=np.linalg.inv(Kff); a=np.einsum('eij,ej->ei',Ki,R[:nv].reshape(nv,2*T))
        loo=np.mean((a/np.einsum('eii->ei',Ki))**2)      # GP leave-one-out residual (per coordinate)
        if verbose: print(f"   lam={lam:g} LOO={loo:.5f} nug={ng:.4f} K0={kd.round(3)}",flush=True)
        if best is None or loo<best[0]: best=(loo,w,ng,kd)
    _,w,nug,kdiag=best
    # test
    wte=world(r,Ete,T,S,G,L,D,sf); p0=r.uniform(-1,1,(Ete,2)); psiF=behaviour(r,Ete,T)
    psiC=behaviour(r,Ete,T) if query=="indep" else psiF+np.radians(query)*r.standard_normal((Ete,T))
    PF=rollout(wte,p0,psiF); PC=rollout(wte,p0,psiC)
    base=PC-predict("model",p0,psiF,PF,psiC,None,0)
    out={}
    ko=oracle_kernel(r,S,G,L,D)
    for name,meth,kp,ng in[("replay","replay",None,0),("post(learned)","post",(w,K,OM,Dv),nug),("twin(learned)","twin",(w,K,OM,Dv),nug),
                        ("post(oracleK)","post",ko,max(sf*sf,1e-4)),("twin(oracleK)","twin",ko,max(sf*sf,1e-4))]:
        e=PC-predict(meth,p0,psiF,PF,psiC,kp,ng)
        out[name]=[(e[:,t]**2).sum(-1).mean()/(base[:,t]**2).sum(-1).mean() for t in (1,2,4,T)]
    bd=None
    for ell in (1,2,4,8,16,32):
        e=PC-predict("disc",p0,psiF,PF,psiC,ell,0); v=[(e[:,t]**2).sum(-1).mean()/(base[:,t]**2).sum(-1).mean() for t in (1,2,4,T)]
        if bd is None or v[-1]<bd[-1]: bd=v
    out["disc-replay(best ell)"]=bd
    if verbose:
        print(flush=True,end="");print(f"S={S} G={G} L={L} D={D} sf={sf} query={query} | fitted nugget={nug:.4f} (true {sf*sf:.4f}) K(0)diag={kdiag.round(3)} fit {time.time()-t0:.0f}s")
        for k,v in out.items(): print(f"   {k:22s} rel.risk vs model-only at t=1,2,4,{T}: "+" ".join(f"{x:6.3f}" for x in v))
    return out
if __name__=="__main__":
    for L in (1000,10,5,2.5):
        for sf in (0.0,0.3):
            run(0.5,0.4,L,0.3,sf,Etr=2500,Ete=1000)
    run(0.,0.,10,0.3,0.4)            # fresh only: must reduce to replay
    run(0.5,0.4,5,0.3,0.3,query=30)  # local counterfactuals
