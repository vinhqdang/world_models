import numpy as np, torch
from t2 import feats,r2,krig
rng=np.random.default_rng(2)
def coef_fit(a,r):
    f=feats(a); p1=f.shape[1]; iu=np.triu_indices(p1); Q=f[:,iu[0]]*f[:,iu[1]]
    d=r.shape[1]; T=(r[:,:,None]*r[:,None,:]).reshape(len(r),-1)
    C=np.linalg.lstsq(Q,T,rcond=None)[0].reshape(-1,d,d); C=(C+C.transpose(0,2,1))/2
    return C,iu,p1
def lift(C,iu,p1,rank,restarts=8,iters=400):
    d=C.shape[1]; Ct=torch.tensor(C); J=torch.tensor(iu[0]); K=torch.tensor(iu[1])
    w=torch.where(J==K,1.0,1.0)
    best=None
    for s in range(restarts):
        torch.manual_seed(s)
        G=(C.max()**0.5*torch.randn(p1,d,rank,dtype=torch.float64)).requires_grad_()
        opt=torch.optim.LBFGS([G],lr=1,max_iter=iters,line_search_fn='strong_wolfe',tolerance_grad=1e-12,tolerance_change=1e-14)
        def loss_fn():
            P=G[J]@G[K].transpose(1,2); P=torch.where((J==K)[:,None,None],P,P+P.transpose(1,2))
            return ((P-Ct)**2).sum()
        def closure():
            opt.zero_grad(); l=loss_fn(); l.backward(); return l
        opt.step(closure); l=loss_fn().item()
        if best is None or l<best[0]: best=(l,G.detach().numpy().copy())
    return best
def run(name,F,p,n=40000,nt=5000,rank=1,sampleT=None,obs_noise=0.0):
    a=rng.uniform(-1,1,(n,p)); T=sampleT(n); y=F(a,T); y=y+obs_noise*rng.standard_normal(y.shape)
    X=feats(a); B=np.linalg.lstsq(X,y,rcond=None)[0]; r=y-X@B
    C,iu,p1=coef_fit(a,r); l,G=lift(C,iu,p1,rank)
    at=rng.uniform(-1,1,(nt,p)); ap=rng.uniform(-1,1,(nt,p)); Tt=sampleT(nt)
    yt,yc=F(at,Tt),F(ap,Tt); yt=yt+obs_noise*rng.standard_normal(yt.shape); mu_a,mu_p=feats(at)@B,feats(ap)@B; rt=yt-mu_a
    d=y.shape[1]; tau=1e-3*np.mean(r**2)+obs_noise**2
    res=dict(oracle=r2(mu_p,yc),RR=r2(yt+mu_p-mu_a,yc),PK=r2(mu_p+krig(G,np.zeros(d),at,ap,rt,tau),yc),fitloss=l/np.sum(C**2))
    print(name,{k:round(float(v),4) for k,v in res.items()})
tanhT=lambda n: np.tanh(rng.standard_normal((n,1)))
if __name__=="__main__":
    for lam in [0,0.5,1.0]:
        m=(1-lam)**2+lam**2/3; s=1/(1+np.sqrt(3*0.394*m/0.98)); g=0.98*(1-s); d=np.array([1.,-1.])
        F=lambda a,T,lam=lam,s=s,g=g: g*a+s*T*d*((1-lam)+lam*a)
        run(f"hand-like lam={lam}",F,2,sampleT=tanhT)
        run(f"hand-like lam={lam} +obs noise 0.02",F,2,sampleT=tanhT,obs_noise=0.02)
    F=lambda a,T: 0.5*a@np.array([[1.,0.3,0.1]])+T*(np.array([1.,0.,0.5])+a*np.array([0.,1.,-0.5]))
    run("rank1 nonparallel d=3,p=1",F,1,sampleT=tanhT)
    G0=rng.standard_normal((6,2)); G1=rng.standard_normal((6,2)); G2=rng.standard_normal((6,2))
    F=lambda a,T: T@G0.T+a[:,:1]*(T@G1.T)+a[:,1:2]*(T@G2.T)
    run("tall d=6 r=2 p=2",F,2,rank=2,sampleT=lambda n: rng.standard_normal((n,2)))
    for d_,r_ in [(2,2),(3,2),(4,2),(4,3),(5,3),(6,3),(3,3)]:
        G0=rng.standard_normal((d_,r_)); G1=rng.standard_normal((d_,r_))
        F=lambda a,T,G0=G0,G1=G1: T@G0.T+a[:,:1]*(T@G1.T)
        run(f"generic d={d_} r={r_} p=1 (2r-d={2*r_-d_})",F,1,rank=r_,sampleT=lambda n,r_=r_: rng.standard_normal((n,r_)),n=200000)
    for d_,r_ in [(2,2),(3,3)]:
        Gs=[rng.standard_normal((d_,r_)) for _ in range(4)]
        F=lambda a,T,Gs=Gs: T@Gs[0].T+sum(a[:,j:j+1]*(T@Gs[j+1].T) for j in range(3))
        run(f"generic d={d_} r={r_} p=3",F,3,rank=r_,sampleT=lambda n,r_=r_: rng.standard_normal((n,r_)),n=200000)
