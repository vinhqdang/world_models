# Forward replanning (the paper's Table 17 protocol in miniature): 2 behaviour prefix transitions, then 3 planned
# transitions towards a goal; persistent / fresh / none test conditions; one training law with both components.
import numpy as np, torch, neural
def err(s,a): return 0.08*np.sin(3*s+2*a)                  # imperfection of the factual predictor
def mu(s,a): return 0.8*s+a+err(s,a)                       # imperfect factual predictor (truth has no err term)
def step(s,a,xi,eta): return 0.8*s+a*(1+xi[:,0])+xi[:,1]*np.sin(2*s)+eta
def main(seed=0,Etr=6000,E=4000,sf=0.2,tol=0.1):
    r=np.random.default_rng(seed); T=5
    xi=np.stack([0.4*r.standard_normal(Etr),0.3*r.standard_normal(Etr)],1); s=r.uniform(-1,1,Etr); X=[];R=[]
    for t in range(T):
        a=r.uniform(-1,1,Etr); s2=step(s,a,xi,sf*r.standard_normal(Etr)); X.append(np.stack([s,a],-1)); R.append(s2-mu(s,a)); s=np.clip(s2,-3,3)
    X=np.stack(X,1); R=np.stack(R,1)[...,None]
    model=neural.fit(X,R,m=4,epochs=80)
    n=model.logn.exp().item(); agrid=np.linspace(-1,1,201)
    print(f"learned nugget {n:.4f} (fresh var {sf*sf:.4f} + predictor error var {0.08**2/2:.4f})")
    for cond in("persistent","fresh","none","mixed(train law)"):
        xi=np.stack([0.4*r.standard_normal(E),0.3*r.standard_normal(E)],1)*(cond in("persistent","mixed(train law)"))
        fs=sf*(cond in("fresh","mixed(train law)")); goal=r.uniform(-0.8,0.8,E); s0=r.uniform(-1,1,E)
        pre=r.uniform(-1,1,(E,2)); eta=fs*r.standard_normal((E,T)); out={}
        for meth in("pred","reuse-last","ours-filter","oracle"):
            s=s0.copy(); Xh=[];Rh=[]
            for t in range(T):
                if t<2: a=pre[:,t]
                else:
                    S=np.repeat(s[:,None],201,1); A=np.repeat(agrid[None],E,0)
                    if meth=="oracle": pred=0.8*S+A*(1+xi[:,:1])+xi[:,1:]*np.sin(2*S)
                    else:
                        pred=mu(S,A)
                        if meth=="reuse-last": pred=pred+Rh[-1][:,None]
                        if meth=="ours-filter":
                            xh,_=neural.posterior(model,np.stack(Xh,1),np.stack(Rh,1)[...,None])
                            with torch.no_grad(): F=model(torch.tensor(np.stack([S,A],-1),dtype=torch.float32))[...,0,:]
                            pred=pred+(F*xh[:,None,:]).sum(-1).numpy()
                    a=agrid[np.argmin((pred-goal[:,None])**2,1)]
                s2=step(s,a,xi,eta[:,t]); Xh.append(np.stack([s,a],-1)); Rh.append(s2-mu(s,a)); s=s2
            out[meth]=np.mean(np.abs(s-goal)<tol)
        print(f"  {cond:18s} success after 3 replans (|s-goal|<{tol}): "+"  ".join(f"{k} {v:.3f}" for k,v in out.items()),flush=True)
main()
