# Hindsight planning on the gain/load-fault plant: after a factual T-step trajectory, find the action sequence that would have
# reached the endpoint attained by another sequence under the same persistent fault and the same fresh kicks.
import numpy as np, torch, neural
def step(s,a,xi,eta): return 0.8*s+a*(1+xi[...,0])+xi[...,1]*np.sin(2*s)+eta
def main(sf,seed=0,T=3,Etr=6000,E=400,C=2000,tol=0.1):
    r=np.random.default_rng(seed)
    xi=np.stack([0.4*r.standard_normal(Etr),0.3*r.standard_normal(Etr)],1); s=r.uniform(-1,1,Etr); X=[];R=[]
    for t in range(T):
        a=r.uniform(-1,1,Etr); s2=step(s,a,xi,sf*r.standard_normal(Etr)); X.append(np.stack([s,a],-1)); R.append(s2-0.8*s-a); s=s2
    model=neural.fit(np.stack(X,1),np.stack(R,1)[...,None],m=4,epochs=50)
    xi=np.stack([0.4*r.standard_normal(E),0.3*r.standard_normal(E)],1); eta=sf*r.standard_normal((E,T)); s0=r.uniform(-1,1,E)
    AF=r.uniform(-1,1,(E,T)); AG=r.uniform(-1,1,(E,T))
    def roll(A):   # A (E,C,T) true replay
        s=np.repeat(s0[:,None],A.shape[1],1)
        for t in range(T): s=step(s,A[:,:,t],xi[:,None,:],eta[:,t:t+1])
        return s
    s=s0.copy(); X=[];R=[]
    for t in range(T): s2=step(s,AF[:,t],xi,eta[:,t]); X.append(np.stack([s,AF[:,t]],-1)); R.append(s2-0.8*s-AF[:,t]); s=s2
    X=np.stack(X,1); R=np.stack(R,1); goal=roll(AG[:,None])[:,0]
    xh,inn=neural.posterior(model,X,R[...,None]); inn=inn[...,0].numpy()
    cand=r.uniform(-1,1,(E,C,T)); res={}
    for meth in("model","replay","post","twin"):
        s=np.repeat(s0[:,None],C,1)
        for t in range(T):
            d=0
            if meth=="replay": d=R[:,t:t+1]
            if meth in("post","twin"):
                with torch.no_grad(): F=model(torch.tensor(np.stack([s,cand[:,:,t]],-1),dtype=torch.float32))[...,0,:]
                d=(F*xh[:,None,:]).sum(-1).numpy()+(inn[:,t:t+1] if meth=="twin" else 0)
            s=0.8*s+cand[:,:,t]+d
        best=cand[np.arange(E),np.argmin((s-goal[:,None])**2,1)]
        res[meth]=np.mean(np.abs(roll(best[:,None])[:,0]-goal)<tol)
    res["factual sequence"]=np.mean(np.abs(roll(AF[:,None])[:,0]-goal)<tol)
    print(f"1-D hindsight planning, T={T}, fresh sd {sf}: "+"  ".join(f"{k} {v:.3f}" for k,v in res.items()),flush=True)
for sf in (0.0,0.1,0.3): main(sf)
