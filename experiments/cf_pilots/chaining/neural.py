# Learned low-rank persistent-field model  r_t = Phi_theta(x_t) xi + eta_t,  xi ~ N(0, I_m), eta ~ N(0, n I),
# fitted on factual episodes by the exact joint Gaussian likelihood with path-dependent Gram (valid under feedback).
import numpy as np, torch, torch.nn as nn, time
torch.set_num_threads(1)
class Phi(nn.Module):
    def __init__(s,dx,p,m,h=128):
        super().__init__(); s.p,s.m=p,m
        s.net=nn.Sequential(nn.Linear(dx,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,p*m)); s.logn=nn.Parameter(torch.tensor(-2.))
    def forward(s,x): return s.net(x).reshape(*x.shape[:-1],s.p,s.m)
def nll(model,X,R):
    E,T,p=R.shape; F=model(X).reshape(E,T*p,model.m); n=model.logn.exp()
    C=F@F.transpose(1,2)+n*torch.eye(T*p); L=torch.linalg.cholesky(C)
    a=torch.cholesky_solve(R.reshape(E,T*p,1),L)
    return (0.5*(R.reshape(E,1,T*p)@a).squeeze()+torch.log(torch.diagonal(L,dim1=1,dim2=2)).sum(1)).mean()/(T*p)
def fit(X,R,m,epochs=300,lr=2e-3,bs=256,seed=0):
    torch.manual_seed(seed); X=torch.tensor(X,dtype=torch.float32); R=torch.tensor(R,dtype=torch.float32)
    model=Phi(X.shape[-1],R.shape[-1],m); opt=torch.optim.Adam(model.parameters(),lr=lr)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    for ep in range(epochs):
        perm=torch.randperm(len(X))
        for i in range(0,len(X),bs):
            idx=perm[i:i+bs]; l=nll(model,X[idx],R[idx]); opt.zero_grad(); l.backward(); opt.step()
        sched.step()
    return model
@torch.no_grad()
def posterior(model,XF,R):
    """smoothed mean of xi given the whole factual trajectory; returns xi_hat (E,m) and innovations (E,T,p)."""
    XF=torch.tensor(XF,dtype=torch.float32); R=torch.tensor(R,dtype=torch.float32); E,T,p=R.shape
    F=model(XF).reshape(E,T*p,model.m); n=model.logn.exp()
    A=F.transpose(1,2)@F+n*torch.eye(model.m); xi=torch.linalg.solve(A,F.transpose(1,2)@R.reshape(E,T*p,1))[...,0]
    inn=R-(F@xi[...,None]).reshape(E,T,p)
    return xi,inn
@torch.no_grad()
def field(model,x,xi): return (model(torch.tensor(x,dtype=torch.float32))@xi[...,None])[...,0].numpy()

def gain_demo(seed=0,T=5,Etr=4000,Ete=4000,sf=0.1):
    """1-D plant with a persistent gain fault, a persistent state-dependent load and a fresh kick (exact means used):
       s' = 0.8 s + a (1 + xi1) + xi2 sin(2 s) + eta.   mu(s,a) = 0.8 s + a."""
    r=np.random.default_rng(seed)
    def sim(E,A,s0,xi,eta):
        S=[s0]; s=s0
        for t in range(T): s=0.8*s+A[:,t]*(1+xi[:,0])+xi[:,1]*np.sin(2*s)+eta[:,t]; S.append(s)
        return np.stack(S,1)
    def draw(E): return r.uniform(-1,1,E), r.uniform(-1,1,(E,T)), np.stack([0.4*r.standard_normal(E),0.3*r.standard_normal(E)],1), sf*r.standard_normal((E,T))
    s0,A,xi,eta=draw(Etr); S=sim(Etr,A,s0,xi,eta); R=(S[:,1:]-0.8*S[:,:-1]-A)[...,None]; X=np.stack([S[:,:-1],A],-1)
    model=fit(X,R,m=4,epochs=60)
    s0,A,xi,eta=draw(Ete); S=sim(Ete,A,s0,xi,eta); R=(S[:,1:]-0.8*S[:,:-1]-A)[...,None]; X=np.stack([S[:,:-1],A],-1)
    A2=r.uniform(-1,1,(Ete,T)); S2=sim(Ete,A2,s0,xi,eta)
    xih,inn=posterior(model,X,R)
    res={}
    for meth in("model","replay","post","twin"):
        s=s0.copy(); P=[s]
        for t in range(T):
            if meth=="model": d=0
            elif meth=="replay": d=R[:,t,0]
            else:
                d=field(model,np.stack([s,A2[:,t]],-1),xih)[:,0]+(inn[:,t,0].numpy() if meth=="twin" else 0)
            s=0.8*s+A2[:,t]+d; P.append(s)
        res[meth]=np.mean((np.stack(P,1)-S2)**2,0)
    print(f"gain/load fault demo, fresh sd {sf}, learned nugget {model.logn.exp().item():.4f} (true {sf*sf:.4f})")
    for k in("replay","post","twin"): print(f"   {k:7s} rel.risk vs model-only at t=1..{T}: "+" ".join(f"{a/b:6.3f}" for a,b in zip(res[k][1:],res["model"][1:])))
if __name__=="__main__":
    for sf in (0.0,0.1,0.3): gain_demo(sf=sf)
