# Multi-step re-creation of the boat of arXiv 2611.24778 App. J (velocity/advect follow cwm/sims/boat.py), batched,
# with T transitions, an optional fresh gust per transition, and counterfactual replays under the same field+gusts.
import numpy as np
MODES=3
def eddies(r, G, L, D, E):
    size=(E,MODES)
    ang=r.uniform(0,2*np.pi,size); wn=2*np.pi/r.uniform(L,2*L,size)
    return dict(k=wn[...,None]*np.stack([np.cos(ang),np.sin(ang)],-1), phase=r.uniform(0,2*np.pi,size),
                rate=r.uniform(-D,D,size), amp=G*r.standard_normal(size)/np.sqrt(MODES))
def velocity(p,t,mean,eddy):
    arg=np.einsum("ed,ekd->ek",p,eddy["k"])+eddy["phase"]+eddy["rate"]*t
    across=np.stack([eddy["k"][...,1],-eddy["k"][...,0]],-1)/np.linalg.norm(eddy["k"],axis=-1,keepdims=True)
    return np.einsum("ek,ekd->ed",eddy["amp"]*np.cos(arg),across)+mean
def advect(p,psi,t,mean,eddy,n=8):
    v=np.stack([np.cos(psi),np.sin(psi)],-1); h=1/n
    f=lambda q,s: v+velocity(q,s,mean,eddy)
    for i in range(n):
        s=t+i*h
        k1=f(p,s);k2=f(p+h/2*k1,s+h/2);k3=f(p+h/2*k2,s+h/2);k4=f(p+h*k3,s+h)
        p=p+h/6*(k1+2*k2+2*k3+k4)
    return p
def world(r,E,T,S,G,L,D,sf):
    ed=eddies(r,G,L,D,E)
    mean=np.stack([S*np.tanh(r.standard_normal(E)),np.zeros(E)],-1)
    gust=sf*r.standard_normal((E,T,2))
    return dict(ed=ed,mean=mean,gust=gust)
def rollout(w,p0,psi):  # psi (E,T) headings; gust is a uniform extra water velocity held for the transition
    E,T=psi.shape; P=[p0]; p=p0
    for t in range(T):
        p=advect(p,psi[:,t],float(t),w["mean"]+w["gust"][:,t],w["ed"]); P.append(p)
    return np.stack(P,1)
def behaviour(r,E,T,kind="walk"):
    if kind=="walk":
        psi=np.cumsum(np.concatenate([r.uniform(-np.pi,np.pi,(E,1)),np.radians(50)*r.standard_normal((E,T-1))],1),1)
    else: psi=r.uniform(-np.pi,np.pi,(E,T))
    return psi
def U(psi): return np.stack([np.cos(psi),np.sin(psi)],-1)
