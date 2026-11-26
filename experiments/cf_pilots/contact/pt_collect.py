import sys, numpy as np, time
from pt_sim import *
mode, n, seed0, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
sim=Sim(); t0=time.time()
if mode=="train":
    S=np.empty((n,11)); A=np.empty((n,2)); Y=np.empty((n,11)); EP=np.empty(n); C=np.empty(n)
    for i in range(n):
        hist,s=prefix(sim, seed0+i); rng=np.random.default_rng([seed0+i,337])
        a=rng.uniform(-LIM,LIM,2); e=eps_of(rng.standard_normal()); y,c=sim.step(a,e)
        S[i],A[i],Y[i],EP[i],C[i]=s,a,y,e,c
    np.savez(out,S=S,A=A,Y=Y,EP=EP,C=C)
else:
    E,K=int(sys.argv[5]),4
    S=np.empty((n,11)); A=np.empty((n,K,2)); Y=np.empty((n,E,K,11)); EP=np.empty((n,E)); C=np.empty((n,E,K))
    for i in range(n):
        hist,s=prefix(sim, seed0+i); rng=np.random.default_rng([seed0+i,337])
        A[i]=rng.uniform(-LIM,LIM,(K,2)); EP[i]=eps_of(rng.standard_normal(E)); S[i]=s
        for j in range(E):
            for k in range(K):
                replay(sim, seed0+i, hist); st=sim.state(); assert np.allclose(st,s,atol=1e-8)
                Y[i,j,k],C[i,j,k]=sim.step(A[i,k],EP[i,j])
    np.savez(out,S=S,A=A,Y=Y,EP=EP,C=C)
print(mode,n,"done in %.1fs"%(time.time()-t0))
