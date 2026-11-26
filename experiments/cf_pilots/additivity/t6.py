# short-time rates on a damped pendulum with (i) additive torque offset, (ii) gain fault, (iii) dial; exact MC means, K from factual variance function (quadratic fit in a per source state)
import numpy as np
rng=np.random.default_rng(5)
def sim(s,a,eps,h,lam,n=200):
    th,om=s; dt=h/n
    for _ in range(n):
        u=a+0.5*eps*((1-lam)+lam*a)        # effective torque, a in [-1,1]
        acc=-np.sin(th)-0.3*om+2.0*u
        om=om+dt*acc; th=th+dt*om
    return np.stack([th,om],-1)
def experiment(h,lam,nA=41,nE=400,npairs=300):
    s=(np.array(0.7),np.array(0.2))
    A=np.linspace(-1,1,nA); E=np.tanh(rng.standard_normal(nE))
    Y=sim((s[0]*np.ones((nA,nE)),s[1]*np.ones((nA,nE))),A[:,None]*np.ones((1,nE)),E[None]*np.ones((nA,1)),h,lam)  # (nA,nE,2)
    mu=Y.mean(1); R=Y-mu[:,None]                           # factual laws (exact up to MC) for each a
    # factual covariance function -> coefficient fit quadratic in a (per source), then rank-1 lift via closed form: use PSD of stacked coefficient? simple: fit G(a)=g0+a g1 by alternating LS on Sigma(a)
    Sig=np.einsum('aei,aej->aij',R,R)/nE
    X=np.stack([np.ones(nA),A,A**2],1); C=np.linalg.lstsq(X,Sig.reshape(nA,-1),rcond=None)[0].reshape(3,2,2)
    # block matrix M=[[C0, C1/2 + S],[C1/2 - S, C2]] with S antisym unknown; choose S minimizing rank (d=2: scan scalar)
    J=np.array([[0,1.],[-1,0]]); best=None
    for c in np.linspace(-3,3,2401)*np.sqrt(abs(np.trace(C[0])*np.trace(C[2]))+1e-30):
        M=np.block([[C[0],C[1]/2+c*J],[C[1]/2-c*J,C[2]]]); w=np.linalg.eigvalsh(M)
        score=np.sum(np.sort(np.abs(w))[:3])   # want rank 1 => three smallest ~0
        if best is None or score<best[0]: best=(score,M)
    M=best[1]; w,V=np.linalg.eigh(M); M=(V*np.clip(w,0,None))@V.T
    Phi=lambda a: np.concatenate([np.eye(2),a*np.eye(2)],1)
    i=rng.integers(0,nA,npairs); j=rng.integers(0,nA,npairs)
    eRR=eP=eO=0.
    for ii,jj in zip(i,j):
        if ii==jj: continue
        Kaa=Phi(A[ii])@M@Phi(A[ii]).T; Kpa=Phi(A[jj])@M@Phi(A[ii]).T
        L=Kpa@np.linalg.pinv(Kaa+1e-6*np.trace(Kaa)*np.eye(2))
        r=R[ii]; tgt=R[jj]
        eRR+=((r-tgt)**2).sum(); eP+=((r@L.T-tgt)**2).sum(); eO+=(tgt**2).sum()
    return eRR/eO,eP/eO
for lam in [0,0.5,1.0]:
    for h in [0.4,0.2,0.1,0.05]:
        a,b=experiment(h,lam); print(f"lam={lam} h={h}: rho_RR={a:.4f} rho_PCK={b:.5f}")
