import numpy as np, torch
rng=np.random.default_rng(1)
def feats(a): return np.c_[np.ones(len(a)),a]
def r2(pred,truth): return 1-((pred-truth)**2).sum()/((truth-truth.mean(0))**2).sum()
def fit_factor(a,r,rank,steps=3000,lr=0.02,nug=True):
    """Sigma(a)=G(a)G(a)^T + diag(nug), G(a)=sum_j f_j(a) G_j ; fit by matching r r^T (Frobenius)"""
    f=torch.tensor(feats(a),dtype=torch.float64); R=torch.tensor(r,dtype=torch.float64)
    d=R.shape[1]; p1=f.shape[1]
    torch.manual_seed(0)
    G=(0.1*torch.randn(p1,d,rank,dtype=torch.float64)).requires_grad_()
    ln=torch.full((d,),-6.,dtype=torch.float64,requires_grad=True)
    opt=torch.optim.Adam([G,ln],lr=lr)
    for i in range(steps):
        idx=torch.randint(0,len(f),(2048,))
        Ga=torch.einsum('nj,jdr->ndr',f[idx],G)
        u=torch.einsum('ndr,nd->nr',Ga,R[idx])
        # || rr^T - GG^T - D ||_F^2 expanded
        S=Ga@Ga.transpose(1,2)+ (torch.diag(torch.exp(ln)) if nug else 0)
        loss=((R[idx,:,None]*R[idx,None,:]-S)**2).sum((1,2)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return G.detach().numpy(), np.exp(ln.detach().numpy())
def krig(G,nug,a,ap,r,tau=1e-4):
    Ga=np.einsum('nj,jdr->ndr',feats(a),G); Gp=np.einsum('nj,jdr->ndr',feats(ap),G)
    Kaa=Ga@Ga.transpose(0,2,1)+np.diag(nug)+tau*np.eye(Ga.shape[1]); Kpa=Gp@Ga.transpose(0,2,1)
    return np.einsum('nij,nj->ni',Kpa,np.linalg.solve(Kaa,r[...,None])[...,0])
def sym_polar(a,r):
    """unconstrained symmetric polarization: Sigma(a)=sum_{j<=k} f_j f_k C_jk, K(a,a')=sum_jk f_j f'_k M_jk with M_jk=M_kj sym"""
    f=feats(a); p1=f.shape[1]; iu=np.triu_indices(p1); Q=f[:,iu[0]]*f[:,iu[1]]
    d=r.shape[1]; T=(r[:,:,None]*r[:,None,:]).reshape(len(r),-1)
    C=np.linalg.lstsq(Q,T,rcond=None)[0]
    M=np.zeros((p1,p1,d,d))
    for c,(j,k) in enumerate(zip(*iu)):
        m=C[c].reshape(d,d); m=(m+m.T)/2
        if j==k: M[j,j]=m
        else: M[j,k]=M[k,j]=m/2
    return M
def krig_M(M,a,ap,r,tau):
    f,fp=feats(a),feats(ap)
    Kaa=np.einsum('nj,nk,jkde->nde',f,f,M); Kpa=np.einsum('nj,nk,jkde->nde',fp,f,M)
    Kaa=Kaa+tau*np.eye(Kaa.shape[1])
    return np.einsum('nij,nj->ni',Kpa,np.linalg.solve(Kaa,r[...,None])[...,0])
def run(name,F,p,n=40000,nt=5000,rank=1,sampleT=None):
    a=rng.uniform(-1,1,(n,p)); T=sampleT(n); y=F(a,T)
    X=feats(a); B=np.linalg.lstsq(X,y,rcond=None)[0]; r=y-X@B
    G,nug=fit_factor(a,r,rank)
    at=rng.uniform(-1,1,(nt,p)); ap=rng.uniform(-1,1,(nt,p)); Tt=sampleT(nt)
    yt,yc=F(at,Tt),F(ap,Tt); mu_a,mu_p=feats(at)@B,feats(ap)@B; rt=yt-mu_a
    M=sym_polar(a,r); tau=1e-3*np.mean(r**2)
    res=dict(oracle=r2(mu_p,yc),RR=r2(yt+mu_p-mu_a,yc),PK_factor=r2(mu_p+krig(G,nug,at,ap,rt,tau),yc),PK_sym=r2(mu_p+krig_M(M,at,ap,rt,tau),yc))
    print(name,{k:round(float(v),3) for k,v in res.items()})
tanhT=lambda n: np.tanh(rng.standard_normal((n,1)))
# (A) HandReach-like: 2 actions, 2 coords, one shared scalar eps with pattern d=(+1,-1), lam dial
for lam in [0,0.5,1.0]:
    m=(1-lam)**2+lam**2/3; s=1/(1+np.sqrt(3*0.394*m/0.98)); g=0.98*(1-s); d=np.array([1.,-1.])
    F=lambda a,T,lam=lam,s=s,g=g: g*a+s*T*d*((1-lam)+lam*a)
    run(f"hand-like lam={lam}",F,2,sampleT=tanhT)
# (B) non-parallel rank one: Y=(g0+a g1) T with g0 not parallel g1 (antisymmetric part matters)
F=lambda a,T: 0.5*a@np.array([[1.,0.3,0.1]])+T*(np.array([1.,0.,0.5])+a*np.array([0.,1.,-0.5]))
run("rank1 nonparallel d=3,p=1",F,1,sampleT=tanhT)
# (C) two independent disturbances (offset eps0, gain eps1), d=1: best linear only
F=lambda a,T: 0.5*a+0.5*T[:,:1]+0.8*a*T[:,1:2]
run("offset+gain indep d=1",F,1,rank=1,sampleT=lambda n: rng.standard_normal((n,2)))
run("offset+gain indep d=1 rank2",F,1,rank=2,sampleT=lambda n: rng.standard_normal((n,2)))
# (D) rotation gauge counterexample d=2,r=2: Y=(I+aJ)xi
J=np.array([[0,-1.],[1,0]])
F=lambda a,T: T+a*(T@J.T)
run("rotation J d=2 r=2 (unidentified)",F,1,rank=2,sampleT=lambda n: rng.standard_normal((n,2)))
# (E) tall: d=6, r=2 random
G0=rng.standard_normal((6,2)); G1=rng.standard_normal((6,2)); G2=rng.standard_normal((6,2))
F=lambda a,T: T@G0.T+a[:,:1]*(T@G1.T)+a[:,1:2]*(T@G2.T)
run("tall d=6 r=2 p=2",F,2,rank=2,sampleT=lambda n: rng.standard_normal((n,2)))
run("tall d=6 r=2 p=2 (uniform noise)",F,2,rank=2,sampleT=lambda n: rng.uniform(-1.7,1.7,(n,2)))
