import torch,numpy as np,sys
torch.manual_seed(0); torch.set_num_threads(4)
dz,p,d,r=2,2,3,int(sys.argv[1]) if len(sys.argv)>1 else 1
W=[torch.randn(d,r) for _ in range(3)]; Az=[torch.randn(d,r,dz) for _ in range(3)]
def Gtrue(z,a):
    f=torch.cat([torch.ones(len(a),1),a],1)                       # (n,3)
    Gj=torch.stack([W[j][None]+0.7*torch.einsum('drk,nk->ndr',Az[j],torch.sin(2*z)) for j in range(3)],1)  # (n,3,d,r)
    return torch.einsum('nj,njdr->ndr',f,Gj)
def m(z,a): return torch.stack([torch.sin(z[:,0]+a[:,0]),z[:,1]*a[:,1]+a[:,0]**2,torch.tanh(a[:,0]-a[:,1])+z[:,0]],1)
def noise(n): return torch.tanh(torch.randn(n,r))/0.63
def F(z,a,xi): return m(z,a)+0.4*torch.einsum('ndr,nr->nd',Gtrue(z,a),xi)
n=60000; z=torch.rand(n,dz)*2-1; a=torch.rand(n,p)*2-1; y=F(z,a,noise(n))
def mlp(i,o,w=256): return torch.nn.Sequential(torch.nn.Linear(i,w),torch.nn.SiLU(),torch.nn.Linear(w,w),torch.nn.SiLU(),torch.nn.Linear(w,o))
mu=mlp(dz+p,d); opt=torch.optim.Adam(mu.parameters(),1e-3)
for it in range(4000):
    i=torch.randint(0,n,(1024,)); l=((mu(torch.cat([z[i],a[i]],1))-y[i])**2).mean(); opt.zero_grad(); l.backward(); opt.step()
with torch.no_grad(): res=y-mu(torch.cat([z,a],1))
R=r+1  # over-specified rank
gh=mlp(dz,(p+1)*d*R); logn=torch.full((d,),-4.,requires_grad=True)
opt=torch.optim.Adam(list(gh.parameters())+[logn],1e-3)
def Gof(zz,aa):
    f=torch.cat([torch.ones(len(aa),1),aa],1); return torch.einsum('nj,njdr->ndr',f,gh(zz).view(-1,p+1,d,R))
def nll(G,nug,rr):
    # low-rank + diag Gaussian NLL via Woodbury
    Dinv=1/nug; A=torch.eye(G.shape[2])+torch.einsum('ndr,d,nds->nrs',G,Dinv,G)
    Lc=torch.linalg.cholesky(A); u=torch.einsum('ndr,d,nd->nr',G,Dinv,rr)
    sol=torch.cholesky_solve(u[...,None],Lc)[...,0]
    quad=(rr**2*Dinv).sum(1)-(u*sol).sum(1); logdet=torch.log(nug).sum()+2*torch.log(torch.diagonal(Lc,dim1=1,dim2=2)).sum(1)
    return 0.5*(quad+logdet).mean()
for it in range(6000):
    i=torch.randint(0,n,(1024,)); l=nll(Gof(z[i],a[i]),torch.exp(logn)+1e-4,res[i]); opt.zero_grad(); l.backward(); opt.step()
nt=5000; zt=torch.rand(nt,dz)*2-1; at=torch.rand(nt,p)*2-1; ap=torch.rand(nt,p)*2-1; xi=noise(nt)
yt,yc=F(zt,at,xi),F(zt,ap,xi)
with torch.no_grad():
    ma,mp=mu(torch.cat([zt,at],1)),mu(torch.cat([zt,ap],1)); rt=yt-ma
    Ga,Gp=Gof(zt,at),Gof(zt,ap); nug=torch.exp(logn)+1e-4
    Kaa=Ga@Ga.transpose(1,2)+torch.diag(nug); pk=mp+torch.einsum('ndr,nr->nd',Gp,torch.einsum('ner,ne->nr',Ga,torch.linalg.solve(Kaa,rt)))
    sdr=mp+torch.sqrt(torch.diagonal(Gp@Gp.transpose(1,2),dim1=1,dim2=2)/torch.diagonal(Kaa,dim1=1,dim2=2))*rt
    Gta,Gtp=Gtrue(zt,at),Gtrue(zt,ap)
    orc=m(zt,ap)+torch.einsum('ndr,nr->nd',Gtp,torch.linalg.lstsq(Gta,(yt-m(zt,at))[...,None]).solution[...,0])
def r2(pred): return (1-((pred-yc)**2).sum(0)/((yc-yc.mean(0))**2).sum(0)).mean().item()
print(f"r={r}: trained_predictor={r2(mp):.3f} oracle_no_outcome={r2(m(zt,ap)):.3f} RR={r2(yt+mp-ma):.3f} sd_ratio={r2(sdr):.3f} PCK={r2(pk):.3f} exact-structure={r2(orc):.3f} nug={nug.numpy().round(4)}")
