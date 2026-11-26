"""Wall world: learn the pre-contact (input-channel) latent model d = G(s, a+eta), eta ~ nu, from FACTUAL data only
(reweighted wake-sleep / importance-sampled marginal likelihood), then abduct-transport-push for counterfactuals."""
import sys, numpy as np, torch, torch.nn as nn, math, time
from wall_oracle import *
torch.set_num_threads(2); seed=int(sys.argv[1]) if len(sys.argv)>1 else 0; torch.manual_seed(seed)
NTR=int(sys.argv[2]) if len(sys.argv)>2 else 40000; ITERS=int(sys.argv[3]) if len(sys.argv)>3 else 6000
r=np.random.default_rng(100+seed)
s=parents(r,NTR); a=r.uniform(-1,1,(NTR,2)); e=eps_draw(r,(NTR,2)); y,_=step(s,a,e); d=(y-s)/STEP
S,A,Dd=[torch.tensor(v,dtype=torch.float32) for v in (s,a,d)]
def mlp(i,o,h=256): return nn.Sequential(nn.Linear(i,h),nn.GELU(),nn.Linear(h,h),nn.GELU(),nn.Linear(h,h),nn.GELU(),nn.Linear(h,o))
G=mlp(4,2); Q=mlp(6,4); MU=mlp(4,2)
M=8; pi_l=nn.Parameter(torch.zeros(M)); mu_c=nn.Parameter(0.5*torch.randn(M,2)); ls_c=nn.Parameter(torch.full((M,2),-0.7))
def nu_logp(eta):
    z=(eta[...,None,:]-mu_c)/ls_c.exp(); lp=-0.5*z.pow(2).sum(-1)-ls_c.sum(-1)-math.log(2*math.pi)
    return torch.logsumexp(lp+torch.log_softmax(pi_l,0),-1)
def nu_sample(shape):
    k=torch.multinomial(torch.softmax(pi_l,0).detach(),int(np.prod(shape)),True).reshape(shape)
    return (mu_c[k]+ls_c.exp()[k]*torch.randn(*shape,2)).detach()
def q_params(s_,d_,a_):
    o=Q(torch.cat([s_,d_,a_],-1)); return a_+o[...,:2], o[...,2:].clamp(-4,1)
def gauss_logp(x,m,ls): return (-0.5*((x-m)/ls.exp()).pow(2)-ls-0.5*math.log(2*math.pi)).sum(-1)
def particles(s_,d_,a_,K,sy):
    """defensive mixture proposal: half prior, half encoder; returns u (B,K,2) and log-weights"""
    B=len(s_); qm,qls=q_params(s_,d_,a_); qm,qls=qm.detach(),qls.detach()
    u1=a_[:,None]+nu_sample((B,K//2)); u2=qm[:,None]+qls.exp()[:,None]*torch.randn(B,K//2,2)
    u=torch.cat([u1,u2],1)
    lprior=nu_logp(u-a_[:,None]); lq=gauss_logp(u,qm[:,None],qls[:,None])
    lr=torch.logsumexp(torch.stack([lprior.detach(),lq]),0)-math.log(2)
    pred=G(torch.cat([s_[:,None].expand(-1,K,-1),u],-1))
    llik=(-0.5*((d_[:,None]-pred)/sy).pow(2)-math.log(sy)-0.5*math.log(2*math.pi)).sum(-1)
    return u,pred,llik+lprior-lr
opt=torch.optim.Adam(list(G.parameters())+[pi_l,mu_c,ls_c],lr=1e-3); optq=torch.optim.Adam(Q.parameters(),lr=1e-3); optm=torch.optim.Adam(MU.parameters(),lr=1e-3)
t0=time.time(); K=64
for it in range(ITERS):
    sy=max(0.05,0.4*(0.05/0.4)**(it/(0.6*ITERS)))
    i=torch.randint(0,NTR,(256,)); u,pred,lw=particles(S[i],Dd[i],A[i],K,sy)
    loss=-(torch.logsumexp(lw,1)-math.log(K)).mean(); opt.zero_grad(); loss.backward(); opt.step()
    w=torch.softmax(lw.detach(),1); qm,qls=q_params(S[i],Dd[i],A[i])
    lq=-(w*gauss_logp(u,qm[:,None],qls[:,None])).sum(1).mean(); optq.zero_grad(); lq.backward(); optq.step()
    lm=(MU(torch.cat([S[i],A[i]],-1))-Dd[i]).pow(2).mean(); optm.zero_grad(); lm.backward(); optm.step()
    if it%1000==0: print(it,"nll %.3f sy %.3f mse_mu %.4f  %.0fs"%(loss.item(),sy,lm.item(),time.time()-t0),flush=True)
# ---------- evaluation ----------
r=np.random.default_rng(7); N=4000; E=256
s=parents(r,N); a=r.uniform(-1,1,(N,2)); ap=r.uniform(-1,1,(N,2)); e=eps_draw(r,(N,2))
y,bf=step(s,a,e); yp,bc=step(s,ap,e); em=eps_draw(r,(N,E,2))
mu_a=step(s[:,None],a[:,None],em)[0].mean(1); mu_ap=step(s[:,None],ap[:,None],em)[0].mean(1)
big=eps_draw(r,(N,2048,2)); _,bb=step(s[:,None],a[:,None],big); ycf,_=step(s[:,None],ap[:,None],big)
w=bb/np.maximum(bb.sum(1,keepdims=True),1); bayes=np.where(bf[:,None],(w[...,None]*ycf).sum(1),step(s,ap,(y-s)/STEP-a)[0])
with torch.no_grad():
    St,At,Apt,Dt=[torch.tensor(v,dtype=torch.float32) for v in (s,a,ap,(y-s)/STEP)]
    out=[]
    for j in range(0,N,500):
        sl=slice(j,j+500); u,pred,lw=particles(St[sl],Dt[sl],At[sl],1024,0.05); wt=torch.softmax(lw,1)
        up=u+(Apt[sl]-At[sl])[:,None]; dcf=G(torch.cat([St[sl][:,None].expand(-1,1024,-1),up],-1))
        rec=(wt[...,None]*pred).sum(1)
        out.append(torch.cat([(wt[...,None]*dcf).sum(1), rec],-1))
    out=torch.cat(out).numpy(); pct=s+STEP*out[:,:2]; pct_res=pct+STEP*((y-s)/STEP-out[:,2:])
    mh_a=MU(torch.cat([St,At],-1)).numpy(); mh_ap=MU(torch.cat([St,Apt],-1)).numpy()
def r2(p,m): dd=(yp-s); return 1-((p-yp)[m]**2).sum()/((dd-dd.mean(0))**2).sum()*len(yp)/max(m.sum(),1)
sw=bf!=bc; al=np.ones(N,bool)
print("seed %d ntr %d | switch frac %.3f"%(seed,NTR,sw.mean()))
for nm,p in [("oracle no outcome",mu_ap),("trained predictor",s+STEP*mh_ap),("RR exact means",y+mu_ap-mu_a),("RR learned",y+STEP*(mh_ap-mh_a)),
             ("PCT learned (factual only)",pct),("PCT learned + recon resid",pct_res),("PCT true map (Bayes)",bayes)]:
    print("%-28s R2 all %.3f | 1-MSE/totvar: same-mode %.3f  switch %.3f"%(nm,r2(p,al),r2(p,~sw),r2(p,sw)))
print("learned nu: std of eta %.3f (true %.3f)"%(nu_sample((20000,)).std().item(), e.std()))
