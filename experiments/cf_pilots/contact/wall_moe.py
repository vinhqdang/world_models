"""Wall world, structured learner (factual data only):
 1) location-family inverse dynamics on REGULAR outcomes -> effective command e(s,d) and shared noise shape nu
 2) decoder C_free(s,e) on regular outcomes
 3) switching gate g(s,u)=P(collapsed|s,u) by marginal likelihood: regular: 1-g(s,e); collapsed: mean_k g(s,a+eta_k), eta_k~nu
 4) counterfactual: regular factual -> u'=e+a'-a ; collapsed factual -> truncated posterior particles."""
import sys, numpy as np, torch, torch.nn as nn, math, time
from wall_oracle import *
torch.set_num_threads(2); seed=int(sys.argv[1]) if len(sys.argv)>1 else 0; torch.manual_seed(seed)
NTR=int(sys.argv[2]) if len(sys.argv)>2 else 40000; IT=int(sys.argv[3]) if len(sys.argv)>3 else 6000
r=np.random.default_rng(100+seed)
s=parents(r,NTR); a=r.uniform(-1,1,(NTR,2)); e=eps_draw(r,(NTR,2)); y,_=step(s,a,e); d=(y-s)/STEP
T32=lambda v: torch.tensor(v,dtype=torch.float32)
S,A,Dd=T32(s),T32(a),T32(d); col=(Dd.abs().sum(-1)<1e-9); reg=~col           # mode label read off the outcome (atom at zero displacement)
print("collapsed frac %.3f"%col.float().mean().item())
def mlp(i,o,h=256): return nn.Sequential(nn.Linear(i,h),nn.GELU(),nn.Linear(h,h),nn.GELU(),nn.Linear(h,h),nn.GELU(),nn.Linear(h,o))
Enc=mlp(4,2); Dec=mlp(4,2); Gate=mlp(4,1); MU=mlp(4,2)
M=8; pi_l=nn.Parameter(torch.zeros(M)); mu_c=nn.Parameter(0.5*torch.randn(M,2)); ls_c=nn.Parameter(torch.full((M,2),-0.7))
def nu_logp(eta):
    z=(eta[...,None,:]-mu_c)/ls_c.exp(); lp=-0.5*z.pow(2).sum(-1)-ls_c.sum(-1)-math.log(2*math.pi)
    return torch.logsumexp(lp+torch.log_softmax(pi_l,0),-1)
def nu_sample(shape):
    k=torch.multinomial(torch.softmax(pi_l,0).detach(),int(np.prod(shape)),True).reshape(shape)
    return (mu_c[k]+ls_c.exp()[k]*torch.randn(*shape,2)).detach()
g=torch.linspace(-1,1,25); g=(g[:-1]+g[1:])/2; GR=torch.stack(torch.meshgrid(g,g,indexing="ij"),-1).reshape(-1,2)
Sr,Ar,Dr=S[reg],A[reg],Dd[reg]; Sc,Ac=S[col],A[col]; nr,nc=len(Sr),len(Sc)
opt=torch.optim.Adam(list(Enc.parameters())+list(Dec.parameters())+[pi_l,mu_c,ls_c]+list(MU.parameters()),lr=1e-3); t0=time.time()
for it in range(IT):                                                         # stages 1-2 (+ factual predictor for the RR baseline)
    i=torch.randint(0,nr,(512,)); eh=Enc(torch.cat([Sr[i],Dr[i]],-1))
    logZ=torch.logsumexp(nu_logp(eh[:,None]-GR[None]),1)+math.log(4/len(GR))
    l_idm=-(nu_logp(eh-Ar[i])-logZ).mean(); l_dec=(Dec(torch.cat([Sr[i],eh],-1))-Dr[i]).pow(2).mean()
    j=torch.randint(0,NTR,(512,)); l_mu=(MU(torch.cat([S[j],A[j]],-1))-Dd[j]).pow(2).mean()
    loss=l_idm+10*l_dec+l_mu; opt.zero_grad(); loss.backward(); opt.step()
    if it%2000==0: print(it,"idm %.3f dec %.5f mu %.4f %.0fs"%(l_idm.item(),l_dec.item(),l_mu.item(),time.time()-t0),flush=True)
optg=torch.optim.Adam(Gate.parameters(),lr=1e-3)
with torch.no_grad(): Er=Enc(torch.cat([Sr,Dr],-1))
for it in range(IT):                                                         # stage 3: switching surface in (s,u)
    i=torch.randint(0,nr,(512,)); j=torch.randint(0,nc,(128,))
    l_reg=-nn.functional.logsigmoid(-Gate(torch.cat([Sr[i],Er[i]],-1))).mean()
    u=Ac[j][:,None]+nu_sample((128,64)); lg=nn.functional.logsigmoid(Gate(torch.cat([Sc[j][:,None].expand(-1,64,-1),u],-1))[...,0])
    l_col=-(torch.logsumexp(lg,1)-math.log(64)).mean()
    loss=(nr*l_reg+nc*l_col)/NTR; optg.zero_grad(); loss.backward(); optg.step()
print("gate trained %.0fs"%(time.time()-t0))
# ---------------- evaluation ----------------
r=np.random.default_rng(7); N=4000; E=256
s=parents(r,N); a=r.uniform(-1,1,(N,2)); ap=r.uniform(-1,1,(N,2)); e=eps_draw(r,(N,2))
y,bf=step(s,a,e); yp,bc=step(s,ap,e); em=eps_draw(r,(N,E,2))
mu_a=step(s[:,None],a[:,None],em)[0].mean(1); mu_ap=step(s[:,None],ap[:,None],em)[0].mean(1)
big=eps_draw(r,(N,2048,2)); _,bb=step(s[:,None],a[:,None],big); ycf,_=step(s[:,None],ap[:,None],big)
w=bb/np.maximum(bb.sum(1,keepdims=True),1); bayes=np.where(bf[:,None],(w[...,None]*ycf).sum(1),step(s,ap,(y-s)/STEP-a)[0])
with torch.no_grad():
    St,At,Apt,Dt=T32(s),T32(a),T32(ap),T32((y-s)/STEP); colf=(Dt.abs().sum(-1)<1e-9)
    def push(S_,U_):                                                         # hybrid decoder: gate picks stay (atom) vs free expert
        p=torch.sigmoid(Gate(torch.cat([S_,U_],-1))); return (p<0.5).float()*Dec(torch.cat([S_,U_],-1)), p
    eh=Enc(torch.cat([St,Dt],-1)); dreg,_=push(St,eh+Apt-At)
    K=1024; U=At[:,None]+nu_sample((N,K)); SK=St[:,None].expand(-1,K,-1)
    _,pk=push(SK,U); wk=pk[...,0]/pk[...,0].sum(1,keepdim=True).clamp_min(1e-9)
    dk,_=push(SK,U+(Apt-At)[:,None]); dcol=(wk[...,None]*dk).sum(1)
    pct=s+STEP*torch.where(colf[:,None],dcol,dreg).numpy()
    mh_a=MU(torch.cat([St,At],-1)).numpy(); mh_ap=MU(torch.cat([St,Apt],-1)).numpy()
    print("effective-command recovery on regular outcomes: rmse(e - (a+eps) - mean shift) %.4f"%((eh.numpy()-(a+e))[~bf]-((eh.numpy()-(a+e))[~bf]).mean(0)).std())
def r2(p,m): dd=(yp-s); return 1-((p-yp)[m]**2).sum()/((dd-dd.mean(0))**2).sum()*len(yp)/max(m.sum(),1)
sw=bf!=bc; al=np.ones(N,bool)
print("seed %d ntr %d | switch frac %.3f"%(seed,NTR,sw.mean()))
for nm,p in [("oracle no outcome",mu_ap),("trained predictor",s+STEP*mh_ap),("RR exact means",y+mu_ap-mu_a),("RR learned",y+STEP*(mh_ap-mh_a)),
             ("PCT learned (factual only)",pct),("PCT true map (Bayes)",bayes)]:
    print("%-28s R2 all %.3f | 1-MSE/totvar: same-mode %.3f  switch %.3f | free->blocked %.3f  blocked->free %.3f"%(nm,r2(p,al),r2(p,~sw),r2(p,sw),r2(p,(~bf)&bc),r2(p,bf&~bc)))
