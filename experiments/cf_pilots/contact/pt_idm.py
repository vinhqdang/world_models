"""Push-T, no anchor knowledge: location-family inverse dynamics (LF-IDM) abducts the effective command from the FULL
outcome; decoder C(s,e) regresses the outcome on it; counterfactual = C(s, e + a' - a). Factual data only."""
import sys, os, math, numpy as np, torch, torch.nn as nn, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","repo","cwm"))
from cwm import metrics
torch.set_num_threads(2); seed=int(sys.argv[1]) if len(sys.argv)>1 else 0; torch.manual_seed(seed)
NTR=int(sys.argv[2]) if len(sys.argv)>2 else 60000; EP=int(sys.argv[3]) if len(sys.argv)>3 else 25
OUT=[int(c) for c in sys.argv[4].split(",")] if len(sys.argv)>4 else list(range(11))   # outcome coords shown to the IDM
tr=np.load("pt_train.npz"); ev=np.load("pt_eval.npz"); LIM=0.35
S,A,Y=tr["S"][:NTR],tr["A"][:NTR]/LIM,tr["Y"][:NTR]
def feat(s):
    s=np.asarray(s,float); return np.concatenate([(s[...,0:2]-256)/100,(s[...,2:4]-s[...,0:2])/100,s[...,4:6],s[...,6:8]/100,s[...,8:10]/100,s[...,10:11]],-1)
def tgt(y,s):
    return np.concatenate([(y[...,0:2]-s[...,0:2])/100,(y[...,2:4]-s[...,2:4])/100,y[...,4:6],y[...,6:8]/100,y[...,8:10]/100,y[...,10:11]],-1)
T32=lambda v: torch.tensor(v,dtype=torch.float32)
FS,TY,At=T32(feat(S)),T32(tgt(Y,S)),T32(A); my,sy_=TY.mean(0),TY.std(0)+1e-6; mf,sf=FS.mean(0),FS.std(0)+1e-6
def mlp(i,o,h=512): return nn.Sequential(nn.Linear(i,h),nn.GELU(),nn.Linear(h,h),nn.GELU(),nn.Linear(h,h),nn.GELU(),nn.Linear(h,o))
Enc=mlp(11+len(OUT),2); Dec=mlp(11+2,11)
M=6; pi_l=nn.Parameter(torch.zeros(M)); mu_c=nn.Parameter(0.3*torch.randn(M,2)); ls_c=nn.Parameter(torch.full((M,2),-1.0)); Lm=nn.Parameter(torch.eye(2))
def nu_logp(eta):       # eta (...,2): density of L xi, xi ~ diag mixture
    xi=torch.linalg.solve(Lm, eta.unsqueeze(-1)).squeeze(-1)
    ls=ls_c.clamp(min=math.log(0.02)); z=(xi[...,None,:]-mu_c)/ls.exp()
    lp=-0.5*z.pow(2).sum(-1)-ls.sum(-1)-math.log(2*math.pi)
    return torch.logsumexp(lp+torch.log_softmax(pi_l,0),-1)-torch.linalg.slogdet(Lm)[1]
g=torch.linspace(-1,1,25); g=(g[:-1]+g[1:])/2; GR=torch.stack(torch.meshgrid(g,g,indexing="ij"),-1).reshape(-1,2)   # 576 cell centres
def enc(fs,ty): return Enc(torch.cat([(fs-mf)/sf,((ty-my)/sy_)[:,OUT]],-1))
def dec(fs,e): return Dec(torch.cat([(fs-mf)/sf,e],-1))*sy_+my
opt=torch.optim.AdamW(list(Enc.parameters())+list(Dec.parameters())+[pi_l,mu_c,ls_c,Lm],lr=1e-3,weight_decay=1e-5)
steps=EP*(NTR//512); sched=torch.optim.lr_scheduler.OneCycleLR(opt,1e-3,total_steps=steps); t0=time.time()
for it in range(steps):
    i=torch.randint(0,NTR,(512,)); e=enc(FS[i],TY[i])
    ll=nu_logp(e-At[i]); logZ=torch.logsumexp(nu_logp(e[:,None]-GR[None]),1)+math.log(4/len(GR))
    l_idm=-(ll-logZ).mean()                                    # truncated location-family likelihood of the action
    l_dec=(((dec(FS[i],e)-TY[i])/sy_)**2).mean()               # sufficiency: outcome from (state, effective command)
    loss=l_idm+10*l_dec; opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if it%1000==0: print(it,"idm nll %.3f  dec mse %.4f  %.0fs"%(l_idm.item(),l_dec.item(),time.time()-t0),flush=True)
Enc.eval(); Dec.eval()
Se,Ae,Ye=ev["S"],ev["A"]/LIM,ev["Y"]; N,E,K,D=Ye.shape; W=np.random.default_rng(0).multinomial(N,np.full(N,1/N),size=64)
Te=tgt(Ye,Se[:,None,None,:]); obs=Te[:,:,0]
with torch.no_grad():
    fs=T32(feat(Se))[:,None].expand(N,E,11).reshape(-1,11); ob=T32(obs.reshape(-1,11))
    e=enc(fs,ob); rec=dec(fs,e); P=[]
    for k in range(K):
        da=T32(Ae[:,k]-Ae[:,0])[:,None].expand(N,E,2).reshape(-1,2)
        P.append(dec(fs,e+da))
    P=torch.stack(P,1); Pres=P+(ob-rec)[:,None]
    P=P.reshape(N,E,K,11).numpy().astype(np.float64); Pres=Pres.reshape(N,E,K,11).numpy().astype(np.float64)
    # how well does the abducted location track the true effective command?  u = a + eps*(1,-1)/(GAIN*LIM)
    GAIN=1.472860284876826; utrue=Ae[:,None,0]+ev["EP"][:,:,None]*np.array([1.,-1.])/(GAIN*LIM)
    eh=e.reshape(N,E,2).numpy(); Xd=np.c_[eh.reshape(-1,2),np.ones(N*E)]; coef=np.linalg.lstsq(Xd,utrue.reshape(-1,2),rcond=None)[0]
    print("effective-command recovery: R2 of affine fit %.4f ; slope matrix\n"%(1-((Xd@coef-utrue.reshape(-1,2))**2).sum()/((utrue-utrue.mean((0,1)))**2).sum()),coef[:2].round(3))
groups={"pusher xy":[0,1],"block xy":[2,3],"block cos,sin":[4,5],"block vx":[8],"block vy":[9],"block w":[10]}
print("%-34s"%("LF-IDM seed %d n %d out %s"%(seed,NTR,len(OUT)))+"".join("%14s"%g for g in groups))
for nm,p in [("IDT (decoder at shifted location)",P),("IDT + recon residual",Pres)]:
    print("%-34s"%nm+"".join("%14.1f"%(100*metrics.scores(p[...,g],Te[...,g],obs[...,g],W)["r2"]) for g in groups.values()))
