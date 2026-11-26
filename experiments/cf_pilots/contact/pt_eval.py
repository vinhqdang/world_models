import sys, os, numpy as np, torch, torch.nn as nn, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","repo","cwm"))
from cwm import metrics
torch.set_num_threads(2); torch.manual_seed(int(sys.argv[1]) if len(sys.argv)>1 else 0)
NTR = int(sys.argv[2]) if len(sys.argv)>2 else 60000
ev=np.load("pt_eval.npz"); t1=np.load("pt_train.npz"); t2=np.load("pt_train_big.npz")
tr={k:np.concatenate([t1[k],t2[k]]) for k in ("S","A","Y","C")}
HO={k:v[-5000:] for k,v in tr.items()}
S,A,Y=tr["S"][:NTR],tr["A"][:NTR],tr["Y"][:NTR]; Cn=tr["C"][:NTR]
def feat(s):  # state features: positions relative to pusher, scaled
    s=np.asarray(s,float); return np.concatenate([(s[...,0:2]-256)/100,(s[...,2:4]-s[...,0:2])/100,s[...,4:6],s[...,6:8]/100,s[...,8:10]/100,s[...,10:11]],-1)
def tgt(y,s):  # outcome in displacement / scaled units
    return np.concatenate([(y[...,0:2]-s[...,0:2])/100,(y[...,2:4]-s[...,2:4])/100,y[...,4:6],y[...,6:8]/100,y[...,8:10]/100,y[...,10:11]],-1)
def untgt(t,s):
    return np.concatenate([t[...,0:2]*100+s[...,0:2],t[...,2:4]*100+s[...,2:4],t[...,4:6],t[...,6:8]*100,t[...,8:10]*100,t[...,10:11]],-1)
class Net(nn.Module):
    def __init__(s,i,o,h=512):
        super().__init__(); s.f=nn.Sequential(nn.Linear(i,h),nn.GELU(),nn.Linear(h,h),nn.GELU(),nn.Linear(h,h),nn.GELU(),nn.Linear(h,o))
    def forward(s,x): return s.f(x)
def fit(X,T,epochs=40,bce=False,lr=1e-3):
    X=torch.tensor(X,dtype=torch.float32); T=torch.tensor(T,dtype=torch.float32)
    mx,sx=X.mean(0),X.std(0)+1e-6; mt,stt=(T.mean(0),T.std(0)+1e-6) if not bce else (0.,1.)
    net=Net(X.shape[1],T.shape[1]); opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=1e-5)
    n=len(X); steps=6000; sched=torch.optim.lr_scheduler.OneCycleLR(opt,lr,total_steps=steps)
    for it in range(steps):
        i=torch.randint(0,n,(512,)); p=net((X[i]-mx)/sx)
        loss=nn.functional.binary_cross_entropy_with_logits(p,T[i]) if bce else ((p-(T[i]-mt)/stt)**2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval()
    def f(x):
        with torch.no_grad():
            x=torch.tensor(x,dtype=torch.float32); sh=x.shape[:-1]; o=net((x.reshape(-1,x.shape[-1])-mx)/sx)
            o=o if bce else o*stt+mt
            return o.reshape(*sh,-1).numpy().astype(np.float64)
    return f
t0=time.time()
FS,TY=feat(S),tgt(Y,S); AI=[0,1]         # anchor: pusher end position (displacement units)
mu = fit(np.concatenate([FS,A],-1), TY)                                   # factual predictor mu(s,a)
Cmap = fit(np.concatenate([FS,TY[:,AI]],-1), TY)                           # anchor-to-outcome map C(s, y_A)
rA = TY[:,AI]-mu(np.concatenate([FS,A],-1))[:,AI]
Gres = fit(np.concatenate([FS,A,rA],-1), TY)                               # residual-conditioned predictor G(s,a,r_A)
gate = fit(np.concatenate([FS,A],-1), (Cn>0).astype(np.float32)[:,None], bce=True)   # contact classifier (privileged labels)
print("fits %.0fs  NTR %d"%(time.time()-t0,NTR))
hf,ht=feat(HO["S"]),tgt(HO["Y"],HO["S"]); pm=mu(np.concatenate([hf,HO["A"]],-1)); pc=Cmap(np.concatenate([hf,ht[:,AI]],-1))
f2=lambda p,j: 100*(1-((p[:,j]-ht[:,j])**2).sum()/((ht[:,j]-ht[:,j].mean(0))**2).sum())
print("held-out FACTUAL R2  mu(s,a): block xy %.1f vx %.1f vy %.1f w %.1f | C(s,y_A): block xy %.1f vx %.1f vy %.1f w %.1f"%(f2(pm,[2,3]),f2(pm,[8]),f2(pm,[9]),f2(pm,[10]),f2(pc,[2,3]),f2(pc,[8]),f2(pc,[9]),f2(pc,[10])))
Se,Ae,Ye=ev["S"],ev["A"],ev["Y"]; N,E,K,D=Ye.shape; W=np.random.default_rng(0).multinomial(N,np.full(N,1/N),size=64)
FSe=feat(Se); Te=tgt(Ye,Se[:,None,None,:])
fa=np.concatenate([np.broadcast_to(FSe[:,None],(N,K,FSe.shape[1])),Ae],-1); MU=mu(fa)        # (N,K,11) learned means
MUx=Te.mean(1)                                                                               # exact within-parent means
obs=Te[:,:,0]                                                                                # factual outcomes (N,E,11)
preds={}
preds["trained predictor"]=np.broadcast_to(MU[:,None],Te.shape)
preds["RR learned"]=obs[:,:,None]+MU[:,None]-MU[:,None,:1]
preds["RR exact means"]=obs[:,:,None]+MUx[:,None]-MUx[:,None,:1]
pg=gate(fa)[...,0]>0; same=(pg[:,None,:]==pg[:,None,:1])                                     # predicted-mode gate
preds["mode-gated RR"]=np.where(same[...,None],preds["RR learned"],preds["trained predictor"])
fb=np.broadcast_to(FSe[:,None,None],(N,E,K,FSe.shape[1]))
for nm,M in [("PCT-A (learned mu_A)",MU),("PCT-A (exact mu_A)",MUx)]:
    yA=obs[:,:,None,AI]+M[:,None,:,AI]-M[:,None,:1,AI]                                       # anchor transported by residual reuse
    p=Cmap(np.concatenate([fb,yA],-1)); p[...,AI]=yA; preds[nm]=p
    preds[nm+" +resid"]=p+(obs-Cmap(np.concatenate([fb[:,:,0],obs[...,AI]],-1)))[:,:,None]   # nested form: add factual reconstruction residual
r=obs[...,AI]-MU[:,None,0,AI]
preds["resid-conditioned G(s,a',r_A)"]=Gres(np.concatenate([fb,np.broadcast_to(Ae[:,None],(N,E,K,2)),np.broadcast_to(r[:,:,None],(N,E,K,2))],-1))
groups={"pusher xy":[0,1],"block xy":[2,3],"block cos,sin":[4,5],"block vx":[8],"block vy":[9],"block w":[10]}
so=metrics.simulator_only(Te[...,[8]],W)
print("%-34s"%"method"+"".join("%14s"%g for g in groups))
def row(nm,f): print("%-34s"%nm+"".join("%14.1f"%(100*f(g)) for g in groups.values()))
row("oracle w/o outcome",lambda g: metrics.simulator_only(Te[...,g],W)["oracle_r2"])
row("copy",lambda g: metrics.simulator_only(Te[...,g],W)["copy_r2"])
row("ideal RR (their metric)",lambda g: metrics.simulator_only(Te[...,g],W)["ideal_r2"])
print("%-34s"%"rho"+"".join("%14.3f"%metrics.simulator_only(Te[...,g],W)["rho"] for g in groups.values()))
for nm,p in preds.items(): row(nm,lambda g: metrics.scores(p[...,g],Te[...,g],obs[...,g],W)["r2"])
# contact-mode breakdown for block velocities
Ce=ev["C"]>0; sw=(Ce[:,:,1:]!=Ce[:,:,:1])
print("contact frac %.3f  mode-switch frac %.3f"%(Ce.mean(),sw.mean()))
g=[8,9,10]; tv=Te[:,:,1:][...,g]; tot=((tv-tv.mean((0,1,2)))**2).sum(-1).mean()
for nm in ["RR exact means","trained predictor","mode-gated RR","PCT-A (learned mu_A)","resid-conditioned G(s,a',r_A)"]:
    e2=((preds[nm][:,:,1:][...,g]-tv)**2).sum(-1)
    print("%-34s MSE/totvar same-mode %.3f  switch %.3f"%(nm,e2[~sw].mean()/tot,e2[sw].mean()/tot))
# model-implied interaction ratio (factual-only estimate of rho under the channel assumption) vs the simulator's rho
pm_=preds["PCT-A (learned mu_A)"]
print("%-34s"%"rho-hat from learned PCT model"+"".join("%14.3f"%metrics.interaction(pm_[...,g_],W)["rho"] for g_ in groups.values()))
print("%-34s"%"rho (simulator, paired)"+"".join("%14.3f"%metrics.interaction(Te[...,g_],W)["rho"] for g_ in groups.values()))
