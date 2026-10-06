import numpy as np, torch, time, pilot_b as pb
t=time.time(); r=np.random.RandomState(0)
tasks=[pb.sample_task(np.random.RandomState(9999+i)) for i in range(80)]
n=24000
S=np.stack([r.uniform(-0.6,0.6,n), r.uniform(-0.45,0.45,n)],1); A=r.uniform(-1,1,(n,2)); S2=pb.env_step(S,A)
ens=pb.Ens(); pb.fit(ens,[list(S),list(A),list(S2)],0,epochs=120)
with torch.no_grad():
    St,At=torch.tensor(S,dtype=torch.float32),torch.tensor(A,dtype=torch.float32); P=ens.mean(St,At).numpy()
blocked=(np.abs(S2-S).sum(1)<1e-9)&(np.abs(S[:,0])<0.13)
print(n,'epochs120 success',pb.evaluate(ens,tasks,0),'blocked |err|',np.abs(P-S2)[blocked].mean(),'free |err|',np.abs(P-S2)[~blocked].mean(),round(time.time()-t),flush=True)
