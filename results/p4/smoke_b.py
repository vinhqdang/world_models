import time, pilot_b as pb
t=time.time()
for st in ['crossregret','dagger','disagree','random']:
    c=pb.run(st,0,rounds=1,B=204,n0=300); print(st,c,round(time.time()-t),flush=True)
