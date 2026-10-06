import time, sys, numpy as np, torch
from common import *
t=time.time()
src=ObsMap(8,1,'tanh')
E,P=train_jepa(src,4,2000,5,0,steps=2500)
print('train',time.time()-t)
Pm=Pred(P).P
f=lambda o: E(o)
sr,fd=run_episodes(E,P,src,40,0)
print('oracle src success',sr,fd,time.time()-t)
