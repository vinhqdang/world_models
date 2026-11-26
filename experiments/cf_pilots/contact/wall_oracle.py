"""Wall world with an actuator-offset disturbance: structure-level headroom (no learning).
y = W(s, s + STEP*(a+eps)); wall x=0,|y|<0.4, blocked -> stay (results/p4/pilot_b.py rule)."""
import numpy as np
STEP=0.12; SIG=0.9
def W(s, sn):
    sn = np.clip(sn, -1, 1)
    cross = (s[...,0]*sn[...,0] < 0) | (np.abs(sn[...,0]) < 1e-9)
    t = np.where(np.abs(sn[...,0]-s[...,0])>1e-12, -s[...,0]/(sn[...,0]-s[...,0]+1e-12), 0.0)
    yc = s[...,1] + t*(sn[...,1]-s[...,1])
    blocked = cross & (np.abs(yc) < 0.4)
    return np.where(blocked[...,None], s, sn), blocked
def step(s,a,e): return W(s, s+STEP*(a+e))
def eps_draw(r, shape): return SIG*np.tanh(r.standard_normal(shape))
def parents(r,n): return np.stack([r.uniform(-0.3,0.3,n), r.uniform(-0.6,0.6,n)],-1)
if __name__=="__main__":
    r=np.random.default_rng(0); N=4000; E=256
    s=parents(r,N); a=r.uniform(-1,1,(N,2)); ap=r.uniform(-1,1,(N,2)); e=eps_draw(r,(N,2))
    y,bf=step(s,a,e); yp,bc=step(s,ap,e)
    # exact means by MC
    em=eps_draw(r,(N,E,2))
    mu_a = step(s[:,None],a[:,None],em)[0].mean(1); mu_ap = step(s[:,None],ap[:,None],em)[0].mean(1)
    rr = y+mu_ap-mu_a
    # PCT with true map: free -> exact abduction; blocked -> truncated posterior by MC rejection
    pct = np.empty_like(y)
    ehat = (y-s)/STEP - a                      # exact when not blocked and not outer-clipped
    pct_free,_ = step(s,ap,ehat)
    big = eps_draw(r,(N,4096,2))
    yb,bb = step(s[:,None],a[:,None],big)       # which draws reproduce 'blocked'
    ycf_all,_ = step(s[:,None],ap[:,None],big)
    w = bb.astype(float); w = w/np.maximum(w.sum(1,keepdims=True),1)
    post = (w[...,None]*ycf_all).sum(1)
    pct = np.where(bf[:,None], post, pct_free)
    def r2(p, m=None):
        m = np.ones(N,bool) if m is None else m
        d = (yp-s)[m]; return 1-((p-yp)[m]**2).sum()/((d-d.mean(0))**2).sum()
    sw = bf!=bc
    print("frac factual blocked %.3f  cf blocked %.3f  mode switch %.3f"%(bf.mean(),bc.mean(),sw.mean()))
    for nm,p in [("oracle no outcome",mu_ap),("copy",y),("ideal residual reuse",rr),("PCT true map (Bayes)",pct)]:
        print("%-24s all %.3f | same-mode %.3f | switch %.3f | fact free->cf blocked %.3f | fact blocked->cf free %.3f"%(
            nm, r2(p), r2(p,~sw), r2(p,sw), r2(p,(~bf)&bc), r2(p,bf&(~bc))))
