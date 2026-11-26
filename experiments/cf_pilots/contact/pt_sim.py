"""Push-T (vendored simulator of the 2611.24778 code release) driven as in cwm/sims/pusht.py + controller.py, without rendering.
Deviation from the paper: the two prefix transitions use a scripted policy (toward the block, 25% uniform) instead of the
LeRobot diffusion policy (needs CUDA)."""
import os, sys, numpy as np
os.environ.setdefault("SDL_VIDEODRIVER","dummy"); os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT","1")
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE,"..","pylibs")); sys.path.insert(0, os.path.join(HERE,"..","repo","cwm"))
from pymunk.vec2d import Vec2d
from cwm.sims.vendor.pusht_env import PushTEnv
GAIN=1.472860284876826; SCALE=0.4739784696868478; LIM=0.35; NSTEPS=4; HZ=10
class Sim:
    def __init__(s):
        s.env=PushTEnv(render_size=96, relative=True, with_velocity=False, legacy=False, action_scale=100.)
        s.env.control_hz=HZ
    def reset(s, seed):
        e=s.env; e.seed(seed); e._setup(); rs=e.random_state
        st=np.array([rs.randint(50,450),rs.randint(50,450),rs.randint(100,400),rs.randint(100,400),rs.randn()*2*np.pi-np.pi])
        e._set_state(st); return s.state()
    def state(s):
        e=s.env; a,b,th=e.agent.position,e.block.position,e.block.angle
        return np.array([a[0],a[1],b[0],b[1],np.cos(th),np.sin(th),*e.agent.velocity,*e.block.velocity,e.block.angular_velocity])
    def native(s, goal):
        e=s.env; dt=1.0/e.sim_hz; e.n_contact_points=0
        for _ in range(e.sim_hz//e.control_hz):
            acc=e.k_p*(Vec2d(*goal)-e.agent.position)+e.k_v*(Vec2d(0,0)-e.agent.velocity)
            e.agent.velocity+=acc*dt; e.space.step(dt)
        return e.n_contact_points
    def step(s, a, eps):
        goal=256.+216.*(GAIN*np.asarray(a,float)+eps*np.array([1.,-1.])); c=0
        for _ in range(NSTEPS): c+=s.native(goal)
        return s.state(), c
    def behavior(s, rng):
        if rng.random()<0.25: return rng.uniform(-LIM,LIM,2)
        st=s.state(); tgt=st[2:4]+rng.normal(scale=25.,size=2)
        return np.clip((tgt-256.)/(216.*GAIN)+rng.normal(scale=.03,size=2),-LIM,LIM)
def eps_of(g): return SCALE*np.tanh(g)
def prefix(sim, seed):
    rng=np.random.default_rng([seed,317]); sim.reset(seed); hist=[]
    for t in range(2):
        a=sim.behavior(rng); e=eps_of(rng.standard_normal()); hist.append((a,e)); sim.step(a,e)
    return hist, sim.state()
def replay(sim, seed, hist):
    sim.reset(seed)
    for a,e in hist: sim.step(a,e)
