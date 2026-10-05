"""N1: cross-fitted ("propose on one noise fold, decide on another") CEM for stochastic latent world models.

Roles of the particle noise in a sampling-based planner
  (R1) ranking candidates / defining elites, i.e. the proposal distribution of the next CEM iteration,
  (R2) assessing and finally choosing the plan that is executed.
The baseline uses the same noise for both. Here the CEM iterations (R1) run on fold A (any coupling scheme, e.g.
common random numbers chained over replans), and the executed plan is chosen on a fresh fold B (R2) among a short list
S = {elite mean} U {elites of the last iteration}. Fold B is drawn independently of fold A and of everything that
selected S, so conditionally on S the fold-B estimate of every member of S is unbiased (Thm 1 in results/n1/REPORT.md).
Within fold B the noise is common across the members of S (CRN), which is free because S was already chosen.

Budget accounting (predictor rows): baseline iters*N*M; here sum_t N*MA[t] + |S|*MB. `equal_mb` returns the MB that
makes the totals equal.
"""
import math
import torch
from .d6_model import CoupledLatentModel
from .d6_noise import NoiseSource
from .risk_plan import OracleModel


def equal_mb(iters, N, M, MA, S):
    """fold-B particles per short-list member such that total rollouts equal iters*N*M."""
    rest = iters * N * M - N * sum(MA)
    assert rest > 0 and rest % S == 0, (rest, S)
    return rest // S


class CFLatent(CoupledLatentModel):
    """Learned energy-score latent model; rollouts driven by an explicit noise tensor u (E,Nb,M,H,K)."""

    def rollout_u(self, z0, acts, u_all, goal):
        with torch.no_grad():
            E, N, H, A = acts.shape
            Me = u_all.shape[2]
            z = z0[:, None, None, :].expand(E, N, Me, z0.shape[-1])
            acc = torch.zeros(E, N, Me, device=z0.device)
            failed = torch.zeros(E, N, Me, device=z0.device)
            for t in range(H):
                a = acts[:, :, t, None, :].expand(E, N, Me, A)
                u = u_all[:, :, :, t].expand(E, N, Me, self.m.noise_dim)
                z = self.spread_next(z, a, u)
                if self.stage_w > 0:
                    acc = acc + ((z - goal[:, None, None, :]) ** 2).sum(-1) / H
                if self.anchor is not None:
                    failed = torch.maximum(failed, (((z - self.anchor.z) ** 2).sum(-1) < self.anchor.tau).float())
            return torch.cat([z, acc[..., None], failed[..., None]], -1)

    def noise_dim_k(self):
        return self.m.noise_dim


class CFOracle(OracleModel):
    """True simulator; the noise tensor carries Gaussian u[...,0] whose sign is the wind direction (so that the same
    NoiseSource machinery (CRN, chain) applies)."""
    FAIL_COST = 1.5

    def begin_replan(self, reset_mask=None):
        self.noise.begin_replan(reset_mask)

    def rollout_u(self, p0, acts, u_all, goal):
        E, N, H, _ = acts.shape
        Me = u_all.shape[2]
        acc = torch.zeros(E, N, Me, device=p0.device)
        p = p0[:, None, None, :].expand(E, N, Me, 2).clone()
        stuck = torch.zeros(E, N, Me, dtype=torch.bool, device=p0.device)
        for t in range(H):
            a = acts[:, :, t, None, :].clamp(-1, 1)
            pn = p + self.step * a
            sign = torch.where(u_all[:, :, :, t, 0] >= 0, 1.0, -1.0).expand(E, N, Me)
            pn[..., 1] = pn[..., 1] + self.wind * sign
            pn = pn.clamp(0, 1)
            pn = torch.where(stuck[..., None], p, pn)
            stuck = stuck | self._in_pit(pn)
            p = pn
            if self.stage_w > 0:
                d = ((p - goal[:, None, None, :]) ** 2).sum(-1)
                acc = acc + torch.where(stuck, torch.full_like(d, self.FAIL_COST), d) / H
        out = torch.cat([p, stuck[..., None].float()], -1)
        return torch.cat([out, acc[..., None]], -1) if self.stage_w > 0 else out


@torch.no_grad()
def cf_cem(model, obs0, goal, H, N, iters, MA, MB, noiseA, gen, noise_k, n_short=3, elite_frac=0.1, init_std=0.7,
           decide='cf', short='mean+elites', return_diag=False, rho=0.0):
    """Cross-fitted CEM.

    model.rollout_u(obs0, acts (E,N,H,A), u (E,Nb,M,H,K), goal) -> features;  model.cost(feat, goal) -> (E,N,M)
    noiseA : NoiseSource for fold A (shared/independent, chained or not); sampled once per iteration at max(MA) particles
             and sliced to MA[t] (so that sticky sets stay valid when MA varies).
    MA     : list of length iters, particles per candidate in the proposal phase.
    MB     : fold-B particles per short-list member (fresh iid noise, common across the short list).
    rho    : correlation between fold-B noise and the (tiled) fold-A noise of the last iteration; 0 = independent folds
             (the proposed planner), 1 = replay of the proposal scenarios (no new information). Requires shared (CRN) fold A.
    decide : 'cf'   choose argmin of the fold-B score over S   (the proposed planner)
             'mean' return the elite mean (baseline CEM behaviour; fold B unused)
    Returns plan (E,H,A) [, diag dict].
    """
    E, A = obs0.shape[0], 2
    dv = obs0.device
    mu = torch.zeros(E, 1, H, A, device=dv)
    sd = torch.full((E, 1, H, A), init_std, device=dv)
    ne = max(n_short, int(N * elite_frac))
    Mmax = max(MA)
    el = None
    for t in range(iters):
        acts = (mu + sd * torch.randn(E, N, H, A, device=dv, generator=gen)).clamp(-1, 1)
        uA_full = noiseA.get(E, N, Mmax, H, gen, dv)
        u = uA_full[:, :, :MA[t]]
        c = model.cost(model.rollout_u(obs0, acts, u, goal), goal)                      # (E,N,MA[t])
        s = c.mean(-1)
        sv, idx = s.topk(ne, dim=1, largest=False)
        el = torch.gather(acts, 1, idx[:, :, None, None].expand(-1, -1, H, A))           # (E,ne,H,A) best first
        mu = el.mean(1, keepdim=True)
        sd = el.std(1, keepdim=True).clamp_min(0.05)
    if decide == 'mean':
        return (mu[:, 0], dict(sA_best=sv[:, 0], elite0=el[:, 0])) if return_diag else mu[:, 0]
    # ---- decision phase on a fresh fold
    cand = torch.cat([mu, el[:, :n_short]], 1) if short == 'mean+elites' else el[:, :n_short]
    uB = torch.randn(E, 1, MB, H, noise_k, device=dv, generator=gen)                     # fresh, common across S
    if rho > 0:
        assert uA_full.shape[1] == 1, 'rho > 0 needs shared (CRN) fold A'
        tile = uA_full[:, :, torch.arange(MB, device=dv) % MA[-1]]
        uB = math.sqrt(1 - rho * rho) * uB + rho * tile
    cB = model.cost(model.rollout_u(obs0, cand, uB, goal), goal)                         # (E,|S|,MB)
    sB = cB.mean(-1)
    best = sB.argmin(1)
    plan = torch.gather(cand, 1, best[:, None, None, None].expand(-1, 1, H, A))[:, 0]
    if not return_diag:
        return plan
    # optimism diagnostic: fold-A score of the best elite (in-sample, selected on) versus its fold-B score
    off = 1 if short == 'mean+elites' else 0
    diag = dict(elite0=el[:, 0], choice=best, sA_best=sv[:, 0], sB_of_Abest=sB[:, off], sB_min=sB.min(1).values, sB_mu=sB[:, 0] if off else None)
    return plan, diag
