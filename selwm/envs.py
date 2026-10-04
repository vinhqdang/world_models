"""Batched 2D navigation environments with exact dynamics (ground truth available)."""
import torch


class WallMaze:
    """Point agent in [0,1]^2 with thin axis-aligned walls.

    x' = x + step * clip(a, -1, 1); a move is rejected (agent stays) if the
    straight segment crosses a wall rectangle. Cost is squared distance to goal.
    """

    def __init__(self, walls=None, step=0.08, sub=8):
        # walls: list of (x0, y0, x1, y1)
        self.walls = walls if walls is not None else [(0.48, 0.0, 0.52, 0.70)]
        self.step = step
        self.sub = sub
        self.W = torch.tensor(self.walls, dtype=torch.float32)

    def _inside(self, p):  # p: (..., 2) -> bool (...)
        W = self.W.to(p.device)
        x, y = p[..., 0:1], p[..., 1:2]
        hit = (x >= W[:, 0]) & (x <= W[:, 2]) & (y >= W[:, 1]) & (y <= W[:, 3])
        return hit.any(-1)

    def transition(self, x, a):
        a = a.clamp(-1, 1)
        d = a * self.step
        blocked = torch.zeros(x.shape[:-1], dtype=torch.bool, device=x.device)
        for k in range(1, self.sub + 1):
            blocked |= self._inside(x + d * (k / self.sub))
        xn = torch.where(blocked.unsqueeze(-1), x, x + d)
        out = (xn < 0) | (xn > 1)
        xn = torch.where(out, x, xn)
        return xn

    def rollout(self, x0, acts):
        """x0: (B,2), acts: (B,H,2) -> states (B,H+1,2)."""
        xs = [x0]
        x = x0
        for t in range(acts.shape[1]):
            x = self.transition(x, acts[:, t])
            xs.append(x)
        return torch.stack(xs, 1)

    @staticmethod
    def cost(x, g):
        return ((x - g) ** 2).sum(-1)

    def sample_start_goal(self, n, gen):
        """Start left of wall, goal right of wall (needs a detour via the gap)."""
        s = torch.rand(n, 2, generator=gen) * torch.tensor([0.3, 0.5]) + torch.tensor([0.1, 0.05])
        g = torch.rand(n, 2, generator=gen) * torch.tensor([0.3, 0.5]) + torch.tensor([0.6, 0.05])
        return s, g

    def collect(self, n, gen, restrict=None):
        """Random-action exploration data (x, a, x')."""
        x = torch.rand(n, 2, generator=gen)
        if restrict is not None:
            x = restrict(x)
        a = torch.randn(n, 2, generator=gen).clamp(-1, 1)
        return x, a, self.transition(x, a)
