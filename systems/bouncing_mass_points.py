import torch
import numpy as np
from pytorch_lightning import seed_everything
from .rigid_body import RigidBody, BodyGraph
from utils import Animation
from matplotlib import collections as mc
from matplotlib.patches import Circle

class BouncingMassPoints(RigidBody):
    dt = 0.01
    integration_time = 2.0

    def __init__(self, n_balls=2, m=None, l=None, g=9.81, mu=0.1, cor=0.0):
        assert n_balls >= 1
        self.body_graph = BodyGraph()
        self.arg_str = f"n{n_balls}m{m or 'r'}l{l or 'r'}"
        seed_everything(0)
        ms = [0.6 + 0.8*np.random.rand() for _ in range(n_balls)] if m is None else n_balls*[m]
        ls = [0.06 + 0.08*np.random.rand() for _ in range(n_balls)] if l is None else n_balls*[l]
        for i in range(0, n_balls):
            self.body_graph.add_extended_body(i, ms[i], d=2)
        self.ms, self.ls, self.n, self.d, self.D, self.g = ms, ls, n_balls, 2, n_balls, g
        self.bdry_lin_coef = torch.tensor([[1, 0, 0],
                                              [0, 1, 0],
                                              [-1, 0, 1],
                                              [0, -1, 1]], dtype=torch.float32)
        # mu
        self.mus = mu * torch.ones(n_balls*(n_balls-1)//2 + n_balls*self.bdry_lin_coef.shape[0], dtype=torch.float32)
        self.cors = cor * torch.ones_like(self.mus)


    def __str__(self):
        return f"{self.__class__.__name__}{self.arg_str}"
    
    def potential(self, x):
        # x: (bs, n, d)
        M = self.M.to(x.device, x.dtype)
        # g = self.g.to(x.device, x.dtype)
        return self.g * (M @ x)[..., 1].sum(1)

    def sample_initial_conditions(self, N):
        n = len(self.body_graph.nodes)
        x_list = []
        ptr = 0
        while ptr < N:
            x0 = torch.rand(N, n, 2)
            is_collide, *_ = self.check_collision(x0)
            x_list.append(x0[torch.logical_not(is_collide)])
            ptr += sum(torch.logical_not(is_collide))
        x0 = torch.cat(x_list, dim=0)[0:N]
        x_dot0 = torch.randn(N, n, 2) * 0.2
        return torch.stack([x0, x_dot0], dim=1) # (N, 2, n, d) 

    def check_collision(self, x):
        is_cld_0, is_cld_ij, dist_ij = self.check_ball_collision(x)
        is_cld_1, is_cld_bdry, dist_bdry = self.check_boundry_collision(x)
        return torch.logical_or(is_cld_0, is_cld_1), is_cld_ij, is_cld_bdry, dist_ij, dist_bdry

    def check_ball_collision(self, x):
        # x: (bs, n, 2)
        bs, n, _ = x.shape
        is_collide = torch.zeros(bs, dtype=torch.bool, device=x.device)
        dist_ij = torch.zeros(bs, n, n, dtype=x.dtype, device=x.device)
        is_collide_ij = torch.zeros(bs, n, n, dtype=torch.bool, device=x.device)
        if n == 1:
            return is_collide, is_collide_ij, dist_ij
        for i in range(n-1):
            for j in range(i+1, n):
                dist_ij[:, i, j] = ((x[:, i] - x[:, j]) ** 2).sum(1).sqrt() - (self.ls[i] + self.ls[j])
                is_collide_ij[:, i, j] =  dist_ij[:, i, j] < 0
        is_collide = is_collide_ij.sum([1, 2]) > 0
        return is_collide, is_collide_ij, dist_ij

    def check_boundry_collision(self, x):
        coef = self.bdry_lin_coef / (self.bdry_lin_coef[:, 0:1] ** 2 + 
                    self.bdry_lin_coef[:, 1:2] ** 2).sqrt() # n_bdry, 3
        x_one = torch.cat(
            [x, torch.ones(*x.shape[:-1], 1, dtype=x.dtype, device=x.device)],
            dim=-1
        ).unsqueeze(-2) # bs, n, 1, 3
        dist = (x_one * coef).sum(-1) # bs, n, n_bdry
        dist_bdry = dist - torch.tensor(self.ls, dtype=torch.float32).unsqueeze(-1)
        is_collide_bdry = dist_bdry < 0 # bs, n, n_bdry
        is_collide = is_collide_bdry.sum([1, 2]) > 0
        return is_collide, is_collide_bdry, dist_bdry

    def cld_2did_to_1did(self, cld_ij_ids, cld_bdry_ids):
        n = self.n
        i, j = cld_ij_ids.unbind(dim=1)
        ij_1d_ids = j + n*i - (i+1) * (i+2) // 2
        ball_i, bdry_i = cld_bdry_ids.unbind(dim=1)
        bdry_1d_ids = ball_i * self.bdry_lin_coef.shape[0] + bdry_i
        bdry_1d_ids += n * (n-1) // 2
        return torch.cat([ij_1d_ids, bdry_1d_ids], dim=0)

    @property
    def animator(self):
        return BouncingMassPointsAnimation

class BouncingMassPointsAnimation(Animation):
    def __init__(self, qt, body):
        # at: T, n, d
        super().__init__(qt, body)
        # draw boundaries

        self.ax.set_xlim(-0.1, 1.1)
        self.ax.set_ylim(-0.1, 1.1)

        empty = self.qt.shape[-1] * [[]]
        self.objects["links"] = []
        self.objects["pts"] = sum(
            [self.ax.plot(*empty, "o", ms=body.ls[i], c=self.colors[i]) for i in range(qt.shape[1])], []
        )
        self.circles = [Circle(empty, body.ls[i], color=self.colors[i]) for i in range(qt.shape[1])] + []

        [self.ax.add_artist(circle) for circle in self.circles]


        lines = [[(0,0),(0,1)], [(0,1),(1,1)], [(1,1),(1,0)], [(1,0),(0,0)]]
        # c = np.array([(1,2,)])

        lc = mc.LineCollection(lines, linewidths=2)
        self.ax.add_collection(lc)

    def update(self, i=0):
        T, n, d = self.qt.shape
        for j in range(n):
            self.circles[j].center = self.qt[i, j][0], self.qt[i, j][1]
        return super().update(i)