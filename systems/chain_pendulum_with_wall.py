from systems.rigid_body import RigidBody, BodyGraph
from utils import Animation
from pytorch_lightning import seed_everything
import numpy as np
import torch
import networkx as nx
from matplotlib import collections as mc
from matplotlib.patches import Circle

class ChainPendulum_w_Wall(RigidBody):
    dt = 0.01
    integration_time = 1.0

    def __init__(self, n_links=2, m=None, l=None, radius=0.1, mu=0.1, cor=0.0, lb=1, rb=1):
        self.body_graph = BodyGraph()
        self.arg_str = f"n{n_links}m{m or 'r'}l{l or 'r'}"
        seed_everything(0)
        ms = [0.6 + 0.8*np.random.rand() for _ in range(n_links)] if m is None else n_links*[m]
        ls = [0.6 + 0.8*np.random.rand() for _ in range(n_links)] if l is None else n_links*[l]
        self.body_graph.add_extended_body(0, ms[0], d=0, tether=(torch.zeros(2), ls[0]))
        for i in range(1, n_links):
            self.body_graph.add_extended_body(i, ms[i], d=0)
            self.body_graph.add_edge(i-1, i, l=ls[i])
        self.ms, self.n, self.d, self.D = ms, n_links, 2, n_links
        self.angular_dims = range(n_links)

        self.bdry_lin_coef = torch.tensor([[1, 0, lb],
                                              [-1, 0, rb]], dtype=torch.float32)
        self.radius = radius
        self.mus = mu * torch.ones(n_links*self.bdry_lin_coef.shape[0], dtype=torch.float32)
        self.cors = cor * torch.ones_like(self.mus)
        self.lb, self.rb = lb, rb


    def __str__(self):
        return f"{self.__class__.__name__}{self.arg_str}"

    def potential(self, r):
        M = self.M.to(dtype=r.dtype)
        return 9.81 * (M @ r)[..., 1].sum(1) 

    def sample_initial_conditions(self, N, dtype=torch.float64):
        n = len(self.body_graph.nodes)
        xv0_list = []
        ptr = 0
        while ptr < N:
            q0 = torch.rand([N, n], dtype=dtype) * 3.14
            q_dot0 = torch.randn([N, n], dtype=dtype)
            q_q_dot0 = torch.stack([q0, q_dot0], dim=1)
            xv0 = self.angle_to_global_cartesian(q_q_dot0) # (N, 2, n, d)
            is_collide, *_ = self.check_collision(xv0[:, 0])
            xv0_list.append(xv0[torch.logical_not(is_collide)])
            ptr += sum(torch.logical_not(is_collide))
        return torch.cat(xv0_list, dim=0)[0:N]

    def check_collision(self, x):
        bs, n, _ = x.shape
        is_cld, is_cld_bdry, dist_bdry = self.check_boundary_collision(x)
        is_cld_ij = torch.zeros(bs, n, n, dtype=torch.bool, device=x.device)
        dist_ij = torch.zeros(bs, n, n, dtype=x.dtype, device=x.device)
        return is_cld, is_cld_ij, is_cld_bdry, dist_ij, dist_bdry

    def check_boundary_collision(self, x):
        coef = self.bdry_lin_coef / (self.bdry_lin_coef[:, 0:1] ** 2 + 
                    self.bdry_lin_coef[:, 1:2] ** 2).sqrt() # n_bdry, 3
        x_one = torch.cat(
            [x, torch.ones(*x.shape[:-1], 1, dtype=x.dtype, device=x.device)],
            dim=-1
        ).unsqueeze(-2) # bs, n, 1, 3
        dist = (x_one * coef).sum(-1) # bs, n, n_bdry
        dist_bdry = dist - self.radius
        is_collide_bdry = dist_bdry < 0 # bs, n, n_bdry
        is_collide = is_collide_bdry.sum([1, 2]) > 0
        return is_collide, is_collide_bdry, dist_bdry

    def cld_2did_to_1did(self, cld_ij_ids, cld_bdry_ids):
        n = self.n
        link_i, bdry_i = cld_bdry_ids.unbind(dim=1)
        bdry_1d_ids = link_i * self.bdry_lin_coef.shape[0] + bdry_i
        return bdry_1d_ids

    def angle_to_global_cartesian(self, q_q_dot):
        *N2, n = q_q_dot.shape
        d = 2
        r_r_dot = torch.zeros(*N2, n, d, device=q_q_dot.device, dtype=q_q_dot.dtype)
        joint_r_r_dot = torch.zeros(*N2, d, device=q_q_dot.device, dtype=q_q_dot.dtype)

        l0 = self.body_graph.nodes[0]["tether"][1]
        joint_r_r_dot[..., 0, :] = self.body_graph.nodes[0]["tether"][0]
        joint_r_r_dot += self.angle_to_local_cartesian(l0, q_q_dot[..., 0])
        r_r_dot[..., 0, :] = joint_r_r_dot
        for (i_prev, i), l in nx.get_edge_attributes(self.body_graph, "l").items():
            joint_r_r_dot += self.angle_to_local_cartesian(l, q_q_dot[..., i])
            r_r_dot[..., i, :] += joint_r_r_dot
        return r_r_dot
    
    def angle_to_local_cartesian(self, l, q_q_dot):
        # input *bsT, 2, output *bsT, 2, 2
        *bsT, _ = q_q_dot.shape
        r_r_dot = torch.zeros(*bsT, 2, 2, device=q_q_dot.device, dtype=q_q_dot.dtype)
        r_r_dot[..., 0, 0] = l * q_q_dot[..., 0].sin() # x
        r_r_dot[..., 0, 1] = - l * q_q_dot[..., 0].cos() # y
        r_r_dot[..., 1, 0] = l * q_q_dot[..., 0].cos() * q_q_dot[..., 1] # x_dot
        r_r_dot[..., 1, 1] = l * q_q_dot[..., 0].sin() * q_q_dot[..., 1] # y_dot
        return r_r_dot

    def global_cartesian_to_angle(self, r_r_dot):
        *NT2, n, d = r_r_dot.shape
        q_q_dot = torch.zeros(*NT2, n, device=r_r_dot.device, dtype=r_r_dot.dtype)
        joint_r_r_dot = torch.zeros(*NT2, d, device=r_r_dot.device, dtype=r_r_dot.dtype)
        joint_r_r_dot[..., 0, :] = self.body_graph.nodes[0]["tether"][0] # position
        rel_r_r_dot = r_r_dot[..., 0, :] - joint_r_r_dot # *NT2, d
        q_q_dot[..., 0] += self.local_cartesian_to_angle(rel_r_r_dot)
        joint_r_r_dot += rel_r_r_dot
        for (i_prev, i), l in nx.get_edge_attributes(self.body_graph, "l").items():
            rel_r_r_dot = r_r_dot[..., i, :] - joint_r_r_dot
            q_q_dot[..., i] += self.local_cartesian_to_angle(rel_r_r_dot)
            joint_r_r_dot += rel_r_r_dot
        return q_q_dot
        
    def local_cartesian_to_angle(self, rel_r_r_dot):
        assert rel_r_r_dot.ndim >= 4
        x, y = rel_r_r_dot[..., 0, :].chunk(2, dim=-1) # *NT1, *NT1
        vx, vy = rel_r_r_dot[..., 1, :].chunk(2, dim=-1)
        q = torch.atan2(x, -y)
        q_dot = torch.where(q < 1e-2, vx / (-y), vy / x)
        q_unwrapped = torch.from_numpy(np.unwrap(q.detach().cpu().numpy(), axis=-2)).to(x.device, x.dtype)
        return torch.cat([q_unwrapped, q_dot], dim=-1) # *NT2

    def angle_to_cos_sin(self, q_q_dot):
        # input shape (*NT, 2, n)
        q, q_dot = q_q_dot.chunk(2, dim=-2)
        return torch.stack([q.cos(), q.sin(), q_dot], dim=-2)

    @property
    def animator(self):
        return Pendulum_w_Wall_Animation
    
class Pendulum_w_Wall_Animation(Animation):
    def __init__(self, qt, body):
        # qt: T, n, d
        super().__init__(qt, body)

        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()
        x_min = x_min if x_min < -body.lb-0.1 else -body.lb-0.1
        x_max = x_max if x_max > body.rb+0.1 else body.rb+0.1
        self.ax.set_xlim(x_min, x_max)

        # self.ax.set_xlim()
        # self.ax.set_ylim()
        # self.body = body
        self.G = body.body_graph
        empty = self.qt.shape[-1] * [[]]
        n_links = len(nx.get_node_attributes(self.G, "tether")) + len(self.G.edges)
        self.objects["links"] = sum([self.ax.plot(*empty, "-", color='k') for _ in range(n_links)], [])
        self.objects["pts"] = sum(
            [self.ax.plot(*empty, "o", ms=10*body.ms[i], c=self.colors[i]) for i in range(qt.shape[1])], []
        )
        self.circles = [Circle(empty, body.radius, color=self.colors[i]) for i in range(qt.shape[1])] + []

        [self.ax.add_artist(circle) for circle in self.circles]

        lines = [[(-body.lb, y_min), (-body.lb, y_max)], [(body.rb, y_min), (body.rb, y_max)]]
        lc = mc.LineCollection(lines, linewidths=2)
        self.ax.add_collection(lc)

    def update(self, i=0):
        links = [
            np.stack([loc.detach().cpu().numpy(), self.qt[i, k, :]], axis=1)
            for k, (loc, l) in nx.get_node_attributes(self.G, "tether").items()
        ] + [
            np.stack([self.qt[i, k, :], self.qt[i, l, :]], axis=1)
            for (k, l) in self.G.edges
        ]
        for link, link_line in zip(links, self.objects["links"]):
            link_line.set_data(*link[:2])
            if self.qt.shape[-1] == 3:
                link_line.set_3d_properties(link[2])
        T, n, d = self.qt.shape
        for j in range(n):
            self.circles[j].center = self.qt[i, j][0], self.qt[i, j][1]
        return super().update(i)
