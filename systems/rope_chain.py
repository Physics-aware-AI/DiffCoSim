"""
Non-commercial Use License

Copyright (c) 2021 Siemens Technology

This software, along with associated documentation files (the "Software"), is 
provided for the sole purpose of providing Proof of Concept. Any commercial 
uses of the Software including, but not limited to, the rights to sublicense, 
and/or sell copies of the Software are prohibited and are subject to a 
separate licensing agreement with Siemens. This software may be proprietary 
to Siemens and may be covered by patent and copyright laws. Processes 
controlled by the Software are patent pending.

The above copyright notice and this permission notice shall remain attached 
to the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from systems.rigid_body import RigidBody, BodyGraph
from utils import Animation
from pytorch_lightning import seed_everything
import numpy as np
import torch
import networkx as nx
from matplotlib import collections as mc
from matplotlib.patches import Circle
from models.impulse import ImpulseSolver
from models.impulse_mujoco import ImpulseSolverMujoco

PI = 3.1415927410125732

class RopeChain(RigidBody):
    dt = 0.005
    integration_time = 0.1

    def __init__(
        self, 
        kwargs_file_name="default",
        n_o=10, 
        g=9.81,
        ms=[0.1]*10, 
        ls=[0.05]*10, 
        radii=[0.01], 
        mus=[0.0], 
        cors=[0.0], 
        bdry_lin_coef=[[0, 0, 0]],
        angle_limit=0.3,
        is_homo=True,
        is_mujoco_like=False,
        dtype=torch.float64
    ):
        assert is_homo and n_o >= 2
        self.kwargs_file_name = kwargs_file_name
        self.ms = torch.tensor(ms, dtype=dtype)
        self.ls = torch.tensor(ls, dtype=dtype)
        self.radii = torch.tensor(radii, dtype=dtype)
        self.g = g
        self.n_o, self.n_p, self.d = n_o, 1, 2
        self.n = self.n_o * self.n_p
        self.angle_limit = angle_limit

        self.bdry_lin_coef = torch.tensor(bdry_lin_coef, dtype=dtype)
        self.n_c = n_o - 1
        self.mus = torch.tensor(mus*self.n_c, dtype=torch.float64)
        self.cors = torch.tensor(cors*self.n_c, dtype=torch.float64)
        self.is_homo = is_homo
        self.is_mujoco_like = is_mujoco_like
        
        self.body_graph = BodyGraph()
        self.body_graph.add_extended_body(0, ms[0], d=0, tether=(torch.zeros(2), ls[0]))
        for i in range(1, n_o):
            self.body_graph.add_extended_body(i, ms[i], d=0)
            self.body_graph.add_edge(i-1, i, l=ls[i])

        if not is_mujoco_like:
            self.impulse_solver = ImpulseSolver(
                dt = self.dt,
                n_o = self.n_o,
                n_p = self.n_p,
                d = self.d,
                ls = self.ls,
                bdry_lin_coef = self.bdry_lin_coef,
                check_collision = self.check_collision,
                cld_2did_to_1did = self.cld_2did_to_1did,
                DPhi = self.DPhi,
                get_limit_e_for_Jac=self.get_limit_e_for_Jac
            )        
        else:
            self.impulse_solver = ImpulseSolverMujoco(
                dt = self.dt,
                n_o = self.n_o,
                n_p = self.n_p,
                d = self.d,
                ls = self.ls,
                bdry_lin_coef = self.bdry_lin_coef,
                check_collision = self.check_collision,
                cld_2did_to_1did = self.cld_2did_to_1did,
                DPhi = self.DPhi,
                get_limit_e_for_Jac=self.get_limit_e_for_Jac
            )   


    def __str__(self):
        if self.is_mujoco_like:
            return f"{self.__class__.__name__}{self.kwargs_file_name}_mujoco"
        else:
            return f"{self.__class__.__name__}{self.kwargs_file_name}"

    def potential(self, x):
        M = self.M.to(dtype=x.dtype)
        return self.g * (M @ x)[..., 1].sum(1) 

    def sample_initial_conditions(self, N, dtype=torch.float64):
        n = len(self.body_graph.nodes)
        xv0_list = []
        ptr = 0
        while ptr < N:
            q0 = [torch.rand([N], dtype=dtype) * 3.14]
            for i in range(n-1):
                q0_i = torch.rand([N], dtype=dtype) * 2 * self.angle_limit - self.angle_limit
                q0.append(q0[-1]+q0_i)
            q0 = torch.stack(q0, dim=-1)
            q_dot0 = torch.randn([N, n], dtype=dtype)
            q_q_dot0 = torch.stack([q0, q_dot0], dim=1)
            xv0 = self.angle_to_global_cartesian(q_q_dot0) # (N, 2, n, d)
            is_collide, *_ = self.check_collision(xv0[:, 0])
            xv0_list.append(xv0[torch.logical_not(is_collide)])
            ptr += sum(torch.logical_not(is_collide))
        return torch.cat(xv0_list, dim=0)[0:N]

    def check_collision(self, x):
        bs, n, _ = x.shape
        is_cld_ij = torch.zeros(bs, n, n, dtype=torch.bool, device=x.device)
        dist_ij = torch.zeros(bs, n, n, dtype=x.dtype, device=x.device)
        is_cld_bdry = torch.zeros(bs, n, 0, dtype=torch.bool, device=x.device)
        dist_bdry = torch.zeros(bs, n, 0).type_as(x)
        is_cld, is_cld_limit, dist_limit = self.check_limit(x)
        return is_cld, is_cld_ij, is_cld_bdry, is_cld_limit, dist_ij, dist_bdry, dist_limit

    def check_limit(self, x):
        bs, n, _ = x.shape
        x_dot = torch.zeros_like(x)
        q = self.global_cartesian_to_angle(torch.stack([x, x_dot], dim=1))[:,0] # (bs, n)
        delta_q = q[:, 1:] - q[:, :-1] # (bs, n-1)
        # make sure in range -pi to pi
        delta_q = torch.atan2(delta_q.sin(), delta_q.cos())
        dist_1 = self.angle_limit - delta_q # angle of the latter link is large
        dist_2 = self.angle_limit + delta_q # angle of the former link is large
        dist_limit = torch.stack([dist_1, dist_2], dim=-1) # (bs, n-1, 2)
        is_cld_limit = torch.stack([dist_1 < 0, dist_2 < 0], dim=-1) # (bs, n-1, 2)
        is_cld = is_cld_limit.sum([1, 2]) > 0
        return is_cld, is_cld_limit, dist_limit

    def get_limit_e_for_Jac(self, x):
        bs, n, _ = x.shape
        x_dot = torch.zeros_like(x)
        q = self.global_cartesian_to_angle(torch.stack([x, x_dot], dim=1))[:,0] # (bs, n)
        e_n = torch.stack([q.cos(), q.sin()], dim=-1) # (bs, n, 2)
        ls = self.get_actual_ls(x)
        e_n_div_l = e_n / ls[..., None]
        e_t_div_l = torch.zeros_like(e_n)
        e_t_div_l[..., 0], e_t_div_l[..., 1] = -e_n_div_l[..., 1], e_n_div_l[..., 0]
        return e_n_div_l, e_t_div_l

    def get_actual_ls(self, x):
        *bsT, n, _ = x.shape
        diff = x[..., 1:, :] - x[..., :-1, :] # (*bsT, n-1, 2)
        diff = torch.cat([x[..., 0:1, :], diff], dim=-2) # (*bsT, n, 2)
        ls = (diff[..., 0]**2 + diff[..., 1]**2).sqrt()
        return ls

    # def check_boundary_collision(self, x):
    #     coef = self.bdry_lin_coef / (self.bdry_lin_coef[:, 0:1] ** 2 + 
    #                 self.bdry_lin_coef[:, 1:2] ** 2).sqrt() # n_bdry, 3
    #     x_one = torch.cat(
    #         [x, torch.ones(*x.shape[:-1], 1, dtype=x.dtype, device=x.device)],
    #         dim=-1
    #     ).unsqueeze(-2) # bs, n, 1, 3
    #     dist = (x_one * coef).sum(-1) # bs, n, n_bdry
    #     dist_bdry = dist - self.radii[:, None]
    #     is_collide_bdry = dist_bdry < 0 # bs, n, n_bdry
    #     is_collide = is_collide_bdry.sum([1, 2]) > 0
    #     return is_collide, is_collide_bdry, dist_bdry

    def cld_2did_to_1did(self, cld_ij_ids, cld_bdry_ids, cld_limit_ids):
        return cld_limit_ids[:,0]

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
        assert rel_r_r_dot.ndim >= 3
        x, y = rel_r_r_dot[..., 0, :].chunk(2, dim=-1) # *NT1, *NT1
        vx, vy = rel_r_r_dot[..., 1, :].chunk(2, dim=-1)
        q = torch.atan2(x, -y)
        q_dot = torch.where(q < 1e-2, vx / (-y), vy / x)
        # q_unwrapped = torch.from_numpy(np.unwrap(q.detach().cpu().numpy(), axis=-2)).to(x.device, x.dtype)
        return torch.cat([q, q_dot], dim=-1) # *NT2

    def angle_to_cos_sin(self, q_q_dot):
        # input shape (*NT, 2, n)
        q, q_dot = q_q_dot.chunk(2, dim=-2)
        return torch.stack([q.cos(), q.sin(), q_dot], dim=-2)

    @property
    def animator(self):
        return PendulumAnimation
    
class PendulumAnimation(Animation):
    def __init__(self, qt, body):
        # qt: T, n, d
        super().__init__(qt, body)
        self.body = body
        self.n_o = body.n_o
        self.n_p = body.n_p
        self.G = body.body_graph
        empty = self.qt.shape[-1] * [[]]
        n_links = len(nx.get_node_attributes(self.G, "tether")) + len(self.G.edges)
        self.objects["links"] = sum([self.ax.plot(*empty, "-", color='k') for _ in range(n_links)], [])
        self.objects["pts"] = sum(
            [self.ax.plot(*empty, "o", ms=10*body.ms[i], c=self.colors[i]) for i in range(qt.shape[1])], []
        )

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
        return super().update(i)

