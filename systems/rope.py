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
import matplotlib.pyplot as plt
from models.impulse import ImpulseSolver
from models.impulse_mujoco import ImpulseSolverMujoco

PI = 3.1415927410125732

class Rope(RigidBody):
    dt = 0.005
    integration_time = 0.1

    def __init__(
        self, 
        kwargs_file_name="default",
        n_o=10, 
        g=9.81,
        ms=[0.1], 
        ls=[0.05], 
        radii=[0.01], 
        mus=[0.0], 
        cors=[0.0], 
        bdry_lin_coef=[[0, 0, 0]],
        angle_limit=0.3,
        min_stretch=0.8,
        max_stretch=1.2,
        spring_k = 50,
        is_homo=True,
        is_mujoco_like=False,
        is_lcp_data=False,
        is_lcp_model=False,
        dtype=torch.float64
    ):
        assert is_homo and n_o >= 2
        self.kwargs_file_name = kwargs_file_name
        self.ms = torch.tensor(ms*n_o, dtype=dtype)
        self.ls = torch.tensor(ls*n_o, dtype=dtype)
        self.radii = torch.tensor(radii, dtype=dtype)
        self.g = g
        self.n_o, self.n_p, self.d = n_o, 1, 2
        self.n = self.n_o * self.n_p
        self.angle_limit = angle_limit
        self.min_stretch = min_stretch
        self.max_stretch = max_stretch
        self.spring_k = spring_k

        self.bdry_lin_coef = torch.tensor(bdry_lin_coef, dtype=dtype)
        self.n_c = 2*n_o - 1 # bending and strectch
        self.mus = torch.tensor(mus*self.n_c, dtype=torch.float64)
        self.cors = torch.tensor(cors*self.n_c, dtype=torch.float64)
        self.is_homo = is_homo
        self.is_mujoco_like = is_mujoco_like
        
        self.body_graph = BodyGraph()
        # self.body_graph.add_extended_body(0, ms[0], d=0, tether=(torch.zeros(2), ls[0]))
        # for i in range(1, n_o):
        #     self.body_graph.add_extended_body(i, ms[i], d=0)
        #     self.body_graph.add_edge(i-1, i, l=ls[i])
        for i in range(0, n_o):
            self.body_graph.add_extended_body(i, ms[0], d=0)

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
            return f"{self.__class__.__name__}_{self.kwargs_file_name}_mujoco"
        else:
            return f"{self.__class__.__name__}_{self.kwargs_file_name}"
        # return f"{self.__class__.__name__}{self.kwargs_file_name}"

    def potential(self, x):
        # x: (bs, n, d)
        # output V: (bs,)
        M = self.M.to(dtype=x.dtype)
        actual_ls = self.get_actual_ls(x) # (bs, n)
        spring_potential = (self.spring_k * (actual_ls - self.ls.type_as(actual_ls))**2).sum(1)
        gravity_potential = self.g * (M @ x)[..., 1].sum(1) 
        return gravity_potential + spring_potential

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
            xv0 = self.initial_angle_to_global_cartesian(q_q_dot0) # (N, 2, n, d)
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
        x_dot = torch.zeros_like(x) # bs, n, 2
        # first get angle limit
        q = self.global_cartesian_to_angle(torch.stack([x, x_dot], dim=1))[:,0] # (bs, n)
        delta_q = q[:, 1:] - q[:, :-1] # (bs, n-1)
        # make sure in range -pi to pi
        delta_q = torch.atan2(delta_q.sin(), delta_q.cos())
        dist_1_angles = self.angle_limit - delta_q # angle of the latter link is large
        dist_2_angles = self.angle_limit + delta_q # angle of the former link is large
        dist_angles_limit = torch.stack([dist_1_angles, dist_2_angles], dim=-1) # (bs, n-1, 2)
        # is_cld_angles_limit = dist_angles_limit < 0 # (bs, n-1, 2)
        # then get length limit
        delta_x = x[:, 1:] - x[:, :-1] # (bs, n-1, 2)
        delta_x = torch.cat([x[:, 0:1], delta_x], dim=1) # (bs, n, 2)
        actual_ls = self.get_actual_ls(x) # (bs, n)
        dist_1_ls = self.max_stretch*self.ls.type_as(x) - actual_ls # maximum stretch
        dist_2_ls = actual_ls - self.min_stretch*self.ls.type_as(x) # minimun stretch
        dist_ls_limit = torch.stack([dist_1_ls, dist_2_ls], dim=-1) # (bs, n, 2)
        # is_cld_ls_limit = dist_ls_limit < 0

        dist_limit = torch.cat([dist_angles_limit, dist_ls_limit], dim=1) # (bs, 2n-1, 2)
        is_cld_limit = dist_limit < 0 # (bs, 2n-1, 2)

        is_cld = is_cld_limit.sum([1, 2]) > 0
        return is_cld, is_cld_limit, dist_limit

    def get_limit_e_for_Jac(self, x):
        bs, n, _ = x.shape
        x_dot = torch.zeros_like(x)
        q = self.global_cartesian_to_angle(torch.stack([x, x_dot], dim=1))[:,0] # (bs, n)
        e_n_angles = torch.stack([q.cos(), q.sin()], dim=-1) # (bs, n, 2)
        e_n_ls = torch.stack([q.sin(), -q.cos()], dim=-1) # (bs, n, 2)
        ls = self.get_actual_ls(x)
        e_n_angles_div_ls = e_n_angles / ls[..., None]
        e_n = torch.cat([e_n_angles_div_ls, e_n_ls], dim=1) # (bs, 2n, 2)
        e_t = torch.zeros_like(e_n)
        e_t[..., 0], e_t[..., 1] = -e_n[..., 1], e_n[..., 0]
        return e_n, e_t

    def get_actual_ls(self, x):
        *bsT, n, _ = x.shape
        diff = x[..., 1:, :] - x[..., :-1, :] # (*bsT, n-1, 2)
        diff = torch.cat([x[..., 0:1, :], diff], dim=-2) # (*bsT, n, 2)
        ls = (diff**2).sum(-1).sqrt()
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

    def initial_angle_to_global_cartesian(self, q_q_dot):
        *N2, n = q_q_dot.shape
        d = 2
        r_r_dot = torch.zeros(*N2, n, d, device=q_q_dot.device, dtype=q_q_dot.dtype)
        ls = self.ls.type_as(q_q_dot)

        l0 = ls[0]
        joint_r_r_dot = self.initial_angle_to_local_cartesian(l0, q_q_dot[..., 0])
        r_r_dot[..., 0, :] = joint_r_r_dot
        for i in range(1, n):
            joint_r_r_dot += self.initial_angle_to_local_cartesian(ls[i], q_q_dot[..., i])
            r_r_dot[..., i, :] += joint_r_r_dot
        return r_r_dot
    
    def initial_angle_to_local_cartesian(self, l, q_q_dot):
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
        # joint_r_r_dot[..., 0, :] = self.body_graph.nodes[0]["tether"][0] # position
        rel_r_r_dot = r_r_dot[..., 0, :] - joint_r_r_dot # *NT2, d
        q_q_dot[..., 0] += self.local_cartesian_to_angle(rel_r_r_dot)
        joint_r_r_dot += rel_r_r_dot
        for i in range(1, n):
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

    # def angle_to_cos_sin(self, q_q_dot):
    #     # input shape (*NT, 2, n)
    #     q, q_dot = q_q_dot.chunk(2, dim=-2)
    #     return torch.stack([q.cos(), q.sin(), q_dot], dim=-2)

    @property
    def animator(self):
        return RopeAnimation
    
class RopeAnimation(Animation):
    def __init__(self, qt, body):
        # qt: T, n, d
        self.qt = qt.detach().cpu().numpy()
        T, n, d = qt.shape
        assert d in (2, 3)
        self.fig = plt.figure()
        self.ax = self.fig.add_axes([0, 0, 1, 1], projection='3d') if d==3 else self.fig.add_axes([0,0,1,1])

        empty = d * [[]]
        # self.colors = np.random.choice([f"C{i}" for i in range(15)], size=n, replace=False)
        if body.n_o < 15:
            self.colors = [f"C{i}" for i in range(15)]
            self.objects = {
                'pts': sum([self.ax.plot(*empty, ms=6, color=self.colors[i]) for i in range(n)], []),
                'trails': sum([self.ax.plot(*empty, "-", color=self.colors[i]) for i in range(n)], [])
            }
        else:
            self.objects = {
                'pts': sum([self.ax.plot(*empty, ms=6, color='k') for i in range(n)], []),
                'trails': sum([self.ax.plot(*empty, "-", color='gray') for i in range(n)], [])
            }
        self.body = body
        self.n_o = body.n_o
        self.n_p = body.n_p

        # self.ax.set_xlim(-0.53, 0.53)
        # self.ax.set_ylim(-0.59, 0.47)
        # self.ax.set_xlim(-0.1, 0.53)
        # self.ax.set_ylim(-0.2, 0.47)
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.axis("off"),
        self.fig.set_size_inches(10.5, 10.5)

        self.G = body.body_graph
        self.objects["links"] = sum([self.ax.plot([], [], "-", color='k', linewidth=4.5) for _ in range(self.n_o)], [])
        self.objects["pts"] = sum(
            [self.ax.plot([], [], "o", ms=10*body.ms[i], c="tab:blue") for i in range(qt.shape[1])], []
        )

    def update(self, i=0):
        # links = [
        #     np.stack([loc.detach().cpu().numpy(), self.qt[i, k, :]], axis=1)
        #     for k, (loc, l) in nx.get_node_attributes(self.G, "tether").items()
        # ] + [
        #     np.stack([self.qt[i, k, :], self.qt[i, l, :]], axis=1)
        #     for (k, l) in self.G.edges
        # ]
        self.objects["links"][0].set_data([0, self.qt[i, 0, 0]], [0, self.qt[i, 0, 1]])
        for j in range(1, self.n_o):
            self.objects["links"][j].set_data([self.qt[i, j-1, 0], self.qt[i, j, 0]], [self.qt[i, j-1, 1], self.qt[i, j, 1]])
        return super().update(i)

