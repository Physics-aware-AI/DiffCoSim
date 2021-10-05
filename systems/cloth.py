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
from baselines.lcp.impulse_lcp import ImpulseSolverLCP
from scipy.spatial.transform import Rotation

class Cloth(RigidBody):
    dt = 0.005
    integration_time = 0.2

    def __init__(
        self, 
        kwargs_file_name="default",
        n_side=8, 
        ms=[0.1], 
        ls=[0.5], 
        radii=[0.01], 
        mus=[0.05], 
        cors=[0.05], 
        bdry_lin_coef=[[0, 0, 0, 0]],
        spring_k = 200,
        min_stretch=0.98,
        max_stretch=1.02,
        g=9.81,
        is_homo=True,
        is_mujoco_like=False,
        is_lcp_data=False,
        is_lcp_model=False,
        dtype=torch.float64
    ):
        n_o = n_side * n_side
        # assert n_o == len(ms) == len(ls) == len(radii)
        assert is_homo
        assert not (is_mujoco_like and is_lcp_model)
        self.body_graph = BodyGraph()
        self.kwargs_file_name = kwargs_file_name
        self.ls = torch.tensor(ls[0], dtype=dtype)
        self.ms = torch.tensor(ms*(n_side**2), dtype=dtype)
        self.radii = torch.tensor(radii*(n_side**2), dtype=dtype)
        # self.body_graph.add_extended_body(0, ms[0], d=0, tether=(torch.zeros(3), 0))
        for i in range(0, n_o):
            self.body_graph.add_extended_body(i, ms[0], d=0)
        # fix the first point in the inertial frame
        # self.body_graph.add_joint(
        #     key1=0,
        #     pos1=torch.tensor([0.,0.,0], dtype=dtype),
        #     pos2=torch.tensor([0.,0.,0], dtype=dtype))

        # for i in range(n_side-1):
        #     for j in range(n_side-1):
        #         idx = i * n_side + j
        #         self.body_graph.add_edge(idx, idx+1, l=ls[0])
        #         self.body_graph.add_edge(idx, idx+n_side, l=ls[0])
        #         self.body_graph.add_edge(idx+1, idx+n_side, l=ls[0]*np.sqrt(2))
        # for i in range(n_side-1):
        #     self.body_graph.add_edge(n_side*(n_side-1)+i, n_side*(n_side-1)+i+1, l=ls[0])
        #     self.body_graph.add_edge(n_side-1 + i*n_side, n_side-1 + (i+1)*n_side, l=ls[0])
        self.populated_ls = torch.cat([
            torch.tensor(ls * (1 + 2 * (n_side-1) * n_side), dtype=dtype),
            torch.tensor(ls * ((n_side-1) * (n_side-1)), dtype=dtype) * torch.tensor(2, dtype=dtype).sqrt()
        ])

        matrix = torch.tensor([i for i in range(n_o)], dtype=torch.int).reshape(n_side, n_side)
        # the first constraint is the cloth attach to the origin with a spring
        idx_to_nodes0 = torch.zeros((1, 2), dtype=torch.int)
        idx_to_nodes1 = torch.stack([matrix[1:, :].reshape(-1), matrix[:-1, :].reshape(-1)], dim=-1)
        idx_to_nodes2 = torch.stack([matrix[:, 1:].reshape(-1), matrix[:, :-1].reshape(-1)], dim=-1)
        idx_to_nodes3 = torch.stack([matrix[1:, :-1].reshape(-1), matrix[:-1, 1:].reshape(-1)], dim=-1)
        # idx_to_nodes4 = torch.stack([matrix[1:, 1:].reshape(-1), matrix[:-1, :-1].reshape(-1)], dim=-1)

        limit_idx_to_o_idx = torch.cat(
            [idx_to_nodes0, idx_to_nodes1, idx_to_nodes2, idx_to_nodes3], dim=0
        )
        assert limit_idx_to_o_idx.shape[0] == 1 + 2 * (n_side-1) * n_side + (n_side-1) * (n_side-1)
        self.limit_idx_to_o_idx = limit_idx_to_o_idx

        self.n_o, self.n_p, self.d = n_o, 1, 3
        self.n = self.n_o * self.n_p
        self.n_side = n_side
        self.min_stretch = min_stretch
        self.max_stretch = max_stretch
        self.spring_k = spring_k
        self.g = g

        self.bdry_lin_coef = torch.tensor(bdry_lin_coef, dtype=dtype)
        self.n_c = 1 + 2 * (n_side-1) * n_side + (n_side-1) * (n_side-1) 
        assert len(mus) == len(cors) == 1
        self.mus = torch.tensor(mus*self.n_c, dtype=torch.float64)
        self.cors = torch.tensor(cors*self.n_c, dtype=torch.float64)
        self.is_homo = is_homo
        self.is_mujoco_like = is_mujoco_like
        self.is_lcp_model = is_lcp_model
        self.is_lcp_data = is_lcp_data

        if is_lcp_model:
            self.impulse_solver = ImpulseSolverLCP(
                dt = self.dt,
                n_o = self.n_o,
                n_p = self.n_p,
                d = self.d,
                ls = self.ls,
                bdry_lin_coef = self.bdry_lin_coef,
                check_collision = self.check_collision,
                cld_2did_to_1did = self.cld_2did_to_1did,
                DPhi = self.DPhi,
                get_limit_e_for_Jac = self.get_limit_e_for_Jac,
                limit_idx_to_o_idx = self.limit_idx_to_o_idx
            )     
        elif is_mujoco_like:
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
                get_limit_e_for_Jac = self.get_limit_e_for_Jac,
                limit_idx_to_o_idx = self.limit_idx_to_o_idx
            )   
        else:
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
                get_limit_e_for_Jac = self.get_limit_e_for_Jac,
                limit_idx_to_o_idx = self.limit_idx_to_o_idx
            )   

    def __str__(self):
        if self.is_mujoco_like:
            return f"{self.__class__.__name__}_{self.kwargs_file_name}_mujoco"
        elif self.is_lcp_data:
            return f"{self.__class__.__name__}_{self.kwargs_file_name}_lcp"
        else:
            return f"{self.__class__.__name__}_{self.kwargs_file_name}"
        # return f"{self.__class__.__name__}{self.kwargs_file_name}"

    def potential(self, x):
        M = self.M.to(dtype=x.dtype)
        gravity_potential = self.g * (M @ x)[..., 2].sum(1) 
        actual_ls = self.get_actual_ls(x)
        spring_potential = (self.spring_k * (actual_ls - self.populated_ls.type_as(actual_ls))**2).sum(1)
        return gravity_potential + spring_potential

    def sample_initial_conditions(self, N, dtype=torch.float64):
        n = len(self.body_graph.nodes)
        xv_list = []
        ptr = 0
        while ptr < N:
            euler = np.zeros((N, 3))
            euler[:, 0] = np.random.rand(N) * 2 * np.pi
            euler[:, 1] = (np.random.rand(N) - 0.5) * 2 * np.pi / 4 - np.pi / 2
            euler[:, 2] = np.random.rand(N) * 0.2
            
            R = Rotation.from_euler("ZXZ", euler).as_matrix() # N, 3, 3
            R = torch.from_numpy(R)
            # x0 is the position of the 0th node.
            x0 = torch.randn(N, 3) 
            x0 = x0 / (x0**2).sum(dim=1, keepdim=True).sqrt() * self.ls 
            # x0 [:, 2] = (torch.rand(N) - 0.5)*0.5 + 0.5
            x = torch.zeros(N, self.n_o, 3)
            for i in range(self.n_side):
                for j in range(self.n_side):
                    idx = i*self.n_side + j
                    x[:, idx] = x0 + i * self.ls * R[..., 0] + j * self.ls * R[..., 1]

            # com = x.sum(1) / self.n_o # N, 3
            # x[..., 0] = x[..., 0] - com[:, 0:1]
            # x[..., 1] = x[..., 1] - com[:, 1:2]
            # v = (torch.rand(N, self.n_o, 1) - 0.5)*0.01 * R[..., 2].unsqueeze(1) # N, n_o, 3
            v = torch.zeros(N, self.n_o, 3)
            is_collide, *_ = self.check_collision(x)
            xv = torch.stack([x, v], dim=1)
            xv_list.append(xv[torch.logical_not(is_collide)])
            ptr += sum(torch.logical_not(is_collide))
        return torch.cat(xv_list, dim=0)[0:N]

    def check_collision(self, x):
        bs, n, _ = x.shape
        # is_cld, is_cld_bdry, dist_bdry = self.check_boundary_collision(x)
        is_cld_bdry = torch.zeros(bs, n, 0, dtype=torch.bool, device=x.device)
        dist_bdry = torch.zeros(bs, n, 0, dtype=x.dtype, device=x.device)
        is_cld_ij = torch.zeros(bs, n, n, dtype=torch.bool, device=x.device)
        dist_ij = torch.zeros(bs, n, n, dtype=x.dtype, device=x.device)
        is_cld, is_cld_limit, dist_limit = self.check_limit(x)
        return is_cld, is_cld_ij, is_cld_bdry, is_cld_limit, dist_ij, dist_bdry, dist_limit

    def check_limit(self, x):
        # get length limit
        actual_ls = self.get_actual_ls(x)
        dist_1_ls = self.max_stretch * self.populated_ls.type_as(x) - actual_ls # maximum stretch
        dist_2_ls = actual_ls - self.min_stretch * self.populated_ls.type_as(x)
        dist_ls_limit = torch.stack([dist_1_ls, dist_2_ls], dim=-1) # (bs, n_c, 2)
        is_cld_limit = dist_ls_limit < 0 # (bs, n_c, 2)

        is_cld = is_cld_limit.sum([1, 2]) > 0
        return is_cld, is_cld_limit, dist_ls_limit

    # def check_boundary_collision(self, x):
    #     coef = self.bdry_lin_coef / (self.bdry_lin_coef[:, 0:1] ** 2 + 
    #                 self.bdry_lin_coef[:, 1:2] ** 2 + self.bdry_lin_coef[:, 2:3]**2).sqrt() # n_bdry, 4
    #     x_one = torch.cat(
    #         [x, torch.ones(*x.shape[:-1], 1, dtype=x.dtype, device=x.device)],
    #         dim=-1
    #     ).unsqueeze(-2) # bs, n, 1, 4
    #     dist = (x_one * coef.type_as(x)).sum(-1) # bs, n, n_bdry
    #     dist_bdry = dist - self.radii[:, None].type_as(x)
    #     is_collide_bdry = dist_bdry < 0 # bs, n, n_bdry
    #     is_collide = is_collide_bdry.sum([1, 2]) > 0
    #     return is_collide, is_collide_bdry, dist_bdry

    def get_diff_ls(self, x):
        *bsT, n, _ = x.shape
        x_view = x.view(*bsT, self.n_side, self.n_side, 3)
        diff_0 = x_view[..., 0, 0, :]
        diff_1 = x_view[..., 1:, :, :] - x_view[..., :-1, :, :] # (*bsT, n_side-1, n_side, 3)
        diff_2 = x_view[..., :, 1:, :] - x_view[..., :, :-1, :] # (*bsT, n_side, n_side-1, 3)
        diff_3 = x_view[..., 1:, :-1, :] - x_view[..., :-1, 1:, :] # (*bsT, n_side-1, n_side-1, 3)
        # diff_4 = x_view[..., 1:, 1:, :] - x_view[..., :-1, :-1, :] # (*bsT, n_side-1, n_side-1, 3)
        diff = torch.cat([
            diff_0.reshape(*bsT, -1, 3),
            diff_1.reshape(*bsT, -1, 3),
            diff_2.reshape(*bsT, -1, 3),
            diff_3.reshape(*bsT, -1, 3),
            # diff_4.reshape(*bsT, -1, 3)
        ], dim=-2) # (*bsT, n_c, 3)   
        return diff 

    def get_limit_e_for_Jac(self, x):
        diff = self.get_diff_ls(x)
        e_n_ls = diff # (*bsT, n_c, 3)  
        e_t1_ls = torch.zeros_like(e_n_ls)
        e_t1_ls[..., 1], e_t1_ls[..., 2] = -e_n_ls[..., 2], e_n_ls[..., 1]
        e_t2_ls = torch.cross(e_n_ls, e_t1_ls, dim=-1)
        e_n_ls = e_n_ls / (e_n_ls**2).sum(-1, keepdim=True).sqrt()
        e_t1_ls = e_t1_ls / (e_t1_ls**2).sum(-1, keepdim=True).sqrt()
        e_t2_ls = e_t2_ls / (e_t2_ls**2).sum(-1, keepdim=True).sqrt()
        return e_n_ls, e_t1_ls, e_t2_ls


    def get_actual_ls(self, x):
        diff = self.get_diff_ls(x)
        ls = (diff**2).sum(-1).sqrt()
        return ls

    def cld_2did_to_1did(self, cld_ij_ids, cld_bdry_ids, cld_limit_ids):
        # return cld_bdry_ids[:, 0]
        return cld_limit_ids[:, 0]

    @property
    def animator(self):
        return ClothAnimation
    
class ClothAnimation(Animation):
    def __init__(self, qt, body):
        # qt: T, n, d
        # from the super code
        self.qt = qt.detach().cpu().numpy()
        T, n, d = qt.shape
        assert d in (2, 3)
        self.fig = plt.figure()
        self.ax = self.fig.add_axes([0, 0, 1, 1], projection='3d') if d==3 else self.fig.add_axes([0,0,1,1])

        if d!=3: self.ax.set_aspect("equal")

        empty = d * [[]]
        self.objects = {
            'pts': sum([self.ax.plot(*empty, ms=10, color='k') for i in range(n)], []),
            'trails': sum([self.ax.plot(*empty, "-", color='lightgray') for i in range(n)], [])
        }

        ############
        self.body = body
        self.n_o = body.n_o
        self.n_p = body.n_p

        # plot the ground
        corner = 1.2
        xx = np.array([[-corner, corner],
                        [-corner, corner]])
        y_coor = body.bdry_lin_coef[0, 3].numpy()
        zz = np.array([[0.0, 0.0],
                        [0.0, 0.0]])
        yy = np.array([[-corner, -corner],
                        [corner, corner]])
        # self.ax.plot_surface(xx, yy, zz, color="azure", zorder=-1)

        # plot the cloth
        # TODO
        # x_min, x_max = self.ax.get_xlim()
        # y_min, y_max = self.ax.get_ylim()
        # x_min = x_min if x_min < -body.lb-0.1 else -body.lb-0.1
        # x_max = x_max if x_max > body.rb+0.1 else body.rb+0.1
        # self.ax.set_xlim(x_min, x_max)

        # x_min, y_min, x_max, y_max = -1.1, -1.1, 1.1, 1.1
        # self.ax.set_xlim(x_min, x_max)
        # self.ax.set_ylim(y_min, y_max)
        # self.ax.axis("off"),
        # self.fig.set_size_inches(10.5, 10.5)

        # set view angle 
        self.ax.view_init(elev=-0.0, azim=00)
        self.ax.dist = 15 # the larger the farther
        self.ax.axis("off")
        self.fig.set_size_inches(13.5, 13.5)

        # self.body = body
        self.G = body.body_graph
        empty = self.qt.shape[-1] * [[]]
        self.objects["links"] = sum([self.ax.plot(*empty, "-", color='k', linewidth=4) for _ in range(body.n_c)], [])
        # self.objects["pts"] = sum(
        #     [self.ax.plot(*empty, "o", ms=10*body.ms[i], c=self.colors[i]) for i in range(qt.shape[1])], []
        # )

    def update(self, i=0):
        T, n, d = self.qt.shape
        # plot links
        links = [
            np.stack([np.array([0, 0, 0]), self.qt[i, 0, :]], axis=1)
        ] 
        for idx in range(1, self.body.n_c):
            k, l = self.body.limit_idx_to_o_idx[idx, 0], self.body.limit_idx_to_o_idx[idx, 1]
            links.append(np.stack([self.qt[i, k, :], self.qt[i, l, :]], axis=1))
        for link, link_line in zip(links, self.objects["links"]):
            link_line.set_data(*link[:2])
            link_line.set_3d_properties(link[2])

        return super().update(i)

