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

import torch
import numpy as np
from pytorch_lightning import seed_everything
from .rigid_body import RigidBody, BodyGraph
from utils import Animation
from matplotlib import collections as mc
from matplotlib.patches import Circle
from models.contact_model import ContactModel
from models.contact_model_reg import ContactModelReg
from baselines.lcp.contact_model_lcp import ContactModelLCP

class BouncingDisks(RigidBody):
    dt = 0.01
    integration_time = 1.0

    def __init__(
        self,
        kwargs_file_name="default_args",
        n_o=1, 
        g=9.81,
        ms=[0.1], 
        ls=[0.1], 
        mus=[0.0, 0.0, 0.0, 0.0], 
        cors=[0.0, 0.0, 0.0, 0.0],
        bdry_lin_coef=[[1, 0, 0], [0, 1, 0], [-1, 0, 1], [0, -1, 1]],
        is_homo=False,
        is_reg_data=False,
        is_reg_model=False,
        is_lcp_data=False,
        is_lcp_model=False,
        dtype=torch.float64
    ):
        assert n_o == len(ms) == len(ls)
        assert not (is_reg_model and is_lcp_model)
        self.body_graph = BodyGraph()
        self.kwargs_file_name = kwargs_file_name
        self.ms = torch.tensor(ms, dtype=torch.float64)
        self.ls = torch.tensor(ls, dtype=torch.float64)
        for i in range(0, n_o):
            self.body_graph.add_extended_body(i, self.ms[i], d=2,
                    moments= self.ls[i]**2 /4 * torch.ones(2, dtype=torch.float64))
        self.g = g
        self.n_o, self.n_p, self.d = n_o, 3, 2
        self.n = self.n_o * self.n_p
        self.bdry_lin_coef = torch.tensor(bdry_lin_coef, dtype=torch.float64)
        self.n_c = n_o * (n_o - 1) // 2 + n_o * self.bdry_lin_coef.shape[0]
        if is_homo:
            assert len(mus) == len(cors) == 1
            self.mus = torch.tensor(mus*self.n_c, dtype=torch.float64)
            self.cors = torch.tensor(cors*self.n_c, dtype=torch.float64)
        else:
            assert len(mus) == len(cors) == self.n_c
            self.mus = torch.tensor(mus, dtype=torch.float64)
            self.cors = torch.tensor(cors, dtype=torch.float64)
        self.is_homo = is_homo
        self.is_reg_model = is_reg_model
        self.is_reg_data = is_reg_data
        self.is_lcp_model = is_lcp_model
        self.is_lcp_data = is_lcp_data

        delta = torch.tensor([[-1, 1, 0], [-1, 0, 1]], dtype=torch.float64) # 2, 3

        if is_lcp_model:
            self.impulse_solver = ContactModelLCP(
                dt = self.dt,
                n_o = self.n_o,
                n_p = self.n_p,
                d = self.d,
                ls = self.ls,
                bdry_lin_coef = self.bdry_lin_coef,
                check_collision = self.check_collision,
                cld_2did_to_1did = self.cld_2did_to_1did,
                DPhi = self.DPhi,
                delta=delta
            )
        elif is_reg_model:
            self.impulse_solver = ContactModelReg(
                dt = self.dt,
                n_o = self.n_o,
                n_p = self.n_p,
                d = self.d,
                ls = self.ls,
                bdry_lin_coef = self.bdry_lin_coef,
                check_collision = self.check_collision,
                cld_2did_to_1did = self.cld_2did_to_1did,
                DPhi = self.DPhi,
                delta=delta
            )
        else:
            self.impulse_solver = ContactModel(
                dt = self.dt,
                n_o = self.n_o,
                n_p = self.n_p,
                d = self.d,
                ls = self.ls,
                bdry_lin_coef = self.bdry_lin_coef,
                check_collision = self.check_collision,
                cld_2did_to_1did = self.cld_2did_to_1did,
                DPhi = self.DPhi,
                delta=delta
            )

    def __str__(self):
        if self.is_reg_data:
            return f"{self.__class__.__name__}_{self.kwargs_file_name}_reg"
        elif self.is_lcp_data:
            return f"{self.__class__.__name__}_{self.kwargs_file_name}_lcp"
        else:
            return f"{self.__class__.__name__}_{self.kwargs_file_name}"

    def potential(self, x):
        # x: (bs, n, d)
        M = self.M.to(x.device, x.dtype)
        # g = self.g.to(x.device, x.dtype)
        return self.g * (M @ x)[..., 1].sum(1)

    def sample_initial_conditions(self, N):
        n_o = self.n_o
        # com initial location
        r_list = []
        ptr = 0
        while ptr < N:
            r0 = torch.rand(N, n_o, 2)
            is_collide, *_ = self.check_collision(r0)
            r_list.append(r0[torch.logical_not(is_collide)])
            ptr += sum(torch.logical_not(is_collide))
        r0 = torch.cat(r_list, dim=0)[0:N]
        r_dot0 = torch.randn(N, n_o, 2) * 0.5 
        theta0 = torch.rand(N, n_o, 1) * 3.14 * 2
        theta_dot0 = torch.randn(N, n_o, 1) * 0.5
        r_theta0 = torch.cat([r0, theta0], dim=2)
        r_theta_dot0 = torch.cat([r_dot0, theta_dot0], dim=2)
        r_thetas = torch.stack([r_theta0, r_theta_dot0], dim=1) # (N, 2, n_o, 3)
        return self.generalized_to_cartesian(r_thetas)

    def generalized_to_cartesian(self, r_thetas):
        """ input: (*bsT, 2, n_o, 3), output: (*bsT, 2, n_o, n_p, d) """
        *bsT, _, n_o, _ = r_thetas.shape
        xv_com = r_thetas[..., 0:2] # (*bsT, 2, n_o, 2)
        theta = r_thetas[..., 0, :, 2] # (*bsT, n_o)
        theta_dot = r_thetas[..., 1, :, 2] # (*bsT, n_o)
        Rdot_RT = torch.zeros([*bsT, n_o, 2, 2], dtype=xv_com.dtype, device=xv_com.device)
        Rdot_RT[..., 0, 1] = - theta_dot
        Rdot_RT[..., 1, 0] = theta_dot
        R = torch.zeros_like(Rdot_RT)
        R[..., 0, 0] = theta.cos()
        R[..., 0, 1] = - theta.sin()
        R[..., 1, 0] = theta.sin()
        R[..., 1, 1] = theta.cos()
        Rdot = Rdot_RT @ R # (*bsT, n_o, 2, 2)
        xv_u_vec_body = torch.stack([R, Rdot], dim=-4).transpose(-2, -1) # (*bsT, 2, n_o, n_p-1, d)
        xv_u_vec = xv_u_vec_body + xv_com[..., None, :]
        return torch.cat([xv_com[..., None, :], xv_u_vec], dim=-2).reshape(*bsT, 2, self.n, 2)

    def check_collision(self, x):
        bs = x.shape[0]
        is_cld_0, is_cld_ij, dist_ij = self.check_ij_collision(x)
        is_cld_1, is_cld_bdry, dist_bdry = self.check_boundry_collision(x)
        is_cld_limit = torch.zeros(bs, 0, 2, dtype=torch.bool, device=x.device)
        dist_limit = torch.zeros(bs, 0, 2).type_as(x)
        return torch.logical_or(is_cld_0, is_cld_1), is_cld_ij, is_cld_bdry, is_cld_limit, dist_ij, dist_bdry, dist_limit

    def check_ij_collision(self, x):
        # x: (bs, n_o, 2) or (bs, n_o*n_p, 2)
        bs, n_o, n_p = x.shape[0], self.n_o, self.n_p
        x_l = x.reshape(bs, n_o, -1, 2)[..., 0, :]

        is_collide = torch.zeros(bs, dtype=torch.bool, device=x.device)
        dist_ij = torch.zeros(bs, n_o, n_o, dtype=x.dtype, device=x.device)
        is_collide_ij = torch.zeros(bs, n_o, n_o, dtype=torch.bool, device=x.device)
        if n_o == 1:
            return is_collide, is_collide_ij, dist_ij
        ls = self.ls.to(device=x.device, dtype=x.dtype)
        for i in range(n_o-1):
            for j in range(i+1, n_o):
                dist_ij[:, i, j] = ((x_l[:, i] - x_l[:, j]) ** 2).sum(1).sqrt() - (ls[i] + ls[j])
                is_collide_ij[:, i, j] =  dist_ij[:, i, j] < 0
        is_collide = is_collide_ij.sum([1, 2]) > 0
        return is_collide, is_collide_ij, dist_ij

    def check_boundry_collision(self, x):
        # x: (bs, n_o, 2) or (bs, n_o*n_p, 2)
        bs, n_o, n_p = x.shape[0], self.n_o, self.n_p
        x_l = x.reshape(bs, n_o, -1, 2)[..., 0, :]
        bdry_lin_coef = self.bdry_lin_coef.to(x.device, x.dtype)
        ls = self.ls.to(x.device, x.dtype)
        coef = bdry_lin_coef / (bdry_lin_coef[:, 0:1] ** 2 + 
                    bdry_lin_coef[:, 1:2] ** 2).sqrt() # n_bdry, 3
        x_one = torch.cat(
            [x_l, torch.ones(*x_l.shape[:-1], 1, dtype=x.dtype, device=x.device)],
            dim=-1
        ).unsqueeze(-2) # bs, n_o, 1, 3
        dist = (x_one * coef).sum(-1) # bs, n_o, n_bdry
        dist_bdry = dist - ls[..., None]
        is_collide_bdry = dist_bdry < 0 # bs, n_o, n_bdry
        is_collide = is_collide_bdry.sum([1, 2]) > 0
        return is_collide, is_collide_bdry, dist_bdry

    def cld_2did_to_1did(self, cld_ij_ids, cld_bdry_ids, cld_limit_ids):
        n_o = self.n_o
        i, j = cld_ij_ids.unbind(dim=1)
        ij_1d_ids = j + n_o*i - (i+1) * (i+2) // 2
        ball_i, bdry_i = cld_bdry_ids.unbind(dim=1)
        bdry_1d_ids = ball_i * self.bdry_lin_coef.shape[0] + bdry_i
        bdry_1d_ids += n_o * (n_o-1) // 2
        return torch.cat([ij_1d_ids, bdry_1d_ids], dim=0)

    @property
    def animator(self):
        if not hasattr(self, "_animator"):
            self._animator = BouncingDisksAnimation
        return self._animator
    
    @animator.setter
    def animator(self, val):
        self._animator = val

class BouncingDisksAnimation(Animation):
    def __init__(self, qt, body):
        # qt: T, n, d
        super().__init__(qt, body)
        # draw boundaries
        self.body = body
        self.n_o = body.n_o
        self.n_p = body.n_p

        self.ax.set_xlim(-0.1, 1.1)
        self.ax.set_ylim(-0.1, 1.1)
        self.ax.axis("off"),
        self.fig.set_size_inches(10.5, 10.5)

        empty = self.qt.shape[-1] * [[]]
        self.objects["links"] = []
        self.objects["pts"] = sum(
            [self.ax.plot(*empty, "o", ms=body.ls[i], c=self.colors[i]) for i in range(self.n_o)], []
        )
        self.circles = [Circle([[0], [0]], body.ls[i], color=self.colors[i]) for i in range(self.n_o)] + []

        [self.ax.add_artist(circle) for circle in self.circles]


        lines = [[(0,0),(0,1)], [(0,1),(1,1)], [(1,1),(1,0)], [(1,0),(0,0)]]
        # c = np.array([(1,2,)])

        lc = mc.LineCollection(lines, linewidths=4, color="k")
        self.ax.add_collection(lc)
        segments = [np.zeros([2, 2]) for i in range(self.n_o)]
        self.disk_orientation = mc.LineCollection(segments=segments, linewidths=4, color='k')
        self.ax.add_collection(self.disk_orientation)

    def update(self, i=0):
        T, n, d = self.qt.shape
        qt = self.qt.reshape(T, self.n_o, self.n_p, d)
        segments = []
        for j in range(self.n_o):
            self.circles[j].center = qt[i, j, 0, 0], qt[i, j, 0, 1]
            l = self.body.ls[j].detach().cpu().numpy()
            delta = (qt[i, j, 1] - qt[i, j, 0]) * l
            segments.append([qt[i, j, 0], qt[i, j, 0]+delta])

        self.disk_orientation.set_segments(segments)
        return super().update(i)


class ThrowAnimation(BouncingDisksAnimation):
    def __init__(self, qt, body, task, target, init_pos):
        super().__init__(qt, body)
        # initial position should be present for both tasks
        if isinstance(init_pos, torch.Tensor):
            init_pos = init_pos.detach().cpu().numpy()

        init_circle = Circle(
            [[init_pos[0]], [init_pos[1]]],
            radius=body.ls[0],
            color=self.colors[0],
            linestyle="-",
            linewidth=7,
            fill=False
        )
        self.ax.add_artist(init_circle)

        if task == "hit":
            if isinstance(target, torch.Tensor):
                target_pos = target.detach().cpu().numpy()
            else:
                target_pos = target
            target_circle = Circle(
                [[target_pos[0]], [target_pos[1]]], 
                radius=body.ls[0], 
                color="k", 
                linestyle="--",
                linewidth=6,
                fill=False,
            )
            self.ax.add_artist(target_circle)
        # this part is coded based on qt
        if task == "vertical_nospin":
            for t in range(qt.shape[0]):
                y = self.qt[t, 0, 1] 
                if y < 0.5 and y > self.qt[t-1, 0, 1]:
                    break
            self.ax.plot(
                self.qt[0:t, 0, 0], self.qt[0:t, 0, 1], 
                linestyle="--",
                linewidth=7,
                color=self.colors[0]
            )
            self.ax.plot(
                self.qt[t-1, 0, 0] * np.ones(2,), self.qt[t-1, 0, 1] + np.linspace(0, 0.25, 2), 
                linestyle="--",
                linewidth=7,
                color='k'
            )