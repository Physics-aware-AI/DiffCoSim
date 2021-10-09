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
import matplotlib.pyplot as plt
from models.contact_model import ContactModel
from models.contact_model_reg import ContactModelReg
from baselines.lcp.contact_model_lcp import ContactModelLCP
import matplotlib.animation as animation

class Billiards(RigidBody):
    dt = 0.003
    integration_time = 0.003 * 1024

    def __init__(
        self, 
        kwargs_file_name="default_args",
        billiard_layers=4, 
        ms=[1], 
        ls=[0.03], 
        mus=[0.0], 
        cors=[0.8],
        bdry_lin_coef=[[0, 0, 0]], # no boundary
        goal=[0.9, 0.75],
        is_homo=True,
        is_reg_data=False,
        is_reg_model=False,
        is_lcp_data=False,
        is_lcp_model=False,
        dtype=torch.float64
    ):
        self.goal = goal
        assert not (is_reg_model and is_lcp_model)
        self.body_graph = BodyGraph()
        self.kwargs_file_name = kwargs_file_name
        self.n_o = 1 + (1 + billiard_layers) * billiard_layers // 2
        self.ms = torch.tensor(ms*self.n_o, dtype=torch.float64)
        self.ls = torch.tensor(ls*self.n_o, dtype=torch.float64)
        for i in range(0, self.n_o):
            self.body_graph.add_extended_body(i, ms[0], d=0)
        self.n_p, self.d = 1, 2
        self.n = self.n_o * self.n_p
        self.billiard_layers = billiard_layers
        self.bdry_lin_coef = torch.tensor(bdry_lin_coef, dtype=torch.float64) 
        self.n_c = self.n_o * (self.n_o - 1) // 2
        if is_homo:
            assert len(mus) == len(cors) == 1
            self.mus = torch.tensor(mus*self.n_c, dtype=torch.float64)
            self.cors = torch.tensor(cors*self.n_c, dtype=torch.float64)
        else:
            assert len(mus) == len(cors) == self.n_c
            self.mus = torch.tensor(mus, dtype=torch.float64)
            self.cors = torch.tensor(cors, dtype=torch.float64)
        self.is_homo = is_homo
        self.is_reg_data = is_reg_data
        self.is_reg_model = is_reg_model   
        self.is_lcp_data = is_lcp_data
        self.is_lcp_model = is_lcp_model

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
                DPhi = self.DPhi
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
                DPhi = self.DPhi
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
                DPhi = self.DPhi
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
        # TODO: optimize this
        return 0 * (M @ x)[..., 1].sum(1)

    def sample_initial_conditions(self, N):
        print("Note: we only have one initial condition in billiards example...")
        assert N == 1
        return self.get_initial_conditions()

    def get_initial_conditions(self):
        n = len(self.body_graph.nodes)
        x0 = torch.zeros(1, n, 2)
        x0[0, 0, 0] = 0.1
        x0[0, 0, 1] = 0.5
        count = 0 
        for i in range(self.billiard_layers):
            for j in range(i+1):
                count += 1
                x0[0, count, 0] = i * 2 * self.ls[0] + 0.5
                x0[0, count, 1] = j * 2 * self.ls[0] + 0.5 - i * self.ls[0] * 0.7
        x_dot0 = torch.zeros(1, n, 2)
        x_dot0[0, 0, 0] = 0.3
        x_dot0[0, 0, 1] = 0.0
        return torch.stack([x0, x_dot0], dim=1) # (1, 2, n, 2)

    def check_collision(self, x):
        bs = x.shape[0]
        is_cld_0, is_cld_ij, dist_ij = self.check_ball_collision(x)
        is_cld_bdry = torch.zeros(bs, self.n, 0, dtype=torch.bool, device=x.device)
        dist_bdry = torch.zeros(bs, self.n, 0).type_as(x)
        is_cld_limit = torch.zeros(bs, 0, 2, dtype=torch.bool, device=x.device)
        dist_limit = torch.zeros(bs, 0, 2).type_as(x)
        return is_cld_0, is_cld_ij, is_cld_bdry, is_cld_limit, dist_ij, dist_bdry, dist_limit

    def check_ball_collision(self, x):
        # x: (bs, n, 2)
        bs, n, _ = x.shape
        is_collide = torch.zeros(bs, dtype=torch.bool, device=x.device)
        dist_ij = torch.zeros(bs, n, n, dtype=x.dtype, device=x.device)
        is_collide_ij = torch.zeros(bs, n, n, dtype=torch.bool, device=x.device)
        if n == 1:
            return is_collide, is_collide_ij, dist_ij
        ls = self.ls.to(x.device, x.dtype)
        for i in range(n-1):
            for j in range(i+1, n):
                dist_ij[:, i, j] = ((x[:, i] - x[:, j]) ** 2).sum(1).sqrt() - (ls[i] + ls[j])
                is_collide_ij[:, i, j] =  dist_ij[:, i, j] < 0
        is_collide = is_collide_ij.sum([1, 2]) > 0
        return is_collide, is_collide_ij, dist_ij

    def cld_2did_to_1did(self, cld_ij_ids, cld_bdry_ids, cld_limit_ids):
        n = self.n
        i, j = cld_ij_ids.unbind(dim=1)
        ij_1d_ids = j + n*i - (i+1) * (i+2) // 2
        return ij_1d_ids

    @property
    def animator(self):
        return BilliardsAnimation

class BilliardsAnimation(Animation):
    def __init__(self, qt, body):
        # at: T, n, d
        ###############  from super, we want to remove trail
        self.qt = qt.detach().cpu().numpy()
        T, n, d = qt.shape
        assert d in (2, 3)
        self.fig = plt.figure()
        self.ax = self.fig.add_axes([0, 0, 1, 1], projection='3d') if d==3 else self.fig.add_axes([0,0,1,1])
        if d!=3: self.ax.set_aspect("equal")

        empty = d * [[]]
        # self.colors = np.random.choice([f"C{i}" for i in range(15)], size=n, replace=False)
        self.colors = [f"C{i}" for i in range(15)]
        self.objects = {
            'pts': sum([self.ax.plot(*empty, ms=6, color=self.colors[i]) for i in range(n)], [])
        }
        ################
        self.body = body
        self.n_o = body.n_o
        self.n_p = body.n_p

        self.ax.set_xlim(0.0, 1.0)
        self.ax.set_ylim(0.0, 1.0)
        self.ax.axis("off"),
        # self.ax.set_facecolor('green')
        self.fig.set_size_inches(10.5, 10.5)
        self.fig.patch.set_facecolor('#3C733F')

        empty = self.qt.shape[-1] * [[]]
        # self.objects["pts"] = sum(
        #     [self.ax.plot(*empty, c=self.colors[i]) for i in range(0, qt.shape[1])], []
        # )
        self.circles = [Circle([[0], [0]], body.ls[0], color='#CCCCCC', lw=None)] + \
            [Circle([[0], [0]], body.ls[i], color='#F20530', lw=None) for i in range(1, qt.shape[1]-1)] + \
            [Circle([[0], [0]], body.ls[-1], color="#3344cc", lw=None)] + \
            [Circle([[self.body.goal[0]], [self.body.goal[1]]], body.ls[0]/2, color='#000000', lw=None)] # this is the goal

        [self.ax.add_artist(circle) for circle in self.circles]


    def update(self, i=0):
        T, n, d = self.qt.shape
        for j in range(n):
            self.circles[j].center = self.qt[i, j][0], self.qt[i, j][1]
        # qt = self.qt.reshape(T, self.n_o, self.n_p, d)
        # for j in range(self.n_o):
        #     # draw trails
        #     xyz = qt[i: i+1, j, 0, :]
        #     # draw points
        #     self.objects['pts'][j].set_data(*xyz[-1:,...,:2].T)
        #     if d==3: self.objects['pts'][j].set_3d_properties(xyz[-1:,...,2].T)
        # return sum(self.objects.values(), []) 
        return self.circles

    # def animate(self):
    #     return animation.FuncAnimation(self.fig, self.update, frames=self.qt.shape[0],
    #                 interval=64, init_func=self.init, blit=True,)#.save("test.gif")#.to_html5_video()


class BilliardsDummyAnimation(BilliardsAnimation):
    def __init__(self, qt, body, vx, vy):
        super().__init__(qt, body)
        # self.ax.set_xlim(-0.05, 1.0)
        # self.ax.set_ylim(0.35, 0.8)
        self.ax.set_xlim(-0.1, 1)
        self.ax.set_ylim(-0.05, 1.05)

        self.initial_circle = Circle(
            [[0.1], [0.5]], body.ls[0], color='#CCCCCC', fill=False, linestyle="--", linewidth=4
        )
        self.ax.add_artist(self.initial_circle)
        self.ax.arrow(0.1, 0.5, 0.3/3, 0.0/3, color='#CCCCCC',
            head_width=0.015, head_length=0.03, linestyle="--", linewidth=4, zorder=10
        )
        self.ax.arrow(qt[0, 0, 0], qt[0, 0, 1], vx/3, vy/3, color='#CCCCCC',
            head_width=0.015, head_length=0.03, linestyle="-", linewidth=4, zorder=10
        )
