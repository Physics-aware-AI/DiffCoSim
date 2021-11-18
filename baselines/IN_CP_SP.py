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

from models.lagrangian import CLNNwC
import torch
import torch.nn as nn
from utils import mlp
from torchdiffeq import odeint
from .interaction_network import InteractionNetwork

class IN_CP_SP(CLNNwC):
    def __init__(
        self,
        body_graph,
        impulse_solver,
        n_c,
        d,
        is_homo=False,
        hidden_size: int = 256,
        num_layers: int = 3,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32,
        body=None,
        R_net_hidden_size=300,
        O_net_hidden_size=200,
        **kwargs
    ):
        super().__init__(
            body_graph,
            impulse_solver,
            n_c,
            d,
            is_homo,
            hidden_size,
            num_layers,
            device,
            dtype,
            **kwargs 
        )
        self.mu_params = None
        self.cor_params = None
        self.impulse_solver = None
        self.body = body
        self.velocity_impulse = InteractionNetwork(body, R_net_hidden_size, O_net_hidden_size)
        is_mass_point = "BP" in self.body.kwargs_file_name or \
                        "CP" in self.body.kwargs_file_name or \
                        "Rope" in self.body.kwargs_file_name 
        if is_mass_point:
            self.inv_moments = torch.div(1, self.body.ms)
        elif "BD" in self.body.kwargs_file_name:
            ms = self.body.ms
            ls = self.body.ls
            moments = ms * ls *ls / 4
            m_and_m = torch.stack([ms, moments, moments], dim=1)
            self.inv_moments = torch.div(1, m_and_m.reshape(-1))
        elif "Gyro" in self.body.kwargs_file_name:
            m = self.body.m    
            m_and_m = torch.cat(
                [torch.tensor(m).reshape(1).type_as(self.body.moments), m*self.body.moments],
                dim=0
            )
            self.inv_moments = torch.div(1, m_and_m)
        else:
            raise NotImplementedError

    def get_f_external(self, bs, n, d):
        # f_external (bs, n, d)
        if "BP" in self.body.kwargs_file_name:
            return torch.zeros(bs, n, d)
        elif "BD" in self.body.kwargs_file_name:
            return torch.zeros(bs, n, d)
        elif "CP" in self.body.kwargs_file_name:
            # inv_moments = torch.div(1, self.body.ms)
            g = self.body.g
            f_external = torch.stack(
                [torch.zeros(bs, n), -g * self.body.ms * torch.ones(bs, n)],
                dim=-1
            )
            return f_external
        elif "Rope" in self.body.kwargs_file_name:
            g = self.body.g
            # inv_moments = torch.div(1, self.body.ms)
            f_external = torch.stack(
                [torch.zeros(bs, n), -g * self.body.ms * torch.ones(bs, n)],
                dim=-1
            )
            return f_external
        elif "Gyro" in self.body.kwargs_file_name:
            f_external = torch.zeros(bs, 4, 3)
            f_external[:, 0, 2] = f_external[:, 0, 2] - self.body.m * 9.81
            return f_external
        else:
            raise NotImplementedError

    def integrate(self, z0, ts, tol=1e-4, method="rk4"):
        """
        input:
            z0: bs, 2, n, d
            ts: length T
        returns:
            a tensor of size bs, T, 2, n, d
        """
        assert (z0.ndim == 4) and (ts.ndim == 1)
        assert (z0.shape[-1] == self.d) and z0.shape[-2] == self.n
        # assert len(ts) == 2

        bs, _, n, d = z0.shape
        # true Minv
        f_external = self.get_f_external(bs, n, d)
        zt = z0
        zT = torch.zeros(bs, len(ts), 2, n, d).type_as(z0)
        zT[:, 0] = zt
        for i in range(len(ts)-1):
            vt_n = self.velocity_impulse(zt, self.inv_moments.type_as(z0), f_external) # bs, n, d
            xt = zt[:, 0] ; vt = zt[:, 1]
            xt_n = xt + vt * (ts[i+1] - ts[i])
            zt_n = torch.stack([xt_n, vt_n], dim=1)
            zt = zt_n
            zT[:, i+1] = zt
        return zT