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

class IN_CP_CLNN(CLNNwC):
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
        R_net_hidden_size=150,
        O_net_hidden_size=100,
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

        bs= z0.shape[0]
        with torch.enable_grad():
            z0 = torch.zeros_like(z0, requires_grad=True) + z0
            zt = z0.reshape(bs, -1)
            zT = torch.zeros([bs, len(ts), zt.shape[1]]).type_as(z0)
            zT[:, 0] = zt

            # get mass
            # inv_moments = self.get_inv_moments()
            d = int(list(self.m_params.keys())[0])
            inv_moments = torch.exp(-self.m_params[str(d)]).reshape(-1) # n_o*n_p
            for i in range(len(ts)-1):
                zt_n = odeint(self, zt, ts[i:i+2], rtol=tol, method=method)[1]
                zt_n = zt_n.reshape(bs, 2, self.n, self.d)
                xt_n = zt_n[:, 0] ; vt_n = zt_n[:, 1]

                # add walls
                # get potential velocity
                V = self.potential(xt_n)
                dV = torch.autograd.grad(V.sum(), xt_n, create_graph=True)[0] # bs, n, d

                delta_v = self.velocity_impulse(zt_n, inv_moments, dV)
                vt_n = vt_n + delta_v.reshape(bs, self.n, self.d)
                zt_n = torch.stack([xt_n, vt_n], dim=1).reshape(bs, -1)
                zt = zt_n
                zT[:, i+1] = zt
        return zT.reshape(bs, len(ts), 2, self.n, self.d)

    # def get_inv_moments(self):
    #     if "BP" in self.body.kwargs_file_name:
    #         return inv_moments
    #     elif "BD" in self.body.kwargs_file_name:
    #         d = int(list(self.m_params.keys())[0])
    #         inv_m = torch.exp(-self.m_params[str(d)]).reshape(-1) # n_o*n_p

    #         return inv_moments
