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

"""
This module performed bad and is not reported in the final paper. 
"""

from models.lagrangian import CLNNwC
import torch
import torch.nn as nn
from utils import mlp
from torchdiffeq import odeint

class MLP_CLNN(CLNNwC):
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
        sizes = [2*self.n*self.d] + num_layers * [hidden_size] + [self.n*self.d]
        self.velocity_impulse = mlp(sizes, nn.ReLU, orthogonal_init=True)


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
        assert len(ts) == 2

        bs = z0.shape[0]

        z1 = odeint(self, z0.reshape(bs, -1), ts, rtol=tol, method="rk4")[1]
        delta_v = self.velocity_impulse(z1)
        z1 = z1.reshape(bs, 2, self.n, self.d)
        x1 = z1[:, 0] ; v1 = z1[:, 1]
        v1 = v1 + delta_v.reshape(bs, self.n, self.d)
        z1 = torch.stack([x1, v1], dim=1)
        return torch.stack([z0, z1], dim=1)