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

# Standard library imports
from argparse import ArgumentParser, Namespace
import os, sys
import json
from networkx.algorithms.planar_drawing import set_position
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(PARENT_DIR)

# Third party imports
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from torchdiffeq import odeint

# local application imports
# from datasets.datasets import RigidBodyDataset
# from systems.chain_pendulum import ChainPendulum
# from systems.gyroscope import Gyroscope
# from systems.bouncing_mass_points import BouncingMassPoints
# from systems.bouncing_disks import BouncingDisks
# from systems.chain_pendulum_with_contact import ChainPendulumWithContact
# from systems.rope_chain import RopeChain
# from systems.elastic_rope import ElasticRope
# from systems.gyroscope_with_wall import GyroscopeWithWall
# from models.hamiltonian import CHNN, HNN_Struct, HNN_Struct_Angle, HNN, HNN_Angle
from systems.billiards import Billiards
from models.lagrangian import CLNNwC
from models.hamiltonian import CHNNwC
from models.dynamics import ConstrainedLagrangianDynamics
from baselines.CLNN_MLP import CLNN_MLP
from baselines.CLNN_CD_MLP import CLNN_CD_MLP
from baselines.CLNN_IN import CLNN_IN
from baselines.IN import IN

from utils import dummy_dataloader
from trainer import Model as Dynamics_pl_model

seed_everything(0)

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

def collect_tensors(field, outputs):
    res = torch.stack([log[field] for log in outputs], dim=0)
    if res.ndim == 1:
        return res
    else:
        return res.flatten(0, 1)

class Model(pl.LightningModule):
    def __init__(self, hparams, **kwargs):
        super().__init__()
        hparams = Namespace(**hparams) if type(hparams) is dict else hparams
        vars(hparams).update(**kwargs)

        assert hparams.body_kwargs_file == ""
        body = str_to_class(hparams.body_class)()

        vars(hparams).update(
            dt=body.dt, 
            integration_time=body.integration_time,
            is_homo=body.is_homo,
            body=body
        )

        ##### target
        self.register_buffer("goal", torch.tensor(body.goal))
        # initial condition and time step
        self.register_buffer("ts", torch.arange(
            0, body.integration_time, body.dt
        ))
        self.initial_xy = nn.Parameter(torch.tensor([0.1, 0.5]))
        self.initial_vxvy = nn.Parameter(torch.tensor([0.3, 0.0]))
        ## we build initial velocity inside training step
        # get constant 
        self.register_buffer("Minv", body.Minv.to(torch.float32))
        self.register_buffer("mus", body.mus.to(torch.float32))
        self.register_buffer("cors", body.cors.to(torch.float32))
        self.potential = body.potential
        self.Minv_op = body.Minv_op
        ##############
        # self.Minv_op = lambda p: self.Minv.to(p.device, p.dtype) @ p

        self.dynamics = ConstrainedLagrangianDynamics(
            self.potential,
            self.Minv_op,
            body.DPhi,
            (body.n, body.d)
        )
        #############
        #############
        self.hparams = hparams
        self.body = body
        self.history = []
        self.history_loss = []

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.hparams.optimizer_class)(
            self.parameters(),
            lr = self.hparams.lr,
            weight_decay = self.hparams.weight_decay
        )
        if self.hparams.SGDR:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams.max_epochs)
            return [optimizer], [scheduler]
        else:
            return optimizer

    def train_dataloader(self):
        return dummy_dataloader()
    
    def val_dataloader(self) :
        return dummy_dataloader()
    
    def test_dataloader(self):
        return dummy_dataloader()

    def get_z0(self, x0, v0):
        """ x0: (3,)
            v0: (3,)
        """
        z0 = self.body.get_initial_conditions().to(device=self.device, dtype=self.dtype)
        z0[0, 0, 0, :] = x0 # the learnable position of the white ball
        z0[0, 1, 0, :] = v0 # the learnable position of the white ball
        return z0

    def simulate(self):
        # generate a trajectory based on the parametrized initial condition
        ################################
        ##### These are fixed 
        self.body.impulse_solver.to(device=self.device)
        # training 
        mus = self.mus
        cors = self.cors
        Minv = self.Minv 
        dynamics = self.dynamics
        z0 = self.get_z0(self.initial_xy, self.initial_vxvy)
        zt = z0.reshape(1, -1)
        zT = torch.zeros([1, len(self.ts), zt.shape[1]], device=zt.device, dtype=zt.dtype)
        zT[:, 0] = zt
        ##### integration
        for i in range(len(self.ts)-1):
            zt_n = odeint(dynamics, zt, self.ts[i:i+2], method=self.hparams.solver)[1]
            zt_n, _ = self.body.impulse_solver.add_impulse(zt_n, mus, cors, Minv)
            zt = zt_n
            zT[:, i+1] = zt
        ##### compute loss
        final_states = zT[:, -1].reshape(1, 2, -1, 2)
        loss = (final_states[0, 0, -1, 0] - self.goal[0]) ** 2 + (final_states[0, 0, -1, 1] - self.goal[1]) ** 2
        return zT, loss

    def training_step(self, batch, batch_idx):
        *_, loss = self.simulate()
        self.log('train/loss', loss, prog_bar=True)
        self.train_loss = loss.item()
        return loss
    
    def validation_step(self, batch, batch_idx):
        scaler_loss = getattr(self, 'train_loss', 0)
        self.log('val/loss', scaler_loss, prog_bar=True)
        self.log('x0', self.initial_xy[0], prog_bar=True)
        self.log('y0', self.initial_xy[1], prog_bar=True)
        self.log('vx0', self.initial_vxvy[0], prog_bar=True)
        self.log('vy0', self.initial_vxvy[1], prog_bar=True)
        self.history.append(torch.stack([self.initial_xy, self.initial_vxvy], dim=0).detach().cpu().numpy())
        self.history_loss.append(scaler_loss)
        return scaler_loss

    def test_step(self, batch, batch_idx):
        return self.simulate()

    def on_save_checkpoint(self, checkpoint):
        checkpoint['history'] = self.history
        checkpoint['history_loss'] = self.history_loss

    def on_load_checkpoint(self, checkpoint):
        self.history = checkpoint['history']
        self.history_loss = checkpoint['history_loss']

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # parser.add_argument("--task", type=str, default="hit")
        # parser.add_argument("--initial-position", type=float, nargs=3, default=[0.25, 0.65, 0.0])
        parser.add_argument("--target-xy", type=float, nargs=2, default=[0.9, 0.75])
        # parser.add_argument("--use-learned-properties", action="store_true", default=False)
        # parser.add_argument("--ckpt-path", type=str, default="")
        # dataset 
        parser.add_argument("--body-class", type=str, default="Billiards")
        parser.add_argument("--body-kwargs-file", type=str, default="")
        # optimizer
        parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
        parser.add_argument("--optimizer-class", type=str, default="SGD")
        parser.add_argument("--weight-decay", type=float, default=0.0)
        parser.add_argument("--SGDR", action="store_true")
        parser.add_argument("--no-SGDR", action="store_false", dest='SGDR')
        parser.set_defaults(SGDR=False)
        # model
        parser.add_argument("--solver", type=str, default="euler")
    
        return parser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = Model.add_model_specific_args(parser)
    hparams = parser.parse_args()
    model = Model(hparams)

    savedir = os.path.join(".", "logs", "billiards")
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=savedir, name='')

    checkpoint = ModelCheckpoint(monitor="val/loss",
                                 save_top_k=1,
                                 save_last=True,
                                 dirpath=tb_logger.log_dir
                                 )

    trainer = Trainer.from_argparse_args(
        hparams,
        deterministic=True,
        terminate_on_nan=True,
        callbacks=[checkpoint],
        logger=[tb_logger],
        max_epochs=200
    )

    trainer.fit(model)
