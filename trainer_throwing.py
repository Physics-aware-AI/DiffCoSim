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
from systems.bouncing_disks import BouncingDisks
# from systems.chain_pendulum_with_contact import ChainPendulumWithContact
# from systems.rope_chain import RopeChain
# from systems.elastic_rope import ElasticRope
# from systems.gyroscope_with_wall import GyroscopeWithWall
# from models.hamiltonian import CHNN, HNN_Struct, HNN_Struct_Angle, HNN, HNN_Angle
from models.lagrangian import CLNNwC
from models.hamiltonian import CHNNwC
from models.dynamics import ConstrainedLagrangianDynamics
from baselines.CLNN_MLP import CLNN_MLP
from baselines.CLNN_CD_MLP import CLNN_CD_MLP
from baselines.CLNN_IN import CLNN_IN
from baselines.IN import IN
from find_bad_grad import BadGradFinder

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
        if not hasattr(hparams, "is_mujoco_like"):
            hparams.is_mujoco_like = False
        if not hasattr(hparams, "is_base_full"):
            hparams.is_base_full = False
        if not hasattr(hparams, "noise_std"):
            hparams.noise_std = 0.0
        if not hasattr(hparams, "reg"):
            hparams.reg = 0.01
        if not hasattr(hparams, "is_lcp_data"):
            hparams.is_lcp_data = False
        if not hasattr(hparams, "is_lcp_model"):
            hparams.is_lcp_model = False

        if hparams.body_kwargs_file == "":
            body = str_to_class(hparams.body_class)()
        else:
            with open(os.path.join(THIS_DIR, "examples", hparams.body_kwargs_file+".json"), "r") as file:
                body_kwargs = json.load(file)
            body = str_to_class(hparams.body_class)(hparams.body_kwargs_file, 
                                                    is_mujoco_like=hparams.is_mujoco_like, 
                                                    is_lcp_data=hparams.is_lcp_data,
                                                    is_lcp_model=hparams.is_lcp_model,
                                                    **body_kwargs)
            vars(hparams).update(**body_kwargs)
        vars(hparams).update(
            dt=body.dt, 
            integration_time=body.integration_time,
            is_homo=body.is_homo,
            body=body
        )

        # self.model = str_to_class(hparams.network_class)(body_graph=body.body_graph, 
        #                                                  impulse_solver=body.impulse_solver,
        #                                                  d=body.d,
        #                                                  n_c=body.n_c,
        #                                                  device=self.device,
        #                                                  dtype=self.dtype,
        #                                                  **vars(hparams))
        ##### target
        self.register_buffer("target_xy", torch.tensor(hparams.target_xy))
        # initial condition and time step
        self.register_buffer("one_time_step", torch.tensor([0, body.dt]))
        self.register_buffer("initial_position", torch.tensor(hparams.initial_position))
        if hparams.task == "hit":
            self.initial_velocity = nn.Parameter(torch.zeros(3))
        else:
            self.register_buffer("initial_vxvy", torch.tensor(hparams.initial_vxvy))
            self.initial_w = nn.Parameter(torch.zeros(1))
            ## we build initial velocity inside training step
        # get constant 
        self.register_buffer("Minv", body.Minv.to(torch.float32))
        self.register_buffer("mus", body.mus.to(torch.float32))
        self.register_buffer("cors", body.cors.to(torch.float32))
        self.potential = body.potential
        self.Minv_op = body.Minv_op
        ##############
        self.dynamics = ConstrainedLagrangianDynamics(
            self.potential,
            self.Minv_op,
            body.DPhi,
            (body.n, body.d)
        )
        #############
        #############
        if hparams.use_learned_properties:
            if not hparams.ckpt_path:
                raise ValueError("must provide a path to the checkpoint when setting --use-learned-properties to be true")
            dynamics_pl_model = Dynamics_pl_model.load_from_checkpoint(hparams.ckpt_path)
            assert isinstance(dynamics_pl_model.model, CLNNwC)
            dynamics_pl_model.freeze() # very important to freeze the model so that the parameters are fixed
            self.register_buffer("learned_mus", F.relu(dynamics_pl_model.model.mu_params * torch.ones(4)))
            self.register_buffer("learned_cors", F.hardsigmoid(dynamics_pl_model.model.cor_params * torch.ones(4)))
            self.register_buffer("learned_Minv", dynamics_pl_model.model.Minv)
            self.learned_potential = dynamics_pl_model.model.potential # need to make sure the model is freezed
            self.learned_Minv_op = dynamics_pl_model.model.Minv_op # need to make sure the model is freezed
            self.learned_dynamics = ConstrainedLagrangianDynamics(
                self.learned_potential,
                self.learned_Minv_op,
                body.DPhi,
                (body.n, body.d)
            )
            self.dynamics_pl_mnodel = dynamics_pl_model # add a pointer to check require_grad info
        self.hparams = hparams
        self.body = body
        self.history = []

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

    def mae(self, pred_zts, true_zts):
        return (pred_zts - true_zts).abs().mean()

    def get_z0(self, x0, v0):
        """ x0: (3,)
            v0: (3,)
        """
        xv0 = torch.stack([x0, v0], dim=0)[None, :, None, :]
        return self.body.generalized_to_cartesian(xv0)

    def simulate(self):
        # generate a trajectory based on the parametrized initial condition
        ################################
        ##### These are fixed 
        # training 
        if self.training and self.hparams.use_learned_properties:
            mus = self.learned_mus
            cors = self.learned_cors
            Minv = self.learned_Minv
            dynamics = self.learned_dynamics
        else:
            mus = self.mus
            cors = self.cors
            Minv = self.Minv 
            dynamics = self.dynamics
        ##### Initial conditions are learnable
        if self.hparams.task == "vertical":
            self.initial_velocity = torch.cat([self.initial_vxvy, self.initial_w])
        z0 = self.get_z0(self.initial_position, self.initial_velocity)
        ##### integration
        zt = z0.reshape(1, -1)
        zT = [zt]
        vy = [zt.reshape(1, 2, 1, 3, 2)[0, 1, 0, 0, 1]] # bs, 2, n_o, n_p, d
        # status -1: if vy[0] is positive
        #        0: going down
        #        1: from hit the groud to the top set_position
        #        2: from the top position to the second time that hit the ground
        status = [0] if self.initial_velocity[1] < 0 else [-1]
        while not self.is_terminate(status):
            zt_n = odeint(dynamics, zt, self.one_time_step, method=self.hparams.solver)[1]
            zt_n, _ = self.body.impulse_solver.add_impulse(zt_n, mus, cors, Minv)
            zt = zt_n
            zT.append(zt)
            vy.append(zt.reshape(1, 2, 1, 3, 2)[0, 1, 0, 0, 1]) 
            status.append(status[-1]+1 if vy[-1]*vy[-2] < 0 or vy[-1] == 0 else status[-1])
        # calculate the loss model.hparams.task
        if self.hparams.task == "vertical" or self.hparams.task == "vertical_nospin":
            # get those in zT such that status == 1
            for i in range(len(status)):
                if status[i] == 1:
                    idx = i-1
                    break
            # ground_xy = zT[idx].reshape(1, 2, 1, 3, 2)[0, 0, 0, 0]
            final_x = zT[-2].reshape(1, 2, 1, 3, 2)[0, 0, 0, 0, 0]
            vx_after_contact = zT[idx+2].reshape(1, 2, 1, 3, 2)[0, 1, 0, 0, 0]
            x_after_contact = zT[idx+2].reshape(1, 2, 1, 3, 2)[0, 0, 0, 0, 0]

            loss = self.mae(vx_after_contact, torch.zeros_like(vx_after_contact)) + self.mae(x_after_contact, final_x)
            # if self.hparams.task == "vertical_nospin":
            #     # penalize rotation
            #     vs = zT[idx].reshape(1, 2, 1, 3, 2)[0, 1, 0, :]
            #     loss = loss + self.mae(vs[1], vs[0]) + self.mae(vs[2], vs[0])
        elif self.hparams.task == "hit":
            # get the last point where status == 2
            assert status[-2] == 2
            pred_xy = zT[-2].reshape(1, 2, 1, 3, 2)[0, 0, 0, 0]
            loss = self.mae(pred_xy, self.target_xy)
        else:
            raise NotImplementedError
        return zT, status, loss

    def training_step(self, batch, batch_idx):
        *_, loss = self.simulate()
        self.log('train/loss', loss, prog_bar=True)
        self.train_loss = loss.item()
        return loss
    
    def validation_step(self, batch, batch_idx):
        if self.hparams.use_learned_properties:
            *_, loss = self.simulate()
            scaler_loss = loss.item()
        else: # here validation would be the same as training
            scaler_loss = getattr(self, 'train_loss', 0)
        if hasattr(self, "initial_velocity"): # make pt-lightning happy
            self.log('val/loss', scaler_loss, prog_bar=True)
            self.log('vx0', self.initial_velocity[0], prog_bar=True)
            self.log('vy0', self.initial_velocity[1], prog_bar=True)
            self.log('w0', self.initial_velocity[2], prog_bar=True)
            self.history.append(self.initial_velocity.clone().detach().cpu().numpy())
        return scaler_loss

    def test_step(self, batch, batch_idx):
        return self.simulate()

    def is_terminate(self, status):
        if self.hparams.task == "vertical" or self.hparams.task == "vertical_nospin":
            return True if status[-1] == 2 else False
        elif self.hparams.task == "hit":
            return True if status[-1] == 3 else False
        else:
            raise NotImplementedError

    def on_save_checkpoint(self, checkpoint):
        checkpoint['history'] = self.history

    def on_load_checkpoint(self, checkpoint):
        self.history = checkpoint['history']

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--task", type=str, default="hit")
        parser.add_argument("--initial-position", type=float, nargs=3, default=[0.25, 0.65, 0.0])
        parser.add_argument("--target-xy", type=float, nargs=2, default=[0.75, 0.1])
        parser.add_argument("--initial-vxvy", type=float, nargs=2, default=[0.6, 0.4])
        parser.add_argument("--use-learned-properties", action="store_true", default=False)
        parser.add_argument("--ckpt-path", type=str, default="")
        # dataset 
        parser.add_argument("--body-class", type=str, default="BouncingDisks")
        parser.add_argument("--body-kwargs-file", type=str, default="BD1_homo_cor0.8_mu0.2")
        parser.add_argument("--dataset-class", type=str, default="RigidBodyDataset")
        parser.add_argument("--is-mujoco-like", action="store_true", default=False)
        parser.add_argument("--is-lcp-data", action="store_true", default=False)
        # optimizer
        parser.add_argument("--lr", type=float, default=1e-1, help="learning rate")
        parser.add_argument("--optimizer-class", type=str, default="AdamW")
        parser.add_argument("--weight-decay", type=float, default=1e-4)
        parser.add_argument("--SGDR", action="store_true")
        parser.add_argument("--no-SGDR", action="store_false", dest='SGDR')
        parser.set_defaults(SGDR=True)
        # model
        parser.add_argument("--solver", type=str, default="rk4")
        # parser.add_argument("--learned-model-path", type=str, default="")
    
        return parser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = Model.add_model_specific_args(parser)
    hparams = parser.parse_args()
    model = Model(hparams)

    is_mujoco = "_mujoco" if hparams.is_mujoco_like else ""
    is_lcp_model = "_lcp" if hparams.is_lcp_model else ""
    is_learned_model = "_learned" if hparams.ckpt_path else ""
    savedir = os.path.join(".", "logs", 
                          hparams.task + "_" + hparams.body_kwargs_file + is_mujoco + is_lcp_model)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=savedir, name='')

    checkpoint = ModelCheckpoint(monitor="val/loss",
                                 save_top_k=1,
                                 save_last=True,
                                 dirpath=tb_logger.log_dir
                                 )

    trainer = Trainer.from_argparse_args(hparams,
                                         deterministic=True,
                                         terminate_on_nan=True,
                                         callbacks=[checkpoint],
                                         logger=[tb_logger])

    trainer.fit(model)

    # with torch.no_grad():
    #     trainer.test()
