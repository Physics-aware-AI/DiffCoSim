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
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
from operator import add
from functools import reduce

# Third party imports
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import _logger as log
from torchdiffeq import odeint

# local application imports
from datasets.datasets import RigidBodyDataset
from systems.bouncing_point_masses import BouncingPointMasses
from systems.bouncing_disks import BouncingDisks
from systems.chain_pendulum_with_contact import ChainPendulumWithContact
from systems.rope import Rope
from systems.gyroscope_with_wall import GyroscopeWithWall
from models.lagrangian import CLNNwC
from models.hamiltonian import CHNNwC
from baselines.MLP_CD_CLNN import MLP_CD_CLNN
from baselines.IN_CP_CLNN import IN_CP_CLNN
from baselines.IN_CP_SP import IN_CP_SP

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

        if hparams.body_kwargs_file == "":
            body = str_to_class(hparams.body_class)()
        else:
            with open(os.path.join(THIS_DIR, "examples", hparams.body_kwargs_file+".json"), "r") as file:
                body_kwargs = json.load(file)
            body = str_to_class(hparams.body_class)(hparams.body_kwargs_file, 
                                                    is_reg_data=hparams.is_reg_data,
                                                    is_reg_model=hparams.is_reg_model, 
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

        # load/generate data
        train_dataset = str_to_class(hparams.dataset_class)(
            mode = "train",
            n_traj = hparams.n_train,
            body = body,
            dtype = self.dtype,
            chunk_len = hparams.chunk_len,
            noise_std = hparams.noise_std,
        )

        val_dataset = str_to_class(hparams.dataset_class)(
            mode = "val",
            n_traj = hparams.n_val,
            body = body,
            dtype = self.dtype,
            chunk_len = hparams.chunk_len,
            noise_std = hparams.noise_std,
        )

        test_dataset = str_to_class(hparams.dataset_class)(
            mode = "test",
            n_traj = hparams.n_test,
            body = body,
            dtype = self.dtype,
            chunk_len = hparams.chunk_len,
            noise_std = hparams.noise_std,
        )

        datasets = {"train": train_dataset, "val": val_dataset, "test": test_dataset}

        self.model = str_to_class(hparams.network_class)(body_graph=body.body_graph, 
                                                         impulse_solver=body.impulse_solver,
                                                         d=body.d,
                                                         n_c=body.n_c,
                                                         device=self.device,
                                                         dtype=self.dtype,
                                                         **vars(hparams))
        self.hparams = hparams
        self.body = body
        self.datasets = datasets

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
        return DataLoader(self.datasets["train"], 
                    batch_size=self.hparams.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.datasets["val"], 
                    batch_size=self.hparams.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.datasets["test"], 
                    batch_size=self.hparams.batch_size, shuffle=False)

    def traj_mae(self, pred_zts, true_zts):
        return (pred_zts - true_zts).abs().mean()

    def one_batch(self, batch, batch_idx):
        # z0: (bs, 2, n, d), zts: (bs, T, 2, n, d), ts: (bs, T)
        (z0, ts), zts, is_clds = batch

        if self.hparams.network_class != "CLNNwC" and self.hparams.network_class != "CHNNwC":
            # reshape data to predict only one step forward for ablation study
            bs, T, _, n, d = zts.shape
            z0 = zts[:, :-1].reshape(bs*(T-1), 2, n, d)
            zts = torch.stack([z0, zts[:, 1:].reshape(bs*(T-1), 2, n, d)], dim=1)
            ts = ts[:, 0:2]
            assert zts.shape == (bs*(T-1), 2, 2, n, d) and ts.shape == (bs, 2)

        ts = ts[0] - ts[0,0]

        pred_zts = self.model.integrate(z0, ts, tol=self.hparams.tol, method=self.hparams.solver)
        loss = self.traj_mae(pred_zts, zts)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.one_batch(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss = self.one_batch(batch, batch_idx).item()
        return loss

    def validation_epoch_end(self, outputs):
        val_loss = reduce(add, outputs) / len(outputs)
        self.log("val/loss", val_loss, prog_bar=True)
        if self.body.is_homo and self.hparams.network_class in ['CLNNwC', 'CHNNwC']:
            mu = F.relu(self.model.mu_params)
            cor = F.hardsigmoid(self.model.cor_params)
            self.log("mu", mu, prog_bar=True)
            self.log("cor", cor, prog_bar=True)

    def on_after_backward(self):
        # only skip samples that cause nan if we don't explicitly want to terminate when nan appears
        if not self.hparams.terminate_on_nan:
            # skip samples that cause inf/nan gradients/loss
            # https://github.com/PyTorchLightning/pytorch-lightning/issues/4956
            valid_gradients = True
            for name, param in self.named_parameters():
                if param.grad is not None:
                    valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                    if not valid_gradients:
                        break

            if not valid_gradients:
                log.warning(f'detected inf or nan values in gradients. not updating model parameters')
                self.zero_grad()

    def set_requires_grad(self, train_mu_cor, train_m_V):
        self.model.mu_params.requires_grad = train_mu_cor
        self.model.cor_params.requires_grad = train_mu_cor
        for param in self.model.V_net.parameters():
            param.requires_grad = train_m_V
        for param in self.model.m_params.parameters():
            param.requires_grad = train_m_V

    def test_step(self, batch, batch_idx, integration_time=None):
        (z0, _), zT, _ = batch
        if integration_time is None:
            integration_time = max(self.body.integration_time, self.body.dt*100)
        ts = torch.arange(0.0, integration_time, self.body.dt).type_as(z0)
        pred_zts = self.model.integrate(z0, ts, method='rk4')
        true_zts, _ = self.body.integrate(z0, ts, method='rk4') # (bs, T, 2, n, d)

        sq_diff = (pred_zts - true_zts).pow(2).sum((2,3,4))
        sq_true = true_zts.pow(2).sum((2,3,4))
        sq_pred = pred_zts.pow(2).sum((2,3,4))
        # (bs, T)
        rel_err = sq_diff.div(sq_true).sqrt()
        bounded_rel_err = sq_diff.div(sq_true+sq_pred).sqrt()
        abs_err = sq_diff.sqrt()

        loss = self.traj_mae(pred_zts, true_zts)
        pred_zts_true_energy = self.true_energy(pred_zts)
        true_zts_true_energy = self.true_energy(true_zts)

        return {
            "traj_mae": loss.detach(),
            "true_zts": true_zts.detach(),
            "pred_zts": pred_zts.detach(),
            "abs_err": abs_err.detach(),
            "rel_err": rel_err.detach(),
            "bounded_rel_err": bounded_rel_err.detach(),
            "true_zts_true_energy": true_zts_true_energy.detach(),
            "pred_zts_true_energy": pred_zts_true_energy.detach(),
        }

    def test_epoch_end(self, outputs):
        log, save = self._collect_test_steps(outputs)
        self.log("test_loss", log["traj_mae"])
        for k, v in log.items():
            self.log(f"test/{k}", v)

    def _collect_test_steps(sef, outputs):
        loss = collect_tensors("traj_mae", outputs).mean(0).item()
        # collect batch errors from minibatches (BS, T)
        abs_err = collect_tensors("abs_err", outputs)
        rel_err = collect_tensors("rel_err", outputs)
        bounded_rel_err = collect_tensors("bounded_rel_err", outputs)

        pred_zts_true_energy = collect_tensors("pred_zts_true_energy", outputs) # (BS, T)
        true_zts_true_energy = collect_tensors("true_zts_true_energy", outputs)

        true_zts = collect_tensors("true_zts", outputs)
        pred_zts = collect_tensors("pred_zts", outputs)

        log = {
            "traj_mae" : loss,
            "mean_abs_err": abs_err.sum(1).mean(0),
            "mean_rel_err": rel_err.sum(1).mean(0),
            "mean_bounded_rel_err": bounded_rel_err.sum(1).mean(0),
            "mean_true_zts_true_energy": true_zts_true_energy.sum(1).mean(0),
            "mean_pred_zts_true_energy": pred_zts_true_energy.sum(1).mean(0),
        }
        save = {"true_zts": true_zts, "pred_zts": pred_zts}
        return log, save

    def true_energy(self, zts):
        N, T = zts.shape[:2]
        x, v = zts.chunk(2, dim=2)
        p_x = self.body.M.type_as(v) @ v
        zts = torch.cat([x, p_x], dim=2)
        energy = self.body.hamiltonian(None, zts.reshape(N*T, -1))
        return energy.reshape(N, T)


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # dataset 
        parser.add_argument("--body-class", type=str, default="BouncingPointMasses")
        parser.add_argument("--body-kwargs-file", type=str, default="default")
        parser.add_argument("--dataset-class", type=str, default="RigidBodyDataset")
        parser.add_argument("--n-train", type=int, default=800, help="number of train trajectories")
        parser.add_argument("--n-val", type=int, default=100, help="number of validation trajectories")
        parser.add_argument("--n-test", type=int, default=100, help="number of test trajectories")
        parser.add_argument("--is-reg-data", action="store_true", default=False)
        parser.add_argument("--is-lcp-data", action="store_true", default=False)
        parser.add_argument("--noise-std", type=float, default=0.0)
        # optimizer
        parser.add_argument("--chunk-len", type=int, default=5)
        parser.add_argument("--batch-size", type=int, default=200)
        parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
        parser.add_argument("--optimizer-class", type=str, default="AdamW")
        parser.add_argument("--weight-decay", type=float, default=1e-4)
        parser.add_argument("--SGDR", action="store_true", default=False)
        # model
        parser.add_argument("--hidden-size", type=int, default=256, help="number of hidden units")
        parser.add_argument("--num-layers", type=int, default=3, help="number of hidden layers")
        parser.add_argument("--tol", type=float, default=1e-7)
        parser.add_argument("--solver", type=str, default="rk4")
        parser.add_argument("--network-class", type=str, help="dynamical model",
                            choices=[
                                "CLNNwC", "CHNNwC", "MLP_CD_CLNN", "IN_CP_SP", "IN_CP_CLNN"
                            ], default="CLNNwC")
        parser.add_argument("--is-lcp-model", action="store_true", default=False)
        parser.add_argument("--is-reg-model", action="store_true", default=False)
        # set a negative reg to enable a learnable regularizer, as suggested by a reviewer
        parser.add_argument("--reg", type=float, default=0.01)
    
        return parser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = Model.add_model_specific_args(parser)
    hparams = parser.parse_args()
    model = Model(hparams)

    is_reg_model = "_reg" if hparams.is_reg_model else ""
    is_lcp_model = "_lcp" if hparams.is_lcp_model else ""
    noise_std_str = "" if hparams.noise_std < 0.0000001 else f"_{hparams.noise_std}"
    savedir = os.path.join(".", "logs", 
                          hparams.body_kwargs_file + f"_{hparams.network_class}" 
                          + is_reg_model + is_lcp_model + f"_N{hparams.n_train}" + noise_std_str)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=savedir, name='')

    checkpoint = ModelCheckpoint(monitor="val/loss",
                                 save_top_k=1,
                                 save_last=True,
                                 dirpath=tb_logger.log_dir
                                 )

    trainer = Trainer.from_argparse_args(hparams,
                                         deterministic=True,
                                         callbacks=[checkpoint],
                                         logger=[tb_logger],
                                        )

    trainer.fit(model)