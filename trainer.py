# Standard library imports
from argparse import ArgumentParser, Namespace
import os, sys
import json
# THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(PARENT_DIR)

# Third party imports
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from torchdiffeq import odeint

# local application imports
from datasets.datasets import RigidBodyDataset
# from systems.chain_pendulum import ChainPendulum
# from systems.gyroscope import Gyroscope
from systems.bouncing_mass_points import BouncingMassPoints
# from models.hamiltonian import CHNN, HNN_Struct, HNN_Struct_Angle, HNN, HNN_Angle
from models.lagrangian import CLNNwC

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

        with open(os.path.join(".", "examples", hparams.body_kwargs_file+".json"), "r") as file:
            body_kwargs = json.load(file)

        body = str_to_class(hparams.body_class)(**body_kwargs)
        vars(hparams).update(dt=body.dt, integration_time=body.integration_time)

        # load/generate data
        train_dataset = str_to_class(hparams.dataset_class)(
            mode = "train",
            n_traj = hparams.n_train,
            body = body,
            dtype = self.dtype,
            chunk_len = hparams.chunk_len,
        )

        val_dataset = str_to_class(hparams.dataset_class)(
            mode = "val",
            n_traj = hparams.n_val,
            body = body,
            dtype = self.dtype,
            chunk_len = hparams.chunk_len,
        )

        test_dataset = str_to_class(hparams.dataset_class)(
            mode = "test",
            n_traj = hparams.n_test,
            body = body,
            dtype = self.dtype,
            chunk_len = hparams.chunk_len,
        )

        datasets = {"train": train_dataset, "val": val_dataset, "test": test_dataset}

        self.model = str_to_class(hparams.network_class)(body_graph=body.body_graph, 
                                                         impulse_solver=body.impulse_solver,
                                                         d = body.d,
                                                         n_c = body.n_c,
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

    def training_step(self, batch, batch_idx):
        (z0, ts), zts = batch
        ts = ts[0] - ts[0,0]
        # integrate 
        pred_zts = self.model.integrate(z0, ts, tol=self.hparams.tol, method=self.hparams.solver)
        # MAE error
        loss = self.traj_mae(pred_zts, zts)
        logs = {"train/loss": loss, "train/nfe": self.model.nfe}
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/nfe", self.model.nfe, prog_bar=True)
        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # dataset 
        parser.add_argument("--body-class", type=str, default="BouncingMassPoints")
        parser.add_argument("--body-kwargs-file", type=str, default="BM3_homo_cor1_mu0")
        parser.add_argument("--dataset-class", type=str, default="RigidBodyDataset")
        parser.add_argument("--n-train", type=int, default=800, help="number of train trajectories")
        parser.add_argument("--n-val", type=int, default=100, help="number of validation trajectories")
        parser.add_argument("--n-test", type=int, default=100, help="number of test trajectories")
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
        parser.add_argument("--network-class", type=str, help="dynamical model",
                            choices=[
                                "CLNNwC"
                            ], default="CLNNwC")
        parser.add_argument("--tol", type=float, default=1e-7)
        parser.add_argument("--solver", type=str, default="rk4")
    
        return parser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = Model.add_model_specific_args(parser)
    hparams = parser.parse_args()
    model = Model(hparams)

    savedir = os.path.join(".", "logs", 
                          hparams.body_kwargs_file + f"_N{hparams.n_train}")
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=savedir, name='')

    checkpoint = ModelCheckpoint(monitor="train/loss",
                                 save_top_k=1,
                                 save_last=True,
                                 dirpath=tb_logger.log_dir
                                 )

    trainer = Trainer.from_argparse_args(hparams,
                                         deterministic=True,
                                         callbacks=[checkpoint],
                                         logger=[tb_logger])

    trainer.fit(model)

    # with torch.no_grad():
    #     trainer.test()