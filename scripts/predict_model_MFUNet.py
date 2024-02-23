"""This script will run nowcasting prediction for the L-CNN model implementation
"""
import argparse
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from utils import load_config, setup_logging
from utils import LagrangianHDF5Writer
from models import MFUNET
from datamodules import SHMUDataModule


def run(checkpointpath, configpath, predict_list) -> None:

    confpath = Path("config") / configpath
    dsconf = load_config(confpath / "datasets.yaml")
    outputconf = load_config(confpath / "output.yaml")
    modelconf = load_config(confpath / "MFUNET.yaml")

    setup_logging(outputconf.logging)

    datamodel = SHMUDataModule(
        dsconf, modelconf.train_params, predict_list=predict_list
    )

    model = MFUNET(modelconf).load_from_checkpoint(checkpointpath, config=modelconf, map_location=torch.device('cpu'))

    output_writer = LagrangianHDF5Writer(**modelconf.prediction_output)

    logger = WandbLogger(save_dir=f"checkpoints/{modelconf.train_params.savefile}/wandb/predictions", project=modelconf.train_params.savefile, log_model=True)
    trainer = pl.Trainer(
        profiler="pytorch",
        logger=logger,
        devices=modelconf.train_params.gpus,
        callbacks=[output_writer],
    )

    # Predictions are written in HDF5 file
    trainer.predict(model, datamodel, return_predictions=False)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    argparser.add_argument(
        "config",
        type=str,
        help="Configuration folder path",
    )
    argparser.add_argument(
        "-l",
        "--list",
        type=str,
        default="predict",
        help="Name of predicted list (replaces {split} in dataset settings).",
    )

    args = argparser.parse_args()

    run(args.checkpoint, args.config, args.list)
