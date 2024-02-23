"""A datamodule for working with SHMU dataset for training, validation and testing that has files in h5 format and already transformed to Lagrangian coordinates by using `transform_shmu_to_lagrangian.py` script."""
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets import LagrangianSHMUDataset


class LagrangianSHMUDataModule(pl.LightningDataModule):
    def __init__(self, dsconfig, train_params, predict_list="predict"):
        super().__init__()
        self.dsconfig = dsconfig
        self.train_params = train_params
        self.predict_list = predict_list

    def prepare_data(self):
        # called only on 1 GPU
        pass

    def setup(self, stage):
        # called on every GPU
        if stage == "fit":
            self.train_dataset = LagrangianSHMUDataset(
                split="train", **self.dsconfig.SHMUDataset
            )
            self.valid_dataset = LagrangianSHMUDataset(
                split="valid", **self.dsconfig.SHMUDataset
            )
        if stage == "test":
            self.test_dataset = LagrangianSHMUDataset(
                split="test", **self.dsconfig.SHMUDataset
            )
        if stage == "predict":
            self.predict_dataset = LagrangianSHMUDataset(
                split=self.predict_list, predicting=True, **self.dsconfig.SHMUDataset
            )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_params.train_batch_size,
            num_workers=self.train_params.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=_collate_fn,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.train_params.valid_batch_size,
            num_workers=self.train_params.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=_collate_fn,
        )
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.train_params.test_batch_size,
            num_workers=self.train_params.num_workers,
            shuffle=False,
            collate_fn=_collate_fn,
        )
        return test_loader

    def predict_dataloader(self):
        predict_loader = DataLoader(
            self.predict_dataset,
            batch_size=self.train_params.predict_batch_size,
            num_workers=self.train_params.num_workers,
            shuffle=False,
            collate_fn=_collate_fn,
        )
        return predict_loader

def _collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

