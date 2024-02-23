# DataModules

This subfolder contains custom PyTorch-Lightning LightningDataModules.

The datasets should extend `pytorch_lighning.LightningDataModule`, for example:

```python
class CustomDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()

    def __init__(self, **kwargs):
        """Initialize Dataset."""
        super().__init__()
        ...

    def prepare_data(self):
        # called only on 1 GPU
        pass

    def setup(self, stage):
        # called on every GPU
        self.train_dataset = CustomDataset(**dataset_kwargs)
        self.valid_dataset = CustomDataset(**dataset_kwargs)
        self.test_dataset = CustomDataset(**dataset_kwargs)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            **params,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            self.valid_dataset,
            **params,
        )
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_dataset,
            **params,
        )
        return test_loader
```

Each datamodule should be imported in `__init__.py` in order to be available to use in other directories.

The code here may import

- `datasets`
- `utils`

## Implemented DataModules

### `LagrangianSHMUDataModule`
A datamodule for working with SHMU dataset for training, validation and testing that has files in h5 format and already transformed to Lagrangian coordinates by using `transform_shmu_to_lagrangian.py` script.

### `SHMUDataModule`
A datamodule for working with SHMU dataset for training, validation and testing that has files in h5 format and not transformed to Lagrangian coordinates.
