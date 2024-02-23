# Datasets

In this subfolder you can find custom PyTorch dataset modules.

The datasets should extend `torch.utils.data.Dataset`, for example:

```python
class SHMUDataset(Dataset):
    """A dataset for working with SHMU radar files that are in h5 format."""

    def __init__(self, **kwargs):
        """Initialize Dataset."""
        super().__init__()
        ...

    def __len__(self):
        """Mandatory property for Dataset."""
        return self.len

    def __getitem__(self, idx):
        """Mandatory property for fetching data."""
        ...
        return inputs, outputs, idx
```

Each dataset should be imported in `__init__.py` in order to be available to use in other directories.

The code here may import

- `utils`


## Dataset implementations

### `SHMUDataset`

PyTorch Dataset for working with SHMU radar files that are in h5 format.

### `LagrangianSHMUDataset`

PyTorch Dataset for working with SHMU radar files that are in h5 format and transformed to Lagrangian coordinates.
