# Model Components

This subfolder contains neural network model components implemented using PyTorch.

The components should extend `torch.nn.Module`. For example:

```python
class NNComponent(nn.Module):
    """Model that implements some NN component"""

    def __init__(self, **kwargs):
        """Initialize the class instance."""
        super().__init__()
        ...
```

Each module should be imported in `__init__.py` in order to be available to use in other directories.

The code here may import

- `utils`

## Implementations

### `RainNet`

A module implementing a RainNet convolutional neural network architecture for precipitation nowcasting. RainNet architecture was originally published in article [_RainNet: a convolutional neural network for radar-based precipitation nowcasting_](https://publishup.uni-potsdam.de/opus4-ubp/frontdoor/deliver/index/docId/47294/file/pmnr964.pdf).

