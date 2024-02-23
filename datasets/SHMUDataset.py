"""PyTorch Dataset for working with SHMU radar files that are in h5 format."""
from pathlib import Path
import logging
import numpy as np
from skimage.measure import block_reduce
import torch
from torch.utils.data import Dataset
import h5py
import datetime


class SHMUDataset(Dataset):
    """A dataset for working with SHMU radar files that are in h5 format."""

    def __init__(
        self,
        split="train",
        date_list=None,
        path=None,
        filename=None,
        importer=None,
        input_block_length=None,
        prediction_block_length=None,
        timestep=None,
        bbox=None,
        image_size=None,
        bbox_image_size=None,
        input_image_size=None,
        upsampling_method=None,
        max_val=70.0,
        min_val=-16.948421478271484,
        transform_to_grayscale=True,
        predicting=False,
        normalization_method='none',
    ):
        """Initialize the dataset.

        Parameters
        ----------
        split : {'train', 'test', 'valid'}
            The type of the dataset: training, testing or validation.
        date_list : str
            Defines the name format of the date list file. The string is expected
            to contain the '{split}' keyword, where the value of the 'split'
            argument is substituted.
        path : str
            Format of the data path. May contain the tokens {year:*}, {month:*},
            {day:*}, {hour:*}, {minute:*}, {second:*} that are substituted when
            going through the dates.
        filename : str
            Format of the data file names. May contain the tokens {year:*},
            {month:*}, {day:*}, {hour:*}, {minute:*}, {second:*} that are
            substituted when going through the dates.
        importer : {'h5'}
            The importer to use for reading the files.
        input_block_length : int
            The number of frames to be used as input to the models.
        prediction_block_length : int
            The number of frames that are predicted and tested against the
            observations.
        timestep : int
            Time step of the data (minutes).
        bbox : str
            Bounding box of the data in the format '[x1, x2, y1, x2]'.
        image_size : str
            Shape of the original images without bounding box in the format
            '[width, height]'.
        image_size : str
            Shape of the images after the bounding box in the format
            '[width, height]'.
        input_image_size : str
            Shape of the input images supplied to the models in the format
            '[width, height]' after upsampling.
        upsampling_method : {'average'}
            The method to use for upsampling the input images.
        max_val : float
            The maximum value to use when scaling the data between 0 and 1.
        min_val : float
            The minimum value to use when scaling the data between 0 and 1.
        """
        assert date_list is not None, "No date list for radar files provided!"
        assert path is not None, "No path to radar files provided!"
        assert filename is not None, "No filename format for radar files provided!"
        assert importer is not None, "No importer for radar files provided!"

        # Inherit from parent class
        super().__init__()

        # Get data times
        with h5py.File(date_list.format(split=split), 'r') as hf:
            self.date_list = [item.decode() for item in hf['timestamps'][()]]
            self.date_list = np.array([datetime.datetime.strptime(item, '%Y-%m-%dT%H:%M:%S.%fZ') for item in self.date_list])

        self.path = path
        self.filename = filename

        # Get correct importer function
        if importer == "h5":
            self.importer = read_h5_composite
        else:
            raise NotImplementedError(f"Importer {importer} not implemented!")

        self.upsampling_method = upsampling_method

        self.max_val = max_val
        self.min_val = min_val

        self.image_size = image_size
        self.bbox_image_size = bbox_image_size
        self.input_image_size = input_image_size
        if bbox is None:
            self.use_bbox = False
        else:
            self.use_bbox = True
            self.bbox_x_slice = slice(bbox[0], bbox[1])
            self.bbox_y_slice = slice(bbox[2], bbox[3])

        self.num_frames_input = input_block_length
        self.num_frames_output = prediction_block_length
        self.num_frames = input_block_length + prediction_block_length

        self.transform_to_grayscale = transform_to_grayscale

        # Get windows
        self.timestep = timestep
        self.windows = np.lib.stride_tricks.sliding_window_view(self.date_list, self.num_frames)
        self.filtered_windows = []
        for self.window in self.windows:
            if (self.window[-1] - self.window[0]).seconds / (self.timestep * 60) == (self.num_frames - 1):
                self.filtered_windows.append(self.window)
        self.windows = np.array(self.filtered_windows)

        self.common_time_index = self.num_frames_input - 1

        self.transform_to_grayscale = transform_to_grayscale

        # If we're predicting now
        self.predicting = predicting

        if normalization_method not in ["log", "log_unit", "none", "log_unit_diff"]:
            raise NotImplementedError(
                f"data normalization method {normalization_method} not implemented"
            )
        else:
            self.normalization = normalization_method

    def __len__(self):
        """Mandatory property for Dataset."""
        return self.windows.shape[0]

    def __getitem__(self, idx):
        """Mandatory property for fetching data."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        window = self.windows[idx, ...]
        data = np.empty((self.num_frames, *self.input_image_size))

        # Check that window has correct length
        if (window[-1] - window[0]).seconds / (self.timestep * 60) != (
            self.num_frames - 1
        ):
            logging.info(f"Window {window[0]} - {window[-1]} wrong!")

        for i, date in enumerate(window):
            fn = Path(
                self.path.format(
                    year=date.year,
                    month=date.month,
                    day=date.day,
                    hour=date.hour,
                    minute=date.minute,
                    second=date.second,
                )
            ) / Path(
                self.filename.format(
                    year=date.year,
                    month=date.month,
                    day=date.day,
                    hour=date.hour,
                    minute=date.minute,
                    second=date.second,
                )
            )

            im = self.importer(fn)
            if self.use_bbox:
                im = im[self.bbox_x_slice, self.bbox_y_slice]

            if (
                im.shape[0] != self.input_image_size[0]
                or im.shape[1] != self.input_image_size[1]
            ):
                # Upsample image
                # Calculate window size
                block_x = int(im.shape[0] / self.input_image_size[0])
                block_y = int(im.shape[1] / self.input_image_size[1])
                if self.upsampling_method == "average":
                    # Upsample by averaging
                    im = block_reduce(
                        im, func=np.nanmean, cval=0, block_size=(block_x, block_y)
                    )
                else:
                    raise NotImplementedError(
                        f"Upsampling {self.upsampling_method} not yet implemented!"
                    )

            data[i, ...] = im

        data = data[..., np.newaxis]
        
        inputs, outputs = self.postprocessing(data)

        return inputs, outputs, idx

    def to_grayscale(self, data):
        """Transform image from dBZ to grayscale (the 0-1 range)."""
        return (data - self.min_val) / (self.max_val - self.min_val)

    def from_grayscale(self, data):
        """Transform from grayscale to dBZ (the 0-1 range)."""
        return data * (self.max_val - self.min_val) + self.min_val

    def from_transformed(self, data, scaled=True):
        if scaled:
            data = self.invScaler(data)  # to mm/h
        data = 200 * data ** (1.6)  # to z
        data = 10 * torch.log10(data + 1)  # to dBZ

        return data
    
    def postprocessing(self, data_in: np.ndarray):
        data = torch.Tensor(data_in)
        if self.transform_to_grayscale:
            # data of shape (window_size, im.shape[0], im.shape[1])
            # dbZ to mm/h
            data = 10 ** (data * 0.1)
            data = (data / 200) ** (1 / 1.6)  # fixed

        if self.transform_to_grayscale:
            # mm / h to log-transformed
            data = self.scaler(data)

        # Divide to input & output
        # Use output frame number, since that is constant whether we apply differencing or not
        if self.num_frames_output == 0:
            inputs = data
            outputs = torch.empty((0, data.shape[1], data.shape[2]))
        else:
            inputs = data[: -self.num_frames_output, ...].permute(0, 3, 1, 2).contiguous()
            outputs = data[-self.num_frames_output :, ...].permute(0, 3, 1, 2).contiguous()

        return inputs, outputs

    # def get_window(self, index):
    #     return self.windows[index, ...]
    
    def get_window(self, index):
        """Utility function to get window."""
        if isinstance(index, int):
            return self.windows[index]
        elif index.numel() == 1:
            return self.windows[index.item()]
        else:
            return self.windows[index]

    def get_common_time(self, index):
        window = self.get_window(index)
        return window[self.common_time_index]
    
    def scaler(self, data: torch.Tensor):
        if self.normalization == "log_unit_diff":
            data[data > self.log_unit_diff_cutoff] = self.log_unit_diff_cutoff
            data[data < -self.log_unit_diff_cutoff] = -self.log_unit_diff_cutoff
            return (data / self.log_unit_diff_cutoff + 1) / 2
        if self.normalization == "log_unit":
            return (torch.log(data + 0.01) + 5) / 10
        if self.normalization == "log":
            return torch.log(data + 0.01)
        if self.normalization == "none":
            return data

    def invScaler(self, data: torch.Tensor):
        if self.normalization == "log_unit_diff":
            return (data * 2 - 1) * self.log_unit_diff_cutoff
        if self.normalization == "log_unit":
            return torch.exp((data * 10) - 5) - 0.01
        if self.normalization == "log":
            return torch.exp(data) - 0.01
        if self.normalization == "none":
            return data


def read_h5_composite(filename):
    """"Read h5 composite."""
    with h5py.File(filename, "r") as hf:
        data = hf['precipitation_map'][()]
        data = data.astype(np.float64)
    return data
        
        