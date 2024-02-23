# Utils

In this subfolder you may find utility functions that are of use when training neural network.

Subfolder contains following modules:
* `config.py`:
  * `load_config`: Loads configurations from yaml-file as `attrdict.AttrDict`.
* `logging.py`:
  * `setup_logging`: Sets up logging with `logging` module. Required parameters, given in dictionary, are `level` for logging level,  `format` for log message format, `dateformat` for date format in log message, and `filename` for output file name (if not given, logging outputs to `std`).
* `lagrangian_transform.py`:
  * `transform_to_eulerian`: Transforms the input time series to Eulerian coordinates using the given advection field.
  * `transform_to_lagrangian`: Transforms the input time series to Lagrangian coordinates using the given advection field.
  * `plot_lagrangian_fields`: Plots Lagrangian precipitation fields.
  * `read_advection_fields_from_h5`: Reads advection fields from HDF5 file.
  * `save_lagrangian_fields_h5_with_advfields`: Saves Lagrangian precipitation fields into HDF5 files.
* `prediction_writers.py`:
  * `LagrangianHDF5Writer`: Class BasePredictionWriters for use when creating predictions from Lagrangian coordinates to HDF5 file using PyTorch Lightning library.
