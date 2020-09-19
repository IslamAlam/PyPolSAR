import contextlib
import itertools
import math
import os.path
import pickle
import shutil
import sys
import tempfile
import warnings
from contextlib import ExitStack
from io import BytesIO
from pathlib import Path
from typing import Optional

import pytest

import numpy as np
import pandas as pd
from pandas.errors import OutOfBoundsDatetime

try:
    import netCDF4 as nc4
except ImportError:
    pass

try:
    import dask
    import dask.array as da

    dask_version = dask.__version__
except ImportError:
    # needed for xfailed tests when dask < 2.4.0
    # remove when min dask > 2.4.0
    dask_version = "10.0"

ON_WINDOWS = sys.platform == "win32"
default_value = object()


class DatasetIOBase:
    engine: Optional[str] = None
    file_format: Optional[str] = None

    def create_store(self):
        raise NotImplementedError()

    def test_roundtrip_example_1_netcdf(self):
        with open_example_dataset("example_1.nc") as expected:
            with self.roundtrip(expected) as actual:
                # we allow the attributes to differ since that
                # will depend on the encoding used.  For example,
                # without CF encoding 'actual' will end up with
                # a dtype attribute.
                assert_equal(expected, actual)
