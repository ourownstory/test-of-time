import logging

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Type

import pandas as pd
from .dataset_loader import DatasetLoaderCSV

log = logging.getLogger("tot.dataset")


@dataclass
class Dataset:
    """
    >>> dataset = Dataset(
    >>>     dataset = AirPassengersDataset(),
    >>>     name = "air_passengers",
    >>>     freq = "MS",
    >>>     seasonalities = [365.25,], # yearly seasonality
    >>>     seasonality_mode = "multiplicative",
    >>> ),
    """

    dataset: DatasetLoaderCSV
    name: str
    freq: Optional[str] = None
    seasonalities: Optional[List] = field(default_factory=list)
    seasonality_mode: Optional[str] = "additive"
