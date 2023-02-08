import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Type

import pandas as pd

from .dataset_loader import DatasetLoaderCSV, DatasetMetadataLoader

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

    df: pd.DataFrame  # only temporary
    dataset: DatasetLoaderCSV = None
    name: Optional[str] = None
    freq: Optional[str] = None
    seasonalities: Optional[List] = field(default_factory=list)
    seasonality_mode: Optional[str] = "additive"
    # TODO: add future attributes that need to be mapped to metadata

    def __post_init__(self):
        if self.dataset is None:
            self.dataset = DatasetLoaderCSV(
                metadata=DatasetMetadataLoader(
                    name=self.name,
                    freq=self.freq,
                    seasonalities=self.seasonalities,
                    seasonality_mode=self.seasonality_mode,
                )
            )
