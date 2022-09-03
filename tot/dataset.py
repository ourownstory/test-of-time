import logging

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Type

import pandas as pd

log = logging.getLogger("tot.dataset")


@dataclass
class Dataset:
    """
    example use:
    >>> dataset = Dataset(
    >>>     df = pd.read_csv('air_passengers.csv'),
    >>>     name = "air_passengers",
    >>>     freq = "MS",
    >>>     seasonalities = [365.25,], # yearly seasonality
    >>>     seasonality_mode = "multiplicative",
    >>> ),
    """

    df: pd.DataFrame
    name: str
    freq: str
    seasonalities: List = field(default_factory=list)
    seasonality_mode: Optional[str] = None
