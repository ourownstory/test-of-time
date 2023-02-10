# make classes available upon package import
from .dataset import Dataset  # noqa: F401 to evade flake8
from .datasets import (  # noqa: F401 to evade flake8
    AirPassengersDataset,
    AusBeerDataset,
    SunspotsNoMissing,
    TourismYearly,
)
