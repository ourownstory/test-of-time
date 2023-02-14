"""
Datasets
--------
Popular datasets. Lists all datasets that are available. For adding a new dataset, setup a new instance and
fill respective information.
"""

# imports
from dataclasses import dataclass

from .dataset_loader import DatasetLoaderCSV, DatasetLoaderTSF, DatasetMetadataLoader

"""
    Overall usage of this package:
    from tot.datasets import AirPassengersDataset
    time series: df = AirPassengersDataset.load()
"""
# TODO: add logging
# logger = get_logger(__name__)


# example for loading local .csv
@dataclass
class AirPassengersDataset(DatasetLoaderCSV):
    """
    Monthly Air Passengers Dataset, from 1949 to 1960.
    References
    ----------
    .. [1] https://www.kaggle.com/datasets/chirag19/air-passengers
    """

    metadata: DatasetMetadataLoader = DatasetMetadataLoader(
        name="air_passengers.csv",
        freq="MS",
        start_timestamp="1949-01-01",
        horizon=3,
        missing=False,
        equallength=True,
        multivariate=False,
        seasonality_mode="multiplicative",
    )


# example for downloading .csv
@dataclass
class AusBeerDataset(DatasetLoaderCSV):
    """
    Total quarterly beer production in Australia (in megalitres) from 1956:Q1 to 2008:Q3 [1]_.
    References
    ----------
    .. [1] https://rdrr.io/cran/fpp/man/ausbeer.html
    """

    metadata: DatasetMetadataLoader = DatasetMetadataLoader(
        "ausbeer.csv",
        url="https://raw.githubusercontent.com/unit8co/darts/master/datasets/ausbeer.csv",
        header_time="date",
        format_time="%Y-%m-%d",
    )


# example for downloading .tsf
@dataclass
class SunspotsNoMissing(DatasetLoaderTSF):
    """
    Contains the single daily time series representing the sunspot numbers from 08/01/1818 to 31/05/2020. As the
    dataset contains missing values, a LOCF-imputed version is included.
    ----------
    .. [1] http://doi.org/10.5281/zenodo.4654722
    """

    metadata: DatasetMetadataLoader = DatasetMetadataLoader(
        name="sunspot_dataset_without_missing_values.tsf",
        url="https://zenodo.org/record/4654722/files/sunspot_dataset_without_missing_values.zip",
    )


# example for loading local .tsf
@dataclass
class TourismYearly(DatasetLoaderTSF):
    """
    This dataset originates from a Kaggle competition and contains 518 yearly time series related to tourism.
    ----------
    .. [1] http://doi.org/10.5281/zenodo.4656103
    """

    metadata: DatasetMetadataLoader = DatasetMetadataLoader(
        name="tourism_yearly_dataset.tsf",
        # url="https://zenodo.org/record/4656103/files/tourism_yearly_dataset.zip",
    )
