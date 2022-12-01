import logging
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from distutils.util import strtobool
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, List, Optional
from urllib import request
from zipfile import ZipFile

import pandas as pd

# TODO: adapt logger
log = logging.getLogger("ds")
log.setLevel("INFO")


@dataclass
class DatasetMetadataLoader:
    """
    Class that is used for loading the dataset metadata.
    """

    # name of the dataset file, including extension
    name: str
    # url of the dataset, expects a publicly available file. Otherwise dataset needs to be stored in test-of-time/datasets
    url: Optional[str] = None
    # used to indicate the freq when we already know it
    freq: Optional[str] = None
    # not sure if we need it
    header_time: Optional[str] = None
    # used to convert the string date to pd.Datetime
    # https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
    format_time: Optional[str] = None
    # not sure if we need
    start_timestamp: Optional[str] = None
    # recommended prediction horizon
    horizon: Optional[int] = (None,)
    # indicate whether dataset contains missing values
    missing: Optional[bool] = (None,)
    # indicate whether time series are equal length
    equallength: Optional[bool] = None
    # multivariate
    multivariate: Optional[bool] = None
    # if seasonalities are known
    seasonalities: Optional[List] = field(default_factory=list)
    # if seasonality mode is known
    seasonality_mode: Optional[str] = None
    # a custom function to handling non-csv based datasets
    pre_process_zipped_csv_fn: Optional[Callable] = None


class DatasetLoadingException(BaseException):
    pass


@dataclass
class DatasetLoader(ABC):
    """
    Class that loads/ downloads a dataset and stores it locally.
    Assumes that the file can be downloaded (i.e. publicly available via a URL) or is already stores locally
    """

    metadata: DatasetMetadataLoader
    _root_path: Optional[Path] = None
    _DEFAULT_DIRECTORY = os.path.join(Path(__file__).parent.parent.parent.absolute(), "datasets")

    def __post_init__(self):
        if self._root_path is None:
            self._root_path: Path = self._DEFAULT_DIRECTORY

    def load(self) -> pd.DataFrame:
        """
        Load the dataset in memory, as a pd.Dataframe.
        Downloads the dataset if it is not present already.

        Returns
        -------
        dataframe: pd.Dataframe
            A dataframe that contains the dataset
        """

        if self.metadata.url is not None:  # replace with: if not self._is_already_downloaded():
            if self.metadata.url.endswith(".zip"):
                self._download_zip_dataset()
            else:
                self._download_dataset_file()

        # TODO: add integrity check  self._check_dataset_integrity_or_raise(
        return self._load_from_disk(self._get_path_dataset(), self.metadata)

    def _download_dataset_file(self):
        """
        Downloads the dataset in the root_path directory

        Raises
        -------
        DatasetLoadingException
            if downloading or writing the file to disk fails
        """
        os.makedirs(self._root_path, exist_ok=True)

        try:
            with TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                dataset_path = self._download(temp_path)
                # archive.extractall(path=temp_path)
                shutil.copy(
                    os.path.join(temp_path, self.metadata.name), os.path.join(self._root_path, self.metadata.name)
                )
                shutil.rmtree(temp_dir)  # delete directory
        except Exception as e:
            raise DatasetLoadingException("Could not download the dataset. Reason:" + e.__repr__()) from None

    def _download_zip_dataset(self):
        """
        Downloads the dataset .zip and extracts it to the root_path directory.
        """
        os.makedirs(self._root_path, exist_ok=True)

        try:
            with TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                with ZipFile(self._download(temp_path)) as archive:
                    archive.extractall(path=temp_path)
                    shutil.copy(
                        os.path.join(temp_path, self.metadata.name), os.path.join(self._root_path, self.metadata.name)
                    )
                    shutil.rmtree(temp_dir)  # delete directory
        except Exception as e:
            raise DatasetLoadingException("Could not download the dataset. Reason:" + e.__repr__()) from None

    def _download(self, path: Path):  # TODO: move to utility functions file
        """
        Downloads file at given path.
        """
        if self.metadata.name.endswith("tsf"):
            metadata_name = self.metadata.name.replace("tsf", "zip")
        else:
            metadata_name = self.metadata.name
        file_path = path / metadata_name
        # TODO: add progress bar and itshook
        request.urlretrieve(
            self.metadata.url,
            filename=file_path,
        )
        return file_path

    @abstractmethod
    def _load_from_disk(self, path_to_file: Path, metadata: DatasetMetadataLoader) -> pd.DataFrame:
        """
        Given a Path to the file and a DataLoaderMetadata object, return a pd.Dataframe.
        One can assume that the file exists before this function is called

        Parameters
        ----------
        path_to_file: Path
            A Path object where the dataset is located
        metadata: Metadata
            The dataset's metadata

        Returns
        -------
        df: pd.Dataframe
            a TimeSeries object that contains the whole dataset
        """
        pass

    def _get_path_dataset(self) -> Path:
        return Path(os.path.join(self._root_path, self.metadata.name))

    def _is_already_downloaded(self) -> bool:
        return os.path.isfile(self._get_path_dataset())

    def _format_time_column(self, df):
        df[self.metadata.header_time] = pd.to_datetime(
            df[self.metadata.header_time],
            format=self.metadata.format_time,
            errors="raise",
        )
        return df

    def convert_nested_df_to_df(self, df) -> pd.DataFrame:  # move to utility function
        pass


@dataclass()
class DatasetLoaderCSV(DatasetLoader):
    def _load_from_disk(self, path_to_file: Path, metadata: DatasetMetadataLoader) -> pd.DataFrame:
        """
        Given a Path to the .csv file and a DataLoaderMetadata object, return a pd.Dataframe.
        One can assume that the file exists before this function is called

        Parameters
        ----------
        path_to_file: Path
            A Path object where the dataset is located
        metadata: Metadata
            The dataset's metadata

        Returns
        -------
        df: pd.Dataframe
            a TimeSeries object that contains the whole dataset
        """
        df = pd.read_csv(path_to_file)
        if metadata.format_time is not None:
            df = self._format_time_column(df)
        if metadata.header_time is not None:
            df.rename(columns={self.metadata.header_time: "ds"}, inplace=True)
        df.sort_index(inplace=True)

        return df


@dataclass()
class DatasetLoaderTSF(DatasetLoader):
    def _load_from_disk(self, path_to_file: Path, metadata: DatasetMetadataLoader) -> pd.DataFrame:
        """
        Given a Path to the .tsf file and a DataLoaderMetadata object, return a pd.Dataframe.
        One can assume that the file exists before this function is called

        Parameters
        ----------
        path_to_file: Path
            A Path object where the dataset is located
        metadata: Metadata
            The dataset's metadata

        Returns
        -------
        df: pd.Dataframe
            a TimeSeries object that contains the whole dataset
        """

        (
            df,
            self.metadata.freq,
            self.metadata.horizon,
            self.metadata.equallength,
            self.metadata.missing,
        ) = self._convert_tsf_to_dataframe(
            full_file_path_and_name=path_to_file,
            replace_missing_vals_with="NaN",
            value_column_name="y",  # TODO: generalize for multivariate
        )
        # extract time series of multivariate data to common df
        # TODO: add query if multivariate
        df = self.convert_nested_df_to_df(df)

    def _convert_tsf_to_dataframe(
        self,
        full_file_path_and_name,
        replace_missing_vals_with="NaN",
        value_column_name="series_value",
    ) -> (pd.DataFrame, int, int, bool, bool):
        """
        Converts the contents in a .tsf file into a dataframe and returns it along with other metadata
        of the dataset.

        Parameters
         ----------
        full_file_path_and_name: str
            complete .tsf file path
        replace_missing_vals_with: str
            a term to indicate the missing values in series in the returning dataframe
        value_column_name: str
            Any name that is preferred to have as the name of the column containing series values
        """
        col_names = []
        col_types = []
        all_data = {}
        line_count = 0
        frequency = None
        forecast_horizon = None
        contain_missing_values = None
        contain_equal_length = None
        found_data_tag = False
        found_data_section = False
        started_reading_data_section = False

        with open(full_file_path_and_name, "r", encoding="cp1252") as file:
            for line in file:
                # Strip white space from start/end of line
                line = line.strip()

                if line:
                    if line.startswith("@"):  # Read meta-data
                        if not line.startswith("@data"):
                            line_content = line.split(" ")
                            if line.startswith("@attribute"):
                                if len(line_content) != 3:  # Attributes have both name and type
                                    raise Exception("Invalid meta-data specification.")

                                col_names.append(line_content[1])
                                col_types.append(line_content[2])
                            else:
                                if len(line_content) != 2:  # Other meta-data have only values
                                    raise Exception("Invalid meta-data specification.")

                                if line.startswith("@frequency"):
                                    frequency = line_content[1]  # TODO: convert to pandas frequency table
                                elif line.startswith("@horizon"):
                                    forecast_horizon = int(line_content[1])
                                elif line.startswith("@missing"):
                                    contain_missing_values = bool(strtobool(line_content[1]))
                                elif line.startswith("@equallength"):
                                    contain_equal_length = bool(strtobool(line_content[1]))

                        else:
                            if len(col_names) == 0:
                                raise Exception("Missing attribute section. Attribute section must come before data.")

                            found_data_tag = True
                    elif not line.startswith("#"):
                        if len(col_names) == 0:
                            raise Exception("Missing attribute section. Attribute section must come before data.")
                        elif not found_data_tag:
                            raise Exception("Missing @data tag.")
                        else:
                            if not started_reading_data_section:
                                started_reading_data_section = True
                                found_data_section = True
                                all_series = []

                                for col in col_names:
                                    all_data[col] = []

                            full_info = line.split(":")

                            if len(full_info) != (len(col_names) + 1):
                                raise Exception("Missing attributes/values in series.")

                            series = full_info[len(full_info) - 1]
                            series = series.split(",")

                            if len(series) == 0:
                                raise Exception(
                                    "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                                )

                            numeric_series = []

                            for val in series:
                                if val == "?":
                                    numeric_series.append(replace_missing_vals_with)
                                else:
                                    numeric_series.append(float(val))

                            if numeric_series.count(replace_missing_vals_with) == len(numeric_series):
                                raise Exception(
                                    "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                                )

                            all_series.append(pd.Series(numeric_series).array)

                            for i in range(len(col_names)):
                                att_val = None
                                if col_types[i] == "numeric":
                                    att_val = int(full_info[i])
                                elif col_types[i] == "string":
                                    att_val = str(full_info[i])
                                elif col_types[i] == "date":
                                    att_val = datetime.strptime(full_info[i], "%Y-%m-%d %H-%M-%S")
                                else:
                                    raise Exception(
                                        "Invalid attribute type."
                                    )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                                if att_val is None:
                                    raise Exception("Invalid attribute value.")
                                else:
                                    all_data[col_names[i]].append(att_val)

                    line_count = line_count + 1

            if line_count == 0:
                raise Exception("Empty file.")
            if len(col_names) == 0:
                raise Exception("Missing attribute section.")
            if not found_data_section:
                raise Exception("Missing series information under data section.")

            all_data[value_column_name] = all_series
            loaded_data = pd.DataFrame(all_data)

            return (
                loaded_data,
                frequency,
                forecast_horizon,
                contain_missing_values,
                contain_equal_length,
            )

    def convert_nested_df_to_df(self, df) -> pd.DataFrame:
        """
        Converts the series in a df with multivariate data to one common df.
        """
        df_converted = pd.DataFrame()
        for index, row in df.iterrows():
            # TODO: check if format of df is suited for neuralprophet
            ID = row.series_name
            series = row.y
            dti = pd.date_range(row.start_timestamp, periods=row.y.size, freq="D")  # TODO: adopt frequency
            df_converted = pd.concat(
                [df_converted, pd.DataFrame({"ds": dti, "y": series, "ID": ID})], ignore_index=True
            )

        df_converted.sort_values(by="ds", inplace=True)
        return df_converted


@dataclass
class MaunalDataset(DatasetLoader):
    """
    >>> dataset = Dataset(
    >>>     filename = "air_passengers.csv",
    >>>     name = "AirPassengers",
    >>>     freq = "MS",
    >>>     seasonalities = [365.25,], # yearly seasonality
    >>>     seasonality_mode = "multiplicative",
    >>> ),
    """

    metadata: DatasetMetadataLoader = None
    _root_path: Optional[Path] = None
    filename: str = None
    name: Optional[str] = None
    freq: Optional[str] = None
    seasonalities: Optional[List] = field(default_factory=list)
    seasonality_mode: Optional[str] = "additive"

    def __post_init__(self):
        self.metadata = DatasetMetadataLoader(name=self.filename)

    def _load_from_disk(self, path_to_file: Path, metadata: DatasetMetadataLoader) -> pd.DataFrame:
        pass
