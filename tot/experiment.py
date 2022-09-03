import gc
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from multiprocessing.pool import Pool
from typing import List, Optional, Tuple, Type

import numpy as np
import pandas as pd
from neuralprophet import df_utils

from tot.dataset import Dataset
from tot.models import Model
from tot.metrics import ERROR_FUNCTIONS


log = logging.getLogger("tot.benchmark")
log.debug(
    "Note: The Test of Time benchmarking framework is in early development."
    "Please help us by reporting any bugs and adding documentation."
    "Multiprocessing is not covered by tests and may break on your device."
    "If you use multiprocessing, only run one benchmark per python script."
)


@dataclass
class Experiment(ABC):
    model_class: Model
    params: dict
    data: Dataset
    metrics: List[str]
    test_percentage: float
    experiment_name: Optional[str] = None
    metadata: Optional[dict] = None
    save_dir: Optional[str] = None
    num_processes: int = 1

    def __post_init__(self):

        data_params = {}
        if len(self.data.seasonalities) > 0:
            data_params["seasonalities"] = self.data.seasonalities
        if hasattr(self.data, "seasonality_mode") and self.data.seasonality_mode is not None:
            data_params["seasonality_mode"] = self.data.seasonality_mode
        self.params.update({"_data_params": data_params})
        if not hasattr(self, "experiment_name") or self.experiment_name is None:
            self.experiment_name = "{}_{}{}".format(
                self.data.name,
                self.model_class.model_name,
                "".join(["_{0}_{1}".format(k, v) for k, v in self.params.items()]),
            )
        if not hasattr(self, "metadata") or self.metadata is None:
            self.metadata = {
                "data": self.data.name,
                "model": self.model_class.model_name,
                "params": str(self.params),
                "experiment": self.experiment_name,
            }

    def write_results_to_csv(self, df, prefix, current_fold=None):
        # save fcst and create dir if necessary
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        name = self.experiment_name
        if current_fold is not None:
            name = name + "_fold_" + str(current_fold)
        name = prefix + "_" + name + ".csv"
        df.to_csv(os.path.join(self.save_dir, name), encoding="utf-8", index=False)

    def _evaluate_model(self, model, df_train, df_test, current_fold=None):
        df_test = model.maybe_add_first_inputs_to_df(df_train, df_test)
        min_length = model.n_lags + model.n_forecasts
        if min_length > len(df_train):
            raise ValueError("Not enough training data to create a single input sample.")
        elif len(df_train) - min_length < 5:
            log.warning("Less than 5 training samples")
        if min_length > len(df_test):
            raise ValueError("Not enough test data to create a single input sample.")
        elif len(df_test) - min_length < 5:
            log.warning("Less than 5 test samples")
        fcst_train = model.predict(df_train)
        fcst_test = model.predict(df_test)
        # remove added input lags
        fcst_train, df_train = model.maybe_drop_first_forecasts(fcst_train, df_train)
        fcst_test, df_test = model.maybe_drop_first_forecasts(fcst_test, df_test)
        # remove interpolated dates
        fcst_train, df_train = model.maybe_drop_added_dates(fcst_train, df_train)
        fcst_test, df_test = model.maybe_drop_added_dates(fcst_test, df_test)

        result_train = self.metadata.copy()
        result_test = self.metadata.copy()
        for metric in self.metrics:
            # todo: parallelize
            n_yhats_train = sum(["yhat" in colname for colname in fcst_train.columns])
            n_yhats_test = sum(["yhat" in colname for colname in fcst_test.columns])

            assert n_yhats_train == n_yhats_test, "Dimensions of fcst dataframe faulty."

            metric_train_list = []
            metric_test_list = []

            fcst_train = fcst_train.fillna(value=np.nan)
            df_train = df_train.fillna(value=np.nan)
            fcst_test = fcst_test.fillna(value=np.nan)
            df_test = df_test.fillna(value=np.nan)

            for x in range(1, n_yhats_train + 1):
                metric_train_list.append(
                    ERROR_FUNCTIONS[metric](
                        predictions=fcst_train["yhat{}".format(x)].values,
                        truth=df_train["y"].values,
                        truth_train=df_train["y"].values,
                    )
                )
                metric_test_list.append(
                    ERROR_FUNCTIONS[metric](
                        predictions=fcst_test["yhat{}".format(x)].values,
                        truth=df_test["y"].values,
                        truth_train=df_train["y"].values,
                    )
                )
            result_train[metric] = np.nanmean(metric_train_list, dtype="float32")
            result_test[metric] = np.nanmean(metric_test_list, dtype="float32")

        if self.save_dir is not None:
            self.write_results_to_csv(fcst_train, prefix="predicted_train", current_fold=current_fold)
            self.write_results_to_csv(fcst_test, prefix="predicted_test", current_fold=current_fold)
        del fcst_train
        del fcst_test
        gc.collect()
        return result_train, result_test

    @abstractmethod
    def run(self):
        pass


@dataclass
class SimpleExperiment(Experiment):
    """
    use example:
    >>> ts = Dataset(df = air_passengers_df, name = "air_passengers", freq = "MS")
    >>> params = {"seasonality_mode": "multiplicative"}
    >>> exp = SimpleExperiment(
    >>>     model_class=NeuralProphetModel,
    >>>     params=params,
    >>>     data=ts,
    >>>     metrics=["MAE", "MSE"],
    >>>     test_percentage=25,
    >>>     save_dir='./benchmark_logging',
    >>> )
    >>> result_train, result_val = exp.run()
    """

    def run(self):
        df_train, df_test = df_utils.split_df(
            df=self.data.df,
            n_lags=0,
            n_forecasts=1,
            valid_p=self.test_percentage / 100.0,
        )
        model = self.model_class(self.params)
        model.fit(df=df_train, freq=self.data.freq)
        result_train, result_test = self._evaluate_model(model, df_train, df_test)
        return result_train, result_test


@dataclass
class CrossValidationExperiment(Experiment):
    """
    >>> ts = Dataset(df = air_passengers_df, name = "air_passengers", freq = "MS")
    >>> params = {"seasonality_mode": "multiplicative"}
    >>> exp = CrossValidationExperiment(
    >>>     model_class=NeuralProphetModel,
    >>>     params=params,
    >>>     data=ts,
    >>>     metrics=["MAE", "MSE"],
    >>>     test_percentage=10,
    >>>     num_folds=3,
    >>>     fold_overlap_pct=0,
    >>>     save_dir="./benchmark_logging/",
    >>> )
    >>> result_train, result_train, result_val = exp.run()
    """

    num_folds: int = 5
    fold_overlap_pct: float = 0
    global_model_cv_type: str = "global-time"
    # results_cv_train: dict = field(init=False)
    # results_cv_test: dict = field(init=False)

    def _run_fold(self, args):
        df_train, df_test, current_fold = args
        model = self.model_class(self.params)
        model.fit(df=df_train, freq=self.data.freq)
        result_train, result_test = self._evaluate_model(model, df_train, df_test, current_fold=current_fold)
        del model
        gc.collect()
        return (result_train, result_test)

    def _log_results(self, results):
        if type(results) != list:
            results = [results]
        for res in results:
            result_train, result_test = res
            for m in self.metrics:
                self.results_cv_train[m].append(result_train[m])
                self.results_cv_test[m].append(result_test[m])

    def _log_error(self, error):
        log.error(repr(error))

    def run(self):
        folds = df_utils.crossvalidation_split_df(
            df=self.data.df,
            n_lags=0,
            n_forecasts=1,
            k=self.num_folds,
            fold_pct=self.test_percentage / 100.0,
            fold_overlap_pct=self.fold_overlap_pct / 100.0,
            global_model_cv_type=self.global_model_cv_type,
        )
        # init empty dicts with list for fold-wise metrics
        self.results_cv_train = self.metadata.copy()
        self.results_cv_test = self.metadata.copy()
        for m in self.metrics:
            self.results_cv_train[m] = []
            self.results_cv_test[m] = []
        if self.num_processes > 1 and self.num_folds > 1:
            with Pool(self.num_processes) as pool:
                args = [(df_train, df_test, current_fold) for current_fold, (df_train, df_test) in enumerate(folds)]
                pool.map_async(self._run_fold, args, callback=self._log_results, error_callback=self._log_error)
                pool.close()
                pool.join()
            gc.collect()
        else:
            for current_fold, (df_train, df_test) in enumerate(folds):
                args = (df_train, df_test, current_fold)
                self._log_results(self._run_fold(args))

        if self.save_dir is not None:
            results_cv_test_df = pd.DataFrame()
            results_cv_train_df = pd.DataFrame()
            results_cv_test_df = pd.concat(
                [results_cv_test_df, pd.DataFrame([self.results_cv_test])], ignore_index=True
            )
            results_cv_train_df = pd.concat(
                [results_cv_train_df, pd.DataFrame([self.results_cv_train])], ignore_index=True
            )
            self.write_results_to_csv(results_cv_test_df, prefix="summary_test")
            self.write_results_to_csv(results_cv_train_df, prefix="summary_train")

        return self.results_cv_train, self.results_cv_test
