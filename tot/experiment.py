import gc
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocessing.pool import Pool
from typing import List, Optional

import pandas as pd
from neuralprophet import set_random_seed

from tot.dataset import Dataset
from tot.df_utils import (
    check_dataframe,
    crossvalidation_split_df,
    handle_missing_data,
    maybe_drop_added_dates,
    prep_or_copy_df,
    split_df,
)
from tot.exp_utils import evaluate_forecast
from tot.models import Model

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
        if hasattr(self.data, "freq") and self.data.freq is not None:
            data_params["freq"] = self.data.freq
        self.params.update({"_data_params": data_params})
        if not hasattr(self, "experiment_name") or self.experiment_name is None:
            self.experiment_name = "{}_{}{}".format(
                self.data.name,
                self.model_class.model_name,
                r"".join([r"_{0}_{1}".format(k, v) for k, v in self.params.items()]).replace("'", "").replace(":", "_"),
            )
        if not hasattr(self, "metadata") or self.metadata is None:
            self.metadata = {
                "data": self.data.name,
                "model": self.model_class.model_name,
                "params": str(self.params),
                "experiment": self.experiment_name,
            }

    def write_results_to_csv(self, df, prefix, current_fold=None):
        """
        Write evaluation results to a CSV file.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing the results to be written to file.
        prefix : str
            Prefix to be added to the filename.
        current_fold : int, optional
            Fold number, to be included in the filename if specified.

        Returns
        -------
        None
        """
        # save fcst and create dir if necessary
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        name = self.experiment_name
        if current_fold is not None:
            name = name + "_fold_" + str(current_fold)
        name = prefix + "_" + name + ".csv"
        df.to_csv(os.path.join(self.save_dir, name), encoding="utf-8", index=False)

    def _make_forecast(self, model, df_train, df_test, current_fold=None):
        """
        Make predictions using the given model on the train and test data.

        Parameters
        ----------
        model : object
            The model to be used for prediction.
        df_train : pandas.DataFrame
            The train data.
        df_test : pandas.DataFrame
            The test data.
        current_fold : int, optional
            Fold number, to be included in the filename if saving results to disk.

        Returns
        -------
        fcst_train : pandas.DataFrame
            Predictions on the train data.
        fcst_test : pandas.DataFrame
            Predictions on the test data.
        """
        fcst_train = model.predict(df=df_train)
        fcst_test = model.predict(df=df_test, df_historic=df_train)
        if self.save_dir is not None:
            self.write_results_to_csv(fcst_train, prefix="predicted_train", current_fold=current_fold)
            self.write_results_to_csv(fcst_test, prefix="predicted_test", current_fold=current_fold)
        # del fcst_train #TODO save as optional
        # del fcst_test
        # gc.collect()
        return fcst_train, fcst_test

    def _evaluate_model(self, fcst_train, fcst_test):
        """
        Evaluate a forecast by computing various metrics on both the train and test sets.

        Parameters
        ----------
        fcst_train : DataFrame
            DataFrame of forecast results on the training set.
        fcst_test : DataFrame
            DataFrame of forecast results on the test set.
        metrics : list of str
            List of metrics to compute, such as 'mse' or 'mae'.
        metadata : dict-like
            Metadata to add to the result.

        Returns
        -------
        result_train : DataFrame
            DataFrame of results for the training set.
        result_test : DataFrame
            DataFrame of results for the test set.
        """

        metadata = self.metadata.copy()
        metrics = self.metrics
        result_train, result_test = evaluate_forecast(
            fcst_train=fcst_train, fcst_test=fcst_test, metrics=metrics, metadata=metadata
        )

        return result_train, result_test

    @abstractmethod
    def run(self):
        """
        Runs the experiment.
        """
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
        """
        Runs the experiment.

        Returns:
            tuple: (fcst_train, fcst_test, result_train, result_test)
        """
        # data-specific pre-processing
        set_random_seed(42)
        df, received_ID_col, received_single_time_series, _ = prep_or_copy_df(self.data.df)
        df = check_dataframe(df, check_y=True)
        # add infer frequency
        df = handle_missing_data(df, freq=self.data.freq)
        df_train, df_test = split_df(
            df=df,
            test_percentage=self.test_percentage,
        )
        # fit model
        model = self.model_class(self.params)
        model.fit(df=df_train, freq=self.data.freq)
        # predict model
        fcst_train, fcst_test = self._make_forecast(model=model, df_train=df_train, df_test=df_test)
        # data-specific post-processing
        fcst_train, df_train = maybe_drop_added_dates(fcst_train, df_train)
        fcst_test, df_test = maybe_drop_added_dates(fcst_test, df_test)
        # evaluation
        result_train, result_test = self._evaluate_model(fcst_train, fcst_test)
        return fcst_train, fcst_test, result_train, result_test


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
        """
        Function to run a single fold of the cross-validation experiment.

        Parameters
        ----------
        args: Tuple
            A tuple of (df_train, df_test, current_fold).
            df_train: pandas DataFrame
                The training data for the current fold.
            df_test: pandas DataFrame
                The test data for the current fold.
            current_fold: int, optional
                The index of the current fold.

        Returns
        -------
        Tuple
            Tuple of 4 elements: fcst_train, fcst_test, result_train, result_test.
            fcst_train: pandas DataFrame
                Forecast values for the training data of the current fold.
            fcst_test: pandas DataFrame
                Forecast values for the test data of the current fold.
            result_train: dict
                Dictionary containing the evaluation metrics for the training data of the current fold.
            result_test: dict
                Dictionary containing the evaluation metrics for the test data of the current fold.
        """
        set_random_seed(42)
        df_train, df_test, current_fold = args
        # fit model
        model = self.model_class(self.params)
        model.fit(df=df_train, freq=self.data.freq)
        # predict model
        fcst_train, fcst_test = self._make_forecast(
            model=model, df_train=df_train, df_test=df_test, current_fold=current_fold
        )
        # data-specific post-processing
        fcst_train, df_train = maybe_drop_added_dates(fcst_train, df_train)
        fcst_test, df_test = maybe_drop_added_dates(fcst_test, df_test)
        # evaluation
        result_train, result_test = self._evaluate_model(fcst_train, fcst_test)
        del model
        gc.collect()
        return (fcst_train, fcst_test, result_train, result_test)

    def _log_results(self, results):
        """Log the results of a model's run.

        Parameters
        ----------
        results : list of tuple
            Results of model run, which includes fcst_train, fcst_test,
            result_train, and result_test.

        Returns
        -------
            None
        """

        if type(results) != list:
            results = [results]
        for res in results:
            fcst_train, fcst_test, result_train, result_test = res
            for m in self.metrics:
                self.results_cv_train[m].append(result_train[m])
                self.results_cv_test[m].append(result_test[m])
            self.fcst_train.append(fcst_train)
            self.fcst_test.append(fcst_test)

    def _log_error(self, error):
        """
        Logs the errors.

        """
        log.error(repr(error))

    def run(self):
        """
        Runs the experiment.

        Returns:
            tuple: (fcst_train, fcst_test, results_cv_train, results_cv_test)
        """
        set_random_seed(42)
        # data-specific pre-processing
        df, received_ID_col, received_single_time_series, _ = prep_or_copy_df(self.data.df)
        df = check_dataframe(df, check_y=True)
        # add infer frequency
        df = handle_missing_data(df, freq=self.data.freq)
        folds = crossvalidation_split_df(
            df=self.data.df,
            freq=self.data.freq,
            k=self.num_folds,
            fold_pct=self.test_percentage,
            fold_overlap_pct=self.fold_overlap_pct,
        )
        # init empty dicts with list for fold-wise metrics
        self.results_cv_train = self.metadata.copy()
        self.results_cv_test = self.metadata.copy()
        for m in self.metrics:
            self.results_cv_train[m] = []
            self.results_cv_test[m] = []
        self.fcst_train = []
        self.fcst_test = []
        if self.num_processes > 1 and self.num_folds > 1:
            with Pool(self.num_processes) as pool:
                args = [(df_train, df_test, current_fold) for current_fold, (df_train, df_test) in enumerate(folds)]
                pool.map_async(
                    self._run_fold,
                    args,
                    callback=self._log_results,
                    error_callback=self._log_error,
                )
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
                [results_cv_test_df, pd.DataFrame([self.results_cv_test])],
                ignore_index=True,
            )
            results_cv_train_df = pd.concat(
                [results_cv_train_df, pd.DataFrame([self.results_cv_train])],
                ignore_index=True,
            )
            self.write_results_to_csv(results_cv_test_df, prefix="summary_test")
            self.write_results_to_csv(results_cv_train_df, prefix="summary_train")

        return self.fcst_train, self.fcst_test, self.results_cv_train, self.results_cv_test
