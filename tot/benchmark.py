import datetime
import gc
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from multiprocessing.pool import Pool
from typing import List, Optional, Tuple, Type
import numpy as np
import pandas as pd

from tot.models import Model
from tot.dataset import Dataset
from tot.experiment import Experiment, SimpleExperiment, CrossValidationExperiment


log = logging.getLogger("tot.benchmark")
log.info(
    "Note: The Test of Time benchmarking framework is in early development."
    "Please help us by reporting any bugs and adding documentation."
    "Multiprocessing is not covered by tests and may break on your device."
    "If you use multiprocessing, only run one benchmark per python script."
)


@dataclass
class Benchmark(ABC):
    """Abstract Benchmarking class"""

    metrics: List[str]

    # df_metrics_train: pd.DataFrame = field(init=False)
    # df_metrics_test: pd.DataFrame = field(init=False)

    def __post_init__(self):
        if not hasattr(self, "experiments"):
            self.experiments = self.setup_experiments()
        if not hasattr(self, "num_processes"):
            self.num_processes = 1
        if not hasattr(self, "save_dir"):
            self.save_dir = None

    def setup_experiments(self):
        if self.save_dir is not None:
            for e in self.experiments:
                if e.save_dir is None:
                    e.save_dir = self.save_dir
        return self.experiments

    # def _run_exp(self, exp, verbose=False, exp_num=0):
    def _run_exp(self, args):
        exp, verbose, exp_num = args
        if verbose:
            log.info("--------------------------------------------------------")
            log.info("starting exp {}: {}".format(exp_num, exp.experiment_name))
            log.info("--------------------------------------------------------")
        exp.metrics = self.metrics
        res_train, res_test = exp.run()
        if verbose:
            log.info("--------------------------------------------------------")
            log.info("finished exp {}: {}".format(exp_num, exp.experiment_name))
            log.info("test results {}: {}".format(exp_num, res_test))
            log.info("--------------------------------------------------------")
        # del exp
        # gc.collect()
        return (res_train, res_test)

    def _log_result(self, results):
        if type(results) != list:
            results = [results]
        for res in results:
            res_train, res_test = res
            self.df_metrics_train = pd.concat([self.df_metrics_train, pd.DataFrame([res_train])], ignore_index=True)
            self.df_metrics_test = pd.concat([self.df_metrics_test, pd.DataFrame([res_test])], ignore_index=True)

    def _log_error(self, error):
        log.error(repr(error))

    def run(self, verbose=True):
        # setup DataFrame to store each experiment in a row
        cols = list(self.experiments[0].metadata.keys()) + self.metrics
        self.df_metrics_train = pd.DataFrame(columns=cols)
        self.df_metrics_test = pd.DataFrame(columns=cols)

        if verbose:
            log.info("Experiment list:")
            for i, exp in enumerate(self.experiments):
                log.info("exp {}/{}: {}".format(i + 1, len(self.experiments), exp.experiment_name))
        log.info("---- Staring Series of {} Experiments ----".format(len(self.experiments)))
        if self.num_processes > 1 and len(self.experiments) > 1:
            if not all([exp.num_processes == 1 for exp in self.experiments]):
                raise ValueError("can not set multiprocessing in experiments and Benchmark.")
            with Pool(self.num_processes) as pool:
                args_list = [(exp, verbose, i + 1) for i, exp in enumerate(self.experiments)]
                pool.map_async(self._run_exp, args_list, callback=self._log_result, error_callback=self._log_error)
                pool.close()
                pool.join()
            gc.collect()
        else:
            args_list = [(exp, verbose, i + 1) for i, exp in enumerate(self.experiments)]
            for args in args_list:
                self._log_result(self._run_exp(args))
                gc.collect()

        return self.df_metrics_train, self.df_metrics_test


@dataclass
class CVBenchmark(Benchmark, ABC):
    """Abstract Crossvalidation Benchmarking class"""

    def write_summary_to_csv(self, df_summary, save_dir):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        models = [
            "{}-{}".format(e.metadata["model"], "".join(["_{0}_{1}".format(k, v) for k, v in e.params.items()]))
            for e in self.experiments
        ]
        models = "_".join(list(set(models)))
        stamp = str(datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S-%f"))
        name = "metrics_summary_" + models + stamp + ".csv"
        log.debug(name)
        df_summary.to_csv(os.path.join(save_dir, name), encoding="utf-8", index=False)

    def _summarize_cv_metrics(self, df_metrics, name=None):
        df_metrics_summary = df_metrics.copy(deep=True)
        name = "" if name is None else "_{}".format(name)
        for metric in self.metrics:
            df_metrics_summary[metric + name] = df_metrics[metric].copy(deep=True).apply(lambda x: np.array(x).mean())
            df_metrics_summary[metric + "_std" + name] = (
                df_metrics[metric].copy(deep=True).apply(lambda x: np.array(x).std())
            )
        return df_metrics_summary

    def run(self, verbose=True):
        df_metrics_train, df_metrics_test = super().run(verbose=verbose)
        df_metrics_summary_train = self._summarize_cv_metrics(df_metrics_train)
        df_metrics_summary_train["split"] = "train"
        df_metrics_summary_test = self._summarize_cv_metrics(df_metrics_test)
        df_metrics_summary_test["split"] = "test"
        df_metrics_summary = pd.concat([df_metrics_summary_train, df_metrics_summary_test])
        if self.save_dir is not None:
            self.write_summary_to_csv(df_metrics_summary, save_dir=self.save_dir)
        return df_metrics_summary, df_metrics_train, df_metrics_test


@dataclass
class ManualBenchmark(Benchmark):
    """Manual Benchmarking class
    use example:
    >>> benchmark = ManualBenchmark(
    >>>     metrics=["MAE", "MSE"],
    >>>     experiments=experiment_list, # iterate over this list of experiments
    >>>     save_dir="./logs"
    >>> )
    >>> results_train, results_val = benchmark.run()
    """

    save_dir: Optional[str] = None
    experiments: List[Experiment] = None
    num_processes: int = 1


@dataclass
class ManualCVBenchmark(CVBenchmark):
    """Manual Crossvalidation Benchmarking class
    use example:
    >>> benchmark = ManualCVBenchmark(
    >>>     metrics=["MAE", "MSE"],
    >>>     experiments=cv_experiment_list, # iterate over this list of experiments
    >>>     save_dir="./logs"
    >>> )
    >>> results_train, results_val = benchmark.run()
    """

    save_dir: Optional[str] = None
    experiments: List[Experiment] = None
    num_processes: int = 1


@dataclass
class SimpleBenchmark(Benchmark):
    """
    use example:
    >>> benchmark = SimpleBenchmark(
    >>>     model_classes_and_params=model_classes_and_params, # iterate over this list of tuples
    >>>     datasets=dataset_list, # iterate over this list
    >>>     metrics=["MAE", "MSE"],
    >>>     test_percentage=25,
    >>>     save_dir='./benchmark_logging',
    >>> )
    >>> results_train, results_val = benchmark.run()
    """

    model_classes_and_params: List[Tuple[Model, dict]]
    datasets: List[Dataset]
    test_percentage: float
    save_dir: Optional[str] = None
    num_processes: int = 1

    def setup_experiments(self):
        experiments = []
        for ts in self.datasets:
            for model_class, params in self.model_classes_and_params:
                exp = SimpleExperiment(
                    model_class=model_class,
                    params=params,
                    data=ts,
                    metrics=self.metrics,
                    test_percentage=self.test_percentage,
                    save_dir=self.save_dir,
                )
                experiments.append(exp)
        return experiments


@dataclass
class CrossValidationBenchmark(CVBenchmark):
    """
    example use:
    >>> benchmark_cv = CrossValidationBenchmark(
    >>>     metrics=["MAE", "MSE"],
    >>>     model_classes_and_params=model_classes_and_params, # iterate over this list of tuples
    >>>     datasets=dataset_list, # iterate over this list
    >>>     test_percentage=10,
    >>>     num_folds=3,
    >>>     fold_overlap_pct=0,
    >>>     save_dir="./benchmark_logging/",
    >>> )
    >>> results_summary, results_train, results_val = benchmark_cv.run()
    """

    model_classes_and_params: List[Tuple[Model, dict]]
    datasets: List[Dataset]
    test_percentage: float
    num_folds: int = 5
    fold_overlap_pct: float = 0
    save_dir: Optional[str] = None
    num_processes: int = 1

    def setup_experiments(self):
        experiments = []
        for ts in self.datasets:
            for model_class, params in self.model_classes_and_params:
                exp = CrossValidationExperiment(
                    model_class=model_class,
                    params=params,
                    data=ts,
                    metrics=self.metrics,
                    test_percentage=self.test_percentage,
                    num_folds=self.num_folds,
                    fold_overlap_pct=self.fold_overlap_pct,
                    save_dir=self.save_dir,
                    num_processes=1,
                )
                experiments.append(exp)
        return experiments
