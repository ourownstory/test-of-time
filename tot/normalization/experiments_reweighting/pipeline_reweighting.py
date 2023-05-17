import os
import json
import pathlib

from neuralprophet import set_log_level

from tot.benchmark import SimpleBenchmark
from tot.datasets import Dataset
from tot.normalization.experiments_reweighting.utils import (
    gen_model_and_params, plot, plot_forecast
)

set_log_level("ERROR")


def run(df, df_name, freq, model_class, model_params, save=False, dir_name=None, show=False):
    base_dir_name = pathlib.Path(__file__).parent.absolute()
    exp_dir_name = os.path.join(base_dir_name, f"results/{dir_name}")

    if save:
        try:
            os.mkdir(exp_dir_name)
        except OSError as e:
            print(e)

    data_file_name = os.path.join(exp_dir_name, f"{df_name}.png")
    plot(df, show=show, save=save, file_name=data_file_name)

    dataset_list = [
        Dataset(df=df, name=df_name, freq=freq),
    ]

    model_classes_and_params = gen_model_and_params(model_params, model_class)

    benchmark = SimpleBenchmark(
        model_classes_and_params=model_classes_and_params,
        datasets=dataset_list,
        metrics=["MAE", "RMSE", "MAPE", "MASE"],
        test_percentage=0.25,
    )

    results_train, results_test = benchmark.run(verbose=True)
    print("results train", results_train)
    print("results test", results_test)

    for exp, fcst_test in zip(benchmark.experiments, benchmark.fcst_test):
        plot_file_name = os.path.join(exp_dir_name, f"{exp.experiment_name}.png")
        plot_forecast(fcst_test, show=show, save=save, file_name=plot_file_name)

    if save:
        config_file_name = os.path.join(exp_dir_name, f"{df_name}.json")
        with open(config_file_name, "w") as file:
            json.dump(model_params, file)

        results_file_name = os.path.join(exp_dir_name, f"{df_name}.csv")
        results_test.to_csv(results_file_name)
