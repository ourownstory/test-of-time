import os
import pathlib

from neuralprophet.utils import set_log_level

from experiments.utils import (
    gen_model_and_params,
    plot_and_save,
    plot_forecasts,
    save_params,
    save_results,
)
from tot.benchmark import SimpleBenchmark
from tot.datasets import Dataset

set_log_level("ERROR")


def run(
    df,
    df_name,
    freq,
    model_class,
    model_params,
    metrics=["MAE", "RMSE", "MASE"],
    test_percentage=0.25,
    save=True,
    plot=False,
    dir_name=None,
    scalers="default",
    scaling_levels="default",
    reweight_loss=True,
    num_processes=19,
    model_and_params_generator=gen_model_and_params,
):
    base_dir_name = pathlib.Path(__file__).parent.absolute()
    results_dir_name = os.path.join(base_dir_name, f"results")
    exp_dir_name = os.path.join(results_dir_name, f"{dir_name}")
    data_file_name = os.path.join(exp_dir_name, "data.png")
    param_file_name = os.path.join(exp_dir_name, "model_classes_and_params.csv")
    if save:
        try:
            os.mkdir(exp_dir_name)
        except OSError as e:
            print(e)

    plot_and_save(df, plot=plot, save=save, file_name=data_file_name)

    dataset_list = [
        Dataset(df=df, name=df_name, freq=freq),
    ]
    model_classes_and_params = model_and_params_generator(
        model_params, model_class, scalers, scaling_levels, reweight_loss
    )
    benchmark = SimpleBenchmark(
        model_classes_and_params=model_classes_and_params,
        datasets=dataset_list,
        metrics=metrics,
        test_percentage=test_percentage,
        num_processes=num_processes,
    )
    results_train, results_test = benchmark.run(verbose=True)
    print("results test", results_test)

    plot_forecasts(benchmark=benchmark, dir=exp_dir_name, plot=plot, save=save)
    save_results(benchmark=benchmark, metrics=metrics, freq=freq, dir=exp_dir_name, save=True)
    save_params(params=model_classes_and_params, dir=exp_dir_name, df_name=df_name, save=True)
