import os

from neuralprophet.utils import set_log_level

from experiments.pipeline.helpers.misc import save_results, save_params
from experiments.pipeline.helpers.plotting import plot_and_save, plot_forecasts
from experiments.pipeline.models import params_generators
from experiments.pipeline.models.models import get_tot_model_class
from experiments.pipeline.models.params import get_params_for_model, get_num_processes
from experiments.pipeline.models.params_generators import validate_param_generator
from tot.benchmark import SimpleBenchmark
from tot.datasets import Dataset

set_log_level("ERROR")


class Pipeline:
    def __init__(
        self,
        model_name,
        params_name,
        data,
        base_dir_name,
        pipeline_name,
        freq="H",
        df_name="synthetic",
    ):
        self.model_class = get_tot_model_class(model_name)
        self.model_params = get_params_for_model(model_name, params_name)
        self.num_processes = get_num_processes(params_name)

        self.results_dir = os.path.join(base_dir_name, "results")

        self.model_name = model_name
        self.pipeline_name = pipeline_name
        self.data = data
        self.df_name = df_name
        self.freq = freq
        self.results_train, self.results_test = None, None
        self.kwargs = None

    def reset(self):
        self.results_train, self.results_test = None, None
        self.kwargs = None

    def run(
        self,
        params_generator_name,
        test_percentage,
        with_scalers=False,
        metrics=None,
        verbose=True,
        plot=False,
        save=False,
        name="",
        **kwargs,
    ):
        self.reset()
        if metrics is None:
            metrics = ["MAE", "RMSE", "MASE"]
        self.kwargs = kwargs

        exp_dir_name = os.path.join(self.results_dir, f"{self.pipeline_name}_{name}")
        data_file_name = os.path.join(exp_dir_name, "data.png")

        if save:
            try:
                os.mkdir(exp_dir_name)
            except OSError as e:
                print(e)

        plot_and_save(self.data, plot=plot, save=save, file_name=data_file_name)

        dataset_list = [
            Dataset(df=self.data, name=self.df_name, freq=self.freq),
        ]

        validate_param_generator(self.model_class, params_generator_name)
        params_generator = getattr(params_generators, params_generator_name)
        model_classes_and_params = params_generator(self.model_params, self.model_class, with_scalers, **kwargs)

        benchmark = SimpleBenchmark(
            model_classes_and_params=model_classes_and_params,
            datasets=dataset_list,
            metrics=metrics,
            test_percentage=test_percentage,
            num_processes=self.num_processes,
        )

        self.results_train, self.results_test = benchmark.run(verbose=verbose)

        plot_forecasts(benchmark=benchmark, dir_name=exp_dir_name, plot=plot, save=save)
        save_results(benchmark=benchmark, metrics=metrics, freq=self.freq, dir=exp_dir_name, save=save)
        save_params(params=model_classes_and_params, dir_name=exp_dir_name, save=save)

    def summary(self):
        print(f"Pipeline {self.pipeline_name}.")
        print(f"Model: {self.model_name} with kwargs: {self.kwargs}")
        print(f"Dataset: {self.df_name}")
        print(f"Results train:")
        print(f"{self.results_train}")
        print(f"Results test:")
        print(f"{self.results_test}")
