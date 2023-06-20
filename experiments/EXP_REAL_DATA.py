import multiprocessing
import pathlib as pathlib
import time

from experiments.pipeline.helpers.arg_parsers import get_real_data_arg_parser
from experiments.pipeline.helpers.data_loaders import DATASETS
from experiments.pipeline.helpers.misc import build_real_data_name
from experiments.pipeline.pipeline import Pipeline

if __name__ == "__main__":
    args = get_real_data_arg_parser()

    # Freezing support for multiprocessing
    multiprocessing.freeze_support()

    data = DATASETS[args.dataset]["load"]()

    pipeline_name = build_real_data_name(args.dataset, args.gen_func)

    pipeline = Pipeline(
        model_name=args.model,
        params_name=args.params,
        data=data,
        freq=DATASETS[args.dataset]["freq"],
        pipeline_name=pipeline_name,
        base_dir_name=pathlib.Path(__file__).parent.absolute(),
    )

    # kwargs could contain:
    #             scalers,
    #             scaling_levels,
    #             weighted_loss,
    #             norm_types,
    #             norm_modes,
    #             norm_affines,
    # e.g. kwargs = {"scalers": [StandardScaler()], "scaling_levels": ["per_time_series"]}
    kwargs = {}

    start_time = time.time()
    pipeline.run(
        save=True, test_percentage=0.25, params_generator_name=args.gen_func, with_scalers=args.with_scalers, **kwargs
    )
    end_time = time.time()

    print("Pipeline execution time: ", end_time - start_time)
    pipeline.summary()
