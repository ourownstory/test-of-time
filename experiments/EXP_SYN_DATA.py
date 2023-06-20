import multiprocessing
import pathlib as pathlib
import time

import pandas as pd

from experiments.pipeline.helpers.arg_parsers import get_synth_data_arg_parser
from experiments.pipeline.helpers.misc import build_synth_data_name
from experiments.pipeline.pipeline import Pipeline
from experiments.pipeline.synthetic_data.base_generator import generate

if __name__ == "__main__":
    args = get_synth_data_arg_parser()

    # Freezing support for multiprocessing
    multiprocessing.freeze_support()

    # post-processing args
    data_n_ts_groups = [int(i) for i in args.data_n_ts_groups.split(",")]
    data_offset_per_group = [int(i) for i in args.data_offset_per_group.split(",")]
    data_amplitude_per_group = [int(i) for i in args.data_amplitude_per_group.split(",")]
    data_trend_gradient_per_group = (
        [float(i) for i in args.data_trend_gradient_per_group.split(",")]
        if args.data_trend_gradient_per_group is not None
        else None
    )
    proportion_break = [int(i) for i in args.proportion_break.split(",")] if args.proportion_break is not None else None

    freq = "H"
    series_length = 24 * 7 * 15
    series_start = pd.to_datetime("2011-01-01 01:00:00")

    df = generate(
        series_start=series_start,
        series_length=series_length,
        data_trend_gradient_per_group=data_trend_gradient_per_group,
        data_func=args.data_func,
        n_ts_groups=data_n_ts_groups,
        offset_per_group=data_offset_per_group,
        amplitude_per_group=data_amplitude_per_group,
        proportion_break=proportion_break,
        freq=freq,
    )

    pipeline_name = build_synth_data_name(
        args.data_func,
        args.params,
        data_n_ts_groups,
        data_amplitude_per_group,
        data_offset_per_group,
        data_trend_gradient_per_group,
        proportion_break,
    )

    pipeline = Pipeline(
        model_name=args.model,
        params_name=args.params,
        data=df,
        freq=freq,
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
