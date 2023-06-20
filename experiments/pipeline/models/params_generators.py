import logging

from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

__all__ = [
    "gen_model_and_params_default",
    "gen_model_and_params_pytorch_batch_norm",
    "gen_model_and_params_norm",  # norm layers all
    "gen_model_and_params_none",
    "gen_model_and_params_scalers",
    "gen_model_and_params_scalers_reweighting",
]

from experiments.pipeline.helpers.scalers import LogTransformer
from tot.models import NeuralProphetModel

log = logging.getLogger("experiments")


def validate_param_generator(model_class, param_gen_name):
    default = ["gen_model_and_params_none", "gen_model_and_params_scalers", "gen_model_and_params_default"]
    allowed = default
    if model_class == NeuralProphetModel:
        allowed = __all__
    if param_gen_name not in allowed:
        raise ValueError(f"Invalid gen_fun for {model_class.__name__}. Allowed generators {allowed}")


SCALERS_DEFAULT = [
    None,
    StandardScaler(),
    # MinMaxScaler(feature_range=(-0.5, 0.5)),
    MinMaxScaler(feature_range=(0, 1)),
    RobustScaler(quantile_range=(5, 95)),
    # ShiftedBoxCoxTransformer(),
    PowerTransformer(method="yeo-johnson", standardize=True),
    QuantileTransformer(output_distribution="normal"),
    LogTransformer(),
]

SCALING_LEVELS_DEFAULT = ["per_time_series", "per_dataset"]
WEIGHTED_LOSS_DEFAULT = ["none", "std"]  # "std*avg", "std"
NORM_MODES_DEFAULT = ["revin", "pytorch"]
NORM_TYPES_DEFAULT = ["batch", "instance"]
NORM_AFFINES_DEFAULT = [True, False]


def gen_model_and_params(
    common_params,
    model_class,
    with_scalers,
    scalers=None,
    scaling_levels=None,
    weighted_loss=None,
    norm_types=None,
    norm_modes=None,
    norm_affines=None,
):
    if scalers is None:
        scalers = [None]
    if scaling_levels is None:
        scaling_levels = ["per_time_series"]
    if norm_affines is None:
        norm_affines = [False]
    if norm_modes is None:
        norm_modes = [None]
    if norm_types is None:
        norm_types = [None]
    if weighted_loss is None:
        weighted_loss = [None]

    if with_scalers:
        scalers = SCALERS_DEFAULT
        scaling_levels = SCALING_LEVELS_DEFAULT

    model_classes_and_params = [(model_class, common_params)]
    for norm_type in norm_types:
        for norm_mode in norm_modes:
            if norm_mode == "pytorch" and norm_type == "instance":
                continue
            for norm_affine in norm_affines:
                for scaler in scalers:
                    if scaler is not None:
                        for scaling_level in scaling_levels:
                            if scaling_level == "per_time_series":
                                for weighting in weighted_loss:
                                    params = common_params.copy()
                                    params.update(
                                        {
                                            "scaler": scaler,
                                            "scaling_level": scaling_level,
                                            "weighted_loss": weighting,
                                            "norm_mode": norm_mode,
                                            "norm_type": norm_type,
                                            "norm_affine": norm_affine,
                                        }
                                    )
                                    model_classes_and_params.append((model_class, params))
                            else:
                                params = common_params.copy()
                                params.update(
                                    {
                                        "scaler": scaler,
                                        "scaling_level": scaling_level,
                                        "norm_mode": norm_mode,
                                        "norm_type": norm_type,
                                        "norm_affine": norm_affine,
                                    }
                                )
                                model_classes_and_params.append((model_class, params))
                    else:
                        if norm_mode is None:
                            continue
                        params = common_params.copy()
                        params.update(
                            {
                                "norm_mode": norm_mode,
                                "norm_type": norm_type,
                                "norm_affine": norm_affine,
                            }
                        )
                        model_classes_and_params.append((model_class, params))

    if any(x in str(model_classes_and_params[0][0]) for x in ["NeuralProphetModel", "TorchProphetModel"]):
        model_classes_and_params[0][1].update({"learning_rate": 0.03})

    return model_classes_and_params


def gen_model_and_params_default(common_params, model_class, with_scalers, **kwargs):
    log.warning("Using gen_model_and_params_default. Skipping params check - might result in unexpected behavior.")
    return gen_model_and_params(
        common_params,
        model_class,
        with_scalers=with_scalers,
        **kwargs,
    )


def gen_model_and_params_scalers(common_params, model_class, with_scalers, **kwargs):
    log.info("Using gen_model_and_params_scalers")
    return gen_model_and_params(
        common_params,
        model_class,
        with_scalers=True,
        **kwargs,
    )


def gen_model_and_params_scalers_reweighting(common_params, model_class, with_scalers, **kwargs):
    log.info("Using gen_model_and_params_scalers")
    kwargs.pop("weighted_loss", None)
    return gen_model_and_params(
        common_params,
        model_class,
        with_scalers=True,
        weighted_loss=WEIGHTED_LOSS_DEFAULT,
        **kwargs,
    )


def gen_model_and_params_pytorch_batch_norm(common_params, model_class, with_scalers, **kwargs):
    print("Using gen_model_and_params_pytorch_batch_norm")

    kwargs.pop("norm_types", None)
    kwargs.pop("norm_modes", None)
    kwargs.pop("norm_affines", None)

    return gen_model_and_params(
        common_params,
        model_class,
        with_scalers,
        norm_types=["batch"],
        norm_modes=["pytorch"],
        norm_affines=[True, False],
        **kwargs,
    )


def gen_model_and_params_norm(common_params, model_class, with_scalers, **kwargs):
    log.info("Using gen_model_and_params_norm")

    kwargs.pop("norm_types", None)
    kwargs.pop("norm_modes", None)
    kwargs.pop("norm_affines", None)

    return gen_model_and_params(
        common_params,
        model_class,
        with_scalers,
        norm_types=NORM_TYPES_DEFAULT,
        norm_modes=NORM_MODES_DEFAULT,
        norm_affines=NORM_AFFINES_DEFAULT,
        **kwargs,
    )


def gen_model_and_params_none(common_params, model_class, with_scalers, **kwargs):
    log.info("Using gen_model_and_params_none")
    model_classes_and_params = [(model_class, common_params)]
    return model_classes_and_params
