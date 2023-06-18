import logging

from tot.models import NaiveModel, NeuralProphetModel, SeasonalNaiveModel, TorchProphetModel
from tot.models.models_darts import DartsForecastingModel

__all__ = ["get_tot_model_class", "SUPPORTED_MODELS"]

log = logging.getLogger("experiments")

TOT_MODELS = {
    "NeuralProphetModel": NeuralProphetModel,
    "TorchProphetModel": TorchProphetModel,
    "LightGBMModel": DartsForecastingModel,
    "RNNModel": DartsForecastingModel,
    "TransformerModel": DartsForecastingModel,
    "NaiveModel": NaiveModel,
    "SeasonalNaiveModel": SeasonalNaiveModel,
}

SUPPORTED_MODELS = list(TOT_MODELS.keys())


def get_tot_model_class(model_name):
    try:
        return TOT_MODELS[model_name]
    except KeyError as e:
        log.error(e, f"Model not supported. Supported models: {SUPPORTED_MODELS}")
