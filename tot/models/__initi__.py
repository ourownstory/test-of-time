# make classes available upon package import
from .models_naive import NaiveModel, SeasonalNaiveModel  # noqa: F401 to evade flake8
from .models_neuralprophet import NeuralProphetModel, TorchProphetModel  # noqa: F401 to evade flake8
from .models_prophet import ProphetModel  # noqa: F401 to evade flake8
from .models_simple import LinearRegressionModel  # noqa: F401 to evade flake8
