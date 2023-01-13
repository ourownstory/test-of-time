import logging

# make version number accessible
from ._version import __version__  # noqa: F401
from .benchmark import (  # noqa: F401
    CrossValidationBenchmark,
    ManualBenchmark,
    ManualCVBenchmark,
    SimpleBenchmark,
)

# make classes available upon package import
from .dataset import Dataset  # noqa: F401
from .experiment import (  # noqa: F401
    # CrossValidationExperiment,  # noqa: F401
    SimpleExperiment,  # noqa: F401
)  # noqa: F401
from .models import (  # noqa: F401
    NaiveModel,  # noqa: F401
    NeuralProphetModel,  # noqa: F401
    ProphetModel,  # noqa: F401
    SeasonalNaiveModel,  # noqa: F401
)

log = logging.getLogger("dv")
log.setLevel("INFO")

c_handler = logging.StreamHandler()
c_format = logging.Formatter(
    "%(levelname)s - (%(name)s.%(funcName)s) - %(message)s"
)
c_handler.setFormatter(c_format)
log.addHandler(c_handler)

logging.captureWarnings(True)
warnings_log = logging.getLogger("py.warnings")
warnings_log.addHandler(c_handler)

write_log_file = False
if write_log_file:
    f_handler = logging.FileHandler("logs.log", "w+")
    # f_handler.setLevel("ERROR")
    f_format = logging.Formatter(
        "%(asctime)s; %(levelname)s; %(name)s; %(funcName)s; %(message)s"
    )
    f_handler.setFormatter(f_format)
    log.addHandler(f_handler)
    warnings_log.addHandler(f_handler)
