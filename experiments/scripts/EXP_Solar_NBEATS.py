from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler

from experiments.pipeline_experiment import run
from experiments.utils import LogTransformer, load_Solar
from tot.models.models_darts import DartsForecastingModel
from darts.models.forecasting.nbeats import NBEATSModel
from darts.utils.losses import SmapeLoss


PLOT = False
DIR_NAME = "Solar_NBEATS"
FREQ = "10min"
MODEL = DartsForecastingModel
MODEL_PARAMS = {
    "model": NBEATSModel,
    "n_forecasts": 1,
    "output_chunk_length": 1,
    "input_chunk_length":24,
    "n_lags":24,
    'num_stacks':20,
    'num_blocks':1,
    'num_layers':2,
    'layer_widths':136,
    'expansion_coefficient_dim':11,
    'loss_fn':SmapeLoss(),
    'batch_size':1024,
    'optimizer_kwargs':{'lr':0.001},
    'pl_trainer_kwargs':{'accelerator':'gpu', 'gpus':-1, 'auto_select_gpus': True},
    "_data_params": {},
}

scalers = [
    StandardScaler(),
    MinMaxScaler(feature_range=(-1, 1)),
    MinMaxScaler(feature_range=(0, 1)),
    RobustScaler(quantile_range=(25, 75)),
    PowerTransformer(method="box-cox", standardize=True),
    PowerTransformer(method="yeo-johnson", standardize=True),
    QuantileTransformer(output_distribution="normal"),
    LogTransformer(),
]

run(
    dir_name=DIR_NAME,
    save=True,
    df=load_Solar(n_ids=10, n_samples=40000),
    df_name=DIR_NAME,
    freq=FREQ,
    model_class=MODEL,
    model_params=MODEL_PARAMS,
    scalers=scalers,
    scaling_levels="default",
    reweight_loss=True,
    metrics=["MAE", "RMSE", "MAPE", "MASE"],
    test_percentage=0.25,
    plot=PLOT,
)
