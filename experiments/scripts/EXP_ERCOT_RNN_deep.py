from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler

from experiments.pipeline_experiment import run
from experiments.utils import LogTransformer, load_ERCOT
from tot.models.models_darts import DartsForecastingModel
from darts.models.forecasting.rnn_model import RNNModel

PLOT = False
DIR_NAME = "ERCOT_RNN_deep"
FREQ = "H"
MODEL = DartsForecastingModel
MODEL_PARAMS = {
    "model": RNNModel,
    "input_chunk_length": 24,
    'hidden_dim':25,
    'n_rnn_layers': 20,
    'batch_size':128,
    'n_epochs':20,
    'random_state':0,
    'training_length':24,
    'force_reset':True,
    'lags': 24,
    'n_forecasts': 1,
    '_data_params':{},
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
    df=load_ERCOT(),
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
