import os
import pathlib

import numpy as np
import optuna
import pandas as pd
from neuralprophet import NeuralProphet, set_random_seed

# find reference here: https://unit8co.github.io/darts/examples/17-hyperparameter-optimization.html#


### use AirPassenger dataset ###
DIR = pathlib.Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(DIR, "tests", "test-data")
AIR_FILE = os.path.join(DATA_DIR, "air_passengers.csv")
df = pd.read_csv(AIR_FILE)


# convert to float32
df["y"] = df.loc[:, "y"].astype(np.float32)

# Split in train/val/test
VAL_LEN = 58  # 30 months = 40%
df_train = df.iloc[:-VAL_LEN]
df_test = df.iloc[-VAL_LEN:]


df_train.plot()
# plt.show()


def objective(trial):

    # hyperparameters
    n_lags = trial.suggest_int("n_lags", 6, 12, log=True)
    num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 10)
    d_hidden = trial.suggest_int("d_hidden", 5, 10)

    # reproducibility
    # torch.manual_seed(42)
    set_random_seed(42)

    # build and train the model with these hyper-parameters:
    model = NeuralProphet(
        n_lags=n_lags,
        n_forecasts=3,
        # learning_rate=learning_rate,
        # batch_size=64,
        epochs=100,
        # ar_reg = ar_reg,
        num_hidden_layers=num_hidden_layers,
        d_hidden=d_hidden,
        # yearly_seasonality = yearly_seasonality,
        # daily_seasonality = daily_seasonality,
        yearly_seasonality=True,
        seasonality_mode="multiplicative",
    )

    # train the model
    metrics = model.fit(
        df=df_train,
        # validation_df=val,
        early_stopping=True,
        checkpointing=True,
        freq="MS",
    )

    # Evaluate how good it is on the validation set
    preds = model.predict(df=df_test)

    # test-of-time evaluation procedure
    n_yhats_test = sum(["yhat" in colname for colname in preds.columns])
    metric_test_list = []
    preds = preds.fillna(value=np.nan)

    for x in range(1, n_yhats_test + 1):
        metric_test_list.append(
            _calc_mape(
                predictions=preds["yhat{}".format(x)].values,
                truth=df_test["y"].values,
            )
        )
    mape = np.nanmean(metric_test_list, dtype="float32")

    return mape


def print_callback(study, trial):
    # for convenience, print some optimization trials information
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")


def _calc_smape(
    predictions: np.ndarray,
    truth: np.ndarray,
    # truth_train: np.ndarray = None,
) -> float:
    """Calculates SMAPE error."""
    absolute_error = np.abs(np.subtract(truth, predictions))
    absolute_sum = np.abs(truth) + np.abs(predictions)
    error_relative_sym = np.divide(absolute_error, absolute_sum)
    # return 100.0 * np.nanmean(error_relative_sym, dtype="float32")
    return error_relative_sym


def _calc_mape(
    predictions: np.ndarray,
    truth: np.ndarray,
    truth_train: np.ndarray = None,
) -> float:
    """Calculates MAPE error."""
    error = np.subtract(truth, predictions)
    error_relative = np.abs(np.divide(error, truth))
    return 100.0 * np.nanmean(error_relative, dtype="float32")



# execute script
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30, callbacks=[print_callback])


