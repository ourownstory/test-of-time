from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Scaler:
    transformer: object

    def __post_init__(self):
        is_transformer_valid = (
            callable(getattr(self.transformer, "fit", None))
            and callable(getattr(self.transformer, "transform", None))
            and callable(getattr(self.transformer, "inverse_transform", None))
        )
        if not is_transformer_valid:
            raise ValueError(
                "Transformer provided to the Scaler must implement fit, transform and inverse_transform " "methods"
            )

    def _transform(self, df: pd.DataFrame, fit):
        values = []
        for df_name, df_i in df.groupby("ID"):
            aux = df_i.copy(deep=True)
            values.append(aux["y"])

        values = np.transpose(values)
        if fit:
            self.transformer.fit(values)
        values_transformed = np.transpose(self.transformer.transform(values))

        result = pd.DataFrame()
        for (_, df_i), values_row in zip(df.groupby("ID"), values_transformed):
            group = df_i.copy(deep=True)
            group.loc[:, "y"] = values_row
            result = pd.concat((result, group))

        return result

    def _inverse_transform(self, df: pd.DataFrame):
        values_pred = []
        values_true = []
        for df_name, df_i in df.groupby("ID"):
            values_pred.append(df_i["yhat1"])
            values_true.append(df_i["y"])

        values_pred = np.transpose(values_pred)
        values_pred_transformed = np.transpose(self.transformer.inverse_transform(values_pred))
        values_true = np.transpose(values_true)
        values_true_transformed = np.transpose(self.transformer.inverse_transform(values_true))

        result = pd.DataFrame()
        for (_, df_i), pred_row, true_row in zip(df.groupby("ID"), values_pred_transformed, values_true_transformed):
            group = df_i.copy(deep=True)
            group.loc[:, "yhat1"] = pred_row
            group.loc[:, "y"] = true_row
            result = pd.concat((result, group))

        return result

    def transform(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        return self._transform(df_train, fit=True), self._transform(df_test, fit=False)

    def inverse_transform(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        return self._inverse_transform(df_train), self._inverse_transform(df_test)
