import pandas as pd
from tot.evaluation.metrics import ERROR_FUNCTIONS


# def calc_scaled_metric_abs(MAE_gt, MAE_t):
#     avg_t = np.mean(MAE_t)
#     avg_gt = np.mean(MAE_gt)
#     MAE_abs = avg_t/avg_gt
#     return MAE_abs

# def calc_scaled_metric_rel(MAE_gt, MAE_t): # move to evaluation itself
#     scaled_err_individual = np.divide(MAE_t, MAE_gt)
#     MAE_rel = np.mean(scaled_err_individual)
#     return MAE_rel

def calc_scaled_metric_avg(metrics_per_series):
        metrics_per_dataset = metrics_per_series.mean(axis=1)
        return metrics_per_dataset


def  calc_per_series_metrics(df, metrics):
    metrics_df = pd.concat(
        [
            df.groupby('ID').apply(
                lambda x: ERROR_FUNCTIONS[metric](
                    predictions=df["y_ideal"].values,
                    truth=df["y"].values,
                    freq='H',
                )
            )
            for metric in metrics
        ],
        axis=1,
        keys=metrics,
    )
    return metrics_df

def cal_dataset_metrics(df, metrics):
        metrics_per_series = calc_per_series_metrics(df=df, metrics= metrics)
        metrics_abs = calc_scaled_metric_avg(metrics_per_series)
        return metrics_abs

