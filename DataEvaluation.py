"""
Different methods for data evaluation
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

METRICS = ["RMSE", "MSE"]


def get_metrics(real_df: pd.DataFrame, synth_df: pd.DataFrame, params: list):
    # Estraggo gli anni presenti nel dataset
    years = real_df.index.year.unique()
    cut_df = []

    # Divido il dataframe per anno
    for year in years:
        cut_df.append(real_df[real_df.index.year == year])

    metrics_val = {}

    for metric in METRICS:
        metrics_val[metric] = {}
        for parameter in params:
            metrics_val[metric][parameter] = []

    for metric in METRICS:
        for parameter in params:
            tmp = metrics_val[metric][parameter]
            for year in cut_df:
                tmp.append(evaluate(year.head(synth_df.shape[0])[parameter].reset_index(drop=True), synth_df[parameter],
                                    metric))
                break  # Rimuovere per fare la media sugli anni del dataset

            metrics_val[metric][parameter] = np.mean(np.array(tmp))

    return pd.DataFrame.from_dict(metrics_val, orient='columns')


def evaluate(real, synth, metric):
    if metric == "SMAPE":
        return smape(real, synth)
    elif metric == "MAPE":
        return mape(real, synth)
    elif metric == "RMSE":
        return rmse(real, synth)
    elif metric == "MSE":
        return mse(real, synth)


def prepare_data(real, synth):
    # Check if is an array, if not fix it
    if not all([isinstance(real, np.ndarray),
                isinstance(synth, np.ndarray)]):
        real, synth = np.array(real), np.array(synth)

    return real, synth


def rmse(real, synth):
    real, synth = prepare_data(real, synth)
    rmse_val = mean_squared_error(y_true=real, y_pred=synth, squared=False)

    return rmse_val


def mse(real, synth):
    real, synth = prepare_data(real, synth)
    mse_val = mean_squared_error(y_true=real, y_pred=synth, squared=True)

    return mse_val


def mape(real, synth):
    real, synth = prepare_data(real, synth)
    mape_val = round(np.mean(np.abs(synth - real) / (np.abs(synth))) * 100, 3)
    return mape_val


def smape(real, synth):
    real, synth = prepare_data(real, synth)
    smape_val = round(np.mean(np.abs(synth - real) / ((np.abs(synth)) + np.abs(real)) / 2) * 100, 3)
    return smape_val
