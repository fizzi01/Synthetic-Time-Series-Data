import calendar
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import List
from datetime import datetime
from joblib import dump, load

"""
This class contains the preprocessing and postprocessing functions for training and
visualizing syntethic data.
"""


class DataProcessor:

    def __init__(self, seq_lenght: int = 24, numerical_columns: List[str] = None,
                 categorical_columns: List[str] = None):
        self.scaler = None
        self.seq_lenght = seq_lenght
        self.numerical_col = numerical_columns
        self.categorical_col = categorical_columns

    def fit(self, data: np.ndarray):
        # Fit scaler to data
        self.scaler = MinMaxScaler().fit(data)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        # Normilizing data
        ori_data = self.scaler.transform(data)
        return ori_data

    def preprocess(self, data: np.ndarray):
        """Required for TimeGAN"""
        pass

    def load_metadata(self, path: str) -> 'DataProcessor':
        """
        Load processor metadata
        :param path: file path
        :return: DataProcessor
        """
        processor = load(path)
        self.scaler = processor['scaler']
        self.seq_lenght = processor['sequence_length']
        self.numerical_col = processor['measurement_cols_metadata']
        self.categorical_col = processor['categorical_cols_metadata']
        return self

    def save_metadata(self, path: str) -> "DataProcessor":
        """
        Saves processor scaler as file
        :return:
        """
        if self.scaler is not None:
            dump({
                "scaler": self.scaler,
                "measurement_cols_metadata": self.numerical_col,
                "categorical_cols_metadata": self.categorical_col,
                "sequence_length": self.seq_lenght,
            }, path)
        else:
            raise Exception("Missing scaler.")

        return self

    def reverse_transform(self, synth_df: pd.DataFrame) -> pd.DataFrame:
        # Reversing data
        synth_df_tmp = self.scaler.inverse_transform(synth_df[self.numerical_col].values)
        synth_df_cat = pd.DataFrame(synth_df[self.categorical_col], columns=self.categorical_col).reset_index(drop=True)
        synth_df_num = pd.DataFrame(synth_df_tmp, columns=self.numerical_col).reset_index(drop=True)
        synth_df = pd.concat([synth_df_num, synth_df_cat], axis=1)

        # Ordering by month and index
        if "Mese" in self.categorical_col:
            synth_df["Mese"] = synth_df["Mese"].apply(lambda x: datetime.strptime(x, '%B').month)
            synth_df = synth_df.rename_axis("index").sort_values(by=["Mese", "index"])
            synth_df["Mese"] = synth_df["Mese"].apply(lambda x: calendar.month_name[x])
            # Resetting index order
            synth_df = synth_df.reset_index(drop=True)

        # Remove 'temperature_2m (°C)' from columns clipping
        cols_to_clip = [col for col in self.numerical_col if col != 'temperature_2m (°C)']

        # Cleaning negative values (out bounds)
        synth_df[cols_to_clip] = synth_df[cols_to_clip].clip(lower=0)

        return synth_df
