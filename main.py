import os.path
import sys
import pandas as pd

from tqdm import trange
from joblib import dump, load

from ydata_synthetic.synthesizers.timeseries import TimeSeriesSynthesizer
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

from tqdm import trange
import calendar
from datetime import datetime

import DataLoad as dl
import DataProcessor as dprc
import DataPlot as dp
import PvModel as pvm
from DataEvaluation import *

from pvlib import pvsystem
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.pvsystem import PVSystem
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

LOCATION = Location(40.3548, 18, 0, 0, 'Lecce')

BATCH = 128
LR = 0.001
LATENT_DIM = 24
EPOCHS = 850
SEQUENCE_LENGTH = 24
SAMPLE_LENGHT = 8

loader = dl.Data(data_path="data/training/open-meteo-Lecce.csv")

# Defining the categorical and numerical columns
categorical_cols = ["Mese"]
numerical_cols = [col for col in loader.data.columns if col not in categorical_cols]

model_args = ModelParameters(batch_size=BATCH,
                             lr=LR,
                             betas=(0.2, 0.9),
                             latent_dim=LATENT_DIM,
                             gp_lambda=2,
                             pac=1)

train_args = TrainParameters(epochs=EPOCHS,
                             sequence_length=SEQUENCE_LENGTH,
                             sample_length=SAMPLE_LENGHT,
                             rounds=1,
                             measurement_cols=numerical_cols)

# Training the DoppelGANger synthesizer
retrain = input('Retrain? (y/n) ').lower().replace(" ", "")
model_path = "model/modelv4"
processor_path = "model/metav4"

if retrain == 'n':
    if path.exists(model_path) and path.exists(processor_path):
        # Loading model
        model_dop_gan = TimeSeriesSynthesizer.load(model_path)

        # Loading processor
        processor = dprc.DataProcessor().load_metadata(processor_path)
    else:
        raise FileExistsError("Files not found")

else:
    # Preparing data for training
    mba_data = loader.prepare_data(num_col=numerical_cols, cat_col=categorical_cols)
    processor = loader.processor

    # Saving real data metadata
    loader.save(processor_path)

    model_dop_gan = TimeSeriesSynthesizer(modelname='doppelganger', model_parameters=model_args)
    model_dop_gan.fit(mba_data, train_args, num_cols=numerical_cols, cat_cols=categorical_cols)
    model_dop_gan.save(f"{model_path}")

synth_data = []
for _ in trange(1, file=sys.stdout, desc="Generating samples"):
    synth_data = model_dop_gan.sample(n_samples=int(samples_n))
    dump(synth_data, f"Data_samples_doppel.df")

synth_df = pd.concat(synth_data, axis=0)
synth_df = processor.reverse_transform(synth_df)
synth_df.to_csv("model_output_data.csv", index=False)

metrics_val = get_metrics(real_df=loader.data, synth_df=synth_df, params=processor.numerical_col)
metrics_val.to_csv("metrics.csv", index=False)

# dp.plty_plot_dataframes([synth_df],title="Synth DoppelGANger")

# Generating PV data

# PV module
module = pvm.PVModel(location=LOCATION)

energy_sample = module.run_model(data, synth_df)
energy_sample.to_csv("pv_outputd_data.csv", index=False)
# dp.plty_plot_dataframes([energy_sample], title="Syntethic Energy Production Data")
