import pandas as pd

from pvlib import pvsystem
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.pvsystem import PVSystem
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS


class PVModel:
    def __init__(self, location: Location):
        self.location = location
        # module parameters for the Offgridtec Mono 20W 3-01-001560:
        self._MODULE_PARAMETERS = {
            'Name': 'Offgridtec_001560',
            'BIPV': 'ESG_glass',
            'Date': '04/05/2023',
            'celltype': 'monocrystalline',  # technology
            'Bifacial': True,  # True/False
            'T_NOCT': 45,
            'STC1': 1000,  # Standard Test Conditions power (w/m2)
            'STC2': "AM1.5 Spectrum",  # Standard Test Conditions power
            'STC3': 25,  # Standard Test Conditions power c (Temprature)
            # 'PTC': XXX, # PVUSA Test Condition power
            # 'A_c': XXX, #module cells area in m²
            'N_s': 36,  # number of cells in series
            'Array': "2 * 18",
            'Pmax': 20,  # Wp
            'I_sc_ref': 1.21,  # Short circuit current (ISC)
            'V_oc_ref': 22.3,  # Open circuit voltage (VOC)
            'I_mp_ref': 1.12,  # max. current (IMP)
            'V_mp_ref': 17.8,
            'MSV': 600,  # Maximum system voltage
            'alpha_sc': 0.0045,
            # short circuit current temperature coefficient in A/Δ°C          *** PV producers do not provice this data
            # 'beta_oc': -XXX, #open circuit voltage temperature coefficient in V/Δ°C
            'a_ref': 1.3825,
            # diode ideality factor                                                *** PV producers do not provice this data
            'I_L_ref': 1.21,
            # light or photogenerated current at reference condition in A         *** PV producers do not provice this data
            'I_o_ref': 883e-10,
            # diode saturation current at reference condition in A            *** PV producers do not provice this data
            'R_s': 0.9987,
            # series resistance in Ω                                                 *** PV producers do not provice this data
            'R_sh_ref': 3941.111,
            # shunt resistance at reference condition in Ω                      *** PV producers do not provice this data
            # 'Adjust': XXX, #adjustment to short circuit temperature coefficient in %
            # 'gamma_r': -XXX, #power temperature coefficient at reference condition in %/Δ°
            'Tolerance1': "-3% to 3%",  # Power tolerance
            'Tolerance2': "-5% to 5%",  # Elect. parameters tolerance
            'NOCT': 45,  # +/- 2 degree C
            'temp. coefficient Voc': -0.45,  # %/℃
            'temp. coefficient Isc': -0.45,  # %/℃
            'temp. coefficient P': -0.45,  # %/℃
        }
        self._INVERTER_PARAMS = pvsystem.retrieve_sam('cecinverter')['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
        self._pvsystem = PVSystem(surface_tilt=20, surface_azimuth=180,
                                  module_parameters=self._MODULE_PARAMETERS,
                                  inverter_parameters=self._INVERTER_PARAMS,
                                  temperature_model_parameters=TEMPERATURE_MODEL_PARAMETERS['sapm'][
                                      'open_rack_glass_glass'])
        self._modelchain = ModelChain(self._pvsystem, self.location, aoi_model='physical', spectral_model='no_loss')

    def get_module_params(self):
        return self._MODULE_PARAMETERS

    def get_inverter_params(self):
        return self._INVERTER_PARAMS

    def run_model(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame = None):
        if synthetic_df is not None:
            synthetic_df.index = real_df.head(synthetic_df.shape[0]).index
            df = synthetic_df
        else:
            df = real_df

        # Preparazione dei dati atmosferici
        weather = pd.DataFrame({
            'ghi': df['shortwave_radiation_instant (W/m²)'],  # Global Horizontal Irradiance
            'dhi': df['diffuse_radiation_instant (W/m²)'],  # Diffuse Horizontal Irradiance
            'dni': df['direct_normal_irradiance_instant (W/m²)'],  # Direct Normal Irradiance
            'temp_air': df['temperature_2m (°C)'],
            'wind_speed': df['wind_speed_10m (m/s)']
        }, index=df.index)

        # Calcolo della produzione energetica
        self._modelchain.run_model(weather)
        return pd.DataFrame(self._modelchain.results.dc)
