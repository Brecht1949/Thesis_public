import pvlib
import pandas as pd
from pathlib import Path


"""

REMARK: Make sure DWD satellite is already in 15 min weighted when passing to physical model, will crash otherwise.
REMARK 2: IRRADIANCE VALUES ARE PRECEDING!!! SHOULD PROBABLY BE PAST AS UPCOMING.
"""
### Put as class in order to be able to create one instance to multithread.
class PhysicalModelPVLib:
    def __init__(self, latitude, longitude, system_size_w, orientation_deg, tilt_deg):
        self.latitude = latitude
        self.longitude = longitude
        self.system_size_w = system_size_w
        self.orientation_deg = orientation_deg
        self.tilt_deg = tilt_deg

        ### Set defaults, just use some typical params
        self.noct = 45
        self.eff_STC = 0.2
        self.temp_coeff = -0.004
        self.transmittance_absorptance = 0.9
        self.array_height = 1 ### 2 for taller buildings, 1 for smaller buildings. (Amount of stories of approx 3m.)
        self.mount_standoff = 4

        ### Fitting factor
        self.fitting_factor = 1 ### SET TO ONE FOR NOW#0.8 ### It might be better however to adjust through inverter efficiency or something. Have to look at overall performance over multiple houses.


    def irradiance_poa(self, weather_df):
        times = weather_df.index
        if times.tz is not None:
            tz = times.tz
        else:
            raise RuntimeError("Weather data index must be timezone-aware. Please ensure the index has a timezone.")
        location = pvlib.location.Location(self.latitude, self.longitude, tz=tz)
        solar_position_array = location.get_solarposition(times)
        solar_elevation_array = 90 - solar_position_array["apparent_zenith"] ### Solar elevation in degrees. Like in PV_forecast updated. Is for nans at night.
        irradiance_array = pvlib.irradiance.get_total_irradiance(
            surface_tilt=self.tilt_deg,
            surface_azimuth=self.orientation_deg,
            solar_zenith=solar_position_array["apparent_zenith"],
            solar_azimuth=solar_position_array["azimuth"],
            dni=weather_df["direct_normal_irradiance"],
            ghi=weather_df["shortwave_radiation"],
            dhi=weather_df["diffuse_radiation"],
            albedo=0.15,
            dni_extra=pvlib.irradiance.get_extra_radiation(times), ### Required for haydavies and perez. Not for isotropic. Models how much diffuse radiation landas on panel.
            model="haydavies",
        )
        return irradiance_array, solar_elevation_array
    
    def temperature_and_efficiency(self, irradiance_array, weather_df): ### For panel
        ### Check that they all have the same index, otherwise raise
        if not irradiance_array.index.equals(weather_df.index):
            raise ValueError("The indices of irradiance_array and weather_df do not match. Please ensure they have the same timestamps.")
        Temperature_array = pvlib.temperature.noct_sam(
            poa_global=irradiance_array["poa_global"],
            temp_air=weather_df["temperature_2m"],
            wind_speed=weather_df["wind_speed_10m"] / 3.6,  ### Convert from km/h to m/s, SHOULD CHECK IF STILL CORRECT.
            noct=self.noct, ### Use the defaults
            module_efficiency=self.eff_STC,
            transmittance_absorptance=self.transmittance_absorptance,
            array_height=self.array_height,
            mount_standoff=self.mount_standoff
        )
        Efficiency_array = 1 * (1 + self.temp_coeff * (Temperature_array - 25)) ### NOT TIMES SELF.EFF_STC HERE, we're usign system size later on.
        return Efficiency_array
    
    def DC_power_output(self, irradiance_array, Efficiency_array):
        power_output_panels = irradiance_array["poa_global"] * (self.system_size_w / 1000) * Efficiency_array / 1000 ### /1000 because STC 1000W/m^2 and put in kW
        return power_output_panels
    
    def DC_energy_yield(self, power_output_array):
        ### Make sure dt_s is 15min, otherwise raise
        dt_s = (power_output_array.index[1] - power_output_array.index[0]).total_seconds()
        if dt_s != 900:
            raise ValueError(f"Make sure time step is 15 minutes. First two timestamps are {power_output_array.index[0]} and {power_output_array.index[1]}, with a difference of {dt_s/60} minutes.")
        energy_kwh_15min = power_output_array * 0.25
        return energy_kwh_15min
    
    def inverter_power_output(self, power_output_array):
        inverter_max_dc_power = 1.2 * (self.system_size_w / 1000) ### Will do this for now
        inverter_power_output = pvlib.inverter.pvwatts(pdc=power_output_array, pdc0=inverter_max_dc_power, eta_inv_nom=0.971) ### Of my inverter :)
        return inverter_power_output

    def inverter_energy_yield(self, inverter_power_output_array):
        dt_s = (inverter_power_output_array.index[1] - inverter_power_output_array.index[0]).total_seconds()
        if dt_s != 900:
            raise ValueError(f"Make sure time step is 15 minutes. First two timestamps are {inverter_power_output_array.index[0]} and {inverter_power_output_array.index[1]}, with a difference of {dt_s/60} minutes.")
        energy_kwh_15min = inverter_power_output_array * 0.25
        return energy_kwh_15min
    

    def fitting_factor_power(self, inverter_power_output):
        return inverter_power_output * self.fitting_factor
    
    def fitting_factor_energy(self, fitting_factor_power_array):
        dt_s = (fitting_factor_power_array.index[1] - fitting_factor_power_array.index[0]).total_seconds()
        if dt_s != 900:
            raise ValueError(f"Make sure time step is 15 minutes. First two timestamps are {fitting_factor_power_array.index[0]} and {fitting_factor_power_array.index[1]}, with a difference of {dt_s/60} minutes.")
        fitting_factor_energy = fitting_factor_power_array * 0.25
        return fitting_factor_energy



    def clear_sky_model(self, weather_df): ### Basically the exact same but will use clear sky model for irradiance
        clear_weather_df = weather_df.copy()
        times = clear_weather_df.index ### Check not needed, will crash on irradiance_poa already

        location = pvlib.location.Location(self.latitude, self.longitude, tz=times.tz)
        clear_sky_df = location.get_clearsky(times) ### Returns df w ghi, dni, dhi
        clear_weather_df["shortwave_radiation"] = clear_sky_df["ghi"]
        clear_weather_df["diffuse_radiation"] = clear_sky_df["dhi"]
        clear_weather_df["direct_normal_irradiance"] = clear_sky_df["dni"]

        ### Rest is same
        irradiance_array, solar_elevation_array = self.irradiance_poa(clear_weather_df)
        Efficiency_array = self.temperature_and_efficiency(irradiance_array, clear_weather_df)
        power_output_array = self.DC_power_output(irradiance_array, Efficiency_array)
        DC_energy_yield_array = self.DC_energy_yield(power_output_array)
        inverter_power_output_array = self.inverter_power_output(power_output_array)
        inverter_energy_yield_array = self.inverter_energy_yield(inverter_power_output_array)
        fitting_factor_power_array = self.fitting_factor_power(inverter_power_output_array)
        fitting_factor_energy_array = self.fitting_factor_energy(fitting_factor_power_array)

        output_df = pd.DataFrame(index=weather_df.index)
        output_df["clear_sky_fitting_factor_power"] = fitting_factor_power_array
        output_df["clear_sky_fitting_factor_energy"] = fitting_factor_energy_array

        return output_df

    
    def run(self, weather_df):
        try: 
            irradiance_array, solar_elevation_array = self.irradiance_poa(weather_df)
            Efficiency_array = self.temperature_and_efficiency(irradiance_array, weather_df)
            power_output_array = self.DC_power_output(irradiance_array, Efficiency_array)
            DC_energy_yield_array = self.DC_energy_yield(power_output_array)
            inverter_power_output_array = self.inverter_power_output(power_output_array)
            inverter_energy_yield_array = self.inverter_energy_yield(inverter_power_output_array)
            fitting_factor_power_array = self.fitting_factor_power(inverter_power_output_array)
            fitting_factor_energy_array = self.fitting_factor_energy(fitting_factor_power_array)

            clear_sky_df = self.clear_sky_model(weather_df)
        except Exception as e:
            raise RuntimeError(f"Physics model failed:\n{e}")
        
        output_df = pd.DataFrame(index=weather_df.index)
        #output_df["poa_global"] = irradiance_array["poa_global"]
        #output_df["poa_direct"] = irradiance_array["poa_direct"]
        #output_df["poa_diffuse"] = irradiance_array["poa_diffuse"]
        #output_df["Efficiency"] = Efficiency_array
        #output_df["DC_power_output"] = power_output_array
        #output_df["DC_energy_yield"] = DC_energy_yield_array
        #output_df["inverter_power_output"] = inverter_power_output_array
        #output_df["inverter_energy_yield"] = inverter_energy_yield_array

        output_df["solar_elevation"] = solar_elevation_array
        output_df["fitting_factor_power"] = fitting_factor_power_array ### For now, only keep these
        output_df["fitting_factor_energy"] = fitting_factor_energy_array

        output_df["clear_sky_fitting_factor_power"] = clear_sky_df["clear_sky_fitting_factor_power"] ### The power that would be generated if the sky was ideal
        output_df["clear_sky_fitting_factor_energy"] = clear_sky_df["clear_sky_fitting_factor_energy"]

        return output_df
    



if __name__ == "__main__":
    ### Example
    latitude = 51.0
    longitude = 4.0
    system_size_w = 3000
    orientation_deg = 180
    tilt_deg = 30

    physical_model = PhysicalModelPVLib(latitude, longitude, system_size_w, orientation_deg, tilt_deg)
    FILE = Path('XXX')
    weather_df = pd.read_parquet(FILE)
    output_df = physical_model.run(weather_df)
    print(output_df.head())