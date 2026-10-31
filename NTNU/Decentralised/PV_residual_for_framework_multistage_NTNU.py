import sys
import pandas as pd
from pathlib import Path
import numpy as np
from dataclasses import dataclass
import joblib
from geopy.distance import geodesic
from termcolor import colored
import time

from Physical_model_pvlib_NTNU import PhysicalModelPVLib

"""
Rewritten version of Residual_searching_v4_virtual_house_parallel.py, but now all in class for usage in general framework. Input should thus basically be virtual house and timestep.
Doesn't include plotting or eval.
Code reqs:
    This code assumes the avvailability of the joblib files created by Residual_searching_v4_virtual_house_parallel.py.
"""

PV_HORIZON_STEPS = 48*4
PV_HORIZON_TAG = "48h"

SCRATCH_DIR = Path("/cluster/work/brechtg")
MARKET_MPC_DIR = SCRATCH_DIR / "Market_MPC"
PV_DATA_DIR = SCRATCH_DIR / "Fluvius PV"

PVOUTPUT_METADATA_FILE = PV_DATA_DIR / "PVOutput" / "PVOutput_metadata" / "PVOutput_metadata_cleaned.parquet"
PVOUTPUT_EXCLUDE_SYSTEMS_FOLDER = PV_DATA_DIR / "PVOutput"
VIRTUAL_HOUSES_SINGLE_RUN_FOLDER = PV_DATA_DIR / "Open-meteo Single Run" / "Weather_saved" / "Building_Flanders_houses" / "icon_seamless"
VIRTUAL_HOUSES_SATELLITE_FOLDER = PV_DATA_DIR / "Open-meteo Satellite" / "Satellite_weather_data" / "Flanders_buildings"

FLANDERS_BUILDINGS_ASSIGNED_FILE = MARKET_MPC_DIR / "Households_w_battery_dataset" / "random_sample_1_with_load_pv_and_bess_assigned.parquet"
PVOUTPUT_RESIDUALS_FOLDER = MARKET_MPC_DIR / "Overall" / f"PVOutput_Residuals_{PV_HORIZON_TAG}"
PVOUTPUT_RESIDUALS_PRECOMPUTE_CACHE_FOLDER = MARKET_MPC_DIR / "Overall" / f"PVOutput_Residuals_Precomputed_cache_{PV_HORIZON_TAG}"


@dataclass
class PV_residual_behaviour_params:
    CACHE_ALL_RESIDUALS_TO_RAM: bool = True
    IMMEDIATELY_FILTER_RESIDUALS: bool = True

    # Probably not needed:
    ONLY_FIRST_POINT_FOR_EVAL: bool = True
    SAVE_PVOUTPUT_RESIDUALS_CACHE_TO_FILE = True ### To joblib
    FORCE_REBUILD_CACHE = False


    ### Paths
    PVOutput_metadata_file: Path = PVOUTPUT_METADATA_FILE
    PVOutput_residuals_folder: Path = PVOUTPUT_RESIDUALS_FOLDER
    PVOutput_residuals_precompute_cache_folder: Path = PVOUTPUT_RESIDUALS_PRECOMPUTE_CACHE_FOLDER
    Flanders_buildings_assigned_file: Path = FLANDERS_BUILDINGS_ASSIGNED_FILE
    Virtual_houses_single_run_folder: Path = VIRTUAL_HOUSES_SINGLE_RUN_FOLDER
    Virtual_houses_satellite_folder: Path = VIRTUAL_HOUSES_SATELLITE_FOLDER
    Pvoutput_exclude_systems_folder: Path = PVOUTPUT_EXCLUDE_SYSTEMS_FOLDER

    ### Options
    LOOKBACK_DAYS: int = 30
    #AMOUNT_OF_SCENARIOS: int = 10 ### Will be defined in framework for reducing scenarios as time advances
    EXCLUDE_PVOUTPUT_SHADOWS: bool = True
    EXCLUDE_PVOUTPUT_CHOPPED: bool = True
    EXCLUDE_PVOUTPUT_EW_ORIENTED: bool = True

    USE_PV_RESIDUAL_SCENARIOS: bool = True ### Added later for avoiding cache



class Virtual_household_residual_searching:
    FLANDERS_BUILDINGS_CACHE = None ### Will be shared for every class inside each process!
    PVOUTPUT_METADATA_CACHE = None

    def __init__(self, params: PV_residual_behaviour_params, virtual_household: int):
        self.load_id = virtual_household
        self.params = params
        self.pvoutput_residuals_folder = params.PVOutput_residuals_folder

        if Virtual_household_residual_searching.FLANDERS_BUILDINGS_CACHE is None:
            Virtual_household_residual_searching.FLANDERS_BUILDINGS_CACHE = pd.read_parquet(params.Flanders_buildings_assigned_file).set_index("load_id")
        PV_params_virtual_household, self.virtual_house_battery_capacity = self.init_retrieve_flanders_household_metadata(load_id=virtual_household, flanders_buildings_df=Virtual_household_residual_searching.FLANDERS_BUILDINGS_CACHE)

        #PV_params_virtual_household, self.virtual_house_battery_capacity = self.init_retrieve_flanders_household_metadata(load_id=virtual_household, flanders_buildings_df=pd.read_parquet(params.Flanders_buildings_assigned_file).set_index("load_id"))
        self.PV_params_virtual_household = PV_params_virtual_household
        
        self.physical_model = PhysicalModelPVLib(**self.PV_params_virtual_household)
        self.df_weather = self.init_load_weather_data(params.Virtual_houses_single_run_folder / f"Weather_forecasts_saved_{virtual_household}.parquet", state="upcoming") ### Do not cache yet to save RAM
        #self.df_weather_with_physical = self.init_precompute_physical_forecast(self.df_weather) ### This is the eventual df used. No sense to cache raw weather_df
        ### Previous line, see explanation next to new function after last init function
        self.df_weather_with_physical_cache = {}

        if self.params.USE_PV_RESIDUAL_SCENARIOS: ### This is not enough
            if Virtual_household_residual_searching.PVOUTPUT_METADATA_CACHE is None:
                Virtual_household_residual_searching.PVOUTPUT_METADATA_CACHE = pd.read_parquet(params.PVOutput_metadata_file)
            self.pvoutput_metadata_df = Virtual_household_residual_searching.PVOUTPUT_METADATA_CACHE

            #self.pvoutput_metadata_df = pd.read_parquet(params.PVOutput_metadata_file)        
            self.virtual_houses_satellite_df = pd.read_parquet(params.Virtual_houses_satellite_folder / f"load_id_{virtual_household}.0_satellite_weather.parquet")
            self.satellite_power_array = self.init_precompute_satellite_power(self.df_weather)

            self.nearby_houses = self.get_nearby_pvoutput_houses(radius_km=30) ### Moves this here, stays the same for each virtual house
            ### Now change the caching. Only cache the residuals for the nearby houses
            self.pvout_residuals_cache = self.init_cache_nearby_pvoutput_houses(params.PVOutput_residuals_precompute_cache_folder, params.IMMEDIATELY_FILTER_RESIDUALS)
        else:
            self.pvoutput_metadata_df = None
            self.virtual_houses_satellite_df = None
            self.satellite_power_array = None
            self.nearby_houses = {} ### Easiest
            self.pvout_residuals_cache = None



    def init_cache_nearby_pvoutput_houses(self, pvoutput_residuals_precompute_cache_folder: Path, immediately_filter_residuals: bool):
        pvoutput_residuals_cache = {}
        
        length_nearby_houses = len(self.nearby_houses)
        print(f"Initializing cache for {length_nearby_houses} nearby PVOutput houses...")
        for i, nearby_sid in enumerate(self.nearby_houses.keys(), start=1):
            print(f"\rProcessing nearby PVOutput house {i}/{length_nearby_houses} cache", end="")
            nearby_sid_key = str(nearby_sid)

            if immediately_filter_residuals:
                cache_file_path = pvoutput_residuals_precompute_cache_folder / f"Residuals_cache_system_{nearby_sid_key}_filtered.joblib"
                #print(f"File path: {cache_file_path}")
            else:
                raise NotImplementedError("Currently only immediately_filter_residuals=True is implemented")
            
            if not cache_file_path.exists():
                raise FileNotFoundError(f"Joblib pvoutput residual cache file for system {nearby_sid_key} not found. For now, required.")
            
            cached_object = joblib.load(cache_file_path) ### This has modification time and residual_cache inside
            # Skip mod time integrity check
            pvoutput_residuals_cache[nearby_sid_key] = cached_object["residual_cache"]
        print("\nCache initialization complete.")
        return pvoutput_residuals_cache

            

        
    def init_precompute_satellite_power(self, df_weather):
        satellite_df = self.virtual_houses_satellite_df.copy()

        #satellite_df["temperature_2m"] = 20.0 ### THIS HAS TO BE PORTED FROM THE SINGLE-RUN DATA!! ------>>>>> TO DO
        #satellite_df["wind_speed_10m"] = 5.0 ### THIS HAS TO BE PORTED FROM THE SINGLE-RUN DATA!! ------>>>>> TO DO

        ### Added in v4 too
        weather_temp_and_wind = df_weather[["forecasted_time", "temperature_2m", "wind_speed_10m"]].copy() ### Fill in
        weather_temp_and_wind = weather_temp_and_wind.drop_duplicates("forecasted_time", keep="last").set_index("forecasted_time") ### Only keep last available forecast

        satellite_df = satellite_df.loc[satellite_df.index.isin(weather_temp_and_wind.index)].copy() ### Fix by chat for misalignemtn

        ### This assumes reasonably good prediction of temperature and wind speed but good enough as the impact is not so high.
        satellite_df["temperature_2m"] = weather_temp_and_wind["temperature_2m"].reindex(satellite_df.index).to_numpy()
        satellite_df["wind_speed_10m"] = weather_temp_and_wind["wind_speed_10m"].reindex(satellite_df.index).to_numpy()

        if satellite_df[["temperature_2m", "wind_speed_10m"]].isna().any().any():
            raise ValueError(f"Could not allign temperature and wind speed from weather data to satellite data for load_id {self.load_id} because of nan")

        satellite_output_df = self.run_physical_model(satellite_df, index_already_set=True)

        return satellite_output_df["fitting_factor_power"]


    def init_precompute_physical_forecast(self, df_weather):
        output_df = self.run_physical_model(df_weather, index_already_set=False)
        output_df = output_df.reset_index()

        merged_df = self.merge_output_with_weather(df_weather, output_df)

        eps = 0.01 ### To avoid division by zero and to zero instead of nan. Cleaner
        valid_clear_sky_mask = merged_df["clear_sky_fitting_factor_power"] > eps
        merged_df["forecasted_clear_sky_factor"] = 0.0
        merged_df.loc[valid_clear_sky_mask, "forecasted_clear_sky_factor"] = merged_df.loc[valid_clear_sky_mask, "forecasted_power_kw"] / merged_df.loc[valid_clear_sky_mask, "clear_sky_fitting_factor_power"]
        
        return merged_df

    


    def init_retrieve_flanders_household_metadata(self, load_id, flanders_buildings_df):
        #flanders_buildings_df = flanders_buildings_df.set_index("load_id") ### Done outside loop already
        virtual_house = flanders_buildings_df.loc[load_id]
        virtual_house_latitude = virtual_house["latitude"]
        virtual_house_longitude = virtual_house["longitude"]
        virtual_house_orientation = virtual_house["pv_orientation_deg"]
        virtual_house_tilt = virtual_house["pv_tilt_deg"]
        virtual_house_system_size = virtual_house["pv_system_size_w"]

        ### Immediately retrieve battery capacity for the MPC framework to avoid loading double
        virtual_house_battery_capacity = virtual_house["Assigned_BESS_capacity_kwh"]

        PV_params = {
                "latitude": virtual_house_latitude,
                "longitude": virtual_house_longitude,
                "system_size_w": virtual_house_system_size,
                "orientation_deg": virtual_house_orientation,
                "tilt_deg": virtual_house_tilt
            }
        return PV_params, virtual_house_battery_capacity


    def init_load_weather_data(self, file_path, state): ### state should be "preceding" or "upcoming"!!! --> Will shift the values accordingly
        weather_df = pd.read_parquet(file_path)
        weather_df["_origin"] = weather_df.index
        weather_df = weather_df.sort_values(["_origin", "forecasted_time"]) ### Have to sort in this way
        weather_df = weather_df.set_index("_origin")
        ### Cut from on 2025-08-01
        weather_df = weather_df[weather_df.index >= "2025-08-01"]
        ### A few checks
        if weather_df.isna().any().any():
            raise ValueError(f"Weather data contains NaN values. Please check the data in {file_path}.")
        required_columns = ["shortwave_radiation", "diffuse_radiation", "direct_normal_irradiance", "temperature_2m", "wind_speed_10m"]
        if not all(col in weather_df.columns for col in required_columns):
            raise ValueError(f"Weather data is missing required columns. Please check the data in {file_path}. Required columns: {required_columns}")
        if state not in ["preceding", "upcoming"]:
            raise ValueError(f"State must be either 'preceding' or 'upcoming'. Provided state: {state}")

        if state == "preceding":
            pass
        if state == "upcoming":
            weather_df["forecasted_time"] = weather_df["forecasted_time"] - pd.Timedelta(minutes=15) ### Shift back by 15 minutes. Now, each timestamp is the average of what WILL happen in the next 15 minutes.

        return weather_df
    
    
    def get_weather_with_physical_for_root_timestep(self, root_timestep: pd.Timestamp): ### Function in between init and normal ones. Debug showed running the whole physical model for the whole df_weather is way too heavy as frameworks have to be re-built for each instance as processes can't share objects.
        weather_df_cut = self.df_weather[self.df_weather.index <= root_timestep]
        if weather_df_cut.empty:
            raise ValueError(f"No weather data available for root timestep {root_timestep}. Please check the weather data.")
        
        last_origin = weather_df_cut.index.max()
        if last_origin in self.df_weather_with_physical_cache:
            return self.df_weather_with_physical_cache[last_origin].copy()
        
        weather_df_one_origin = self.df_weather[self.df_weather.index == last_origin].copy()
        weather_df_with_physical = self.init_precompute_physical_forecast(weather_df_one_origin)
        #self.df_weather_with_physical_cache[last_origin] = weather_df_with_physical
        self.df_weather_with_physical_cache = {last_origin: weather_df_with_physical} ### Clear previous cache to save RAM, only keep the last one
        return weather_df_with_physical.copy()






    def run_physical_model(self, weather_df, index_already_set=False):
            run_weather_df = weather_df.copy()
            ### Set forecasted_time as index
            if not index_already_set:
                run_weather_df = run_weather_df.set_index("forecasted_time")

            output_df = self.physical_model.run(run_weather_df) ### Use forecasted_time as the time column, not index now, that's the origin
            return output_df
    

    def merge_output_with_weather(self, weather_df, output_df):
            weather_df = weather_df.copy()
            output_df = output_df.copy()

            if len(weather_df) != len(output_df):
                raise ValueError(f"Length mismatch: weather_df={len(weather_df)}, output_df={len(output_df)}")

            ### Check just the allignment, position is only factor kept in order to merge 
            weather_times = weather_df["forecasted_time"].reset_index(drop=True)
            output_times = output_df["forecasted_time"].reset_index(drop=True)

            if not weather_times.equals(output_times):
                raise ValueError("forecasted_time is not aligned row-by-row.")

            ### Now allign and hope for best xd
            weather_df["forecasted_power_kw"] = output_df["fitting_factor_power"].to_numpy()
            weather_df["forecasted_energy_kwh"] = output_df["fitting_factor_energy"].to_numpy()
            weather_df["solar_elevation"] = output_df["solar_elevation"].to_numpy()
            weather_df["clear_sky_fitting_factor_power"] = output_df["clear_sky_fitting_factor_power"].to_numpy()
            weather_df["clear_sky_fitting_factor_energy"] = output_df["clear_sky_fitting_factor_energy"].to_numpy()

            return weather_df
    

    def get_nearby_pvoutput_houses(self, radius_km=30): ### Get all PVOutput houses in a radius of x kms around the virtual house
        to_exclude_sids = []
        if self.params.EXCLUDE_PVOUTPUT_SHADOWS:
            shadow_txt = self.params.Pvoutput_exclude_systems_folder / "pvoutput_shadow_sids.txt"
            shadow_sids = shadow_txt.read_text().splitlines()
            shadow_sids = [sid.strip() for sid in shadow_sids if sid.strip()]

            to_exclude_sids.extend(shadow_sids)

        if self.params.EXCLUDE_PVOUTPUT_CHOPPED:
            chopped_txt = self.params.Pvoutput_exclude_systems_folder / "pvoutput_chopped_sids.txt"
            chopped_sids = chopped_txt.read_text().splitlines()
            chopped_sids = [sid.strip() for sid in chopped_sids if sid.strip()]

            to_exclude_sids.extend(chopped_sids)

        if self.params.EXCLUDE_PVOUTPUT_EW_ORIENTED:
            ew_txt = self.params.Pvoutput_exclude_systems_folder / "pvoutput_EW_orientation_sids.txt"
            ew_sids = ew_txt.read_text().splitlines()
            ew_sids = [sid.strip() for sid in ew_sids if sid.strip()]

            to_exclude_sids.extend(ew_sids)

        nearby_houses = {}
        for idx, row in self.pvoutput_metadata_df.iterrows():
            pvoutput_house_sid = row["system_id"]
            if str(pvoutput_house_sid) in to_exclude_sids:
                continue
            secondary_orientation = str(row["secondary_orientation"]).strip().lower()
            secondary_panels = float(row["secondary_panels"])
            if secondary_panels > 0 or secondary_orientation not in ["", "nan", "none", "null"]: ### Pvoutput has no consistent param here. System with secondary system should be excluded! Gives problems with ridiculous clear sky factors and thus very bad scenarios
                print(f"Excluding PVOutput system {pvoutput_house_sid} due to secondary system or orientation: secondary_orientation={secondary_orientation}, secondary_panels={secondary_panels}")
                continue
            pvoutput_house_coordinates = (row["latitude"], row["longitude"]) ### Must be tuple
            virtual_house_coordinates = (self.PV_params_virtual_household["latitude"], self.PV_params_virtual_household["longitude"])
            distance_km = geodesic(virtual_house_coordinates, pvoutput_house_coordinates).km
            if distance_km <= radius_km:
                nearby_houses[pvoutput_house_sid] = distance_km
        return nearby_houses
    


    def get_clear_sky_residuals(self, nearby_houses, now_simulation_timestep, current_time, current_trajectory, include_distance_term, n_scenarios, only_first_point, verbose):
        candidate_scores = []
        current_values = current_trajectory["forecasted_clear_sky_factor"].to_numpy()

        horizon_steps = len(current_trajectory)

        if n_scenarios == 0:
            scenario_df = current_trajectory[["forecasted_time", "forecasted_power_kw", "clear_sky_fitting_factor_power"]].copy()
            return scenario_df, pd.DataFrame() ### Return empty candidate scores df

        for nearby_sid in nearby_houses.keys(): ### Changes from on here
            nearby_sid_key = str(nearby_sid)
            residuals_file = self.pvoutput_residuals_folder / f"Residuals_merged_system_{nearby_sid}.parquet"
            if self.pvout_residuals_cache is None:
                raise ValueError("pvout_residuals_cache is None. Enable caching")
            if nearby_sid_key not in self.pvout_residuals_cache:
                if verbose >= 1:
                    print(colored(f"Residuals for system_id {nearby_sid} not found in cache. Skipping.", "yellow"))
                continue
            
            residual_cache = self.pvout_residuals_cache[nearby_sid_key]
            # Current time is still simulation timestep now
            ### Load in matrix
            candidate_group = residual_cache["candidate_matrix_by_time"].get(current_time)
            if candidate_group is None:
                continue
            candidate_start_times = candidate_group["candidate_starts"]
            candidate_origins = candidate_group["candidate_origins"]
            forecasted_clear_sky_factor_matrix = candidate_group["forecasted_clear_sky_factor_matrix"]
            clear_sky_residual_matrix = candidate_group["clear_sky_residual_matrix"]

            if forecasted_clear_sky_factor_matrix.shape[1] < horizon_steps: ### Little check for debug by chat
                raise ValueError(f"PV residual cache has {forecasted_clear_sky_factor_matrix.shape[1]} steps, but {horizon_steps} are needed.")

            #start_times = residual_cache["start_times_by_forecasted_time"].get(current_time, pd.Series(dtype='datetime64[ns, Europe/Brussels]')) ### If the current time does not exist, return empty serie, by chat. Now returns all historical candidate start times w same time of day as current timestep
            # Old: start_times = df_residuals_filtered.loc[df_residuals_filtered["forecasted_time"].dt.time == current_time, "forecasted_time"].drop_duplicates().sort_values() ### Retrieve all unique forecasted_time values for the current time of day

            
            if self.params.LOOKBACK_DAYS is not None:
                min_start_time = now_simulation_timestep - pd.Timedelta(days=self.params.LOOKBACK_DAYS)
                valid_mask = (candidate_start_times >= min_start_time) & (candidate_start_times < now_simulation_timestep)
                #start_times = start_times[(start_times >= (now_simulation_timestep - pd.Timedelta(days=self.params.LOOKBACK_DAYS))) & (start_times < now_simulation_timestep)] ### Should do it
            else:
                valid_mask = (candidate_start_times < now_simulation_timestep) ### No limit
            

            valid_mask_2 = valid_mask & ((candidate_start_times + pd.Timedelta(minutes=15 * horizon_steps)) <= now_simulation_timestep)
            if not valid_mask_2.any():
                if verbose >= 1:
                    print(colored(f"No valid candidate start times found for load_id {self.load_id} and nearby_sid {nearby_sid}.", "yellow"))
                continue

            selected_starts = candidate_start_times[valid_mask_2]
            selected_origins = candidate_origins[valid_mask_2]
            selected_forecasted_clear_sky_factor_matrix = forecasted_clear_sky_factor_matrix[valid_mask_2, :horizon_steps] ### Immediately applies to rows
            selected_clear_sky_residual_matrix = clear_sky_residual_matrix[valid_mask_2, :horizon_steps]
            
            # The magic. Set float32 instead of 64, by chat
            rmse_values = np.sqrt(np.mean((selected_forecasted_clear_sky_factor_matrix - current_values.astype(np.float32)) ** 2, axis=1)) ### Axis 1 because over columns. rows are candidates, columns are the 96 forecasted values
            
            ### And assign in the same way as before for compatibility
            for j in range(len(rmse_values)):
                candidate_scores.append({
                    "load_id": self.load_id,
                    "nearby_sid": nearby_sid,
                    "candidate_origin": selected_origins[j],
                    "candidate_start": selected_starts[j],
                    "rmse": rmse_values[j],
                    "distance_km": nearby_houses[nearby_sid], ### Put this in here already for speedup not to have to retrieve it again
                    "clear_sky_residuals": selected_clear_sky_residual_matrix[j]
                })
            
                
        candidate_scores_df = pd.DataFrame(candidate_scores) ### Put in a df
        if candidate_scores_df.empty:
            if verbose >= 1:
                print(colored(f"No analog candidates found for load_id {self.load_id}", "red"))
            scenario_df = current_trajectory[["forecasted_time", "forecasted_power_kw", "clear_sky_fitting_factor_power"]].copy()
            return scenario_df, candidate_scores_df ### Return empty df

        if not include_distance_term:
            candidate_scores_df = candidate_scores_df.sort_values("rmse").head(n_scenarios).reset_index(drop=True) ### Get the best AMOUNT_OF_SCENARIOS candidates. Maybe this shouldn't be done yet until combining with full loss function but good for now
        else:
            candidate_scores_df["combined_score"] = candidate_scores_df["rmse"] + 0.01 * candidate_scores_df["distance_km"] ### I have absolutely no idea yet on weights
            candidate_scores_df = candidate_scores_df.sort_values("combined_score").head(n_scenarios).reset_index(drop=True) 

        ### Create scenarios, similar to load scenario building

        if only_first_point:
            scenario_df = current_trajectory[["forecasted_time", "forecasted_power_kw", "clear_sky_fitting_factor_power"]].iloc[[0]].copy() ### Only first point to avoid calculating mean over everything, rest can stay the same
            for i, row in candidate_scores_df.iterrows():
                first_residual = row["clear_sky_residuals"][0]
                scenario_df[f"scenario_{i}_power_kw"] = scenario_df["forecasted_power_kw"] + first_residual * scenario_df["clear_sky_fitting_factor_power"]
                scenario_df[f"scenario_{i}_power_kw"] = scenario_df[f"scenario_{i}_power_kw"].clip(lower=0)
        else:
            scenario_df = current_trajectory[["forecasted_time", "forecasted_power_kw", "clear_sky_fitting_factor_power"]].copy()
            for i, row in candidate_scores_df.iterrows():
                scenario_df[f"scenario_{i}_power_kw"] = scenario_df["forecasted_power_kw"] + row["clear_sky_residuals"] * scenario_df["clear_sky_fitting_factor_power"] ### See file of PVOutput Residuals.py for the logic or on paper --> Should double check this however
                scenario_df[f"scenario_{i}_power_kw"] = scenario_df[f"scenario_{i}_power_kw"].clip(lower=0) ### Clip to 0, no negative power

                #### DEBUG
                print("DEGUUUUG")
                max_reasonable_power_kw = 1.2 * self.PV_params_virtual_household["system_size_w"] / 1000.0
                scenario_power = scenario_df[f"scenario_{i}_power_kw"]

                if scenario_power.max() > max_reasonable_power_kw + 1.0:
                    worst_position = scenario_power.to_numpy().argmax()

                    print("\nPV SCENARIO TOO HIGH DEBUG")
                    print("load_id:", self.load_id)
                    print("virtual_system_size_kw:", self.PV_params_virtual_household["system_size_w"] / 1000.0)
                    print("max_reasonable_power_kw:", max_reasonable_power_kw)
                    print("scenario:", i)
                    print("nearby_sid:", row["nearby_sid"])
                    print("candidate_origin:", row["candidate_origin"])
                    print("candidate_start:", row["candidate_start"])
                    print("forecasted_time:", scenario_df["forecasted_time"].iloc[worst_position])
                    print("forecasted_power_kw:", scenario_df["forecasted_power_kw"].iloc[worst_position])
                    print("clear_sky_fitting_factor_power:", scenario_df["clear_sky_fitting_factor_power"].iloc[worst_position])
                    print("clear_sky_residual:", row["clear_sky_residuals"][worst_position])
                    print("scenario_power_kw:", scenario_power.iloc[worst_position])
                    print("END PV SCENARIO TOO HIGH DEBUG\n")
                print("DEBUUUUG END")
                ###### DEBUUUUG END


        if verbose >= 2:
            print(candidate_scores_df[["nearby_sid", "candidate_origin", "rmse", "distance_km"]]) ### DEBUG
            print(scenario_df.head())



        return scenario_df, candidate_scores_df ### Needed for easy plotting as it keeps info on sid link
    



    def add_satellite_scenario(self, scenario_df): ### Called in run_one_timestep
        scenario_df = scenario_df.copy()

        scenario_df["satellite_power_kw"] = self.satellite_power_array.reindex(scenario_df["forecasted_time"]).to_numpy() ### Reindex can now immediately used to make the alignment. Faster and simpler

        return scenario_df
    


    def run_for_timestep(self, timestep: pd.Timestamp, end_timestep: pd.Timestamp, n_scenarios: int, verbose: int = 3, only_first_point: bool = False): ### Still an option but MPC will need full 24h trajectory
        # Check if in brussels timezone
        if timestep.tzinfo is None or str(timestep.tzinfo) != "Europe/Brussels":
            raise ValueError(f"Timestamp must be timezone-aware and in Europe/Brussels timezone. Provided timestamp: {timestep}")
        now_simulation_timestep = timestep
        end_simulation_timestep = end_timestep

        """weather_df_cut = self.df_weather_with_physical[self.df_weather_with_physical.index <= now_simulation_timestep].copy() # V2 ### Keep the df with only the last origin before the now_simulation_timestep, use index smaller than and max
        if weather_df_cut.empty:
            raise ValueError(colored(f"No weather data available before the simulation timestep {now_simulation_timestep} for load_id {self.load_id}. Please check the weather data.", "red"))
        last_origin = weather_df_cut.index.max()
        weather_df_cut = weather_df_cut[weather_df_cut.index == last_origin]"""
        weather_df_cut = self.get_weather_with_physical_for_root_timestep(now_simulation_timestep) ### new function
        last_origin = weather_df_cut.index.max()
        # Keep only upcoming 24 hours
        weather_df_cut = weather_df_cut[weather_df_cut["forecasted_time"] >= now_simulation_timestep]
        weather_df_cut = weather_df_cut[weather_df_cut["forecasted_time"] < end_simulation_timestep]
        if weather_df_cut.empty:
            raise ValueError(colored(f"No weather data found for the last origin {last_origin} before the simulation timestep {now_simulation_timestep} for load_id {self.load_id}. Please check the weather data.", "red"))
        # Check if every value in forecasted_time is unique as an extra check
        if weather_df_cut["forecasted_time"].duplicated().any():
            duplicated_times = weather_df_cut[weather_df_cut["forecasted_time"].duplicated(keep=False)]
            raise ValueError(colored(f"Duplicated forecasted_time values found in the weather data for load_id {self.load_id}:\n{duplicated_times}", "red"))
     

        merged_df = weather_df_cut.copy() 
        nearby_houses = self.nearby_houses
        if len(nearby_houses) < 3 and verbose >= 1:
            print(colored(f"Found low amount of nearby PVOutput houses for virtual house {self.load_id}.", "yellow"))
        else:
            if verbose >= 2:
                print(f"Found {len(nearby_houses)} nearby PVOutput houses for virtual house {self.load_id}.")
        

        current_time = now_simulation_timestep.tz_convert("UTC").time() ### By chta, convert to UTC as cache builder also groups by UTC in df["forecasted_time_utc_time"] = df["forecasted_time"].dt.tz_convert("UTC").dt.time
        current_trajectory = merged_df.sort_values("forecasted_time").reset_index(drop=True) ### Just to be sure, should already be sorted. Fix by Chat too, otherwise calc of rmse might be wrongly aligned

        scenario_df, candidate_scores_df = self.get_clear_sky_residuals(nearby_houses, now_simulation_timestep, current_time, current_trajectory, include_distance_term=False, n_scenarios=n_scenarios, only_first_point=only_first_point, verbose=verbose) ### Get the best 5 scenarios based on RMSE
        #scenario_df_with_satellite = self.add_satellite_scenario(scenario_df) ### Add the satellite-based trajectory to the scenario_df
        if self.satellite_power_array is not None:
            scenario_df = self.add_satellite_scenario(scenario_df)


        return scenario_df, candidate_scores_df ### Make it return them for analysis
    



    def run_for_branch_timestep(self, root_timestep: pd.Timestamp, branch_timestep: pd.Timestamp, end_timestep: pd.Timestamp, n_scenarios: int, verbose: int = 3, only_first_point: bool = False):
        if root_timestep.tzinfo is None or str(root_timestep.tzinfo) != "Europe/Brussels":
            raise ValueError(f"Root timestamp must be timezone-aware and in Europe/Brussels timezone. Provided timestamp: {root_timestep}")
        if branch_timestep.tzinfo is None or str(branch_timestep.tzinfo) != "Europe/Brussels":
            raise ValueError(f"Branch timestamp must be timezone-aware and in Europe/Brussels timezone. Provided timestamp: {branch_timestep}")
        
        if branch_timestep < root_timestep:
            raise ValueError(f"Branch timestep {branch_timestep} cannot be earlier than root timestep {root_timestep}.") ### To avoid shit from before
        
        end_time = end_timestep
        """weather_df_cut = self.df_weather_with_physical[self.df_weather_with_physical.index <= root_timestep].copy() ### Keep similar structure

        if weather_df_cut.empty:
            raise ValueError(colored(f"No weather data available before the root timestep {root_timestep} for load_id {self.load_id}. Please check the weather data.", "red"))
        last_origin = weather_df_cut.index.max() 
        weather_df_cut = weather_df_cut[weather_df_cut.index == last_origin].copy() ### Do copy here because gonna change"""
        ### Replaced by new code:
        weather_df_cut = self.get_weather_with_physical_for_root_timestep(root_timestep)
        last_origin = weather_df_cut.index.max()

        weather_df_cut = weather_df_cut[weather_df_cut["forecasted_time"] >= branch_timestep] ### Re cut
        weather_df_cut = weather_df_cut[weather_df_cut["forecasted_time"] < end_time]
        if weather_df_cut.empty:
            raise ValueError(f"No forecasted_time values from {branch_timestep} to {end_time} using weather origin {last_origin}")
        # Check length too
        if len(weather_df_cut) != int((end_time - branch_timestep) / pd.Timedelta(minutes=15)):
            raise ValueError(f"Expected {PV_HORIZON_STEPS} forecasted_time values from {branch_timestep} to {end_time} using weather origin {last_origin}, but got {len(weather_df_cut)}. Please check the weather data.")
        
        if weather_df_cut["forecasted_time"].duplicated().any(): ### Same check
            duplicated_times = weather_df_cut[weather_df_cut["forecasted_time"].duplicated(keep=False)]
            raise ValueError(colored(f"Duplicated forecasted_time values found in the weather data for load_id {self.load_id}:\n{duplicated_times}", "red"))
        
        current_time = branch_timestep.tz_convert("UTC").time() ### By chat
        current_trajectory = weather_df_cut.sort_values("forecasted_time").reset_index(drop=True) ### Skip merged_df immediately

        scenario_df, candidate_scores_df = self.get_clear_sky_residuals(nearby_houses=self.nearby_houses, now_simulation_timestep=root_timestep, current_time=current_time, current_trajectory=current_trajectory, include_distance_term=False, n_scenarios=n_scenarios, only_first_point=only_first_point, verbose=verbose)

        return scenario_df, candidate_scores_df


    


if __name__ == "__main__":
    params = PV_residual_behaviour_params()
    virtual_household = 301
    vh_residual_search = Virtual_household_residual_searching(params, virtual_household)
    timestep = pd.Timestamp("2025-11-24 02:00", tz="Europe/Brussels")
    ts_2 = pd.Timestamp("2025-11-24 02:15", tz="Europe/Brussels")
    scenario_df_with_satellite, candidate_scores_df = vh_residual_search.run_for_timestep(timestep, n_scenarios=5, verbose=3, only_first_point=False)
    # Second ts to check performance
    start_time = time.perf_counter()
    scenario_df_with_satellite_2, candidate_scores_df_2 = vh_residual_search.run_for_timestep(ts_2, n_scenarios=5, verbose=3, only_first_point=False) ### 0.0621 seconds
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for second timestep: {elapsed_time:.4f} seconds")
    print(scenario_df_with_satellite.head())