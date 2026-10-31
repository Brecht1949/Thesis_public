import openmeteo_requests
from pathlib import Path
import pandas as pd
import requests_cache
from retry_requests import retry
from dataclasses import dataclass, field, asdict
import sys
import numpy as np
from termcolor import colored


"""
Code to fetch satellite weather data from Open-Meteo API for Flanders buildings, using DWD, JMA, and EUMETSAT models. The data is resampled to 15-minute intervals and combined into a single DataFrame, with bad days filled in using EUMETSAT data.
"""
@dataclass
class Weather_params_DWD: ### Available from on 16-02-2026 ### DWD ALSO HAS A CLEAR SKY PARAMETER. ### March tenth is problematic
    latitude: float
    longitude: float
    apikey: str

    start_date: str
    end_date: str
    url: str = "https://satellite-api.open-meteo.com/v1/archive"
    timezone: str = "Europe/Brussels"
    temporal_resolution: str = "native" ### To keep 10 min for DWD or 15 min for EUMETSAT.
    
    
    models: str = "dwd_sis_europe_africa_v4"
    hourly: list = field(default_factory=lambda: ["shortwave_radiation", "diffuse_radiation", "direct_normal_irradiance", "temperature_2m", "wind_speed_10m"]) ### Have to do this again in order to make exclusive object per dataclass as list is mutable.
    ### Called hourly but native resolution is packed inside



@dataclass
class Weather_params_JMA: ### Available up until (and including) 2026-03-09
    latitude: float
    longitude: float
    apikey: str

    start_date: str
    end_date: str
    url: str = "https://satellite-api.open-meteo.com/v1/archive"
    timezone: str = "Europe/Brussels"
    temporal_resolution: str = "native" ### To keep 10 min for DWD or 15 min for EUMETSAT.
    
    
    models: str = "jma_jaxa_mtg_fci"
    hourly: list = field(default_factory=lambda: ["shortwave_radiation", "diffuse_radiation", "direct_normal_irradiance", "temperature_2m", "wind_speed_10m"]) ### Have to do this again in order to make exclusive object per dataclass as list is mutable.



@dataclass
class Weather_params_EUMETSAT: ### Will be used to fill in broken days.
    latitude: float
    longitude: float
    apikey: str

    start_date: str
    end_date: str
    url: str = "https://satellite-api.open-meteo.com/v1/archive"
    timezone: str = "Europe/Brussels"
    temporal_resolution: str = "native" ### To keep 10 min for DWD or 15 min for EUMETSAT.
    
    
    models: str = "eumetsat_lsa_saf_msg"
    hourly: list = field(default_factory=lambda: ["shortwave_radiation", "diffuse_radiation", "direct_normal_irradiance", "temperature_2m", "wind_speed_10m"]) ### Have to do this again in order to make exclusive object per dataclass as list is mutable.



class SatelliteWeatherFetcher():
    def __init__(self, params):
        self.url = params.pop("url") ### Does this work?
        self.params = params


    def fetch_weather_data(self):
        current_directory = Path(__file__).resolve().parent
        cache_dir = current_directory / "Weather_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_db = cache_dir / "openmeteo_cache.sqlite" ### Fix for linux

        cache_session = requests_cache.CachedSession(str(cache_db), expire_after=1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        try:
            responses = openmeteo.weather_api(self.url, params=self.params)
            weather_df = self.weather_data_to_dataframe(responses)
            return weather_df ### To avoid cached data in script
        except Exception as e:
            raise RuntimeError(f"Failed to fetch weather data: {e}")
        

    def weather_data_to_dataframe(self, responses):
        response = responses[0]
        hourly = response.Hourly() ### If want to add hourly or daily variables, adjust here as needed!!

        idx_utc = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )

        idx = idx_utc.tz_convert(self.params["timezone"]) ### Fix by chat

        data = {"date": idx}
        for i, var in enumerate(self.params["hourly"]):
            data[var] = hourly.Variables(i).ValuesAsNumpy()

        return pd.DataFrame(data).set_index("date")
    

    def run(self):
        try: 
            weather_df = self.fetch_weather_data()
            return weather_df
        except Exception as e:
            raise RuntimeError(f"Weather model failed:\n{e}")
        



def resample_10_to_5_to_15min(df):
    ### Check that it is 10 min data, otherwise raise
    dt_s = (df.index[1] - df.index[0]).total_seconds()
    if dt_s != 600:
        raise ValueError(f"Make sure time step is 10 minutes. First two timestamps are {df.index[0]} and {df.index[1]}, with a difference of {dt_s/60} minutes.")
    df = df.sort_index()

    df_5min_a = df.copy()
    df_5min_b = df.copy()
    df_5min_b.index = df_5min_b.index - pd.Timedelta(minutes=5) ### Good little trick

    df_5min = pd.concat([df_5min_a, df_5min_b]).sort_index() ### Now have 5 min data

    df_15min = df_5min.resample("15min", label="right", closed="right").mean() ### Preceding!

    return df_15min






if __name__ == "__main__":
    if sys.platform == "darwin":
        API_KEY = pd.read_csv("/Users/brechtgoethals/API_OPENMETEO.txt", header=None).iloc[0, 0]
    elif sys.platform == "win32":
        API_KEY = pd.read_csv(r"C:\Users\brech\API_OPENMETEO.txt", header=None).iloc[0, 0]

    END_SATELLITE = (pd.Timestamp.now(tz="Europe/Brussels").normalize() - pd.Timedelta(days=3)).strftime("%Y-%m-%d")

    ENABLE_FLANDERS_BUILDINGS = True
    ENABLE_PVOUTPUT_HOUSES = True

    if ENABLE_FLANDERS_BUILDINGS:
        ### Has to be loaded in only for FLANDERS_BUILDINGS. Maybe later for fitting factor?
        ###'/Users/brechtgoethals/Library/CloudStorage/GoogleDrive-brecht.goethals23@gmail.com/My Drive/T - GDRIVE/Virtual_houses/Households_w_battery_dataset/random_sample_1_with_load_pv_and_bess_assigned.parquet'
        FLANDERS_BUILDINGS = Path(Path(__file__).parent.parent.parent / "Virtual_houses" / "Households_w_battery_dataset" / "random_sample_1_with_load_pv_and_bess_assigned.parquet")
        FLANDERS_BUILDINGS_df = pd.read_parquet(FLANDERS_BUILDINGS)

        FLANDERS_BUILDINGS_df = FLANDERS_BUILDINGS_df[["load_id", "latitude", "longitude"]] ### Load id is from Fluvius 2400 dataset




        start_DWD = "2026-02-16"
        #end_DWD = (pd.Timestamp.now(tz="Europe/Brussels").normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d") 
        end_DWD = END_SATELLITE #"2026-06-23"
        ### Read bad dates for DWD
        bad_dwd_txt = Path(__file__).parent / "Bad_days_DWD.txt" ### Dates with an X behind it should be filled in, not fully deleted. If no X, completely nan that day
        bad_dwd_lines = bad_dwd_txt.read_text().splitlines()
        bad_dwd_lines = [line.strip() for line in bad_dwd_lines]
        bad_X_dwd_dates = [line.replace("X", "").strip() for line in bad_dwd_lines if line.endswith("X")]
        bad_dwd_dates = [line.strip() for line in bad_dwd_lines if not line.endswith("X")]

        start_JMA = "2025-07-30"
        end_JMA = "2026-03-08" ### 9th is already incomplete
        ### Read bad dates for JMA
        Bad_days_JMA = Path(__file__).parent / "Bad_days_JMA.txt" ### Dates with an X behind it should be filled in, not fully deleted. If no X, completely nan that day
        bad_jma_lines = Bad_days_JMA.read_text().splitlines()
        bad_jma_lines = [line.strip() for line in bad_jma_lines]
        bad_X_jma_dates = [line.replace("X", "").strip() for line in bad_jma_lines if line.endswith("X")] ### do fillna immediately
        bad_jma_dates = [line.strip() for line in bad_jma_lines if not line.endswith("X")] ### delete full day, then fillna

        ### For testing, only keep first one
        #FLANDERS_BUILDINGS_df = FLANDERS_BUILDINGS_df.head(1)
        total_len = len(FLANDERS_BUILDINGS_df)
        counter = 0
        for index, row in FLANDERS_BUILDINGS_df.iterrows():
            load_id = row["load_id"] ### Have to define first
            latitude = row["latitude"]
            longitude = row["longitude"]
            save_path = Path(__file__).parent / "Satellite_weather_data" / "Flanders_buildings" / f"load_id_{load_id}_satellite_weather.parquet"
            
            ### Improved saving
            wanted_start = pd.Timestamp(start_JMA, tz="Europe/Brussels") # Move up
            wanted_end = pd.Timestamp(end_DWD, tz="Europe/Brussels") + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
            wanted_index = pd.date_range(start=wanted_start, end=wanted_end, freq="15min")

            existing_df = None
            missing_start = wanted_start
            
            if save_path.exists():
                existing_df = pd.read_parquet(save_path)
                existing_df.index = pd.to_datetime(existing_df.index)

                if existing_df.index.tz is None:
                    existing_df.index = existing_df.index.tz_localize("Europe/Brussels")
                else:
                    existing_df.index = existing_df.index.tz_convert("Europe/Brussels")
                
                existing_df = existing_df.sort_index()
                missing_timestamps = wanted_index.difference(existing_df.index)

                if len(missing_timestamps) == 0:
                    print(f"Satellite weather data for load_id {load_id} already exists and is complete. Skipping.")
                    counter += 1
                    continue

                missing_start = missing_timestamps.min()
                print(colored(f"Satellite weather data for load_id {load_id} exists but is incomplete. Missing data starts from {missing_start}. Fetching missing data.", "yellow"))
            counter += 1



            start_DWD_fetch = max(missing_start.normalize(), pd.Timestamp(start_DWD, tz="Europe/Brussels")).strftime("%Y-%m-%d")
            start_JMA_fetch = max(missing_start.normalize(), pd.Timestamp(start_JMA, tz="Europe/Brussels")).strftime("%Y-%m-%d")
            start_EUMETSAT_fetch = missing_start.normalize().strftime("%Y-%m-%d")

            ### Retrieve both, cut JMA to beginning of DWD
            weather_params_dwd = Weather_params_DWD(latitude=latitude, longitude=longitude, apikey=API_KEY, start_date=start_DWD_fetch, end_date=end_DWD)
            weather_params_jma = Weather_params_JMA(latitude=latitude, longitude=longitude, apikey=API_KEY, start_date=start_JMA_fetch, end_date=end_JMA)
            weather_params_eumetsat = Weather_params_EUMETSAT(latitude=latitude, longitude=longitude, apikey=API_KEY, start_date=start_EUMETSAT_fetch, end_date=end_DWD) ### Just get the whole period, have enough credits

            fetcher_dwd = SatelliteWeatherFetcher(asdict(weather_params_dwd))
            fetcher_jma = SatelliteWeatherFetcher(asdict(weather_params_jma))
            fetcher_eumetsat = SatelliteWeatherFetcher(asdict(weather_params_eumetsat))

            dwd_df = fetcher_dwd.run()
            dwd_df_15min = resample_10_to_5_to_15min(dwd_df)
            print("DWD data retrieved and resampled.")
            #print(dwd_df.head()) ### temperature_2m, wind_speed_10m not retrieved. Has to be retrieved from other source!

            if pd.Timestamp(start_JMA_fetch) <= pd.Timestamp(end_JMA):
                jma_df = fetcher_jma.run()
                jma_df_15min = resample_10_to_5_to_15min(jma_df)
                print("JMA data retrieved and resampled.")
                #print(jma_df.head()) ### temperature_2m, wind_speed_10m also not retrieved. Has to be retrieved from other source!
            else:
                jma_df_15min = pd.DataFrame(columns=dwd_df_15min.columns) ### Just empty df with same columns to avoid breaking code
                print("JMA not needed for this missing period.")

            eumetsat_df = fetcher_eumetsat.run()
            eumetsat_df_15min = eumetsat_df
            print("EUMETSAT data retrieved.")
            #eumetsat_df_15min = resample_10_to_5_to_15min(eumetsat_df) ### IS NATIVELY 15MIN
            #print(eumetsat_df.head()) 
            #print(eumetsat_df.head()) ### temperature_2m, wind_speed_10m also not retrieved. Has to be retrieved from other source!


            ### Make overall df
            start_overall = missing_start
            end_overall = pd.Timestamp(end_DWD, tz="Europe/Brussels") + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)

            index_overall = pd.date_range(start=start_overall, end=end_overall, freq="15min")
            cols = dwd_df_15min.columns.union(jma_df_15min.columns) ### Took union but could be either one or intersection
            satellite_combined_df = pd.DataFrame(index=index_overall, columns=cols, dtype=float) ### Filled with nans
            satellite_combined_df["source"] = pd.Series(index=index_overall, dtype="string") ### Fix by chat
            satellite_combined_df["source"] = "None"

            dwd_aligned = dwd_df_15min.reindex(index_overall)
            dwd_fill_indices = satellite_combined_df[cols].isna() & dwd_aligned[cols].notna()
            satellite_combined_df[cols] = satellite_combined_df[cols].fillna(dwd_aligned[cols])
            satellite_combined_df.loc[dwd_fill_indices.any(axis=1), "source"] = "DWD" ### Fix by chat

            jma_aligned = jma_df_15min.reindex(index_overall)
            jma_fill_indices = satellite_combined_df[cols].isna() & jma_aligned[cols].notna()
            satellite_combined_df[cols] = satellite_combined_df[cols].fillna(jma_aligned[cols])
            satellite_combined_df.loc[jma_fill_indices.any(axis=1), "source"] = "JMA" ### Fix by chat


            eumetsat_aligned = eumetsat_df_15min.reindex(index_overall)
            ### For bad days in DWD, make them nan
            for bad_date in bad_dwd_dates:
                day_indices = satellite_combined_df.index.date == pd.to_datetime(bad_date).date()
                satellite_combined_df.loc[day_indices, cols] = np.nan
                satellite_combined_df.loc[day_indices, cols] = satellite_combined_df.loc[day_indices, cols].fillna(eumetsat_aligned.loc[day_indices, cols]) #### Now fill in of course
                satellite_combined_df.loc[day_indices, "source"] = "EUMETSAT_REPLACE"
            for bad_date in bad_X_dwd_dates:
                day_indices = satellite_combined_df.index.date == pd.to_datetime(bad_date).date()
                fill_indices = satellite_combined_df.loc[day_indices, cols].isna() & eumetsat_aligned.loc[day_indices, cols].notna() ### For source logic
                satellite_combined_df.loc[day_indices, cols] = satellite_combined_df.loc[day_indices, cols].fillna(eumetsat_aligned.loc[day_indices, cols]) 
                satellite_combined_df.loc[day_indices, "source"] = np.where(fill_indices.any(axis=1), f"EUMETSAT_FILL", satellite_combined_df.loc[day_indices, "source"]) ### Any axis debug by chat

            
            ### For bad days in JMA, make them nan
            for bad_date in bad_jma_dates:
                day_indices = satellite_combined_df.index.date == pd.to_datetime(bad_date).date()
                satellite_combined_df.loc[day_indices, cols] = np.nan
                satellite_combined_df.loc[day_indices, cols] = satellite_combined_df.loc[day_indices, cols].fillna(eumetsat_aligned.loc[day_indices, cols])
                satellite_combined_df.loc[day_indices, "source"] = "EUMETSAT_REPLACE"
            for bad_date in bad_X_jma_dates:
                day_indices = satellite_combined_df.index.date == pd.to_datetime(bad_date).date()
                fill_indices = satellite_combined_df.loc[day_indices, cols].isna() & eumetsat_aligned.loc[day_indices, cols].notna()
                satellite_combined_df.loc[day_indices, cols] = satellite_combined_df.loc[day_indices, cols].fillna(eumetsat_aligned.loc[day_indices, cols])
                satellite_combined_df.loc[day_indices, "source"] = np.where(fill_indices.any(axis=1), f"EUMETSAT_FILL", satellite_combined_df.loc[day_indices, "source"])

            

            ### Temporary nan fillings: 
            # take out all nans where shortwave_radiation is 0 and diffuse and direct are nan, put them to zero too
            zero_indices = (satellite_combined_df["shortwave_radiation"] == 0) & (satellite_combined_df["diffuse_radiation"].isna()) & (satellite_combined_df["direct_normal_irradiance"].isna())
            satellite_combined_df.loc[zero_indices, ["diffuse_radiation", "direct_normal_irradiance"]] = 0
            ### if has nans between 00:00 and 06:00, fill them with 0 as well
            night_indices = (satellite_combined_df.index.time >= pd.to_datetime("00:00").time()) & (satellite_combined_df.index.time < pd.to_datetime("06:00").time())
            night_nan_indices = night_indices & (satellite_combined_df["shortwave_radiation"].isna())
            satellite_combined_df.loc[night_nan_indices, ["shortwave_radiation", "diffuse_radiation", "direct_normal_irradiance"]] = 0

            ### And rest can be filled with fillna for irradiance columns, I checked it manually
            irradiance_cols = ["shortwave_radiation", "diffuse_radiation", "direct_normal_irradiance"]
            satellite_combined_df[irradiance_cols] = satellite_combined_df[irradiance_cols].fillna(0)

            print(satellite_combined_df.head())

        
            ### Print nans as rows, ignoring temp and wind
            nan_rows = satellite_combined_df[satellite_combined_df[["shortwave_radiation", "diffuse_radiation", "direct_normal_irradiance"]].isna().any(axis=1)]
            for idx, row in nan_rows.iterrows():
                print(f"NaN at {idx} for load_id {load_id}: shortwave_radiation={row['shortwave_radiation']}, diffuse_radiation={row['diffuse_radiation']}, direct_normal_irradiance={row['direct_normal_irradiance']}")



            ### Save overall df
            # Make intersection
            save_path.parent.mkdir(parents=True, exist_ok=True)
            if existing_df is not None:
                satellite_combined_df = pd.concat([existing_df, satellite_combined_df]).sort_index()
                satellite_combined_df = satellite_combined_df[~satellite_combined_df.index.duplicated(keep="last")] ### Last but should be same
            final_df = satellite_combined_df.reindex(wanted_index)
            # Small check
            missing_timestamps_after = wanted_index.difference(satellite_combined_df.index)

            if len(missing_timestamps_after) > 0:
                raise ValueError(f"After saving, there are still missing timestamps for load_id {load_id}: {missing_timestamps_after}")
            
            final_df.to_parquet(save_path)
            print(colored(f"Saved satellite weather data for load_id {load_id} to {save_path} ({counter}/{total_len})", "green"))


    ENABLE_FLANDERS_BUILDINGS_ALTERNATIVE = True

    if ENABLE_FLANDERS_BUILDINGS_ALTERNATIVE:
        FLANDERS_BUILDINGS_ALT = (
            Path(__file__).parent.parent.parent
            / "Virtual_houses"
            / "Building_addresses_Flanders"
            / "Alternative_virtual_houses_samples"
            / "random_sample_MAX_DISTANCE_15_rs_40_nclosest_3_samples_1200_with_loads_and_pv_and_bess_assigned.parquet"
        )

        FLANDERS_BUILDINGS_df = pd.read_parquet(FLANDERS_BUILDINGS_ALT)
        FLANDERS_BUILDINGS_df = FLANDERS_BUILDINGS_df[["load_id", "latitude", "longitude"]]

        start_DWD = "2026-02-16"
        end_DWD = END_SATELLITE
        start_JMA = "2025-07-30"
        end_JMA = "2026-03-08"

        bad_dwd_txt = Path(__file__).parent / "Bad_days_DWD.txt"
        bad_dwd_lines = [line.strip() for line in bad_dwd_txt.read_text().splitlines()]
        bad_X_dwd_dates = [line.replace("X", "").strip() for line in bad_dwd_lines if line.endswith("X")]
        bad_dwd_dates = [line.strip() for line in bad_dwd_lines if not line.endswith("X")]

        Bad_days_JMA = Path(__file__).parent / "Bad_days_JMA.txt"
        bad_jma_lines = [line.strip() for line in Bad_days_JMA.read_text().splitlines()]
        bad_X_jma_dates = [line.replace("X", "").strip() for line in bad_jma_lines if line.endswith("X")]
        bad_jma_dates = [line.strip() for line in bad_jma_lines if not line.endswith("X")]

        total_len = len(FLANDERS_BUILDINGS_df)
        counter = 0

        for index, row in FLANDERS_BUILDINGS_df.iterrows():
            load_id = row["load_id"]
            latitude = row["latitude"]
            longitude = row["longitude"]

            save_path = (
                Path(__file__).parent
                / "Satellite_weather_data"
                / "Flanders_buildings_ALTERNATIVE"
                / f"load_id_{load_id}_satellite_weather.parquet"
            )

            wanted_start = pd.Timestamp(start_JMA, tz="Europe/Brussels")
            wanted_end = pd.Timestamp(end_DWD, tz="Europe/Brussels") + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
            wanted_index = pd.date_range(start=wanted_start, end=wanted_end, freq="15min")

            existing_df = None
            missing_start = wanted_start

            if save_path.exists():
                existing_df = pd.read_parquet(save_path)
                existing_df.index = pd.to_datetime(existing_df.index)

                if existing_df.index.tz is None:
                    existing_df.index = existing_df.index.tz_localize("Europe/Brussels")
                else:
                    existing_df.index = existing_df.index.tz_convert("Europe/Brussels")

                existing_df = existing_df.sort_index()
                missing_timestamps = wanted_index.difference(existing_df.index)

                if len(missing_timestamps) == 0:
                    print(f"Satellite weather data for alternative load_id {load_id} already exists and is complete. Skipping.")
                    counter += 1
                    continue

                missing_start = missing_timestamps.min()
                print(colored(f"Alternative load_id {load_id} incomplete. Missing data starts from {missing_start}.", "yellow"))

            counter += 1

            start_DWD_fetch = max(missing_start.normalize(), pd.Timestamp(start_DWD, tz="Europe/Brussels")).strftime("%Y-%m-%d")
            start_JMA_fetch = max(missing_start.normalize(), pd.Timestamp(start_JMA, tz="Europe/Brussels")).strftime("%Y-%m-%d")
            start_EUMETSAT_fetch = missing_start.normalize().strftime("%Y-%m-%d")

            weather_params_dwd = Weather_params_DWD(latitude=latitude, longitude=longitude, apikey=API_KEY, start_date=start_DWD_fetch, end_date=end_DWD)
            weather_params_jma = Weather_params_JMA(latitude=latitude, longitude=longitude, apikey=API_KEY, start_date=start_JMA_fetch, end_date=end_JMA)
            weather_params_eumetsat = Weather_params_EUMETSAT(latitude=latitude, longitude=longitude, apikey=API_KEY, start_date=start_EUMETSAT_fetch, end_date=end_DWD)

            fetcher_dwd = SatelliteWeatherFetcher(asdict(weather_params_dwd))
            fetcher_jma = SatelliteWeatherFetcher(asdict(weather_params_jma))
            fetcher_eumetsat = SatelliteWeatherFetcher(asdict(weather_params_eumetsat))

            dwd_df = fetcher_dwd.run()
            dwd_df_15min = resample_10_to_5_to_15min(dwd_df)
            print("DWD data retrieved and resampled.")

            if pd.Timestamp(start_JMA_fetch) <= pd.Timestamp(end_JMA):
                jma_df = fetcher_jma.run()
                jma_df_15min = resample_10_to_5_to_15min(jma_df)
                print("JMA data retrieved and resampled.")
            else:
                jma_df_15min = pd.DataFrame(columns=dwd_df_15min.columns)
                print("JMA not needed for this missing period.")

            eumetsat_df = fetcher_eumetsat.run()
            eumetsat_df_15min = eumetsat_df
            print("EUMETSAT data retrieved.")

            start_overall = missing_start
            end_overall = pd.Timestamp(end_DWD, tz="Europe/Brussels") + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)

            index_overall = pd.date_range(start=start_overall, end=end_overall, freq="15min")
            cols = dwd_df_15min.columns.union(jma_df_15min.columns)

            satellite_combined_df = pd.DataFrame(index=index_overall, columns=cols, dtype=float)
            satellite_combined_df["source"] = "None"

            dwd_aligned = dwd_df_15min.reindex(index_overall)
            dwd_fill_indices = satellite_combined_df[cols].isna() & dwd_aligned[cols].notna()
            satellite_combined_df[cols] = satellite_combined_df[cols].fillna(dwd_aligned[cols])
            satellite_combined_df.loc[dwd_fill_indices.any(axis=1), "source"] = "DWD"

            jma_aligned = jma_df_15min.reindex(index_overall)
            jma_fill_indices = satellite_combined_df[cols].isna() & jma_aligned[cols].notna()
            satellite_combined_df[cols] = satellite_combined_df[cols].fillna(jma_aligned[cols])
            satellite_combined_df.loc[jma_fill_indices.any(axis=1), "source"] = "JMA"

            eumetsat_aligned = eumetsat_df_15min.reindex(index_overall)

            for bad_date in bad_dwd_dates + bad_jma_dates:
                day_indices = satellite_combined_df.index.date == pd.to_datetime(bad_date).date()
                satellite_combined_df.loc[day_indices, cols] = np.nan
                satellite_combined_df.loc[day_indices, cols] = satellite_combined_df.loc[day_indices, cols].fillna(eumetsat_aligned.loc[day_indices, cols])
                satellite_combined_df.loc[day_indices, "source"] = "EUMETSAT_REPLACE"

            for bad_date in bad_X_dwd_dates + bad_X_jma_dates:
                day_indices = satellite_combined_df.index.date == pd.to_datetime(bad_date).date()
                fill_indices = satellite_combined_df.loc[day_indices, cols].isna() & eumetsat_aligned.loc[day_indices, cols].notna()
                satellite_combined_df.loc[day_indices, cols] = satellite_combined_df.loc[day_indices, cols].fillna(eumetsat_aligned.loc[day_indices, cols])
                satellite_combined_df.loc[day_indices, "source"] = np.where(
                    fill_indices.any(axis=1),
                    "EUMETSAT_FILL",
                    satellite_combined_df.loc[day_indices, "source"]
                )

            zero_indices = (
                (satellite_combined_df["shortwave_radiation"] == 0)
                & satellite_combined_df["diffuse_radiation"].isna()
                & satellite_combined_df["direct_normal_irradiance"].isna()
            )
            satellite_combined_df.loc[zero_indices, ["diffuse_radiation", "direct_normal_irradiance"]] = 0

            night_indices = (
                (satellite_combined_df.index.time >= pd.to_datetime("00:00").time())
                & (satellite_combined_df.index.time < pd.to_datetime("06:00").time())
            )
            night_nan_indices = night_indices & satellite_combined_df["shortwave_radiation"].isna()
            satellite_combined_df.loc[night_nan_indices, ["shortwave_radiation", "diffuse_radiation", "direct_normal_irradiance"]] = 0

            irradiance_cols = ["shortwave_radiation", "diffuse_radiation", "direct_normal_irradiance"]
            satellite_combined_df[irradiance_cols] = satellite_combined_df[irradiance_cols].fillna(0)

            save_path.parent.mkdir(parents=True, exist_ok=True)

            if existing_df is not None:
                satellite_combined_df = pd.concat([existing_df, satellite_combined_df]).sort_index()
                satellite_combined_df = satellite_combined_df[~satellite_combined_df.index.duplicated(keep="last")]

            final_df = satellite_combined_df.reindex(wanted_index)

            missing_timestamps_after = wanted_index.difference(satellite_combined_df.index)

            if len(missing_timestamps_after) > 0:
                raise ValueError(f"After saving, there are still missing timestamps for alternative load_id {load_id}: {missing_timestamps_after}")

            final_df.to_parquet(save_path)
            print(colored(f"Saved satellite weather data for alternative load_id {load_id} to {save_path} ({counter}/{total_len})", "green"))


    if ENABLE_PVOUTPUT_HOUSES:
        ### Has to be loaded in only for FLANDERS_BUILDINGS. Maybe later for fitting factor?
        ###'/Users/brechtgoethals/Library/CloudStorage/GoogleDrive-brecht.goethals23@gmail.com/My Drive/T - GDRIVE/Virtual_houses/Households_w_battery_dataset/random_sample_1_with_load_pv_and_bess_assigned.parquet'
        PVOUTPUT_HOUSES = Path(Path(__file__).parent.parent / "PVOutput" / "PVOutput_metadata" / "PVOutput_metadata_cleaned.parquet")
        PVOUTPUT_HOUSES_df = pd.read_parquet(PVOUTPUT_HOUSES)

        PVOUTPUT_HOUSES_df = PVOUTPUT_HOUSES_df[["system_id", "latitude", "longitude"]] ### Load id is now system_id




        start_DWD = "2026-02-16"
        #end_DWD = (pd.Timestamp.now(tz="Europe/Brussels").normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d") 
        end_DWD = END_SATELLITE #"2026-06-14"
        ### Read bad dates for DWD
        bad_dwd_txt = Path(__file__).parent / "Bad_days_DWD.txt" ### Dates with an X behind it should be filled in, not fully deleted. If no X, completely nan that day
        bad_dwd_lines = bad_dwd_txt.read_text().splitlines()
        bad_dwd_lines = [line.strip() for line in bad_dwd_lines]
        bad_X_dwd_dates = [line.replace("X", "").strip() for line in bad_dwd_lines if line.endswith("X")]
        bad_dwd_dates = [line.strip() for line in bad_dwd_lines if not line.endswith("X")]

        start_JMA = "2025-07-30"
        end_JMA = "2026-03-08" ### 9th is already incomplete
        ### Read bad dates for JMA
        Bad_days_JMA = Path(__file__).parent / "Bad_days_JMA.txt" ### Dates with an X behind it should be filled in, not fully deleted. If no X, completely nan that day
        bad_jma_lines = Bad_days_JMA.read_text().splitlines()
        bad_jma_lines = [line.strip() for line in bad_jma_lines]
        bad_X_jma_dates = [line.replace("X", "").strip() for line in bad_jma_lines if line.endswith("X")] ### do fillna immediately
        bad_jma_dates = [line.strip() for line in bad_jma_lines if not line.endswith("X")] ### delete full day, then fillna

        ### For testing, only keep first one
        #FLANDERS_BUILDINGS_df = FLANDERS_BUILDINGS_df.head(1)
        total_len = len(PVOUTPUT_HOUSES_df)
        counter = 0
        for index, row in PVOUTPUT_HOUSES_df.iterrows():
            system_id = row["system_id"]
            latitude = row["latitude"]
            longitude = row["longitude"]
            save_path = Path(__file__).parent / "Satellite_weather_data" / "PVOutput_houses" / f"system_id_{system_id}_satellite_weather.parquet"

            ### Improved saving
            wanted_start = pd.Timestamp(start_JMA, tz="Europe/Brussels") # Move up
            wanted_end = pd.Timestamp(end_DWD, tz="Europe/Brussels") + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
            wanted_index = pd.date_range(start=wanted_start, end=wanted_end, freq="15min")

            existing_df = None
            missing_start = wanted_start

            if save_path.exists():
                existing_df = pd.read_parquet(save_path)
                existing_df.index = pd.to_datetime(existing_df.index)
                if existing_df.index.tz is None:
                    existing_df.index = existing_df.index.tz_localize("Europe/Brussels")
                else:
                    existing_df.index = existing_df.index.tz_convert("Europe/Brussels")
                existing_df = existing_df.sort_index()
                missing_timestamps = wanted_index.difference(existing_df.index)

                if len(missing_timestamps) == 0:
                    print(f"Satellite weather data for system_id {system_id} already exists and is complete. Skipping.")
                    counter += 1
                    continue

                missing_start = missing_timestamps.min()
                print(colored(f"Satellite weather data for system_id {system_id} exists but is incomplete. Missing data starts from {missing_start}. Fetching missing data.", "yellow"))
            counter += 1
            

            start_DWD_fetch = max(missing_start.normalize(), pd.Timestamp(start_DWD, tz="Europe/Brussels")).strftime("%Y-%m-%d")
            start_JMA_fetch = max(missing_start.normalize(), pd.Timestamp(start_JMA, tz="Europe/Brussels")).strftime("%Y-%m-%d")
            start_EUMETSAT_fetch = missing_start.normalize().strftime("%Y-%m-%d")

            ### Retrieve both, cut JMA to beginning of DWD
            weather_params_dwd = Weather_params_DWD(latitude=latitude, longitude=longitude, apikey=API_KEY, start_date=start_DWD_fetch, end_date=end_DWD)
            weather_params_jma = Weather_params_JMA(latitude=latitude, longitude=longitude, apikey=API_KEY, start_date=start_JMA_fetch, end_date=end_JMA)
            weather_params_eumetsat = Weather_params_EUMETSAT(latitude=latitude, longitude=longitude, apikey=API_KEY, start_date=start_EUMETSAT_fetch, end_date=end_DWD) ### Just get the whole period, have enough credits

            fetcher_dwd = SatelliteWeatherFetcher(asdict(weather_params_dwd))
            fetcher_jma = SatelliteWeatherFetcher(asdict(weather_params_jma))
            fetcher_eumetsat = SatelliteWeatherFetcher(asdict(weather_params_eumetsat))

            dwd_df = fetcher_dwd.run()
            dwd_df_15min = resample_10_to_5_to_15min(dwd_df)
            print("DWD data retrieved and resampled.")
            #print(dwd_df.head()) ### temperature_2m, wind_speed_10m not retrieved. Has to be retrieved from other source!

            if pd.Timestamp(start_JMA_fetch) <= pd.Timestamp(end_JMA):
                jma_df = fetcher_jma.run()
                jma_df_15min = resample_10_to_5_to_15min(jma_df)
                print("JMA data retrieved and resampled.")
                #print(jma_df.head()) ### temperature_2m, wind_speed_10m also not retrieved. Has to be retrieved from other source!
            else:
                jma_df_15min = pd.DataFrame(columns=dwd_df_15min.columns) ### Just empty df with same columns to avoid breaking code
                print("JMA not needed for this missing period.")
                
            eumetsat_df = fetcher_eumetsat.run()
            eumetsat_df_15min = eumetsat_df
            print("EUMETSAT data retrieved.")
            #eumetsat_df_15min = resample_10_to_5_to_15min(eumetsat_df) ### IS NATIVELY 15MIN
            #print(eumetsat_df.head()) 
            #print(eumetsat_df.head()) ### temperature_2m, wind_speed_10m also not retrieved. Has to be retrieved from other source!


            ### Make overall df
            start_overall = missing_start
            end_overall = pd.Timestamp(end_DWD, tz="Europe/Brussels") + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)

            index_overall = pd.date_range(start=start_overall, end=end_overall, freq="15min")
            cols = dwd_df_15min.columns.union(jma_df_15min.columns) ### Took union but could be either one or intersection
            satellite_combined_df = pd.DataFrame(index=index_overall, columns=cols, dtype=float) ### Filled with nans
            satellite_combined_df["source"] = pd.Series(index=index_overall, dtype="string") ### Fix by chat
            satellite_combined_df["source"] = "None"

            dwd_aligned = dwd_df_15min.reindex(index_overall)
            dwd_fill_indices = satellite_combined_df[cols].isna() & dwd_aligned[cols].notna()
            satellite_combined_df[cols] = satellite_combined_df[cols].fillna(dwd_aligned[cols])
            satellite_combined_df.loc[dwd_fill_indices.any(axis=1), "source"] = "DWD" ### Fix by chat

            jma_aligned = jma_df_15min.reindex(index_overall)
            jma_fill_indices = satellite_combined_df[cols].isna() & jma_aligned[cols].notna()
            satellite_combined_df[cols] = satellite_combined_df[cols].fillna(jma_aligned[cols])
            satellite_combined_df.loc[jma_fill_indices.any(axis=1), "source"] = "JMA" ### Fix by chat


            eumetsat_aligned = eumetsat_df_15min.reindex(index_overall)
            ### For bad days in DWD, make them nan
            for bad_date in bad_dwd_dates:
                day_indices = satellite_combined_df.index.date == pd.to_datetime(bad_date).date()
                satellite_combined_df.loc[day_indices, cols] = np.nan
                satellite_combined_df.loc[day_indices, cols] = satellite_combined_df.loc[day_indices, cols].fillna(eumetsat_aligned.loc[day_indices, cols]) #### Now fill in of course
                satellite_combined_df.loc[day_indices, "source"] = "EUMETSAT_REPLACE"
            for bad_date in bad_X_dwd_dates:
                day_indices = satellite_combined_df.index.date == pd.to_datetime(bad_date).date()
                fill_indices = satellite_combined_df.loc[day_indices, cols].isna() & eumetsat_aligned.loc[day_indices, cols].notna() ### For source logic
                satellite_combined_df.loc[day_indices, cols] = satellite_combined_df.loc[day_indices, cols].fillna(eumetsat_aligned.loc[day_indices, cols]) 
                satellite_combined_df.loc[day_indices, "source"] = np.where(fill_indices.any(axis=1), f"EUMETSAT_FILL", satellite_combined_df.loc[day_indices, "source"]) ### Any axis debug by chat

            
            ### For bad days in JMA, make them nan
            for bad_date in bad_jma_dates:
                day_indices = satellite_combined_df.index.date == pd.to_datetime(bad_date).date()
                satellite_combined_df.loc[day_indices, cols] = np.nan
                satellite_combined_df.loc[day_indices, cols] = satellite_combined_df.loc[day_indices, cols].fillna(eumetsat_aligned.loc[day_indices, cols])
                satellite_combined_df.loc[day_indices, "source"] = "EUMETSAT_REPLACE"
            for bad_date in bad_X_jma_dates:
                day_indices = satellite_combined_df.index.date == pd.to_datetime(bad_date).date()
                fill_indices = satellite_combined_df.loc[day_indices, cols].isna() & eumetsat_aligned.loc[day_indices, cols].notna()
                satellite_combined_df.loc[day_indices, cols] = satellite_combined_df.loc[day_indices, cols].fillna(eumetsat_aligned.loc[day_indices, cols])
                satellite_combined_df.loc[day_indices, "source"] = np.where(fill_indices.any(axis=1), f"EUMETSAT_FILL", satellite_combined_df.loc[day_indices, "source"])

            

            ### Temporary nan fillings: 
            # take out all nans where shortwave_radiation is 0 and diffuse and direct are nan, put them to zero too
            zero_indices = (satellite_combined_df["shortwave_radiation"] == 0) & (satellite_combined_df["diffuse_radiation"].isna()) & (satellite_combined_df["direct_normal_irradiance"].isna())
            satellite_combined_df.loc[zero_indices, ["diffuse_radiation", "direct_normal_irradiance"]] = 0
            ### if has nans between 00:00 and 06:00, fill them with 0 as well
            night_indices = (satellite_combined_df.index.time >= pd.to_datetime("00:00").time()) & (satellite_combined_df.index.time < pd.to_datetime("06:00").time())
            night_nan_indices = night_indices & (satellite_combined_df["shortwave_radiation"].isna())
            satellite_combined_df.loc[night_nan_indices, ["shortwave_radiation", "diffuse_radiation", "direct_normal_irradiance"]] = 0

            ### And rest can be filled with fillna for irradiance columns, I checked it manually
            irradiance_cols = ["shortwave_radiation", "diffuse_radiation", "direct_normal_irradiance"]
            satellite_combined_df[irradiance_cols] = satellite_combined_df[irradiance_cols].fillna(0)

            print(satellite_combined_df.head())

        
            ### Print nans as rows, ignoring temp and wind
            nan_rows = satellite_combined_df[satellite_combined_df[["shortwave_radiation", "diffuse_radiation", "direct_normal_irradiance"]].isna().any(axis=1)]
            for idx, row in nan_rows.iterrows():
                print(f"NaN at {idx} for system_id {system_id}: shortwave_radiation={row['shortwave_radiation']}, diffuse_radiation={row['diffuse_radiation']}, direct_normal_irradiance={row['direct_normal_irradiance']}")



            ### Save overall df
            
            save_path.parent.mkdir(parents=True, exist_ok=True)
            # Save logic
            if existing_df is not None:
                satellite_combined_df = pd.concat([existing_df, satellite_combined_df]).sort_index()
                satellite_combined_df = satellite_combined_df[~satellite_combined_df.index.duplicated(keep="last")] ### Last but should be same
            final_df = satellite_combined_df.reindex(wanted_index)
            # Small check 
            missing_timestamps_after = wanted_index.difference(satellite_combined_df.index)
            if len(missing_timestamps_after) > 0:
                raise ValueError(f"After saving, there are still missing timestamps for system_id {system_id}: {missing_timestamps_after}")
            
            final_df.to_parquet(save_path)
            print(colored(f"Saved satellite weather data for system_id {system_id} to {save_path} ({counter}/{total_len})", "green"))