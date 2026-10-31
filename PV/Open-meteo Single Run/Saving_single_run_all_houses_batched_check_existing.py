import openmeteo_requests
from pathlib import Path
import pandas as pd
import requests_cache
from retry_requests import retry
from dataclasses import dataclass, field, asdict
import sys


@dataclass
class Weather_params:
    latitude: float
    longitude: float
    apikey: str
    #url: str = "https://single-runs-api.open-meteo.com/v1/forecast"
    url: str = "https://customer-single-runs-api.open-meteo.com/v1/forecast" ### Commercial url
    timezone: str = "Europe/Brussels"
    
    models: str = "icon_seamless"
    minutely_15: list = field(default_factory=lambda: ["shortwave_radiation", "diffuse_radiation", "direct_normal_irradiance", "temperature_2m", "wind_speed_10m"]) ### Have to do this again in order to make exclusive object per dataclass as list is mutable.
    forecast_days: int = 3




class Single_run_fetcher():
    def __init__(self, params: Weather_params):
        self.url = params.pop("url") ### Does this work?
        self.params = params


    def fetch_weather_data(self, run_origin):
        current_directory = Path(__file__).resolve().parent
        cache_dir = current_directory / "Weather_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_db = cache_dir / "openmeteo_cache.sqlite" ### Fix for linux

        cache_session = requests_cache.CachedSession(str(cache_db), expire_after=1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        try:
            params_with_origin = self.params.copy()
            params_with_origin["run"] = run_origin
            responses = openmeteo.weather_api(self.url, params=params_with_origin)
            weather_df = self.weather_data_to_dataframe(responses)
            return weather_df ### To avoid cached data in script
        except Exception as e:
            raise RuntimeError(f"Failed to fetch weather data: {e}")
        

    def weather_data_to_dataframe(self, responses):
        response = responses[0]
        minutely_15 = response.Minutely15() ### If want to add hourly or daily variables, adjust here as needed!!

        idx_utc = pd.date_range(
            start=pd.to_datetime(minutely_15.Time(), unit="s", utc=True),
            end=pd.to_datetime(minutely_15.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=minutely_15.Interval()),
            inclusive="left",
        )

        idx = idx_utc.tz_convert(self.params["timezone"]) ### Fix by chat

        data = {"date": idx}
        for i, var in enumerate(self.params["minutely_15"]):
            data[var] = minutely_15.Variables(i).ValuesAsNumpy()

        return pd.DataFrame(data).set_index("date")
    

    def run(self, run_origin, to_brussels_time=True):
        try: 
            weather_df = self.fetch_weather_data(run_origin)
            if to_brussels_time:
                weather_df.index = weather_df.index.tz_convert("Europe/Brussels")
            return weather_df
        except Exception as e:
            raise RuntimeError(f"Weather model failed:\n{e}")
        

    
    def process_df(self, weather_df, origin):
        df_to_save = weather_df.copy()
        df_to_save = df_to_save.reset_index().rename(columns={"date": "forecasted_time"})
        df_to_save.index = pd.Index([origin] * len(df_to_save), name="timestamp")
        ### Remove entry of origin
        df_to_save = df_to_save[df_to_save["forecasted_time"] != origin]
        return df_to_save
    

    def fetch_weather_batch(self, batch_df, run_origin): ### By Chat
        current_directory = Path(__file__).resolve().parent
        cache_dir = current_directory / "Weather_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_db = cache_dir / "openmeteo_cache.sqlite"

        cache_session = requests_cache.CachedSession(str(cache_db), expire_after=1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        params_with_origin = self.params.copy()
        params_with_origin["latitude"] = ",".join(batch_df["latitude"].astype(str))
        params_with_origin["longitude"] = ",".join(batch_df["longitude"].astype(str))
        params_with_origin["run"] = run_origin

        responses = openmeteo.weather_api(self.url, params=params_with_origin)

        weather_by_system_id = {}

        for system_id, response in zip(batch_df["system_id"], responses):
            weather_df = self.weather_data_to_dataframe([response])
            weather_by_system_id[system_id] = weather_df

        return weather_by_system_id


        




def make_batches(df, batch_size):
    n_rows = len(df)

    for start_idx in range(0, n_rows, batch_size):
        end_idx = start_idx + batch_size
        batch_df = df.iloc[start_idx:end_idx]

        yield batch_df ### Doesn't terminate the function, saves it state still


def get_save_file(folder_name, model_name, house_id):
    return (
        Path(__file__).parent
        / "Weather_saved"
        / folder_name
        / f"{model_name}"
        / f"Weather_forecasts_saved_{house_id}.parquet"
    )


def origin_already_saved(save_file, origin):
    if not save_file.exists():
        return False

    existing_data = pd.read_parquet(save_file)
    return origin in existing_data.index


def keep_only_missing_houses(batch_df, origin, folder_name, model_name):
    keep_rows = []

    for _, row in batch_df.iterrows():
        house_id = row["system_id"]
        save_file = get_save_file(folder_name, model_name, house_id)

        if origin_already_saved(save_file, origin):
            print(f"Data for {house_id} at {origin} already exists, not requesting.")
            keep_rows.append(False)
        else:
            keep_rows.append(True)

    return batch_df.loc[keep_rows].copy()

if __name__ == "__main__":
    if sys.platform == "darwin":
        API_KEY = pd.read_csv("/Users/brechtgoethals/API_OPENMETEO.txt", header=None).iloc[0, 0]
    elif sys.platform == "win32":
        API_KEY = pd.read_csv(r"C:\Users\brech\API_OPENMETEO.txt", header=None).iloc[0, 0]
    ### Get systems and coordinates. Flanders_buildings are saved with the "load_id", corresponding to fluvius 2400 dataset.
    FILE_SAMPLE_FLANDERS_BUILDINGS = Path(__file__).parent.parent.parent / "Virtual_houses" / "Building_addresses_Flanders" / "random_sample_1_with_load_assigned.parquet"
    FILE_METADATA_PVOUTPUT = Path(__file__).parent.parent / "PVOutput" / "PVOutput_metadata" / "PVOutput_metadata.parquet"



    df_sample_flanders_buildings = pd.read_parquet(FILE_SAMPLE_FLANDERS_BUILDINGS)
    ### Only keep "load_id", "latitude", "longitude" columns
    df_sample_flanders_buildings = df_sample_flanders_buildings[["load_id", "latitude", "longitude"]] ### Already has int, float, float

    df_metadata_pvoutput = pd.read_parquet(FILE_METADATA_PVOUTPUT)
    ### Only keep "system_id", "latitude", "longitude" columns
    df_metadata_pvoutput = df_metadata_pvoutput[["system_id", "latitude", "longitude"]]
    ### Lat and lon to floats, system_id to int
    df_metadata_pvoutput["latitude"] = df_metadata_pvoutput["latitude"].astype(float)
    df_metadata_pvoutput["longitude"] = df_metadata_pvoutput["longitude"].astype(float)
    df_metadata_pvoutput["system_id"] = df_metadata_pvoutput["system_id"].astype(int)


    params = Weather_params(latitude=0, longitude=0, apikey=API_KEY)
    fetcher = Single_run_fetcher(asdict(params))
        

    start_timestamp = pd.Timestamp("2026-06-20 00:00", tz="UTC")#"2025-11-24 02:00"#"2025-07-22 02:00" ### Manually adjust this according to Retrieveindex_ranges.py script
    end_timestamp = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=params.forecast_days+1)
    print(f"Retrieving until {end_timestamp}")

    #end_timestamp = pd.Timestamp("2026-06-23 00:00", tz="UTC") ### Set for now to not retrieve everything again each time. ------ HAS TO BE REMOVED
    
    origins_utc = pd.date_range(start=start_timestamp, end=end_timestamp.floor("6h"), freq="6h") ### Have to put this in UTC, otherwise big crash in winter after DST change

    error_strings = ["The requested model run is not available", "No such key", "No value associated with key", "Unexpected error while streaming data"]


    ### First cache pvoutput based on systemid, longitude, latitude.
    BATCH_SIZE = 150

    

    ENABLE_PV = True
    if ENABLE_PV:
        print("Sample of Flanders buildings:")
        print(df_sample_flanders_buildings.head())
        for origin_utc_ in origins_utc:
            run_origin = origin_utc_.strftime("%Y-%m-%dT%H:%M")
            origin = origin_utc_.tz_convert("Europe/Brussels")

            print(f"Fetching batched weather at origin {origin} and run_origin {run_origin} for PVOutput households")

            for batch_df in make_batches(df_metadata_pvoutput, BATCH_SIZE):

                batch_df = keep_only_missing_houses(
                    batch_df=batch_df,
                    origin=origin,
                    folder_name="PVOutput_houses",
                    model_name=params.models,
                )

                if batch_df.empty:
                    print(f"All PVOutput houses in this batch already have {origin}, skipping API request.")
                    continue

                try:
                    weather_by_system_id = fetcher.fetch_weather_batch(batch_df, run_origin)

                except Exception as e:
                    if any(error_string in str(e) for error_string in error_strings):
                        #print(f"Skipping unavailable/problematic run {run_origin}: {e}")
                        continue
                    else:
                        raise

                for system_id, weather_df in weather_by_system_id.items():
                    SAVE_FILE = (
                        Path(__file__).parent / "Weather_saved" / "PVOutput_houses" / f"{params.models}" / f"Weather_forecasts_saved_{system_id}.parquet")

                    df_to_save = fetcher.process_df(weather_df, origin)

                    if SAVE_FILE.exists():
                        existing_data = pd.read_parquet(SAVE_FILE)

                        if origin in existing_data.index:
                            #print(f"Data for system {system_id} at {origin} already exists, skipping.")
                            continue

                        combined_data = pd.concat([existing_data, df_to_save], axis=0)
                        combined_data.to_parquet(SAVE_FILE)

                    else:
                        SAVE_FILE.parent.mkdir(parents=True, exist_ok=True)
                        df_to_save.to_parquet(SAVE_FILE)

                    print(f"Saved weather for system {system_id} at origin {origin}")


    ENABLE_FLANDERS_BUILDINGS = True
    if ENABLE_FLANDERS_BUILDINGS:
        print("Metadata of PVOutput systems:")
        print(df_metadata_pvoutput.head())
        ### Secondly, cache Flanders buildings

        df_flanders_for_weather = df_sample_flanders_buildings.rename(
            columns={"load_id": "system_id"}
        )

        for origin_utc_ in origins_utc:
            run_origin = origin_utc_.strftime("%Y-%m-%dT%H:%M")
            origin = origin_utc_.tz_convert("Europe/Brussels")

            print(f"Fetching batched weather for Flanders buildings at origin {origin} and run_origin {run_origin}")

            for batch_df in make_batches(df_flanders_for_weather, BATCH_SIZE):

                batch_df = keep_only_missing_houses(
                    batch_df=batch_df,
                    origin=origin,
                    folder_name="Building_Flanders_houses",
                    model_name=params.models,
                )

                if batch_df.empty:
                    print(f"All Flanders buildings in this batch already have {origin}, skipping API request.")
                    continue

                try:
                    weather_by_load_id = fetcher.fetch_weather_batch(batch_df, run_origin)

                except Exception as e:
                    if any(error_string in str(e) for error_string in error_strings):
                        #print(f"Skipping unavailable/problematic run {run_origin}: {e}")
                        continue
                    else:
                        raise

                for load_id, weather_df in weather_by_load_id.items():
                    SAVE_FILE = (
                        Path(__file__).parent
                        / "Weather_saved"
                        / "Building_Flanders_houses"
                        / f"{params.models}"
                        / f"Weather_forecasts_saved_{load_id}.parquet"
                    )

                    df_to_save = fetcher.process_df(weather_df, origin)

                    if SAVE_FILE.exists():
                        existing_data = pd.read_parquet(SAVE_FILE)

                        if origin in existing_data.index:
                            #print(f"Data for building {load_id} at {origin} already exists, skipping.")
                            continue

                        combined_data = pd.concat([existing_data, df_to_save], axis=0)
                        combined_data.to_parquet(SAVE_FILE)

                    else:
                        SAVE_FILE.parent.mkdir(parents=True, exist_ok=True)
                        df_to_save.to_parquet(SAVE_FILE)

                    print(f"Saved weather for Flanders building {load_id} at origin {origin}")



    ENABLE_FLANDERS_BUILDINGS_ALTERNATIVE = True

    if ENABLE_FLANDERS_BUILDINGS_ALTERNATIVE:
        FILE_SAMPLE_FLANDERS_BUILDINGS_ALTERNATIVE = Path(__file__).parent.parent.parent / "Virtual_houses" / "Building_addresses_Flanders" / "Alternative_virtual_houses_samples" / "random_sample_MAX_DISTANCE_15_rs_40_nclosest_3_samples_1200_with_loads_and_pv_and_bess_assigned.parquet"
        df_sample_flanders_buildings_alternative = pd.read_parquet(FILE_SAMPLE_FLANDERS_BUILDINGS_ALTERNATIVE)
        ### Only keep "load_id", "latitude", "longitude" columns
        df_sample_flanders_buildings_alternative = df_sample_flanders_buildings_alternative[["load_id", "latitude", "longitude"]] ### Already has int, float, float
        print("Sample of Flanders buildings alternative:")
        print(df_sample_flanders_buildings_alternative.head())

        df_flanders_for_weather = df_sample_flanders_buildings_alternative.rename(
            columns={"load_id": "system_id"}
        )

        for origin_utc_ in origins_utc:
            run_origin = origin_utc_.strftime("%Y-%m-%dT%H:%M")
            origin = origin_utc_.tz_convert("Europe/Brussels")

            print(f"Fetching batched weather for Flanders buildings at origin {origin} and run_origin {run_origin}")

            for batch_df in make_batches(df_flanders_for_weather, BATCH_SIZE):

                batch_df = keep_only_missing_houses(
                    batch_df=batch_df,
                    origin=origin,
                    folder_name="Building_Flanders_houses_ALTERNATIVE",
                    model_name=params.models,
                )

                if batch_df.empty:
                    print(f"All Flanders buildings in this batch already have {origin}, skipping API request.")
                    continue

                try:
                    weather_by_load_id = fetcher.fetch_weather_batch(batch_df, run_origin)

                except Exception as e:
                    if any(error_string in str(e) for error_string in error_strings):
                        #print(f"Skipping unavailable/problematic run {run_origin}: {e}")
                        continue
                    else:
                        raise

                for load_id, weather_df in weather_by_load_id.items():
                    SAVE_FILE = (Path(__file__).parent / "Weather_saved" / "Building_Flanders_houses_ALTERNATIVE" / f"{params.models}" / f"Weather_forecasts_saved_{load_id}.parquet")

                    df_to_save = fetcher.process_df(weather_df, origin)

                    if SAVE_FILE.exists():
                        existing_data = pd.read_parquet(SAVE_FILE)

                        if origin in existing_data.index:
                            #print(f"Data for building {load_id} at {origin} already exists, skipping.")
                            continue

                        combined_data = pd.concat([existing_data, df_to_save], axis=0)
                        combined_data.to_parquet(SAVE_FILE)

                    else:
                        SAVE_FILE.parent.mkdir(parents=True, exist_ok=True)
                        df_to_save.to_parquet(SAVE_FILE)

                    print(f"Saved weather for Flanders building {load_id} at origin {origin}")