import pandas as pd
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import time
import sys 
from concurrent.futures import ProcessPoolExecutor, as_completed



@dataclass
class Params_model:
    start_training: str
    end_training: str

    HORIZON_STEPS: int = 48*4 ### 48 hours ahead with 15-minute data
    OUTPUT_DIR: Path = Path(Path(__file__).parent.parent / "Dataframes 48h")
    period: str = "15min"




class build_training_df:
    def __init__(self, params: Params_model):
        self.params = params



    def load_data(self, FILE, household):
        df = pd.read_parquet(FILE, columns=[str(household)]) ### Will have to be called per household, has to be put as str for some reason
        ### Data has been preprocessed so not much will have to be done

        ### Check for negative values and nans
        if (df[household] < 0).any():
            raise ValueError(f"Data contains negative values: {df[df[household] < 0]}")
        if df.isna().any().any():
            raise ValueError(f"Data contains nans: {df.isna().sum()} at {df[df.isna().any(axis=1)]}")

        ### Set target
        self.y = df[household]


    def make_features(self):
        self.params.steps_per_hour = int(pd.Timedelta("1h") / pd.Timedelta(self.params.period))
        idx = self.y.index ### Already datetimeindex
        self.X = pd.DataFrame(index=idx)
        ### Add lag features
        self.params.lags = [
            1,2,3,4, 
            self.params.steps_per_hour,
            self.params.steps_per_hour*24, ### 1 day lag
            self.params.steps_per_hour*24*7, ### 1 week lag
        ]
        for lag in self.params.lags:
            self.X[f"lag_{lag}"] = self.y.shift(lag)
            col_name = f"lag_{lag}"
            self.X[f"{col_name}_missing"] = self.X[col_name].isna().astype(int) ### Indicating missing value with int
        self.X["lag_0"] = self.y ### Add zero lag, when called at 22:02 for example, lag_0 brings the data of 21:45 until 22:00!!
        
        ### Difference features
        self.X["diff_0"] = self.y - self.y.shift(1) 
        self.X["diff_1"] = self.y.shift(1) - self.y.shift(2)
        self.X["diff_0_2"] = self.y - self.y.shift(2) ### Added extra

        ### Rolling features
        self.params.roll_windows = [ ### May add more
            self.params.steps_per_hour, 
            self.params.steps_per_hour*24
        ]
        for roll in self.params.roll_windows:
            self.X[f"roll_mean_{roll}"] = self.y.rolling(roll).mean()
            self.X[f"roll_std_{roll}"] = self.y.rolling(roll).std()
            self.X[f"roll_min_{roll}"] = self.y.rolling(roll).min()
            self.X[f"roll_max_{roll}"] = self.y.rolling(roll).max()


        ### Add time features
        dow = idx.dayofweek
        self.X["is_weekend"] = (dow >= 5).astype(int)
        self.X["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        self.X["dow_cos"] = np.cos(2 * np.pi * dow / 7)

        hour = idx.hour
        minute = idx.minute
        min_of_day = hour * 60 + minute
        self.X["mod_sin"] = np.sin(2 * np.pi * min_of_day / (24 * 60))
        self.X["mod_cos"] = np.cos(2 * np.pi * min_of_day / (24 * 60))

        doy = idx.dayofyear
        self.X["doy_sin"] = np.sin(2 * np.pi * doy / 365)
        self.X["doy_cos"] = np.cos(2 * np.pi * doy / 365)

        ### Specific features
        self.X["is_morning_peak"] = ((hour >= 7) & (hour < 10)).astype(int)
        self.X["is_evening_peak"] = ((hour >= 17) & (hour < 20)).astype(int)


        ### No Intraday features because simulation file is already at 15min

    
    def make_future_features_original(self, origins): ### Better name would probably be flatten but will keep adopted from previous script, also outdated, does not return origin times.
        X_rows = []
        Y_rows = []
        future_times = []

        idx = self.y.index
        target_values = self.y.values

        for origin in origins:
            origin_features_dict = self.X.iloc[origin].to_dict() ### Was implemented with iloc before, loc would be safer but will keep for now.
            ### Take it out of inner loop for efficiency of course!!

            for h in range(1, self.params.HORIZON_STEPS + 1):
                future_idx = origin + h ### If implementing with loc thsi has to change to pandas offset
                future_time = idx[future_idx]

                row = origin_features_dict.copy()

                ### Time 
                future_dow = future_time.dayofweek
                row["future_is_weekend"] = int(future_dow >= 5)
                row["future_dow_sin"] = np.sin(2 * np.pi * future_dow / 7)
                row["future_dow_cos"] = np.cos(2 * np.pi * future_dow / 7)

                future_hour = future_time.hour
                future_minute = future_time.minute

                min_of_day = future_hour * 60 + future_minute
                row["future_mod_sin"] = np.sin(2 * np.pi * min_of_day / (1440))
                row["future_mod_cos"] = np.cos(2 * np.pi * min_of_day / (1440))
                
                row["future_doy_sin"] = np.sin(2 * np.pi * future_time.dayofyear / 365)
                row["future_doy_cos"] = np.cos(2 * np.pi * future_time.dayofyear / 365)

                ### Step features
                row["h_step"] = h
                row["h_step_sin"] = np.sin(2 * np.pi * h / self.params.HORIZON_STEPS)
                row["h_step_cos"] = np.cos(2 * np.pi * h / self.params.HORIZON_STEPS)
                row["h_norm"] = h / self.params.HORIZON_STEPS ### Normalized step, added by CHAT
                row["h_hours"] = h * (pd.Timedelta(self.params.period).total_seconds() / 3600) ### Step in hours, added by CHAT

                ### Morning and evening peaks
                row["future_is_morning_peak"] = int((future_hour >= 7) & (future_hour < 10))
                row["future_is_evening_peak"] = int((future_hour >= 17) & (future_hour < 20))

                X_rows.append(row)
                Y_rows.append(target_values[future_idx])
                future_times.append(future_time)

        X_with_future = pd.DataFrame(X_rows)
        Y_with_future = pd.Series(Y_rows, name="target")
        future_times = pd.DatetimeIndex(future_times) ### For plotting if wanted


        # Check for nans
        if X_with_future.isna().any().any():
            raise ValueError(f"X_with_future contains nans: {X_with_future.isna().sum()} at {X_with_future[X_with_future.isna().any(axis=1)]}")
        if Y_with_future.isna().any():
            raise ValueError(f"Y_with_future contains nans: {Y_with_future.isna().sum()} at {Y_with_future[Y_with_future.isna()]}")
        
        return X_with_future, Y_with_future, future_times
        

    def make_future_features_vectorized(self, origins): ### This is the same function but rewritten by chat to avoid computationally expensive for origin -and h loop. Gives massive improvement
        idx = self.y.index
        target_values = self.y.to_numpy()

        H = self.params.HORIZON_STEPS
        origins = np.asarray(origins, dtype=np.int64)

        ### Make one shot without iterations
        origin_rep = np.repeat(origins, H) # origin_rep = [100,100,...,100,101,101,...,101]
        h_steps = np.tile(np.arange(1, H + 1, dtype=np.int16), len(origins)) # h_steps = [1,2,...,96,1,2,...,96]

        future_idx = origin_rep + h_steps

        if future_idx.max() >= len(idx):
            raise ValueError("future_idx goes beyond available dataframe length.")

        ### This replaces:
        X_with_future = self.X.iloc[origin_rep].reset_index(drop=True).copy() ### Set row for each repeated index, look at origin_rep to understand

        origin_times = pd.DatetimeIndex(idx[origin_rep]) ### Store them too in order to be able to slice correctly
        future_times = pd.DatetimeIndex(idx[future_idx])

        ### Target
        Y_with_future = pd.Series(target_values[future_idx], name="target") ### Is now of length origins*H

        ### Future time features, vectorized
        future_dow = future_times.dayofweek.to_numpy()
        future_hour = future_times.hour.to_numpy()
        future_minute = future_times.minute.to_numpy()
        future_doy = future_times.dayofyear.to_numpy()

        min_of_day = future_hour * 60 + future_minute

        X_with_future["future_is_weekend"] = (future_dow >= 5).astype(np.int8)
        X_with_future["future_dow_sin"] = np.sin(2 * np.pi * future_dow / 7)
        X_with_future["future_dow_cos"] = np.cos(2 * np.pi * future_dow / 7)

        X_with_future["future_mod_sin"] = np.sin(2 * np.pi * min_of_day / 1440)
        X_with_future["future_mod_cos"] = np.cos(2 * np.pi * min_of_day / 1440)

        X_with_future["future_doy_sin"] = np.sin(2 * np.pi * future_doy / 365)
        X_with_future["future_doy_cos"] = np.cos(2 * np.pi * future_doy / 365)

        ### Step features
        X_with_future["h_step"] = h_steps
        X_with_future["h_step_sin"] = np.sin(2 * np.pi * h_steps / H)
        X_with_future["h_step_cos"] = np.cos(2 * np.pi * h_steps / H)
        X_with_future["h_norm"] = h_steps / H
        X_with_future["h_hours"] = h_steps * (pd.Timedelta(self.params.period).total_seconds() / 3600)

        ### Morning and evening peaks
        X_with_future["future_is_morning_peak"] = ((future_hour >= 7) & (future_hour < 10)).astype(np.int8)
        X_with_future["future_is_evening_peak"] = ((future_hour >= 17) & (future_hour < 20)).astype(np.int8)


        ### Leave nan logic, although might be heavy on crash
        # Check for nans
        if X_with_future.isna().any().any():
            raise ValueError(f"X_with_future contains nans: {X_with_future.isna().sum()} at {X_with_future[X_with_future.isna().any(axis=1)]}")
        if Y_with_future.isna().any():
            raise ValueError(f"Y_with_future contains nans: {Y_with_future.isna().sum()} at {Y_with_future[Y_with_future.isna()]}")

        return X_with_future, Y_with_future, future_times, origin_times
    

    def build_origins(self): ### Version of this function was made by chat in earlier version
        idx = self.y.index
        positions = np.arange(len(idx) - self.params.HORIZON_STEPS) ### To stay within the possible origins
        origin_times = idx[positions]

        horizon_delta = self.params.HORIZON_STEPS * pd.Timedelta(self.params.period)

        train_start_time = pd.to_datetime(self.params.start_training).tz_localize(idx.tz) 
        train_end_time = pd.to_datetime(self.params.end_training).tz_localize(idx.tz) - horizon_delta
        self.train_origins = positions[(origin_times >= train_start_time) & (origin_times < train_end_time)]

        if len(self.train_origins) == 0:
            raise ValueError("No training origins were created. Check start_training/end_training and HORIZON_STEPS.")
        
        ### Don't even bother about eval and test origins


    def Save_df(self, X, Y, future_times, origin_times, household):
        output_folder = self.params.OUTPUT_DIR / str(household)
        output_folder.mkdir(parents=True, exist_ok=True)

        df_out = X.copy()

        df_out["target"] = Y.to_numpy()
        df_out["future_time"] = future_times
        df_out["origin_time"] = origin_times

        output_file = output_folder / f"training_df_household_{household}_{self.params.HORIZON_STEPS}.parquet"
        #print(f"Head of df to save for household {household}:\n{df_out.head()}")
        df_out.to_parquet(output_file, index=False)

        print(f"Saved training dataframe for household {household}")


    def build_pipeline(self, FILE, household):
        start_time = time.time()
        print("Started for household:", household)

        self.load_data(FILE, household)
        print("Data loaded")
        self.make_features()
        print("Features created")
        self.build_origins()
        print("Origins built")
        X, Y, future_times, origin_times = self.make_future_features_vectorized(self.train_origins) ### Future times is to reconstruct back to original timestamps if wanted.
        print("Future features created")
        self.Save_df(X, Y, future_times, origin_times, household)
        print("Training dataframe saved")

        


        end_time = time.time()
        print(f"Time taken to build pipeline: {end_time - start_time:.2f} seconds")


        """print(f"X head: {X.head()}")
        print(f"Shape of X: {X.shape}")
        print(f"Y head: {Y.head()}")
        print(f"Shape of Y: {Y.shape}")
        print(f"Future times head: {future_times[:100]}")
        print(f"Shape of future times: {future_times.shape}")
        print(f"Range of X: {X.index.min()} - {X.index.max()}")
        print(f"Range of Y: {Y.index.min()} - {Y.index.max()}")
        print(f"Number of samples: {len(X)}")"""






def get_completed_households(params: Params_model):
    completed_households = []

    if not params.OUTPUT_DIR.exists():
        return completed_households
    
    ### Check for each household if the file exists by iteration over the output directory
    for household_folder in params.OUTPUT_DIR.iterdir():
        if household_folder.is_dir():
            household = household_folder.name
            output_file = household_folder / f"training_df_household_{household}_{params.HORIZON_STEPS}.parquet"
            if output_file.exists():
                completed_households.append(int(household))
    return completed_households



def process_one_household(household, params, FILE): ### For multiprocessing, needs a worker function
    builder = build_training_df(params)
    builder.build_pipeline(FILE, household)
    return household





if __name__ == "__main__":

    """### Build full pool of households to build training dfs for
    households_load = [i for i in range(301, 601)] ### 301-600 are load households
    households_HP = [i for i in range(901, 1201)] ### 901-1200 are heat pump households
    households_EV = [i for i in range(1501, 1801)] ### 1501-1800 are EV households
    #households_HP_EV = [i for i in range(2101, 2401)] ### 2101-2400 are HP+EV households

    all_households = households_load + households_HP + households_EV #a+ households_HP_EV
    completed_households = get_completed_households(Params_model(start_training="", end_training=""))

    households_to_process = list(set(all_households) - set(completed_households))"""


    households_to_process = [9999900] ### Overwrite
    

    length_to_process = len(households_to_process)
    print(f"Number of households to process: {len(households_to_process)}")

    start_training="2023-01-20 00:00:00"
    end_training="2024-12-31 23:59:59"
    FILE = Path(Path(__file__).parent.parent / "Fluvius 2400" / "Prepped" / "All_prepared_2_years_with_aggregate.parquet")

    params = Params_model(start_training=start_training, end_training=end_training) ### Start from on february, first timestamp will be around 2024-01-20 00:00:00 which will then use around 11 months of data.


    MULTIPROCESSING = True
    if not MULTIPROCESSING:
        counter = 0
        for household in households_to_process:
            counter += 1
            print(f"\nProcessing household {household} ({counter}/{length_to_process})")
            builder = build_training_df(params)
            builder.build_pipeline(FILE, household)


    else:
        MAX_WORKERS = 2 if sys.platform == "win32" else 2 if sys.platform == "darwin" else 2
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor: ### Uses newer process pool executor instead of multiprocessing, should be more efficient and easier to use.
            futures = {executor.submit(process_one_household, household, params, FILE): household for household in households_to_process}

            for counter, future in enumerate(as_completed(futures), 1):
                household = futures[future]
                try:
                    result = future.result()
                    print(f"\nCompleted household {result} ({counter}/{length_to_process})")
                except Exception as e:
                    raise RuntimeError(f"Error processing household {household}: {e}")

    