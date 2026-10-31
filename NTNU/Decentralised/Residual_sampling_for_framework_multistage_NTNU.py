import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import json
import joblib
import time
import os
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRATCH_DIR = Path("/cluster/work/brechtg")
LOAD_TRAINING_DIR = SCRATCH_DIR / "thesis_load_training"
LOAD_MODEL_BASE_DIR = LOAD_TRAINING_DIR / "Models" / "48h" / "Deterministic"
LOAD_TIMESTAMPS_FILE = LOAD_TRAINING_DIR / "Forecast" / "Timestamps.json"
LOAD_DATA_FILE = LOAD_TRAINING_DIR / "Fluvius 2400" / "Prepped" / "All_prepared_2_years_with_aggregate.parquet"

"""
Code similar to Save_forecasts.py but for framework.

Notes:
Don't cache in all files for one virtual house, MPC will be slow so better to leave ram usage low.

Preceding codes:
Code relies on existing joblib models, these are made by Train_models_Deterministic_multiprocessing.py
"""
@dataclass
class Load_Forecaster_Params: ### Can be moved easily
    ### Only training dates are required. Eval/test dates are optional.
    start_training: str
    end_training: str

    HORIZON_STEPS: int = 48*4 ### 48 hours
    OUTPUT_DIR: Path = Path(Path(__file__).parent / "Models" / "Load")
    period: str = "15min"

    boosting_type: str = "gbdt" ### gbdt or dart
    objective: str = "regression"
    n_estimators: int = 1000
    learning_rate: float = 0.06
    num_leaves: int = 31
    max_bin: int = 31

    max_depth: int = 25
    min_child_samples: int = 20#50### 2
    min_child_weight: float = 1e-3
    subsample: float = 0.8 ### Use % of data for each tree
    subsample_freq: int = 1 ### Perform subsampling at every x iteration (0 is disable)
    colsample_bytree: float = 0.8 ### Use % of features for each tree.
    reg_alpha: float = 0.0 ### L1 regularization
    reg_lambda: float = 2.0 ### L2 regularization

    random_state: int = 42
    verbose: int = -1
    early_stopping_rounds: int = 500 ### Not used but will leave in case I want to use it later
    
    eval_metric: str = "rmse"


    ### Nan logic won't have to be used anymore.
    Use_extra_weigth_recent_data: bool = True
    alpha_extra_weight_recent_data: float = 0.06 # 0.25 ### Formula: weight = np.exp(-alpha * (h_step - 1)), higher --> more weight on recent data 

    n_jobs: int = -1



"""
USES 48H FILES NOW
For a root_forecast, use predictions.parquet file and the code from Residual_sampling.py to get scenarios for the next 48h. These should then all be cut off at the branching timestep (one before).
Then, should use the original load data file (2 years prepped file and read in with the specific column and cut to two weeks before origin let's say), then to make children from parent, use code from Training_dfs.py and stitch the original load data until origin together with the parent trajectory until branching timestep. Then make the features for the branching timestep, predict and add scenarios.
"""


class LoadResidualSampler:
    def __init__(self, household: int, exclude_before_ts: str): ### Just as a reminder to myself to exclude that part from simulation
        self.household = household
        self.params = Load_Forecaster_Params(start_training=None, end_training=None) ### Doesn't matter for params string
        params_string = f"{self.params.boosting_type}_{self.params.objective}_{self.params.n_estimators}_{self.params.learning_rate}_{self.params.num_leaves}_{self.params.max_bin}_{self.params.max_depth}_{self.params.min_child_samples}_{self.params.min_child_weight}_{self.params.subsample}_{self.params.subsample_freq}_{self.params.colsample_bytree}_{self.params.reg_alpha}_{self.params.reg_lambda}"
        self.model_folder = LOAD_MODEL_BASE_DIR / params_string / f"Household_{household}" ### And then according to training timestamps.
        self.timestamps = self.init_make_training_timestamps(exclude_before_ts) ### Loaded in as DatetimeIndex

        ### For root forecaster. 
        # Predictions file was the one with all cols like pred_h_1, target_h_1 etc. Holds columns like analyse_timestamp and last_training_timestamp 
        self.predictions_df = self.init_cache_predictions_file(Path(self.model_folder / "predictions.parquet"))

        ### For batch forecaster.
        raw_load_data_file = LOAD_DATA_FILE
        self.raw_load_data_df = pd.read_parquet(raw_load_data_file, columns=[str(self.household)]) ### Has to be read in as string although they are saved as integers. Still a df, not a series
        self.raw_load_data_df.columns = self.raw_load_data_df.columns.astype(str) ### Just add this to avoid bullshit once and for all

        
        


    def init_cache_predictions_file(self, file_path: Path):
        if not file_path.exists():
            raise FileNotFoundError(f"Predictions file not found: {file_path}")
        predictions_df = pd.read_parquet(file_path)
        predictions_df["analyse_timestamp"] = pd.to_datetime(predictions_df["analyse_timestamp"])
        predictions_df = predictions_df.set_index("analyse_timestamp").sort_index()
        is_feb_29 = (predictions_df.index.month == 2) & (predictions_df.index.day == 29) ### Have to take it out again for shifting
        #return predictions_df[~is_feb_29].copy() 
        return predictions_df.copy()



    def init_make_training_timestamps(self, exclude_before_ts: str):
        JSON_FILE = LOAD_TIMESTAMPS_FILE
        retry_counter = 0
        while True: ### Fails from time to time, just have to try again a few times
            try:
                with open(JSON_FILE, 'r') as f:
                    timestamps_dict = json.load(f)
                break
            except OSError: ### FileNotFoundError is a subclass of OSError so this catches both
                print(f"JSON file not found: {JSON_FILE}. Retrying in 5 seconds...")
                time.sleep(5)
                retry_counter += 1
                if retry_counter >= 10:
                    raise FileNotFoundError(f"Failed to find JSON file after {retry_counter} attempts: {JSON_FILE}")
                continue

        Forecast_Hour = timestamps_dict["Forecast_Hour"]
        Timezone = timestamps_dict["Timezone"]
        training_days = timestamps_dict["Training_Timestamps"]

        training_timestamps = []
        for month, days in training_days.items():
            for day in days:
                ts = pd.Timestamp(f"2024-{month}-{day} {Forecast_Hour}", tz=Timezone)
                training_timestamps.append(ts)
        
        timestamps = pd.DatetimeIndex(training_timestamps).sort_values()
        ### Filter out timestamps before exclude_before_ts
        timestamps = timestamps[timestamps >= pd.Timestamp(exclude_before_ts, tz=Timezone)]


        return timestamps
    



    def predict_for_root_timestamp_with_analogs(self, root_timestamp: pd.Timestamp, n_scenarios: int, same_time: bool): ### Can just use predictions file
        row = self.predictions_df.loc[root_timestamp]

        pred_cols = [f"pred_h_{h}" for h in range(1, self.params.HORIZON_STEPS + 1)]
        target_cols = [f"target_h_{h}" for h in range(1, self.params.HORIZON_STEPS + 1)]

        forecast = row[pred_cols].to_numpy(dtype=float)
        target = row[target_cols].to_numpy(dtype=float)
        last_training_timestamp = row["last_training_timestamp"]

        future_index = pd.date_range(start=root_timestamp + pd.Timedelta(self.params.period), periods=self.params.HORIZON_STEPS, freq=self.params.period, tz="Europe/Brussels")
        future_index.name = "future_time"
        if n_scenarios == 0:
            current_forecast_df = pd.DataFrame({"current_forecast": forecast}, index=future_index)
            target_df = pd.DataFrame({"target": target}, index=future_index)
            return None, None, None, current_forecast_df, target_df ### No analogs requested
        
        ### Analogs for the root timestamp.
        prediction_df = self.predictions_df.copy() ### Remmeber that index is already analyse_timestamp
        # Everything after root timestamp should be put a year back, do this by naive like before. Should be checked maybe for dst changes
        root_timestamp_naive = root_timestamp.tz_localize(None)
        prediction_df["pseudo_analyse_timestamp"] = prediction_df.index.tz_localize(None)
        mask_after_root = prediction_df.index > root_timestamp
        prediction_df.loc[mask_after_root, "pseudo_analyse_timestamp"] = prediction_df.loc[mask_after_root, "pseudo_analyse_timestamp"] - pd.DateOffset(years=1) ### Shift back one year

        start_interval = root_timestamp_naive - pd.DateOffset(months=11)
        end_timestamp = root_timestamp_naive - pd.DateOffset(days=1) ### Make sure forecasts are complete

        archive_df = prediction_df[(prediction_df["pseudo_analyse_timestamp"] >= start_interval) & (prediction_df["pseudo_analyse_timestamp"] <= end_timestamp) & (prediction_df.index != root_timestamp)].copy() ### Exclude current forecast of course
        if same_time:
            archive_df = archive_df[archive_df["pseudo_analyse_timestamp"].dt.time == root_timestamp_naive.time()] ### Only keep the same time of day as the root timestamp

        if archive_df.empty:
            raise ValueError(f"No analogs found for root timestamp {root_timestamp}. Check the data and the time range.")

        ### Get closest analogs based on RMSE
        forecast_matrix = archive_df[pred_cols].to_numpy(dtype=float)
        rmse_per_row = np.sqrt(np.mean((forecast_matrix - forecast) ** 2, axis=1)) ### Make sure to put across column axis

        archive_df["similiarity_RMSE"] = rmse_per_row
        closest = archive_df.sort_values("similiarity_RMSE").head(n_scenarios)

        residuals = (closest[target_cols].to_numpy(dtype=float) - closest[pred_cols].to_numpy(dtype=float)) ### Still on the same row from predictions_df (shape (n_scenarios, 96))

        scenarios = forecast.reshape(1, -1) + residuals ### Easy by chat, turn (96,) into (1, 96)

        # Assign index immediately, made before for early return

        ### Now all convert to df
        current_forecast_df = pd.DataFrame({"current_forecast": forecast}, index=future_index) ### This always has to happen, also for further scenarios!
        target_df = pd.DataFrame({"target": target}, index=future_index)
        residuals_df = pd.DataFrame(residuals.T, index=future_index, columns=[f"residual_{i+1}" for i in range(residuals.shape[0])])
        scenarios_df = pd.DataFrame(scenarios.T, index=future_index, columns=[f"scenario_{i+1}" for i in range(scenarios.shape[0])])

        return closest, residuals_df, scenarios_df, current_forecast_df, target_df



        

    


    def predict_for_branch_timestamp_with_analogs(self, origin_timestamp: pd.Timestamp, branch_timestamp: pd.Timestamp, parent_trajectory: pd.DataFrame, n_scenarios: int, same_time: bool): ### Have to add parent trajectory here. Or add multiple immediately for batch. Will use several times same data up until the origin
        # Load the model that will be used. Should be based on origin
        #last_training_timestep = self.timestamps[self.timestamps <= origin_timestep].max() ### Get the last trained model. Not good, creates future leakage when model changes (model trained on that data)
        knowledge_timestamp = origin_timestamp - pd.Timedelta(self.params.period)
        last_training_timestep = self.timestamps[self.timestamps <= knowledge_timestamp].max() 
        if pd.isna(last_training_timestep):
            raise ValueError(f"No training timestamp found before origin timestamp {origin_timestamp}. Check the timestamps and the data.") ### Extra check by chat
        model_file = self.model_folder / f"{last_training_timestep.strftime('%Y-%m-%d_%H-%M-%S')}" / "model.joblib"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        model = joblib.load(model_file)
        expected_cols = list(model.feature_name_) ### Check I made once
        #print(expected_cols)


        # Load the raw load data up until the origin timestamp but not including (right? CHECK THIS)
        raw_load_data_cut = self.raw_load_data_df.loc[:origin_timestamp - pd.Timedelta(self.params.period)].copy()

        # Add parent trajectory, its column is called current_forecast
        if not (raw_load_data_cut.index.tz == parent_trajectory.index.tz):
            raise ValueError("Timezone mismatch between raw load data and parent trajectory.")
        parent_trajectory = parent_trajectory.rename(columns={"current_forecast": str(self.household)}) ### Rename to match the raw load data column name, also fix inconsistency of int and str
        # Cut parent trajectory to only run up to the branch timestamp
        parent_trajectory = parent_trajectory[:branch_timestamp - pd.Timedelta(self.params.period)]
        raw_load_data_until_branch = pd.concat([raw_load_data_cut, parent_trajectory], axis=0).sort_index() 
        raw_load_data_until_branch = raw_load_data_until_branch[~raw_load_data_until_branch.index.duplicated(keep="last")] ### By chat
        #print(raw_load_data_until_branch.head(10))

        ### Build new index to check integrity
        verification_index = pd.date_range(start=raw_load_data_cut.index.min(), end=branch_timestamp - pd.Timedelta(self.params.period), freq=self.params.period, tz="Europe/Brussels")
        # This index should be equal to the one after adding the parent trajectory
        if not raw_load_data_until_branch.index.equals(verification_index):
            print(f"They are not equal")
            print(f"Indices that do not exist in raw_load_data_until_branch:\n{verification_index.difference(raw_load_data_until_branch.index)}")
            print(f"Indices that do not exist in verification_index:\n{raw_load_data_until_branch.index.difference(verification_index)}")


        ### Now have to make features
        prediction_row = self.make_origin_feature_row_for_branch_timestep(raw_load_data_until_branch, (branch_timestamp - pd.Timedelta(self.params.period)))
        prediction_frame_future, future_times  = self.make_future_feature_row_for_branch_timestep(prediction_row, (branch_timestamp - pd.Timedelta(self.params.period)))

        ### Just a stability check if were to work on features
        missing_cols = [col for col in expected_cols if col not in prediction_frame_future.columns]
        if missing_cols:
            raise ValueError(f"Missing expected columns in prediction_frame_future: {missing_cols}") ### Nan cehck is already implemented
        prediction_frame_future = prediction_frame_future[expected_cols]

        ### And predict
        forecast_branch_timestamp_numpy_array = model.predict(prediction_frame_future)
        #print(f"Eventual forecast for branch timestamp {branch_timestamp}:\n{forecast_branch_timestamp_numpy_array}")

        
        
        branch_forecast_df = pd.DataFrame({"current_forecast": forecast_branch_timestamp_numpy_array}, index=future_times) ### For neater use afterwards
        # Build target too for consistency with root
        target = self.raw_load_data_df.loc[future_times, str(self.household)].to_numpy(dtype=float)
        target_df = pd.DataFrame({"target": target}, index=future_times)
        ### Now make analogs
        closest, residuals_df, scenarios_df = self.sample_analogs_for_branch_timestep(forecast=forecast_branch_timestamp_numpy_array, branch_timestamp=branch_timestamp, knowledge_timestamp=knowledge_timestamp, future_index=future_times, n_scenarios=n_scenarios, same_time=same_time)

        ### Return all
        return closest, residuals_df, scenarios_df, branch_forecast_df, target_df ### For root origin was: closest, residuals_df, scenarios_df, current_forecast_df, target_df

        



        """
        Features to be made:
        ['lag_1', 'lag_1_missing', 'lag_2', 'lag_2_missing', 'lag_3', 'lag_3_missing', 'lag_4', 'lag_4_missing', 'lag_96', 'lag_96_missing', 'lag_672', 'lag_672_missing', 'lag_0', 
        'diff_0', 'diff_1', 'diff_0_2', 'roll_mean_4', 'roll_std_4', 'roll_min_4', 'roll_max_4', 'roll_mean_96', 'roll_std_96', 'roll_min_96', 'roll_max_96', 'is_weekend', 
        'dow_sin', 'dow_cos', 'mod_sin', 'mod_cos', 'doy_sin', 'doy_cos', 'is_morning_peak', 'is_evening_peak', 'future_is_weekend', 'future_dow_sin', 'future_dow_cos', 'future_mod_sin', 'future_mod_cos', 'future_doy_sin', 'future_doy_cos', 
        'h_step', 'h_step_sin', 'h_step_cos', 'h_norm', 'h_hours', 'future_is_morning_peak', 'future_is_evening_peak']
        """


    def make_origin_feature_row_for_branch_timestep(self, raw_load_data_until_branch: pd.DataFrame, one_before_branch_timestep: pd.Timestamp): ### Whole function just comes from Training_dfs.py make_features()
        steps_per_hour = int(pd.Timedelta("1h") / pd.Timedelta(self.params.period))
        # Make index of one timestamp
        idx = pd.DatetimeIndex([one_before_branch_timestep]) ### Bit ridiculous but well xd
        X = pd.DataFrame(index=idx)
        Y = raw_load_data_until_branch[str(self.household)]

        self.params.lags = [
            1,2,3,4,
            steps_per_hour,
            steps_per_hour*24, ### 1 day lag
            steps_per_hour*24*7, ### 1 week lag
        ]
        for lag in self.params.lags:
            X[f"lag_{lag}"] = Y.shift(lag)
            col_name = f"lag_{lag}"
            X[f"{col_name}_missing"] = X[col_name].isna().astype(int) ### Indicating missing value with int
        X["lag_0"] = Y ### Add zero lag, when called at 22:02 for example, lag_0 brings the data of 21:45 until 22:00!!

        ### Difference features
        X["diff_0"] = Y - Y.shift(1)
        X["diff_1"] = Y.shift(1) - Y.shift(2)
        X["diff_0_2"] = Y - Y.shift(2) ### Added extra

        ### Rolling features
        self.params.roll_windows = [ ### May add more
            steps_per_hour,
            steps_per_hour*24
        ]
        for roll in self.params.roll_windows:
            X[f"roll_mean_{roll}"] = Y.rolling(roll).mean()
            X[f"roll_std_{roll}"] = Y.rolling(roll).std()
            X[f"roll_min_{roll}"] = Y.rolling(roll).min()
            X[f"roll_max_{roll}"] = Y.rolling(roll).max()

        ### Add time features
        dow = idx.dayofweek
        X["is_weekend"] = (dow >= 5).astype(int)
        X["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        X["dow_cos"] = np.cos(2 * np.pi * dow / 7)

        hour = idx.hour
        minute = idx.minute
        min_of_day = hour * 60 + minute
        X["mod_sin"] = np.sin(2 * np.pi * min_of_day / (24 * 60))
        X["mod_cos"] = np.cos(2 * np.pi * min_of_day / (24 * 60))

        doy = idx.dayofyear
        X["doy_sin"] = np.sin(2 * np.pi * doy / 365)
        X["doy_cos"] = np.cos(2 * np.pi * doy / 365)

        ### Specific features
        X["is_morning_peak"] = ((hour >= 7) & (hour < 10)).astype(int)
        X["is_evening_peak"] = ((hour >= 17) & (hour < 20)).astype(int)

        ### No intraday features because simulation file is already at 15min

        #print(f"X: \n{X.head(10)}")
        return X


    def make_future_feature_row_for_branch_timestep(self, origin_feature_row, one_before_branch_timestep: pd.Timestamp):
        X_rows = []
        future_times = []

        ### idx was previously formed by the index of y but don't have that here. Could do it though from the raw_load_data but then I would have to reform code. Good for now
        #idx_future = pd.date_range(start=one_before_branch_timestep + pd.Timedelta(self.params.period), end=one_before_branch_timestep + pd.Timedelta(self.params.period) * self.params.HORIZON_STEPS, freq=self.params.period, inclusive="left") # Not needed actually
        
        origin_features_dict = origin_feature_row.iloc[0].to_dict() ### Just keep this, otuer loop not needed anymore

        for h in range(1, self.params.HORIZON_STEPS + 1):
            future_time = one_before_branch_timestep + pd.Timedelta(self.params.period) * h ### Easier
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
            future_times.append(future_time)

        X_with_future = pd.DataFrame(X_rows)
        future_times = pd.DatetimeIndex(future_times) ### For plotting if wanted

        if X_with_future.isna().any().any():
            raise ValueError(f"X_with_future contains nans: {X_with_future.isna().sum()} at {X_with_future[X_with_future.isna().any(axis=1)]}")
        
        #print(f"X_with_future: \n{X_with_future.head(10)}")
        
        return X_with_future, future_times
    

    def sample_analogs_for_branch_timestep(self, forecast, branch_timestamp, knowledge_timestamp, future_index, n_scenarios, same_time: bool):
        branch_timestamp_as_origin_for_branch = branch_timestamp - pd.Timedelta(self.params.period) ### Now use in the same sense as origin for the root timestep analogs 
        branch_timestamp_as_origin_for_branch_naive = branch_timestamp_as_origin_for_branch.tz_localize(None) ### For same_time=True
        ### Very similar to the root timestep analogs
        pred_cols = [f"pred_h_{h}" for h in range(1, self.params.HORIZON_STEPS + 1)]
        target_cols = [f"target_h_{h}" for h in range(1, self.params.HORIZON_STEPS + 1)]

        prediction_df = self.predictions_df.copy() ### Remmeber that index is already analyse_timestamp

        knowledge_timestamp_naive = knowledge_timestamp.tz_localize(None) ### instead of root_timestep_naive
        prediction_df["pseudo_analyse_timestamp"] = prediction_df.index.tz_localize(None)

        mask_after_knowledge = prediction_df.index > knowledge_timestamp
        prediction_df.loc[mask_after_knowledge, "pseudo_analyse_timestamp"] = prediction_df.loc[mask_after_knowledge, "pseudo_analyse_timestamp"] - pd.DateOffset(years=1) ### Shift back one year everything that comes after the knowledge timestamp

        start_interval = knowledge_timestamp_naive - pd.DateOffset(months=11)
        end_timestamp = knowledge_timestamp_naive - pd.DateOffset(days=1) ### Make sure forecasts are complete

        archive_df = prediction_df[(prediction_df["pseudo_analyse_timestamp"] >= start_interval) & (prediction_df["pseudo_analyse_timestamp"] <= end_timestamp) & (prediction_df.index != knowledge_timestamp)].copy() ### Don't think last one is still needed as it differs in time anyway, leave in for if same_time=False. Can put this to branch_timestamp_as_origin_for_branch_naive but df is already cut at knowledge_timestamp so doesn't matter
        if same_time:
            archive_df = archive_df[archive_df["pseudo_analyse_timestamp"].dt.time == branch_timestamp_as_origin_for_branch_naive.time()] 

        if archive_df.empty:
            raise ValueError(f"No analogs found for branch timestamp {branch_timestamp}. Check the data and the time range.")

        ### forecast should be numpy array!
        forecast_matrix = archive_df[pred_cols].to_numpy(dtype=float)
        rmse_per_row = np.sqrt(np.mean((forecast_matrix - forecast) ** 2, axis=1)) ### Make sure to put across column axis

        archive_df["similiarity_RMSE"] = rmse_per_row
        closest = archive_df.sort_values("similiarity_RMSE").head(n_scenarios)

        residuals = (closest[target_cols].to_numpy(dtype=float) - closest[pred_cols].to_numpy(dtype=float)) ### Still on the same row from predictions_df (shape (n_scenarios, 96))
        scenarios = forecast.reshape(1, -1) + residuals ### Easy by chat, turn (96,) into (1, 96)

        residuals_df = pd.DataFrame(residuals.T, index=future_index, columns=[f"residual_{i+1}" for i in range(residuals.shape[0])]) ### Amount of rows --> n_scenarios
        scenarios_df = pd.DataFrame(scenarios.T, index=future_index, columns=[f"scenario_{i+1}" for i in range(scenarios.shape[0])])

        return closest, residuals_df, scenarios_df























if __name__ == "__main__":
    xx = LoadResidualSampler(household=301, exclude_before_ts="2024-01-20 00:00:00")
    root_timestamp = pd.Timestamp("2024-09-01 00:00:00", tz="Europe/Brussels")
    closest, residuals_df, scenarios_df, current_forecast_df, target_df = xx.predict_for_root_timestamp_with_analogs(root_timestamp=(root_timestamp - pd.Timedelta(xx.params.period)), n_scenarios=5, same_time=True)
    print(current_forecast_df.head())
    xx.predict_for_branch_timestamp_with_analogs(origin_timestamp=pd.Timestamp("2024-09-01 00:00:00", tz="Europe/Brussels"), branch_timestamp=pd.Timestamp("2024-09-01 03:00:00", tz="Europe/Brussels"), parent_trajectory=current_forecast_df, n_scenarios=10, same_time=True)
