import pandas as pd
from pathlib import Path
import joblib
from dataclasses import dataclass, field
import json
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score ### Not used, maybe make script for evaluation of average scores for a household over a year?
import time
import matplotlib.pyplot as plt
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

"""
What should code do: 

For each household, create folder with its ID. Then, folders should be created for each timestamp according to a pre-defined resolution. First start with one week retrain each time and move on from there maybe. But maybe work with a type of grid that can be adjusted. E.g. pre-define each day to be trained, bit of stupid hard-coding but is okay. 
Set time always at 00:00:00 for training end? 
Model should be put under folder with the parameters listed. 
Put the four categories of households separate such that several PC's can train together. Implement a check that checks each round which models exist already.

Train them under deterministic for trial, then go to quantile?
"""

@dataclass
class Load_Forecaster_Params: ### Can be moved easily
    ### Only training dates are required. Eval/test dates are optional.
    start_training: str
    end_training: str

    HORIZON_STEPS: int = 48*4 ### 24 hours ahead with 5-minute data
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




class model_trainer():
    def __init__(self, params: Load_Forecaster_Params):
        self.params = params

    
    def make_training_timestamps(self, exclude_before_ts: str):
        JSON_FILE = Path(Path(__file__).parent / "Timestamps.json")
        with open(JSON_FILE, 'r') as f:
            timestamps_dict = json.load(f)

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
    

    def train_model(self, household_id: str, training_timestamp: pd.Timestamp, Models_folder: Path, household_df : pd.DataFrame):

        ### Cut dataframe to training timestamp
        known_until = training_timestamp
        start_training = known_until - pd.DateOffset(months=11)

        ### Cut on origin time and future
        df_train = household_df[(household_df["origin_time"] >= start_training) & (household_df["future_time"] <= known_until)].copy()

        if df_train.empty: ### Small check from Chat
            raise ValueError(
                f"No training data for household {household_id} at {training_timestamp}. "
                f"origin_time range: {household_df['origin_time'].min()} - {household_df['origin_time'].max()}, "
                f"future_time range: {household_df['future_time'].min()} - {household_df['future_time'].max()}"
            )

        #print( ### Previous printing was incorrect, have to print until future time, not max of origin time
            #f"Training timestamp: {training_timestamp}:", f"Origin range: {df_train['origin_time'].min()} - {df_train['origin_time'].max()}", f"Future target range: {df_train['future_time'].min()} - {df_train['future_time'].max()}\n"
        #)

        X = df_train.drop(columns=["target", "origin_time", "future_time"])
        Y = df_train["target"]

        ### Train model
        model = LGBMRegressor(
            boosting_type=self.params.boosting_type,
            objective=self.params.objective,
            n_estimators=self.params.n_estimators,
            learning_rate=self.params.learning_rate,
            num_leaves=self.params.num_leaves,
            max_bin=self.params.max_bin,
            max_depth=self.params.max_depth,
            min_child_samples=self.params.min_child_samples,
            min_child_weight=self.params.min_child_weight,
            subsample=self.params.subsample,
            subsample_freq=self.params.subsample_freq,
            colsample_bytree=self.params.colsample_bytree,
            reg_alpha=self.params.reg_alpha,
            reg_lambda=self.params.reg_lambda,
            random_state=self.params.random_state,
            verbose=self.params.verbose,
            #early_stopping_rounds=self.params.early_stopping_rounds,
            eval_metric=self.params.eval_metric,
            n_jobs=self.params.n_jobs
        )
        model.fit(X, Y)


        ### Save model
        params_string = f"{self.params.boosting_type}_{self.params.objective}_{self.params.n_estimators}_{self.params.learning_rate}_{self.params.num_leaves}_{self.params.max_bin}_{self.params.max_depth}_{self.params.min_child_samples}_{self.params.min_child_weight}_{self.params.subsample}_{self.params.subsample_freq}_{self.params.colsample_bytree}_{self.params.reg_alpha}_{self.params.reg_lambda}"

        model_folder = Path(Models_folder / params_string / f"Household_{household_id}" / f"{training_timestamp.strftime('%Y-%m-%d_%H-%M-%S')}")
        model_folder.mkdir(parents=True, exist_ok=True)
        model_file = model_folder / "model.joblib"
        joblib.dump(model, model_file)

        return model



    



### Multiprocessing worker function
def process_one_household(household, training_timestamps_original, params, Models_folder, Dataframes_folder):
    trainer = model_trainer(params)

    Dataframe_file = Dataframes_folder / f"{household}" / f"training_df_household_{household}_{params.HORIZON_STEPS}.parquet"

    df_household = pd.read_parquet(Dataframe_file)
    df_household["origin_time"] = pd.to_datetime(df_household["origin_time"])
    df_household["future_time"] = pd.to_datetime(df_household["future_time"])

    params_string = f"{params.boosting_type}_{params.objective}_{params.n_estimators}_{params.learning_rate}_{params.num_leaves}_{params.max_bin}_{params.max_depth}_{params.min_child_samples}_{params.min_child_weight}_{params.subsample}_{params.subsample_freq}_{params.colsample_bytree}_{params.reg_alpha}_{params.reg_lambda}"

    trained_count = 0
    skipped_count = 0

    for training_timestamp in training_timestamps_original:
        print(f"\nProcessing household {household} and training timestamp {training_timestamp}")
        model_folder = Path(Models_folder / params_string / f"Household_{household}" / f"{training_timestamp.strftime('%Y-%m-%d_%H-%M-%S')}")
        model_file = model_folder / "model.joblib"
        if model_file.exists():
            print(f"Model already exists for household {household} and training timestamp {training_timestamp}, skipping training.")
            skipped_count += 1
            continue

        else: 
            start_time = time.time()
            trainer.train_model(household_id=household, training_timestamp=training_timestamp, Models_folder=Models_folder, household_df=df_household)
            end_time = time.time()

            print(f"Time taken to train model for one household and one timestamp: {round(end_time - start_time, 2)} seconds")
            trained_count += 1
    
    return household, trained_count, skipped_count



if __name__ == "__main__":

    start_training = "2024-01-20 00:00:00"
    end_training = "2024-12-31 23:59:59" ### These are exclusively for building the timestamps!

    from_amount_to_train_of_each = -40 ### MAKE SURE THIS DOESN'T INTERFERE WITH DELFTBLUE
    to_amount_to_train_of_each = -1

    if sys.platform == "darwin": ### Currently -60 to -50 is running on Mac.
        households_load = [i for i in range(301, 601)] ### 301-600 are load households THIS IS RUNNING ON MAC
        households_HP = [i for i in range(901, 1201)] ### 901-1200 are heat pump households
        households_EV = [i for i in range(1501, 1801)] ### 1501-1800 are EV households
        households_HP_EV = [i for i in range(2101, 2401)] ### 2101-2400 are HP+EV households

        all_households = sorted(households_load[from_amount_to_train_of_each:to_amount_to_train_of_each] + households_HP[from_amount_to_train_of_each:to_amount_to_train_of_each] + households_EV[from_amount_to_train_of_each:to_amount_to_train_of_each] + households_HP_EV[from_amount_to_train_of_each:to_amount_to_train_of_each])

    elif sys.platform.startswith("linux"):
        households_load = [i for i in range(301, 601)]
        households_HP = [i for i in range(901, 1201)]
        households_EV = [i for i in range(1501, 1801)]
        households_HP_EV = [i for i in range(2101, 2401)]

        all_households = sorted(households_load + households_HP + households_EV + households_HP_EV)
    
    elif sys.platform == "win32": ### Currently -40 to -1 is running on Lenovo.
        households_load = [i for i in range(301, 601)] ### 301-600 are load households THIS IS RUNNING ON MAC
        households_HP = [i for i in range(901, 1201)] ### 901-1200 are heat pump households THIS IS RUNNING ON WINDOWS
        households_EV = [i for i in range(1501, 1801)] ### 1501-1800 are EV households
        households_HP_EV = [i for i in range(2101, 2401)] ### 2101-2400 are HP+EV households THIS IS RUNNING ON WINDOWS_YOUNAS


        all_households = sorted(households_load[from_amount_to_train_of_each:to_amount_to_train_of_each] + households_HP[from_amount_to_train_of_each:to_amount_to_train_of_each] + households_EV[from_amount_to_train_of_each:to_amount_to_train_of_each] + households_HP_EV[from_amount_to_train_of_each:to_amount_to_train_of_each])

    

    ### OVERWRITE
    all_households = [9999900] ### These were missing for some reason, should check if they can be trained or if something is wrong
    total_length = len(all_households)


    params = Load_Forecaster_Params(start_training=start_training, end_training=end_training)
    trainer = model_trainer(params) ### For timestamp making
    training_timestamps_original = trainer.make_training_timestamps(exclude_before_ts=start_training)

    hours_horizon = int(params.HORIZON_STEPS * (pd.Timedelta(params.period)/pd.Timedelta("1h"))) 
    Dataframes_folder = Path(Path(__file__).parent.parent / f"Dataframes {hours_horizon}h")
    Models_folder = Path(Path(__file__).parent.parent / "Models" / f"{hours_horizon}h" / "Deterministic")
    print(sys.platform)
    MAX_WORKERS = 1 if sys.platform == "darwin" else 1 if sys.platform == "win32" else 2
    
    params.n_jobs = 2 if sys.platform == "darwin" else 16 if sys.platform == "win32" else 4

    try:    
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(process_one_household, household, training_timestamps_original, params, Models_folder, Dataframes_folder): household for household in all_households
            }

            for counter, future in enumerate(as_completed(futures), 1):
                household = futures[future]
                try:
                    finished_household, trained_count, skipped_count = future.result()

                    print(f"Finished household {finished_household}. ({counter}/{total_length}) Trained: {trained_count}, Skipped: {skipped_count}")

                except Exception as e:
                    print(f"Error processing household {household}:\n{e}")
                    raise
    except KeyboardInterrupt:
        print("Process interrupted by user. Shutting down...") ### Have to add this because code keeps running in background

        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception as e:
            print(f"Error during shutdown: {e}")

        raise