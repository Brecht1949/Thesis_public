import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import json
import joblib
import time
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from termcolor import colored


"""
Preceding codes:
Code relies on existing joblib models, these are made by Train_models_Deterministic_multiprocessing.py
"""
@dataclass
class Load_Forecaster_Params: ### Can be moved easily
    ### Only training dates are required. Eval/test dates are optional.
    start_training: str
    end_training: str

    HORIZON_STEPS: int = 48*4 ### 48 hours with 15-min data
    #OUTPUT_DIR: Path = Path(Path(__file__).parent / "Models" / "Load")
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





def make_features(dataframe_indexed, plot_timestamp, expected_cols):
    df_cut = dataframe_indexed.loc[[plot_timestamp]]

    targets = df_cut["target"].to_numpy()
    df_forecast = df_cut[expected_cols]

    return df_forecast, targets





    


def make_training_timestamps(exclude_before_ts: str):
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







def process_one_household(household, referral_dict, Models_folder, params_string, horizon_steps):
    start_time = time.time()
    training_file_household = Path(Path(__file__).parent.parent / "Dataframes 48h" / f"{household}" / f"training_df_household_{household}_{horizon_steps}.parquet")
    training_df_household = pd.read_parquet(training_file_household)
    training_df_household = training_df_household.sort_values(["origin_time", "h_step"])
    training_df_household = training_df_household.set_index("origin_time")
    max_h_step = training_df_household["h_step"].max()
    if max_h_step < horizon_steps:
        raise ValueError(f"Training dataframe for household {household} only goes to h_step={max_h_step}, but horizon_steps={horizon_steps}. Rebuild the 48h training dataframe first.")

    rows = []

    household_model_folder = Path(Models_folder / params_string / f"Household_{household}")
    output_file = household_model_folder / "predictions.parquet"

    for last_training_timestamp_ in referral_dict.keys():
        model_file = Path(Models_folder / params_string / f"Household_{household}" / f"{last_training_timestamp_.strftime('%Y-%m-%d_%H-%M-%S')}" / "model.joblib")
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found for household {household} and training timestamp {last_training_timestamp_}: {model_file}")

        model = joblib.load(model_file)
        expected_cols = list(model.feature_name_)

        ### Do not loop individually anymore, direct for whole batch immediately per model
        anaylse_timestamps_model = referral_dict[last_training_timestamp_] ### Select all the timestamps that belong to this model. This will be in 15min steps until the next training timestamp (next model).
        df_batch = training_df_household.loc[anaylse_timestamps_model].copy() ### Index was changed to origin_time

        targets_batch = df_batch["target"].to_numpy() ### Performance improv by chat
        df_forecast_batch = df_batch[expected_cols]

        forecast_batch = model.predict(df_forecast_batch)
        df_batch["forecast"] = forecast_batch ### Add back to train df

        for analyse_timestamp, group in df_batch.groupby(level=0): ### Group by index. 

            group = group.sort_values("h_step") 
            forecast = group["forecast"].to_numpy()
            targets = group["target"].to_numpy()

            if len(forecast) != (horizon_steps): ### Hard coded for now
                raise ValueError(f"Expected {horizon_steps} forecasted values for analyse_timestamp {analyse_timestamp} but got {len(forecast)}")
            
            row = {"analyse_timestamp": analyse_timestamp, "last_training_timestamp": last_training_timestamp_}
            for h in range(horizon_steps):
                row[f"pred_h_{h+1}"] = forecast[h]
                row[f"target_h_{h+1}"] = targets[h]

            rows.append(row)

    predictions_df = pd.DataFrame(rows).sort_values("analyse_timestamp")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_parquet(output_file, index=False)
    time_taken = time.time() - start_time
    print(colored(f"Saved predictions for household {household} to {output_file} in {time_taken:.2f} seconds", "green"))


    return household, output_file





if __name__ == "__main__":
    Models_folder = Path(Path(__file__).parent.parent / "Models" / "48h" / "Deterministic")

    params = Load_Forecaster_Params(start_training=None, end_training=None) ### Doesn't matter for params string
    params_string = f"{params.boosting_type}_{params.objective}_{params.n_estimators}_{params.learning_rate}_{params.num_leaves}_{params.max_bin}_{params.max_depth}_{params.min_child_samples}_{params.min_child_weight}_{params.subsample}_{params.subsample_freq}_{params.colsample_bytree}_{params.reg_alpha}_{params.reg_lambda}"

    exclude_before_ts = "2024-01-20 00:00:00"
    timestamps = make_training_timestamps(exclude_before_ts) ### Use start=timestamps.min() because otherwise no model can be loaded
    ### CAN'T RUN UNTIL "2024-12-30 23:59:59" ANYMORE, LAST MODEL BASICALLY USELESS
    analyse_timestamps = pd.date_range(start=timestamps.min() + pd.Timedelta(minutes=15), end=pd.Timestamp("2024-12-29 23:59:59", tz="Europe/Brussels"), freq="15min") ### Do one day less because after that, no target values anymore. + pd.Timedelta(minutes=15) because of fix of chat in last_training_timestamp = timestamps[timestamps <= analyse_timestamp - pd.Timedelta(minutes=15)].max()

    referral_dict = {} ### Make a dict with key: last_training_timestamp, value: list of analyse_timestamps that belong to that model
    for analyse_timestamp in analyse_timestamps:
        last_training_timestamp = timestamps[timestamps <= analyse_timestamp - pd.Timedelta(minutes=15)].max() ### FIX BY CHAT!!! Otherwise model still trained on that last timestamp. Tbh, bit confusing with using the timestamp before.
        #print(f"Plot timestamp: {plot_timestamp}, last training timestamp: {last_training_timestamp}")

        ### Use nested dict with last_training_timestamp as key and list of plot_timestamps as value
        if last_training_timestamp not in referral_dict:
            referral_dict[last_training_timestamp] = [analyse_timestamp]
        else:
            referral_dict[last_training_timestamp].append(analyse_timestamp)


    ### Check which files have been processed already
    from_amount = 0
    to_amount = None ### Don't use -1, skips the last one xd
    ### Now to run on Delftblue
    households_load = [i for i in range(301, 601)] ### 301-600 are load households THIS IS RUNNING ON MAC
    households_HP = [i for i in range(901, 1201)] ### 901-1200 are heat pump households THIS IS RUNNING ON WINDOWS
    households_EV = [i for i in range(1501, 1801)] ### 1501-1800 are EV households
    households_HP_EV = [i for i in range(2101, 2401)] ### 2101-2400 are HP+EV households

    all_households = sorted(households_HP[from_amount:to_amount] + households_load[from_amount:to_amount] + households_EV[from_amount:to_amount] + households_HP_EV[from_amount:to_amount])
    
    
    total_length = len(all_households)
    HOUSEHOLDS = all_households
    completed_households = set()
    """
    RESET = False
    if not RESET:
        for household in HOUSEHOLDS:
            prediction_file = Path(Models_folder / params_string / f"Household_{household}" / "predictions.parquet")
            if prediction_file.exists():
                completed_households.add(household)
    

    required_training_timestamps =  sorted(referral_dict.keys())
    households_to_process = []
    for household in HOUSEHOLDS:
        if household in completed_households:
            continue
        
        missing_count = 0
        for training_timestamp in required_training_timestamps:
            model_file = Path(Models_folder / params_string / f"Household_{household}" / f"{training_timestamp.strftime('%Y-%m-%d_%H-%M-%S')}" / "model.joblib")
            if not model_file.exists():
                missing_count += 1
            
        if missing_count == 0:
            households_to_process.append(household)
        else: 
            print(colored(f"Skipping household {household} because {missing_count}/{len(required_training_timestamps)} model files are missing.", "yellow"))
    """


    ### Overwrite
    households_to_process = [9999900]
    max_workers = min(2, len(households_to_process)) ### 12 needs more memory
    if len(households_to_process) == 0: ### If all households are already processed, skip multiprocessing and just do analysis
        pass
    else:
        overall_dict = {}
        print("Starting multiprocessing for households...")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_one_household, household, referral_dict, Models_folder, params_string, params.HORIZON_STEPS): household for household in households_to_process}

            for counter, future in enumerate(as_completed(futures), 1):
                household = futures[future]
                try:
                    household, household_output_file = future.result()
                    print(f"Completed processing for household {household} ({counter}/{len(households_to_process)})")
                except Exception as e:
                    print(f"Error processing household {household}: {e}")