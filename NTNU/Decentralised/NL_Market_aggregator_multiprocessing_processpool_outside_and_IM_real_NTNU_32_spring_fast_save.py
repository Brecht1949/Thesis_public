import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
import time
from termcolor import colored
import pyomo.environ as pyo
from pyomo.opt import SolverFactory ### Disable if switching to gurobipy
import joblib
from NL_Framework_v2_NTNU import Framework
from NL_MPC_NTNU import MPC, BatteryData ### Update: import batterydata for file naming
#from MPC_gurobipy import MPC, SolverFactory ### Performance check
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import platform ### Easier alternative to sys
import numpy as np
import json

print(colored(f"SCRIPT USES C_DEG = {BatteryData.c_deg}", "red"))
"""
This version is almost the same but processpool is outside the sim steps. This way, cache can be maintained for the households as they work with fixed chuncks. 
HOWEVER, this will use too much ram for many households. Only good for small amounts.
"""


class MarketDetails:
    market_step_kwh: float = 25 ### 100 kw * 0.25
    rounding_threshold: float = 0.8 ### Percentage needed of volume in order to make bid
    bid_price_margin_eur_per_kwh: float = 25 / 1000 ### Margin for bid price, should be optimised



@dataclass
class MarketAggregate: ### Completely re-made, was becoming a mess. Separate different types of commitment now for each market, much easier.
    DA_raw_commitment: pd.Series = field(default_factory=lambda: pd.Series(dtype=float, index=pd.DatetimeIndex([], dtype='datetime64[ns, Europe/Brussels]'))) ### Datetimeindex by chat, error regarding comparison of timestamp with np array
    IDA1_raw_commitment: pd.Series = field(default_factory=lambda: pd.Series(dtype=float, index=pd.DatetimeIndex([], dtype='datetime64[ns, Europe/Brussels]')))
    IDA2_raw_commitment: pd.Series = field(default_factory=lambda: pd.Series(dtype=float, index=pd.DatetimeIndex([], dtype='datetime64[ns, Europe/Brussels]')))
    IDA3_raw_commitment: pd.Series = field(default_factory=lambda: pd.Series(dtype=float, index=pd.DatetimeIndex([], dtype='datetime64[ns, Europe/Brussels]')))

    DA_rounded_commitment: pd.Series = field(default_factory=lambda: pd.Series(dtype=float, index=pd.DatetimeIndex([], dtype='datetime64[ns, Europe/Brussels]')))
    IDA1_rounded_commitment: pd.Series = field(default_factory=lambda: pd.Series(dtype=float, index=pd.DatetimeIndex([], dtype='datetime64[ns, Europe/Brussels]')))
    IDA2_rounded_commitment: pd.Series = field(default_factory=lambda: pd.Series(dtype=float, index=pd.DatetimeIndex([], dtype='datetime64[ns, Europe/Brussels]')))
    IDA3_rounded_commitment: pd.Series = field(default_factory=lambda: pd.Series(dtype=float, index=pd.DatetimeIndex([], dtype='datetime64[ns, Europe/Brussels]')))

    DA_cleared_commitment: pd.Series = field(default_factory=lambda: pd.Series(dtype=float, index=pd.DatetimeIndex([], dtype='datetime64[ns, Europe/Brussels]')))
    IDA1_cleared_commitment: pd.Series = field(default_factory=lambda: pd.Series(dtype=float, index=pd.DatetimeIndex([], dtype='datetime64[ns, Europe/Brussels]')))
    IDA2_cleared_commitment: pd.Series = field(default_factory=lambda: pd.Series(dtype=float, index=pd.DatetimeIndex([], dtype='datetime64[ns, Europe/Brussels]')))
    IDA3_cleared_commitment: pd.Series = field(default_factory=lambda: pd.Series(dtype=float, index=pd.DatetimeIndex([], dtype='datetime64[ns, Europe/Brussels]')))


    def total_commitment(self):
        total_commitment = pd.Series(dtype=float, index=pd.DatetimeIndex([], dtype='datetime64[ns, Europe/Brussels]')) ### No need for field, not stored in class

        for commitment_ in [ ### Just list them all
            self.DA_raw_commitment, self.IDA1_raw_commitment, self.IDA2_raw_commitment, self.IDA3_raw_commitment,
            self.DA_rounded_commitment, self.IDA1_rounded_commitment, self.IDA2_rounded_commitment, self.IDA3_rounded_commitment,
            self.DA_cleared_commitment, self.IDA1_cleared_commitment, self.IDA2_cleared_commitment, self.IDA3_cleared_commitment
        ]:
            if len(commitment_) == 0:
                continue ### Skip empty series, causes crashes
            total_commitment = total_commitment.add(commitment_, fill_value=0) ### Don't put nans, problem for ida3

        return total_commitment.sort_index()
    

    def set_raw_commitment(self, market_name, commitment_series):
        commitment_series = commitment_series.sort_index() ### To be sure

        if market_name == "DA":
            self.DA_raw_commitment = commitment_series
        elif market_name == "IDA1":
            self.IDA1_raw_commitment = commitment_series
        elif market_name == "IDA2":
            self.IDA2_raw_commitment = commitment_series
        elif market_name == "IDA3":
            self.IDA3_raw_commitment = commitment_series
        else: 
            raise ValueError(f"Invalid market name in raw: {market_name}.")
        
    
    def set_rounded_commitment(self, market_name, commitment_series):
        commitment_series = commitment_series.sort_index() ### To be sure

        if market_name == "DA":
            self.DA_raw_commitment = pd.Series(dtype=float, index=pd.DatetimeIndex([], dtype='datetime64[ns, Europe/Brussels]')) ### Reset raw commitment to empty
            self.DA_rounded_commitment = commitment_series
        elif market_name == "IDA1":
            self.IDA1_raw_commitment = pd.Series(dtype=float, index=pd.DatetimeIndex([], dtype='datetime64[ns, Europe/Brussels]')) ### Reset raw commitment to empty
            self.IDA1_rounded_commitment = commitment_series
        elif market_name == "IDA2":
            self.IDA2_raw_commitment = pd.Series(dtype=float, index=pd.DatetimeIndex([], dtype='datetime64[ns, Europe/Brussels]')) ### Reset raw commitment to empty
            self.IDA2_rounded_commitment = commitment_series
        elif market_name == "IDA3":
            self.IDA3_raw_commitment = pd.Series(dtype=float, index=pd.DatetimeIndex([], dtype='datetime64[ns, Europe/Brussels]')) ### Reset raw commitment to empty
            self.IDA3_rounded_commitment = commitment_series
        else: 
            raise ValueError(f"Invalid market name in rounded: {market_name}.")
        

    def set_cleared_commitment(self, market_name, commitment_series):
        commitment_series = commitment_series.sort_index() ### To be sure

        if market_name == "DA":
            self.DA_rounded_commitment = pd.Series(dtype=float, index=pd.DatetimeIndex([], dtype='datetime64[ns, Europe/Brussels]')) ### Reset rounded commitment to empty
            self.DA_cleared_commitment = self.update_cleared_commitment_serie(self.DA_cleared_commitment, commitment_series)
        elif market_name == "IDA1":
            self.IDA1_rounded_commitment = pd.Series(dtype=float, index=pd.DatetimeIndex([], dtype='datetime64[ns, Europe/Brussels]')) ### Reset rounded commitment to empty
            self.IDA1_cleared_commitment = self.update_cleared_commitment_serie(self.IDA1_cleared_commitment, commitment_series)
        elif market_name == "IDA2":
            self.IDA2_rounded_commitment = pd.Series(dtype=float, index=pd.DatetimeIndex([], dtype='datetime64[ns, Europe/Brussels]')) ### Reset rounded commitment to empty
            self.IDA2_cleared_commitment = self.update_cleared_commitment_serie(self.IDA2_cleared_commitment, commitment_series)
        elif market_name == "IDA3":
            self.IDA3_rounded_commitment = pd.Series(dtype=float, index=pd.DatetimeIndex([], dtype='datetime64[ns, Europe/Brussels]')) ### Reset rounded commitment to empty
            self.IDA3_cleared_commitment = self.update_cleared_commitment_serie(self.IDA3_cleared_commitment, commitment_series)
        else: 
            raise ValueError(f"Invalid market name in cleared: {market_name}.")

        

    @staticmethod
    def update_cleared_commitment_serie(old_commitment, new_commitment): ### To not lose cleared DA, used in current day timesteps of MPC and later for logging.
        if new_commitment is None or len(new_commitment) == 0:
            return old_commitment.sort_index() ### Nothing to update
        
        ### Few checks for debug
        if old_commitment.index.duplicated().any():
            raise ValueError("Old commitment series has duplicated timestamps.")
        if new_commitment.index.duplicated().any():
            raise ValueError("New commitment series has duplicated timestamps.")
        
        ### Flag for overlap, spot misalignment
        overlap = old_commitment.index.intersection(new_commitment.index) ### Better, by chat
        if len(overlap) > 0:
            raise ValueError(f"New commitment series overlaps with old commitment series at timestamps: {overlap}.")
        

        ### If all good, return liek before
        return pd.concat([old_commitment, new_commitment]).sort_index()
    

    def remove_old_commitments(self, simulation_step): ### Avoid overlap between old and new commitments
        self.DA_raw_commitment = self.DA_raw_commitment[self.DA_raw_commitment.index >= simulation_step]
        self.IDA1_raw_commitment = self.IDA1_raw_commitment[self.IDA1_raw_commitment.index >= simulation_step]
        self.IDA2_raw_commitment = self.IDA2_raw_commitment[self.IDA2_raw_commitment.index >= simulation_step]
        self.IDA3_raw_commitment = self.IDA3_raw_commitment[self.IDA3_raw_commitment.index >= simulation_step]

        self.DA_rounded_commitment = self.DA_rounded_commitment[self.DA_rounded_commitment.index >= simulation_step]
        self.IDA1_rounded_commitment = self.IDA1_rounded_commitment[self.IDA1_rounded_commitment.index >= simulation_step]
        self.IDA2_rounded_commitment = self.IDA2_rounded_commitment[self.IDA2_rounded_commitment.index >= simulation_step]
        self.IDA3_rounded_commitment = self.IDA3_rounded_commitment[self.IDA3_rounded_commitment.index >= simulation_step]

        self.DA_cleared_commitment = self.DA_cleared_commitment[self.DA_cleared_commitment.index >= simulation_step]
        self.IDA1_cleared_commitment = self.IDA1_cleared_commitment[self.IDA1_cleared_commitment.index >= simulation_step]
        self.IDA2_cleared_commitment = self.IDA2_cleared_commitment[self.IDA2_cleared_commitment.index >= simulation_step]
        self.IDA3_cleared_commitment = self.IDA3_cleared_commitment[self.IDA3_cleared_commitment.index >= simulation_step]
            
        
##### BY CHAT: SURPRESS EVAL_METRIC WARNING, his is no error but forgot to take it out at training, no harm but annoying prints
import lightgbm as lgb
class LightGBMWarningFilter:
    def info(self, msg):
        if "Unknown parameter: eval_metric" not in msg:
            print(msg)

    def warning(self, msg):
        if "Unknown parameter: eval_metric" not in msg:
            print(msg)
lgb.register_logger(LightGBMWarningFilter())


def should_linearise_battery_constraint(tree_data, battery_data, contract_data, markets_to_check=None): ### Deviation_cost_penalty hardcoded for now, should be synced to MPC script
    ### Check if linearising can happen safe 100% for this interval, using constraint related to a priori paper
    EPS = 1e-6
    if battery_data.eta**2 >= 1 - EPS:
        print(colored(f"Battery eff approximately 1. No binaries needed as long as degredation cost penalty is present.", "yellow"))
        return [], pd.DataFrame()
    
    RHS = - battery_data.c_deg * (1 + battery_data.eta**2) / (1 - battery_data.eta**2)

    if markets_to_check is None:
        markets_to_check = tree_data["market_prices_by_timestamp"].keys() ### Check for all markets. Not too sure about this but better to be too conservative
    
    ### Get all timestamps
    timestamps = sorted(tree_data["bid_step_to_timestamp"].values())
    
    rows = []
    battery_binary_needed_timestamps = []
    for market_name in markets_to_check:
        for timestamp in timestamps:
            market_price = tree_data["market_prices_by_timestamp"][market_name].get(timestamp, float("nan")) ### Update: Check if price is available regarding market cancellation
            if pd.isna(market_price):
                continue  ### No price available, skip this timestamp for this market

            if contract_data.SALDEREN: ### Make distinction between salderen and non-salderen
                LHS_IMPORT = market_price + contract_data.inkoopvergoeding + contract_data.import_belasting
                LHS_EXPORT = market_price + contract_data.verkoopvergoeding + contract_data.import_belasting ### The higher for export, more DEmotivation of simultaneous, see paper
            else:
                LHS_IMPORT = market_price + contract_data.inkoopvergoeding + contract_data.import_belasting
                LHS_EXPORT = market_price + contract_data.verkoopvergoeding

            
            potential_violation = (LHS_IMPORT < RHS) or (LHS_EXPORT < RHS)
            if potential_violation:
                rows.append({
                    "timestamp": timestamp,
                    "market_name": market_name,
                    "market_price": market_price, ### For printing, debug, analysis
                    "Salderen_bool": contract_data.SALDEREN,
                    "LHS_IMPORT": LHS_IMPORT,
                    "LHS_EXPORT": LHS_EXPORT,
                    "RHS": RHS,
                    "potential_violation": potential_violation
                })
                battery_binary_needed_timestamps.append(timestamp)
    
    potential_violations_df = pd.DataFrame(rows)

    if len(potential_violations_df) == 0:
        if VERBOSITY >= 0:
            print(colored(f"No violations found PRICE WISE for simultaneous charge/discharge. Will TRY LP.", "green"))
        return [], potential_violations_df
    
    battery_binary_needed_timestamps = sorted(set(battery_binary_needed_timestamps))
    #print(colored(f"Potential violations found for battery linearisation check at {len(battery_binary_needed_timestamps)} timestamps.", "yellow"))
    #print(colored(f"Violations:\n{potential_violations_df.head(10).to_string()}", "yellow"))

    return battery_binary_needed_timestamps, potential_violations_df


def extract_next_step_action(model, tree_data):
    ### Have to get some instance of NST (node,stage,local_time). They are all linked (see MPC.py, constraint on first step)
    first_stage = min(tree_data["stages"])
    first_local_time = min(tree_data["stage_local_times"][first_stage])
    first_node = tree_data["nodes_per_stage_dict"][first_stage][0] ### Doesn't matter which noe, all same

    E_charge = pyo.value(model.E_charge[first_node, first_stage, first_local_time])
    E_discharge = pyo.value(model.E_discharge[first_node, first_stage, first_local_time])
    E_grid_import = pyo.value(model.E_grid_import[first_node, first_stage, first_local_time])
    E_grid_export = pyo.value(model.E_grid_export[first_node, first_stage, first_local_time])
    SOC_start = pyo.value(model.SOC[first_node, first_stage, first_local_time]) ### remember significance of SOC, SOC is at start of interval

    return {"E_charge": E_charge, "E_discharge": E_discharge, "E_grid_import": E_grid_import, "E_grid_export": E_grid_export, "SOC_start": SOC_start}

def update_soc(battery_data, next_step_action_dict): ### Like in NTNU script, was also external. NOISE CAN BE ADDED HERE LIKE BEFORE.
    soc_next = next_step_action_dict["SOC_start"] + (battery_data.eta * next_step_action_dict["E_charge"] - (1 / battery_data.eta) * next_step_action_dict["E_discharge"]) / battery_data.Vmax
    return soc_next 


### Now impelment the balance delta
def prepare_balance_delta_df(balance_delta_df):
    balance_delta_df = balance_delta_df.copy()
    ### Just liek before from Convert_to_parquet.py
    balance_delta_df["Timeinterval Start Loc"] = pd.to_datetime(balance_delta_df["Timeinterval Start Loc"], format="mixed")
    balance_delta_df["Timeinterval End Loc"] = pd.to_datetime(balance_delta_df["Timeinterval End Loc"], format="mixed")
    balance_delta_df["available_at_ts"] = balance_delta_df["Timeinterval End Loc"] + BALANCE_DELTA_DELAY ### Maybe should place this variable before this function, will see

    ### Just in case, by chat
    balance_delta_df = balance_delta_df.sort_values("available_at_ts").reset_index(drop=True)
    return balance_delta_df

def get_balance_delta_for_interval(balance_delta_df, simulation_step):
    interval_end = simulation_step + pd.Timedelta(minutes=15)
    return balance_delta_df[(balance_delta_df["available_at_ts"] >= simulation_step) & (balance_delta_df["available_at_ts"] < interval_end)].sort_values("available_at_ts").copy()


def apply_balance_delta_after_solving(next_step_action, battery_data, simulation_step, household, fixed_market_commitment, balance_delta_interval_df, real_load_kwh, real_pv_kwh):
    action = next_step_action.copy() ### Copy to not change original
    ### Instead, calculate real imbalance
    commitment_kwh = float(fixed_market_commitment.get(simulation_step, 0.0))
    household_position_kwh = real_pv_kwh - real_load_kwh + action["E_discharge"] - action["E_charge"]
    base_imbalance_kwh = household_position_kwh - commitment_kwh

    action["E_balance_delta_correction"] = 0.0 ### Mkae sure these values exist in any case to avoid crash in eval
    action["imbalance_before_balance_delta"] = base_imbalance_kwh
    action["imbalance_after_balance_delta"] = base_imbalance_kwh ### Will be updated below, keep strucutre
    action["SOC_next_without_balance_delta"] = update_soc(battery_data=battery_data, next_step_action_dict=action) ### Easier comparison afterwards in eval file

    if not USE_BALANCE_DELTA or len(balance_delta_interval_df) == 0:
        soc_next = update_soc(battery_data=battery_data, next_step_action_dict=action)
        return action, pd.DataFrame(), soc_next
    
    balance_delta_interval_df = balance_delta_interval_df.copy()
    #balance_delta_interval_df["duration_in_hours"] = (balance_delta_interval_df["Timeinterval End Loc"] - balance_delta_interval_df["Timeinterval Start Loc"]).dt.total_seconds() / 3600 ### Easier eventually, by chat
    interval_end = simulation_step + pd.Timedelta(minutes=15) ### Fix by chat, crash on transition, have to use actually used time
    next_available = balance_delta_interval_df["available_at_ts"].shift(-1).fillna(interval_end)
    balance_delta_interval_df["duration_in_hours"] = (next_available - balance_delta_interval_df["available_at_ts"]).dt.total_seconds() / 3600
    
    total_duration = balance_delta_interval_df["duration_in_hours"].sum()
    # And implement check of chat too as a safeguard
    if abs(total_duration - 0.25) > 1e-6:
        raise ValueError(f"Total duration of balance delta interval is not 15 minutes. Total duration: {total_duration} hours.")
    
    soc = action["SOC_start"]
    soc_without_balance_delta = action["SOC_start"] ### At same frequency as steps. Follows the original MPC setpoints
    total_charge_kwh = 0.0
    total_discharge_kwh = 0.0
    total_correction_kwh = 0.0

    detail_rows = []
    previous_measured_imbalance_kw = 0.0
    previous_correction_kw = 0.0
    total_measured_imbalance_kwh = 0.0
    for _, row in balance_delta_interval_df.iterrows(): ### Already sorted
        dt_h = float(row["duration_in_hours"])
        row_fraction = dt_h / total_duration ### Have to do fraction of course because in kwh. Is 1/15 for 1min data, 1/75 for 12s data
        real_load_substep_kwh = real_load_kwh * row_fraction
        real_pv_substep_kwh = real_pv_kwh * row_fraction
        commitment_substep_kwh = commitment_kwh * row_fraction

        base_charge_kwh = action["E_charge"] * row_fraction
        base_discharge_kwh = action["E_discharge"] * row_fraction
        base_grid_import_kwh = action["E_grid_import"] * row_fraction
        base_grid_export_kwh = action["E_grid_export"] * row_fraction
        #commitment_substep_kwh = commitment_kwh * row_fraction ### Not used anymore but can be written for base_imbalance_substep_kwh

        base_net_battery_kwh = base_discharge_kwh - base_charge_kwh
        base_household_position_substep_kwh = base_grid_export_kwh - base_grid_import_kwh
        #base_imbalance_substep_kwh = base_household_position_substep_kwh - commitment_substep_kwh ### Positive means we are long (up)
        base_imbalance_substep_kwh = base_imbalance_kwh * row_fraction


        highest_price = row["Highest Upward Regulation Price"]
        lowest_price = row["Lowest Downward Regulation Price"]
        mid_price = row["Mid Price"]

        ### Make the cases three cases and then subdivide with extreme prices trhough threshold
        #up_price_strength = max(0.0, float(highest_price) - float(mid_price)) if not pd.isna(highest_price) else 0.0
        #down_price_strength = max(0.0, float(mid_price) - float(lowest_price)) if not pd.isna(lowest_price) else 0.0

        ### Implement with abs values now regarding reverse pricing, direction based on diff instead of strength now
        up_price_difference = float(highest_price) - float(mid_price) if not pd.isna(highest_price) else 0.0
        down_price_difference = float(mid_price) - float(lowest_price) if not pd.isna(lowest_price) else 0.0
        up_price_strength = abs(up_price_difference)
        down_price_strength = abs(down_price_difference)

        direction = 0 ### 1: discharge, -1: charge
        price_strength = 0.0
        extreme_price = False

        ### Upward case
        if not pd.isna(highest_price) and pd.isna(lowest_price):
            direction = 1
            price_strength = up_price_strength
            extreme_price = up_price_difference >= EXTREME_PRICE_THRESHOLD_EUR_PER_MWH ### Don't use the abs value, by chat

            ### Keep original imbalance direction for not extreme prices
            if not extreme_price and previous_measured_imbalance_kw >= 0:
                direction = 0
        ### Downward case
        elif pd.isna(highest_price) and not pd.isna(lowest_price):
            direction = -1
            price_strength = down_price_strength
            extreme_price = down_price_difference >= EXTREME_PRICE_THRESHOLD_EUR_PER_MWH

            if not extreme_price and previous_measured_imbalance_kw <= 0:
                direction = 0
        ### Both prices cases
        elif not pd.isna(highest_price) and not pd.isna(lowest_price):
            up_extreme_strength = max(0.0, up_price_difference) ### Keep like original, will have to be in correct direction for extreme price
            down_extreme_strength = max(0.0, down_price_difference)
            if up_extreme_strength >= EXTREME_PRICE_THRESHOLD_EUR_PER_MWH or down_extreme_strength >= EXTREME_PRICE_THRESHOLD_EUR_PER_MWH:
                extreme_price = True
                if up_extreme_strength > down_extreme_strength:
                    direction = 1
                    price_strength = up_extreme_strength
                else:
                    direction = -1
                    price_strength = down_extreme_strength
            ### Neither is extreme, reduce original imbalance
            elif previous_measured_imbalance_kw < 0:
                direction = 1
                price_strength = up_price_strength
            elif previous_measured_imbalance_kw > 0:
                direction = -1
                price_strength = down_price_strength
        ### No prices
        else:
            direction = 0
            price_strength = 0.0
            extreme_price = False
        
        ### Updating logic
        soc_start_substep = soc
        soc_without_balance_delta_start = soc_without_balance_delta
        max_discharge_kwh = min(battery_data.dmax * dt_h, max(0.0, (soc - battery_data.soc_min) * battery_data.Vmax * battery_data.eta)) ### See mpc.py, keep soc constraints
        max_charge_kwh = min(battery_data.cmax * dt_h, max(0.0, (battery_data.soc_max - soc) * battery_data.Vmax / battery_data.eta))

        price_score = 0.0
        soc_score = 0.0
        response_fraction = 0.0
        wanted_correction_kw = previous_correction_kw
        wanted_correction_kwh = wanted_correction_kw * dt_h
        target_net_battery_kwh = base_net_battery_kwh

        if direction != 0 and price_strength > 0: ### HAVE TO LOOK AT REVERSE PRICING CASE. Occurs when the highest downward bid is higher than the lowest upward bid and thus the mid price is higher than some of the upward bids and will thus create a negative price strength, even though the direction is still upward
            if direction == 1:
                soc_score = (soc - battery_data.soc_min) / (battery_data.soc_max - battery_data.soc_min)
                battery_headroom_kwh = max(0.0, max_discharge_kwh - base_net_battery_kwh) ### remember base net = discharge - charge
            else:
                soc_score = (battery_data.soc_max - soc) / (battery_data.soc_max - battery_data.soc_min)
                battery_headroom_kwh = max(0.0, max_charge_kwh + base_net_battery_kwh) ### base net battery is negative for charging

            soc_score = max(0.0, min(1.0, soc_score)) ### Just in case
            price_score = 1.0 - np.exp(-price_strength / PRICE_SCALE_EUR_PER_MWH)
            response_fraction = price_score * (soc_score ** SOC_SENSITIVITY)

            if extreme_price:
                ### We can cross the threshold
                wanted_correction_kwh = direction * battery_headroom_kwh * response_fraction ### Can cross boundary
            
            else: ### For normal prices, can't cross bound
                wanted_correction_kw = previous_correction_kw + direction * abs(previous_measured_imbalance_kw) * response_fraction  ### Corrected by chat, multiply by fraction, updated
                wanted_correction_kwh = wanted_correction_kw * dt_h
        
        target_net_battery_kwh = base_net_battery_kwh + wanted_correction_kwh
        new_net_battery_kwh = max(-max_charge_kwh, min(max_discharge_kwh, target_net_battery_kwh))
        actual_correction_kwh = new_net_battery_kwh - base_net_battery_kwh
        previous_correction_kw = actual_correction_kwh / dt_h

        ### Separate again
        new_charge_kwh = max(0.0, -new_net_battery_kwh)
        new_discharge_kwh = max(0.0, new_net_battery_kwh)

        # Update soc
        soc = soc + (battery_data.eta * new_charge_kwh - new_discharge_kwh / battery_data.eta) / battery_data.Vmax
        soc_without_balance_delta = soc_without_balance_delta + (battery_data.eta * base_charge_kwh - base_discharge_kwh / battery_data.eta) / battery_data.Vmax

        total_charge_kwh += new_charge_kwh
        total_discharge_kwh += new_discharge_kwh
        total_correction_kwh += actual_correction_kwh

        actual_position_substep_kwh = real_pv_substep_kwh - real_load_substep_kwh + new_discharge_kwh - new_charge_kwh
        actual_imbalance_substep_kwh = actual_position_substep_kwh - commitment_substep_kwh

        previous_measured_imbalance_kw = actual_imbalance_substep_kwh / dt_h ### pass for the next step
        total_measured_imbalance_kwh += actual_imbalance_substep_kwh

        ### Make clearer for saving
        if direction == 1:
            selected_direction = "UP"
        elif direction == -1:
            selected_direction = "DOWN"
        else:
            selected_direction = "NONE"

        ### Save results for this substep. Will probably delete some columns later for clarity, but keep for now for debug and analysis
        detail_rows.append({
            "simulation_step": simulation_step,
            "household": household,
            "substep_start": row["Timeinterval Start Loc"],
            "substep_end": row["Timeinterval End Loc"],
            "available_at_ts": row["available_at_ts"],
            "selected_direction": selected_direction,
            "SOC_start": soc_start_substep,
            "SOC_end": soc,
            "SOC_without_balance_delta_start": soc_without_balance_delta_start,
            "SOC_without_balance_delta_end": soc_without_balance_delta,
            "base_imbalance_kwh": base_imbalance_substep_kwh, ### This one doesn't change
            "imbalance_after_balance_delta_kwh": actual_imbalance_substep_kwh,
            "actual_position_substep_kwh": actual_position_substep_kwh,
        })
    
    balance_delta_saving_df = pd.DataFrame(detail_rows)

    corrected_household_position_kwh = real_pv_kwh - real_load_kwh + total_discharge_kwh - total_charge_kwh
    final_imbalance_kwh = total_measured_imbalance_kwh

    action["E_charge"] = total_charge_kwh
    action["E_discharge"] = total_discharge_kwh
    action["E_grid_import"] = max(0.0, -corrected_household_position_kwh)
    action["E_grid_export"] = max(0.0, corrected_household_position_kwh)
    action["E_balance_delta_correction"] = total_correction_kwh
    action["imbalance_after_balance_delta"] = final_imbalance_kwh
    action["SOC_next_without_balance_delta"] = soc_without_balance_delta ### The one update along the substeps with original setpoints, don't get mistaken


    return action, balance_delta_saving_df, soc


        






def extract_market_bid(model, tree_data):
    out = {}

    for bid_step in model.MARKET_BID_STEPS:
        timestamp = tree_data["bid_step_to_timestamp"][int(bid_step)]
        out[timestamp] = pyo.value(model.E_market_bid[bid_step])

    if len(out) == 0:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="Europe/Brussels")) ### Fix by chat regarding cancelled market
    
    return pd.Series(out).sort_index()

def extract_market_bid_prices(tree_data, market_name, bid_start, bid_end):
    final_price_dict = {}

    for bid_step, timestamp in tree_data["bid_step_to_timestamp"].items():
        if bid_start <= timestamp < bid_end:
            price_value = tree_data["market_prices_by_timestamp"][market_name].get(timestamp, float("nan")) ### Update: Check if price is available regarding market cancellation
            if not pd.isna(price_value):
                final_price_dict[timestamp] = price_value

    if len(final_price_dict) == 0:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="Europe/Brussels")) ### Fix by chat regarding cancelled market
    final_price_series = pd.Series(final_price_dict).sort_index()
    return final_price_series


def get_market_for_timestep(simulation_step):
    DAY = simulation_step.normalize()
    TIME = simulation_step.strftime("%H:%M")

    if TIME == "12:00":
        market_name = "DA"
        bid_start = DAY + pd.DateOffset(days=1)
        bid_end = DAY + pd.DateOffset(days=2)
    elif TIME == "15:00":
        market_name = "IDA1"
        bid_start = DAY + pd.DateOffset(days=1)
        bid_end = DAY + pd.DateOffset(days=2)
    elif TIME == "22:00":
        market_name = "IDA2"
        bid_start = DAY + pd.DateOffset(days=1)
        bid_end = DAY + pd.DateOffset(days=2)
    elif TIME == "10:00":
        market_name = "IDA3"
        bid_start = DAY.replace(hour=12)
        bid_end = DAY + pd.DateOffset(days=1)
    else:
        market_name = None
        bid_start = None
        bid_end = None

    return market_name, bid_start, bid_end


def get_rounding_instances_for_timestep(simulation_step):
    TIME = simulation_step.strftime("%H:%M")

    if TIME == "12:15": ### Easier, way more realistic implementation. Makes it possible even to perform a optimisation for the rounding residual distribution over the households.
        return "DA"
    elif TIME == "15:15":
        return "IDA1"
    elif TIME == "22:15":
        return "IDA2"
    elif TIME == "10:15":
        return "IDA3"
    else:
        return None


def get_clearing_instances_for_timestep(simulation_step):
    TIME = simulation_step.strftime("%H:%M")

    if TIME == "13:00":
        return "DA"
    elif TIME == "15:30":
        return "IDA1"
    elif TIME == "22:30":
        return "IDA2"
    elif TIME == "10:30":
        return "IDA3"
    else:
        return None
    



def aggregate_household_bids(household_bids_dict):
    portfolio_bid = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="Europe/Brussels"))

    for bid in household_bids_dict.values():
        portfolio_bid = portfolio_bid.add(bid, fill_value=0)

    return portfolio_bid.sort_index()


def round_bid_value(bid_array, market_details: MarketDetails):
    ### Nothing to do with pricing yet, only volume
    # SHould treat as absolute value, easier, by chat
    bid_series_abs = bid_array.abs()

    lower_bound = (bid_series_abs // market_details.market_step_kwh) * market_details.market_step_kwh
    residual_series = bid_series_abs - lower_bound

    rounded_abs = lower_bound.copy()
    round_up_mask = (residual_series >= market_details.rounding_threshold * market_details.market_step_kwh)
    rounded_abs[round_up_mask] = rounded_abs[round_up_mask] + market_details.market_step_kwh

    sign_series = bid_array.copy()
    sign_series[sign_series >= 0] = 1
    sign_series[sign_series < 0] = -1

    final_bid_array = rounded_abs * sign_series

    ### Immediately get residual array
    residual_array = final_bid_array - bid_array ### Positive, bid too high, negative, bid too low
    return final_bid_array.sort_index(), residual_array.sort_index()


def simulate_submitted_bid_prices(forecast_price_series, portfolio_bid_series, market_details: MarketDetails):
    ### Bid slightly lower for sell bids, slightly higher for buy bids to increase acceptance probability
    submitted_price_serie = forecast_price_series.copy()

    sell_mask = portfolio_bid_series > 0
    buy_mask = portfolio_bid_series < 0

    submitted_price_serie[sell_mask] = submitted_price_serie[sell_mask] - market_details.bid_price_margin_eur_per_kwh
    submitted_price_serie[buy_mask] = submitted_price_serie[buy_mask] + market_details.bid_price_margin_eur_per_kwh

    return submitted_price_serie.sort_index()


def allocate_residuals_to_households(household_bids_dict, residual_array):
    ### This logic can be adjusted. Can even be turned into an optimisation problem if wanted according to some weighting criteria.
    corrected_household_bids_dict = {}
    n_households = len(household_bids_dict)

    correction_per_household_array = residual_array / n_households
    for household, raw_bid in household_bids_dict.items():
        corrected_household_bids_dict[household] = raw_bid.add(correction_per_household_array, fill_value=0).sort_index()

    return corrected_household_bids_dict


def get_cleared_market_prices(framework, market_name, timestamps):
    if len(timestamps) == 0:
        return None ### Little hack in order to trigger the print statement 
    ### In order not to rebuild the whole tree just for the prices, access framework directly
    if market_name == "DA":
        price_df = framework.DA_prices_df
    elif market_name == "IDA1":
        price_df = framework.IDA1_prices_df
    elif market_name == "IDA2":
        price_df = framework.IDA2_prices_df
    elif market_name == "IDA3":
        price_df = framework.IDA3_prices_df
    else:
        raise ValueError(f"Invalid market name in get_cleared_market_prices: {market_name}.")
    
    prices = price_df["price_eur_per_kwh"].reindex(timestamps)
    if prices.isna().any():
        #raise ValueError(f"Cleared market prices for {market_name} contain NaN values for timestamps: {prices[prices.isna()].index}.") ### To be sure
        ### Update: markets (IDA's) can be cancelled/partially decoupled. For those, they should be skipped basically --> Set acceptance mask to 0 everywhere later on.
        missing_timestamps = prices[prices.isna()].index
        print(colored(f'{market_name} was not successful for timestamps:\n. Setting acceptance mask to 0 for those.\n  {missing_timestamps}', 'red'))

        return None
    
    return prices.sort_index()


def get_accepted_bids_mask(rounded_portfolio_bid, submitted_limit_prices, cleared_prices): ### Need mask, not bids direclty
    accepted_bids_mask = pd.Series(False, index=rounded_portfolio_bid.index)
    eps = 1e-6 ### Fix by chat, have to distinct zero volume bids from pos or neg volume bids

    for timestamp, bid_volume in rounded_portfolio_bid.items():
        submitted_limit_price = submitted_limit_prices[timestamp]
        published_price = cleared_prices[timestamp]

        if bid_volume > eps: ### Sell/export bid
            if submitted_limit_price <= published_price:
                accepted_bids_mask[timestamp] = True

        elif bid_volume < -eps: ### Buy/import bid
            if submitted_limit_price >= published_price:
                accepted_bids_mask[timestamp] = True
        
        else:
            accepted_bids_mask[timestamp] = False ### Zero volume bids are not accepted anyway

    return accepted_bids_mask.sort_index()


def final_household_commitment_according_to_clearing(rounded_household_bids_serie, accepted_bids_mask):
    accepted_mask_adjusted = accepted_bids_mask.reindex(rounded_household_bids_serie.index).fillna(False)
    cleared_commitment = rounded_household_bids_serie.where(accepted_mask_adjusted, 0.0).sort_index() ### Use float, not int. Np where easier, by chat

    return cleared_commitment


### Check whether sim charging/ discharging occurred
def check_simultaneous_battery_charge_discharge(model, tree_data, eps_bat=1e-4, eps_grid=1e-2):
    rows = []

    for name, stage, local_time in model.NST:
        E_charge = pyo.value(model.E_charge[name, stage, local_time])
        E_discharge = pyo.value(model.E_discharge[name, stage, local_time])

        E_grid_import = pyo.value(model.E_grid_import[name, stage, local_time])
        E_grid_export = pyo.value(model.E_grid_export[name, stage, local_time])

        if E_charge > eps_bat and E_discharge > eps_bat:
            timestamp = tree_data["local_time_to_timestamps"][int(stage)][int(local_time)]
            row = {
                "violation_type": "battery_simultaneous",
                "node": name,
                "local_time": int(local_time),
                "stage": int(stage),
                "timestamp": timestamp,
                "E_charge": E_charge,
                "E_discharge": E_discharge,
                "simultaneous_amount": min(E_charge, E_discharge)
            }
            for market_name, price_series in tree_data["market_prices_by_timestamp"].items():
                row[f"market_price_{market_name}"] = price_series[timestamp]
            
            rows.append(row)
        
        ### Grid should never be simultaneous --> Raise if so
        if E_grid_export > eps_grid and E_grid_import > eps_grid:
            timestamp = tree_data["local_time_to_timestamps"][int(stage)][int(local_time)]
            #raise ValueError(f"Simultaneous grid import/export detected at {timestamp}: E_grid_import={E_grid_import}, E_grid_export={E_grid_export}. This should never happen.")
            rows.append({
                "violation_type": "grid_simultaneous",
                "node": name,
                "local_time": int(local_time),
                "stage": int(stage),
                "timestamp": timestamp,
                "E_charge": E_charge, ### Put these to avoid crash
                "E_discharge": E_discharge,
                "E_grid_import": E_grid_import,
                "E_grid_export": E_grid_export,
                "simultaneous_amount": min(E_grid_import, E_grid_export)
            })


    sim_df = pd.DataFrame(rows)

    if len(sim_df) > 0:
        sim_summary = sim_df.groupby("timestamp").agg( ### otherwise way too many rows, by chat
            n_nodes=("node", "count"),
            mean_charge=("E_charge", "mean"),
            mean_discharge=("E_discharge", "mean"),
            max_simultaneous=("simultaneous_amount", "max")
        ).sort_index()
        for col in sim_df.columns:
            if col.startswith("market_price_"):
                sim_summary[col] = sim_df.groupby("timestamp")[col].first() ### Just take first, all same for a given timestamp

        if VERBOSITY >= 0:
            print(colored(f"WARNING: simultaneous battery charge/discharge in {len(sim_df)} node-timesteps.", "red"))
        #print(f"Simultaneous battery charge/discharge summary:\n{sim_summary.head(30)}")
        #print(sim_df.head(20).to_string())
    else:
        if VERBOSITY >= 0:
            print(colored("No simultaneous battery charge/discharge detected.", "green"))

    return sim_df


### Warm start functions, one for load and one for apply
def save_warm_start(model, tree_data, household, model_type):
    if not USE_WARM_START:
        return False

    warm_start = {"model_type": model_type, "variable_values": {}, "last_variable_values": {}}

    for var in model.component_objects(pyo.Var, active=True):
        for index in var:
            value = pyo.value(var[index])
            if value is None: 
                continue

            if isinstance(index, tuple) and len(index) == 3:
                node_name, stage, local_time = index
                timestamp = tree_data["local_time_to_timestamps"][int(stage)][int(local_time)]
                key = (var.name, node_name, timestamp)
                

            elif var.name == "E_market_bid":
                timestamp = tree_data["bid_step_to_timestamp"][int(index)]
                key = (var.name, timestamp)
                
            else: 
                continue

            warm_start["variable_values"][key] = float(value)
    
    warm_start_file = WARM_START_DIR / f"warm_start_household_{household}.joblib"
    #joblib.dump(warm_start, warm_start_file)
    safe_joblib_dump(warm_start, warm_start_file)
    return True


def apply_warm_start(model, tree_data, household, model_type):
    warm_start_file = WARM_START_DIR / f"warm_start_household_{household}.joblib"

    if not USE_WARM_START or not warm_start_file.exists():
        return False
    
    warm_start_joblib = joblib.load(warm_start_file)

    if WARM_START_ONLY_SAME_MODEL_TYPE and warm_start_joblib["model_type"] != model_type:
        return False
    
    old_values = warm_start_joblib["variable_values"]

    for var in model.component_objects(pyo.Var, active=True): ### Model already built
        for index in var:
            value_to_set = None

            if isinstance(index, tuple) and len(index) == 3: ### Good trick by chat. For all NST-type indexed vars
                node_name, stage, local_time = index
                ### Have to store per timestamp now and stage becomes redundant then too
                timestamp = tree_data["local_time_to_timestamps"][int(stage)][int(local_time)] ### debug, use int
                key = (var.name, node_name, timestamp) ### This can be unpacked later again without destroying now
                ### Forget about the last value, doesn't have to be set and is a headache. If wanted, add it here with a forward fill from last known stored value
            
            elif var.name == "E_market_bid": ### See mpc file again, has separate index, not NST
                timestamp = tree_data["bid_step_to_timestamp"][int(index)]
                key = (var.name, timestamp)

            else:
                continue


            value_to_set = old_values.get(key)
            ### If no exact timestamp match, use ffill (last horizon step)
            if value_to_set is None:
                continue

            if var[index].is_binary():
                value_to_set = round(value_to_set)
            
            try:
                var[index].set_value(value_to_set) ### Main block. try bu chat
            except ValueError:
                pass  ### Ignore, not all variables are present in the new model
    return True ### Successfull

def safe_to_parquet(df, path, index=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    if index is None:
        df.to_parquet(tmp_path)
    else:
        df.to_parquet(tmp_path, index=index)
    tmp_path.replace(path)


def safe_joblib_dump(obj, path):
    tmp_path = path.with_name(path.name + ".tmp")
    joblib.dump(obj, tmp_path)
    tmp_path.replace(path)


### Saving market events for post
def make_portfolio_wide_row(event_timestamp, market_name, event_type, values_series, fill_value=0.0):
    market_values_series = values_series.sort_index() ### 00:00 to 23:45, the market interval

    full_horizon_index_rebuilt = pd.date_range(start=event_timestamp, periods=PORTFOLIO_WIDE_N_STEPS, freq="15min", tz="Europe/Brussels")
    values_full = market_values_series.reindex(full_horizon_index_rebuilt, fill_value=fill_value) ### zero everywhere poutside market

    row = {
        "event_timestamp": event_timestamp,
        "market_name": market_name,
        "event_type": event_type,
        "horizon_start": full_horizon_index_rebuilt.min(),
        "horizon_end": full_horizon_index_rebuilt.max() + pd.Timedelta(minutes=15), ### Delta 15min by chat
        "market_delivery_start": market_values_series.index.min() if len(market_values_series) > 0 else pd.NaT, ### Keep it numerical
        "market_delivery_end": market_values_series.index.max() + pd.Timedelta(minutes=15) if len(market_values_series) > 0 else pd.NaT,
    }
    for ts_, value in values_full.items():
        row[ts_.strftime("%Y-%m-%d %H:%M:%S%z")] = float(value)

    row_as_df = pd.DataFrame([row]).set_index("event_timestamp")
    return row_as_df

def save_portfolio_wide_row(event_timestamp, market_name, event_type, values_series, fill_value=0.0):
    new_row_df = make_portfolio_wide_row(event_timestamp, market_name, event_type, values_series, fill_value=fill_value)
    if PORTFOLIO_SUBMISSIONS_FILE.exists():
        old_df = pd.read_parquet(PORTFOLIO_SUBMISSIONS_FILE)

        ### By chat, careful for overlap
        same_row_mask = ((old_df.index == event_timestamp) & (old_df["market_name"] == market_name) & (old_df["event_type"] == event_type))

        old_df = old_df[~same_row_mask] ### Remove any existing row with same event_timestamp, market_name, and event_type
        combined_df = pd.concat([old_df, new_row_df]).sort_index()
    else:
        combined_df = new_row_df.sort_index()
    
    #combined_df.to_parquet(PORTFOLIO_SUBMISSIONS_FILE)
    safe_to_parquet(combined_df, PORTFOLIO_SUBMISSIONS_FILE)




VERBOSITY = -1
USE_APRIORI_PARTIAL_BATTERY_BINARIES = False  ### Still kept this logic for deterministic case, should be able to predict perfectly when binaries are needed and where not, purely based on price.
AUTO_RERUN_MILP = True ### If LP contains simultaneous charge/discharge, rerun MILP with binaries
FORCE_FIRST_TIME_AS_LP = True ### For checking solve times
RESUME_FROM_CHECKPOINT = True ### Make optional to make individual runs
SAVE_RESULTS = True
USE_WARM_START = False
WARM_START_ONLY_SAME_MODEL_TYPE = True


SCENARIO_COUNTS = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)] ### First is PV, second is load. MAYBE CAN EXCLUDE DETERMINISTIC FROM LOAD.
USE_PV_RESIDUAL_SCENARIOS = any(n_pv > 0 for n_pv, n_load in SCENARIO_COUNTS) ### By chat, a lot shorter than looping
if USE_PV_RESIDUAL_SCENARIOS and VERBOSITY > 1:
    print(colored(f"Using PV residual scenarios for market aggregation.", "yellow"))
elif not USE_PV_RESIDUAL_SCENARIOS and VERBOSITY > 1:
    print(colored(f"Not using PV residual scenarios for market aggregation.", "yellow"))

DEVIATION_QUADRATIC_PENALTY = 0.0 ### ALWAYS LEAVE AT ZERO FOR OUR SIMULATION, destroys solving time. Will leave it here if I want to benchmark it later.
CLEAR_SKY_EXPORT_FRACTION = 1.0 ### Leave to one, better. 
### Instead of quadratic, implement two levels
DEVIATION_COST_PENALTY_LOW = 0.15 ### The former dev_pen
DEVIATION_COST_PENALTY_HIGH = 0.8 ### FOR NOW, FIRST SEE EFFECT OF CLEARSKY #0.2 ### Note that this comes additionally on top of the low penalty
DEVIATION_THRESHOLD_KWH = 0.075 ### 1kw JUST PRESS START IN THE MORNING, MPC c_deg HAS BEEN SET

### Balance delta
USE_BALANCE_DELTA = True
# Params to tune
PRICE_SCALE_EUR_PER_MWH = 50.0 ### Should probably be set around 50.0
SOC_SENSITIVITY = 0.3 ### Higher is more conservative
EXTREME_PRICE_THRESHOLD_EUR_PER_MWH = 150.0 ### 200 is upper 5%, do note that this is the difference with the mid price. Do ±90 for the raw value

### Add some reservation on the battery soc for imbalance
UPWARD_RESERVE_SOC = 0.20 ### self.battery_data.soc_min + upward_reserve_soc, soc_min is currently at 12%
DOWNWARD_RESERVE_SOC = 0.05 ### self.battery_data.soc_max - downward_reserve_soc
WEIGHT_SOC_SOFT = 20.0 ### Weight for the soft constraint. Should probably be set to 10.0

TO_AMOUNT = 8

def folder_number(value):### by chat
    return f"{value:g}".replace(".", "_")
### Automate the naming of the run
if all(n_pv == 0 and n_load == 0 for n_pv, n_load in SCENARIO_COUNTS):
    scenario_folder = "No_residuals"
else:
    scenario_folder = "With_residuals"


dev_pen_folder = f"devpen_L_{folder_number(DEVIATION_COST_PENALTY_LOW)}_H_{folder_number(DEVIATION_COST_PENALTY_HIGH)}_th_{folder_number(DEVIATION_THRESHOLD_KWH)}_cs_{folder_number(CLEAR_SKY_EXPORT_FRACTION)}"
reserve_soc_folder = f"res_soc_up_{folder_number(UPWARD_RESERVE_SOC)}_down_{folder_number(DOWNWARD_RESERVE_SOC)}_soft_w_{folder_number(WEIGHT_SOC_SOFT)}" ### Make sure paths aren't too long
balance_delta_folder = "with_BD" if USE_BALANCE_DELTA else "without_BD"

if USE_BALANCE_DELTA:
    RUN_NAME = f"Price_{folder_number(PRICE_SCALE_EUR_PER_MWH)}_Soc_{folder_number(SOC_SENSITIVITY)}_th_{folder_number(EXTREME_PRICE_THRESHOLD_EUR_PER_MWH)}"
else:
    RUN_NAME = "No_BD"


#RESULTS_DIR = Path(Path(__file__).parent) / "Results_MPC" / RUN_NAME ### Gives problems with gdrive, write locally and copy
SCRATCH_FOLDER = Path("/cluster/work/brechtg")
HOME_FOLDER = Path("/cluster/home/brechtg")

#RESULTS_ROOT = SCRATCH_FOLDER / "Market_MPC" / f"NL_Results_MPC_cdeg_{folder_number(BatteryData.c_deg)}"
RESULTS_ROOT = HOME_FOLDER / "Market_MPC" / f"NL_Results_MPC_cdeg_{folder_number(BatteryData.c_deg)}"
RESULTS_DIR = RESULTS_ROOT / f"H_{folder_number(4*TO_AMOUNT)}" / "Imb_real_personal_spring_fast_save" / scenario_folder / dev_pen_folder / reserve_soc_folder / balance_delta_folder / RUN_NAME
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WARM_START_DIR = RESULTS_DIR / "warm_starts"
WARM_START_DIR.mkdir(parents=True, exist_ok=True)

### The data files
HOUSEHOLD_RESULTS_DIR = RESULTS_DIR / "household_results"
HOUSEHOLD_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FAST_HOUSEHOLD_RESULTS_DIR = RESULTS_DIR / "fast_household_results"
FAST_HOUSEHOLD_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_SOC_FILE = HOUSEHOLD_RESULTS_DIR / "system_soc.parquet"
SOLVER_RESULTS_DIR = RESULTS_DIR / "solver_results"
SOLVER_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SIM_BATTERY_EVENTS_DIR = RESULTS_DIR / "sim_battery_events"
SIM_BATTERY_EVENTS_DIR.mkdir(parents=True, exist_ok=True)
MARKET_BID_RESULTS_DIR = RESULTS_DIR / "market_bid_results"
MARKET_BID_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PORTFOLIO_RESULTS_DIR = RESULTS_DIR / "portfolio_results"
PORTFOLIO_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PORTFOLIO_SUBMISSIONS_FILE = PORTFOLIO_RESULTS_DIR / "portfolio_submissions.parquet"
PORTFOLIO_WIDE_N_STEPS = 48*4 ### Can be adapted to what interval we wanna keep


### Balance delta
BALANCE_DELTA_FILE = SCRATCH_FOLDER / "Market_MPC" / "Markets" / "Markets Data_NL" / "Balancing_delta" / "Parquets" / "overall_df.parquet"
BALANCE_DELTA_DELAY = pd.Timedelta(minutes=2)

### The real data
REAL_LOAD_FILE = SCRATCH_FOLDER / "thesis_load_training" / "Fluvius 2400" / "Prepped" / "All_prepared_2_years_with_aggregate.parquet"
REAL_PV_FOLDER = SCRATCH_FOLDER / "Fluvius PV" / "Open-meteo Satellite" / "Satellite_Power_Virtual_Houses"

BALANCE_DELTA_DIR = RESULTS_DIR / "balance_delta_results"
BALANCE_DELTA_DIR.mkdir(parents=True, exist_ok=True)

### Write all params to a json file to keep track of the settings
params_run_dict = {
    "RUN_NAME": RUN_NAME,
    "SCENARIO_COUNTS": SCENARIO_COUNTS,
    "DEVIATION_COST_PENALTY": DEVIATION_COST_PENALTY_LOW,
    "DEVIATION_COST_PENALTY_HIGH": DEVIATION_COST_PENALTY_HIGH,
    "DEVIATION_THRESHOLD_KWH": DEVIATION_THRESHOLD_KWH,
    "DEVIATION_QUADRATIC_PENALTY": DEVIATION_QUADRATIC_PENALTY,
    "CLEAR_SKY_EXPORT_FRACTION": CLEAR_SKY_EXPORT_FRACTION,
    "USE_BALANCE_DELTA": USE_BALANCE_DELTA,
    "PRICE_SCALE_EUR_PER_MWH": PRICE_SCALE_EUR_PER_MWH,
    "SOC_SENSITIVITY": SOC_SENSITIVITY,
    "EXTREME_PRICE_THRESHOLD_EUR_PER_MWH": EXTREME_PRICE_THRESHOLD_EUR_PER_MWH,
    "UPWARD_RESERVE_SOC": UPWARD_RESERVE_SOC,
    "DOWNWARD_RESERVE_SOC": DOWNWARD_RESERVE_SOC,
    "WEIGHT_SOC_SOFT": WEIGHT_SOC_SOFT,
}



def ensure_result_dirs_exist():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    WARM_START_DIR.mkdir(parents=True, exist_ok=True)
    HOUSEHOLD_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SOLVER_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SIM_BATTERY_EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    MARKET_BID_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PORTFOLIO_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    BALANCE_DELTA_DIR.mkdir(parents=True, exist_ok=True)
    FAST_HOUSEHOLD_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
### Parallelisation
N_HOUSEHOLD_WORKERS = 56 ### Put this to 8 for delftblue probably or even 16
### Some solver settings, set inside solve for one household now
GUROBI_THREADS = 2 ### Cleanup will be done afterwards
MILP_RERUN_GUROBI_THREADS = 4
MIPGAP = 0.02 #### SET NONE FOR DEFAULT




WORKER_OBJECT_CACHE = {}

def get_worker_objects(household):
    if household not in WORKER_OBJECT_CACHE:
        framework = Framework(virtual_household=household, exclude_before_ts="2024-01-22 00:00", use_pv_residual_scenarios=USE_PV_RESIDUAL_SCENARIOS)
        mpc = MPC(battery_data=framework.battery_data, contract_data=framework.contract_data, pv_system_power_kw=framework.pv_system_power_kw)
        WORKER_OBJECT_CACHE[household] = (framework, mpc)
    
    return WORKER_OBJECT_CACHE[household]

def split_household_chunks(households, n_chunks): ### Every worker each time holds the same chunck of households
    return [households[i::n_chunks] for i in range(n_chunks)] ### Easy, by chat.


def solve_one_household_for_timestep(args_worker): ### Rewrite without all the print, becomes too unstructured
        household = args_worker["household"]
        simulation_step = args_worker["simulation_step"]
        horizon_end = args_worker["horizon_end"]
        market_name = args_worker["market_name"]
        bid_start = args_worker["bid_start"]
        bid_end = args_worker["bid_end"]
        fixed_market_commitment = args_worker["fixed_market_commitment"]
        soc_init = args_worker["soc_init"]
        balance_delta_interval_df = args_worker["balance_delta_interval_df"]
        real_load_kwh = args_worker["real_load_kwh"]
        real_pv_kwh = args_worker["real_pv_kwh"]

        ### Move caching to per household for each timestep, becomes too heavy for many households
        framework, mpc = get_worker_objects(household)
        framework.battery_data.soc_init_mpc_run = soc_init ### Have to immediately overwrite, we don't keep the framework cached anymore

        start_time = time.time()
        nodes = framework.build_scenario_tree(
            simulation_step=simulation_step,
            horizon_end=horizon_end,
            branch_hours_utc=[0, 6, 12, 18],
            scenario_counts=SCENARIO_COUNTS,
            include_deterministic_load=True,
            include_deterministic_pv=True,
            extend_last_nonzero_branch_to_horizon_end=True
        )

        tree_data = framework.nodes_to_pyomo_per_stage(nodes)
        tree_data = framework.add_market_prices_to_tree(tree_data=tree_data, simulation_step=simulation_step)

        battery_binary_needed_timestamps, potential_violations_df = should_linearise_battery_constraint(tree_data=tree_data, battery_data=framework.battery_data, contract_data=framework.contract_data)

        if FORCE_FIRST_TIME_AS_LP:
            linearise_battery_constraint = True
            battery_binary_timestamps = []
        elif len(battery_binary_needed_timestamps) == 0:
            linearise_battery_constraint = True
            battery_binary_timestamps = []
        else:
            if USE_APRIORI_PARTIAL_BATTERY_BINARIES:
                linearise_battery_constraint = True
                battery_binary_timestamps = battery_binary_needed_timestamps
            else:
                linearise_battery_constraint = False
                battery_binary_timestamps = None
        

        ### For warm-starting, detect current model
        if linearise_battery_constraint and len(battery_binary_timestamps) == 0:
            current_model_type = "LP"
        elif linearise_battery_constraint and len(battery_binary_timestamps) > 0:
            current_model_type = "MILP_PARTIAL_BINARIES"
        else:
            current_model_type = "MILP_FULL_BINARIES"

        ### For printing at the end
        if current_model_type == "LP":
            model_used_during_ts_run = "LP"
        else:
            model_used_during_ts_run = "MILP_FROM_START"

        full_mpc_start_time = time.time()
        model = mpc.Build_MPC_model(
            tree_data=tree_data,
            market_name=market_name,
            bid_start=bid_start,
            bid_end=bid_end,
            fixed_market_commitment=fixed_market_commitment, ### Comes externally
            deviation_cost_penalty_low=DEVIATION_COST_PENALTY_LOW, ### Still has to be tuned
            deviation_cost_penalty_high=DEVIATION_COST_PENALTY_HIGH,
            deviation_threshold_kwh=DEVIATION_THRESHOLD_KWH,
            deviation_quadratic_penalty=DEVIATION_QUADRATIC_PENALTY,
            clear_sky_export_fraction=CLEAR_SKY_EXPORT_FRACTION,
            upward_reserve_soc=UPWARD_RESERVE_SOC,
            downward_reserve_soc=DOWNWARD_RESERVE_SOC,
            weight_soc_soft=WEIGHT_SOC_SOFT,
            linearise_grid_constraint=True,
            linearise_battery_constraint=linearise_battery_constraint,
            battery_binary_timestamps=battery_binary_timestamps
        )

        warm_start_used_bool = apply_warm_start(model=model, tree_data=tree_data, household=household, model_type=current_model_type)
        if warm_start_used_bool and VERBOSITY >= 1:
            print(colored(f"Warm start applied for household {household} at {simulation_step}.", "green"))
        elif not warm_start_used_bool and VERBOSITY >= 1:
            print(colored(f"No warm start applied for household {household} at {simulation_step}.", "yellow"))

        local_solver = SolverFactory("gurobi")
        local_solver.options["Threads"] = GUROBI_THREADS
        if MIPGAP is not None:
            local_solver.options["MIPGap"] = MIPGAP
        
        result = local_solver.solve(model, tee=False, load_solutions=True, warmstart=warm_start_used_bool)
        full_mpc_end_time = time.time()
        how_terminated = str(result.solver.termination_condition)
        if how_terminated not in ["optimal", "feasible"]:
            raise RuntimeError(f"Solver did not terminate optimally for household {household} at timestep {simulation_step}. Termination: {how_terminated}")
        if VERBOSITY >= 1:
            print(colored(f"Household {household} solved for timestep {simulation_step} with model type {current_model_type}. Termination: {how_terminated}. Solve time: {full_mpc_end_time - full_mpc_start_time:.2f}s", "blue"))

        sim_battery_df = check_simultaneous_battery_charge_discharge(model=model, tree_data=tree_data, eps_bat=1e-4, eps_grid=1e-2)
        if len(sim_battery_df) > 0: ### Add metadata to sim_battery_df
            sim_battery_df = sim_battery_df.copy()
            sim_battery_df["simulation_step"] = simulation_step
            sim_battery_df["household"] = household
            sim_battery_df["model_type"] = current_model_type


        grid_violation_detected = (len(sim_battery_df) > 0 and "violation_type" in sim_battery_df.columns and (sim_battery_df["violation_type"] == "grid_simultaneous").any())
        battery_violation_detected = (len(sim_battery_df) > 0 and "violation_type" in sim_battery_df.columns and (sim_battery_df["violation_type"] == "battery_simultaneous").any())
        if AUTO_RERUN_MILP and (battery_violation_detected or grid_violation_detected):
            if VERBOSITY >= 0:
                print("Rerunning with MILP due to simultaneous battery/grid operation detected.")
            current_model_type = "MILP_FULL_BINARIES"
            model_used_during_ts_run = "RERUN_MILP"
            full_mpc_start_time = time.time()
            model = mpc.Build_MPC_model(
                tree_data=tree_data,
                market_name=market_name,
                bid_start=bid_start,
                bid_end=bid_end,
                fixed_market_commitment=fixed_market_commitment,
                deviation_cost_penalty_low=DEVIATION_COST_PENALTY_LOW,
                deviation_cost_penalty_high=DEVIATION_COST_PENALTY_HIGH,
                deviation_threshold_kwh=DEVIATION_THRESHOLD_KWH,
                deviation_quadratic_penalty=DEVIATION_QUADRATIC_PENALTY,
                clear_sky_export_fraction=CLEAR_SKY_EXPORT_FRACTION,
                upward_reserve_soc=UPWARD_RESERVE_SOC,
                downward_reserve_soc=DOWNWARD_RESERVE_SOC,
                weight_soc_soft=WEIGHT_SOC_SOFT,
                linearise_grid_constraint= not grid_violation_detected,
                linearise_battery_constraint= not battery_violation_detected,
                battery_binary_timestamps=None
            )

            warm_start_used_bool = apply_warm_start(model=model, tree_data=tree_data, household=household, model_type=current_model_type)
            if warm_start_used_bool and VERBOSITY >= 1:
                print(colored(f"Warm start applied for household {household} at {simulation_step} for MILP rerun.", "green"))
            elif not warm_start_used_bool and VERBOSITY >= 1:
                print(colored(f"No warm start applied for household {household} at {simulation_step} for MILP rerun.", "yellow"))
            local_solver.options["Threads"] = MILP_RERUN_GUROBI_THREADS ### Be careful with this, I think it can hurt too if too many threads are used and acting against eachother.
            result = local_solver.solve(model, tee=False, load_solutions=True, warmstart=warm_start_used_bool)
            full_mpc_end_time = time.time()
            how_terminated = str(result.solver.termination_condition)
            if how_terminated not in ["optimal", "feasible"]:
                raise RuntimeError(f"Solver did not terminate optimally for household {household} at timestep {simulation_step}. Termination: {how_terminated}")
            if VERBOSITY >= 0:
                print(colored(f"Household {household} rerun with MILP for timestep {simulation_step}. Termination: {how_terminated}. Solve time: {full_mpc_end_time - full_mpc_start_time:.2f}s", "blue"))
            sim_battery_df = check_simultaneous_battery_charge_discharge(model=model, tree_data=tree_data, eps_bat=1e-4, eps_grid=1e-2)

        
        save_warm_start(model=model, tree_data=tree_data, household=household, model_type=current_model_type)

        next_step_action = extract_next_step_action(model=model, tree_data=tree_data)
        ### Now implement balance delta
        ### Pass real data
        next_step_action, balance_delta_saving_df, soc_next = apply_balance_delta_after_solving(next_step_action=next_step_action, battery_data=framework.battery_data, simulation_step=simulation_step, household=household, fixed_market_commitment=fixed_market_commitment, balance_delta_interval_df=balance_delta_interval_df, real_load_kwh=real_load_kwh, real_pv_kwh=real_pv_kwh)
        #soc_next = update_soc(battery_data=framework.battery_data, next_step_action_dict=next_step_action)


        Household_saving_df = pd.DataFrame([{ ### Add balance delta details too
            "simulation_step": simulation_step,
            "household": household,
            "SOC_start": next_step_action["SOC_start"],
            "E_charge": next_step_action["E_charge"],
            "E_discharge": next_step_action["E_discharge"],
            "E_grid_import": next_step_action["E_grid_import"],
            "E_grid_export": next_step_action["E_grid_export"],
            "SOC_next": soc_next, ### This is after balance delta now!!!
            "battery_capacity_kwh": framework.battery_data.Vmax, ### Just add it here, will be repeated for every row but doesn't matter for now, avoids having to import all separate files in post
            "model_type": current_model_type,
            "E_balance_delta_correction": next_step_action["E_balance_delta_correction"],
            "imbalance_before_balance_delta": next_step_action["imbalance_before_balance_delta"],
            "imbalance_after_balance_delta": next_step_action["imbalance_after_balance_delta"],
            "SOC_next_without_balance_delta": next_step_action["SOC_next_without_balance_delta"]
        }])
        ### Do not export yet, do this in parent. Can create I/O problems, by chat

        Solver_saving_df = pd.DataFrame([{
            "simulation_step": simulation_step,
            "household": household,
            "model_type": current_model_type,
            "model_used_during_ts_run": model_used_during_ts_run,
            "warm_start_used": warm_start_used_bool,
            "termination_condition": how_terminated,
            "solver_time_seconds": full_mpc_end_time - full_mpc_start_time
        }])

        ### Extract market bid for this household
        household_bid = None ### Make sure they are None for non-market timesteps
        forecasted_prices_for_bid = None

        market_bid_rows = []
        if market_name is not None:
            household_bid = extract_market_bid(model=model, tree_data=tree_data) ### Have to bring to outer loop too, don't write into household bids overall dict here
            forecasted_prices_for_bid = extract_market_bid_prices(tree_data=tree_data, market_name=market_name, bid_start=bid_start, bid_end=bid_end) ### Made by reference framework now
            
            for delivery_ts, bid_volume in household_bid.items():
                market_bid_rows.append({
                    "simulation_step": simulation_step,
                    "household": household,
                    "market_name": market_name,
                    "delivery_timestamp": delivery_ts,
                    "raw_household_bid_kwh": bid_volume
                })
            market_bid_household_df = pd.DataFrame(market_bid_rows)
        
        else:
            market_bid_household_df = pd.DataFrame()
        
        if len(sim_battery_df) == 0:
            sim_battery_df = pd.DataFrame()


        
        ### HAVE TO TRY SOMETHING ELSE, MEMORY IS NOT RELEASED PROPERLY
        out = { ### Pack all info to be saved or needed for next run
            "household": household,
            "next_step_action": next_step_action,
            "soc_next": soc_next,
            "household_bid": household_bid,
            "forecasted_prices_for_bid": forecasted_prices_for_bid,
            "household_saving_df": Household_saving_df,
            "solver_saving_df": Solver_saving_df,
            "sim_battery_df": sim_battery_df,
            "market_bid_household_df": market_bid_household_df,
            "balance_delta_saving_df": balance_delta_saving_df
        }
        try:
            del model
        except NameError:
            pass
        try:
            del result
        except NameError:
            pass
        try:
            del local_solver
        except NameError:
            pass
        try:
            del nodes
        except NameError:
            pass
        try:
            del tree_data
        except NameError:
            pass
        try:
            del potential_violations_df
        except NameError:
            pass

        gc.collect()

        return out




def solve_household_chunck_for_timestep(args_chunck): ### Wrapper around solve_one_household_for_timestep like previous
    
    current_households = set(int(args_worker["household"]) for args_worker in args_chunck["worker_args_list"])

    for cache_key in list(WORKER_OBJECT_CACHE.keys()): ### By chat, to try fix memory leak
        cached_household = cache_key[0] if isinstance(cache_key, tuple) else cache_key
        if int(cached_household) not in current_households:
            del WORKER_OBJECT_CACHE[cache_key]

    gc.collect()

    results = []
    for args_worker in args_chunck["worker_args_list"]:
        result = solve_one_household_for_timestep(args_worker)
        results.append(result)

    return results




if __name__ == "__main__":
    MAX_TIME_SECONDS = 23 * 3600 + 30 * 60
    run_start_perf = time.perf_counter()
    ensure_result_dirs_exist()
    if USE_BALANCE_DELTA:
        balance_delta_df = prepare_balance_delta_df(pd.read_parquet(BALANCE_DELTA_FILE))
    else:
        balance_delta_df = pd.DataFrame()
    from_amount = 0
    to_amount = TO_AMOUNT ### Total gives four times the difference between these two

    households_load = [i for i in range(301, 601)] ### 301-600 are load households
    households_HP = [i for i in range(901, 1201)] ### 901-1200 are heat pump households
    households_EV = [i for i in range(1501, 1801)] ### 1501-1800 are EV households
    households_HP_EV = [i for i in range(2101, 2401)] ### 2101-2400 are HP+EV households

    all_households = sorted(households_load[from_amount:to_amount] + households_HP[from_amount:to_amount] + households_EV[from_amount:to_amount] + households_HP_EV[from_amount:to_amount])

    ### OVERWRITE
    #all_households = [301, 302]

    actual_workers = min(N_HOUSEHOLD_WORKERS, len(all_households)) ### Deosn't have to be recalculated
    household_chunks = split_household_chunks(all_households, actual_workers)

    ### Real data
    real_load_df = pd.read_parquet(REAL_LOAD_FILE)
    real_load_df = real_load_df[all_households]

    real_pv_by_household = {}
    for household in all_households:
        pv_file = REAL_PV_FOLDER / f"SatellitePower_load_id_{household}.parquet"
        pv_power_kw = pd.read_parquet(pv_file, columns=["satellite_fitting_power_kw"])["satellite_fitting_power_kw"]
        real_pv_by_household[household] = pv_power_kw * 0.25




    ### Initial setup for multiprocessing
    household_market_commitments_dict = {}
    soc_by_household = {}

    for household in all_households:
        household_market_commitments_dict[household] = MarketAggregate()
        soc_by_household[household] = 0.5 ### Easier to make separate dict

    ### One framework for parent to read prices, loads full DA, IDA1-3 price
    reference_framework_for_prices = Framework(virtual_household=all_households[0], exclude_before_ts="2024-01-22 00:00", use_pv_residual_scenarios=False)

    print(colored(f"Initialised parent framework for cleared prices with household {all_households[0]}.", "green"))




    raw_household_bids_by_market = {}
    rounded_household_bids_by_market = {}

    raw_portfolio_bid_by_market = {}
    rounded_portfolio_bid_by_market = {}

    submitted_limit_prices_by_market = {}


    household_results = []
    portfolio_results = []
    clearing_results = []

    solver_results = []
    sim_battery_events = []
    market_bid_results = []

    last_solution_by_household = {}


    market_details = MarketDetails() ### Simple class to store info on market volume step and rounding threshold

    HORIZON_HOURS = 48
    ### Should start on a normal timestep, just to be sure.
    SIMULATION_STEPS = pd.date_range(start="2026-01-01 02:00:00", end="2026-06-30 00:00:00", freq="15min", tz="Europe/Brussels", inclusive="left") ### From on September to have load scenarios (and pv)
    params_run_dict["from_amount"] = from_amount
    params_run_dict["to_amount"] = to_amount
    params_run_dict["SIM_START_TS"] = str(SIMULATION_STEPS[0])
    params_run_dict["SIM_END_TS"] = str(SIMULATION_STEPS[-1])
    
    # Save in results dir
    json_path = RESULTS_DIR / "run_params.json"
    if not json_path.exists():
        with open(json_path, "w") as f:
            json.dump(params_run_dict, f, indent=4)
    else:
        print(colored(f"Run parameters file already exists at {json_path}. Not overwriting.", "yellow"))

    exclude_start_ts = pd.Timestamp("2025-12-30 00:00:00", tz="Europe/Brussels")
    exclude_end_ts = pd.Timestamp("2026-01-22 00:30:00", tz="Europe/Brussels") ### Small mistake made so shifted 15min, not the accuracy of the models, they are correctly alligned, just that new models become available on timestep later
    exclude_mask = (SIMULATION_STEPS >= exclude_start_ts) & (SIMULATION_STEPS < exclude_end_ts)
    SIMULATION_STEPS = SIMULATION_STEPS[~exclude_mask] ### Exclude the period where we don't have load scenarios
    ### Continuation logic
    checkpoint_file = RESULTS_DIR / "checkpoint.joblib" ### Per timestep
    if RESUME_FROM_CHECKPOINT and checkpoint_file.exists():
        checkpoint_joblib = joblib.load(checkpoint_file)

        soc_by_household = checkpoint_joblib["soc_by_household"] ### Just load in once for all households

        household_market_commitments_dict = checkpoint_joblib["household_market_commitments_dict"]
        raw_household_bids_by_market = checkpoint_joblib["raw_household_bids_by_market"]
        rounded_household_bids_by_market = checkpoint_joblib["rounded_household_bids_by_market"]
        raw_portfolio_bid_by_market = checkpoint_joblib["raw_portfolio_bid_by_market"]
        rounded_portfolio_bid_by_market = checkpoint_joblib["rounded_portfolio_bid_by_market"]
        submitted_limit_prices_by_market = checkpoint_joblib["submitted_limit_prices_by_market"]

        ### One-time repair of checkpoint created with DST-unsafe market windows
        if checkpoint_joblib["next_simulation_step"] == pd.Timestamp("2026-03-30 13:00:00", tz="Europe/Brussels"):
            bad_start = pd.Timestamp("2026-03-31 00:00:00", tz="Europe/Brussels")
            bad_end = pd.Timestamp("2026-03-31 01:00:00", tz="Europe/Brussels")

            removed = 0
            for market_commitments in household_market_commitments_dict.values():
                for attr in ["DA_cleared_commitment", "IDA1_cleared_commitment", "IDA2_cleared_commitment"]:
                    serie = getattr(market_commitments, attr)
                    bad_mask = (serie.index >= bad_start) & (serie.index < bad_end)
                    removed += int(bad_mask.sum())
                    setattr(market_commitments, attr, serie[~bad_mask].sort_index())

            print(colored(f"Repaired spring-DST checkpoint: removed {removed} erroneous cleared commitment entries.", "yellow"))

        SIMULATION_STEPS = SIMULATION_STEPS[SIMULATION_STEPS >= checkpoint_joblib["next_simulation_step"]]

        print(colored(f"Resuming simulation from checkpoint at {checkpoint_joblib['next_simulation_step']}. Remaining steps: {len(SIMULATION_STEPS)}", "green"))




    with ProcessPoolExecutor(max_workers=actual_workers) as executor:
        ### The simulation loop, NOW INSIDE PROCESSPOOL
        for simulation_step in SIMULATION_STEPS:
            elapsed_time_since_start = time.perf_counter() - run_start_perf
                                    
            """if elapsed_time_since_start >= MAX_TIME_SECONDS:
                print(colored("\nMaximum simulation time reached. Stopping before starting another simulation step.", "red"))
                break"""
            print(colored(f"\nStarting simulation step {simulation_step}", "green"))

            time_start_step = time.time()

            ### Check if we're starting again in January
            if simulation_step == exclude_end_ts:
                for household in all_households:
                    household_market_commitments_dict[household] = MarketAggregate()
                    soc_by_household[household] = 0.5 ### Easier to make separate dict
                raw_household_bids_by_market.clear()
                rounded_household_bids_by_market.clear()
                raw_portfolio_bid_by_market.clear()
                rounded_portfolio_bid_by_market.clear()
                submitted_limit_prices_by_market.clear()

                print(colored(f"Restarted simulation with empty market state at {simulation_step}.", "yellow"))


            #######################
            ##### CLEARING TS #####
            #######################
            ### First check if any market gets cleared at this ts
            clearing_market_name = get_clearing_instances_for_timestep(simulation_step) ### Returns DA, IDA1, IDA2, IDA3 or None if timestamp falls on 13:00, 15:30, 22:30 or 10:30. This is hardcoded for now according to TS. If every lowered in period, should put this to the next timestep after market publicatiuon.

            if clearing_market_name is not None: ### It is a bidding timestep for a market
                if clearing_market_name in rounded_portfolio_bid_by_market: 
                    reference_framework = reference_framework_for_prices ### Already got one framework at init

                    published_prices = get_cleared_market_prices(framework = reference_framework, market_name=clearing_market_name, timestamps=rounded_portfolio_bid_by_market[clearing_market_name].index)
                    ### Now pass limited prices instead of raw
                    #accepted_bids_mask = get_accepted_bids_mask(rounded_portfolio_bid=rounded_portfolio_bid_by_market[clearing_market_name], submitted_limit_prices=submitted_limit_prices_by_market[clearing_market_name], cleared_prices=published_prices)
                    ### Update logic for unsuccessful markets
                    if published_prices is None:
                        #accepted_bids_mask = pd.Series(False, index=rounded_portfolio_bid_by_market[clearing_market_name].index) ### All rejected
                        #cleared_portfolio_bid = pd.Series(0.0, index=rounded_portfolio_bid_by_market[clearing_market_name].index) ### All rejected
                        #market_successful_mask = pd.Series(0.0, index=rounded_portfolio_bid_by_market[clearing_market_name].index) ### This will be saved as a marker for unsuccessful markets
                        ### Update: Make this just simpler
                        market_index = rounded_portfolio_bid_by_market[clearing_market_name].index
                        if len(market_index) == 0:
                            DAY = simulation_step.normalize()
                            if clearing_market_name == "IDA3":
                                market_index = pd.date_range(start=DAY.replace(hour=12), end=DAY + pd.DateOffset(days=1), freq="15min", tz="Europe/Brussels", inclusive="left")
                            else:
                                market_index = pd.date_range(start=DAY + pd.DateOffset(days=1), end=DAY + pd.DateOffset(days=2), freq="15min", tz="Europe/Brussels", inclusive="left")

                        accepted_bids_mask = pd.Series(False, index=market_index) ### All rejected
                        cleared_portfolio_bid = pd.Series(0.0, index=market_index) ### All rejected
                        market_successful_mask = pd.Series(0.0, index=market_index) ### This will be saved as a marker for unsuccessful markets


                        save_portfolio_wide_row(event_timestamp=simulation_step, market_name=clearing_market_name, event_type="cleared_portfolio_bid", values_series=cleared_portfolio_bid)
                        save_portfolio_wide_row(event_timestamp=simulation_step, market_name=clearing_market_name, event_type="accepted_bids_mask", values_series=accepted_bids_mask.astype(float), fill_value=float("nan"))
                        save_portfolio_wide_row(event_timestamp=simulation_step, market_name=clearing_market_name, event_type="market_successful_mask", values_series=market_successful_mask, fill_value=float("nan")) ### Update: Save with nans outside of market timesteps

                        for household in all_households:
                            cleared_commitment = final_household_commitment_according_to_clearing(rounded_household_bids_serie=rounded_household_bids_by_market[clearing_market_name][household], accepted_bids_mask=accepted_bids_mask)
                            household_market_commitments_dict[household].set_cleared_commitment(market_name=clearing_market_name, commitment_series=cleared_commitment)

                    
                        print(colored(f"{clearing_market_name} skipped at {simulation_step} due to unsuccessful market. All bids rejected.", "red"))
                    

                    else:
                        accepted_bids_mask = get_accepted_bids_mask(rounded_portfolio_bid=rounded_portfolio_bid_by_market[clearing_market_name], submitted_limit_prices=submitted_limit_prices_by_market[clearing_market_name], cleared_prices=published_prices)
                        cleared_portfolio_bid = rounded_portfolio_bid_by_market[clearing_market_name].where(accepted_bids_mask, 0.0).sort_index()
                        market_successful_mask = pd.Series(1.0, index=rounded_portfolio_bid_by_market[clearing_market_name].index)

                        save_portfolio_wide_row(event_timestamp=simulation_step, market_name=clearing_market_name, event_type="cleared_portfolio_bid", values_series=cleared_portfolio_bid)
                        save_portfolio_wide_row(event_timestamp=simulation_step, market_name=clearing_market_name, event_type="accepted_bids_mask", values_series=accepted_bids_mask.astype(float), fill_value=float("nan"))
                        save_portfolio_wide_row(event_timestamp=simulation_step, market_name=clearing_market_name, event_type="market_successful_mask", values_series=market_successful_mask, fill_value=float("nan"))

                        for household in all_households:
                            cleared_commitment = final_household_commitment_according_to_clearing(rounded_household_bids_serie=rounded_household_bids_by_market[clearing_market_name][household], accepted_bids_mask=accepted_bids_mask)
                            household_market_commitments_dict[household].set_cleared_commitment(market_name=clearing_market_name, commitment_series=cleared_commitment)
                        
                        if VERBOSITY >= 0:
                            print(colored(f"{clearing_market_name} cleared at {simulation_step}.", "yellow"))
                
                else:
                    print(colored(f"No rounded portfolio bid found for {clearing_market_name} at {simulation_step}. Cannot perform clearing.", "red"))



            ######################
            #### ROUNDING TS #####
            ######################
            ### Check if markets should be rounded for this ts. Simulates real life behaviour pretty good I think
            rounding_market_name = get_rounding_instances_for_timestep(simulation_step)

            if rounding_market_name is not None:
                if rounding_market_name in raw_portfolio_bid_by_market:
                    portfolio_raw_bid = raw_portfolio_bid_by_market[rounding_market_name] ### Make sure this is always synced to current time instance
                    
                    rounded_portfolio_bid, residual_array = round_bid_value(portfolio_raw_bid, market_details)

                    corrected_household_bids_dict = allocate_residuals_to_households(household_bids_dict=raw_household_bids_by_market[rounding_market_name], residual_array=residual_array)

                    rounded_household_bids_by_market[rounding_market_name] = corrected_household_bids_dict ### Overwrite
                    rounded_portfolio_bid_by_market[rounding_market_name] = rounded_portfolio_bid ### Overwrite

                    ### Save rounded portfolio bid for this market
                    save_portfolio_wide_row(event_timestamp=simulation_step, market_name=rounding_market_name, event_type="rounded_portfolio_bid", values_series=rounded_portfolio_bid)

                    for household in all_households:
                        household_market_commitments_dict[household].set_rounded_commitment(market_name=rounding_market_name, commitment_series=corrected_household_bids_dict[household])
                    
                    portfolio_results.append({
                        "simulation_step": simulation_step,
                        "market_name": rounding_market_name
                        ### ADD STATISTICS HERE LATER
                    })
                    if VERBOSITY >= 0:
                        print(colored(f"{rounding_market_name} rounded at {simulation_step}", "yellow"))
                else:
                    if VERBOSITY >= 0:
                        print(colored(f"No raw portfolio bid found for {rounding_market_name} at {simulation_step}. Cannot perform rounding.", "red"))


            #####################
            #### BIDDING TS #####
            #####################
            ### Check if any market is open for bidding at this ts.
            market_name, bid_start, bid_end = get_market_for_timestep(simulation_step)
            ### Fix for gapping the exclude interval. Have to empty bids and reset soc
            if market_name is not None and exclude_start_ts <= bid_start < exclude_end_ts: ### Use bid_start, not simulation_step, fix by chat
                raw_household_bids_by_market.pop(market_name, None)
                rounded_household_bids_by_market.pop(market_name, None)
                raw_portfolio_bid_by_market.pop(market_name, None)
                rounded_portfolio_bid_by_market.pop(market_name, None)
                submitted_limit_prices_by_market.pop(market_name, None)

                market_name = None ### Makes that the rest of the code doesn't make bids etc
                bid_start = None
                bid_end = None
            horizon_end = simulation_step + pd.Timedelta(hours=HORIZON_HOURS) ### 48h horizon strict!






            ### Solve for each household. Should be parallelised later
            if USE_BALANCE_DELTA:
                balance_delta_interval_df = get_balance_delta_for_interval(balance_delta_df=balance_delta_df, simulation_step=simulation_step)
            else:
                balance_delta_interval_df = pd.DataFrame()

            real_load_timestamp = simulation_step.replace(year=2024) ### Same changes as in distributed reduction script
            worker_args_list = []
            for household in all_households:
                worker_args_list.append({
                    "household": household,
                    "simulation_step": simulation_step,
                    "horizon_end": horizon_end,
                    "market_name": market_name,
                    "bid_start": bid_start,
                    "bid_end": bid_end,
                    "fixed_market_commitment": household_market_commitments_dict[household].total_commitment(), ### Now load it here from parent dict
                    "soc_init": soc_by_household[household],
                    "balance_delta_interval_df": balance_delta_interval_df,
                    "real_load_kwh": float(real_load_df.at[real_load_timestamp, household]),
                    "real_pv_kwh": float(real_pv_by_household[household].at[simulation_step])
                })

            household_bids = {}
            household_actions = {}
            household_saving_dfs = []
            solver_saving_dfs = []
            sim_battery_dfs = []
            market_bid_saving_dfs = []
            balance_delta_saving_dfs = []
            forecasted_prices_for_bid = None


            args_by_household = {args["household"]: args for args in worker_args_list}

            futures = []
            for household_chunck in household_chunks:
                args_chunck = {"worker_args_list": [args_by_household[household] for household in household_chunck]}
                futures.append(executor.submit(solve_household_chunck_for_timestep, args_chunck))

            received_households = set()
            for future in as_completed(futures):
                chunck_results = future.result()
                for result in chunck_results: ### All the same
                    household = result["household"]

                    received_households.add(household)
                    missing_households = sorted(set(all_households) - received_households)
                    #print(colored(f"Missing households: {missing_households}", "yellow")) ### Bit sketchy but well

                    household_actions[household] = result["next_step_action"]
                    soc_by_household[household] = result["soc_next"]
                    household_saving_dfs.append(result["household_saving_df"])
                    solver_saving_dfs.append(result["solver_saving_df"])

                    if len(result["sim_battery_df"]) > 0:
                        sim_battery_dfs.append(result["sim_battery_df"])
                    if market_name is not None:
                        household_bids[household] = result["household_bid"]
                        market_bid_saving_dfs.append(result["market_bid_household_df"])

                        if forecasted_prices_for_bid is None:
                            forecasted_prices_for_bid = result["forecasted_prices_for_bid"]
                    
                    if len(result["balance_delta_saving_df"]) > 0:
                        balance_delta_saving_dfs.append(result["balance_delta_saving_df"])

                    
                    if VERBOSITY >= 1:
                        print(colored(f"Household {household} completed for timestep {simulation_step}.", "cyan"))
            """futures = [executor.submit(solve_one_household_for_timestep, args) for args in worker_args_list]

            for future in as_completed(futures): ### Function waits until completion of all
                result = future.result()
                household = result["household"]
                household_actions[household] = result["next_step_action"]
                soc_by_household[household] = result["soc_next"]
                household_saving_dfs.append(result["household_saving_df"])
                solver_saving_dfs.append(result["solver_saving_df"])

                if len(result["sim_battery_df"]) > 0:
                    sim_battery_dfs.append(result["sim_battery_df"])
                if market_name is not None:
                    household_bids[household] = result["household_bid"]
                    market_bid_saving_dfs.append(result["market_bid_household_df"])

                    if forecasted_prices_for_bid is None:
                        forecasted_prices_for_bid = result["forecasted_prices_for_bid"] ### Just take the first one, all should be the same for a given market

                if VERBOSITY >= 1:
                    print(colored(f"Household {household} completed for timestep {simulation_step}.", "cyan"))"""
                
                
            ### For printing in the end
            model_used_during_ts_run_summary = {} ### To keep track how many households used which model type during this timestep, for printing

            ### Save results for this timestep
            if len(household_saving_dfs) > 0:
                household_results_df = pd.concat(household_saving_dfs, ignore_index=True) ### Combine all parquets, can be done without loop, by chat
                total_capacity_kwh = household_results_df["battery_capacity_kwh"].sum()
                stored_kwh_start = (household_results_df["SOC_start"] * household_results_df["battery_capacity_kwh"]).sum()
                stored_kwh_next = (household_results_df["SOC_next"] * household_results_df["battery_capacity_kwh"]).sum()

                system_soc_df = pd.DataFrame([{
                    "simulation_step": simulation_step,
                    "stored_kwh_start": stored_kwh_start,
                    "stored_kwh_next": stored_kwh_next,
                    "total_capacity_kwh": total_capacity_kwh,
                    "system_soc_start": stored_kwh_start / total_capacity_kwh,
                    "system_soc_next": stored_kwh_next / total_capacity_kwh,
                    "n_households": len(household_results_df)
                }])

                if SYSTEM_SOC_FILE.exists():
                    old_system_soc_df = pd.read_parquet(SYSTEM_SOC_FILE)
                    old_system_soc_df = old_system_soc_df[old_system_soc_df["simulation_step"] < simulation_step] ### Only keep older steps
                    system_soc_df = pd.concat([old_system_soc_df, system_soc_df], ignore_index=True).sort_values("simulation_step")
                #system_soc_df.to_parquet(SYSTEM_SOC_FILE, index=False)
                safe_to_parquet(system_soc_df, SYSTEM_SOC_FILE, index=False)

                """for household, household_df in household_results_df.groupby("household", sort=False): ### Sort falsee by chat
                    household_file = HOUSEHOLD_RESULTS_DIR / f"{int(household)}_results.parquet"
                    if household_file.exists():
                        old_household_df = pd.read_parquet(household_file)
                        old_household_df = old_household_df[old_household_df["simulation_step"] < simulation_step] ### Only keep older steps
                        household_df = pd.concat([old_household_df, household_df], ignore_index=True).sort_values("simulation_step")
                    #household_df.to_parquet(household_file, index=False)
                    safe_to_parquet(household_df, household_file, index=False)"""
                fast_household_file = FAST_HOUSEHOLD_RESULTS_DIR / f"{simulation_step.strftime('%Y%m%d_%H%M_%z')}_households.parquet"
                safe_to_parquet(household_results_df, fast_household_file, index=False)
                    


            if len(market_bid_saving_dfs) > 0:
                market_bid_results_df = pd.concat(market_bid_saving_dfs, ignore_index=True)
                #market_bid_results_df.to_parquet(MARKET_BID_RESULTS_DIR / f"{simulation_step.strftime('%Y%m%d_%H%M')}_market_bids.parquet", index=False)
                safe_to_parquet(market_bid_results_df, MARKET_BID_RESULTS_DIR / f"{simulation_step.strftime('%Y%m%d_%H%M')}_market_bids.parquet", index=False)

            if len(solver_saving_dfs) > 0:
                solver_results_df = pd.concat(solver_saving_dfs, ignore_index=True)
                model_used_during_ts_run_summary = solver_results_df["model_used_during_ts_run"].value_counts().to_dict()

                for household, solver_df_household in solver_results_df.groupby("household", sort=False):
                    solver_file = SOLVER_RESULTS_DIR / f"{int(household)}_solver_info.parquet"
                    pass ### No savinf for now
                    """if solver_file.exists():
                        old_solver_df = pd.read_parquet(solver_file)
                        old_solver_df = old_solver_df[old_solver_df["simulation_step"] < simulation_step]
                        solver_df_household = pd.concat([old_solver_df, solver_df_household], ignore_index=True).sort_values("simulation_step")

                    #solver_df_household.to_parquet(solver_file, index=False)
                    safe_to_parquet(solver_df_household, solver_file, index=False)"""

            if len(sim_battery_dfs) > 0:
                sim_battery_events_df = pd.concat(sim_battery_dfs, ignore_index=True)
                #sim_battery_events_df.to_parquet(SIM_BATTERY_EVENTS_DIR / f"{simulation_step.strftime('%Y%m%d_%H%M')}_sim_battery_events.parquet", index=False)
                safe_to_parquet(sim_battery_events_df, SIM_BATTERY_EVENTS_DIR / f"{simulation_step.strftime('%Y%m%d_%H%M')}_sim_battery_events.parquet", index=False)

            
            if len(balance_delta_saving_dfs) > 0:
                #pass
                balance_delta_results_df = pd.concat(balance_delta_saving_dfs, ignore_index=True)
                #balance_delta_results_df.to_parquet(BALANCE_DELTA_DIR / f"{simulation_step.strftime('%Y%m%d_%H%M')}_balance_delta_results.parquet", index=False)
                safe_to_parquet(balance_delta_results_df, BALANCE_DELTA_DIR / f"{simulation_step.strftime('%Y%m%d_%H%M')}_balance_delta_results.parquet", index=False)


            ### Aggregate household bids into portfolio bid
            if market_name is not None:
                if forecasted_prices_for_bid is None:
                    raise ValueError(f"No forecasted_prices_for_bid returned for market {market_name} at timestep {simulation_step}")
                
                raw_household_bids_by_market[market_name] = household_bids.copy()
                raw_portfolio_bid_by_market[market_name] = aggregate_household_bids(household_bids)

                ### Save raw portfolio bid for this market
                save_portfolio_wide_row(event_timestamp=simulation_step, market_name=market_name, event_type="raw_portfolio_bid", values_series=raw_portfolio_bid_by_market[market_name])
                
                #forecasted_prices_for_bid = extract_market_bid_prices(tree_data=reference_tree_data, market_name=market_name, bid_start=bid_start, bid_end=bid_end)
                
                submitted_limit_prices_by_market[market_name] = simulate_submitted_bid_prices(forecast_price_series=forecasted_prices_for_bid, portfolio_bid_series=raw_portfolio_bid_by_market[market_name], market_details=market_details)
                save_portfolio_wide_row(event_timestamp=simulation_step, market_name=market_name, event_type="submitted_limit_price", values_series=submitted_limit_prices_by_market[market_name])
                
                ### Set household raw commitments for this market
                for household in all_households:
                    household_market_commitments_dict[household].set_raw_commitment(market_name=market_name, commitment_series=household_bids[household])



            ### Update to next step SOC for each household
            for household in all_households: ### Rest is already handled
                household_market_commitments_dict[household].remove_old_commitments(simulation_step=simulation_step) ### Remove commitments that are before the current simulation step
                ### SOC already updated as future as completed
            

            ### Save checkpoint
            if SAVE_RESULTS:
                checkpoint_data = {
                    "next_simulation_step": simulation_step + pd.Timedelta(minutes=15),
                    "soc_by_household": soc_by_household, ### Object already made
                    "household_market_commitments_dict": household_market_commitments_dict,
                    "raw_household_bids_by_market": raw_household_bids_by_market,
                    "rounded_household_bids_by_market": rounded_household_bids_by_market,
                    "raw_portfolio_bid_by_market": raw_portfolio_bid_by_market,
                    "rounded_portfolio_bid_by_market": rounded_portfolio_bid_by_market,
                    "submitted_limit_prices_by_market": submitted_limit_prices_by_market
                }
                #joblib.dump(checkpoint_data, checkpoint_file)
                safe_joblib_dump(checkpoint_data, checkpoint_file)
                if VERBOSITY >= 0:
                    print(colored(f"\n\n\nSaved checkpoint at {simulation_step}.", "green"))


            time_end_step = time.time()
            if VERBOSITY >= -1:
                print(colored(f"\nCompleted simulation step {simulation_step} in {time_end_step - time_start_step:.2f}s. Used models: {model_used_during_ts_run_summary} / {len(all_households)}", "green"))
                print("---------------------------------------------------------------")