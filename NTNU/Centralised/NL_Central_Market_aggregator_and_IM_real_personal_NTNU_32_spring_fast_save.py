import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
import time
from termcolor import colored
import pyomo.environ as pyo
from pyomo.opt import SolverFactory ### Disable if switching to gurobipy
import joblib
from NL_Central_framework_v2_NTNU import CentralFramework
from NL_Central_MPC_NTNU import MPC, BatteryData
import platform ### Easier alternative to sys
import numpy as np
import json


"""
Can be simplified a lot, everything gets solved in one run, no need to run over several households.
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



def extract_next_step_actions(model, tree_data):
    first_stage = min(tree_data["stages"])
    first_local_time = min(tree_data["stage_local_times"][first_stage])
    first_node = tree_data["nodes_per_stage_dict"][first_stage][0]

    actions_by_household = {}

    for household in model.HOUSEHOLDS:
        household = int(household)

        actions_by_household[household] = {
            "E_charge": pyo.value(model.E_charge[household, first_node, first_stage, first_local_time]),
            "E_discharge": pyo.value(model.E_discharge[household, first_node, first_stage, first_local_time]),
            "E_grid_import": pyo.value(model.E_grid_import[household, first_node, first_stage, first_local_time]),
            "E_grid_export": pyo.value(model.E_grid_export[household, first_node, first_stage, first_local_time]),
            "SOC_start": pyo.value(model.SOC[household, first_node, first_stage, first_local_time])
        }

    return actions_by_household



def update_soc(battery_data, next_step_action_dict): ### Like in NTNU script, was also external. NOISE CAN BE ADDED HERE LIKE BEFORE.
    soc_next = next_step_action_dict["SOC_start"] + (battery_data.eta * next_step_action_dict["E_charge"] - (1 / battery_data.eta) * next_step_action_dict["E_discharge"]) / battery_data.Vmax
    return soc_next 

### Can copy whole prepare_balance_delta_df and apply_balance_delta_after_solving from distributed IM system script
### Now impelment the balance delta

def get_balance_delta_for_interval(balance_delta_df, simulation_step):
    interval_end = simulation_step + pd.Timedelta(minutes=15)
    return balance_delta_df[(balance_delta_df["available_at_ts"] >= simulation_step) & (balance_delta_df["available_at_ts"] < interval_end)].sort_values("available_at_ts").copy()


def apply_balance_delta_after_solving(household_actions, battery_data_by_household, simulation_step, fixed_market_commitment, balance_delta_interval_df, real_load_kwh_by_household, real_pv_kwh_by_household): ### No commitment necessary anymore per household
    """
    Will now implement individual strategy too, very similar to decentralised case.
    """
    ### Same as before but now every var is a dict by household
    actions = {household: action.copy() for household, action in household_actions.items()}
    households = sorted(actions)
    n_households = len(households)

    household_position_kwh_by_household = {}
    commitment_kwh_by_household = {}
    base_imbalance_kwh_by_household = {}
    soc_by_household = {}
    soc_without_balance_delta_by_household = {}

    total_charge_kwh_by_household = {household: 0.0 for household in households}
    total_discharge_kwh_by_household = {household: 0.0 for household in households}
    total_correction_kwh_by_household = {household: 0.0 for household in households}

    portfolio_commitment_kwh = float(fixed_market_commitment.get(simulation_step, 0.0))
    planned_position_kwh_by_household = {household: actions[household]["E_grid_export"] - actions[household]["E_grid_import"] for household in households}
    planned_portfolio_position_kwh = sum(planned_position_kwh_by_household.values())

    plan_residual_kwh = portfolio_commitment_kwh - planned_portfolio_position_kwh ### Difference between planned and committed
    plan_residual_kwh_per_household = plan_residual_kwh / n_households

    for household in households: ### Rest stays all same
        action = actions[household]
        battery_data = battery_data_by_household[household]

        #household_position_kwh_by_household[household] = action["E_grid_export"] - action["E_grid_import"] ### keep convention analysis file
        household_position_kwh_by_household[household] = real_pv_kwh_by_household[household] - real_load_kwh_by_household[household] + action["E_discharge"] - action["E_charge"] ### Take the real imbalance
        commitment_kwh_by_household[household] = planned_position_kwh_by_household[household] + plan_residual_kwh_per_household
        base_imbalance_kwh_by_household[household] = household_position_kwh_by_household[household] - commitment_kwh_by_household[household]

        action["E_balance_delta_correction"] = 0.0
        action["imbalance_before_balance_delta"] = base_imbalance_kwh_by_household[household]
        action["imbalance_after_balance_delta"] = base_imbalance_kwh_by_household[household]
        action["SOC_next_without_balance_delta"] = update_soc(battery_data=battery_data, next_step_action_dict=action)

        soc_by_household[household] = action["SOC_start"]
        soc_without_balance_delta_by_household[household] = action["SOC_start"]

    system_base_imbalance_kwh = sum(base_imbalance_kwh_by_household.values()) ### Easier than looping, by chat

    if not USE_BALANCE_DELTA or len(balance_delta_interval_df) == 0:
        soc_next_by_household = {household: update_soc(battery_data=battery_data_by_household[household], next_step_action_dict=actions[household]) for household in households}
        return actions, pd.DataFrame(), soc_next_by_household

    
    balance_delta_interval_df = balance_delta_interval_df.copy()
    #balance_delta_interval_df["duration_in_hours"] = (balance_delta_interval_df["Timeinterval End Loc"] - balance_delta_interval_df["Timeinterval Start Loc"]).dt.total_seconds() / 3600 ### Easier eventually, by chat
    interval_end = simulation_step + pd.Timedelta(minutes=15) ### Fix by chat, crash on transition, have to use actually used time
    next_available = balance_delta_interval_df["available_at_ts"].shift(-1).fillna(interval_end)
    balance_delta_interval_df["duration_in_hours"] = (next_available - balance_delta_interval_df["available_at_ts"]).dt.total_seconds() / 3600
    
    total_duration = balance_delta_interval_df["duration_in_hours"].sum()
    # And implement check of chat too as a safeguard
    if abs(total_duration - 0.25) > 1e-6:
        raise ValueError(f"Total duration of balance delta interval is not 15 minutes. Total duration: {total_duration} hours.")

    detail_rows = []
    previous_measured_system_imbalance_kw = 0.0
    previous_system_correction_kw = 0.0
    previous_measured_imbalance_kw_by_household = {household: 0.0 for household in households} ### Equivalent to decentralised case
    previous_household_correction_kw_by_household = {household: 0.0 for household in households}

    for _, row in balance_delta_interval_df.iterrows(): ### Already sorted
        dt_h = float(row["duration_in_hours"])
        row_fraction = dt_h / total_duration ### Have to do fraction of course because in kwh. Is 1/15 for 1min data, 1/75 for 12s data

        
        base_charge_by_household = {}
        base_discharge_by_household = {}
        base_net_battery_by_household = {}
        base_position_substep_by_household = {}
        base_imbalance_substep_by_household = {}

        max_charge_by_household = {} ### Easier to pull them out
        max_discharge_by_household = {}
        upward_headroom_by_household = {}
        downward_headroom_by_household = {}

        for household in households:
            action = actions[household]
            battery_data = battery_data_by_household[household]

            base_charge_by_household[household] = action["E_charge"] * row_fraction
            base_discharge_by_household[household] = action["E_discharge"] * row_fraction
            base_net_battery_by_household[household] = base_discharge_by_household[household] - base_charge_by_household[household]

            base_position_substep_by_household[household] = household_position_kwh_by_household[household] * row_fraction
            base_imbalance_substep_by_household[household] = base_imbalance_kwh_by_household[household] * row_fraction

            max_discharge_kwh = min(battery_data.dmax * dt_h, max(0.0, (soc_by_household[household] - battery_data.soc_min) * battery_data.Vmax * battery_data.eta))
            max_charge_kwh = min(battery_data.cmax * dt_h, max(0.0, (battery_data.soc_max - soc_by_household[household]) * battery_data.Vmax / battery_data.eta))

            max_discharge_by_household[household] = max_discharge_kwh
            max_charge_by_household[household] = max_charge_kwh

            upward_headroom_by_household[household] = max(0.0, max_discharge_kwh - base_net_battery_by_household[household])
            downward_headroom_by_household[household] = max(0.0, max_charge_kwh + base_net_battery_by_household[household]) ### base net battery is negative for charging

        system_base_imbalance_substep_kwh = sum(base_imbalance_substep_by_household.values())


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

        ### Change from on here, first household level correction
        extreme_price = False

        if not pd.isna(highest_price) and pd.isna(lowest_price):
            extreme_price = up_price_difference >= EXTREME_PRICE_THRESHOLD_EUR_PER_MWH
        elif pd.isna(highest_price) and not pd.isna(lowest_price):
            extreme_price = down_price_difference >= EXTREME_PRICE_THRESHOLD_EUR_PER_MWH
        elif not pd.isna(highest_price) and not pd.isna(lowest_price):
            up_extreme_strength = max(0.0, up_price_difference)
            down_extreme_strength = max(0.0, down_price_difference)
            extreme_price = (up_extreme_strength >= EXTREME_PRICE_THRESHOLD_EUR_PER_MWH or down_extreme_strength >= EXTREME_PRICE_THRESHOLD_EUR_PER_MWH)

        household_correction_kwh_by_household = {}
        net_battery_after_household_correction = {}
        household_direction_by_household = {}

        ### For saving later
        soc_start_substep_by_household = {}
        soc_without_balance_delta_start_by_household = {}

        for household in households:
            battery_data = battery_data_by_household[household]
            soc_start_substep_by_household[household] = soc_by_household[household]
            soc_without_balance_delta_start_by_household[household] = soc_without_balance_delta_by_household[household]

            base_net_battery_kwh = base_net_battery_by_household[household]

            ### At extreme prices, skip household level correction, just use system level correction
            ### Now, don't skip
            #if extreme_price:
            if False: ### Easy
                direction_household = 0
                actual_household_correction_kwh = 0.0
                new_net_battery_kwh = base_net_battery_kwh
            else:
                previous_measured_imbalance_kw = previous_measured_imbalance_kw_by_household[household]
                direction_household = 0
                price_strength_household = 0.0

                ### Upward case
                if not pd.isna(highest_price) and pd.isna(lowest_price):
                    direction_household = 1
                    price_strength_household = up_price_strength

                    #if previous_measured_imbalance_kw >= 0:
                    if not extreme_price and previous_measured_imbalance_kw >= 0:
                        direction_household = 0

                ### Downward case
                elif pd.isna(highest_price) and not pd.isna(lowest_price):
                    direction_household = -1
                    price_strength_household = down_price_strength

                    #if previous_measured_imbalance_kw <= 0:
                    if not extreme_price and previous_measured_imbalance_kw <= 0:
                        direction_household = 0

                ### Both upward and downward prices
                elif not pd.isna(highest_price) and not pd.isna(lowest_price):
                    if extreme_price:
                        if up_extreme_strength > down_extreme_strength:
                            direction_household = 1
                            price_strength_household = up_extreme_strength
                        else:
                            direction_household = -1
                            price_strength_household = down_extreme_strength

                    elif previous_measured_imbalance_kw < 0:
                        direction_household = 1
                        price_strength_household = up_price_strength

                    elif previous_measured_imbalance_kw > 0:
                        direction_household = -1
                        price_strength_household = down_price_strength

                ### Same correction as decentralised now
                wanted_household_correction_kw = previous_household_correction_kw_by_household[household]
                wanted_household_correction_kwh = wanted_household_correction_kw * dt_h

                if direction_household != 0 and price_strength_household > 0:
                    if direction_household == 1:
                        soc_score = (soc_by_household[household] - battery_data.soc_min) / (battery_data.soc_max - battery_data.soc_min)
                        battery_headroom_kwh = max(0.0, max_discharge_by_household[household] - base_net_battery_kwh)
                    else:
                        soc_score = (battery_data.soc_max - soc_by_household[household]) / (battery_data.soc_max - battery_data.soc_min)
                        battery_headroom_kwh = max(0.0, max_charge_by_household[household] + base_net_battery_kwh)

                    soc_score = max(0.0, min(1.0, soc_score))
                    price_score = 1.0 - np.exp(-price_strength_household / PRICE_SCALE_EUR_PER_MWH)
                    response_fraction = price_score * (soc_score ** SOC_SENSITIVITY)

                    #wanted_household_correction_kw = (previous_household_correction_kw_by_household[household] + direction_household * abs(previous_measured_imbalance_kw) * response_fraction)
                    #wanted_household_correction_kwh = wanted_household_correction_kw * dt_h
                    if extreme_price:
                        wanted_household_correction_kwh = direction_household * battery_headroom_kwh * response_fraction
                    else:
                        wanted_household_correction_kw = previous_household_correction_kw_by_household[household] + direction_household * abs(previous_measured_imbalance_kw) * response_fraction
                        wanted_household_correction_kwh = wanted_household_correction_kw * dt_h

                target_net_battery_kwh = base_net_battery_kwh + wanted_household_correction_kwh
                new_net_battery_kwh = max(-max_charge_by_household[household], min(max_discharge_by_household[household], target_net_battery_kwh))
                actual_household_correction_kwh = new_net_battery_kwh - base_net_battery_kwh

            household_correction_kwh_by_household[household] = actual_household_correction_kwh
            net_battery_after_household_correction[household] = new_net_battery_kwh
            household_direction_by_household[household] = direction_household

        ### Now, determine remaining portfolio imbalance
        current_total_household_correction_kw = sum(household_correction_kwh_by_household[household] / dt_h for household in households)

        previous_total_household_correction_kw = sum(previous_household_correction_kw_by_household.values())
        remaining_system_imbalance_kw = previous_measured_system_imbalance_kw + current_total_household_correction_kw - previous_total_household_correction_kw


        ### Now apply the system level correction like before on the rest
        direction = 0
        price_strength = 0.0

        ### Upward case
        if not pd.isna(highest_price) and pd.isna(lowest_price):
            direction = 1
            price_strength = up_price_strength
            if not extreme_price and remaining_system_imbalance_kw >= 0:
                direction = 0

        ### Downward case
        elif pd.isna(highest_price) and not pd.isna(lowest_price):
            direction = -1
            price_strength = down_price_strength
            if not extreme_price and remaining_system_imbalance_kw <= 0:
                direction = 0

        ### Both prices cases
        elif not pd.isna(highest_price) and not pd.isna(lowest_price):
            up_extreme_strength = max(0.0, up_price_difference)
            down_extreme_strength = max(0.0, down_price_difference)

            if extreme_price:
                if up_extreme_strength > down_extreme_strength:
                    direction = 1
                    price_strength = up_extreme_strength
                else:
                    direction = -1
                    price_strength = down_extreme_strength
            elif remaining_system_imbalance_kw < 0:
                direction = 1
                price_strength = up_price_strength
            elif remaining_system_imbalance_kw > 0:
                direction = -1
                price_strength = down_price_strength

        wanted_system_correction_kw = previous_system_correction_kw
        wanted_system_correction_kwh = wanted_system_correction_kw * dt_h

        if direction != 0 and price_strength > 0:
            if direction == 1:
                soc_scores = [(soc_by_household[household] - battery_data_by_household[household].soc_min) / (battery_data_by_household[household].soc_max - battery_data_by_household[household].soc_min) for household in households]
                total_headroom_kwh = sum(max(0.0, max_discharge_by_household[household] - net_battery_after_household_correction[household]) for household in households)
            else:
                soc_scores = [(battery_data_by_household[household].soc_max - soc_by_household[household]) / (battery_data_by_household[household].soc_max - battery_data_by_household[household].soc_min) for household in households]
                total_headroom_kwh = sum(max(0.0, max_charge_by_household[household] + net_battery_after_household_correction[household]) for household in households)

            soc_scores = [max(0.0, min(1.0, score)) for score in soc_scores]
            system_soc_score = float(np.mean(soc_scores))
            price_score = 1.0 - np.exp(-price_strength / PRICE_SCALE_EUR_PER_MWH)
            system_response_fraction = price_score * (system_soc_score ** SOC_SENSITIVITY)

            if extreme_price:
                wanted_system_correction_kwh = direction * total_headroom_kwh * system_response_fraction
            else:
                wanted_system_correction_kw = (previous_system_correction_kw + direction * abs(remaining_system_imbalance_kw) * system_response_fraction)
                wanted_system_correction_kwh = wanted_system_correction_kw * dt_h


        requested_correction_per_household_kwh = wanted_system_correction_kwh / n_households
        ### OVERWRITE NOW, SKIP THE WHOLE PORTFOLIO DISTRIBUTION, JUST APPLY THE HOUSEHOLD CORRECTION UNDER NORMAL PRICING.
        #if not extreme_price:
        requested_correction_per_household_kwh = 0.0
        direction = 0
        actual_system_correction_per_household_kwh = {}
        actual_correction_per_household_kwh = {}

        for household in households:
            battery_data = battery_data_by_household[household]

            personal_net_battery_kwh = net_battery_after_household_correction[household]
            target_final_net_battery_kwh = personal_net_battery_kwh + requested_correction_per_household_kwh
            final_net_battery_kwh = max(-max_charge_by_household[household], min(max_discharge_by_household[household], target_final_net_battery_kwh))

            actual_system_correction_kwh = final_net_battery_kwh - personal_net_battery_kwh
            total_actual_correction_kwh = household_correction_kwh_by_household[household] + actual_system_correction_kwh

            actual_system_correction_per_household_kwh[household] = actual_system_correction_kwh
            actual_correction_per_household_kwh[household] = total_actual_correction_kwh

            new_charge_kwh = max(0.0, -final_net_battery_kwh)
            new_discharge_kwh = max(0.0, final_net_battery_kwh)

            soc_by_household[household] = soc_by_household[household] + (battery_data.eta * new_charge_kwh - (1 / battery_data.eta) * new_discharge_kwh) / battery_data.Vmax

            soc_without_balance_delta_by_household[household] = soc_without_balance_delta_by_household[household] + (battery_data.eta * base_charge_by_household[household] - (1 / battery_data.eta) * base_discharge_by_household[household]) / battery_data.Vmax

            total_charge_kwh_by_household[household] += new_charge_kwh
            total_discharge_kwh_by_household[household] += new_discharge_kwh
            total_correction_kwh_by_household[household] += total_actual_correction_kwh


        ### Update previous measured imbalance and correction for next substep
        system_imbalance_after_balance_delta_kwh = system_base_imbalance_substep_kwh + sum(actual_correction_per_household_kwh.values())
        actual_system_correction_kwh = sum(actual_system_correction_per_household_kwh.values())

        for household in households:
            household_imbalance_after_balance_delta_kwh = base_imbalance_substep_by_household[household] + actual_correction_per_household_kwh[household]
            previous_measured_imbalance_kw_by_household[household] = household_imbalance_after_balance_delta_kwh / dt_h
            previous_household_correction_kw_by_household[household] = household_correction_kwh_by_household[household] / dt_h

        previous_measured_system_imbalance_kw = system_imbalance_after_balance_delta_kwh / dt_h
        previous_system_correction_kw = actual_system_correction_kwh / dt_h
            


        ### Save results per household and per substep
        for household in households:
            actual_position_substep_kwh = base_position_substep_by_household[household] + actual_correction_per_household_kwh[household]
            household_imbalance_after_kwh = base_imbalance_substep_by_household[household] + actual_correction_per_household_kwh[household]

            ### Make distinction now between household and system direction for debug, will probably remove later
            if household_direction_by_household[household] == 1:
                household_selected_direction = "UP"
            elif household_direction_by_household[household] == -1:
                household_selected_direction = "DOWN"
            else:
                household_selected_direction = "NONE"

            if direction == 1:
                system_selected_direction = "UP"
            elif direction == -1:
                system_selected_direction = "DOWN"
            else:
                system_selected_direction = "NONE"

            ### Just put everything in for debug still. WIll probably remove some later
            detail_rows.append({
                "simulation_step": simulation_step,
                "household": household,
                "substep_start": row["Timeinterval Start Loc"],
                "substep_end": row["Timeinterval End Loc"],
                "available_at_ts": row["available_at_ts"],
                "household_selected_direction": household_selected_direction,
                "system_selected_direction": system_selected_direction,
                "SOC_start": soc_start_substep_by_household[household],
                "SOC_end": soc_by_household[household],
                "SOC_without_balance_delta_start": soc_without_balance_delta_start_by_household[household],
                "SOC_without_balance_delta_end": soc_without_balance_delta_by_household[household],
                "household_base_imbalance_kwh": base_imbalance_substep_by_household[household],
                "household_correction_kwh": household_correction_kwh_by_household[household],
                "remaining_system_imbalance_kw": remaining_system_imbalance_kw,
                "requested_system_correction_per_household_kwh": requested_correction_per_household_kwh,
                "actual_system_correction_kwh": actual_system_correction_per_household_kwh[household],
                "total_balance_delta_correction_kwh": actual_correction_per_household_kwh[household],
                "household_imbalance_after_balance_delta_kwh": household_imbalance_after_kwh,
                "system_base_imbalance_kwh": system_base_imbalance_substep_kwh,
                "system_imbalance_after_balance_delta_kwh": system_imbalance_after_balance_delta_kwh,
                "actual_position_substep_kwh": actual_position_substep_kwh
            })
        
    balance_delta_saving_df = pd.DataFrame(detail_rows)
    ### Load into next step actions
    for household in households:
        corrected_household_position_kwh = household_position_kwh_by_household[household] + total_correction_kwh_by_household[household]

        actions[household]["E_charge"] = total_charge_kwh_by_household[household]
        actions[household]["E_discharge"] = total_discharge_kwh_by_household[household]
        actions[household]["E_grid_import"] = max(0.0, -corrected_household_position_kwh)
        actions[household]["E_grid_export"] = max(0.0, corrected_household_position_kwh)
        actions[household]["E_balance_delta_correction"] = total_correction_kwh_by_household[household]
        actions[household]["imbalance_after_balance_delta"] = base_imbalance_kwh_by_household[household] + total_correction_kwh_by_household[household]
        actions[household]["SOC_next_without_balance_delta"] = soc_without_balance_delta_by_household[household]

    soc_next_by_household = {household: soc_by_household[household] for household in households}

    return actions, balance_delta_saving_df, soc_next_by_household



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
    




def simulate_submitted_bid_prices(forecast_price_series, portfolio_bid_series, market_details: MarketDetails):
    ### Bid slightly lower for sell bids, slightly higher for buy bids to increase acceptance probability
    submitted_price_serie = forecast_price_series.copy()

    sell_mask = portfolio_bid_series > 0
    buy_mask = portfolio_bid_series < 0

    submitted_price_serie[sell_mask] = submitted_price_serie[sell_mask] - market_details.bid_price_margin_eur_per_kwh
    submitted_price_serie[buy_mask] = submitted_price_serie[buy_mask] + market_details.bid_price_margin_eur_per_kwh

    return submitted_price_serie.sort_index()





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


TO_AMOUNT = 8


DEVIATION_QUADRATIC_PENALTY = 0.0 ### ALWAYS LEAVE AT ZERO FOR OUR SIMULATION, destroys solving time. Will leave it here if I want to benchmark it later.
CLEAR_SKY_EXPORT_FRACTION = 1.0 ### Leave to one, better. 
### Instead of quadratic, implement two levels
DEVIATION_COST_PENALTY_LOW = 0.15 ### The former dev_pen, see if it keeps increasing
DEVIATION_COST_PENALTY_HIGH = 0.8 ### FOR NOW, FIRST SEE EFFECT OF CLEARSKY #0.2 ### Note that this comes additionally on top of the low penalty
DEVIATION_THRESHOLD_KWH = 0.075 * (4*TO_AMOUNT) * 0.5 ### 0.02kw, adjusted this as forecasting also improves a lot on aggregate level. This should be set to 0.075 but don't know if it would have made a big difference anyway for the central version. SHould also investigate if the fraction improves results.

### Balance delta
USE_BALANCE_DELTA = True
# Params to tune
PRICE_SCALE_EUR_PER_MWH = 50.0 ### Should check if not too agressive, 50 as a start. Note that this is now only for the extreme prices!
SOC_SENSITIVITY = 0.3 ### Higher is more conservative
EXTREME_PRICE_THRESHOLD_EUR_PER_MWH = 150.0 ### 200 is upper 5%, do note that this is the difference with the mid price. Do ±90 for the raw value

### Add some reservation on the battery soc for imbalance
UPWARD_RESERVE_SOC = 0.2 ### self.battery_data.soc_min + upward_reserve_soc, soc_min is currently at 12%
DOWNWARD_RESERVE_SOC = 0.05 ### self.battery_data.soc_max - downward_reserve_soc
WEIGHT_SOC_SOFT = 20.0 ### Weight for the soft constraint



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
WORK_FOLDER = Path("/cluster/work/brechtg")
HOME_FOLDER = Path("/cluster/home/brechtg")

RESULTS_DIR = HOME_FOLDER / "Market_MPC" / f"NL_Results_Central_adjusted_all_equal_MPC_cdeg_{folder_number(BatteryData.c_deg)}" / f"H_{folder_number(4*TO_AMOUNT)}" / "Imb_real_personal_spring_fast_save" / scenario_folder / dev_pen_folder / reserve_soc_folder / balance_delta_folder / RUN_NAME
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WARM_START_DIR = RESULTS_DIR / "warm_starts"
WARM_START_DIR.mkdir(parents=True, exist_ok=True)

### The data files
HOUSEHOLD_RESULTS_DIR = RESULTS_DIR / "household_results"
HOUSEHOLD_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FAST_HOUSEHOLD_SAVE_DIR = RESULTS_DIR / "fast_save"
FAST_HOUSEHOLD_SAVE_DIR.mkdir(parents=True, exist_ok=True)

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
BALANCE_DELTA_FILE = WORK_FOLDER / "Market_MPC" / "Markets" / "Markets Data_NL" / "Balancing_delta" / "Parquets" / "overall_df.parquet"
BALANCE_DELTA_DELAY = pd.Timedelta(minutes=2)

### Real load and PV
REAL_LOAD_FILE = WORK_FOLDER / "thesis_load_training" / "Fluvius 2400" / "Prepped" / "All_prepared_2_years_with_aggregate.parquet"
REAL_PV_FOLDER = WORK_FOLDER / "Fluvius PV" / "Open-meteo Satellite" / "Satellite_Power_Virtual_Houses"

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
    FAST_HOUSEHOLD_SAVE_DIR.mkdir(parents=True, exist_ok=True)


### Some solver settings, set inside solve for one household now
GUROBI_THREADS = 56 ### Cleanup will be done afterwards
TIME_LIMIT_SECONDS = 1000 ### Just in case, to get the buggy runs out for running hours and barely improving
MIPGAP = 0.025 #### SET NONE FOR DEFAULT


### Same as before but for households loop too
def check_simultaneous_battery_and_grid(model, tree_data, eps_bat=1e-4, eps_grid=1e-2):
    rows = []

    for household, name, stage, local_time in model.HNST:
        household = int(household)
        stage = int(stage)
        local_time = int(local_time)

        E_charge = pyo.value(model.E_charge[household, name, stage, local_time])
        E_discharge = pyo.value(model.E_discharge[household, name, stage, local_time])
        E_grid_import = pyo.value(model.E_grid_import[household, name, stage, local_time])
        E_grid_export = pyo.value(model.E_grid_export[household, name, stage, local_time])

        timestamp = tree_data["local_time_to_timestamps"][stage][local_time]

        if E_charge > eps_bat and E_discharge > eps_bat:
            rows.append({
                "violation_type": "battery_simultaneous",
                "household": household,
                "node": name,
                "stage": stage,
                "local_time": local_time,
                "timestamp": timestamp,
                "E_charge": E_charge,
                "E_discharge": E_discharge,
                "E_grid_import": E_grid_import,
                "E_grid_export": E_grid_export,
                "simultaneous_amount": min(E_charge, E_discharge)
            })

        if E_grid_import > eps_grid and E_grid_export > eps_grid:
            rows.append({
                "violation_type": "grid_simultaneous",
                "household": household,
                "node": name,
                "stage": stage,
                "local_time": local_time,
                "timestamp": timestamp,
                "E_charge": E_charge,
                "E_discharge": E_discharge,
                "E_grid_import": E_grid_import,
                "E_grid_export": E_grid_export,
                "simultaneous_amount": min(E_grid_import, E_grid_export)
            })

    return pd.DataFrame(rows)



def solve_central_for_timestep(central_framework, central_mpc, all_households, simulation_step, horizon_end, market_name, bid_start, bid_end, fixed_market_commitment, soc_by_household, balance_delta_interval_df, real_load_kwh_by_household, real_pv_kwh_by_household): ### For now just pass everything needed, should probably pack this like args worker but good for now
    ### Follow solve_one_household_for_timestep function but then for central
    for household in all_households:
        central_framework.battery_data_by_household[household].soc_init_mpc_run = soc_by_household[household] ### Update the initial soc for the household before running the MPC

    nodes = central_framework.build_scenario_tree(
        simulation_step=simulation_step,
        horizon_end=horizon_end,
        branch_hours_utc=[0, 6, 12, 18],
        scenario_counts=SCENARIO_COUNTS,
        include_deterministic_load=True,
        include_deterministic_pv=True,
        extend_last_nonzero_branch_to_horizon_end=True
    )

    tree_data = central_framework.nodes_to_pyomo_per_stage(nodes)
    tree_data = central_framework.add_market_prices_to_tree(tree_data=tree_data, simulation_step=simulation_step)


    linearise_grid_constraint = True
    linearise_battery_constraint = True

    solver_rows = []
    simultaneous_dfs = []
    ### Make it loop until no simultaneous charge/discharge is found
    for solve_attempt in range(1, 4): ### Three cases: (lin,lin), (one not-lin), (not-lin,not-lin)
        if linearise_grid_constraint and linearise_battery_constraint:
            current_model_type = "Central_both_linearised"
        elif not linearise_grid_constraint and linearise_battery_constraint:
            current_model_type = "Central_grid_not_linearised"
        elif linearise_grid_constraint and not linearise_battery_constraint:
            current_model_type = "Central_battery_not_linearised"
        else:
            current_model_type = "Central_full_MILP"
        

        ### For now, just run the full model without linearising. Should check how to do this still.
        time_start = time.time()
        model = central_mpc.Build_MPC_model(
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
            linearise_grid_constraint=linearise_grid_constraint,
            linearise_battery_constraint=linearise_battery_constraint,
            battery_binary_timestamps=None
        )
        print(f"Model built in {time.time() - time_start:.2f} seconds")
        solver = pyo.SolverFactory("gurobi")
        ### Is one problem so always use max threads
        solver.options["Threads"] = GUROBI_THREADS
        solver.options["TimeLimit"] = TIME_LIMIT_SECONDS
        if MIPGAP is not None: ### This is only relevant for MILP, but doesn't hurt for LP so can just leave always
            solver.options["MIPGap"] = MIPGAP

        solve_start_time = time.time()
        ### if simulation step is on a market event, set tee=True, just to get a bit better idea of how optimisation is going.
        if market_name is not None:
            solver.options["DisplayInterval"] = 30 ### Reduced printing in terminal, print every 30 seconds
            result = solver.solve(model, tee=True, load_solutions=True)
        else:
            result = solver.solve(model, tee=False, load_solutions=True)
        solve_end_time = time.time()
        print(f"Solver finished in {solve_end_time - solve_start_time:.2f} seconds")

        termination_condition = str(result.solver.termination_condition)
        if termination_condition not in ["optimal", "feasible", "maxTimeLimit"]:
            raise ValueError(f"Solver did not find an optimal or feasible solution. Termination condition: {termination_condition}")

        ### Check simultaneous
        simultaneous_df = check_simultaneous_battery_and_grid(model=model, tree_data=tree_data, eps_bat=1e-4, eps_grid=1e-2)
        bat_violation_detected = (len(simultaneous_df) > 0) and (simultaneous_df["violation_type"] == "battery_simultaneous").any()
        grid_violation_detected = (len(simultaneous_df) > 0) and (simultaneous_df["violation_type"] == "grid_simultaneous").any()

        solver_rows.append({
            "simulation_step": simulation_step,
            "solve_attempt": solve_attempt,
            "model_type": current_model_type,
            "linearise_battery_constraint": linearise_battery_constraint,
            "linearise_grid_constraint": linearise_grid_constraint,
            "battery_violation_detected": bat_violation_detected,
            "grid_violation_detected": grid_violation_detected,
            "termination_condition": termination_condition,
            "solve_time_seconds": solve_end_time - solve_start_time,
            "n_households": len(all_households)
        })

        if len(simultaneous_df) > 0:
            simultaneous_df = simultaneous_df.copy()
            simultaneous_df["simulation_step"] = simulation_step
            simultaneous_df["solve_attempt"] = solve_attempt
            simultaneous_df["model_type"] = current_model_type
            simultaneous_dfs.append(simultaneous_df)


        if not bat_violation_detected and not grid_violation_detected:
            print(colored(f"\nNo simultaneous charge/discharge violations detected at {simulation_step}: {current_model_type}.", "green"))
            break ### No violations, exit the loop
        else: 
            print(colored(f"Simultaneous detected. Battery: {bat_violation_detected}, Grid: {grid_violation_detected}", "red"))
        
        old_linearise_battery_constraint = linearise_battery_constraint
        old_linearise_grid_constraint = linearise_grid_constraint

        if grid_violation_detected:
            linearise_grid_constraint = False
        if bat_violation_detected:
            linearise_battery_constraint = False
        ### Extra check by chat, should be impossible though but well
        if (old_linearise_grid_constraint == linearise_grid_constraint and old_linearise_battery_constraint == linearise_battery_constraint):
            raise ValueError(f"Simultaneous operation remained at {simulation_step}, even though the required binary constraints were enabled.")
        
        if VERBOSITY >= 0:
            print(colored(f"Rerunning central MPC at {simulation_step} due to Battery violation: {bat_violation_detected}, Grid violation: {grid_violation_detected}.", "yellow"))


    final_model_type = current_model_type
    actions_by_household = extract_next_step_actions(model=model, tree_data=tree_data)
    ### APPLY BALANCE DELTA
    ### Pass real data now
    actions_by_household, balance_delta_results_df, updated_soc_by_household = apply_balance_delta_after_solving(household_actions=actions_by_household, battery_data_by_household=central_framework.battery_data_by_household, simulation_step=simulation_step, fixed_market_commitment=fixed_market_commitment, balance_delta_interval_df=balance_delta_interval_df, real_load_kwh_by_household=real_load_kwh_by_household, real_pv_kwh_by_household=real_pv_kwh_by_household)
    household_rows = []

    for household in all_households:
        action = actions_by_household[household]
        battery_data = central_framework.battery_data_by_household[household]
        #soc_next = update_soc(battery_data=battery_data, next_step_action_dict=action) ### Disable, happens through the balance delta function now
        soc_next = updated_soc_by_household[household]  ### Already updated in the balance delta function, so just use that value. Note that this is the value after the balance delta
        #updated_soc_by_household[household] = soc_next

        household_rows.append({
            "simulation_step": simulation_step,
            "household": household,
            "SOC_start": action["SOC_start"],
            "E_charge": action["E_charge"],
            "E_discharge": action["E_discharge"],
            "E_grid_import": action["E_grid_import"],
            "E_grid_export": action["E_grid_export"],
            "SOC_next": soc_next,
            "battery_capacity_kwh": battery_data.Vmax,
            "model_type": final_model_type,
            "E_balance_delta_correction": action["E_balance_delta_correction"],
            "imbalance_before_balance_delta": action["imbalance_before_balance_delta"],
            "imbalance_after_balance_delta": action["imbalance_after_balance_delta"],
            "SOC_next_without_balance_delta": action["SOC_next_without_balance_delta"]
        })

    household_results_df = pd.DataFrame(household_rows)
    solver_results_df = pd.DataFrame(solver_rows)

    if len(simultaneous_dfs) > 0:
        simultaneous_events_df = pd.concat(simultaneous_dfs, ignore_index=True)
    else:
        simultaneous_events_df = pd.DataFrame()

    portfolio_bid = None
    forecasted_prices_for_bid = None
    if market_name is not None:
        portfolio_bid = extract_market_bid(model=model, tree_data=tree_data)
        forecasted_prices_for_bid = extract_market_bid_prices(tree_data=tree_data, market_name=market_name, bid_start=bid_start, bid_end=bid_end)

    print(f"End of function in {time.time() - solve_end_time:.2f} seconds")
    return {
        "soc_by_household": updated_soc_by_household,
        "household_results_df": household_results_df,
        "solver_results_df": solver_results_df,
        "simultaneous_events_df": simultaneous_events_df,
        "portfolio_bid": portfolio_bid,
        "forecasted_prices_for_bid": forecasted_prices_for_bid,
        "termination_condition": termination_condition,
        "final_model_type": final_model_type,
        "balance_delta_results_df": balance_delta_results_df
    }







if __name__ == "__main__":
    ensure_result_dirs_exist()
    if USE_BALANCE_DELTA:
        balance_delta_df = prepare_balance_delta_df(pd.read_parquet(BALANCE_DELTA_FILE))
    else:
        balance_delta_df = pd.DataFrame()
    from_amount = 0 ### HAVE TO LEAVE THIS TO 0
    to_amount = TO_AMOUNT ### Total gives four times the difference between these two

    households_load = [i for i in range(301, 601)] ### 301-600 are load households
    households_HP = [i for i in range(901, 1201)] ### 901-1200 are heat pump households
    households_EV = [i for i in range(1501, 1801)] ### 1501-1800 are EV households
    households_HP_EV = [i for i in range(2101, 2401)] ### 2101-2400 are HP+EV households

    all_households = sorted(households_load[from_amount:to_amount] + households_HP[from_amount:to_amount] + households_EV[from_amount:to_amount] + households_HP_EV[from_amount:to_amount])

    ### OVERWRITE
    #all_households = [301, 302]

    ### Read realised load
    real_load_df = pd.read_parquet(REAL_LOAD_FILE)
    real_load_df = real_load_df[all_households]

    ### Read realised PV
    real_pv_by_household = {}

    for household in all_households:
        pv_file = REAL_PV_FOLDER / f"SatellitePower_load_id_{household}.parquet"
        pv_power_kw = pd.read_parquet(pv_file, columns=["satellite_fitting_power_kw"])["satellite_fitting_power_kw"]
        real_pv_by_household[household] = pv_power_kw * 0.25



    central_framework = CentralFramework(households=all_households, to_amount=to_amount, exclude_before_ts="2024-01-22 00:00", perfect_DA_forecast=False)
    effective_c_deg_values = {household: central_framework.battery_data_by_household[household].c_deg for household in all_households} ### Have to get the actual c_deg used, comes from the distributed MPC.py
    unique_c_deg_values = sorted(set(effective_c_deg_values.values()))
    if len(unique_c_deg_values) != 1:
        raise ValueError(f"Households have different c_deg values: {unique_c_deg_values}")
    print(colored(f"Effective c_deg used by all household batteries: {unique_c_deg_values[0]} €/kWh throughput", "red"))
    central_mpc = MPC(battery_data_by_household=central_framework.battery_data_by_household, contract_data=central_framework.contract_data, pv_system_power_kw_by_household=central_framework.pv_system_power_kw_by_household)



    ### Initial setup for multiprocessing
    portfolio_market_commitments = MarketAggregate() ### For portfolio level, not per household
    soc_by_household = {household: 0.5 for household in all_households}

    print(colored(f"Initialised parent framework for cleared prices with household {all_households[0]}.", "green"))


    raw_portfolio_bid_by_market = {}
    rounded_portfolio_bid_by_market = {}
    submitted_limit_prices_by_market = {}


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
    exclude_end_ts = pd.Timestamp("2026-01-22 00:30:00", tz="Europe/Brussels")
    exclude_mask = (SIMULATION_STEPS >= exclude_start_ts) & (SIMULATION_STEPS < exclude_end_ts)
    SIMULATION_STEPS = SIMULATION_STEPS[~exclude_mask] ### Exclude the period where we don't have load scenarios
    ### Continuation logic
    checkpoint_file = RESULTS_DIR / "checkpoint.joblib" ### Per timestep
    if RESUME_FROM_CHECKPOINT and checkpoint_file.exists():
        checkpoint_joblib = joblib.load(checkpoint_file)

        soc_by_household = checkpoint_joblib["soc_by_household"] ### Just load in once for all households. Has to stay this way

        portfolio_market_commitments = checkpoint_joblib["portfolio_market_commitments"]
        raw_portfolio_bid_by_market = checkpoint_joblib["raw_portfolio_bid_by_market"]
        rounded_portfolio_bid_by_market = checkpoint_joblib["rounded_portfolio_bid_by_market"]
        submitted_limit_prices_by_market = checkpoint_joblib["submitted_limit_prices_by_market"]


        SIMULATION_STEPS = SIMULATION_STEPS[SIMULATION_STEPS >= checkpoint_joblib["next_simulation_step"]]

        print(colored(f"Resuming simulation from checkpoint at {checkpoint_joblib['next_simulation_step']}. Remaining steps: {len(SIMULATION_STEPS)}", "green"))




    for simulation_step in SIMULATION_STEPS:
        print(colored(f"\nStarting simulation step {simulation_step}", "green"))

        time_start_step = time.time()

        if simulation_step == exclude_end_ts:
            portfolio_market_commitments = MarketAggregate() ### For portfolio level, not per household
            soc_by_household = {household: 0.5 for household in all_households}
            raw_portfolio_bid_by_market.clear()
            rounded_portfolio_bid_by_market.clear()
            submitted_limit_prices_by_market.clear()

            print(colored(f"Resetting portfolio market commitments and SOCs at {simulation_step} after excluded period.", "yellow"))


        #######################
        ##### CLEARING TS #####
        #######################
        ### First check if any market gets cleared at this ts
        clearing_market_name = get_clearing_instances_for_timestep(simulation_step) ### Returns DA, IDA1, IDA2, IDA3 or None if timestamp falls on 13:00, 15:30, 22:30 or 10:30. This is hardcoded for now according to TS. If every lowered in period, should put this to the next timestep after market publicatiuon.

        if clearing_market_name is not None: ### It is a bidding timestep for a market
            if clearing_market_name in rounded_portfolio_bid_by_market:
                published_prices = get_cleared_market_prices(framework=central_framework, market_name=clearing_market_name, timestamps=rounded_portfolio_bid_by_market[clearing_market_name].index)
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

                    portfolio_market_commitments.set_cleared_commitment(market_name=clearing_market_name, commitment_series=cleared_portfolio_bid)

                    print(colored(f"{clearing_market_name} skipped at {simulation_step} due to unsuccessful market. All bids rejected.", "red"))
                

                else:
                    accepted_bids_mask = get_accepted_bids_mask(rounded_portfolio_bid=rounded_portfolio_bid_by_market[clearing_market_name], submitted_limit_prices=submitted_limit_prices_by_market[clearing_market_name], cleared_prices=published_prices)
                    cleared_portfolio_bid = rounded_portfolio_bid_by_market[clearing_market_name].where(accepted_bids_mask, 0.0).sort_index()
                    market_successful_mask = pd.Series(1.0, index=rounded_portfolio_bid_by_market[clearing_market_name].index)

                    save_portfolio_wide_row(event_timestamp=simulation_step, market_name=clearing_market_name, event_type="cleared_portfolio_bid", values_series=cleared_portfolio_bid)
                    save_portfolio_wide_row(event_timestamp=simulation_step, market_name=clearing_market_name, event_type="accepted_bids_mask", values_series=accepted_bids_mask.astype(float), fill_value=float("nan"))
                    save_portfolio_wide_row(event_timestamp=simulation_step, market_name=clearing_market_name, event_type="market_successful_mask", values_series=market_successful_mask, fill_value=float("nan"))

                    portfolio_market_commitments.set_cleared_commitment(market_name=clearing_market_name, commitment_series=cleared_portfolio_bid)
                    
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
                portfolio_raw_bid = raw_portfolio_bid_by_market[rounding_market_name]

                ### Small check from chat to ensure we are bidding in multiples of the market step size
                bid_steps = portfolio_raw_bid / market_details.market_step_kwh
                if not np.allclose(bid_steps.to_numpy(), np.round(bid_steps.to_numpy()), atol=1e-4):
                    raise ValueError(f"The central {rounding_market_name} bid does not consist of {market_details.market_step_kwh} kWh steps.")
                portfolio_raw_bid = np.round(bid_steps) * market_details.market_step_kwh ### There can be ever so small numerical errors and the above raised once with eps 1e-6.

                rounded_portfolio_bid_by_market[rounding_market_name] = portfolio_raw_bid.copy()
                portfolio_market_commitments.set_rounded_commitment(market_name=rounding_market_name, commitment_series=rounded_portfolio_bid_by_market[rounding_market_name])

                save_portfolio_wide_row(event_timestamp=simulation_step, market_name=rounding_market_name, event_type="rounded_portfolio_bid", values_series=rounded_portfolio_bid_by_market[rounding_market_name])

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
        if market_name is not None and exclude_start_ts <= bid_start < exclude_end_ts:
            raw_portfolio_bid_by_market.pop(market_name, None)
            rounded_portfolio_bid_by_market.pop(market_name, None)
            submitted_limit_prices_by_market.pop(market_name, None)

            market_name = None
            bid_start = None
            bid_end = None

        horizon_end = simulation_step + pd.Timedelta(hours=HORIZON_HOURS) ### 48h horizon strict!

        fixed_market_commitment = portfolio_market_commitments.total_commitment()

        if USE_BALANCE_DELTA:
            balance_delta_interval_df = get_balance_delta_for_interval(balance_delta_df=balance_delta_df, simulation_step=simulation_step)
        else:
            balance_delta_interval_df = pd.DataFrame()

        ### Prepare real dtaa fro passing
        real_load_timestamp = simulation_step.replace(year=2024)
        real_load_kwh_by_household = {household: float(real_load_df.at[real_load_timestamp, household]) for household in all_households}
        real_pv_kwh_by_household = {household: float(real_pv_by_household[household].at[simulation_step]) for household in all_households}

        central_solve_results = solve_central_for_timestep(
            central_framework=central_framework,
            central_mpc=central_mpc,
            all_households=all_households,
            simulation_step=simulation_step,
            horizon_end=horizon_end,
            market_name=market_name,
            bid_start=bid_start,
            bid_end=bid_end,
            fixed_market_commitment=fixed_market_commitment,
            soc_by_household=soc_by_household,
            balance_delta_interval_df=balance_delta_interval_df,
            real_load_kwh_by_household=real_load_kwh_by_household,
            real_pv_kwh_by_household=real_pv_kwh_by_household
        )

        soc_by_household = central_solve_results["soc_by_household"]
        household_results_df = central_solve_results["household_results_df"]
        solver_results_df = central_solve_results["solver_results_df"]
        termination_condition = central_solve_results["termination_condition"]
        simultaneous_events_df = central_solve_results["simultaneous_events_df"]
        balance_delta_results_df = central_solve_results["balance_delta_results_df"]

        if market_name is not None: ### There was a bid
            raw_portfolio_bid_by_market[market_name] = central_solve_results["portfolio_bid"]
            forecasted_prices_for_bid = central_solve_results["forecasted_prices_for_bid"]

            save_portfolio_wide_row(event_timestamp=simulation_step, market_name=market_name, event_type="raw_portfolio_bid", values_series=raw_portfolio_bid_by_market[market_name])

            submitted_limit_prices_by_market[market_name] = simulate_submitted_bid_prices(forecast_price_series=forecasted_prices_for_bid, portfolio_bid_series=raw_portfolio_bid_by_market[market_name], market_details=market_details)

            save_portfolio_wide_row(event_timestamp=simulation_step, market_name=market_name, event_type="submitted_limit_price", values_series=submitted_limit_prices_by_market[market_name])

            portfolio_market_commitments.set_raw_commitment(market_name=market_name, commitment_series=raw_portfolio_bid_by_market[market_name])



        ### Make whole new saving part
        total_capacity_kwh = household_results_df["battery_capacity_kwh"].sum() ### Can be copied exactly
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
        fast_household_file = FAST_HOUSEHOLD_SAVE_DIR / f"{simulation_step.strftime('%Y%m%d_%H%M_%z')}_households.parquet"
        safe_to_parquet(household_results_df, fast_household_file, index=False)


        solver_file = SOLVER_RESULTS_DIR / "central_solver_info.parquet"
        if solver_file.exists():
            old_solver_results_df = pd.read_parquet(solver_file)
            old_solver_results_df = old_solver_results_df[old_solver_results_df["simulation_step"] < simulation_step]
            solver_results_df = pd.concat([old_solver_results_df, solver_results_df], ignore_index=True).sort_values("simulation_step")
        safe_to_parquet(solver_results_df, solver_file, index=False)

        if len(simultaneous_events_df) > 0:
            simultaneous_file = SIM_BATTERY_EVENTS_DIR / f"{simulation_step.strftime('%Y%m%d_%H%M')}_simultaneous_events.parquet"
            safe_to_parquet(simultaneous_events_df, simultaneous_file, index=False)

        if len(balance_delta_results_df) > 0:
            pass ### For now disabled
            #safe_to_parquet(balance_delta_results_df, BALANCE_DELTA_DIR / f"{simulation_step.strftime('%Y%m%d_%H%M')}_balance_delta_results.parquet", index=False)


        portfolio_market_commitments.remove_old_commitments(simulation_step=simulation_step)

        if SAVE_RESULTS:
            checkpoint_data = {
                "next_simulation_step": simulation_step + pd.Timedelta(minutes=15),
                "soc_by_household": soc_by_household,
                "portfolio_market_commitments": portfolio_market_commitments,
                "raw_portfolio_bid_by_market": raw_portfolio_bid_by_market,
                "rounded_portfolio_bid_by_market": rounded_portfolio_bid_by_market,
                "submitted_limit_prices_by_market": submitted_limit_prices_by_market
            }
            safe_joblib_dump(checkpoint_data, checkpoint_file)

            if VERBOSITY >= 0:
                print(colored(f"Saved checkpoint at {simulation_step}.", "green"))

        time_end_step = time.time()

        if VERBOSITY >= -1:
            print(colored(f"\nCompleted central simulation step {simulation_step} in {time_end_step - time_start_step:.2f}s. Solver: {termination_condition}. Households: {len(all_households)}.", "green"))
            print("---------------------------------------------------------------")