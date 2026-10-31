import pandas as pd
from pathlib import Path
from dataclasses import dataclass
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import os
from termcolor import colored
import random ### If want to keep randomness for SOC
import math


"""
Note:
    MPC will be formulated as if calculation time was 0. E.g. at 12h, you calculate the whole MPC, then you immediately apply it for interval 12:00-12:15, at 12:15 you do the same.
    SOC means the SOC at the beginning of the interval, so at 12:00, SOC is the SOC at 12:00
"""

@dataclass
class BatteryData:
    Vmax: float ### kwh
    soc_init_mpc_run: float = 0.5 ### Initial SOC for the MPC run, has to be updated each time

    eta: float = 0.95 ### Efficiency --> 0.95*0.95 = 0.9025 round trip efficiency
    soc_min: float = 0.12 ### LOWERED for imbalance reserve, was 0.2
    soc_max: float = 0.95
    c_deg: float = 0.02 #1.1e-3 ### Was 1e-3. This whole implementation should be replaced by degradation model

    def __post_init__(self): ### Not necessary at all but wanted to try xd
        ### This assumes 0.7C. This can be done better but good for now
        self.cmax = 0.7 * self.Vmax ### kW, max charge power
        self.dmax = 0.7 * self.Vmax ### kW, max discharge power


@dataclass
class ContractData:
    inkoopvergoeding: float = 0.02 ### Zonneplan, incl VAT, in €/kwh
    verkoopvergoeding: float = 0.02 #-0.02 ### Zonneplan still has this as positive btu is the only one, incl VAT, in €/kwh
    import_belasting: float = 0.111 ### Incl VAT, in €/kwh
    jaarlijkse_belastingsvermindering: float = 629.0 ### Per year, for 2026. Incl. VAT

    SALDEREN: bool = False ### I'll put it in here for now



class MPC:
    def __init__(self, battery_data_by_household: dict, contract_data: ContractData, pv_system_power_kw_by_household: dict, period: str = "15min"):
        self.households = sorted(battery_data_by_household.keys()) ### No need to pass them separately
        self.battery_data_by_household = battery_data_by_household ### dict full of BatteryData objects
        self.contract_data = contract_data ### Same contract for all households.
        self.pv_system_power_kw_by_household = pv_system_power_kw_by_household ### Should be made per household now of course
        self.dt = pd.Timedelta(period).total_seconds() / 3600.0 ### 0.25


    def Build_MPC_model(self, tree_data, market_name=None, bid_start=None, bid_end=None, fixed_market_commitment=None, deviation_cost_penalty_low=0.1, deviation_cost_penalty_high=0.2, deviation_threshold_kwh=0.25, deviation_quadratic_penalty=0.04, weight_soc_soft=1.0, clear_sky_export_fraction=1.0, upward_reserve_soc=0.15, downward_reserve_soc=0.10, linearise_grid_constraint=False, linearise_battery_constraint=False, battery_binary_timestamps=None): ### Period is str "15min". In update, market strategy is per build made and market dependent. HAVE TO TUNE AND SET PENALTY TERM!!!
        m = pyo.ConcreteModel()
        ### Multidemensional index now: (name, stage, local_time)
        m.NST = pyo.Set(initialize=tree_data["NST"], dimen=3) ### All integrated into one now instead of separate sets for each stage. Should be the only index needed now
        ### Look back at NST construction, is also just list of tuples. Can add households without changeing the original framework file
        ### Expand to being household dependent
        m.HOUSEHOLDS = pyo.Set(initialize=self.households)
        HNST = [(household, name, stage, local_time) for household in self.households for name, stage, local_time in tree_data["NST"]]
        m.HNST = pyo.Set(initialize=HNST, dimen=4)
        ### Extra sets for aid
        m.LEAF_NODES = pyo.Set(initialize=tree_data["leaf_nodes"]) ### For SOC terminal constraint. Do this for proper value setting. Still have to implement proper SOC terminal value for day to day but for now is enough just be setting soc + action to be between boundaries of soc

        if fixed_market_commitment is None: 
            fixed_market_commitment = pd.Series(dtype=float)
        market_bid_steps = []
        if market_name is not None:

            for bid_step, timestamp in tree_data["bid_step_to_timestamp"].items(): ### bid step is integer, like local_time, but for all timestamps
                if bid_start <= timestamp < bid_end:
                    price_value = tree_data["market_prices_by_timestamp"][market_name].get(timestamp, float("nan")) ### Update: Check if price is available regarding market cancellation
                    if not pd.isna(price_value):
                        market_bid_steps.append(bid_step)
        market_bid_steps_set = set(market_bid_steps) ### we now have an exclusive list of the bid steps for the market
        m.MARKET_BID_STEPS = pyo.Set(initialize=market_bid_steps) ### use market_bids_steps, not the set! Warning by pyomo

        market_bid_timestamps = {tree_data["bid_step_to_timestamp"][bid_step] for bid_step in market_bid_steps} ### Market bid steps is already a set. The current market bid timestamps, separation for IDA3, by chat
        fixed_commitment_timestamps = set(fixed_market_commitment.index) ### timestamps of previously commited market positions, cleared after each market day, still have to implement this.

        commitment_timestamps = market_bid_timestamps.union(fixed_commitment_timestamps)
        commitment_NST = [(name, stage, local_time) for name, stage, local_time in tree_data["NST"] if tree_data["local_time_to_timestamps"][stage][local_time] in commitment_timestamps] ### Removes whole bunch of unnecessary vars
        m.COMMITMENT_NST = pyo.Set(initialize=commitment_NST, dimen=3)

        ### Make MILP with binaries only when necessary and only at those timestamps
        if battery_binary_timestamps is None:
            battery_binary_timestamps = []
        battery_binary_timsetamp_dict = {pd.Timestamp(timestamp): True for timestamp in battery_binary_timestamps} ### For SET init, needs a dict (not strictly but from past experience, the way to go)

        if linearise_battery_constraint: ### By chat, only apply if you don't want all as MILP of course
            battery_binary_NST = [(name, stage, local_time) for name, stage, local_time in tree_data["NST"] if tree_data["local_time_to_timestamps"][stage][local_time] in battery_binary_timsetamp_dict]
        else:
            battery_binary_NST = list(tree_data["NST"]) ### All timestamps and its Nodes

        ### The partial milp was never used but will leave the logic, expand in same way as before
        battery_binary_HNST = [(household, name, stage, local_time) for household in self.households for name, stage, local_time in battery_binary_NST]
        m.BATTERY_BINARY_HNST = pyo.Set(initialize=battery_binary_HNST, dimen=4)
        


        ### Parameters
        m.Vmax = pyo.Param(m.HOUSEHOLDS, initialize={household: self.battery_data_by_household[household].Vmax for household in self.households}) ### In kwh, have to manually rebuild a dictionary. Dict has batterydata objects inside of it
        m.eta = pyo.Param(m.HOUSEHOLDS, initialize={household: self.battery_data_by_household[household].eta for household in self.households}) ### Efficiency, same for all households
        m.cmax = pyo.Param(m.HOUSEHOLDS, initialize={household: self.battery_data_by_household[household].cmax  * self.dt for household in self.households}) ### In kwh
        m.dmax = pyo.Param(m.HOUSEHOLDS, initialize={household: self.battery_data_by_household[household].dmax * self.dt for household in self.households}) ### In kwh
        m.soc_min = pyo.Param(m.HOUSEHOLDS, initialize={household: self.battery_data_by_household[household].soc_min for household in self.households})
        m.soc_max = pyo.Param(m.HOUSEHOLDS, initialize={household: self.battery_data_by_household[household].soc_max for household in self.households})
        market_soc_min_dict = {}
        market_soc_max_dict = {}
        for household in m.HOUSEHOLDS:
            market_soc_min = self.battery_data_by_household[household].soc_min + upward_reserve_soc
            market_soc_max = self.battery_data_by_household[household].soc_max - downward_reserve_soc

            if market_soc_max - market_soc_min < 0.2:
                raise ValueError(f"Market SOC range for household {household} is too small. Please adjust upward_reserve_soc and downward_reserve_soc.")
            market_soc_min_dict[household] = market_soc_min
            market_soc_max_dict[household] = market_soc_max
        m.market_soc_min = pyo.Param(m.HOUSEHOLDS, initialize=market_soc_min_dict)
        m.market_soc_max = pyo.Param(m.HOUSEHOLDS, initialize=market_soc_max_dict)

        m.soc_init = pyo.Param(m.HOUSEHOLDS, initialize={household: self.battery_data_by_household[household].soc_init_mpc_run for household in self.households}) ### HAS TO BE CHANGED EACH TIME
        m.c_deg = pyo.Param(m.HOUSEHOLDS, initialize={household: self.battery_data_by_household[household].c_deg for household in self.households}) ### SHould be adjusted now according to KWH vs KW!!! Don't do * self.dt, in obj function, E_charge and E_discharge are already in kwh
        m.max_grid_import = pyo.Param(initialize=34.5 * self.dt) ### A standard 3x25A connection. Same for each household so can be kept the same
        m.max_grid_export = pyo.Param(initialize=34.5 * self.dt)
        #maxmimum_market_export_kw = min(17.25, self.pv_system_power_kw + self.battery_data.dmax) ### Have to limit export better. Unrealistic bidding. Still not good. Will use the clear sky approach.
        #m.max_market_export = pyo.Param(initialize=maxmimum_market_export_kw * self.dt)

        ### Clear sky pv + battery export limit approach, has to be summed too over hosueholds
        portfolio_pv_power_kw = sum(self.pv_system_power_kw_by_household.values())
        portfolio_battery_discharge_kwh = sum(self.battery_data_by_household[household].dmax * self.dt for household in self.households)
        portfolio_grid_limit_kwh = len(self.households) * 34.5 * self.dt
        max_market_export_dict = {}
        for bid_step in market_bid_steps_set:
            timestamp = tree_data["bid_step_to_timestamp"][bid_step]
            aggregate_clear_sky_pv_kwh = float(tree_data["clear_sky_pv_by_timestamp"][timestamp])

            available_pv_kwh = min(portfolio_pv_power_kw * self.dt, clear_sky_export_fraction * aggregate_clear_sky_pv_kwh)
            max_market_export_dict[bid_step] = min(portfolio_grid_limit_kwh, available_pv_kwh + portfolio_battery_discharge_kwh) ### Limit by grid and available PV + battery

        m.max_market_export = pyo.Param(m.MARKET_BID_STEPS, initialize=max_market_export_dict)
        m.max_market_import = pyo.Param(initialize=portfolio_grid_limit_kwh)
        
        m.inkoopvergoeding = pyo.Param(initialize=self.contract_data.inkoopvergoeding)
        m.verkoopvergoeding = pyo.Param(initialize=self.contract_data.verkoopvergoeding)
        m.import_belasting = pyo.Param(initialize=self.contract_data.import_belasting)
        m.jaarlijkse_belastingsvermindering = pyo.Param(initialize=self.contract_data.jaarlijkse_belastingsvermindering)
        m.SALDEREN = pyo.Param(initialize=self.contract_data.SALDEREN)
        # Still have to implement full length of simulation to get the yearly belastingvermindering

        ### Separate net loads
        m.individual_net_load = pyo.Param(m.HNST, initialize=tree_data["individual_net_load"], within=pyo.Reals) ### In kwh, can be negative. Defined as load - pv. Will now come from central framework
        m.aggregated_net_load = pyo.Param(m.NST, initialize=tree_data["aggregate_net_load"], within=pyo.Reals)


        ### Add the determinstic net load forecast independently
        #deterministic_net_load_dict = {}
        fixed_market_commitment_by_bid_step_dict = {} ### Not striclty necessary but more clearer
        for bid_step in market_bid_steps_set:
            timestamp = tree_data["bid_step_to_timestamp"][bid_step]
            #deterministic_net_load_dict[bid_step] = float(tree_data["deterministic_net_load_by_timestamp"][timestamp])
            fixed_market_commitment_by_bid_step_dict[bid_step] = float(fixed_market_commitment.get(timestamp, 0.0))

        #m.deterministic_net_load = pyo.Param(m.MARKET_BID_STEPS, initialize=deterministic_net_load_dict, within=pyo.Reals) ### Was commented out but leave here just in case
        m.fixed_market_commitment_by_bid_step = pyo.Param(m.MARKET_BID_STEPS, initialize=fixed_market_commitment_by_bid_step_dict, within=pyo.Reals)


        fixed_market_commitment_dict = {} ### Map all the current commitments to the nst index
        for name, stage, local_time in m.COMMITMENT_NST: ### Don't use the full NST anymore
            timestamp = tree_data["local_time_to_timestamps"][stage][local_time]
            ### Don't convert to bid step here, can leave as nst, by chat
            fixed_market_commitment_dict[name, stage, local_time] = float(fixed_market_commitment.get(timestamp, 0.0)) ### If not in series, default to 0.0 for not existing commitment yet.
        m.fixed_market_commitment = pyo.Param(m.COMMITMENT_NST, initialize=fixed_market_commitment_dict, within=pyo.Reals) ### Have to keep nst index for imbalance deviatipon penalty

        market_bid_price_dict = {} ### Load in the market prices for this specific market of this build. 
        for bid_step in market_bid_steps_set:
            timestamp = tree_data["bid_step_to_timestamp"][bid_step]
            price_value = float(tree_data["market_prices_by_timestamp"][market_name][timestamp]) ### Not over NST anymore, just timestamp, much lighter
            if pd.isna(price_value):
                raise ValueError(f"Missing price for {market_name} at {timestamp}.")
            market_bid_price_dict[bid_step] = price_value ### Same end result.
        m.market_bid_price = pyo.Param(m.MARKET_BID_STEPS, initialize=market_bid_price_dict, within=pyo.Reals)

        m.deviation_cost_penalty_low = pyo.Param(initialize=deviation_cost_penalty_low, within=pyo.NonNegativeReals)
        m.deviation_cost_penalty_high = pyo.Param(initialize=deviation_cost_penalty_high, within=pyo.NonNegativeReals)
        m.deviation_threshold_kwh = pyo.Param(initialize=deviation_threshold_kwh, within=pyo.NonNegativeReals)
        
        ### Introduce the quadratic penalty
        m.deviation_quadratic_penalty = pyo.Param(initialize=deviation_quadratic_penalty, within=pyo.NonNegativeReals) ### DON'T USE FOR NOW

        # Weights
        m.W_SOC = pyo.Param(initialize=weight_soc_soft, within=pyo.NonNegativeReals) ### SOC deviation penalty for soft constraint. HAS TO BE TUNED ACCORDING TO AMOUNT OF VIOLATIONS!


        ### Vars
        # For charge and discharge, have to look at different formulations! Can make into one var too and neglect efficiency of one of them, see article.
        # Can keep them separate and with price initiative with criterium can keep linear!! HAS TO BE DONE
        m.E_charge = pyo.Var(m.HNST, domain=pyo.NonNegativeReals) ### PUT IN KWH NOW IMMEDIATELY, was power in MPC single run
        m.E_discharge = pyo.Var(m.HNST, domain=pyo.NonNegativeReals) ### PUT IN KWH
        m.SOC = pyo.Var(m.HNST, domain=pyo.NonNegativeReals, bounds=(0.0, 1.0)) ### As fraction as in MPC single run. Now put as soft constraints!
        
        ### Should be implemented as one eventually and use formulation of Dario! However, from linearisation script, binaries were not needed due to tax.
        ### The physical grid vars:
        m.E_grid_import = pyo.Var(m.HNST, domain=pyo.NonNegativeReals) ### In KWH, just bound on 20kwh, doesn't really matter too much for now
        m.E_grid_export = pyo.Var(m.HNST, domain=pyo.NonNegativeReals) ### In KWH, just bound on 20kwh, doesn't really matter too much for now

        ### The market vars
        #m.E_market_bid = pyo.Var(m.MARKET_BID_STEPS, domain=pyo.Reals) ### Should be bounded also by physical grid limits --> NOt anymore, otherwise to go from +7kw to -5 for example you can have problem in correction
        def market_step_bounds(m, bid_step):
            fixed_commitment = pyo.value(m.fixed_market_commitment_by_bid_step[bid_step])
            lower_bound = math.ceil((-pyo.value(m.max_market_import) - fixed_commitment) / 25.0)
            upper_bound = math.floor((pyo.value(m.max_market_export[bid_step]) - fixed_commitment) / 25.0)
            return lower_bound, upper_bound

        m.market_bid_step_var = pyo.Var(m.MARKET_BID_STEPS, domain=pyo.Integers, bounds=market_step_bounds)
        m.E_market_bid = pyo.Expression(m.MARKET_BID_STEPS, rule=lambda m, bid_step: 25.0 * m.market_bid_step_var[bid_step])


        ### Deviation vars for soft-constraints, keeps solution feasible.
        m.E_deviation_positive = pyo.Var(m.COMMITMENT_NST, domain=pyo.NonNegativeReals) ### Separation done by chat, gurobi indeed doesn't want abs() in objective. MAYBE CAN BE SOLVED BY SAME LOGIC AS FOR DARIO PRICING TRICK
        m.E_deviation_negative = pyo.Var(m.COMMITMENT_NST, domain=pyo.NonNegativeReals) ### These can now be only on the commitment_NST

        ### Weights for soft constraints
        m.slack_soc_low = pyo.Var(m.HNST, domain=pyo.NonNegativeReals) ### Slack variable for SOC lower bound
        m.slack_soc_high = pyo.Var(m.HNST, domain=pyo.NonNegativeReals) ### Slack variable for SOC upper bound

        ### New formulation in two levels instead of one or quadratic dev_pen
        m.Excess_deviation_after_threshold = pyo.Var(m.COMMITMENT_NST, domain=pyo.NonNegativeReals)

        ### Curtailable PV is left out, net_load is already constructed inside nods, would be too complicated now

        ### Binaries
        m.battery_indicator = pyo.Var(m.BATTERY_BINARY_HNST, domain=pyo.Binary) ### Now depends on set
        if not linearise_grid_constraint:
            m.grid_indicator = pyo.Var(m.HNST, domain=pyo.Binary)
        


        ### Constraints
        def energy_balance_rule(m, household, name, stage, local_time):
            return m.E_grid_import[household, name, stage, local_time] - m.E_grid_export[household, name, stage, local_time] == m.E_charge[household, name, stage, local_time] - m.E_discharge[household, name, stage, local_time] + m.individual_net_load[household, name, stage, local_time]
        m.energy_balance_constraint = pyo.Constraint(m.HNST, rule=energy_balance_rule)

        ### Market constraint, be careful
        def market_commitment_rule(m, name, stage, local_time): ### Can stay almost exactly the same, just change to sum of all households.
            bid_step = tree_data["node_time_to_bid_step"][name, stage, local_time]

            if bid_step in market_bid_steps_set: ### Use the set, not the index object, fix by chat. commitment_nst contains all nst for which there is a market commitment so also of the day of the sim step.
                new_market_bid = m.E_market_bid[bid_step]
            else: 
                new_market_bid = 0.0

            portfolio_position = sum(m.E_discharge[household, name, stage, local_time] - m.E_charge[household, name, stage, local_time] for household in self.households) - m.aggregated_net_load[name, stage, local_time] ### Very good catch from chat, do not use E_grid_export - E_grid_import and sum ove rhouseholds because the net load has to come from the aggregated forecast as this is what we are commiting to the market!
            committed_market_position = m.fixed_market_commitment[name, stage, local_time] + new_market_bid ### Add new bid. Was both zero for timestamps outside of market interval. Was also wrong because later on, in the balance, objective was penalized for the physical commitment.

            return portfolio_position - committed_market_position == m.E_deviation_positive[name, stage, local_time] - m.E_deviation_negative[name, stage, local_time]
        m.market_commitment_constraint = pyo.Constraint(m.COMMITMENT_NST, rule=market_commitment_rule)


        ### Of course have to add cumulative bound by grid import and export!!!! System was making infinite bids
        def cumulative_market_commitment_grid_rule(m, bid_step):
            total_market_commitment = m.fixed_market_commitment_by_bid_step[bid_step] + m.E_market_bid[bid_step]
            return pyo.inequality(-m.max_market_import, total_market_commitment, m.max_market_export[bid_step])
        
        m.cumulative_market_commitment_constraint = pyo.Constraint(m.MARKET_BID_STEPS, rule=cumulative_market_commitment_grid_rule)
        """def cumulative_market_commitment_grid_rule(m, bid_step):
            total_market_commitment = m.fixed_market_commitment_by_bid_step[bid_step] + m.E_market_bid[bid_step]
            return pyo.inequality(-m.max_grid_import, total_market_commitment, m.max_grid_export)
        m.cumulative_market_commitment_constraint = pyo.Constraint(m.MARKET_BID_STEPS, rule=cumulative_market_commitment_grid_rule)"""

        """### And based on the deterministic net forecast ### Problems with this. It pulls the solution to the deterministic forecast and creates infeasibilities. Bettr to just use the cumulative bound by grid constraint
        def deterministic_market_commitment_grid_rule(m, bid_step):
            total_market_commitment = m.fixed_market_commitment_by_bid_step[bid_step] + m.E_market_bid[bid_step]
            return pyo.inequality((-m.cmax - m.deterministic_net_load[bid_step]), total_market_commitment, (m.dmax - m.deterministic_net_load[bid_step]))
        m.deterministic_market_commitment_constraint = pyo.Constraint(m.MARKET_BID_STEPS, rule=deterministic_market_commitment_grid_rule)"""

        def deviation_excess_rule(m, name, stage, local_time): ### Just to give value to Excess_deviation_after_threshold
            sum_deviation = m.E_deviation_positive[name, stage, local_time] + m.E_deviation_negative[name, stage, local_time] ### They shouldn't be nonzero at same time (could in theory happen but would be suboptimal)
            return m.Excess_deviation_after_threshold[name, stage, local_time] >= sum_deviation - m.deviation_threshold_kwh
        m.deviation_excess_constraint = pyo.Constraint(m.COMMITMENT_NST, rule=deviation_excess_rule)


        """### Now the whole knowledge part of the central controller, bring in the market step requirement
        m.market_bid_step_var = pyo.Var(m.MARKET_BID_STEPS, domain=pyo.Integers)
        def market_step_rule(m, bid_step):
            return m.E_market_bid[bid_step] == m.market_bid_step_var[bid_step] * 25.0 ### 25kwh steps == 100kw.
        m.market_step_constraint = pyo.Constraint(m.MARKET_BID_STEPS, rule=market_step_rule)"""
        ### Replace by: (and commented out m.E_market_bid = pyo.Var(m.MARKET_BID_STEPS, domain=pyo.Reals) too)

        ### Move everything to household dependency
        def soc_update_rule(m, household, name, stage, local_time): ### MAKE SURE TO FOLLOW CONVENTION OF LAST SCRIPT, SOC AT BEGINNING OF INTERVAL
            if local_time == 0: 
                parent = tree_data["parent"][name] ### Had to be added to tree_data dict
                if parent == "root": ### Root node is just a point actually
                    return m.SOC[household, name, stage, local_time] == m.soc_init[household] ### First time step of this MPC instance, has to be updated each mpc run!!!
                else: 
                    parent_stage = stage - 1
                    parent_last_local_time = max(tree_data["stage_local_times"][parent_stage]) ### Easier logic by chat, just get the whole list and max, parent end are built on the same branch intervals
                    previous_soc = m.SOC[household, parent, parent_stage, parent_last_local_time]

                    ### Apply last action to previous soc! Equal to prev_T
                    return m.SOC[household, name, stage, local_time] == previous_soc + (m.eta[household] * m.E_charge[household, parent, parent_stage, parent_last_local_time] - (1 / m.eta[household]) * m.E_discharge[household, parent, parent_stage, parent_last_local_time]) / m.Vmax[household]

            else:
                return m.SOC[household, name, stage, local_time] == m.SOC[household, name, stage, local_time - 1] + (m.eta[household] * m.E_charge[household, name, stage, local_time - 1] - (1 / m.eta[household]) * m.E_discharge[household, name, stage, local_time - 1]) / m.Vmax[household]
        m.soc_update_constraint = pyo.Constraint(m.HNST, rule=soc_update_rule)


        
        # SOC soft constraints
        def soc_soft_lower_soft_rule(m, household, name, stage, local_time):
            return m.SOC[household, name, stage, local_time] >= m.market_soc_min[household] - m.slack_soc_low[household, name, stage, local_time]
        m.soc_soft_lower_constraint = pyo.Constraint(m.HNST, rule=soc_soft_lower_soft_rule)

        def soc_soft_upper_soft_rule(m, household, name, stage, local_time):
            return m.SOC[household, name, stage, local_time] <= m.market_soc_max[household] + m.slack_soc_high[household, name, stage, local_time]
        m.soc_soft_upper_constraint = pyo.Constraint(m.HNST, rule=soc_soft_upper_soft_rule)


        ### these should all be linearised. Grid can be done just by neglecting, charging has to be done along criterium
        if not linearise_grid_constraint:
            def grid_exclusive_import(m, household, name, stage, local_time):
                return m.E_grid_import[household, name, stage, local_time] <= m.max_grid_import * m.grid_indicator[household, name, stage, local_time]
            m.grid_exclusive_import_constraint = pyo.Constraint(m.HNST, rule=grid_exclusive_import)

            def grid_exclusive_export(m, household, name, stage, local_time):
                return m.E_grid_export[household, name, stage, local_time] <= m.max_grid_export * (1 - m.grid_indicator[household, name, stage, local_time])
            m.grid_exclusive_export_constraint = pyo.Constraint(m.HNST, rule=grid_exclusive_export)
        else: 
            def grid_import_constraint(m, household, name, stage, local_time):
                return m.E_grid_import[household, name, stage, local_time] <= m.max_grid_import
            m.grid_import_constraint = pyo.Constraint(m.HNST, rule=grid_import_constraint)

            def grid_export_constraint(m, household, name, stage, local_time):
                return m.E_grid_export[household, name, stage, local_time] <= m.max_grid_export
            m.grid_export_constraint = pyo.Constraint(m.HNST, rule=grid_export_constraint)


        def max_charge_constraint(m, household, name, stage, local_time):
            return m.E_charge[household, name, stage, local_time] <= m.cmax[household] ### The normal constraint
        m.max_charge_constraint = pyo.Constraint(m.HNST, rule=max_charge_constraint)

        def max_discharge_constraint(m, household, name, stage, local_time):
            return m.E_discharge[household, name, stage, local_time] <= m.dmax[household] ### The normal constraint
        m.max_discharge_constraint = pyo.Constraint(m.HNST, rule=max_discharge_constraint)

        def charge_exclusive_constraint(m, household, name, stage, local_time): ### Additional constraint if MILP is needed for certain timesteps
            return m.E_charge[household, name, stage, local_time] <= m.cmax[household] * m.battery_indicator[household, name, stage, local_time]
        m.charge_exclusive_constraint = pyo.Constraint(m.BATTERY_BINARY_HNST, rule=charge_exclusive_constraint)

        def discharge_exclusive_constraint(m, household, name, stage, local_time): ### Additional constraint if MILP is needed for certain timesteps
            return m.E_discharge[household, name, stage, local_time] <= m.dmax[household] * (1 - m.battery_indicator[household, name, stage, local_time])
        m.discharge_exclusive_constraint = pyo.Constraint(m.BATTERY_BINARY_HNST, rule=discharge_exclusive_constraint)



        
        
        
        ### Still have to implement nonanticipativity for current step, has to be same action for all
        first_stage = min(tree_data["stages"]) ### Root was excluded from tree_data
        first_local_time = min(tree_data["stage_local_times"][first_stage])
        first_nodes = tree_data["nodes_per_stage_dict"][first_stage]
        # Easier to pick a reference node, proposed by chat
        reference_node = first_nodes[0]
        m.nonanticipativity_constraints = pyo.ConstraintList() ### Create the object first, method for adding constraints is different than for normal constraints
        for household in m.HOUSEHOLDS:
            for name in first_nodes[1:]: ### Exclude reference node
                m.nonanticipativity_constraints.add(m.E_charge[household, name, first_stage, first_local_time] == m.E_charge[household, reference_node, first_stage, first_local_time]) ### The choosable actions have to be added here.
                m.nonanticipativity_constraints.add(m.E_discharge[household, name, first_stage, first_local_time] == m.E_discharge[household, reference_node, first_stage, first_local_time])


        ### Still have to put SOC_terminal constraint: TERMINAL CONSTRAINT STILL PUT AS HARD CONSTRAINT, CAN BE RELAXED IF WANTED.
        last_stage = max(tree_data["stages"])
        last_local_time = max(tree_data["stage_local_times"][last_stage])
        def SOC_terminal_rule(m, household, name): ### Indexed on leaf nodes!
            soc_after_last_action = m.SOC[household, name, last_stage, last_local_time] + (m.eta[household] * m.E_charge[household, name, last_stage, last_local_time] - (1 / m.eta[household]) * m.E_discharge[household, name, last_stage, last_local_time]) / m.Vmax[household]
            return pyo.inequality(m.soc_min[household], soc_after_last_action, m.soc_max[household]) ### Same as before
        m.SOC_terminal_constraint = pyo.Constraint(m.HOUSEHOLDS, m.LEAF_NODES, rule=SOC_terminal_rule)

        ### Update: Have to constrain market bids before 2025-10-01 to be one hourly, make 15min intervals in same hour equal.
        da_hourly_bid_pairs = []
        if (market_name == "DA") and (bid_start is not None) and (bid_start < pd.Timestamp("2025-10-01 00:00:00", tz="Europe/Brussels")):
            sorted_bid_steps = sorted(market_bid_steps, key=lambda bid_step: tree_data["bid_step_to_timestamp"][bid_step]) ### key sorting, by chat, safer to sort on timestamp direclty

            for previous_bid_step, current_bid_step in zip(sorted_bid_steps[:-1], sorted_bid_steps[1:]):
                previous_timestamp = tree_data["bid_step_to_timestamp"][previous_bid_step]
                current_timestamp = tree_data["bid_step_to_timestamp"][current_bid_step]

                if (previous_timestamp.floor("h") == current_timestamp.floor("h")):
                    da_hourly_bid_pairs.append((previous_bid_step, current_bid_step)) ### 00:15 gets linked to 00:00, 00:30 to 00:15, 00:45 to 00:30
        m.DA_HOURLY_BID_PAIRS = pyo.Set(initialize=da_hourly_bid_pairs, dimen=2)

        def hourly_bid_before_october_rule(m, previous_bid_step, current_bid_step):
            return m.E_market_bid[previous_bid_step] == m.E_market_bid[current_bid_step]
        m.hourly_bid_before_october_constraint = pyo.Constraint(m.DA_HOURLY_BID_PAIRS, rule=hourly_bid_before_october_rule)

                



        ### Objective function
        def OBJ(m):
            ### Implement same logic as MPC_single_run
            total = 0
            for household, name, stage, local_time in m.HNST:
                probability_ = tree_data["probability"][name] ### Probability is indexed by node name only, see framework_v2.py

                if not self.contract_data.SALDEREN: ### We are maximizing instead of minimizing so flip signs
                    total += probability_ * (-(m.inkoopvergoeding + m.import_belasting) * m.E_grid_import[household, name, stage, local_time] + (m.verkoopvergoeding) * m.E_grid_export[household, name, stage, local_time])
                
                else: ### Met salderen ### Immediately add term here, it was actually equivalent to putting it separate like for single run
                    total += probability_ * (-(m.inkoopvergoeding) * m.E_grid_import[household, name, stage, local_time] + (m.verkoopvergoeding) * m.E_grid_export[household, name, stage, local_time] - (m.import_belasting) * (m.E_grid_import[household, name, stage, local_time] - m.E_grid_export[household, name, stage, local_time])) 


            ### REMEMBER TO ADD BELASTINGSVERMINDERING IN POST!!!!

            market_revenue = sum(m.market_bid_price[bid_step] * m.E_market_bid[bid_step] for bid_step in m.MARKET_BID_STEPS)
            ### Note that probability was indexed by node name only, see framework_v2.py
            deviation_cost = sum(tree_data["probability"][name] * m.deviation_cost_penalty_low * (m.E_deviation_positive[name, stage, local_time] + m.E_deviation_negative[name, stage, local_time]) for name, stage, local_time in m.COMMITMENT_NST) ### Deviation only valid over market interval now
            additional_deviation_cost = sum(tree_data["probability"][name] * m.deviation_cost_penalty_high * m.Excess_deviation_after_threshold[name, stage, local_time] for name, stage, local_time in m.COMMITMENT_NST)
            ### For debug, make it conditional
            if deviation_quadratic_penalty > 0.0:
                #quadratic_deviation_cost = sum(tree_data["probability"][name] * m.deviation_quadratic_penalty * (m.E_deviation_positive[name, stage, local_time] + m.E_deviation_negative[name, stage, local_time])**2 for name, stage, local_time in m.COMMITMENT_NST) ### Quadratic penalty for deviation to control amplitude
                ### Write it as sum of squares instead of square of sum for debug, the inner product might bug gurobi, debug. This makes the model nonlinear because of inner product I think.
                quadratic_deviation_cost = sum(tree_data["probability"][name] * m.deviation_quadratic_penalty * (m.E_deviation_positive[name, stage, local_time]**2 + m.E_deviation_negative[name, stage, local_time]**2) for name, stage, local_time in m.COMMITMENT_NST) ### Quadratic penalty for deviation to control amplitude
            else:
                quadratic_deviation_cost = 0

            degradation_cost = sum(tree_data["probability"][name] * m.c_deg[household] * (m.E_charge[household, name, stage, local_time] + m.E_discharge[household, name, stage, local_time]) for household, name, stage, local_time in m.HNST) ### Lile before but now multidimensional. Keep sum of two
            #soc_soft_cost = sum(tree_data["probability"][name] * m.W_SOC * (m.slack_soc_low[household, name, stage, local_time]**2 + m.slack_soc_high[household, name, stage, local_time]**2) for household, name, stage, local_time in m.HNST) ### Squared penalty for soft constraints, see article Cai
            ### DO NOTE THAT THIS TURNS THE PROBLEM INTO A QP/MIQP INSTEAD OF LP/MILP!!!
            ### Keep it linear now
            soc_soft_cost = sum(tree_data["probability"][name] * m.W_SOC * (m.slack_soc_low[household, name, stage, local_time] + m.slack_soc_high[household, name, stage, local_time]) for household, name, stage, local_time in m.HNST) ### Squared penalty for soft constraints, see article Cai
            return market_revenue + total - deviation_cost - additional_deviation_cost - quadratic_deviation_cost - degradation_cost - soc_soft_cost
        m.OBJ = pyo.Objective(rule=OBJ, sense=pyo.maximize)



        return m