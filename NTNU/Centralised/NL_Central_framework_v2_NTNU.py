import pandas as pd
from pathlib import Path
from copy import copy
from NL_Framework_v2_NTNU import Framework as HouseholdFramework, Node
from Residual_sampling_for_framework_multistage_NTNU import LoadResidualSampler

"""
Will make central approach now that is as compatible as possible with the market aggregator.
"""

### For PV, make separate class although it won't do much (for now), as PV analogs are currently not used. Makes it easy however to later insert necessary code. For Load, we can re-use the old sampler.
### Only take deterministic forecast for both from the individual households
class AggregatePV:
    def __init__(self, central_framework):
        self.central_framework = central_framework

    def make_pv_df(self, start, end):
        delivery_index = pd.date_range(start=start, end=end, freq="15min", inclusive="left")
        aggregate_pv_kwh = self.central_framework.tree_framework.deterministic_pv_forecast.reindex(delivery_index)

        if aggregate_pv_kwh.isna().any():
            raise ValueError(f"Missing aggregate PV forecast between {start} and {end}.")

        pv_df = pd.DataFrame({
            "forecasted_time": delivery_index,
            "forecasted_power_kw": aggregate_pv_kwh.to_numpy() / 0.25 ### Convert back to kw, make_child_scenarios() expects pv power in kw
        })
        return pv_df

    def run_for_timestep(self, timestep, end_timestep, n_scenarios, verbose=0, only_first_point=False):
        if n_scenarios != 0:
            raise ValueError("Aggregate PV does not support scenarios. Set n_scenarios to 0.")

        pv_df = self.make_pv_df(start=timestep, end=end_timestep)
        if only_first_point:
            pv_df = pv_df.iloc[[0]].copy()

        return pv_df, pd.DataFrame() ### Return empty DataFrame for scenarios

    def run_for_branch_timestep(self, root_timestep, branch_timestep, end_timestep, n_scenarios, verbose=0, only_first_point=False):
        if n_scenarios != 0:
            raise ValueError("Aggregate PV does not support scenarios. Set n_scenarios to 0.")

        pv_df = self.make_pv_df(start=branch_timestep, end=end_timestep)
        if only_first_point:
            pv_df = pv_df.iloc[[0]].copy()

        return pv_df, pd.DataFrame() ### Return empty DataFrame for scenarios




class CentralFramework:
    def __init__(self, households, to_amount, exclude_before_ts, perfect_DA_forecast=False):
        self.households = sorted(households)
        self.to_amount = to_amount
        if len(self.households) == 1:
            self.aggregate_household_naming = self.households[0] ### Just for the one household case for a second
        else:
            expected_number_of_households = 4 * to_amount
            if len(self.households) != expected_number_of_households:
                raise ValueError(f"to_amount={to_amount} corresponds to {expected_number_of_households} households, but {len(self.households)} households were provided.")

            self.aggregate_household_naming = int(str(9999) + str(4*to_amount))

        self.exclude_before_ts = exclude_before_ts
        ### Load in all household frameworks, should check how ram handles this
        self.household_frameworks = {} ### In the end, will only be used to call build_determininstic_pv_and_load_forecast()
        for household in self.households:
            self.household_frameworks[household] = HouseholdFramework(virtual_household=household, exclude_before_ts=exclude_before_ts, perfect_DA_forecast=perfect_DA_forecast, use_pv_residual_scenarios=False)


        self.reference_framework = self.household_frameworks[self.households[0]] ### Select e reference for the prices, just like we did in market aggregator

        self.tree_framework = copy(self.reference_framework) ### By chat in order to keep the same tree structure but then for the aggregated framework

        ### Have to take out load and pv samplers
        # Load
        #self.aggregate_load_residual_sampler = LoadResidualSampler(household=self.aggregate_household_naming, exclude_before_ts=self.exclude_before_ts)
        if len(self.households) == 1:
            self.aggregate_load_residual_sampler = self.household_frameworks[self.households[0]].load_residual_sampler
        else:
            self.aggregate_load_residual_sampler = LoadResidualSampler(household=self.aggregate_household_naming, exclude_before_ts=self.exclude_before_ts)
        ### Overwrite to connect to the aggregate tree structure
        self.tree_framework.load_residual_sampler = self.aggregate_load_residual_sampler
        self.tree_framework.load_forecaster_params = self.aggregate_load_residual_sampler.params
        # PV
        self.tree_framework.pv_residual_sampler = AggregatePV(self)
        self.tree_framework.pv_forecaster_params.USE_PV_RESIDUAL_SCENARIOS = False


        ### Add this here too for the MPC, expects battery_data_by_household, pv_system_power_kw_by_household, contract_data. Were separately obtained before too.
        self.battery_data_by_household = {household: framework.battery_data for household, framework in self.household_frameworks.items()}
        self.pv_system_power_kw_by_household = {household: framework.pv_system_power_kw for household, framework in self.household_frameworks.items()}
        self.contract_data = self.reference_framework.contract_data

        self.DA_prices_df = self.reference_framework.DA_prices_df
        self.IDA1_prices_df = self.reference_framework.IDA1_prices_df
        self.IDA2_prices_df = self.reference_framework.IDA2_prices_df
        self.IDA3_prices_df = self.reference_framework.IDA3_prices_df


    def build_all_deterministic_forecasts(self, simulation_step, horizon_end):
        ### Have to look at build_deterministic_pv_and_load_forecasts(), do exactly the same thing but include the aggregate
        delivery_index = pd.date_range(start=simulation_step, end=horizon_end, freq="15min", inclusive="left")

        self.individual_load_forecasts = {}
        self.individual_pv_forecasts = {}
        self.individual_net_load_forecasts = {}
        self.individual_clear_sky_pv_forecasts = {}

        aggregate_pv_forecast = pd.Series(0.0, index=delivery_index)
        aggregate_clear_sky_pv_forecast = pd.Series(0.0, index=delivery_index)

        for household, framework in self.household_frameworks.items():
            household_load_forecast, household_pv_forecast, household_net_load_forecast, household_clear_sky_pv_forecast = framework.build_deterministic_pv_and_load_forecasts(simulation_step=simulation_step, horizon_end=horizon_end)

            household_load_forecast = household_load_forecast.reindex(delivery_index)
            household_pv_forecast = household_pv_forecast.reindex(delivery_index)
            household_net_load_forecast = household_net_load_forecast.reindex(delivery_index)
            household_clear_sky_pv_forecast = household_clear_sky_pv_forecast.reindex(delivery_index)

            if household_load_forecast.isna().any():
                raise ValueError(f"Missing deterministic load forecast for household {household}.")
            if household_pv_forecast.isna().any():
                raise ValueError(f"Missing deterministic PV forecast for household {household}.")
            if household_net_load_forecast.isna().any():
                raise ValueError(f"Missing deterministic net-load forecast for household {household}.")
            if household_clear_sky_pv_forecast.isna().any():
                raise ValueError(f"Missing clear-sky PV forecast for household {household}.")
            
            self.individual_load_forecasts[household] = household_load_forecast
            self.individual_pv_forecasts[household] = household_pv_forecast
            self.individual_net_load_forecasts[household] = household_net_load_forecast
            self.individual_clear_sky_pv_forecasts[household] = household_clear_sky_pv_forecast

            aggregate_pv_forecast = aggregate_pv_forecast.add(household_pv_forecast)
            aggregate_clear_sky_pv_forecast = aggregate_clear_sky_pv_forecast.add(household_clear_sky_pv_forecast)

        aggregate_pv_forecast.name = "aggregate_pv_forecast"
        aggregate_clear_sky_pv_forecast.name = "aggregate_clear_sky_pv_forecast"

        if aggregate_pv_forecast.isna().any():
            raise ValueError(f"Missing aggregate PV forecast between {simulation_step} and {horizon_end}.")
        if aggregate_clear_sky_pv_forecast.isna().any():
            raise ValueError(f"Missing aggregate clear sky PV forecast between {simulation_step} and {horizon_end}.")

        ### Now make the aggregate load forecast
        load_root_origin = simulation_step.replace(year=2024) - pd.Timedelta(minutes=15)
        _,_,_, aggregate_load_forecast_df, _ = self.aggregate_load_residual_sampler.predict_for_root_timestamp_with_analogs(root_timestamp=load_root_origin, n_scenarios=0, same_time=True)
        load_start = simulation_step.replace(year=2024)
        horizon_duration = horizon_end.tz_convert("UTC") - simulation_step.tz_convert("UTC") ### This was because of DST handling, this code all comes from prementioned function
        load_end = load_start + horizon_duration

        aggregate_load_forecast = aggregate_load_forecast_df.loc[(aggregate_load_forecast_df.index >= load_start) & (aggregate_load_forecast_df.index < load_end), "current_forecast"].copy()

        if len(aggregate_load_forecast) != len(delivery_index):
            raise ValueError(f"Aggregate load forecast has {len(aggregate_load_forecast)} values, but {len(delivery_index)} were expected.")

        aggregate_load_forecast.index = delivery_index ### Set back to 2025/2026 index
        aggregate_load_forecast.name = "deterministic_aggregate_load_forecast"

        if aggregate_load_forecast.isna().any():
            raise ValueError(f"Missing aggregate load forecast between {simulation_step} and {horizon_end}.")

        aggregate_net_load_forecast = aggregate_load_forecast - aggregate_pv_forecast
        aggregate_net_load_forecast.name = "deterministic_aggregate_net_load_forecast"

        self.tree_framework.deterministic_load_forecast = aggregate_load_forecast
        self.tree_framework.deterministic_pv_forecast = aggregate_pv_forecast ### Use for AggregatePV class
        self.tree_framework.deterministic_net_load_forecast = aggregate_net_load_forecast
        self.tree_framework.clear_sky_pv_forecast = aggregate_clear_sky_pv_forecast




    def build_scenario_tree(self, simulation_step: pd.Timestamp, horizon_end=None, branch_hours_utc: list = [0, 6, 12, 18], scenario_counts=None, include_deterministic_load: bool = True, include_deterministic_pv: bool = True, extend_last_nonzero_branch_to_horizon_end: bool = False):
        ### Just copy whole previous function from Framework_V2.py
        if horizon_end is None:
            horizon_end = simulation_step + pd.Timedelta("48h")
        if not include_deterministic_pv: ### Little safeguard, by chat
            raise ValueError("Central framework requires deterministic PV forecast to be included.")

        self.build_all_deterministic_forecasts(simulation_step=simulation_step, horizon_end=horizon_end)
        self.tree_framework.load_tree_cache = {}
        self.tree_framework.pv_tree_cache = {}

        ### Copy all this crap from framework file
        ### Don't forget that root is actually also a branch
        branch_times = [simulation_step]

        utc_start = simulation_step.tz_convert("UTC")
        utc_end = horizon_end.tz_convert("UTC")

        utc_days = pd.date_range(start=utc_start.normalize(), end=utc_end.normalize(), freq="D", tz="UTC") #### Much easier logic, by chat

        for utc_day in utc_days:
            for hour in branch_hours_utc:
                utc_branch_time = pd.Timestamp(year=utc_day.year, month=utc_day.month, day=utc_day.day, hour=hour, minute=0, second=0, tz="UTC")
                if utc_start < utc_branch_time < utc_end:
                    branch_times.append(utc_branch_time.tz_convert("Europe/Brussels")) ### Convert back to Brussels time for the rest of the code


        branch_times.append(horizon_end) ### Otherwise tree stops at last branch time and not at the end of the horizon
        branch_times = sorted(list(dict.fromkeys(branch_times))) ### by chat fix, removes duplicates while preserving order. To zavoid if sim step is at branch time

        ### Make the branching dynamic now
        if scenario_counts is None:
            scenario_counts = [(0, 4)] * (len(branch_times) - 1) ### Fix by chat, -1 because branch_times are the start moments of the intervals (e.g. 4 branch times, 3 intervals)/ horizon_end is included
        if len(scenario_counts) < len(branch_times) - 1:
            scenario_counts = scenario_counts + [(0,0)] * ((len(branch_times) - 1) - len(scenario_counts))
        if len(scenario_counts) > len(branch_times) - 1:
            scenario_counts = scenario_counts[:len(branch_times) - 1]

        if any(n_pv != 0 for n_pv, _ in scenario_counts): ### Little check, by chat
            raise ValueError("Central framework does not support PV scenarios. Set n_pv to 0 in scenario_counts.")

        ### Implement extend to horizon end logic
        if extend_last_nonzero_branch_to_horizon_end:
            nonzero_indices = [i for i, (n_pv, n_load) in enumerate(scenario_counts) if n_pv > 0 or n_load > 0] ### By chat

            if len(nonzero_indices) > 0: ### Manipulate branch_times and scenario_counts
                last_nonzero_index = nonzero_indices[-1]
                
                if last_nonzero_index < len(scenario_counts) - 1:
                    branch_times = branch_times[:last_nonzero_index + 1]  + [horizon_end] ### End slicing is exclusive xd, do +1, by chat
                    scenario_counts = scenario_counts[:last_nonzero_index + 1]
            else: 
                branch_times = [simulation_step, horizon_end]
                scenario_counts = [(0, 0)]

        root_node = Node( ### Have to import Node too from Framework_V2.py
            name="root",
            parent=None,
            probability=1.0,
            start=simulation_step,
            end=simulation_step,
            stage=0,
            pv=pd.Series(dtype=float),
            load=pd.Series(dtype=float),
            net_load=pd.Series(dtype=float),
            load_trajectory_until_node_end=pd.Series(dtype=float),
            load_trajectory_key="root" ### Just call it root for now, not used anyway in its own block
        )

        nodes = {"root": root_node} ### Easiest for now
        current_parent_nodes = [root_node]

        for stage in range(len(branch_times)-1):
            child_start = branch_times[stage]
            child_end = branch_times[stage + 1]
            n_pv_scenarios, n_load_scenarios = scenario_counts[stage]
            if (stage == len(branch_times) - 2): ### If it is the last stage, do not save the load_trajectory_until_node_end, as it will never be used as a parent. To save RAM
                is_last_stage = True
            else:
                is_last_stage = False

            next_parent_nodes = []

            for parent_node in current_parent_nodes:
                children = self.tree_framework.make_child_scenarios(simulation_step=simulation_step, parent_node=parent_node, child_start=child_start, child_end=child_end, n_pv_scenarios=n_pv_scenarios, n_load_scenarios=n_load_scenarios, include_deterministic_load=include_deterministic_load, include_deterministic_pv=include_deterministic_pv, is_last_stage=is_last_stage)

                for child in children:
                    nodes[child.name] = child
                    next_parent_nodes.append(child)
            
            current_parent_nodes = next_parent_nodes
        
        return nodes



    def nodes_to_pyomo_per_stage(self, nodes): ### Make spearate function such that we can store individual net load and then the aggregate. This fucnction gives tree_data as output
        """Already returned from old function:
        return {
            "stages": stages,
            "nodes_per_stage_dict": nodes_per_stage_dict, ### Indexed by stage
            "stage_local_times": stage_times, ### Indexed by stage, list of local time indices
            "NST": NST, ### The index
            "net_load": net_load,
            "parent": parent,
            "probability": probability,
            "local_time_to_timestamps": local_time_to_timestamps, ### For reverse conversion to timestamps again
            #"node_last_local_time": node_last_local_time
            "leaf_nodes": leaf_nodes,
            "bid_steps": list(bid_step_to_timestamp.keys()), ### Added later, easier for index
            "bid_step_to_timestamp": bid_step_to_timestamp,
            "timestamp_to_bid_step": timestamp_to_bid_step,
            "node_time_to_bid_step": node_time_to_bid_step,
            "deterministic_load_by_timestamp": self.deterministic_load_forecast.to_dict(),
            "deterministic_pv_by_timestamp": self.deterministic_pv_forecast.to_dict(),
            "deterministic_net_load_by_timestamp": self.deterministic_net_load_forecast.to_dict(),
            "clear_sky_pv_by_timestamp": self.clear_sky_pv_forecast.to_dict()
        }"""
        tree_data = self.tree_framework.nodes_to_pyomo_per_stage(nodes)

        ### Should rename net_load as it is now of the central aggregate
        tree_data["aggregate_net_load"] = tree_data.pop("net_load")

        individual_net_load = {}
        ### No need to create HNST. Will do in MPC
        for household in self.households:
            household_net_load_forecast = self.individual_net_load_forecasts[household]

            for name, stage, local_time in tree_data["NST"]:
                timestamp = tree_data["local_time_to_timestamps"][stage][local_time]
                if timestamp not in household_net_load_forecast.index:
                    raise ValueError(f"Timestamp is missing from deterministic net-load forecast for household {household}.")

                value = household_net_load_forecast.loc[timestamp]
                if pd.isna(value):
                    raise ValueError(f"Deterministic net-load forecast for household {household} has NaN value at timestamp {timestamp}.")

                individual_net_load[household, name, stage, local_time] = float(value)

        tree_data["individual_net_load"] = individual_net_load

        return tree_data


    def add_market_prices_to_tree(self, tree_data, simulation_step):
        return self.reference_framework.add_market_prices_to_tree(tree_data=tree_data, simulation_step=simulation_step)
    