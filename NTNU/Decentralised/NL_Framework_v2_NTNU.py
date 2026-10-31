import sys
import pandas as pd
from pathlib import Path
import numpy as np
from termcolor import colored
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import time


from Residual_sampling_for_framework_multistage_NTNU import Load_Forecaster_Params, LoadResidualSampler
from PV_residual_for_framework_multistage_NTNU import PV_residual_behaviour_params, Virtual_household_residual_searching
from NL_MPC_NTNU import BatteryData, ContractData

SCRATCH_DIR = Path("/cluster/work/brechtg")
MARKET_DATA_DIR = SCRATCH_DIR / "Market_MPC" / "Markets" / "Markets Data_NL"
EPEXPREDICTOR_FILE = MARKET_DATA_DIR / "Epexpredictor_forecasted_prices" / "epexpredictor_NL_daily_train_15min_forecasts_PATCHED.parquet"
PROCESSED_MARKET_DIR = MARKET_DATA_DIR / "Processed"

"""
V2: 
Cache the pv scenarios, they ae re-calculated for each stage but they stay the same.
Cache the load scenarios (more limited due to dependency on parent node) but still quite some redundancy.
Make option to include deterministic forecast for load and pv separately.
    Note that n_scenarios is still the number of scenarios excluding the deterministic forecast. By including deterministic, you get n_scenarios + 1
The combination of the two is also made more efficient by taking out the initial preparation of the load and pv data
"""
"""
The general framework code for simulation of the virtual houses. 
Have to get load model, make prediction, get scenarios for load. See Residual_sampling.py
Have to get weather_df for the virtual house, make prediction, get scenarios by means of PVOutput. ### Check out Residual_searching_v4_virtual_house_parallel.py
Have to make multistage MPC optimisation for the virtual house. Branching on PV weather df updates and in theory also on DA, IDA1, IDA2 and IDA3 revelation but as I use deterministic forecast, this is basically doing it with one scenario.


Notes: 
    For now will just make perfect price forecasting. This should then be replaced with epexpredictor input.
    For each time branching, should make new prediction for load and PV of course. Can be made computationally cheaper by making less scenarios as you branch later in the future. 
    Might make an option to choose how many steps are impelmented by MPC. If you do 2, then two 15 min steps are implemented immediately and heavily reducing computational time.


To do:
    Will fail when there is a long history missing for a pvoutput house.    
    Can do plotting maybe with networkx as nx (see github of multi-stage-stachastic-programming)
"""


@dataclass
class Framework_Params: ### Should be deleted
    single_run_branch_times: list = field(default_factory=lambda: [0, 6, 12, 18]) ### The times IN UTC at which the MPC will branch on the PV weather forecast.



### The dynamic classes 
@dataclass
class Node:
    name: str
    parent: str | None
    probability: float
    start: pd.Timestamp
    end: pd.Timestamp
    stage: int

    pv: pd.Series ### To keep timestamps. Might change later to numpy array for speed/ram.
    load: pd.Series
    net_load: pd.Series ### For combining pv and load

    load_trajectory_until_node_end: pd.Series

    load_trajectory_key: str ### For caching!


class Framework: ### All-holding class for one household
    MARKET_PRICES_CACHE = None ### These vars will be shared across all instances of the Framework class within one process.
    EPEXPREDICTOR_FORECASTS_CACHE = None

    def __init__(self, virtual_household: int, exclude_before_ts: str, perfect_DA_forecast: bool = False, use_pv_residual_scenarios: bool = True):
        self.virtual_household = virtual_household

        # Setup load
        ### Note: relies on the predicitions.parquet files in the forecast Models folder, created by Save_forecasts.py. These files do take around 42gb of space so this should be revised.
        self.load_residual_sampler = LoadResidualSampler(household=virtual_household, exclude_before_ts=exclude_before_ts) ### Note: relies on the predicitions.parquet files in the forecast Models folder, created by Save_forecasts.py. These files do take around 42gb of space.
        self.load_forecaster_params = self.load_residual_sampler.params
        #print(colored(f"Load residual sampler for household {virtual_household} initialized.", "green"))

        # Setup PV. Will still write script but only input should be virtual house and timestep. Loading the pv cache is not very ram intensive as now it's binded to only loading the nearby houses
        self.pv_forecaster_params = PV_residual_behaviour_params() ### Also holds Flanders_buildings_assigned_file. From here, should load battery data
        self.pv_forecaster_params.USE_PV_RESIDUAL_SCENARIOS = use_pv_residual_scenarios ### If false, will just use deterministic forecast and skip caching to save memory
        self.pv_residual_sampler = Virtual_household_residual_searching(self.pv_forecaster_params, virtual_household=virtual_household)
        #print(colored(f"PV residual sampler for household {virtual_household} initialized.", "green"))

        # Load in prices
        if perfect_DA_forecast:
            self.perfect_DA_forecast = True
            ### DA prices always have to be loaded of course
        else: 
            self.perfect_DA_forecast = False
            if Framework.EPEXPREDICTOR_FORECASTS_CACHE is None:
                Framework.EPEXPREDICTOR_FORECASTS_CACHE = self.init_load_epexpredictor_forecasts(EPEXPREDICTOR_FILE)
            self.epexpredictor_forecasts_df = Framework.EPEXPREDICTOR_FORECASTS_CACHE
        
        if Framework.MARKET_PRICES_CACHE is None:
            Framework.MARKET_PRICES_CACHE = {
                "DA": pd.read_parquet(PROCESSED_MARKET_DIR / "DA_processed.parquet"),
                "IDA1": pd.read_parquet(PROCESSED_MARKET_DIR / "IDA1_processed.parquet"),
                "IDA2": pd.read_parquet(PROCESSED_MARKET_DIR / "IDA2_processed.parquet"),
                "IDA3": pd.read_parquet(PROCESSED_MARKET_DIR / "IDA3_processed.parquet")
            }

        self.DA_prices_df = Framework.MARKET_PRICES_CACHE["DA"]
        self.IDA1_prices_df = Framework.MARKET_PRICES_CACHE["IDA1"]
        self.IDA2_prices_df = Framework.MARKET_PRICES_CACHE["IDA2"]
        self.IDA3_prices_df = Framework.MARKET_PRICES_CACHE["IDA3"]
        #print(colored(f"Market prices loaded.", "green"))

        # Load in battery data for the virtual house. Already made available in the PV_residual_for_framework.py class
        self.battery_data = BatteryData(Vmax=self.pv_residual_sampler.virtual_house_battery_capacity)
        self.contract_data = ContractData()
        self.pv_system_power_kw = self.pv_residual_sampler.PV_params_virtual_household["system_size_w"] / 1000



    def init_load_epexpredictor_forecasts(self, file_path: Path):
        epexp_df = pd.read_parquet(file_path, columns=["model_origin", "forecast_origin", "timestamp", "price_forecast_eur_mwh"])
        convert_cols = ["model_origin", "forecast_origin", "timestamp"]
        for col in convert_cols:
            epexp_df[col] = pd.to_datetime(epexp_df[col], utc=True).dt.tz_convert("Europe/Brussels")
        return epexp_df


    ### Get the deterministic forecasts independently for the market commitment constraints. This may not be optimal but good for now.
    ### This approach didn't work well. However, we will re-use this part and leave it anyway. We will just pass the clear sky power on and use that + dmax as the export limit. This is already a lot better than a fixed limit.
    def build_deterministic_pv_and_load_forecasts(self, simulation_step: pd.Timestamp, horizon_end: pd.Timestamp):
        delivery_index = pd.date_range(start=simulation_step, end=horizon_end, freq="15min", inclusive="left")

        ### Load, just exactly like before
        load_root_origin = simulation_step.replace(year=2024) - pd.Timedelta(self.load_forecaster_params.period)
        _, _, _, load_forecast_df, _ = self.load_residual_sampler.predict_for_root_timestamp_with_analogs(root_timestamp=load_root_origin, n_scenarios=0, same_time=True)
        load_start = simulation_step.replace(year=2024)
        horizon_end_duration = horizon_end.tz_convert("UTC") - simulation_step.tz_convert("UTC")
        load_end = load_start + horizon_end_duration
        deterministic_load_serie = load_forecast_df.loc[(load_forecast_df.index >= load_start) & (load_forecast_df.index < load_end), "current_forecast"].copy()

        if len(deterministic_load_serie) != len(delivery_index):
            raise ValueError(f"Deterministic load forecast length {len(deterministic_load_serie)} does not match delivery index length {len(delivery_index)}.")

        deterministic_load_serie.index = delivery_index

        ### PV
        # The old version:
        pv_forecast_df, _ = self.pv_residual_sampler.run_for_timestep(timestep=simulation_step, end_timestep=horizon_end, n_scenarios=0, verbose=0, only_first_point=False)
        """
        deterministic_pv_serie = pv_forecast_df.drop_duplicates(subset="forecasted_time", keep="last").set_index("forecasted_time")["forecasted_power_kw"].reindex(delivery_index) * 0.25 ### kwh, by chta
        if deterministic_pv_serie.isna().any():
            raise ValueError(f"NaNs in deterministic PV forecast for {simulation_step} to {horizon_end}.")"""
        # We now only need the clear sky power
        pv_forecast_df = pv_forecast_df.drop_duplicates(subset="forecasted_time", keep="last").set_index("forecasted_time")
        deterministic_pv_serie = pv_forecast_df["forecasted_power_kw"].reindex(delivery_index) * 0.25 ### kwh
        clear_sky_pv_serie = pv_forecast_df["clear_sky_fitting_factor_power"].reindex(delivery_index) * 0.25

        if deterministic_pv_serie.isna().any() or clear_sky_pv_serie.isna().any():
            raise ValueError(f"NaNs in deterministic PV forecast or clear sky power for {simulation_step} to {horizon_end}.")
        
            

        deterministic_net_load_serie = deterministic_load_serie - deterministic_pv_serie
        deterministic_load_serie.name = "deterministic_load_kwh"
        deterministic_pv_serie.name = "deterministic_pv_kwh"
        clear_sky_pv_serie.name = "clear_sky_power_kwh"
        deterministic_net_load_serie.name = "deterministic_net_load_kwh"

        return deterministic_load_serie, deterministic_pv_serie, deterministic_net_load_serie, clear_sky_pv_serie
        
    

    ### Now have to build whole scenario tree logic
    def make_child_scenarios(self, simulation_step: pd.Timestamp, parent_node: Node, child_start, child_end, n_pv_scenarios, n_load_scenarios, include_deterministic_load: bool, include_deterministic_pv: bool, is_last_stage: bool = False): ### Just hardcode to 15min by now
        ### Include current sim_timestep in order to be careful for future leakage. This is the moment until where information is known.



        ### Mark if parent is the root
        parent_is_root = True if parent_node.name == "root" else False

        ### Load
        # Part changes to include caching
        load_child_start = child_start.replace(year=2024)
        #load_child_end = child_end.replace(year=2024) ### Update: this gives problems with DST
        load_child_duration = child_end.tz_convert("UTC") - child_start.tz_convert("UTC")
        load_child_end = load_child_start + load_child_duration
        load_cache_key = (parent_node.load_trajectory_key, load_child_start, load_child_end, n_load_scenarios, include_deterministic_load, parent_is_root)

        if load_cache_key in self.load_tree_cache:
            load_scenarios_df = self.load_tree_cache[load_cache_key].copy()
            parent_trajectory_load = parent_node.load_trajectory_until_node_end ### Cache in for concatenation later again
        else:

            if parent_is_root:
                load_root_origin = simulation_step.replace(year=2024) - pd.Timedelta(self.load_forecaster_params.period)
                _, _, load_scenarios_df, load_forecast_df, _ = self.load_residual_sampler.predict_for_root_timestamp_with_analogs(root_timestamp=load_root_origin, n_scenarios=n_load_scenarios, same_time=True)
                
                if load_scenarios_df is None:
                    load_scenarios_df = pd.DataFrame(index=load_forecast_df.index) ### Empty df with correct index, fix for scenario count (0,0)
                if include_deterministic_load:
                    load_scenarios_df["Deterministic forecast"] = load_forecast_df["current_forecast"] ### Called current forecast now, not det forecast

                parent_trajectory_load = pd.Series(dtype=float) ### Set empty because it is root

            else:
                parent_trajectory_load_df = parent_node.load_trajectory_until_node_end.to_frame(name="current_forecast") ### Has to conicide with logic from the load residual sampler
                _, _, load_scenarios_df, load_forecast_df, _ = self.load_residual_sampler.predict_for_branch_timestamp_with_analogs(origin_timestamp=simulation_step.replace(year=2024), branch_timestamp=child_start.replace(year=2024), parent_trajectory=parent_trajectory_load_df, n_scenarios=n_load_scenarios, same_time=True)
                
                if load_scenarios_df is None:
                    load_scenarios_df = pd.DataFrame(index=load_forecast_df.index) ### Empty df with correct index, fix for scenario count (0,0)
                if include_deterministic_load:
                    load_scenarios_df["Deterministic forecast"] = load_forecast_df["current_forecast"]

                parent_trajectory_load = parent_node.load_trajectory_until_node_end ### Cache in for concatenation later

            load_scenarios_df = load_scenarios_df.loc[(load_scenarios_df.index >= load_child_start) & (load_scenarios_df.index < load_child_end)] ### Cut to correct interval, year fixed by chat
            
            self.load_tree_cache[load_cache_key] = load_scenarios_df.copy() ### Add to cache

        load_cols = [col for col in load_scenarios_df.columns if col.startswith("scenario_")] ### Check syntax from load residual sampler
        load_cols = sorted(load_cols, key=lambda col: int(col.split("_")[1])) ### Sorting by chat
        # Add deterministic forecast
        if include_deterministic_load:
            load_cols.append("Deterministic forecast")
        if len(load_cols) == 0:
            raise ValueError(f"No load scenario columns found for {child_start} to {child_end}.")


        ### PV
        if n_pv_scenarios > 0 and not self.pv_forecaster_params.USE_PV_RESIDUAL_SCENARIOS:
            raise ValueError(f"n_pv_scenarios={n_pv_scenarios}, but USE_PV_RESIDUAL_SCENARIOS=False.")
        pv_cache_key = (child_start, child_end, n_pv_scenarios, include_deterministic_pv, parent_is_root)
        if pv_cache_key in self.pv_tree_cache: ### Created in building
            pv_scenarios_df = self.pv_tree_cache[pv_cache_key].copy()
        else:
            if parent_is_root:
                pv_scenarios_df, _ = self.pv_residual_sampler.run_for_timestep(timestep=simulation_step, end_timestep=child_end, n_scenarios=n_pv_scenarios, verbose=0, only_first_point=False) ### pass end_timestep now for 48h adaptation
            else:
                pv_scenarios_df, _ = self.pv_residual_sampler.run_for_branch_timestep(root_timestep=simulation_step, branch_timestep=child_start, end_timestep=child_end, n_scenarios=n_pv_scenarios, verbose=0, only_first_point=False)
            
            pv_scenarios_df = pv_scenarios_df.set_index("forecasted_time")
            pv_scenarios_df = pv_scenarios_df.loc[(pv_scenarios_df.index >= child_start) & (pv_scenarios_df.index < child_end)] ### Cut to correct interval

            self.pv_tree_cache[pv_cache_key] = pv_scenarios_df.copy() ### Add to cache

        if include_deterministic_pv:
            pv_scenarios_df["Deterministic forecast"] = pv_scenarios_df["forecasted_power_kw"]

        pv_cols = [col for col in pv_scenarios_df.columns if col.startswith("scenario_") and col.endswith("_power_kw")] ### See get clear sky function
        pv_cols = sorted(pv_cols, key=lambda col: int(col.split("_")[1])) ### Sorting by chat
        if include_deterministic_pv:
            pv_cols.append("Deterministic forecast")
        if len(pv_cols) == 0:
            raise ValueError(f"No PV scenario columns found for {child_start} to {child_end}.")
        

        ### Now have to combine load and pv scenarios
        child_index = pd.date_range(start=child_start, end=child_end, freq="15min", inclusive="left") ### Move out of loop
        children = []

        # Take out load loop
        load_items = []
        for i_load, load_col in enumerate(load_cols):
            load_scenario_serie = load_scenarios_df[load_col].copy() ### Still in 2024 format
            if len(load_scenario_serie) != len(child_index): ### If no bug, should return same index but in different year
                raise ValueError(f"Load scenario series length {len(load_scenario_serie)} does not match child index length {len(child_index)}.")
            
            load_simulation_serie = load_scenario_serie.copy() ### LOAD IS IN KWH PERO 15MIN, RESULTS FROM ALL_PREPPED_2_YEARS STILL
            load_simulation_serie.index = child_index ### Now in 2026/2026 format
            load_simulation_serie.name = "load"

            if load_simulation_serie.isna().any():
                raise ValueError(f"NaNs in load simulation series for {child_start} to {child_end}.")
            
            ### Empty concat warning pandas
            if parent_trajectory_load.empty:
                new_load_trajectory_until_node_end = load_scenario_serie.sort_index()
            elif load_scenario_serie.empty:
                new_load_trajectory_until_node_end = parent_trajectory_load.sort_index()
            else: 
                new_load_trajectory_until_node_end = pd.concat([parent_trajectory_load, load_scenario_serie]).sort_index() ### HAVE TO UPDATE OF COURSE. HAVE TO USE SCENARIO SERIE, NOT SIM SERIE, always has to be kept in 2024

            new_load_trajectory_until_node_end = new_load_trajectory_until_node_end[~new_load_trajectory_until_node_end.index.duplicated(keep='last')] ### Remove duplicates but should check this
            new_load_trajectory_key = f"{parent_node.load_trajectory_key}_{load_col}" ### Has to be retracable to the parent node and the load scenario

            load_items.append((i_load, load_col, load_simulation_serie, new_load_trajectory_until_node_end, new_load_trajectory_key)) ### Save as a list of tuples

        # Take out pv loop
        pv_items = []
        for i_pv, pv_col in enumerate(pv_cols):
            pv_scenario_serie = pv_scenarios_df[pv_col].copy()
            pv_simulation_serie = pv_scenario_serie.reindex(child_index)
            pv_simulation_serie.name = "pv"

            if pv_simulation_serie.isna().any():
                raise ValueError(f"NaNs in PV simulation series for {child_start} to {child_end}.")
            
            pv_energy_serie = pv_simulation_serie * 0.25 ### kwh, as 15min intervals, like LOAD!!
            pv_energy_serie.name = "pv_kwh"

            pv_items.append((i_pv, pv_col, pv_energy_serie)) ### Save as a list of tuples

        # Now combine load and pv items
        for i_load, load_col, load_simulation_serie, new_load_trajectory_until_node_end, new_load_trajectory_key in load_items: ### Load the whole tuple
            for i_pv, pv_col, pv_energy_serie in pv_items: ### Load the whole tuple
                net_load_serie = load_simulation_serie - pv_energy_serie ### SHOULD CHECK THIS TO CONFIRM 100%
                net_load_serie.name = "net_load_kwh"

                if net_load_serie.isna().any():
                    raise ValueError(f"NaNs in net load series for {child_start} to {child_end}.")

                ### Now able to build full child
                """child = Node(
                    name=f"{parent_node.name}_Load{i_load}_PV{i_pv}",
                    parent=parent_node.name,
                    probability=parent_node.probability * (1 / len(load_cols)) * (1 / len(pv_cols)), ### Assuming equal probability for now but this will change
                    start=child_start,
                    end=child_end,
                    stage=parent_node.stage + 1,
                    pv=pv_energy_serie,
                    load=load_simulation_serie.copy(),
                    net_load=net_load_serie.copy(),
                    load_trajectory_until_node_end=new_load_trajectory_until_node_end,
                    load_trajectory_key=new_load_trajectory_key
                )"""
                if is_last_stage:
                    stored_load_trajectory_until_node_end = pd.Series(dtype=float) ### Empty series to save RAM, not used
                else:
                    stored_load_trajectory_until_node_end = new_load_trajectory_until_node_end ### Save full trajectory until node end for next stage

                child = Node( ### Only net_load is used eventually, don't save load and pv to save RAM
                    name=f"{parent_node.name}_Load{i_load}_PV{i_pv}",
                    parent=parent_node.name,
                    probability=parent_node.probability * (1 / len(load_cols)) * (1 / len(pv_cols)), ### Assuming equal probability for now but this will change
                    start=child_start,
                    end=child_end,
                    stage=parent_node.stage + 1,
                    pv=pd.Series(dtype=float), ### Empty series to save RAM, not used
                    load=pd.Series(dtype=float), ### Empty series to save RAM, not used
                    net_load=net_load_serie.copy().astype("float32"), ### Save RAM by using float32 instead of float64
                    load_trajectory_until_node_end=stored_load_trajectory_until_node_end,
                    load_trajectory_key=new_load_trajectory_key
                )

                children.append(child)

        return children
    



    def build_scenario_tree(self, simulation_step: pd.Timestamp, horizon_end=None, branch_hours_utc: list = [0, 6, 12, 18], scenario_counts=None, include_deterministic_load: bool = True, include_deterministic_pv: bool = True, extend_last_nonzero_branch_to_horizon_end: bool = False): ### Scenario counts is actually a list of tuples. I want to integrate decreasing amount of scenarios in order to keep computation tractable.
        ### NOTE THAT FOR NOW THE DETERMINISTIC FORECAST IS ALSO INCLUDED --> WE GET 5 + 1
        ### Something like this. SHould be evaluated for performance and results
        self.pv_tree_cache = {} ### V2
        self.load_tree_cache = {} ### V2
        if horizon_end is None:
            horizon_end = simulation_step + pd.Timedelta("48h")

        ### Update: retrieve separately the deterministic forecasts for the market commitment constraints, 
        self.deterministic_load_forecast, self.deterministic_pv_forecast, self.deterministic_net_load_forecast, self.clear_sky_pv_forecast = self.build_deterministic_pv_and_load_forecasts(simulation_step=simulation_step, horizon_end=horizon_end)
        
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
            scenario_counts = [(4, 4)] * (len(branch_times) - 1) ### Fix by chat, -1 because branch_times are the start moments of the intervals (e.g. 4 branch times, 3 intervals)/ horizon_end is included
        if len(scenario_counts) < len(branch_times) - 1:
            scenario_counts = scenario_counts + [(0,0)] * ((len(branch_times) - 1) - len(scenario_counts))
        if len(scenario_counts) > len(branch_times) - 1:
            scenario_counts = scenario_counts[:len(branch_times) - 1]

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


        root_node = Node(
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
            start_perf_time = time.time()
            child_start = branch_times[stage]
            child_end = branch_times[stage + 1]
            n_pv_scenarios, n_load_scenarios = scenario_counts[stage]
            if (stage == len(branch_times) - 2): ### If it is the last stage, do not save the load_trajectory_until_node_end, as it will never be used as a parent. To save RAM
                is_last_stage = True
            else:
                is_last_stage = False

            next_parent_nodes = []

            for parent_node in current_parent_nodes:
                children = self.make_child_scenarios(simulation_step=simulation_step, parent_node=parent_node, child_start=child_start, child_end=child_end, n_pv_scenarios=n_pv_scenarios, n_load_scenarios=n_load_scenarios, include_deterministic_load=include_deterministic_load, include_deterministic_pv=include_deterministic_pv, is_last_stage=is_last_stage)

                for child in children:
                    nodes[child.name] = child
                    next_parent_nodes.append(child)
            
            current_parent_nodes = next_parent_nodes
            end_perf_time = time.time()
            ### THIS IS INTERESTING FOR PERFORMANCE TESTING
            #print(colored(f"Stage {stage+1} completed: {child_start} to {child_end} with {len(current_parent_nodes)} nodes in {end_perf_time - start_perf_time:.2f} seconds.", "blue"))
        
        return nodes
                


    #@staticmethod ### To avoid the class putting self in the function call, gives error called with one argument instead of two, fix by chat
    ### Take out the staticmethod as now we will be including the deterministic forecast in the tree as well
    def nodes_to_pyomo_per_stage(self, nodes: dict): ### Follow example of NTNU script for two-stage
        nodes_except_root = {name: node for name, node in nodes.items() if name != "root"}

        stages = sorted(set(node.stage for node in nodes_except_root.values())) ### use set 

        nodes_per_stage_dict = {stage: [name for name, node in nodes_except_root.items() if node.stage == stage] for stage in stages} ### map the nodes of that stage to the stage

        stage_times = {}
        local_time_to_timestamps = {}
        net_load = {}

        for stage in stages:
            timestamps = sorted(set(ts for name in nodes_per_stage_dict[stage] for ts in nodes_except_root[name].net_load.index)) ### Can use net load or load or pv, use set to avoid duplicates
            stage_times[stage] = list(range(len(timestamps))) ### Like in sdp script of ntnu, don't use real timestamps, can convert back later
            local_time_to_timestamps[stage] = {t: ts for t, ts in enumerate(timestamps)} ### t will be the local time index for that stage, ts is the real timestamp
            timestamps_to_local_time = {ts: t for t, ts in local_time_to_timestamps[stage].items()} ### Reverse mapping for later use also needed later on, easier

            for name in nodes_per_stage_dict[stage]:
                for ts, value in nodes_except_root[name].net_load.items():
                    t = timestamps_to_local_time[ts]
                    net_load[name, stage, t] = float(value) ### Keep this mapping for pyomo, float to avoid numpy float issues
        

        NST = [(name, stage, t) for stage in stages for name in nodes_per_stage_dict[stage] for t in stage_times[stage]] ### For pyomo indexing

        """### Needed for soc update constraint, keeps things easier
        node_last_local_time = {}
        for name, stage, local_time in NST:
            node_last_local_time[name] = max(node_last_local_time.get(name), local_time) ### Could do get(name, default) but shouldn't be necessary."""
        
        ### Needed for SOC terminal rule, easier. Mark the leaf nodes
        leaf_nodes = nodes_per_stage_dict[max(stages)]

        ### Only keep what's really important for MPC
        parent = {name: node.parent for name, node in nodes_except_root.items()} ### For soc transfer
        probability = {name: float(node.probability) for name, node in nodes_except_root.items()}


        ### Market update: separate market bid from scenario physical
        all_timestamps = sorted(set(local_time_to_timestamps[stage][local_time] for name, stage, local_time in NST)) ### Set for all timestamps
        # Do like before to make easy conversion
        bid_step_to_timestamp = {bid_step: ts for bid_step, ts in enumerate(all_timestamps)} ### Make bid step just like local time, but for all timestamps, not per stage
        timestamp_to_bid_step = {ts: bid_step for bid_step, ts in bid_step_to_timestamp.items()}

        node_time_to_bid_step = {}
        for name, stage, local_time in NST:
            ts = local_time_to_timestamps[stage][local_time]
            node_time_to_bid_step[name, stage, local_time] = timestamp_to_bid_step[ts]

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
        }
    

    def add_market_prices_to_tree(self, tree_data, simulation_step):
        market_series_dict = {"DA": self.DA_prices_df, "IDA1": self.IDA1_prices_df, "IDA2": self.IDA2_prices_df, "IDA3": self.IDA3_prices_df} ### Make sure they are stored as euro/kwh and indexed by pd timestamp
        ### Make spearate retrieve function from now on.

        ### Updated version, do not index by nst anymore but just by timestamp, much lighter
        price_by_market_dict = {}
        source_by_market_dict = {}
        all_timestamps = sorted(tree_data["bid_step_to_timestamp"].values())
        for market_name, market_series in market_series_dict.items():
            ### Cache first, then retrieve
            known_timestamps = []
            forecast_timestamps = []
            for delivery_ts in all_timestamps:
                if self.price_is_known(market_name, simulation_step, delivery_ts):
                    known_timestamps.append(delivery_ts)
                else:
                    forecast_timestamps.append(delivery_ts)

            known_prices = market_series.reindex(known_timestamps)
            if len(forecast_timestamps) > 0:
                forecast_prices = self.predict_market_prices(market_name, simulation_step, forecast_timestamps)
            else:
                #forecast_prices = pd.Series(dtype=float, name="price_eur_per_kwh").to_frame() ### Don't pass empty series, pass empty dataframe due to updated logic in price forecast.
                forecast_prices = market_series.iloc[0:0].copy() ### Empty dataframe with same columns as market_series, better, by chat
                
            combined_prices = pd.concat([known_prices, forecast_prices]).sort_index() ### Bring it all together again
            price_series = combined_prices["price_eur_per_kwh"].reindex(all_timestamps)

            price_by_market_dict[market_name] = {timestamp_ : float(price) for timestamp_, price in price_series.items()}
            ### Don't loop over nst anymore
            known_timestamps_set = set(known_timestamps)
            source_by_market_dict[market_name] = {timestamp_: "known" if timestamp_ in known_timestamps_set else "forecast" for timestamp_ in all_timestamps} ### Same as before

        
        tree_data["market_prices_by_timestamp"] = price_by_market_dict
        tree_data["market_price_source_by_timestamp"] = source_by_market_dict
        
        return tree_data


    def price_is_known(self, market_name, simulation_step, delivery_timestamp):
        delivery_day = delivery_timestamp.normalize()
        previous_day = delivery_day - pd.DateOffset(days=1)

        if market_name == "DA":
            publication_time = previous_day.replace(hour=13, minute=0)
        elif market_name == "IDA1":
            publication_time = previous_day.replace(hour=15, minute=20)
        elif market_name == "IDA2":
            publication_time = previous_day.replace(hour=22, minute=20)
        elif market_name == "IDA3":
            publication_time = delivery_day.replace(hour=10, minute=20)
        else:
            raise ValueError(f"Invalid market name: {market_name}")

        if simulation_step >= publication_time:
            return True
        else:
            return False
        
    
    def predict_market_prices(self, market_name, simulation_step, forecast_timestamps):
        if self.perfect_DA_forecast or market_name != "DA":
            ### For now, perfect forecast
            if market_name == "DA":
                price_series = self.DA_prices_df
            elif market_name == "IDA1":
                price_series = self.IDA1_prices_df
            elif market_name == "IDA2":
                price_series = self.IDA2_prices_df
            elif market_name == "IDA3":
                price_series = self.IDA3_prices_df
        
            return price_series.reindex(forecast_timestamps)
        
        else: 
            
            epexpredictor_forecasts_df_filtered = self.epexpredictor_forecasts_df.loc[(self.epexpredictor_forecasts_df["model_origin"] <= simulation_step) & (self.epexpredictor_forecasts_df["forecast_origin"] <= simulation_step)]

            if epexpredictor_forecasts_df_filtered.empty:
                raise ValueError(f"No EPEX predictor forecasts available for simulation step {simulation_step}.")
            
            last_model_origin = epexpredictor_forecasts_df_filtered["model_origin"].max()
            epexpredictor_forecasts_df_filtered = epexpredictor_forecasts_df_filtered.loc[epexpredictor_forecasts_df_filtered["model_origin"] == last_model_origin].copy()

            last_forecast_origin = epexpredictor_forecasts_df_filtered["forecast_origin"].max()
            epexpredictor_forecasts_df_filtered = epexpredictor_forecasts_df_filtered.loc[epexpredictor_forecasts_df_filtered["forecast_origin"] == last_forecast_origin].copy()

            if epexpredictor_forecasts_df_filtered["timestamp"].duplicated().any(): ### Shouldn't be any duplicates, if this flags, implement drop_duplicates last
                raise ValueError(f"Duplicated timestamps found in EPEX predictor forecasts for simulation step {simulation_step}.")
            
            epexpredictor_forecasts_df_filtered = epexpredictor_forecasts_df_filtered.set_index("timestamp").sort_index()

            epexpr_forecast_df = pd.DataFrame(index=forecast_timestamps)
            epexpr_forecast_df["price_eur_per_kwh"] = epexpredictor_forecasts_df_filtered["price_forecast_eur_mwh"].reindex(forecast_timestamps) / 1000.0
            if epexpr_forecast_df["price_eur_per_kwh"].isna().any():
                raise ValueError(f"NaN values found in EPEX predictor price forecasts for simulation step {simulation_step}.")
            
            return epexpr_forecast_df

            
            













if __name__ == "__main__":
    HOUSEHOLD = 301
    predict_dt = pd.Timestamp("2025-09-01 00:00:00", tz="Europe/Brussels") ### This should run from on 01/09/2025 until today more or less. However, load is assumed to be ported to this year from 2024.


    ### create class instance
    
    # Should review the storage required by the predictions file


    ### Load the PV forecaster params


    # Exclude before was #exclude_before_ts = "2024-01-20 00:00:00" but in timestamps json was from on the 22nd --> Run from on 22nd of January 2024
    General_framework = Framework(virtual_household=HOUSEHOLD, exclude_before_ts="2024-01-22 00:00:00")
    
    #predict_dt_to_2024 = (predict_dt - pd.Timedelta(General_framework.load_forecaster_params.period)).replace(year=2024)
    #_, _, scenarios_df, current_forecast_df, target_df = General_framework.load_residual_sampler.get_scenarios(predict_dt_to_2024, amount_analogs=10)
    #scenario_df_with_satellite, candidate_scores_df = General_framework.pv_residual_sampler.run_for_timestep(predict_dt, verbose=1, only_first_point=False)


    simulation_step = pd.Timestamp("2025-09-01 00:00:00", tz="Europe/Brussels")
    nodes = General_framework.build_scenario_tree(
    simulation_step=simulation_step,
    branch_hours_utc=[0, 6, 12, 18],
    scenario_counts=[(4, 4), (4, 4), (3, 3), (0, 0), (0, 0)],
    include_deterministic_load=True,
    include_deterministic_pv=True, 
    extend_last_nonzero_branch_to_horizon_end=False
    )

    tree_data = General_framework.nodes_to_pyomo_per_stage(nodes)
    tree_data_with_prices = General_framework.add_market_prices_to_tree(tree_data, simulation_step)