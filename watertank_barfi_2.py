# -*- coding: utf-8 -*-
"""
Created on Thu May 29 21:15:04 2025

@author: marti
"""

# -*- coding: utf-8 -*-
"""
Thermal Storage Tank Simulation - Simplified Version
"""

import streamlit as st
from barfi.flow import Block, ComputeEngine
from barfi.flow.streamlit import st_flow
import numpy as np
import pandas as pd
import altair as alt
from Watertank import ThermalStorageTank



# Title and description
st.title("Thermal Storage Tank Simulation")
st.markdown("Interactive simulation of a stratified thermal storage tank with inlet flows.")

# 1. Tank Parameters Block
tank_block = Block(name="Tank Parameters")
tank_block.add_output(name="tank_params_out")
tank_block.add_option(name="tank_height", type="number", value=4.0, label="Tank Height (m)")
tank_block.add_option(name="tank_diameter", type="number", value=2.0, label="Tank Diameter (m)")
tank_block.add_option(name="num_nodes", type="number", value=50, label="Number of Nodes")
tank_block.add_option(name="initial_temp", type="number", value=20.0, label="Initial Temp (°C)")

def tank_block_func(self):
    params = {
        'tank_height': self.get_option("tank_height"),
        'tank_diameter': self.get_option("tank_diameter"),
        'num_nodes': int(self.get_option("num_nodes")),
        'initial_temp': self.get_option("initial_temp")
    }
    self.set_interface(name="tank_params_out", value=params)

tank_block.add_compute(tank_block_func)

# 2. Flow Configuration Block
flow_block = Block(name="Flow Configuration")
flow_block.add_output(name="flow_config_out")
flow_block.add_option(name="hot_inlet_height", type="number", value=4.0, label="Hot Inlet Height (m)")
flow_block.add_option(name="hot_inlet_flow", type="number", value=10.0, label="Hot Flow Rate (kg/s)")
flow_block.add_option(name="hot_inlet_temp", type="number", value=90.0, label="Hot Inlet Temp (°C)")
flow_block.add_option(name="cold_inlet_height", type="number", value=0.0, label="Cold Inlet Height (m)")
flow_block.add_option(name="cold_inlet_flow", type="number", value=5.0, label="Cold Flow Rate (kg/s)")
flow_block.add_option(name="cold_inlet_temp", type="number", value=70.0, label="Cold Inlet Temp (°C)")
flow_block.add_option(name="outlet_height", type="number", value=2.0, label="Outlet Height (m)")
flow_block.add_option(name="outlet_flow", type="number", value=15.0, label="Outlet Flow Rate (kg/s)")

def flow_block_func(self):
    config = {
        'hot_inlet_height': self.get_option("hot_inlet_height"),
        'hot_inlet_flow': self.get_option("hot_inlet_flow"),
        'hot_inlet_temp': self.get_option("hot_inlet_temp"),
        'cold_inlet_height': self.get_option("cold_inlet_height"),
        'cold_inlet_flow': self.get_option("cold_inlet_flow"),
        'cold_inlet_temp': self.get_option("cold_inlet_temp"),
        'outlet_height': self.get_option("outlet_height"),
        'outlet_flow': self.get_option("outlet_flow")
    }
    self.set_interface(name="flow_config_out", value=config)

flow_block.add_compute(flow_block_func)

# 3. Simulation Block
sim_block = Block(name="Run Simulation")
sim_block.add_input(name="tank_params_in")
sim_block.add_input(name="flow_config_in")
sim_block.add_output(name="results_out")

def sim_block_func(self):
    try:
        # Get all inputs
        tank_params = self.get_interface("tank_params_in")
        flow_config = self.get_interface("flow_config_in")

        # Create tank parameters dictionary
        params = {
            'tank_height': tank_params['tank_height'],
            'tank_diameter': tank_params['tank_diameter'],
            'C_fl': 4186,  # Fixed water heat capacity
            'k_fl': 0.6,   # Fixed thermal conductivity
            'delta_k_eff': 0.0,
            'UA_i': 0.0,
            'UA_gfl': 0.0,
            'epsilon': 0.5,
            'T_env': 20 + 273.15,
            'T_gfl': 15 + 273.15,
            'T_initial': tank_params['initial_temp'],
            'heat_exchangers': []  # No heat exchangers in this simplified version
        }

        # Create tank instance
        tank = ThermalStorageTank(tank_params['num_nodes'], params)

        # Set up flow specs
        flow_specs = {
            'inlets': [
                (flow_config['hot_inlet_height'], flow_config['hot_inlet_flow'], flow_config['hot_inlet_temp'], "Hot Inlet"),
                (flow_config['cold_inlet_height'], flow_config['cold_inlet_flow'], flow_config['cold_inlet_temp'], "Cold Inlet")
            ],
            'outlets': [
                (flow_config['outlet_height'], flow_config['outlet_flow'], "Mixed Outlet")
            ]
        }

        # Run simulation
        t_span = np.linspace(0, 3600, 100)
        solution = tank.solve(t_span, flow_specs)

        # Prepare results
        final_temps = solution[-1, :] - 273.15  # Convert to °C
        node_heights = np.linspace(tank_params['tank_height'], 0, tank_params['num_nodes'])

        # Calculate outlet temperature
        outlet_node = tank.get_node_at_height(flow_config['outlet_height'])
        outlet_temp = final_temps[outlet_node]

        # Calculate flow distribution at outlet
        node_flow = {'flow_in': 0.0, 'flow_out': 0.0, 'T_in': 0.0}
        for height, flow_rate, temp, name in flow_specs['inlets']:
            if outlet_node == tank.get_node_at_height(height):
                node_flow['flow_in'] += flow_rate
                node_flow['T_in'] = temp
        for height, flow_rate, name in flow_specs['outlets']:
            if outlet_node == tank.get_node_at_height(height):
                node_flow['flow_out'] += flow_rate

        flows = tank.calculate_flows(node_flow, outlet_node)

        results = {
            "node_heights": node_heights,
            "final_temps": final_temps,
            "outlet_info": {
                "height": flow_config['outlet_height'],
                "temperature": outlet_temp,
                "flow_rate": flow_config['outlet_flow'],
                "upward_flow": flows['FL6'],
                "downward_flow": flows['FL4'],
                "net_flow": flows['FL8']
            }
        }

        self.set_interface(name="results_out", value=results)

    except Exception as e:
        st.error(f"Simulation failed: {e}")

sim_block.add_compute(sim_block_func)

# 4. Results Display Block
results_block = Block(name="Results Display")
results_block.add_input(name="results_in")

def results_block_func(self):
    results = self.get_interface("results_in")
    if results:
        # Create a DataFrame for the temperature profile
        df = pd.DataFrame({
            'Height (m)': results['node_heights'],
            'Temperature (°C)': results['final_temps']
        })
        
        # Plot temperature profile
        chart = alt.Chart(df).mark_line().encode(
            x='Temperature (°C)',
            y='Height (m)',
            tooltip=['Height (m)', 'Temperature (°C)']
        ).properties(
            width=600,
            height=400,
            title="Temperature Profile in Tank"
        )
        
        st.altair_chart(chart, use_container_width=True)
        
        # Display outlet information
        st.subheader("Outlet Information")
        col1, col2, col3 = st.columns(3)
        col1.metric("Outlet Temperature", f"{results['outlet_info']['temperature']:.2f} °C")
        col2.metric("Specified Flow Rate", f"{results['outlet_info']['flow_rate']:.2f} kg/s")
        col3.metric("Height Position", f"{results['outlet_info']['height']:.2f} m")
        
        # Display flow distribution
        st.subheader("Flow Distribution at Outlet Node")
        col1, col2, col3 = st.columns(3)
        col1.metric("Upward Flow", f"{results['outlet_info']['upward_flow']:.2f} kg/s")
        col2.metric("Downward Flow", f"{results['outlet_info']['downward_flow']:.2f} kg/s")
        col3.metric("Net Flow", f"{results['outlet_info']['net_flow']:.2f} kg/s")

results_block.add_compute(results_block_func)

# Create the flow with all blocks
base_blocks = [tank_block, flow_block, sim_block, results_block]
barfi_result = st_flow(base_blocks)

# Initialize compute engine and execute the schema
compute_engine = ComputeEngine(base_blocks)
if barfi_result.editor_schema:
    compute_engine.execute(barfi_result.editor_schema)
