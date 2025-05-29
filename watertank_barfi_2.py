# -*- coding: utf-8 -*-
"""
Created on Tue May 27 22:17:16 2025

@author: marti
"""

import streamlit as st
from barfi import Block, ComputeEngine, SchemaManager
from barfi.streamlit import st_flow
import numpy as np
import pandas as pd
import altair as alt
from Watertank import ThermalStorageTank  # Ensure this module is correctly implemented

# Set page configuration (only call this once)
st.set_page_config(page_title="Thermal Storage Tank Simulation", layout="wide")

# Title and description
st.title("Thermal Storage Tank Simulation")
st.markdown("Interactive simulation of a stratified thermal storage tank with heat exchangers.")

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

# 2. Fluid Properties Block
fluid_block = Block(name="Fluid Properties")
fluid_block.add_output(name="fluid_props_out")
fluid_block.add_option(name="Cp", type="number", value=4186.0, label="Heat Capacity (J/kg·K)")
fluid_block.add_option(name="k_fl", type="number", value=0.6, label="Thermal Conductivity (W/m·K)")
fluid_block.add_option(name="delta_k_eff", type="number", value=0.0, label="Enhanced Conductivity (W/m·K)")

def fluid_block_func(self):
    props = {
        'Cp': self.get_option("Cp"),
        'k_fl': self.get_option("k_fl"),
        'delta_k_eff': self.get_option("delta_k_eff")
    }
    self.set_interface(name="fluid_props_out", value=props)

fluid_block.add_compute(fluid_block_func)

# 3. Heat Exchanger Block
hx_block = Block(name="Heat Exchanger")
hx_block.add_output(name="hx_config_out")
hx_block.add_option(name="hx_type", type="select", items=["Tube", "Plate"], value="Plate", label="HX Type")
hx_block.add_option(name="hx_height", type="number", value=1.0, label="Height Position (m)")
hx_block.add_option(name="U_value", type="number", value=3000.0, label="U Value (W/m²·K)")
hx_block.add_option(name="m_dot_secondary", type="number", value=10.0, label="Secondary Flow (kg/s)")
hx_block.add_option(name="T_in_secondary", type="number", value=90.0, label="Secondary Inlet Temp (°C)")

def hx_block_func(self):
    config = {
        'hx_type': self.get_option("hx_type"),
        'hx_height': self.get_option("hx_height"),
        'U_value': self.get_option("U_value"),
        'm_dot_secondary': self.get_option("m_dot_secondary"),
        'T_in_secondary': self.get_option("T_in_secondary")
    }
    self.set_interface(name="hx_config_out", value=config)

hx_block.add_compute(hx_block_func)

# 4. Flow Configuration Block
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

# 5. Simulation Block
sim_block = Block(name="Run Simulation")
sim_block.add_input(name="tank_params_in")
sim_block.add_input(name="fluid_props_in")
sim_block.add_input(name="hx_config_in")
sim_block.add_input(name="flow_config_in")
sim_block.add_output(name="results_out")

def sim_block_func(self):
    try:
        # Get all inputs
        tank_params = self.get_interface("tank_params_in")
        fluid_props = self.get_interface("fluid_props_in")
        hx_config = self.get_interface("hx_config_in")
        flow_config = self.get_interface("flow_config_in")

        # Create tank parameters dictionary
        params = {
            'tank_height': tank_params['tank_height'],
            'tank_diameter': tank_params['tank_diameter'],
            'C_fl': fluid_props['Cp'],
            'k_fl': fluid_props['k_fl'],
            'delta_k_eff': fluid_props['delta_k_eff'],
            'UA_i': 0.0,
            'UA_gfl': 0.0,
            'epsilon': 0.5,
            'T_env': 20 + 273.15,
            'T_gfl': 15 + 273.15,
            'T_initial': tank_params['initial_temp'],
            'heat_exchangers': [{
                'type': hx_config['hx_type'].lower(),
                'height': hx_config['hx_height'],
                'area': 2.5 if hx_config['hx_type'] == "Plate" else None,
                'length': 0.2 if hx_config['hx_type'] == "Tube" else None,
                'diameter': 0.05 if hx_config['hx_type'] == "Tube" else None,
                'num_tubes': 40 if hx_config['hx_type'] == "Tube" else None,
                'U': hx_config['U_value'],
                'fluid': 'secondary',
                'm_dot_secondary': hx_config['m_dot_secondary'],
                'Cp_secondary': fluid_props['Cp'],
                'T_in_HE': hx_config['T_in_secondary'],
                'flow_arrangement': 'counterflow'
            }]
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

        # Calculate HX performance
        hx_node = tank.get_node_at_height(hx_config['hx_height'])
        flows = tank.calculate_flows({
            'flow_in': sum(f[1] for f in flow_specs['inlets'] if int(tank.get_node_at_height(f[0])) == hx_node),
            'flow_out': sum(f[1] for f in flow_specs['outlets'] if int(tank.get_node_at_height(f[0])) == hx_node),
            'T_in': 0  # Dummy, you can refine this if needed
        }, hx_node)

        m_dot_local = abs(flows['FL4'] - flows['FL6'])
        Q_hx, T_out = tank.heat_exchangers[0]['hx'].calculate_heat_transfer(solution[-1, hx_node], m_dot_local)

        results = {
            "node_heights": node_heights,
            "final_temps": final_temps,
            "hx_performance": {
                "heat_transfer": Q_hx / 1000,  # kW
                "outlet_temp": T_out - 273.15,  # °C
                "node_temp": final_temps[hx_node]
            },
            "flow_rates": {
                "upward": flows['FL6'],
                "downward": flows['FL4'],
                "net": flows['FL8']
            }
        }

        self.set_interface(name="results_out", value=results)

    except Exception as e:
        st.error(f"Simulation failed: {e}")

sim_block.add_compute(sim_block_func)

# 6. Results Display Block
results_block = Block(name="Results Display")
results_block.add_input(name="results_in")
results_block.add_option(name="display-option", type="display", value="Simulation Results")

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
        
        # Display HX performance
        st.subheader("Heat Exchanger Performance")
        col1, col2, col3 = st.columns(3)
        col1.metric("Heat Transfer Rate", f"{results['hx_performance']['heat_transfer']:.2f} kW")
        col2.metric("Outlet Temperature", f"{results['hx_performance']['outlet_temp']:.2f} °C")
        col3.metric("Node Temperature", f"{results['hx_performance']['node_temp']:.2f} °C")
        
        # Display flow rates
        st.subheader("Flow Rates at HX Node")
        col1, col2, col3 = st.columns(3)
        col1.metric("Upward Flow", f"{results['flow_rates']['upward']:.2f} kg/s")
        col2.metric("Downward Flow", f"{results['flow_rates']['downward']:.2f} kg/s")
        col3.metric("Net Flow", f"{results['flow_rates']['net']:.2f} kg/s")

results_block.add_compute(results_block_func)

# Create the flow with all blocks
base_blocks = [tank_block, fluid_block, hx_block, flow_block, sim_block, results_block]
barfi_result = st_flow(base_blocks)

# Initialize compute engine and execute the schema
compute_engine = ComputeEngine(base_blocks)
if barfi_result.editor_schema:
    compute_engine.execute(barfi_result.editor_schema)
