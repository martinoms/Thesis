# -*- coding: utf-8 -*-
"""
Simplified Thermal Storage Tank Simulation
"""

import streamlit as st
from barfi.flow import Block, ComputeEngine
from barfi.flow.streamlit import st_flow
import numpy as np
import pandas as pd
import altair as alt
from Watertank import ThermalStorageTank

# Set page configuration
st.set_page_config(page_title="Thermal Tank Flow Sim", layout="wide")

# Title and description
st.title("Thermal Tank Flow Simulation")
st.markdown("Connect flow inputs to tank nodes and visualize the results")

# 1. Flow Block (can be connected to tank inputs)
flow_block = Block(name="Flow Input")
flow_block.add_output(name="flow_out")  # This will connect to tank inputs
flow_block.add_option(name="flow_temp", type="input", value=80.0, label="Temperature (°C)")
flow_block.add_option(name="flow_rate", type="input", value=5.0, label="Flow Rate (kg/s)")
# Using 'input' type for text input since 'text' type isn't available
flow_block.add_option(name="flow_name", type="input", value="Flow", label="Flow Name")

def flow_block_func(self):
    flow_data = {
        'temperature': float(self.get_option("flow_temp")),
        'flow_rate': float(self.get_option("flow_rate")),
        'name': str(self.get_option("flow_name"))
    }
    self.set_interface(name="flow_out", value=flow_data)

flow_block.add_compute(flow_block_func)

# 2. Tank Block (with configurable inputs/outputs)
tank_block = Block(name="Tank")
tank_block.add_option(name="tank_height", type="input", value=4.0, label="Height (m)")
tank_block.add_option(name="tank_diameter", type="input", value=2.0, label="Diameter (m)")
tank_block.add_option(name="num_nodes", type="input", value=20, label="Number of Nodes")
tank_block.add_option(name="num_inputs", type="input", value=2, label="Number of Inputs")
tank_block.add_option(name="num_outputs", type="input", value=1, label="Number of Outputs")

# Dynamic inputs/outputs will be added based on the options
tank_block.add_output(name="tank_out")  # Main tank output

def tank_block_func(self):
    # Get tank parameters
    params = {
        'tank_height': float(self.get_option("tank_height")),
        'tank_diameter': float(self.get_option("tank_diameter")),
        'num_nodes': int(float(self.get_option("num_nodes"))),
        'initial_temp': 20.0,  # Default initial temperature
        'C_fl': 4186,  # Water heat capacity
        'k_fl': 0.6    # Thermal conductivity
    }
    
    # Create tank instance
    tank = ThermalStorageTank(params['num_nodes'], params)
    
    # Collect all connected flows
    inlets = []
    outlets = []
    
    # Get dynamic inputs (added by user)
    num_inputs = int(float(self.get_option("num_inputs")))
    for i in range(num_inputs):
        input_name = f"flow_in_{i}"
        if self.get_interface(input_name):
            flow_data = self.get_interface(input_name)
            # For simplicity, place inputs at evenly spaced heights
            height = params['tank_height'] * (i + 1) / (num_inputs + 1)
            inlets.append((height, flow_data['flow_rate'], flow_data['temperature'], flow_data['name']))
    
    # Get dynamic outputs (added by user)
    num_outputs = int(float(self.get_option("num_outputs")))
    for i in range(num_outputs):
        output_name = f"flow_out_{i}"
        if self.get_interface(output_name):
            flow_data = self.get_interface(output_name)
            # For simplicity, place outputs at evenly spaced heights
            height = params['tank_height'] * (i + 1) / (num_outputs + 1)
            outlets.append((height, flow_data['flow_rate'], flow_data['name']))
    
    # If no outputs connected, create a default one
    if not outlets:
        outlets.append((params['tank_height']/2, sum(f[1] for f in inlets) if inlets else 5.0, "Default Outlet"))
    
    # Run simulation
    t_span = np.linspace(0, 3600, 100)
    flow_specs = {'inlets': inlets, 'outlets': outlets}
    solution = tank.solve(t_span, flow_specs)
    
    # Prepare results
    final_temps = solution[-1, :] - 273.15  # Convert to °C
    node_heights = np.linspace(params['tank_height'], 0, params['num_nodes'])
    
    # Calculate outlet temperatures
    outlet_temps = []
    for height, flow_rate, name in outlets:
        node_idx = tank.get_node_at_height(height)
        outlet_temps.append({
            'name': name,
            'height': height,
            'temperature': final_temps[node_idx],
            'flow_rate': flow_rate
        })
    
    results = {
        "node_heights": node_heights,
        "temperatures": final_temps,
        "outlets": outlet_temps,
        "inlets": inlets
    }
    
    self.set_interface(name="tank_out", value=results)

tank_block.add_compute(tank_block_func)

# 3. Results Block
results_block = Block(name="Results")
results_block.add_input(name="results_in")

def results_block_func(self):
    results = self.get_interface("results_in")
    if results:
        # Temperature profile plot
        df = pd.DataFrame({
            'Height (m)': results["node_heights"],
            'Temperature (°C)': results["temperatures"]
        })
        
        chart = alt.Chart(df).mark_line().encode(
            x='Temperature (°C)',
            y='Height (m)',
            tooltip=['Height (m)', 'Temperature (°C)']
        ).properties(
            width=600,
            height=400,
            title="Temperature Profile"
        )
        st.altair_chart(chart, use_container_width=True)
        
        # Inlets/outlets information
        st.subheader("Flow Connections")
        
        cols = st.columns(2)
        with cols[0]:
            st.write("**Inlets**")
            for inlet in results["inlets"]:
                st.write(f"- {inlet[3]}: {inlet[1]} kg/s at {inlet[2]}°C (height: {inlet[0]:.1f}m)")
        
        with cols[1]:
            st.write("**Outlets**")
            for outlet in results["outlets"]:
                st.write(f"- {outlet['name']}: {outlet['flow_rate']} kg/s at {outlet['temperature']:.1f}°C (height: {outlet['height']:.1f}m)")

results_block.add_compute(results_block_func)

# Create the flow with all blocks
base_blocks = [flow_block, tank_block, results_block]
barfi_result = st_flow(base_blocks)

# Initialize compute engine
compute_engine = ComputeEngine(base_blocks)

# Function to dynamically add inputs/outputs to tank based on user selection
def update_tank_connections():
    if barfi_result.editor_schema:
        # Find the tank block in the schema
        for block_id, block_data in barfi_result.editor_schema['blocks'].items():
            if block_data['block_name'] == 'Tank':
                # Clear existing inputs/outputs (except main output)
                block_data['interfaces'] = {
                    'input': {},
                    'output': {'tank_out': {'name': 'tank_out'}}
                }
                
                # Add dynamic inputs
                num_inputs = block_data['options']['num_inputs']['value']
                for i in range(int(float(num_inputs))):
                    input_name = f"flow_in_{i}"
                    block_data['interfaces']['input'][input_name] = {'name': input_name}
                
                # Add dynamic outputs
                num_outputs = block_data['options']['num_outputs']['value']
                for i in range(int(float(num_outputs))):
                    output_name = f"flow_out_{i}"
                    block_data['interfaces']['output'][output_name] = {'name': output_name}
                
                break

# Update connections when options change
if barfi_result.editor_schema:
    update_tank_connections()
    compute_engine.execute(barfi_result.editor_schema)
