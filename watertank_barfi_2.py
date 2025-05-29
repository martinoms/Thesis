# -*- coding: utf-8 -*-
"""
Thermal Tank Flow Simulation with Dynamic Flow Connections
"""

import streamlit as st
from barfi.flow import Block, ComputeEngine
from barfi.flow.streamlit import st_flow
import numpy as np
import pandas as pd
import altair as alt
from Watertank import ThermalStorageTank

# Set page config
st.set_page_config(page_title="Thermal Tank Flow Sim", layout="wide")

st.title("Thermal Tank Flow Simulation")
st.markdown("Configure flow connections and simulate stratified tank behavior")

# Tank configuration options
st.sidebar.header("Tank Configuration")
tank_height = st.sidebar.number_input("Tank Height (m)", value=4.0)
tank_diameter = st.sidebar.number_input("Tank Diameter (m)", value=2.0)
num_nodes = st.sidebar.number_input("Number of Nodes", min_value=5, max_value=100, value=20)
num_inputs = st.sidebar.number_input("Number of Flow Inputs", min_value=1, max_value=5, value=2)
num_outputs = st.sidebar.number_input("Number of Flow Outputs", min_value=1, max_value=5, value=1)

# Flow Block factory
def create_flow_block(i):
    fb = Block(name=f"Flow Input {i+1}")
    fb.add_output(name="flow_out")
    fb.add_option(name="flow_temp", type="input", value="80.0", label="Temperature (°C)")
    fb.add_option(name="flow_rate", type="input", value="5.0", label="Flow Rate (kg/s)")
    fb.add_option(name="flow_name", type="input", value=f"Flow {i+1}", label="Flow Name")

    def fb_func(self):
        self.set_interface("flow_out", {
            'temperature': float(self.get_option("flow_temp")),
            'flow_rate': float(self.get_option("flow_rate")),
            'name': self.get_option("flow_name")
        })

    fb.add_compute(fb_func)
    return fb

# Create multiple flow blocks
flow_blocks = [create_flow_block(i) for i in range(num_inputs)]

# Tank Block

tank_block = Block(name="Tank")
tank_block.add_option(name="tank_height", type="input", value=str(tank_height), label="Tank Height")
tank_block.add_option(name="tank_diameter", type="input", value=str(tank_diameter), label="Tank Diameter")
tank_block.add_option(name="num_nodes", type="input", value=str(num_nodes), label="Nodes")

# Add tank output
tank_block.add_output(name="tank_out")

# Add dynamic inputs to tank
for i in range(num_inputs):
    tank_block.add_input(name=f"flow_in_{i}")

# Add dynamic (optional) outputs for completeness
for j in range(num_outputs):
    tank_block.add_output(name=f"flow_out_{j}")

# Tank compute function
def tank_block_func(self):
    params = {
        'tank_height': float(self.get_option("tank_height")),
        'tank_diameter': float(self.get_option("tank_diameter")),
        'num_nodes': int(float(self.get_option("num_nodes"))),
        'initial_temp': 20.0,
        'C_fl': 4186,
        'k_fl': 0.6
    }

    tank = ThermalStorageTank(params['num_nodes'], params)
    inlets = []
    outlets = []

    for i in range(num_inputs):
        input_name = f"flow_in_{i}"
        if self.get_interface(input_name):
            data = self.get_interface(input_name)
            height = params['tank_height'] * (i + 1) / (num_inputs + 1)
            inlets.append((height, data['flow_rate'], data['temperature'], data['name']))

    for j in range(num_outputs):
        height = params['tank_height'] * (j + 1) / (num_outputs + 1)
        flow_rate = sum(f[1] for f in inlets) / num_outputs if inlets else 5.0
        outlets.append((height, flow_rate, f"Outlet {j+1}"))

    t_span = np.linspace(0, 3600, 100)
    solution = tank.solve(t_span, {'inlets': inlets, 'outlets': outlets})
    final_temps = solution[-1, :] - 273.15
    node_heights = np.linspace(params['tank_height'], 0, params['num_nodes'])

    outlet_temps = []
    for height, flow_rate, name in outlets:
        idx = tank.get_node_at_height(height)
        outlet_temps.append({
            'name': name,
            'height': height,
            'temperature': final_temps[idx],
            'flow_rate': flow_rate
        })

    self.set_interface("tank_out", {
        "node_heights": node_heights,
        "temperatures": final_temps,
        "outlets": outlet_temps,
        "inlets": inlets,
        "params": params
    })

tank_block.add_compute(tank_block_func)

# Results Block
results_block = Block(name="Results")
results_block.add_input(name="results_in")

def results_block_func(self):
    results = self.get_interface("results_in")
    if not results:
        return

    st.subheader("Temperature Profile")
    df = pd.DataFrame({
        'Height (m)': results['node_heights'],
        'Temperature (°C)': results['temperatures']
    })

    chart = alt.Chart(df).mark_line().encode(
        x='Temperature (°C)',
        y='Height (m)',
        tooltip=['Height (m)', 'Temperature (°C)']
    ).properties(width=600, height=400)

    st.altair_chart(chart, use_container_width=True)

    st.subheader("Inlets")
    for inlet in results['inlets']:
        st.write(f"🌡️ {inlet[3]} — {inlet[2]}°C at {inlet[1]} kg/s (Height: {inlet[0]:.2f} m)")

    st.subheader("Outlets")
    for outlet in results['outlets']:
        st.write(f"💧 {outlet['name']} — {outlet['temperature']:.1f}°C at {outlet['flow_rate']} kg/s (Height: {outlet['height']:.2f} m)")

results_block.add_compute(results_block_func)

# Create final flow
flow = st_flow(flow_blocks + [tank_block, results_block])

# Compute
engine = ComputeEngine(flow_blocks + [tank_block, results_block])
if flow and flow.schema:
    engine.execute(flow.schema)
