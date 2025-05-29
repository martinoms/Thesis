# -*- coding: utf-8 -*-
"""
Enhanced Thermal Storage Tank Simulation with Visible Inputs/Options
"""

import streamlit as st
from barfi.flow import Block, ComputeEngine
from barfi.flow.streamlit import st_flow
import numpy as np
import pandas as pd
import altair as alt
from Watertank import ThermalStorageTank
import json

# Set page configuration
st.set_page_config(page_title="Thermal Tank Flow Sim", layout="wide")

# Title and description
st.title("Thermal Tank Flow Simulation")
st.markdown("Connect flow inputs to tank nodes and visualize the results")

# 1. Flow Block (can be connected to tank inputs)
flow_block = Block(name="Flow Input")
flow_block.add_output(name="flow_out")
flow_block.add_option(name="flow_temp", type="input", value="80.0", label="Temperature (°C)")
flow_block.add_option(name="flow_rate", type="input", value="5.0", label="Flow Rate (kg/s)")
flow_block.add_option(name="flow_name", type="input", value="Flow", label="Flow Name")
flow_block.add_option(name="flow_height", type="input", value="2.0", label="Height (m)")

def flow_block_func(self):
    flow_data = {
        'temperature': float(self.get_option("flow_temp")),
        'flow_rate': float(self.get_option("flow_rate")),
        'name': str(self.get_option("flow_name")),
        'height': float(self.get_option("flow_height"))
    }
    self.set_interface(name="flow_out", value=flow_data)

flow_block.add_compute(flow_block_func)

# 2. Tank Block
tank_block = Block(name="Tank")
tank_block.add_option(name="tank_height", type="input", value="4.0", label="Height (m)")
tank_block.add_option(name="tank_diameter", type="input", value="2.0", label="Diameter (m)")
tank_block.add_option(name="num_nodes", type="input", value="20", label="Number of Nodes")
tank_block.add_option(name="num_inputs", type="input", value="2", label="Number of Inputs")
tank_block.add_option(name="num_outputs", type="input", value="1", label="Number of Outputs")
tank_block.add_output(name="tank_out")

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

    num_inputs = int(float(self.get_option("num_inputs")))
    for i in range(num_inputs):
        input_name = f"flow_in_{i}"
        if self.get_interface(input_name):
            flow_data = self.get_interface(input_name)
            height = flow_data.get('height', params['tank_height'] * (i + 1) / (num_inputs + 1))
            inlets.append((height, flow_data['flow_rate'], flow_data['temperature'], flow_data['name']))

    num_outputs = int(float(self.get_option("num_outputs")))
    for i in range(num_outputs):
        output_name = f"flow_out_{i}"
        if self.get_interface(output_name):
            flow_data = self.get_interface(output_name)
            height = flow_data.get('height', params['tank_height'] * (i + 1) / (num_outputs + 1))
            outlets.append((height, flow_data['flow_rate'], flow_data['name']))

    if not outlets:
        outlets.append((params['tank_height']/2, sum(f[1] for f in inlets) if inlets else 5.0, "Default Outlet"))

    t_span = np.linspace(0, 3600, 100)
    flow_specs = {'inlets': inlets, 'outlets': outlets}
    solution = tank.solve(t_span, flow_specs)

    final_temps = solution[-1, :] - 273.15
    node_heights = np.linspace(params['tank_height'], 0, params['num_nodes'])

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
        "inlets": inlets,
        "params": params
    }

    self.set_interface(name="tank_out", value=results)

tank_block.add_compute(tank_block_func)

# 3. Results Block
results_block = Block(name="Results")
results_block.add_input(name="results_in")

def results_block_func(self):
    results = self.get_interface("results_in")
    if results:
        with st.expander("Tank Configuration", expanded=True):
            cols = st.columns(3)
            cols[0].metric("Height", f"{results['params']['tank_height']} m")
            cols[1].metric("Diameter", f"{results['params']['tank_diameter']} m")
            cols[2].metric("Nodes", results['params']['num_nodes'])

        st.subheader("Temperature Profile")
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
            height=400
        )
        st.altair_chart(chart, use_container_width=True)

        st.subheader("Flow Connections")
        with st.expander("Input Flows", expanded=True):
            if results["inlets"]:
                for i, inlet in enumerate(results["inlets"]):
                    cols = st.columns(3)
                    cols[0].metric(f"Input {i+1} Name", inlet[3])
                    cols[1].metric("Flow Rate", f"{inlet[1]} kg/s")
                    cols[2].metric("Temperature", f"{inlet[2]}°C")
                    st.caption(f"Height: {inlet[0]:.2f} m")
            else:
                st.warning("No input flows connected")

        with st.expander("Output Flows", expanded=True):
            if results["outlets"]:
                for i, outlet in enumerate(results["outlets"]):
                    cols = st.columns(3)
                    cols[0].metric(f"Output {i+1} Name", outlet['name'])
                    cols[1].metric("Flow Rate", f"{outlet['flow_rate']} kg/s")
                    cols[2].metric("Temperature", f"{outlet['temperature']:.1f}°C")
                    st.caption(f"Height: {outlet['height']:.2f} m")
            else:
                st.warning("No output flows connected")

results_block.add_compute(results_block_func)

base_blocks = [flow_block, tank_block, results_block]
barfi_result = st_flow(base_blocks)

compute_engine = ComputeEngine(base_blocks)

if barfi_result.editor_schema:
    compute_engine.execute(barfi_result.editor_schema)

st.sidebar.header("Current Connections")
if barfi_result.editor_schema:
    for connection in barfi_result.editor_schema['connections']:
        source_block = barfi_result.editor_schema['blocks'][connection['source_id']]['block_name']
        source_interface = connection['source_interface']
        target_block = barfi_result.editor_schema['blocks'][connection['target_id']]['block_name']
        target_interface = connection['target_interface']
        st.sidebar.write(f"🔗 {source_block} ({source_interface}) → {target_block} ({target_interface})")
else:
    st.sidebar.write("No connections yet")
