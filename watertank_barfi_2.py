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
st.markdown("Configure flow connections and simulate tank behavior")

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
    tank_block.add_input(name=f"flow in {i}")

# Add dynamic (optional) outputs for completeness
for j in range(num_outputs):
    tank_block.add_output(name=f"flow out {j}")

# Tank compute function
from barfi import Block
import numpy as np
from your_module import ThermalStorageTank  # Ensure correct import

def create_tank_block(num_inputs, num_outputs):
    tank_block = Block(name="Tank")

    # User-configurable tank properties
    tank_block.add_option("tank_height", type="input", value="4.0", label="Tank Height (m)")
    tank_block.add_option("tank_diameter", type="input", value="2.0", label="Tank Diameter (m)")
    tank_block.add_option("num_nodes", type="input", value="50", label="Number of Nodes")
    tank_block.add_option("initial_temp", type="input", value="92", label="Initial Temperature (°C)")

    # Fixed advanced parameters (can also be converted to options if needed)
    advanced_params = {
        'C_fl': 4186,
        'k_fl': 0.0,
        'delta_k_eff': 0.0,
        'UA_i': 0.0,
        'UA_gfl': 0.0,
        'epsilon': 0.5,
        'T_env': 293.15,   # 20°C
        'T_gfl': 288.15,   # 15°C
    }

    # Outputs
    tank_block.add_output(name="tank_out")

    # Dynamic flow inputs and outputs
    for i in range(num_inputs):
        tank_block.add_input(name=f"flow_in_{i}")

    for j in range(num_outputs):
        tank_block.add_output(name=f"flow_out_{j}")

    def tank_block_func(self):
        # Gather options
        H = float(self.get_option("tank_height"))
        D = float(self.get_option("tank_diameter"))
        N = int(float(self.get_option("num_nodes")))
        T_init = float(self.get_option("initial_temp")) + 273.15

        # Build parameter dict
        params = {
            'tank_height': H,
            'tank_diameter': D,
            'T_initial': T_init,
            'heat_exchangers': [
                {
                    'type': 'plate',
                    'height': 1.0,
                    'area': 2.5,
                    'U': 3000,
                    'fluid': 'secondary',
                    'm_dot_secondary': 10,
                    'Cp_secondary': 4186,
                    'T_in_HE': 90,
                    'flow_arrangement': 'counterflow'
                }
            ],
            **advanced_params
        }

        tank = ThermalStorageTank(N, params)

        # Build inlets
        inlets = []
        for i in range(num_inputs):
            key = f"flow_in_{i}"
            interface = self.get_interface(key)
            if interface:
                height = H * (i + 1) / (num_inputs + 1)
                inlets.append((height, interface['flow_rate'], interface['temperature'], interface['name']))

        # Build outlets
        outlets = []
        if inlets:
            total_flow = sum(f[1] for f in inlets)
        else:
            total_flow = 5.0 * num_outputs  # default

        for j in range(num_outputs):
            height = H * (j + 1) / (num_outputs + 1)
            outlets.append((height, total_flow / num_outputs, f"Outlet {j+1}"))

        # Time simulation
        t_span = np.linspace(0, 3600, 100)
        solution = tank.solve(t_span, {'inlets': inlets, 'outlets': outlets})
        final_temps = solution[-1, :] - 273.15
        node_heights = np.linspace(H, 0, N)

        outlet_temps = []
        for height, flow_rate, name in outlets:
            idx = tank.get_node_at_height(height)
            outlet_temps.append({
                'name': name,
                'height': height,
                'temperature': final_temps[idx],
                'flow_rate': flow_rate
            })

        # Output full simulation results
        self.set_interface("tank_out", {
            "node_heights": node_heights,
            "temperatures": final_temps,
            "outlets": outlet_temps,
            "inlets": inlets,
            "params": params
        })

    tank_block.add_compute(tank_block_func)
    return tank_block


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
