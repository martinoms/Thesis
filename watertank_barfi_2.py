import streamlit as st
from barfi.flow import Block, ComputeEngine
from barfi.flow.streamlit import st_flow
import numpy as np
import pandas as pd
import altair as alt
from Watertank import ThermalStorageTank

# Page setup
st.set_page_config(page_title="Thermal Tank Flow Sim", layout="wide")
st.title("Thermal Tank Flow Simulation")
st.markdown("Configure flow connections and simulate tank behavior")

# Sidebar config
st.sidebar.header("Tank Configuration")
tank_height = st.sidebar.number_input("Tank Height (m)", value=4.0)
tank_diameter = st.sidebar.number_input("Tank Diameter (m)", value=2.0)
num_nodes = st.sidebar.number_input("Number of Nodes", min_value=5, max_value=100, value=20)
num_inputs = st.sidebar.number_input("Number of Flow Inputs", min_value=1, max_value=5, value=2)
num_outputs = st.sidebar.number_input("Number of Flow Outputs", min_value=1, max_value=5, value=2)

# Flow block generator
def create_flow_block(i):
    fb = Block(name=f"Flow Input {i+1}")
    fb.add_output(name="flow_out")
    fb.add_option("flow_temp", type="input", value="80.0", label="Temperature (°C)")
    fb.add_option("flow_rate", type="input", value="5.0", label="Flow Rate (kg/s)")
    fb.add_option("flow_name", type="input", value=f"Flow {i+1}", label="Flow Name")
    fb.add_option("flow_height", type="input", value="1.0", label="Height (m)")

    def fb_func(self):
        self.set_interface("flow_out", {
            'temperature': float(self.get_option("flow_temp")),
            'flow_rate': float(self.get_option("flow_rate")),
            'name': self.get_option("flow_name"),
            'height': float(self.get_option("flow_height"))
        })

    fb.add_compute(fb_func)
    return fb

# Create multiple input blocks
flow_blocks = [create_flow_block(i) for i in range(num_inputs)]

# Tank block
def create_tank_block(num_inputs, num_outputs):
    tank_block = Block(name="Tank")

    tank_block.add_option("tank_height", type="input", value="4.0", label="Tank Height (m)")
    tank_block.add_option("tank_diameter", type="input", value="2.0", label="Tank Diameter (m)")
    tank_block.add_option("num_nodes", type="input", value="50", label="Number of Nodes")
    tank_block.add_option("initial_temp", type="input", value="92", label="Initial Temperature (°C)")

    for i in range(num_inputs):
        tank_block.add_input(name=f"flow_in_{i}")

    for j in range(num_outputs):
        tank_block.add_option(f"output_height_{j}", type="input", value=str((j + 1) * 1.0), label=f"Output {j+1} Height (m)")
        tank_block.add_option(f"output_flowrate_{j}", type="input", value="5.0", label=f"Output {j+1} Flowrate (kg/s)")
        tank_block.add_option(f"output_name_{j}", type="input", value=f"Outlet {j+1}", label=f"Output {j+1} Name")
        tank_block.add_output(name=f"flow_out_{j}")

    tank_block.add_output(name="tank_out")

    def tank_block_func(self):
        H = float(self.get_option("tank_height"))
        D = float(self.get_option("tank_diameter"))
        N = int(float(self.get_option("num_nodes")))
        T_init = float(self.get_option("initial_temp")) + 273.15

        advanced_params = {
            'C_fl': 4186,
            'k_fl': 0.0,
            'delta_k_eff': 0.0,
            'UA_i': 0.0,
            'UA_gfl': 0.0,
            'epsilon': 0.5,
            'T_env': 293.15,
            'T_gfl': 288.15,
        }

        tank = ThermalStorageTank(N, {
            'tank_height': H,
            'tank_diameter': D,
            'T_initial': T_init,
            'heat_exchangers': [],
            **advanced_params
        })

        inlets = []
        for i in range(num_inputs):
            interface = self.get_interface(f"flow_in_{i}")
            if interface:
                inlets.append((
                    interface.get("height", H * (i + 1) / (num_inputs + 1)),
                    interface["flow_rate"],
                    interface["temperature"],
                    interface["name"]
                ))

        outlets = []
        for j in range(num_outputs):
            height = float(self.get_option(f"output_height_{j}"))
            flowrate = float(self.get_option(f"output_flowrate_{j}"))
            name = self.get_option(f"output_name_{j}")
            outlets.append((height, flowrate, name))

        # Check: Mass flow in = out
        total_in = sum(flow[1] for flow in inlets)
        total_out = sum(flow[1] for flow in outlets)

        if abs(total_in - total_out) > 1e-6:
            st.error(f"⚠️ Incoming mass flow ({total_in:.2f} kg/s) ≠ Outgoing mass flow ({total_out:.2f} kg/s)")
            return

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

        self.set_interface("tank_out", {
            "node_heights": node_heights,
            "temperatures": final_temps,
            "outlets": outlet_temps,
            "inlets": inlets
        })

    tank_block.add_compute(tank_block_func)
    return tank_block

# Create tank block
tank_block = create_tank_block(num_inputs, num_outputs)

# Results block
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

# Create and run flow
flow = st_flow(flow_blocks + [tank_block, results_block])

engine = ComputeEngine(flow_blocks + [tank_block, results_block])
if flow and flow.schema:
    engine.execute(flow.schema)
