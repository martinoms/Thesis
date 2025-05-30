#aanpas test, test123
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
st.markdown("By Martijn Stynen & Senne Schepens")
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

    def fb_func(self):
        self.set_interface("flow_out", {
            'temperature': float(self.get_option("flow_temp")),
            'flow_rate': float(self.get_option("flow_rate")),
            'name': self.get_option("flow_name")
        })

    fb.add_compute(fb_func)
    return fb

flow_blocks = [create_flow_block(i) for i in range(num_inputs)]

# Tank block
def create_tank_block(num_inputs, num_outputs):
    tank_block = Block(name="Tank")

    tank_block.add_option("tank_height", type="input", value="", label="Tank Height (m)")
    tank_block.add_option("tank_diameter", type="input", value="", label="Tank Diameter (m)")
    tank_block.add_option("num_nodes", type="input", value="", label="Number of Nodes")
    tank_block.add_option("initial_temp", type="input", value="", label="Initial Temperature (°C)")

    for i in range(num_inputs):
        tank_block.add_option(f"input_height_{i}", type="input", value=str((i + 1) * 1.0), label=f"Input {i+1} Height (m)")
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

        tank = ThermalStorageTank(N, {
            'tank_height': H,
            'tank_diameter': D,
            'T_initial': T_init,
            'heat_exchangers': [],
            'C_fl': 4186,
            'k_fl': 0.6,
            'UA_i': 0.0,
            'UA_gfl': 0.0,
            'epsilon': 0.5,
            'T_env': 293.15,
            'T_gfl': 288.15,
        })

        # Get inlet configurations
        inlets = []
        for i in range(num_inputs):
            interface = self.get_interface(f"flow_in_{i}")
            height = float(self.get_option(f"input_height_{i}"))
            if interface:
                inlets.append((height, interface["flow_rate"], interface["temperature"], interface["name"]))

        # Get outlet configurations
        outlets = []
        for j in range(num_outputs):
            height = float(self.get_option(f"output_height_{j}"))
            flowrate = float(self.get_option(f"output_flowrate_{j}"))
            name = self.get_option(f"output_name_{j}")
            outlets.append((height, flowrate, name))

        # Check mass balance
        total_in = sum(flow[1] for flow in inlets)
        total_out = sum(flow[1] for flow in outlets)
        if abs(total_in - total_out) > 1e-6:
            st.error(f"⚠️ Mass flow imbalance: In={total_in:.2f} kg/s, Out={total_out:.2f} kg/s")
            return

        # Run simulation
        t_span = np.linspace(0, 10*3600, 100)  # 1 hour
        solution = tank.solve(t_span, {'inlets': inlets, 'outlets': outlets})

        # Set output interfaces and build output data
        outlet_data = []
        for j, (height, flow_rate, name) in enumerate(outlets):
            temp, _ = tank.get_outlet_conditions(solution, {'inlets': inlets, 'outlets': outlets}, name)
            outlet_data.append({
                'name': name,
                'height': height,
                'temperature': temp,
                'flow_rate': flow_rate
            })

            self.set_interface(f"flow_out_{j}", {
                "temperature": float(temp),
                "flow_rate": float(flow_rate)
            })

        self.set_interface("tank_out", {
            "outlets": outlet_data
        })

    tank_block.add_compute(tank_block_func)
    return tank_block

tank_block = create_tank_block(num_inputs, num_outputs)

# Results block
results_block = Block(name="Results")
results_block.add_input(name="results_in")

def results_block_func(self):
    results = self.get_interface("results_in")
    if not results:
        return

    st.subheader("Tank Outlet Conditions")
    for outlet in results['outlets']:
        st.write(f"""
        **{outlet['name']}**  
        - Height: {outlet['height']:.2f} m  
        - Temperature: {outlet['temperature']:.1f} °C  
        - Mass flow rate: {outlet['flow_rate']:.2f} kg/s  
        """)
        st.write("---")

results_block.add_compute(results_block_func)

blocks = flow_blocks + [tank_block, results_block]

# Barfi flow UI
barfi_result = st_flow(blocks)

# Execute flow
if barfi_result and barfi_result.editor_schema:
    compute_engine = ComputeEngine(blocks)
    compute_engine.execute(barfi_result.editor_schema)

    node_labels = [node.label for node in barfi_result.editor_schema.nodes]
    if "Tank" in node_labels and "Results" in node_labels:
        try:
            tank_block_node = barfi_result.editor_schema.block(node_label="Tank")
            tank_output = tank_block_node.get_interface("tank_out")

            result_block_node = barfi_result.editor_schema.block(node_label="Results")

            if not result_block_node.get_interface("results_in") and tank_output:
                result_block_node.set_interface("results_in", tank_output)

            result_data = result_block_node.get_interface("results_in")
            if result_data:
                st.write("Results from Results block:", result_data)

        except Exception as e:
            st.error(f"⚠️ Error during result processing: {e}")
    else:
        st.warning("⚠️ Please add and connect both the **Tank** and **Results** blocks in the editor.")
