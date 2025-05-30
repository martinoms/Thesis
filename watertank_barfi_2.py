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

    tank_block.add_option("tank_height", type="input", value="4.0", label="Tank Height (m)")
    tank_block.add_option("tank_diameter", type="input", value="2.0", label="Tank Diameter (m)")
    tank_block.add_option("num_nodes", type="input", value="50", label="Number of Nodes")
    tank_block.add_option("initial_temp", type="input", value="30", label="Initial Temperature (°C)")
    tank_block.add_option("display", type="output", value="", label="Outlet 1 Display")

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

    advanced_params = {
        'C_fl': 4186,
        'k_fl': 0.6,
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
        height = float(self.get_option(f"input_height_{i}"))
        if interface:
            inlets.append((height, interface["flow_rate"], interface["temperature"], interface["name"]))

    outlets = []
    for j in range(num_outputs):
        height = float(self.get_option(f"output_height_{j}"))
        flowrate = float(self.get_option(f"output_flowrate_{j}"))
        name = self.get_option(f"output_name_{j}")
        outlets.append((height, flowrate, name))

    total_in = sum(flow[1] for flow in inlets)
    total_out = sum(flow[1] for flow in outlets)
    if abs(total_in - total_out) > 1e-6:
        st.error(f"⚠️ Incoming mass flow ({total_in:.2f} kg/s) ≠ Outgoing mass flow ({total_out:.2f} kg/s)")
        return

    t_span = np.linspace(0, 3600, 100)
    solution = tank.solve(t_span, {'inlets': inlets, 'outlets': outlets})
    final_temps = solution[-1, :] - 273.15
    node_heights = np.linspace(H, 0, N)

    # Get outlet conditions using the new method
    outlet_data = []
    for j, (height, flow_rate, name) in enumerate(outlets):
        # Use get_outlet_conditions to get the temperature
        temp, flow = tank.get_outlet_conditions(solution, {'inlets': inlets, 'outlets': outlets}, name)
        
        outlet_data.append({
            'name': name,
            'height': height,
            'temperature': temp,
            'flow_rate': flow_rate
        })

        # Set the output interface with consistent structure
        self.set_interface(f"flow_out_{j}", {
            "temperature": float(temp),
            "flow_rate": float(flow_rate),
            "name": name,
            "height": float(height)
        })

    # Set display for the first outlet
    if outlet_data:
        label = f"{outlet_data[0]['temperature']:.1f}°C @ {outlet_data[0]['flow_rate']:.1f} kg/s"
        self.set_option("display", label)

    # Set the tank output with all data
    self.set_interface("tank_out", {
        "node_heights": node_heights.tolist(),  # Convert to list for JSON serialization
        "temperatures": final_temps.tolist(),
        "outlets": outlet_data,
        "inlets": [{
            'height': i[0],
            'flow_rate': i[1],
            'temperature': i[2],
            'name': i[3]
        } for i in inlets],
        "solution": solution.tolist(),  # Include full solution if needed
        "time_span": t_span.tolist()
    })

tank_block = create_tank_block(num_inputs, num_outputs)

# Results block
results_block = Block(name="Results")
results_block.add_input(name="results_in")

def results_block_func(self):
    results = self.get_interface("results_in")
    if not results:
        return

    # Create temperature profile chart
    st.subheader("Temperature Profile")
    df = pd.DataFrame({
        'Height (m)': results['node_heights'],
        'Temperature (°C)': results['temperatures']
    })

    # Add markers for inlets and outlets
    inlet_points = pd.DataFrame([{
        'Height (m)': i['height'],
        'Temperature (°C)': i['temperature'],
        'Type': 'Inlet',
        'Name': i['name']
    } for i in results['inlets']])
    
    outlet_points = pd.DataFrame([{
        'Height (m)': o['height'],
        'Temperature (°C)': o['temperature'],
        'Type': 'Outlet',
        'Name': o['name']
    } for o in results['outlets']])
    
    points = pd.concat([inlet_points, outlet_points])
    
    line = alt.Chart(df).mark_line().encode(
        x='Temperature (°C)',
        y='Height (m)'
    )
    
    points_chart = alt.Chart(points).mark_point(size=100, filled=True).encode(
        x='Temperature (°C)',
        y='Height (m)',
        color='Type',
        tooltip=['Name', 'Temperature (°C)', 'Height (m)']
    )
    
    chart = (line + points_chart).properties(width=600, height=400)
    st.altair_chart(chart, use_container_width=True)

    # Display detailed information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Inlets")
        for inlet in results['inlets']:
            st.metric(
                label=f"{inlet['name']} (Height: {inlet['height']:.2f}m",
                value=f"{inlet['temperature']:.1f}°C",
                delta=f"{inlet['flow_rate']:.2f} kg/s"
            )
    
    with col2:
        st.subheader("Outlets")
        for outlet in results['outlets']:
            st.metric(
                label=f"{outlet['name']} (Height: {outlet['height']:.2f}m",
                value=f"{outlet['temperature']:.1f}°C",
                delta=f"{outlet['flow_rate']:.2f} kg/s"
            )

    # Optional: Show time evolution for selected nodes
    if 'solution' in results and 'time_span' in results:
        st.subheader("Time Evolution")
        selected_nodes = st.multiselect(
            "Select nodes to display",
            options=[f"{h:.2f}m" for h in results['node_heights']],
            default=[f"{results['node_heights'][0]:.2f}m", f"{results['node_heights'][-1]:.2f}m"]
        )
        
        if selected_nodes:
            # Convert solution back to numpy array
            solution = np.array(results['solution'])
            time_span = np.array(results['time_span'])
            
            # Get indices of selected nodes
            selected_indices = [results['node_heights'].index(float(h[:-1])) for h in selected_nodes]
            
            # Prepare data for plotting
            plot_data = []
            for idx in selected_indices:
                for t, temp in zip(time_span, solution[:, idx]):
                    plot_data.append({
                        'Time (s)': t,
                        'Temperature (°C)': temp - 273.15,
                        'Node': f"{results['node_heights'][idx]:.2f}m"
                    })
            
            df_time = pd.DataFrame(plot_data)
            time_chart = alt.Chart(df_time).mark_line().encode(
                x='Time (s)',
                y='Temperature (°C)',
                color='Node',
                tooltip=['Time (s)', 'Temperature (°C)', 'Node']
            ).properties(width=600, height=300)
            
            st.altair_chart(time_chart, use_container_width=True)

results_block.add_compute(results_block_func)

# Assemble flow
flow = st_flow(flow_blocks + [tank_block, results_block])
engine = ComputeEngine(flow_blocks + [tank_block, results_block])
if flow and flow.schema:
    engine.execute(flow.schema)
