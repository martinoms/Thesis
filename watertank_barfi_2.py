import streamlit as st
from barfi.flow import Block, ComputeEngine
from barfi.flow.streamlit import st_flow
import numpy as np
from Watertank import ThermalStorageTank, TubeHeatExchanger, PlateHeatExchanger

# --- App Setup ---
st.set_page_config(page_title="Thermal Tank Flow Sim", layout="wide")
st.title("Thermal Tank Flow Simulation")
st.markdown("Configure flow connections and simulate tank behavior")
st.markdown("By Martijn Stynen & Senne Schepens")

# --- Page Selector ---
page = st.sidebar.radio("Navigation", ["Configure Tank", "Launch Simulation"])

# --- Tank Configuration Step ---
if page == "Configure Tank":
    st.sidebar.header("Tank Configuration")
    tank_height = st.sidebar.number_input("Tank Height (m)", value=4.0)
    tank_diameter = st.sidebar.number_input("Tank Diameter (m)", value=2.0)
    num_nodes = st.sidebar.number_input("Number of Nodes", min_value=5, max_value=100, value=50)
    num_inputs = st.sidebar.number_input("Number of Flow Inputs", min_value=1, max_value=5, value=2)
    num_outputs = st.sidebar.number_input("Number of Flow Outputs", min_value=1, max_value=5, value=2)
    
    # Heat Exchanger Configuration
    st.sidebar.header("Heat Exchanger Configuration")
    use_hx = st.sidebar.checkbox("Include Heat Exchanger", value=False)
    
    hx_config = {}
    if use_hx:
        hx_type = st.sidebar.selectbox("Heat Exchanger Type", ["Tube", "Plate"])
        hx_config = {
            'type': hx_type.lower(),
            'height': st.sidebar.number_input("HX Height (m)", min_value=0.0, 
                                             max_value=float(tank_height), value=2.0),
            'm_dot_secondary': st.sidebar.number_input("Secondary Flow Rate (kg/s)", value=2.5),
            'T_in_HE': st.sidebar.number_input("Secondary Inlet Temp (°C)", value=60.0),
            'U': st.sidebar.number_input("U Value (W/m²K)", value=850.0),
            'Cp_secondary': st.sidebar.number_input("Cp secondary (W/kgK)", value=4186),
            'fluid': 'secondary'
            
        }
        
        if hx_type == "Tube":
            hx_config.update({
                'length': st.sidebar.number_input("Tube Length (m)", value=3.0),
                'diameter': st.sidebar.number_input("Tube Diameter (m)", value=0.05),
                'num_tubes': st.sidebar.number_input("Number of Tubes", value=10)
            })
        else:  # Plate
            hx_config.update({
                'num_plates': st.sidebar.number_input("Number of Plates", value=30),
                'plate_width': st.sidebar.number_input("Plate Width (m)", value=0.5),
                'plate_height': st.sidebar.number_input("Plate Height (m)", value=1.0),
                'flow_arrangement': st.sidebar.selectbox("Flow Arrangement", 
                                                       ["counterflow", "parallel"])
            })

    st.session_state['tank_config'] = {
        'tank_height': tank_height,
        'tank_diameter': tank_diameter,
        'num_nodes': num_nodes,
        'num_inputs': num_inputs,
        'num_outputs': num_outputs,
        'hx_config': hx_config if use_hx else None
    }

    st.success("Configuration saved. Switch to 'Launch Simulation' to begin.")

# --- Simulation + Barfi Editor Step ---
elif page == "Launch Simulation":
    config = st.session_state.get('tank_config', None)

    if not config:
        st.warning("⚠️ Please configure the tank first in the 'Configure Tank' menu.")
        st.stop()

    # --- Flow Block Generator ---
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

    # --- Heat Exchanger Block ---
    def create_heat_exchanger_block():
        if not config['hx_config']:
            return None
            
        hx_block = Block(name="Heat Exchanger")
        hx_block.add_input(name="secondary_in")  # Secondary fluid input
        hx_block.add_output(name="hx_out")      # Connection to tank
        
        # Display configured parameters
        hx_type = config['hx_config']['type'].title()
        hx_block.add_option("info", type="display", 
                          value=f"Pre-configured {hx_type} HX at {config['hx_config']['height']}m",
                          label="Configuration")
        
        def hx_block_func(self):
            # Get secondary input conditions if connected
            secondary_in = self.get_interface("secondary_in")
            hx_params = config['hx_config'].copy()
            
            if secondary_in:
                hx_params['T_in_HE'] = secondary_in.get('temperature', hx_params['T_in_HE'])
                hx_params['m_dot_secondary'] = secondary_in.get('flow_rate', hx_params['m_dot_secondary'])
            
            # Pass through the pre-configured parameters
            self.set_interface("hx_out", {
                "hx_params": hx_params,
                "connected": True
            })
        
        hx_block.add_compute(hx_block_func)
        return hx_block

    # --- Tank Block Generator ---
    def create_tank_block(num_inputs, num_outputs):
        tank_block = Block(name="Tank")

        tank_block.add_option("tank_height", type="input", value=str(config['tank_height']), label="Tank Height (m)")
        tank_block.add_option("tank_diameter", type="input", value=str(config['tank_diameter']), label="Tank Diameter (m)")
        tank_block.add_option("num_nodes", type="input", value=str(config['num_nodes']), label="Number of Nodes")
        tank_block.add_option("initial_temp", type="input", value="60.0", label="Initial Temperature (°C)")

        # Add regular flow inputs
        for i in range(num_inputs):
            tank_block.add_input(name=f"flow_in_{i}")
            tank_block.add_option(f"input_height_{i}", type="input", value="", label=f"Input {i+1} Height (m)")

        # Add dedicated heat exchanger input if configured
        if config['hx_config']:
            tank_block.add_input(name="hx_in")
            tank_block.add_option("hx_height", type="input", value=str(config['hx_config']['height']), label="HX Height (m)")

        # Add outputs
        for j in range(num_outputs):
            tank_block.add_output(name=f"flow_out_{j}")
            tank_block.add_option(f"output_height_{j}", type="input", value="", label=f"Output {j+1} Height (m)")
            tank_block.add_option(f"output_flowrate_{j}", type="input", value="", label=f"Output {j+1} Flowrate (kg/s)")
            tank_block.add_option(f"output_name_{j}", type="input", value=f"Outlet {j+1}", label=f"Output {j+1} Name")

        tank_block.add_output(name="tank_out")

        def tank_block_func(self):
            # Get tank parameters
            H = float(self.get_option("tank_height"))
            D = float(self.get_option("tank_diameter"))
            N = int(float(self.get_option("num_nodes")))
            T_init = float(self.get_option("initial_temp")) + 273.15

            # Initialize tank with basic parameters
            tank_params = {
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
            }

            # Process regular flow inlets
            inlets = []
            for i in range(num_inputs):
                interface = self.get_interface(f"flow_in_{i}")
                height = float(self.get_option(f"input_height_{i}"))
                
                if interface:
                    inlets.append((
                        height, 
                        interface["flow_rate"], 
                        interface["temperature"], 
                        interface.get("name", f"Inlet {i+1}")
                    ))

            # Process heat exchanger if configured
            heat_exchangers = []
            if config['hx_config']:
                hx_interface = self.get_interface("hx_in")
                if hx_interface and 'hx_params' in hx_interface:
                    hx_params = hx_interface['hx_params']
                    hx_height = float(self.get_option("hx_height"))
                    
                    heat_exchangers.append({
                        'hx': TubeHeatExchanger(hx_params, 4186) if hx_params['type'] == 'tube' 
                              else PlateHeatExchanger(hx_params, 4186),
                        'node_idx': int(hx_height / H * N),
                        'type': hx_params['type']
                    })

            # Add heat exchangers to tank parameters
            tank_params['heat_exchangers'] = [
                {
                    'height': hx['hx'].params['height'],
                    'type': hx['type'],
                    **hx['hx'].params
                } for hx in heat_exchangers
            ]

            # Create tank instance
            tank = ThermalStorageTank(N, tank_params)

            # Process outlets
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
            t_span = np.linspace(0, 10*3600, 100)
            solution = tank.solve(t_span, {'inlets': inlets, 'outlets': outlets})

            # Prepare outlet data
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

            # Prepare heat exchanger results
            hx_results = []
            for idx, hx in enumerate(heat_exchangers):
                T_out, m_dot, Q = tank.get_heat_exchanger_conditions(
                    solution, {'inlets': inlets, 'outlets': outlets}, 
                    hx_index=idx
                )
                hx_results.append({
                    'type': hx['type'],
                    'height': hx['hx'].params['height'],
                    'm_dot_secondary': hx['hx'].params['m_dot_secondary'],
                    'T_in_HE': hx['hx'].params['T_in_HE'],
                    'T_out_HE': T_out,
                    'heat_transfer': Q,
                    'power_kW': Q / 1000,
                    'efficiency': (hx['hx'].params['T_in_HE'] - T_out) / 
                                 (hx['hx'].params['T_in_HE'] - solution[-1, hx['node_idx']] + 273.15)
                })

            self.set_interface("tank_out", {
                "outlets": outlet_data,
                "heat_exchangers": hx_results,
                "tank_params": tank_params,
                "node_temperatures": solution[-1, :] - 273.15  # Final temps in °C
            })

        tank_block.add_compute(tank_block_func)
        return tank_block

    # --- Results Block ---
    def create_results_block():
        results_block = Block(name="Results")
        results_block.add_input(name="results_in")

        def results_block_func(self):
            results = self.get_interface("results_in")
            if not results:
                return

            st.subheader("Tank Configuration")
            cols = st.columns(3)
            cols[0].metric("Height", f"{results['tank_params']['tank_height']} m")
            cols[1].metric("Diameter", f"{results['tank_params']['tank_diameter']} m")
            cols[2].metric("Nodes", results['tank_params'].get('num_nodes', 'N/A'))
            
            st.subheader("Tank Outlet Conditions")
            for outlet in results.get('outlets', []):
                with st.expander(f"{outlet['name']} at {outlet['height']:.2f}m"):
                    st.metric("Temperature", f"{outlet['temperature']:.1f} °C")
                    st.metric("Flow Rate", f"{outlet['flow_rate']:.2f} kg/s")

            if results.get('heat_exchangers'):
                st.subheader("Heat Exchanger Performance")
                for hx in results['heat_exchangers']:
                    with st.expander(f"{hx['type'].title()} HX at {hx['height']:.2f}m"):
                        cols = st.columns(2)
                        cols[0].metric("Inlet Temp", f"{hx['T_in_HE']:.1f} °C")
                        cols[1].metric("Outlet Temp", f"{hx['T_out_HE']:.1f} °C")
                        cols[0].metric("Flow Rate", f"{hx['m_dot_secondary']:.2f} kg/s")
                        cols[1].metric("Power", f"{hx['power_kW']:.2f} kW")
                        if 0<hx['efficiency']<1:
                            st.progress(min(1.0, hx['efficiency']), 
                                f"Efficiency: {abs(hx['efficiency'])*100:.1f}%")
                        
                        
                            
            else:
                st.info("No heat exchanger configured")

            # Temperature profile visualization
            st.subheader("Temperature Profile")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(results['node_temperatures'], 
                   np.linspace(0, config['tank_height'], config['num_nodes']))
            ax.set_xlabel("Temperature (°C)")
            ax.set_ylabel("Height (m)")
            ax.grid(True)
            st.pyplot(fig)

        results_block.add_compute(results_block_func)
        return results_block

    # --- Assemble Blocks ---
    flow_blocks = [create_flow_block(i) for i in range(config['num_inputs'])]
    tank_block = create_tank_block(config['num_inputs'], config['num_outputs'])
    hx_block = create_heat_exchanger_block()
    results_block = create_results_block()

    blocks = flow_blocks + [tank_block]
    if hx_block:
        blocks.append(hx_block)
    blocks.append(results_block)

    # --- Run Flow Editor ---
    barfi_result = st_flow(blocks)

    if barfi_result and barfi_result.editor_schema:
        compute_engine = ComputeEngine(blocks)
        compute_engine.execute(barfi_result.editor_schema)

        try:
            tank_node = barfi_result.editor_schema.block(node_label="Tank")
            tank_output = tank_node.get_interface("tank_out")
            results_node = barfi_result.editor_schema.block(node_label="Results")

            if not results_node.get_interface("results_in") and tank_output:
                results_node.set_interface("results_in", tank_output)

        except Exception as e:
            st.error(f"⚠️ Error during result processing: {e}")
