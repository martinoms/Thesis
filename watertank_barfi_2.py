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
    num_nodes = st.sidebar.number_input("Number of Nodes", min_value=5, max_value=100, value=20)
    num_inputs = st.sidebar.number_input("Number of Flow Inputs", min_value=1, max_value=5, value=2)
    num_outputs = st.sidebar.number_input("Number of Flow Outputs", min_value=1, max_value=5, value=2)
    
    # Heat Exchanger Configuration
    st.sidebar.header("Heat Exchanger Configuration")
    hx_type = st.sidebar.selectbox("Heat Exchanger Type", 
                                  ["None", "Tube", "Plate"],
                                  index=0)
    
    hx_config = {}
    if hx_type != "None":
        hx_config['type'] = hx_type.lower()
        hx_config['height'] = st.sidebar.number_input("HX Height (m)", 
                                                     min_value=0.0, 
                                                     max_value=float(tank_height), 
                                                     value=2.0)
        hx_config['m_dot_secondary'] = st.sidebar.number_input("Secondary Flow Rate (kg/s)", 
                                                              value=2.5)
        hx_config['T_in_HE'] = st.sidebar.number_input("Secondary Inlet Temp (°C)", 
                                                      value=60.0)
        hx_config['U_value'] = st.sidebar.number_input("U Value (W/m²K)", 
                                                      value=850.0)
        
        if hx_type == "Tube":
            hx_config['length'] = st.sidebar.number_input("Tube Length (m)", value=3.0)
            hx_config['diameter'] = st.sidebar.number_input("Tube Diameter (m)", value=0.05)
            hx_config['num_tubes'] = st.sidebar.number_input("Number of Tubes", value=10)
        else:  # Plate
            hx_config['num_plates'] = st.sidebar.number_input("Number of Plates", value=30)
            hx_config['plate_width'] = st.sidebar.number_input("Plate Width (m)", value=0.5)
            hx_config['plate_height'] = st.sidebar.number_input("Plate Height (m)", value=1.0)
            hx_config['flow_arrangement'] = st.sidebar.selectbox("Flow Arrangement", 
                                                                ["counterflow", "parallel"])

    st.session_state['tank_config'] = {
        'tank_height': tank_height,
        'tank_diameter': tank_diameter,
        'num_nodes': num_nodes,
        'num_inputs': num_inputs,
        'num_outputs': num_outputs,
        'hx_config': hx_config if hx_type != "None" else None
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
        hx_block.add_output(name="hx_out")
        
        # Display configured parameters
        hx_type = config['hx_config']['type'].title()
        hx_block.add_option("info", type="display", 
                          value=f"Pre-configured {hx_type} HX at {config['hx_config']['height']}m",
                          label="Configuration")
        
        def hx_block_func(self):
            # Pass through the pre-configured parameters
            self.set_interface("hx_out", {
                "hx_params": config['hx_config'],
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

        # Add inputs and outputs
        for i in range(num_inputs):
            tank_block.add_input(name=f"flow_in_{i}")
            tank_block.add_option(f"input_height_{i}", type="input", value=str((i + 1) * 1.0), label=f"Input {i+1} Height (m)")

        for j in range(num_outputs):
            tank_block.add_output(name=f"flow_out_{j}")
            tank_block.add_option(f"output_height_{j}", type="input", value=str((j + 1) * 1.0), label=f"Output {j+1} Height (m)")
            tank_block.add_option(f"output_flowrate_{j}", type="input", value="5.0", label=f"Output {j+1} Flowrate (kg/s)")
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

            # Process inlets and heat exchangers
            inlets = []
            heat_exchangers = []
            
            for i in range(num_inputs):
                interface = self.get_interface(f"flow_in_{i}")
                height = float(self.get_option(f"input_height_{i}"))
                
                if interface:
                    if 'hx_params' in interface:  # This is a heat exchanger connection
                        hx_params = interface['hx_params']
                        heat_exchangers.append({
                            'hx': TubeHeatExchanger(hx_params, 4186) if hx_params['type'] == 'tube' 
                                  else PlateHeatExchanger(hx_params, 4186),
                            'node_idx': int(height / H * N),  # Convert height to node index
                            'type': hx_params['type']
                        })
                    else:  # Regular flow input
                        inlets.append((
                            height, 
                            interface["flow_rate"], 
                            interface["temperature"], 
                            interface.get("name", f"Inlet {i+1}")
                        ))

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
                    'power_kW': Q / 1000  # Add power in kW
                })

            self.set_interface("tank_out", {
                "outlets": outlet_data,
                "heat_exchangers": hx_results,
                "tank_params": tank_params
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
            st.write(f"Height: {results['tank_params']['tank_height']} m")
            st.write(f"Diameter: {results['tank_params']['tank_diameter']} m")
            st.write(f"Nodes: {results['tank_params'].get('num_nodes', 'N/A')}")
            
            st.subheader("Tank Outlet Conditions")
            for outlet in results.get('outlets', []):
                st.write(f"""
                **{outlet['name']}**  
                - Height: {outlet['height']:.2f} m  
                - Temperature: {outlet['temperature']:.1f} °C  
                - Mass flow rate: {outlet['flow_rate']:.2f} kg/s  
                """)
                st.write("---")

            if results.get('heat_exchangers'):
                st.subheader("Heat Exchanger Performance")
                for hx in results['heat_exchangers']:
                    st.write(f"""
                    **{hx['type'].title()} Heat Exchanger at {hx['height']:.2f}m**  
                    - Secondary Flow Rate: {hx['m_dot_secondary']:.2f} kg/s  
                    - Inlet Temperature: {hx['T_in_HE']:.1f} °C  
                    - Outlet Temperature: {hx['T_out_HE']:.1f} °C  
                    - Heat Transfer Power: {hx['power_kW']:.2f} kW  
                    """)
                    st.write("---")
            else:
                st.write("No heat exchanger configured")

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
