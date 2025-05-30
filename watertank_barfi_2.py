import streamlit as st
from barfi.flow import Block, ComputeEngine
from barfi.flow.streamlit import st_flow
import numpy as np
from Watertank import ThermalStorageTank

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
    
    st.sidebar.header("Heat Exchanger Options")
    max_heat_exchangers = st.sidebar.number_input("Maximum Heat Exchangers", min_value=0, max_value=5, value=1)

    st.session_state['tank_config'] = {
        'tank_height': tank_height,
        'tank_diameter': tank_diameter,
        'num_nodes': num_nodes,
        'num_inputs': num_inputs,
        'num_outputs': num_outputs,
        'max_heat_exchangers': max_heat_exchangers
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

    # --- Heat Exchanger Block Generator ---
   def create_heat_exchanger_block(i):
    hx = Block(name=f"Heat Exchanger {i+1}")
    
    # Inputs
    hx.add_input(name="primary_in")
    hx.add_output(name="primary_out")
    
    # Options - Fixed the select options by properly specifying items
    hx.add_option("hx_type", type="select", 
                 items=["Tube", "Plate"], value="Tube", label="Type")
    hx.add_option("height", type="input", value=str((i+1)*1.0), label="Height in Tank (m)")
    hx.add_option("fluid_side", type="select", 
                 items=["Primary", "Secondary"], value="Primary", label="Fluid Side")
    hx.add_option("U_value", type="input", value="1000.0", label="U Value (W/m²K)")
    hx.add_option("length", type="input", value="5.0", label="Length (m)")
    hx.add_option("diameter", type="input", value="0.05", label="Diameter (m)")
    hx.add_option("num_tubes", type="input", value="10", label="Number of Tubes")
    hx.add_option("m_dot_secondary", type="input", value="5.0", label="Secondary Flow (kg/s)")
    hx.add_option("Cp_secondary", type="input", value="4186", label="Secondary Cp (J/kgK)")
    hx.add_option("T_in_secondary", type="input", value="60.0", label="Secondary Inlet Temp (°C)")
    hx.add_option("flow_arrangement", type="select", 
                 items=["Counterflow", "Parallel"], value="Counterflow", label="Flow Arrangement")
    
    def hx_func(self):
        # Get primary flow conditions if connected
        primary_flow = self.get_interface("primary_in")
        
        if primary_flow:
            # Create heat exchanger parameters
            hx_params = {
                'type': self.get_option("hx_type").lower(),
                'height': float(self.get_option("height")),
                'U': float(self.get_option("U_value")),
                'fluid': 'primary' if self.get_option("fluid_side") == "Primary" else 'secondary',
                'm_dot_secondary': float(self.get_option("m_dot_secondary")),
                'Cp_secondary': float(self.get_option("Cp_secondary")),
                'T_in_HE': float(self.get_option("T_in_secondary")),
                'flow_arrangement': self.get_option("flow_arrangement").lower()
            }
            
            # Add type-specific parameters
            if hx_params['type'] == 'tube':
                hx_params.update({
                    'length': float(self.get_option("length")),
                    'diameter': float(self.get_option("diameter")),
                    'num_tubes': int(float(self.get_option("num_tubes")))
                })
            else:  # plate
                hx_params.update({
                    'num_plates': int(float(self.get_option("num_tubes")) + 2),  # Approximate
                    'plate_width': float(self.get_option("length")),
                    'plate_height': float(self.get_option("diameter")) * 5  # Approximate aspect ratio
                })
            
            # Store parameters for tank block to use
            self.set_interface("primary_out", {
                'hx_params': hx_params,
                'connected': True
            })
        else:
            self.set_interface("primary_out", {
                'connected': False
            })
            
    hx.add_compute(hx_func)
    return hx

    # --- Tank Block Generator ---
    def create_tank_block(num_inputs, num_outputs, max_heat_exchangers):
        tank_block = Block(name="Tank")

        tank_block.add_option("tank_height", type="input", value=str(config['tank_height']), label="Tank Height (m)")
        tank_block.add_option("tank_diameter", type="input", value=str(config['tank_diameter']), label="Tank Diameter (m)")
        tank_block.add_option("num_nodes", type="input", value=str(config['num_nodes']), label="Number of Nodes")
        tank_block.add_option("initial_temp", type="input", value="60.0", label="Initial Temperature (°C)")
        tank_block.add_option("k_fl", type="input", value="0.6", label="Fluid Conductivity (W/mK)")
        tank_block.add_option("UA_i", type="input", value="0.0", label="UA Insulation (W/K)")
        tank_block.add_option("T_env", type="input", value="20.0", label="Environment Temp (°C)")

        # Add inputs for flow connections
        for i in range(num_inputs):
            tank_block.add_option(f"input_height_{i}", type="input", value=str((i + 1) * 1.0), label=f"Input {i+1} Height (m)")
            tank_block.add_input(name=f"flow_in_{i}")

        # Add outputs for flow connections
        for j in range(num_outputs):
            tank_block.add_option(f"output_height_{j}", type="input", value=str((j + 1) * 1.0), label=f"Output {j+1} Height (m)")
            tank_block.add_option(f"output_flowrate_{j}", type="input", value="5.0", label=f"Output {j+1} Flowrate (kg/s)")
            tank_block.add_option(f"output_name_{j}", type="input", value=f"Outlet {j+1}", label=f"Output {j+1} Name")
            tank_block.add_output(name=f"flow_out_{j}")

        # Add inputs for heat exchangers
        for k in range(max_heat_exchangers):
            tank_block.add_input(name=f"hx_in_{k}")

        tank_block.add_output(name="tank_out")

        def tank_block_func(self):
            H = float(self.get_option("tank_height"))
            D = float(self.get_option("tank_diameter"))
            N = int(float(self.get_option("num_nodes")))
            T_init = float(self.get_option("initial_temp")) + 273.15

            # Get heat exchanger configurations from connected blocks
            heat_exchangers = []
            for k in range(max_heat_exchangers):
                hx_interface = self.get_interface(f"hx_in_{k}")
                if hx_interface and hx_interface.get('connected', False):
                    heat_exchangers.append(hx_interface['hx_params'])

            tank = ThermalStorageTank(N, {
                'tank_height': H,
                'tank_diameter': D,
                'T_initial': T_init,
                'heat_exchangers': heat_exchangers,
                'C_fl': 4186,
                'k_fl': float(self.get_option("k_fl")),
                'UA_i': float(self.get_option("UA_i")),
                'UA_gfl': 0.0,
                'epsilon': 0.5,
                'T_env': float(self.get_option("T_env")) + 273.15,
                'T_gfl': 288.15,
            })

            # Process inlets
            inlets = []
            for i in range(num_inputs):
                interface = self.get_interface(f"flow_in_{i}")
                height = float(self.get_option(f"input_height_{i}"))
                if interface:
                    inlets.append((height, interface["flow_rate"], interface["temperature"], interface["name"]))

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

            self.set_interface("tank_out", {
                "outlets": outlet_data,
                "solution": solution,
                "t_span": t_span,
                "flow_specs": {'inlets': inlets, 'outlets': outlets}
            })

        tank_block.add_compute(tank_block_func)
        return tank_block

    # --- Results Block ---
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

        # Add visualization button
        if st.button("Visualize Results"):
            tank = ThermalStorageTank(1, {'tank_height': 1, 'tank_diameter': 1, 'T_initial': 293.15})
            tank.visualize_results(
                results['solution'], 
                results['t_span'], 
                results['flow_specs']
            )

    results_block.add_compute(results_block_func)

    # --- Assemble Blocks ---
    flow_blocks = [create_flow_block(i) for i in range(config['num_inputs'])]
    heat_exchanger_blocks = [create_heat_exchanger_block(i) for i in range(config['max_heat_exchangers'])]
    tank_block = create_tank_block(config['num_inputs'], config['num_outputs'], config['max_heat_exchangers'])
    blocks = flow_blocks + heat_exchanger_blocks + [tank_block, results_block]

    # Display the flow diagram
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
