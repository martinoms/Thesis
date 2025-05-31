"""
Martijn Stynen & Senne Schepens
Centrale warmteverdeling van een industriële organische vergistingsinstallatie met behulp van een watervat

Finale 1D transient code
Als programmeerhulp is in dit programma chatGPT gebruikt.
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

class TubeHeatExchanger:
    def __init__(self, params, C_fl):
        """
        Tube heat exchanger implementation
        
        Args:
            params (dict): {
                'height': position in tank [m],
                'length': length of tubes [m],
                'diameter': tube diameter [m],
                'num_tubes': number of tubes [],
                'U': overall heat transfer coefficient [W/m²K],
                'fluid': 'primary' or 'secondary',
                'm_dot_secondary': secondary flow rate [kg/s],
                'Cp_secondary': secondary fluid heat capacity [J/kgK],
                'T_in_HE': secondary inlet temp [°C],
                'effectiveness': optional fixed effectiveness
            }
            C_fl: Fluid heat capacity [J/kgK]
        """
        self.params = params
        self.C_fl = C_fl
        self.A = np.pi * params['diameter'] * params['length'] * params['num_tubes']
        self.fixed_effectiveness = params.get('effectiveness', None)
        
    def calculate_heat_transfer(self, T_tank_node, m_dot_tank):
        """
        Calculate heat transfer for tube heat exchanger
        Returns:
            Q (float): Heat transfer rate [W]
            T_out_secondary (float): Secondary outlet temperature [K]
            effectiveness (float): Heat exchanger effectiveness [0-1]
        """
        p = self.params
        T_secondary_in = p['T_in_HE'] + 273.15  # Convert to K
        
        # Capacity rates
        C_secondary = p['m_dot_secondary'] * p['Cp_secondary']
        C_tank = m_dot_tank * self.C_fl if m_dot_tank > 1e-6 else 1e-6
        
        # Determine C_min and capacity ratio
        C_min = min(C_tank, C_secondary)
        C_max = max(C_tank, C_secondary)
        C_r = C_min / C_max if C_max > 0 else 0
        
        # Calculate NTU
        NTU = (p['U'] * self.A) / C_min if C_min > 0 else 0
        
        # Effectiveness calculation
        if self.fixed_effectiveness is not None:
            effectiveness = self.fixed_effectiveness
        else:
            if C_r < 1e-6:  # One fluid has infinite capacity
                effectiveness = 1 - np.exp(-NTU)
            else:
                effectiveness = (1 - np.exp(-NTU * (1 - C_r))) / (1 - C_r * np.exp(-NTU * (1 - C_r)))
        
        # Heat transfer calculation
        if p['fluid'] == 'primary':
            Q_max = C_min * (T_secondary_in - T_tank_node)
        else:
            Q_max = C_min * (T_tank_node - T_secondary_in)
            
        Q = effectiveness * Q_max
        
        # Secondary outlet temperature
        T_out_secondary = T_secondary_in - Q/C_secondary if C_secondary > 0 else T_secondary_in
        
        return Q, T_out_secondary, effectiveness  # Now returns effectiveness

class PlateHeatExchanger:
    def __init__(self, params, C_fl):
        """
        Plate heat exchanger implementation
        
        Args:
            params (dict): {
                'height': position in tank [m],
                'area': heat transfer area [m²],  # Optioneel als 'num_plates' is opgegeven
                'U': overall heat transfer coefficient [W/m²K],
                'fluid': 'primary' or 'secondary',
                'm_dot_secondary': secondary flow rate [kg/s],
                'Cp_secondary': secondary fluid heat capacity [J/kgK],
                'T_in_HE': secondary inlet temp [°C],
                'effectiveness': optional fixed effectiveness,
                'flow_arrangement': 'counterflow' or 'parallel',
                'num_plates': number of plates [],  # Nieuw: aantal platen
                'plate_width': width of plates [m],  # Nieuw: breedte van platen
                'plate_height': height of plates [m]  # Nieuw: hoogte van platen
            }
            C_fl: Fluid heat capacity [J/kgK]
        """
        self.params = params
        self.C_fl = C_fl
        
        # Bereken het warmteoverdrachtsoppervlak op basis van aantal platen of gebruik direct opgegeven area
        if 'num_plates' in params:
            # Bereken area op basis van aantal platen (2 zijden per plaat, minus 2 eindplaten)
            self.area = (params['num_plates'] - 2) * 2 * params['plate_width'] * params['plate_height']
        else:
            self.area = params['area']
            
        self.fixed_effectiveness = params.get('effectiveness', None)
        
    def calculate_heat_transfer(self, T_tank_node, m_dot_tank):
        """
        Calculate heat transfer for tube heat exchanger
        Returns:
            Q (float): Heat transfer rate [W]
            T_out_secondary (float): Secondary outlet temperature [K]
            effectiveness (float): Heat exchanger effectiveness [0-1]
        """
        p = self.params
        T_secondary_in = p['T_in_HE'] + 273.15  # Convert to K
        
        # Capacity rates
        C_secondary = p['m_dot_secondary'] * p['Cp_secondary']
        C_tank = m_dot_tank * self.C_fl if m_dot_tank > 1e-6 else 1e-6
        
        # Determine C_min and capacity ratio
        C_min = min(C_tank, C_secondary)
        C_max = max(C_tank, C_secondary)
        C_r = C_min / C_max if C_max > 0 else 0
        
        # Calculate NTU
        NTU = (p['U'] * self.area) / C_min if C_min > 0 else 0
        
        # Effectiveness calculation
        if self.fixed_effectiveness is not None:
            effectiveness = self.fixed_effectiveness
        else:
            if C_r < 1e-6:  # One fluid has infinite capacity
                effectiveness = 1 - np.exp(-NTU)
            else:
                effectiveness = (1 - np.exp(-NTU * (1 - C_r))) / (1 - C_r * np.exp(-NTU * (1 - C_r)))
        
        # Heat transfer calculation
        if p['fluid'] == 'primary':
            Q_max = C_min * (T_secondary_in - T_tank_node)
        else:
            Q_max = C_min * (T_tank_node - T_secondary_in)
            
        Q = effectiveness * Q_max
        
        # Secondary outlet temperature
        T_out_secondary = T_secondary_in - Q/C_secondary if C_secondary > 0 else T_secondary_in
        
        return Q, T_out_secondary, effectiveness  # Now returns effectiveness
class ThermalStorageTank:
    def __init__(self, num_nodes, params):
        """Initialize the thermal storage tank"""
        self.num_nodes = num_nodes
        self.params = params
        
        # Geometry calculations
        self.tank_height = params['tank_height']
        self.node_heights = np.linspace(self.tank_height, 0, num_nodes)
        self.node_volumes = np.full(num_nodes, 
                                   np.pi * (params['tank_diameter']/2)**2 * 
                                   self.tank_height / num_nodes)
        
        
        initial_T = params.get('T_initial', None)
        if initial_T is None:
            self.T = np.full(num_nodes, 20 + 273.15)  # Standaard uniforme temperatuur
        elif isinstance(initial_T, (int, float)):
            self.T = np.full(num_nodes, initial_T + 273.15)  # Uniforme temperatuur
        elif isinstance(initial_T, (list, np.ndarray)):
            if len(initial_T) == num_nodes:
                self.T = np.array(initial_T) + 273.15  # Specifieke temperatuurverdeling
            else:
                raise ValueError("Length of initial_T array must match num_nodes")
        
        # Flow variables
        self.FL4S = 0.0  # Downward flow storage
        self.FL6S = 0.0  # Upward flow storage
        
        # Initialize heat exchangers
        self.heat_exchangers = []
        if 'heat_exchangers' in params:
            for hx_params in params['heat_exchangers']:
                if hx_params.get('type', 'tube') == 'plate':
                    hx_class = PlateHeatExchanger
                else:
                    hx_class = TubeHeatExchanger
                
                self.heat_exchangers.append({
                    'hx': hx_class(hx_params, params['C_fl']),
                    'node_idx': self.get_node_at_height(hx_params['height']),
                    'type': hx_params.get('type', 'tube')
                })

        # Geometry calculations
        self.tank_height = params['tank_height']
        self.node_heights = np.linspace(self.tank_height, 0, num_nodes)
        self.node_volumes = np.full(num_nodes, 
                                   np.pi * (params['tank_diameter']/2)**2 * 
                                   self.tank_height / num_nodes)
        
        # Initialize temperatures
        self.T = np.full(num_nodes, params.get('T_initial', 20)) + 273.15
        
    def get_density(self, T):
        """Calculate water density [kg/m³] as function of temperature [K]"""
        T_C = T - 273.15
        return (999.974950 * (1 - (3.983035 * (T_C - 301.797)**2) / 
                ((T_C + 522528.9) * (T_C + 69.34881))))
    
    def get_node_at_height(self, height):
        """Get node index closest to specified height"""
        return np.argmin(np.abs(self.node_heights - height))
    
    def get_outlet_conditions(self, solution, flow_specs, outlet_name, time_idx=-1):
        """
        Get the temperature and mass flow rate of a specified outlet at a given time index.
        
        Args:
            solution (ndarray): Temperature solution array from solve()
            flow_specs (dict): Flow specifications dictionary
            outlet_name (str): Name of the outlet to query (must match name in flow_specs)
            time_idx (int): Time index to query (-1 for last time step by default)
            
        Returns:
            tuple: (temperature in °C, mass flow rate in kg/s)
            
        Raises:
            ValueError: If outlet name is not found
        """
        # Find the outlet in flow_specs
        outlet = None
        for height, flow_rate, name in flow_specs['outlets']:
            if name == outlet_name:
                outlet = (height, flow_rate, name)
                break
                
        if outlet is None:
            raise ValueError(f"Outlet '{outlet_name}' not found in flow specifications")
        
        height, flow_rate, name = outlet
        node_idx = self.get_node_at_height(height)
        
        # Get temperature (convert from K to °C)
        temperature = solution[time_idx, node_idx] - 273.15
        
        return temperature, flow_rate
    
    def get_heat_exchanger_conditions(self, solution, flow_specs, hx_index=0, time_idx=-1):
        """
        Get the output conditions of a specified heat exchanger at a given time index.
        
        Args:
            solution (ndarray): Temperature solution array from solve()
            flow_specs (dict): Flow specifications dictionary
            hx_index (int): Index of the heat exchanger to query (0-based)
            time_idx (int): Time index to query (-1 for last time step by default)
            
        Returns:
            tuple: (secondary outlet temperature in °C, 
                    secondary mass flow rate in kg/s,
                    heat transfer rate in W,
                    effectiveness [0-1])
        """
        if hx_index >= len(self.heat_exchangers):
            raise IndexError(f"Heat exchanger index {hx_index} out of range (0-{len(self.heat_exchangers)-1})")
        
        hx_info = self.heat_exchangers[hx_index]
        node_idx = hx_info['node_idx']
        
        # Get node flow conditions at the specified time
        node_flow = {'flow_in': 0.0, 'flow_out': 0.0, 'T_in': 0.0}
        for height, flow_rate, temp, name in flow_specs['inlets']:
            if node_idx == self.get_node_at_height(height):
                node_flow['flow_in'] += flow_rate
        for height, flow_rate, name in flow_specs['outlets']:
            if node_idx == self.get_node_at_height(height):
                node_flow['flow_out'] += flow_rate
        
        flows = self.calculate_flows(node_flow, node_idx)
        m_dot_local = abs(flows['FL4'] - flows['FL6'])
        
        # Calculate heat exchanger performance (now returns effectiveness)
        Q, T_out_secondary, effectiveness = hx_info['hx'].calculate_heat_transfer(
            solution[time_idx, node_idx], 
            m_dot_local
        )
        
        # Convert to appropriate units
        return (
            T_out_secondary - 273.15,  # Convert K to °C
            hx_info['hx'].params['m_dot_secondary'],
            Q,
            effectiveness  # Directly from calculate_heat_transfer
        )
    def calculate_flows(self, external_flow, current_node):
        """Calculate flow distribution for current node"""
        FL3 = external_flow['flow_in'] - external_flow['flow_out']
        flows = {'FL4': 0.0, 'FL5': 0.0, 'FL6': 0.0, 'FL7': 0.0, 'FL8': 0.0}
        
        if current_node == 0:  # Top node
            if FL3 < 0:
                flows['FL5'] = -FL3
            else:
                flows['FL7'] = FL3
            self.FL4S = flows['FL7']
            self.FL6S = flows['FL5']
        elif current_node == self.num_nodes - 1:  # Bottom node
            flows['FL4'] = self.FL4S
            flows['FL6'] = self.FL6S
            flows['FL8'] = flows['FL4'] - flows['FL6'] + FL3
        else:  # Middle nodes
            flows['FL4'] = self.FL4S
            flows['FL6'] = self.FL6S
            flows['FL8'] = flows['FL4'] - flows['FL6'] + FL3
            if flows['FL8'] < 0:
                flows['FL5'] = -flows['FL8']
            else:
                flows['FL7'] = flows['FL8']
            self.FL4S = flows['FL7']
            self.FL6S = flows['FL5']
        
        return flows
    
    def energy_balance(self, T, t, flow_specs):
        """Calculate temperature derivatives for all nodes"""
        p = self.params
        rho_fl = p.get('rho_fl') or self.get_density(T)
        
        # Initialize external flows
        external_flows = []
        for i in range(self.num_nodes):
            node_flow = {'flow_in': 0.0, 'flow_out': 0.0, 'T_in': 0.0}
            for height, flow_rate, temp, name in flow_specs['inlets']:
                if i == self.get_node_at_height(height):
                    node_flow['flow_in'] += flow_rate
                    node_flow['T_in'] = temp + 273.15
            for height, flow_rate, name in flow_specs['outlets']:
                if i == self.get_node_at_height(height):
                    node_flow['flow_out'] += flow_rate
            external_flows.append(node_flow)
        
        dTdt = np.zeros(self.num_nodes)
        
        for i in range(self.num_nodes):
            # Get density (scalar or array)
            rho_fl_i = rho_fl if isinstance(rho_fl, float) else rho_fl[i]
            
            # Calculate flows
            node_flow = external_flows[i]
            node_flows = self.calculate_flows(node_flow, i)
            
            # Boundary conditions
            T_above = T[i-1] if i > 0 else T[i]
            T_below = T[i+1] if i < self.num_nodes-1 else T[i]
            delta_above = self.tank_height/self.num_nodes if i > 0 else 0
            delta_below = self.tank_height/self.num_nodes if i < self.num_nodes-1 else 0
            
            # Flow terms
            flow_term = 0.0
            if node_flow['flow_in'] > 0:
                flow_term += node_flow['flow_in'] * node_flow['T_in']
            if node_flow['flow_out'] > 0:
                flow_term -= node_flow['flow_out'] * T[i]
            flow_term += (node_flows['FL4'] * T_above + 
                         node_flows['FL5'] * T_below - 
                         node_flows['FL6'] * T[i] - 
                         node_flows['FL7'] * T[i])
            
            # Conduction
            k_eff = p['k_fl'] + p.get('delta_k_eff', 0)
            if delta_above > 0:
                conduction = k_eff * np.pi * (p['tank_diameter']/2)**2 * (T_above - T[i]) / delta_above
            else:
                conduction = 0
            if delta_below > 0:
                conduction += k_eff * np.pi * (p['tank_diameter']/2)**2 * (T[i] - T_below) / delta_below
            
            # Heat losses
            node_surface = np.pi * p['tank_diameter'] * (self.tank_height/self.num_nodes)
            losses = (-p['UA_i'] * node_surface * (T[i] - p['T_env']) - 
                     (1 - p['epsilon']) * p['UA_gfl'] * node_surface * (T[i] - p['T_gfl']))
            
            # Combine terms
            dTdt[i] = (p['C_fl'] * flow_term + conduction + losses) / (rho_fl_i * p['C_fl'] * self.node_volumes[i])
            
            # Add heaters
            if 'heater_positions' in p:
                for height, power, set_temp in p['heater_positions']:
                    if i == self.get_node_at_height(height) and T[i] < set_temp + 273.15:
                        dTdt[i] -= power / (rho_fl_i * p['C_fl'] * self.node_volumes[i])
            
            # Add heat exchangers with local flow rate
            for hx in self.heat_exchangers:
                if i == hx['node_idx']:
                    m_dot_local = abs(node_flows['FL4'] - node_flows['FL6'])
                    Q_hx, _, _ = hx['hx'].calculate_heat_transfer(T[i], m_dot_local)
                    print("Q_hx:", Q_hx)
                    dTdt[i] -= Q_hx / (rho_fl_i * p['C_fl'] * self.node_volumes[i])
        
        return dTdt
    
    def solve(self, t_span, flow_specs):
        """Solve the system over time"""
        sol = odeint(self.energy_balance, self.T, t_span, args=(flow_specs,))
        return sol
        
        
    
    def visualize_results(self, solution, t_span, flow_specs, save_path=None):
        """
        Enhanced visualization with heat exchanger performance metrics:
        - Outlet temperature of secondary fluid
        - Heat transfer rate in the heat exchanger
        """
        # Set style parameters with larger fonts
        plt.style.use('seaborn-v0_8-white')
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'font.size': 14,               # Base font size
            'axes.labelsize': 16,          # Axis labels
            'axes.titlesize': 18,          # Subplot titles
            'legend.fontsize': 14,         # Legend
            'xtick.labelsize': 14,         # X-axis ticks
            'ytick.labelsize': 14,         # Y-axis ticks
            'figure.titlesize': 20,        # Figure title
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'lines.linewidth': 2.5,        # Default line width
            'axes.linewidth': 2,           # Axis line width
            'grid.linewidth': 1.5,         # Grid line width
            'patch.linewidth': 1.5         # Patch/bar line width
        })
    
        # Convert K to °C for plotting
        solution_c = solution - 273.15
    
        # ================= FIGURE 1: TIME EVOLUTION =================
        fig1 = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 2, figure=fig1, height_ratios=[3, 1])
        
        # Main temperature plot
        ax1 = fig1.add_subplot(gs[0, :])
        
        # Find outlet node(s)
        outlet_nodes = []
        for height, flow_rate, name in flow_specs['outlets']:
            outlet_nodes.append(self.get_node_at_height(height))
        
        # Find heat exchanger nodes
        hx_nodes = [hx['node_idx'] for hx in self.heat_exchangers]
        
        # Select 8 nodes to plot, always including outlet and heat exchanger nodes
        num_display_nodes = 8
        nodes_to_plot = set(outlet_nodes + hx_nodes)  # Start with important nodes
        
        # Add evenly distributed nodes to reach total of 8
        remaining_nodes = num_display_nodes - len(nodes_to_plot)
        if remaining_nodes > 0:
            # Get evenly spaced nodes excluding already selected nodes
            all_nodes = set(range(self.num_nodes))
            available_nodes = list(all_nodes - nodes_to_plot)
            additional_nodes = np.linspace(0, len(available_nodes)-1, remaining_nodes, dtype=int)
            for idx in additional_nodes:
                nodes_to_plot.add(available_nodes[idx])
        
        # Convert to sorted list
        nodes_to_plot = sorted(nodes_to_plot)
        
        # Plot temperature vs time for selected nodes
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for i, color in zip(nodes_to_plot, colors):
            label = f'{self.node_heights[i]:.1f} m'
            if i in outlet_nodes:
                label += ' (Outlet)'  # Mark outlet nodes
            if i in hx_nodes:
                label += ' (HX)'      # Mark heat exchanger nodes
            ax1.plot(t_span/3600, solution_c[:, i], color=color, 
                    linewidth=3.5, label=label)
        
        ax1.set_xlabel('Time (hours)', fontsize=16)
        ax1.set_ylabel('Temperature (°C)', fontsize=16)
        ax1.set_title('Temperature Evolution at Different Heights', fontsize=20, pad=20)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Make legend bigger
        legend = ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                           title="Height", title_fontsize=14, fontsize=12)
        
        # Adjust spines and tick parameters
        ax1.tick_params(axis='both', which='major', length=10, width=2)
        for spine in ax1.spines.values():
            spine.set_linewidth(2)
        
        # Heat exchanger performance plot
        if self.heat_exchangers:
            ax1b = fig1.add_subplot(gs[1, :])
            
            # Calculate heat exchanger performance over time
            for hx in self.heat_exchangers:
                Q_hx = []
                T_out_secondary = []
                
                for t_idx in range(len(t_span)):
                    node_idx = hx['node_idx']
                    node_flow = {'flow_in': 0.0, 'flow_out': 0.0, 'T_in': 0.0}
                    for height, flow_rate, temp, name in flow_specs['inlets']:
                        if node_idx == self.get_node_at_height(height):
                            node_flow['flow_in'] += flow_rate
                    for height, flow_rate, name in flow_specs['outlets']:
                        if node_idx == self.get_node_at_height(height):
                            node_flow['flow_out'] += flow_rate
                    
                    flows = self.calculate_flows(node_flow, node_idx)
                    m_dot_local = abs(flows['FL4'] - flows['FL6'])
                    
                    # Get heat transfer and outlet temperature
                    Q, T_out, _ = hx['hx'].calculate_heat_transfer(solution[t_idx, node_idx], m_dot_local)
                    Q_hx.append(Q/1000)  # Convert to kW
                    T_out_secondary.append(T_out - 273.15)  # Convert to °C
                
                # Plot heat transfer rate
                ax1b.plot(t_span/3600, Q_hx, 'r-', linewidth=3.5, 
                         label=f'HX at {self.node_heights[hx["node_idx"]]:.1f}m: Heat Transfer')
                
                # Plot secondary outlet temperature
                ax1b.plot(t_span/3600, T_out_secondary, 'b--', linewidth=3.5,
                         label=f'HX at {self.node_heights[hx["node_idx"]]:.1f}m: Outlet Temp')
            
            ax1b.set_xlabel('Time (hours)', fontsize=16)
            ax1b.set_ylabel('Heat (kW) / Temp (°C)', fontsize=16)
            ax1b.set_title('Heat Exchanger Performance', fontsize=20, pad=20)
            ax1b.grid(True, linestyle='--', alpha=0.7)
            ax1b.legend(fontsize=12, loc='upper right')
            ax1b.tick_params(axis='both', which='major', length=10, width=2)
            for spine in ax1b.spines.values():
                spine.set_linewidth(2)
        
        fig1.tight_layout()
        
        if save_path:
            fig1.savefig(f"{save_path}_time_evolution.pdf", bbox_inches='tight', format='pdf')
    
        # ================= FIGURE 2: COMBINED PROFILE =================
        fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(20, 10), sharey=True)
        
        # Left subplot: Temperature profile
        final_idx = -1
        ax2.plot(solution_c[final_idx, :], self.node_heights, 
                color='blue', linewidth=4)
        
        ax2.set_xlabel('Temperature (°C)', fontsize=20)
        ax2.set_ylabel('Height (m)', fontsize=20)
        ax2.set_title('Final Temperature Profile', fontsize=20, pad=25)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.tick_params(axis='both', which='major', length=10, width=2)
        
        # Annotations with white background for better readability
        x_min, x_max = ax2.get_xlim()
        bbox_props = dict(boxstyle="round,pad=0.3", fc="w", ec="none", alpha=0.8)
        
        for height, flow_rate, temp, name in flow_specs['inlets']:
            ax2.axhline(y=height, color='blue', linestyle='--', alpha=0.7, linewidth=2.5)
            ax2.text(x_min + 0.5, height, 
                    f"{name}\nIn: {temp:.2f}°C\n{flow_rate:.2f} kg/s", 
                    color='blue', fontsize=20, fontweight='bold',
                    verticalalignment='center', 
                    horizontalalignment='left',
                    bbox=bbox_props)
        
        for height, flow_rate, name in flow_specs['outlets']:
            node_idx = self.get_node_at_height(height)
            outlet_temp = solution_c[final_idx, node_idx]
            ax2.axhline(y=height, color='red', linestyle='--', alpha=0.7, linewidth=2.5)
            ax2.text(x_max - 0.5, height, 
                    f"{name}\nOut: {outlet_temp:.2f}°C\n{flow_rate:.2f} kg/s", 
                    color='red', fontsize=20, fontweight='bold',
                    verticalalignment='center', 
                    horizontalalignment='right',
                    bbox=bbox_props)
        
        # Mark heat exchanger locations and show performance
        for hx in self.heat_exchangers:
            height = self.node_heights[hx['node_idx']]
            node_idx = hx['node_idx']
            
            node_flow = {'flow_in': 0.0, 'flow_out': 0.0, 'T_in': 0.0}
            for h, flow_rate, temp, name in flow_specs['inlets']:
                if node_idx == self.get_node_at_height(h):
                    node_flow['flow_in'] += flow_rate
            for h, flow_rate, name in flow_specs['outlets']:
                if node_idx == self.get_node_at_height(h):
                    node_flow['flow_out'] += flow_rate
            
            flows = self.calculate_flows(node_flow, node_idx)
            m_dot_local = abs(flows['FL4'] - flows['FL6'])
            Q_hx, T_out, _ = hx['hx'].calculate_heat_transfer(solution[final_idx, node_idx], m_dot_local)
            
            ax2.axhline(y=height, color='green', linestyle=':', alpha=0.7, linewidth=2.5)
            ax2.text(x_min + (x_max-x_min)/2, height, 
                    f"HX: {Q_hx/1000:.1f} kW\nOut: {T_out-273.15:.1f}°C", 
                    color='green', fontsize=20, fontweight='bold',
                    verticalalignment='center', 
                    horizontalalignment='center',
                    bbox=bbox_props)
        
        # Right subplot: Mass flow rates
        upward_flows = np.zeros(self.num_nodes)
        downward_flows = np.zeros(self.num_nodes)
        net_flows = np.zeros(self.num_nodes)
        
        for i in range(self.num_nodes):
            node_flow = {'flow_in': 0.0, 'flow_out': 0.0, 'T_in': 0.0}
            for height, flow_rate, temp, name in flow_specs['inlets']:
                if i == self.get_node_at_height(height):
                    node_flow['flow_in'] += flow_rate
            for height, flow_rate, name in flow_specs['outlets']:
                if i == self.get_node_at_height(height):
                    node_flow['flow_out'] += flow_rate
        
            flows = self.calculate_flows(node_flow, i)
            upward_flows[i] = flows['FL6']
            downward_flows[i] = flows['FL4']
            net_flows[i] = flows['FL8']
        
        # Plot flows with consistent styling
        ax3.plot(upward_flows, self.node_heights, 
                label="Upward", color='green', linestyle=':', linewidth=4)
        ax3.plot(downward_flows, self.node_heights, 
                label="Downward", color='orange', linestyle=':', linewidth=4)
        ax3.plot(net_flows, self.node_heights, 
                label="Net", color='blue', linestyle='-', linewidth=4)
        
        ax3.set_xlabel('Mass Flow Rate (kg/s)', fontsize=20)
        ax3.set_title('Mass Flow Distribution', fontsize=20, pad=25)
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.tick_params(axis='both', which='major', length=10, width=2)
        
        # Enhanced legend
        legend = ax3.legend(loc='upper right', framealpha=1, fontsize=20)
        for line in legend.get_lines():
            line.set_linewidth(4)
        
        # Adjust spines for both subplots
        for ax in [ax2, ax3]:
            for spine in ax.spines.values():
                spine.set_linewidth(2.5)
        
        fig2.tight_layout(pad=5)
        
        if save_path:
            fig2.savefig(f"{save_path}_combined_profile.pdf", bbox_inches='tight', format='pdf')
            
        # ================= FIGURE 3: TEMPERATURE PROFILES AT DIFFERENT TIMES =================
        fig3 = plt.figure(figsize=(12, 8))
        ax3 = fig3.add_subplot(111)
        
        # Select time indices to plot (initial, some intermediates, final)
        time_indices = [
            0,
            len(t_span) // 6,
            len(t_span) // 3,
            len(t_span) // 2,
            len(t_span) * 2 // 3,
            len(t_span) * 5 // 6,
            -1
        ]
        
        # Maak een aangepaste colormap van oranje naar rood
        orange_to_darkred = LinearSegmentedColormap.from_list(
            'orange_to_darkred',
            ['#FFDAB9', '#FF8C00', '#B22222']  # PeachPuff → DarkOrange → FireBrick
        )
        
        # Genereer kleuren op basis van het aantal tijdstappen
        colors = orange_to_darkred(np.linspace(0, 1, len(time_indices)))
        
        
        # Plot temperature profiles for selected times
        for idx, color in zip(time_indices, colors):
            time_label = f'{t_span[idx]:.1f} seconds'
            ax3.plot(solution_c[idx, :], self.node_heights, 
                     color=color, linewidth=3, label=time_label)
        
        ax3.set_xlabel('Temperature (°C)', fontsize=16)
        ax3.set_ylabel('Height (m)', fontsize=16)
        ax3.set_title('Temperature Profiles at Different Times', fontsize=20, pad=20)
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend(fontsize=14, title="Time", title_fontsize=14)
        
        # Mark inlets, outlets and heat exchangers
        for height, flow_rate, temp, name in flow_specs['inlets']:
            ax3.axhline(y=height, color='blue', linestyle='--', alpha=0.5, linewidth=2)
        
        for height, flow_rate, name in flow_specs['outlets']:
            ax3.axhline(y=height, color='red', linestyle='--', alpha=0.5, linewidth=2)
        
        for hx in self.heat_exchangers:
            ax3.axhline(y=self.node_heights[hx['node_idx']], 
                        color='green', linestyle=':', alpha=0.5, linewidth=2)
        
        # Adjust spines and tick parameters
        ax3.tick_params(axis='both', which='major', length=10, width=2)
        for spine in ax3.spines.values():
            spine.set_linewidth(2)
        
        fig3.tight_layout()
        
        if save_path:
            fig3.savefig(f"{save_path}_time_profiles.pdf", bbox_inches='tight', format='pdf')
        plt.show()

# Example usage
if __name__ == "__main__":
    num_nodes = 50
    H = 4.0
    tube_hx_params = {
    'type': 'tube',  # Specify this is a tube heat exchanger
    'height': 2.5,   # Installation height in the tank [m]
    'length': 3.0,   # Length of each tube [m]
    'diameter': 0.05,  # Diameter of tubes [m]
    'num_tubes': 10,  # Number of parallel tubes
    'U': 850,        # Overall heat transfer coefficient [W/m²K]
    'fluid': 'primary',  # 'primary' (tank fluid) or 'secondary' (external fluid)
    'm_dot_secondary': 2.5,  # Secondary fluid flow rate [kg/s]
    'Cp_secondary': 4186,    # Secondary fluid heat capacity [J/kgK]
    'T_in_HE': 60.0,         # Secondary fluid inlet temperature [°C]
    # Optional parameters:
    'effectiveness': None,   # If specified, fixes effectiveness (None for calculated)
    'name': "Primary Tube HX"  # Optional identifier
}
    plate_hx_params = {
    'type': 'plate',  # Specify this is a plate heat exchanger
    'height': 1.0,   # Installation height in the tank [m]
    'num_plates': 30,  # Number of plates
    'plate_width': 0.5,  # Width of each plate [m]
    'plate_height': 1.0,  # Height of each plate [m]
    'U': 850,        # Overall heat transfer coefficient [W/m²K]
    'fluid': 'secondary',  # 'primary' or 'secondary'
    'm_dot_secondary': 2.5,  # Secondary fluid flow rate [kg/s]
    'Cp_secondary': 4186,    # Secondary fluid heat capacity [J/kgK] (e.g., glycol mix)
    'T_in_HE': 60.0,         # Secondary fluid inlet temperature [°C]
    'flow_arrangement': 'counterflow',  # 'counterflow' or 'parallel'
    # Optional parameters:
    'effectiveness': None,   # If specified, fixes effectiveness
    'name': "Secondary Plate HX"  # Optional identifier
}     
    # Creëer een gestratificeerde initiële temperatuurverdeling
    node_heights = np.linspace(2.0, 0, num_nodes)  # Tankhoogte van 2m
    initial_T = np.where(node_heights > 1.0, 90, 70)  # 80°C boven 1m, 20°C eronder
    initial_T = 92
    params = {
        'tank_height': H,
        'tank_diameter': 2.0,
        'C_fl': 4186,
        'k_fl': 0.6,
        'delta_k_eff': 0.0,
        'UA_i': 0.0,
        'UA_gfl': 0.0,
        'epsilon': 0.5,
        'T_env': 20 + 273.15,
        'T_gfl': 15 + 273.15,
        'T_initial': initial_T,  
        'heat_exchangers': [plate_hx_params
            
            ]
        }

     
    tank = ThermalStorageTank(num_nodes, params)
    
    flow_specs = {
        'inlets': [
            (H, 5.0, 80.0, "Hot Inlet"),
            (0.0, 5.0, 40.0, "Cold Inlet")
        ],
        'outlets': [
            (3.0, 8.0, "Mixed Outlet"),
            (1.0, 2.0, "Mixed Outlet" )
        ]
    }
    
    t_span = np.linspace(0, 3*3600, 100)
    solution = tank.solve(t_span, flow_specs)
    tank.visualize_results(solution, t_span, flow_specs) 

    # Get outlet conditions (defaults to last time step)
    temp, flow = tank.get_outlet_conditions(solution, flow_specs, "Mixed Outlet")
    print(f"Outlet temperature: {temp:.2f}°C, Flow rate: {flow:.2f} kg/s")
    
    # Get at specific time index
    temp, flow = tank.get_outlet_conditions(solution, flow_specs, "Mixed Outlet", time_idx=50)
    print(f"At time index 50 - Temperature: {temp:.2f}°C, Flow rate: {flow:.2f} kg/s")

    T_out, m_dot, Q, effectiveness = tank.get_heat_exchanger_conditions(solution, flow_specs, hx_index=0, time_idx=50)
    print(f"Outlet Temp: {T_out:.2f}°C, Flow: {m_dot:.2f} kg/s, Heat Transfer: {Q/1000:.2f} kW, Effectiveness: {effectiveness*100:.2f}%")
