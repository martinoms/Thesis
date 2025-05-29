# -*- coding: utf-8 -*-
"""
Martijn Stynen & Senne Schepens
Centrale warmteverdeling van een industriële organische vergistingsinstallatie met behulp van een watervat

Finale 2D code
De vervolgcode (en dus finale code) is het 1D transient model.
Als programmeerhulp is in dit programma chatGPT gebruikt.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class HeatDistributionTank:
    def __init__(self, diameter, height, dx, dy, initial_temperature, heat_input, desired_temp_increase, 
                 rho=998.0, specific_heat_cap=4186, k=0.6, h_warm_factor=2.0, h_cool_factor=0.5):
        self.diameter = diameter
        self.height = height
        self.dx = float(dx)
        self.dy = float(dy)
        self.l = self.dx

        # Grid resolution
        self.N = int(height / self.dy)  # Number of rows
        self.M = int(diameter / self.dx)  # Number of columns

        # Physical properties
        self.rho = rho
        self.specific_heat_cap = specific_heat_cap
        self.k = k

        # Heat transfer coefficients (now as arrays)
        self.h_right = np.zeros((self.N, self.M), dtype=float)
        self.h_left = np.zeros((self.N, self.M), dtype=float)
        self.h_up = np.zeros((self.N, self.M), dtype=float)
        self.h_down = np.zeros((self.N, self.M), dtype=float)
        
        # Parameters for side coefficients near inputs/outputs
        self.h_near_input = 200.0  # High value near inputs
        self.h_near_output = 200.0  # Different value near outputs
        self.h_far_input = 0.0
        self.input_effect_radius = 3
        self.output_effect_radius = 3
        
        self.h_base = 10
        self.h_factor = 50

        # Heat input and losses
        self.Qdot_input = heat_input
        self.Qdot_input_grid = np.zeros((self.N, self.M))  # For distributed heat inputs
        self.dT = desired_temp_increase
        self.Qdot_loss_left = 0
        self.Qdot_loss_right = 0
        self.Qdot_loss_bottom = 0
        self.Qdot_loss_up = 0

        # Top layer volume and mass calculation
        self.Volume_top_layer = ((self.dx * (self.dy / 2)) * (self.M - 2) + 2 * ((self.dx / 2) * (self.dy / 2)))
        self.mass_top_layer = self.Volume_top_layer * self.rho
        
        # Initialize temperature grid
        self.T_initial = initial_temperature
        self.T_grid = np.full((self.N, self.M), self.T_initial, dtype=float)
        
        # Matrix initialization
        self.Size_of_matrix = self.M * self.N
        self.A = np.zeros((self.Size_of_matrix, self.Size_of_matrix), dtype=float)
        self.b = np.zeros(self.Size_of_matrix, dtype=float)
        
        # Mass inputs and outputs
        self.mass_inputs = {}  # Key: (row, col), Value: (mass_flow, Tin)
        self.mass_outputs = {}  # Key: (row, col), Value: mass_flow
        
        # Wall properties
        self.t = 0.005
        self.k_wall = 50  # steel
        self.U_wall = self.k_wall/self.t
        self.U_wall = 0
        self.T_amb = 20  # Ambient temperature
        self.A_side = self.dy
        self.A_tb = self.dx
        self.A_corner = (self.dx + self.dy)/2

        # Flow storage variables
        self.column_flows = np.zeros(self.M)  # Track flow for each column
        self.flow_down = np.zeros((self.N, self.M))
        self.flow_up = np.zeros((self.N, self.M))
        
        self.heat_exchangers = []  # Store heat exchanger data

    def thermal_expansion_coefficient(self, T):
        """Calculate the thermal expansion coefficient (beta) for water."""
        return 0.0002

    def density(self, T):
        """Calculate the density of water as a function of temperature."""
        return -3.784 * 0.001 * T**2 + 2.010 * T + 733.5

    def dynamic_viscosity(self, T):
        """Calculate the dynamic viscosity (mu) for water."""
        return 0.2271 * 0.000001 * T**2 - 0.1567 * 0.001 * T + 0.02743
    
    def kinematic_viscosity(self, T):
        """Calculate the kinematic viscosity (nu) for water."""
        mu = self.dynamic_viscosity(T)
        rho = self.density(T)
        return mu / rho

    def prandtl_number(self, T):
        """Calculate the Prandtl number (Pr) for water."""
        return 50000 / (T**2 + 155 * T + 3700)

    def grasshof_number(self, T_s, T_inf, L):
        """Calculate the Grashof number."""
        g = 9.81
        beta = self.thermal_expansion_coefficient(T_s)
        nu = self.kinematic_viscosity(T_s)
        return (g * beta * abs(T_s - T_inf) * L**3) / nu**2
    
    def reynolds_number(self, mass_flow, T):
        """Calculate Reynolds number based on mass flow and temperature."""
        mu = self.dynamic_viscosity(T)
        perimeter = self.diameter * np.pi
        Re = 4 * (mass_flow) / (perimeter * mu)
        return Re

    def nusselt_number(self, Gr, Pr):
        """Calculate the Nusselt number using the Churchill-Chu correlation."""
        Ra = Gr * Pr
        if Ra < 1e9:
            Nu = 0.68 + (0.67 * Ra**0.25) / (1 + (0.492 / Pr)**(9/16))**(4/9)
        else:
            Nu = (0.825 + (0.387 * Ra**(1/6)) / (1 + (0.492 / Pr)**(9/16))**(8/27))**2
        return Nu

    def calculate_heat_transfer_coefficient(self, i, j, T_mean):
        """
        Calculate heat transfer coefficients by comparing cell temperature to mean temperature.
        Returns h_up, h_down adjusted based on whether cell is warmer/cooler than average.
        """
        L = self.height
        T_cell = self.T_grid[i, j]
        Gr = self.grasshof_number(T_cell, T_mean, L)
        Pr = self.prandtl_number(T_cell)
        Nu = self.nusselt_number(Gr, Pr)
        h_base = (Nu * self.k) / L
        
        h_up = 0
        h_down = 0
        
        if T_cell > T_mean:
            h_up = h_base 
        elif T_cell < T_mean:
            h_down = h_base
        
        return h_up, h_down

    def update_side_coefficients(self):
        """Update h_left and h_right based on proximity to mass inputs and outputs,
        with enhanced mixing zone at right outlet (half-circle region)"""
        self.h_left.fill(self.h_far_input)
        self.h_right.fill(self.h_far_input)
        
        # Process mass inputs (left wall) - only enhance h_right
        for (i, j), (mass_flow, Tin) in self.mass_inputs.items():
            if j == 0:  # Left wall inputs
                for dj in range(min(self.input_effect_radius, self.M)):
                    current_j = j + dj
                    if current_j < self.M:
                        distance_factor = 1.0 / (1.0+ dj)
                        self.h_left[i, current_j] = max(
                            self.h_left[i, current_j],
                            self.h_near_input * distance_factor
                        )
        
        # Process mass outputs (right wall) - create mixing zone
        for (i, j), mass_flow in self.mass_outputs.items():
            if j == self.M - 1:  # Right wall outputs
                # Define mixing zone as half-circle at right outlet
                radius = min(self.M//3, self.N//2)  # Radius of mixing zone
                center_i = i
                center_j = j
                
                # Enhance coefficients in half-circle region
                for di in range(-radius, radius+1):
                    for dj in range(-radius, 0+1):  # Only right half (dj <= 0)
                        current_i = center_i + di
                        current_j = center_j + dj
                        
                        # Check if within grid bounds
                        if 0 <= current_i < self.N and 0 <= current_j < self.M:
                            # Calculate distance from outlet point
                            distance = np.sqrt(di**2 + dj**2)
                            
                            if distance <= radius:
                                # Enhance all coefficients in mixing zone
                                distance_factor = 10.0 - (distance / radius)  # 1 at center, 0 at edge
                                enhancement = 1000.0  # Strong mixing in this zone
                                
                                # Apply enhanced coefficients
                                self.h_up[current_i, current_j] = max(
                                    self.h_up[current_i, current_j],
                                    enhancement * distance_factor
                                )
                                self.h_down[current_i, current_j] = max(
                                    self.h_down[current_i, current_j],
                                    enhancement * distance_factor
                                )
                                self.h_left[current_i, current_j] = max(
                                    self.h_left[current_i, current_j],
                                    enhancement * distance_factor
                                )
                                # Don't enhance h_right at right edge (wall)
                                if current_j < self.M - 1:
                                    self.h_right[current_i, current_j] = max(
                                        self.h_right[current_i, current_j],
                                        enhancement * distance_factor
                                )
    def get_index(self, i, j):
        return i * self.M + j

    def set_mass_input_by_height(self, height, total_mass_flow, Tin):
        """Set mass inputs at a specific height, distributed across the diameter"""
        if height < 0 or height > self.height:
            raise ValueError("Height must be between 0 and tank height")
            
        # Find closest row index
        row = int(np.round(height / self.height * (self.N - 1)))
        
        # Distribute flow evenly across all columns at this height
        flow_per_column = total_mass_flow / self.M
        for col in range(self.M):
            self.set_mass_input(row, col, flow_per_column, Tin)
    
    def set_mass_output_by_height(self, height, total_mass_flow):
        """Set mass outputs at a specific height, distributed across the diameter"""
        if height < 0 or height > self.height:
            raise ValueError("Height must be between 0 and tank height")
            
        # Find closest row index
        row = int(np.round(height / self.height * (self.N - 1)))
        
        # Distribute flow evenly across all columns at this height
        flow_per_column = total_mass_flow / self.M
        for col in range(self.M):
            self.set_mass_output(row, col, flow_per_column)
            
    def set_mass_input(self, row, column, mass_flow, Tin):
        """Set a mass input at a specific grid location."""
        if 0 <= row < self.N and 0 <= column < self.M:
            self.mass_inputs[(row, column)] = (float(mass_flow), float(Tin))
        else:
            raise ValueError("Row or column index out of bounds.")
        return self.mass_inputs
    
    def set_mass_output(self, row, column, mass_flow):
        """Set a mass output at a specific grid location."""
        if 0 <= row < self.N and 0 <= column < self.M:
            self.mass_outputs[(row, column)] = float(mass_flow)
        else:
            raise ValueError("Row or column index out of bounds.")
        return self.mass_outputs
    
    def set_heat_input_by_height(self, height, heat_power):
        """Set heat input at a specific height (in meters) with given power in Watts"""
        if height < 0 or height > self.height:
            raise ValueError("Height must be between 0 and tank height")
            
        # Find closest row index
        row = int(np.round(height / self.height * (self.N - 1)))
        
        # Distribute heat evenly across all columns at this height
        self.Qdot_input_grid[row, :] += heat_power / self.M
    
    def add_heat_exchanger(self, height, length, flow_rate, T_in, effectiveness):
        """Adds a heat exchanger and returns initial T_out estimate."""
        # Find affected rows
        start_row = int(np.round((height - length/2) / self.height * (self.N - 1)))
        end_row = int(np.round((height + length/2) / self.height * (self.N - 1)))
        start_row = max(0, start_row)
        end_row = min(self.N - 1, end_row)
    
        # Calculate average tank temperature in the region
        T_tank = np.mean(self.T_grid[start_row:end_row+1, :])
        
        # Heat exchanger performance
        T_out = T_in + effectiveness * (T_tank - T_in)
        heat_power = flow_rate * self.specific_heat_cap * (T_out - T_in)
    
        # Apply heat to the affected rows
        for row in range(start_row, end_row+1):
            self.Qdot_input_grid[row, :] += heat_power / (end_row - start_row + 1) / self.M
    
        # Store heat exchanger data for later retrieval
        self.heat_exchangers.append({
            'height': height,
            'length': length,
            'flow_rate': flow_rate,
            'T_in': T_in,
            'effectiveness': effectiveness,
            'start_row': start_row,
            'end_row': end_row,
            'initial_T_out': T_out  # Initial estimate
        })
        
        return T_out  # Return initial estimate
    
    def get_heat_exchanger_outlet_temp(self, index=0):
        """Returns the updated outlet temperature after solving the system.
        Args:
            index (int): Which heat exchanger to check (default: first one added).
        Returns:
            float: Outlet temperature (°C) based on final tank temperatures.
        """
        if not self.heat_exchangers or index >= len(self.heat_exchangers):
            raise ValueError("No heat exchanger found at the specified index.")
        
        hx = self.heat_exchangers[index]
        start_row, end_row = hx['start_row'], hx['end_row']
        
        # Recompute with final tank temperatures
        T_tank_final = np.mean(self.T_grid[start_row:end_row+1, :])
        T_out_final = hx['T_in'] + hx['effectiveness'] * (T_tank_final - hx['T_in'])
        
        return T_out_final

    def setup_boundary_conditions(self):
        """Apply boundary conditions to the temperature grid."""
        # Original top layer heating (optional)
        for j in range(self.M):
            self.T_grid[0, j] += self.Qdot_input / (self.mass_top_layer * self.specific_heat_cap)

    def calculate_flows(self, external_flow, current_node, current_column, column_flow, next_FL4, next_FL6):
        """Calculate vertical-only flows within a column (no cross-column interference)."""
        
        # Update the cumulative flow for this column
        node_net_flow = external_flow['flow_in'] - external_flow['flow_out']
        column_flow += node_net_flow
    
        flows = {
            'FL4': next_FL4,      # Flow from above
            'FL5': 0.0,           # Flow from below
            'FL6': next_FL6,      # Flow to above
            'FL7': 0.0,           # Flow to below
            'FL8': column_flow    # Net flow in column
        }
    
        # Determine direction of vertical flow
        if current_node == 0:
            flows['FL7'] = max(0, column_flow)  # Top node: only downward
        elif current_node == self.N - 1:
            flows['FL5'] = max(0, -column_flow)  # Bottom node: only upward
        else:
            if column_flow > 0: 
                flows['FL7'] = column_flow  # Downward
            elif column_flow < 0:
                flows['FL5'] = -column_flow  # Upward
    
        return flows, column_flow, flows['FL7'], flows['FL5']  # Also return next FL4, FL6

    def check_mass_conservation(self):
        total_input = sum(flow for (flow, _) in self.mass_inputs.values())
        total_output = sum(flow for flow in self.mass_outputs.values())
        return abs(total_input - total_output) < 1e-6

    def fill_matrices(self):
        """Matrix assembly with improved vertical flow handling"""
        self.update_side_coefficients()
       
        # Initialize flow tracking for each column
        column_flows = [0.0] * self.M  # Net flow for each column
        
        for j in range(self.M):
            # Reset vertical flows for this column
            flow_down = 0.0  # Flow coming from above
            flow_up = 0.0    # Flow coming from below
            
            for i in range(self.N):
                # Initialize external flow parameters
                external_flow_in = 0.0
                external_flow_out = 0.0
                external_T_in = self.T_initial
                h_up, h_down = self.h_up[i, j], self.h_down[i, j]
                h_up, h_down = 0, 0
                # Check for mass inputs/outputs at this node
                if (i, j) in self.mass_inputs:
                    external_flow_in, external_T_in = self.mass_inputs[(i, j)]
                if (i, j) in self.mass_outputs:
                    external_flow_out = self.mass_outputs[(i, j)]
                
                # Update column net flow
                column_flows[j] += (external_flow_in - external_flow_out)
                
                # Determine vertical flows
                if i == 0:  # Top node
                    # Only allow downward flow if positive net flow
                    flow_down = 0.0
                    flow_up = 0.0
                elif i == self.N - 1:  # Bottom node
                    # Only allow upward flow if negative net flow
                    flow_up = max(0, -column_flows[j])
                    flow_down = 0.0
                else:  # Middle nodes
                    if column_flows[j] > 0:
                        flow_down = column_flows[j]
                        flow_up = 0.0
                    else:
                        flow_up = -column_flows[j]
                        flow_down = 0.0

                # Get heat transfer coefficients
                #h_up, h_down = 0, 0  # Can be calculated if needed
                
                #h_left = self.h_left[i, j]
                #h_right = self.h_right[i, j]
                
                # Get heat input for this cell
                heat_input = self.Qdot_input_grid[i,j] if hasattr(self, 'Qdot_input_grid') else 0
                
                # Initialize neighbor coefficients
                T_down = T_left = T_right = T_up = 0
                Q_dot_loss = 0
               
                h_left = 0
                h_right = 0
                # Boundary conditions$0.1
                # Bottom-left corner
                if i == self.N - 1 and j == 0:
                    T_up = h_up * (self.dx / 2) + self.k * (self.dx/2)/ (self.dy)
                    T_right = h_right * (self.dy / 2) + self.k * (self.dy/2) / (self.dx)
                    Q_dot_loss = self.U_wall * self.A_corner * self.T_amb
    
                # Bottom-right corner
                elif i == self.N - 1 and j == self.M - 1:
                    T_up = h_up * (self.dx/2) + self.k * (self.dx/2) / (self.dy)
                    T_left = h_left * (self.dy) + self.k * (self.dy/2) / (self.dx)
                    Q_dot_loss = self.U_wall * self.A_corner * self.T_amb
    
                # Top-left corner
                elif (i == 0) and (j == 0):
                    T_down = h_down * (self.dx / 2) + self.k * self.dx / (2*self.dy)
                    T_right = h_right * (self.dy / 2) + self.k * self.dy / (2*self.dx)
                    Q_dot_loss = self.U_wall * self.A_corner * self.T_amb
                    
                # Top-right corner
                elif i == 0 and j == self.M - 1:
                    T_down = h_down * (self.dx / 2) + self.k * self.dx / (2* self.dy)
                    T_left = h_left * (self.dy / 2) + self.k * self.dy / (2*self.dx)
                    Q_dot_loss = self.U_wall * self.A_corner * self.T_amb
                
                # Top layer
                elif i == 0:
                    T_down = h_down * self.dx + self.k * self.dx / self.dy
                    T_left = h_left * (self.dy / 2) + self.k * self.dy / (2 * self.dx)
                    T_right = h_right * (self.dy / 2) + self.k * self.dy / (2 * self.dx)
                    Q_dot_loss = self.U_wall * self.A_tb * self.T_amb
        
                # Bottom layer
                elif i == self.N - 1:
                    T_up = h_up * self.dx + self.k * self.dx / self.dy
                    T_left = h_left * (self.dy / 2) + self.k * self.dy / (2 * self.dx)
                    T_right = h_right * (self.dy / 2) + self.k * self.dy / (2 * self.dx)
                    Q_dot_loss = self.U_wall * self.A_tb * self.T_amb
        
                # Left edge
                elif j == 0 and 0 < i < self.N - 1:
                    T_up = h_up * (self.dx / 2) + self.k * self.dx / (2 * self.dy)
                    T_right = h_right * self.dy + self.k * self.dy / self.dx
                    T_down = h_down * (self.dx / 2) + self.k * self.dx / (2 * self.dy)
                    Q_dot_loss = self.U_wall * self.A_side * self.T_amb
         
                # Right edge
                elif j == self.M - 1 and 0 < i < self.N - 1:
                    T_up = h_up * (self.dx / 2) + self.k * self.dx / (2 * self.dy)
                    T_left = h_left * self.dy + self.k * self.dy / self.dx
                    T_down = h_down * (self.dx / 2) + self.k * self.dx / (2 * self.dy)
                    Q_dot_loss = self.U_wall * self.A_side * self.T_amb
             
                # Internal nodes
                elif 0 < i < self.N - 1 and 0 < j < self.M - 1:
                    T_up = h_up * self.dx + self.k * self.dx / self.dy
                    T_left = h_left * self.dy + self.k * self.dy / self.dx
                    T_down = h_down * self.dx + self.k * self.dx / self.dy
                    T_right = h_right * self.dy + self.k * self.dy / self.dx
                
                # Self term
                T_self = -(T_up + T_left + T_down + T_right)
                T_self -= external_flow_in * self.specific_heat_cap
                T_self -= flow_up * self.specific_heat_cap
                T_self -= flow_down * self.specific_heat_cap
                
                # Matrix assembly
                index = self.get_index(i, j)
                self.A[index, index] = T_self
                self.b[index] = (-external_flow_in * self.specific_heat_cap * external_T_in 
                                - Q_dot_loss 
                                + heat_input)
                
                # Neighbor coefficients
                if j > 0:
                    self.A[index, self.get_index(i, j-1)] = T_left
                if j < self.M - 1:
                    self.A[index, self.get_index(i, j+1)] = T_right
                if i < self.N - 1:
                    self.A[index, self.get_index(i+1, j)] = T_down + flow_up * self.specific_heat_cap
                if i > 0:
                    self.A[index, self.get_index(i-1, j)] = T_up + flow_down * self.specific_heat_cap
        
        return self.A, self.b

    def solve_temperature_distribution(self, max_iter=20, tolerance=1e-3):
        """Solve the temperature distribution iteratively."""
        h_up_history = []
        h_down_history = []
        
        for iteration in range(max_iter):
            print(f"Iteration {iteration + 1}/{max_iter}")
            
            # Calculate mean temperature of the tank
            T_mean = np.mean(self.T_grid)
            
            # Update convection coefficients based on mean temperature
            for i in range(self.N):
                for j in range(self.M):
                    self.h_up[i,j], self.h_down[i,j] = self.calculate_heat_transfer_coefficient(i, j, T_mean)
            
            h_up_history.append(np.mean(self.h_up, axis=1))
            h_down_history.append(np.mean(self.h_down, axis=1))
            
            # Fill matrices and solve
            self.A.fill(0)
            self.b.fill(0)
            self.fill_matrices()
            x = np.linalg.solve(self.A, self.b)
            
            # Calculate change in temperature for convergence check
            new_T_grid = np.copy(self.T_grid)
            for i in range(self.N):
                new_T_grid[i] = x[i * self.M:(i + 1) * self.M]
            
            delta_T = np.max(np.abs(new_T_grid - self.T_grid))
            if delta_T < tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break
                
            self.T_grid = new_T_grid
        
        return x, self.T_grid, h_up_history, h_down_history

    def visualize_temperature_grid(self, save_path=None):
        """Visualize the temperature distribution as a heatmap with improved styling"""
        plt.style.use('seaborn-v0_8-white')
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'font.size': 10,
            'axes.labelsize': 10,
            'axes.titlesize': 12,
            'legend.fontsize': 9,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'figure.dpi': 300,
            'savefig.dpi': 300
        })
        
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Create heatmap with better color scaling
        im = ax.imshow(self.T_grid, cmap='inferno', origin='lower', aspect='auto',
                       extent=[0, self.diameter, 0, self.height])
        
        # Add colorbar with proper label
        cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
        cbar.set_label('Temperature (°C)', rotation=270, labelpad=15)
        
        # Set labels and title
        ax.set_xlabel('Width (m)')
        ax.set_ylabel('Height (m)')
        ax.set_title('Temperature Distribution in Storage Tank')
        
        # Add grid lines
        ax.grid(False)
        
        # Adjust layout
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', format='pdf')
        plt.show()
    
    def visualize_heat_inputs(self, save_path=None):
        """Show where heat inputs are applied in the tank"""
        if not hasattr(self, 'Qdot_input_grid') or np.max(self.Qdot_input_grid) == 0:
            print("No heat inputs defined")
            return
            
        plt.style.use('seaborn-v0_8-white')
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'font.size': 10,
            'axes.labelsize': 10,
            'axes.titlesize': 12,
            'legend.fontsize': 9,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'figure.dpi': 300,
            'savefig.dpi': 300
        })
        
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Create custom colormap for heat inputs
        cmap = LinearSegmentedColormap.from_list('heat_inputs', ['white', 'red'])
        
        # Plot heat input locations
        im = ax.imshow(self.Qdot_input_grid, cmap=cmap, origin='lower', aspect='auto',
                      extent=[0, self.diameter, 0, self.height], vmin=0)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
        cbar.set_label('Heat Input (W)', rotation=270, labelpad=15)
        
        # Set labels and title
        ax.set_xlabel('Width (m)')
        ax.set_ylabel('Height (m)')
        ax.set_title('Heat Input Locations in Tank')
        
        # Add grid
        ax.grid(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', format='pdf')
        plt.show()
    
    def plot_temperature_vs_height(self, save_path=None):
        """Plot the average temperature at each height with improved styling,
        showing total mass flows at each height level"""
        plt.style.use('seaborn-v0_8-white')
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'font.size': 10,
            'axes.labelsize': 10,
            'axes.titlesize': 12,
            'legend.fontsize': 9,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'figure.dpi': 300,
            'savefig.dpi': 300
        })
        
        avg_temperature_per_height = np.mean(self.T_grid, axis=1)
        height_points = np.linspace(0, self.height, self.N)
        
        # Calculate total input/output flows at each height
        input_flows = np.zeros(self.N)
        input_temps = np.zeros(self.N)
        output_flows = np.zeros(self.N)
        
        for (i, j), (flow, Tin) in self.mass_inputs.items():
            input_flows[i] += flow
            input_temps[i] = Tin  # Will be overwritten if multiple inputs at same height
            
        for (i, j), flow in self.mass_outputs.items():
            output_flows[i] += flow
        
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Main plot with line
        ax.plot(avg_temperature_per_height, height_points, 
                color='#1f77b4', linewidth=1.5, label='Average Temperature')
        
        # Add labels and grid
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Height (m)')
        ax.set_title('Vertical Temperature Profile')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Mark input/output heights
        for i in range(self.N):
            if input_flows[i] > 0:
                ax.axhline(y=height_points[i], color='blue', linestyle='--', alpha=0.5, linewidth=0.8)
                # ax.text(np.min(avg_temperature_per_height)+1, height_points[i], 
                #         f"In: {input_flows[i]:.1f} kg/s\nT={input_temps[i]:.1f}°C", 
                #         color='blue', fontsize=8,
                #         verticalalignment='center')
            
            if output_flows[i] > 0:
                outlet_temp = np.mean(self.T_grid[i, :])  # Average across width
                ax.axhline(y=height_points[i], color='red', linestyle='--', alpha=0.5, linewidth=0.8)
                # ax.text(np.max(avg_temperature_per_height)-1, height_points[i], 
                #         f"Out: {output_flows[i]:.1f} kg/s\nT={outlet_temp:.1f}°C", 
                #         color='red', fontsize=8,
                #         verticalalignment='center',
                #         horizontalalignment='right')
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Adjust spines
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', format='pdf')
        plt.show()
    
    def plot_flows_vs_height(self, save_path=None):
        """Plot the vertical flows at each height showing total flows across the width"""
        plt.style.use('seaborn-v0_8-white')
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'font.size': 10,
            'axes.labelsize': 10,
            'axes.titlesize': 12,
            'legend.fontsize': 9,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'figure.dpi': 300,
            'savefig.dpi': 300
        })
        
        # Initialize flow arrays
        height_points = np.linspace(0, self.height, self.N)
        total_downward = np.zeros(self.N)
        total_upward = np.zeros(self.N)
        total_net = np.zeros(self.N)
        
        # Calculate external flows per height
        external_flows = np.zeros((self.N, self.M))
        for (i,j), (flow, _) in self.mass_inputs.items():
            external_flows[i,j] += flow
        for (i,j), flow in self.mass_outputs.items():
            external_flows[i,j] -= flow
        
        # Calculate column-wise flows
        for j in range(self.M):
            column_flow = 0.0
            for i in range(self.N):
                column_flow += external_flows[i,j]
                
                if i == 0:  # Top node
                    total_downward[i] += max(0, column_flow)
                elif i == self.N-1:  # Bottom node
                    total_upward[i] += max(0, -column_flow)
                else:  # Middle nodes
                    if column_flow > 0:
                        total_downward[i] += column_flow
                    else:
                        total_upward[i] += -column_flow
        
        # Calculate net flows
        total_net = total_downward - total_upward
        
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Plot total flows with improved styling
        ax.plot(total_downward, height_points, 
                label=f"Downward (max={total_downward.max():.2f} kg/s)", 
                color='#ff7f0e', linestyle='--', linewidth=1.5)
        ax.plot(total_upward, height_points, 
                label=f"Upward (max={total_upward.max():.2f} kg/s)",
                color='#2ca02c', linestyle='--', linewidth=1.5)
        ax.plot(total_net, height_points, 
                label=f"Net flow (max={abs(total_net).max():.2f} kg/s)",
                color='#1f77b4', linestyle='-', linewidth=2)
        
        # Add markers for input/output locations
        input_heights = set()
        output_heights = set()
        
        for (i,j), _ in self.mass_inputs.items():
            input_heights.add(height_points[i])
        for (i,j), _ in self.mass_outputs.items():
            output_heights.add(height_points[i])
        
        for h in input_heights:
            ax.axhline(h, color='blue', linestyle=':', alpha=0.3, linewidth=0.8)
        for h in output_heights:
            ax.axhline(h, color='red', linestyle=':', alpha=0.3, linewidth=0.8)
        
        # Add labels and title
        ax.set_xlabel('Total Mass Flow Rate (kg/s)')
        ax.set_ylabel('Height (m)')
        ax.set_title('Total Vertical Flow Distribution')
        ax.grid(True, linestyle='--', alpha=0.4)
        
        # Add legend with max flow values
        ax.legend()
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', format='pdf')
        plt.show()
    def plot_heat_transfer_coefficients(self, h_up_history, h_down_history, save_path=None):
        """Plot the evolution of heat transfer coefficients with publication quality."""
        plt.style.use('seaborn-v0_8-white')
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'font.size': 10,
            'axes.labelsize': 10,
            'axes.titlesize': 12,
            'legend.fontsize': 9,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'figure.dpi': 300,
            'savefig.dpi': 300
        })
        
        height_points = np.linspace(self.height, 0, self.N)
        
        fig, ax = plt.subplots(figsize=(5, 5))
        
        # Plot initial and final states clearly
        ax.plot(h_up_history[0], height_points, 'b-', alpha=0.5, linewidth=1, 
                label='Upward (initial)')
        ax.plot(h_down_history[0], height_points, 'r-', alpha=0.5, linewidth=1, 
                label='Downward (initial)')
        
        ax.plot(h_up_history[-1], height_points, 'b--', linewidth=1.5, 
                label='Upward (final)')
        ax.plot(h_down_history[-1], height_points, 'r--', linewidth=1.5, 
                label='Downward (final)')
        
        # Add intermediate states with transparency
        for h_up, h_down in zip(h_up_history[1:-1], h_down_history[1:-1]):
            ax.plot(h_up, height_points, 'b-', alpha=0.1, linewidth=0.5)
            ax.plot(h_down, height_points, 'r-', alpha=0.1, linewidth=0.5)
        
        # Add labels and legend
        ax.set_xlabel('Heat Transfer Coefficient (W/m²K)')
        ax.set_ylabel('Height (m)')
        ax.set_title('Evolution of Convection Coefficients')
        ax.legend(frameon=True, framealpha=1, edgecolor='black')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Adjust spines
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', format='pdf')
        plt.show()

    def plot_side_coefficients(self, save_path=None):
        """Visualize h_left and h_right coefficients with publication quality"""
        plt.style.use('seaborn-v0_8-white')
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'font.size': 10,
            'axes.labelsize': 10,
            'axes.titlesize': 12,
            'legend.fontsize': 9,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'figure.dpi': 300,
            'savefig.dpi': 300
        })
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        
        # Plot h_left
        im1 = ax1.imshow(self.h_left, cmap='viridis', origin='lower', aspect='auto',
                        extent=[0, self.diameter, 0, self.height])
        ax1.set_title('Left Side Coefficients (h_left)')
        ax1.set_xlabel('Width (m)')
        ax1.set_ylabel('Height (m)')
        fig.colorbar(im1, ax=ax1, label='Heat Transfer Coefficient (W/m²K)')
        
        # Plot h_right
        im2 = ax2.imshow(self.h_right, cmap='viridis', origin='lower', aspect='auto',
                         extent=[0, self.diameter, 0, self.height])
        ax2.set_title('Right Side Coefficients (h_right)')
        ax2.set_xlabel('Width (m)')
        fig.colorbar(im2, ax=ax2, label='Heat Transfer Coefficient (W/m²K)')
        
        # Mark input/output locations
        for ax in (ax1, ax2):
            # Mark inputs
            for (i,j), (flow, Tin) in self.mass_inputs.items():
                y_pos = i * self.dy
                x_pos = j * self.dx
                ax.plot(x_pos, y_pos, 'bo', markersize=5, alpha=0.7)
            
            # Mark outputs
            for (i,j), flow in self.mass_outputs.items():
                y_pos = i * self.dy
                x_pos = j * self.dx
                ax.plot(x_pos, y_pos, 'rx', markersize=6, alpha=0.7)
        
        # Add legend
        ax1.plot([], [], 'bo', label='Mass Input')
        ax1.plot([], [], 'rx', label='Mass Output')
        ax1.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', format='pdf')
        plt.show()
# Tank setup
height = 1
diameter = 1
dx = diameter/10
dy = 0.01

tank = HeatDistributionTank(
    diameter=diameter,
    height=height, 
    dx=dx,
    dy=dy,
    initial_temperature=20,
    heat_input=0,
    desired_temp_increase=10
)



# Set inputs/outputs by height (in meters)
tank.set_mass_input_by_height(height=1.0, total_mass_flow=10, Tin=90)  # Top inputs
#tank.set_mass_input_by_height(height=0.5, total_mass_flow=10, Tin=80)  # Middle inputs
tank.set_mass_input_by_height(height=0.0, total_mass_flow=10, Tin=50)  # Bottom inputs

# Set outputs by height
#tank.set_mass_output_by_height(height=1.0, total_mass_flow=10)  # Lower outputs
#tank.set_mass_output_by_height(height=1.0, total_mass_flow=10)  # Lower outputs
tank.set_mass_output_by_height(height=0.5, total_mass_flow=20)  # Upper outputs

# Set heat exchangers at different heights (in Watts)
#tank.set_heat_input_by_height(height=0.2, heat_power=100000)  # Lower heat exchanger
#tank.set_heat_input_by_height(height=0.7, heat_power=3000)  # Upper heat exchanger

#height, length, mass flow, Tin, effectiveness
T_heat = tank.add_heat_exchanger(height=0.05,length= 0.2,flow_rate= 20, T_in= 30, effectiveness=1)


# Solve
sol, T_grid, h_up, h_down = tank.solve_temperature_distribution(max_iter=200)
tank.visualize_temperature_grid()
tank.plot_temperature_vs_height()
tank.plot_flows_vs_height()
tank.plot_heat_transfer_coefficients(h_up, h_down)
#tank.plot_horizontal_convection()
tank.plot_side_coefficients()

# Get the final outlet temperature
final_T_out = tank.get_heat_exchanger_outlet_temp(index=0)
print(f"Final T_out after solving: {final_T_out:.2f} °C")