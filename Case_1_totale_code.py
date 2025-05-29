"""
Martijn Stynen & Senne Schepens
Centrale warmteverdeling van een industriële organische vergistingsinstallatie met behulp van een watervat

Finale code case 1
Als programmeerhulp is in dit programma chatGPT gebruikt.
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



        
class Case1:
    def __init__(self):
        self.num_eq = 8
        self.num_vars = 8
        self.variables = sp.symbols(f'x1:{self.num_vars+1}')
        self.equations = []
        self.A = None
        self.b = None
        self.augmented_matrix = None
        self.solution = None
        self.free_vars = None
        self.init_constants()
        self.create_equations()
        self.convert_to_matrix()
        self.gaussian_elimination()
        self.back_substitution()
        self.calculate_mass_flows()
        

    def init_constants(self):
        self.Thot = 90
        self.cp = 4186
        
        self.T1i = 65
        self.T2i = 55
        self.T3i = 45
        
        self.T1o = self.Thot
        self.T2o = 80 #output temperatuur tak 2
        self.T3o = 70 #output temperatuur tak 3
        
        ####∟CHECK DE BEREKENINGEN MET DE DEBIETE, VOLGENS MIJ KLOPT ER IETS NIET
        self.mdot1 = 60 #massadebiet tak 1 (aan de output)
        self.mdot2 = 20  #massadebiet tak 2
        self.mdot3 = 5  #massadebiet tak 3
        self.mdot_tot = self.mdot1 + self.mdot2 + self.mdot3
        
        #range of x8 (cold flow of output 3)
        #self.x8 = 24.44444444 #run eerst de code om te weten welke deze waarde is, deze wordt berekend bij plot_....
        self.x8_min = 0
        self.x8_max = self.mdot3
        
        self.Tmeng = (self.mdot1*self.T1i + self.mdot2*self.T2i + self.mdot3*self.T3i) / (self.mdot1 + self.mdot2 + self.mdot3)
        self.Tcold = self.Tmeng
        
        self.factor2 = (self.T2o - self.Thot) / (self.Tcold - self.T2o) + 1 # +1 omdat de vergelijking bestaat uit de vorm mdot_h2 + ... + factor1*mdot_h2 + ... = ...
        self.factor3 = (self.T3o - self.Thot) / (self.Tcold - self.T3o) + 1
        
        self.tolerance = 0.01         
        self.toleranceT2 = self.tolerance
        self.toleranceT3 = self.tolerance
            
        # if self.mdot2 < 0.0000001:
        #     self.toleranceT2 = 10
        #     self.toleranceT3 = 10
        # elif self.mdot3 < 0.0000001:
        #     self.toleranceT2 = 10
        #     self.toleranceT3 = 10            
        
        self.resolution = 1000

    def create_equations(self):
        eqs = [
            "-1*x1 + 1*x2 + 1*x3 + 1*x4 + 0*x5 + 0*x6 + 0*x7 + 0*x8 = 0",
            "0*x1 + 0*x2 + 0*x3 + 0*x4 + -1*x5 + 1*x6 + 1*x7 + 1*x8 = 0",
            f"0*x1 + 1*x2 + 0*x3 + 0*x4 + 0*x5 + 1*x6 + 0*x7 + 0*x8 = {self.mdot1}",
            f"0*x1 + 0*x2 + 1*x3 + 0*x4 + 0*x5 + 0*x6 + 1*x7 + 0*x8 = {self.mdot2}",
            f"0*x1 + 0*x2 + 0*x3 + 1*x4 + 0*x5 + 0*x6 + 0*x7 + 1*x8 = {self.mdot3}",
            f"1*x1 + 0*x2 + 0*x3 + 0*x4 + 1*x5 + 0*x6 + 0*x7 + 0*x8 = {self.mdot_tot}",
            f"0*x1 + 1*x2 + {self.factor2}*x3 + {self.factor3}*x4 + 0*x5 + 0*x6 + 0*x7 + 0*x8 = {self.mdot_tot}",
            "0*x1 + 0*x2 + 0*x3 + 0*x4 + 0*x5 + 1*x6 + 0*x7 + 0*x8 = 0"
        ]
        self.equations = eqs

    def convert_to_matrix(self):
        A, b = [], []
        for eq_str in self.equations:
            left, right = eq_str.split('=')
            left_expr = sp.sympify(left.strip(), locals={str(var): var for var in self.variables})
            right_expr = sp.sympify(right.strip(), locals={str(var): var for var in self.variables})
            coeffs = [left_expr.coeff(var) for var in self.variables]
            A.append(coeffs)
            b.append(float(right_expr))
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.augmented_matrix = np.column_stack((self.A, self.b))

    def gaussian_elimination(self):
        n, m = self.augmented_matrix.shape
        for i in range(min(n, m - 1)):
            if self.augmented_matrix[i, i] == 0:
                for j in range(i + 1, n):
                    if self.augmented_matrix[j, i] != 0:
                        self.augmented_matrix[[i, j]] = self.augmented_matrix[[j, i]]
                        break
            if self.augmented_matrix[i, i] == 0:
                continue
            self.augmented_matrix[i] /= self.augmented_matrix[i, i]
            for j in range(i + 1, n):
                self.augmented_matrix[j] -= self.augmented_matrix[j, i] * self.augmented_matrix[i]

    def back_substitution(self):
        rows, cols = self.augmented_matrix.shape
        cols -= 1
        solutions = {var: None for var in self.variables}
        free_vars = []
        for i in range(rows - 1, -1, -1):
            leading_idx = np.where(self.augmented_matrix[i, :-1] != 0)[0]
            if len(leading_idx) == 0:
                continue
            pivot = leading_idx[0]
            if pivot >= cols:
                continue
            dependent_vars = [self.variables[j] for j in leading_idx[1:]]
            if dependent_vars:
                free_vars.extend(dependent_vars)
                solutions[self.variables[pivot]] = self.augmented_matrix[i, -1] - sum(
                    self.augmented_matrix[i, j] * (solutions[self.variables[j]] if solutions[self.variables[j]] is not None else self.variables[j])
                    for j in leading_idx[1:]
                )
            else:
                solutions[self.variables[pivot]] = self.augmented_matrix[i, -1]
        for var in solutions:
            if solutions[var] is None:
                solutions[var] = var
        self.solution = solutions
        self.free_vars = list(set(free_vars))

    def solve(self):
        self.gaussian_elimination()
        self.back_substitution()
        if self.free_vars:
            print("The system has infinitely many solutions:")
            print(f"for Tmeng = {self.Tmeng} with:\n Ti1 = {self.T1i} and T1o = {self.T1o} and mdot1 = {self.mdot1}")
            print(f" Ti2 = {self.T2i} and T2o = {self.T2o} and mdot2 = {self.mdot2}")
            print(f" Ti3 = {self.T3i} and T2o = {self.T3o} and mdot3 = {self.mdot3}")
        else:
            print("The system has a unique solution:")
        for var, expr in self.solution.items():
            print(f"{var} = {expr}")

    def find_negative_solutions(self):
        x8_values = np.linspace(self.x8_min, self.x8_max, num=100)
        for x8_value in x8_values:
            evaluated_solution = {var: float(self.solution[var].subs(self.variables[-1], x8_value).evalf()) for var in self.solution}
            negative_vars = [var for var, val in evaluated_solution.items() if val < 0]
            if negative_vars:
                print(f"For x8 = {x8_value:.4f}, the following variables are negative: {', '.join(map(str, negative_vars))}")
                break
        else:
            print("All variables remain positive for all values of x8 in the given range.")

    def calculate_mass_flows(self, T2_target=None, T3_target=None, tolerance=None, mdot_c3_range=None, resolution=None):
        """Calculates all mass flow rates based on the solved system."""
        if self.solution is None:
            print("No solution found yet. Run solve() first.")
            return
        
        try:
            matching_mdot_c3, mdot_c3_values, T2_values, T3_values = self.find_matching_mdot_c3(
                T2_target, T3_target, tolerance, mdot_c3_range, resolution
            )
            if matching_mdot_c3.size > 0:
                middle = len(matching_mdot_c3) // 2  # Get middle index
            
            self.x8 = matching_mdot_c3[middle]
            evaluated_solution = {var: self.solution[var].subs(self.variables[-1], self.x8).evalf() for var in self.solution}
            
            self.mdot_hot = evaluated_solution[self.variables[0]]
            self.mdot_h1 = evaluated_solution[self.variables[1]]
            self.mdot_h2 = evaluated_solution[self.variables[2]]
            self.mdot_h3 = evaluated_solution[self.variables[3]]
            self.mdot_cold = evaluated_solution[self.variables[4]]
            self.mdot_c1 = evaluated_solution[self.variables[5]]
            self.mdot_c2 = evaluated_solution[self.variables[6]]
            self.mdot_c3 = evaluated_solution[self.variables[7]]  # x8
            self.x8 = self.mdot_c3
        except KeyError as e:
            print(f"Error: Variable {e} not found in solution. Ensure solve() has been executed.")
        
        
    def print_mass_flows(self):
        """Prints the calculated mass flow values."""
        print(f"\ncalculations for massflows with mdot_c3 = {self.x8}:")
        print(f"mdot_hot = {self.mdot_hot} kg/s")
        print(f"mdot_h1 = {self.mdot_h1} kg/s")
        print(f"mdot_h2 = {self.mdot_h2} kg/s")
        print(f"mdot_h3 = {self.mdot_h3} kg/s")
        print(f"mdot_cold = {self.mdot_cold} kg/s")
        print(f"mdot_c1 = {self.mdot_c1} kg/s")
        print(f"mdot_c2 = {self.mdot_c2} kg/s")
        print(f"mdot_c3 = {self.mdot_c3} kg/s\n")
        
    def calculate_T2_T3(self, mdot_c3):
        """Calculates T2 and T3 based on mdot_c3."""
        Thot, Tcold = self.Thot, self.Tcold
        
        evaluated_solution = {var: self.solution[var].subs(self.variables[-1], mdot_c3).evalf() for var in self.solution}
        
        mdot_h2 = evaluated_solution[self.variables[2]]
        mdot_h3 = evaluated_solution[self.variables[3]]
        mdot_c2 = evaluated_solution[self.variables[6]]
        
        tolerance = 0.000001
        # if mdot_h2 < tolerance or mdot_c2 < tolerance:
        #     T2 = 0
        #     T3 = (Thot * mdot_h3 + Tcold * mdot_c3) / (mdot_h3 + mdot_c3)
        # elif mdot_h3 < tolerance:
        #     T2 = (Thot * mdot_h2 + Tcold * mdot_c2) / (mdot_h2 + mdot_c2)
        #     T3 = 0
        # else:
        T2 = (Thot * mdot_h2 + Tcold * mdot_c2) / (mdot_h2 + mdot_c2)
        T3 = (Thot * mdot_h3 + Tcold * mdot_c3) / (mdot_h3 + mdot_c3)
        
        return T2, T3
    
    def calculate_T2o_T3o_for_matching_mdot_c3(self):
        """Calculates and prints T2 and T3 only for the matching mdot_c3 values."""
        matching_mdot_c3, _, _, _ = self.find_matching_mdot_c3()
    
        if not matching_mdot_c3.size:  # Check if there are no matching values
            print("No matching mdot_c3 values found within the given tolerance.")
            return None
    
        for mdot_c3 in matching_mdot_c3:
            T2, T3 = self.calculate_T2_T3(mdot_c3)
            print(f"Matching mdot_c3 = {mdot_c3:.6f} → T2o = {T2:.2f}°C, T3o = {T3:.2f}°C")
    
        return matching_mdot_c3

    def find_matching_mdot_c3(self, T2_target=None, T3_target=None, tolerance=None, mdot_c3_range=None, resolution=None):
        """Finds mdot_c3 values where T2 and T3 match the target values."""
        if T2_target is None:
            T2_target = self.T2o
        if T3_target is None:
            T3_target = self.T3o
        if mdot_c3_range is None:
            mdot_c3_range = (self.x8_min, self.x8_max)
        if tolerance == None:
            toleranceT2 =self.toleranceT2
            toleranceT3 =self.toleranceT3
        if resolution == None:
            resolution = self.resolution

        mdot_c3_values = np.linspace(mdot_c3_range[0], mdot_c3_range[1], resolution)

        T2_values, T3_values = [], []
        
        for mdot_c3 in mdot_c3_values:
            T2, T3 = self.calculate_T2_T3(mdot_c3)
            T2_values.append(T2)
            T3_values.append(T3)
        
        T2_values, T3_values = np.array(T2_values), np.array(T3_values)
        matching_indices = np.where((np.abs(T2_values - T2_target) <= toleranceT2) & (np.abs(T3_values - T3_target) <= toleranceT3))
        matching_mdot_c3 = mdot_c3_values[matching_indices]

        return matching_mdot_c3, mdot_c3_values, T2_values, T3_values

    def plot_temperature_vs_mdot_c3(self, T2_target=None, T3_target=None, tolerance=None, mdot_c3_range=None, resolution=None):
        """Plots the calculated temperatures T2 and T3 as functions of mdot_c3."""
        matching_mdot_c3, mdot_c3_values, T2_values, T3_values = self.find_matching_mdot_c3(
            T2_target, T3_target, tolerance, mdot_c3_range, resolution
        )
        
        if T2_target is None:
            T2_target = self.T2o
        if T3_target is None:
            T3_target = self.T3o
        if tolerance == None:
            tolerance = self.tolerance
        if resolution == None:
            resolution = self.resolution
        
        plt.figure(figsize=(10, 6))
        plt.plot(mdot_c3_values, T2_values, label="T2_calculated", color="orange")
        plt.plot(mdot_c3_values, T3_values, label="T3_calculated", color="blue")
        plt.axhline(y=T2_target, color="orange", linestyle="dashed", label=f"T2_target ({T2_target}°C)")
        plt.axhline(y=T3_target, color="blue", linestyle="dashed", label=f"T3_target ({T3_target}°C)")
        
        if len(matching_mdot_c3) > 0:
            # Plotting the matching mdot_c3 points for T2 and T3
            plt.scatter(matching_mdot_c3, [T2_target]*len(matching_mdot_c3), color="black", zorder=3, label=r"$\dot{m}_{\text{c,3}}$ (T2)")
            plt.scatter(matching_mdot_c3, [T3_target]*len(matching_mdot_c3), color="black", zorder=3, label=r"$\dot{m}_{\text{c,3}}$ (T3)")
            
            # Add vertical lines at matching mdot_c3 positions with LaTeX formatting
            for mdot in matching_mdot_c3:
                plt.axvline(x=mdot, color='red', linestyle='dotted', label=f"Matching $\dot{{m}}_{{c,3}}$ = {mdot:.2f}")
                # Add the value of matching mdot_c3 to the x-axis
                plt.text(mdot, -5, f'{mdot:.2f}', ha='center', va='bottom', fontsize=10, color='red')  # Adjust the vertical position as needed
            
        plt.xlabel(r"$\dot{m}_{\text{c,3}}$ in kg/s", fontsize=12)  # LaTeX formatted x-axis label
        plt.ylabel("Temperature (°C)")
        plt.title(r"Finding the Best $\dot{m}_{\text{c,3}}$: Temperature vs. $\dot{m}_{\text{c,3}}$")
        plt.legend(loc='upper right')  # Move the legend to the top right corner
        plt.grid(True)
        plt.show()
        
        return matching_mdot_c3


    
    def calculate_direct_heating_of_streams(self):
        Q1 = self.mdot1*self.cp*(self.T1o - self.T1i)
        Q2 = self.mdot2*self.cp*(self.T2o - self.T2i)
        Q3 = self.mdot3*self.cp*(self.T3o - self.T3i)
        
        return Q1, Q2, Q3
    
    def calculate_buffer_heating(self):
        evaluated_solution = {var: self.solution[var].subs(self.variables[-1], self.x8).evalf() for var in self.solution}
        self.mdot_hot = evaluated_solution[self.variables[0]]
        
        Q = self.mdot_hot*self.cp*(self.Thot - self.Tmeng)
        
        return Q
    
    def calculate_difference_in_heating(self):
        Q1, Q2, Q3 = self.calculate_direct_heating_of_streams()
        Qdir_tot = Q1 + Q2 + Q3
        Q = self.calculate_buffer_heating()
        
        diff_heat = Q - Qdir_tot
        
        return diff_heat
    
    def calculate_BufferPercentage(self):
        buff = self.calculate_buffer_heating()
        diff = self.calculate_difference_in_heating()
        percentage = (diff/buff)*100
        return percentage
    
    def plot_BufferPercentage_over_massflow_M1_M3(self):
        m1 = np.linspace(0, 100, 200)
        m3 = np.linspace(0, 100, 200)
        
        M1, M3 = np.meshgrid(m1, m3)
        percentage = np.zeros_like(M1)
        
        for i in range(M1.shape[0]):
            for j in range(M1.shape[1]):
                # Temporarily adjust mdot1 and mdot2
                self.mdot1 = M1[i, j]
                self.mdot3 = M3[i, j]
                
                # Calculate buffer percentage
                percentage[i, j] = self.calculate_BufferPercentage()
        
        # Restore original values (optional)
        self.mdot1 = m1[0]
        self.mdot3 = m3[0]
        
        # Plotting
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(M1, M3, percentage, cmap='viridis')
        
        ax.set_xlabel('Mass flow m1')
        ax.set_ylabel('Mass flow m3')
        ax.set_zlabel('Buffer Percentage')
        ax.set_title('Buffer Percentage over Mass Flow')
    
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()




system = Case1()
system.solve()
system.find_negative_solutions()
matching_mdot_c3 = system.plot_temperature_vs_mdot_c3()
system.calculate_T2o_T3o_for_matching_mdot_c3()
print("\nMatching mdot_c3 values:", matching_mdot_c3)
system.print_mass_flows()

dir_heating = system.calculate_direct_heating_of_streams()
buffer_heating = system.calculate_buffer_heating()
difference_heating = system.calculate_difference_in_heating()
percentage = system.calculate_BufferPercentage()
print('\nheat input when heating the streams directly: ',dir_heating, ', sum:', sum(dir_heating))
print('heat input when using buffer: ',buffer_heating)
print('difference in heating between buffer and direct heating: ',difference_heating,"and percentage of heat input of buffer =",percentage,'%' )

#system.plot_BufferPercentage_over_massflow_M1_M3()

###############
# DEZE CODE IS ENKEL GELDIG VOOR EEN Tmeng < laagste T aan de outputs => LET OP MET DE INPUT TEMPERATUREN
# OOK ZIJN DE FORMULES VOOR DE MASSADEBIETEN AFHANKELIJK VAN DE OUTPUT TEMPERATUREN, indien deze veranderen veranderen de formules voor de massadebieten (indien output temperatuur veranderd zal er geen matching mdot_c3 zijn)
# FORMULES VOOR DE MASSADEBIETEN ZIJN AFHANKELIJK VAN DE INPUT DEBIETEN
###############

"""
systeem bestaat een een mengtemperatuur met 65°C (= Tmeng) 
en de output temperaturen zijn T1o = 92°C, T2o = 80°C en T3o = 70°C
deze code bepaalt voor welke massadebieten de output temperaturen kunnen worden behaald
"""
