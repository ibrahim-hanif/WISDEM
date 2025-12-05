"""
mainshaft_openmdao.py
OpenMDAO components for main bearing & shaft sizing for IEA 15MW TLP
Compatible with WISDEM framework from NREL
Author: Claude (converted from MATLAB)
Date: 2025-11-03
"""
#%%
import numpy as np
import openmdao.api as om
from scipy.stats import weibull_min
from scipy.integrate import cumulative_trapezoid
from wisdem.drivetrainse.drive_structure import analytical_MB_Forces
from wisdem.drivetrainse.drive_components import MainBearing
import os
from wisdem.commonse.utilities import mainshaft_loads_from_mat_to_dict

#%%
class MainShaftLoadsComponent(om.ExplicitComponent):
    """
    #TODO: no use? input loads from main script ('F_aero_hub')
    Component to process FLS and ULS loads from OpenFAST
    """
    def initialize(self):
        self.options.declare('fileloc_ms_loads', desc='full filepath to the location of mainshaft / hub loads (from global simulations, openFAST)')
    
    def setup(self):
        fileloc_ms_loads = self.options['fileloc_ms_loads']
        F_dict = mainshaft_loads_from_mat_to_dict(loc_loads_mat_file=fileloc_ms_loads)
        # Inputs - FLS loads
        self.add_input('Fx_FLS', val=np.zeros((720000, 10)), desc='FLS force X time series', units='N')
        self.add_input('Fy_FLS', val=np.zeros((720000, 10)), desc='FLS force Y time series', units='N')
        self.add_input('Fz_FLS', val=np.zeros((720000, 10)), desc='FLS force Z time series', units='N')
        self.add_input('Mx_FLS', val=np.zeros((720000, 10)), desc='FLS moment X time series', units='N*m')
        self.add_input('My_FLS', val=np.zeros((720000, 10)), desc='FLS moment Y time series', units='N*m')
        self.add_input('Mz_FLS', val=np.zeros((720000, 10)), desc='FLS moment Z time series', units='N*m')
        
        # Inputs - ULS loads
        self.add_input('Fx_max', val=0.0, desc='ULS max force X', units='N')
        self.add_input('Fy_max', val=0.0, desc='ULS max force Y', units='N')
        self.add_input('Fz_mean', val=0.0, desc='ULS mean force Z', units='N')
        self.add_input('Mx_max', val=0.0, desc='ULS max moment X', units='N*m')
        self.add_input('My_max', val=0.0, desc='ULS max moment Y', units='N*m')
        self.add_input('Mz_max', val=0.0, desc='ULS max moment Z', units='N*m')
        
        # Outputs
        self.add_output('loads_processed', val=True, desc='Flag indicating loads are processed')
        
    def compute(self, inputs, outputs):
        # Simple pass-through component for now
        # In a full implementation, this could perform load processing
        outputs['loads_processed'] = True


class BearingPropertiesComponent(om.ExplicitComponent): #TODO: use MainBearing from wisdem
    """
    #NOTE: no use, rather use MainBearing after implementing all needed
    Component to compute bearing properties based on diameter
    """
    
    def setup(self):
        # Inputs
        self.add_input('D_mb1', val=2.0, desc='Main bearing 1 diameter', units='m')
        self.add_input('D_mb2', val=2.0, desc='Main bearing 2 diameter', units='m')
        
        # Outputs
        self.add_output('Cr_mb1', val=0.0, desc='Dynamic load rating MB1', units='N')
        self.add_output('Cr_mb2', val=0.0, desc='Dynamic load rating MB2', units='N')
        self.add_output('FW_mb1', val=0.0, desc='Face width MB1', units='m')
        self.add_output('FW_mb2', val=0.0, desc='Face width MB2', units='m')
        self.add_output('W_mb1', val=0.0, desc='Weight MB1', units='kg')
        self.add_output('W_mb2', val=0.0, desc='Weight MB2', units='kg')
        
        self.declare_partials('*', '*', method='fd')
        
    def compute(self, inputs, outputs):
        D_mb1 = inputs['D_mb1']
        D_mb2 = inputs['D_mb2']
        
        # CRB bearing properties
        outputs['Cr_mb1'] = 4526.5e3 * D_mb1 ** 0.9556
        outputs['FW_mb1'] = 0.1570 * D_mb1 + 0.0849
        outputs['W_mb1'] = 1070.8 * D_mb1 ** 1.8278
        
        # 2TRB bearing properties
        outputs['Cr_mb2'] = 6579.9e3 * D_mb2 ** 0.8592
        outputs['FW_mb2'] = 0.1541 * D_mb2 + 0.2087
        outputs['W_mb2'] = 1442.6 * D_mb2 ** 1.8932


class Analytical_FLS_Bearing_Life( om.ExplicitComponent ):
    """
    Component to compute bearing fatigue life (FLS)
    """
    
    def initialize(self):
        self.options.declare('nBins', default=100, desc='Number of bins for histogram')
        
    def setup(self):
        # Inputs
        # - loads
        self.add_input('Fx_FLS', val=np.zeros((720000, 10)), desc='X-dir force, mainshaft input', units='N')
        self.add_input('Fy_FLS', val=np.zeros((720000, 10)), desc='Y-dir force, mainshaft input', units='N')
        self.add_input('Fz_FLS', val=np.zeros((720000, 10)), desc='Z-dir force, mainshaft input', units='N')
        self.add_input('Mx_FLS', val=np.zeros((720000, 10)), desc='X-dir moment, mainshaft input', units='N*m')
        self.add_input('My_FLS', val=np.zeros((720000, 10)), desc='Y-dir moment, mainshaft input', units='N*m')
        self.add_input('Mz_FLS', val=np.zeros((720000, 10)), desc='Z-dir moment, mainshaft input', units='N*m')
        # - 1. LSS parameters (from Layout, Hub_Rotor_LSS_Frame)
        self.add_input('L_12', val=6.0, desc='Main bearing span', units='m')
        self.add_input('L_h1', val=4.25, desc='Rotor bearing distance', units='m')
        # - 2. bearing parameters (from MainBearing)
        self.add_input('Cr_mb1', val=1e7, desc='Dynamic load rating MB1', units='N')
        self.add_input('Cr_mb2', val=1e7, desc='Dynamic load rating MB2', units='N')
        self.add_input('p_mb', val=10/3, desc='Bearing life exponent')
        self.add_input('e_mb', val=4.0, desc='Bearing limiting factor, load ratio')
        self.add_input('X1_mb', val=0.0, desc='Bearing light coefficient for P calculation')
        self.add_input('Y1_mb', val=0.0, desc='Bearing light coefficient for P calculation')
        self.add_input('X2_mb', val=0.0, desc='Bearing heavy coefficient for P calculation')
        self.add_input('Y2_mb', val=0.0, desc='Bearing heavy coefficient for P calculation')
        # - operational
        self.add_input('rated_rpm', val=7.56, desc='Nominal/rated rotational speed', units='rpm')
        self.add_input('design_life', val=20.0, desc='Wind turbine design life')

        # Outputs
        self.add_output('L10h_mb1', val=0.0, desc='L10 life MB1', units='h')
        self.add_output('L10h_mb2', val=0.0, desc='L10 life MB2', units='h')
        self.add_output('constr_L_mb1', val=0.0, desc='Safety factor MB1')
        self.add_output('constr_L_mb2', val=0.0, desc='Safety factor MB2')
        self.add_output('constr_L_mb_all', val=0.0, desc='Minimum safety factor')
        
        self.declare_partials('*', '*', method='fd')
        
    def compute(self, inputs, outputs):
        # ISO 281 parameters (from MainBearing)
        e, p = inputs['e_mb'], inputs['p_mb']
        X1, Y1 = inputs['X1_mb'], inputs['Y1_mb']
        X2, Y2 = inputs['Y1_mb'], inputs['Y2_mb']
        
        # Wind statistics
        ws = np.arange(5, 25, 2) #note: til 23, in py last '25' not incl.
        ws_pdf_utsira = weibull_min.pdf(ws, c=1.95, scale=11.6)
        
        Lm = inputs['L_12']
        L_rb = inputs['L_h1']
        n0 = inputs['rated_rpm']
        
        Fx = inputs['Fx_FLS']
        Fy = inputs['Fy_FLS']
        Fz = inputs['Fz_FLS']
        Mx = inputs['Mx_FLS']
        My = inputs['My_FLS']
        Mz = inputs['Mz_FLS']
        
        # Bearing loads (analytical) calculation
        F_mb1_x,F_mb1_y,F_mb1_z,F_mb1_rad, F_mb2_x,F_mb2_y,F_mb2_z,F_mb2_rad = analytical_MB_Forces(
            Fx,Fy,Fz,Mx,My,Mz,L_rb,Lm
            )

        # Equivalent loads MB2
        ratio = F_mb2_x / np.maximum(F_mb2_rad, np.finfo(float).eps)
        light = np.abs(ratio) <= e
        
        P_mb2 = np.zeros_like(F_mb2_rad)
        P_mb2[light] = X1 * F_mb2_rad[light] + Y1 * F_mb2_x[light]
        P_mb2[~light] = X2 * F_mb2_rad[~light] + Y2 * F_mb2_x[~light]
        # Equivalent loads MB1 (= radial loads coz radial bearing CRB)
        P_mb1 = F_mb1_rad

        # Histogram processing
        nBins = self.options['nBins']
        
        P1max = np.max(P_mb1) if np.max(P_mb1) > 0 else np.finfo(float).eps
        P2max = np.max(P_mb2) if np.max(P_mb2) > 0 else np.finfo(float).eps
        
        edges_mb1 = np.linspace(0, P1max, nBins + 1)
        edges_mb2 = np.linspace(0, P2max, nBins + 1)
        centers_mb1 = (edges_mb1[:-1] + edges_mb1[1:]) / 2
        centers_mb2 = (edges_mb2[:-1] + edges_mb2[1:]) / 2
        
        P_mb1_hist_utsira = np.zeros(nBins)
        P_mb2_hist_utsira = np.zeros(nBins)

        for ec in range(len(ws)):
            hist1, _ = np.histogram(P_mb1[:, ec], bins=edges_mb1, density=True) #density, for it is PDF
            hist1 = hist1 / np.sum(hist1) if np.sum(hist1) > 0 else hist1 #divide to normalize (sum=1)
            P_mb1_hist_utsira += ws_pdf_utsira[ec] * hist1
            
            hist2, _ = np.histogram(P_mb2[:, ec], bins=edges_mb2, density=True)
            hist2 = hist2 / np.sum(hist2) if np.sum(hist2) > 0 else hist2
            P_mb2_hist_utsira += ws_pdf_utsira[ec] * hist2

        P_mb1_sum = (np.sum((centers_mb1 ** (p)) * P_mb1_hist_utsira)) ** (1/p)
        P_mb2_sum = (np.sum((centers_mb2 ** (p)) * P_mb2_hist_utsira)) ** (1/p)
        
        # Life calculation
        Cr1 = inputs['Cr_mb1']
        Cr2 = inputs['Cr_mb2']
        
        L10_mb1 = (Cr1 / P_mb1_sum) ** (p)
        L10_mb2 = (Cr2 / P_mb2_sum) ** (p)
        
        outputs['L10h_mb1'] = L10_mb1 * (1e6 / n0 / 60)
        outputs['L10h_mb2'] = L10_mb2 * (1e6 / n0 / 60)
        
        # Safety factors (20 years = 20*8766 hours)
        L_design = inputs['design_life']
        outputs['constr_L_mb1'] = (outputs['L10h_mb1'] / (L_design * 8766)) ** (1/p)
        outputs['constr_L_mb2'] = (outputs['L10h_mb2'] / (L_design * 8766)) ** (1/p)
        outputs['constr_L_mb_all'] = min(outputs['constr_L_mb1'], outputs['constr_L_mb2'])


class ULSShaftStressComponent(om.ExplicitComponent):
    """
    NOTE: no use, Hub_Rotor_LSS_Frame does the same but better (higher fidelity with pyFrame3DD)
    Component to compute shaft stress and deflection under ULS loads
    """
    
    def initialize(self):
        self.options.declare('Nx', default=100, desc='Number of points along shaft')
        
    def setup(self):
        # Inputs - loads
        self.add_input('Fx_max', val=0.0, desc='ULS max force X', units='N')
        self.add_input('Fy_max', val=0.0, desc='ULS max force Y', units='N')
        self.add_input('Fz_mean', val=0.0, desc='ULS mean force Z', units='N')
        self.add_input('Mx_max', val=0.0, desc='ULS max moment X', units='N*m')
        self.add_input('My_max', val=0.0, desc='ULS max moment Y', units='N*m')
        self.add_input('Mz_max', val=0.0, desc='ULS max moment Z', units='N*m')
        
        # Inputs - geometry
        self.add_input('L_mb', val=6.0, desc='Main bearing span', units='m')
        self.add_input('L_rb', val=4.25, desc='Rotor bearing distance', units='m')
        self.add_input('D_mb1', val=2.0, desc='Outer diameter MB1', units='m')
        self.add_input('D_mb2', val=2.0, desc='Outer diameter MB2', units='m')
        self.add_input('t_mb1', val=0.3, desc='Wall thickness MB1', units='m')
        self.add_input('t_mb2', val=0.3, desc='Wall thickness MB2', units='m')
        
        # Inputs - material
        self.add_input('S_y', val=379e6, desc='Yield strength', units='Pa')
        self.add_input('E', val=168e9, desc='Young\'s modulus', units='Pa')
        self.add_input('k', val=1.1, desc='Safety factor')
        
        # Inputs - constraints
        self.add_input('nu_y_per', val=0.8e-3, desc='Allowable slope', units='rad')
        
        # Outputs
        self.add_output('sigma_eq_max', val=0.0, desc='Max equivalent stress', units='Pa')
        self.add_output('nu_y_max', val=0.0, desc='Max slope', units='rad')
        self.add_output('S_ULS', val=0.0, desc='ULS safety factor')
        self.add_output('S_nuy', val=0.0, desc='Slope safety factor')
        
        self.declare_partials('*', '*', method='fd')
        
    def compute(self, inputs, outputs):
        Nx = self.options['Nx']
        x_local = np.linspace(0, 1, Nx)
        
        Lm = inputs['L_mb']
        L_rb = inputs['L_rb']
        
        # ULS loads
        Fx_max = inputs['Fx_max']
        Fy_max = inputs['Fy_max']
        Fz_max = inputs['Fz_mean']
        Mx_max = inputs['Mx_max']
        My_max = inputs['My_max']
        Mz_max = inputs['Mz_max']
        
        # Bearing reactions (analytical)
        F_mb1_x,F_mb1_y,F_mb1_z,F_mb1_rad, F_mb2_x,F_mb2_y,F_mb2_z,F_mb2_rad = analytical_MB_Forces(
            Fx_max,Fy_max,Fz_max,Mx_max,My_max,Mz_max,L_rb,Lm
            )
        
        # Geometry along shaft
        D1 = inputs['D_mb1']
        D2 = inputs['D_mb2']
        t1 = inputs['t_mb1']
        t2 = inputs['t_mb2']
        d1 = D1 - t1
        d2 = D2 - t2
        
        x_vals = x_local * Lm
        Dx = (D2 - D1) * x_local / Lm + D1
        dx = (d2 - d1) * x_local / Lm + d1
        
        # Moments along shaft
        My_vec = -Fz_max * (x_vals + L_rb) - My_max - F_mb1_z * x_vals
        Mz_vec = -Fy_max * (x_vals + L_rb) - Mz_max - F_mb1_y * x_vals
        
        # Section properties
        I = (np.pi / 64) * (Dx**4 - dx**4)
        J = (np.pi / 32) * (Dx**4 - dx**4)
        
        # Stresses
        sigma_y = (My_vec * Dx) / (2 * I)
        sigma_z = (Mz_vec * Dx) / (2 * I)
        tau = (Mx_max * Dx) / (2 * J)
        
        sigma_eq = np.sqrt(sigma_y**2 + sigma_z**2 + 3*tau**2)
        outputs['sigma_eq_max'] = np.max(sigma_eq)
        
        # Deflection
        E = inputs['E']
        invEI = My_vec / (E * I)
        
        # Numerical integration using cumulative trapezoid
        theta = cumulative_trapezoid(invEI, x_vals, initial=0)
        v = cumulative_trapezoid(theta, x_vals, initial=0)
        
        vend = v[-1]
        v_adj = v - (x_vals / Lm) * vend
        theta_adj = theta - vend / Lm
        
        outputs['nu_y_max'] = np.max(np.abs(theta_adj))
        
        # Safety factors
        sigma_per = inputs['S_y'] / inputs['k']
        outputs['S_ULS'] = sigma_per / outputs['sigma_eq_max']
        outputs['S_nuy'] = inputs['nu_y_per'] / outputs['nu_y_max']


class ShaftWeightComponent(om.ExplicitComponent):
    """
    Component to compute shaft and bearing assembly weight
    """
    
    def setup(self):
        # Inputs - geometry
        self.add_input('L_mb', val=6.0, desc='Main bearing span', units='m')
        self.add_input('D_mb1', val=2.0, desc='Outer diameter MB1', units='m')
        self.add_input('D_mb2', val=2.0, desc='Outer diameter MB2', units='m')
        self.add_input('t_mb1', val=0.3, desc='Wall thickness MB1', units='m')
        self.add_input('t_mb2', val=0.3, desc='Wall thickness MB2', units='m')
        self.add_input('FW_mb1', val=0.3, desc='Face width MB1', units='m')
        self.add_input('FW_mb2', val=0.3, desc='Face width MB2', units='m')
        
        # Inputs - weights
        self.add_input('W_mb1', val=0.0, desc='Weight MB1', units='kg')
        self.add_input('W_mb2', val=0.0, desc='Weight MB2', units='kg')
        
        # Inputs - material
        self.add_input('rho', val=7100.0, desc='Material density', units='kg/m**3')
        
        # Outputs
        self.add_output('W_shaft', val=0.0, desc='Shaft weight', units='kg')
        self.add_output('W_total', val=0.0, desc='Total assembly weight', units='kg')
        
        self.declare_partials('*', '*', method='fd')
        
    def compute(self, inputs, outputs):
        Lm = inputs['L_mb']
        D1 = inputs['D_mb1']
        D2 = inputs['D_mb2']
        t1 = inputs['t_mb1']
        t2 = inputs['t_mb2']
        d1 = D1 - t1
        d2 = D2 - t2
        
        FW1 = inputs['FW_mb1']
        FW2 = inputs['FW_mb2']
        rho = inputs['rho']
        
        # Bearing seats
        seat1 = (np.pi / 4) * (D1**2 - d1**2) * FW1
        seat2 = (np.pi / 4) * (D2**2 - d2**2) * FW2
        
        # Tapered section
        segLen = Lm - FW1/2 - FW2/2
        taper_outer = segLen * (np.pi / 12) * (D1**2 + D2**2 + D1*D2)
        taper_inner = segLen * (np.pi / 12) * (d1**2 + d2**2 + d1*d2)
        
        outputs['W_shaft'] = 1.33 * (taper_outer - taper_inner + seat1 + seat2) * rho
        outputs['W_total'] = outputs['W_shaft'] + inputs['W_mb1'] + inputs['W_mb2']


class MainShaftGroup(om.Group):
    """
    Group combining all main shaft components with optimization
    """
    
    def setup(self):
        # Add components
        self.add_subsystem('bearing_props', BearingPropertiesComponent(),
                          promotes_inputs=['D_mb1', 'D_mb2'],
                          promotes_outputs=['Cr_mb1', 'Cr_mb2', 'FW_mb1', 'FW_mb2', 'W_mb1', 'W_mb2'])
        
        self.add_subsystem('fls_life', Analytical_FLS_Bearing_Life(),
                          promotes_inputs=['L_mb', 'Cr_mb1', 'Cr_mb2'])
        
        self.add_subsystem('uls_stress', ULSShaftStressComponent(),
                          promotes_inputs=['L_mb', 'D_mb1', 'D_mb2', 't_mb1', 't_mb2'])
        
        self.add_subsystem('weight', ShaftWeightComponent(),
                          promotes_inputs=['L_mb', 'D_mb1', 'D_mb2', 't_mb1', 't_mb2',
                                         'FW_mb1', 'FW_mb2', 'W_mb1', 'W_mb2', 'W_total'])
        
        # Add design variables
        self.add_design_var('L_mb', lower=4.0, upper=10.0)
        self.add_design_var('D_mb1', lower=1.0, upper=4.0)
        self.add_design_var('D_mb2', lower=1.0, upper=4.0)
        self.add_design_var('t_mb1', lower=0.1, upper=0.5)
        self.add_design_var('t_mb2', lower=0.1, upper=0.5)
        
        # Add constraints
        self.add_constraint('fls_life.S_mb', lower=1.0)
        self.add_constraint('uls_stress.S_ULS', lower=1.0)
        self.add_constraint('uls_stress.S_nuy', lower=1.0)
        
        # Add objective
        self.add_objective('weight.W_total')

#%%
def setup_main_shaft_optimization():
    """
    Setup function to create and configure the main shaft optimization problem
    Returns configured OpenMDAO problem
    """
    prob = om.Problem()
    
    # Create model
    prob.model = MainShaftGroup()
    
    # Setup optimizer
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1e-6
    prob.driver.options['maxiter'] = 200
    
    # Setup recorder (optional)
    prob.driver.add_recorder(om.SqliteRecorder('main_shaft_opt.sql'))
    
    # Setup problem
    prob.setup()
    
    # Set initial values
    prob.set_val('L_mb', 6.0)
    prob.set_val('D_mb1', 2.0)
    prob.set_val('D_mb2', 2.0)
    prob.set_val('t_mb1', 0.3)
    prob.set_val('t_mb2', 0.3)
    
    return prob

#%%
if __name__ == '__main__':
    # Example usage
    prob = setup_main_shaft_optimization()
    
    # Run optimization
    prob.run_driver()
    
    # Print results
    print('\nOptimization Results:')
    print(f"L_mb = {prob.get_val('L_mb')[0]:.3f} m")
    print(f"D_mb1 = {prob.get_val('D_mb1')[0]:.3f} m")
    print(f"D_mb2 = {prob.get_val('D_mb2')[0]:.3f} m")
    print(f"t_mb1 = {prob.get_val('t_mb1')[0]:.3f} m")
    print(f"t_mb2 = {prob.get_val('t_mb2')[0]:.3f} m")
    print(f"\nTotal Weight = {prob.get_val('weight.W_total')[0]:.2f} kg")
    print(f"S_mb (FLS) = {prob.get_val('fls_life.S_mb')[0]:.3f}")
    print(f"S_ULS = {prob.get_val('uls_stress.S_ULS')[0]:.3f}")
    print(f"S_nuy = {prob.get_val('uls_stress.S_nuy')[0]:.3f}")
    
    prob.cleanup()

# %%[markdown]
# test loading `.mat` load files
import numpy as np
from wisdem.commonse.utilities import mainshaft_loads_from_mat_to_dict

results_path = 'm:\\Vasudev_Gupta\\WISDEM\\examples\\06_drivetrain\\M4W_production_runs\\02_results'
loc_FLS_loads_mat_file = os.path.join(results_path, "Load_input\mainshaft_loads_FLS.mat")
loc_ULS_loads_mat_file = os.path.join(results_path, "Load_input\mainshaft_loads_ULS.mat")

Snew, keys_all = mainshaft_loads_from_mat_to_dict(loc_FLS_loads_mat_file, loc_ULS_loads_mat_file)
#%%
Fx, Fy, Fz = Snew['Fx'], Snew['Fy'], Snew['Fz']
Mx, My, Mz = Snew['Mx'], Snew['My'], Snew['Mz']
F_aero_hub = np.array( [Snew['Fx_max'], Snew['Fy_max'], Snew['Fz_mean']] )
M_aero_hub = np.array( [Snew['Mx_max'], Snew['My_max'], Snew['Mz_max']] )
L_rb = 4.25
Lm = 6.0
# %%
