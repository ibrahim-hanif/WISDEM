# %% [markdown]
# # Drivetrain optimization (`WISDEM`)
# single component optimization (as suggested by `WEIS` ppt)
# 
# ### current version:
# LSS sizing only: (`Hub_Rotor_LSS_Frame` main)
#
# - objective: (1) `nacelle_mass` minimization (`NacelleSystemAdder`)
#
# - DVs: (4) `L_h1, L_12, lss_diameter, lss_wall_thickness`
#
# - constraints: (3) lss stresses, deflections (linear, angle)
#
#
# ### TODO:
# 
# references
# 1. 2020_Wang_NTNU - on design modelling and analysis of 10MW
# 2. IEA 15MW=baseline
# 3. Task2.1

# %% [markdown]
# imports
import os
import numpy as np
import openmdao.api as om

# %%
from wisdem.drivetrainse.drivetrain import DriveMaterials

from wisdem.drivetrainse.hub import Hub_System
from wisdem.drivetrainse.gearbox import Gearbox

import wisdem.drivetrainse.layout as lay

import wisdem.drivetrainse.drive_components as dc

import wisdem.drivetrainse.drive_structure as ds

from wisdem.commonse.utilities import get_recorder_results
import pickle

# %% [markdown]
# ### Defining results directory and files
results_dir = "01_results"
script_dir = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(script_dir, results_dir)
os.makedirs(results_path, exist_ok=True)

loc_cases = os.path.join(results_path, "cases_recorded.sql")
# if os.path.exists( loc_cases ):
#     os.remove( loc_cases)


loc_doe = os.path.join(results_path, "DOE.sql")
loc_n2 = os.path.join(results_path, "n2.html")
loc_scaling_report = os.path.join(results_path, 'scaling_report.html')

# %% [markdown]
# ### Defining options (modelling_options), flags

opt_flag = True
flag_DOE = False

# define modelling_options
opts = {}

opts["WISDEM"] = {}
opts["WISDEM"]["n_dlc"] = 1
opts["WISDEM"]["DriveSE"] = {}
# NOTE "hub": 'Hub_System' component are NOT included in the 'LSS_layout' component 
opts["WISDEM"]["DriveSE"]["hub"] = {}
opts["WISDEM"]["DriveSE"]["hub"]["hub_gamma"] = 2.0
opts["WISDEM"]["DriveSE"]["hub"]["spinner_gamma"] = 1.5

opts["WISDEM"]["DriveSE"]["direct"] = False
opts["WISDEM"]["DriveSE"]["use_gb_torque_density"] = True # False =(GB  optim, in-capabale)

opts["WISDEM"]["DriveSE"]["gamma_f"] = 1.35 #IEC-1, 7.6.2.2a, pg.57
opts["WISDEM"]["DriveSE"]["gamma_m"] = 1.3  #IEC-1, 7.6.2.4, pg.59
opts["WISDEM"]["DriveSE"]["gamma_n"] = 1.0  #IEC-1, 7.6.1.3, pg.55
# used as: gamma = gamma_f * gamma_m * gamma_n (within TODO)

opts["WISDEM"]["RotorSE"] = {}
opts["WISDEM"]["RotorSE"]["n_pc"] = 2 #cf. RPM_Input in drive_components.py

opts["materials"] = {}
opts["materials"]["n_mat"] = 4

opts["flags"] = {}
dogen = opts["flags"]["generator"] = False
dohub = opts["flags"]["hub"] = False #(v)

# %% [markdown]
# ### Defining the model (problem)
# as an openMDAO group that uses DrivetrainSE classes as components
class LSS_layout( om.Group ):
    """
    Group containing components for the layout of the LSS components
    """
    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        opt = self.options["modeling_options"]["WISDEM"]["DriveSE"]
        n_dlcs = self.options["modeling_options"]["WISDEM"]["n_dlc"]
        direct = opt["direct"]
        if direct:
            use_gb_torque_density = False
        else:
            use_gb_torque_density = opt["use_gb_torque_density"]
            
        dogen = self.options["modeling_options"]["flags"]["generator"]
        n_pc = self.options["modeling_options"]["WISDEM"]["RotorSE"]["n_pc"]
        flag_hub = self.options["modeling_options"]["flags"]["hub"]
        
        # print flag information
        print("=== Problem 'LSS_layout' setting up ===")
        print(f"flag info: direct={direct}, use_gb_torque_density={use_gb_torque_density}, dogen={dogen}, flag_hub={flag_hub}")

        # self.set_input_defaults("machine_rating", units="kW")
        #self.set_input_defaults("hvac_mass_coeff", 0.025, units="kg/kW/m")

        # Materials prep
        self.add_subsystem(
            "mat",
            DriveMaterials(direct=direct, n_mat=self.options["modeling_options"]["materials"]["n_mat"]),
                promotes=["*"]
            )
        # - for 'layout' component: need = lss_rho, bedplate_rho, hss_rho 

        # Before the layout, need to do these first
        # 1. hub system (perf hub system optimization)
        if flag_hub: # bypass rn, TODO later
            self.add_subsystem(
                "hub", Hub_System(modeling_options=opt["hub"]),
                    promotes=["*"]
                )
        
        # # 2. gearbox
        self.add_subsystem(
            "gear", Gearbox(direct_drive=direct, use_gb_torque_density=use_gb_torque_density),
                promotes=["*"]
            )

        # Layout (just discretization of DT and each compn, output 's_drive', etc.)
        #if not direct:
        self.add_subsystem(
            'layout', lay.GearedLayout(),
                promotes=["*"]
            )
        
        # Hub_Rotor_LSS_Frame:
        self.add_subsystem(
            "lss", ds.Hub_Rotor_LSS_Frame(n_dlcs=n_dlcs, modeling_options=opt, direct_drive=direct),
                promotes=["*"]
            )

        # MBs for defl constraints
        # self.add_subsystem("bear1", dc.MainBearing())
        # self.add_subsystem("bear2", dc.MainBearing())

        # # Final tallying (mass summation)
        self.add_subsystem(
            "misc", dc.MiscNacelleComponents(direct_drive=direct),
            promotes=["*"]
            )
        self.add_subsystem(
            "nac", dc.NacelleSystemAdder(direct_drive=direct),
            promotes=["*"]
            )
        # self.add_subsystem("rna", dc.RNA_Adder(), promotes=["*"])

        # # Bedplate_IBeam_Frame:
        # self.add_subsystem(
        #     "bed", ds.Bedplate_IBeam_Frame(modeling_options=opt, n_dlcs=n_dlcs),
        #         promotes=["*"]
        #     )
        # # = bear(1,2) -to- Bedplate_IBeam_Frame
        # self.connect("bear1.mb_mass", "mb1_mass")
        # self.connect("bear1.mb_I", "mb1_I")
        # self.connect("bear1.mb_max_defl_ang", "mb1_max_defl_ang")
        # self.connect("bear2.mb_mass", "mb2_mass")
        # self.connect("bear2.mb_I", "mb2_I")
        # self.connect("bear2.mb_max_defl_ang", "mb2_max_defl_ang")

        if False:
            # Hub_Rotor_LSS_Frame: components required =
            # 1. brake system
            self.add_subsystem(
                "brake", dc.Brake(direct_drive=direct),
                    promotes=["*"]
                )
            
            # 2. generator
            self.add_subsystem(
                "rpm", dc.RPM_Input(n_pc=n_pc),
                    promotes=["*"]
                )
            self.add_subsystem(
                "gensimp", dc.GeneratorSimple(direct_drive=direct, n_pc=n_pc),
                    promotes=["*"]
                )

            # Hub_Rotor_LSS_Frame:
            self.add_subsystem(
                "lss", ds.Hub_Rotor_LSS_Frame(n_dlcs=n_dlcs, modeling_options=opt, direct_drive=direct),
                    promotes=["*"]
                )
            
            # MBs for defl constraints
            self.add_subsystem("bear1", dc.MainBearing())
            self.add_subsystem("bear2", dc.MainBearing())

            # TODO: add HSS_Frame: perform HSS side optimization

            # Bedplate_IBeam_Frame:
            self.add_subsystem(
                "bed", ds.Bedplate_IBeam_Frame(modeling_options=opt, n_dlcs=n_dlcs),
                    promotes=["*"]
                )
            
            # NacelleSystemAdder: for nacelle_mass #TODO: add later
            # - NOTE: yaw_mass default (0.0) used, coz dc.YawSystem not included (dc.Electronics also, for transformer_ and converter_mass)
            self.add_subsystem(
                "nac", dc.NacelleSystemAdder(direct_drive=direct),
                    promotes=["*"]
                )

            # Output-to-input connections
            # = mat -to- hub
            if flag_hub:
                self.connect("bedplate_rho", ["pitch_system.rho", "spinner.metal_rho"])
                self.connect("bedplate_Xy", ["pitch_system.Xy", "spinner.Xy"])
                self.connect("bedplate_mat_cost", "spinner.metal_cost")
                self.connect("hub_rho", "hub_shell.rho")
                self.connect("hub_Xy", "hub_shell.Xy")
                self.connect("hub_mat_cost", "hub_shell.metal_cost")
                self.connect("spinner_rho", "spinner.composite_rho")
                self.connect("spinner_Xt", "spinner.composite_Xt")
                self.connect("spinner_mat_cost", "spinner.composite_cost")

            # self.connect("hub_rho", "rho_castiron")
            # self.connect("spinner_rho", "rho_fiberglass")

# %% [markdown]
# ### Setup the problem
# Define the problem
prob = om.Problem(reports=False)

# Define the model
prob.model = LSS_layout(modeling_options=opts) # an instance of the LSS_layout problem defined above

# %%[markdown]
# ### Optimization
# If performing optimization, set up the optimizer and problem formulation
if opt_flag:
    # Choose the optimizer to use
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"
    prob.driver.options["tol"] = 1e-6 # comment to default (1e-6?)
    prob.driver.options["maxiter"] = 5 * 4
    prob.driver.options["disp"] = True
    prob.driver.options["debug_print"] = ["desvars", "objs", "nl_cons", "ln_cons"]

    # prob.set_solver_print(level=2)
    recorder = om.SqliteRecorder( loc_cases )
    prob.driver.add_recorder( recorder=recorder )

    # Add objective
    prob.model.add_objective("nacelle_mass", ref=1e5)               #DONE: 'nacelle_mass' minimization (def: 'torq_deflection')
    # - NOTE: effectively 'lss_mass' minimization

    # Add design variables
    prob.model.add_design_var("L_12", lower=1.0, upper=10.0, ref=10.0)
    prob.model.add_design_var("L_h1", lower=1.0, upper=10.0, ref=10.0)
    prob.model.add_design_var("lss_diameter", lower=1.0, upper=4.0, ref=5.0)
    prob.model.add_design_var("lss_wall_thickness", lower=0.05, upper=0.9)#, ref=1e-2)

    # Add constraints
    
    # 1. von Mises stress util
    prob.model.add_constraint("constr_lss_vonmises", upper=1.0) #TODO: add next
    
    # 2. deflection (main bearing: max allowed as an angle)
    prob.model.add_constraint("constr_shaft_deflection", upper=1.0)     #TODO: add next
    prob.model.add_constraint("constr_shaft_angle", upper=1.0)          #TODO: add next

#%%[markdown]
# ### DOE setup
if flag_DOE: #NOTE: running 72 mins! on my PC for 6 levels x 4 DVs (khayr insha'Allah)

    prob.driver = om.DOEDriver(om.FullFactorialGenerator(levels=6))
    recorder = om.SqliteRecorder( loc_doe )
    prob.driver.add_recorder( recorder )

    # Add objective
    prob.model.add_objective("torq_deflection")#, ref=1e6)

    # Add design variables
    prob.model.add_design_var("L_12", lower=1.0, upper=10.0, ref=10.0)
    prob.model.add_design_var("L_h1", lower=1.0, upper=10.0, ref=10.0)
    prob.model.add_design_var("lss_diameter", lower=1.0, upper=4.0, ref=5.0)
    prob.model.add_design_var("lss_wall_thickness", lower=0.05, upper=0.9)

# %%
# Setup the problem
prob.setup()

# %%[markdown]
# Print objectives, design variables, and constraints in a concise readable form

print("\n=== All needed inputs to the model ===\n")
# for name, meta in prob.model.list_inputs(out_stream=None, val=False):
#     print(name)
prob.model.list_inputs();

print("\n=== All outputs from the model ===\n")
# for name, meta in prob.model.list_outputs(out_stream=None, val=False):
#     print(name)
prob.model.list_outputs();

# %% [markdown]
# ### Defining input values
# after calling prob.setup() (on the openMDAO 'prob' defined) and before calling prob.run_driver()

# 1. High-level Inputs
prob.set_val("machine_rating", 15.0, units="MW")
prob["rotor_diameter"] = 240.0
prob["rated_torque"] = 4308926.79641971

prob["upwind"] = True
prob["D_top"] = 6.5 #tower top diameter
# prob["minimum_rpm"] = 5
# prob["rated_rpm"] = 7.56
prob["hub_diameter"] = 7.94
prob["overhang"] = 11.35 #ref.2
prob["tilt"] = 6.0 #[deg] ref.3

# Loading from rotor
#(v) actual ULS loads (source: copied felix main_shafft_siz code 20251031)
# - NOTE: these are predscribed 50-yr extremes from extr DLCs (5.1,6.1,6.3) 
prob["F_aero_hub"] = np.array([5.3995*1e6, 1.3697*1e6, 5.5742*1e6]).reshape((3, 1))
prob["M_aero_hub"] = np.array([5.2515*1e7, 1.0747*1e8, 9.9481*1e7]).reshape((3, 1))
# ---

# %% [markdown]
# 2. Blade properties and hub design options
# - cf. `opts["flags"]["hub"]`

# Hub_Rotor_LSS_Frame inputs
# TODO: copied from made4wind_geared (IEA-15MW = ref), change to made4wind specs
if True:
    blade_mass = 65252.0
    n_blades = 3
    prob["blades_mass"] = n_blades * blade_mass
    prob["blades_cm"] = 2.46175
    prob["blades_I"] = np.r_[3.48453857e+08, 1.74226928e+08, 1.74226928e+08, np.zeros(3)]

    # if run HUB module within DriveSE
    if dohub:
        prob["flange_t2shell_t"] = 6.0
        prob["flange_OD2hub_D"] = 0.6
        prob["flange_ID2flange_OD"] = 0.8
        prob["hub_in2out_circ"] = 1.2
        prob["hub_stress_concentration"] = 3.0
        prob["n_front_brackets"] = 5
        prob["n_rear_brackets"] = 5
        prob["clearance_hub_spinner"] = 0.5
        prob["spin_hole_incr"] = 1.2
        prob["blade_root_diameter"] = 5.2

        prob["n_blades"] = 3
        prob["blade_mass"] = 65252.0
        prob["blades_mass"] = prob["n_blades"] * prob["blade_mass"]
        prob["blades_cm"] = 2.46175
        prob["blades_I"] = np.r_[3.48453857e+08, 1.74226928e+08, 1.74226928e+08, np.zeros(3)]

        prob["pitch_system.BRFM"] = 26648449.0
        prob["pitch_system_scaling_factor"] = 0.75

        prob["spinner_gust_ws"] = 70.0

    else:
        # run made4wind_geared.py with opt_flag = false and copy the following values from drivetrain_example.csv
        prob["hub_system_mass"] = 62561.91718921
        prob["hub_system_cm"] = 3.35947759
        prob["hub_system_I"] = np.array([[865503.52531197, 567289.77714803, 567289.77714803],[0., 0., 0.]])

# TODO: replace hub.py and user input hub inputs for LSS
# RNA_mass = 1017 (tons) (cf. tab. ES-1, ref.2)
# nacelle mass = 820.888 (tons) (cf. tab. 5-1, ref.2)
# nacelle mass minus hub = 630.888 (tons)
# so, hub mass = 190 (tons)

# %% [markdown]
# 3. Drivetrain configuration and sizing inputs

myones = np.ones(2)
 
# - init condn for some design vars
# prob["bear1.bearing_type"] = "CRB" # 1. floating MB
# prob["bear2.bearing_type"] = "TRB2" # 2. fixed MB
# prob["bear1.D_shaft"] = 4.0
# prob["bear2.D_shaft"] = 3.2

prob["L_h1"] = 4.25 #1.0; cf. L_rb in main_shaft_sizing code
prob["L_12"] = 7.1 #1.2
prob["lss_diameter"] = 4.0 * myones #1.0
prob["lss_wall_thickness"] = 0.288 * myones

# Gearbox inputs
# prob["L_gearbox"] = 1.5 #(v) calc in gearbox.py
# prob["gear_configuration"] = "eee"
# prob["planet_numbers"] = np.array([5, 3, 0]) #ref.1
prob["gear_ratio"] = 50 #.039
#prob["gearbox_mass_user"] = 0.0 #(cf. defined default 0.0 line 156, gearbox.py)
prob["gearbox_torque_density"] = 200.0 # (cf. line 210, gearbox.py)

prob["L_hss"] = 1.5
prob["hss_diameter"] = 0.5 * myones
prob["hss_wall_thickness"] = 0.1 * myones

# Generator inputs (TODO: add compn later)
# - needed by Bedplate_IBeam_Frame in drive_structure.py, output of HSS_Frame
# - copied from made4wind_geared.py's output drivetrain_example.csv
# prob["R_generator"] = 1.7999999999999998
prob["L_generator"] = 11.98398883842414 # maybe calc in generator.py
# prob["generator_cm"] = -0.09998102618633065
prob["generator_rotor_mass"] = 26437.71371233699
prob["generator_rotor_I"] = np.array([42829.09621398592, 31598.575743266098, 31598.575743266098])
# prob["F_generator"] = np.array([[-55905.04536116102], [-0.0], [-531900.9765713954]])
# prob["M_generator"] = np.array([[420611.2199999999], [-1687869.5522841304], [-0.0]])

# 'drive_height' : derive from the high-level inputs (output= 5.95522 m)
# - needed by layout.py (line 122)
def calc_drive_height(prob):
    L_fl = 0.358 #ref.2: Hub flange length 
    L2n = 0.9 #ref.2: Distance of downwind bearing from bedplate flange
    L_lss = prob["L_h1"]+prob["L_12"]+L2n
    H_nose = 4.875 #ref.2: Nose height (from tower top to bottom of bedplate flange)
    drive_height = H_nose + ( np.sin(np.deg2rad(prob["tilt"]))*( (prob["hub_diameter"]*np.sqrt(3/4))+L_fl+L_lss ) )
    print( f'    Calculated drive height: {drive_height} m' ) #5.95522 m
    return drive_height
prob["drive_height"] = calc_drive_height(prob)

# bedplate: Hub:_Rotor_LSS_Frame, Bedplate_IBeam_Frame inputs
# prob["bedplate_flange_width"] = 1.0
# prob["bedplate_flange_thickness"] = 0.1
# prob["bedplate_web_thickness"] = 0.1

prob["shaft_deflection_allowable"] = 1e-4 # within Hub_Rotor_LSS_Frame (below): Deflections and rotations at GB attachment
prob["shaft_angle_allowable"] = 1e-3
# prob["stator_deflection_allowable"] = 1e-4 # within Bedplate_IBeam_Frame (below)
# prob["stator_angle_allowable"] = 1e-3

# %% [markdown]
# 4. Material properties (discrete_inputs to `DriveMaterials`)

# DriveMaterials inputs
prob["E_mat"] = np.c_[200e9 * np.ones(3), 205e9 * np.ones(3), 118e9 * np.ones(3), [4.46e10, 1.7e10, 1.67e10]].T
prob["G_mat"] = np.c_[79.3e9 * np.ones(3), 80e9 * np.ones(3), 47.6e9 * np.ones(3), [3.27e9, 3.48e9, 3.5e9]].T
prob["Xt_mat"] = np.c_[450e6 * np.ones(3), 814e6 * np.ones(3), 310e6 * np.ones(3), [6.092e8, 3.81e7, 1.529e7]].T
# - (v, note) these would be  -np.c_-> (4,3) -.T-> (3,4) array
prob["rho_mat"] = np.r_[7800.0, 7850.0, 7200.0, 1940.0]
prob["Xy_mat"] = np.r_[345e6, 485e6, 265e6, 18.9e6]
prob["wohler_exp_mat"] = 1e1 * np.ones(4)
prob["wohler_A_mat"] = 1e1 * np.ones(4)
prob["unit_cost_mat"] = np.r_[0.7, 0.9, 0.5, 1.9]
# - Material assignment
prob["lss_material"] = prob["hss_material"] = "steel_drive"
prob["bedplate_material"] = "steel"
prob["hub_material"] = "cast_iron"
prob["spinner_material"] = "glass_uni"
prob["material_names"] = ["steel", "steel_drive", "cast_iron", "glass_uni"]
# ---

#%%[markdown]
# ### Final check before running
print("\n=== Final input check ===\n")
prob.model.list_inputs();
om.n2(prob, outfile=loc_n2, show_browser=True);

# %% [markdown]
# ### Running
# _model (analysis) or _driver (optimization)

if opt_flag:
    # set up the optimization and problem formulation
    # NOTE: objective "nacelle_mass" = 409215.9885880007, from drivetrain_example.csv

    # Run the optimization
    prob.model.approx_totals()
    prob.run_driver()

elif flag_DOE:
    # Run the DOE
    prob.run_driver()

else:
    # Run the analysis
    prob.run_model()

# %%[markdown]
# Print the results
print( prob["L_h1"], prob["L_12"], prob["lss_diameter"], prob["lss_wall_thickness"] )
# [3.48132032] [1.] [4. 4.] [0.32635334 0.29289825]
list_driver_vars = prob.list_driver_vars()

#%%
from wisdem.commonse.fileIO import save_data
loc_save_data = os.path.join(results_path, "01")
save_data(loc_save_data, prob)

#%%[markdown]
# Driver scaling report 
prob.driver.scaling_report(outfile=loc_scaling_report)

#%%
# ### Recorded cases
print("\n=== Recorded cases from the optimization ===\n")
results_dict = get_recorder_results( loc_cases, None, True )
results_dict
# %%
