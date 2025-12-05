# %% [markdown]
# # _Drivetrain optimization_ (`WISDEM`)
# single component optimization (as suggested by `WEIS` ppt)
# 
# ### current version:
# DT layout (without M4W GB optim):
#
# - objective: (1) `nacelle_mass` minimization (`NacelleSystemAdder`)
#
# - DVs: (4) `L_h1, L_12, lss_diameter, lss_wall_thickness`
#
# - constraints: (5) lss stresses, deflections (linear, angle), DT dims, L10 MBs FLS
#
# ### TODO:
# - think 🤔 and add DVs, constrs etc.
# - test LSS DVs and add 2 more constrs: mb* defl_ang
# - add M4W generator data (now using simple empirical gen)
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
# import scipy.io as sio # --- not used in here, but within imports
# import pickle

# %%
from wisdem.drivetrainse.drivetrain import DriveMaterials

from wisdem.drivetrainse.hub import Hub_System
from wisdem.drivetrainse.gearbox import Gearbox

import wisdem.drivetrainse.layout as lay

import wisdem.drivetrainse.drive_components as dc

import wisdem.drivetrainse.drive_structure as ds

from wisdem.commonse.utilities import get_recorder_results, mainshaft_loads_from_mat_to_dict, load_all_mat_to_dict
from wisdem.commonse.fileIO import save_data

# %% [markdown]
# ### Defining results directory and files
results_dir = "03_results"
script_dir = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(script_dir, results_dir)
os.makedirs(results_path, exist_ok=True)

# loc_cases = os.path.join(results_path, "cases_recorded.sql") #TODO
# if os.path.exists( loc_cases ):
#     os.remove( loc_cases)


loc_doe = os.path.join(results_path, "DOE_recorded.sql")
loc_n2 = os.path.join(results_path, "n2.html")
loc_scaling_report = os.path.join(results_path, 'scaling_report.html')
loc_save_data = os.path.join(results_path, "03")

#%% Loading `openFAST` hub loads from a saved file
part_loads = True 
load_fls_loads = False
# False: full loads (72e4,10) (200 Hz sampled, 60mins)
# True: part loads (72e3,11) (20 Hz sampled, 60mins)

dir_loads = "M:\Vasudev_Gupta\outputs_mainshaft_loads"

if part_loads: # define paths
    loc_all_loads_mat_file = os.path.join(dir_loads, "mainshaft_loads.mat")
    S_all, keys_all = load_all_mat_to_dict(loc_all_loads_mat_file)

else: # define paths
    loc_FLS_loads_mat_file = os.path.join(dir_loads, "mainshaft_loads_FLS_full.mat")
    loc_ULS_loads_mat_file = os.path.join(dir_loads, "mainshaft_loads_ULS.mat")
    if load_fls_loads: # load from paths
        Snew, keys_new = mainshaft_loads_from_mat_to_dict(
            loc_FLS_loads_mat_file, loc_ULS_loads_mat_file)

# %% [markdown]
# ### Defining options (`modelling_options`), flags

flag_opt_GBO = True        # GBO: gradient based optimizer
flag_DOE = False        # DOE: design of experiments
flag_opt_GFO = False     # GFO: gradient free optimizer

# define `modelling_options`
opts = {}

opts["WISDEM"] = {}
opts["WISDEM"]["n_dlc"] = 1
opts["WISDEM"]["DriveSE"] = {}
# NOTE "hub": 'Hub_System' component are NOT included in the 'DrivetrainSE_M4W' component 
opts["WISDEM"]["DriveSE"]["hub"] = {}
opts["WISDEM"]["DriveSE"]["hub"]["hub_gamma"] = 2.0
opts["WISDEM"]["DriveSE"]["hub"]["spinner_gamma"] = 1.5

opts["WISDEM"]["DriveSE"]["direct"] = False
opts["WISDEM"]["DriveSE"]["use_gb_torque_density"] = True # False =(GB  optim, in-capabale)

opts["WISDEM"]["DriveSE"]["gamma_f"] = 1.35 #IEC-1, 7.6.2.2a, pg.57
opts["WISDEM"]["DriveSE"]["gamma_m"] = 1.3  #IEC-1, 7.6.2.4, pg.59
opts["WISDEM"]["DriveSE"]["gamma_n"] = 1.0  #IEC-1, 7.6.1.3, pg.55
opts["WISDEM"]["DriveSE"]["nBins"] = 100    #used by (new) Analytical_FLS_Bearing_Life; =Number of bins for histogram MB FLS
# used as: gamma = gamma_f * gamma_m * gamma_n (within TODO)

opts["WISDEM"]["RotorSE"] = {}
opts["WISDEM"]["RotorSE"]["n_pc"] = 2 #cf. RPM_Input in drive_components.py
            # `n_pc`: Number of wind speeds to compute the power curve
opts["materials"] = {}
opts["materials"]["n_mat"] = 4

opts["flags"] = {}
dogen = opts["flags"]["generator"] = False
dohub = opts["flags"]["hub"] = False #(v)

opts["OpenFAST"] = {}
opts["OpenFAST"]["simulation"] = {}
opts["OpenFAST"]["simulation"]["DT"] = 0.05
# dir(ectory) where MS loads are stored .csv (?)
if part_loads:
    opts["OpenFAST"]["openfast_dir"] = loc_all_loads_mat_file
else:
    ValueError('Full loads not defined in openfast_dir<-OpenFAST<-modelling_options. Please define it first. jazakumAllahu khayr.')

opts["DLC_driver"] = {}
opts["DLC_driver"]["DLCs"] = [{}]
opts["DLC_driver"]["DLCs"][0]["DLC"] = "1.2"
opts["DLC_driver"]["DLCs"][0]["wind_speed"] = [ 5.,  7.,  9., 11., 13., 15., 17., 19., 21., 23., 25.]
opts["DLC_driver"]["DLCs"][0]["probabilities"] = [0.06541262, 0.14245179, 0.14299681, 0.12940412, 0.10735197, 0.0824332 , 0.05894909, 0.03942148, 0.02472593, 0.01457773, 0.00466888]

# %% [markdown]
# ### Defining the model `problem class`:
# as an openMDAO group that uses DrivetrainSE classes as components
class DrivetrainSE_M4W( om.Group ):
    """
    Group containing components for the layout of the LSS components
    """
    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        opt_drivese = self.options["modeling_options"]["WISDEM"]["DriveSE"]
        # OpenFAST: containing 1. simulation DT and 2. MS loads dir
        opt_openfast = self.options["modeling_options"]["OpenFAST"]
        # DLC: only 1 used '[0]': containing "wind_speed" and "probabilities"
        opt_DLC = self.options["modeling_options"]["DLC_driver"]["DLCs"][0]

        n_dlcs = self.options["modeling_options"]["WISDEM"]["n_dlc"]
        direct = opt_drivese["direct"]
        if direct:
            use_gb_torque_density = False
        else:
            use_gb_torque_density = opt_drivese["use_gb_torque_density"]
            
        dogen = self.options["modeling_options"]["flags"]["generator"]
        n_pc = self.options["modeling_options"]["WISDEM"]["RotorSE"]["n_pc"]
        flag_hub = self.options["modeling_options"]["flags"]["hub"] #TODO: this modified; remove and add hub as legacy
        
        # print flag information
        print("=== Problem 'DrivetrainSE_M4W' setting up ===")
        print(f"flag info: use_gb_torque_density={use_gb_torque_density}, dogen={dogen}, flag_hub={flag_hub}, direct={direct}")

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
                "hub", Hub_System(modeling_options=opt_drivese["hub"]),
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
        
        # All smaller components (from `dc`; empirical no load analysis)
        # - required by `Hub_Rotor_LSS_Frame`
        # 0. Main Bearings
        self.add_subsystem("bear1", dc.MainBearing())
        self.add_subsystem("bear2", dc.MainBearing())
        # -connecting = GearedLayout -to- bear(1,2) (NEW)
        self.connect("Dshaft_mb1", "bear1.D_shaft") #DONE: impl later
        self.connect("Dshaft_mb2", "bear2.D_shaft") #DONE: impl later
        # 1. brake system
        self.add_subsystem(
            "brake", dc.Brake(direct_drive=direct),
                promotes=["*"]
            )
        # 2. electronics
        self.add_subsystem(
            "elec", dc.Electronics(),
            promotes=["*"]
            )
        # 3. yaw system
        self.add_subsystem(
            "yaw", dc.YawSystem(),
            promotes=["yaw_mass", "yaw_mass_user", "yaw_I", "yaw_cm", "rotor_diameter", "D_top"]
            )
        
        # Generator (simple for now)
        self.add_subsystem(
            "rpm", dc.RPM_Input(n_pc=n_pc),
            promotes=["*"]
            )
        # - TODO: add M4W gen data / `if dogen:`
        self.add_subsystem(
            "gensimp", dc.GeneratorSimple(direct_drive=direct, n_pc=n_pc),
            promotes=["*"]
            )

        # Hub_Rotor_LSS_Frame:
        self.add_subsystem(
            "lss", ds.Hub_Rotor_LSS_Frame(n_dlcs=n_dlcs, modeling_options=opt_drivese),
                promotes=["*"]
            )
        # -connecting = bear(1,2) -to- Hub_Rotor_LSS_Frame (NEW)
        self.connect("bear1.face_width", "mb1_face_width") # mb_fw(s) shifted from GearedLayout to Hub_* to avoid cycle
        self.connect("bear2.face_width", "mb2_face_width")
        self.connect("bear1.mb_Reactions", "mb1_Reactions")
        self.connect("bear2.mb_Reactions", "mb2_Reactions")
        
        # FLS MBs (Analytical); TODO: input opt_openfast and opt_DLC.
        self.add_subsystem(
            "mb_fls", ds.Analytical_FLS_Bearing_Life(
                modeling_options=opt_drivese,
                openfast_options=opt_openfast,
                dlc_options=opt_DLC
                ),
            promotes_inputs=["L_h1","L_12", "rated_rpm","lifetime"],
            promotes_outputs=["constr_L10_mb1","constr_L10_mb2"]
        )
        # -connecting = bear(1,2) -to- Analy_*
        self.connect("bear2.mb_e", "mb_fls.e_mb") # same for both MBs ---
        self.connect("bear2.mb_p", "mb_fls.p_mb")
        self.connect("bear2.mb_X1", "mb_fls.X1_mb")
        self.connect("bear2.mb_Y1", "mb_fls.Y1_mb")
        self.connect("bear2.mb_X2", "mb_fls.X2_mb")
        self.connect("bear2.mb_Y2", "mb_fls.Y2_mb") # ---
        self.connect("bear1.mb_Cr", "mb_fls.Cr_mb1")
        self.connect("bear2.mb_Cr", "mb_fls.Cr_mb2")

        # HSS
        self.add_subsystem(
            "hss", ds.HSS_Frame(modeling_options=opt_drivese, n_dlcs=n_dlcs),
            promotes=["*"]
            )

        # Final tallying (mass summation)
        self.add_subsystem(
            "misc", dc.MiscNacelleComponents(direct_drive=direct),
            promotes=["*"]
            )
        self.add_subsystem(
            "nac", dc.NacelleSystemAdder(direct_drive=direct),
            promotes=["*"]
            )
        # -connecting NacelleSystemAdder to Layout
        self.connect("s_mb1", "mb1_cm") # mb*_cm is the s_* itself
        self.connect("s_mb2", "mb2_cm")
        self.connect("s_gearbox", "gearbox_cm")
        self.connect("s_generator", "generator_cm")
        # -connecting = bear(1,2) -to- NacelleSystemAdder
        # -- already done with Bedplate_* (below; to avoid a cycle)
        self.add_subsystem(
            "rna", dc.RNA_Adder(),
            promotes=["*"]
            )
        
        # Bedplate_IBeam_Frame:
        self.add_subsystem(
            "bed", ds.Bedplate_IBeam_Frame(modeling_options=opt_drivese, n_dlcs=n_dlcs),
                promotes=["*"]
            )
        # -connecting = bear(1,2) -to- Bedplate_*
        self.connect("bear1.mb_mass", "mb1_mass")
        # self.connect("bear1.mb_cm", "mb1_cm")
        self.connect("bear1.mb_I", "mb1_I")
        self.connect("bear1.mb_max_defl_ang", "mb1_max_defl_ang")
        self.connect("bear2.mb_mass", "mb2_mass")
        # self.connect("bear2.mb_cm", "mb2_cm")
        self.connect("bear2.mb_I", "mb2_I")
        self.connect("bear2.mb_max_defl_ang", "mb2_max_defl_ang")
        # -connecting = Bedplate_* to Yaw*
        self.connect("bedplate_rho", "yaw.rho")

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

            self.connect("hub_rho", "rho_castiron")
            self.connect("spinner_rho", "rho_fiberglass")

# %% [markdown]
# ### Setup the problem
# Define the problem
prob = om.Problem(reports=False)

# Define the model
prob.model = DrivetrainSE_M4W(modeling_options=opts) # an instance of the DrivetrainSE_M4W problem defined above

# %%[markdown]
# ### Optimization / DOE setup
# If performing optimization, set up the optimizer and settings

if flag_opt_GBO:
    print("=== running GBO ===")
    # Choose the (GBO) optimizer to use
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"
    prob.driver.options["tol"] = 1e-4 # default: 1e-6
    prob.driver.options["maxiter"] = 5 * 4
    prob.driver.options["disp"] = True
    prob.driver.options["debug_print"] = ["desvars", "objs", "nl_cons", "ln_cons"]
    # prob.driver.options # disp for debugging
    # prob.set_solver_print(level=2)

    # recorder = om.SqliteRecorder( loc_cases )     #TODO: add in final setup (full problem)
    # prob.driver.add_recorder( recorder=recorder )

elif flag_opt_GFO:
    print("=== running GFO ===")
    # GFO: gradient free optimizer
    prob.driver = om.SimpleGADriver()
    # prob.driver = om.DifferentialEvolutionDriver()
    prob.driver.options["debug_print"] = ["desvars", "objs", "nl_cons", "ln_cons"]
    # OSError: 'lss' <class Hub_Rotor_LSS_Frame>: Error calling compute(), exception: access violation reading 0x000001CB530B5FB0

elif flag_DOE: #NOTE: running 194 mins! on my PC for 10 levels x 4 DVs (khayr insha'Allah)
               # TAKES HOUR(S), with just 10 levels !!!!!!!!!!!! why?
    print("=== running DOE ===")
    prob.driver = om.DOEDriver(om.FullFactorialGenerator(levels=10))
    recorder = om.SqliteRecorder( loc_doe )
    prob.driver.add_recorder( recorder )

else:
    print("=== running analysis only (`run_model()`) ===")

#%%
# setup optimization: objs, desvars, cons
# - TODO: scaling (is better).
if flag_opt_GBO or flag_opt_GFO or flag_DOE:
    # === Add objective ===
    prob.model.add_objective("nacelle_mass", ref=1e5)               #DONE: 'nacelle_mass' minimization
    
    # === Add design variables === 
    # 1. LSS
    prob.model.add_design_var("L_12", lower=0.5, upper=10.0, ref=10.0, ref0=0.5)
    prob.model.add_design_var("L_h1", lower=0.5, upper=5.0, ref=5.0, ref0=0.5)
    prob.model.add_design_var("lss_diameter", lower=0.5, upper=4.0, ref=4.0, ref0=0.5)
    prob.model.add_design_var("lss_wall_thickness", lower=4e-3, upper=0.9, ref=1e-1) #DONE: scaled so driver sees lb=0, ub=1 (why? 0.05 causes probs)

    # 2. HSS (TODO: add later if needed)

    # 3. Bedplate (TODO: add later if needed)
    # prob.model.add_design_var("bedplate_web_thickness", lower=4e-3, upper=5e-1, ref=1e-2)
    # prob.model.add_design_var("bedplate_flange_thickness", lower=4e-3, upper=5e-1, ref=1e-2)
    # prob.model.add_design_var("bedplate_flange_width", lower=0.1, upper=2.0)

    # === Add constraints ===    
    if flag_DOE: pass # DOE: no constraints

    # 1. von Mises stress util
    prob.model.add_constraint("constr_lss_vonmises", upper=1.0)         #DONE: add next
    prob.model.add_constraint("constr_bedplate_vonmises", upper=1.0)    #TODO: add if needed

    # 2. deflection #NOTE: scaling is better
    prob.model.add_constraint("constr_shaft_deflection", upper=1.0)         #DONE: add next
    prob.model.add_constraint("constr_shaft_angle", upper=1.0, ref=1e-3)    #DONE: add next
    # --- MBs (main bearing: max perm is angle, + fls)
    prob.model.add_constraint("constr_L10_mb1", lower=1.0)                  #DONE: add next
    prob.model.add_constraint("constr_L10_mb2", lower=1.0)                  #DONE: add next
    # ----- bearing angular deflections (from Bedplate_*)
    prob.model.add_constraint("constr_mb1_defl", upper=1.0)                 #DONE: add next
    prob.model.add_constraint("constr_mb2_defl", upper=1.0)                 #DONE: add next
    # --- bedplate # TODO: add later if needed (gen stator / max bedplate end defl)
    # prob.model.add_constraint("constr_stator_deflection", upper=1.0)
    # prob.model.add_constraint("constr_stator_angle", upper=1.0)

    # 3. length: target overhang, hub height and LSS wrt. MBs
    prob.model.add_constraint("constr_length", lower=0.0)               #DONE: add later
    prob.model.add_constraint("constr_height", lower=0.0, ref=1e1)      #DONE: add later
    prob.model.add_constraint("constr_Lh1_MB1fw", lower=0.0, ref=1e1)   #DONE: add later
    prob.model.add_constraint("constr_L12_MBsFW", lower=0.0, ref=1e0)   #DONE: add later

# %%
# Setup the problem
prob.setup()

# %%[markdown]
# ###
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
# after calling `prob.setup()` (on the openMDAO `prob` defined) and before calling `prob.run_driver()`

# 1. High-level Inputs
# - TODO: check windIO (02_ref WTs) data and change below
prob.set_val("machine_rating", 15.0, units="MW")
prob["rotor_diameter"] = 240.0 # TODO: ref.1 = 240, geo_schema = 241.35064632
prob["rated_torque"] = 4308926.79641971
prob["minimum_rpm"] = 5.0 # needed by RPM_Input
prob["rated_rpm"] = 7.56
prob["lifetime"] = 25.0 #design life in years ('lifetime' from WEIS, WindIO)

prob["upwind"] = True
prob["D_top"] = 6.5 #tower top diameter
prob["hub_diameter"] = 7.94
prob["overhang"] = 11.35 #ref.2 = 11.35 ; geo_schema = 12.0313 
prob["tilt"] = 6.0 #[deg] ref.3

#%%[markdown]
# Loading `openFAST` hub loads from a saved file
#
# Snew, keys_all = mainshaft_loads_from_mat_to_dict(loc_FLS_loads_mat_file, loc_ULS_loads_mat_file)
#%% Loads assignment (ULS, FLS)
# ## ULS load loads (xD), input to Analy_*
# - NOTE: these are predscribed 50-yr extremes from extr DLCs (5.1,6.1,6.3) 
# prob["F_aero_hub"] = np.array([5.3995*1e6, 1.3697*1e6, 5.5742*1e6]).reshape((3, 1))
# prob["M_aero_hub"] = np.array([5.2515*1e7, 1.0747*1e8, 9.9481*1e7]).reshape((3, 1))
# TODO: change here for testing
# 1. partial loads (S_all)
if part_loads:
    prob['F_aero_hub'] = np.array( [S_all['Fx_max'], S_all['Fy_max'], S_all['Fz_max']] ).reshape((3, 1))
    prob['M_aero_hub'] = np.array( [S_all['Mx_max'], S_all['My_max'], S_all['Mz_max']] ).reshape((3, 1))
# 2. full loads (Snew)
else:
    prob['F_aero_hub'] = np.array( [Snew['Fx_max'], Snew['Fy_max'], Snew['Fz_mean']] ).reshape((3, 1))
    prob['M_aero_hub'] = np.array( [Snew['Mx_max'], Snew['My_max'], Snew['Mz_max']] ).reshape((3, 1))

# ## FLS load loads (xD), input to Analy_*; (72e4, 10)
# TODO: change here for testing
# 1. partial loads (S_all)
if load_fls_loads:
    if part_loads:
        prob['Fx_FLS'],prob['Fy_FLS'],prob['Fz_FLS'] = S_all['Fx'],S_all['Fy'],S_all['Fz']
        prob['Mx_FLS'],prob['My_FLS'],prob['Mz_FLS'] = S_all['Mx'],S_all['My'],S_all['Mz']
        prob['rot_speed'] = S_all['rot_speed']
        prob['mean_wind_speed'] = S_all['mean_wind_speed'][0]
        prob['Time'] = S_all['Time'][0]
    # 2. full loads (Snew)
    else:
        prob['Fx_FLS'],prob['Fy_FLS'],prob['Fz_FLS'] = Snew['Fx'],Snew['Fy'],Snew['Fz']
        prob['Mx_FLS'],prob['My_FLS'],prob['Mz_FLS'] = Snew['Mx'],Snew['My'],Snew['Mz']
        prob['rot_speed'] = Snew['rot_speed']
        prob['mean_wind_speed'] = Snew['ws']
        prob['Time'] = Snew['Time']

# TODO: make nice PPT with flow/chart of om.Problem here
# TODO: update using pCrunch's rainflow
# ---

# %% [markdown]
# 2. Blade properties and hub design options
# - cf. `opts["flags"]["hub"]`

# Hub_Rotor_LSS_Frame inputs
# TODO: from made4wind_geared (IEA-15MW = ref), change to made4wind specs
if True: #NOTE: True with `Hub_*`
    blade_mass = 65250 # from ref.2, tab. ES-2 (= made4wind specs also)
    n_blades = 3
    prob["blades_mass"] = n_blades * blade_mass
    prob["blades_cm"] = 2.46175
    prob["blades_I"] = np.r_[3.48453857e+08, 1.74226928e+08, 1.74226928e+08, np.zeros(3)]

    # if run HUB module within DrivetrainSE
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
        # run made4wind_geared.py with flag_opt_GBO = false and copy the following values from drivetrain_example.csv
        prob["hub_system_mass"] = 190e3 # from ref.2, tab. 5-1
        prob["hub_system_cm"] = 3.35947759
        prob["hub_system_I"] = np.array([[865503.52531197, 567289.77714803, 567289.77714803],[0., 0., 0.]])

# TODO: cm & I (hub_system_ & blades_) will change with DVs (L in lss)

# %% [markdown]
# 3. Drivetrain configuration and sizing inputs

myones = np.ones(2)
# - init condn for some design vars

# Main Bearing inputs
prob["bear1.bearing_type"] = "TRB2" # 1. floating MB
prob["bear2.bearing_type"] = "CRB" # 2. fixed MB
# prob["bear1.D_shaft"] = 2.0 #(def:2.0), 4.0
# prob["bear2.D_shaft"] = 2.0 #(def:2.0), 3.2
prob["bear1.mb_e"] = 3.5 # from 3.5-4.0 (TODO: find ref.)
prob["bear2.mb_e"] = 3.5
# prob["bear1.mb_p"] = 3.33
# prob["bear2.mb_p"] = 3.33

# Layout / lss inputs
prob["L_h1"] = 0.5 #(def: 2.0), 4.25; cf. L_rb in main_shaft_sizing code
prob["L_12"] = 7.0 #(def:1.2), 7.1
prob["lss_diameter"] = myones * 2.0 #(def:1.0), 4.0
prob["lss_wall_thickness"] = myones * 0.1 #(def:0.1), 0.3

# Gearbox inputs
# prob["L_gearbox"] = 1.5 #(v) calc in gearbox.py
# prob["gear_configuration"] = "eee"
# prob["planet_numbers"] = np.array([5, 3, 0]) #ref.1
prob["gear_ratio"] = 50 #.039
#prob["gearbox_mass_user"] = 0.0 #(cf. defined default 0.0 line 156, gearbox.py)
prob["gearbox_torque_density"] = 200.0 # (cf. line 210, gearbox.py)

# HSS (TODO: consider as DV if needed)
prob["L_hss"] = 1.5
prob["hss_diameter"] = 0.5 * myones
prob["hss_wall_thickness"] = 0.1 * myones

# Generator inputs (TODO: add compn later)
# - needed by Bedplate_IBeam_Frame in drive_structure.py, output of HSS_Frame
# - copied from made4wind_geared.py's output drivetrain_example.csv
# prob["R_generator"] = 1.7999999999999998
prob["L_generator"] = 2.15 #TODO: opts: 1. input from gen design (indar), 2. maybe calc in generator.py?, 3. 11.98398883842414 (from drivetrain_example.csv), 4. 2.0 (drivetrain_geared) or 2.15 (drivetrain_direct)
# prob["generator_cm"] = -0.09998102618633065
# prob["generator_rotor_mass"] = 26437.71371233699
# prob["generator_rotor_I"] = np.array([42829.09621398592, 31598.575743266098, 31598.575743266098])
# prob["F_generator"] = np.array([[-55905.04536116102], [-0.0], [-531900.9765713954]])
# prob["M_generator"] = np.array([[420611.2199999999], [-1687869.5522841304], [-0.0]])
generator_mass_375rpm = 14482 #[kg] (cf. Made4Wind D5.1, Tab.9)

# 'drive_height' : derive from the high-level inputs
# - needed by layout.py (line 123)
# - (def: 5.614 for 15MW DD)
def calc_drive_height(prob):
    L_fl = 0.358 #ref.2: Hub flange length 
    L2n = 0.9 #ref.2: Distance of downwind bearing from bedplate flange
    L_lss = prob["L_h1"]+prob["L_12"]+L2n
    H_nose = 4.875 #ref.2: Nose height (from tower top to bottom of bedplate flange)
    drive_height = H_nose + ( np.sin(np.deg2rad(prob["tilt"]))*( (prob["hub_diameter"]*np.sqrt(3/4))+L_fl+L_lss ) )
    print( f'    Calculated drive height: {drive_height} m' ) #5.95522 m
    return drive_height
# prob["drive_height"] = calc_drive_height(prob) #(output= 5.95522 m)
prob["drive_height"] = 5.614 # (def: 5.614 for 15MW DD)

# bedplate: Hub:_Rotor_LSS_Frame, Bedplate_IBeam_Frame inputs
prob["bedplate_flange_width"] = 1.0
prob["bedplate_flange_thickness"] = 0.1
prob["bedplate_web_thickness"] = 0.1

# `Hub_*` requires:
prob["shaft_deflection_allowable"] = 1e-4 # within Hub_Rotor_LSS_Frame (below): Deflections and rotations at GB attachment
prob["shaft_angle_allowable"] = 1e-3
# `Bedplate_IBeam_Frame` requires:
prob["stator_deflection_allowable"] = 1e-2 #(def: 1e-4 m; 1e-2)
prob["stator_angle_allowable"] = 1e-1 #(def: 1e-3 deg; 1e-1)

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
# ### Run: Optimization / DOE / Analysis
# `_driver` (optimization) / `_model` (analysis)

if flag_opt_GBO or flag_DOE:
    # Run GBO or DOE
    prob.model.approx_totals() # TODO.
    prob.run_driver()

elif flag_opt_GFO:
    # Run the GFO
    prob.run_driver()

else:
    # Run the analysis
    prob.run_model()

# %%[markdown]
# Print the results
print("LSS desvars:")
print(" ", prob["L_h1"], prob["L_12"], prob["lss_diameter"], prob["lss_wall_thickness"] )
# [3.48132032] [1.] [4. 4.] [0.32635334 0.29289825]
print("F_mb*:")
print(" ", prob["F_mb1"], prob["F_mb2"] )
print("M_mb*:")
print(" ", prob["M_mb1"], prob["M_mb2"] )
print("constr_L10_mb(1,2):", prob["constr_L10_mb1"], prob["constr_L10_mb2"] )
#
print(f"MSA mass: {prob["msa_mass"]}")
print(f"nacelle mass: {prob["nacelle_mass"]}")


list_driver_vars = prob.list_driver_vars()

#%%[markdown]
# Driver scaling report 
prob.driver.scaling_report(outfile=loc_scaling_report)

#%%
### Recorded cases #TODO
# print("\n=== Recorded cases from the optimization ===\n")
# results_dict = get_recorder_results( loc_cases, None, True )
# results_dict
# ===============================================================

#%%
save_data(loc_save_data, prob)

#%%
