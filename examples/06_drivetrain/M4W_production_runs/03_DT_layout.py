# %% [markdown]
# # _Drivetrain optimization_ (`WISDEM`)
# single component optimization (as suggested by `WEIS` ppt)
# 
# ### current version:
# DT layout (without M4W GB optim):
#
# - objective: (1) `nacelle_mass` minimization (`NacelleSystemAdder`)
#
# - DVs (10): dims of LSS, HSS, Bedplate
#
# - constraints (13): all, except bedplate-end (stator) defl and ang 
#
# - CONVERGED Alhamdolillah! need 80 optim iters!
#
# ### TODO:
# - 
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
import time
import matplotlib.pyplot as plt
# import scipy.io as sio # --- not used in here, but within imports
# import pickle

# %%
from wisdem.drivetrainse.drivetrain import DriveMaterials, DrivetrainSE_M4W

from wisdem.drivetrainse.hub import Hub_System
from wisdem.drivetrainse.gearbox import Gearbox

import wisdem.drivetrainse.layout as lay

import wisdem.drivetrainse.drive_components as dc

import wisdem.drivetrainse.drive_structure as ds

from wisdem.commonse.utilities import get_recorder_results, mainshaft_loads_from_mat_to_dict, load_all_mat_to_dict
from wisdem.commonse.fileIO import save_data
# %% [markdown]
# ### Define flags
# post-processing results
make_xdsm = False       # html-show or detailed pdf
record_cases = False    #TODO: add in final setup (full problem)
plot_cases = False      #NOTE: saved, not changing now (commented)
flag_scaling_show_browser = False
flag_save_new_data = False

# Loading `openFAST` hub loads from a saved file
part_loads = True 
load_fls_loads = False
# False: full loads (72e4,10) (200 Hz sampled, 60mins)
# True: part loads (72e3,11) (20 Hz sampled, 60mins)
dir_loads = "M:\Vasudev_Gupta\outputs_mainshaft_loads"

# Optimization flags
flag_opt_GBO = True     # GBO: gradient based optimizer
flag_DOE = False        # DOE: design of experiments
flag_opt_GFO = False    # GFO: gradient free optimizer

# Parametric study
flag_study_parametric = True
param_for_study = "LDD"     # "MB" (types) / "LDD" (MS' L_*)
meth_Peq = "DEL".lower()    # "LRD" or "DEL"

# %% [markdown]
# ### Defining results directory and files
results_dir = "03_results"
script_dir = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(script_dir, results_dir)
os.makedirs(results_path, exist_ok=True)

loc_doe = os.path.join(results_path, "DOE_recorded.sql")
loc_n2 = os.path.join(results_path, "n2.html")
loc_scaling_report = os.path.join(results_path, 'scaling_report.html')
loc_save_data = os.path.join(results_path, "03")

if make_xdsm: loc_xdsm = os.path.join(results_path, 'xdsm_03')

# Record results?
if record_cases:
    print(" ---- Recording cases using `SqliteRecorder` ---- ")
    loc_cases = os.path.join(results_path,
        "cases_recorded_"+meth_Peq+".sql")
    if os.path.exists( loc_cases ):
        os.remove( loc_cases )

#%% Loading `openFAST` hub loads from a saved file
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
dohub = opts["flags"]["hub"] = True #(v)
doMBfls = opts["flags"]["mb_fls"] = True

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
    prob.driver.options["maxiter"] = 5 * 20 # needs 80 iters to converge
    prob.driver.options["disp"] = True
    prob.driver.options["debug_print"] = ["desvars", "objs", "nl_cons", "ln_cons"]
    # prob.driver.options # disp for debugging
    # prob.set_solver_print(level=2)

    if record_cases:
        recorder = om.SqliteRecorder( loc_cases )
        prob.driver.add_recorder( recorder=recorder )

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
    prob.model.add_objective("nacelle_mass", ref=1e6)               #DONE: 'nacelle_mass' minimization
    
    # === Add design variables === 
    # 1. LSS
    prob.model.add_design_var("L_h1", lower=0.2, upper=5.0, ref=5.0, ref0=0.2)
    prob.model.add_design_var("L_12", lower=0.5, upper=10.0, ref=10.0, ref0=0.5)
    prob.model.add_design_var("lss_diameter", lower=0.5, upper=4.0, ref=4.0, ref0=0.5)
    prob.model.add_design_var("lss_wall_thickness", lower=4e-3, upper=0.9, ref=0.9, ref0=4e-3) #DONE: scaled so driver sees lb=0, ub=1 (why? 0.05 causes probs)

    # 2. HSS (TODO: add later if needed)
    prob.model.add_design_var("L_hss", lower=0.1, upper=5.0, ref=5.0, ref0=0.1)
    prob.model.add_design_var("hss_diameter", lower=0.5, upper=6.0, ref=6.0, ref0=0.5)
    prob.model.add_design_var("hss_wall_thickness", lower=4e-3, upper=0.5, ref=0.5, ref0=4e-3)

    # 3. Bedplate (TODO: add later if needed)
    prob.model.add_design_var("bedplate_web_thickness", lower=4e-3, upper=5e-1, ref=5e-1, ref0=4e-3)
    prob.model.add_design_var("bedplate_flange_thickness", lower=4e-3, upper=5e-1, ref=5e-1, ref0=4e-3)
    prob.model.add_design_var("bedplate_flange_width", lower=0.1, upper=2.0, ref=2.0, ref0=0.1)

    # 4. hub
    # prob.model.add_design_var("hub_diameter", lower=2.0, upper=5.0)

    # === Add constraints ===    
    if flag_DOE: pass # DOE: no constraints

    # 1. von Mises stress util
    prob.model.add_constraint("constr_lss_vonmises", upper=1.0)         #DONE: add next
    prob.model.add_constraint("constr_bedplate_vonmises", upper=1.0)    #TODO: add if needed
    prob.model.add_constraint("constr_hss_vonmises", upper=1.0)

    # 2. deflection #NOTE: scaling is better
    prob.model.add_constraint("constr_shaft_deflection", upper=1.0)         #DONE: add next
    prob.model.add_constraint("constr_shaft_angle", upper=1.0, ref=1e-3)    #DONE: add next
    # --- MBs (main bearing: max perm is angle, + fls)
    if doMBfls:
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

    # 4. hub
    # - hub dia to accom. blades' roots
    # prob.model.add_constraint("constr_hub_diameter", lower=0.0)

# %%
# Setup the problem
prob.setup()

#%%[markdown]
### pyXDSM trial
#%%
if make_xdsm:
    from omxdsm import write_xdsm
    write_xdsm(
        prob,
        filename=loc_xdsm,
        out_format='pdf', # pdf / html
        show_browser=True,
        quiet=False,
        output_side='left',
        include_indepvarcomps=False,
        class_names=False
    )
# -----

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
prob["rated_torque"] = 21.03*1e6 # [Nm] ref.2, tab.5-4
prob["minimum_rpm"] = 5.0 # needed by RPM_Input
prob["rated_rpm"] = 7.56
if doMBfls:
    prob["lifetime"] = 25.0 #design life in years ('lifetime' from WEIS, WindIO)

prob["upwind"] = True
prob["D_top"] = 6.5 #tower top diameter
prob["hub_diameter"] = 7.94
prob["overhang"] = 12.0313 #ref.2 = 11.35 ; geo_schema = 12.0313 
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
        prob["flange_t2shell_t"] = 6.0      # ---- flange MS data ----
        prob["flange_OD2hub_D"] = 0.6
        prob["flange_ID2flange_OD"] = 0.8   # ----
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
prob["bear1.bearing_type"] = "CRB" # 1. floating MB
prob["bear2.bearing_type"] = "TRB2" # 2. fixed MB
prob["bear1.mb_e"] = 3.5 # from 3.5-4.0 (TODO: find ref.)
prob["bear2.mb_e"] = 3.5
if doMBfls:
    prob["mb_fls.e_mb"] = prob["bear2.mb_e"]

# Layout / lss inputs
prob["L_h1"] = 0.301 #(def: 2.0), 4.25; cf. L_rb in main_shaft_sizing code
prob["L_12"] = 5.063 #(def:1.2), 7.1
prob["lss_diameter"] = np.array([3.402, 1.816]) #(def:1.0), 4.0
prob["lss_wall_thickness"] = np.array([0.011, 0.102]) #(def:0.1), 0.3

# Gearbox inputs
# prob["L_gearbox"] = 1.5 #(v) calc in gearbox.py
# prob["gear_configuration"] = "eee"
# prob["planet_numbers"] = np.array([5, 3, 0]) #ref.1
prob["gear_ratio"] = 50 #.039
prob["gearbox_mass_user"] = 135.5*1e3 # D5.1 R2
# prob["gearbox_torque_density"] = 200.0 # (cf. line 210, gearbox.py)

# HSS (DONE: consider as DV if needed)
prob["L_hss"] = 0.101
prob["hss_diameter"] = np.array([0.638, 1.095])
prob["hss_wall_thickness"] = np.array([0.034, 0.047])

# === Generator inputs (DONE: add compn later)
# - needed by Bedplate_IBeam_Frame in drive_structure.py, output of HSS_Frame
# TODO: opts:
# --- 1. input from gen design (indar's ismael),
# --- 2. maybe calc in generator.py?,
# --- 3. 11.98398883842414 (from drivetrain_example.csv),
# --- 4. 2.0 (drivetrain_geared) or 2.15 (drivetrain_direct)

# prob["R_generator"] = 1.7999999999999998
# prob["generator_cm"] = -0.09998102618633065
# prob["generator_rotor_mass"] = 26437.71371233699
# prob["generator_rotor_I"] = np.array([42829.09621398592, 31598.575743266098, 31598.575743266098])
# prob["F_generator"] = np.array([[-55905.04536116102], [-0.0], [-531900.9765713954]])
# prob["M_generator"] = np.array([[420611.2199999999], [-1687869.5522841304], [-0.0]])

# TODO: Indar generator dimensions (D5.4):
prob["generator_mass_user"] = 34.8*1e3 # D5.4, tab.9
# # Overall dimensions TODO: wrong! correct!!
# H_generator, W_generator, L_generator = 2.4, 0.8, 4.2 #[m]
prob["L_generator"] = 2.15
# # -- make an equivalent cylinder from the cuboid with the SAME (mass) MoI
# prob["R_generator"] = np.sqrt( (H_generator**2 + W_generator**2)/6 ) # 1.0328
gen_eff = 0.9805
prob["generator_efficiency_user"] = np.array([ [0.0,1.0],[gen_eff,gen_eff] ])

# === Electronics input (ING: converter, transformer)
# converter mass = 3 Tn per 8MW conversion line (ING Bidane's email)
prob["converter_mass_user"] = (3*1e3*15)/8 # 5,625 [kg]
# overall dims (est. very preliminary): 2400x800x4200 mm [HxWxL]
H_converter, W_converter, L_converter = 2.4, 0.8, 4.2 # [m]

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
# --- below vals from ONLY bedplate optim (desvars, constr) for nacelle mass min
prob["bedplate_flange_width"] = 1.998
prob["bedplate_flange_thickness"] = 0.023
prob["bedplate_web_thickness"] = 0.023

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
prob["bedplate_material"] = "steel_drive" # steel -> steel_drive
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
#%%
if flag_opt_GBO or flag_DOE:
    # Run GBO or DOE
    t0 = time.time()
    # main GBO
    prob.model.approx_totals() # TODO.
    prob.run_driver()
    
    t1 = time.time()
    print(" - WISDEM run completed in,", t1-t0, "seconds")

elif flag_opt_GFO:
    # Run the GFO
    prob.run_driver()

else:
    # Run the analysis
    prob.run_model()

# %%[markdown]
# # _____ Post-processing _____

# %%
# Print the results
print("LSS desvars:")
print(" ", prob["L_h1"], prob["L_12"], prob["lss_diameter"], prob["lss_wall_thickness"] )
# TODO: for flange mass, dohub (cf. var `flange_t2shell_t`)
print("HSS desvars:")
print(" ", prob["L_hss"], prob["hss_diameter"], prob["hss_wall_thickness"] )
print("Bedplate desvars:")
print(" ", prob["bedplate_flange_width"], prob["bedplate_flange_thickness"], prob["bedplate_web_thickness"] )
print(" ")
print("F_mb*:")
print(" ", prob["F_mb1"], prob["F_mb2"] )
print("M_mb*:")
print(" ", prob["M_mb1"], prob["M_mb2"] )
if doMBfls:
    print("constr_L10_mb(1,2):", prob["constr_L10_mb1"], prob["constr_L10_mb2"] )
print("--- constr_ max ---")
print("- lss: ",
      np.max(prob["constr_lss_vonmises"])
      )
print("- hss: ",
      np.max(prob["constr_hss_vonmises"])
      )
print("- bedplate: ",
      np.max(prob["constr_bedplate_vonmises"])
      )
#
print("--- obj: masses ---")
print(f"MSA mass: {prob["msa_mass"]}")
print(f"nacelle mass: {prob["nacelle_mass"]}")


# list_driver_vars = prob.list_driver_vars()

#%%[markdown]
# Driver scaling report 
prob.driver.scaling_report(
    outfile=loc_scaling_report,show_browser=flag_scaling_show_browser
);
#%%
### Recorded cases
if record_cases:
    print("\n=== Recorded cases from the optimization ===\n")
    results_dict = get_recorder_results( loc_cases, None, True )
    print(results_dict);

# ### Plot recorded results TODO

#%%
if flag_save_new_data: save_data(loc_save_data, prob)
# ===============================================================
#%%[markdown]
# # Plot recorded results
#%%
# main colors
clr_blueDark = '#313694'
clr_blueLight = '#A6CAEC'
clr_redDark = '#C00000'
clr_redLight = 'r'
# -------------------------
# options: Journal polish
# plot rc params
params_plot_rc = {
        "font.size": 24,
        "axes.labelsize": 24,
        "legend.fontsize": 24, # 16 for pdf of `var_with_iter` plot
        "lines.linewidth": 2,
        "lines.markersize": 6,
    }
plt.rcParams.update( params_plot_rc )

fontsize = 18

#%%[markdown]
# ### Drivetrain mass comparison (IEA and M4W)
#%%
if plot_cases:
    # --------------------------------------------------
    # Data (example values, replace with your real ones)
    # --------------------------------------------------
    components = [
        "Main shaft",
        "Turret nose",
        "Main bearings",
        "Gearbox",
        "High-speed shaft",
        "Brake",
        "Generator",
        "Converter",
        "Transformer",
        "Misc. components",
        "Bedplate",
        "Yaw system",
    ]

    # Masses in tonnes [t]
    mass_IEA = {
        "Main shaft":       15.734,
        "Turret nose":      11.394,
        "Main bearings":    7.894,
        "Gearbox":          0.0,
        "High-speed shaft": 0.0,
        "Brake":            0.0,        # TODO
        "Generator":        371.592,
        "Converter":        0.0,        # TODO
        "Transformer":      0.0,        # TODO
        "Misc. components": 50.0,
        "Bedplate":         70.329,
        "Yaw system":       100.0,
    }

    mass_M4W = {
        "Main shaft":       prob["lss_mass"][0] / 1e3,
        "Turret nose":      0.0,
        "Main bearings":    2.0*prob["mean_bearing_mass"][0] / 1e3,
        "Gearbox":          prob["gearbox_mass"][0] / 1e3,
        "High-speed shaft": prob["hss_mass"][0] / 1e3,
        "Brake":            prob["brake_mass"][0] / 1e3,
        "Generator":        prob["generator_mass"][0] / 1e3,
        "Converter":        prob["converter_mass"][0] / 1e3,
        "Transformer":      prob["transformer_mass"][0] / 1e3,
        "Misc. components": (prob["hvac_mass"][0]+prob["platform_mass"][0]+prob["cover_mass"][0]) / 1e3,
        "Bedplate":         prob["bedplate_mass"][0] / 1e3,
        "Yaw system":       prob["yaw_mass"][0] / 1e3,
    }

    total_IEA = sum(mass_IEA.values())
    total_M4W = sum(mass_M4W.values())

    # --------------------------------------------------
    # Styling (colors + hatching)
    # --------------------------------------------------
    # Consistent hatching / coloring
    hatches = ['/', '\\', 'x', '-', '+', 'o', 'O', '.', '*', '//', 'xx', '++']
    tab10 = plt.cm.tab10.colors
    colors = list(tab10) + list(tab10[:2])  # extend to 12 components

    # --------------------------------------------------
    # Figure
    # --------------------------------------------------
    # --- Figure setup ---
    fig, ax = plt.subplots(figsize=(9, 16))

    x = np.array([0, 1])
    labels = ["IEA 15 MW", "MADE4WIND 15 MW"]
    bar_width = 0.45

    # --- Stacking ---
    bottom_IEA = 0.0
    bottom_M4W = 0.0
    tops_IEA, tops_M4W = [], []

    for i, comp in enumerate(components):
        ax.bar(
            x[0], mass_IEA[comp], bottom=bottom_IEA,
            width=bar_width, color=colors[i],
            hatch=hatches[i], edgecolor="black",
            label=comp,
        )

        ax.bar(
            x[1], mass_M4W[comp], bottom=bottom_M4W,
            width=bar_width, color=colors[i],
            hatch=hatches[i], edgecolor="black",
        )

        tops_IEA.append(bottom_IEA + mass_IEA[comp])
        tops_M4W.append(bottom_M4W + mass_M4W[comp])

        bottom_IEA += mass_IEA[comp]
        bottom_M4W += mass_M4W[comp]

    # --- Dotted connectors (top of each component) ---
    for y_iea, y_m4w in zip(tops_IEA, tops_M4W):
        ax.plot(
            [x[0] + bar_width / 2, x[1] - bar_width / 2],
            [y_iea, y_m4w],
            linestyle=":", color="black", linewidth=1.2
        )

    # --- Total mass labels ---
    total_IEA = sum(mass_IEA.values())
    total_M4W = sum(mass_M4W.values())
    offset = 8.0

    ax.text(x[0], total_IEA + offset, rf"${total_IEA:.0f}\,\mathrm{{t}}$",
            ha="center", va="bottom", fontsize=fontsize, fontweight="bold")
    ax.text(x[1], total_M4W + offset, rf"${total_M4W:.0f}\,\mathrm{{t}}$",
            ha="center", va="bottom", fontsize=fontsize, fontweight="bold")

    # --- Formatting ---
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"Mass [t]")
    ax.set_title("Comparison of nacelle mass distribution")
    ax.legend(
        loc="upper right",
        fontsize=fontsize, frameon=True
    )
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    # -------------------------
    # save
    # -------------------------
    plot_path = os.path.join(results_path,
            "compare_mass_"+meth_Peq+".png")
    # plt.savefig(plot_path) # NOTE: saved, so don't change now 

    plt.show()

#%%[markdown]
# ### Convergence/parametric study setup
# 1. vary chosen GRs (and rspt. GB and gen weights)
# -- and save results for nacelle mass optim
# -- so varied= `gear_ratio`, `gearbox_mass_user`, `generator_mass_user`
#%%
gearbox_ratios = [300, 375, 500, 600]/7.56
gearbox_weights = [98842.8719028457, 99161.5464738645, 99766.6156499942, 100767.901889412]
generator_weights = [ 43.1, 34.8, 26.69, 22.57 ] * 1e3 #[kg] (D5.4, tab.10)