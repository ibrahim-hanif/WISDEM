# %% [markdown]
# Alhamdolillah
# # _Drivetrain-Tower optimization_ (`WISDEM`)
# ### main script for:
# 1. two-scope optimization (as suggested by `WEIS` ppt)
# 2. single optimization run
# 3. save results in csv file for further studies (eg. parallel)
# 
# ### current version: TODO
# - copying from 03_DT_layout & tower_direct
# - next step: setup the om problem
#
# ### goal of current version:
# DT layout (cf. 03_) + Tower (cf. 01_):
# - objective: (1) `NT_mass` minimization
# - DVs (10+T): dims of LSS, HSS, Bedplate, tower dia and thickness
# - constraints (13+T): all, except bedplate-end (stator) defl and ang + tower
#
# - CONVERGED? need ?? optim iters?
#
# ### TODO:

# %%
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
from wisdem.commonse.fileIO import save_data, load_data, get_variable_list
import wisdem.commonse.fileIO as IO

from wisdem.towerse.tower import TowerSE

# %%
# ### Define flags - DT TODO

# pre-processing; Loading `openFAST` hub loads from a saved file
part_loads = True 
load_fls_loads = False
# False: full loads (72e4,10) (200 Hz sampled, 60mins)
# True: part loads (72e3,11) (20 Hz sampled, 60mins)
dir_loads = "M:\\Vasudev_Gupta\\outputs_mainshaft_loads"

# Optimization flags
flag_opt_GBO = True     # GBO: gradient based optimizer
flag_debug_print = True
flag_parallel = False

make_xdsm = False       # html-show or detailed pdf
record_cases = False    #TODO: add in final setup (full problem)

# post-processing results
plot_cases = False      #NOTE: saved, not changing now (commented)
flag_scaling_show_browser = False
flag_save_new_data = False

# Parametric study
flag_study_parametric = False
# if True: init drive prob with 1 iter and reset to require (80) iters
maxIter_param = 1
maxIter_req = 5 * 16 # 80
if flag_study_parametric: maxIter = maxIter_param
else: maxIter = maxIter_req

#%%
# ### Defining results directory and files
# - results main dir
results_dir = "M4W_03b_results"
script_dir = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(script_dir, results_dir)
os.makedirs(results_path, exist_ok=True)

# - used within optimization
loc_record_doe = os.path.join(results_path, "DOE_recorded.sql")
loc_n2 = os.path.join(results_path, "n2.html")
# -- XDSM?
if make_xdsm:
    loc_xdsm = os.path.join(results_path, 'xdsm')
# -- Record results?
if record_cases:
    print(" ---- Recording cases using `SqliteRecorder` ---- ")
    loc_cases = os.path.join(results_path,
        "cases_recorded.sql")
    if os.path.exists( loc_cases ):
        os.remove( loc_cases )

# - post-processing
loc_scaling_report = os.path.join(results_path, 'scaling_report.html')
loc_save_data = os.path.join(results_path, "saved_data")
load_from_saved_data = False
if os.path.exists(loc_save_data+".csv"): load_from_saved_data = True

#%% Loading `openFAST` hub loads from a saved file
if part_loads: # define paths
    loc_all_loads_mat_file = os.path.join(dir_loads, "hub_loads_M4w.mat")
    S_all, keys_all = load_all_mat_to_dict(loc_all_loads_mat_file)

else: # define paths
    loc_FLS_loads_mat_file = os.path.join(dir_loads, "mainshaft_loads_FLS_full.mat")
    loc_ULS_loads_mat_file = os.path.join(dir_loads, "mainshaft_loads_ULS.mat")
    if load_fls_loads: # load from paths
        Snew, keys_new = mainshaft_loads_from_mat_to_dict(
            loc_FLS_loads_mat_file, loc_ULS_loads_mat_file)

#%%
# Set analysis and optimization options and define geometry
plot_flag = True
opt_flag = True

n_control_points = 3
n_materials = 4 + 1 #(DT+tower)
n_load_cases = 1 # in examples: 1 - DT; 2 - tower

h_param = np.diff(np.linspace(0.0, 87.6, n_control_points))
d_param = np.linspace(8.0, 3.87, n_control_points)
t_param = np.linspace(0.08, 0.02, n_control_points)
max_diam = 9.0

#%%
# ### Defining options (`modelling_options`) - DT TODO

# define `modelling_options`
opts = {}

opts["WISDEM"] = {}
opts["WISDEM"]["n_dlc"] = n_load_cases
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
opts["materials"]["n_mat"] = n_materials

opts["flags"] = {}
dogen = opts["flags"]["generator"] = False
dohub = opts["flags"]["hub"] = False
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
opts["DLC_driver"]["DLCs"][0]["wind_speed"] = [ 5.,  7.,  9., 11., 13., 15., 17., 19., 21., 23.]
opts["DLC_driver"]["DLCs"][0]["probabilities"] = [0.06541262, 0.14245179, 0.14299681, 0.12940412, 0.10735197, 0.0824332, 0.05894909, 0.03942148, 0.02472593, 0.0083042]

# TowerSE
opts["WISDEM"]["TowerSE"] = {}
opts["WISDEM"]["TowerSE"]["buckling_method"] = "eurocode"
opts["WISDEM"]["TowerSE"]["buckling_length"] = 15.0
opts["WISDEM"]["TowerSE"]["n_refine"] = 3
# ---- safety factors
opts["WISDEM"]["TowerSE"]["gamma_f"] = 1.35
opts["WISDEM"]["TowerSE"]["gamma_m"] = 1.3
opts["WISDEM"]["TowerSE"]["gamma_n"] = 1.0
opts["WISDEM"]["TowerSE"]["gamma_b"] = 1.1
opts["WISDEM"]["TowerSE"]["gamma_fatigue"] = 1.35 * 1.3 * 1.0
# ----  Frame3DD options
opts["WISDEM"]["TowerSE"]["frame3dd"] = {}
opts["WISDEM"]["TowerSE"]["frame3dd"]["shear"] = True
opts["WISDEM"]["TowerSE"]["frame3dd"]["geom"] = True
opts["WISDEM"]["TowerSE"]["frame3dd"]["tol"] = 1e-9
opts["WISDEM"]["TowerSE"]["frame3dd"]["modal_method"] = 1
opts["WISDEM"]["TowerSE"]["rank_and_file"] = True
# ----  geo?
opts["WISDEM"]["TowerSE"]["n_height"] = n_control_points
opts["WISDEM"]["TowerSE"]["n_layers"] = 1
opts["WISDEM"]["TowerSE"]["wind"] = "PowerWind"

#%%
# ### Setup the problem
# Define the problem
prob = om.Problem(reports=False)

# Define the model TODO
# = make group with DT and tower: cf. WT_RNTA in wisdem

prob.model = DrivetrainSE_M4W(opts=opts) # an instance of the DrivetrainSE_M4W problem defined above
prob.model = TowerSE(modeling_options=opts)

#%%
# ### Optimization / DOE setup
# If performing optimization, set up the optimizer and settings

if flag_opt_GBO:
    print("=== running GBO ===")
    # Choose the (GBO) optimizer to use
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"
    prob.driver.options["tol"] = 1e-4 # 1e-4; def: 1e-6
    prob.driver.options["maxiter"] = maxIter # needs 80 iters to converge
    prob.driver.options["disp"] = True
    if flag_debug_print:
        prob.driver.options["debug_print"] = [
            "desvars", "objs", "nl_cons", "ln_cons"]
    # prob.driver.options # disp for debugging
    # prob.set_solver_print(level=2)

    if record_cases:
        recorder = om.SqliteRecorder( loc_cases )
        prob.driver.add_recorder( recorder=recorder )

else:
    print("=== running analysis only (`run_model()`) ===")

#%%
# setup optimization: objs, desvars, cons
# - TODO: scaling (is better).
if flag_opt_GBO:
    # === Add objective ===
    prob.model.add_objective("nacelle_mass", ref=1e6) #TODO, make NT grp up
    
    # === Add design variables === 
    # 1. LSS
    prob.model.add_design_var("L_h1", lower=0.1, upper=5.0, ref=5.0, ref0=0.1)
    prob.model.add_design_var("L_12", lower=0.1, upper=8.0, ref=8.0, ref0=0.1)
    # prob.model.add_design_var("delta", lower=0.1, upper=5.0, ref=5.0, ref0=0.1)
    prob.model.add_design_var("lss_diameter", lower=1.0, upper=5.0, ref=5.0, ref0=1.0)
    prob.model.add_design_var("lss_wall_thickness", lower=4e-3, upper=1.0, ref=1.0, ref0=4e-3) #DONE: scaled so driver sees lb=0, ub=1 (why? 0.05 causes probs)

    # 2. HSS
    prob.model.add_design_var("L_hss", lower=0.1, upper=5.0, ref=5.0, ref0=0.1)
    prob.model.add_design_var("hss_diameter", lower=0.5, upper=5.0, ref=5.0, ref0=0.5)
    prob.model.add_design_var("hss_wall_thickness", lower=4e-3, upper=0.2, ref=0.2, ref0=4e-3)

    # 3. Bedplate
    prob.model.add_design_var("bedplate_web_thickness", lower=5e-3, upper=1.0, ref=1.0, ref0=5e-3)
    prob.model.add_design_var("bedplate_flange_thickness", lower=5e-3, upper=1.0, ref=1.0, ref0=5e-3)
    prob.model.add_design_var("bedplate_flange_width", lower=0.01, upper=3.0, ref=3.0, ref0=0.01)

    # 4. Tower (TODO: use vals from 01_'s analy opts)
    prob.model.add_design_var("tower_outer_diameter_in", lower=3.87, upper=max_diam)
    prob.model.add_design_var("tower_layer_thickness", lower=4e-3, upper=2e-1)

    # === Add constraints ===    
    # 1. von Mises stress util
    prob.model.add_constraint("constr_lss_vonmises", upper=1.0)         #DONE: add next
    prob.model.add_constraint("constr_bedplate_vonmises", upper=1.0)    #DONE: add if needed
    prob.model.add_constraint("constr_hss_vonmises", upper=1.0)

    # 2. deflection
    prob.model.add_constraint("constr_shaft_deflection", upper=1.0)         #DONE: add next
    prob.model.add_constraint("constr_shaft_angle", upper=1.0, ref=1e-3)    #DONE: add next
    # --- MBs (main bearing: max perm is angle, + fls)
    if doMBfls:
        prob.model.add_constraint("constr_L10_mb1", lower=1.0)                  #DONE: add next
        prob.model.add_constraint("constr_L10_mb2", lower=1.0)                  #DONE: add next
    # ----- bearing angular deflections (from Bedplate_*)
    prob.model.add_constraint("constr_mb1_defl", upper=1.0)                 #DONE: add next
    prob.model.add_constraint("constr_mb2_defl", upper=1.0)                 #DONE: add next
    # --- bedplate 
    # prob.model.add_constraint("constr_stator_deflection", upper=1.0)      # TODO: add later if needed (gen stator / max bedplate end defl)
    prob.model.add_constraint("constr_stator_angle", upper=1.0)

    # 3. length: target overhang, hub height and LSS wrt. MBs
    prob.model.add_constraint("constr_length", lower=0.0)               #DONE: add later
    prob.model.add_constraint("constr_height", lower=0.0, ref=1e1)      #DONE: add later
    prob.model.add_constraint("constr_Lh1_MB1fw", lower=0.0)#, ref=1e1)   #DONE: add later
    prob.model.add_constraint("constr_L12_MBsFW", lower=0.0)#, ref=1e0)   #DONE: add later
    # prob.model.add_constraint("constr_del_MB2fw", lower=0.0)#, ref=1e0)            #DONE: add later

    # 4. tower (TODO: use vals from 01_'s analy opts)
    prob.model.add_constraint("post.constr_stress", upper=1.0)
    prob.model.add_constraint("post.constr_global_buckling", upper=1.0)
    prob.model.add_constraint("post.constr_shell_buckling", upper=1.0)
    prob.model.add_constraint("constr_d_to_t", lower=80.0, upper=500.0)
    prob.model.add_constraint("constr_taper", lower=0.2)
    prob.model.add_constraint("slope", upper=1.0)
    prob.model.add_constraint("tower.f1", lower=0.13, upper=0.40)

# %%
# Setup the problem
prob.setup()

#%%
# Defining input values - TODO: read from saved data from 03_DT and 01_tower_semisub
