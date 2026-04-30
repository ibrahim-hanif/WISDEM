# %% [markdown]
# # _Drivetrain optimization_ (`WISDEM`)
# ### main script for:
# 1. single component optimization (as suggested by `WEIS` ppt)
# 2. single optimization run
# 3. save results in csv file for further studies (eg. parallel)
# 4. NOT for (parallel) parametric runs: that is in another script!
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
# - 1. estimate (accurate) lengths of GB and gen (detailed from partners)
# - 2. input converter dims, so modify `Electronics` in `drive_components` 
# - 3. add total `nacelle_cm` as obj? (MOO)
# - 4. change `constr_length`? wrt. 0.0: upper, equal, or lower?
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
import pandas as pd

# %%
from wisdem.drivetrainse.drivetrain import DriveMaterials, DrivetrainSE_M4W

from wisdem.drivetrainse.hub import Hub_System
from wisdem.drivetrainse.gearbox import Gearbox

import wisdem.drivetrainse.layout as lay

import wisdem.drivetrainse.drive_components as dc

import wisdem.drivetrainse.drive_structure as ds

from wisdem.commonse.utilities import get_recorder_results, mainshaft_loads_from_mat_to_dict, load_all_mat_to_dict
from wisdem.commonse.fileIO import save_data, load_data, get_variable_list, var_df2dict
# import the utilities_drivetrain module as utilsDT
import utilities_drivetrain as utilsDT

# %% [markdown]
# ### Define flags
suffix = "_m4w"

# pre-processing; Loading `openFAST` hub loads from a saved file
part_loads = True 
load_fls_loads = False
# False: full loads (72e4,10) (200 Hz sampled, 60mins)
# True: part loads (72e3,11) (20 Hz sampled, 60mins)
dir_loads = "M:\\Vasudev_Gupta\\outputs_mainshaft_loads"

# Optimization flags
flag_opt_GBO = False     # GBO: gradient based optimizer
flag_DOE = False        # DOE: design of experiments
flag_opt_GFO = False    # GFO: gradient free optimizer
flag_debug_print = True
flag_parallel = False

make_xdsm = False       # html-show or detailed pdf
record_cases = False    #TODO: add in final setup (full problem)

# post-processing results
plot_cases = False      #NOTE: saved, not changing now (commented)
flag_scaling_show_browser = False
flag_save_new_data = False
load_from_saved_data = True

# Parametric study
flag_study_parametric = False
# if True: init drive prob with 1 iter and reset to require (80) iters
maxIter_param = 1
maxIter_req = 5 * 16 # 80
if flag_study_parametric: maxIter = maxIter_param
else: maxIter = maxIter_req

param_for_study = "LDD"     # "MB" (types) / "LDD" (MS' L_*)
meth_Peq = "DEL".lower()    # "LRD" or "DEL"

# %% [markdown]
# ### Defining results directory and files
# - results main dir
results_dir = "03_results"
script_dir = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(script_dir, results_dir)
os.makedirs(results_path, exist_ok=True)

# - used within optimization
loc_record_doe = os.path.join(results_path, "DOE_recorded.sql")
loc_n2 = os.path.join(results_path, "n2.html")
# -- XDSM?
if make_xdsm:
    loc_xdsm = os.path.join(results_path, 'xdsm_03')
# -- Record results?
if record_cases:
    print(" ---- Recording cases using `SqliteRecorder` ---- ")
    loc_cases = os.path.join(results_path,
        "cases_recorded_"+meth_Peq+".sql")
    if os.path.exists( loc_cases ):
        os.remove( loc_cases )

# - post-processing
loc_scaling_report = os.path.join(results_path, 'scaling_report.html')
loc_save_data = os.path.join(results_path, "03"+suffix) # "03"+suffix

# - load from saved data
# 02_ data
# if load_from_saved_data: loc_saved_02_data = os.path.join(results_path, "02"+suffix) # 02newULS
# 03_ data
if os.path.exists(loc_save_data+".csv"): load_from_saved_data = True # TODO

# - DOE
loc_DOEcsv_GBgen = os.path.join(script_dir, "04_results", "DOE_GBgen_updated.csv")
flag_load_from_DOEcsv = False

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
opts["WISDEM"]["DriveSE"]["own_hub_loads"] = True
# opts["WISDEM"]["DriveSE"]["nBins"] = 100    #used by (new) Analytical_FLS_Bearing_Life; =Number of bins for histogram MB FLS
# used as: gamma = gamma_f * gamma_m * gamma_n (within TODO)

opts["WISDEM"]["RotorSE"] = {}
opts["WISDEM"]["RotorSE"]["n_pc"] = 2 #cf. RPM_Input in drive_components.py
            # `n_pc`: Number of wind speeds to compute the power curve
opts["materials"] = {}
opts["materials"]["n_mat"] = 4

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

elif flag_opt_GFO:
    print("=== running GFO ===")
    # GFO: gradient free optimizer
    prob.driver = om.SimpleGADriver()
    # prob.driver = om.DifferentialEvolutionDriver()
    if flag_debug_print:
        prob.driver.options["debug_print"] = [
            "desvars", "objs", "nl_cons", "ln_cons"]
    # OSError: 'lss' <class Hub_Rotor_LSS_Frame>: Error calling compute(), exception: access violation reading 0x000001CB530B5FB0

elif flag_DOE: # NOTE: DOEDriver doesn't optimize (so `run_model`) and enforces constraints (unlike `run_driver`)
        #TODO: running 194 mins! on my PC for 10 levels x 4 DVs (khayr insha'Allah)
        # TAKES HOUR(S), with just 10 levels !!!!!!!!!!!! why?
    print("=== running DOE ===")
    prob.driver = om.DOEDriver(om.FullFactorialGenerator(levels=10))
    if record_cases:
        recorder = om.SqliteRecorder( loc_record_doe )
        prob.driver.add_recorder( recorder )

else:
    print("=== running analysis only (`run_model()`) ===")

#%%
# setup optimization: objs, desvars, cons
# - TODO: scaling (is better).
if flag_opt_GBO or flag_opt_GFO or flag_DOE:
    # === Add objective ===
    prob.model.add_objective("nacelle_mass", ref=1e6) #DONE: 'nacelle_mass' minimization
    # prob.model.add_objective("nacelle_moo", ref=1e1) #TODO
    
    # === Add design variables === 
    # 1. LSS
    prob.model.add_design_var("L_h1", lower=0.1, upper=5.0, ref=5.0, ref0=0.1)
    prob.model.add_design_var("L_12", lower=0.1, upper=8.0, ref=8.0, ref0=0.1)
    # prob.model.add_design_var("delta", lower=0.1, upper=5.0, ref=5.0, ref0=0.1)
    prob.model.add_design_var("lss_diameter", lower=1.0, upper=5.0, ref=5.0, ref0=1.0)
    prob.model.add_design_var("lss_wall_thickness", lower=4e-3, upper=1.0, ref=1.0, ref0=4e-3) #DONE: scaled so driver sees lb=0, ub=1 (why? 0.05 causes probs)

    # 2. HSS (TODO: add later if needed)
    prob.model.add_design_var("L_hss", lower=0.1, upper=5.0, ref=5.0, ref0=0.1)
    prob.model.add_design_var("hss_diameter", lower=0.5, upper=5.0, ref=5.0, ref0=0.5)
    prob.model.add_design_var("hss_wall_thickness", lower=4e-3, upper=0.2, ref=0.2, ref0=4e-3)

    # 3. Bedplate (TODO: add later if needed)
    prob.model.add_design_var("bedplate_web_thickness", lower=5e-3, upper=1.0, ref=1.0, ref0=5e-3)
    prob.model.add_design_var("bedplate_flange_thickness", lower=5e-3, upper=1.0, ref=1.0, ref0=5e-3)
    prob.model.add_design_var("bedplate_flange_width", lower=0.01, upper=3.0, ref=3.0, ref0=0.01)

    # 4. hub
    # prob.model.add_design_var("hub_diameter", lower=2.0, upper=5.0)

    # === Add constraints ===    
    if flag_DOE: pass # DOE: no constraints

    # 1. von Mises stress util
    prob.model.add_constraint("constr_lss_vonmises", upper=1.0)         #DONE: add next
    prob.model.add_constraint("constr_bedplate_vonmises", upper=1.0)    #DONE: add if needed
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
    # prob.model.add_constraint("constr_stator_deflection", upper=1.0)      #TODO: add next -> results saved in `03newULS_wConstrStatorDefl`: nacelle_mass=684 t.
    prob.model.add_constraint("constr_stator_angle", upper=1.0)

    # 3. length: target overhang, hub height and LSS wrt. MBs
    prob.model.add_constraint("constr_length", lower=0.0)               #DONE: add later
    prob.model.add_constraint("constr_height", lower=0.0, ref=1e1)      #DONE: add later
    prob.model.add_constraint("constr_Lh1_MB1fw", lower=0.0)#, ref=1e1)   #DONE: add later
    prob.model.add_constraint("constr_L12_MBsFW", lower=0.0)#, ref=1e0)   #DONE: add later
    # prob.model.add_constraint("constr_del_MB2fw", lower=0.0)#, ref=1e0)            #DONE: add later

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
#
# #### `NOTE`: if loading from saved data (`.csv`), variables below will be overwritten
# check the flag `load_from_saved_data`
#%%
# 1. High-level Inputs
# - TODO: check windIO (02_ref WTs) data and change below
prob.set_val("machine_rating", 15.0, units="MW")
prob["rotor_diameter"] = 240.0 # TODO: ref.1 = 240, geo_schema = 241.35064632
prob["rated_torque"] = 21.03*1e6 # [Nm] ref.2, tab.5-4
prob["minimum_rpm"] = 5.0 # needed by RPM_Input
rated_rpm = prob["rated_rpm"] = 7.56
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
# TODO: change to made4wind specs
# (old) run made4wind_geared.py with flag_opt_GBO = false and copy the following values from drivetrain_example.csv
# (new) updated using runWISDEM with orig def blades, hub in geo yaml

if True: #NOTE: True with `Hub_*`
    blade_mass = 65250 # from ref.2, tab. ES-2 (= made4wind specs also)
    n_blades = 3 
    # ---- updated using runWISDEM with orig def blades, hub
    prob["blades_mass"] = 203480.8003090195 # n_blades * blade_mass
    prob["blades_cm"] = 2.450999236350028 # 2.46175
    prob["blades_I"] = [342920565.8181109, 171460282.90905544, 171460282.90905544, 0.0, 0.0, 0.0] # np.r_[3.48453857e+08, 1.74226928e+08, 1.74226928e+08, np.zeros(3)]

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
        prob["hub_system_mass"] = 73097.29755948295 # 190e3 # from ref.2, tab. 5-1
        prob["hub_system_cm"] = 3.3540366555461496 # 3.35947759
        prob["hub_system_I"] = np.array([[1033618.0649506741, 648827.3159275538, 648827.3159275538],[0., 0., 0.]])

# TODO: cm & I (hub_system_ & blades_) will change with DVs (L in lss)

# %% [markdown]
# 3. Drivetrain configuration and sizing inputs
prob['moo_weight'] = 0.5

myones = np.ones(2)
# - init condn for some design vars

# Main Bearing inputs
prob["bear1.bearing_type"] = "CRB" # 1. floating MB
prob["bear2.bearing_type"] = "TRB2" # 2. fixed MB
prob["bear1.mb_e"] = 0.4 # from 0.3-0.4 
prob["bear2.mb_e"] = 0.4
# prob["bear2.mb_k"] = 0.0 #3e10
if doMBfls:
    prob["mb_fls.e_mb"] = prob["bear2.mb_e"]

# Layout / lss inputs
prob["L_h1"] = 0.3 #(def: 2.0), 4.25
prob["L_12"] = 1.24 #(def:1.2), 7.1
prob["delta"] = 0.5
prob["lss_diameter"] = np.array([3.45, 3.21]) #(def:1.0), 4.0
prob["lss_wall_thickness"] = np.array([0.19, 0.01]) #(def:0.1), 0.3

# Gearbox inputs
prob["gear_ratio"] = (375 / rated_rpm)
prob["gearbox_mass_user"] = 138.7286451*1e3 # incl housing (from DOE_GBgen_updated.csv)
widths_flanks = np.array([400,260,240])/1e3 # 0.9 sum of PLC lengths/flank widths (tab.6, D5.1 R2)
# prob["gearbox_length_user"] = (widths_flanks[0] + 2*np.sum(widths_flanks)) * 1.1 # 2.42 (with 10% margin)
prob["gearbox_length_user"] = 2.512381653*1.1 # (from DOE_GBgen_updated.csv, +10% margin)
# prob["gearbox_radius_user"] = (5000/2)/1e3 # outer diameter is roughly (m_n*z_r=25*194=4850 mm) plus some margin for housing
prob["gearbox_radius_user"] = 2.10184*1.1 #(from DOE_GBgen_updated.csv, +10% margin)

# HSS (DONE: consider as DV if needed)
prob["L_hss"] = 1.0
prob["hss_diameter"] = np.array([0.5, 0.5])
prob["hss_wall_thickness"] = np.array([0.1, 0.1])

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
prob["generator_radius_user"] = 2.8 / 2 # = stator outer diameter

# Generator Length Est. (Total cylindrical)
# prob["L_generator"] = 2.15
# ----- 1. formula
def est_generator_length( D_rotor_outer, pole_pairs, len_active,
                        margin_str=0.1 ):
    # INPUTS:
    # - all: in meters [m]
    # - margin_str: structural margins (def: 10%)
    D_stator_inner = D_rotor_outer/(1-0.002)
    tau_p = (np.pi*D_stator_inner)/(2*pole_pairs)
    len_end_axial = 0.5*tau_p # 0.3 - 0.5
    L_generator = (len_active + 2*len_end_axial) * (1+margin_str)
    print( f'    Est. generator length: {L_generator} m' )
    return L_generator
# prob["L_generator"] = est_generator_length(2.375,24,0.866, 0.1) # 1.124 m
# ----- 2. formula
def Lgen( L_active, L_endWind, margin_str=0.1 ):
    return (L_active+(2*L_endWind)) * (1+margin_str)
Lactive = [1085,866,650,545]
LendWind = 271.88
Lgen_list = Lgen( np.array(Lactive), LendWind )
prob["L_generator"] = Lgen_list[1]/1e3 # 1.550 m

# # -- make an equivalent cylinder from the cuboid with the SAME (mass) MoI
# prob["R_generator"] = np.sqrt( (H_generator**2 + W_generator**2)/6 ) # 1.0328
gen_eff = 0.9805
prob["generator_efficiency_user"] = np.array([ [0.0,1.0],[gen_eff,gen_eff] ])

# === Electronics input (ING: converter, transformer)
# converter mass = 3 Tn per 8MW conversion line (ING Bidane's email)
prob["converter_mass_user"] = (3*1e3*15)/8 # 5,625 [kg]
# overall dims (est. very preliminary): TODO
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
prob["bedplate_flange_width"] = 0.5 #1.724
prob["bedplate_flange_thickness"] = 0.02 #0.028
prob["bedplate_web_thickness"] = 0.02 #0.029

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

#%% overwrite variables from saved data
if load_from_saved_data:
    prob = load_data( loc_save_data+".csv", prob )

#%% load DOE case ( GR = 49 ) ?
if flag_load_from_DOEcsv:
    import pandas as pd
    cases = pd.read_csv( loc_DOEcsv_GBgen )
    this_case = cases.loc[1]
    prob = utilsDT.read_df_to_prob( this_case, prob )

#%%[markdown]
# ### Final check before running
print("\n=== Final input check ===\n")
prob.model.list_inputs();
om.n2(prob, outfile=loc_n2, show_browser=True);

# %% [markdown]
# ### Run: Optimization / DOE / Analysis
# `_driver` (optimization) / `_model` (analysis)
#%%
if (flag_opt_GBO or flag_DOE): # and not flag_study_parametric:
    # ---- time it ;)
    t0 = time.time()
    # Run GBO or DOE
    prob.model.approx_totals() # TODO.
    prob.run_driver()
    # ---- time it ;)
    t1 = time.time()
    status_optim = prob.driver.get_exit_status()
    print(" - ",status_optim,": WISDEM run completed in,", t1-t0, "seconds")

elif flag_opt_GFO:
    # Run the GFO
    prob.run_driver()

else:
    # Run the analysis (also when `flag_study_param` True)
    prob.run_model()

# %%[markdown]
# # _____ Post-processing _____

# %%
# Print the results
print("F_aero_hub:")
print(" ", prob["F_aero_hub"] )
print("M_aero_hub:")
print(" ", prob["M_aero_hub"], "\n" )

print("LSS desvars:")
print(" ", prob["L_h1"], prob["L_12"], prob["lss_diameter"], prob["lss_wall_thickness"] )
# TODO: for flange mass, dohub (cf. var `flange_t2shell_t`)
print("HSS desvars:")
print(" ", prob["L_hss"], prob["hss_diameter"], prob["hss_wall_thickness"] )
print("Bedplate desvars (w_f, t_f, t_w):")
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
print(f"nacelle cm: {prob["nacelle_cm"]}")

print("\n--- RNA properties ---")
print(f"RNA mass: {prob["rna_mass"]}") # drivese.rna_mass
print(f"RNA cm: {prob["rna_cm"]}") # drivese.rna_cm
print(f"RNA MoI: {prob["rna_I_TT"]}") # drivese.rna_I_TT
#
print("\nTower-top / drivetrain bedplate base loads:")
print(" - base_F: ", prob['base_F']) # drivese.base_F
print(" - base_M: ", prob['base_M']) # drivese.base_M
# -----------------------------------------------------------------------

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
from my_util_tools import util_funcs
loc_clr_scheme_m4w = util_funcs.loc_clr_scheme_m4w
clrs_m4w = util_funcs.read_color_scheme(loc_clr_scheme_m4w)
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
    len_compns = len(components)

    # Masses in tonnes [t]
    mass_IEA = {
        "Main shaft":       15.734,
        "Turret nose":      11.394,
        "Main bearings":    7.894, # 2.230 + 5.664
        "Gearbox":          0.0,
        "High-speed shaft": 0.0,
        "Brake":            25.6560,        # wisdem empirical
        "Generator":        371.592,
        "Converter":        30.0, # (wisdem empirical=11.98385 ; M4W data_collect indar=30.0)
        "Transformer":      25.0, # (wisdem empirical=30.6350 ; M4W data_collect indar=25.0)
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
    # Colors:
    # ---- tab10
    tab10 = plt.cm.tab10.colors
    colors = list(tab10) + list(tab10[:2])  # extend to 12 components
    # ----- Made4Wind
    colors = []
    for key,val in clrs_m4w.items():
        colors.append(val)
    colors = np.flip(colors)
    if len_compns > len(colors):
        # mul = np.ceil( len_compns/len(colors), 0)
        colors *= 2

    # --------------------------------------------------
    # Figure
    # --------------------------------------------------
    # --- Figure setup ---
    fig, ax = plt.subplots(figsize=(14, 14))

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
        loc="center",
        fontsize=fontsize, frameon=True
    )
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    # -------------------------
    # save
    # -------------------------
    plot_path = os.path.join(results_path,
            "compare_mass"+suffix+".pdf")
    # plt.savefig(plot_path) # NOTE: saved, so don't change now 

    plt.show()

#%%[markdown]
# ===============================================================
# ### Convergence/parametric study setup
# 1. vary chosen GRs (and rspt. GB and gen weights)
#
# -- and save results for nacelle mass optim
#
# -- so varied= `gear_ratio`, `gearbox_mass_user`, `generator_mass_user`

#%%
if flag_study_parametric and flag_opt_GBO:
    print("===== parametric study =====")
    # ==== setup again: prob and driver options
    # set max iterations to required
    prob.driver.options['maxiter'] = maxIter_req
    prob.driver.options['debug_print'] = ['objs'] # TODO ?
    # ==== Initialize: Combinations to study ====
    import pandas as pd
    cases = pd.read_csv( results_path+'\\DOE_GBgen_cleaned.csv' )
    len_steps = cases.shape[0]

    outs_recorded = utilsDT.init_case_dict_from_prob( prob, len_steps )

    # param study loop: TODO parallelization ;)
    for i in range(len_steps):
        prob = utilsDT.read_df_to_prob( cases, i, prob )
        print(f"=== gear ratio: {prob['gear_ratio'][0]} ===")
        print("-------------------- v ------------------")

        # ===== RUN driver (GBO) =====
        # ---- time it ;)
        t0 = time.time()
        # Run GBO or DOE
        prob.model.approx_totals()
        prob.run_driver()
        # ---- time it ;)
        t1 = time.time()
        tcomp = t1-t0
        status_driver_exit = prob.driver.get_exit_status()
        print(" - ",status_driver_exit,": WISDEM run completed in,",
              tcomp, "seconds")
        
        # ===== post-processing =====
        # save outputs        
        outs_recorded = utilsDT.fill_case_dict_from_prob( i, prob, outs_recorded, tcomp )

print(outs_recorded);

# %%
"""
RESULTS for [300,375,500,600]:
(with empirical GB dims)
outs_recorded = {
'L_h1': array([[0.31658344],
       [0.31749984],
       [0.37970139],
       [0.27922879]]),
'L_12': array([[5.81192846],
       [5.91330855],
       [5.7834232 ],
       [5.99463453]]),
'lss_diameter': array([[3.17722253, 1.64129992],
       [3.13190868, 1.71124384],
       [3.20065704, 1.65092328],
       [3.10959507, 1.62197863]]),
'lss_wall_thickness': array([[0.004     , 0.14120271],
       [0.00855137, 0.11865994],
       [0.00404864, 0.12950184],
       [0.00400519, 0.13767114]]),
'L_hss': array([[0.12811803],
       [0.12232293],
       [0.11261694],
       [0.10442458]]),
'hss_diameter': array([[0.5       , 0.6599612 ],
       [0.85694189, 0.55393866],
       [0.82360998, 0.53900728],
       [0.79480517, 0.52379174]]),
'hss_wall_thickness': array([[0.05423808, 0.004     ],
       [0.0323228 , 0.004     ],
       [0.02387966, 0.004     ],
       [0.00429321, 0.004     ]]),
'bedplate_web_thickness': array([[0.02677534],
       [0.01907214],
       [0.02493869],
       [0.02515716]]),
'bedplate_flange_thickness': array([[0.0271102 ],
       [0.01570331],
       [0.02218383],
       [0.02361556]]),
'bedplate_flange_width': array([[1.8241057 ],
       [1.85661121],
       [1.98619811],
       [1.95898837]]),
'nacelle_mass': array([[389047.75040603],
       [362942.68967398],
       [368715.24413073],
       [365450.30131209]])}
"""
#%%
tst = cases.copy()
for i in range(len_steps):
    tst = utilsDT.write_dict_to_df( tst, i, outs_recorded )

tst
# tst.to_csv( results_path+'\\DOE_GBgen_results.csv', index=False )
# %%
