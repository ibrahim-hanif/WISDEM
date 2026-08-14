# %% [markdown]
# # _Reliability-based optimization of the MBSA_ (`WISDEM`+`openturns`)
# 
# ### current version:
# copy of `02_fls_lss_layout.py`
#
# ### TODO:
# 1. 
# 

#%%
import sys
import scipy.io as sio

venv_needed = 'rbdo_wisdem_env'

use_base = input("Are you using the "+venv_needed+" venv with 'openturns' installed? [y/N]: ")
if use_base.strip().lower() not in ("y", "yes"):
    print("Please activate the "+venv_needed+" environment with sklearn installed and rerun.")
    sys.exit('exit')

try:
    import openturns as ot
    import openturns.viewer as otv
except ImportError:
    print("openturns is not available. Please activate the "+venv_needed+" environment with openturns installed.")
    sys.exit('exit')

#%%
# imports
import os
import numpy as np
import openmdao.api as om
import matplotlib.pyplot as plt
import time
# import scipy.io as sio # --- not used in here, but within imports
# import pickle
import pandas as pd

# %%
from wisdem.drivetrainse.drivetrain import MBSA
from wisdem.commonse.utilities import get_recorder_results, mainshaft_loads_from_mat_to_dict, load_all_mat_to_dict, pdf_norm_int_using_cdf
from wisdem.commonse.fileIO import save_data, load_data, var_df2dict
from wisdem.commonse.cross_sections import Tube
import Drive4Wind.utilities.utilities_drivetrain as utilsDT

# %%
# ### Define flags
suffix = "_sima" # _noMBfls

# Optimization flags
flag_opt_GBO = False     # GBO: gradient based optimizer

# post-processing results
make_xdsm, xdsm_type = False, "html"       # html-show or detailed pdf
record_cases = False    #TODO: add in final setup (full problem)
plot_cases = True      #NOTE: saved, not changing now (commented)
flag_scaling_show_browser = False
flag_save_new_data = False
flag_load_from_data = True
flag_load_from_02data = True

# Loading `openFAST` hub loads from a saved file
# TODO: dont even need to do this now, coz `Load_Own_Hub_Loads` component does it internally and outputs the needed loads for the DT component. So, can just set `own_hub_loads=True` in `modelling_options` and not worry about loading the loads here in the script. JazakumAllahu khayr.
part_loads = True 
load_fls_loads = False
# False: full loads (72e4,10) (200 Hz sampled, 60mins)
# True: part loads (72e3,11) (20 Hz sampled, 60mins)
dir_loads = "M:\\Vasudev_Gupta\\outputs_mainshaft_loads"
loc_all_loads_mat_file = os.path.join(dir_loads, "hub_loads_M4W.mat")
if "sima" in suffix:
    loc_all_loads_mat_file = "C://SIMA_M4W_loads//all_main_shaft_loads.mat" # TODO: sima loads
loc_FLS_loads_mat_file = os.path.join(dir_loads, "mainshaft_loads_FLS_full.mat")
loc_ULS_loads_mat_file = os.path.join(dir_loads, "mainshaft_loads_ULS.mat")

#%%
# ### Defining results directory and files
results_dir = "02b_results"
script_dir = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(script_dir, results_dir)
os.makedirs(results_path, exist_ok=True)

results02_dir = "02_results"
results02_path = os.path.join(script_dir, results02_dir)

# base case turbine csv
basecaseCSVpath = os.path.join(
    os.path.dirname(os.path.dirname(script_dir)),
    "02_reference_turbines","M4W_production_runs","outputs",
    "basecase_NOoptim.csv")
basecaseDF = pd.read_csv(basecaseCSVpath)
basecaseDict = var_df2dict(basecaseDF)

# outputs
loc_doe = os.path.join(results_path, "DOE_recorded.sql")
loc_n2 = os.path.join(results_path, "n2.html")
loc_scaling_report = os.path.join(results_path, 'scaling_report.html')
loc_save_data = os.path.join(results_path, "02"+suffix)

if flag_load_from_data:
    if flag_load_from_02data:
        loc_load_saved_data = os.path.join(results02_path, "02"+suffix) # 02newULS
    else:
        loc_load_saved_data = os.path.join(results_path, "02"+suffix) # 02b

loc_xdsm = os.path.join(results_path, 'xdsm_02')

# Record results?
if record_cases:
    print(" ---- Recording cases using `SqliteRecorder` ---- ")
    loc_cases = os.path.join(results_path,
        "cases_recorded_"+suffix+".sql")
    if os.path.exists( loc_cases ):
        os.remove( loc_cases )

#%% Loading `openFAST` hub loads from a saved file
if part_loads: # define paths
    S_all, keys_all = load_all_mat_to_dict(loc_all_loads_mat_file)

else: # define paths
    if load_fls_loads: # load from paths
        Snew, keys_new = mainshaft_loads_from_mat_to_dict(
            loc_FLS_loads_mat_file, loc_ULS_loads_mat_file)
# %% [markdown]
# ### Defining options (`modelling_options`), flags
#%%
# define `modelling_options`
# TODO: probabs check with wind site
from utilities_drivetrain import define_modeling_options_dict_for_drivetrainSE as defModelOpts
opts = defModelOpts(loc_all_loads_mat_file)

doMBfls = opts["flags"]["mb_fls"]
dohub = opts["flags"]["hub"]
dogen = opts["flags"]["generator"]
# %% [markdown]
# ### Defining the model `problem class`:

# %%
# ### Setup the problem
# Define the problem
prob = om.Problem(reports=False)

# Define the model
prob.model = MBSA(modeling_options=opts) # an instance of the LSS_layout problem defined above

# %%
# ### Optimization / DOE setup
# If performing optimization, set up the optimizer and settings
# - NOTE: all 3 need gradient information
# - OSError: 'lss' <class Hub_Rotor_LSS_Frame>: Error calling compute(), exception: access violation reading 0x000001CB530B5FB0

if flag_opt_GBO:
    print("=== running GBO ===")
    # Choose the (GBO) optimizer to use
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"
    prob.driver.options["tol"] = 1e-6 # comment to default (1e-6?)
    prob.driver.options["maxiter"] = 5 * 4
    prob.driver.options["disp"] = True
    prob.driver.options["debug_print"] = ["desvars", "objs", "nl_cons", "ln_cons"]
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
    # Add objective
    prob.model.add_objective("msa_mass", ref=1e6)               #DONE: 'msa_mass' minimization
    # - NOTE: effectively 'lss_mass' minimization
    
    # Add design variables
    prob.model.add_design_var("L_h1", lower=0.1, upper=5.0, ref=5.0, ref0=0.1)
    prob.model.add_design_var("L_12", lower=0.1, upper=8.0, ref=8.0, ref0=0.1)
    # prob.model.add_design_var("delta", lower=0.1, upper=5.0, ref=5.0, ref0=0.1)
    prob.model.add_design_var("lss_diameter", lower=1.0, upper=5.0, ref=5.0, ref0=1.0)
    prob.model.add_design_var("lss_wall_thickness", lower=4e-3, upper=0.5, ref=1.0, ref0=4e-3) #DONE: scaled so driver sees lb=0, ub=1 (why? 0.05 causes probs)

    # Add constraints
    
    # 1. von Mises stress util
    prob.model.add_constraint("constr_lss_vonmises", upper=1.0)         #DONE: add next
    
    # 2. deflection (main bearing: max perm is angle, + fls) #NOTE: scaling is better
    prob.model.add_constraint("constr_shaft_deflection", upper=1.0)#, ref=1e-2)     #DONE: add next
    prob.model.add_constraint("constr_shaft_angle", upper=1.0, ref=1e-3)          #DONE: add next
    if doMBfls:
        prob.model.add_constraint("constr_L10_mb1", lower=1.0)
        prob.model.add_constraint("constr_L10_mb2", lower=1.0)

    # 3. target overhang and hub height
    # prob.model.add_constraint("constr_length", lower=0.0)               #DONE: add later
    # prob.model.add_constraint("constr_height", lower=0.0, ref=1e1)               #DONE: add later
    prob.model.add_constraint("constr_Lh1_MB1fw", lower=0.0)#, ref=1e1)            #DONE: add later
    prob.model.add_constraint("constr_L12_MBsFW", lower=0.0)#, ref=1e0)            #DONE: add later
    # prob.model.add_constraint("constr_del_MB2fw", lower=0.0)#, ref=1e0)            #TODO: add later
    # prob.model.add_constraint("L_lss", upper=7.0)#, ref=1e0)            #TODO: add later

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
        out_format= xdsm_type, # html or pdf
        show_browser=True,
        quiet=False,
        output_side='left',
        include_indepvarcomps=False,
        class_names=False
    )
# -----

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
# after calling `prob.setup()` (on the openMDAO `prob` defined) and before calling `prob.run_driver()`
#%%
if not flag_load_from_data:
    print(" user defined prob vars")
    # ==== 1. High-level Inputs ====
    machine_rating = float(basecaseDict["drivese.machine_rating"])
    prob.set_val("machine_rating",machine_rating,"kW")
    D_rotor = prob["rotor_diameter"] = float(basecaseDict["drivese.rotor_diameter"])
    prob["rated_torque"] = float(basecaseDict["drivese.rated_torque"]) # 21.3 * 1e6 # Nm
    # prob["minimum_rpm"] = 5
    rated_rpm = prob["rated_rpm"] = float(basecaseDict["drivese.rated_rpm"]) #7.56
    prob["lifetime"] = float(basecaseDict["drivese.lifetime"]) #design life in years ('lifetime' from WEIS, WindIO)

    prob["upwind"] = True
    prob["D_top"] = float(basecaseDict["drivese.D_top"]) #tower top diameter
    prob["hub_diameter"] = float(basecaseDict["drivese.hub_diameter"])
    prob["overhang"] = float(basecaseDict["drivese.overhang"]) #ref.2
    prob["tilt"] = float(basecaseDict["drivese.tilt"]) #[deg] ref.3

    # ==== Loading `openFAST` hub loads from a saved file ====
    # Loads assignment (ULS, FLS)
    #
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

    # ==== 2. Blade properties and hub design options ====
    # - cf. `opts["flags"]["hub"]`

    # Hub_Rotor_LSS_Frame inputs
    # TODO: from made4wind_geared (IEA-15MW = ref), change to made4wind specs
    if True: #NOTE: True with `Hub_*`
        blade_mass = float(basecaseDict["drivese.blade_mass"]) #65250 # from ref.2, tab. ES-2 (= made4wind specs also)
        n_blades = 3
        prob["blades_mass"] = float(basecaseDict["drivese.blades_mass"]) #n_blades * blade_mass
        prob["blades_cm"] = float(basecaseDict["drivese.blades_cm"])
        prob["blades_I"] = eval(basecaseDict["drivese.blades_I"])

        # if run HUB module within DrivetrainSE
        # # - mostly False so not taken from base case
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
            prob["hub_system_mass"] = float(basecaseDict["drivese.hub_system_mass"]) # 190e3; from ref.2, tab. 5-1
            prob["hub_system_cm"] = float(basecaseDict["drivese.hub_system_cm"])
            prob["hub_system_I"] = eval(basecaseDict["drivese.hub_system_I"])

    # TODO: cm & I (hub_system_ & blades_) will change with DVs (L in lss)

    # ==== 3. Drivetrain configuration and sizing inputs ====

    myones = np.ones(2)
    # - init condn for some design vars

    # Main Bearing inputs
    prob["bear1.bearing_type"] = "CRB" # 1. floating MB
    prob["bear2.bearing_type"] = "TRB2" # 2. fixed MB
    prob["bear1.mb_e"] = 0.4 # from 3.5-4.0 (TODO: find ref.)
    prob["bear2.mb_e"] = 0.4
    prob["bear2.mb_k"] = 3.0*1e10 - 6e8
    if doMBfls:
        prob["mb_fls.e_mb"] = prob["bear2.mb_e"]
        prob["mb_fls.k_mb2"] = prob["bear2.mb_k"]

    # Layout / lss inputs
    prob["L_h1"] = 0.1 #(def: 0.5), 4.25; converg: 0.264
    prob["L_12"] = 2.0 #(def: 2.0), 7.1; converg: 6.936
    prob["lss_diameter"] = np.array([3.0, 3.0]) #(def:2.0), 4.0; converg: np.array([2.907, 1.679])
    prob["lss_wall_thickness"] = np.array([0.1, 0.1]) #(def:0.1), 0.3; converg: np.array([0.006, 0.123])

    flange_MS_length = 0.3*(D_rotor/100)**2 - 0.1*(D_rotor/100) + 0.4
    print(f"   - flange length at main-shaft = {flange_MS_length}")

    # Gearbox inputs
    # prob["L_gearbox"] = 1.5 #(v) calc in gearbox.py
    # prob["gear_configuration"] = "eee"
    # prob["planet_numbers"] = np.array([5, 3, 0]) #ref.1
    prob["gear_ratio"] = (375 / rated_rpm)
    prob["gearbox_mass_user"] = 138.73*1e3 # 138.728645e3: incl housing (from DOE_GBgen_updated.csv)
    # prob["gearbox_torque_density"] = 200.0 # (cf. line 210, gearbox.py)

    prob["L_hss"] = 1.5
    prob["hss_diameter"] = 0.5 * myones
    prob["hss_wall_thickness"] = 0.1 * myones

    # Generator inputs (TODO: add compn later)
    # - needed by Bedplate_IBeam_Frame in drive_structure.py, output of HSS_Frame
    # - copied from made4wind_geared.py's output drivetrain_example.csv
    # prob["R_generator"] = 1.7999999999999998
    # prob["L_generator"] = 4.2
    # TODO: opts:
    # --- 1. input from gen design (ingeteam),
    # --- 2. maybe calc in generator.py?,
    # --- 3. 11.98398883842414 (from drivetrain_example.csv),
    # --- 4. 2.0 (drivetrain_geared) or 2.15 (drivetrain_direct)

    # TODO: Ingeteam generator dimensions
    # prob["generator_mass_user"] = 34.8*1e3 # D5.4, tab.9
    # prob["generator_radius_user"] = 2.8 / 2 # = stator outer diameter
    prob["L_generator"] = 1.550

    # === Electronics input (ING: converter, transformer)
    # converter mass = 3 Tn per 8MW conversion line (ING Bidane's email)
    # prob["converter_mass_user"] = (3*1e3*15)/8 # 5,625 [kg]
    # overall dims (est. very preliminary): TODO
    H_converter, W_converter, L_converter = 2.4, 0.8, 4.2 # [m]

    prob["drive_height"] = float(basecaseDict["drivese.drive_height"])

    # bedplate: Hub:_Rotor_LSS_Frame, Bedplate_IBeam_Frame inputs
    # prob["bedplate_flange_width"] = 1.0
    # prob["bedplate_flange_thickness"] = 0.1
    # prob["bedplate_web_thickness"] = 0.1

    # NOTE: True with `Hub_*`
    prob["shaft_deflection_allowable"] = 1e-4 # within Hub_Rotor_LSS_Frame (below): Deflections and rotations at GB attachment
    prob["shaft_angle_allowable"] = 1e-3

    # prob["stator_deflection_allowable"] = 1e-4 # within Bedplate_IBeam_Frame (below)
    # prob["stator_angle_allowable"] = 1e-3

    # ==== 4. Material properties ==== 
    #  (discrete_inputs to `DriveMaterials`)

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
else:
    print(" loading prob vars from saved csv")
    prob = load_data( loc_load_saved_data+".csv", prob )

#%%
# ### Final check before running
print("\n=== Final input check ===\n")
prob.model.list_inputs();
om.n2(prob, outfile=loc_n2, show_browser=True);

#%%
# ### Run: Optimization / DOE / Analysis
# `_driver` (optimization) / `_model` (analysis)

if flag_opt_GBO:
    # Run GBO or DOE
    t0 = time.time()
    # main GBO
    prob.model.approx_totals() # TODO.
    prob.run_driver()
    
    t1 = time.time()
    print(" - WISDEM run completed in,", t1-t0, "seconds")

else:
    # Run the analysis
    t0 = time.time()
    # run
    prob.run_model()
    t1 = time.time()
    print(" - WISDEM run completed in,", t1-t0, "seconds")

# %%[markdown]
# # _____ Post-processing _____
#%%
# Print the results
print("F_aero_hub:")
print(" ", prob["F_aero_hub"] )
print("M_aero_hub:")
print(" ", prob["M_aero_hub"], "\n" )

print("LSS desvars:")
print(" ", prob["L_h1"], prob["L_12"], prob["lss_diameter"], prob["lss_wall_thickness"], "\n" )
# [3.48132032] [1.] [4. 4.] [0.32635334 0.29289825]

print("--- constr_ max ---")
print("- lss: ",
      np.max(prob["constr_lss_vonmises"])
      )
if doMBfls:
    print("constr_L10_mb(1,2):", prob["constr_L10_mb1"], prob["constr_L10_mb2"], "\n" )
#
print("--- obj: masses ---")
print(f"MSA mass: {prob["msa_mass"]}")

# list_driver_vars = prob.list_driver_vars()
# ==========================================================
#%%
### Recorded cases
if record_cases:
    print("\n=== Recorded cases from the optimization ===\n")
    results_dict = get_recorder_results( loc_cases, None, True )
    print(results_dict);

#%%
# Driver scaling report 
try:
    prob.driver.scaling_report(
        outfile=loc_scaling_report,show_browser=flag_scaling_show_browser
    );
except:
    pass
#%%
if flag_save_new_data: save_data(loc_save_data, prob)
# ===============================================================

#%%[markdown]
# ### Plot recorded results
#%%
# main colors
from Drive4Wind.post_processing import color_schemes
loc_clr_scheme_m4w = color_schemes.loc_clr_scheme_m4w
clrs_m4w = color_schemes.read_color_scheme(loc_clr_scheme_m4w)
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

# %%
if record_cases and plot_cases:
    print(" NOTE: DV converg iter plot for testing now; not being saved")
    # -------------------------
    # Extract and squeeze data
    # -------------------------
    res = results_dict

    msa_mass = res['msa_mass'].squeeze()
    scale_m_msa = 1e5;
    msa_mass = msa_mass / scale_m_msa

    L_12 = res['L_12'].squeeze()
    L_h1 = res['L_h1'].squeeze()

    lss_diam = res['lss_diameter']
    lss_t = res['lss_wall_thickness']

    L10_mb1 = res['constr_L10_mb1'].squeeze()
    L10_mb2 = res['constr_L10_mb2'].squeeze()

    iters = np.arange(1, len(msa_mass) + 1)
    # -------------------------
    # obj func val converg: diff in change per iter; TODO
    del_obj_func = msa_mass
    del_obj_func = del_obj_func[1:] - del_obj_func[:-1]

    # -------------------------
    # Figure and layout
    # -------------------------
    fig = plt.figure(figsize=(24, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.25, wspace=0.5)

    # ========= Row 1 (span both columns): msa_mass =========
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(iters, msa_mass,
            marker='o', linewidth=2, color= clrs_m4w["Dark_Blue"],
            label=r'$m_{msa}$')
    ax1.set_ylabel(r'Mass [$\cdot 10^2$ t]')
    # ax1.set_xlabel('Iteration')
    ax1.set_xticks(iters)
    ax1.grid(True)
    ax1.legend()

    # ========= Row 2, Col 1: L_12 and 10*L_h1 =========
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(iters, 10.0 * L_h1,
            marker='s', color = clrs_m4w['Dark_Teal'],
            label=r'$L_{h1} \times 10$')
    ax2.plot(iters, L_12,
            marker='o', color = clrs_m4w['Aqua'],
            label=r'$L_{12}$')
    ax2.set_ylabel(r'Length [m]')
    # ax2.set_xlabel('Iteration')
    # ax2.set_xticks(iters)
    ax2.grid(True)
    ax2.legend()

    # ========= Row 2, Col 2: diameter and thickness =========
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(iters, lss_diam[:, 0],
            marker='o', color = clrs_m4w['Dark_Teal'],
            label=r'$D_{lss,1}$')
    ax3.plot(iters, lss_diam[:, 1],
            marker='o', color = clrs_m4w['Aqua'],
            label=r'$D_{lss,2}$')
    ax3.plot(iters, 10.0 * lss_t[:, 0],
            marker='s', color = clrs_m4w['Dark_Red'],
            label=r'$t_{lss,1} \times 10$')
    ax3.plot(iters, 10.0 * lss_t[:, 1],
            marker='s', color = clrs_m4w['Red'],
            label=r'$t_{lss,2} \times 10$')
    ax3.set_ylabel(r'Cross-section [m]')
    # ax3.set_xlabel('Iteration')
    # ax3.set_xticks(iters)
    ax3.grid(True)
    # ax3.legend(ncol=2)
    ax3.legend(loc='center left',bbox_to_anchor=(-0.5,0.5))

    # ========= Row 3 (span both columns): L10 constraints =========
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(iters, L10_mb1,
            marker='o', linewidth=2, color = clrs_m4w['Dark_Teal'],
            label=r'$L_{10}^{mb1}$')
    ax4.plot(iters, L10_mb2,
            marker='s', linewidth=2, color = clrs_m4w['Aqua'],
            label=r'$L_{10}^{mb2}$')
    ax4.axhline(1.0, color='k', linestyle='--', linewidth=1)
    ax4.set_ylabel(r'$ g\_L_{10} $ [-]')
    ax4.set_xlabel('Optimizer function evaluations')
    ax4.set_xticks(iters)
    ax4.grid(True)
    ax4.legend(loc='upper right')#,bbox_to_anchor=(1,0.5))

    # -------------------------
    # Final layout
    # -------------------------
    fig.tight_layout()

    # -------------------------
    # save
    # -------------------------
    plot_path = os.path.join(results_path,
            "vars_with_iter_"+suffix+".png")
    # plt.savefig(plot_path) # NOTE: saved, so don't change now 

    plt.show()

# %% [markdown]
# ### Reliability-based design optimization (RBDO)
# -------------------------- using `openturns` --------------------------

# %%
class ReliabiltyComponent( om.ExplicitComponent ):

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        # init
        opts = self.options["modeling_options"]
        n_dlcs = opts["n_dlcs"]

        # add inputs TODO: only the design variables

        # add ouputs TODO: reliability results as constraints
        self.add_output("beta", val=3.0, desc="reliability index, min of all if many")
        self.add_output("Pf", val=1e-6, desc="failure probability, min of all if many")

    def compute(self, inputs, outputs):
        opts = self.options["modeling_options"]

        F_hub = inputs["F_aero_hub"]
        M_hub = inputs["M_aero_hub"]
        E = float(inputs["lss_E"][0])
        mb1_e = inputs["mb1_e"]
        mb2_e = inputs["mb2_e"]

        # --------------
        # MBSA Problem
        # --------------
        prob = om.Problem(reports=False)
        # Define the model
        prob.model = MBSA(modeling_options=opts) # an instance of the LSS_layout problem defined above
        # Setup the problem
        prob.setup()
        # Assign input values
        print("- loading prob vars from saved csv ...") # TODO: rmv or verbose flag
        prob = load_data( loc_load_saved_data+".csv", prob );

        # --------------
        # Random variables
        # --------------
        # define the model and random var dist; TODO add self. + testing
        model = ot.PythonFunction(inputDim=3,outputDim=1,func=g_mbsa)
        distribution = make_distribution_of_mbsa_inputs()

        # Create the event whose probability we want to estimate.
        vect = ot.RandomVector(distribution)
        G = ot.CompositeRandomVector(model, vect)
        event = ot.ThresholdEvent(G, ot.Greater(), 1.0) # TODO for each g
        event.setName("lss_vonmises_deviation")

        # --------------
        # Reliability analysis (FORM / MCS)
        # --------------
        maxCalls = 5.e1
        Pf, beta = run_FORM( distribution, event, maxCallsNum=maxCalls )
        # MCS
        Pf_mcs = run_MCS( event, numSamples=maxCalls )

        # Probability
        outputs["Pf"] = Pf
        # # Hasofer reliability index
        outputs["beta"] = beta

    class MBSAEvaluator:
        def __init__(self, opts, loc_load_saved_data):
            self.prob = om.Problem()
            self.prob.model = MBSA(modeling_options=opts)
            self.prob.setup()
            self.prob = load_data(loc_load_saved_data+".csv", self.prob)

        def evaluate(self, X):
            # TODO: equal to g_mbsa below
            self.prob.set_val(...)
            self.prob.run_model()
            return ...

    def g_mbsa( X ):
        """
        Inputs
        _______
        X : float[ 8,1 ]
            with elements
            1. F_aero_hub[0]
            2. F_aero_hub[1]
            3. F_aero_hub[2]
            4. M_aero_hub[0]
            5. M_aero_hub[1]
            6. M_aero_hub[2]
            7. lss_E
            8. mb2_e
            9. (not used) mb1_e

        Outputs
        _______
        Y : float[ 5,1 ]
            all the (relevant) MBSA constraints
            1. constr_lss_vonmises
            2. constr_shaft_deflection
            3. constr_shaft_angle
            4. constr_L10_mb1
            5. constr_L10_mb2
            6. (not used) constr_Lh1_MB1fw
            7. (not used) constr_L12_MBsFW

        Progress
        _______
        1. DONE: implement a hard-coded version, with fixed inputDims, outDims
        2. TODO: automated Dims
        """
        # Inputs: parse
        F_aero_hub = np.zeros((3,1)) # TODO: testing -----
        F_aero_hub[0] = X[0]
        F_aero_hub[1] = X[1]
        F_aero_hub[2] = X[2]
        # M_aero_hub = np.zeros((3,1))
        # M_aero_hub[0] = X[3]
        # M_aero_hub[1] = X[4]
        # M_aero_hub[2] = X[5] # -----
        # lss_E = X
        # mb2_e = X[1] # TODO testing
        # -- init
        lstIns = ["F_aero_hub","M_aero_hub",
                  "lss_E","bear1.mb_e","bear2.mb_e"]
        lenIns = len(lstIns)
        
        # -- from user
        prob["F_aero_hub"] = F_aero_hub # TODO: testing
        # prob["M_aero_hub"] = M_aero_hub
        # prob["lss_E"] = lss_E # TODO: testing
        # prob["bear2.mb_e"] = mb2_e
        # Run model
        prob.run_model()
        # Parse outputs
        # -- init
        lstOuts = ["constr_lss_vonmises",
            "constr_shaft_deflection",
            "constr_shaft_angle",
            "constr_L10_mb1",
            "constr_L10_mb2"]
        lenOuts = len(lstOuts)
        Y = np.zeros(lenOuts,)
        # -- extract
        for i in range(lenOuts):
            name = lstOuts[i]
            imax = max( np.array(prob[name]) )
            if "_vonmises" in name: imax = imax[0] # arr to float
            iOut = float(imax)
            Y[i] = iOut

        # return
        Ytest = Y[0] # TODO: testing with 1
        return [Ytest]

    def make_distribution_of_mbsa_inputs():
        """
        Inputs
        _______
        same inputs and dims as g_mbsa

        Progress
        _______
        1. DONE: implement a basic working example inshaAllah
        2. TODO: define standard devs of each random var correctly
        """
        # From saved csv
        savedDataDF = pd.read_csv(loc_load_saved_data+".csv")
        savedDataDict = var_df2dict(savedDataDF)
        # Inputs: fill
        F_aero_hub = np.array(eval(savedDataDict["F_aero_hub"])).reshape(3,)
        M_aero_hub = np.array(eval(savedDataDict["M_aero_hub"])).reshape(3,)
        lss_E = float(savedDataDict["lss_E"])
        mb1_e = float(savedDataDict["bear1.mb_e"])
        mb2_e = float(savedDataDict["bear2.mb_e"])

        dims = 7
        # F_aero_hub
        F_aero_cov = M_aero_cov = 0.2 # 10-20 %
        F_aero_sigma = F_aero_hub * F_aero_cov
        # - 1.
        F1 = ot.LogNormal()  # in N
        F1.setParameter(ot.LogNormalMuSigma()(
            [F_aero_hub[0], F_aero_sigma[0], 0.0]
            ))
        # F1.setDescription("Fx")
        F1.setName("F_aero_hub_x")
        # - 2.
        F2 = ot.LogNormal()  # in N
        F2.setParameter(ot.LogNormalMuSigma()(
            [F_aero_hub[1], F_aero_sigma[1], 0.0] # TODO
            ))
        # F2.setDescription("Fy")
        F2.setName("F_aero_hub_y")
        # - 3.
        F3 = ot.LogNormal()  # in N
        F3.setParameter(ot.LogNormalMuSigma()(
            [F_aero_hub[2], F_aero_sigma[2], 0.0] # TODO
            ))
        # F3.setDescription("Fz")
        F3.setName("F_aero_hub_z")

        # M_aero_hub
        M_aero_sigma = M_aero_hub * M_aero_cov
        # - 1.
        M1 = ot.LogNormal()  # in N
        M1.setParameter(ot.LogNormalMuSigma()(
            [M_aero_hub[0], M_aero_sigma[0], 0.0] # TODO
            ))
        # M1.setDescription("Mx")
        M1.setName("M_aero_hub_x")
        # - 2.
        M2 = ot.LogNormal()  # in N
        M2.setParameter(ot.LogNormalMuSigma()(
            [M_aero_hub[1], M_aero_sigma[1], 0.0] # TODO
            ))
        # M2.setDescription("My")
        M2.setName("M_aero_hub_y")
        # - 3.
        M3 = ot.LogNormal()  # in N
        M3.setParameter(ot.LogNormalMuSigma()(
            [M_aero_hub[2], M_aero_sigma[2], 0.0] # TODO
            ))
        # M3.setDescription("Mz")
        M3.setName("M_aero_hub_z")

        # # Young's modulus E (in N/m^2)
        # - 1. real
        E = ot.LogNormal()  # in N
        E_cov = 0.03 # 2-3 %
        E_sigma = E_cov*lss_E
        E.setParameter(ot.LogNormalMuSigma()(
            [lss_E, E_sigma, 0.0] # TODO
            ))
        # - 2. test TODO 
        # E = ot.Beta(0.9, 3.5, lss_E*0.999, lss_E*1.001)
        E.setDescription("E")
        E.setName("Young modulus")

        # MB's e
        # # - 1. 
        # e_mb1 = ot.Normal(mb1_e, 0.1)  # TODO
        # e_mb1.setDescription("e")
        # e_mb1.setName("mb1_e")
        # # - 2. 
        # e_mb2 = ot.Normal(mb2_e, 0.1)  # TODO
        # e_mb2.setDescription("e")
        # e_mb2.setName("mb2_e")

        # correlation matrix TODO
        # - Aerodynamic loads are highly correlated.
        # - Aeroelastic simulations can estimate these.
        # - Ignoring correlation can produce very misleading reliability indices.
        R = ot.CorrelationMatrix(dims)
        R[1,4] = 0.7 # Fy and My correlated, 0.7-0.9
        R[2,5] = 0.7 # Fz and Mz correlated, 0.7-0.9
        copula = ot.NormalCopula(
            ot.NormalCopula.GetCorrelationFromSpearmanCorrelation(R)
        )
        distribution = ot.JointDistribution(
            [F1,F2,F3, M1,M2,M3, E], copula
        )

        # TODO testing: correlation matrix of 2
        dims = 3
        R = ot.CorrelationMatrix(dims)
        copula = ot.NormalCopula(
            ot.NormalCopula.GetCorrelationFromSpearmanCorrelation(R)
        )
        distribution = ot.JointDistribution(
            [F1,F2,F3], copula
        )

        return distribution

    def run_MCS( event, numSamples=1e5 ):
        """
        Run a Monte-Carlo Simulation (MCS) experiment for a given `event`
        """
        ts = time.time()
        # Create a Monte Carlo algorithm.
        experiment = ot.MonteCarloExperiment()
        algo = ot.ProbabilitySimulationAlgorithm(event, experiment) #(v) Pf = P(E) = P(G(X) < 0)
        algo.setMaximumCoefficientOfVariation(0.05)
        algo.setMaximumOuterSampling(int(numSamples))
        algo.setKeepSample(True)
        algo.run()
        # Retrieve results.
        result = algo.getResult()
        probability = result.getProbabilityEstimate()
        # calc beta also TODO
        #
        te = time.time()
        t_total = te-ts 
        print(f"! Results (MCS) in {t_total}s : Pf = {probability}.")
        return probability

    def run_FORM( distribution, event, maxCallsNum=1e4 ):
        """
        Run a First-Order Reliability Method (FORM) analysis
         on a given `distribution` and a given `event`
        """
        ts = time.time()
        # Define a solver, here we use a :class:`~openturns.MultiStart` optimization based on :class:`~openturns.Cobyla`
        startingSample = distribution.getSample(10)
        optimAlgo = ot.MultiStart(ot.Cobyla(), startingSample)
        optimAlgo.setMaximumCallsNumber( int(maxCallsNum) )
        maxError = 1.e-3
        optimAlgo.setMaximumAbsoluteError(maxError)
        optimAlgo.setMaximumRelativeError(maxError)
        optimAlgo.setMaximumResidualError(maxError)
        optimAlgo.setMaximumConstraintError(maxError)
        # Run FORM
        algo = ot.FORM(optimAlgo, event)
        algo.run()
        # Retrieve results.
        result = algo.getResult()
        probability = result.getEventProbability()
        beta = result.getHasoferReliabilityIndex()
        #
        te = time.time()
        t_total = te-ts
        print(f"! Results (FORM) in {t_total}s : Pf = {probability}, beta = {beta}.")
        return probability, beta

class MBSA_RBDO( om.Group ):
    """
    Group containing components for the layout of the LSS components
    """
    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        opts = self.options["modeling_options"]

        opt_drivese = opts["WISDEM"]["DriveSE"]
        # OpenFAST: containing 1. simulation DT and 2. MS loads dir
        opt_openfast = opts["OpenFAST"]
        # DLC: only 1 used '[0]': containing "wind_speed" and "probabilities"
        opt_DLC = opts["DLC_driver"]["DLCs"][0]

        n_dlcs = opts["WISDEM"]["n_dlc"]
        direct = opt_drivese["direct"]
        if direct:
            gearbox_torque_density = 0.0
        else:
            gearbox_torque_density = opt_drivese["gearbox_torque_density"]
            
        dogen = opts["flags"]["generator"]
        n_pc = opts["WISDEM"]["RotorSE"]["n_pc"]
        flag_hub = opts["flags"]["hub"]
        doMBfls = opts["flags"]["mb_fls"]

        # print flag information
        print("=== Problem 'LSS_layout' setting up ===")
        print(f"flag info: doMBfls={doMBfls}, gearbox_torque_density={gearbox_torque_density}, dogen={dogen}, flag_hub={flag_hub}, direct={direct}")

        # Materials prep
        self.add_subsystem(
            "mbsa",
            MBSA(modeling_options=opts),
                promotes=["*"]
            )
        
        self.add_subsystem(
            "reliab",
            ReliabiltyComponent(TODO),
                promotes=["*"]
            )
        
        self.connect("bear1.mb_e", "mb1_e")
        self.connect("bear2.mb_e", "mb2_e")

#%%
X = np.zeros(9,)
for i in range(3):
    X[i] = prob['F_aero_hub'][i][0]
    X[i+3] = prob['M_aero_hub'][i][0]
X[-3] = prob["lss_E"][0]
X[-2] = prob["bear1.mb_e"][0]
X[-1] = prob["bear2.mb_e"][0]

# Xtest = X[-3] # lss_E
Xtest = X[:3] # F_aero_hub

Y = g_mbsa(Xtest)
Y
# %%[markdown]
# from `copilot`
# ==============================================================
#%%
class MBSAEvaluator:

    def __init__(
        self,
        modeling_options,
        saved_data_csv,
    ):

        self.prob = om.Problem(
            reports=False
        )

        self.prob.model = MBSA(
            modeling_options=modeling_options
        )

        self.prob.setup()

        print("- loading prob vars from saved csv ...")
        self.prob = load_data(
            saved_data_csv,
            self.prob
        )

        # store constr names, operator and limits in dict
        self.constr_info = { 
            "constr_lss_vonmises":{
                "operator": ot.Greater(),
                "threshold": 1.0
            },
            "constr_shaft_deflection":{
                "operator": ot.Greater(),
                "threshold": 1.0
            },
            "constr_shaft_angle":{
                "operator": ot.Greater(),
                "threshold": 1.0
            },
            "constr_L10_mb1":{
                "operator": ot.Less(),
                "threshold": 1.0
            },
            "constr_L10_mb2":{
                "operator": ot.Less(),
                "threshold": 1.0
            }
        }

        self.cache = {}

    def evaluate(
        self,
        F_aero_hub,
        M_aero_hub,
        lss_E,
        mb2_e,
        X_fls
    ):

        prob = self.prob

        # print("- F_aero_hub = ", F_aero_hub) # NOTE: testing
        prob.set_val("F_aero_hub", F_aero_hub)
        prob.set_val("M_aero_hub", M_aero_hub)
        prob.set_val("lss_E", lss_E)
        prob.set_val("bear2.mb_e", mb2_e)
        prob.set_val("mb_fls.X_fls", X_fls)

        prob.run_model()

        outputs = {

            "constr_lss_vonmises":
                float(
                    np.max(
                        prob[
                        "constr_lss_vonmises"
                        ]
                    )
                ),

            "constr_shaft_deflection":
                float(
                    np.max(
                        prob[
                        "constr_shaft_deflection"
                        ]
                    )
                ),

            "constr_shaft_angle":
                float(
                    np.max(
                        prob[
                        "constr_shaft_angle"
                        ]
                    )
                ),

            "constr_L10_mb1":
                float(
                    prob[
                    "constr_L10_mb1"
                    ][0]
                ),

            "constr_L10_mb2":
                float(
                    prob[
                    "constr_L10_mb2"
                    ][0]
                ),
        }

        return outputs

    def cached_evaluate(
        self,
        F_aero_hub,
        M_aero_hub,
        lss_E,
        mb2_e,
        X_fls
    ):

        key = tuple(
            np.round(
                np.hstack([
                    F_aero_hub,
                    M_aero_hub,
                    [lss_E],
                    [mb2_e],
                    [X_fls]
                ]),
                9
            )
        )

        if key in self.cache: return self.cache[key]

        result = self.evaluate(
            F_aero_hub,
            M_aero_hub,
            lss_E,
            mb2_e,
            X_fls
        )

        self.cache[key] = result

        return result

#%%
class MBSALimitState:

    def __init__(
        self,
        evaluator,
        response_name,
    ):

        self.evaluator     = evaluator
        self.response_name = response_name

    def __call__(self, X):

        F = np.array(X[0:3])

        M = np.array(X[3:6])

        E = X[6]

        e = X[7]

        X_fls = X[8]

        result = self.evaluator.cached_evaluate( # TODO cached eval
            F_aero_hub=F,
            M_aero_hub=M,
            lss_E=E,
            mb2_e=e,
            X_fls=X_fls
        )

        return [
            result[
                self.response_name
            ]
        ]
# %%
def compute_beta(
    distribution,
    evaluator,
    response_name,
):
    # ----------------
    # Distribution
    # ----------------
    startingPoint = distribution.getMean()
    sample = distribution.getSample(20)

    # ----------------
    # Define the limit state function using the MBSALimitState class
    # ----------------
    g = ot.PythonFunction(
        inputDim=9,
        outputDim=1,
        func=MBSALimitState(
            evaluator,
            response_name
        )
    )

    # ----------------
    # Create a random vector and a composite random vector
    # ----------------
    vect = ot.RandomVector(
        distribution
    )
    G = ot.CompositeRandomVector(
        g,
        vect
    )

    # ----------------
    # Define the event corresponding to the limit state function
    # ----------------
    constr_info = evaluator.constr_info[response_name]
    print(f"- {response_name} | operator:{constr_info['operator']}, threshold:{constr_info['threshold']}") # testing
    event = ot.ThresholdEvent(
        G,
        constr_info['operator'],
        constr_info['threshold']
    )
    event.setName("deviation")

    # ----------------
    # Rough estimate beta beforehand to skip inactive constraints
    # ----------------
    vals = np.array(
        [g(x)[0] for x in sample]
        )    
    mu = vals.mean()
    sigma = vals.std()
    if bool(np.ptp(vals) < 1.e-8) or bool(sigma < 1e-16):
        print(f"-- constraint appears deterministic: limit state constant wrt. uncertain vars: sigma={sigma}; returning safe values.")
        return np.inf, 0.0, None
    # beta_est
    threshold = constr_info['threshold']
    if str(constr_info['operator']) == '<':
        beta_est = (mu-threshold)/sigma
    elif str(constr_info['operator']) == '>':
        beta_est = (threshold-mu)/sigma
    if bool(beta_est > 8.0):
        print(f"-- inactive reliab constr: estimate beta {beta_est} > 8.0; returning safe values.")
        return np.inf, 0.0, None

    # ----------------
    # FORM
    # ----------------
    optimAlgo = ot.Cobyla()
    optimAlgo.setStartingPoint( startingPoint )
    optimAlgo.setMaximumCallsNumber( 1000 ) # TODO
    maxError = 1.e-3
    optimAlgo.setMaximumAbsoluteError(maxError)
    optimAlgo.setMaximumRelativeError(maxError)
    optimAlgo.setMaximumResidualError(maxError)
    optimAlgo.setMaximumConstraintError(maxError)

    algo = ot.FORM(
        optimAlgo,
        event,
        startingPoint
    )

    algo.run()

    result = algo.getResult()

    return (
        result.getHasoferReliabilityIndex(),
        result.getEventProbability(),
        result
    )

# %%
def make_distribution_of_mbsa_inputs():
    """
    Inputs
    _______
    same inputs and dims as g_mbsa

    Progress
    _______
    1. DONE: implement a basic working example inshaAllah
    2. TODO: define standard devs of each random var correctly
    """
    # From saved csv
    savedDataDF = pd.read_csv(loc_load_saved_data+".csv")
    savedDataDict = var_df2dict(savedDataDF)
    # Inputs: fill
    F_aero_hub = np.array(eval(savedDataDict["F_aero_hub"])).reshape(3,)
    M_aero_hub = np.array(eval(savedDataDict["M_aero_hub"])).reshape(3,)
    lss_E = float(savedDataDict["lss_E"])
    mb1_e = float(savedDataDict["bear1.mb_e"])
    mb2_e = float(savedDataDict["bear2.mb_e"])
    # Inputs: random distibutions saved
    all_loads_dict = sio.loadmat(loc_all_loads_mat_file)
    str_max_dist = "_max_mu_norm_sigma"

    dims = 9

    # F_aero_hub
    # - 1.
    F1_dist = all_loads_dict["Fx"+str_max_dist]
    F1 = F_aero_hub[0] * ot.LogNormal( F1_dist[0,0], F1_dist[0,1] ) # in N
    F1.setDescription([r"$F_x^{ULS}$"])
    F1.setName("F_aero_hub_x")
    # - 2.
    F2_dist = all_loads_dict["Fy"+str_max_dist]
    F2 = F_aero_hub[1] * ot.LogNormal( F2_dist[0,0], F2_dist[0,1] ) # in N
    F2.setDescription([r"$F_y^{ULS}$"])
    F2.setName("F_aero_hub_y")
    # - 3.
    F3_dist = all_loads_dict["Fz"+str_max_dist]
    F3 = F_aero_hub[2] * ot.LogNormal( F3_dist[0,0], F3_dist[0,1] ) # in N
    F3.setDescription([r"$F_z^{ULS}$"])
    F3.setName("F_aero_hub_z")

    # M_aero_hub
    # - 1.
    M1_dist = all_loads_dict["Mx"+str_max_dist]
    M1 = M_aero_hub[0] * ot.LogNormal( M1_dist[0,0], M1_dist[0,1] ) # in N
    M1.setDescription([r"$M_x^{ULS}$"])
    M1.setName("M_aero_hub_x")
    # - 2.
    M2_dist = all_loads_dict["My"+str_max_dist]
    M2 = M_aero_hub[1] * ot.LogNormal( M2_dist[0,0], M2_dist[0,1] ) # in N
    M2.setDescription([r"$M_y^{ULS}$"])
    M2.setName("M_aero_hub_y")
    # - 3.
    M3_dist = all_loads_dict["My"+str_max_dist]
    M3 = M_aero_hub[2] * ot.LogNormal( M3_dist[0,0], M3_dist[0,1] ) # in N
    M3.setDescription([r"$M_z^{ULS}$"])
    M3.setName("M_aero_hub_z")

    # # Young's modulus E (in N/m^2)
    # - 1. real
    E_cov = 0.02 # 2-3 %
    E = lss_E * ot.Normal(1, E_cov)  # in N
    # - 2. test TODO 
    # E = ot.Beta(0.9, 3.5, lss_E*0.999, lss_E*1.001)
    E.setDescription(r"$E$")
    E.setName("Young modulus")

    # MB2's e (not mb1, coz its assumed only-radial reacting so no 'e' used)
    e_mb2 = ot.TruncatedDistribution(
        ot.Normal(
            mb2_e,
            0.02*mb2_e
        ),
        0.26,   # typical value of e might be 0.26 (contact angle = 10 deg) to
        1.5     # 1.5 (contact angle = 45 deg) as e = 1.5*tan(alpha)
    )
    e_mb2.setDescription([r"$e^{MB2}$"])
    e_mb2.setName("mb2_e")

    # FLS loads
    # = X_aero & X_dyn (cf. 2014_Nejad-On long term)
    X_fls = ot.LogNormal()
    X_fls.setParameter(
        ot.LogNormalMuSigma()(
            [1.0, 0.111915, 0.0] # TODO: 0.05 test
        )
    )
    X_fls.setDescription([r"$X^{FLS}$"]) # TODO: rename to \xi
    X_fls.setName("X_FLS_hub_load")

    # correlation matrix TODO
    # - Aerodynamic loads are highly correlated.
    # - Aeroelastic simulations can estimate these.
    # - Ignoring correlation can produce very misleading reliability indices.
    R = ot.CorrelationMatrix(dims)
    R[1,4] = 0.7 # Fy and My correlated, 0.7-0.9
    R[2,5] = 0.7 # Fz and Mz correlated, 0.7-0.9
    copula = ot.NormalCopula(
        ot.NormalCopula.GetCorrelationFromSpearmanCorrelation(R)
    )
    distribution = ot.JointDistribution(
        [F1,F2,F3, M1,M2,M3, E, e_mb2, X_fls], copula
    )

    # TODO testing: correlation matrix of 3
    # dims = 3
    # R = ot.CorrelationMatrix(dims)
    # copula = ot.NormalCopula(
    #     ot.NormalCopula.GetCorrelationFromSpearmanCorrelation(R)
    # )
    # distribution = ot.JointDistribution(
    #     [F1,F2,F3], copula
    # )

    return distribution

#%%
# post-processing functions

# Importance factors
def draw_importance_factors( results ):
    graph = results.drawImportanceFactors()
    view = otv.View(graph)

#%%
# Add uncertainty to FLS hub loads # TODO
opts["WISDEM"]["DriveSE"]["reliability"] = True

#%%
distribution = make_distribution_of_mbsa_inputs()
evaluator = MBSAEvaluator(opts,loc_load_saved_data+".csv")

#%%
beta_vm, pf_vm, results_vm = compute_beta(
    distribution,
    evaluator,
    "constr_lss_vonmises"
)

beta_shaft_defl, pf_shaft_defl, results_shaft_defl = compute_beta(
    distribution,
    evaluator,
    "constr_shaft_deflection"
)

beta_shaft_angle, pf_shaft_angle, results_shaft_angle = compute_beta(
    distribution,
    evaluator,
    "constr_shaft_angle"
)

beta_mb1, pf_mb1, results_mb1 = compute_beta(
    distribution,
    evaluator,
    "constr_L10_mb1"
)
#%%
beta_mb2, pf_mb2, results_mb2 = compute_beta(
    distribution,
    evaluator,
    "constr_L10_mb2"
)

#%%
class ReliabiltyComponent( om.ExplicitComponent ):
    """
    OpenMDAO Optimizer
        │
        ▼
    ReliabilityComponent
        │
        ├── FORM(g1)  -> beta_vonmises
        ├── FORM(g2)  -> beta_mb1
        ├── FORM(g3)  -> beta_mb2
        └── ...
                │
                ▼
        MBSAEvaluator
                │
                ▼
        persistent MBSA Problem
    """
    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        # init
        opts = self.options["modeling_options"]
        n_dlcs = opts["n_dlcs"]

        self.distribution = make_distribution_of_mbsa_inputs()
        self.evaluator = MBSAEvaluator(opts,loc_load_saved_data+".csv")

        # add inputs TODO: only the design variables

        # add ouputs TODO: reliability results as constraints
        self.add_output("beta", val=3.0, desc="reliability index, min of all if many")
        self.add_output("Pf", val=1e-6, desc="failure probability, min of all if many")

        self.add_output("beta_vonmises")
        self.add_output("beta_shaft_defl")
        self.add_output("beta_shaft_angle")
        self.add_output("beta_mb1")
        self.add_output("beta_mb2")
        self.add_output("beta_min")

    def compute(self, inputs, outputs):

        distribution = self.distribution
        evaluator = self.evaluator

        beta_vm, pf_vm = compute_beta(
            distribution,
            evaluator,
            "constr_lss_vonmises"
        )

        beta_shaft_defl, pf_shaft_defl = compute_beta(
            distribution,
            evaluator,
            "constr_shaft_deflection"
        )

        beta_shaft_angle, pf_shaft_angle = compute_beta(
            distribution,
            evaluator,
            "constr_shaft_angle"
        )

        beta_mb1, pf_mb1 = compute_beta(
            distribution,
            evaluator,
            "constr_L10_mb1"
        )

        beta_mb2, pf_mb2 = compute_beta(
            distribution,
            evaluator,
            "constr_L10_mb2"
        )

        outputs["beta_vonmises"] = beta_vm
        outputs["beta_shaft_defl"] = beta_shaft_defl
        outputs["beta_shaft_angle"] = beta_shaft_angle
        outputs["beta_mb1"]      = beta_mb1
        outputs["beta_mb2"]      = beta_mb2

        outputs["beta_min"] = min([
            beta_vm,
            beta_shaft_defl,
            beta_shaft_angle,
            beta_mb1,
            beta_mb2
        ])

# %%
# FLS uncertain class =======================
#%%
from wisdem.drivetrainse.drive_structure import Analytical_FLS_Bearing_Life

class Analytical_FLS_Bearing_Life_RBDO(
    Analytical_FLS_Bearing_Life
):
    """
    RBDO extension of the deterministic WISDEM
    Analytical_FLS_Bearing_Life component.

    The parent performs the deterministic FLS bearing-life
    calculation.

    This subclass only provides an interface for applying
    stochastic load factors before the parent calculation.
    """

    def initialize(self):
        super().initialize()

        self.options.declare(
            "rbdo_load_factors",
            default=None,
            allow_none=True,
        )

    def setup(self):
        super().setup()

        # RBDO load multipliers
        self.add_input("lambda_Fx", val=1.0)
        self.add_input("lambda_Fy", val=1.0)
        self.add_input("lambda_Fz", val=1.0)

        self.add_input("lambda_Mx", val=1.0)
        self.add_input("lambda_My", val=1.0)
        self.add_input("lambda_Mz", val=1.0)
