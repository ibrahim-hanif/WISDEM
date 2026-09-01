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
flag_opt_GFO = False

# post-processing results
make_xdsm, xdsm_type = False, "html"       # html-show or detailed pdf
record_cases = False    #TODO: add in final setup (full problem)
plot_cases = True      #NOTE: saved, not changing now (commented)
flag_scaling_show_browser = False
flag_save_new_data = False
flag_load_from_data = True
flag_load_from_02data = False

# Loading `openFAST` hub loads from a saved file
# TODO: dont even need to do this now, coz `Load_Own_Hub_Loads` component does it internally and outputs the needed loads for the DT component. So, can just set `own_hub_loads=True` in `modelling_options` and not worry about loading the loads here in the script. JazakumAllahu khayr.
part_loads = True 
load_fls_loads = False
# False: full loads (72e4,10) (200 Hz sampled, 60mins)
# True: part loads (72e3,11) (20 Hz sampled, 60mins)
dir_loads = "M:\\Vasudev_Gupta\\outputs_mainshaft_loads"
loc_all_loads_mat_file = os.path.join(dir_loads, "hub_loads_M4W.mat")
if "sima" in suffix: # NOTE: RBDO sima loads
    loc_all_loads_mat_file = "C://SIMA_M4W_loads//rbdo_main_shaft_loads.mat"

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
        "cases_recorded"+suffix+".sql")
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
if False: #record_cases:
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

# %% [markdown]
# ### Reliability-based design optimization (RBDO)
# -------------------------- using `openturns` --------------------------

#%%
# Pf = Phi( -beta )
Phi = ot.Normal(0,1) # standard normal distribution
beta_s = np.asarray([1.28, 2.33, 3.09, 3.72, 4.26, 4.75, 5.2, 10.0])
pf_s = np.asarray([ Phi.computeCDF(-beta_i) for beta_i in beta_s ])

graphCDF = Phi.drawCDF() #(xMin=-1.28,xMax=-5.2,logScale=True)
graphCDF.setLegends(['normal cdf'])
otv.View(graphCDF)

graphPDF = Phi.drawPDF() #(xMin=-1.28,xMax=-5.2,logScale=True)
graphPDF.setLegends(['normal pdf'])
otv.View(graphPDF)

#%%
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

#%%
# post-processing functions

# Importance factors
def draw_importance_factors( results ):
    graph = results.drawImportanceFactors()
    view = otv.View(graph)

def draw_beta_sensitities( results ):
    marginalSensitivity, otherSensitivity = results.drawHasoferReliabilityIndexSensitivity()
    marginalSensitivity.setLegends([
        "Fx", "Fy", "Fz",
        "Mx", "My", "Mz",
        "E", "e", "X_fls"
    ])
    marginalSensitivity.setLegendPosition("bottom")
    view = otv.View(marginalSensitivity)

def draw_event_probab_sensitities( results ):
    marginalSensitivity, otherSensitivity = results.drawEventProbabilitySensitivity()
    marginalSensitivity.setLegends([
            "Fx", "Fy", "Fz",
            "Mx", "My", "Mz",
            "E", "e", "X_fls"
        ])
    marginalSensitivity.setLegendPosition("bottom")
    view = otv.View(marginalSensitivity)

def draw_optim_error_history( results) :
    # Error history
    optimResult = results.getOptimizationResult()
    graphErrors = optimResult.drawErrorHistory()
    graphErrors.setLegendPosition("bottom")
    graphErrors.setYMargin(0.0)
    view = otv.View(graphErrors)

# %%[markdown]
# from `copilot`
# ==============================================================

#%%
# Add uncertainty to FLS hub loads # TODO
opts["WISDEM"]["DriveSE"]["reliability"] = True
doMBfls = opts["flags"]["mb_fls"] = True

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
    str_max_dist = "_max_mean_std"
    str_max = "_max"

    # F_aero_hub
    # CoV = std / mean = sigma / mu
    CoV_uls = 0.01 # 0.01 test; 0.1 cf. 2021_Al-Sanad
    def make_dist_lognormal( sigma, mu=1.0 ):    
        X_uls = ot.LogNormal()
        X_uls.setParameter(
            ot.LogNormalMuSigma()(
                [ mu, sigma, 0.0]
            )
        )
        return X_uls
    X_uls = make_dist_lognormal( CoV_uls )
    # - 1.
    F1_mean = float(all_loads_dict["Fx"+str_max_dist][0,0])
    F1_sigma = float(all_loads_dict["Fx"+str_max_dist][0,1])
    F1 = make_dist_lognormal( F1_sigma, F1_mean )
    # F1 = F_aero_hub[0] * X_uls # TODO: * _dist OR * X_uls
    F1.setDescription([r"$F_x^{ULS}$"])
    F1.setName("F_aero_hub_x")
    # - 2.
    F2_mean = float(all_loads_dict["Fy"+str_max_dist][0,0])
    F2_sigma = float(all_loads_dict["Fy"+str_max_dist][0,1])
    F2 = make_dist_lognormal( F2_sigma, F2_mean )
    # F2 = F_aero_hub[1] * X_uls # * _dist OR * X_uls
    F2.setDescription([r"$F_y^{ULS}$"])
    F2.setName("F_aero_hub_y")
    # - 3.
    F3_mean = float(all_loads_dict["Fz"+str_max_dist][0,0])
    F3_sigma = float(all_loads_dict["Fz"+str_max_dist][0,1])
    F3 = make_dist_lognormal( F3_sigma, F3_mean )
    # F3 = F_aero_hub[2] * X_uls # * _dist OR * X_uls
    F3.setDescription([r"$F_z^{ULS}$"])
    F3.setName("F_aero_hub_z")

    # M_aero_hub
    # - 1.
    M1_mean = float(all_loads_dict["Mx"+str_max_dist][0,0])
    M1_sigma = float(all_loads_dict["Mx"+str_max_dist][0,1])
    M1 = make_dist_lognormal( M1_sigma, M1_mean )
    # M1 = M_aero_hub[0] * X_uls # * _dist OR * X_uls
    M1.setDescription([r"$M_x^{ULS}$"])
    M1.setName("M_aero_hub_x")
    # - 2.
    M2_mean = float(all_loads_dict["My"+str_max_dist][0,0])
    M2_sigma = float(all_loads_dict["My"+str_max_dist][0,1])
    M2 = make_dist_lognormal( M2_sigma, M2_mean )
    # M2 = M_aero_hub[1] * X_uls # * _dist OR * X_uls
    M2.setDescription([r"$M_y^{ULS}$"])
    M2.setName("M_aero_hub_y")
    # - 3.
    M3_mean = float(all_loads_dict["Mz"+str_max_dist][0,0])
    M3_sigma = float(all_loads_dict["Mz"+str_max_dist][0,1])
    M3 = make_dist_lognormal( M3_sigma, M3_mean )
    # M3 = M_aero_hub[2] * X_uls # * _dist OR * X_uls
    M3.setDescription([r"$M_z^{ULS}$"])
    M3.setName("M_aero_hub_z")

    # # Young's modulus E (in N/m^2)
    # - 1. real
    E_cov = 0.05 # 2-3 % cf. 2021_Al-Sanad
    E = lss_E * ot.Normal(1, E_cov)  # in N
    # - 2. test TODO 
    # E = ot.Beta(0.9, 3.5, lss_E*0.999, lss_E*1.001)
    E.setDescription([r"$E$"])
    E.setName("Young modulus")

    # MB2's e (not mb1, coz its assumed only-radial reacting so no 'e' used)
    e_mb2 = mb2_e * ot.TruncatedDistribution(
        ot.Normal(
            1,
            0.01
        ),
        0.26,   # typical value of e might be 0.26 (contact angle = 10 deg) to
        1.5     # 1.5 (contact angle = 45 deg) as e = 1.5*tan(alpha)
    )
    e_mb2.setDescription([r"$e^{MB2}$"])
    e_mb2.setName("mb2_e")


    # X uncertain
    # - tab.4.1 (2014_Torp-Safety_Factors_IEC_61400-1_ed_4_-_background_document.pdf)
    X_exp = make_dist_lognormal(0.15,1.0)
    X_dyn = make_dist_lognormal(0.05,1.0)
    X_aero= ot.Gumbel()
    X_aero.setParameter(ot.GumbelMuSigma()([1.0,0.1]))
    X_all = X_exp*X_dyn*X_aero
    # X_all.getSample(10)

    # FLS loads
    # = X_aero & X_dyn (cf. 2014_Nejad-On long term)
    X_fls = ot.LogNormal()
    X_fls.setParameter(
        ot.LogNormalMuSigma()( # NOTE: Good if physical mean = 1.0, physical std = 0.111915.
            [1.0, 0.05, 0.0] # TODO: 0.01 test; 0.111915 actual
        )
    )
    X_fls.setDescription([r"$\chi^{FLS}$"]) # TODO: rename to \xi
    X_fls.setName("X_FLS_hub_load")

    # ----
    dims = 8

    # correlation matrix TODO
    # - Aerodynamic loads are highly correlated.
    # - Aeroelastic simulations can estimate these.
    # - Ignoring correlation can produce very misleading reliability indices.
    R = ot.CorrelationMatrix(dims)
    R[1,4] = 0. # Fy and My correlated, 0.7-0.9
    R[2,5] = 0. # Fz and Mz correlated, 0.7-0.9
    copula = ot.NormalCopula(
        ot.NormalCopula.GetCorrelationFromSpearmanCorrelation(R)
    )
    distribution = ot.JointDistribution(
        [
            F1,
            F2,
            F3,
            M1,
            M2,
            M3,
            E,
            # e_mb2,
            X_all
        ], copula
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

# %%[markdown]
# ### RBDO with improvements
# =============================================================================
# =============================================================================

#%%
class MBSA_Evaluator:
    """
    deterministic MBSA evaluator

    Responsible for:
    > Given `X` and `design_variables`, run WISDEM once and
    calculate all reliability limit states.
    """

    def __init__(
        self,
        modeling_options,
        saved_data_csv,
    ):

        model_opts = self.model_opts = modeling_options
        doMBfls = model_opts["flags"]["mb_fls"]

        # ------------------------------------------------------------
        # Build WISDEM/OpenMDAO problem
        # ------------------------------------------------------------
        prob = om.Problem(
            reports=False
        )

        prob.model = MBSA(
            modeling_options=model_opts
        )

        prob.setup()

        print("- loading prob vars from saved csv ...")
        prob = load_data(
            saved_data_csv,
            prob
        )

        self.prob = prob

        # ------------------------------------------------------------
        # Cache
        # ------------------------------------------------------------
        self.cache = {}

        # ------------------------------------------------------------
        # Reliability responses ONLY
        #
        # g >= 0 : safe
        # g <  0 : failure
        #
        # g = m * response + c
        # ------------------------------------------------------------
        # store constr names, operator and limits in dict
        # - in the form of: g = m * result + c
        # - with tuple (m,c) defined of each
        self.constr_info = { 
            "constr_lss_vonmises": # Greater than 1.0 is fail
            (-1.0, 1.0),
            "constr_shaft_deflection": # Greater than 1.0 is fail
            (-1.0, 1.0),
            "constr_shaft_angle": # Greater than 1.0 is fail
            (-1.0, 1.0),
            #
            "msa_mass":
            (1.0, 0.0)
        }
        if doMBfls:
            self.constr_info.update({
                "constr_L10_mb1": # Less than 1.0 is fail
                (1.0, -1.0),
                "constr_L10_mb2": # Less than 1.0 is fail
                (1.0, -1.0),
            })

        # ------------------------------------------------------------
        # Objective responses
        # ------------------------------------------------------------
        self.objective_names = [
            "msa_mass"
        ]

    # ================================================================
    # Evaluate
    # ================================================================
    def evaluate(self, X, design_variables):

        # init
        constr_info = self.constr_info
        model_opts = self.model_opts

        X = np.asarray(X, dtype=float)

        L_h1 = float(design_variables["L_h1"][0])
        L_12 = float(design_variables["L_12"][0])
        D_lss = design_variables["lss_diameter"]
        t_lss = design_variables["lss_wall_thickness"]

        # ------------------------------------------------
        # Cache key MUST include design variables: TODO
        # ------------------------------------------------
        cache_key = tuple(
            np.asarray( # np.round; NOTE: introduce rounding if you have evidence that floating-point noise is preventing useful cache hits
                np.concatenate([
                    np.asarray(X),
                    np.asarray([L_h1, L_12]),
                    np.asarray(D_lss),
                    np.asarray(t_lss) # already arrays
                ]),
                dtype=float, #12,
            )
        )

        # ------------------------------------------------
        # RETURN CACHED RESULTS
        # ------------------------------------------------

        if cache_key in self.cache:
            return self.cache[cache_key]

        # ------------------------------------------------
        # Uncertain variables
        # ------------------------------------------------
        Fx = X[0]
        Fy = X[1]
        Fz = X[2]
        F_aero_hub = np.array([Fx, Fy, Fz]).reshape(3, 1)

        Mx = X[3]
        My = X[4]
        Mz = X[5]
        M_aero_hub = np.array([Mx, My, Mz]).reshape(3, 1)

        E_lss = float(X[6])
        # mb2_e = float(X[7])
        X_fls = float(X[7])        

        # ------------------------------------------------
        # Design variables
        # ------------------------------------------------
        prob = self.prob
        # Use the actual WISDEM/OpenMDAO variable names
        prob.set_val("L_h1", L_h1)
        prob.set_val("L_12", L_12)
        prob.set_val("lss_diameter", D_lss)
        prob.set_val("lss_wall_thickness", t_lss)

        # ------------------------------------------------
        # Random variables
        # ------------------------------------------------

        prob.set_val("F_aero_hub", F_aero_hub )
        prob.set_val("M_aero_hub", M_aero_hub )
        prob.set_val("lss_E", E_lss)
        # prob.set_val("bear2.mb_e", mb2_e)
        if model_opts["flags"]["mb_fls"]: prob.set_val("mb_fls.X_fls", X_fls) # X_fls is handled by your Analytical_* component

        # ------------------------------------------------
        # Run WISDEM/MBSA
        # ------------------------------------------------
        prob.run_model()

        # ------------------------------------------------
        # Retrieve responses
        # ------------------------------------------------
        results = {}

        for response_name, mc in constr_info.items():
            mbsa_result = float(np.max(np.asarray(
                                prob[ response_name ]
                            )))
            m, c = mc
            g = m * mbsa_result + c

            results[ response_name ] = g

        # ------------------------------------------------
        # Store COMPLETE result dictionary
        # ------------------------------------------------

        self.cache[cache_key] = results

        return results

#%%
class MBSA_LimitStateModel:
    """
    adapter between MBSA and OpenTURNS
    
    Responsible for:
    > Give OpenTURNS the particular `g(X)` that I want.

    OpenTURNS model returning all MBSA limit-state functions.

    Output convention:
        0 = von Mises
        1 = shaft deflection
        2 = shaft angle
        3 = MB1 L10
        4 = MB2 L10
    """

    def __init__(self, evaluator, design_variables, response_name):
        self.evaluator = evaluator
        self.design_variables = design_variables
        self.response_name = response_name

        # Validation on the 'response_name'
        if response_name not in evaluator.constr_info:
            raise ValueError(
                f"Unknown response '{response_name}'. "
                f"Available responses: "
                f"{list(evaluator.constr_info.keys())}"
            )

    def __call__(self, X):
        X = np.asarray(X, dtype=float)

        results = self.evaluator.evaluate(
            X,
            design_variables=self.design_variables,
        )

        # constr_info = self.evaluator.constr_info
        # g = []
        # for key, _ in constr_info.items():
        #     g.append( results[key] )
            
            # NOTE: below reference only, not incl in the code  :)
            # [
            #     results["constr_lss_vonmises"],
            #     results["constr_shaft_deflection"],
            #     results["constr_shaft_angle"],
            #     results["constr_L10_mb1"],
            #     results["constr_L10_mb2"],
            # ]

        return [ results[ self.response_name ]]

#%%
def compute_beta_new(
    distribution, dist_samples, dist_mean,
    evaluator,
    response_name,
    design_variables,
    max_calls=1.E3,
):
    """
    reliability analysis
    """

    constraint_index = { # TODO: better use evaluator.constr_info ?
        "constr_lss_vonmises": 0,
        "constr_shaft_deflection": 1,
        "constr_shaft_angle": 2,
        "constr_L10_mb1": 3,
        "constr_L10_mb2": 4,
    }

    # idx = constraint_index[response_name]
    dimDist = int(distribution.getDimension())

    # --------------------------------------------------
    # Vector-valued MBSA model
    # --------------------------------------------------

    model_wrapper = MBSA_LimitStateModel(
        evaluator=evaluator,
        design_variables=design_variables,
        response_name=response_name
    )

    model = ot.PythonFunction(
        inputDim=dimDist,     # random variables only
        outputDim=1,    # all 5 reliability responses
        func=model_wrapper
    )

    # --------------------------------------------------
    # Select requested limit state
    # --------------------------------------------------

    # scalar_model = model.getMarginal([idx])

    # --------------------------------------------------
    # Random vector
    # --------------------------------------------------

    X = ot.RandomVector(distribution)

    G = ot.CompositeRandomVector(
        model,
        X,
    )

    # --------------------------------------------------
    # Failure event
    #
    # ALL g-functions use:
    #
    #     g > 0 : safe
    #     g <= 0: failure
    # --------------------------------------------------

    event = ot.ThresholdEvent(
        G,
        ot.Less(),
        0.0,
    )

    event.setName(response_name)

    # --------------------------------------------------
    # Beta: initial screening
    # - quick response-space screening estimate
    # - beforehand to skip inactive constraints
    # --------------------------------------------------
    # sample = distribution.getSample(20) # handled by dist_samples
    values = model( dist_samples )

    # i = constraint_index[ response_name ]

    vals = np.asarray(values)
    mu = np.mean(vals)
    sigma = np.std(vals)

    if bool(np.ptp(vals) < 1.e-8) or bool(sigma < 1e-16):
        print(f"-- constraint appears deterministic: limit state constant wrt. uncertain vars: sigma={sigma}; returning safe values.")
        return 8.0, 0.0, None
    # beta_est
    # = correct direction for a quick response-space screening estimate.
    beta_est = mu / sigma
    if bool(beta_est > 8.0):
        print(f"-- inactive reliab constr: estimate beta {beta_est} > 8.0; returning safe values.")
        return 8.0, 0.0, None

    # --------------------------------------------------
    # FORM
    # --------------------------------------------------

    # starting_point = distribution.getMean()
    starting_point = dist_mean

    optimAlgo = ot.Cobyla()

    optimAlgo.setStartingPoint(
        starting_point
    )

    optimAlgo.setMaximumCallsNumber(
        int(max_calls)
    )

    maxError = 1.0e-3

    optimAlgo.setMaximumAbsoluteError(maxError)
    optimAlgo.setMaximumRelativeError(maxError)
    optimAlgo.setMaximumResidualError(maxError)
    optimAlgo.setMaximumConstraintError(maxError)

    algo = ot.FORM(
        optimAlgo,
        event,
        starting_point,
    )

    algo.run()

    result = algo.getResult()
    # Pf
    pf = result.getEventProbability()
    # print(f" --- result Pf = {pf}") # test
    # Beta
    beta = result.getHasoferReliabilityIndex()
    # print(f" --- result beta = {beta}") # test

    normal = ot.Normal()
    beta_calc = -normal.computeQuantile(pf)[0]
    # print(f" --- calculated beta (from pf) = {beta_calc}") # test
    pf_calc = normal.computeCDF(-beta)
    # print(f" --- calculated pf (from beta) = {pf_calc}") # test

    return beta, pf_calc, result

#%%
# TEST
distribution = make_distribution_of_mbsa_inputs()
N_samples = 20
dist_samples = distribution.getSample( N_samples )
dist_mean = distribution.getMean()

evaluator = MBSA_Evaluator(opts,loc_load_saved_data+".csv")

design_variables = {
    "L_h1": np.array([4.95916667]), #prob["L_h1"],
    "L_12": np.array([5.56416667]), #prob["L_12"],
    "lss_diameter": np.array([2.03333333, 2.56666667]), #prob["lss_diameter"],
    "lss_wall_thickness": np.array([0.28093333, 0.08253333]) #prob["lss_wall_thickness"]
}



#%%
# Calculate the reliablity for each constraint

# ---- 
response_name = "msa_mass"
results_msa_mass = evaluator.evaluate(
    X=distribution.getMean(),
    design_variables=design_variables
)
print(f"{response_name}: { results_msa_mass[response_name] }")

# ---- 
response_name = "constr_lss_vonmises"
beta_vm, pf_vm, results_vm = compute_beta_new(
    distribution, dist_samples, dist_mean,
    evaluator,
    response_name,
    design_variables
)
print(f"{response_name}: beta={beta_vm}, pf={pf_vm}")

# ---- 
response_name = "constr_shaft_deflection"
beta_shaft_defl, pf_shaft_defl, results_shaft_defl = compute_beta_new(
    distribution, dist_samples, dist_mean,
    evaluator,
    response_name,
    design_variables
)
print(f"{response_name}: beta={beta_shaft_defl}, pf={pf_shaft_defl}")

# ---- 
response_name = "constr_shaft_angle"
beta_shaft_angle, pf_shaft_angle, results_shaft_angle = compute_beta_new(
    distribution, dist_samples, dist_mean,
    evaluator,
    response_name,
    design_variables
)
print(f"{response_name}: beta={beta_shaft_angle}, pf={pf_shaft_angle}")

# ---- 
if doMBfls:
    response_name = "constr_L10_mb1"
    beta_mb1, pf_mb1, results_mb1 = compute_beta_new(
        distribution, dist_samples, dist_mean,
        evaluator,
        response_name,
        design_variables
    )
    print(f"{response_name}: beta={beta_mb1}, pf={pf_mb1}")

# ---- 
if doMBfls:
    response_name = "constr_L10_mb2"
    beta_mb2, pf_mb2, results_mb2 = compute_beta_new(
        distribution, dist_samples, dist_mean,
        evaluator,
        response_name,
        design_variables
    )
    print(f"{response_name}: beta={beta_mb2}, pf={pf_mb2}")

#%%
# post-process results from FORM
if results_vm is not None: draw_importance_factors(results_vm)
if results_shaft_defl is not None: draw_importance_factors(results_shaft_defl)
if results_shaft_angle is not None: draw_importance_factors(results_shaft_angle)
if doMBfls and results_mb1 is not None: draw_importance_factors(results_mb1)
if doMBfls and results_mb2 is not None: draw_importance_factors(results_mb2)

#%%[markdown]
# ------------------------------ RBDO ----------------------------------
#%%
class ReliabilityComponent_new( om.ExplicitComponent ):
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
        n_dlcs = opts["WISDEM"]["n_dlc"]
        doMBfls = opts["flags"]["mb_fls"]

        self.distribution = make_distribution_of_mbsa_inputs()
        self.dist_samples = self.distribution.getSample( N_samples )
        self.dist_mean = self.distribution.getMean()

        self.evaluator = MBSA_Evaluator(opts,loc_load_saved_data+".csv")

        # -----------------------------------------
        # Design variables
        # -----------------------------------------
        self.add_input('L_12', val=0.0, desc='Main bearing span', units='m')
        self.add_input('L_h1', val=0.0, desc='Rotor bearing distance', units='m')
        self.add_input("lss_diameter", val=np.zeros(2), units="m") #(v) a DV to be optim
        self.add_input("lss_wall_thickness", val=np.zeros(2), units="m") #(v) a DV to be optim

        # -----------------------------------------
        # Reliability outputs
        # -----------------------------------------
        def_beta, def_pf = 3.0, 1.e-3
        # Beta
        self.add_output( "beta_vonmises", val=def_beta )
        self.add_output( "beta_shaft_deflection", val=def_beta )
        self.add_output( "beta_shaft_angle", val=def_beta )
        if doMBfls:
            self.add_output( "beta_mb1", val=def_beta )
            self.add_output( "beta_mb2", val=def_beta )
        # Pf
        self.add_output( "pf_vonmises", val=def_pf )
        self.add_output( "pf_shaft_deflection", val=def_pf )
        self.add_output( "pf_shaft_angle", val=def_pf )
        if doMBfls:
            self.add_output( "pf_mb1", val=def_pf )
            self.add_output( "pf_mb2", val=def_pf )
        # obj
        self.add_output("msa_mass", val=0.0)

    def compute(self, inputs, outputs):

        design_variables = {
            "L_h1": inputs["L_h1"],
            "L_12": inputs["L_12"],
            "lss_diameter": inputs["lss_diameter"],
            "lss_wall_thickness": inputs["lss_wall_thickness"]
        }

        # Calculate the objective
        obj_name = "msa_mass"
        results_msa_mass = evaluator.evaluate(
            X=self.dist_mean,
            design_variables=design_variables
        )
        obj_val = results_msa_mass[obj_name]
        print(f"{ obj_name }: { obj_val }") # test
        outputs[obj_name] = obj_val

        constraints = [
            ("constr_lss_vonmises",
            "beta_vonmises",
            "pf_vonmises"),

            ("constr_shaft_deflection",
            "beta_shaft_deflection",
            "pf_shaft_deflection"),

            ("constr_shaft_angle",
            "beta_shaft_angle",
            "pf_shaft_angle"),
        ]
        if doMBfls:
            constraints.append(
                ("constr_L10_mb1",
                "beta_mb1",
                "pf_mb1"),
            )
    
            constraints.append(
                ("constr_L10_mb2",
                "beta_mb2",
                "pf_mb2")
            )

        for constr_name, beta_name, pf_name in constraints:

            beta, pf, result = compute_beta_new(
                self.distribution, self.dist_samples, self.dist_mean,
                self.evaluator,
                constr_name,
                design_variables,
                max_calls=1.E3,
            )

            # test
            print(f"{constr_name}: beta={beta}, pf={pf}")

            outputs[beta_name] = beta
            outputs[pf_name] = pf

# %%[markdown]
# ### RBDO (reliability based design optimization)
# %%
flag_opt_GBO = False
flag_opt_GFO = True
flag_DOE = False

# ### The problem
# Define the problem
prob_rbdo = om.Problem(reports=False)
# Define the model
prob_rbdo.model = om.Group()
prob_rbdo.model.add_subsystem(
    "rbdo", ReliabilityComponent_new(modeling_options=opts),
    promotes=["*"])

# Optimization driver
# ---- 
if flag_opt_GBO:
    print("=== running GBO ===\n")
    # Choose the (GBO) optimizer to use
    prob_rbdo.driver = om.ScipyOptimizeDriver()
    prob_rbdo.driver.options["optimizer"] = "SLSQP"
    prob_rbdo.driver.options["tol"] = 1e-6 # comment to default (1e-6?)
    prob_rbdo.driver.options["maxiter"] = 5 #5 * 4 # TODO
    prob_rbdo.driver.options["disp"] = True
    prob_rbdo.driver.options["debug_print"] = ["desvars", "objs", "nl_cons", "ln_cons"]
    # prob_rbdo.driver.options # disp for debugging
    # prob_rbdo.set_solver_print(level=2)

elif flag_opt_GFO:
    print("=== running GFO ===\n")
    # GFO: gradient free optimizer
    # ---- Simple GA (Genetic Alg.)
    # prob_rbdo.driver = om.SimpleGADriver()
    # ---- Evolution
    # prob_rbdo.driver = om.DifferentialEvolutionDriver()
    # prob_rbdo.driver.options['max_gen'] = 400
    # prob_rbdo.driver.options['Pc'] = 0.5
    # prob_rbdo.driver.options['F'] = 0.5
    # ---- NSGA2
    from wisdem.optimization_drivers.nsga2_driver import NSGA2Driver
    prob_rbdo.driver = NSGA2Driver()
    prob_rbdo.driver.options["max_gen"] = 5 * 1 # * 4 # TODO
    prob_rbdo.driver.options["run_parallel"] = True
    prob_rbdo.driver.options["procs_per_model"] = 2 # TODO

    prob_rbdo.driver.options["debug_print"] = ["desvars", "objs", "nl_cons", "ln_cons"]

elif flag_DOE:
    print("=== running DOE ===\n")
    prob_rbdo.driver = om.DOEDriver(
        om.UniformGenerator(num_samples=5)
    )
    if record_cases:
        prob_rbdo.driver.add_recorder(
            om.SqliteRecorder( loc_cases )
        )

else:
    print("=== running analysis only (`run_model()`) ===\n")

# ---- 
if flag_opt_GBO or flag_opt_GFO or flag_DOE:
    print(" --- setting optimization obj, desvars, constrs --- \n")
    # Add objective
    prob_rbdo.model.add_objective("msa_mass", ref=1e6)
    # Add design variables
    prob_rbdo.model.add_design_var("L_h1", lower=0.1, upper=5.0, ref=5.0, ref0=0.1)
    prob_rbdo.model.add_design_var("L_12", lower=0.1, upper=8.0, ref=8.0, ref0=0.1)
    prob_rbdo.model.add_design_var("lss_diameter", lower=1.0, upper=5.0, ref=5.0, ref0=1.0)
    prob_rbdo.model.add_design_var("lss_wall_thickness", lower=4e-3, upper=0.5, ref=1.0, ref0=4e-3) #DONE: scaled so driver sees lb=0, ub=1 (why? 0.05 causes probs)
    # Add constraints
    prob_rbdo.model.add_constraint("beta_vonmises", lower=3.0)
    # prob_rbdo.model.add_constraint("beta_shaft_defl", lower=3.0)
    # prob_rbdo.model.add_constraint("beta_shaft_angle", lower=3.0)
    if doMBfls:
        prob_rbdo.model.add_constraint("beta_mb1", lower=3.0)
        prob_rbdo.model.add_constraint("beta_mb2", lower=3.0)

    if record_cases:
        recorder = om.SqliteRecorder( loc_cases )
        prob_rbdo.driver.add_recorder( recorder=recorder )

# Setup the problem
prob_rbdo.setup()

# Print objectives, design variables, and constraints in a concise readable form
print("\n=== All needed inputs to the model ===\n")
prob_rbdo.model.list_inputs();

print("\n=== All outputs from the model ===\n")
prob_rbdo.model.list_outputs();

# Set values of DVs
prob_rbdo.set_val("L_h1", float(prob["L_h1"][0]) )
prob_rbdo.set_val("L_12", float(prob["L_12"][0]) )
prob_rbdo.set_val("lss_diameter", prob["lss_diameter"])
prob_rbdo.set_val("lss_wall_thickness", prob["lss_wall_thickness"])

#%%
# ### Run: Optimization / DOE / Analysis
# `_driver` (optimization) / `_model` (analysis)

t0 = time.time()
if flag_opt_GBO:
    # Run GBO or DOE
    # prob_rbdo.model.approx_totals() # TODO.
    prob_rbdo.run_driver()

elif flag_opt_GFO or flag_DOE:
    # Run the GFO
    prob_rbdo.run_driver()

else:
    # Run the analysis
    prob_rbdo.run_model()

t1 = time.time()
print(" - WISDEM RBDO run completed in,", t1-t0, "seconds")

# %%
# post-processing
print("msa_mass", prob_rbdo["msa_mass"])
print("beta", prob_rbdo["beta_vonmises"])
print("pf", prob_rbdo["pf_vonmises"])

# %%
cr = om.CaseReader( loc_cases )
cases = cr.list_cases('driver')

values = []
for case in cases:
    outputs = cr.get_case(case).outputs
    values.append((outputs['x'].item(), outputs['y'].item(), outputs['f_xy'].item()))

print("\n".join(["x: %5.2f, y: %5.2f, f_xy: %6.2f" % xyf for xyf in values]))

# %%
if record_cases:
    print("\n=== Recorded cases from the optimization ===\n")
    results_dict = get_recorder_results( loc_cases, None, True )
    print(results_dict);
# %%
