#%%[markdown]
# # Base case: IEA 15 MW Direct-Drive VolturnUS Turbine
# 
# ### current version:
# Drivetrain Analysis: Replica of IEA-15-MW report for reference
# 
# references
# 1. IEA 15MW report
# 2. NOTE: all dimensions are from the latest iea15mw wisdem-weis windIO
#
# TODO
# 1. update wind speeds and probs, wrt sima data, + use wisdem's Weibull func?
# 2. use latest sima hub loads

#%%
# Import needed libraries
import os
import numpy as np
import openmdao.api as om
import time
import matplotlib.pyplot as plt
import pandas as pd

from wisdem.commonse.fileIO import save_data, load_data, var_df2dict
from wisdem.drivetrainse.drivetrain import DrivetrainSE, DrivetrainSE_M4W
from wisdem.commonse.utilities import load_all_mat_to_dict, pdf_norm_int_using_cdf
import Drive4Wind.utilities.utilities_drivetrain as utilsDT

#%%
# ### Define flags
suffix = "_sima_loads"
# information
# 1. '_old_loads':  old hub loads from felix' openfast (wrong) model
# 2. '_sima_loads': sima loads from seraj's sima (correct) model

opt_flag = True
opt_hub = False # (def: False) if to optimize hub, its Compn incl if dohub
flag_save_new_data = True
load_from_saved_data = False
flag_save_RNAprops4tower = True

# post-processing results
plot_cases = True
save_new_plot = True #NOTE: saved, not changing now (commented)

# -------
# Loading `openFAST` hub loads from a saved file
dir_loads = "M:\\Vasudev_Gupta\\outputs_mainshaft_loads"
loc_all_loads_mat_file = os.path.join(dir_loads, "hub_loads_M4w.mat")
if "sima" in suffix:
    loc_all_loads_mat_file = "C://SIMA_M4W_loads//all_main_shaft_loads.mat" # TODO: sima loads

# - results main dir
results_dir = "results"
script_dir = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(script_dir, results_dir)
os.makedirs(results_path, exist_ok=True)

# base case turbine csv
basecaseCSVpath = os.path.join(
    os.path.dirname(os.path.dirname(script_dir)),
    "02_reference_turbines","M4W_production_runs","outputs",
    "basecase_NOoptim.csv")
basecaseDF = pd.read_csv(basecaseCSVpath)
basecaseDict = var_df2dict(basecaseDF)

loc_save_data = os.path.join(results_path, "m4w_base_case_DT"+suffix)
if flag_save_RNAprops4tower:
    loc_save_RNAprops4tower = os.path.join(
        results_path, "RNA_props_model_for_tower"+suffix+".yaml")

# %% [markdown]
# ### Defining results directory and files
#%% Loading `openFAST` hub loads from a saved file
S_all, keys_all = load_all_mat_to_dict(loc_all_loads_mat_file)

#%%
# Set input options (modeling options dictionary)
# TODO: probabs check with wind site
from Drive4Wind.utilities.utilities_drivetrain import define_modeling_options_dict_for_drivetrainSE as defModelOpts
opts = defModelOpts(loc_all_loads_mat_file)

# base case DD specific
direct = opts["WISDEM"]["DriveSE"]["direct"] = True
opts["WISDEM"]["RotorSE"]["n_pc"] = 20

doMBfls = opts["flags"]["mb_fls"]
dohub = opts["flags"]["hub"] = True
dogen = opts["flags"]["generator"]

#%%
# Initialize OpenMDAO problem
prob = om.Problem(reports=False)
prob.model = DrivetrainSE_M4W(modeling_options=opts)
# ---

#%%
# If performing optimization, set up the optimizer and problem formulation
if opt_flag:
    print(" === running GBO === ")
    # Choose the optimizer to use
    prob.driver = om.ScipyOptimizeDriver() # selecting optimzer
    prob.driver.options["optimizer"] = "SLSQP"# configuring it
    prob.driver.options["tol"] = 1e-5
    prob.driver.options["maxiter"] = 5 * 20 # needs 80 iters to converge
    prob.driver.options["disp"] = True
    prob.driver.options["debug_print"] = ["desvars", "objs", "nl_cons", "ln_cons"]

    # Add objective
    prob.model.add_objective("nacelle_mass", scaler=1e-6) #scale to order 1 for better convergence behaviour

    # Add design variables, in this case the drivetrain diameters and wall thicknesses
    if opt_hub: prob.model.add_design_var("hub_diameter", lower=3.0, upper=15.0)
    prob.model.add_design_var("L_12", lower=0.1, upper=5.0)
    prob.model.add_design_var("L_h1", lower=0.1, upper=8.0)
    prob.model.add_design_var("lss_diameter", lower=0.5, upper=6.0)
    prob.model.add_design_var("lss_wall_thickness", lower=4e-3, upper=1.0, ref=1e-2)
    prob.model.add_design_var("nose_diameter", lower=0.5, upper=6.0)
    prob.model.add_design_var("nose_wall_thickness", lower=4e-3, upper=1.0, ref=1e-2)
    prob.model.add_design_var("bedplate_wall_thickness", lower=4e-3, upper=1.0, ref=1e-2)

    # Add constraints on the tower design
    # 1. von Mises stress util
    prob.model.add_constraint("constr_lss_vonmises", upper=1.0)
    prob.model.add_constraint("constr_bedplate_vonmises", upper=1.0)
    # 2. main bearing defl (allowed, max, as an angle) 
    prob.model.add_constraint("constr_mb1_defl", upper=1.0)
    prob.model.add_constraint("constr_mb2_defl", upper=1.0)
    prob.model.add_constraint("constr_shaft_deflection", upper=1.0)
    prob.model.add_constraint("constr_shaft_angle", upper=1.0)
    # prob.model.add_constraint("constr_stator_deflection", upper=1.0) #TODO: add
    prob.model.add_constraint("constr_stator_angle", upper=1.0)
    if doMBfls:
        prob.model.add_constraint("constr_L10_mb1", lower=1.0)
        prob.model.add_constraint("constr_L10_mb2", lower=1.0)
    # 3. hub dia to accom. blades' root radius 
    if opt_hub: prob.model.add_constraint("constr_hub_diameter", lower=0.0)
    # 4. target overhand and hub height
    prob.model.add_constraint("constr_length", lower=0.0)
    prob.model.add_constraint("constr_height", lower=0.0)
    prob.model.add_constraint("constr_ecc", lower=0.0)
    prob.model.add_constraint("L_lss", lower=0.1)
    prob.model.add_constraint("L_nose", lower=0.1)
    # prob.model.add_constraint("constr_Lh1_MB1fw", lower=0.0)#, ref=1e1) # TODO: cant incl in optim, fails
    # prob.model.add_constraint("constr_L12_MBsFW", lower=0.0)#, ref=1e0) # TODO: same as above
    # 5. maintainance access
    prob.model.add_constraint("constr_access", lower=0.0)
    # ---

else:
    print("=== running analysis only (`run_model()`) ===")

# Set up the OpenMDAO problem
prob.setup()
# ----
#%% overwrite variables from saved data
if load_from_saved_data:
    print(" loading prob vars from saved csv")
    prob = load_data( loc_save_data+".csv", prob )
#%%
# Set high-level input values (that desc the turbine)
machine_rating = float(basecaseDict["drivese.machine_rating"])
prob.set_val("machine_rating",machine_rating,"kW")
prob["upwind"] = bool(basecaseDict["drivese.upwind"])
D_rotor = prob["rotor_diameter"] = float(basecaseDict["drivese.rotor_diameter"])
prob["D_top"] = float(basecaseDict["drivese.D_top"]) #tower top diameter
prob["minimum_rpm"] = float(basecaseDict["drivese.minimum_rpm"])
rated_rpm = prob["rated_rpm"] = float(basecaseDict["drivese.rated_rpm"]) #7.56
prob["rated_torque"] = float(basecaseDict["drivese.rated_torque"]) # 21.3 * 1e6 # Nm
if doMBfls:
    prob["lifetime"] = float(basecaseDict["drivese.lifetime"]) #design life in years ('lifetime' from WEIS, WindIO)
prob["overhang"] = float(basecaseDict["drivese.overhang"]) #ref.2
prob["drive_height"] = float(basecaseDict["drivese.drive_height"])
prob["tilt"] = float(basecaseDict["drivese.tilt"]) #[deg] ref.3

# Loading from rotor
# prob["F_aero_hub"] = np.array([2517580.0, -27669.0, 3204.0]).reshape((3, 1))
# prob["M_aero_hub"] = np.array([21030561.0, 7414045.0, 1450946.0]).reshape((3, 1))
prob['F_aero_hub'] = np.array( [S_all['Fx_max'], S_all['Fy_max'], S_all['Fz_max']] ).reshape((3, 1))
prob['M_aero_hub'] = np.array( [S_all['Mx_max'], S_all['My_max'], S_all['Mz_max']] ).reshape((3, 1))
# ----

# Blade properties and hub design options
# --- NOTE: copied from M4W_base_case_driver's NOoptim csv
prob["hub_diameter"] = float(basecaseDict["drivese.hub_diameter"])
n_blades = prob["n_blades"] = float(basecaseDict["drivese.n_blades"])
blade_mass = prob["blade_mass"] = float(basecaseDict["drivese.blade_mass"])
prob["blades_mass"] = float(basecaseDict["drivese.blades_mass"]) #n_blades * blade_mass
prob["blades_cm"] = float(basecaseDict["drivese.blades_cm"])
prob["blades_I"] = eval(basecaseDict["drivese.blades_I"])

lstHub = ["flange_t2shell_t",
          "flange_OD2hub_D",
          "flange_ID2flange_OD",
          "hub_in2out_circ",
          "hub_stress_concentration",
          "n_front_brackets",
          "n_rear_brackets",
          "clearance_hub_spinner",
          "spin_hole_incr",
          "blade_root_diameter",

          "pitch_system.BRFM",
          "pitch_system_scaling_factor",

          "spinner_gust_ws"]
for name in lstHub:
    prob[name] = float(basecaseDict["drivese."+name])
# ----
#%%
# Drivetrain configuration and sizing inputs
prob["bear1.bearing_type"] = "CARB" # iea15 report: TRB2; latest wisdem: CARB
prob["bear2.bearing_type"] = "SRB"  
prob["bear1.D_shaft"] = 2.2
prob["bear2.D_shaft"] = 2.2
if doMBfls:
    prob["bear1.mb_e"] = 0.4 # from 0.3-0.4 
    prob["bear2.mb_e"] = 0.4
    # prob["bear2.mb_k"] = 0.0 #3e10
    prob["mb_fls.e_mb"] = prob["bear2.mb_e"]
# - init condn for some design vars
myones = np.ones(2)
prob["L_h1"] = 1.0
prob["L_12"] = 1.2
prob["lss_diameter"] = 3.0 * myones #* 2
prob["lss_wall_thickness"] = 0.1 * myones #* 2

prob["nose_diameter"] = 2.2 * myones #* 2
prob["nose_wall_thickness"] = 0.1 * myones #* 2

prob["L_generator"] = float(basecaseDict["drivese.L_generator"])  # core length
prob["generator_mass_user"] = float(basecaseDict["drivese.generator_mass"])
prob["generator_radius_user"] = float(basecaseDict["drivese.R_generator"]) # air gap radius

prob["access_diameter"] = float(basecaseDict["drivese.access_diameter"])

prob["bedplate_wall_thickness"] = 0.05 * np.ones(4) # same mass (as report): use 0.0925

prob["yaw_system_mass_user"] = 0.0 # report = 100e3

prob["shaft_deflection_allowable"] = 1e-4
prob["shaft_angle_allowable"] = 1e-3
prob["stator_deflection_allowable"] = 1e-2 #(def: 1e-4 m; 1e-2)
prob["stator_angle_allowable"] = 1e-1 #(def: 1e-3 deg; 1e-1)
# ----

# Material properties (4 materials defined, cf. "n_mat"=4)
prob["E_mat"] = np.c_[200e9 * np.ones(3), 205e9 * np.ones(3), 118e9 * np.ones(3), [4.46e10, 1.7e10, 1.67e10]].T
# - (v, note) these would be  -np.c_-> (4,3) -.T-> (3,4) array
prob["G_mat"] = np.c_[79.3e9 * np.ones(3), 80e9 * np.ones(3), 47.6e9 * np.ones(3), [3.27e9, 3.48e9, 3.5e9]].T
prob["Xt_mat"] = np.c_[450e6 * np.ones(3), 814e6 * np.ones(3), 310e6 * np.ones(3), [6.092e8, 3.81e7, 1.529e7]].T
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
# ----

# %%
# ### Print inputs and outputs to the model `Problem`

print("\n=== All needed inputs to the model ===\n")
# for name, meta in prob.model.list_inputs(out_stream=None, val=False):
#     print(name)
prob.model.list_inputs();

print("\n=== All outputs from the model ===\n")
# for name, meta in prob.model.list_outputs(out_stream=None, val=False):
#     print(name)
prob.model.list_outputs();

#%%
# Run the analysis or optimization
# ---- time it ;)
t0 = time.time()

if opt_flag:
    # Run GBO or DOE
    prob.model.approx_totals() # TODO.
    prob.run_driver()
    status_optim = prob.driver.get_exit_status()
else:
    prob.run_model()
    status_optim = 'ANALYSIS'

# ---- time it ;)
t1 = time.time()
print(" - ",status_optim,": WISDEM run completed in,", (t1-t0)/60, "minutes")
# ----

#%%
if flag_save_new_data: save_data(loc_save_data, prob)

#%%
# Results to match
print(" ---------- Results to match, from report ----------")
print(" - Rotor nacelle assembly mass:  1,017 t")
print(" - Annual energy production:     77.4 GWh")
print(" - Bedplate mass:                70,329 kg")
print(" ---------------------------------------------------")

#%%
# Print the results
print("F_aero_hub:")
print(" ", prob["F_aero_hub"] )
print("M_aero_hub:")
print(" ", prob["M_aero_hub"], "\n" )

print("LSS desvars:")
print(" ", prob["L_h1"], prob["L_12"], prob["lss_diameter"], prob["lss_wall_thickness"] )
print("Bedplate wall thickness:")
print(" ", prob["bedplate_wall_thickness"])
print(" ")
print("F_mb*:")
print(" ", prob["F_mb1"], prob["F_mb2"] )
print("M_mb*:")
print(" ", prob["M_mb1"], prob["M_mb2"] )

print("\n--- constr_ max ---")
if doMBfls:
    print("- constr_L10_mb(1,2):", prob["constr_L10_mb1"], prob["constr_L10_mb2"] )
print("- lss: ", np.max(prob["constr_lss_vonmises"]) )
print("- bedplate: ", np.max(prob["constr_bedplate_vonmises"]) )
print("- defl mb1: ", np.max(prob["constr_mb1_defl"]) )
print("- defl mb2: ", np.max(prob["constr_mb2_defl"]) )
print("- constr_Lh1_MB1fw: ", prob["constr_Lh1_MB1fw"] )
print("- constr_L12_MBsFW: ", prob["constr_L12_MBsFW"] )

print("- constr_shaft_deflection:", prob["constr_shaft_deflection"])
print("- constr_shaft_angle:", prob["constr_shaft_angle"])
print("- constr_stator_deflection:", prob["constr_stator_deflection"])
print("- constr_stator_angle:", prob["constr_stator_angle"])
print("- constr_hub_diameter:", prob["constr_hub_diameter"])
print("- constr_length:", prob["constr_length"])
print("- constr_height:", prob["constr_height"])
print("- constr_access:", prob["constr_access"])
print("- constr_ecc:", prob["constr_ecc"])

#
print("")
print("Masses of drivetrain components")
print(" - lss mass:", prob["lss_mass"][0] )
print(" - nose-turret mass:", prob["nose_mass"][0] )
mb1_mass = prob["mb1_mass"][0]
mb2_mass = prob["mb2_mass"][0]
print(f" - mb masses = {mb1_mass+mb2_mass}; mb1 = {mb1_mass}, mb2 = {mb2_mass}")
print(f" - MSA mass: {prob["msa_mass"]}")
print(" - generator mass:", prob["generator_mass"][0] )
print(" - bedplate mass: ", prob["bedplate_mass"][0] )
print(" - misc. components: ", (prob["hvac_mass"][0]+prob["platform_mass"][0]+prob["cover_mass"][0]) )
print(" - yaw system mass: ", prob["yaw_mass"][0] )
print(" - nacelle_mass:", prob["nacelle_mass"][0] )
print(f" - nacelle cm: {prob["nacelle_cm"]}")

print("\n--- RNA properties ---")
print(f" - RNA mass: {prob["rna_mass"]}") # drivese.rna_mass
print(f" - RNA cm: {prob["rna_cm"]}") # drivese.rna_cm
print(f" - RNA MoI: {prob["rna_I_TT"]}") # drivese.rna_I_TT
#
print("\nTower-top / drivetrain bedplate base loads:")
print(" - base_F: ", prob['base_F']) # drivese.base_F
print(" - base_M: ", prob['base_M']) # drivese.base_M
# -----------------------------------------------------------------------

# %%
# Save rna properties into `yaml` file for next tower optimization
if flag_save_RNAprops4tower:
    utilsDT.write_yaml_of_drivetrain_properties(
        prob, loc_save_RNAprops4tower, direct=True )
# ===============================================================
#%%
if plot_cases:
    # save plot loc
    loc_save_img = None
    if save_new_plot:
        loc_save_img = os.path.join( results_path,
                        "iea15DD_compare_mass"+suffix+".pdf" )
    # plot via func
    utilsDT.plot_drivetrain_mass_comparison(
        loc_save_data+".csv", os.path.join(results_path, "iea_report_DT.csv"),
        m4w_label="IEA 15MW (UN)", iea_label="IEA 15MW (report)",
        flag_WTnamespace=False, loc_save_img=loc_save_img )
# %%
