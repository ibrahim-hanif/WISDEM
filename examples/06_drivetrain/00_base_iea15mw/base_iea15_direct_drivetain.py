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

from wisdem.commonse.fileIO import save_data, load_data
from wisdem.drivetrainse.drivetrain import DrivetrainSE, DrivetrainSE_M4W
from wisdem.commonse.utilities import load_all_mat_to_dict, pdf_norm_int_using_cdf
import Drive4Wind.utilities.utilities_drivetrain as utilsDT

#%%
# ### Define flags
suffix = "_old_loads"
# information
# 1. '_old_loads':  old hub loads from felix' openfast (wrong) model
# 2. '_sima_loads': sima loads from seraj's sima (correct) model

opt_flag = True
opt_hub = False # (def: False) if to optimize hub, its Compn incl if dohub
flag_save_new_data = False
load_from_saved_data = True
flag_save_RNAprops4tower = True

# Loading `openFAST` hub loads from a saved file
part_loads = True 
dir_loads = "M:\\Vasudev_Gupta\\outputs_mainshaft_loads"
loc_all_loads_mat_file = os.path.join(dir_loads, "hub_loads_M4w.mat")
if suffix == "_sima_loads":
    loc_all_loads_mat_file = "C://SIMA_M4W_loads//all_main_shaft_loads.mat" # TODO: sima loads

# - results main dir
results_dir = "results"
script_dir = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(script_dir, results_dir)
os.makedirs(results_path, exist_ok=True)

loc_save_data = os.path.join(results_path, "m4w_base_case_DT"+suffix)
if flag_save_RNAprops4tower:
    loc_save_RNAprops4tower = os.path.join(
        results_path, "RNA_props_model_for_tower"+suffix+".yaml")

# %% [markdown]
# ### Defining results directory and files
#%% Loading `openFAST` hub loads from a saved file
if part_loads: # define paths
    S_all, keys_all = load_all_mat_to_dict(loc_all_loads_mat_file)

# Wind speed and probabilies: auto parse loads dict
ws = S_all["mean_wind_speed"][0,:].tolist()
pdf_ws = pdf_norm_int_using_cdf(ws).tolist()
# ----
#%%
# Set input options (modeling options dictionary)
opts = {}

opts["WISDEM"] = {}
opts["WISDEM"]["n_dlc"] = 1
opts["WISDEM"]["DriveSE"] = {}
# NOTE "hub": 'Hub_System' component are NOT included in the 'DrivetrainSE_M4W' component 
opts["WISDEM"]["DriveSE"]["hub"] = {}
opts["WISDEM"]["DriveSE"]["hub"]["hub_gamma"] = 2.0
opts["WISDEM"]["DriveSE"]["hub"]["spinner_gamma"] = 1.5

direct = opts["WISDEM"]["DriveSE"]["direct"] = True
opts["WISDEM"]["DriveSE"]["gearbox_torque_density"] = 0.0 # False =(GB  optim, in-capabale)

opts["WISDEM"]["DriveSE"]["gamma_f"] = 1.35 #IEC-1, 7.6.2.2a, pg.57
opts["WISDEM"]["DriveSE"]["gamma_m"] = 1.3  #IEC-1, 7.6.2.4, pg.59
opts["WISDEM"]["DriveSE"]["gamma_n"] = 1.0  #IEC-1, 7.6.1.3, pg.55
opts["WISDEM"]["DriveSE"]["own_hub_loads"] = True
# used as: gamma = gamma_f * gamma_m * gamma_n (within TODO)

opts["WISDEM"]["RotorSE"] = {}
opts["WISDEM"]["RotorSE"]["n_pc"] = 20 #cf. RPM_Input in drive_components.py
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
opts["DLC_driver"]["DLCs"][0]["wind_speed"] = ws
opts["DLC_driver"]["DLCs"][0]["probabilities"] = pdf_ws
# ---

#%%
# Initialize OpenMDAO problem
prob = om.Problem(reports=False)
prob.model = DrivetrainSE_M4W(modeling_options=opts)
# ---

#%%
# If performing optimization, set up the optimizer and problem formulation
if opt_flag:
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


# Set up the OpenMDAO problem
prob.setup()
# ----
#%% overwrite variables from saved data
if load_from_saved_data:
    print(" loading prob vars from saved csv")
    prob = load_data( loc_save_data+".csv", prob )
#%%
# Set high-level input values (that desc the turbine)
prob.set_val("machine_rating", 15.0, units="MW")
prob["upwind"] = True
n_blades = 3
prob["rotor_diameter"] = 241.35064632 # latest: 241.35064632; old: 240.0
prob["D_top"] = 6.5 #tower top diameter
prob["minimum_rpm"] = 5.0
prob["rated_rpm"] = 7.56
prob["rated_torque"] = 21.3 * 1e6 # 19947034.78543754
prob["overhang"] = 12.0313 # 11.35
prob["drive_height"] = 5.614
prob["tilt"] = 6.0

# Loading from rotor
# prob["F_aero_hub"] = np.array([2517580.0, -27669.0, 3204.0]).reshape((3, 1))
# prob["M_aero_hub"] = np.array([21030561.0, 7414045.0, 1450946.0]).reshape((3, 1))
prob['F_aero_hub'] = np.array( [S_all['Fx_max'], S_all['Fy_max'], S_all['Fz_max']] ).reshape((3, 1))
prob['M_aero_hub'] = np.array( [S_all['Mx_max'], S_all['My_max'], S_all['Mz_max']] ).reshape((3, 1))
# ----

# Blade properties and hub design options
# --- NOTE: copied from M4W_base_case_driver's NOoptim csv
prob["hub_diameter"] = 7.94
prob["blades_cm"] = 2.1853055315151138
blade_mass = 68233.0936092383
prob["blades_mass"] = n_blades * blade_mass
prob["blades_I"] = np.r_[348506332.76071006, 174253166.38035503, 174253166.38035503, 0.0, 0.0, 0.0]
prob["pitch_system.BRFM"] = 117585772.28432259
prob["pitch_system_scaling_factor"] = 0.75
prob["blade_root_diameter"] = 5.20
prob["flange_t2shell_t"] = 6.0
prob["flange_OD2hub_D"] = 0.6
prob["flange_ID2flange_OD"] = 0.8 # ? ----+ down: TODO
prob["hub_in2out_circ"] = 1.2
prob["hub_stress_concentration"] = 3.0
prob["n_front_brackets"] = 5
prob["n_rear_brackets"] = 5
prob["clearance_hub_spinner"] = 0.5
prob["spin_hole_incr"] = 1.2
prob["spinner_gust_ws"] = 70.0
# ----

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

prob["L_generator"] = 2.15  # core length
prob["generator_mass_user"] = 368839.0
prob["generator_radius_user"] = 5.309 # air gap radius

prob["access_diameter"] = 2.0

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
    utilsDT.write_yaml_of_drivetrain_properties( prob, loc_save_RNAprops4tower, direct=True )
# ===============================================================

# %%
# TODO: change nacelle mass compr plot to read saved csv for iea15