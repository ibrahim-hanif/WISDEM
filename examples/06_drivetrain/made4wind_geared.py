#%% [markdown] MADE4WIND FIRST DRIVETRAIN ITERATION (ULS)
#!/usr/bin/env python3

# references
# 1. 2020_Wang_NTNU - on design modelling and analysis of 10MW
# 2. IEA 15MW=baseline
# 3. Task2.1
#%%
# Import needed libraries
import numpy as np
import openmdao.api as om

from wisdem.commonse.fileIO import save_data
from wisdem.drivetrainse.drivetrain import DrivetrainSE

#%%[markdown]
# use Recorder to track optimzer progress (format if .sql)
# generte plots from it by running: (#433 github: https://github.com/WISDEM/WISDEM/issues/433#issuecomment-1510475996)
"""from wisdem.glue_code.gc_RunTools import PlotRecorder
import wisdem.inputs as sch
import os

mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
fname_analysis_options = mydir + os.sep + "analysis_options.yaml"
analysis_options = sch.load_analysis_yaml(fname_analysis_options)
wt_opt = om.Problem(model=PlotRecorder(opt_options=analysis_options))
wt_opt.setup(derivatives=False)
wt_opt.run_model()"""

#%%
opt_flag = True # False: analysis; True: optimization
# ---

# Set input options
opt = {}
opt["WISDEM"] = {}
opt["WISDEM"]["n_dlc"] = 1

opt["WISDEM"]["DriveSE"] = {}
opt["WISDEM"]["DriveSE"]["direct"] = False
opt["WISDEM"]["DriveSE"]["use_gb_torque_density"] = True

opt["WISDEM"]["DriveSE"]["hub"] = {}
opt["WISDEM"]["DriveSE"]["hub"]["hub_gamma"] = 2.0
opt["WISDEM"]["DriveSE"]["hub"]["spinner_gamma"] = 1.5
opt["WISDEM"]["DriveSE"]["gamma_f"] = 1.35 #IEC-1, 7.6.2.2a, pg.57
opt["WISDEM"]["DriveSE"]["gamma_m"] = 1.3  #IEC-1, 7.6.2.4, pg.59
opt["WISDEM"]["DriveSE"]["gamma_n"] = 1.0  #IEC-1, 7.6.1.3, pg.55

opt["WISDEM"]["RotorSE"] = {}
opt["WISDEM"]["RotorSE"]["n_pc"] = 20

opt["materials"] = {}
opt["materials"]["n_mat"] = 4

opt["flags"] = {}
opt["flags"]["generator"] = False #run detailed generator design?
# ---

#%%
# Initialize OpenMDAO problem
prob = om.Problem(reports=False)
prob.model = DrivetrainSE(modeling_options=opt)
# ---

# If performing optimization, set up the optimizer and problem formulation
if opt_flag:
    # Choose the optimizer to use
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"
    prob.driver.options["tol"] = 1e-2
    prob.driver.options["maxiter"] = 5 * 100
    prob.driver.options["disp"] = True
    #prob.driver.options["debug_print"] = ["desvars", "ln_cons", "nl_cons", "objs"]

    # Add objective
    prob.model.add_objective("nacelle_mass", scaler=1e-6)

    # Add design variables, in this case the drivetrain diameters and wall thicknesses
    # prob.model.add_design_var("hub_diameter", lower=2.0, upper=5.0) #TODO: comment out for no hub optimization

    prob.model.add_design_var("L_12", lower=0.1, upper=5.0)
    prob.model.add_design_var("L_h1", lower=0.1, upper=5.0)
    prob.model.add_design_var("lss_diameter", lower=0.5, upper=6.0)
    prob.model.add_design_var("lss_wall_thickness", lower=4e-3, upper=5e-1, ref=1e-2)

    prob.model.add_design_var("L_hss", lower=0.1, upper=5.0)
    prob.model.add_design_var("hss_diameter", lower=0.5, upper=6.0)
    prob.model.add_design_var("hss_wall_thickness", lower=4e-3, upper=5e-1, ref=1e-2)
    
    prob.model.add_design_var("bedplate_web_thickness", lower=4e-3, upper=5e-1, ref=1e-2)
    prob.model.add_design_var("bedplate_flange_thickness", lower=4e-3, upper=5e-1, ref=1e-2)
    prob.model.add_design_var("bedplate_flange_width", lower=0.1, upper=2.0)

    # Add constraints on the tower design
    # 1. von Mises stress util
    prob.model.add_constraint("constr_lss_vonmises", upper=1.0)
    prob.model.add_constraint("constr_hss_vonmises", upper=1.0)
    prob.model.add_constraint("constr_bedplate_vonmises", upper=1.0)
    # 2. main bearing defl (allowed, max, as an angle)
    prob.model.add_constraint("constr_mb1_defl", upper=1.0)
    prob.model.add_constraint("constr_mb2_defl", upper=1.0)
    prob.model.add_constraint("constr_shaft_deflection", upper=1.0)
    prob.model.add_constraint("constr_shaft_angle", upper=1.0)
    prob.model.add_constraint("constr_stator_deflection", upper=1.0)
    prob.model.add_constraint("constr_stator_angle", upper=1.0)
    # 3. hub dia to accom. blades' roots
    # prob.model.add_constraint("constr_hub_diameter", lower=0.0) #TODO: comment out for no hub optimization
    # 4. target overhang and hub height
    prob.model.add_constraint("constr_length", lower=0.0)
    prob.model.add_constraint("constr_height", lower=0.0)
    # ---

#%%
# Set up the OpenMDAO problem
prob.setup()
# ---

# %%
print("All needed inputs to the model:\n")
for name, meta in prob.model.list_inputs(out_stream=None, val=False):
    print(name)

print("All outputs from the model:\n")
for name, meta in prob.model.list_outputs(out_stream=None, val=False):
    print(name)

#%%
# Set the high-level input values
# - ref.2
prob.set_val("machine_rating", 15.0, units="MW")
prob["upwind"] = True
prob["n_blades"] = 3
D_rotor = prob["rotor_diameter"] = 240.0
prob["D_top"] = 6.5 #tower top diameter
prob["minimum_rpm"] = 5.0 
prob["rated_rpm"] = 7.56 

# Loading from rotor
#(v) actual ULS loads (source: copied felix main_shafft_siz code 20251031)
# - NOTE: these are predscribed 50-yr extremes from extr DLCs (5.1,6.1,6.3) 
prob["F_aero_hub"] = np.array([5.3995*1e6, 1.3697*1e6, 5.5742*1e6]).reshape((3, 1))
prob["M_aero_hub"] = np.array([5.2515*1e7, 1.0747*1e8, 9.9481*1e7]).reshape((3, 1))
# ---

# Blade properties and hub design options
# -(v) copied from drivetrain_direct (IEA-15MW = ref)
prob["blades_cm"] = 2.46175
prob["blade_mass"] = 65252.0
prob["blades_mass"] = 3 * prob["blade_mass"]
prob["blades_I"] = np.r_[3.48453857e+08, 1.74226928e+08, 1.74226928e+08, np.zeros(3)]
prob["pitch_system.BRFM"] = 26648449.0
prob["pitch_system_scaling_factor"] = 0.75
prob["blade_root_diameter"] = 5.2
prob["flange_t2shell_t"] = 6.0
prob["flange_OD2hub_D"] = 0.6
prob["flange_ID2flange_OD"] = 0.8
prob["hub_in2out_circ"] = 1.2
prob["hub_stress_concentration"] = 3.0
prob["n_front_brackets"] = 5
prob["n_rear_brackets"] = 5
prob["clearance_hub_spinner"] = 0.5
prob["spin_hole_incr"] = 1.2
prob["spinner_gust_ws"] = 70.0
prob["hub_diameter"] = 7.94
# ---

# Drivetrain configuration and sizing inputs
#(v) notes: 1. hub flange length = 0.358m, 

prob["bear1.bearing_type"] = "CRB" # 1. fixed MB; default "CARB"
prob["bear2.bearing_type"] = "TRB2" # 2. floating MB; default "SRB"
# - init condn for some design vars
prob["bear1.D_shaft"] = 4.0 #2.2
prob["bear2.D_shaft"] = 3.2 #2.2

prob["L_h1"] = 4.25 #1.0; cf. L_rb in main_shaft_sizing code
prob["L_12"] = 7.1 #1.2
myones = np.ones(2)
prob["lss_diameter"] = 4.0 * myones #1.0
prob["lss_wall_thickness"] = 0.288 * myones
# Gearbox
prob["L_gearbox"] = 0.0 #1.5, or 0.0 for default scaling wrt. D_rotor
prob["planet_numbers"] = np.array([4, 5, 4]) #ref.1
prob["gear_configuration"] = "eee"
prob["gear_ratio"] = 50 #.039

prob["L_hss"] = 1.5
prob["hss_diameter"] = 0.5 * myones
prob["hss_wall_thickness"] = 0.1 * myones
prob["L_generator"] = 2.15

prob["overhang"] = 11.35 #ref.2
prob["tilt"] = 6.0 #ref.3

# 'drive_height' : derive from the high-level inputs (output= 5.95522 m)
# - needed by layout.py (line 122)
L_fl = 0.358 #ref.2: Hub flange length 
# L_fl = 0.3*(D_rotor/100)**2 - 0.1*(D_rotor/100)+0.4 #1.888; cf. 2015_Guo-Analytical
L2n = 0.9 #ref.2: Distance of downwind bearing from bedplate flange
L_lss = prob["L_h1"]+prob["L_12"]+L2n
H_nose = 4.875 #ref.2: Nose height (from tower top to bottom of bedplate flange)
prob["drive_height"] = H_nose + ( np.sin(np.deg2rad(prob["tilt"]))*( (prob["hub_diameter"]*np.sqrt(3/4))+L_fl+L_lss ) )
#print( prob["drive_height"] ) # = 5.95522 m

prob["bedplate_web_thickness"] = 0.1
prob["bedplate_flange_thickness"] = 0.1
prob["bedplate_flange_width"] = 1.0

prob["shaft_deflection_allowable"] = 1e-4
prob["shaft_angle_allowable"] = 1e-3
prob["stator_deflection_allowable"] = 1e-4
prob["stator_angle_allowable"] = 1e-3
# ---

# Material properties (4 materials defined, cf. "n_mat"=4)
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

#%%
# Run the analysis or optimization
if opt_flag:
    prob.model.approx_totals()
    prob.run_driver()
else:
    prob.run_model()
save_data("drivetrain_example", prob)
# ---

#%%
# Display all inputs and outputs
# prob.model.list_inputs(units=True)
# prob.model.list_outputs(units=True)

# Print out the objective, design variables and constraints
print("nacelle_mass:", prob["nacelle_mass"])
print("")
print("hub_diameter:", prob["hub_diameter"])
print("hub_system_mass:", prob["hub_system_mass"])
print("hub_system_cm:", prob["hub_system_cm"])
print("hub_system_I:", prob["hub_system_I"])

print("L_h1:", prob["L_h1"])
print("L_12:", prob["L_12"])
print("L_lss:", prob["L_lss"])
print("lss_diameter:", prob["lss_diameter"])
print("lss_wall_thickness:", prob["lss_wall_thickness"])

print("L_hss:", prob["L_hss"])
print("hss_diameter:", prob["hss_diameter"])
print("hss_wall_thickness:", prob["hss_wall_thickness"])

print("L_generator:", prob["L_generator"])
print("L_gearbox:", prob["L_gearbox"])
print("L_bedplate:", prob["L_bedplate"])
print("H_bedplate:", prob["H_bedplate"])
print("bedplate_web_thickness:", prob["bedplate_web_thickness"])
print("bedplate_flange_thickness:", prob["bedplate_flange_thickness"])
print("bedplate_flange_width:", prob["bedplate_flange_width"])
print("")
print("constr_lss_vonmises:", prob["constr_lss_vonmises"].flatten())
print("constr_hss_vonmises:", prob["constr_hss_vonmises"].flatten())
print("constr_bedplate_vonmises:", prob["constr_bedplate_vonmises"].flatten())
print("constr_mb1_defl:", prob["constr_mb1_defl"])
print("constr_mb2_defl:", prob["constr_mb2_defl"])
print("constr_shaft_deflection:", prob["constr_shaft_deflection"])
print("constr_shaft_angle:", prob["constr_shaft_angle"])
print("constr_stator_deflection:", prob["constr_stator_deflection"])
print("constr_stator_angle:", prob["constr_stator_angle"])
print("constr_hub_diameter:", prob["constr_hub_diameter"])
print("constr_length:", prob["constr_length"])
print("constr_height:", prob["constr_height"])
print("") #(v) cf. drivetrain_example.csv (in WISDEM folder)
print("planet_numbers:", prob["planet_numbers"])
print("stage_ratios:", prob["stage_ratios"])
# ---

#%% [markdown]
# OUTPUT (analysis, opt_flag=False)
"""
nacelle_mass: [387043.7821504]

hub_diameter: [7.94]
hub_system_mass: [62561.91718921]
hub_system_cm: [3.35947759]
hub_system_I: [865503.52531197 567289.77714803 567289.77714803      0.
      0.              0.        ]
L_h1: [1.]
L_12: [1.2]
L_lss: [2.3]
lss_diameter: [1. 1.]
lss_wall_thickness: [0.288 0.288]
L_hss: [1.5]
hss_diameter: [0.5 0.5]
hss_wall_thickness: [0.1 0.1]
L_generator: [2.15]
L_gearbox: [3.6]
L_bedplate: [13.44593603]
H_bedplate: [4.54199758]
bedplate_web_thickness: [0.1]
bedplate_flange_thickness: [0.1]
bedplate_flange_width: [1.]

constr_lss_vonmises: [0.69384564 0.70331889 0.70445963 0.76080822]
constr_hss_vonmises: [0.12442389 0.12340084]
constr_bedplate_vonmises: [2.02772686e-03 1.02684704e-02 2.08612714e-01 2.05787255e-01
 2.08954376e-01 7.34362945e-02 7.30933582e-02 5.86895829e-02
 5.62196427e-02 9.39759135e-04 1.59508718e-08 2.02772645e-03
 1.98698837e-02 3.43068266e-01 3.39593300e-01 3.30182256e-01
 7.33541758e-02 7.26533120e-02 5.87241675e-02 5.62274022e-02
 9.39708888e-04 8.51218697e-09]
constr_mb1_defl: [0.00349732]
constr_mb2_defl: [0.03161487]
constr_shaft_deflection: [2.9008074]
constr_shaft_angle: [1.4814362e-06]
constr_stator_deflection: [26.56707208]
constr_stator_angle: [0.47870511]
constr_hub_diameter: [0.73466864]
constr_length: [-1.15406397]
constr_height: [4.54199758]

planet_numbers: [5 3 0]
stage_ratios: [3.6840315 3.6840315 3.6840315]
"""

# OUTPUT (optimization, opt_flag=True)
"""
Iteration limit reached    (Exit mode 9)
            Current function value: 1.106578994156817
            Iterations: 20
            Function evaluations: 88
            Gradient evaluations: 20
Optimization FAILED.
Iteration limit reached
-----------------------------------
nacelle_mass: [1106578.99415682]

hub_diameter: [7.94]
hub_system_mass: [62561.91718921]
hub_system_cm: [3.35947759]
hub_system_I: [865503.52531197 567289.77714803 567289.77714803      0.
      0.              0.        ]
L_h1: [4.52298871]
L_12: [0.10037979]
L_lss: [4.7233685]
lss_diameter: [3.40346473 0.5       ]
lss_wall_thickness: [0.08443051 0.00406725]
L_hss: [0.21240112]
hss_diameter: [1.41452557 0.6013716 ]
hss_wall_thickness: [0.15552484 0.08437647]
L_generator: [2.15]
L_gearbox: [3.6]
L_bedplate: [14.6]
H_bedplate: [4.49835402]
bedplate_web_thickness: [0.49999999]
bedplate_flange_thickness: [0.5]
bedplate_flange_width: [1.99999999]

constr_lss_vonmises: [0.0952554  0.11900281 0.12256263 0.80382188]
constr_hss_vonmises: [0.01181233 0.03733804]
constr_bedplate_vonmises: [3.80250287e-03 6.27326683e-03 6.53216372e-03 6.78627650e-03
 1.13143726e-02 2.63892498e-02 2.60905752e-02 2.16621628e-02
 2.15152620e-02 5.15622807e-03 2.27789755e-10 5.53003483e-03
 8.00609609e-03 8.24895145e-03 8.50056601e-03 1.30291494e-02
 2.65247052e-02 2.61738414e-02 2.16625610e-02 2.15153930e-02
 5.15622768e-03 1.03360649e-09]
constr_mb1_defl: [-0.00014236]
constr_mb2_defl: [-0.00125425]
constr_shaft_deflection: [0.00183954]
constr_shaft_angle: [8.08601894e-08]
constr_stator_deflection: [2.18040316]
constr_stator_angle: [0.00626111]
constr_hub_diameter: [0.73466864]
constr_length: [-3.7566803e-09]
constr_height: [4.49835402]

planet_numbers: [5 3 0]
stage_ratios: [3.6840315 3.6840315 3.6840315]
"""