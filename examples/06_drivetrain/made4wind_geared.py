##(v) MADE4WIND FIRST DRIVETRAIN ITERATION (ULS)
#!/usr/bin/env python3

# references
# 1. 2020_Wang_NTNU - on design modelling and analysis of 10MW
# 2. IEA 15MW=baseline
# 3. Task2.1

# Import needed libraries
import numpy as np
import openmdao.api as om

from wisdem.commonse.fileIO import save_data
from wisdem.drivetrainse.drivetrain import DrivetrainSE

opt_flag = False
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
    prob.driver.options["maxiter"] = 5 * 1

    # Add objective
    prob.model.add_objective("nacelle_mass", scaler=1e-6)

    # Add design variables, in this case the drivetrain diameters and wall thicknesses
    #prob.model.add_design_var("hub_diameter", lower=2.0, upper=5.0)

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
    #prob.model.add_constraint("constr_hub_diameter", lower=0.0)
    # 4. target overhang and hub height
    prob.model.add_constraint("constr_length", lower=0.0)
    prob.model.add_constraint("constr_height", lower=0.0)
    # ---


# Set up the OpenMDAO problem
prob.setup()
# ---

# Set the high-level input values
# - ref.2
prob.set_val("machine_rating", 15.0, units="MW")
prob["upwind"] = True
prob["n_blades"] = 3
prob["rotor_diameter"] = 240.0
prob["D_top"] = 6.5 #tower top diameter
prob["minimum_rpm"] = 5.0 
prob["rated_rpm"] = 7.56 

# Loading from rotor
#(v) TODO: actual ULS loads, not copied from drivetrain_direct
prob["F_aero_hub"] = np.array([2517580.0, -27669.0, 3204.0]).reshape((3, 1))
prob["M_aero_hub"] = np.array([21030561.0, 7414045.0, 1450946.0]).reshape((3, 1))
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

prob["bear1.bearing_type"] = "SRB" # 1. fixed MB; default "CARB"
prob["bear2.bearing_type"] = "CARB" # 2. floating MB; default "SRB"
# - init condn for some design vars
prob["bear1.D_shaft"] = 2.2
prob["bear2.D_shaft"] = 2.2

prob["L_h1"] = 1.0
prob["L_12"] = 1.2
myones = np.ones(2)
prob["lss_diameter"] = 1.0 * myones
prob["lss_wall_thickness"] = 0.288 * myones

prob["L_gearbox"] = 1.5
prob["planet_numbers"] = np.array([5, 3, 0]) #ref.1
prob["gear_configuration"] = "eep"
prob["gear_ratio"] = 50 #.039

prob["L_hss"] = 1.5
prob["hss_diameter"] = 0.5 * myones
prob["hss_wall_thickness"] = 0.1 * myones
prob["L_generator"] = 2.15

prob["overhang"] = 11.35 #ref.2
prob["tilt"] = 5.0 #ref.3
D_hub2flange = 0.358 #ref.2: derive hub height = 5.95522 m
L2n = 0.9
prob["drive_height"] = 4.875 + ( np.sin(prob["tilt"]*np.pi/180)*( L2n+prob["L_12"]+prob["L_h1"]+D_hub2flange+(prob["hub_diameter"]*np.sqrt(3/4)) ) ) # = 5.95522 m

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

# Run the analysis or optimization
if opt_flag:
    prob.model.approx_totals()
    prob.run_driver()
else:
    prob.run_model()
save_data("drivetrain_example", prob)
# ---

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

# OUTPUT
"""
Iteration limit reached    (Exit mode 9)
            Current function value: 0.3937423644722116
            Iterations: 5
            Function evaluations: 25
            Gradient evaluations: 5
Optimization FAILED.
Iteration limit reached
-----------------------------------
nacelle_mass: [393742.36447221]

hub_diameter: [7.94]
L_h1: [0.1048798]
L_12: [0.74098698]
L_lss: [0.94586677]
lss_diameter: [0.64090397 1.87807032]
lss_wall_thickness: [0.28772139 0.28798542]
L_hss: [3.98990285]
hss_diameter: [0.5 0.5]
hss_wall_thickness: [0.0999969 0.0999969]
L_generator: [2.15]
L_gearbox: [3.6]
L_bedplate: [14.6]
H_bedplate: [4.33666551]
bedplate_web_thickness: [0.10138695]
bedplate_flange_thickness: [0.10080672]
bedplate_flange_width: [0.52487507]

constr_lss_vonmises: [0.00194033 0.08606653 0.03097886 0.03872102]
constr_hss_vonmises: [0.0071324  0.00011819]
constr_bedplate_vonmises: [1.74351020e-03 1.73956959e-02 1.79675656e-02 1.20682608e-01
 9.40095833e-02 7.48137812e-02 7.40222255e-02 6.67264544e-02
 6.58290779e-02 8.42461461e-05 2.43365707e-08 1.74350991e-03
 1.74034081e-02 1.79773974e-02 1.20886902e-01 9.42931242e-02
 7.47588820e-02 7.39599094e-02 6.67264041e-02 6.58291202e-02
 8.42235064e-05 4.86680463e-08]
constr_mb1_defl: [-0.02343549]
constr_mb2_defl: [-0.00259453]
constr_shaft_deflection: [0.84049439]
constr_shaft_angle: [3.39296474e-06]
constr_stator_deflection: [21.40211536]
constr_stator_angle: [-0.01783059]
constr_hub_diameter: [0.73466864]
constr_length: [-8.24980972e-10]
constr_height: [4.33666551]

planet_numbers: [5 3 0]
stage_ratios: [3.6849891 3.6849891 3.6849891]
"""