##(v) cf. https://wisdem.readthedocs.io/en/master/examples/06_drivetrain/tutorial.html#geared-design
#!/usr/bin/env python3
# Import needed libraries
import numpy as np
import openmdao.api as om

from wisdem.commonse.fileIO import save_data
from wisdem.drivetrainse.drivetrain import DrivetrainSE

opt_flag = True
# ---

# Set input options
opt = {}
opt["WISDEM"] = {}
opt["WISDEM"]["n_dlc"] = 1

opt["WISDEM"]["DriveSE"] = {}
opt["WISDEM"]["DriveSE"]["direct"] = False
opt["WISDEM"]["DriveSE"]["use_gb_torque_density"] = False
opt["WISDEM"]["DriveSE"]["hub"] = {}
opt["WISDEM"]["DriveSE"]["hub"]["hub_gamma"] = 2.0
opt["WISDEM"]["DriveSE"]["hub"]["spinner_gamma"] = 1.5
opt["WISDEM"]["DriveSE"]["gamma_f"] = 1.35
opt["WISDEM"]["DriveSE"]["gamma_m"] = 1.3
opt["WISDEM"]["DriveSE"]["gamma_n"] = 1.0

opt["WISDEM"]["RotorSE"] = {}
opt["WISDEM"]["RotorSE"]["n_pc"] = 20

opt["materials"] = {}
opt["materials"]["n_mat"] = 4

opt["flags"] = {}
opt["flags"]["generator"] = False
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
    prob.model.add_design_var("L_12", lower=0.1, upper=5.0)
    prob.model.add_design_var("L_h1", lower=0.1, upper=5.0)
    prob.model.add_design_var("L_hss", lower=0.1, upper=5.0)
    prob.model.add_design_var("hub_diameter", lower=2.0, upper=5.0)
    prob.model.add_design_var("lss_diameter", lower=0.5, upper=6.0)
    prob.model.add_design_var("lss_wall_thickness", lower=4e-3, upper=5e-1, ref=1e-2)

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
    prob.model.add_constraint("constr_hub_diameter", lower=0.0)
    # 4. target overhand and hub height
    prob.model.add_constraint("constr_length", lower=0.0)
    prob.model.add_constraint("constr_height", lower=0.0)
    # ---


# Set up the OpenMDAO problem
prob.setup()
# ---

# Set the high-level input values
prob.set_val("machine_rating", 5.0, units="MW")
prob["upwind"] = True
prob["n_blades"] = 3
prob["rotor_diameter"] = 126.0
prob["D_top"] = 3.87 #tower top diameter
prob["minimum_rpm"] = 6.9
prob["rated_rpm"] = 12.1

# Loading from rotor
prob["F_hub"] = np.array([1125044.07614847, -7098.0872533, -7022.79756034]).reshape((3, 1))
prob["M_hub"] = np.array([10515165.10636333, 945938.60268626, 1042828.16100417]).reshape((3, 1))
# ---

# Blade properties and hub design options
prob["blade_mass"] = 16403.0
prob["blades_mass"] = 3 * prob["blade_mass"]
prob["pitch_system.BRFM"] = 14239550.0
prob["pitch_system_scaling_factor"] = 0.54
prob["blade_root_diameter"] = 3.542
prob["flange_t2shell_t"] = 4.0
prob["flange_OD2hub_D"] = 0.5
prob["flange_ID2flange_OD"] = 0.8
prob["hub_in2out_circ"] = 1.2
prob["hub_stress_concentration"] = 2.5
prob["n_front_brackets"] = 3
prob["n_rear_brackets"] = 3
prob["clearance_hub_spinner"] = 1.0
prob["spin_hole_incr"] = 1.2
prob["spinner_gust_ws"] = 70.0
prob["hub_diameter"] = 3.0
prob["blades_I"] = np.r_[36494351.0, 17549243.0, 14423664.0, np.zeros(3)]
# ---

# Drivetrain configuration and sizing inputs
prob["bear1.bearing_type"] = "CARB"
prob["bear2.bearing_type"] = "SRB"
# - init condn for some design vars
prob["L_12"] = 1.912 #0.368
prob["L_h1"] = 0.368 #1.912
prob["L_hss"] = 1.5
prob["L_generator"] = 2.0
prob["L_gearbox"] = 1.5
prob["overhang"] = 5.0
prob["drive_height"] = 2.3
prob["tilt"] = 5.0

prob["planet_numbers"] = np.array([3, 3, 0])
prob["gear_configuration"] = "eep"
prob["gear_ratio"] = 96.0

myones = np.ones(2)
prob["lss_diameter"] = 1.0 * myones
prob["hss_diameter"] = 0.5 * myones
prob["lss_wall_thickness"] = 0.288 * myones
prob["hss_wall_thickness"] = 0.1 * myones
prob["bedplate_web_thickness"] = 0.1
prob["bedplate_flange_thickness"] = 0.1
prob["bedplate_flange_width"] = 1.0
prob["bear1.D_shaft"] = 2.2
prob["bear2.D_shaft"] = 2.2
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
print("L_h1:", prob["L_h1"])
print("L_12:", prob["L_12"])
print("L_lss:", prob["L_lss"])
print("L_hss:", prob["L_hss"])
print("L_generator:", prob["L_generator"])
print("L_gearbox:", prob["L_gearbox"])
print("L_bedplate:", prob["L_bedplate"])
print("H_bedplate:", prob["H_bedplate"])
print("hub_diameter:", prob["hub_diameter"])
print("lss_diameter:", prob["lss_diameter"])
print("lss_wall_thickness:", prob["lss_wall_thickness"])
print("hss_diameter:", prob["hss_diameter"])
print("hss_wall_thickness:", prob["hss_wall_thickness"])
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
# test 1: Lh1 = 1.912, L12=0.368
"""
Iteration limit reached    (Exit mode 9)
            Current function value: 0.16922729776973358
            Iterations: 50
            Function evaluations: 287
            Gradient evaluations: 50
Optimization FAILED.
Iteration limit reached
-----------------------------------
nacelle_mass: [169227.29776973]

L_h1: [1.70733376]
L_12: [0.251258]
L_lss: [2.05859176]
L_hss: [1.00003623]
L_generator: [2.]
L_gearbox: [1.89]
L_bedplate: [6.92218637]
H_bedplate: [1.4784902]
hub_diameter: [4.9542797]
lss_diameter: [1.02498861 1.25015858]
lss_wall_thickness: [0.32975047 0.32763904]
hss_diameter: [0.53006557 0.71300914]
hss_wall_thickness: [0.06203973 0.09931631]
bedplate_web_thickness: [0.09883529]
bedplate_flange_thickness: [0.02891908]
bedplate_flange_width: [1.07419507]

constr_lss_vonmises: [0.3121217  0.30743701 0.29627995 0.23238659]
constr_hss_vonmises: [0.02799162 0.01644432]
constr_bedplate_vonmises: [1.95194229e-03 1.11392285e-02 1.02916886e-02 1.22462192e-02
 1.91713986e-02 7.42623387e-02 7.50575872e-02 5.59164734e-02
 5.45388789e-02 1.64972253e-03 2.50047087e-08 1.95196133e-03
 2.33686045e-02 2.42824588e-02 2.60635027e-02 3.22618086e-02
 7.74082344e-02 8.07556137e-02 5.59276880e-02 5.45420272e-02
 1.64973498e-03 1.89446636e-08]
constr_mb1_defl: [0.00200438]
constr_mb2_defl: [0.00024692]
constr_shaft_deflection: [0.07385187]
constr_shaft_angle: [1.45949314e-06]
constr_stator_deflection: [1.0130119]
constr_stator_angle: [0.03543997]
constr_hub_diameter: [0.04634053]
constr_length: [2.45489995]
constr_height: [1.4784902]

planet_numbers: [3 3 0]
stage_ratios: [4.57885697 4.57885697 4.57885697]
"""

# test 2: Lh1 = 0.368, L12=1.912
"""
Values in x were outside bounds during a minimize step, clipping to boundsIteration limit reached    (Exit mode 9)
            Current function value: 0.16102988038039678
            Iterations: 50
            Function evaluations: 392
            Gradient evaluations: 50
Optimization FAILED.
Iteration limit reached
-----------------------------------
nacelle_mass: [161029.8803804]

L_h1: [0.21991087]
L_12: [1.0023408]
L_lss: [1.32225167]
L_hss: [1.72402842]
L_generator: [2.]
L_gearbox: [1.89]
L_bedplate: [6.90988544]
H_bedplate: [1.47933922]
hub_diameter: [4.95949286]
lss_diameter: [1.65597109 0.61935843]
lss_wall_thickness: [0.28534298 0.28122109]
hss_diameter: [0.62791059 0.52248553]
hss_wall_thickness: [0.08671673 0.1061336 ]
bedplate_web_thickness: [0.0861382]
bedplate_flange_thickness: [0.10692674]
bedplate_flange_width: [0.28581524]

constr_lss_vonmises: [0.0964616  0.14978977 0.36995273 0.9816    ]
constr_hss_vonmises: [0.02218026 0.02513031]
constr_bedplate_vonmises: [1.89816775e-03 1.75669646e-02 1.29498299e-02 1.69039905e-02
 2.75258958e-02 7.73470406e-02 7.87371029e-02 6.39350419e-02
 5.57029879e-02 1.98980029e-04 1.25272740e-08 1.89824780e-03
 2.83397442e-02 2.86499611e-02 3.30177300e-02 4.19004748e-02
 8.31997010e-02 8.84133553e-02 6.40281936e-02 5.57220254e-02
 1.99004880e-04 3.66976014e-08]
constr_mb1_defl: [0.00048895]
constr_mb2_defl: [0.00034595]
constr_shaft_deflection: [0.80148367]
constr_shaft_angle: [4.72027239e-07]
constr_stator_deflection: [1.78941947]
constr_stator_angle: [0.04238008]
constr_hub_diameter: [0.05155369]
constr_length: [2.44519569]
constr_height: [1.47933922]

planet_numbers: [3 3 0]
stage_ratios: [4.57885697 4.57885697 4.57885697]
"""