##(v) cf. https://wisdem.readthedocs.io/en/master/examples/06_drivetrain/tutorial.html#geared-design
#!/usr/bin/env python3
# Import needed libraries
import numpy as np
import openmdao.api as om

from wisdem.commonse.fileIO import save_data
from wisdem.drivetrainse.drivetrain import DrivetrainSE

opt_flag = True
# ---

# Set input "modelling" options (dictionary; selective for the drivetrain model optimization)
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
    prob.driver.options["disp"] = True
    #prob.driver.options["debug_print"] = ["desvars", "ln_cons", "nl_cons", "objs"]

    # Add objective
    prob.model.add_objective("nacelle_mass", scaler=1e-6)

    # Add design variables, in this case the drivetrain diameters and wall thicknesses
    prob.model.add_design_var("hub_diameter", lower=2.0, upper=5.0)

    prob.model.add_design_var("L_h1", lower=0.1, upper=5.0)
    prob.model.add_design_var("L_12", lower=0.1, upper=5.0)
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
    prob.model.add_constraint("constr_hub_diameter", lower=0.0)
    # 4. target overhang and hub height
    prob.model.add_constraint("constr_length", lower=0.0)
    prob.model.add_constraint("constr_height", lower=0.0)
    # ---


# Set up the OpenMDAO problem
prob.setup()
#prob.set_solver_print(level=2)
# ---

# Set the high-level input values
prob.set_val("machine_rating", 5.0, units="MW")
prob["upwind"] = True
prob["n_blades"] = 3
prob["rotor_diameter"] = 126.0
prob["D_top"] = 3.87 #tower top diameter
prob["minimum_rpm"] = 6.9
prob["rated_rpm"] = 12.1
prob["rated_torque"] = 4308926.79641971
prob["overhang"] = 5.0
prob["drive_height"] = 2.3 #(v) report: 2.4 m
prob["tilt"] = 5.0

# Loading from rotor
prob["F_aero_hub"] = np.array([1125044.07614847, -7098.0872533, -7022.79756034]).reshape((3, 1))
prob["M_aero_hub"] = np.array([10515165.10636333, 945938.60268626, 1042828.16100417]).reshape((3, 1))
# ---

# Blade properties and hub design options
prob["blades_cm"] = 0.99847077
prob["blade_mass"] = 16403.0 #(v) report: hub mass = 56,780 kg
prob["blades_mass"] = 3 * prob["blade_mass"]
prob["blades_I"] = np.r_[36494351.0, 17549243.0, 14423664.0, np.zeros(3)]
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
# ---

# Drivetrain configuration and sizing inputs
prob["bear1.bearing_type"] = "SRB" # 1. fixed MB; default "CARB"
prob["bear2.bearing_type"] = "CARB" # 2. floating MB; default "SRB"
# - init condn for some design vars
prob["bear1.D_shaft"] = 2.2
prob["bear2.D_shaft"] = 2.2

prob["L_h1"] = 1.912 #(v) report: hub center to MB, not flange! also RWT =3pt suspension
prob["L_12"] = 0.368
myones = np.ones(2)
prob["lss_diameter"] = 1.0 * myones
prob["lss_wall_thickness"] = 0.288 * myones

prob["L_gearbox"] = 1.5
prob["planet_numbers"] = np.array([3, 3, 0])
prob["gear_configuration"] = "eep"
prob["gear_ratio"] = 96.0 #(v) report = 97

prob["L_hss"] = 1.5
prob["L_generator"] = 2.0
prob["hss_diameter"] = 0.5 * myones
prob["hss_wall_thickness"] = 0.1 * myones

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
# test 1: maxiter = 100 (L_h1 = 1.912, L_12 = 0.368)
"""
Optimization terminated successfully    (Exit mode 0)
            Current function value: 0.2800365326879702
            Iterations: 21
            Function evaluations: 71
            Gradient evaluations: 21
Optimization Complete
-----------------------------------
nacelle_mass: [280036.53268797]

L_h1: [0.98700811]
L_12: [0.14248346]
L_lss: [1.22949157]
L_hss: [0.10189485]
L_generator: [2.]
L_gearbox: [1.89]
L_bedplate: [7.69200422]
H_bedplate: [1.62703683]
hub_diameter: [5.]
lss_diameter: [0.84109957 0.68091051]
lss_wall_thickness: [0.28202845 0.27139887]
hss_diameter: [0.93633721 1.07883486]
hss_wall_thickness: [0.15150088 0.1424637 ]
bedplate_web_thickness: [0.30592616]
bedplate_flange_thickness: [0.1786935]
bedplate_flange_width: [1.35453364]

constr_lss_vonmises: [0.59788466 0.62962778 0.65081657 0.84502496]
constr_hss_vonmises: [0.00749502 0.00423346]
constr_bedplate_vonmises: [5.25284543e-03 8.14276095e-03 8.42624564e-03 8.61197297e-03
 1.23542396e-02 3.72733017e-02 3.68210304e-02 2.59750009e-02
 2.56966517e-02 1.18267485e-03 1.73289935e-09 8.86278004e-03
 1.17818084e-02 1.18560214e-02 1.20260423e-02 1.57640014e-02
 3.63103944e-02 3.54833110e-02 2.59756973e-02 2.56968542e-02
 1.18267905e-03 6.25132740e-10]
constr_mb1_defl: [-0.0001797]
constr_mb2_defl: [-0.00158303]
constr_shaft_deflection: [0.06377591]
constr_shaft_angle: [0.00079599]
constr_stator_deflection: [0.99650798]
constr_stator_angle: [0.02038174]
constr_hub_diameter: [0.09206083]
constr_length: [0.75700422]
constr_height: [1.62703683]

planet_numbers: [3 3 0]
stage_ratios: [3.39744738 3.21443045 8.79051271]
"""