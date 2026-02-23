# %% [markdown]
# # _Drivetrain optimization_ (`WISDEM`)
# single component optimization (as suggested by `WEIS` ppt)
# 
# ### current version:
# LSS sizing only: (`Hub_Rotor_LSS_Frame` main)
#
# - objective: (1) `msa_mass` minimization (`NacelleSystemAdder`)
#
# - DVs: (4) `L_h1, L_12, lss_diameter, lss_wall_thickness`
#
# - constraints: (5) lss stresses, deflections (linear, angle), DT dims, L10 MBs FLS
#
# ### TODO:
# 1. test on all loads shape=(72e4,10); TODO: change `vals=()` in fls `om` compn's setup method
# 3. optim: add recorder to final setup
# 4. try DOE run (cf. docs), maybe in `00_testing.py`
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
import matplotlib.pyplot as plt
import time
# import scipy.io as sio # --- not used in here, but within imports
# import pickle
import pandas as pd

# %%
from wisdem.drivetrainse.drivetrain import DriveMaterials

from wisdem.drivetrainse.hub import Hub_System
from wisdem.drivetrainse.gearbox import Gearbox

import wisdem.drivetrainse.layout as lay

import wisdem.drivetrainse.drive_components as dc

import wisdem.drivetrainse.drive_structure as ds

from wisdem.commonse.utilities import get_recorder_results, mainshaft_loads_from_mat_to_dict, load_all_mat_to_dict, read_color_scheme
from wisdem.commonse.fileIO import save_data
from wisdem.commonse.cross_sections import Tube
import utilities_drivetrain as utilsDT
# %% [markdown]
# ### Define flags
# post-processing results
make_xdsm, xdsm_type = False, "html"       # html-show or detailed pdf
record_cases = False    #TODO: add in final setup (full problem)
plot_cases = False      #NOTE: saved, not changing now (commented)
flag_scaling_show_browser = False
flag_save_new_data = False

# Loading `openFAST` hub loads from a saved file
part_loads = True 
load_fls_loads = False
# False: full loads (72e4,10) (200 Hz sampled, 60mins)
# True: part loads (72e3,11) (20 Hz sampled, 60mins)
dir_loads = "M:\Vasudev_Gupta\outputs_mainshaft_loads"

# Optimization flags
flag_opt_GBO = True     # GBO: gradient based optimizer
flag_DOE = False        # DOE: design of experiments
flag_opt_GFO = False    # GFO: gradient free optimizer

# Parametric study
flag_study_parametric = True
param_for_study = "mb"     # "MB" (types) / "LDD" (MS' L_*)
meth_Peq = "DEL".lower()    # "LRD" or "DEL"

#%%[markdown]
# ### Defining results directory and files
results_dir = "02_results"
script_dir = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(script_dir, results_dir)
os.makedirs(results_path, exist_ok=True)

loc_doe = os.path.join(results_path, "DOE_recorded.sql")
loc_n2 = os.path.join(results_path, "n2.html")
loc_scaling_report = os.path.join(results_path, 'scaling_report.html')
loc_save_data = os.path.join(results_path, "02")
loc_xdsm = os.path.join(results_path, 'xdsm_02')

loc_DOEcsv_MBtype = os.path.join(results_path, "DOE_MBtype.csv")

# Record results?
if record_cases:
    print(" ---- Recording cases using `SqliteRecorder` ---- ")
    loc_cases = os.path.join(results_path,
        "cases_recorded_"+meth_Peq+".sql")
    if os.path.exists( loc_cases ):
        os.remove( loc_cases )

#%% Loading `openFAST` hub loads from a saved file
if part_loads: # define paths
    # TODO: mainshaft_loads: (old) "." , (newULS) "_M4W"
    loc_all_loads_mat_file = os.path.join(dir_loads, "mainshaft_loads.mat")
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
opts["WISDEM"]["DriveSE"]["nBins"] = 100    #used by (new) Analytical_FLS_Bearing_Life; =Number of bins for histogram MB FLS
# used as: gamma = gamma_f * gamma_m * gamma_n (within TODO)

opts["WISDEM"]["RotorSE"] = {}
opts["WISDEM"]["RotorSE"]["n_pc"] = 2 #cf. RPM_Input in drive_components.py
            # `n_pc`: Number of wind speeds to compute the power curve
opts["materials"] = {}
opts["materials"]["n_mat"] = 4

opts["flags"] = {}
dogen = opts["flags"]["generator"] = False
dohub = opts["flags"]["hub"] = False #(v)
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
opts["DLC_driver"]["DLCs"][0]["wind_speed"] = [ 5.,  7.,  9., 11., 13., 15., 17., 19., 21., 23., 25.]
opts["DLC_driver"]["DLCs"][0]["probabilities"] = [0.06541262, 0.14245179, 0.14299681, 0.12940412, 0.10735197, 0.0824332 , 0.05894909, 0.03942148, 0.02472593, 0.01457773, 0.00466888]

# %% [markdown]
# ### Defining the model `problem class`:
# as an openMDAO group that uses DrivetrainSE classes as components
class LSS_layout( om.Group ):
    """
    Group containing components for the layout of the LSS components
    """
    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        opt_drivese = self.options["modeling_options"]["WISDEM"]["DriveSE"]
        # OpenFAST: containing 1. simulation DT and 2. MS loads dir
        opt_openfast = self.options["modeling_options"]["OpenFAST"]
        # DLC: only 1 used '[0]': containing "wind_speed" and "probabilities"
        opt_DLC = self.options["modeling_options"]["DLC_driver"]["DLCs"][0]

        n_dlcs = self.options["modeling_options"]["WISDEM"]["n_dlc"]
        direct = opt_drivese["direct"]
        if direct:
            use_gb_torque_density = False
        else:
            use_gb_torque_density = opt_drivese["use_gb_torque_density"]
            
        dogen = self.options["modeling_options"]["flags"]["generator"]
        n_pc = self.options["modeling_options"]["WISDEM"]["RotorSE"]["n_pc"]
        flag_hub = self.options["modeling_options"]["flags"]["hub"]
        doMBfls = self.options["modeling_options"]["flags"]["mb_fls"]

        # print flag information
        print("=== Problem 'LSS_layout' setting up ===")
        print(f"flag info: doMBfls={doMBfls}, use_gb_torque_density={use_gb_torque_density}, dogen={dogen}, flag_hub={flag_hub}, direct={direct}")

        # self.set_input_defaults("machine_rating", units="kW")
        #self.set_input_defaults("hvac_mass_coeff", 0.025, units="kg/kW/m")

        # Materials prep
        self.add_subsystem(
            "mat",
            DriveMaterials(direct=direct, n_mat=self.options["modeling_options"]["materials"]["n_mat"]),
                promotes=["*"]
            )
        # - for 'layout' component: need = lss_rho, bedplate_rho, hss_rho 

        # Before the layout, need to do these first
        # 1. hub system (perf hub system optimization)
        if flag_hub: # bypass rn, TODO later
            self.add_subsystem(
                "hub", Hub_System(modeling_options=opt_drivese["hub"]),
                    promotes=["*"]
                )
        
        # # 2. gearbox
        self.add_subsystem(
            "gear", Gearbox(direct_drive=direct, use_gb_torque_density=use_gb_torque_density),
                promotes=["*"]
            )

        # Layout (just discretization of DT and each compn, output 's_drive', etc.)
        #if not direct:
        self.add_subsystem(
            'layout', lay.GearedLayout(),
                promotes=["*"]
            )
        
        # Main Bearings
        self.add_subsystem("bear1", dc.MainBearing_withDerivatives())
        self.add_subsystem("bear2", dc.MainBearing_withDerivatives())
        # -connecting = GearedLayout -to- bear(1,2) (NEW)
        self.connect("Dshaft_mb1", "bear1.D_shaft") #DONE: impl later
        self.connect("Dshaft_mb2", "bear2.D_shaft") #DONE: impl later
        
        # Hub_Rotor_LSS_Frame:
        self.add_subsystem(
            "lss", ds.Hub_Rotor_LSS_Frame(n_dlcs=n_dlcs, modeling_options=opt_drivese),
                promotes=["*"]
            )
        # -connecting = bear(1,2) -to- Hub_Rotor_LSS_Frame (NEW)
        self.connect("bear1.face_width", "mb1_face_width") # mb_fw(s) shifted from GearedLayout to Hub_* to avoid cycle
        self.connect("bear2.face_width", "mb2_face_width")
        self.connect("bear1.mb_Reactions", "mb1_Reactions")
        self.connect("bear2.mb_Reactions", "mb2_Reactions")
        
        # FLS MBs (Analytical)
        if doMBfls:
            self.add_subsystem(
                "mb_fls", ds.Analytical_FLS_Bearing_Life(
                    modeling_options=opt_drivese,
                    openfast_options=opt_openfast,
                    dlc_options=opt_DLC
                    ),
                promotes_inputs=["L_h1","L_12", "rated_rpm","lifetime","carrier_mass","tilt","s_lss"],
                promotes_outputs=["constr_L10_mb1","constr_L10_mb2"]
            )
            # -connecting = bear(1,2) -to- Analy_*
            self.connect("bear2.mb_p", "mb_fls.p_mb") # same for both MBs ---
            self.connect("bear2.mb_X1", "mb_fls.X1_mb")
            self.connect("bear2.mb_Y1", "mb_fls.Y1_mb")
            self.connect("bear2.mb_X2", "mb_fls.X2_mb")
            self.connect("bear2.mb_Y2", "mb_fls.Y2_mb") # ---
            self.connect("bear1.mb_Cr", "mb_fls.Cr_mb1")
            self.connect("bear2.mb_Cr", "mb_fls.Cr_mb2")


        # # Final tallying (mass summation)
        self.add_subsystem(
            "misc", dc.MiscNacelleComponents(direct_drive=direct),
            promotes=["*"]
            )
        self.add_subsystem(
            "nac", dc.NacelleSystemAdder(direct_drive=direct),
            promotes=["*"]
            )
        # self.add_subsystem("rna", dc.RNA_Adder(), promotes=["*"])
        # -connecting = bear(1,2) -to- NacelleSystemAdder
        self.connect("bear1.mb_mass", "mb1_mass")
        # self.connect("bear1.mb_cm", "mb1_cm")
        self.connect("bear1.mb_I", "mb1_I")
        self.connect("bear2.mb_mass", "mb2_mass")
        # self.connect("bear2.mb_cm", "mb2_cm")
        self.connect("bear2.mb_I", "mb2_I")

        # # Bedplate_IBeam_Frame:
        # self.add_subsystem(
        #     "bed", ds.Bedplate_IBeam_Frame(modeling_options=opt_drivese, n_dlcs=n_dlcs),
        #         promotes=["*"]
        #     )
        # -connecting = bear(1,2) -to- Bedplate_*
        # self.connect("bear1.mb_max_defl_ang", "mb1_max_defl_ang")
        # self.connect("bear2.mb_max_defl_ang", "mb2_max_defl_ang")

        if False:
            # Hub_Rotor_LSS_Frame: components required =
            # 1. brake system
            self.add_subsystem(
                "brake", dc.Brake(direct_drive=direct),
                    promotes=["*"]
                )
            
            # 2. generator
            self.add_subsystem(
                "rpm", dc.RPM_Input(n_pc=n_pc),
                    promotes=["*"]
                )
            self.add_subsystem(
                "gensimp", dc.GeneratorSimple(direct_drive=direct, n_pc=n_pc),
                    promotes=["*"]
                )

            # Hub_Rotor_LSS_Frame:
            self.add_subsystem(
                "lss", ds.Hub_Rotor_LSS_Frame(n_dlcs=n_dlcs, modeling_options=opt_drivese, direct_drive=direct),
                    promotes=["*"]
                )
            
            # MBs for defl constraints
            self.add_subsystem("bear1", dc.MainBearing())
            self.add_subsystem("bear2", dc.MainBearing())

            # TODO: add HSS_Frame: perform HSS side optimization

            # Bedplate_IBeam_Frame:
            self.add_subsystem(
                "bed", ds.Bedplate_IBeam_Frame(modeling_options=opt_drivese, n_dlcs=n_dlcs),
                    promotes=["*"]
                )
            
            # NacelleSystemAdder: for nacelle_mass #TODO: add later
            # - NOTE: yaw_mass default (0.0) used, coz dc.YawSystem not included (dc.Electronics also, for transformer_ and converter_mass)
            self.add_subsystem(
                "nac", dc.NacelleSystemAdder(direct_drive=direct),
                    promotes=["*"]
                )

            # Output-to-input connections
            # = mat -to- hub
            if flag_hub:
                self.connect("bedplate_rho", ["pitch_system.rho", "spinner.metal_rho"])
                self.connect("bedplate_Xy", ["pitch_system.Xy", "spinner.Xy"])
                self.connect("bedplate_mat_cost", "spinner.metal_cost")
                self.connect("hub_rho", "hub_shell.rho")
                self.connect("hub_Xy", "hub_shell.Xy")
                self.connect("hub_mat_cost", "hub_shell.metal_cost")
                self.connect("spinner_rho", "spinner.composite_rho")
                self.connect("spinner_Xt", "spinner.composite_Xt")
                self.connect("spinner_mat_cost", "spinner.composite_cost")

            # self.connect("hub_rho", "rho_castiron")
            # self.connect("spinner_rho", "rho_fiberglass")

# %% [markdown]
# ### Setup the problem
# Define the problem
prob = om.Problem(reports=False)

# Define the model
prob.model = LSS_layout(modeling_options=opts) # an instance of the LSS_layout problem defined above

# %%[markdown]
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
    prob.driver.options["maxiter"] = 5 * 6
    prob.driver.options["disp"] = True
    prob.driver.options["debug_print"] = ["desvars", "objs", "nl_cons", "ln_cons"]
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
    prob.driver.options["debug_print"] = ["desvars", "objs", "nl_cons", "ln_cons"]
    # OSError: 'lss' <class Hub_Rotor_LSS_Frame>: Error calling compute(), exception: access violation reading 0x000001CB530B5FB0

elif flag_DOE: #NOTE: running 194 mins! on my PC for 10 levels x 4 DVs (khayr insha'Allah)
               # TAKES HOUR(S), with just 10 levels !!!!!!!!!!!! why?
    print("=== running DOE ===")
    prob.driver = om.DOEDriver(om.FullFactorialGenerator(levels=10))
    recorder = om.SqliteRecorder( loc_doe )
    prob.driver.add_recorder( recorder )

else:
    print("=== running analysis only (`run_model()`) ===")

#%%
# setup optimization: objs, desvars, cons
# - TODO: scaling (is better).
if flag_opt_GBO or flag_opt_GFO or flag_DOE:
    # Add objective
    prob.model.add_objective("msa_mass", ref=1e5)               #DONE: 'msa_mass' minimization
    # - NOTE: effectively 'lss_mass' minimization
    
    # Add design variables
    prob.model.add_design_var("L_h1", lower=0.1, upper=5.0, ref=5.0, ref0=0.2)
    prob.model.add_design_var("L_12", lower=0.1, upper=10.0, ref=10.0, ref0=0.5)
    prob.model.add_design_var("lss_diameter", lower=0.5, upper=4.0, ref=4.0, ref0=0.5)
    prob.model.add_design_var("lss_wall_thickness", lower=4e-3, upper=0.9, ref=1e-1) #DONE: scaled so driver sees lb=0, ub=1 (why? 0.05 causes probs)

    if flag_DOE: pass # DOE: no constraints
    
    # Add constraints
    
    # 1. von Mises stress util
    prob.model.add_constraint("constr_lss_vonmises", upper=1.0)         #DONE: add next
    
    # 2. deflection (main bearing: max perm is angle, + fls) #NOTE: scaling is better
    prob.model.add_constraint("constr_shaft_deflection", upper=1.0)     #DONE: add next
    prob.model.add_constraint("constr_shaft_angle", upper=1.0, ref=1e-3)          #DONE: add next
    if doMBfls:
        prob.model.add_constraint("constr_L10_mb1", lower=1.0)
        prob.model.add_constraint("constr_L10_mb2", lower=1.0)

    # 3. target overhang and hub height
    # prob.model.add_constraint("constr_length", lower=0.0)               #DONE: add later
    # prob.model.add_constraint("constr_height", lower=0.0, ref=1e1)               #DONE: add later
    prob.model.add_constraint("constr_Lh1_MB1fw", lower=0.0, ref=1e1)            #DONE: add later
    prob.model.add_constraint("constr_L12_MBsFW", lower=0.0, ref=1e0)            #DONE: add later

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
# ==== 1. High-level Inputs ====
prob.set_val("machine_rating", 15.0, units="MW")
D_rotor = prob["rotor_diameter"] = 240.0
prob["rated_torque"] = 21.03*1e6 # [Nm] ref.2, tab.5-4
# prob["minimum_rpm"] = 5
prob["rated_rpm"] = 7.56
prob["lifetime"] = 25.0 #design life in years ('lifetime' from WEIS, WindIO)

prob["upwind"] = True
prob["D_top"] = 6.5 #tower top diameter
prob["hub_diameter"] = 7.94
prob["overhang"] = 12.0313 #ref.2
prob["tilt"] = 6.0 #[deg] ref.3

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
    blade_mass = 65250 # from ref.2, tab. ES-2 (= made4wind specs also)
    n_blades = 3
    prob["blades_mass"] = n_blades * blade_mass
    prob["blades_cm"] = 2.46175
    prob["blades_I"] = np.r_[3.48453857e+08, 1.74226928e+08, 1.74226928e+08, np.zeros(3)]

    # if run HUB module within DrivetrainSE
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
        prob["hub_system_mass"] = 190e3 # from ref.2, tab. 5-1
        prob["hub_system_cm"] = 3.35947759
        prob["hub_system_I"] = np.array([[865503.52531197, 567289.77714803, 567289.77714803],[0., 0., 0.]])

# TODO: cm & I (hub_system_ & blades_) will change with DVs (L in lss)

# ==== 3. Drivetrain configuration and sizing inputs ====

myones = np.ones(2)
# - init condn for some design vars

# Main Bearing inputs
prob["bear1.bearing_type"] = "CRB" # 1. floating MB
prob["bear2.bearing_type"] = "TRB2" # 2. fixed MB
# prob["bear1.D_shaft"] = 2.0 #(def:2.0), 4.0
# prob["bear2.D_shaft"] = 2.0 #(def:2.0), 3.2
prob["bear1.mb_e"] = 3.5 # from 3.5-4.0 (TODO: find ref.)
prob["bear2.mb_e"] = 3.5
if doMBfls:
    prob["mb_fls.e_mb"] = prob["bear2.mb_e"]

# Layout / lss inputs
prob["L_h1"] = 0.5 #(def: 0.5), 4.25; converg: 0.264
prob["L_12"] = 2.0 #(def: 2.0), 7.1; converg: 6.936
prob["lss_diameter"] = np.array([2.0, 2.0]) #(def:2.0), 4.0; converg: np.array([2.907, 1.679])
prob["lss_wall_thickness"] = np.array([0.1, 0.1]) #(def:0.1), 0.3; converg: np.array([0.006, 0.123])

flange_MS_length = 0.3*(D_rotor/100)**2 - 0.1*(D_rotor/100) + 0.4
print(f"   - flange length at main-shaft = {flange_MS_length}")

# Gearbox inputs
# prob["L_gearbox"] = 1.5 #(v) calc in gearbox.py
# prob["gear_configuration"] = "eee"
# prob["planet_numbers"] = np.array([5, 3, 0]) #ref.1
prob["gear_ratio"] = 50 #.039
prob["gearbox_mass_user"] = 135.5*1e3 # D5.1 R2
# prob["gearbox_torque_density"] = 200.0 # (cf. line 210, gearbox.py)

prob["L_hss"] = 1.5
prob["hss_diameter"] = 0.5 * myones
prob["hss_wall_thickness"] = 0.1 * myones

# Generator inputs (TODO: add compn later)
# - needed by Bedplate_IBeam_Frame in drive_structure.py, output of HSS_Frame
# - copied from made4wind_geared.py's output drivetrain_example.csv
# prob["R_generator"] = 1.7999999999999998
prob["L_generator"] = 4.2
# TODO: opts:
# --- 1. input from gen design (ingeteam),
# --- 2. maybe calc in generator.py?,
# --- 3. 11.98398883842414 (from drivetrain_example.csv),
# --- 4. 2.0 (drivetrain_geared) or 2.15 (drivetrain_direct)

# prob["generator_cm"] = -0.09998102618633065
# prob["generator_rotor_mass"] = 26437.71371233699
# prob["generator_rotor_I"] = np.array([42829.09621398592, 31598.575743266098, 31598.575743266098])
# prob["F_generator"] = np.array([[-55905.04536116102], [-0.0], [-531900.9765713954]])
# prob["M_generator"] = np.array([[420611.2199999999], [-1687869.5522841304], [-0.0]])
generator_mass_375rpm = 14482 #[kg] (cf. Made4Wind D5.1, Tab.9)

# TODO: Ingeteam generator dimensions (email 15.12.25 from Bidane):
# Mass [kg] = 3 Tn per 8MW conversion line
generator_mass_user = (3*1e3/8)*(prob["machine_rating"]/1e3)
# Overall dimensions (est. very preliminary): 2400x800x4200 mm [HxWxL]
H_generator, W_generator, L_generator = 2.4, 0.8, 4.2 # [m]

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

#%%[markdown]
# ### Final check before running
print("\n=== Final input check ===\n")
prob.model.list_inputs();
om.n2(prob, outfile=loc_n2, show_browser=True);

# %% [markdown]
# ### Run: Optimization / DOE / Analysis
# `_driver` (optimization) / `_model` (analysis)

if flag_opt_GBO or flag_DOE:
    # Run GBO or DOE
    t0 = time.time()
    # main GBO
    prob.model.approx_totals() # TODO.
    prob.run_driver()
    
    t1 = time.time()
    print(" - WISDEM run completed in,", t1-t0, "seconds")

elif flag_opt_GFO:
    # Run the GFO
    prob.run_driver()

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
print("LSS desvars:")
print(" ", prob["L_h1"], prob["L_12"], prob["lss_diameter"], prob["lss_wall_thickness"] )
# [3.48132032] [1.] [4. 4.] [0.32635334 0.29289825]
flangeCyl = Tube( prob["lss_diameter"][0], prob["lss_wall_thickness"][0] )
flange_mass = (flangeCyl.Area * flange_MS_length * prob["lss_rho"])[0]
print(f"   flange mass, est.: {flange_mass} kg. use `dohub` for accurate est.")

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
#
print("--- obj: masses ---")
print(f"MSA mass: {prob["msa_mass"]}")
# Hub_*
# [[4803196.5959757 ] [1369699.99999982] [ -99247.94301496]]
# [[-0.] [-0.] [-0.]]
# - NOTE: "Upwind bearing restricts translational", meth has 'M_mb' also

# Analytical_*
# [[        0.        ] [ 11821817.6056338 ] [-24047488.73239437]]
# [[  5399500.        ] [-13191517.6056338 ] [ 18473288.73239437]]

list_driver_vars = prob.list_driver_vars()
# ==========================================================
#%%
### Recorded cases
if record_cases:
    print("\n=== Recorded cases from the optimization ===\n")
    results_dict = get_recorder_results( loc_cases, None, True )
    print(results_dict);

#%%[markdown]
# ### Plot recorded results
#%%
# main colors
loc_clr_scheme_m4w = "C:\\Users\\vasudevg\\OneDrive - NTNU\\R&D\\Made4Wind\\pics_vids_templates_etc\\color-scheme-made4wind.csv"
clrs_m4w = read_color_scheme(loc_clr_scheme_m4w)
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
    scale_m_msa = 1e3;
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
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    # ========= Row 1 (span both columns): msa_mass =========
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(iters, msa_mass,
            marker='o', linewidth=2, color= clrs_m4w["Dark_Blue"],
            label=r'$m_{msa}$')
    ax1.set_ylabel(r'Mass [t]')
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
    ax3.set_ylabel(r'Dimensions [m]')
    # ax3.set_xlabel('Iteration')
    # ax3.set_xticks(iters)
    ax3.grid(True)
    # ax3.legend(ncol=2)
    ax3.legend(loc='center left',bbox_to_anchor=(1,0.5))

    # ========= Row 3 (span both columns): L10 constraints =========
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(iters, L10_mb1,
            marker='o', linewidth=2, color = clrs_m4w['Dark_Teal'],
            label=r'$L_{10}^{mb1}$')
    ax4.plot(iters, L10_mb2,
            marker='s', linewidth=2, color = clrs_m4w['Aqua'],
            label=r'$L_{10}^{mb2}$')
    ax4.axhline(1.0, color='k', linestyle='--', linewidth=1)
    ax4.set_ylabel(r'Life constraint [-]')
    ax4.set_xlabel('Optimizer iterations')
    ax4.set_xticks(iters)
    ax4.grid(True)
    ax4.legend(loc='center left',bbox_to_anchor=(1,0.5))

    # -------------------------
    # Final layout
    # -------------------------
    fig.tight_layout()

    # -------------------------
    # save
    # -------------------------
    plot_path = os.path.join(results_path,
            "vars_with_iter_"+meth_Peq+".png")
    # plt.savefig(plot_path) # NOTE: saved, so don't change now 

    plt.show()

#%%[markdown]
# Driver scaling report 
prob.driver.scaling_report(
    outfile=loc_scaling_report,show_browser=flag_scaling_show_browser
);
#%%
if flag_save_new_data: save_data(loc_save_data, prob)
# ===============================================================

# %% [markdown]
# ### Convergence/parametric study setup
# 1. for LDD 'inconsistency' check
# 2. for MB types / layout optimization
# - vary DVs over a grid, and record outputs
#%%
if flag_study_parametric and flag_opt_GBO:
    print("===== parametric study =====")

    # ==== Initialize: Combinations to study ====
    if param_for_study.lower() == "mb":
        cases = pd.read_csv( loc_DOEcsv_MBtype )
        
        steps_MBtype = [
            ("CRB","TRB2"),
            ("CARB","TRB2"),
            ("CRB","SRB"),
            ("CARB","SRB")
            ]
        print(" - MB types: ", steps_MBtype);
        # length: total num of param varying steps
        len_steps = len(cases)

    elif param_for_study.lower() == "ldd":
        # L_h1 (0,5.0)      : 1.25, 3.75
        # L_12 (0,10.0))    : 2.5, 7.5

        # TODO: make grid later ?
        # inputs: param vary stepping values
        # steps_L_h1 = np.array([2.0,4.0])
        # steps_L_12 = np.array([1.2,7.0])
        # steps_lss_diameter = np.array([[1.0,4.0],[1.0,4.0]])
        # steps_lss_wall_thickness = np.array([[0.1,0.3],[0.1,0.3]])

        steps_L = [
            (1.25, 2.5),
            (3.75, 2.5),
            (1.25, 7.5),
            (3.75, 7.5)
        ]
        print("- L values: ", steps_L);
        # length: total num of param varying steps
        len_steps = len(steps_L)

        # # Recorded cases: convergence for each parameter in study
        if record_cases:
            # each's recorded cases: save loc and dict
            # TODO: change file name based on analy: ldd or del
            loc_cases_all = [
                "case_"+meth_Peq+"_set_" + str(i) + ".sql" for i in range(
                    1,len_steps+1)
                ]
            # all's
            results_list_of_dicts = [] # array of dict(s)

    else:
        ValueError('Incorrect parameter value for the study: choose "MB" or "LDD"')

    # saving DVs+obj as outputs dict
    # - init to 0
    # - TODO: constr (size 2) are not here, so they become 1 long array
    outs_recorded = {}
    # - status/time
    outs_recorded["status_driver_exit"] = [""]*len_steps
    outs_recorded["time"] = np.zeros( (len_steps,1) )
    # - DVs
    lst_dvs = prob.driver.get_design_var_values()
    for key, val in lst_dvs.items():
        len_dv = int(val.size)
        outs_recorded[key] = np.zeros( (len_steps, len_dv) )
    # - constr
    lst_constr = ["constr_L10_mb1", "constr_L10_mb2"] 
    for key in lst_constr:
        outs_recorded[key] = np.zeros( (len_steps, 1) )
    # - other masses to record
    lst_masses = ["mb1_mass", "mb2_mass", "lss_mass"]
    for key in lst_masses:
        outs_recorded[key] = np.zeros( (len_steps, 1) )
    # - objs
    name_obj = list(prob.model.get_objectives().keys())[0]
    outs_recorded[name_obj] = np.zeros((len_steps,1))
    outs_recorded

    # ==== Run parametric study loop ====
    # set DVs and run driver
    for i in range(len_steps):
        
        # MB param study
        if param_for_study.lower() == "mb":
            # set MB type
            set_MBs = steps_MBtype[i]
            print(f"=== type of bearing: {set_MBs} ===")
            this_case = cases.loc[i]
            prob = utilsDT.read_df_to_prob( this_case, prob )
            print("-------------------- v ------------------")
            # set L val: DONE above
            # prob["L_h1"] = 0.5
            # prob["L_12"] = 7.0

        # LDD param study
        elif param_for_study.lower() == "ldd":

            if record_cases:
                # loc: define for this iter
                loc_case_i = os.path.join(results_path, loc_cases_all[i])
                if os.path.exists( loc_case_i ):
                    os.remove( loc_case_i )
                # driver: change recorder; TODO: problematic with saved sql files
                recorder = om.SqliteRecorder( loc_case_i )
                prob.driver.add_recorder( recorder=recorder )

            # set MB type: DONE above
            # prob["bear1.bearing_type"] = "CRB"
            # prob["bear2.bearing_type"] = "TRB2"
            # set L val
            set_Ls = steps_L[i]
            print(f"=== value of L: {set_Ls} ===")
            prob["L_h1"] = set_Ls[0]
            prob["L_12"] = set_Ls[1]
            print("---------- v ----------")
        
        # redef ? dia/thick :
        # prob["lss_diameter"] = myones * 4.0
        # prob["lss_wall_thickness"] = myones * 0.3

        # ---- time it ;)
        t0 = time.time()
        # ===== RUN driver (GBO) =====
        prob.model.approx_totals()
        prob.run_driver()
        # ---- time it ;)
        t1 = time.time()
        # ---- post-process
        time_optim = t1-t0
        status_optim = prob.driver.get_exit_status()
        print(" - ",status_optim,": MSA optim run completed in,", time_optim, "s.")
        # print( "L_h1 = ", prob["L_h1"] ) # debugging
        
        # ===== post-processing =====
        # save outputs
        # for key, val in outs_recorded.items():
        #     outs_recorded[key][i,:] = prob[key]
        outs_recorded = utilsDT.fill_case_dict_from_prob(
                            i, prob, outs_recorded, time_optim )

        # Recorded cases
        if param_for_study.lower() == "ldd" and record_cases:
            results_list_of_dicts.append(
                get_recorder_results( loc_case_i, None, True ) #out=dict
                )
    # print outputs dict
    print(outs_recorded);
# =====

#%% save in to df and csv
casesOut = cases.copy()
for i in range(len_steps):
    casesOut = utilsDT.write_dict_to_df(casesOut,i,outs_recorded)
# save to csv
casesOut.to_csv(loc_DOEcsv_MBtype, index=False)

#%% [markdown]
# ## Result outputs
# ==== 1. MB type vary
""" 
{'L_12': array([[6.93563233],
       [4.95328602],
       [4.90916032 ],
       [2.41317276]]),
'L_h1': array([[0.26441632],
       [0.2       ],
       [0.30079521],
       [0.2       ]]),
'lss_diameter': array([[2.90729986, 1.67985956],
       [0.82344261, 1.9916245 ],
       [3.43984147, 1.74033424],
       [1.12356334, 1.78972858]]),
'lss_wall_thickness': array([[0.00609344, 0.12302497],
       [0.24916403, 0.10025035],
       [0.004    , 0.1101714],
       [0.22114549, 0.17132604]]),
'msa_mass': array([[68293.12376318],
       [52305.38151471],
       [64650.67674836],
       [38386.64245769]])}
"""
# ==== 1. MB type vary: new ULS
"""
{'L_h1': array([[0.30305484],
       [0.19005296],
       [0.30272963],
       [0.20136395]]),
'L_12': array([[4.85547772],
       [4.06728249],
       [4.86676433],
       [2.16464623]]),
'lss_diameter': array([[3.4551008 , 1.92397095],
       [1.12036565, 2.48709673],
       [3.44842101, 1.94309144],
       [1.04527294, 2.78601022]]),
'lss_wall_thickness': array([[0.02584622, 0.27731484],
       [0.82202806, 0.12166346],
       [0.02551514, 0.2671655 ],
       [0.9       , 0.09527698]]),
'msa_mass': array([[106210.72664515],
       [ 97263.67950993],
       [ 95305.95598093],
       [ 67281.03505691]])}
"""

# ==== 2. L_ vary: LRD (DEL gives same results :D AL)
"""
{'L_h1': array([[0.3030549 ],
       [0.3030549 ],
       [0.30305493],
       [0.30305484]]),
'L_12': array([[4.85547562],
       [4.85547561],
       [4.85547463],
       [4.85547772]]),
'lss_diameter': array([[3.45510162, 1.92397181],
       [3.45510162, 1.92397181],
       [3.4551016 , 1.92397191],
       [3.4551008 , 1.92397095]]),
'lss_wall_thickness': array([[0.02584621, 0.27731437],
       [0.02584621, 0.27731437],
       [0.02584617, 0.27731429],
       [0.02584622, 0.27731484]]),
'msa_mass': array([[106210.69683583],
       [106210.69610393],
       [106210.6729395 ],
       [106210.72664515]])}
"""
# %%
if (param_for_study.lower() == "ldd") and (
    record_cases and plot_cases):
    print(" NOTE: 2D multi-start converg plot for testing now; not being saved")
    # -------------------------
    # Figure
    # -------------------------
    fig, ax = plt.subplots(figsize=(6.5, 6))

    for i, res in enumerate(results_list_of_dicts):

        # Extract & squeeze
        L_h1 = res['L_h1'].squeeze()
        L_12 = res['L_12'].squeeze()

        # Path (iterations)
        ax.plot(
            L_h1,
            L_12,
            color='0.7',
            linewidth=1.5,
            zorder=1
        )

        # Intermediate points
        ax.scatter(
            L_h1[:-1],
            L_12[:-1],
            color='0.7',
            s=25,
            zorder=2
        )

        # Starting point
        ax.scatter(
            L_h1[0],
            L_12[0],
            color = clrs_m4w["Aqua"],
            s=80,
            zorder=4,
            label='Start' if i == 0 else None
        )

        # Final (converged) point
        ax.scatter(
            L_h1[-1],
            L_12[-1],
            color = clrs_m4w["Dark_Blue"],
            marker='x',
            s=100,
            zorder=4,
            label='Converged' if i == 0 else None
        )

    # -------------------------
    # Axes formatting
    # -------------------------
    ax.set_xlabel(r'$L_{h1}\ \mathrm{[m]}$')
    ax.set_ylabel(r'$L_{12}\ \mathrm{[m]}$')

    ax.grid(True)
    ax.legend(loc='center')
    # supported values are 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    # ax.set_title('2D optimization convergence path')
    
    # -------------------------
    # Final layout
    plt.tight_layout()
    # -------------------------
    # Save plot
    plot_path = os.path.join(results_path,
        meth_Peq+"_multiStart_optim_path.png")
    # plt.savefig(plot_path) # NOTE: saved, so don't change now 
    
    plt.show()

# %%
if (param_for_study.lower() == "ldd") and (
    record_cases and plot_cases):
    print(" NOTE: 3D multi-start converg plot for testing now; not being saved")
    print("NOTE: matplotlib [Bug]: zlabel on 3D axes cut off when using `%matplotlib inline` in Jupyter #28117")
    # imports
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    # -------------------------
    # Figure
    # -------------------------
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')

    for i, res in enumerate(results_list_of_dicts):

        # Extract & squeeze
        L_h1 = res['L_h1'].squeeze()
        L_12 = res['L_12'].squeeze()
        m_msa = res['msa_mass'].squeeze()
        msa_scale = 1e3  # kg -> tonnes
        m_msa = m_msa / msa_scale # z_plot

        # Trajectory
        ax.plot(
            L_h1,
            L_12,
            m_msa,
            color='0.7',
            linewidth=1.5,
            zorder=1
        )

        # Intermediate points
        ax.scatter(
            L_h1[:-1],
            L_12[:-1],
            m_msa[:-1],
            color='0.7',
            s=25,
            zorder=2
        )

        # Starting point
        ax.scatter(
            L_h1[0],
            L_12[0],
            m_msa[0],
            color = clrs_m4w["Aqua"],
            s=80,
            zorder=3,
            label='Start' if i == 0 else None
        )

        # Label start with set number
        ax.text(
            L_h1[0],
            L_12[0],
            m_msa[0],
            f'{i+1}',
            fontsize=16,
            color='k'
        )

        # Converged point
        ax.scatter(
            L_h1[-1],
            L_12[-1],
            m_msa[-1],
            color = clrs_m4w["Dark_Blue"],
            marker='x',
            s=100,
            zorder=4,
            depthshade=False,
            label='Converged' if i == 0 else None
        )

    # -------------------------
    # Axes formatting
    # -------------------------
    ax.set_xlabel(r'$L_{h1}\ \mathrm{[m]}$', labelpad=10)
    ax.set_ylabel(r'$L_{12}\ \mathrm{[m]}$')
    ax.set_zlabel(r'$m_{\mathrm{MSA}}\ \mathrm{[t]}$')

    # ax.set_title('3D optimization convergence path')
    ax.legend(loc='best')

    # -------------------------
    # Final layout
    plt.tight_layout()
    # -------------------------
    # rotate view (via camera angles)
    # def: (30,-60), print(ax.elev, ax.azim)
    if meth_Peq=="lrd": ax.view_init(elev=40, azim=-20)
    # plt.ion() # interactive
    # ------------------------
    # Save plot
    plot_path = os.path.join(results_path,
        meth_Peq+"_multiStart3D_optim_path.png")
    # plt.savefig(plot_path) # NOTE: saved, so don't change now 

    plt.show()
# =============================================================

#%%