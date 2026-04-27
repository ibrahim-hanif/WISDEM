# %% [markdown]
# # _Drivetrain optimization_ (`WISDEM`)
# under a wrapper function (like `runWISDEM`) - esp. for parallelized parametric studies
# same as `03_DT_layout.py`, so refer for updates and change here accordingly

# %% [markdown]
# imports
import os
import numpy as np
import openmdao.api as om
import time
import matplotlib.pyplot as plt
import pandas as pd
# import scipy.io as sio # --- not used in here, but within imports
# import pickle
import multiprocessing as mp

# %%
from wisdem.drivetrainse.drivetrain import DriveMaterials, DrivetrainSE_M4W

from wisdem.drivetrainse.hub import Hub_System
from wisdem.drivetrainse.gearbox import Gearbox

import wisdem.drivetrainse.layout as lay

import wisdem.drivetrainse.drive_components as dc

import wisdem.drivetrainse.drive_structure as ds

from wisdem.commonse.utilities import get_recorder_results, mainshaft_loads_from_mat_to_dict, load_all_mat_to_dict
from wisdem.commonse.fileIO import save_data, load_data
import utilities_drivetrain as utilsDT

#%%
# ===== Define flags =====
# pre-processing; Loading `openFAST` hub loads from a saved file
part_loads = True       # True kept always 
load_fls_loads = False  # False, as loaded inside DrivetrainSE_M
# False: full loads (72e4,10) (200 Hz sampled, 60mins)
# True: part loads (72e3,11) (20 Hz sampled, 60mins)
dir_loads = "M:\\Vasudev_Gupta\\outputs_mainshaft_loads"

# Optimization flags
flag_opt_GBO = True     # GBO: gradient based optimizer
flag_debug_print = True
flag_parallel = False
n_procs = 8

make_xdsm = False       # html-show or detailed pdf
record_cases = False    #TODO: add in final setup (full problem)
verbose = False

# post-processing results
plot_cases = False      #NOTE: saved, not changing now (commented)
flag_scaling_show_browser = False
flag_save_new_data = False

# Parametric study
flag_study_parametric = True

param_for_study = "LDD"     # "MB" (types) / "LDD" (MS' L_*)
meth_Peq = "DEL".lower()    # "LRD" or "DEL"

# if param_study: init drive prob with 1 iter and reset to require (80) iters
maxIter_param = 1
maxIter_req = 20*5 # test: 2 | required: 80
if flag_study_parametric: maxIter = maxIter_param
else: maxIter = maxIter_req

# ===== flags dict input =====
# ---- if parallel, remove print to screen
if flag_parallel:
    flag_debug_print = False
    verbose = False
# ----- v1 quick dict -----
flags_optim_dict = {
    "flag_opt_GBO": flag_opt_GBO, "flag_debug_print": flag_debug_print,
    "flag_study_parametric": flag_study_parametric, "maxIter": maxIter,
    "flag_parallel": flag_parallel,
    "part_loads": part_loads, "load_fls_loads": load_fls_loads,
    "make_xdsm": make_xdsm, "record_cases": record_cases,
    "verbose": verbose
}
# ---- make analysis options style input TODO ----
analysis_opts = {}
analysis_opts["driver"] = {}
# - optimization (GBO)
analysis_opts["driver"]["optimization"] = {}
analysis_opts["driver"]["optimization"]["flag"] = flag_opt_GBO
analysis_opts["driver"]["optimization"]["debug_print"] = flag_debug_print
# - DOE
analysis_opts["driver"]["design_of_experiments"] = {}
analysis_opts["driver"]["design_of_experiments"]["flag"] = flag_study_parametric
analysis_opts["driver"]["design_of_experiments"]["run_parallel"] = flag_parallel
# - recorder
analysis_opts["recorder"] = {}
analysis_opts["recorder"]["flag"] = record_cases
# analysis_opts["recorder"]["file_name"] = loc_cases
# =====

#%% locs
loc_all_loads_mat_file = os.path.join(dir_loads, "mainshaft_loads_M4W.mat")
loc_FLS_loads_mat_file = os.path.join(dir_loads, "mainshaft_loads_FLS_full.mat")
loc_ULS_loads_mat_file = os.path.join(dir_loads, "mainshaft_loads_ULS.mat")

# ===== Defining results directory and files =====
# - results main dir
results_dir = "04_results"
script_dir = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(script_dir, results_dir)
os.makedirs(results_path, exist_ok=True)

# - used within optimization
loc_record_doe = os.path.join(results_path, "DOE_recorded.sql")
loc_n2 = os.path.join(results_path, "n2.html")
# -- store
locs_optim_dict={
    "loc_all_loads_mat_file": loc_all_loads_mat_file,
    "loc_FLS_loads_mat_file": loc_FLS_loads_mat_file,
    "loc_ULS_loads_mat_file": loc_ULS_loads_mat_file,
    "loc_record_doe":loc_record_doe,
    "loc_n2":loc_n2
}
# -- XDSM?
if make_xdsm:
    loc_xdsm = os.path.join(results_path, 'xdsm_04')
    locs_optim_dict["loc_xdsm"] = loc_xdsm
# -- Record results?
if record_cases:
    print(" ---- Recording cases using `SqliteRecorder` ---- ")
    loc_cases = os.path.join(results_path,
        "cases_recorded_"+meth_Peq+".sql")
    locs_optim_dict["loc_cases"] = loc_cases
    if os.path.exists( loc_cases ):
        os.remove( loc_cases )

# - post-processing
loc_scaling_report = os.path.join(results_path, 'scaling_report.html')
loc_save_data = os.path.join(results_path, "04")
# -- store
locs_optim_dict["loc_save_data"] = loc_save_data
# -- flag load data
# TODO:
# - for 1. run, use 03_results csv;
# - then, use those saved in 04_ (loc_save_data+".csv")
loc_load_data = os.path.join(script_dir, "03_results", "03newULS.csv")
if os.path.exists( loc_load_data ):
    load_from_saved_data = True
else:
    load_from_saved_data = False
# -- update flags
flags_optim_dict["load_from_saved_data"] = load_from_saved_data
locs_optim_dict["loc_load_data"] = loc_load_data

loc_DOEcsv_GBgen = os.path.join(results_path, "DOE_GBgen_cleaned.csv")
# =====

#%%
# ===== Defining options (`modelling_options`), flags =====

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
# =====

#%% # driver guard: prevent recurring spawn
def driver( flags_dict, locs_dict, opts ):
# ===== first unpack the dicts (flags and locs) =====
# - flags
    flag_opt_GBO = flags_dict["flag_opt_GBO"]
    flag_debug_print = flags_dict["flag_debug_print"]
    flag_study_parametric = flags_dict["flag_study_parametric"]
    maxIter = flags_dict["maxIter"]
    flag_parallel = flags_dict["flag_parallel"]
    part_loads = flags_dict["part_loads"]
    load_fls_loads = flags_dict["load_fls_loads"]
    make_xdsm = flags_dict["make_xdsm"]
    record_cases = flags_dict["record_cases"]
    verbose = flags_dict["verbose"]
    load_from_saved_data = flags_dict["load_from_saved_data"]
# - locs
    # - part_loads
    loc_all_loads_mat_file = locs_dict["loc_all_loads_mat_file"]
    # - else
    loc_FLS_loads_mat_file = locs_dict["loc_FLS_loads_mat_file"]
    loc_ULS_loads_mat_file = locs_dict["loc_ULS_loads_mat_file"]

    loc_record_doe = locs_dict["loc_record_doe"]
    loc_n2 = locs_dict["loc_n2"]
    if make_xdsm: loc_xdsm=locs_dict["loc_xdsm"]
    if record_cases: loc_cases=locs_dict["loc_cases"]
    loc_save_data = locs_dict["loc_save_data"]
    loc_load_data = locs_dict["loc_load_data"]
# =====


# ===== Loading `openFAST` hub loads from a saved file =====
    if part_loads: # define paths
        S_all, keys_all = load_all_mat_to_dict(loc_all_loads_mat_file)
    else: # define paths        
        if load_fls_loads: # load from paths
            Snew, keys_new = mainshaft_loads_from_mat_to_dict(
                loc_FLS_loads_mat_file, loc_ULS_loads_mat_file)
# =====


# ===== MANUAL CREATION, imp pts =====
    # - avoid spacing after comma seperation 
    # - header row  = EXACT var/design var name (promoted; eg. drivese.L_12)
    # - each row    = one DOE case (only numbers, no units no text)
    # ---- verify exact promoted variable names using:
    # prob.model.list_inputs(prom_name=True)
    # prob.model.list_outputs(prom_name=True)

    # ==== AUTOMATED CREATION ====
    # # NOTE: written and saved already
    # # ---- write ----
    # df = pd.DataFrame({
    #     "gear_ratio": gearbox_ratios,
    #     "gearbox_mass_user": gearbox_weights,
    #     "generator_mass_user": generator_weights,
    # })
    # df
    # # ---- save ----
    # df.to_csv(loc_DOEcsv_GBgen, index=False)
    # ---- read cases from saved csv ----
    cases = pd.read_csv( loc_DOEcsv_GBgen )
    len_steps = cases.shape[0]
# =====


# ===== run: initial
    # - base case of GR = 49, for testing
    case_DOE = cases.loc[1]
    # run drivetrain optimization
    prob_init, status_init, time_init = run_drivetrain_optim(
        case_DOE,opts,S_all,flags_optim_dict,locs_optim_dict
    )
    # output
    outs_recorded = utilsDT.init_case_dict_from_prob( prob_init, len_steps )
    if verbose: print( outs_recorded )
# =====


# ===== _____ Post-processing _____ =====
    # Print the results
    if verbose:
        print("LSS desvars:")
        print(" ", prob_init["L_h1"], prob_init["L_12"], prob_init["lss_diameter"], prob_init["lss_wall_thickness"] )
        # TODO: for flange mass, dohub (cf. var `flange_t2shell_t`)
        print("HSS desvars:")
        print(" ", prob_init["L_hss"], prob_init["hss_diameter"], prob_init["hss_wall_thickness"] )
        print("Bedplate desvars (w_f, t_f, t_w):")
        print(" ", prob_init["bedplate_flange_width"], prob_init["bedplate_flange_thickness"], prob_init["bedplate_web_thickness"] )
        print(" ")
        print("F_mb*:")
        print(" ", prob_init["F_mb1"], prob_init["F_mb2"] )
        print("M_mb*:")
        print(" ", prob_init["M_mb1"], prob_init["M_mb2"] )
        if doMBfls:
            print("constr_L10_mb(1,2):", prob_init["constr_L10_mb1"], prob_init["constr_L10_mb2"] )
        print("--- constr_ max ---")
        print("- lss: ",
            np.max(prob_init["constr_lss_vonmises"])
            )
        print("- hss: ",
            np.max(prob_init["constr_hss_vonmises"])
            )
        print("- bedplate: ",
            np.max(prob_init["constr_bedplate_vonmises"])
            )
        #
        print("--- obj: masses ---")
        print(f"MSA mass: {prob_init["msa_mass"]}")
        print(f"nacelle mass: {prob_init["nacelle_mass"]}")
        # list_driver_vars = prob.list_driver_vars()

        # Driver scaling report 
        prob_init.driver.scaling_report(
            outfile=loc_scaling_report,show_browser=flag_scaling_show_browser
        );
# =====


# ===== Recorded cases / save data
    if record_cases:
        results_dict = get_recorder_results( loc_cases, None, True )
        if verbose:
            print("\n=== Recorded cases from the optimization ===\n")
            print(results_dict);
    # ### Plot recorded results TODO

    if flag_save_new_data: save_data(loc_save_data, prob_init)
# =====


# ===== DOE: parallelized parametric studies
    # ### Convergence/parametric study setup
    # 1. vary chosen GRs (and rspt. GB and gen weights)
    # -- and save results for nacelle mass optim
    # -- so varied= `gear_ratio`, `gearbox_mass_user`, `generator_mass_user`

    if flag_study_parametric and flag_opt_GBO:
        print("===== parametric study =====")
        myargs = [] # iterable for parallel ( list of lists, each being [i, args...] )
        probs_list = [] # 
        # change flags for parametric study
        flags_optim_dict["maxIter"] = maxIter_req #TODO
        # ---- loop over all cases ----
        for i,row in cases.iterrows():
            # ---- extract case
            # print(row) # debugging
            print(f" - {i}: gear_ratio = { row["gear_ratio"] }")
            myargs.append(
                [i, row,opts,S_all,flags_optim_dict,locs_optim_dict]
            )
            # === Run case in serial ===
            if not flag_parallel:
                print("===== _sequential DOE_ =====")
                # ---- run drivetrain optimization
                prob, status_driver_exit, time_optim = run_drivetrain_optim(
                    row,opts,S_all,flags_optim_dict,locs_optim_dict )
                # ===== post-processing =====
                # ---- outputs of optimization
                # - store
                outs_recorded = utilsDT.fill_case_dict_from_prob(
                    i,prob,outs_recorded,time_optim)
                # - save entire prob
                probs_list.append( prob )
            
        # === Run cases in parallel === ;)        
        if flag_parallel:
            print("===== _parallelized DOE_ =====")
            # ---- using multiprocessing
            with mp.Pool(processes=n_procs) as pool:
                # ---- run!
                results = pool.starmap(parallel_runner, myargs)
                print(" ----- Finished parallelized runs, Alhamdolillah! ----- ")
                # for results in pool.starmap(parallel_runner, myargs):
                # ===== post-processing =====
                # ---- store outputs of optimization
                for k in results:
                    # unpack outputs
                    i_case = k[0]
                    m_nacelle = k[1] # TODO
                    exit_status = k[2]
                    time_optim = k[3]
                    # print out
                    # m_nacelle = prob["nacelle_mass"][0]
                    print( f" - {i_case}: {exit_status}. nacelle_mass: {m_nacelle}; in {time_optim} s." )
                    # store outputs # TODO: prob rmv as arg of parallel_runner
                    # outs_recorded = utilsDT.fill_case_dict_from_prob(
                    #                     i_case,
                    #                     prob,   # prob
                    #                     outs_recorded,
                    #                     time_optim    # total time for optim
                    #     )
        # output
        if verbose: print(outs_recorded);
# =====
    
    return outs_recorded
# ===== end of func =====

# ===============================================================

#%%
def run_drivetrain_optim( case_DOE,
        opts_model, S_all, flags_dict, locs_dict ):
    
    # ===== first unpack the dicts (flags and locs) =====
    # - flags
    flag_opt_GBO = flags_dict["flag_opt_GBO"]
    flag_debug_print = flags_dict["flag_debug_print"]
    flag_study_parametric = flags_dict["flag_study_parametric"]
    maxIter = flags_dict["maxIter"]
    flag_parallel = flags_dict["flag_parallel"]
    part_loads = flags_dict["part_loads"]
    load_fls_loads = flags_dict["load_fls_loads"]
    make_xdsm = flags_dict["make_xdsm"]
    record_cases = flags_dict["record_cases"]
    verbose = flags_dict["verbose"]
    load_from_saved_data = flags_dict["load_from_saved_data"]
    # - locs
    loc_record_doe = locs_dict["loc_record_doe"]
    loc_n2 = locs_dict["loc_n2"]
    if make_xdsm: loc_xdsm=locs_dict["loc_xdsm"]
    if record_cases: loc_cases=locs_dict["loc_cases"]
    loc_save_data = locs_dict["loc_save_data"]
    loc_load_data = locs_dict["loc_load_data"]
    # =====

    # ===== Setup the problem =====
    # Define the problem
    prob = om.Problem(reports=False)
    # Define the model
    prob.model = DrivetrainSE_M4W(modeling_options=opts_model) # an instance of the DrivetrainSE_M4W problem defined above
    # model options
    dohub = opts_model["flags"]["hub"]
    doMBfls = opts_model["flags"]["mb_fls"]
    # =====
    
    # ===== Optimization / DOE setup =====
    # If performing optimization, set up the optimizer and settings
    if flag_opt_GBO:
        if verbose: print("=== running GBO (`run_driver()`) ===")
        # Choose the (GBO) optimizer to use
        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options["optimizer"] = "SLSQP"
        prob.driver.options["tol"] = 1e-4 # default: 1e-6
        prob.driver.options["maxiter"] = maxIter
        if flag_debug_print:
            prob.driver.options["disp"] = True
            prob.driver.options["debug_print"] = [
                "objs", "desvars", "nl_cons", "ln_cons"]
        # prob.driver.options # disp for debugging
        # prob.set_solver_print(level=2)

        if record_cases:
            recorder = om.SqliteRecorder( loc_cases )
            prob.driver.add_recorder( recorder=recorder )

    else:
        if verbose: print("=== running analysis only (`run_model()`) ===")
    # =====

    # ===== setup optimization: objs, desvars, cons =====
    # - TODO: scaling (is better).
    if flag_opt_GBO:
        # === Add objective ===
        prob.model.add_objective("nacelle_mass", ref=1e6)               #DONE: 'nacelle_mass' minimization
        
        # === Add design variables === 
        # 1. LSS
        prob.model.add_design_var("L_h1", lower=0.1, upper=5.0, ref=5.0, ref0=0.1)
        prob.model.add_design_var("L_12", lower=0.1, upper=10.0, ref=10.0, ref0=0.1)
        prob.model.add_design_var("lss_diameter", lower=0.1, upper=5.0, ref=5.0, ref0=0.1)
        prob.model.add_design_var("lss_wall_thickness", lower=4e-3, upper=1.0, ref=1.0, ref0=4e-3)

        # 2. HSS (DONE: add later if needed)
        prob.model.add_design_var("L_hss", lower=0.1, upper=5.0, ref=5.0, ref0=0.1)
        prob.model.add_design_var("hss_diameter", lower=0.1, upper=5.0, ref=5.0, ref0=0.1)
        prob.model.add_design_var("hss_wall_thickness", lower=4e-3, upper=1.0, ref=1.0, ref0=4e-3)

        # 3. Bedplate (DONE: add later if needed)
        prob.model.add_design_var("bedplate_web_thickness", lower=4e-3, upper=5e-1, ref=5e-1, ref0=4e-3)
        prob.model.add_design_var("bedplate_flange_thickness", lower=4e-3, upper=5e-1, ref=5e-1, ref0=4e-3)
        prob.model.add_design_var("bedplate_flange_width", lower=0.01, upper=3.0, ref=3.0, ref0=0.01)

        # 4. hub
        # prob.model.add_design_var("hub_diameter", lower=2.0, upper=5.0)

        # === Add constraints ===

        # 1. von Mises stress util
        prob.model.add_constraint("constr_lss_vonmises", upper=1.0)         #DONE: add next
        prob.model.add_constraint("constr_bedplate_vonmises", upper=1.0)    #TODO: add if needed
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
        # prob.model.add_constraint("constr_stator_deflection", upper=1.0)
        # prob.model.add_constraint("constr_stator_angle", upper=1.0)

        # 3. length: target overhang, hub height and LSS wrt. MBs
        prob.model.add_constraint("constr_Lh1_MB1fw", lower=0.0, ref=1e1)   #DONE: add later
        prob.model.add_constraint("constr_L12_MBsFW", lower=0.0, ref=1e0)   #DONE: add later
        prob.model.add_constraint("constr_length", lower=0.0)               #TODO: rmv or modify?
        prob.model.add_constraint("constr_height", lower=0.0, ref=1e1)      #DONE: add later

        # 4. hub
        # - hub dia to accom. blades' roots
        # prob.model.add_constraint("constr_hub_diameter", lower=0.0)
    # =====

    # ===== Setup the problem =====
    prob.setup()
    # =====

    # ===== pyXDSM trial
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
    # =====

    # ===== Print to screen (debugging)
    if verbose:
        # - objectives, design variables, and constraints in a concise readable form
        print("\n=== All needed inputs to the model ===\n")
        prob.model.list_inputs();

        print("\n=== All outputs from the model ===\n")
        prob.model.list_outputs();
    # =====

    if not load_from_saved_data:
        if verbose: print(" ---- prob variables: re-definition")
        # ===== Loading `openFAST` hub loads from a saved file =====
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
        # =====

        # ===== Defining input values =====
        
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

        # 2. Blade properties and hub design options
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
                # run made4wind_geared.py with flag_opt_GBO = false and copy the following values from drivetrain_example.csv
                prob["hub_system_mass"] = 190e3 # from ref.2, tab. 5-1
                prob["hub_system_cm"] = 3.35947759
                prob["hub_system_I"] = np.array([[865503.52531197, 567289.77714803, 567289.77714803],[0., 0., 0.]])

        # TODO: cm & I (hub_system_ & blades_) will change with DVs (L in lss)

        # 3. Drivetrain configuration and sizing inputs
        myones = np.ones(2)
        # - init condn for some design vars

        # Main Bearing inputs
        prob["bear1.bearing_type"] = "CRB" # 1. floating MB
        prob["bear2.bearing_type"] = "TRB2" # 2. fixed MB
        prob["bear1.mb_e"] = 3.5 # from 3.5-4.0 (TODO: find ref.)
        prob["bear2.mb_e"] = 3.5
        if doMBfls:
            prob["mb_fls.e_mb"] = prob["bear2.mb_e"]

        # Layout / lss inputs
        prob["L_h1"] = 0.345 #(def: 2.0), 4.25
        prob["L_12"] = 5.673 #(def:1.2), 7.1
        prob["lss_diameter"] = np.array([3.215, 1.637]) #(def:1.0), 4.0
        prob["lss_wall_thickness"] = np.array([0.005, 0.132]) #(def:0.1), 0.3

        # Gearbox inputs
        prob["gear_ratio"] = (375 / rated_rpm)
        prob["gearbox_mass_user"] = 135.5*1e3 # D5.1 R2 [135.5 Tn] TODO
        widths_flanks = np.array([400,260,240])/1e3 # 0.9 sum of PLC lengths/flank widths (tab.6, D5.1 R2)
        prob["gearbox_length_user"] = (widths_flanks[0] + 2*np.sum(widths_flanks)) * 1.1 # 2.42 (with 10% margin)
        prob["gearbox_radius_user"] = (5000/2)/1e3 # outer diameter is roughly (m_n*z_r=25*194=4850 mm) plus some margin for housing

        # HSS (DONE: consider as DV if needed)
        prob["L_hss"] = 0.153
        prob["hss_diameter"] = np.array([0.5, 0.636])
        prob["hss_wall_thickness"] = np.array([0.048, 0.005])

        # TODO: Indar generator dimensions (D5.4):
        prob["generator_mass_user"] = 34.8*1e3 # D5.4, tab.9
        prob["generator_radius_user"] = 2.8 / 2 # = stator outer diameter

        # Length (Total cylindrical)
        # prob["L_generator"] = 2.15 # for testing
        # length estimation TODO
        def est_generator_length( D_rotor_outer, pole_pairs, len_active,
                                margin_str=0.1 ):
            # INPUTS:
            # - all: in meters [m]
            # - margin_str: structural margins (def: 10%)
            D_stator_inner = D_rotor_outer/(1-0.002)
            tau_p = (np.pi*D_stator_inner)/(2*pole_pairs)
            len_end_axial = 0.5*tau_p # 0.3 - 0.5
            L_generator = (len_active + 2*len_end_axial) * (1+margin_str)
            # print( f'    Est. generator length: {L_generator} m' ) # 1.124 m
            return L_generator
        prob["L_generator"] = est_generator_length(2.375,24,0.866, 0.1)
        # # -- make an equivalent cylinder from the cuboid with the SAME (mass) MoI
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
            # print( f'    Calculated drive height: {drive_height} m' ) #5.95522 m
            return drive_height
        # prob["drive_height"] = calc_drive_height(prob) #(output= 5.95522 m)
        prob["drive_height"] = 5.614 # (def: 5.614 for 15MW DD)

        # bedplate: Hub:_Rotor_LSS_Frame, Bedplate_IBeam_Frame inputs
        # --- below vals from ONLY bedplate optim (desvars, constr) for nacelle mass min
        prob["bedplate_flange_width"] = 1.724
        prob["bedplate_flange_thickness"] = 0.028
        prob["bedplate_web_thickness"] = 0.029

        # `Hub_*` requires:
        prob["shaft_deflection_allowable"] = 1e-4 # within Hub_Rotor_LSS_Frame (below): Deflections and rotations at GB attachment
        prob["shaft_angle_allowable"] = 1e-3
        # `Bedplate_IBeam_Frame` requires:
        prob["stator_deflection_allowable"] = 1e-2 #(def: 1e-4 m; 1e-2)
        prob["stator_angle_allowable"] = 1e-1 #(def: 1e-3 deg; 1e-1)

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
        
    else:
        prob = load_data( loc_load_data, prob );
        if verbose: print(" ---- prob variables: loading from saved")
    # =====

    # ===== parallel parametric study =====
    # - change respective variables
    if flag_study_parametric:
        prob = utilsDT.read_df_to_prob( case_DOE, prob )
    # =====

    # ===== Final check before running =====
    elif verbose:
        print("\n=== Final input check ===\n")
        prob.model.list_inputs();
        om.n2(prob, outfile=loc_n2, show_browser=True);
    # =====

    # ===== Run: Optimization / DOE / Analysis =====
    # ---- time it ;)
    t0 = time.time()
    
    # `_driver` (optimization) / `_model` (analysis)
    if flag_opt_GBO:
        # Run GBO or DOE
        prob.model.approx_totals() # TODO.
        prob.run_driver()
    else:
        # Run the analysis (also when `flag_study_param` True)
        prob.run_model()

    # ---- time it ;)
    t1 = time.time()
    status_optim = prob.driver.get_exit_status()
    time_optim = (t1-t0)
    if verbose:
        print(" - ",status_optim,
            ": WISDEM run completed in,", time_optim, "s.")
    # =====

    return prob, status_optim, time_optim
# ===== end of func =====

#%%
def parallel_runner(i_case,
        case_data,opts,S_all,flags_optim_dict,locs_optim_dict):
    # run drivetrain optimization
    prob, status, time = run_drivetrain_optim(
        case_data,opts,S_all,flags_optim_dict,locs_optim_dict)
    m_nacelle = prob["nacelle_mass"][0]
    # return
    return i_case, m_nacelle, status, time # TODO prob removed as 2nd arg for now
# ===== end of func =====

#%%
# === Run ===
if __name__ == '__main__': # necessary, protect the entry point
    outs_dict = driver( flags_optim_dict, locs_optim_dict, opts )

# %%
