# %% [markdown]
# # _Drivetrain optimization_ (`WISDEM`)
# under a wrapper function (like `runWISDEM`) - esp. for parallelized parametric studies
# same as `03_DT_layout.py`, so refer for updates and change here accordingly

#%%
import os
from wisdem import run_wisdem
import numpy as np
import matplotlib.pyplot as plt
from wisdem.commonse.utilities import load_all_mat_to_dict

#%%
wt_m4w = False # turbine to analyse: True = m4w / False = iea15mw

flag_plot = True
save_new_plot = False
verbose = False

opt_flag = False
opt_mbsa = False
flag_scaling_show_browser = False

flag_override_own_hub_loads = True # TODO
flag_override_tower_init = False

#%%
## File management (inputs)
mydir = os.path.dirname(os.path.abspath(__file__))  # get path to this file
dir_examples = os.path.dirname(os.path.dirname(mydir))
dir_02_ref_turbines = dir_examples +os.sep+ "02_reference_turbines" # get path to 02_reference_turbines
dir_02_rwt_m4w = dir_02_ref_turbines +os.sep+"M4W_production_runs"

# M4W run directory
dir_m4w_run = mydir + os.sep + "04_results"

# ---- wind turbine geometry (same init for both iea and m4w)
fname_wt_input = dir_02_rwt_m4w +os.sep + "M4W-15-VolturnUS-WT.yaml"
# fname_wt_input = dir_m4w_run +os.sep + "m4w-DT-towerSemiSub.yaml"
# fname_wt_input = dir_m4w_run + os.sep + "outputs//test_m4w.yaml"

# ---- modelling options
if opt_mbsa:
     fname_model_opts_m4w = dir_m4w_run+os.sep+ "modelOpts_MBSA.yaml"
else:
     fname_model_opts_m4w = dir_m4w_run+os.sep+ "modeling_options.yaml"

fname_modeling_options = fname_model_opts_m4w

# ---- analysis/optimization options
if opt_flag:
     fname_analysis_options = dir_m4w_run + os.sep + "analysis_options.yaml"
     print(" ---- FOWT optimization ---- ")
elif opt_mbsa:
     fname_analysis_options = dir_m4w_run + os.sep + "analysisOpts_MBSA.yaml"
     print(" ---- MBSA optimization ---- ")
else:
     fname_analysis_options = dir_m4w_run + os.sep + "analysis_options_NOopt.yaml"
     print(" ---- Analysis (No optimization) ---- ")

## File Management (outputs)
loc_scaling_report = os.path.join(dir_m4w_run,
      'outputs', 'scaling_report.html')
if opt_mbsa:
     loc_scaling_report = os.path.join(dir_m4w_run,
      'outputs', 'scaling_report_MBSA.html')

#%% Overwrite values ?
overrides = {}

# load hub loads from .mat file (from m4w ULS)
if flag_override_own_hub_loads:
      # read from model_opts yaml
      import wisdem.inputs as sch
      dict_model_opts = sch.load_yaml( fname_model_opts_m4w )
      loc_all_loads_mat_file = dict_model_opts['OpenFAST']['openfast_dir']
      # load
      S_all, keys_all = load_all_mat_to_dict(loc_all_loads_mat_file)
      # reshape
      F_aero_hub = np.array( [S_all['Fx_max'], S_all['Fy_max'], S_all['Fz_max']] ).reshape((3, 1))
      M_aero_hub = np.array( [S_all['Mx_max'], S_all['My_max'], S_all['Mz_max']] ).reshape((3, 1))
      # override
      overrides['drivese.F_aero_hub'] = F_aero_hub
      overrides['drivese.M_aero_hub'] = M_aero_hub

elif flag_override_tower_init:
      overrides['towerse.tower_outer_diameter'] = np.ones((1,20))*15
      overrides['towerse.tower_layer_thickness'] = np.ones((1,20))*100e-3

#%%
wt_opt, analysis_options, opt_options = run_wisdem(
    fname_wt_input, fname_modeling_options, fname_analysis_options,
    overridden_values=overrides
)

# %%[markdown]
# # _____ Post-processing _____

# %%
doMBfls = analysis_options["flags"]["mb_fls"]
print("MB FLS: ", doMBfls)
# Print the results
print("\nF_aero_hub [M-N]:") # NOTE: overwritten with hub loads .mat input
print(" ", wt_opt["drivese.F_aero_hub"]/1e6 )
print("M_aero_hub [M-Nm]:")
print(" ", wt_opt["drivese.M_aero_hub"]/1e6, "\n" )

# ---- 1P and 3P freq ranges
rpm_min = wt_opt['drivese.minimum_rpm'][0]
rpm_rated = wt_opt['drivese.rated_rpm'][0]
freq_range_1P = np.array( [rpm_min, rpm_rated] )/60
freq_range_3P = 3* freq_range_1P
print("1P (blade period) freq ranges:")
print(" ", freq_range_1P, " Hz" )
print("3P (blade passing) freq ranges:")
print(" ", freq_range_3P, " Hz" )
freq_tower = wt_opt["towerse.tower.structural_frequencies"] # towerse.tower OR floatingse.structural_frequencies
print("Tower fore-aft/side-side freq range:")
print(" ", freq_tower[0:2], " Hz \n" )

# ---- drivetrain variables
print("LSS desvars:")
print(" ", wt_opt["drivese.L_h1"], wt_opt["drivese.L_12"], wt_opt["drivese.lss_diameter"], wt_opt["drivese.lss_wall_thickness"] )
#
print("HSS desvars:")
print(" ", wt_opt["drivese.L_hss"], wt_opt["drivese.hss_diameter"], wt_opt["drivese.hss_wall_thickness"] )
print("Bedplate desvars (w_f, t_f, t_w):")
print(" ", wt_opt["drivese.bedplate_flange_width"], wt_opt["drivese.bedplate_flange_thickness"], wt_opt["drivese.bedplate_web_thickness"] )
print(" ")
if doMBfls:
    print("constr_L10_mb(1,2):", wt_opt["drivese.constr_L10_mb1"], wt_opt["drivese.constr_L10_mb2"] )
print("--- constr_ max ---")
print("- lss: ",
      np.max(wt_opt["drivese.constr_lss_vonmises"])
      )
print("- hss: ",
      np.max(wt_opt["drivese.constr_hss_vonmises"])
      )
print("- bedplate: ",
      np.max(wt_opt["drivese.constr_bedplate_vonmises"])
      )
print("- MB1 defl: ", wt_opt["drivese.constr_mb1_defl"] )
print("- MB2 defl: ", wt_opt["drivese.constr_mb2_defl"], "\n" )

print("- tower GL buckling: ",
      np.max( wt_opt["towerse.post.constr_global_buckling"] )
      )
print("- tower Sh buckling: ",
      np.max( wt_opt["towerse.post.constr_shell_buckling"] )
      )
print("- tower stress von-Mises: ",
      np.max( wt_opt["towerse.post.constr_stress"] )
      )

print("\nTower-top / drivetrain bedplate base loads:")
print(" - base_F: ", wt_opt['drivese.base_F'])
print(" - base_M: ", wt_opt['drivese.base_M'])
#
print("\n--- obj: masses ---")
print(f"MSA mass: {wt_opt["drivese.msa_mass"]}")
print(f"nacelle mass: {wt_opt["drivese.nacelle_mass"]}")
print(f"nacelle cm: {wt_opt["drivese.nacelle_cm"]}")

print("\n--- RNA properties ---")
print(f"RNA mass: {wt_opt["drivese.rna_mass"]}")
print(f"RNA cm: {wt_opt["drivese.rna_cm"]}")
#
print("\nTower mass: ", wt_opt['towerse.tower_mass'])
#
print("\nNacelle+Tower mass: ",
      wt_opt['drivese.nacelle_mass'] + wt_opt['towerse.tower_mass']
      )
print("\nRNA+Tower mass: ", wt_opt['towerse.turbine_mass'])
# -----------------------------------------------------------------------
# %%
