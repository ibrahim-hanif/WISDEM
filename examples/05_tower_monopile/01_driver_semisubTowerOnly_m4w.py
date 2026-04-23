# %% [markdown]
# # _Wind Turbine optimization_ (IEA 15MW; full `WISDEM`)
# purpose: copy of `driver_tower_monopile_m4w.py` to test M4W modifications
# 
# ### current version:
# mainly 'tower' optimization with a (given) drivetrain/RNA (result of 03_)
#
# - objective: (1) `tower_mass` minimization
#
# ### TODO:
# - adapt the modified DrivetrainSE code to work with `yaml` files ...
# -- DONE: mb*_e added to geomtry iA
# -- TODO: DLC data used for MB FLS, code into yaml (cf. WEIS schema) -- ref `notion` for notes.   
# - add optim params (copied from `03_DT_layout.py`)
# -- DVs: (4) `L_h1, L_12, lss_diameter, lss_wall_thickness`
# -- constraints: (5) lss stresses, deflections (linear, angle), DT dims, L10 MBs FLS
# - geared TLP

#%%
import os
from wisdem import run_wisdem
import numpy as np
import matplotlib.pyplot as plt
from wisdem.commonse.utilities import load_all_mat_to_dict

#%%
wt_m4w = True # turbine to analyse: True = m4w / False = iea15mw
flag_plot = True
verbose = False

flag_opt_GBO = False
flag_scaling_show_browser = False

flag_override_hub_loads = False # TODO: not working; make a flag in model_opts which removes connections
flag_override_tower_init = False

#%%
## File management (inputs)
mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
dir_examples = os.path.dirname(mydir)
dir_02_ref_turbines = dir_examples +os.sep+ "02_reference_turbines" # get path to 02_reference_turbines
dir_02_rwt_m4w = dir_02_ref_turbines +os.sep+"M4W_production_runs"

# M4W run directory
dir_m4w_run = mydir + os.sep + "M4W_01_semisubTower_only"

# ---- wind turbine geometry (same init for both iea and m4w)
# fname_wt_input = dir_m4w_run +os.sep + "iea15mw_tower_semisub_report.yaml"
# fname_wt_input = dir_m4w_run +os.sep + "iea15mw_tower_semisub_acciona.yaml"
fname_wt_input = dir_m4w_run + os.sep + "outputs//test_m4w.yaml"

# ---- modelling options
dir_m4w_runs_main = mydir +os.sep+ "M4W_production_runs"
fname_model_opts_m4w = dir_m4w_runs_main+os.sep+ "modeling_options_m4w_monopile_only.yaml"
fname_model_opts_iea = dir_m4w_runs_main+os.sep+ "modeling_options_iea15_monopile_only_wisdemV3.yaml"
if wt_m4w:
      fname_modeling_options = fname_model_opts_m4w
else:
     fname_modeling_options = fname_model_opts_iea

# ---- analysis/optimization options
if flag_opt_GBO:
     fname_analysis_options = dir_m4w_run + os.sep + "analysis_options.yaml"
else:
     fname_analysis_options = dir_m4w_run + os.sep + "analysis_options_NOopt.yaml"

## File Management (outputs)
loc_scaling_report = os.path.join(dir_m4w_run,
      'outputs', 'tower_scaling_report.html')

#%% Loads from hub: overwrite values TODO: rotorse overwrites it at run
if flag_override_hub_loads:
      dir_loads = "M:\\Vasudev_Gupta\\outputs_mainshaft_loads"
      loc_all_loads_mat_file = os.path.join(dir_loads, "hub_loads_M4w.mat")
      S_all,_ = load_all_mat_to_dict(loc_all_loads_mat_file)

      F_aero_hub =np.array( [S_all['Fx_max'], S_all['Fy_max'], S_all['Fz_max']] ).reshape((3, 1))
      M_aero_hub =np.array( [S_all['Mx_max'], S_all['My_max'], S_all['Mz_max']] ).reshape((3, 1))
      
      # can't override coz (required) rotorse overwrites them in run
      overrides = {
           'drivese.F_aero_hub': F_aero_hub, 'drivese.M_aero_hub': M_aero_hub
      }

elif flag_override_tower_init:
     overrides = {
          'towerse.tower_outer_diameter': np.ones((1,20))*15,
          'towerse.tower_layer_thickness': np.ones((1,20))*100e-3
          }

else: overrides = None

#%%
wt_opt, analysis_options, opt_options = run_wisdem(
    fname_wt_input, fname_modeling_options, fname_analysis_options,
    overridden_values=overrides
)
# TODO
# 1. overwrite hub loads from saved (ULS) file?
# 2. check iea report for tower util plots... not mentioned?
# 3. ! ~ full DT optimization takes toooo LOOOONG !
# - do full DT optim using 03_
# - restrict to some DT DVs? take vals from 03_
# - check gradients wrt. each DV -- bad scaling?
# %%[markdown]
# # _____ Post-processing _____

# %%
# ---- 1P and 3P freq ranges
rpm_min = 5.0 # wt_opt['drivese.minimum_rpm'][0]
rpm_rated = 7.56 # wt_opt['drivese.rated_rpm'][0]
freq_range_1P = np.array( [rpm_min, rpm_rated] )/60
freq_range_3P = 3* freq_range_1P
print("1P (blade period) freq ranges:")
print(" ", freq_range_1P, " Hz" )
print("3P (blade passing) freq ranges:")
print(" ", freq_range_3P, " Hz" )
freq_tower = wt_opt["towerse.tower.structural_frequencies"]
print("Tower fore-aft/side-side freq range:")
print(" ", freq_tower[0:2], " Hz" )

#
print("\n--- RNA properties ---")
print(f"RNA mass: {wt_opt["towerse.rna_mass"]}") # drivese.rna_mass
print(f"RNA cm: {wt_opt["towerse.rna_cg"]}") # drivese.rna_cm
print(f"RNA MoI: {wt_opt["towerse.rna_I"]}") # drivese.rna_I_TT
#
print("\nTower-top / drivetrain bedplate base loads:")
print(" - base_F: ", wt_opt['towerse.tower.rna_F']) # drivese.base_F
print(" - base_M: ", wt_opt['towerse.tower.rna_M']) # drivese.base_M
#
print("\n Tower mass: ", wt_opt['towerse.tower_mass'])
# -----------------------------------------------------------------------

#%%
# Driver scaling report 
try:
    wt_opt.driver.scaling_report(
        outfile=loc_scaling_report,show_browser=flag_scaling_show_browser
    )
except Exception as e:
    print("Error giving scaling report (maybe coz of analysis, not optim): ", e)

#%% plotting options
# main colors
from my_util_tools import util_funcs
loc_clr_scheme_m4w = util_funcs.loc_clr_scheme_m4w
clrs_m4w = util_funcs.read_color_scheme(loc_clr_scheme_m4w)

#%%[markdown]
# ### Tower utilizations
#%%
def get_tower_utilizations( wt_opt ):
     zs = wt_opt["towerse.z_full"]
     ds = wt_opt["towerse.outer_diameter_full"]
     ts = wt_opt["towerse.t_full"]
     mass = wt_opt["towerse.tower_mass"]
     cg = wt_opt["towerse.tower_center_of_mass"]
     constr_d_to_t = wt_opt["towerse.constr_d_to_t"]
     constr_taper = wt_opt["towerse.constr_taper"]
     wind = wt_opt["towerse.env.Uref"]
     freq = wt_opt["towerse.tower.structural_frequencies"]
     modes_FA = wt_opt["towerse.tower.fore_aft_modes"]
     modes_SS = wt_opt["towerse.tower.side_side_modes"]
     defl_top = wt_opt["towerse.tower.top_deflection"]
     F_tower_base = wt_opt["towerse.tower.turbine_F"]
     M_tower_base = wt_opt["towerse.tower.turbine_M"]
     constr_stress = wt_opt["towerse.post.constr_stress"]
     constr_buckle_GL = wt_opt["towerse.post.constr_global_buckling"]
     constr_buckle_Sh = wt_opt["towerse.post.constr_shell_buckling"]
     tower_mass = wt_opt["towerse.tower_mass"]
     # return all as dict
     return {
           'zs': zs, 'ds': ds, 'ts': ts, 'mass': mass, 'cg': cg,
           'constr_d_to_t': constr_d_to_t, 'constr_taper': constr_taper,
           'wind': wind, 'freq': freq, 'modes_FA': modes_FA, 'modes_SS': modes_SS,
           'defl_top': defl_top, 'F_tower_base': F_tower_base, 'M_tower_base': M_tower_base,
           'constr_stress': constr_stress, 'constr_buckle_GL': constr_buckle_GL,
           'constr_buckle_Sh': constr_buckle_Sh,
           'tower_mass': tower_mass
      }
     

def print_tower_utilizations( dict_tower_utils ):
      # unpack dict
      zs = dict_tower_utils['zs']
      ds = dict_tower_utils['ds']
      ts = dict_tower_utils['ts']
      mass = dict_tower_utils['mass']
      cg = dict_tower_utils['cg']
      constr_d_to_t = dict_tower_utils['constr_d_to_t']
      constr_taper = dict_tower_utils['constr_taper']
      wind = dict_tower_utils['wind']
      freq = dict_tower_utils['freq']
      modes_FA = dict_tower_utils['modes_FA']
      modes_SS = dict_tower_utils['modes_SS']
      defl_top = dict_tower_utils['defl_top']
      F_tower_base = dict_tower_utils['F_tower_base']
      M_tower_base = dict_tower_utils['M_tower_base']
      constr_stress = dict_tower_utils['constr_stress']
      constr_buckle_GL = dict_tower_utils['constr_buckle_GL']
      constr_buckle_Sh = dict_tower_utils['constr_buckle_Sh']
      tower_mass = dict_tower_utils['tower_mass']

      print("zs =", zs)
      print("ds =", ds)
      print("ts =", ts)
      print("mass (kg) =", mass)
      print("cg (m) =", cg)
      print("d:t constraint =", constr_d_to_t)
      print("taper ratio constraint =", constr_taper)

      print("\nwind: ", wind)
      print("freq (Hz) =", freq)
      print("Fore-aft mode shapes =", modes_FA)
      print("Side-side mode shapes =", modes_SS)
      print("top_deflection (m) =", defl_top)
      print("Tower base forces (N) =", F_tower_base)
      print("Tower base moments (Nm) =", M_tower_base)
      print("stress =", constr_stress)
      print("GL buckling =", constr_buckle_GL)
      print("Shell buckling =", constr_buckle_Sh)
      print("\n----------\n")
      print("Tower mass =", tower_mass)

#%%
# plot tower utilization
z = 0.5 * (wt_opt["towerse.z_full"][:-1] + wt_opt["towerse.z_full"][1:])
dict_tower_utils = get_tower_utilizations(wt_opt)
if verbose: print_tower_utilizations( dict_tower_utils)

if flag_plot:
    stress = wt_opt["towerse.post.constr_stress"]
    shellBuckle = wt_opt["towerse.post.constr_shell_buckling"]
    globalBuckle = wt_opt["towerse.post.constr_global_buckling"]

    plt.figure(figsize=(5.0, 3.5))
    plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=3)
    plt.plot(stress, z,
      label="stress", color=clrs_m4w['Aqua'])
#     plt.plot(stress[:, 1], z, label="stress 2")
    plt.plot(shellBuckle, z,
      label="shell buckling", color=clrs_m4w['Red'])
#     plt.plot(shellBuckle[:, 1], z, label="shell buckling 2")
    plt.plot(globalBuckle, z,
      label="global buckling", color=clrs_m4w['Dark_Green'])
#     plt.plot(globalBuckle[:, 1], z, label="global buckling 2")
    plt.axvline(1.0, color='k', linestyle='--', linewidth=1, label='1.0 limit')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc=2)
    plt.xlabel("utilization")
    plt.ylabel("height along tower (m)")
    plt.tight_layout()
    plt.show()

#%%[markdown]
# ### Tower geometry
#%%
from wisdem.postprocessing.plot_tower_data import plot_tower_geo_comparison
#%%
# define yamls and run plot
# Geometry YAML files
# 1. base IEA 15-MW
iea_report_yaml = dir_m4w_run +os.sep + "iea15mw_tower_semisub_report.yaml"
acciona_yaml = dir_m4w_run +os.sep + "iea15mw_tower_semisub_acciona.yaml"
# 2. Made4Wind
m4w_yaml = dir_m4w_run +os.sep+ "outputs" + os.sep+ "test_m4w.yaml"
iea_report_yaml = dir_m4w_run +os.sep+ "outputs" + os.sep+ "test_10m.yaml"
# loc save img
loc_save_img = dir_m4w_run +os.sep+ "outputs" +os.sep+ (
            "geometry_tower_noFreqConstr_m4w&ieaReport.png"
        )
# plot
plot_tower_geo_comparison( m4w_yaml, iea_report_yaml )

#%%[markdown]
# ### Monopile utilizations
#%%
def get_monopile_utilizations( wt_opt ):
      zs = wt_opt["fixedse.z_full"]
      ds = wt_opt["fixedse.outer_diameter_full"]
      ts = wt_opt["fixedse.t_full"]
      mass = wt_opt["fixedse.monopile_mass"]
      cg = wt_opt["fixedse.monopile_z_cg"]
      constr_d_to_t = wt_opt["fixedse.constr_d_to_t"]
      constr_taper = wt_opt["fixedse.constr_taper"]
      wind = wt_opt["fixedse.env.Uref"]
      freq = wt_opt["fixedse.structural_frequencies"]
      modes_FA = wt_opt["fixedse.fore_aft_modes"]
      modes_SS = wt_opt["fixedse.side_side_modes"]
      defl_top = wt_opt["fixedse.monopile.top_deflection"]
      F_mudline = wt_opt["fixedse.monopile.mudline_F"]
      M_mudline = wt_opt["fixedse.monopile.mudline_M"]
      constr_stress = wt_opt["fixedse.post.constr_stress"]
      constr_buckle_GL = wt_opt["fixedse.post.constr_global_buckling"]
      constr_buckle_Sh = wt_opt["fixedse.post.constr_shell_buckling"]
      # return all as dict
      return {
           'zs': zs, 'ds': ds, 'ts': ts, 'mass': mass, 'cg': cg,
           'constr_d_to_t': constr_d_to_t, 'constr_taper': constr_taper,
           'wind': wind, 'freq': freq, 'modes_FA': modes_FA, 'modes_SS': modes_SS,
           'defl_top': defl_top, 'F_mudline': F_mudline, 'M_mudline': M_mudline,
           'constr_stress': constr_stress, 'constr_buckle_GL': constr_buckle_GL,
           'constr_buckle_Sh': constr_buckle_Sh
      }

def print_monopile_utilizations( dict_monopile_utils ):
      # unpack dict
      zs = dict_monopile_utils['zs']
      ds = dict_monopile_utils['ds']
      ts = dict_monopile_utils['ts']
      mass = dict_monopile_utils['mass']
      cg = dict_monopile_utils['cg']
      constr_d_to_t = dict_monopile_utils['constr_d_to_t']
      constr_taper = dict_monopile_utils['constr_taper']
      wind = dict_monopile_utils['wind']
      freq = dict_monopile_utils['freq']
      modes_FA = dict_monopile_utils['modes_FA']
      modes_SS = dict_monopile_utils['modes_SS']
      defl_top = dict_monopile_utils['defl_top']
      F_mudline = dict_monopile_utils['F_mudline']
      M_mudline = dict_monopile_utils['M_mudline']
      constr_stress = dict_monopile_utils['constr_stress']
      constr_buckle_GL = dict_monopile_utils['constr_buckle_GL']
      constr_buckle_Sh = dict_monopile_utils['constr_buckle_Sh']

     # print results from the analysis or optimization
      print("zs =", zs)
      print("ds =", ds)
      print("ts =", ts)
      print("mass (kg) =", mass)
      print("cg (m) =", cg)
      print("d:t constraint =", constr_d_to_t)
      print("taper ratio constraint =", constr_taper)

      print("\nwind: ", wind )
      print("freq (Hz) =", freq)
      print("Fore-aft mode shapes =", modes_FA)
      print("Side-side mode shapes =", modes_SS)
      print("top_deflection (m) =", defl_top)
      print("Tower base forces (N) =", F_mudline)
      print("Tower base moments (Nm) =", M_mudline)
      print("stress =", constr_stress)
      print("GL buckling =", constr_buckle_GL)
      print("Shell buckling =", constr_buckle_Sh)

#%%
z_monopile = 0.5 * (
     wt_opt["fixedse.z_full"][:-1] + wt_opt["fixedse.z_full"][1:] )
dict_monopile_utils = get_monopile_utilizations(wt_opt)
if verbose: print_monopile_utilizations( dict_monopile_utils )

if flag_plot:
    stress = wt_opt["fixedse.post.constr_stress"]
    shellBuckle = wt_opt["fixedse.post.constr_shell_buckling"]
    globalBuckle = wt_opt["fixedse.post.constr_global_buckling"]

    plt.figure(figsize=(5.0, 3.5))
    plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=3)
    plt.plot(stress, z_monopile,
      label="stress", color=clrs_m4w['Aqua'])
#     plt.plot(stress[:, 1], z, label="stress 2")
    plt.plot(shellBuckle, z_monopile,
      label="shell buckling", color=clrs_m4w['Red'])
#     plt.plot(shellBuckle[:, 1], z, label="shell buckling 2")
    plt.plot(globalBuckle, z_monopile,
      label="global buckling", color=clrs_m4w['Dark_Green'])
#     plt.plot(globalBuckle[:, 1], z, label="global buckling 2")
    plt.axvline(1.0, color='k', linestyle='--', linewidth=1, label='1.0 limit')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc=2)
    plt.xlabel("utilization")
    plt.ylabel("height along monopile (m)")
    plt.tight_layout()
    plt.show()

# %%
