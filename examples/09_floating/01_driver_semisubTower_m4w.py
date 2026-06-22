# %% [markdown]
# # _Tower (on semisub) optimization_ (M4W 15MW; full `WISDEM`)
# 
# ### current version:
# mainly 'tower' optimization with given:
# - drivetrain/RNA (result of 03_)
# - iea15mw report's semisub geo 
#
# - objective: (1) `tower_mass` minimization
#
# ### TODO:

#%%
import os
from wisdem import run_wisdem
import numpy as np
import matplotlib.pyplot as plt

#%%
wt_m4w = False # init geo of tower: True = acciona / False = iea report
loads_m4w = True

flag_plot = True
save_new_plot = False
verbose = False

flag_opt_GBO = False
flag_scaling_show_browser = False

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
# - iea15mw ref
# fname_wt_input = dir_02_ref_turbines + os.sep + "M4W-15-240-RWT.yaml"
# - m4w 15mw 
# fname_wt_input = dir_02_rwt_m4w + os.sep + "M4W-15-VolturnUS-WT.yaml"

if wt_m4w:
     fname_wt_input = dir_m4w_run +os.sep + "iea15_towerSemi_acciona.yaml"
else:
      fname_wt_input = dir_m4w_run +os.sep + "iea15_towerSemi_report.yaml"

fname_wt_input = dir_m4w_run + os.sep + "outputs\\test_m4w.yaml"

# ---- modelling options
fname_model_opts_m4w = dir_m4w_run+os.sep+ "modelOpts_m4w.yaml"
fname_model_opts_iea = dir_m4w_run+os.sep+ "modelOpts_iea15.yaml"
if loads_m4w:
      fname_modeling_options = fname_model_opts_m4w
else:
     fname_modeling_options = fname_model_opts_iea

# ---- analysis/optimization options
if flag_opt_GBO:
     fname_analysis_options = dir_m4w_run + os.sep + "analyOpts.yaml"
else:
     fname_analysis_options = dir_m4w_run + os.sep + "analyOpts_NOopt.yaml"

## File Management (outputs)
loc_scaling_report = os.path.join(dir_m4w_run,
      'outputs', 'scaling_report.html')
loc_n2 = os.path.join(dir_m4w_run, 'outputs', 'n2.html')
# OR in runWISDEM, before setup()
# om.n2(wt_opt, outfile=os.path.join(folder_output, 'n2.html'), show_browser=True); #(v) debugging

#%% overwrite values TODO
if flag_override_tower_init:
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
freq_tower = wt_opt["floatingse.structural_frequencies"]
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
from Drive4Wind.post_processing import color_schemes
loc_clr_scheme_m4w = color_schemes.loc_clr_scheme_m4w
clrs_m4w = color_schemes.read_color_scheme(loc_clr_scheme_m4w)

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
     freq = wt_opt["floatingse.structural_frequencies"]
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
    if save_new_plot:
        loc_save_img = dir_m4w_run + os.sep + "outputs" + os.sep + (
            "utils_tower_m4w_noFreqConstr.pdf"
        )
        plt.savefig(loc_save_img, dpi=300, bbox_inches='tight')
    plt.show()

#%%[markdown]
# ### Tower geometry
#%%
if flag_plot:
    from Drive4Wind.utilities.plot_tower_data import plot_tower_geo_comparison
    # define yamls and run plot
    # Geometry YAML files
    # 1. base IEA 15-MW
    iea_report_yaml = dir_m4w_run +os.sep + "iea15_towerSemi_report.yaml"
    acciona_yaml = dir_m4w_run +os.sep + "iea15_towerSemi_acciona.yaml"
    # 2. Made4Wind
    m4w_yaml = dir_m4w_run +os.sep+ "outputs" + os.sep+ "test_m4w_new.yaml"
    # loc save img
    if save_new_plot:
        loc_save_img = dir_m4w_run +os.sep+ "outputs" +os.sep+ (
                    "geometry_tower_noFreqConstr_m4w&ieaReport.png"
                )
    else: loc_save_img = None
    # plot
    plot_tower_geo_comparison( m4w_yaml, iea_report_yaml,
                              loc_save_img=loc_save_img )

#%%
