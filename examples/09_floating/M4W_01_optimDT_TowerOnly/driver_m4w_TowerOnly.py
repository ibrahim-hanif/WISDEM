# %% [markdown]
# # _Tower (on TLP) optimization_ (M4W 15MW; full `WISDEM`)
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
from wisdem.inputs import load_yaml

#%%
wt_innovative = True # init geo of tower: True = acciona / False = iea report

flag_opt = False

flag_load_results_csv = False

flag_scaling_show_browser = False

flag_override_tower_init = False

#%%
## File management (inputs)
mydir = os.path.dirname(os.path.abspath(__file__))  # get path to this file
dir_examples = os.path.dirname(os.path.dirname(mydir))
dir_02_ref_turbines = dir_examples +os.sep+ "02_reference_turbines" # get path to 02_reference_turbines
dir_02_rwt_m4w = dir_02_ref_turbines +os.sep+"M4W_production_runs"

# ---- wind turbine geometry (same init for both iea and m4w)
# - iea15mw ref
# fname_wt_input = dir_02_ref_turbines + os.sep + "M4W-15-240-RWT.yaml"
# - m4w 15mw 
# fname_wt_input = dir_02_rwt_m4w + os.sep + "M4W-15-VolturnUS-WT.yaml"

# ---- modelling options
fname_model_opts_m4w = mydir+os.sep+ "modelOpts_m4w.yaml"
fname_model_opts_iea = mydir+os.sep+ "modelOpts_iea15.yaml"

if True:
      fname_modeling_options = fname_model_opts_m4w
else:
     fname_modeling_options = fname_model_opts_iea

# ---- analysis/optimization options
loc_analy_opt = mydir + os.sep + "analyOpts.yaml"
loc_analy_NOopt = mydir + os.sep + "analyOpts_NOopt.yaml"

if flag_opt:
     fname_analysis_options = loc_analy_opt
else:
     fname_analysis_options = loc_analy_NOopt

# ---- wind turbine geometry (same init for both iea and m4w)
file_geo_iea_baseline = os.path.join(
     os.path.dirname(mydir),
     "M4W_00_basecase_TowerOnly",
     "M4W-15-TLP-base_case-woRNA.yaml"
)
file_geo_iea_tower = os.path.join(
     os.path.dirname(mydir),
     "M4W_00_basecase_TowerOnly",
     "outputs", "optim.yaml"
)

dict_analy_opt = load_yaml(loc_analy_opt)
file_geo_m4w_tower = os.path.join(
     mydir,
     dict_analy_opt["general"]["folder_output"],
     dict_analy_opt["general"]["fname_output"] + ".yaml"
)

if wt_innovative:
    fname_wt_input = file_geo_m4w_tower
    str_geo = "_m4w"
else:
    fname_wt_input = file_geo_iea_tower
    str_geo = "_iea"

## File Management (outputs)
loc_scaling_report = os.path.join(mydir,
      'outputs', 'scaling_report.html')
loc_n2 = os.path.join(mydir, 'outputs', 'n2.html')
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

print(f"{wt_opt.driver.get_exit_status()} WISDEM run. Check outputs in: {
     opt_options["general"]["folder_output"]}"
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
print("Floating tower fore-aft/side-side freq range:")
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
# directories
if not flag_load_results_csv: dict_analy = opt_options.copy()
else: dict_analy = load_yaml(fname_analysis_options)

folder_results = dict_analy["general"]["folder_output"]
fname_results = dict_analy["general"]["fname_output"]
csv_file = os.path.join( mydir, folder_results, fname_results + ".csv" )

from Drive4Wind.post_processing.color_schemes import read_color_scheme, loc_clr_scheme_m4w
clrs_m4w = read_color_scheme(loc_clr_scheme_m4w)

# plt.rcParams.update( plot_rcParams_update )
linewidth = 3
params_plot_rc = {
        "font.size": 24,
        "axes.labelsize": 24,
        "legend.fontsize": 24, # 16 for pdf of `var_with_iter` plot
        "lines.linewidth": linewidth,
        "lines.markersize": 10, #linewidth*3,
    }
plt.rcParams.update( params_plot_rc )

#%%
# Tower constraints
from Drive4Wind.utilities.plot_tower_data import plot_tower_constraints_stress_utils

fig_TowerConstrs, ax = plot_tower_constraints_stress_utils(
    csv_file,
    figsize=(5.0,10.0),
    colors=["tab:blue","tab:green","tab:orange"]
)

if False: #flag_save_plots: 
    path_plot_tower_constr = os.path.join(
            mydir, folder_results, f"utils_tower{str_geo}.png" )
    fig_TowerConstrs.savefig(
        path_plot_tower_constr,
        bbox_inches="tight",
        dpi=300
    )

# fig_TowerConstrs

#%%[markdown]
# ### Tower geometry
#%%
from Drive4Wind.utilities.plot_tower_data import plot_tower_geo_comparison

# plot
fig_TowerGeo, ax_TowerGeo = plot_tower_geo_comparison(
    m4w_yaml=file_geo_m4w_tower, m4w_label="Made4Wind",
    iea15_yaml=file_geo_iea_tower, iea_label="IEA 15MW (UN)",
    colors=[
        "grey", "tab:blue", "darkgreen"
    ]
)

fig_TowerGeo.set_size_inches([13,8])
# 
for iplot in range(2):
    for iline in range(3):
        ax_TowerGeo[ iplot ].lines[ iline ].set_linewidth( linewidth )
        # if iline != 0: ax_TowerGeo[ iplot ].lines[ iline ].set_marker(".")
        # if iline == 1: ax_TowerGeo[ iplot ].lines[ iline ].set_linestyle("--")
# #
ax_TowerGeo[0].legend(loc="upper right")
# ax_TowerGeo[0].legend_.set_bbox_to_anchor((0.75, 0.1))
#
ax_TowerGeo[0].set_ylabel("Height along tower [m]")
# fig_TowerGeo.suptitle("Made4Wind tower optimization",y=1.0)

if False: #flag_save_plots: 
    path_plot_tower_geo = os.path.join(
            mydir, folder_results, "geometry_tower_m4w&iea.png" )
    fig_TowerGeo.savefig(
        path_plot_tower_geo,
        bbox_inches="tight",
        dpi=300
    )

# fig_TowerGeo

#%%
