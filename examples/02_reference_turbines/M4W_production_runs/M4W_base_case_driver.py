# %% [markdown]
# # _Wind Turbine optimization_ (IEA 15MW; full `WISDEM`)
# purpose: copy of `iea15mw_driver.py` to test M4W modifications
# 
# ### current version:
# 1. Drivetrain optimization for the (given, SIMA) hub loads
# - objective: (1) `nacelle_mass` minimization (`NacelleSystemAdder`)
# 2. Tower optimization with a (given) drivetrain/RNA (result of 03_)
# - objective: (1) `turbine_mass` minimization
#
# ### TODO:
# For a final base case - before any drivetrain optim - following must be completed:
# - 1. geo yaml: inc hub height by 2.5m, adjust tower, floater (pt.2)
# - 2. geo yaml: TLP <- finish raft, weis model then input here insha'Allah

#%%
import os
import openmdao.api as om
from wisdem import run_wisdem
import numpy as np
import matplotlib.pyplot as plt
from wisdem.commonse.utilities import load_all_mat_to_dict

#%%
# ---- turbine geo
wt_m4w = False # turbine to analyse: True = m4w / False = iea15mw

# ---- optimization
opt_flag_DT = False
opt_flag_tower = False

# ---- load from saved?
flag_load_from_saved_01_DT = False

flag_plot = True
verbose = False
flag_override_hub_loads = True # TODO: not working; make a flag in model_opts which removes connections

#%%
## File management
mydir = os.path.dirname(os.path.abspath(__file__))  # get path to this file
dir_02_ref_turbines = os.path.dirname(mydir)  # get path to 02_reference_turbines

# ---- wind turbine geometry
fname_wt_m4w = mydir + os.sep + "M4W-15-VolturnUS-WT.yaml"
fname_wt_iea15mw = mydir + os.sep + "IEA-15-VolturnUS-report.yaml"
if wt_m4w:
      fname_wt_input = fname_wt_m4w
      direct = False
else:
      fname_wt_input = fname_wt_iea15mw
      direct = True
      # ---- 01_DT
      if flag_load_from_saved_01_DT:
            fname_wt_input = os.path.join(
                  mydir, "outputs", "01_Drivetrain", "iea15_optim.yaml")
            print(" ---- loaded from saved geo yaml for 01_Drivetrain ---- ")

# ---- modelling options
fname_modeling_options = mydir + os.sep + "modeling_options.yaml"
# ---- analysis/optimization options
if opt_flag_DT:
      fname_analysis_options = mydir + os.sep + "analysis_options_DTopt.yaml"
      print(" ---- 01_Drivetrain optimization on-going ---- ")
else:
      fname_analysis_options = mydir + os.sep + "analysis_options_NOopt.yaml"
      print(" ---- Analysis (no optimization) on-going ---- ")

# ---- hub loads
dir_loads = "M:\\Vasudev_Gupta\\outputs_mainshaft_loads"
loc_all_loads_mat_file = os.path.join(dir_loads, "hub_loads_M4W.mat")

# ---- others
loc_n2 = os.path.join(mydir+os.sep+"outputs", "n2.html")


#%% Loads from hub: overwrite values TODO: rotorse overwrites it at run
if flag_override_hub_loads:
      S_all,_ = load_all_mat_to_dict(loc_all_loads_mat_file)

      F_aero_hub =np.array( [S_all['Fx_max'], S_all['Fy_max'], S_all['Fz_max']] ).reshape((3, 1))
      M_aero_hub =np.array( [S_all['Mx_max'], S_all['My_max'], S_all['Mz_max']] ).reshape((3, 1))
      
      # can't override coz (required) rotorse overwrites them in run
      overrides = {
           'drivese.F_aero_hub': F_aero_hub, 'drivese.M_aero_hub': M_aero_hub
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
doMBfls = analysis_options["flags"]["mb_fls"]
# Print the results
print("F_aero_hub:")
print(" ", wt_opt["drivese.F_aero_hub"]/1e6, " MN" )
print("M_aero_hub:")
print(" ", wt_opt["drivese.M_aero_hub"]/1e6, " MNm \n" )

# ---- 1P and 3P freq ranges
rpm_min = wt_opt['drivese.minimum_rpm'][0]
rpm_rated = wt_opt['drivese.rated_rpm'][0]
freq_range_1P = np.array( [rpm_min, rpm_rated] )/60
freq_range_3P = 3* freq_range_1P
print("1P (blade period) freq ranges:")
print(" ", freq_range_1P, " Hz" )
print("3P (blade passing) freq ranges:")
print(" ", freq_range_3P, " Hz \n" )

print("LSS desvars:")
print(" ", wt_opt["drivese.L_h1"], wt_opt["drivese.L_12"], wt_opt["drivese.lss_diameter"], wt_opt["drivese.lss_wall_thickness"] )
# TODO: for flange mass, dohub (cf. var `flange_t2shell_t`)
if not direct:
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
if not direct:
      print("- hss: ",
            np.max(wt_opt["drivese.constr_hss_vonmises"])
            )
print("- bedplate: ",
      np.max(wt_opt["drivese.constr_bedplate_vonmises"])
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
# -----------------------------------------------------------------------
#%%
# N2 diagram
try:
      om.n2(wt_opt, outfile=loc_n2, show_browser=True);
except: pass

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
     freq = wt_opt["towerse.tower.structural_frequencies"]
     modes_FA = wt_opt["towerse.tower.fore_aft_modes"]
     modes_SS = wt_opt["towerse.tower.side_side_modes"]
     defl_top = wt_opt["towerse.tower.top_deflection"]
     F_tower_base = wt_opt["towerse.tower.turbine_F"]
     M_tower_base = wt_opt["towerse.tower.turbine_M"]
     constr_stress = wt_opt["towerse.post.constr_stress"]
     constr_buckle_GL = wt_opt["towerse.post.constr_global_buckling"]
     constr_buckle_Sh = wt_opt["towerse.post.constr_shell_buckling"]
     # return all as dict
     return {
           'zs': zs, 'ds': ds, 'ts': ts, 'mass': mass, 'cg': cg,
           'constr_d_to_t': constr_d_to_t, 'constr_taper': constr_taper,
           'wind': wind, 'freq': freq, 'modes_FA': modes_FA, 'modes_SS': modes_SS,
           'defl_top': defl_top, 'F_tower_base': F_tower_base, 'M_tower_base': M_tower_base,
           'constr_stress': constr_stress, 'constr_buckle_GL': constr_buckle_GL,
           'constr_buckle_Sh': constr_buckle_Sh
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

#%%
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

# %%
