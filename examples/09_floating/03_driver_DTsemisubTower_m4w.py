# %% [markdown]
# # _Wind Turbine optimization_ (IEA 15MW; full `WISDEM`)
# purpose: copy of `01_driver_semisubTower_m4w.py` to test M4W modifications
# 
# ### current version:
# Drivetrain-Tower Optimization (result of: 03_DT and 01_tower)
#
# - objective: (1) `turbine_mass` minimization (`rna_mass + tower_mass`)
#
# ### TODO:
# 1. why m4w seper optim (DT, tower) has high HSS stress?
# 2. try MPI parallel openMDAO optim (cf. other floating egs.)

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

flag_opt_GBO = False
flag_scaling_show_browser = False

flag_override_own_hub_loads = True # TODO
flag_override_tower_init = False

flag_save_RNAprops4tower = False

#%%
## File management (inputs)
mydir = os.path.dirname(os.path.abspath(__file__))  # get path to this file
dir_examples = os.path.dirname(mydir)
dir_02_ref_turbines = dir_examples +os.sep+ "02_reference_turbines" # get path to 02_reference_turbines
dir_02_rwt_m4w = dir_02_ref_turbines +os.sep+"M4W_production_runs"

# M4W run directory
dir_m4w_run = mydir + os.sep + "M4W_03_DT_towerSemiSub"

# ---- wind turbine geometry (same init for both iea and m4w)
# fname_wt_input = dir_02_rwt_m4w +os.sep + "M4W-15-VolturnUS-WT.yaml"
# fname_wt_input = dir_m4w_run +os.sep + "m4w-DT-towerSemiSub.yaml"
fname_wt_input = dir_m4w_run + os.sep + "outputs//test_m4w.yaml"

# ---- modelling options
fname_model_opts_m4w = dir_m4w_run+os.sep+ "modeling_options_m4w_DTtower.yaml"

fname_modeling_options = fname_model_opts_m4w

# ---- analysis/optimization options
if flag_opt_GBO:
     fname_analysis_options = dir_m4w_run + os.sep + "analysis_options.yaml"
else:
     fname_analysis_options = dir_m4w_run + os.sep + "analysis_options_NOopt.yaml"

## File Management (outputs)
loc_scaling_report = os.path.join(dir_m4w_run,
      'outputs', 'scaling_report.html')
if flag_save_RNAprops4tower:
    loc_save_RNAprops4tower = os.path.join(dir_m4w_run,'outputs',
            "RNA_props_model_for_tower.yaml")

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
# Print the results
print("Optimization driver exited with message: ",
      wt_opt.driver.get_exit_status(), "\n")

doMBfls = analysis_options["flags"]["mb_fls"]
print("MB FLS: ", doMBfls)

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
freq_tower = wt_opt["floatingse.structural_frequencies"] # OR towerse.tower
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
print(f"Nacelle mass: {wt_opt["drivese.nacelle_mass"]}")
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
      iea_report_yaml = dir_02_rwt_m4w +os.sep + "M4W-15-VolturnUS-WT.yaml"
      # ---- iea_report_yaml "ieaReport" plot name
      acciona_yaml = dir_m4w_run +os.sep + "iea15mw_tower_semisub_acciona.yaml"
      # 2. tower-optim M4W 15-MW
      m4w_IC_yaml = dir_m4w_run +os.sep + "m4w-DT-towerSemiSub.yaml"
      # ---- m4w_IC_yaml "onlyTower" plot name
      # 3. DT-tower-optim M4W 15-MW (final run)
      m4w_yaml = dir_m4w_run +os.sep+ "outputs" + os.sep+ "test_m4w.yaml"
      # loc save img
      if save_new_plot:
            loc_save_img = dir_m4w_run +os.sep+ "outputs" +os.sep+ (
                  "geometry_tower_noFreqConstr_m4w&onlyTower.png"
                  )
      else: loc_save_img = None
      # plot
      plot_tower_geo_comparison( m4w_yaml=m4w_yaml, iea15_yaml=m4w_IC_yaml,
                                loc_save_img=loc_save_img,
                                 m4w_label="Integrated",
                                 iea_label="De-coupled" )

#%%[markdown]
# ### Drivetrain mass comparison (IEA and M4W)
#%%
if flag_plot:
    from Drive4Wind.utilities import utilities_drivetrain as utilsDT
    # save plot loc
    loc_save_img = None
    if save_new_plot:
        loc_save_img = dir_m4w_run +os.sep+ "outputs" +os.sep+ (
                        "compare_mass_m4w.png" )
    # plot via func
    utilsDT.plot_drivetrain_mass_comparison(wt_opt, loc_save_img,
        m4w_label="Made4Wind (Integrated)",
        iea_label="IEA 15MW",
        flag_WTnamespace=True)

# %%
# Save rna properties into `yaml` file for next tower optimization
if flag_save_RNAprops4tower:
    utilsDT.write_yaml_of_drivetrain_properties( wt_opt,
            loc_save_RNAprops4tower, flag_WTnamespace=True )
# ===============================================================
# %%
# Compare RNA-TT props btw de-coupled and integrated DT optimization
# 1. de-coupled DT optim
dir_DT_03results =  os.path.join( dir_examples,
      "06_drivetrain","M4W_production_runs","03_results"
)
file_DT = os.path.join(dir_DT_03results,"RNA_props_model_for_tower_m4w_flip.yaml")
# parse
from Drive4Wind.utilities.plot_tower_data import parse_Loading_modelYAML2dict, plot_loads_TT_comparison
dict_DT = parse_Loading_modelYAML2dict(file_DT,flag_yamlFromDrivetrain=True)
dict_WT = parse_Loading_modelYAML2dict(loc_save_RNAprops4tower,flag_yamlFromDrivetrain=True)
# loc_save_img
if save_new_plot:
      loc_save_img = os.path.join(dir_m4w_run,'outputs',
            "compr_RNAprops_DT_decoupl&integrated.pdf")
# plot
plot_loads_TT_comparison(dict_WT, dict_DT,
      m4w_label='Integrated', iea_label='De-coupled',figsize=(11,8),
      loc_save_img=loc_save_img)

# %%
