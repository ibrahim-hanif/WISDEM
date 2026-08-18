"""
utilities_drivetrain.py

written by Vasudev Gupta, IMT NTNU Norway, 2026-05-19
"""
#%%
import openmdao as om
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.io as sio
from Drive4Wind.post_processing.color_schemes import loc_clr_scheme_m4w, read_color_scheme
clrs_m4w = read_color_scheme( loc_clr_scheme_m4w )

from wisdem.commonse.fileIO import var_df2dict
from wisdem.commonse.utilities import load_all_mat_to_dict, pdf_norm_int_using_cdf
from windIO.yaml import load_yaml, write_yaml

#%%
# ==========
def read_df_to_prob( this_case, prob ):
    """
    Parameters
    __________
    this_case : DataFrame row
        output of ( pd.read_csv( csv_file_path ) ).loc[ row ]
    row : int
        row of the case being analysed,
        as each case is listed at one row
    
    Internal Progress
    ______________
    1. DONE : implement
    2. DONE : automate, based on outs_recorded keys
    3. TODO : use `prob.driver.get_design_var_values()` to update from this_case df to prob
    """

    # iter over this_case

    # TODO use prob.driver.get_design_var_values() to update from this_case df to prob
    # lst_dvs = prob.driver.get_design_var_values()
    # for key, val in lst_dvs.items():
    #     # work on DVs
    #     # 1. float type
    #     if type(val) in [float, np.float64]: prob[key] = this_case[key]
    #     # 2. str type for vector DVs or params
    #     elif type(val) == str:
    #         # DV
    #         if this_case[key].startswith('[') and this_case[key].endswith(']'):
    #             prob[key] = np.array(eval( this_case[key] ))
    #         # param
    #         else: prob[key] = str(this_case[key])

    for key, value in this_case.items():
        # skip non-DV keys
        if (
            key in ["status_driver_exit","time"]) or (
                key.startswith("constr_")): continue
        # work on DVs
        # 1. float type
        if type(value) in [float, np.float64]: prob[key] = value
        # 2. str type for vector DVs or params
        elif type(value) == str:
            # DV
            if value.startswith('[') and value.endswith(']'):
                prob[key] = np.array(eval( value ))
            # param
            else: prob[key] = str(value)
    
    return prob
# ==========

# ==========
def init_case_dict_from_prob( prob, len_steps ):
    """
    Note: the dimensions of arrays are (n,1) or (n,m), and NOT (n,); so 2D
    """
    # saving DVs+obj as outputs dict
    # - init to 0
    # - TODO: constr (size 2) are not here, so they become 1 long array
    outs_recorded = {}
    # - driver exit status
    outs_recorded["status_driver_exit"] = ['']*len_steps
    outs_recorded["time"] = np.zeros((len_steps,1))
    # - optim vars
    lst_dvs = prob.driver.get_design_var_values()
    for key, val in lst_dvs.items():
        # make output dict
        len_dv = int(val.size)
        outs_recorded[key] = np.zeros( (len_steps, len_dv) )
        # case sampling (TODO)
    name_obj = list(prob.model.get_objectives().keys())[0]
    outs_recorded[name_obj] = np.zeros((len_steps,1))
    return outs_recorded
# ==========

# ==========
def fill_case_dict_from_prob( i_case, prob, outs_recorded, tcomp ):
    status_driver_exit = prob.driver.get_exit_status()

    for key,_ in outs_recorded.items():
        if key == "status_driver_exit":
            outs_recorded[key][i_case] = np.str_(status_driver_exit)
            continue
        elif key=="time":
            outs_recorded[key][i_case] = tcomp
            continue
        # optim vars
        outs_recorded[key][i_case,:] = prob[key].flatten()

    return outs_recorded
# ==========

# ==========
def write_dict_to_df(  df, row, outs_recorded ):
    """
    function to take values from outs_recorded and store in csv file's case (row)
    """
    for key, val in outs_recorded.items():
        # automatically update df with recorded outs
        # each row
        val = val[row]
        # check type to write in df
        # 1. for status_driver_exit (string)
        if type(val) == np.str_:
            df.at[row,key] = np.str_( val )
        # 2. for time, dvs, obj
        elif type(val) == type( np.empty(1) ):
            if len(val) == 1: # for time, dvs
                df.at[row,key] = val
            elif len(val) > 1: # for vector DVs, constrs
                df.at[row,key] = np.array2string(val, separator=',')
        
    return df
# ==========

# ==========
def write_yaml_of_drivetrain_properties( prob, loc_save_RNAprops4tower,
            direct=False, flag_WTnamespace=False ):
    """
    to save `03_DT_layout.py` results for use in further tower optimization
    
    Inputs
    ______
    prob : OpenMDAO problem
    loc_save_RNAprops4tower : str
        file path to save yaml file with RNA properties
    
    """
    from wisdem.inputs import write_yaml
    
    if flag_WTnamespace: prefix="drivese."
    else: prefix=""

    rna_props = {}
    # ===== geometry options =====
    props_geo = rna_props["geometry_options"] = {}
    # ---- drivetrain
    props_DT = props_geo["drivetrain"] = {}

    # ------ outer shape
    props_outer = props_DT["outer_shape"] = {}
    props_outer["uptilt"] = prob[prefix+"tilt"][0] # windio 2.x needs in deg
    props_outer["distance_tt_hub"] = prob[prefix+"drive_height"][0]
    props_outer["distance_hub_mb"] = prob[prefix+"L_h1"][0]
    props_outer["distance_mb_mb"] = prob[prefix+"L_12"][0]
    props_outer["overhang"] = prob[prefix+"overhang"][0]
    props_outer["cd"] = 0.5
    # ------ gearbox
    props_gb = props_DT["gearbox"] = {}
    props_gb["gear_ratio"] = prob[prefix+"gear_ratio"][0]
    if not direct:
        props_gb["efficiency"] = 0.992 # TODO: hard coded here
        props_gb["mass"] = prob[prefix+"gearbox_mass"][0]
        props_gb["length"] = prob[prefix+"L_gearbox"][0]
        props_gb["radius"] = prob[prefix+"D_gearbox"][0]/2
    else:
        props_gb["efficiency"] = 1.0 # TODO: hard coded here
    # ------ lss
    props_lss = props_DT["lss"] = {}
    props_lss["diameter"] = prob[prefix+"lss_diameter"].tolist()
    props_lss["wall_thickness"] = prob[prefix+"lss_wall_thickness"].tolist()
    props_lss["material"] = prob[prefix+"lss_material"]
    # ------ hss
    props_hss = props_DT["hss"] = {}
    if not direct:
        props_hss["length"] = prob[prefix+"L_hss"][0]
        props_hss["diameter"] = prob[prefix+"hss_diameter"].tolist()
        props_hss["wall_thickness"] = prob[prefix+"hss_wall_thickness"].tolist()
        props_hss["material"] = prob[prefix+"hss_material"]
    else:
    # ------ nose
        props_nose = props_DT["nose"] = {}
        props_nose["diameter"] = prob[prefix+"nose_diameter"]
        props_nose["wall_thickness"] = prob[prefix+"nose_wall_thickness"]
    # ------ bedplate
    props_bed = props_DT["bedplate"] = {}
    if not direct:
        props_bed["flange_width"] = prob[prefix+"bedplate_flange_width"][0]
        props_bed["flange_thickness"] = prob[prefix+"bedplate_flange_thickness"][0]
        props_bed["web_thickness"] = prob[prefix+"bedplate_web_thickness"][0]
    else:
        bed_wt = prob[prefix+"bedplate_wall_thickness"]
        props_bed["wall_thickness"] = {
            "grid": [0.0, 1.0],
            "values": np.array( [bed_wt[0], bed_wt[-1]] ).tolist()
        }
    props_bed["material"] = prob[prefix+"bedplate_material"]
    # ------ other components
    props_other = props_DT["other_components"] = {}
    props_other["mb1Type"] = prob[prefix+"bear1.bearing_type"]
    props_other["mb2Type"] = prob[prefix+"bear2.bearing_type"]
    props_other["mb1_e"] = prob[prefix+"bear1.mb_e"][0]
    props_other["mb2_e"] = prob[prefix+"bear2.mb_e"][0]
    props_other["uptower"] = bool(prob[prefix+"uptower"]) # save uptower boolean as boolean not string
    props_other["converter_mass"] = prob[prefix+"converter_mass"][0]
    # ------ generator
    props_gen = props_DT["generator"] = {}
    props_gen["mass"] = prob[prefix+"generator_mass"][0]
    props_gen["length"] = prob[prefix+"L_generator"][0]
    props_gen["radius"] = prob[prefix+"R_generator"][0]
    gen_eff = prob[prefix+"generator_efficiency"]
    props_gen["rpm_efficiency"] = {
        "grid": [0.0, 1.0],
        "values": np.array( [gen_eff[0], gen_eff[-1]] ).tolist()
    }

    # ===== modeling options =====
    # fill an empty line here in the yaml, without anything for clarity


    props_model = rna_props["modeling_options"] = {}
    props_model["Loading"] = {}

    props_model["Loading"] = {
        "mass": prob[prefix+"rna_mass"][0],
        "center_of_mass": prob[prefix+"rna_cm"].tolist(),
        "moment_of_inertia": prob[prefix+"rna_I_TT"].tolist(), # convert to list for yaml
    }
    rna_loads = props_model["Loading"]["loads"] = [{
        "force": [], # placeholder, will be filled with prob["base_F"]
        "moment": [], # placeholder, will be filled with prob["base_M"]
        "velocity": 11.0, # placeholder, will be filled with 11.0 m/s
    }]
    rna_loads[0]["force"] = prob[prefix+"base_F"][:,0].tolist()   # convert to list for yaml 
    rna_loads[0]["moment"] = prob[prefix+"base_M"][:,0].tolist()   # convert to list for yaml

    write_yaml(rna_props, loc_save_RNAprops4tower)
# ==========

# ==========
def plot_drivetrain_mass_comparison( 
        csv_m4w, csv_iea,
        m4w_label='Made4Wind', iea_label='IEA 15MW',
        flag_WTnamespace=False, loc_save_img=None ):
    """
    plot drivetrain or nacelle mass breakdown comparison between
    IEA 15 MW report and M4W results

    Inputs
    ______
    prob : OpenMDAO problem
        to extract M4W masses for comparison
    flag_WTnamespace : Boolean
        whether to use the complete wind-turbine (WT) namespace for problem variables or not
        False (default): prob["lss_mass"]
        True: prob["drivese.lss_mass"]
    
    Outputs
    _______
    plot : 
        a stacked bar plot comparing the mass breakdown

    Internal Progress
    _______
    1. DONE : copy from 03_ and test
    2. TODO : automate beyond only-DT to complete WT optim (use drivese.* as full namespace of variables)
    """
    if flag_WTnamespace: prefix="drivese."
    else: prefix=""
    # --------------------------------------------------
    # Data (example values, replace with your real ones)
    # --------------------------------------------------
    components = [
        "Main shaft",
        "Turret nose",
        "Main bearings",
        "Gearbox",
        "High-speed shaft",
        "Brake",
        "Generator",
        "Converter",
        "Transformer",
        "Misc. components",
        "Bedplate",
        "Yaw system",
    ]
    len_compns = len(components)

    # iea 15mw
    df_iea = pd.read_csv( csv_iea )
    dict_iea = var_df2dict( df_iea )
    # made4wind
    df_m4w = pd.read_csv( csv_m4w )
    dict_m4w = var_df2dict( df_m4w )

    # Masses in tonnes [t]
    mass_IEA = {
        "Main shaft":       eval(dict_iea[prefix+"lss_mass"]) / 1e3,
        "Turret nose":      eval(dict_iea[prefix+"nose_mass"]) / 1e3,
        "Main bearings":    2.0*eval(dict_iea[prefix+"mean_bearing_mass"]) / 1e3,
        "Gearbox":          eval(dict_iea[prefix+"gearbox_mass"]) / 1e3,
        "High-speed shaft": 0.0,
        "Brake":            eval(dict_iea[prefix+"brake_mass"]) / 1e3,
        "Generator":        eval(dict_iea[prefix+"generator_mass"]) / 1e3,
        "Converter":        eval(dict_iea[prefix+"converter_mass"]) / 1e3,
        "Transformer":      eval(dict_iea[prefix+"transformer_mass"]) / 1e3,
        "Misc. components": (
                            eval(dict_iea[prefix+"hvac_mass"])+
                            eval(dict_iea[prefix+"platform_mass"])+
                            eval(dict_iea[prefix+"cover_mass"])
                            ) / 1e3,
        "Bedplate":         eval(dict_iea[prefix+"bedplate_mass"]) / 1e3,
        "Yaw system":       eval(dict_iea[prefix+"yaw_mass"]) / 1e3,
    }

    mass_M4W = {
        "Main shaft":       eval(dict_m4w[prefix+"lss_mass"]) / 1e3,
        "Turret nose":      0.0,
        "Main bearings":    2.0*eval(dict_m4w[prefix+"mean_bearing_mass"]) / 1e3,
        "Gearbox":          eval(dict_m4w[prefix+"gearbox_mass"]) / 1e3,
        "High-speed shaft": eval(dict_m4w[prefix+"hss_mass"]) / 1e3,
        "Brake":            eval(dict_m4w[prefix+"brake_mass"]) / 1e3,
        "Generator":        eval(dict_m4w[prefix+"generator_mass"]) / 1e3,
        "Converter":        eval(dict_m4w[prefix+"converter_mass"]) / 1e3,
        "Transformer":      eval(dict_m4w[prefix+"transformer_mass"]) / 1e3,
        "Misc. components": (
                            eval(dict_m4w[prefix+"hvac_mass"])+
                            eval(dict_m4w[prefix+"platform_mass"])+
                            eval(dict_m4w[prefix+"cover_mass"])
                            ) / 1e3,
        "Bedplate":         eval(dict_m4w[prefix+"bedplate_mass"]) / 1e3,
        "Yaw system":       eval(dict_m4w[prefix+"yaw_mass"]) / 1e3,
    }

    total_IEA = sum(mass_IEA.values())
    total_M4W = sum(mass_M4W.values())

    # --------------------------------------------------
    # Styling (colors + hatching)
    # --------------------------------------------------
    # Consistent hatching / coloring
    hatches = ['/', '\\', 'x', '-', '+', 'o', 'O', '.', '*', '//', 'xx', '++']
    # Colors:
    # ---- tab10
    tab10 = plt.cm.tab10.colors
    colors = list(tab10) + list(tab10[:2])  # extend to 12 components
    # ----- Made4Wind
    colors = []
    for key,val in clrs_m4w.items():
        colors.append(val)
    colors = np.flip(colors)
    if len_compns > len(colors):
        # mul = np.ceil( len_compns/len(colors), 0)
        colors *= 2
    
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

    fontsize = 18

    # --------------------------------------------------
    # Figure
    # --------------------------------------------------
    # --- Figure setup ---
    fig, ax = plt.subplots(figsize=(14, 14))

    x = np.array([0, 1])
    labels = [iea_label, m4w_label]
    bar_width = 0.45

    # --- Stacking ---
    bottom_IEA = 0.0
    bottom_M4W = 0.0
    tops_IEA, tops_M4W = [], []

    for i, comp in enumerate(components):
        ax.bar(
            x[0], mass_IEA[comp], bottom=bottom_IEA,
            width=bar_width, color=colors[i],
            hatch=hatches[i], edgecolor="black",
            label=comp,
        )

        ax.bar(
            x[1], mass_M4W[comp], bottom=bottom_M4W,
            width=bar_width, color=colors[i],
            hatch=hatches[i], edgecolor="black",
        )

        tops_IEA.append(bottom_IEA + mass_IEA[comp])
        tops_M4W.append(bottom_M4W + mass_M4W[comp])

        bottom_IEA += mass_IEA[comp]
        bottom_M4W += mass_M4W[comp]

    # --- Dotted connectors (top of each component) ---
    for y_iea, y_m4w in zip(tops_IEA, tops_M4W):
        ax.plot(
            [x[0] + bar_width / 2, x[1] - bar_width / 2],
            [y_iea, y_m4w],
            linestyle=":", color="black", linewidth=1.2
        )

    # --- Total mass labels ---
    total_IEA = sum(mass_IEA.values())
    total_M4W = sum(mass_M4W.values())
    offset = 8.0

    ax.text(x[0], total_IEA + offset, rf"${total_IEA:.0f}\,\mathrm{{t}}$",
            ha="center", va="bottom", fontsize=fontsize, fontweight="bold")
    ax.text(x[1], total_M4W + offset, rf"${total_M4W:.0f}\,\mathrm{{t}}$",
            ha="center", va="bottom", fontsize=fontsize, fontweight="bold")

    # --- Formatting ---
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"Mass [t]")
    ax.set_title("Comparison of nacelle mass distribution")
    ax.legend(
        loc="center",
        fontsize=fontsize, frameon=True
    )
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    # -------------------------
    # save
    # -------------------------
    if loc_save_img: plt.savefig( loc_save_img  )
    plt.show()
# ==========
#%%
if __name__ == "__main__":
    mydir = os.path.dirname(os.path.dirname(__file__))
    dir_mainUser = os.path.dirname( os.path.dirname( os.path.dirname( mydir) ) )
    dir_examples = os.path.join( dir_mainUser, "WISDEM","examples" )
    
    dir_DT_03results =  os.path.join( dir_examples,
        "06_drivetrain","M4W_production_runs","03_results"
    )
    dir_WT_03results =  os.path.join( dir_examples,
        "09_floating","M4W_03_DT_towerSemiSub","outputs"
    )
    # model csv files
    file_DT = os.path.join(dir_DT_03results,"RNA_props_model_for_tower_m4w_flip.yaml")
    file_WT = os.path.join(dir_WT_03results,"RNA_props_model_for_tower.yaml")
    # parse
    from Drive4Wind.utilities.plot_tower_data import parse_Loading_modelYAML2dict, plot_loads_TT_comparison
    dict_DT = parse_Loading_modelYAML2dict(file_DT,flag_yamlFromDrivetrain=True)
    dict_WT = parse_Loading_modelYAML2dict(file_WT,flag_yamlFromDrivetrain=True)
    # loc_save_img
    loc_save_img = dir_WT_03results +os.sep+ (
            "compr_RNAprops_iea&m4w.pdf"
        )
    # plot
    plot_loads_TT_comparison(dict_WT, dict_DT,
        m4w_label='Integrated', iea_label='De-coupled',figsize=(8,6))
    
# %%
def parse_rotor_props_from_base_case_csv2dict( loc_csv ):
    """
    Prepare overrides dictionary for rotor system inputs to drivetrainSE
    from saved base-case csv WT file
    """
    df_wt = pd.read_csv( loc_csv )
    dict_wt = var_df2dict( df_wt )

    overrides = {}
    lst_props = [
        # from rotorse or blade
        "drivese.spinner_gust_ws",
        "drivese.rated_rpm",
        "drivese.rated_torque",
        "drivese.pitch_system.BRFM",
        "drivese.blade_root_diameter",
        "drivese.blades_cm",
        "drivese.blade_mass",
        "drivese.blades_mass",
        "drivese.blades_I",
        # towerse
        "drivese.D_top"
    ]
    for name in lst_props:
        overrides[ name ] = eval( dict_wt[name] )
    return overrides

def define_modeling_options_dict_for_drivetrainSE(
        loc_all_loads_mat_file,
        path_modeling_options=None):
    """
    Inputs
    _______
    loc_all_loads_mat_file : string
        path location where hub loads are saved (as an .mat file)
    modeling_options : string
        path to a known options file to overwrite DLC driver vals
    
    Outputs
    _______
    opts : dict
        modeling options for drivetrain-related MDAO
    """
    # Load hub loads
    S_all = sio.loadmat(loc_all_loads_mat_file)
    # S_all, _ = load_all_mat_to_dict(loc_all_loads_mat_file)
    # Auto parse loads dict for
    # - Wind speeds
    ws = S_all["mean_wind_speed"][0,:].tolist()
    # - Probabilities of the wind speeds
    # -- 1. calculate using own function
    pdf_ws_calc = pdf_norm_int_using_cdf(ws).tolist()
    # -- 2. or, use seraj's vals (for consistent compr); cf. Data_collection.xlsx, tab: DLC_driver_UN
    pdf_ws = S_all["probabilities"][0,:].tolist()
    # - Time step
    dt = float(round(S_all["Time"][0,1] - S_all["Time"][0,0],3))
    # ----

    opts = {}

    opts["WISDEM"] = {}
    opts["WISDEM"]["n_dlc"] = 1
    opts["WISDEM"]["DriveSE"] = {}
    # NOTE "hub": 'Hub_System' component are NOT included in the 'DrivetrainSE_M4W' component 
    opts["WISDEM"]["DriveSE"]["hub"] = {}
    opts["WISDEM"]["DriveSE"]["hub"]["hub_gamma"] = 2.0
    opts["WISDEM"]["DriveSE"]["hub"]["spinner_gamma"] = 1.5

    opts["WISDEM"]["DriveSE"]["direct"] = False
    opts["WISDEM"]["DriveSE"]["gearbox_torque_density"] = 0.0

    opts["WISDEM"]["DriveSE"]["gamma_f"] = 1.35 #IEC-1, 7.6.2.2a, pg.57
    opts["WISDEM"]["DriveSE"]["gamma_m"] = 1.3  #IEC-1, 7.6.2.4, pg.59
    opts["WISDEM"]["DriveSE"]["gamma_n"] = 1.0  #IEC-1, 7.6.1.3, pg.55
    opts["WISDEM"]["DriveSE"]["nBins"] = 100    #used by (new) Analytical_FLS_Bearing_Life; =Number of bins for histogram MB FLS
    opts["WISDEM"]["DriveSE"]["own_hub_loads"] = True
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

    # ### Defining options (`modelling_options`), flags
    if path_modeling_options is not None:
        opts = load_yaml(path_modeling_options)

    opts["OpenFAST"] = {}
    opts["OpenFAST"]["simulation"] = {}
    opts["OpenFAST"]["simulation"]["DT"] = dt
    # dir(ectory) where MS loads are stored .csv (?)
    if loc_all_loads_mat_file:
        opts["OpenFAST"]["openfast_dir"] = loc_all_loads_mat_file
    else:
        ValueError('Full loads not defined in openfast_dir<-OpenFAST<-modelling_options. Please define it first. jazakumAllahu khayr.')

    opts["DLC_driver"] = {}
    opts["DLC_driver"]["DLCs"] = [{}]
    opts["DLC_driver"]["DLCs"][0]["DLC"] = "1.2"
    opts["DLC_driver"]["DLCs"][0]["wind_speed"] = ws
    opts["DLC_driver"]["DLCs"][0]["probabilities"] = pdf_ws
    # TODO: probabs check with wind site

    # ### Defining options (`modelling_options`), flags
    if path_modeling_options is not None:
        write_yaml(opts, path_modeling_options)

    return opts

# %%
