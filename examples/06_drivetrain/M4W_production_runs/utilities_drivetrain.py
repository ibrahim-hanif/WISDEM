"""
utilities_drivetrain.py

written by Vasudev Gupta, IMT NTNU Norway, 2026-05-19
"""
import openmdao as om
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from Drive4Wind.post_processing.color_schemes import loc_clr_scheme_m4w, read_color_scheme
clrs_m4w = read_color_scheme( loc_clr_scheme_m4w )

from wisdem.commonse.fileIO import var_df2dict

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
def write_yaml_of_drivetrain_properties( prob, loc_save_RNAprops4tower ):
    """
    to save `03_DT_layout.py` results for use in further tower optimization
    
    Inputs
    ______
    prob : OpenMDAO problem
    loc_save_RNAprops4tower : str
        file path to save yaml file with RNA properties
    
    """
    from wisdem.inputs import write_yaml
    
    rna_props = {}
    # ===== geometry options =====
    props_geo = rna_props["geometry_options"] = {}
    # ---- drivetrain
    props_DT = props_geo["drivetrain"] = {}

    props_DT["uptilt"] = np.deg2rad(prob["tilt"])[0]
    props_DT["distance_tt_hub"] = prob["drive_height"][0]

    props_DT["distance_hub_mb"] = prob["L_h1"][0]
    props_DT["distance_mb_mb"] = prob["L_12"][0]
    props_DT["overhang"] = prob["overhang"][0]
    
    props_DT["gear_ratio"] = prob["gear_ratio"][0]
    props_DT["gearbox_mass_user"] = prob["gearbox_mass_user"][0]
    props_DT["gearbox_efficiency"] = 0.992 # TODO: hard coded here
    props_DT["gearbox_length_user"] = prob["gearbox_length_user"][0]
    props_DT["gearbox_radius_user"] = prob["gearbox_radius_user"][0]

    
    props_DT["lss_diameter"] = prob["lss_diameter"].tolist()
    props_DT["lss_wall_thickness"] = prob["lss_wall_thickness"].tolist()
    props_DT["lss_material"] = prob["lss_material"]

    props_DT["hss_length"] = prob["L_hss"][0]
    props_DT["hss_diameter"] = prob["hss_diameter"].tolist()
    props_DT["hss_wall_thickness"] = prob["hss_wall_thickness"].tolist()
    props_DT["hss_material"] = prob["hss_material"]

    
    props_DT["bedplate_flange_width"] = prob["bedplate_flange_width"][0]
    props_DT["bedplate_flange_thickness"] = prob["bedplate_flange_thickness"][0]
    props_DT["bedplate_web_thickness"] = prob["bedplate_web_thickness"][0]
    props_DT["bedplate_material"] = prob["bedplate_material"]
    
    
    props_DT["mb1Type"] = prob["bear1.bearing_type"]
    props_DT["mb2Type"] = prob["bear2.bearing_type"]
    props_DT["mb1_e"] = prob["bear1.mb_e"][0]
    props_DT["mb2_e"] = prob["bear2.mb_e"][0]

    # save uptower boolean as boolean not string
    props_DT["uptower"] = bool(prob["uptower"])

    props_DT["converter_mass_user"] = prob["converter_mass_user"][0]

    # ---- generator
    props_gen = props_geo["generator"] = {}
    props_gen["generator_mass_user"] = prob["generator_mass_user"][0]
    props_gen["generator_length"] = prob["L_generator"][0]
    props_gen["generator_radius_user"] = prob["generator_radius_user"][0]
    props_gen["generator_rpm_efficiency_user"] = {
        "grid": prob["generator_efficiency_user"][0,:].tolist(),
        "values": prob["generator_efficiency_user"][1,:].tolist()
    }

    # ===== modeling options =====
    # fill an empty line here in the yaml, without anything for clarity


    props_model = rna_props["modeling_options"] = {}
    props_model["Loading"] = {}

    props_model["Loading"] = {
        "mass": prob["rna_mass"][0],
        "center_of_mass": prob["rna_cm"].tolist(),
        "moment_of_inertia": prob["rna_I_TT"].tolist(), # convert to list for yaml
    }
    rna_loads = props_model["Loading"]["loads"] = [{
        "force": [], # placeholder, will be filled with prob["base_F"]
        "moment": [], # placeholder, will be filled with prob["base_M"]
        "velocity": 11.0, # placeholder, will be filled with 11.0 m/s
    }]
    rna_loads[0]["force"] = prob["base_F"][:,0].tolist()   # convert to list for yaml 
    rna_loads[0]["moment"] = prob["base_M"][:,0].tolist()   # convert to list for yaml

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

    dict_iea = var_df2dict( csv_iea )
    dict_m4w = var_df2dict( csv_m4w )

    # Masses in tonnes [t]
    mass_IEA = {
        "Main shaft":       dict_iea[prefix+"lss_mass"][0] / 1e3,
        "Turret nose":      dict_iea[prefix+"nose_mass"][0] / 1e3,
        "Main bearings":    2.0*dict_iea[prefix+"mean_bearing_mass"][0] / 1e3,
        "Gearbox":          dict_iea[prefix+"gearbox_mass"][0] / 1e3,
        "High-speed shaft": 0.0,
        "Brake":            dict_iea[prefix+"brake_mass"][0] / 1e3,
        "Generator":        dict_iea[prefix+"generator_mass"][0] / 1e3,
        "Converter":        dict_iea[prefix+"converter_mass"][0] / 1e3,
        "Transformer":      dict_iea[prefix+"transformer_mass"][0] / 1e3,
        "Misc. components": (
                            dict_iea[prefix+"hvac_mass"][0]+
                            dict_iea[prefix+"platform_mass"][0]+
                            dict_iea[prefix+"cover_mass"][0]
                            ) / 1e3,
        "Bedplate":         dict_iea[prefix+"bedplate_mass"][0] / 1e3,
        "Yaw system":       dict_iea[prefix+"yaw_mass"][0] / 1e3,
    }

    mass_M4W = {
        "Main shaft":       dict_m4w[prefix+"lss_mass"][0] / 1e3,
        "Turret nose":      0.0,
        "Main bearings":    2.0*dict_m4w[prefix+"mean_bearing_mass"][0] / 1e3,
        "Gearbox":          dict_m4w[prefix+"gearbox_mass"][0] / 1e3,
        "High-speed shaft": dict_m4w[prefix+"hss_mass"][0] / 1e3,
        "Brake":            dict_m4w[prefix+"brake_mass"][0] / 1e3,
        "Generator":        dict_m4w[prefix+"generator_mass"][0] / 1e3,
        "Converter":        dict_m4w[prefix+"converter_mass"][0] / 1e3,
        "Transformer":      dict_m4w[prefix+"transformer_mass"][0] / 1e3,
        "Misc. components": (
                            dict_m4w[prefix+"hvac_mass"][0]+
                            dict_m4w[prefix+"platform_mass"][0]+
                            dict_m4w[prefix+"cover_mass"][0]
                            ) / 1e3,
        "Bedplate":         dict_m4w[prefix+"bedplate_mass"][0] / 1e3,
        "Yaw system":       dict_m4w[prefix+"yaw_mass"][0] / 1e3,
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