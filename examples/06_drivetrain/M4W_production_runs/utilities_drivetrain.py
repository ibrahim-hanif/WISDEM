import openmdao as om
import numpy as np
import pandas as pd

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
    2. TODO : automate, based on outs_recorded keys
    """
    # this_case = df.loc[ row ]
    
    # update problem variables TODO: make this automated
    # prob["gear_ratio"] = this_case['gear_ratio']
    # prob["gearbox_mass_user"] = this_case['gearbox_mass_user']
    # prob["generator_mass_user"] = this_case['generator_mass_user']

    # prob['L_h1'] = this_case['L_h1']
    # prob['L_12'] = this_case['L_12']
    # prob['lss_diameter'] = np.array(eval(this_case['lss_diameter']))
    # prob['lss_wall_thickness'] = np.array(eval(this_case['lss_wall_thickness']))

    # prob['L_hss'] = this_case['L_hss']
    # prob['hss_diameter'] = np.array(eval(this_case['hss_diameter']))
    # prob['hss_wall_thickness'] = np.array(eval(this_case['hss_wall_thickness']))

    # prob['bedplate_web_thickness'] = this_case['bedplate_web_thickness']
    # prob['bedplate_flange_thickness'] = this_case['bedplate_flange_thickness']
    # prob['bedplate_flange_width'] = this_case['bedplate_flange_width']
    # TODO: est. L,R of GB and gen (asked)

    # iter over this_case 
    for key, value in this_case.items():
        # skip non-DV keys
        if key in ["status_driver_exit","time"]: continue
        # work on DVs
        # 1. float type
        if type(value) in [float, np.float64]: prob[key] = value
        # 2. str type for vector DVs
        elif type(value) == str: prob[key] = np.array(eval( value ))
    
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
        outs_recorded[key][i_case,:] = prob[key]

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
        if type(val) == type(""):
            df.at[row,key] = np.str_( val )
        # 2. for time, dvs, obj
        elif type(val) == type( np.empty(1) ):
            if len(val) == 1: # for time, dvs
                df.at[row,key] = val
            elif len(val) == 2:
                df.at[row,key] = np.array2string(val, separator=',')
        
    return df
# ==========