#%%
import os
from wisdem import run_wisdem
#%%
## File management
mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
fname_wt_input = mydir + os.sep + "IEA-15-240-RWT.yaml"
fname_modeling_options = mydir + os.sep + "modeling_options.yaml"
fname_analysis_options = mydir + os.sep + "analysis_options.yaml"
#%%
wt_opt, analysis_options, opt_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)
#%%
# OUTPUT (NOTE: this is the Direct-Drive version of IEA-15-240-RWT)
print(wt_opt['components']['nacelle']['drivetrain'])

# =========
# wt.wt_rna
# =========
# NL: NLBGS 1 ; 4.89104182e+11 1
# NL: NLBGS 2 ; 12997708.9 2.65745201e-05
# NL: NLBGS 3 ; 302612.475 6.187076e-07
# NL: NLBGS 4 ; 7634.39772 1.56089398e-08
# NL: NLBGS 5 ; 192.414917 3.93402723e-10
# NL: NLBGS Converged
# ################################################
# Computation of costs of the main turbine components from TurbineCostSE
# Blade cost              759.840 k USD       mass 68208.643 kg
# Pitch system cost       1091.542 k USD       mass 49391.033 kg
# Hub cost                83.499 k USD       mass 21409.881 kg
# Spinner cost            32.606 k USD       mass 2937.439 kg
# ------------------------------------------------
# Rotor cost              3487.167 k USD       mass 278364.281 kg

# LSS cost                187.235 k USD       mass 15734.038 kg
# Main bearing cost       138.535 k USD       mass 30785.643 kg
# Gearbox cost            0.000 k USD       mass 0.000 kg #(v) TODO: GB vals = 0
# HSS cost                0.000 k USD       mass 0.000 kg
# Brake cost              88.018 k USD       mass 24278.014 kg
# Generator cost          5003.335 k USD       mass 368839.407 kg
# Bedplate cost           154.135 k USD       mass 53149.997 kg
# Yaw system cost         233.956 k USD       mass 28187.494 kg
# HVAC cost               1163.279 k USD       mass 9381.281 kg
# Nacelle cover cost      117.160 k USD       mass 20554.306 kg
# Electr connection cost  627.750 k USD
# Controls cost           317.250 k USD
# Other main frame cost   832.291 k USD
# Transformer cost        575.938 k USD       mass 30635.000 kg
# Converter cost          225.296 k USD       mass 11983.850 kg
# ------------------------------------------------
# Nacelle cost            9802.714 k USD       mass 672986.673 kg

# Tower cost              2452.298 k USD       mass 853463.238 kg
# ------------------------------------------------
# ------------------------------------------------
# Turbine cost            15742.178 k USD       mass 1804814.192 kg
# Turbine cost per kW     1049.479 k USD/kW
# ################################################
# ORBIT library intialized at 'C:\Users\vasudevg\.conda\envs\wisdem-env\Lib\site-packages\library'
# ########################################
# Objectives
# Turbine AEP: 77.9000397735 GWh
# Blade Mass:  68208.6425948510 kg
# LCOE:        75.3816041035 USD/MWh
# Tip Defl.:   25.9398537736 m
# ########################################