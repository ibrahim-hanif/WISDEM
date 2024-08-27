import os

from wisdem import run_wisdem

## File management
mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
fname_wt_input = mydir + os.sep + "IEA-15-240-RWT.yaml"
fname_modeling_options = mydir + os.sep + "modeling_options.yaml"
fname_analysis_options = mydir + os.sep + "analysis_options.yaml"

wt_opt, analysis_options, opt_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)

#(v) OUTPUTS:
"""
==============
comp.wt.wt_rna
==============
NL: NLBGS 1 ; 4.8908897e+11 1
NL: NLBGS 2 ; 12033602 2.46041165e-05
NL: NLBGS 3 ; 277408.174 5.67193682e-07
NL: NLBGS 4 ; 6976.91684 1.42651282e-08
NL: NLBGS 5 ; 175.207371 3.58232105e-10
NL: NLBGS Converged
RuntimeWarning: \\home.ansatt.ntnu.no\vasudevg\Vasudev_Gupta\WISDEM\wisdem\commonse\utilization_dnvgl.py:322
The number of calls to function has reached maxfev = 50.ORBIT library intialized at '\\home.ansatt.ntnu.no\vasudevg\Vasudev_Gupta\WISDEM\wisdem\library'
########################################
Objectives
Turbine AEP: 77.9037579237 GWh
Blade Mass:  68206.4068005262 kg
LCOE:        84.4284950361 USD/MWh
Tip Defl.:   25.8318262264 m
########################################
Completed in, 34.94051551818848 seconds
"""