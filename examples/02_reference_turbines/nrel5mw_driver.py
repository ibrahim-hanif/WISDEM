#!/usr/bin/env python3
import os

from wisdem import run_wisdem

## File management
mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
fname_wt_input = mydir + os.sep + "nrel5mw.yaml"
fname_modeling_options = mydir + os.sep + "modeling_options.yaml"
fname_analysis_options = mydir + os.sep + "analysis_options.yaml"

wt_opt, analysis_options, opt_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)
# end

#(v) fixed error: for below line(s)
# - raise KeyError('{}: Variable "{}" not found.'.format(self.msginfo, name))
# - KeyError: '<model> <class Group>: Variable "wt.tcc.blade_mass" not found.'
# checked outputs/refturb_output.csv -> keys are not wt.tcc... but just tcc. (so changed below)
print("blade mass:", wt_opt["tcc.blade_mass"])
print("blade moments of inertia:", wt_opt["drivese.blades_I"])
print("BRFM:", wt_opt["drivese.pitch_system.BRFM"])
print("hub forces:", wt_opt["drivese.F_hub"])
print("hub moments:", wt_opt["drivese.M_hub"])

#(v) OUTPUTS:
"""
==============
comp.wt.wt_rna
==============
NL: NLBGS 1 ; 4.06371192e+11 1
NL: NLBGS 2 ; 4760440.29 1.17145122e-05
NL: NLBGS 3 ; 0 0
NL: NLBGS Converged
######################################
Objectives
Turbine AEP: 23.9127972556 GWh
Blade Mass:  16419.8666989212 kg
LCOE:        51.0804383502 USD/MWh
Tip Defl.:   4.7295143980 m
########################################
Completed in, 33.4929256439209 seconds
blade mass: [16419.86669892]
blade moments of inertia: [35590815.60501074 17795407.80250537 17795407.80250537        0.
        0.                0.        ]
BRFM: [-16256766.58649734]
hub forces: [1171726.24681838    9709.60828864    9930.09366144]
hub moments: [11075137.65779353  1317752.66701242  1200194.31654884]
"""
