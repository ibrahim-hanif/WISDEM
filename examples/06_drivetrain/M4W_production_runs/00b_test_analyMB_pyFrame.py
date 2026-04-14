# %% [markdown]
# _bismillahi ArRahman ArRaheem_
# # `script` to **develop, test and deploy** 💪🏻
#
# ### TODO:
# 1. 

# %% [markdown]
# imports
import os
import numpy as np
import openmdao.api as om
import matplotlib.pyplot as plt
# import scipy.io as sio # --- not used in here, but within imports
import csv
import pandas as pd

# %%
# import needed `WISDEM` modules
# from wisdem.drivetrainse.drivetrain import DriveMaterials

# from wisdem.drivetrainse.hub import Hub_System
# from wisdem.drivetrainse.gearbox import Gearbox

# import wisdem.drivetrainse.layout as lay

# import wisdem.drivetrainse.drive_components as dc

import wisdem.drivetrainse.drive_structure as ds
import wisdem.drivetrainse.drive_components as dc

from wisdem.commonse.utilities import get_recorder_results, mainshaft_loads_from_mat_to_dict, load_all_mat_to_dict, pdf_norm_int_using_cdf, bin_counting_of_load, compute_LRD, compute_LRD_matrix_vectorized
from wisdem.commonse.fileIO import var_df2dict
import utilities_drivetrain as utilsDT

#%%
# paths / locations
results_dir = "00_results"
script_dir = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(script_dir, results_dir)
loc_save_data = os.path.join(results_path, "00")

# 02 results
results_02_dir = "02_results"
results_02_path = os.path.join(script_dir, results_02_dir)
loc_saved_02_data = os.path.join(results_02_path, "02newULS")

#%%
# load and read from saved csv file
flag_load_from_data = True

if flag_load_from_data: df_02results = pd.read_csv( loc_saved_02_data+".csv")

#%% Loading `openFAST` hub loads from a saved file
part_loads = True 
load_fls_loads = False
# False: full loads (72e4,10) (200 Hz sampled, 60mins)
# True: part loads (72e3,11) (20 Hz sampled, 60mins)

dir_loads = "M:\\Vasudev_Gupta\\outputs_mainshaft_loads"
loc_all_loads_mat_file = os.path.join(dir_loads, "hub_loads_M4W.mat")
loc_FLS_loads_mat_file = os.path.join(dir_loads, "mainshaft_loads_FLS_new.mat")
loc_ULS_loads_mat_file = os.path.join(dir_loads, "mainshaft_loads_ULS.mat")

if part_loads: # define paths
    S_all, keys_all = load_all_mat_to_dict(loc_all_loads_mat_file)

    F_uls = np.array( [S_all['Fx_max'], S_all['Fy_max'], S_all['Fz_max']] ).reshape((3, 1))
    M_uls = np.array( [S_all['Mx_max'], S_all['My_max'], S_all['Mz_max']] ).reshape((3, 1))

else: # define paths
    if load_fls_loads: # load from paths
        Snew, keys_new = mainshaft_loads_from_mat_to_dict(
            loc_FLS_loads_mat_file, loc_ULS_loads_mat_file)
        F_uls_full = np.array( [Snew['Fx_max'], Snew['Fy_max'], Snew['Fz_mean']] ).reshape((3, 1))
        M_uls_full = np.array( [Snew['Mx_max'], Snew['My_max'], Snew['Mz_max']] ).reshape((3, 1))

# %%
# Define plotting options
from my_util_tools import analyseWTLoads, util_funcs
loc_clr_scheme_m4w = util_funcs.loc_clr_scheme_m4w
clrs_m4w = util_funcs.read_color_scheme(loc_clr_scheme_m4w)

params_plot_rc = {
        "font.size": 24,
        "axes.labelsize": 24,
        "legend.fontsize": 24, # 16 for pdf of `var_with_iter` plot
        "lines.linewidth": 2,
        "lines.markersize": 6,
    }
plt.rcParams.update( params_plot_rc )
# %%
# Plot hub load statistics
loc_hub_loads_stats = os.path.join(dir_loads, "hub_loads_M4W_stats.pdf")

analyseWTLoads.plot_ms_load_statistics(
    S_all,clrs_m4w["Turquoise"],clrs_m4w["Aqua"], (15,15)
    ) 

#%%
# define `modelling_options`
opts = {}

opts["WISDEM"] = {}
opts["WISDEM"]["n_dlc"] = 1
opts["WISDEM"]["DriveSE"] = {}
# NOTE "hub": 'Hub_System' component are NOT included in the 'DrivetrainSE_M4W' component 
opts["WISDEM"]["DriveSE"]["hub"] = {}
opts["WISDEM"]["DriveSE"]["hub"]["hub_gamma"] = 2.0
opts["WISDEM"]["DriveSE"]["hub"]["spinner_gamma"] = 1.5

opts["WISDEM"]["DriveSE"]["direct"] = False
opts["WISDEM"]["DriveSE"]["use_gb_torque_density"] = True # False =(GB  optim, in-capabale)

opts["WISDEM"]["DriveSE"]["gamma_f"] = 1.35 #IEC-1, 7.6.2.2a, pg.57
opts["WISDEM"]["DriveSE"]["gamma_m"] = 1.3  #IEC-1, 7.6.2.4, pg.59
opts["WISDEM"]["DriveSE"]["gamma_n"] = 1.0  #IEC-1, 7.6.1.3, pg.55
# used as: gamma = gamma_f * gamma_m * gamma_n (within TODO)
# opts["WISDEM"]["DriveSE"]["nBins"] = 100    #used by (new) Analytical_FLS_Bearing_Life; =Number of bins for histogram MB FLS
# opts["WISDEM"]["DriveSE"]["own_hub_loads"] = False

opts["WISDEM"]["RotorSE"] = {}
opts["WISDEM"]["RotorSE"]["n_pc"] = 2 #cf. RPM_Input in drive_components.py
            # `n_pc`: Number of wind speeds to compute the power curve
opts["materials"] = {}
opts["materials"]["n_mat"] = 4

opts["flags"] = {}
dogen = opts["flags"]["generator"] = False
dohub = opts["flags"]["hub"] = False #(v)
doMBfls = opts["flags"]["mb_fls"] = True

opts["OpenFAST"] = {}
opts["OpenFAST"]["simulation"] = {}
opts["OpenFAST"]["simulation"]["DT"] = 0.05
# dir(ectory) where MS loads are stored .csv (?)
if part_loads:
    opts["OpenFAST"]["openfast_dir"] = loc_all_loads_mat_file
else:
    ValueError('Full loads not defined in openfast_dir<-OpenFAST<-modelling_options. Please define it first. jazakumAllahu khayr.')

opts["DLC_driver"] = {}
opts["DLC_driver"]["DLCs"] = [{}]
opts["DLC_driver"]["DLCs"][0]["DLC"] = "1.2"
opts["DLC_driver"]["DLCs"][0]["wind_speed"] = [ 5.,  7.,  9., 11., 13., 15., 17., 19., 21., 23.]
opts["DLC_driver"]["DLCs"][0]["probabilities"] = [0.06541262, 0.14245179, 0.14299681, 0.12940412, 0.10735197, 0.0824332, 0.05894909, 0.03942148, 0.02472593, 0.0083042]

opt_drivese = opts["WISDEM"]["DriveSE"]
# OpenFAST: containing 1. simulation DT and 2. MS loads dir
opt_openfast = opts["OpenFAST"]
# DLC: only 1 used '[0]': containing "wind_speed" and "probabilities"
opt_DLC = opts["DLC_driver"]["DLCs"][0]

#%%
# Define Loads
flag_loads_simple = False
iWSrated = 4 # index @ rated wind speed (10-11 m/s)
# iWSrated = [0,1,2] # prev
startTS = 1000
numTS = 10 # TODO

f = S_all['Fx'][startTS:startTS+numTS,4]
Fx = np.reshape(f,(1,numTS))
# segment loads for testing
if not flag_loads_simple:
    # Forces
    Fx = np.reshape(
        S_all['Fx'][startTS:startTS+numTS,4],
        (1,numTS)
        )
    Fy = np.reshape(
        S_all['Fy'][startTS:startTS+numTS,4],
        (1,numTS)
        )
    Fz = np.reshape(
        S_all['Fz'][startTS:startTS+numTS,4],
        (1,numTS)
        )
    # Moments
    Mx = np.reshape(
        S_all['Mx'][startTS:startTS+numTS,4],
        (1,numTS)
        )
    My = np.reshape(
        S_all['My'][startTS:startTS+numTS,4],
        (1,numTS)
        )
    Mz = np.reshape(
        S_all['Mz'][startTS:startTS+numTS,4],
        (1,numTS)
        )
    # OLD impl
    # Fx, Fy, Fz = S_all['Fx'][:3,4], S_all['Fy'][:3,iWSrated], S_all['Fz'][:3,iWSrated]
    # Mx, My, Mz = S_all['Mx'][:3,iWSrated], S_all['My'][:3,iWSrated], S_all['Mz'][:3,iWSrated]

# test forces
else:
    myForces = np.ones((2,2)) * 5 #(3,2)
    Fx, Fy, Fz = myForces, myForces, myForces
    Mx, My, Mz = myForces, myForces, myForces

#%%[markdown]
# ### Compare mb* loads (F,M) btw analy_*(s) (and Hub_* `pyFrame3DD`)
# -----------------------------------------------------------------
#%%
# define constants and parameters from saved file `df_02results`
var_dict = var_df2dict( df_02results )

# - overall
machine_rating = eval( var_dict['machine_rating'] )*1e3 # W
# - lss
L_h1 = eval( var_dict['L_h1'] ) # test: 2; converg: 0.2550491141298319
L_12 = eval( var_dict['L_12'] ) # test: 5; converg: 7.998456970161276
# - drivetrain
tilt_rad = np.deg2rad(
    eval( var_dict['tilt'] )
)
delta = eval( var_dict['delta'] )
m_carrier = eval( var_dict['carrier_mass'] )
# - materials
E_lss = eval( var_dict['lss_E'] ) # Pa = 1 N/m^2

# %% F_* computation
Fmb1, Fmb2, dFmb1dLh1, dFmb1dL12, dFmb2dLh1, dFmb2dL12 = ds.analytical_MB_Forces(
    Fx,Fy,Fz,Mx,My,Mz,L_h1,L_12, flag_jac=True)

#%%
Fmb1_real, Fmb2_real = ds.analytical_MBforces_realistic(
    Fx,Fy,Fz,Mx,My,Mz,
    m_carrier, delta, tilt_rad,
    L_h1,L_12, flag_jac=False)

# %%[markdown]
# ### Bearing loads for moment-reacting bearing
# %%
s_lss = np.array( eval( var_dict['s_lss'] ) )
lss_diameter = np.array( eval( var_dict['lss_diameter'] ) )
lss_wall_thickness = np.array( eval( var_dict['lss_wall_thickness'] ) )
bear2_Dshaft = eval( var_dict['Dshaft_mb2'] )
bear2_t = eval( var_dict['Tshaft_mb2'] )
#
# mb2's lss tube section
from wisdem.commonse.cross_sections import Tube
lssMB2section = Tube(bear2_Dshaft,bear2_t)
I_lss = lssMB2section.Iyy # m^2
# bending stiffness EI
EI = E_lss*I_lss
# -- bearing torsional stiffness
k_torsional = eval( var_dict['mb_fls.k_mb2'] ) # 3.e10 Nm/rad
# k_torsional = 5.e1
# lambda
lam = (k_torsional*L_12)/(3*EI)

F_mb1_beam, F_mb2_beam, M_mb2_beam = ds.analytical_MBforces_EBbeam(
    Fx,Fy,Fz, Mx,My,Mz, m_carrier, delta, tilt_rad, L_h1, L_12,
    EI, k_torsional, return_M=True
)
#%%
# compare with Hub_* (NOTE: below is 1 / 1e6)
"""
F_mb1 = array([[ -0.        ],
       [-10.93421809],
       [  3.47106296]])

F_mb2 = array([[  2.72010188],
       [ 11.25594415],
       [-10.52081853]])

M_mb2 = array([[-0.        ],
       [12.45775789],
       [13.42588477]])
"""

# F
Fx_uls, Fy_uls, Fz_uls = F_uls
Fx_uls = np.reshape(Fx_uls,(1,1))
Fy_uls = np.reshape(Fy_uls,(1,1))
Fz_uls = np.reshape(Fz_uls,(1,1))
# M
Mx_uls, My_uls, Mz_uls = M_uls
Mx_uls = np.reshape(Mx_uls,(1,1))
My_uls = np.reshape(My_uls,(1,1))
Mz_uls = np.reshape(Mz_uls,(1,1))
# 
F_mb1_beam_uls, F_mb2_beam_uls = ds.analytical_MBforces_EBbeam(
    Fx_uls,Fy_uls,Fx_uls, Mx_uls,My_uls,Mz_uls,
    m_carrier, delta, tilt_rad, L_h1, L_12,
    EI, k_torsional
)
M_mb2_beam_uls = lam*L_12*F_mb1_beam_uls
#%%
print("F_mb1_beam_uls: ", F_mb1_beam_uls/1e6)
print("F_mb2_beam_uls: ", F_mb2_beam_uls/1e6)
print("M_mb2_beam_uls: ", M_mb2_beam_uls/1e6)

#%%[markdown]
# ### `pyFrame3DD` computation for mb* loads
#%%
#%%
# def prob model `Hub_Rotor_LSS_Frame`
class test_Hub_Rotor_LSS_Frame( om.Group ):
    """
    Group containing components for the layout of the LSS components
    """
    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        opt_drivese = self.options["modeling_options"]["WISDEM"]["DriveSE"]
        n_dlcs = self.options["modeling_options"]["WISDEM"]["n_dlc"]
        direct = opt_drivese["direct"]
        opt_openfast = self.options["modeling_options"]["OpenFAST"]

        # Hub_Rotor_LSS_Frame:
        self.add_subsystem(
            "lss", ds.Hub_Rotor_LSS_Frame(
                    n_dlcs=n_dlcs, modeling_options=opt_drivese,
                    direct_drive=direct, openfast_options=opt_openfast
                ),
                promotes=["*"]
            )
        
prob = om.Problem(reports=False)
prob.model = test_Hub_Rotor_LSS_Frame(modeling_options=opts)
prob.setup()
prob.model.list_inputs();

# parse needed inputs
# - overall
prob['upwind'] = eval( var_dict['upwind'] )
# - rotor system
prob['hub_system_mass'] = eval( var_dict['hub_system_mass'] )
prob['hub_system_cm'] = eval( var_dict['hub_system_cm'] )
prob['hub_system_I'] = eval( var_dict['hub_system_I'] )
prob['blades_mass'] = eval( var_dict['blades_mass'] )
prob['blades_cm'] = eval( var_dict['blades_cm'] )
prob['blades_I'] = eval( var_dict['blades_I'] )
prob['s_rotor'] = eval( var_dict['s_rotor'] )
# - drivetrain
prob['tilt'] = eval( var_dict['tilt'] )
prob['s_lss'] = s_lss
prob['lss_diameter'] = lss_diameter
prob['lss_wall_thickness'] = lss_wall_thickness
prob['s_mb1'] = eval( var_dict['s_mb1'] )
prob['s_mb2'] = eval( var_dict['s_mb2'] )
prob['generator_rotor_mass'] = eval( var_dict['generator_rotor_mass'] )
prob['generator_rotor_I'] = eval( var_dict['generator_rotor_I'] )
prob['gearbox_mass'] = eval( var_dict['gearbox_mass'] )
prob['gearbox_I'] = eval( var_dict['gearbox_I'] )
prob['brake_mass'] = eval( var_dict['brake_mass'] )
prob['brake_I'] = eval( var_dict['brake_I'] )
prob['carrier_mass'] = m_carrier
prob['carrier_I'] = eval( var_dict['carrier_I'] )
prob['mb1_face_width'] = eval( var_dict['mb1_face_width'] )
prob['mb2_face_width'] = eval( var_dict['mb2_face_width'] )
prob['mb1_Reactions'] = eval( var_dict['mb1_Reactions'] )
prob['mb2_Reactions'] = eval( var_dict['mb2_Reactions'] )
# - materials
prob['lss_E'] = E_lss
prob['lss_G'] = eval( var_dict['lss_G'] )
prob['lss_rho'] = eval( var_dict['lss_rho'] )
prob['lss_Xy'] = eval( var_dict['lss_Xy'] )
# - constrs
prob['shaft_deflection_allowable'] = eval( var_dict['shaft_deflection_allowable'] )
prob['shaft_angle_allowable'] = eval( var_dict['shaft_angle_allowable'] )

#%%
# Init outputs: loads on MBs
F_mb1_frame = np.zeros((4,numTS)) # x,y,z,rad
F_mb2_frame = np.zeros((4,numTS))
# M_mb1_frame = np.zeros((3,numTS)) # == 0
M_mb2_frame = np.zeros((3,numTS))
# Loop over hub loads
for iF in range(numTS):
    # loads
    prob['F_aero_hub'] = np.array((
        Fx[0,iF], Fy[0,iF], Fz[0,iF]
    ))
    prob['M_aero_hub'] = np.array((
        Mx[0,iF], My[0,iF], Mz[0,iF]
    ))
    # analyse
    prob.run_model()
    # outputs
    # - mb1
    F_mb1_frame[:3,iF] = prob['F_mb1'][:,0]
    # M_mb1_frame[:,iF] = prob['M_mb1'][:,0] # == 0
    # - mb2
    F_mb2_frame[:3,iF] = prob['F_mb2'][:,0]
    M_mb2_frame[:,iF] = prob['M_mb2'][:,0]
# - radial forces
F_mb1_frame[3,:] = np.hypot(F_mb1_frame[1,:], F_mb1_frame[2,:])
F_mb2_frame[3,:] = np.hypot(F_mb2_frame[1,:], F_mb2_frame[2,:])

#%%
# plot options
# main colors
from my_util_tools import util_funcs
loc_clr_scheme_m4w = util_funcs.loc_clr_scheme_m4w
clrs_m4w = util_funcs.read_color_scheme(loc_clr_scheme_m4w)
clr_Frame = clrs_m4w['Dark_Blue']
clr_Beam = clrs_m4w['Aqua']
# -------------------------
# options: Journal polish
# plot rc params
params_plot_rc = {
        "font.size": 24,
        "axes.labelsize": 24,
        "legend.fontsize": 24, # 16 for pdf of `var_with_iter` plot
        "lines.linewidth": 3,
        "lines.markersize": 6,
    }
plt.rcParams.update( params_plot_rc )

#%%
# plot and compare loads from analy_ and Hub_]
# F_mb1_beam.shape# = (4,1,numTS)
fig = plt.figure(figsize=(40, 6))
gs = fig.add_gridspec(1, 5, hspace=0.35, wspace=0.25)
# ---- grid = [ mb1 rad, mb2: ax, rad, My, Mz ]
# ===== mb1 =====
# ----- [0] = rad
ax1 = fig.add_subplot(gs[0])
ax1.plot( F_mb1_beam[3,0,:],
         label="EBbeam", color=clr_Beam )
ax1.plot( F_mb1_frame[3,:],
         label="Frame", color=clr_Frame )
ax1.legend()
ax1.set_title(r"$F_{ax}^{mb1}$")
ax1.set_xticks([])
ax1.set_xlabel(r'$t$')
# ===== mb2 =====
# ----- [1] = x
ax2 = fig.add_subplot(gs[1])
ax2.plot( np.abs(F_mb2_beam[0,0,:]),
         label="EBbeam", color=clr_Beam )
ax2.plot( np.abs(F_mb2_frame[0,:]),
         label="Frame", color=clr_Frame )
# ax2.legend()
ax2.set_title(r"$F_{ax}^{mb2}$")
ax2.set_xticks([])
ax2.set_xlabel(r'$t$')
# ----- [2] = rad
ax3 = fig.add_subplot(gs[2])
ax3.plot( F_mb2_beam[3,0,:],
         label="EBbeam", color=clr_Beam )
ax3.plot( F_mb2_frame[3,:],
         label="Frame", color=clr_Frame )
# ax3.legend()
ax3.set_title(r"$F_{rad}^{mb2}$")
ax3.set_xlabel(r"$t$")
ax3.set_xticks([])
# ----- [3] = My
ax4 = fig.add_subplot(gs[3])
ax4.plot( M_mb2_beam[0,0,:],
         label="EBbeam", color=clr_Beam )
ax4.plot( M_mb2_frame[1,:],
         label="Frame", color=clr_Frame )
# ax4.legend()
ax4.set_title(r"$M_{y}^{mb2}$")
ax4.set_xlabel(r"$t$")
ax4.set_xticks([])
# ----- [4] = Mz
ax5 = fig.add_subplot(gs[4])
ax5.plot( M_mb2_beam[1,0,:],
         label="EBbeam", color=clr_Beam )
ax5.plot( M_mb2_frame[2,:],
         label="Frame", color=clr_Frame )
# ax4.legend()
ax5.set_title(r"$M_{z}^{mb2}$")
ax5.set_xlabel(r"$t$")
ax5.set_xticks([])

# ----------
fig.tight_layout()

plt.show()

# NOTE: const diff
# F_mb1_beam[3,0,:] - F_mb1_frame[3,:] # ~ 7.6 * 1e6
# F_mb2_beam[0,0,:] - F_mb2_frame[0,:] # = 446457 or 390900
# F_mb2_beam[3,0,:] - F_mb2_frame[3,:] # = -1.0 * 1e6
# %%
