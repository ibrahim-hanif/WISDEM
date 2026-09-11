# %% [markdown]
# _bismillahi ArRahman ArRaheem_
# # `script` to validate analytical bearing loads with `pyFrame3DD`
#
# ### TODO:
# 1. `-Fz` works! why?

# %%
# imports
import os
import numpy as np
import matplotlib.pyplot as plt
# import scipy.io as sio # --- not used in here, but within imports
import csv
import pandas as pd
from scipy import stats

# %%
# import needed `WISDEM` modules
import openmdao.api as om

import wisdem.drivetrainse.drive_structure as ds
import wisdem.drivetrainse.drive_components as dc

from wisdem.commonse.utilities import get_recorder_results, mainshaft_loads_from_mat_to_dict, load_all_mat_to_dict, pdf_norm_int_using_cdf, bin_counting_of_load, compute_LRD, compute_LRD_matrix_vectorized
from wisdem.commonse.fileIO import var_df2dict
from Drive4Wind.utilities import utilities_drivetrain as utilsDT
from Drive4Wind.post_processing import analyseWTLoads, color_schemes
from Drive4Wind.utilities import funcs_errors

# Define plotting options
loc_clr_scheme_m4w = color_schemes.loc_clr_scheme_m4w
clrs_m4w = color_schemes.read_color_scheme(loc_clr_scheme_m4w)

params_plot_rc = {
        "font.size": 24,
        "axes.labelsize": 24,
        "legend.fontsize": 24, # 16 for pdf of `var_with_iter` plot
        "lines.linewidth": 2,
        "lines.markersize": 6,
    }
plt.rcParams.update( params_plot_rc )

#%%

suffix = "_sima"
# 1. "_m4w"
# 2. "_sima"

# paths / locations
results_dir = "00b_results"
script_dir = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(script_dir, results_dir)

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
# TODO new
if "sima" in suffix:
    loc_all_loads_mat_file = "C:\\SIMA_M4W_loads\\all_main_shaft_loads.mat"

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
# Plot hub load statistics
loc_hub_loads_stats = os.path.join(dir_loads, "hub_loads_M4W_stats.pdf")
if "sima" in suffix:
    loc_hub_loads_stats = os.path.join(results_path,
            "hub_loads" + suffix.upper() + "_stats.png")

analyseWTLoads.plot_ms_load_statistics(
    S_all,"blue","red", (15,15),
    # loc_save_plot=loc_hub_loads_stats
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
numTS = 100 # TODO

f = S_all['Fx'][startTS:startTS+numTS,iWSrated]
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
    Fz = -np.reshape(                          # TODO: works! why?
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
# ### Define type of analysis
#%%
# TODO: uncomment the desired analysis type
# anaString = "MomentReactingFrame_nonAnalyBeam"; analysis = 1
# anaString = "MomentReacting"; analysis = 2
anaString = "nonMomentReacting"; analysis = 3

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
G_lss = eval( var_dict['lss_G'] )

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
# -- bearing torsional stiffness (Nm/rad)
k_torsional = eval( var_dict['mb_fls.k_mb2'] ) - 6e8 # 3.e10 - 6e8 = 2.94e10 
if analysis != 2: k_torsional *= 0
# lambda
lam = (k_torsional*L_12)/(3*EI); print(f" -- lam = {lam}")
lamL = lam*L_12; print(f" -- lamL = {lamL}")
LonePlusLam = L_12*(1+lam); print(f" -- L_12(1+lam) = {LonePlusLam}")

F_mb1_beam, F_mb2_beam, M_mb2_beam = ds.analytical_MBforces_EBbeam(
    Fx,Fy,Fz, Mx,My,Mz, m_carrier, delta, tilt_rad, L_h1, L_12,
    EI, k_torsional, return_M=True
)
M_mb2_beam_norm = np.hypot(M_mb2_beam[0,0,:], M_mb2_beam[1,0,:])

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
                    n_dlcs=n_dlcs,
                    modeling_options=opt_drivese,
                    direct_drive=direct
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
# --- CRB
mb1_Reactions = prob['mb1_Reactions'] = eval( var_dict['mb1_Reactions'] )
# ----- finite stiffness (springs) and not rigid
# mb1_Reactions = prob['mb1_Reactions'] = 1e9 * np.array([3.53, 8.92, 1.25e1, 0.0, 1.21,8.62e-1])
# --- TRB2
mb2_Reactions = prob['mb2_Reactions'] = eval( var_dict['mb2_Reactions'] )
# ----- non-moment reacting (SRB)
if analysis == 3:
    mb2_Reactions = prob['mb2_Reactions'] = [1.0,1.0,1.0, 0.0,0.0,0.0]
# ----- finite stiffness (springs) and not rigid
# mb2_Reactions = prob['mb2_Reactions'] = 1e9 * np.array([3.39, 5.38, 8.78, 0.0, 5.92e-1, 3.62e-1])
# - materials
prob['lss_E'] = E_lss
prob['lss_G'] = G_lss
rho_lss = prob['lss_rho'] = eval( var_dict['lss_rho'] )
prob['lss_Xy'] = eval( var_dict['lss_Xy'] )
# - constrs
prob['shaft_deflection_allowable'] = eval( var_dict['shaft_deflection_allowable'] )
prob['shaft_angle_allowable'] = eval( var_dict['shaft_angle_allowable'] )

#%%
# Init outputs: loads on MBs
F_mb1_frame = np.zeros((4,numTS)) # x,y,z,rad
F_mb2_frame = np.zeros((4,numTS))
# M_mb1_frame = np.zeros((4,numTS)) # == 0
M_mb2_frame = np.zeros((4,numTS))
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
    M_mb2_frame[:3,iF] = prob['M_mb2'][:,0]
# - radial forces
F_mb1_frame[3,:] = np.hypot(F_mb1_frame[1,:], F_mb1_frame[2,:])
F_mb2_frame[3,:] = np.hypot(F_mb2_frame[1,:], F_mb2_frame[2,:])
# - M_norm
M_mb2_frame[3,:] = np.hypot(M_mb2_frame[1,:], M_mb2_frame[2,:])

#%%
# build_lss_pyframe3dd
import wisdem.pyframe3dd.pyframe3dd as frame3dd
from wisdem.drivetrainse.drive_structure import tube_prop
from wisdem.commonse import gravity

def build_lss_pyframe3dd(
        Fx,Fy,Fz, Mx,My,Mz, m_carrier, delta, tilt,
        L_h1,L_12, D_lss, t_lss,
        E,G,rho, mb1_Reactions,mb2_Reactions, RIGID=1
    ):
    """
    Inputs
    _____
    tilt : rad
    D_lss : "lss_diameter"
    t_lss : "lss_wall_thickness"
    E : "lss_E"
    G : "lss_G"
    rho : "lss_rho"
    mb1_Reactions : "mb1_Reactions"
    mb2_Reactions : "mb2_Reactions"
    
    Outputs
    _____
    results : from `pyframe3dd.run()`, with all data (forces, reactions etc.)
    """
    # -----------------------------
    # Geometry
    # -----------------------------
    x = np.array([0.0, L_h1, (L_h1+L_12), (L_h1+L_12+delta)])
    y = z = r = np.zeros_like(x)
    nnodes = len(x)

    # -----------------------------
    # Node data
    # -----------------------------
    inode = np.arange(1, nnodes+1)
    nodes = frame3dd.NodeData(inode, x, y, z, r)
    # - take out indices
    i1 = 2
    i2 = 3
    itorq = 4

    # -----------------------------
    # Elements (3 beam elements)
    # -----------------------------
    lsscyl = tube_prop(x, D_lss, t_lss)

    ielement = np.arange(1,nnodes) #elem = np.array([1, 2, 3])
    N1 = np.arange(1,nnodes)
    N2 = np.arange(2,nnodes+1)
    roll = np.zeros(nnodes - 1)
    myones = np.ones(nnodes - 1)
    # Section properties
    Ax = lsscyl.Area
    As = lsscyl.Asx
    S = lsscyl.S    #(v) bending modulus for tubular sections: line 114, cross_sections.py
    C = lsscyl.C    #(v) torsional shear constant for tubular sections: line 122, cross_sections.py
    J0 = lsscyl.J0  #(v) polar moment of inertia w.r.t. x-x axis (torsional)
    Jx = lsscyl.Ixx #(v) 2nd area moment of inertia w.r.t. y-y axis (Iyy=Izz for tubes)

    elements = frame3dd.ElementData(
            ielement, N1, N2, Ax, As, As, J0, Jx, Jx, E * myones, G * myones, roll, rho * myones
        )

    # -----------------------------
    # Boundary conditions
    # -----------------------------
    # DOF order: [Tx, Ty, Tz, Rx, Ry, Rz]; num = 6
    FREE = 0 # 1 = fixed, 0 = free
    rnode = np.r_[i1, i2, itorq] #r = np.zeros((nnodes, 6))
    Rx = np.array([mb1_Reactions[0], mb2_Reactions[0], FREE])  # (v, def) RIGID, FREE, FREE: Upwind bearing restricts translational
    Ry = np.array([mb1_Reactions[1], mb2_Reactions[1], FREE])  # (v, def) RIGID, FREE, FREE: Upwind bearing restricts translational
    Rz = np.array([mb1_Reactions[2], mb2_Reactions[2], FREE])  # (v, def) RIGID, FREE, FREE: Upwind bearing restricts translational
    Rxx = np.array([FREE, FREE, RIGID])  # (v, def) FREE, FREE, RIGID: Torque is absorbed by stator, so this is the best way to capture that
    Ryy = np.array([mb1_Reactions[4], mb2_Reactions[4], FREE])  # (v, def) FREE, RIGID, FREE: downwind bearing carry moments
    Rzz = np.array([mb1_Reactions[5], mb2_Reactions[5], FREE])  # (v, def) FREE, RIGID, FREE:  downwind bearing carry moments
    # print("LSS Bearing Reactions (Rx,Ry,Rz,Rxx,Ryy,Rzz): ", np.array((Rx,Ry,Rz,Rxx,Ryy,Rzz))) #(v) debugging
    reactions = frame3dd.ReactionData(rnode, Rx, Ry, Rz, Rxx, Ryy, Rzz, rigid=RIGID)
    

    # ------ options ------------
    shear = geom = True #(v) 1: include shear deformation + geom stiffness
    dx = 1.0
    options = frame3dd.Options(shear, geom, dx)
    # ----------------------------------- 

    # -----------------------------
    # Build frame
    # -----------------------------
    frame = frame3dd.Frame(
        nodes,
        reactions,
        elements,
        options
    )

    # -----------------------------
    # Loads
    # -----------------------------
    gy = 0.0
    gx = gravity * np.sin(tilt)
    gz = -gravity * np.cos(tilt)

    load = frame3dd.StaticLoadCase(0,0,0)
    # Hub loads at Node 1
    # GB carrier mass load at Node 4
    EL = np.array([1,4])
    # - F
    Fx_load = np.array([Fx, m_carrier*gx])
    Fy_load = np.array([Fy, m_carrier*gy])
    Fz_load = np.array([Fz, m_carrier*gz])
    # - M
    Mx_load = np.array([Mx, 0.0])
    My_load = np.array([My, 0.0])
    Mz_load = np.array([Mz, 0.0])

    load.changePointLoads(
        EL, Fx_load,Fy_load,Fz_load, Mx_load,My_load,Mz_load
    )
    frame.addLoadCase(load)

    # -----------------------------
    # Run analysis
    # -----------------------------
    # frame.write() # TODO
    displacements, forces, reactions, internalForces, mass3dd, modal = frame.run()

    return reactions

#%%
# Init outputs: loads on MBs
F_mb1_myframe = np.zeros((4,numTS)) # x,y,z,rad
F_mb2_myframe = np.zeros((4,numTS))
# M_mb1_myframe = np.zeros((4,numTS)) # == 0
M_mb2_myframe = np.zeros((4,numTS))
# Loop over hub loads
for iF in range(numTS):
    # loads
    iFx, iFy, iFz = Fx[0,iF], Fy[0,iF], Fz[0,iF]
    iMx, iMy, iMz = Mx[0,iF], My[0,iF], Mz[0,iF]
    # analyse
    reactions = build_lss_pyframe3dd(
        iFx,iFy,iFz, iMx,iMy,iMz,
        m_carrier,delta,tilt_rad,L_h1,L_12,lss_diameter,lss_wall_thickness,
        E_lss,G_lss,rho_lss,mb1_Reactions,mb2_Reactions,RIGID=1
    )
    # reactions on mbs
    k=0
    # - mb1
    F_mb1_myframe[:3,iF] = np.array([np.abs(reactions.Fx[k,0]), reactions.Fy[k, 0], reactions.Fz[k, 0]])
    # M_mb1_frame[:,iF] = prob['M_mb1'][:,0] # == 0
    # - mb2
    F_mb2_myframe[:3,iF] = np.array([np.abs(reactions.Fx[k, 1]), reactions.Fy[k, 1], reactions.Fz[k, 1]])
    M_mb2_myframe[:3,iF] = np.array([reactions.Mxx[k, 1], reactions.Myy[k, 1], reactions.Mzz[k, 1]])
# - radial forces
F_mb1_myframe[3,:] = np.hypot(F_mb1_myframe[1,:], F_mb1_myframe[2,:])
F_mb2_myframe[3,:] = np.hypot(F_mb2_myframe[1,:], F_mb2_myframe[2,:])
# - M_norm
M_mb2_myframe[3,:] = np.hypot(M_mb2_myframe[1,:], M_mb2_myframe[2,:])

#%%
# plot options
# --- main colors
loc_clr_scheme_m4w = color_schemes.loc_clr_scheme_m4w
clrs_m4w = color_schemes.read_color_scheme(loc_clr_scheme_m4w)
clr_Frame = "k" #clrs_m4w['Dark_Blue']
# --- line options
lineWidth_Frame = 3
lineStyle_Frame = 'dashed'
# --- labels
label_analyMB = " analytical: "
label_Frame = " structural solver: "
# ---- based on analysis: [ analyMB, pyFrame ]
if analysis == 1:
    lstAnaType = ["non-MR", "MR"]
    clr_Beam = "#0000FF" #clrs_m4w['Aqua']

elif analysis == 2:
    lstAnaType = ["MR"]*2
    clr_Beam = "#00FF00" #clrs_m4w['Red']

elif analysis == 3:
    lstAnaType = ["MR "+r"$(k_{\theta}=0)$","non-MR"]
    clr_Beam = "#7FFF00" #clrs_m4w['Red']
# -------------------------
# options: Journal polish
# plot rc params
params_plot_rc = {
        "font.size": 20,
        "axes.labelsize": 20,
        "legend.fontsize": 20, # 16 for pdf of `var_with_iter` plot
        "lines.linewidth": 4.5,
        "lines.markersize": 6,
    }
plt.rcParams.update( params_plot_rc )

#%%
# PUBLICATION plot and compare loads from analy_ and Hub_]
# F_mb1_beam.shape# = (4,1,numTS)
fig = plt.figure(figsize=(14,14))
gs = fig.add_gridspec(5, 2, hspace=0.35, wspace=0.25)
# ---- grid = mb1, mb2
#             [ ax,
#               y,
#               z,
#               rad,
#               M_norm ]
# ==== axial ====
# 0,0 = mb1
ax = fig.add_subplot(gs[0,0])
ax.plot(0, 0,
        label=label_analyMB+lstAnaType[0],
        color=clr_Beam )
ax.plot(0, 0,
        label=label_Frame+lstAnaType[1],
        color=clr_Frame, linestyle=lineStyle_Frame, linewidth=lineWidth_Frame)
ax.set_title("MB1 " + r"$(\times 10^6)$")
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel(r'$ F, ax $')
ax.legend(loc="center")

# 0,1 = mb2
maxFrame = 1e6 #np.max(F_mb2_myframe[0,:])
ax = fig.add_subplot(gs[0,1])
ax.plot( F_mb2_beam[0,0,:] / maxFrame,
         color=clr_Beam )
ax.plot( F_mb2_myframe[0,:] / maxFrame,
         color=clr_Frame, linestyle=lineStyle_Frame, linewidth=lineWidth_Frame )
# ax.legend()
ax.set_title("MB2 "+ r"$(\times 10^6)$")
ax.set_xticks([])
# ax.set_xlabel(r'$t$')

# ==== y ====
# 1,0 = mb1
# maxFrame = np.max(F_mb1_myframe[1,:])
ax = fig.add_subplot(gs[1,0])
ax.plot( F_mb1_beam[1,0,:] / maxFrame,
         color=clr_Beam )
ax.plot( F_mb1_myframe[1,:] / maxFrame,
         color=clr_Frame, linestyle=lineStyle_Frame, linewidth=lineWidth_Frame )
ax.set_ylabel(r"$ F, y $")
ax.set_xticks([])

# 1,1 = mb2
# maxFrame = np.max(F_mb2_myframe[1,:])
ax = fig.add_subplot(gs[1,1])
ax.plot( F_mb2_beam[1,0,:] / maxFrame,
         color=clr_Beam )
ax.plot( F_mb2_myframe[1,:] / maxFrame,
         color=clr_Frame, linestyle=lineStyle_Frame, linewidth=lineWidth_Frame )
# ax3.legend()
# ax.set_title(r"$F_{y}^{mb2}$")
# ax.set_xlabel(r"$t$")
ax.set_xticks([])

# ==== z ====
# 2,0 = mb1
# maxFrame = np.max(F_mb1_myframe[2,:])
ax = fig.add_subplot(gs[2,0])
ax.plot( F_mb1_beam[2,0,:] / maxFrame,
         color=clr_Beam )
ax.plot( F_mb1_myframe[2,:] / maxFrame,
         color=clr_Frame, linestyle=lineStyle_Frame, linewidth=lineWidth_Frame )
# ax3.legend()
ax.set_ylabel(r"$ F, z$")
# ax.set_xlabel(r"$t$")
ax.set_xticks([])

# 2,1 = mb2
# maxFrame = np.max(F_mb2_myframe[2,:])
ax = fig.add_subplot(gs[2,1])
ax.plot( F_mb2_beam[2,0,:] / maxFrame,
         color=clr_Beam )
ax.plot( F_mb2_myframe[2,:] / maxFrame,
         color=clr_Frame, linestyle=lineStyle_Frame, linewidth=lineWidth_Frame )
# ax3.legend()
# ax.set_title(r"$F_{z}^{mb2}$")
# ax.set_xlabel(r"$t$")
ax.set_xticks([])

# ==== radial ====
# 3,0 = mb1
# maxFrame = np.max(F_mb1_myframe[3,:])
ax1 = fig.add_subplot(gs[3,0])
ax1.plot( F_mb1_beam[3,0,:] / maxFrame,
         color=clr_Beam )
ax1.plot( F_mb1_myframe[3,:] / maxFrame,
         color=clr_Frame, linestyle=lineStyle_Frame, linewidth=lineWidth_Frame )
# ax1.legend()
ax1.set_ylabel(r"$ F, rad$")
ax1.set_xticks([])
# ax1.set_xlabel(r'$t$')

# 3,1 = mb2
# maxFrame = np.max(F_mb2_myframe[3,:])
ax1 = fig.add_subplot(gs[3,1])
ax1.plot( F_mb2_beam[3,0,:] / maxFrame,
         color=clr_Beam )
ax1.plot( F_mb2_myframe[3,:] / maxFrame,
         color=clr_Frame, linestyle=lineStyle_Frame, linewidth=lineWidth_Frame )
# ax1.legend()
# ax1.set_ylabel(r"$ F, rad$")
ax1.set_xticks([])

# ==== M_norm ====
# 4,0 = mb1
ax = fig.add_subplot(gs[4,0])
# ax.set_title("MB1")
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel(r'$ M, norm $')
ax.set_xlabel(r"$t$")

# 4,1 = mb2
# maxFrame = np.max(M_mb2_myframe[3,:])
ax = fig.add_subplot(gs[4,1])
ax.plot( M_mb2_beam_norm / maxFrame,
         label="EBbeam", color=clr_Beam )
ax.plot( M_mb2_myframe[3,:] / maxFrame,
         label="Frame", color=clr_Frame, linestyle=lineStyle_Frame, linewidth=lineWidth_Frame )
ax.set_xlabel(r"$t$")
ax.set_xticks([])

# ----------
fig.tight_layout()

plot_path = os.path.join(results_path,
                         "mbReactions_"+anaString+".png")
# plt.savefig(plot_path) # NOTE: saved, so don't change now 

plt.show()

#%%
# Error analysis (analy_MB_EBbeam & `pyFrame3DD`)
# (https://towardsdatascience.com/time-series-forecast-error-metrics-you-should-know-cc88b8c67f27/)

# NOTE: const diff
# F_mb1_rad
err_Fmb1_rad = F_mb1_beam[3,0,:] - F_mb1_myframe[3,:] # - 95600
mape_Fmb1_rad = funcs_errors.mapError( F_mb1_beam[3,0,:], F_mb1_myframe[3,:] )
print(
    f"F_mb1_rad | Error: max= {np.max( err_Fmb1_rad )}; map= {mape_Fmb1_rad}"
)

# F_mb2_ax
err_Fmb2_ax = F_mb2_beam[0,0,:] - F_mb2_myframe[0,:] # = 78546 (due to gravity loads each ele)
mape_Fmb2_ax = funcs_errors.mapError( F_mb2_beam[0,0,:], F_mb2_myframe[0,:] )
print(
    f"F_mb2_ax | Error: max= {np.max( err_Fmb2_ax )}; map= {mape_Fmb2_ax}"
)

# F_mb2_rad
err_Fmb2_rad = F_mb2_beam[3,0,:] - F_mb2_myframe[3,:] # = -1.0 * 1e6
mape_Fmb2_rad = funcs_errors.mapError( F_mb2_beam[3,0,:], F_mb2_myframe[3,:] )
print(
    f"F_mb2_rad | Error: max= {np.max( err_Fmb2_rad )}; map= {mape_Fmb2_rad}"
)

# M_mb2_norm
if mb2_Reactions[-1] > 0.0:
    err_Mmb2_rad = M_mb2_beam_norm - M_mb2_myframe[3,:]
    mape_Mmb2_rad = funcs_errors.mapError( M_mb2_beam_norm, M_mb2_myframe[3,:] )
    print(
        f"M_mb2_rad | Error: max= {np.max( err_Mmb2_rad )}; map= {mape_Mmb2_rad}"
    )
# %%[markdown]
# ===========================================
# ===========================================
#      tiliting stiffness fit for TRB2
# ===========================================
# ===========================================
#%%
bearing_rotatStiff_csv = os.path.join(
    os.path.abspath(__file__), os.pardir,os.pardir,
    "bearing_database", "TRB_with_rotat_stiffness.csv"
)
df_trb = pd.read_csv( bearing_rotatStiff_csv )

# %%
# ============================================================¨
# ====== COPILOT for TRB_with_rotat_stiffness.csv ============
# ============================================================

#%%
# plot the trends
plt.figure(figsize=(8,6))

plt.scatter(
    df_trb["d"],
    df_trb["k_yy"],
    s=80,
    label=r"$k_{yy}$",
    color="blue"
)

plt.scatter(
    df_trb["d"],
    df_trb["k_zz"],
    s=80,
    label=r"$k_{zz}$",
    color="red"
)

plt.yscale("log")

plt.xlabel("D [mm]")
plt.ylabel(r"$k_{\theta}$"+" [Nm/rad]")
plt.title("Database of TRB2 tilting stiffness")
plt.grid(True, which="both", alpha=0.3)
plt.legend(loc="lower right")
plt.tight_layout()

# -- save plot
loc_save_plot_bearing_rotatStiff = os.path.join(
    os.path.dirname(os.path.abspath(bearing_rotatStiff_csv)),
    "plot_rotat_stiff.png")
# plt.savefig(loc_save_plot_bearing_rotatStiff) # TODO 
# --
plt.show()

# %%
# fit the trends to 'd' with linregress
from scipy.stats import linregress

def fit_stiffness(df):

    x = np.log(df["d"])

    model = {}

    for col in ["k_yy", "k_zz"]:

        y = np.log(df[col])

        slope, intercept, r_value, _, _ = linregress(
            x,
            y
        )

        model[col] = {
            "a": np.exp(intercept),
            "b": slope,
            "r2": r_value**2
        }

    return model

def evaluate_stiffness(model, d):

    return {
        "k_yy":
            model["k_yy"]["a"] * d**model["k_yy"]["b"],

        "k_zz":
           model["k_zz"]["a"] * d**model["k_zz"]["b"]
    }

model = fit_stiffness(df_trb)

for key in ["k_yy", "k_zz"]:

    a = model[key]["a"]
    b = model[key]["b"]
    r2 = model[key]["r2"]

    print(
        f"{key}: "
        f"{a:.4e} * d^{b:.4f} "
        f"(R² = {r2:.4f})"
    )
#%%
# test the fit
idx = 0

row = df_trb.iloc[idx]

d_test = row["d"]

pred = evaluate_stiffness(
    model,
    d_test
)

print(f"\nd = {d_test}")

for col in ["k_yy","k_zz"]:

    actual = row[col]

    error = (
        100
        * (pred[col] - actual)
        / actual
    )

    print(
        f"{col}: ",
        f"actual={actual:.3e}",
        f"pred={pred[col]:.3e}",
        f"err={error:+.2f}"
    )

d_grid = np.linspace(
    df_trb["d"].min(),
    df_trb["d"].max(),
    500
)

pred_yy = (
    model["k_yy"]["a"]
    * d_grid**model["k_yy"]["b"]
)

pred_zz = (
    model["k_zz"]["a"]
    * d_grid**model["k_zz"]["b"]
)
#%%
# ===== plot the fit =====

plot_yANDz_stiffnesses = True # TODO

if plot_yANDz_stiffnesses:
    clr_yy = 'black'
    clr_yy_line = "blue"
    label_scat_yy = r"$k_{yy}$"+" data"
    label_pred_yy = r"$k_{yy}$"+" fit"
    plot_name = "plot_fit_rotat_stiff_XY" # NOTE: .png added later
else:
    clr_yy = 'black'
    clr_yy_line = "#00FF00"
    label_scat_yy = 'Database'
    label_pred_yy = "Model prediction"
    plot_name = "plot_fit_rotat_stiff" # NOTE: .png added later

# plot
plt.figure(figsize=(10,8))
# yy
plt.scatter(
    df_trb["d"][1:],    # TODO: saved plot with [1:] ----
    df_trb["k_yy"][1:], # ----
    s=200,
    color=clr_yy,
    marker="x",
    label = label_scat_yy
)

if plot_yANDz_stiffnesses:
    plt.scatter(
        df_trb["d"][1:],    # TODO: saved plot with [1:] ----
        df_trb["k_zz"][1:], # ----
        s=200,
        c="black",
        marker="o",
        label = r"$k_{zz}$"+" data"
    )

plt.plot(
    d_grid,
    pred_yy,
    "-",
    color=clr_yy_line,
    lw=5.0,
    label = label_pred_yy
)

if plot_yANDz_stiffnesses:
    plt.plot(
        d_grid,
        pred_zz,
        c="red",
        lw=5.0,
        label = r"$k_{zz}$"+" fit"
    )

plt.yscale("log")

if plot_yANDz_stiffnesses:
    plt.title(
        "Power-law fit of TRB2 tilting stiffness"
    )
else:
    plt.title(
        f"Power-law fit of TRB2 tilting stiffness \n R²={model['k_yy']['r2']:.3f}"
    )

plt.xlabel("D [mm]")
plt.ylabel(r"$k_{\theta}$"+" [Nm/rad]")
plt.grid(True, which="both", alpha=0.3)
plt.legend(loc="lower right")
plt.tight_layout()
# -- save plot
loc_save_plot_bearing_rotatStiff_fit = os.path.join(
    os.path.dirname(os.path.abspath(bearing_rotatStiff_csv)),
    plot_name + ".png")
# plt.savefig(loc_save_plot_bearing_rotatStiff_fit) # TODO 
# --
plt.show()

# %%
# ==================== NEW mutli-variate ========================
# NOTE:
# 1. uses (d,D,B,C,C0 → k_yy), so csv needs all the first arguments
# 2. model uses sklearn, from the 'base' environment, so much be activated

import sys
use_base = input("Are you using the 'base' venv with 'sklearn' installed? [y/N]: ")
if use_base.strip().lower() not in ("y", "yes"):
    print("Please activate the 'base' environment with sklearn installed and rerun.")
    sys.exit('exit')

try:
    from sklearn.linear_model import LinearRegression
except ImportError:
    print("sklearn is not available. Please activate the 'base' environment with sklearn installed.")
    sys.exit('exit')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
bearing_rotatStiff_csv = os.path.join(
    os.path.abspath(__file__), os.pardir,os.pardir,
    "bearing_database", "TRB_with_rotat_stiffness.csv"
)
df_trb = pd.read_csv( bearing_rotatStiff_csv )

#%%

def fit_log_model(df, target):

    X = np.column_stack(
        [
            np.log(df["d"]),            # inner diameter
            np.log(df["D"]),            # outer diameter
            np.log(df["B"]),            # width
            np.log(df["C_kN"]),         # (Cr) dynamic load capacity
            np.log(df["C0_kN"] + 1e-6), # (C0) static load capacity
        ]
    )

    y = np.log(df[target])

    reg = LinearRegression()
    reg.fit(X, y)

    return {
        "intercept": reg.intercept_,
        "coef": reg.coef_,
        "r2": reg.score(X, y),
        "target": target,
    }

def evaluate_model(
    model,
    d,
    D,
    B,
    C_kN,
    C0_kN,
):

    x = np.array(
        [
            np.log(d),
            np.log(D),
            np.log(B),
            np.log(C_kN),
            np.log(C0_kN + 1e-6),
        ]
    )

    log_y = (
        model["intercept"]
        + np.dot(model["coef"], x)
    )

    return np.exp(log_y)

def print_model(model):

    c = model["coef"]

    print(f"\nTarget: {model['target']}")
    print(f"R² = {model['r2']:.5f}")

    print(
        (
            "y = exp({:.4f}) "
            "* d^{:.4f}"
            "* D^{:.4f}"
            "* B^{:.4f}"
            "* C^{:.4f}"
            "* C0^{:.4f}"
        ).format(
            model["intercept"],
            c[0],
            c[1],
            c[2],
            c[3],
            c[4],
        )
    )

# ====== testing =======

mass_model = fit_log_model(
    df_trb,
    "mass_kg"
)
print_model(mass_model)

C_model = fit_log_model(
    df_trb,
    "C_kN"
)
print_model(C_model)

kyy_model = fit_log_model(
    df_trb,
    "k_yy"
)
print_model(kyy_model)

test_col = df_trb.iloc[0]
evaluate_model(
    kyy_model,
    d=test_col["d"],
    D=test_col["D"],
    B=test_col["B"],
    C_kN=test_col["C_kN"],
    C0_kN=test_col["C0_kN"]
)

from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_percentage_error

def loo_error(df, target):

    loo = LeaveOneOut()

    errors = []

    for train, test in loo.split(df):

        df_train = df.iloc[train]
        df_test = df.iloc[test]

        model = fit_log_model(
            df_train,
            target,
        )

        pred = evaluate_model(
            model,
            df_test["d"].values[0],
            df_test["D"].values[0],
            df_test["B"].values[0],
            df_test["C_kN"].values[0],
            df_test["C0_kN"].values[0],
        )

        actual = df_test[target].values[0]

        errors.append(
            abs(pred-actual)/actual
        )

    return 100*np.mean(errors)

print(
    "kyy CV error = ",
    loo_error(df_trb,"k_yy"),
    "%"
)


# %%
# sort for plotting
df_plot = df_trb.sort_values("d")

pred = []

for _, row in df_plot.iterrows():

    pred.append(
        evaluate_model(
            kyy_model,
            row["d"],
            row["D"],
            row["B"],
            row["C_kN"],
            row["C0_kN"],
        )
    )

pred = np.asarray(pred)

#%%
# plot true and predictions

plt.figure(figsize=(5,4))
# actual values
plt.scatter(
    df_plot["d"],
    df_plot["k_yy"],
    s=80,
    color='orange',
    label="Database"
)

# fitted values
plt.plot(
    df_plot["d"],
    pred,
    "-o",
    lw=2,
    label="Model prediction"
)

plt.yscale("log")

plt.xlabel("D [mm]")
plt.ylabel(r"$k_{\theta}$ [Nm/rad]")

plt.title(
    f"Multivariable fit\nR²={kyy_model['r2']:.3f}"
)

plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
# -- save plot
loc_plot_mb_k_theta_multivariate = os.path.join(
    os.path.dirname(os.path.abspath(bearing_rotatStiff_csv)),
    "plot_fit_rotat_stiff_multivariate.png")
# plt.savefig(loc_plot_mb_k_theta_multivariate) # TODO 
# --
plt.show()

#%%[markdown]
# # Uncertainty in MB `Cr` for `RBDO`
# ==================================================================

#%%
# setup

bearing_type = "TRB2" # TODO: TRB2, SRB

mbClass = dc.MainBearing_withDerivatives()
allMBprops = mbClass.BEARINGS

mbProps = allMBprops[ bearing_type ]


c = float(mbProps["c"])
m = float(mbProps["m"])

# ------------------------------------------------------------
# ============ 2015_Guo TRB2 Cr and X_Cr =====================
# ------------------------------------------------------------

path_guo_trb2 = os.path.join(
    os.path.dirname(os.path.abspath(bearing_rotatStiff_csv)),
    f"2015_Guo_{bearing_type}_bore1m_Cr.csv")

df = pd.read_csv(path_guo_trb2)

# ------------------------------------------------------------
# Empirical relation shown in the figure:
# C_emp(D) = 6579.9 D^0.8592
# ------------------------------------------------------------

df["C_emp_kN"] = c * (df["d"]/1e3)**m

df["X_Cr"] = df["C_kN"] / df["C_emp_kN"]

# Lognormal model for multiplicative model error
log_x = np.log(df["X_Cr"].to_numpy())
mu_ln = log_x.mean()
sigma_ln = log_x.std(ddof=1)

mean_xcr = np.exp(mu_ln + 0.5*sigma_ln**2)
std_xcr = np.sqrt(
    (np.exp(sigma_ln**2)-1)
    * np.exp(2*mu_ln + sigma_ln**2)
)
cov_xcr = std_xcr / mean_xcr

# ------------------------------------------------------------
# Verification plot: detected points over original image
# ------------------------------------------------------------
"""
fig, ax = plt.subplots(figsize=(12, 7))
ax.imshow(img)
ax.scatter(xp, yp, facecolors="none", edgecolors="red", s=100, linewidths=1.2)
for n, xx, yy in zip(df["number"], xp, yp):
    ax.text(xx+5, yy-5, str(n), fontsize=7, color="red")
ax.set_xlim(x0-20, x1+20)
ax.set_ylim(y_bottom+20, y_top-20)
ax.axis("off")
plt.show()

print(f"Detected {len(df)} blue measured points.")
print(f"CSV: {csv_path}")
print(f"Calibration CSV: {calib_path}")
print()
"""
print("TRB2 empirical relation:")
print("  C_emp(D) = 6579.9 * D^0.8592")
print()
print("Multiplicative uncertainty:")
print(f"  ln(X_Cr) ~ Normal(mu={mu_ln:.5f}, sigma={sigma_ln:.5f})")
print(f"  mean(X_Cr) = {mean_xcr:.5f}")
print(f"  std(X_Cr)  = {std_xcr:.5f}")
print(f"  CoV(X_Cr)  = {cov_xcr:.5f}")
print()
print(df[["number", "d", "C_kN", "X_Cr"]].round(4).to_string(index=False))

# %%[markdown]
# ### Uncertainty in Cr of TRB2
# %%

str_plot = f"plot_Guo{bearing_type}_mb_Cr_LogNormal_fit"

# ---------------------------------------------------
# Load database
# ---------------------------------------------------

# df = df_trb # TODO: if True: str_plot = plot_mb_Cr_LogNormal_fit

# ---------------------------------------------------
# Inputs
# ---------------------------------------------------

d_m = df["d"].values / 1000.0      # mm -> m
C_actual = df["C_kN"].values       # kN

# ---------------------------------------------------
# Empirical fit from plot
# ---------------------------------------------------

C_fit = c * d_m**m
df["C_emp_kN"] = C_fit

# ---------------------------------------------------
# Residual uncertainty factor
# ---------------------------------------------------

X_Cr = C_actual / C_fit
df["X_Cr"] = X_Cr

# ---------------------------------------------------
# Statistics
# ---------------------------------------------------

mu = np.mean(X_Cr)
sigma = np.std(X_Cr, ddof=1)
cov = sigma / mu

print("\nX_Cr statistics")
print("----------------")
print(f"mean = {mu:.4f}")
print(f"std  = {sigma:.4f}")
print(f"cov  = {cov:.4f}")

# ---------------------------------------------------
# Fit LogNormal
# ---------------------------------------------------

shape, loc, scale = stats.lognorm.fit(
    X_Cr,
    floc=0.0
)

print("\nLogNormal fit")
print("-------------")
print(f"shape = {shape:.6f}")
print(f"loc   = {loc:.6f}")
print(f"scale = {scale:.6f}")

# ---------------------------------------------------
# Plot
# ---------------------------------------------------

# -------------------------
# options: Journal polish
# plot rc params
params_plot_rc = {
        "font.size": 24,
        "axes.labelsize": 24,
        "legend.fontsize": 24, # 16 for pdf of `var_with_iter` plot
        "lines.linewidth": 4,
        "lines.markersize": 6,
    }
plt.rcParams.update( params_plot_rc )
# -------------------------


fig, axes = plt.subplots(
    1, 2,
    figsize=(15,7)
)

plt.suptitle(
    f"{bearing_type} dynamic load capacity: uncertainty quantification",
    # fontsize=16,
    y=0.95
    )

# ===================================================
# Histogram + fitted LogNormal
# ===================================================

ax = axes[0]

ax.hist(
    X_Cr,
    bins='auto',
    density=True,
    alpha=0.6,
    edgecolor='k',
    label="Uncertainty"
)

x_pdf = np.linspace(
    0.5*X_Cr.min(),
    1.5*X_Cr.max(),
    500
)

pdf_fit = stats.lognorm.pdf(
    x_pdf,
    shape,
    loc=loc,
    scale=scale
)

ax.plot(
    x_pdf,
    pdf_fit,
    'r-',
    # linewidth=3,
    label='Fit'
)

ax.axvline(
    mu,
    color='black',
    linestyle='--',
    # linewidth=3.0,
    label=f'Mean={mu:.2f},\n Std={sigma:.2f}'
)

ax.set_xlabel( r'$\chi_{Cr}$' )
ax.set_ylabel("PDF")
ax.set_title("LogNormal fit") # Residual Factor Distribution
ax.legend(loc="upper right")

# ===================================================
# Check Gaussianity in log-space
# ===================================================

ax = axes[1]

logX = np.log(X_Cr)

ax.hist(
    logX,
    bins=8,#'auto',
    density=True,
    alpha=0.6,
    edgecolor='k',
    # label="Uncertainty"
)

mu_log, sigma_log = stats.norm.fit(logX)

x_log = np.linspace(
    logX.min()*1.1,
    logX.max()*1.1,
    500
)

pdf_norm = stats.norm.pdf(
    x_log,
    mu_log,
    sigma_log
)

ax.plot(
    x_log,
    pdf_norm,
    'r-',
    # linewidth=3,
    # label='Fit'
)

ax.set_xlabel(r"$\log( \chi_{Cr} )$")
ax.set_ylabel("PDF")
ax.set_title("Log-Space Normal fit")
ax.legend(loc="upper right")

plt.tight_layout()

loc_plot_mb_Cr_random = os.path.join(
    os.path.dirname(os.path.abspath(bearing_rotatStiff_csv)),
    str_plot+".png")
# plt.savefig(loc_plot_mb_Cr_random) # TODO 

plt.show()

print(f"X_Cr ~ LogNormal(mean={mu:.3f}, std={sigma:.3f})")

# %%
# residual-vs-bore plot
# - recommended for the paper, iff nice looking and insightful

str_plot_XCrVSd = f"plot_Guo{bearing_type}_XCr_vs_d"

plt.figure(figsize=(8.5,8))

plt.scatter(
    d_m, X_Cr,
    s=80,
    label="Uncertainty"
    )

plt.axhline(
    mu,
    label="Mean",
    color="red",
    linestyle="--"
)

plt.xlabel("Bore diameter (D) [m]")
plt.ylabel(r"$\chi_{Cr}$"+" [-]")
plt.title( f"{bearing_type}: uncertainty vs. bore diameter",
          y=1.05 )
plt.legend()
plt.grid(True)

loc_plot_mb_XCr_vs_d = os.path.join(
    os.path.dirname(os.path.abspath(bearing_rotatStiff_csv)),
    str_plot_XCrVSd+".png")
# plt.savefig( loc_plot_mb_XCr_vs_d ) # TODO 

plt.show()

# %%
