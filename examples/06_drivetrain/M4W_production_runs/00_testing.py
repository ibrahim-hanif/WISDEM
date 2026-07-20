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
from Drive4Wind.utilities import utilities_drivetrain as utilsDT

#%%
# paths / locations
results_dir = "00_results"
script_dir = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(script_dir, results_dir)
loc_save_data = os.path.join(results_path, "00")

# 02 results
results_02_dir = "02_results"
results_02_path = os.path.join(script_dir, results_02_dir)
loc_saved_02_data = os.path.join(results_02_path, "02_m4w")

#%%
# load and read from saved csv file
flag_load_from_data = True

if flag_load_from_data:
    df_02results = pd.read_csv( loc_saved_02_data+".csv")
    var_dict = var_df2dict( df_02results )

#%% Loading `openFAST` hub loads from a saved file
part_loads = True 
load_fls_loads = False
# False: full loads (72e4,10) (200 Hz sampled, 60mins)
# True: part loads (72e3,11) (20 Hz sampled, 60mins)

dir_loads = "M:\\Vasudev_Gupta\\outputs_mainshaft_loads"
loc_all_loads_mat_file = os.path.join(dir_loads, "hub_loads_M4W.mat")
loc_all_loads_mat_file = "C://SIMA_M4W_loads//all_main_shaft_loads.mat" # TODO: sima loads

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
from Drive4Wind.post_processing import analyseWTLoads, color_schemes
from Drive4Wind.utilities import funcs_errors

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
# %%
# Plot hub load statistics
loc_hub_loads_stats = os.path.join(dir_loads, "hub_loads_M4W_stats.pdf")

analyseWTLoads.plot_ms_load_statistics(
    S_all,clrs_m4w["Turquoise"],clrs_m4w["Aqua"], (15,15)
    ) 

# %%
# wind speed probability
ws_full = S_all['mean_wind_speed']; n_w = ws_full.shape[1]
ws = ws_full[0,:n_w]
ws = np.append(ws_full,25.0)
# ws pdf computation
coeff_weibull = (1.95, 11.6)
pdf_ws = pdf_norm_int_using_cdf( ws, coeff_weibull )
pdf_ws_full = pdf_norm_int_using_cdf( ws_full, coeff_weibull )
# print to screen
print(f"ws = {ws}" )
print(f"pdf_ws = {pdf_ws}; sum={np.sum(pdf_ws)}" )

print(f"\nws_full = {ws_full}" )
print(f"pdf_ws_full = {pdf_ws_full}; sum={np.sum(pdf_ws_full)}" )

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
opts["WISDEM"]["DriveSE"]["nBins"] = 100    #used by (new) Analytical_FLS_Bearing_Life; =Number of bins for histogram MB FLS
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

# %%
# P_* computation
P = Fmb1[3,:,:]
n_t, n_w = P.shape[0], P.shape[1]
time = S_all['Time'][:n_t,0]; dt = 0.05
omega = S_all['rot_speed'][:n_t,:n_w]
p = 10/3

dP_dLh1 = dFmb1dLh1[3,:,:]
dP_dL12 = dFmb1dL12[3,:,:]

#%%
# P_eq (LDD, LRD, DEL) computation
P_LDD = bin_counting_of_load( P, ws, pdf_ws )
print(f"P_LDD = {P_LDD}")

P_LRD = compute_LRD_matrix_vectorized( P, dt, omega, pdf_ws, 10/3, 3)
print(f"P_LRD = {P_LRD}")

DEL = ds.del_bearing_computation( P, ws, dt, omega, pdf_ws, p )
print(f"DEL = {DEL}")

#%%
# P=10; ws=np.array([[10]]); dt=0.1
# omega=np.array([[60]]); pdf_ws=np.array([[1]]); p=2; dP_dLh1=np.array([[1]])

DEL, dDEL_dLh1 = ds.del_bearing_computation( P, ws, dt, omega, pdf_ws,
                    p, dP_dLh1 )
print(DEL, dDEL_dLh1)
_, dDEL_dL12 = ds.del_bearing_computation( P, ws, dt, omega, pdf_ws,
                    p, dP_dL12 )
print(dDEL_dL12)

#%%[markdown]
# ## === test `om._Component` ===
# =====================================================================
#%% # MainBearing_withDerivatives
prob = om.Problem( )
prob.model.add_subsystem( 'mb', dc.MainBearing_withDerivatives(), promotes=['*'] )
prob.setup(force_alloc_complex=True)
prob["bearing_type"] = "TRB2"
prob["D_shaft"] = 3.0
prob["mb_e"] = 0.35

prob.check_partials( compact_print=True, method='cs' );

prob.run_model()
#%%[markdown]
# ### Test: Bearing Life Module `Analytical_FLS_Bearing_Life` 

#%%[markdown]
# ### problem: setup and check partials
#%%
prob = om.Problem(reports=False)

prob.model = om.Group()
prob.model.add_subsystem("mb_fls",
    ds.Analytical_FLS_Bearing_Life(
        modeling_options=opt_drivese,
        openfast_options=opt_openfast,
        dlc_options=opt_DLC
        ),
    promotes=['*'])
# prob.model.add_subsystem("bear1", ds.MainBearing())
# prob.model.add_subsystem("bear2", ds.MainBearing())
# connections

prob.setup()

#%%
prob.check_partials( compact_print=True );

#%%[markdown]
# ### problem with derivatives: setup and check partials
#%%
prob_deri = om.Problem(reports=False)
model_deri = prob_deri.model = om.Group()
model_deri.add_subsystem(
    "mb_fls_deri",
    ds.Analytical_FLS_Bearing_Life_Derivatives(
        modeling_options=opt_drivese,
        openfast_options=opt_openfast,
        dlc_options=opt_DLC
    ),
    promotes=['*']
)

prob_deri.setup()
prob_deri.check_partials(compact_print=True);

#%% Loads assignment (ULS, FLS)

# ULS load loads (xD), input to Analy_*
# - NOTE: these are predscribed 50-yr extremes from extr DLCs (5.1,6.1,6.3) 
# prob["F_aero_hub"] = np.array([5.3995*1e6, 1.3697*1e6, 5.5742*1e6]).reshape((3, 1))
# prob["M_aero_hub"] = np.array([5.2515*1e7, 1.0747*1e8, 9.9481*1e7]).reshape((3, 1))
# TODO: change here for testing

# 1. partial loads (S_all)
# if part_loads:
#     prob['F_aero_hub'] = np.array( [S_all['Fx_max'], S_all['Fy_max'], S_all['Fz_max']] ).reshape((3, 1))
#     prob['M_aero_hub'] = np.array( [S_all['Mx_max'], S_all['My_max'], S_all['Mz_max']] ).reshape((3, 1))
# # 2. full loads (Snew)
# else:
#     prob['F_aero_hub'] = np.array( [Snew['Fx_max'], Snew['Fy_max'], Snew['Fz_mean']] ).reshape((3, 1))
#     prob['M_aero_hub'] = np.array( [Snew['Mx_max'], Snew['My_max'], Snew['Mz_max']] ).reshape((3, 1))

# FLS load loads (xD), input to Analy_*; (72e4, 10)
# TODO: change here for testing
# 1. partial loads (S_all)
if load_fls_loads:
    if part_loads:
        prob['Fx_FLS'],prob['Fy_FLS'],prob['Fz_FLS'] = S_all['Fx'],S_all['Fy'],S_all['Fz']
        prob['Mx_FLS'],prob['My_FLS'],prob['Mz_FLS'] = S_all['Mx'],S_all['My'],S_all['Mz']
        prob['rot_speed'] = S_all['rot_speed']
        prob['mean_wind_speed'] = S_all['mean_wind_speed'][0]
        prob['Time'] = S_all['Time'][0]
    # 2. full loads (Snew)
    else:
        prob['Fx_FLS'],prob['Fy_FLS'],prob['Fz_FLS'] = Snew['Fx'],Snew['Fy'],Snew['Fz']
        prob['Mx_FLS'],prob['My_FLS'],prob['Mz_FLS'] = Snew['Mx'],Snew['My'],Snew['Mz']
        prob['rot_speed'] = Snew['rot_speed']
        prob['mean_wind_speed'] = Snew['ws']
        prob['Time'] = Snew['Time']

# TODO: make nice PPT with flow/chart of om.Problem here
# TODO: update using pCrunch's rainflow
# ---

#%%
prob['L_h1'] = 0.5
prob['L_12'] = 7.0
prob['rated_rpm'] = 7.56
prob['lifetime'] = 20.0

# Cr (from Main Bearing component)
D_shaft = 4.0  # m
prob['Cr_mb1'] = Cr_CRB = (4526.5 * D_shaft ** 0.9556) *1e3  # N
prob['Cr_mb2'] = Cr_2TRB = (6579.9 * D_shaft**0.8592) *1e3  # N

print(f"Cr_CRB: {Cr_CRB*1e-6} MN, Cr_2TRB: {Cr_2TRB*1e-6} MN")
# %%
print("\n=== Final input check ===\n")
prob.model.list_inputs();
prob.model.list_outputs();
# %%
prob.run_model()

print("=== Results: Bearing Life Module ===")
print(f"constr_L10_mb1: {prob['constr_L10_mb1']}")
print(f"constr_L10_mb2: {prob['constr_L10_mb2']}")

# %%[markdown]
# ### Test: simple optimization wrt. L_h1 and L_12
#%%
flag_opt = True        # GBO: gradient based optimizer
# flag_opt = False        # GBO: gradient based optimizer
# ---
if flag_opt:
    print("=== running GBO ===")
    # Choose the (GBO) optimizer to use
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"
    prob.driver.options["tol"] = 1e-4 # comment to default (1e-6?)
    prob.driver.options["maxiter"] = 5 * 4
    prob.driver.options["disp"] = True
    prob.driver.options["debug_print"] = ["desvars", "objs", "nl_cons", "ln_cons"]
    
    # Add design variables
    prob.model.add_design_var("L_h1", lower=0.1, upper=5.0)
    prob.model.add_design_var("L_12", lower=0.1, upper=5.0)
    # Add objective (dummy)

#%%
# import standard normal distribution
from scipy.stats import norm
beta = 1.28
Pf = norm.cdf(-beta)
Pf

# %%[markdown]
# ### Analytical derivatives (using `JAX`) and time compr to analytical
# =====================================================================
#%%
import jax.numpy as jnp
from jax import jacfwd
import jax
import time
#%%
# generate a random number using `JAX`'s pure functional RNG
keyRNG = jax.random.PRNGKey(42) # seed = 42
X = jax.random.uniform(keyRNG,shape=(2,2)); dims = (2,2)
# X = jax.random.uniform(keyRNG,shape=(72000,11)); dims = (72000,11)
#%%
def f(X):
    return jnp.sum(X, axis=0)  # sum columns

# X = jnp.array([[1., 2.], [3., 4.]])
t0_jax = time.time()
J = jacfwd(f)(X)
t1_jax = time.time()
print(f"JAX, shape={dims}, in time {t1_jax-t0_jax}s") # J = {J}, 
# Shape: (2, 2, 2) because JAX returns per-element derivative
# You can reshape to (n, m*n) if needed

# %%
def jacobian_sum_columns(m, n):
    # Jacobian shape: (n, m*n)
    J = np.zeros((n, m*n))
    for col in range(n):
        start = col * m
        J[col, start:start+m] = 1
    return J

t0_ana = time.time()
J_analy = jacobian_sum_columns( dims[0], dims[1] )
t1_ana = time.time()
print(f"Analy, shape={dims}, in time {t1_ana-t0_ana}s") # J = {J_analy}, 
# Output:
# [[1. 1. 0. 0.]
#  [0. 0. 1. 1.]]
#%%[markdown]
# ### DOE cases
# =====================================================================
#%%
import pandas as pd
loc_DOEcsv_GBgen = "M:\\Vasudev_Gupta\\WISDEM\\examples\\06_drivetrain\\M4W_production_runs\\03_results\\DOE_GBgen_cleaned.csv"
cases = pd.read_csv( loc_DOEcsv_GBgen )
len_steps = cases.shape[0]

this_case = cases.loc[0]
# iter over this_case 
for key, value in this_case.items():
    # skip non-DV keys
    if key in ["status_driver_exit","time"]: continue
    # work on DVs
    # 1. float type
    if type(value) in [float, np.float64]: prob[key] = value
    # 2. str type for vector DVs
    elif type(value) == str: prob[key] = np.array(eval( value ))

myargs = []
for i,row in cases.iterrows():
    print(f" - {i}: gear_ratio = { row["gear_ratio"] }")
    myargs.append( [i, row] )

for thisIter in myargs:
    print(thisIter, "\n")

# %% # PARALLEL DOE ;)
import multiprocessing as mp
ncore = max(1, mp.cpu_count() - 2)
print(ncore)
pool = mp.Pool(processes=ncore)

# %%
import sys
pathFiles = sys.path
for file in pathFiles:
    print(file)
# %%
import sys
print(sys.executable)

# %%
import pkgutil
print([m.name for m in pkgutil.iter_modules()])

# %%[markdown]
# ### LDD and DEL: plot P_ and compare
# ================================================================
#%%
# method
meth_Peq = "LRD".lower()        # Method: "LRD" or "DEL"
flag_save_newBinPeq = True
# read bins from csv file
loc_csv_nBins = results_path+os.sep+"nBins_P_LDD.csv"
df_nBins = pd.read_csv( loc_csv_nBins )
NnBins = len(df_nBins)
opt_drivese_copy = opt_drivese.copy()
# outer loop for bins
for i in range(NnBins):
    # extract bin
    nBins = df_nBins["nBins"][i]
    opt_drivese_copy["nBins"] = nBins
    # define problem
    prob_ana = om.Problem(reports=False)
    model_ana = prob_ana.model = om.Group()
    model_ana.add_subsystem(
        "mb_fls",
        ds.Analytical_FLS_Bearing_Life(
            modeling_options=opt_drivese_copy,
            openfast_options=opt_openfast,
            dlc_options=opt_DLC
        ),
        promotes=['*']
    )
    # setup
    prob_ana.setup()
    # inputs
    # prob_ana.model.list_inputs();
    # define input params
    # - overall
    prob_ana["rated_rpm"] = eval( var_dict['rated_rpm'] )
    prob_ana["lifetime"] = eval( var_dict['lifetime'] )
    prob_ana["tilt"] = eval( var_dict['tilt'] )
    # - DT
    prob_ana["L_h1"] = eval( var_dict['L_h1'] )
    prob_ana["L_12"] = eval( var_dict['L_12'] )
    prob_ana["Dshaft_mb2"] = eval( var_dict['Dshaft_mb2'] )
    prob_ana["Tshaft_mb2"] = eval( var_dict['Tshaft_mb2'] )
    prob_ana["s_lss"] = eval( var_dict['s_lss'] )
    prob_ana["lss_E"] = eval( var_dict['lss_E'] )
    # - mb_fls
    prob_ana["k_mb2"] = eval( var_dict['mb_fls.k_mb2'] ) # - 6e8
    prob_ana["Cr_mb2"] = eval( var_dict['mb_fls.Cr_mb2'] )
    prob_ana["p_mb"] = eval( var_dict['mb_fls.p_mb'] )
    prob_ana["e_mb"] = eval( var_dict['mb_fls.e_mb'] )
    prob_ana["X1_mb"] = eval( var_dict['mb_fls.X1_mb'] )
    prob_ana["Y1_mb"] = eval( var_dict['mb_fls.Y1_mb'] )
    prob_ana["X2_mb"] = eval( var_dict['mb_fls.X2_mb'] )
    prob_ana["Y2_mb"] = eval( var_dict['mb_fls.Y2_mb'] )
    # - other DTs
    prob_ana["carrier_mass"] = eval( var_dict['carrier_mass'] )
    # run
    prob_ana.run_model()
    # outputs
    P_LDD = prob_ana["P_mb2_sum"]
    print(f"P_LDD [MN] = {P_LDD/1e6}")
    # - store
    if meth_Peq=="lrd": df_nBins.loc[i,"P_LDD"] = P_LDD
    if meth_Peq=="del": df_nBins.loc[i,"P_DEL"] = P_LDD
# save
if flag_save_newBinPeq: df_nBins.to_csv(loc_csv_nBins,index=False)
#%%
# post-processing
# - nBins
lst_nBins = np.array(df_nBins["nBins"].tolist())
# - DEL
P_DEL = df_nBins.loc[0,"P_DEL"]
lst_P_DEL = np.array(df_nBins["P_DEL"].tolist()) #/ P_DEL
# - LDD
lst_P_diff = np.array(df_nBins["P_LDD"].tolist()) - P_DEL

# - opts

# - plot
fig,ax = plt.subplots(figsize=(11, 4))
ax.plot(lst_nBins, lst_P_diff,
        marker='o', color= clrs_m4w["Aqua"])
ax.set_ylabel(r"$ P_{LDD}-P_{DEL} $")
ax.set_xlabel('# Bins')
ax.set_xticks(lst_nBins)
ax.grid(True)
# ax.legend()
plt.tight_layout()
# - save
plot_path = os.path.join(results_path, "Pbins_convergence.png")
# plt.savefig(plot_path) # NOTE: saved, so don't change now 
# - plot
plt.plot()

# %%[markdown]
# ### test `Load_Own_Hub_Loads` component
# ================================================================
#%%
opts["WISDEM"]["DriveSE"]["own_hub_loads"] = True

loadsProb = om.Problem()
loadsProb.model = om.Group()
loadsProb.model.add_subsystem(
        "loads", ds.Load_Own_Hub_Loads(
            openfast_options=opts["OpenFAST"],
            dlc_options=opts["DLC_driver"]["DLCs"][0]
        ), promotes=["*"]
    )
loadsProb.setup()
loadsProb.run_model()
# %%
loadsProb["F_aero_hub"]
# %%
loadsClass = ds.Load_Own_Hub_Loads(
        openfast_options=opts["OpenFAST"],
        dlc_options=opts["DLC_driver"]["DLCs"][0]    
    )
# %%
# retrieve attributes directly from the instance
loadsClass.load_from_file()
loaded_dict = loadsClass.fls_dict

#%%
# check working on the git repo package installed called drive4wind
import Drive4Wind
print(Drive4Wind.__version__)
# %%
