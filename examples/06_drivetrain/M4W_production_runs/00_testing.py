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
# import pickle

# %%
# import needed `WISDEM` modules
# from wisdem.drivetrainse.drivetrain import DriveMaterials

# from wisdem.drivetrainse.hub import Hub_System
# from wisdem.drivetrainse.gearbox import Gearbox

# import wisdem.drivetrainse.layout as lay

# import wisdem.drivetrainse.drive_components as dc

import wisdem.drivetrainse.drive_structure as ds

from wisdem.commonse.utilities import get_recorder_results, mainshaft_loads_from_mat_to_dict, load_all_mat_to_dict, pdf_norm_int_using_cdf, bin_counting_of_load, compute_LRD, compute_LRD_matrix_vectorized
# from wisdem.commonse.fileIO import save_data

#%% Loading `openFAST` hub loads from a saved file
part_loads = True 
load_fls_loads = False
# False: full loads (72e4,10) (200 Hz sampled, 60mins)
# True: part loads (72e3,11) (20 Hz sampled, 60mins)

dir_loads = "M:\Vasudev_Gupta\outputs_mainshaft_loads"

if part_loads: # define paths
    loc_all_loads_mat_file = os.path.join(dir_loads, "mainshaft_loads.mat")
    S_all, keys_all = load_all_mat_to_dict(loc_all_loads_mat_file)

else: # define paths
    loc_FLS_loads_mat_file = os.path.join(dir_loads, "mainshaft_loads_FLS_full.mat")
    loc_ULS_loads_mat_file = os.path.join(dir_loads, "mainshaft_loads_ULS.mat")
    if load_fls_loads: # load from paths
        Snew, keys_new = mainshaft_loads_from_mat_to_dict(
            loc_FLS_loads_mat_file, loc_ULS_loads_mat_file)
#%% segment loads for testing
Fx, Fy, Fz = S_all['Fx'][:3,:2], S_all['Fy'][:3,:2], S_all['Fz'][:3,:2]
Mx, My, Mz = S_all['Mx'][:3,:2], S_all['My'][:3,:2], S_all['Mz'][:3,:2]

#%% test forces
myForces = np.ones((2,2)) * 5 #(3,2)
Fx, Fy, Fz = myForces, myForces, myForces
Mx, My, Mz = myForces, myForces, myForces

# %% F_* computation
L_h1 = 0.264 # test: 2; converg: 0.264
L_12 = 6.935 # test: 5; converg: 6.935
Fmb1, Fmb2, dFmb1dLh1, dFmb1dL12, dFmb2dLh1, dFmb2dL12 = ds.analytical_MB_Forces(
    Fx,Fy,Fz,Mx,My,Mz,L_h1,L_12, flag_jac=True)

#%%
machine_rating = 15e6 #MW
tilt_rad = np.deg2rad(6)
delta = 0.5

m_shrink_disc = (machine_rating*1e-3)/3.0
m_carrier = 8e3
carrier_mass = m_shrink_disc + m_carrier

Fmb1_real, Fmb2_real = ds.analytical_MBforces_realistic(
    Fx,Fy,Fz,Mx,My,Mz,
    m_carrier, delta, tilt_rad,
    L_h1,L_12, flag_jac=False)

# %%
# P_* computation
P = Fmb1[3,:,:]
n_t, n_w = P.shape[0], P.shape[1]
ws_full = S_all['mean_wind_speed']; ws = ws_full[0,:n_w]
time = S_all['Time'][:n_t,0]; dt = 0.05
omega = S_all['rot_speed'][:n_t,:n_w]
p = 10/3

dP_dLh1 = dFmb1dLh1[3,:,:]
dP_dL12 = dFmb1dL12[3,:,:]

# %%
# ws pdf computation
coeff_weibull = (1.95, 11.6)
pdf_ws = pdf_norm_int_using_cdf( ws, coeff_weibull )
pdf_ws_full = pdf_norm_int_using_cdf( ws_full, coeff_weibull )
print(f"pdf_ws = {pdf_ws}" )
print(f"pdf_ws_full = {pdf_ws_full}" )

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
#%%[markdown]
# ### Test: Bearing Life Module `Analytical_FLS_Bearing_Life` 
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
opts["WISDEM"]["DriveSE"]["nBins"] = 100    #used by (new) Analytical_FLS_Bearing_Life; =Number of bins for histogram MB FLS
# used as: gamma = gamma_f * gamma_m * gamma_n (within TODO)

opts["WISDEM"]["RotorSE"] = {}
opts["WISDEM"]["RotorSE"]["n_pc"] = 2 #cf. RPM_Input in drive_components.py
            # `n_pc`: Number of wind speeds to compute the power curve
opts["materials"] = {}
opts["materials"]["n_mat"] = 4

opts["flags"] = {}
dogen = opts["flags"]["generator"] = False
dohub = opts["flags"]["hub"] = False #(v)

opts["OpenFAST"] = {}
opts["OpenFAST"]["simulation"] = {}
opts["OpenFAST"]["simulation"]["DT"] = 0.05
# dir(ectory) where MS loads are stored .csv (?)
if part_loads:
    opts["OpenFAST"]["openfast_dir"] = loc_all_loads_mat_file
else:
    ValueError('Error: full loads not defined in openfast_dir<-OpenFAST<-modelling_options. Please define it first. jazakumAllahu khayr.')

opts["DLC_driver"] = {}
opts["DLC_driver"]["DLCs"] = [{}]
opts["DLC_driver"]["DLCs"][0]["DLC"] = "1.2"
opts["DLC_driver"]["DLCs"][0]["wind_speed"] = [ 5.,  7.,  9., 11., 13., 15., 17., 19., 21., 23., 25.] #TODO
opts["DLC_driver"]["DLCs"][0]["probabilities"] = [0.06541262, 0.14245179, 0.14299681, 0.12940412, 0.10735197, 0.0824332 , 0.05894909, 0.03942148, 0.02472593, 0.01457773, 0.00466888]

opt_drivese = opts["WISDEM"]["DriveSE"]
# OpenFAST: containing 1. simulation DT and 2. MS loads dir
opt_openfast = opts["OpenFAST"]
# DLC: only 1 used '[0]': containing "wind_speed" and "probabilities"
opt_DLC = opts["DLC_driver"]["DLCs"][0]

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

# %%
