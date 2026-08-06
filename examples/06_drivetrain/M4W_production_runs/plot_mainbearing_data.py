#%%[markdown]¨
# # Plot bearing empirical data
# - using `MainBearing_withDerivatives` WISDEM component
# - mass plot w/o housing_factor multiplied is saved as `*_woHouseFac`.
# -- while mass with factor is saved as '*' (no suffix)

#%%
# Imports
import wisdem.drivetrainse.drive_components as dc
import numpy as np
import matplotlib.pyplot as plt
import os
import openmdao.api as om
# main colors
from Drive4Wind.post_processing import color_schemes
loc_clr_scheme_m4w = color_schemes.loc_clr_scheme_m4w
clrs_m4w = color_schemes.read_color_scheme(loc_clr_scheme_m4w)

results_dir = "00_results"
script_dir = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(script_dir, results_dir)
os.makedirs(results_path, exist_ok=True)

#%%
# MainBearing class from WISDEM
# - define as openmdao problem group
mbProb = om.Problem(reports=False)
mbProb.model = om.Group()
mbProb.model.add_subsystem(
    "mb", dc.MainBearing_withDerivatives(),
    promotes=['*']
)
# setup
mbProb.setup()
# - define inputs, which remain constant
mbProb["mb_e"] = 0.4

# - define param vary
nParams = 26 # test = 3; final = 51
diaList = np.linspace(0.0,5.0,nParams); nDia = nParams
mbList = ["CARB", "CRB", "SRB", "TRB", "TRB2"]; nMBs = len(mbList)

# TODO define dict to store
mbData = {}
myzeros = np.zeros(diaList.shape)
N_to_kN = 1e-3

#%%
for mb in mbList:
    # init to zeros
    mbData[mb] = {}
    # mbData[mb]["mass"] = myzeros
    # mbData[mb]["Cr"] = myzeros
    # input mb type
    mbProb["bearing_type"] = str(mb)
    # input dia
    mass = np.copy( myzeros )
    Cr = np.copy( myzeros )
    fw = np.copy( myzeros )

    for iDia in range(len(diaList)):
        mbProb["D_shaft"] = diaList[iDia]
        # compute
        mbProb.run_model()
        # store values
        mass[iDia] = mbProb["mb_mass"][0]
        Cr[iDia] = mbProb["mb_Cr"][0] * N_to_kN # N to kN
        fw[iDia] = mbProb["face_width"][0]
    
    mbData[mb]["mass"] = mass
    mbData[mb]["Cr"] = Cr
    mbData[mb]["width"] = fw

# %% # Plotting options
clrsList = [
    clrs_m4w['Dark_Blue'],
    clrs_m4w['Aqua'],
    clrs_m4w['Dark_Green'],
    clrs_m4w['Light_Green'],
    clrs_m4w['Red']
    ]
mrkerList = ['.','o','x','+','*']

# -------------------------
# Journal polish: plot rc params
params_plot_rc = {
        "font.size": 24,
        "axes.labelsize": 24,
        "legend.fontsize": 24, # 16 for pdf of `var_with_iter` plot
        "lines.linewidth": 3,
        "lines.markersize": 8,
    }
plt.rcParams.update( params_plot_rc )
# grid shape
gridspec = (1,2) # TODO to vary plot: side-side (1,2) or top-down (2,1)
figsize = tuple( [8*x for x in gridspec[::-1] ] )
# - tall fig
tall_fig = False
if gridspec[0] > gridspec[1]:
    gs_ax2 = [1,0]
    tall_fig = True
# - wide fig
else: gs_ax2 = [0,1]

#%% # plt plot
fig = plt.figure(figsize=figsize)
gs = fig.add_gridspec(*gridspec, hspace=0.2)#, wspace=0.25)
# create axis ONCE before loop
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[*gs_ax2])

for mbType, clr, mrkr in zip(mbList, clrsList, mrkerList):
    thisMBdata = mbData[ mbType ]
    # Mass plot
    thisMass = thisMBdata['mass'] / 1e3
    ax1.plot(
        diaList,
        thisMass,
        color= clr,
        marker=mrkr,
        zorder=1,
        label=mbType
        )
    # Cr plot
    thisCr = thisMBdata['Cr'] / 1e3
    ax2.plot(
        diaList,
        thisCr,
        color= clr,
        marker=mrkr,
        zorder=1,
        label=mbType
        )

# 1 Axis Formatting
# ax1.set_xlabel(r'$D\ \mathrm{[m]}$')
ax1.set_title(r'$m\ \mathrm{[t]}$')
if not tall_fig: ax1.set_xlabel(r'$D\ \mathrm{[m]}$')
ax1.grid(True)
ax1.legend(loc='upper left')
# 2 Axis Formatting
ax2.set_xlabel(r'$D\ \mathrm{[m]}$')
ax2.set_title(r'$C\ \mathrm{[MN]}$')
ax2.grid(True)
# ax2.set_xticks( np.arange(0,6) )
ax2.legend(loc='upper left')
# Final
# plt.tight_layout()
# Save
plot_path = os.path.join(results_path, "mb_m&Cr_wide.pdf")
# plt.savefig(plot_path) # NOTE: saved, so don't change now 
# Show
plt.show()

# %%[markdown]
# ### Analyse how `mb_e` varies `X,Y`-load coefficients
# %%
import openmdao.api as om

prob = om.Problem(reports=False)
prob.model = om.Group()
prob.model.add_subsystem( 'mb', dc.MainBearing_withDerivatives(),
                         promotes=['*'])
prob.setup()
# %%
k_torsional = 3e10-6e8 # Nm/rad

prob['mb_e']
prob["bearing_type"] = "TRB2"
prob["D_shaft"] = D_shaft = 3.406783213751883 # m
prob.run_model()
# outputs
mb_mass = prob["mb_mass"] # kg
mb_fw = prob["face_width"] # m
mb_Cr = prob["mb_Cr"] # N
mb_k = prob["mb_k"] # Nm

k_torsional / (mb_Cr*D_shaft) # coeff_torsional = 

#%%
# ----- parse
idx = np.argmin(np.abs(diaList - D_shaft))
mass = mbData['TRB2']['mass'][idx] # kg
Cr = mbData['TRB2']['Cr'][idx] / N_to_kN # N
fw = mbData['TRB2']['width'][idx] # m
# ------------ 1
#a=0.1541, b=0.2087, k=1442.6, n=1.8932, c=6579.9, m=0.8592
c=6579.9
m=0.8592
coeff_torsional = 457.502
# k_tor = K * D * Cr = K * c*D**(m+1)*1e3
denom = c*D_shaft**(m+1) / N_to_kN
k_torsional / denom # = coeff_torsional

# %%
# ------------ 2
# k_tor = K*Cr*(B/D)**0.5 * D
# K = 
denom = Cr * ((fw/D_shaft)**0.5) * D_shaft
k_torsional / denom
# %%
