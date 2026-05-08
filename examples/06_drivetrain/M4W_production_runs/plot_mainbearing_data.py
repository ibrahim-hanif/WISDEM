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
from my_util_tools import util_funcs
loc_clr_scheme_m4w = util_funcs.loc_clr_scheme_m4w
clrs_m4w = util_funcs.read_color_scheme(loc_clr_scheme_m4w)

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

    for iDia in range(len(diaList)):
        mbProb["D_shaft"] = diaList[iDia]
        # compute
        mbProb.run_model()
        # store values
        mass[iDia] = mbProb["mb_mass"]
        Cr[iDia] = mbProb["mb_Cr"] * N_to_kN # N to kN
    
    mbData[mb]["mass"] = mass
    mbData[mb]["Cr"] = Cr

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
figsize=(8,16)

#%% # plt plot
fig = plt.figure(figsize=figsize)
gs = fig.add_gridspec(2,1, hspace=0.2)#, wspace=0.25)
# create axis ONCE before loop
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[1,0])

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
plot_path = os.path.join(results_path, "mb_m&Cr.pdf")
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
prob['mb_e']