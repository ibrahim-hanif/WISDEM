#%%
# Imports
import wisdem.drivetrainse.drive_components as dc
import numpy as np
import matplotlib.pyplot as plt
import os
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
mbClass = dc.MainBearing_withDerivatives()
mbData = mbClass.BEARINGS

# Calculate bearing properties
def def_mb_props( dict, D ):
    # width
    a = dict["a"]
    b = dict["b"]
    width = (a*D) + b
    # mass
    k = dict["k"]
    n = dict["n"]
    mass = k*(D**n)
    # Cr
    c = dict["c"]
    m = dict["m"]
    Cr = c*(D**m)
    return width, mass, Cr

#%% # Diameter: user defined
dia = np.linspace(0.0,5.0,51)

mbList = ["CARB", "CRB", "SRB", "TRB", "TRB2"]
for mb in mbList:
    thisMBdata = mbData[mb]
    mbData[mb]['width'],mbData[mb]['mass'],mbData[mb]['Cr'] = def_mb_props(
        thisMBdata, dia )

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
        "font.size": 20,
        "axes.labelsize": 20,
        "legend.fontsize": 20, # 16 for pdf of `var_with_iter` plot
        "lines.linewidth": 2,
        "lines.markersize": 6,
    }
plt.rcParams.update( params_plot_rc )
figsize=(16, 8)

#%% # plt plot
fig = plt.figure(figsize=figsize)
gs = fig.add_gridspec(1, 2, hspace=0.35, wspace=0.25)
# create axis ONCE before loop
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

for mbType, clr, mrkr in zip(mbList, clrsList, mrkerList):
    thisMBdata = mbData[ mbType ]
    # Mass plot
    mass = thisMBdata['mass'] / 1e3
    ax1.plot(
        dia,
        mass,
        color= clr,
        marker=mrkr,
        linewidth=2,
        zorder=1,
        label=mbType
        )
    # Cr plot
    Cr = thisMBdata['Cr'] / 1e3
    ax2.plot(
        dia,
        Cr,
        color= clr,
        marker=mrkr,
        linewidth=2,
        zorder=1,
        label=mbType
        )

# 1 Axis Formatting
ax1.set_xlabel(r'$D\ \mathrm{[m]}$')
ax1.set_ylabel(r'$m\ \mathrm{[t]}$')
ax1.grid(True)
ax1.legend(loc='upper left')
# 2 Axis Formatting
ax2.set_xlabel(r'$D\ \mathrm{[m]}$')
ax2.set_ylabel(r'$C\ \mathrm{[MN]}$')
ax2.grid(True)
ax2.legend(loc='upper left')
# Final
# plt.tight_layout()
# Save
plot_path = os.path.join(results_path, "mb_m&Cr.png")
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