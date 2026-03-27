#%%
import yaml
import numpy as np
import matplotlib.pyplot as plt
import os
import wisdem.inputs as sch
from my_util_tools.util_funcs import loc_clr_scheme_m4w, read_color_scheme

clrs_m4w = read_color_scheme( loc_clr_scheme_m4w )


plot_rcParams_update = {
        "font.size": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16, # 16 for pdf of `var_with_iter` plot
        "lines.linewidth": 3,
        "lines.markersize": 6,
    }
plt.rcParams.update( plot_rcParams_update )

#%%
def parse_tower_data_from_yaml( yaml_file ):
    # ========================
    # Load YAML
    # ========================
    data = sch.load_yaml( yaml_file )

    tower = data['components']['tower']

    # ========================
    # Extract data
    # ========================
    # Height (z coordinates)
    z = np.array(
        tower['outer_shape_bem']['reference_axis']['z']['values']
    )

    # Outer diameter
    d = np.array(
        tower['outer_shape_bem']['outer_diameter']['values']
    )

    # Thickness (multiple layers → sum them)
    layers = tower['internal_structure_2d_fem']['layers']

    t_list = []
    for layer in layers:
        t_list.append(np.array(layer['thickness']['values']))

    t_total = np.sum(np.vstack(t_list), axis=0)  # [m]
    t_mm = t_total * 1000  # convert to mm

    return z, d, t_mm

def plot_tower_geometry( m4w_yaml, iea15_yaml , clrs=clrs_m4w):
    # ========================
    # Load YAMLs
    # ========================
    # 1. Made4Wind
    z_m4w, d_m4w, t_m4w = parse_tower_data_from_yaml( m4w_yaml )
    z_iea, d_iea, t_iea = parse_tower_data_from_yaml( iea15_yaml )

    # ========================
    # Plot
    # ========================
    fig, axs = plt.subplots(1, 2, figsize=(10, 6), sharey=True)

    # ---- Reference lines ---- 
    waterline = 0.0
    mudline = -30.0
    transition = 15.0

    for ax in axs:
        ax.axhline(transition, linestyle='--',color=clrs['Dark_Green'])
        # ax.axhline(waterline, linestyle='--',color=clrs['Dark_Blue'])
        # ax.axhline(mudline, linestyle='--',color=clrs['Dark_Red'])

    # Labels only once (left plot)
    axs[0].text(d_iea.min(), transition + 2, 'Tower transition')
    # axs[0].text(d_iea.min(), waterline + 2, 'Water line')
    # axs[0].text(d_iea.min(), mudline + 2, 'Mud line')

    # ---- Outer Diameter ----
    axs[0].plot(d_iea, z_iea,
        label='IEA 15MW', color=clrs['Light_Turquoise'], linewidth=2)
    axs[0].plot(d_m4w, z_m4w,
        label='Made4Wind', color=clrs['Aqua'], linewidth=2)
    axs[0].set_xlabel('Outer Diameter [m]')
    axs[0].set_yticks( z_iea )
    axs[0].set_ylabel('Tower Height [m]')
    axs[0].grid(True)

    # ---- Thickness (step plot) ----
    # NOTE: 'mid'=avg. btw x-pos (thickness) <- consistent with internal wisdem tower vector
    axs[1].step(t_iea, z_iea,
        where='mid', color=clrs['Light_Turquoise'], linewidth=2)
    axs[1].step(t_m4w, z_m4w,
        where='mid', color=clrs['Aqua'], linewidth=2)
    axs[1].set_xlabel('Wall Thickness [mm]')
    axs[1].grid(True)

    axs[0].legend(loc='upper center')
    
    plt.tight_layout()
    plt.show()
# ========================

#%%
# Run & plot
if __name__ == "__main__":
    mydir = os.path.dirname(os.path.realpath(__file__))
    
    # Geometry YAML files
    # 1. base IEA 15-MW
    iea_yaml = mydir +os.sep+ "iea15mw_tower_semisub.yaml"
    # 2. Made4Wind
    m4w_yaml = mydir +os.sep+ "outputs" + os.sep+ "test.yaml"
    plot_tower_geometry( m4w_yaml, iea_yaml )
# =======================================================================

# %%[markdown]
# ### Plotting loading conditions at tower-top from `modelling_option` files
#%%
# ======================================================
# 1. READ FUNCTION
# ======================================================
def read_wisdem_loading(yaml_file):
    data = sch.load_yaml( yaml_file )

    loading = data['WISDEM']['Loading']

    out = {}

    # Scalars
    out['mass'] = loading['mass']

    # COM
    out['com'] = np.array(loading['center_of_mass'])

    # Inertia (take principal components only)
    moi = np.array(loading['moment_of_inertia'])
    out['I'] = moi[:3]       # keep if still used elsewhere
    out['I_full'] = moi      # full 6 components



    # Loads (first entry)
    loads = loading['loads'][0]
    out['F'] = np.array(loads['force'])
    out['M'] = np.array(loads['moment'])

    return out


# ======================================================
# 2. PLOTTING FUNCTION
# ======================================================
def plot_comparison(m4w, iea, clrs=clrs_m4w):

    labels = ['IEA 15MW', 'Made4Wind']
    colors = [ clrs['Light_Green'], clrs['Aqua'] ]


    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # --------------------------------------------------
    # (1) Mass
    # --------------------------------------------------
    ax = axs[0, 0]

    categories = [r'$m_{RNA}$']

    m4w_vals = [
        m4w['mass']/1e6,
    ]

    iea_vals = [
        iea['mass']/1e6,
    ]

    x = np.arange(len(categories))
    width = 0.35

    ax.bar(x - width/2, iea_vals,
        width, color=colors[0], label=labels[0])
    ax.bar(x + width/2, m4w_vals,
        width, color=colors[1], label=labels[1])

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel( 'Mass [1e3 t]' )
    ax.set_title('RNA Mass Comparison')
    # ax.legend()
    ax.grid(True)


    # --------------------------------------------------
    # (2) Center of Mass (3D)
    # = removed (TODO? iff needed or show on ppt drawing)
    # = replaced with MoI comparison
    # --------------------------------------------------
    # from mpl_toolkits.mplot3d import Axes3D  # add at top of script

    # fig.delaxes(axs[0, 1])  # remove 2D axis
    # ax = fig.add_subplot(2, 2, 2, projection='3d')

    # # Scatter points
    # ax.scatter(*m4w['com'], label='M4W', s=80)
    # ax.scatter(*iea['com'], label='IEA15MW', s=80)

    # # Labels
    # ax.set_xlabel('X [m]')
    # ax.set_ylabel('Y [m]')
    # ax.set_zlabel('Z [m]')
    # ax.set_title('Center of Mass (3D)')

    # # Optional: connect points (nice visual)
    # ax.plot(
    #     [m4w['com'][0], iea['com'][0]],
    #     [m4w['com'][1], iea['com'][1]],
    #     [m4w['com'][2], iea['com'][2]],
    #     linestyle='--'
    # )

    # ax.legend()

    # --------------------------------------------------
    # (2) Full Moment of Inertia (6 components)
    # --------------------------------------------------
    ax = axs[0, 1]

    labels_I = [
        r'$I_{xx}$', r'$I_{yy}$', r'$I_{zz}$',
        r'$I_{xy}$', r'$I_{xz}$', r'$I_{yz}$'
        ]

    # use full MoI (not just first 3)
    m4w_I = m4w['I_full'] / 1e6
    iea_I = iea['I_full'] / 1e6

    x = np.arange(len(labels_I))
    width = 0.35

    ax.bar(x - width/2, iea_I,
           width, color=colors[0], label=labels[0])
    ax.bar(x + width/2, m4w_I,
           width, color=colors[1], label=labels[1])

    ax.set_xticks(x)
    ax.set_xticklabels(labels_I)
    ax.set_ylabel('MoI [1e6 kg·m²]')
    ax.set_title('Full Inertia Tensor Comparison')
    ax.legend()
    ax.grid(True)


    # --------------------------------------------------
    # (3) Forces
    # --------------------------------------------------
    ax = axs[1, 0]

    labels_F = [r'$F_x$', r'$F_y$', r'$F_z$']

    x = np.arange(3)

    ax.bar(x - width/2, iea['F']/1e6,
           width, color=colors[0])
    ax.bar(x + width/2, m4w['F']/1e6,
           width, color=colors[1])

    ax.set_xticks(x)
    ax.set_xticklabels(labels_F)
    ax.set_ylabel('Force [MN]')
    ax.set_title('Forces Comparison')
    ax.grid(True)


    # --------------------------------------------------
    # (4) Moments
    # --------------------------------------------------
    ax = axs[1, 1]

    labels_M = [r'$M_x$', r'$M_y$', r'$M_z$']

    x = np.arange(3)

    ax.bar(x - width/2, iea['M']/1e6,
           width, color=colors[0])
    ax.bar(x + width/2, m4w['M']/1e6,
           width, color=colors[1])

    ax.set_xticks(x)
    ax.set_xticklabels(labels_M)
    ax.set_ylabel('Moment [MN·m]')
    ax.set_title('Moments Comparison')
    ax.grid(True)


    plt.tight_layout()
    plt.show()

#%%
# ======================================================
# 3. MAIN
# ======================================================
if __name__ == "__main__":

    file_m4w = "modeling_options_m4w_monopile_only.yaml"
    file_iea = "modeling_options_iea15_monopile_only_wisdemV3.yaml"

    m4w = read_wisdem_loading(file_m4w)
    iea = read_wisdem_loading(file_iea)

    plot_comparison(m4w, iea)

# %%
