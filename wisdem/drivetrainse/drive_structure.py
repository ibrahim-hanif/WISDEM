#!/usr/bin/env python
# encoding: utf-8
"""
Adapted by Vasudev Gupta on 2025-11-06. Dept. of Marine Technology, NTNU. All rights reserved.
"""
import numpy as np
import openmdao.api as om

import wisdem.pyframe3dd.pyframe3dd as frame3dd
from wisdem.commonse import gravity
from wisdem.commonse.csystem import DirectionVector
from wisdem.commonse.utilities import find_nearest, nodal2sectional, smooth_abs
from wisdem.commonse.utilities import bin_counting_of_load, load_all_mat_to_dict, compute_LRD_matrix_vectorized #(v)
from wisdem.commonse.cross_sections import Tube, IBeam
from wisdem.commonse.utilization_constraints import TubevonMisesStressUtilization
from scipy.stats import weibull_min #(v)

RIGID = 1
FREE = 0

def tube_prop(s, Di, ti):
    L = s.max() - s.min()

    def equal_pts(xi):
        if len(xi) < len(s) and len(xi) == 2:
            x = np.interp((s - s.min()) / L, [0, 1], xi)
        elif len(xi) == len(s):
            x = xi
        else:
            raise ValueError("Unknown grid of input", str(xi))
        return x

    D = equal_pts(Di)
    t = equal_pts(ti)
    return Tube(nodal2sectional(D)[0], nodal2sectional(t)[0])


class Hub_Rotor_LSS_Frame(om.ExplicitComponent):
    """
    (Geared) Run structural analysis of hub system with the generator rotor and main (LSS) shaft.

    Parameters
    ----------
    tilt : float, [deg]
        Shaft tilt
        = input from: user
    s_lss : numpy array[5], [m]
        Discretized s-coordinates along drivetrain, measured from bedplate (direct) or tower center (geared)
        = input from: layout.py, GearedLayout
    lss_diameter : numpy array[2], [m]
        LSS outer diameter from hub to bearing 2
        = input from: layout.py, GearedLayout
    lss_wall_thickness : numpy array[2], [m]
        LSS wall thickness
        = input from: layout.py, GearedLayout
    hub_system_mass : float, [kg]
        Hub system mass
        = input from: hub.py, TODO: user rn
    hub_system_cm : float, [m]
        Hub system center of mass distance from hub flange
        = input from: hub.py, TODO: user rn
    hub_system_I : numpy array[6], [kg*m**2]
        Hub system moment of inertia
        = input from: hub.py, TODO: user rn
    F_aero_hub : numpy array[3, n_dlcs], [N]
        Aero-only force vector applied to the hub
        = input from: user
    M_aero_hub : numpy array[3, n_dlcs], [N*m]
        Aero-only moment vector applied to the hub
        = input from: user
    blades_mass : float, [kg]
        Mass of all blades
        = input from: user ( = prob["n_blades"] * prob["blade_mass"] )
    blades_cm   : float, [m]
        Center of mass of all blades
        = input from: user
    blades_I    : numpy array[6], [kg*m**2]
        Moment of inertia of all blades
        = input from: user
    s_mb1 : float, [m]
        Bearing 1 s-coordinate along drivetrain, measured from bedplate (direct) or tower center (geared)
        = input from: layout.py, GearedLayout
    s_mb2 : float, [m]
        Bearing 2 s-coordinate along drivetrain, measured from bedplate (direct) or tower center (geared)
        = input from: layout.py, GearedLayout
    s_rotor : float, [m]
        Generator rotor attachment to lss s-coordinate measured from bedplate (direct) or tower center (geared)
        = input from: layout.py, DirectLayout
    generator_rotor_mass : float, [kg]
        Generator rotor mass
        = input from: generator.py / drive_components.py, GeneratorSimple
    generator_rotor_I : numpy array[3], [kg*m**2]
        Generator rotor moment of inertia (measured about its cm)
        = input from: generator.py / drive_components.py, GeneratorSimple
    gearbox_mass : float, [kg]
        Gearbox rotor mass
        = input from: gearbox.py
    gearbox_I : numpy array[3], [kg*m**2]
        Gearbox moment of inertia (measured about its cm)
        = input from: gearbox.py
    brake_mass  : float, [kg]
        Mass of brake system
        = input from: drive_components.py, Brake
    brake_I     : numpy array[6], [kg*m**2]
        Moment of inertia of brake system
        = input from: drive_components.py, Brake
    carrier_mass    : float, [kg]
        Mass of carrier
        = input from: gearbox.py
    carrier_I    : numpy array[6], [kg*m**2]
        Moment of inertia of carrier
        = input from: gearbox.py
    lss_E : float, [Pa]
        modulus of elasticity
        = input from: user (used in drivetrain.py, DriveMaterials)
    lss_G : float, [Pa]
        shear modulus
        = input from: user (used in drivetrain.py, DriveMaterials)
    lss_rho : float, [kg/m**3]
        material density
        = input from: user (used in drivetrain.py, DriveMaterials)
    lss_Xy : float, [Pa]
        yield stress
        = input from: user (used in drivetrain.py, DriveMaterials)
    shaft_deflection_allowable : float, [m]
        Allowable deflection of the shaft at gearbox attachment
        = input from: user
    shaft_angle_allowable : float, [rad]
        Allowable rotation angle of the shaft at gearbox attachment
        = input from: user
    
    ----- extended by Vasudev Gupta (v) -----
    mb1_face_width : float, [m]
        face width of the 1st main bearing
    mb2_face_width : float, [m]
        face width of the 2nd main bearing
    mb1_Reactions : numpy array[6] of int
        Bearing 1 reaction constraints: [Rx,Ry,Rz,Rxx,Ryy,Rzz]
    mb2_Reactions : numpy array[6] of int
        Bearing 2 reaction constraints: [Rx,Ry,Rz,Rxx,Ryy,Rzz]

    Returns
    -------
    lss_spring_constant : float, [N*m/rad]
        Equivalent spring constant for the low speed shaft from T=(G*J/L)*theta
    torq_deflection : float, [m]
        Maximum deflection distance at rotor (direct) or gearbox (geared) attachment
    torq_angle : float, [rad]
        Maximum rotation angle at rotor (direct) or gearbox (geared) attachment
    lss_axial_stress : numpy array[5, n_dlcs], [Pa]
        Axial stress in Curved_beam structure
    lss_shear_stress : numpy array[5, n_dlcs], [Pa]
        Shear stress in Curved_beam structure
    constr_lss_vonmises : numpy array[5, n_dlcs]
        Sigma_y/Von_Mises
    F_mb1 : numpy array[3, n_dlcs], [N]
        Force vector applied to bearing 1 in hub c.s.
    F_mb2 : numpy array[3, n_dlcs], [N]
        Force vector applied to bearing 2 in hub c.s.
    F_torq : numpy array[3, n_dlcs], [N]
        Force vector applied to generator rotor (direct) or gearbox (geared) in hub c.s.
    M_mb1 : numpy array[3, n_dlcs], [N*m]
        Moment vector applied to bearing 1 in hub c.s.
    M_mb2 : numpy array[3, n_dlcs], [N*m]
        Moment vector applied to bearing 2 in hub c.s.
    M_torq : numpy array[3, n_dlcs], [N*m]
        Moment vector applied to generator rotor (direct) or gearbox (geared) in hub c.s.
    lss_axial_load2stress : numpy array[nFull-1,6], [m**2]
        Linear conversion factors between loads [Fx-z; Mx-z] and axial stress; for fatigue
    lss_shear_load2stress : numpy array[nFull-1,6], [m**2]
        Linear conversion factors between loads [Fx-z; Mx-z] and shear stress; for fatigue
    constr_shaft_deflection : float
        Constraint on LSS translational deflection: maximum along the shaft
    constr_shaft_angle : float
        Constraint on LSS anglular deflection: maximum along the shaft
    
    ----- extended by Vasudev Gupta (v) -----
    constr_Lh1_MB1fw : float, [m]
        length constraint that L_h1 must be more than 0.5 * MB1 face width
    constr_L12_MBsFW : float, [m]
        length constraint that L_12 must be more than 0.5 * face widths of both bearings (mb1, mb2)

    """

    def initialize(self):
        self.options.declare("n_dlcs")
        self.options.declare("direct_drive", default=True)
        self.options.declare("modeling_options")

    def setup(self):
        n_dlcs = self.options["n_dlcs"]

        self.add_discrete_input("upwind", True)
        self.add_input("tilt", 0.0, units="deg")
        self.add_input("s_lss", val=np.zeros(5), units="m") #(v) TODO: use layout
        self.add_input("lss_diameter", val=np.zeros(2), units="m") #(v) a DV to be optim
        self.add_input("lss_wall_thickness", val=np.zeros(2), units="m") #(v) a DV to be optim
        self.add_input("hub_system_mass", 0.0, units="kg")
        self.add_input("hub_system_cm", 0.0, units="m")
        self.add_input("hub_system_I", np.zeros(6), units="kg*m**2")
        self.add_input("F_aero_hub", val=np.zeros((3, n_dlcs)), units="N")
        self.add_input("M_aero_hub", val=np.zeros((3, n_dlcs)), units="N*m")
        self.add_input("blades_mass", 0.0, units="kg")
        self.add_input("blades_cm", 0.0, units="m")
        self.add_input("blades_I", np.zeros(6), units="kg*m**2")
        self.add_input("s_mb1", val=0.0, units="m")
        self.add_input("s_mb2", val=0.0, units="m")
        self.add_input("s_rotor", val=0.0, units="m")
        self.add_input("generator_rotor_mass", val=0.0, units="kg")
        self.add_input("generator_rotor_I", val=np.zeros(3), units="kg*m**2")
        self.add_input("gearbox_mass", val=0.0, units="kg") #(v) TODO: use gearbox
        self.add_input("gearbox_I", val=np.zeros(3), units="kg*m**2")
        self.add_input("brake_mass", val=0.0, units="kg")
        self.add_input("brake_I", val=np.zeros(3), units="kg*m**2")
        self.add_input("carrier_mass", val=0.0, units="kg")
        self.add_input("carrier_I", val=np.zeros(3), units="kg*m**2")
        self.add_input("lss_E", val=0.0, units="Pa") #(v) TODO: use materials
        self.add_input("lss_G", val=0.0, units="Pa")
        self.add_input("lss_rho", val=0.0, units="kg/m**3")
        self.add_input("lss_Xy", val=0.0, units="Pa")

        self.add_input("shaft_deflection_allowable", val=1.0, units="m")
        self.add_input("shaft_angle_allowable", val=1.0, units="rad")
        self.add_input("mb1_face_width", val=0.0, units="m", desc="face width of the 1st main bearing") #(v) --- & below ---        
        self.add_input("mb2_face_width", val=0.0, units="m", desc="face width of the 2nd main bearing")
        self.add_input("mb1_Reactions", val=np.zeros(6,dtype=int), desc="Bearing 1 reaction constraints: [Rx,Ry,Rz,Rxx,Ryy,Rzz]")
        self.add_input("mb2_Reactions", val=np.zeros(6,dtype=int), desc="Bearing 2 reaction constraints: [Rx,Ry,Rz,Rxx,Ryy,Rzz]")

        self.add_output("lss_spring_constant", 0.0, units="N*m/rad")
        self.add_output("torq_deflection", val=0.0, units="m") #(v) defl at GB
        self.add_output("torq_angle", val=0.0, units="rad")    #(v) angle at GB
        self.add_output("lss_axial_stress", np.zeros((4, n_dlcs)), units="Pa")
        self.add_output("lss_shear_stress", np.zeros((4, n_dlcs)), units="Pa")
        self.add_output("constr_lss_vonmises", np.zeros((4, n_dlcs)))
        self.add_output("F_mb1", val=np.zeros((3, n_dlcs)), units="N")
        self.add_output("F_mb2", val=np.zeros((3, n_dlcs)), units="N")
        self.add_output("F_torq", val=np.zeros((3, n_dlcs)), units="N")
        self.add_output("M_mb1", val=np.zeros((3, n_dlcs)), units="N*m")
        self.add_output("M_mb2", val=np.zeros((3, n_dlcs)), units="N*m")
        self.add_output("M_torq", val=np.zeros((3, n_dlcs)), units="N*m")
        self.add_output("lss_axial_load2stress", val=np.zeros(6), units="m**2")
        self.add_output("lss_shear_load2stress", val=np.zeros(6), units="m**2")
        self.add_output("constr_shaft_deflection", 0.0)
        self.add_output("constr_shaft_angle", 0.0)
        self.add_output("constr_Lh1_MB1fw", val=0.0, units="m") #(v) --- & below ---
        self.add_output("constr_L12_MBsFW", val=0.0, units="m")

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Unpack inputs
        upwind = discrete_inputs["upwind"]
        Cup = -1.0 if upwind else 1.0
        tilt = float(np.deg2rad(inputs["tilt"][0]))

        s_lss = inputs["s_lss"]
        D_lss = inputs["lss_diameter"]
        t_lss = inputs["lss_wall_thickness"]

        s_mb1 = float(inputs["s_mb1"][0])
        s_mb2 = float(inputs["s_mb2"][0])

        if self.options["direct_drive"]:
            s_rotor = float(inputs["s_rotor"][0])
            m_rotor = float(inputs["generator_rotor_mass"][0])
            I_rotor = inputs["generator_rotor_I"]

            m_brake = float(inputs["brake_mass"][0])
            I_brake = inputs["brake_I"]
        else:
            m_gearbox = float(inputs["gearbox_mass"][0])
            I_gearbox = inputs["gearbox_I"]

            m_carrier = float(inputs["carrier_mass"][0])
            I_carrier = inputs["carrier_I"]

        rho = float(inputs["lss_rho"][0])
        E = float(inputs["lss_E"][0])
        G = float(inputs["lss_G"][0])
        sigma_y = float(inputs["lss_Xy"][0])
        gamma_f = float(self.options["modeling_options"]["gamma_f"])
        gamma_m = float(self.options["modeling_options"]["gamma_m"])
        gamma_n = float(self.options["modeling_options"]["gamma_n"])
        gamma = gamma_f * gamma_m * gamma_n

        m_hub = float(inputs["hub_system_mass"][0])
        cm_hub = Cup*float(inputs["hub_system_cm"][0])
        I_hub = inputs["hub_system_I"]
        m_blades = float(inputs['blades_mass'][0])
        cm_blades = Cup*float(inputs["blades_cm"][0])
        I_blades = inputs["blades_I"]
        m_blades_hub = m_blades + m_hub + 1e-10
        cm_blades_hub = (m_blades*cm_blades + m_hub*cm_hub) / m_blades_hub
        I_blades_hub = I_blades[:3] + I_hub[:3]
        F_hub = inputs["F_aero_hub"]
        M_hub = inputs["M_aero_hub"]

        torq_defl_allow = float(inputs["shaft_deflection_allowable"][0])
        torq_angle_allow = float(inputs["shaft_angle_allowable"][0])

        # ------- node data ----------------
        n = len(s_lss)              #(v) n = 5 (made in GearedLayout)
        inode = np.arange(1, n + 1) #(v) 1,2,3,..., n
        ynode = znode = rnode = np.zeros(n)
        xnode = Cup * s_lss.copy() #(v) 'shallow copy': Changes to copy don't affect original.
        nodes = frame3dd.NodeData(inode, xnode, ynode, znode, rnode)
        # Grab indices for later
        i1 = inode[find_nearest(xnode, Cup * s_mb1)]; #(v) 4
        i2 = inode[find_nearest(xnode, Cup * s_mb2)]; #(v) 2
        iadd = inode[1];                              #(v) 2 (= i2)
        # Differences between direct annd geared
        if self.options["direct_drive"]:
            itorq = inode[find_nearest(xnode, Cup * s_rotor)]
            m_torq = m_rotor
            I_torq = I_rotor

            m_add = m_brake
            I_add = I_brake
        else:
            itorq = inode[0]
            m_torq = m_gearbox - m_carrier
            I_torq = I_gearbox - I_carrier

            m_add = m_carrier
            I_add = I_carrier
        # ------------------------------------

        # ------ reaction data ------------
        # Reactions at main bearings %(v) NOTE: m4w UW CRB (rad), DW TRB (ax,rad,mom)
        mb1_Reactions = inputs["mb1_Reactions"] #(v)
        mb2_Reactions = inputs["mb2_Reactions"] #(v)
        rnode = np.r_[i1, i2, itorq]
        Rx = np.array([mb1_Reactions[0], mb2_Reactions[0], FREE])  # (v, def) RIGID, FREE, FREE: Upwind bearing restricts translational
        Ry = np.array([mb1_Reactions[1], mb2_Reactions[1], FREE])  # (v, def) RIGID, FREE, FREE: Upwind bearing restricts translational
        Rz = np.array([mb1_Reactions[2], mb2_Reactions[2], FREE])  # (v, def) RIGID, FREE, FREE: Upwind bearing restricts translational
        Rxx = np.array([FREE, FREE, RIGID])  # (v, def) FREE, FREE, RIGID: Torque is absorbed by stator, so this is the best way to capture that
        Ryy = np.array([mb1_Reactions[4], mb2_Reactions[4], FREE])  # (v, def) FREE, RIGID, FREE: downwind bearing carry moments
        Rzz = np.array([mb1_Reactions[5], mb2_Reactions[5], FREE])  # (v, def) FREE, RIGID, FREE:  downwind bearing carry moments
        # print("LSS Bearing Reactions (Rx,Ry,Rz,Rxx,Ryy,Rzz): ", np.array((Rx,Ry,Rz,Rxx,Ryy,Rzz))) #(v) debugging
        reactions = frame3dd.ReactionData(rnode, Rx, Ry, Rz, Rxx, Ryy, Rzz, rigid=RIGID)
        # -----------------------------------

        # ------ frame element data ------------
        lsscyl = tube_prop(s_lss, D_lss, t_lss)
        ielement = np.arange(1, n)
        N1 = np.arange(1, n)
        N2 = np.arange(2, n + 1)
        roll = np.zeros(n - 1)
        myones = np.ones(n - 1)
        Ax = lsscyl.Area
        As = lsscyl.Asx
        S = lsscyl.S    #(v) bending modulus for tubular sections: line 114, cross_sections.py
        C = lsscyl.C    #(v) torsional shear constant for tubular sections: line 122, cross_sections.py
        J0 = lsscyl.J0  #(v) polar moment of inertia w.r.t. x-x axis (torsional)
        Jx = lsscyl.Ixx #(v) 2nd area moment of inertia w.r.t. y-y axis (Iyy=Izz for tubes)

        elements = frame3dd.ElementData(
            ielement, N1, N2, Ax, As, As, J0, Jx, Jx, E * myones, G * myones, roll, rho * myones
        )

        lss_stiff = G * J0 / np.diff(s_lss)
        outputs["lss_spring_constant"] = 1.0 / np.sum(1.0 / lss_stiff)
        # -----------------------------------

        # ------ options ------------
        shear = geom = True #(v) 1: include shear deformation + geom stiffness
        dx = -1
        options = frame3dd.Options(shear, geom, dx)
        # -----------------------------------

        # initialize frameDD3 object
        myframe = frame3dd.Frame(nodes, reactions, elements, options)

        # ------ add extra mass: hub and generator rotor (direct) or gearbox (geared) ------------
        # ----- (v) added prev impl under "direct_drive", new (geared) under `else`
        if self.options["direct_drive"]:
            three0 = np.zeros(3).tolist()
            myframe.changeExtraNodeMass(       #(v) NOTE: mb_mass added on bedplate, NOT LSS
                np.r_[inode[-1], itorq, iadd], #(v) TODO: realistic to add itorq (weight of GB-carrier) on shaft DW end?
                [m_blades_hub, m_torq, m_add], #(v) TODO: check if rotor (blades+hub) mass should be added or not (incl in F_aero_hub)
                [I_blades_hub[0], I_torq[0], I_add[0]],
                [I_blades_hub[1], I_torq[1], I_add[1]],
                [I_blades_hub[2], I_torq[2], I_add[2]],
                three0,
                three0,
                three0,
                [cm_blades_hub, 0.0, 0.0],
                three0,
                three0,
                True,
            )
        else:
        # -----(v) -----
        # - 1. removed (GB-carrier) weight from DW end
        # - 2. shifted carrier weight from mb2 to DW end (itorq)
        # - 3. removed hub self-weight loads from UW end (already in F_aero_hub, from openFAST)
            one0 = np.zeros(1).tolist()
            myframe.changeExtraNodeMass(    #(v) NOTE: mb_mass added on bedplate, NOT LSS
                np.r_[itorq],               #(v) DONE: realistic to add itorq (weight of GB-carrier) on shaft DW end? removed.
                [m_add],                    #(v) DONE: check if rotor (blades+hub) mass should be added or not (incl in F_aero_hub), removed
                [I_add[0]],
                [I_add[1]],
                [I_add[2]],
                one0,
                one0,
                one0,
                [0.0],
                one0,
                one0,
                True,
            )
        # ------------------------------------

        # ------- NO dynamic analysis ----------
        # myframe.enableDynamics(NFREQ, discrete_inputs['Mmethod'], discrete_inputs['lump'], float(inputs['tol']), float(inputs['shift']))
        # ----------------------------

        # ------ static load cases ------------
        n_dlcs = self.options["n_dlcs"]
        gy = 0.0
        gx = -gravity * np.sin(tilt)
        gz = -gravity * np.cos(tilt)
        for k in range(n_dlcs):
            # gravity in the X, Y, Z, directions (global)
            load = frame3dd.StaticLoadCase(gx, gy, gz)

            # point loads
            # TODO: Are input loads aligned with the lss? If so they need to be rotated.
            load.changePointLoads(
                [inode[-1]], [F_hub[0, k]], [F_hub[1, k]], [F_hub[2, k]], [M_hub[0, k]], [M_hub[1, k]], [M_hub[2, k]]
            )
            # -----------------------------------

            # Put all together and run
            myframe.addLoadCase(load)

        myframe.write('myframe_lss.3dd') # Debugging
        displacements, forces, reactions, internalForces, mass3dd, modal = myframe.run()

        # ------------ Loop over DLCs and append to outputs ------------
        rotor_gearbox_deflection = np.zeros(n_dlcs)
        rotor_gearbox_angle = np.zeros(n_dlcs)
        outputs["F_mb1"] = np.zeros((3, n_dlcs))
        outputs["F_mb2"] = np.zeros((3, n_dlcs))
        outputs["F_torq"] = np.zeros((3, n_dlcs))
        outputs["M_mb1"] = np.zeros((3, n_dlcs))
        outputs["M_mb2"] = np.zeros((3, n_dlcs))
        outputs["M_torq"] = np.zeros((3, n_dlcs))
        outputs["lss_axial_stress"] = np.zeros((n - 1, n_dlcs))
        outputs["lss_shear_stress"] = np.zeros((n - 1, n_dlcs))
        outputs["constr_lss_vonmises"] = np.zeros((n - 1, n_dlcs))
        for k in range(n_dlcs):
            # Deflections and rotations at torq attachment
            rotor_gearbox_deflection[k] = np.sqrt(
                displacements.dx[k, itorq - 1] ** 2
                + displacements.dy[k, itorq - 1] ** 2
                + displacements.dz[k, itorq - 1] ** 2
            )
            rotor_gearbox_angle[k] = (
                displacements.dxrot[k, itorq - 1]
                + displacements.dyrot[k, itorq - 1]
                + displacements.dzrot[k, itorq - 1]
            )

            # shear and bending, one per element (convert from local to global c.s.)
            Fx = forces.Nx[k, 1::2] #(v) internal force, thru the elements/body in x-dir
            Vy = forces.Vy[k, 1::2] #(v) shear force upwards (y- / z- dir)
            Vz = -forces.Vz[k, 1::2]
            F = np.sqrt(Vz**2 + Vy**2)

            Mxx = forces.Txx[k, 1::2]
            Myy = forces.Myy[k, 1::2]
            Mzz = -forces.Mzz[k, 1::2]
            M = np.sqrt(Myy**2 + Mzz**2)

            # Record total forces and moments
            outputs["F_mb1"][:, k] = -1.0 * np.array([reactions.Fx[k, 0], reactions.Fy[k, 0], reactions.Fz[k, 0]])
            outputs["F_mb2"][:, k] = -1.0 * np.array([reactions.Fx[k, 1], reactions.Fy[k, 1], reactions.Fz[k, 1]])
            outputs["F_torq"][:, k] = -1.0 * np.array([reactions.Fx[k, 2], reactions.Fy[k, 2], reactions.Fz[k, 2]])
            outputs["M_mb1"][:, k] = -1.0 * np.array([reactions.Mxx[k, 0], reactions.Myy[k, 0], reactions.Mzz[k, 0]])
            outputs["M_mb2"][:, k] = -1.0 * np.array([reactions.Mxx[k, 1], reactions.Myy[k, 1], reactions.Mzz[k, 1]])
            outputs["M_torq"][:, k] = -1.0 * np.array([reactions.Mxx[k, 2], reactions.Myy[k, 2], reactions.Mzz[k, 2]])
            outputs["lss_axial_stress"][:, k] = np.abs(Fx) / Ax + M / S
            outputs["lss_shear_stress"][:, k] = 2.0 * F / As + np.abs(Mxx) / C
            hoop = np.zeros(F.shape)

            outputs["constr_lss_vonmises"][:, k] = TubevonMisesStressUtilization(
                outputs["lss_axial_stress"][:, k],
                hoop,
                outputs["lss_shear_stress"][:, k],
                gamma,
                sigma_y,
            )
        outputs["torq_deflection"] = rotor_gearbox_deflection.max()
        outputs["torq_angle"] = rotor_gearbox_angle.max()
        outputs["constr_shaft_deflection"] = gamma * outputs["torq_deflection"] / torq_defl_allow
        outputs["constr_shaft_angle"] = gamma * outputs["torq_angle"] / torq_angle_allow

        # Load->stress conversion for fatigue
        cross_section_xy = 2.0 * np.trapz(t_lss, s_lss)
        ax_load2stress = np.zeros(6)
        ax_load2stress[0] = 1.0 / Ax[0]
        ax_load2stress[1] = 1.0 / cross_section_xy
        ax_load2stress[2] = 1.0 / cross_section_xy
        ax_load2stress[4] = 1.0 / S[0]
        ax_load2stress[5] = 1.0 / S[0]
        sh_load2stress = np.zeros(6)
        sh_load2stress[1] = 1.0 / As[0]
        sh_load2stress[2] = 1.0 / As[0]
        sh_load2stress[3] = 1.0 / C[0]
        outputs["lss_axial_load2stress"] = ax_load2stress
        outputs["lss_shear_load2stress"] = sh_load2stress

        #(v) length constraints for L_h1 and L_12, compared to bearing face widths
        L_12, L_h1 = float(s_lss[3]-s_lss[1]), float(s_lss[4]-s_lss[3])
        mb1_face_width = float(inputs["mb1_face_width"][0]) #(v) NEW
        mb2_face_width = float(inputs["mb2_face_width"][0]) #(v) NEW
        outputs["constr_Lh1_MB1fw"] = L_h1 - (mb1_face_width*0.5) #(v) Should be > 0
        outputs["constr_L12_MBsFW"] = L_12 - (mb1_face_width+mb2_face_width)*0.5 #(v) Should be > 0

class HSS_Frame(om.ExplicitComponent):
    """
    Run structural analysis of high speed shaft (HSS) between gearbox and generator (only for geared configurations).

    Parameters
    ----------
    tilt : float, [deg]
        Shaft tilt
    s_hss : numpy array[3], [m]
        Discretized s-coordinates along drivetrain, measured from bedplate (direct) or tower center (geared)
    hss_diameter : numpy array[2], [m]
        Lss discretized diameter values at coordinates
    hss_wall_thickness : numpy array[2], [m]
        Lss discretized thickness values at coordinates
    M_aero_hub : numpy array[3, n_dlcs], [N*m]
        Aero-only moment vector applied to the hub
    m_generator : float, [kg]
        Gearbox rotor mass
    cm_generator : float, [kg]
        Gearbox center of mass (measured from tower center)
    I_generator : numpy array[3], [kg*m**2]
        Gearbox moment of inertia (measured about its cm)
    hss_E : float, [Pa]
        modulus of elasticity
    hss_G : float, [Pa]
        shear modulus
    hss_rho : float, [kg/m**3]
        material density
    hss_Xy : float, [Pa]
        yield stress

    Returns
    -------
    hss_spring_constant : float, [N*m/rad]
        Equivalent spring constant for the high speed shaft froom T=(G*J/L)*theta
    hss_axial_stress : numpy array[5, n_dlcs], [Pa]
        Axial stress in Curved_beam structure
    hss_shear_stress : numpy array[5, n_dlcs], [Pa]
        Shear stress in Curved_beam structure
    hss_bending_stress : numpy array[5, n_dlcs], [Pa]
        Hoop stress in Curved_beam structure calculated with Roarks formulae
    constr_hss_vonmises : numpy array[5, n_dlcs]
        Sigma_y/Von_Mises
    F_generator : numpy array[3, n_dlcs], [N]
        Force vector applied to generator rotor (direct) or gearbox (geared) in hub c.s.
    M_generator : numpy array[3, n_dlcs], [N*m]
        Moment vector applied to generator rotor (direct) or gearbox (geared) in hub c.s.

    """

    def initialize(self):
        self.options.declare("n_dlcs")
        self.options.declare("modeling_options")

    def setup(self):
        n_dlcs = self.options["n_dlcs"]

        self.add_input("tilt", 0.0, units="deg")
        self.add_input("s_hss", val=np.zeros(3), units="m")
        self.add_input("hss_diameter", val=np.zeros(2), units="m")
        self.add_input("hss_wall_thickness", val=np.zeros(2), units="m")
        self.add_input("M_aero_hub", val=np.zeros((3, n_dlcs)), units="N*m")
        self.add_input("gear_ratio", val=1.0)
        self.add_input("s_generator", val=0.0, units="m")
        self.add_input("generator_mass", val=0.0, units="kg")
        self.add_input("generator_I", val=np.zeros(3), units="kg*m**2")
        self.add_input("brake_mass", val=0.0, units="kg")
        self.add_input("brake_I", val=np.zeros(3), units="kg*m**2")
        self.add_input("hss_E", val=0.0, units="Pa")
        self.add_input("hss_G", val=0.0, units="Pa")
        self.add_input("hss_rho", val=0.0, units="kg/m**3")
        self.add_input("hss_Xy", val=0.0, units="Pa")

        self.add_output("hss_spring_constant", 0.0, units="N*m/rad")
        self.add_output("hss_axial_stress", np.zeros((2, n_dlcs)), units="Pa")
        self.add_output("hss_shear_stress", np.zeros((2, n_dlcs)), units="Pa")
        self.add_output("hss_bending_stress", np.zeros((2, n_dlcs)), units="Pa")
        self.add_output("constr_hss_vonmises", np.zeros((2, n_dlcs)))
        self.add_output("F_generator", val=np.zeros((3, n_dlcs)), units="N")
        self.add_output("M_generator", val=np.zeros((3, n_dlcs)), units="N*m")

    def compute(self, inputs, outputs):
        # Unpack inputs
        tilt = float(np.deg2rad(inputs["tilt"][0]))

        s_hss = inputs["s_hss"]
        D_hss = inputs["hss_diameter"]
        t_hss = inputs["hss_wall_thickness"]

        s_generator = float(inputs["s_generator"][0])
        m_generator = float(inputs["generator_mass"][0])
        I_generator = inputs["generator_I"]

        m_brake = float(inputs["brake_mass"][0])
        I_brake = inputs["brake_I"]

        rho = float(inputs["hss_rho"][0])
        E = float(inputs["hss_E"][0])
        G = float(inputs["hss_G"][0])
        sigma_y = float(inputs["hss_Xy"][0])
        gamma_f = float(self.options["modeling_options"]["gamma_f"])
        gamma_m = float(self.options["modeling_options"]["gamma_m"])
        gamma_n = float(self.options["modeling_options"]["gamma_n"])
        gamma = gamma_f * gamma_m * gamma_n

        M_hub = inputs["M_aero_hub"]
        gear_ratio = float(inputs["gear_ratio"][0])

        # ------- node data ----------------
        n = len(s_hss)
        inode = np.arange(1, n + 1)
        ynode = znode = rnode = np.zeros(n)
        xnode = s_hss.copy()
        nodes = frame3dd.NodeData(inode, xnode, ynode, znode, rnode)
        # ------------------------------------

        # ------ reaction data ------------
        # Reaction at generator attachment
        rnode = [inode[0]]
        Rx = Ry = Rz = Rxx = Ryy = Rzz = np.array([RIGID])
        reactions = frame3dd.ReactionData(rnode, Rx, Ry, Rz, Rxx, Ryy, Rzz, rigid=RIGID)
        # -----------------------------------

        # ------ frame element data ------------
        hsscyl = tube_prop(s_hss, D_hss, t_hss)
        ielement = np.arange(1, n)
        N1 = np.arange(1, n)
        N2 = np.arange(2, n + 1)
        roll = np.zeros(n - 1)
        myones = np.ones(n - 1)
        Ax = hsscyl.Area
        As = hsscyl.Asx
        S = hsscyl.S
        C = hsscyl.C
        J0 = hsscyl.J0
        Jx = hsscyl.Ixx

        elements = frame3dd.ElementData(
            ielement, N1, N2, Ax, As, As, J0, Jx, Jx, E * myones, G * myones, roll, rho * myones
        )

        hss_stiff = G * J0 / np.diff(s_hss)
        outputs["hss_spring_constant"] = 1.0 / np.sum(1.0 / hss_stiff)
        # -----------------------------------

        # ------ options ------------
        shear = geom = True
        dx = -1
        options = frame3dd.Options(shear, geom, dx)
        # -----------------------------------

        # initialize frameDD3 object
        myframe = frame3dd.Frame(nodes, reactions, elements, options)

        # ------ add brake hub and generator rotor (direct) or generator (geared) extra mass ------------
        myframe.changeExtraNodeMass(
            np.r_[inode[1], inode[0]],
            [m_brake, m_generator],
            [I_brake[0], I_generator[0]],
            [I_brake[1], I_generator[1]],
            [I_brake[2], I_generator[2]],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, s_generator - s_hss[-1]],
            [0.0, 0.0],
            [0.0, 0.0],
            True,
        )
        # ------------------------------------

        # ------ static load cases ------------
        n_dlcs = self.options["n_dlcs"]
        gy = 0.0
        gx = -gravity * np.sin(tilt)
        gz = -gravity * np.cos(tilt)
        for k in range(n_dlcs):
            # gravity in the X, Y, Z, directions (global)
            load = frame3dd.StaticLoadCase(gx, gy, gz)

            # point loads
            Fx = Fy = Fz = My = Mz = np.zeros(1)
            Mx = M_hub[0] / gear_ratio
            load.changePointLoads([inode[-1]], Fx, Fy, Fz, Mx, My, Mz)
            # -----------------------------------

            # Put all together and run
            myframe.addLoadCase(load)

        # myframe.write('myframe_hss_frame.3dd') # Debugging
        displacements, forces, reactions, internalForces, mass3dd, modal = myframe.run()

        # Loop over DLCs and append to outputs
        outputs["F_generator"] = np.zeros((3, n_dlcs))
        outputs["M_generator"] = np.zeros((3, n_dlcs))
        outputs["hss_axial_stress"] = np.zeros((n - 1, n_dlcs))
        outputs["hss_shear_stress"] = np.zeros((n - 1, n_dlcs))
        outputs["hss_bending_stress"] = np.zeros((n - 1, n_dlcs))
        outputs["constr_hss_vonmises"] = np.zeros((n - 1, n_dlcs))
        for k in range(n_dlcs):
            # shear and bending, one per element (convert from local to global c.s.)
            Fx = forces.Nx[k, 1::2]
            Vy = forces.Vy[k, 1::2]
            Vz = -forces.Vz[k, 1::2]
            F = np.sqrt(Vz**2 + Vy**2)

            Mxx = forces.Txx[k, 1::2]
            Myy = forces.Myy[k, 1::2]
            Mzz = -forces.Mzz[k, 1::2]
            M = np.sqrt(Myy**2 + Mzz**2)

            # Record total forces and moments
            outputs["F_generator"][:, k] = -1.0 * np.array([reactions.Fx[k, 0], reactions.Fy[k, 0], reactions.Fz[k, 0]])
            outputs["M_generator"][:, k] = -1.0 * np.array(
                [reactions.Mxx[k, 0], reactions.Myy[k, 0], reactions.Mzz[k, 0]]
            )
            outputs["hss_axial_stress"][:, k] = np.abs(Fx) / Ax + M / S
            outputs["hss_shear_stress"][:, k] = 2.0 * F / As + np.abs(Mxx) / C
            hoop = np.zeros(F.shape)

            outputs["constr_hss_vonmises"][:, k] = TubevonMisesStressUtilization(
                outputs["hss_axial_stress"][:, k],
                hoop,
                outputs["hss_shear_stress"][:, k],
                gamma,
                sigma_y,
            )


class Nose_Stator_Bedplate_Frame(om.ExplicitComponent):
    """
    (DD) Run structural analysis of nose/turret with the generator stator and bedplate

    Parameters
    ----------
    upwind : boolean
        Flag whether the design is upwind or downwind
    tilt : float, [deg]
        Lss tilt
    s_nose : numpy array[5], [m]
        Discretized s-coordinates along drivetrain, measured from bedplate
    nose_diameter : numpy array[2], [m]
        Nose outer diameter from bearing 1 to bedplate
    nose_wall_thickness : numpy array[2], [m]
        Nose wall thickness
    x_bedplate : numpy array[12], [m]
        Bedplate centerline x-coordinates
    z_bedplate : numpy array[12], [m]
        Bedplate centerline z-coordinates
    x_bedplate_inner : numpy array[12], [m]
        Bedplate lower curve x-coordinates
    z_bedplate_inner : numpy array[12], [m]
        Bedplate lower curve z-coordinates
    x_bedplate_outer : numpy array[12], [m]
        Bedplate outer curve x-coordinates
    z_bedplate_outer : numpy array[12], [m]
        Bedplate outer curve z-coordinates
    D_bedplate : numpy array[12], [m]
        Bedplate diameters
    t_bedplate : numpy array[12], [m]
        Bedplate wall thickness (mirrors input)
    s_mb1 : float, [m]
        Bearing 1 s-coordinate along drivetrain, measured from bedplate
    s_mb2 : float, [m]
        Bearing 2 s-coordinate along drivetrain, measured from bedplate
    mb1_mass : float, [kg]
        component mass
    mb1_I : numpy array[3], [kg*m**2]
        component I
    mb1_max_defl_ang : float, [rad]
        Maximum allowable deflection angle
    mb2_mass : float, [kg]
        component mass
    mb2_I : numpy array[3], [kg*m**2]
        component I
    mb2_max_defl_ang : float, [rad]
        Maximum allowable deflection angle
    s_stator : float, [m]
        Generator stator attachment to lss s-coordinate measured from bedplate
    generator_stator_mass : float, [kg]
        Generator stator mass
    generator_stator_I : numpy array[3], [kg*m**2]
        Generator stator moment of inertia (measured about cm)
    F_mb1 : numpy array[3, n_dlcs], [N]
        Force vector applied to bearing 1 in hub c.s.
    F_mb2 : numpy array[3, n_dlcs], [N]
        Force vector applied to bearing 2 in hub c.s.
    M_mb1 : numpy array[3, n_dlcs], [N*m]
        Moment vector applied to bearing 1 in hub c.s.
    M_mb2 : numpy array[3, n_dlcs], [N*m]
        Moment vector applied to bearing 2 in hub c.s.
    other_mass : float, [kg]
        Mass of other nacelle components that rest on mainplate
    bedplate_E : float, [Pa]
        modulus of elasticity
    bedplate_G : float, [Pa]
        shear modulus
    bedplate_rho : float, [kg/m**3]
        material density
    bedplate_Xy : float, [Pa]
        yield stress
    stator_deflection_allowable : float, [m]
        Maximum allowable deflection at generator stator attachment
        = input from: user
    stator_angle_allowable : float, [rad]
        Maximum allowable rotation at generator stator attachment
        = input from: user

    Returns
    -------
    mb1_deflection : numpy array[n_dlcs], [m]
        Total deflection distance of bearing 1
    mb2_deflection : numpy array[n_dlcs], [m]
        Total deflection distance of bearing 2
    stator_deflection : float, [m]
        Maximum deflection distance at stator attachment
    mb1_angle : numpy array[n_dlcs], [rad]
        Total rotation angle of bearing 1
    mb2_angle : numpy array[n_dlcs], [rad]
        Total rotation angle of bearing 2
    stator_angle : float, [rad]
        Maximum rotation angle at stator attachment
    base_F : numpy array[3, n_dlcs], [N]
        Total reaction force at bedplate base in tower top coordinate system
    base_M : numpy array[3, n_dlcs], [N*m]
        Total reaction moment at bedplate base in tower top coordinate system
    bedplate_nose_axial_stress : numpy array[12+3, n_dlcs], [Pa]
        Axial stress in Curved_beam structure
    bedplate_nose_shear_stress : numpy array[12+3, n_dlcs], [Pa]
        Shear stress in Curved_beam structure
    bedplate_nose_bending_stress : numpy array[12+3, n_dlcs], [Pa]
        Hoop stress in Curved_beam structure calculated with Roarks formulae
    constr_bedplate_vonmises : numpy array[12+3, n_dlcs]
        Sigma_y/Von_Mises
    constr_mb1_defl : numpy array[n_dlcs]
        Angular deflection relative to limit of bearing 1 (should be <1)
    constr_mb2_defl : numpy array[n_dlcs]
        Angular deflection relative to limit of bearing 2 (should be <1)

    """

    def initialize(self):
        self.options.declare("n_dlcs")
        self.options.declare("modeling_options")

    def setup(self):
        n_dlcs = self.options["n_dlcs"]

        self.add_discrete_input("upwind", True)
        self.add_input("tilt", 0.0, units="deg")
        self.add_input("s_nose", val=np.zeros(5), units="m")
        self.add_input("nose_diameter", np.zeros(2), units="m")
        self.add_input("nose_wall_thickness", np.zeros(2), units="m")
        self.add_input("x_bedplate", val=np.zeros(12), units="m")
        self.add_input("z_bedplate", val=np.zeros(12), units="m")
        self.add_input("x_bedplate_inner", val=np.zeros(12), units="m")
        self.add_input("z_bedplate_inner", val=np.zeros(12), units="m")
        self.add_input("x_bedplate_outer", val=np.zeros(12), units="m")
        self.add_input("z_bedplate_outer", val=np.zeros(12), units="m")
        self.add_input("D_bedplate", val=np.zeros(12), units="m")
        self.add_input("t_bedplate", val=np.zeros(12), units="m")
        self.add_input("s_mb1", val=0.0, units="m")
        self.add_input("s_mb2", val=0.0, units="m")
        self.add_input("mb1_mass", 0.0, units="kg")
        self.add_input("mb1_I", np.zeros(3), units="kg*m**2")
        self.add_input("mb1_max_defl_ang", 0.0, units="rad")
        self.add_input("mb2_mass", 0.0, units="kg")
        self.add_input("mb2_I", np.zeros(3), units="kg*m**2")
        self.add_input("mb2_max_defl_ang", 0.0, units="rad")
        self.add_input("s_stator", val=0.0, units="m")
        self.add_input("generator_stator_mass", val=0.0, units="kg")
        self.add_input("generator_stator_I", val=np.zeros(3), units="kg*m**2")
        self.add_input("F_mb1", val=np.zeros((3, n_dlcs)), units="N")
        self.add_input("F_mb2", val=np.zeros((3, n_dlcs)), units="N")
        self.add_input("M_mb1", val=np.zeros((3, n_dlcs)), units="N*m")
        self.add_input("M_mb2", val=np.zeros((3, n_dlcs)), units="N*m")
        self.add_input("other_mass", val=0.0, units="kg")
        self.add_input("bedplate_E", val=0.0, units="Pa")
        self.add_input("bedplate_G", val=0.0, units="Pa")
        self.add_input("bedplate_rho", val=0.0, units="kg/m**3")
        self.add_input("bedplate_Xy", val=0.0, units="Pa")

        self.add_input("stator_deflection_allowable", val=1.0, units="m")
        self.add_input("stator_angle_allowable", val=1.0, units="rad")

        self.add_output("mb1_deflection", val=np.zeros(n_dlcs), units="m")
        self.add_output("mb2_deflection", val=np.zeros(n_dlcs), units="m")
        self.add_output("stator_deflection", val=0.0, units="m")
        self.add_output("mb1_angle", val=np.zeros(n_dlcs), units="rad")
        self.add_output("mb2_angle", val=np.zeros(n_dlcs), units="rad")
        self.add_output("stator_angle", val=0.0, units="rad")
        self.add_output("base_F", val=np.zeros((3, n_dlcs)), units="N")
        self.add_output("base_M", val=np.zeros((3, n_dlcs)), units="N*m")
        self.add_output("bedplate_nose_axial_stress", np.zeros((12 + 3, n_dlcs)), units="Pa")
        self.add_output("bedplate_nose_shear_stress", np.zeros((12 + 3, n_dlcs)), units="Pa")
        self.add_output("bedplate_nose_bending_stress", np.zeros((12 + 3, n_dlcs)), units="Pa")
        self.add_output("constr_bedplate_vonmises", np.zeros((12 + 3, n_dlcs)))
        self.add_output("constr_mb1_defl", val=np.zeros(n_dlcs))
        self.add_output("constr_mb2_defl", val=np.zeros(n_dlcs))
        self.add_output("constr_stator_deflection", 0.0)
        self.add_output("constr_stator_angle", 0.0)

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Unpack inputs
        upwind = discrete_inputs["upwind"]
        Cup = -1.0 if upwind else 1.0
        tiltD = float(inputs["tilt"][0])
        tiltR = np.deg2rad(tiltD)

        x_c = inputs["x_bedplate"]
        z_c = inputs["z_bedplate"]
        x_inner = inputs["x_bedplate_inner"]
        z_inner = inputs["z_bedplate_inner"]
        x_outer = inputs["x_bedplate_outer"]
        z_outer = inputs["z_bedplate_outer"]
        D_bed = inputs["D_bedplate"]
        t_bed = inputs["t_bedplate"]

        s_nose = inputs["s_nose"][1:]  # First point duplicated with bedplate
        D_nose = inputs["nose_diameter"]
        t_nose = inputs["nose_wall_thickness"]
        x_nose = s_nose.copy()
        x_nose *= Cup
        x_nose += x_c[-1]

        s_mb1 = float(inputs["s_mb1"][0])
        s_mb2 = float(inputs["s_mb2"][0])
        m_mb1 = float(inputs["mb1_mass"][0])
        m_mb2 = float(inputs["mb2_mass"][0])
        I_mb1 = inputs["mb1_I"]
        I_mb2 = inputs["mb2_I"]

        s_stator = float(inputs["s_stator"][0])
        m_stator = float(inputs["generator_stator_mass"][0])
        I_stator = inputs["generator_stator_I"]

        rho = float(inputs["bedplate_rho"][0])
        E = float(inputs["bedplate_E"][0])
        G = float(inputs["bedplate_G"][0])
        sigma_y = float(inputs["bedplate_Xy"][0])
        gamma_f = float(self.options["modeling_options"]["gamma_f"])
        gamma_m = float(self.options["modeling_options"]["gamma_m"])
        gamma_n = float(self.options["modeling_options"]["gamma_n"])
        gamma = gamma_f * gamma_m * gamma_n

        F_mb1 = inputs["F_mb1"]
        F_mb2 = inputs["F_mb2"]
        M_mb1 = inputs["M_mb1"]
        M_mb2 = inputs["M_mb2"]

        m_other = float(inputs["other_mass"][0])
        stator_defl_allow = float(inputs["stator_deflection_allowable"][0])
        stator_angle_allow = float(inputs["stator_angle_allowable"][0])

        # ------- node data ----------------
        n = len(x_c) + len(x_nose)
        inode = np.arange(1, n + 1)
        ynode = rnode = np.zeros(n)
        xnode = np.r_[x_c, x_nose]
        znode = np.r_[z_c, z_c[-1] * np.ones(x_nose.shape)]
        nodes = frame3dd.NodeData(inode, xnode, ynode, znode, rnode)
        # Grab indices for later
        inose = len(x_c)
        istator = inode[find_nearest(xnode, Cup * s_stator + x_c[-1])]
        i1 = inode[find_nearest(xnode, Cup * s_mb1 + x_c[-1])]
        i2 = inode[find_nearest(xnode, Cup * s_mb2 + x_c[-1])]
        # ------------------------------------

        # ------ reaction data ------------
        # Rigid base
        rnode = [int(inode[0])]
        rk = np.array([RIGID])
        reactions = frame3dd.ReactionData(rnode, rk, rk, rk, rk, rk, rk, rigid=RIGID)
        # -----------------------------------

        # ------ frame element data ------------
        bedcyl = tube_prop(x_c, D_bed, t_bed)
        nosecyl = tube_prop(inputs["s_nose"], D_nose, t_nose)
        ielement = np.arange(1, n)
        N1 = np.arange(1, n)
        N2 = np.arange(2, n + 1)
        roll = np.zeros(n - 1)
        myones = np.ones(n - 1)
        Ax = np.r_[bedcyl.Area, nosecyl.Area]
        As = np.r_[bedcyl.Asx, nosecyl.Asx]
        S = np.r_[bedcyl.S, nosecyl.S]
        C = np.r_[bedcyl.C, nosecyl.C]
        J0 = np.r_[bedcyl.J0, nosecyl.J0]
        Ixx = np.r_[bedcyl.Ixx, nosecyl.Ixx]

        elements = frame3dd.ElementData(
            ielement, N1, N2, Ax, As, As, J0, Ixx, Ixx, E * myones, G * myones, roll, rho * myones
        )
        # -----------------------------------

        # ------ options ------------
        shear = geom = True
        dx = -1
        options = frame3dd.Options(shear, geom, dx)
        # -----------------------------------

        # initialize frameDD3 object
        myframe = frame3dd.Frame(nodes, reactions, elements, options)

        # ------ add misc nacelle components at base and stator extra mass ------------
        myframe.changeExtraNodeMass(
            np.r_[inode[0], istator, i1, i2],
            [m_other, m_stator, m_mb1, m_mb2],
            [0.0, I_stator[0], I_mb1[0], I_mb2[0]],
            [0.0, I_stator[1], I_mb1[1], I_mb2[1]],
            [0.0, I_stator[2], I_mb1[2], I_mb2[2]],
            np.zeros(4),
            np.zeros(4),
            np.zeros(4),
            [0.0, 0.0, 0.0, 0.0],
            np.zeros(4),
            np.zeros(4),
            True,
        )
        # ------------------------------------

        # ------- NO dynamic analysis ----------
        # myframe.enableDynamics(NFREQ, discrete_inputs['Mmethod'], discrete_inputs['lump'], float(inputs['tol']), float(inputs['shift']))
        # ----------------------------

        # ------ static load cases ------------
        n_dlcs = self.options["n_dlcs"]
        gy = 0.0
        gx = -gravity * np.sin(tiltR)
        gz = -gravity * np.cos(tiltR)
        for k in range(n_dlcs):
            # gravity in the X, Y, Z, directions (global)
            load = frame3dd.StaticLoadCase(gx, gy, gz)

            # point loads
            F_12 = np.c_[F_mb2[:, k], F_mb1[:, k]]
            M_12 = np.c_[M_mb2[:, k], M_mb1[:, k]]
            load.changePointLoads(np.r_[i2, i1], F_12[0, :], F_12[1, :], F_12[2, :], M_12[0, :], M_12[1, :], M_12[2, :])
            # -----------------------------------

            # Put all together and run
            myframe.addLoadCase(load)

        # myframe.write('myframe_nose_DDstator_bed.3dd') # Debugging
        displacements, forces, reactions, internalForces, mass3dd, modal = myframe.run()

        # ------------ Bedplate "curved beam" geometry for post-processing -------------
        # Need to compute neutral axis, so shift points such that bedplate top is at x=0
        R_c = np.sqrt((x_c - x_c[-1]) ** 2 + z_c**2)
        Ro = np.sqrt((x_outer - x_c[-1]) ** 2 + z_outer**2)
        Ri = np.sqrt((x_inner - x_c[-1]) ** 2 + z_inner**2)
        r_bed_o = 0.5 * D_bed
        r_bed_i = r_bed_o - t_bed
        A_bed = np.pi * (r_bed_o**2 - r_bed_i**2)

        # Radius of the neutral axis
        # http://faculty.fairfield.edu/wdornfeld/ME311/BasicStressEqns-DBWallace.pdf
        R_n = A_bed / (2 * np.pi * (np.sqrt(R_c**2 - r_bed_i**2) - np.sqrt(R_c**2 - r_bed_o**2)))
        e_cn = R_c - R_n
        # ------------------------------------

        # Loop over DLCs and append to outputs
        outputs["mb1_deflection"] = np.zeros(n_dlcs)
        outputs["mb2_deflection"] = np.zeros(n_dlcs)
        stator_deflection = np.zeros(n_dlcs)
        outputs["mb1_angle"] = np.zeros(n_dlcs)
        outputs["mb2_angle"] = np.zeros(n_dlcs)
        stator_angle = np.zeros(n_dlcs)
        outputs["base_F"] = np.zeros((3, n_dlcs))
        outputs["base_M"] = np.zeros((3, n_dlcs))
        outputs["bedplate_nose_axial_stress"] = np.zeros((n - 1, n_dlcs))
        outputs["bedplate_nose_shear_stress"] = np.zeros((n - 1, n_dlcs))
        outputs["bedplate_nose_bending_stress"] = np.zeros((n - 1, n_dlcs))
        outputs["constr_bedplate_vonmises"] = np.zeros((n - 1, n_dlcs))
        for k in range(n_dlcs):
            # Deflections and rotations at bearings- how to sum up rotation angles?
            outputs["mb1_deflection"][k] = np.sqrt(
                displacements.dx[k, i1 - 1] ** 2 + displacements.dy[k, i1 - 1] ** 2 + displacements.dz[k, i1 - 1] ** 2
            )
            outputs["mb2_deflection"][k] = np.sqrt(
                displacements.dx[k, i2 - 1] ** 2 + displacements.dy[k, i2 - 1] ** 2 + displacements.dz[k, i2 - 1] ** 2
            )
            stator_deflection[k] = np.sqrt(
                displacements.dx[k, istator - 1] ** 2
                + displacements.dy[k, istator - 1] ** 2
                + displacements.dz[k, istator - 1] ** 2
            )
            outputs["mb1_angle"][k] = (
                displacements.dxrot[k, i1 - 1] + displacements.dyrot[k, i1 - 1] + displacements.dzrot[k, i1 - 1]
            )
            outputs["mb2_angle"][k] = (
                displacements.dxrot[k, i2 - 1] + displacements.dyrot[k, i2 - 1] + displacements.dzrot[k, i2 - 1]
            )
            stator_angle[k] = (
                displacements.dxrot[k, istator - 1]
                + displacements.dyrot[k, istator - 1]
                + displacements.dzrot[k, istator - 1]
            )

            # shear and bending, one per element (convert from local to global c.s.)
            Fx = forces.Nx[k, 1::2]
            Vy = forces.Vy[k, 1::2]
            Vz = -forces.Vz[k, 1::2]
            F = np.sqrt(Vz**2 + Vy**2)

            Mxx = forces.Txx[k, 1::2]
            Myy = forces.Myy[k, 1::2]
            Mzz = -forces.Mzz[k, 1::2]
            M = np.sqrt(Myy**2 + Mzz**2)

            # Record total forces and moments
            F_base_k = DirectionVector(-reactions.Fx[k, :].sum(), -reactions.Fy[k, :].sum(), -reactions.Fz[k, :].sum())
            M_base_k = DirectionVector(
                -reactions.Mxx[k, :].sum(), -reactions.Myy[k, :].sum(), -reactions.Mzz[k, :].sum()
            )

            # Rotate vector from tilt axes to yaw/tower axes
            outputs["base_F"][:, k] = F_base_k.hubToYaw(-tiltD).toArray()
            outputs["base_M"][:, k] = M_base_k.hubToYaw(-tiltD).toArray()

            outputs["bedplate_nose_axial_stress"][:, k] = np.abs(Fx) / Ax + M / S
            outputs["bedplate_nose_shear_stress"][:, k] = 2.0 * F / As + np.abs(Mxx) / C

            Bending_stress_outer = M[: (inose - 1)] * nodal2sectional((Ro - R_n) / (A_bed * e_cn * Ro))[0]
            # Bending_stress_inner = M[:(inose-1)] * nodal2sectional( (R_n-Ri) / (A_bed*e_cn*Ri) )[0]
            outputs["bedplate_nose_bending_stress"][: (inose - 1), k] = Bending_stress_outer

            outputs["constr_bedplate_vonmises"][:, k] = TubevonMisesStressUtilization(
                outputs["bedplate_nose_axial_stress"][:, k],
                outputs["bedplate_nose_bending_stress"][:, k],
                outputs["bedplate_nose_shear_stress"][:, k],
                gamma,
                sigma_y,
            )

        # Evaluate bearing limits
        outputs["constr_mb1_defl"] = np.abs(outputs["mb1_angle"] / inputs["mb1_max_defl_ang"])
        outputs["constr_mb2_defl"] = np.abs(outputs["mb2_angle"] / inputs["mb2_max_defl_ang"])
        outputs["stator_deflection"] = stator_deflection.max()
        outputs["stator_angle"] = np.abs(stator_angle).max()
        outputs["constr_stator_deflection"] = gamma * outputs["stator_deflection"] / stator_defl_allow
        outputs["constr_stator_angle"] = gamma * outputs["stator_angle"] / stator_angle_allow


class Bedplate_IBeam_Frame(om.ExplicitComponent):
    """
    Run structural analysis of bedplate in geared configuration with 2 X 2 parallel I beams providing support

    Parameters
    ----------
    upwind : boolean
        Flag whether the design is upwind or downwind
    tilt : float, [deg]
        Lss tilt
    D_top : float, [m]
        Diameter tower top
    s_drive : numpy array[?], [m]
        Discretized s-coordinates along drivetrain
    bedplate_flange_width : float, [m]
        Bedplate is two parallel I beams, this is the flange width
    bedplate_flange_thickness : float, [m]
        Bedplate is two parallel I beams, this is the flange thickness
    bedplate_web_thickness : float, [m]
        Bedplate is two parallel I beams, this is the web thickness
    bedplate_web_height : float, [m]
        Bedplate is two parallel I beams, this is the web height
    s_mb1 : float, [m]
        Bearing 1 s-coordinate along drivetrain, measured from bedplate
    s_mb2 : float, [m]
        Bearing 2 s-coordinate along drivetrain, measured from bedplate
    mb1_mass : float, [kg]
        component mass
        = input from: MainBearing.mb_mass, user must 'connect'; (all below)
    mb1_I : numpy array[3], [kg*m**2]
        component I
    mb1_max_defl_ang : float, [rad]
        Maximum allowable deflection angle
    mb2_mass : float, [kg]
        component mass
    mb2_I : numpy array[3], [kg*m**2]
        component I
    mb2_max_defl_ang : float, [rad]
        Maximum allowable deflection angle
    s_gearbox : float, [m]
        Gearbox attachment to lss s-coordinate
    s_generator : float, [m]
        Generator attachment to lss s-coordinate
    F_mb1 : numpy array[3, n_dlcs], [N]
        Force vector applied to bearing 1 in hub c.s.
    F_mb2 : numpy array[3, n_dlcs], [N]
        Force vector applied to bearing 2 in hub c.s.
    F_torq : numpy array[3, n_dlcs], [N]
        Force vector applied to generator rotor (direct) or gearbox (geared) in hub c.s.
    F_generator : numpy array[3, n_dlcs], [N]
        Force vector applied to generator in hub c.s.
        = input from: HSS_Frame, TODO: user rn
    M_mb1 : numpy array[3, n_dlcs], [N*m]
        Moment vector applied to bearing 1 in hub c.s.
        = input from: Hub_Rotor_LSS_Frame (all below)
    M_mb2 : numpy array[3, n_dlcs], [N*m]
        Moment vector applied to bearing 2 in hub c.s.
    M_torq : numpy array[3, n_dlcs], [N*m]
        Moment vector applied to generator rotor (direct) or gearbox (geared) in hub c.s.
    M_generator : numpy array[3, n_dlcs], [N]
        Moment vector applied to generator in hub c.s.
        = input from: HSS_Frame, TODO: user rn
    other_mass : float, [kg]
        Mass of other nacelle components that rest on mainplate
    bedplate_E : float, [Pa]
        modulus of elasticity
        = input from: DriveMaterials (all below)
    bedplate_G : float, [Pa]
        shear modulus
    bedplate_rho : float, [kg/m**3]
        material density
    bedplate_Xy : float, [Pa]
        yield stress

    Returns
    -------
    mb1_deflection : numpy array[n_dlcs], [m]
        Total deflection distance of bearing 1
    mb2_deflection : numpy array[n_dlcs], [m]
        Total deflection distance of bearing 2
    bedplate_deflection : float, [m]
        Maximum deflection distance at bedplate I-beam tip
    mb1_angle : numpy array[n_dlcs], [rad]
        Total rotation angle of bearing 1
    mb2_angle : numpy array[n_dlcs], [rad]
        Total rotation angle of bearing 2
    bedplate_angle : float, [rad]
        Maximum rotation angle at bedplate I-beam tip
    base_F : numpy array[3, n_dlcs], [N]
        Total reaction force at bedplate base in tower top coordinate system
    base_M : numpy array[3, n_dlcs], [N*m]
        Total reaction moment at bedplate base in tower top coordinate system
    bedplate_axial_stress : numpy array[22, n_dlcs], [Pa]
        Axial stress in Curved_beam structure
    bedplate_shear_stress : numpy array[22, n_dlcs], [Pa]
        Shear stress in Curved_beam structure
    bedplate_bending_stress : numpy array[22, n_dlcs], [Pa]
        Hoop stress in Curved_beam structure calculated with Roarks formulae
    constr_bedplate_vonmises : numpy array[22, n_dlcs]
        Sigma_y/Von_Mises
    constr_mb1_defl : numpy array[n_dlcs]
        Angular deflection relative to limit of bearing 1 (should be <1)
    constr_mb2_defl : numpy array[n_dlcs]
        Angular deflection relative to limit of bearing 2 (should be <1)

    """

    def initialize(self):
        self.options.declare("n_dlcs")
        self.options.declare("modeling_options")

    def setup(self):
        n_dlcs = self.options["n_dlcs"]

        self.add_discrete_input("upwind", True)
        self.add_input("tilt", 0.0, units="deg")
        self.add_input("D_top", 0.0, units="m")
        self.add_input("s_drive", val=np.zeros(12), units="m")
        self.add_input("bedplate_flange_width", val=0.0, units="m")
        self.add_input("bedplate_flange_thickness", val=0.0, units="m")
        self.add_input("bedplate_web_height", val=0.0, units="m")
        self.add_input("bedplate_web_thickness", val=0.0, units="m")
        self.add_input("s_mb1", val=0.0, units="m")
        self.add_input("s_mb2", val=0.0, units="m")
        self.add_input("mb1_mass", 0.0, units="kg")
        self.add_input("mb1_I", np.zeros(3), units="kg*m**2")
        self.add_input("mb1_max_defl_ang", 0.0, units="rad")
        self.add_input("mb2_mass", 0.0, units="kg")
        self.add_input("mb2_I", np.zeros(3), units="kg*m**2")
        self.add_input("mb2_max_defl_ang", 0.0, units="rad")
        self.add_input("s_gearbox", val=0.0, units="m")
        self.add_input("s_generator", val=0.0, units="m")
        self.add_input("F_mb1", val=np.zeros((3, n_dlcs)), units="N")
        self.add_input("F_mb2", val=np.zeros((3, n_dlcs)), units="N")
        self.add_input("F_torq", val=np.zeros((3, n_dlcs)), units="N")
        self.add_input("F_generator", val=np.zeros((3, n_dlcs)), units="N")
        self.add_input("M_mb1", val=np.zeros((3, n_dlcs)), units="N*m")
        self.add_input("M_mb2", val=np.zeros((3, n_dlcs)), units="N*m")
        self.add_input("M_torq", val=np.zeros((3, n_dlcs)), units="N*m")
        self.add_input("M_generator", val=np.zeros((3, n_dlcs)), units="N*m")
        self.add_input("other_mass", val=0.0, units="kg")
        self.add_input("bedplate_E", val=0.0, units="Pa")
        self.add_input("bedplate_G", val=0.0, units="Pa")
        self.add_input("bedplate_rho", val=0.0, units="kg/m**3")
        self.add_input("bedplate_Xy", val=0.0, units="Pa")

        self.add_input("stator_deflection_allowable", val=1.0, units="m")
        self.add_input("stator_angle_allowable", val=1.0, units="rad")

        self.add_output("mb1_deflection", val=np.zeros(n_dlcs), units="m")
        self.add_output("mb2_deflection", val=np.zeros(n_dlcs), units="m")
        self.add_output("stator_deflection", val=0.0, units="m")
        self.add_output("mb1_angle", val=np.zeros(n_dlcs), units="rad")
        self.add_output("mb2_angle", val=np.zeros(n_dlcs), units="rad")
        self.add_output("stator_angle", val=0.0, units="rad")
        self.add_output("base_F", val=np.zeros((3, n_dlcs)), units="N")
        self.add_output("base_M", val=np.zeros((3, n_dlcs)), units="N*m")
        self.add_output("bedplate_axial_stress", np.zeros((22, n_dlcs)), units="Pa")
        self.add_output("bedplate_shear_stress", np.zeros((22, n_dlcs)), units="Pa")
        self.add_output("bedplate_bending_stress", np.zeros((22, n_dlcs)), units="Pa")
        self.add_output("constr_bedplate_vonmises", np.zeros((22, n_dlcs)))
        self.add_output("constr_mb1_defl", val=np.zeros(n_dlcs))
        self.add_output("constr_mb2_defl", val=np.zeros(n_dlcs))
        self.add_output("constr_stator_deflection", 0.0)
        self.add_output("constr_stator_angle", 0.0)

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Unpack inputs
        upwind = discrete_inputs["upwind"]
        Cup = -1.0 if upwind else 1.0
        D_top = float(inputs["D_top"][0])
        tiltD = float(inputs["tilt"][0])
        tiltR = np.deg2rad(tiltD)
        s_drive = inputs["s_drive"] #(v) eg. [-0.6374810261863306, 0.43751897381366933, 1.5125189738136693, 2.356110670770005, 3.1997023677263408, 4.999702367726341, 6.799702367726341, 6.8997023677263405, 7.52625494207485, 8.15280751642336, 8.532663244734854, 8.912518973046348].shape = 12

        s_gear = float(inputs["s_gearbox"][0])
        s_gen = float(inputs["s_generator"][0])

        s_mb1 = float(inputs["s_mb1"][0])
        s_mb2 = float(inputs["s_mb2"][0])
        m_mb1 = float(inputs["mb1_mass"][0])
        m_mb2 = float(inputs["mb2_mass"][0])
        I_mb1 = inputs["mb1_I"]
        I_mb2 = inputs["mb2_I"]

        bed_w_flange = float(inputs["bedplate_flange_width"][0])
        bed_t_flange = float(inputs["bedplate_flange_thickness"][0])
        bed_h_web = float(inputs["bedplate_web_height"][0])
        bed_t_web = float(inputs["bedplate_web_thickness"][0])

        rho = float(inputs["bedplate_rho"][0])
        E = float(inputs["bedplate_E"][0])
        G = float(inputs["bedplate_G"][0])
        sigma_y = float(inputs["bedplate_Xy"][0])
        gamma_f = float(self.options["modeling_options"]["gamma_f"])
        gamma_m = float(self.options["modeling_options"]["gamma_m"])
        gamma_n = float(self.options["modeling_options"]["gamma_n"])
        gamma = gamma_f * gamma_m * gamma_n

        F_mb1 = inputs["F_mb1"]
        F_mb2 = inputs["F_mb2"]
        F_gear = inputs["F_torq"]
        F_gen = inputs["F_generator"]
        M_mb1 = inputs["M_mb1"]
        M_mb2 = inputs["M_mb2"]
        M_gear = inputs["M_torq"]
        M_gen = inputs["M_generator"]

        m_other = float(inputs["other_mass"][0])
        stator_defl_allow = float(inputs["stator_deflection_allowable"][0])
        stator_angle_allow = float(inputs["stator_angle_allowable"][0])

        # ------- node data ----------------
        n = len(s_drive) #(v) n = 12
        inode = np.arange(1, 3 * n + 1)
        ynode = 0.25 * D_top * np.r_[np.zeros(n), np.ones(n), -np.ones(n)]
        xnode = s_drive * np.cos(tiltR)
        xnode = Cup * np.r_[xnode, xnode, xnode]
        znode = rnode = np.zeros(3 * n)
        nodes = frame3dd.NodeData(inode, xnode, ynode, znode, rnode)
        # Grab indices for later
        igenerator = inode[find_nearest(s_drive, s_gen)]
        igearbox = inode[find_nearest(s_drive, s_gear)]
        itower = inode[find_nearest(s_drive, 0.0)]
        i1 = inode[find_nearest(s_drive, s_mb1)]
        i2 = inode[find_nearest(s_drive, s_mb2)]
        # ------------------------------------

        # ------ reaction data ------------
        # Rigid base
        rnode = np.int_(np.r_[itower + n, itower + 2 * n])
        rk = np.array([RIGID, RIGID])
        reactions = frame3dd.ReactionData(rnode, rk, rk, rk, rk, rk, rk, rigid=RIGID)
        # -----------------------------------

        # ------ frame element data ------------
        # 2 parallel I-beams, with inner line of points to receive loads & distribute
        # No connections on ghost nodes, just connections to I-beams at same xval
        # Elements connect the lines, then connect the I-beams to the center nodes
        myI = IBeam(bed_w_flange, bed_t_flange, bed_h_web, bed_t_web)

        N1i = np.arange(1, n)
        N2i = np.arange(2, n + 1)
        Ni = np.arange(1, n + 1)
        N1 = np.r_[N1i + n, N1i + 2 * n, Ni + n, Ni + 2 * n]
        N2 = np.r_[N2i + n, N2i + 2 * n, Ni, Ni]
        ielement = np.arange(1, N1.size + 1)
        roll = np.zeros(N1.size)
        myones = np.ones(N1i.size)
        plate1s = np.ones(Ni.size)

        A_plate = As_plate = S_plate = C_plate = I_plate = 1e3 * plate1s
        rho_plate = 1e-6 * plate1s
        E_plate = G_plate = 1e16 * plate1s

        Ax = np.r_[myI.Area * myones, myI.Area * myones, A_plate, A_plate]
        Asy = np.r_[myI.Asx * myones, myI.Asx * myones, As_plate, As_plate]
        Asz = np.r_[myI.Asy * myones, myI.Asy * myones, As_plate, As_plate]
        Sy = np.r_[myI.Sx * myones, myI.Sx * myones, S_plate, S_plate]
        Sz = np.r_[myI.Sy * myones, myI.Sy * myones, S_plate, S_plate]
        C = np.r_[myI.C * myones, myI.C * myones, C_plate, C_plate]
        J0 = np.r_[myI.J0 * myones, myI.J0 * myones, I_plate, I_plate]
        Jy = np.r_[myI.Ixx * myones, myI.Ixx * myones, I_plate, I_plate]
        Jz = np.r_[myI.Iyy * myones, myI.Iyy * myones, I_plate, I_plate]
        myE = np.r_[E * myones, E * myones, E_plate, E_plate]
        myG = np.r_[G * myones, G * myones, G_plate, G_plate]
        myrho = np.r_[rho * myones, rho * myones, rho_plate, rho_plate]

        elements = frame3dd.ElementData(ielement, N1, N2, Ax, Asy, Asz, J0, Jy, Jz, myE, myG, roll, myrho)
        # -----------------------------------

        # ------ options ------------
        shear = geom = True
        dx = -1
        options = frame3dd.Options(shear, geom, dx)
        # -----------------------------------

        # initialize frameDD3 object
        myframe = frame3dd.Frame(nodes, reactions, elements, options)

        # ------ add misc nacelle components at base and stator extra mass ------------
        myframe.changeExtraNodeMass(
            np.r_[itower, i1, i2],
            [m_other, m_mb1, m_mb2],
            [0.0, I_mb1[0], I_mb2[0]],
            [0.0, I_mb1[1], I_mb2[1]],
            [0.0, I_mb1[2], I_mb2[2]],
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            True,
        )
        # ------------------------------------

        # ------- NO dynamic analysis ----------
        # myframe.enableDynamics(NFREQ, discrete_inputs['Mmethod'], discrete_inputs['lump'], float(inputs['tol']), float(inputs['shift']))
        # ----------------------------

        # ------ static load cases ------------
        n_dlcs = self.options["n_dlcs"]
        gx = gy = 0.0
        gz = -gravity
        for k in range(n_dlcs):
            # gravity in the X, Y, Z, directions (global)
            load = frame3dd.StaticLoadCase(gx, gy, gz)

            # point loads
            F_ext = np.c_[F_mb2[:, k], F_mb1[:, k], F_gear[:, k], F_gen[:, k]]
            M_ext = np.c_[M_mb2[:, k], M_mb1[:, k], M_gear[:, k], M_gen[:, k]]
            F_rot = DirectionVector(F_ext[0, :], F_ext[1, :], F_ext[2, :]).hubToYaw(-tiltD).toArray()
            M_rot = DirectionVector(M_ext[0, :], M_ext[1, :], M_ext[2, :]).hubToYaw(-tiltD).toArray()
            load.changePointLoads(
                np.r_[i2, i1, igearbox, igenerator],
                F_rot[:, 0],
                F_rot[:, 1],
                F_rot[:, 2],
                M_rot[:, 0],
                M_rot[:, 1],
                M_rot[:, 2],
            )
            # -----------------------------------

            # Put all together and run
            myframe.addLoadCase(load)

        myframe.write('myframe_bedplate.3dd') # Debugging
        displacements, forces, reactions, internalForces, mass3dd, modal = myframe.run()

        # Loop over DLCs and append to outputs
        outputs["mb1_deflection"] = np.zeros(n_dlcs)
        outputs["mb2_deflection"] = np.zeros(n_dlcs)
        outputs["mb1_angle"] = np.zeros(n_dlcs)
        outputs["mb2_angle"] = np.zeros(n_dlcs)
        outputs["base_F"] = np.zeros((3, n_dlcs))
        outputs["base_M"] = np.zeros((3, n_dlcs))
        outputs["bedplate_axial_stress"] = np.zeros((2 * n - 2, n_dlcs))
        outputs["bedplate_shear_stress"] = np.zeros((2 * n - 2, n_dlcs))
        outputs["bedplate_bending_stress"] = np.zeros((2 * n - 2, n_dlcs))
        outputs["constr_bedplate_vonmises"] = np.zeros((2 * n - 2, n_dlcs))
        for k in range(n_dlcs):
            # Deflections and rotations at bearings- how to sum up rotation angles? TODO
            outputs["mb1_deflection"][k] = np.sqrt(             #(v) TODO: not used as constr for mb1 (linear) defl?
                displacements.dx[k, i1 - 1] ** 2 + displacements.dy[k, i1 - 1] ** 2 + displacements.dz[k, i1 - 1] ** 2
            )
            outputs["mb2_deflection"][k] = np.sqrt(
                displacements.dx[k, i2 - 1] ** 2 + displacements.dy[k, i2 - 1] ** 2 + displacements.dz[k, i2 - 1] ** 2
            )
            bedplate_deflection = np.maximum(
                np.sqrt(displacements.dx[k, n] ** 2 + displacements.dy[k, n] ** 2 + displacements.dz[k, n] ** 2),
                np.sqrt(displacements.dx[k, -1] ** 2 + displacements.dy[k, -1] ** 2 + displacements.dz[k, -1] ** 2),
            )
            outputs["mb1_angle"][k] = (
                displacements.dxrot[k, i1 - 1] + displacements.dyrot[k, i1 - 1] + displacements.dzrot[k, i1 - 1]
            )
            outputs["mb2_angle"][k] = (
                displacements.dxrot[k, i2 - 1] + displacements.dyrot[k, i2 - 1] + displacements.dzrot[k, i2 - 1]
            )
            bedplate_angle = np.maximum(
                displacements.dxrot[k, n] + displacements.dyrot[k, n] + displacements.dzrot[k, n],
                displacements.dxrot[k, -1] + displacements.dyrot[k, -1] + displacements.dzrot[k, -1],
            )

            # shear and bending, one per element (convert from local to global c.s.)
            Fx = forces.Nx[k, 1::2]
            Vy = forces.Vy[k, 1::2]
            Vz = -forces.Vz[k, 1::2]
            # F  =  np.sqrt(Vz**2 + Vy**2)

            Mxx = forces.Txx[k, 1::2]
            Myy = forces.Myy[k, 1::2]
            Mzz = -forces.Mzz[k, 1::2]
            # M   =  np.sqrt(Myy**2 + Mzz**2)

            # Record total forces and moments at base
            outputs["base_F"][:, k] = np.r_[
                -reactions.Fx[k, :].sum(), -reactions.Fy[k, :].sum(), -reactions.Fz[k, :].sum()
            ]
            outputs["base_M"][:, k] = np.r_[
                -reactions.Mxx[k, :].sum(), -reactions.Myy[k, :].sum(), -reactions.Mzz[k, :].sum()
            ]

            outputs["bedplate_axial_stress"][:, k] = (np.abs(Fx) / Ax + np.abs(Myy) / Sy + np.abs(Mzz) / Sz)[
                : (2 * n - 2)
            ]
            outputs["bedplate_shear_stress"][:, k] = (2.0 * (np.abs(Vy) / Asy + np.abs(Vz) / Asz) + np.abs(Mxx) / C)[
                : (2 * n - 2)
            ]
            hoop = np.zeros(2 * n - 2)

            outputs["constr_bedplate_vonmises"][:, k] = TubevonMisesStressUtilization(
                outputs["bedplate_axial_stress"][:, k],
                hoop,
                outputs["bedplate_shear_stress"][:, k],
                gamma,
                sigma_y,
            )

        # Evaluate bearing limits
        outputs["constr_mb1_defl"] = outputs["mb1_angle"] / inputs["mb1_max_defl_ang"] #(v) Should be < 1
        outputs["constr_mb2_defl"] = outputs["mb2_angle"] / inputs["mb2_max_defl_ang"]
        outputs["stator_deflection"] = bedplate_deflection.max()
        outputs["stator_angle"] = bedplate_angle.max()
        outputs["constr_stator_deflection"] = gamma * outputs["stator_deflection"] / stator_defl_allow
        outputs["constr_stator_angle"] = gamma * outputs["stator_angle"] / stator_angle_allow

#%%[markdown]
# Vasudev Gupta: adaptations
# ============================================================
#%%
class FLS_Hub_Rotor_LSS_Frame (Hub_Rotor_LSS_Frame):
    "inherits from `Hub_Rotor_LSS_Frame` class component for its _inputs and compute()"

    def initialize(self):
        super().initialize()
        self.options.declare('batch_size', default=100, 
                           desc='Number of time steps to process at once')

    def setup( self ):
        super().setup() # call the parent setup for its inputs'

        mod_opt = self.options['modeling_options']
        # n_ws = mod_opt.get('n_ws', 10) #TODO: check WEIS for its input names, if any
        # n_t = mod_opt.get('n_time_steps', 72000)  # Reduced for memory

class DrivetrainFLS(Hub_Rotor_LSS_Frame):
    """
    Extends Hub_Rotor_LSS_Frame to compute bearing loads for FLS using
    time-series hub loads from OpenFAST.
    """

    import scipy.io as sio

    def initialize(self):
        super().initialize()
        self.options.declare('hub_loads', types=dict,
                             desc='Dictionary with time-series hub loads: F_aero_hub, M_aero_hub')

    def setup(self):
        # Call parent setup for geometry and material inputs
        super().setup()
        # Add output for aggregated bearing loads
        self.add_output('bearing_loads', shape_by_conn=True) #TODO: shape_ correct?

    def compute(self, inputs, outputs):
        hub_loads = self.options['hub_loads']
        bearing_load_series = []

        # Loop through hub loads and call parent compute
        for F_hub, M_hub in zip(hub_loads['F_aero_hub'], hub_loads['M_aero_hub']):
            # Set loads for this time step
            self.F_hub = F_hub
            self.M_hub = M_hub

            # Call parent compute to calculate F_mb
            super().compute(inputs, outputs)

            # Store bearing loads for fatigue analysis
            bearing_load_series.append(outputs['F_mb'])

        outputs['bearing_loads'] = bearing_load_series

#%% =======================================================
#   ============ analytical implementations ===============
#   =======================================================

# ---------------
def analytical_MB_Forces( Fx,Fy,Fz, Mx,My,Mz, L_h1,L_12, flag_jac=False ):
    """
    Analytical low-fidelity (quick) main bearing force calculation,
    using static equilibrium or moment balance, 
    unlike higher fidelity `Hub_Rotor_LSS_Frame` using pyFrame3DD.
    [depr.]

    Inputs
    -------
    F* : array[ # of time steps , # of wind speeds ]
        Forces (aero) on the hub center/main shaft input; for FLS, shape=(72e4,10)
    M* : array[ # of time steps , # of wind speeds ]
        Moment (aero) on the hub center/main shaft input; for FLS, shape=(72e4,10)
    L_h1 : float
        length along main shaft btw hub and mb1
    L_12 : float
        length along main shaft btw mb1 and mb2
    flag_jac : Boolean
        whether to provide analytical derivatives (True) or not (False)

    Outputs
    -------
    F_mb1 : All forces on the 1. main bearing
        shape = (4, shape(F*) )
    F_mb2 : All forces on the 2. main bearing
        shape = (4, shape(F*) )
    
    --- if flag_jac = True ---
    dFmb1_dLh1 : All derivatives of mb1 forces wrt. L_h1
        shape = (4, shape(F_mb1) )
    dFmb1_dLh1 : All derivatives of mb1 forces wrt. L_12
        shape = (4, shape(F_mb1) )
    dFmb2_dLh1 : All derivatives of mb2 forces wrt. L_h1
        shape = (4, shape(F_mb2) )
    dFmb2_dL12 : All derivatives of mb2 forces wrt. L_12
        shape = (4, shape(F_mb2) )

    Internal Progress
    --------------
    - DONE : implement as a openMDAO Explicit Component
        depr, due to 'known and fixed' shape of F* and M*
    - DONE : implement as a function, general purpose
    - DONE : add analytical gradients
    - TODO : mag of F* and M* is O(6)! exploding jac? how to scale
    """
    
    ### === Loads on bearings ===
    # init
    n_ts, n_ws = Fx.shape[0], Fx.shape[1] # = 72e4, 10
    F_mb1 = np.zeros( (n_ts,n_ws,4) ) # for 4 forces (ax,y,z,rad)
    F_mb2 = np.zeros( (n_ts,n_ws,4) ) # for 4 forces (ax,y,z,rad)

    # ----- MB2 (downwind) reactions
    F_mb2_ax = np.abs(Fx)   # 2TRB typically
    F_mb2_y = (-Mz + Fy * L_h1) / L_12  #
    F_mb2_z = (My + Fz * L_h1) / L_12   #
    # stable, vectorized hypot (handles overflow/underflow)
    F_mb2_rad = np.hypot(F_mb2_y, F_mb2_z) # element-wise

    # ----- MB1 (upwind) reactions
    F_mb1_ax = np.zeros_like( F_mb2_ax )  # CRB typically
    F_mb1_y = -F_mb2_y - Fy #
    F_mb1_z = -F_mb2_z - Fz #
    F_mb1_rad = np.hypot(F_mb1_y, F_mb1_z)

    # ----- collect for outputs
    F_mb1 = np.stack([F_mb1_ax, F_mb1_y, F_mb1_z, F_mb1_rad])
    F_mb2 = np.stack([F_mb2_ax, F_mb2_y, F_mb2_z, F_mb2_rad])

    if not flag_jac:
        return F_mb1, F_mb2
    # ==============================

    else:
    ### === Derivatives of loads wrt. L_* ===
    # init
        dFmb1_dLh1 = F_mb1                # ----- main jac outputs -----
        dFmb1_dL12 = F_mb1
        dFmb2_dLh1 = F_mb2
        dFmb2_dL12 = F_mb2

        # ----- MB2 -----            
        dFmb2ax_dLh1 = np.zeros_like(F_mb2_ax)
        dFmb2ax_dL12 = np.zeros_like(F_mb2_ax)
        # NOTE: no need to use `utils/smooth_abs` coz dFx_dL* = 0 anyway
            
        # ----- mb2_y diff wrt. L_
        dFmb2y_dLh1 = Fy / L_12
        dFmb2y_dL12 = (-1/L_12) * F_mb2_y

        # ----- mb2_z diff wrt. L_
        dFmb2z_dLh1 = Fz / L_12
        dFmb2z_dL12 = (-1/L_12) * F_mb2_z
        
        # ----- mb2_r diff wrt. y, z
        dFmb2rad_dFmb2y = F_mb2_y / F_mb2_rad
        dFmb2rad_dFmb2z = F_mb2_z / F_mb2_rad
        # ----- mb2_r diff wrt. L_
        dFmb2rad_dLh1 = (dFmb2rad_dFmb2y*dFmb2y_dLh1) + (dFmb2rad_dFmb2z*dFmb2z_dLh1)
        dFmb2rad_dL12 = (dFmb2rad_dFmb2y*dFmb2y_dL12) + (dFmb2rad_dFmb2z*dFmb2z_dL12)
        # ---------------

        # ----- MB1 -----
        dFmb1ax_dLh1 = F_mb1_ax # both 0s
        dFmb1ax_dL12 = F_mb1_ax
        
        # ----- mb1_y diff wrt. L_
        dFmb1y_dLh1 = -dFmb2y_dLh1
        dFmb1y_dL12 = -dFmb2y_dL12

        # ----- mb1_z diff wrt. L_
        dFmb1z_dLh1 = -dFmb2z_dLh1
        dFmb1z_dL12 = -dFmb2z_dL12
        
        # ----- mb1_r diff wrt. y, z
        dFmb1rad_dFmb1y = F_mb1_y / F_mb1_rad
        dFmb1rad_dFmb1z = F_mb1_z / F_mb1_rad
        # ----- mb1_r wrt. L_
        dFmb1rad_dLh1 = (dFmb1rad_dFmb1y*dFmb1y_dLh1) + (dFmb1rad_dFmb1z*dFmb1z_dLh1)
        dFmb1rad_dL12 = (dFmb1rad_dFmb1y*dFmb1y_dL12) + (dFmb1rad_dFmb1z*dFmb1z_dL12)
        # ---------------

        # ----- collect for outputs
        dFmb1_dLh1 = np.stack([dFmb1ax_dLh1,dFmb1y_dLh1,dFmb1z_dLh1,dFmb1rad_dLh1])
        dFmb1_dL12 = np.stack([dFmb1ax_dL12,dFmb1y_dL12,dFmb1z_dL12,dFmb1rad_dL12])

        dFmb2_dLh1 = np.stack([dFmb2ax_dLh1,dFmb2y_dLh1,dFmb2z_dLh1,dFmb2rad_dLh1])
        dFmb2_dL12 = np.stack([dFmb2ax_dL12,dFmb2y_dL12,dFmb2z_dL12,dFmb2rad_dL12])

        return F_mb1, F_mb2, dFmb1_dLh1, dFmb1_dL12, dFmb2_dLh1, dFmb2_dL12
    # ==============================

# ---------------
def analytical_MBforces_realistic( Fx,Fy,Fz, Mx,My,Mz, m_carrier,delta,tilt,
        L_h1,L_12, flag_jac=False ):
    """
    Analytical low-fidelity main bearing force calculation,
    using static equilibrium about MB2 (fixed; TRB2)
    Using 4-node formulation, includes carrier mass at shaft-end (GB-input)

    Inputs
    -------
    F* : float array[ # of time steps , # of wind speeds ], [N]
        Forces (aero) on the hub center/main shaft input; for FLS, shape=(72e4,10)
    M* : float array[ # of time steps , # of wind speeds ], [Nm]
        Moment (aero) on the hub center/main shaft input; for FLS, shape=(72e4,10)
    m_carrier : float, [kg]
        mass of the planet carrier (gearbox's first stage): calculated in gearbox.py
    delta : float, [m]
        separation between MB2 and gearbox attachment (constant 0.5 in layout.py)
    tilt : float, [rad]
        drivetrain tilt angle
    L_h1 : float
        length along main shaft btw hub and mb1
    L_12 : float
        length along main shaft btw mb1 and mb2
    flag_jac : Boolean
        whether to provide analytical derivatives (True) or not (False)

    Outputs
    -------
    F_mb1 : All forces on the 1. main bearing
        shape = (4, shape(F*) )
    F_mb2 : All forces on the 2. main bearing
        shape = (4, shape(F*) )
    
    --- if flag_jac = True ---
    dFmb1_dLh1 : All derivatives of mb1 forces wrt. L_h1
        shape = (4, shape(F_mb1) )
    dFmb1_dLh1 : All derivatives of mb1 forces wrt. L_12
        shape = (4, shape(F_mb1) )
    dFmb2_dLh1 : All derivatives of mb2 forces wrt. L_h1
        shape = (4, shape(F_mb2) )
    dFmb2_dL12 : All derivatives of mb2 forces wrt. L_12
        shape = (4, shape(F_mb2) )

    Internal Progress
    --------------
    - DONE : implement as a function, general purpose
    - TODO?: implement as a openMDAO Explicit Component
    - TODO : add analytical gradients
    """
    # init
    g = 9.81 # m^2/s
    x3 = L_h1 + L_12
    n_ts, n_ws = Fx.shape[0], Fx.shape[1] # = 72e4, 10
    ### === Loads on bearings ===
    F_mb1 = np.zeros( (n_ts,n_ws,4) ) # for 4 forces (ax,y,z,rad)
    F_mb2 = np.zeros( (n_ts,n_ws,4) ) # for 4 forces (ax,y,z,rad)
        
    # --- MB1 (CRB) ---
    F_mb1_ax = np.zeros_like(Fx)
    F_mb1_y = ( Mz - (Fy*x3) ) / L_12
    F_mb1_z = ( - My - (Fz*x3) + (m_carrier*g*np.cos(tilt)*delta) ) / L_12
    F_mb1_rad = np.hypot(F_mb1_y, F_mb1_z) # element-wise

    # --- MB2 (DRTRB) ---
    F_mb2_ax = np.abs( -Fx - (m_carrier*g*np.sin(tilt)) ) # `abs` coz mb2 reacts to the axial load, regardless if tensile or compressive.
    F_mb2_y = -Fy - F_mb1_y
    F_mb2_z = -Fz - F_mb1_z + (m_carrier*g*np.cos(tilt))
    F_mb2_rad = np.hypot(F_mb2_y, F_mb2_z) # element-wise

    # ----- collect for outputs
    F_mb1 = np.stack([F_mb1_ax, F_mb1_y, F_mb1_z, F_mb1_rad])
    F_mb2 = np.stack([F_mb2_ax, F_mb2_y, F_mb2_z, F_mb2_rad])

    if not flag_jac:
        return F_mb1, F_mb2
    # ==============================

    else:
    ### === Derivatives of loads wrt. L_* ===
    # init
        dFmb1_dLh1 = F_mb1                # ----- main jac outputs -----
        dFmb1_dL12 = F_mb1
        dFmb2_dLh1 = F_mb2
        dFmb2_dL12 = F_mb2

        # ---- MB1 derivatives ----       # TODO
        # dFmb1y_dLh1 = -Fy / L12
        # dFmb1y_dL12 = -(Fy * L12 - Fy * x3) / L12**2

        # dFmb1z_dLh1 = -Fz / L12
        # dFmb1z_dL12 = -(Fz * L12 - Fz * x3) / L12**2

        # # ---- MB2 depends on MB1 ----
        # dFmb2y_dLh1 = -dFmb1y_dLh1
        # dFmb2y_dL12 = -dFmb1y_dL12

        # dFmb2z_dLh1 = -dFmb1z_dLh1
        # dFmb2z_dL12 = -dFmb1z_dL12

        # ----- collect for outputs
        # dFmb1_dLh1 = np.stack([dFmb1ax_dLh1,dFmb1y_dLh1,dFmb1z_dLh1,dFmb1rad_dLh1])
        # dFmb1_dL12 = np.stack([dFmb1ax_dL12,dFmb1y_dL12,dFmb1z_dL12,dFmb1rad_dL12])

        # dFmb2_dLh1 = np.stack([dFmb2ax_dLh1,dFmb2y_dLh1,dFmb2z_dLh1,dFmb2rad_dLh1])
        # dFmb2_dL12 = np.stack([dFmb2ax_dL12,dFmb2y_dL12,dFmb2z_dL12,dFmb2rad_dL12])

        return F_mb1, F_mb2, dFmb1_dLh1, dFmb1_dL12, dFmb2_dLh1, dFmb2_dL12
    # ==============================

# ---------------
def del_bearing_computation(load_series, ws_bins, t_step, omega,
                    probabilities, p=10/3, dP_dL=[]):
    """
    Compute Damage Equivalent Load (DEL) using full time series method.

    Inputs
    -------
    load_series : array[ # of time steps , # of wind speeds ]
        Time series of loads (forces or moments); for FLS, shape=(72e4,10)
    ws : array[ 1, # of wind speeds ]
        Wind speed bins corresponding to load_series columns, shape=(1,10)
    t_step : float, [s]
        Time step used for integration
    omega : array[ # of time steps , # of wind speeds ]
        Rotational speed time series corresponding to load_series; shape=(72e4,10)
    p : float
        Exponent for equivalent load calculation (default 10/3 for bearings)
    probabilites : array, same size as ws
        or pdf of occurence of each ws
    dP_dL : array, same size as load_series ('P')
        gradient of P wrt. the length segments of the main shaft (L_*)
        if provided, the gradients are computed; otherwise if empty, no gradients
    
    Outputs
    -------
    DEL : float
        Damage Equivalent Load weighted-averaged over all wind speeds
        provided as the only output if no gradients are computed (len(dP_dL)==0)
    dDEL_dL : float
        Gradients of DEL wrt. L (length of main shaft segments)
        returned as the second output if dP_dL is input to the function (size~=0)

    Internal Progress
    --------------
    - DONE : implement as a function, general purpose
    - DONE : add analytical gradients; vectorize more?
    - TODO : remove ws_bins from inputs (not used, probab input directly instead)
    """
    # init
    P = load_series

    # take out dimenstions
    n_t, n_w = omega.shape[0], omega.shape[1] # 72e3, 11
    ws = ws_bins.reshape(1,n_w)
    pdf_ws = probabilities.reshape(1,n_w)

    # start computing DEL
    N = (omega/60 * t_step) # element wise multiplication, broadcasting (72e3,11)
    n = np.sum(N, axis=0).reshape(1, n_w) # (1,11)

    numerator = np.sum((P**p) * N, axis=0).reshape(1,n_w) # (1,11)

    DEL_j = (numerator/n)**(1/p) # (1,11)

    DEL = np.sum( DEL_j**p * pdf_ws )**(1/p) # float
    # provide DEL without gradients
    if len(dP_dL)==0:
        return DEL
    
    # provide DEL with gradients (DEL, gradients)
    else:
        dDELj_dL = (1/p)*(DEL_j**(1-p))*(1/n)*np.sum(
            p*(P**(p-1))*dP_dL*N,
            axis=0
        )
        dDEL_dL = (1/p)*(DEL**(1-p)) * np.sum(
            p*(DEL_j**(p-1))*dDELj_dL*pdf_ws
        )
        return DEL, dDEL_dL

# def compute_partials(self, inputs, partials):
#     """
#     Analytic gradient: d(L10) / d(DEL)
    
#     Given:
#         DEL_w = (Σ DEL[i]^m × p[i])^(1/m)     ... weighted DEL
#         L10 = (Cr / DEL_w)^m                   ... ISO 281
    
#     Want: ∂L10/∂DEL[j]
    
#     Chain rule:
#         ∂L10/∂DEL[j] = (∂L10/∂DEL_w) × (∂DEL_w/∂DEL[j])
    
#     Step 1: ∂L10/∂DEL_w
#         L10 = Cr^m × DEL_w^(-m)
#         ∂L10/∂DEL_w = -m × Cr^m × DEL_w^(-m-1)
    
#     Step 2: ∂DEL_w/∂DEL[j]
#         DEL_w = (Σ DEL[i]^m × p[i])^(1/m)
#         Let S = Σ DEL[i]^m × p[i], so DEL_w = S^(1/m)
#         ∂DEL_w/∂DEL[j] = (1/m) × S^(1/m - 1) × m × DEL[j]^(m-1) × p[j]
#                        = S^(1/m - 1) × DEL[j]^(m-1) × p[j]
#                        = DEL_w^(1-m) × DEL[j]^(m-1) × p[j]
    
#     Combining:
#         ∂L10/∂DEL[j] = -m × Cr^m × DEL_w^(-m-1) × DEL_w^(1-m) × DEL[j]^(m-1) × p[j]
#                      = -m × Cr^m × DEL_w^(-2m) × DEL[j]^(m-1) × p[j]
#     """
#     m = 10.0 / 3.0
#     DEL = inputs['DEL_mb1']
#     ws_pdf = inputs['ws_pdf']
#     Cr = inputs['Cr_mb1']
    
#     # Intermediate values
#     S = np.sum((DEL ** m) * ws_pdf)
#     DEL_w = S ** (1.0 / m)
    
#     # Gradient w.r.t. each DEL[j]
#     dL10_dDEL = -m * (Cr ** m) * (DEL_w ** (-2*m)) * (DEL ** (m-1)) * ws_pdf
    
#     partials['L10_mb1', 'DEL_mb1'] = dL10_dDEL
    
#     # Gradient w.r.t. Cr
#     # L10 = (Cr/DEL_w)^m = Cr^m × DEL_w^(-m)
#     # ∂L10/∂Cr = m × Cr^(m-1) × DEL_w^(-m)
#     partials['L10_mb1', 'Cr_mb1'] = m * (Cr ** (m-1)) * (DEL_w ** (-m))
# ---------------

#%%
# ---------------
class Analytical_FLS_Bearing_Life( om.ExplicitComponent ):
    """
    Component to compute bearing fatigue life (FLS)
    using analytical bearing load calculations and either methods:
    1. bin counting method          : note: bad within optimization
    2. damage equivalent load (DEL) : note: preferred

    Inputs
    -------
    L_12 : float
        Main bearing span / distance between main bearings
    L_h1 : float
        Rotor to main bearing 1 distance

    Outputs
    -------
    constr_L10_mb1 : float
        Safety factor MB1 L10 life compared to design life
    constr_L10_mb2 : float
        Safety factor MB2 L10 life compared to design life
    
    Internal Progress
    --------------
    - DONE : implement as a openMDAO Explicit Component
    - DONE : make DLC load series (yaml; not local stored) compatible with WEIS iA
    - DONE : modify for more_realisitc analy_MBforces: add relevant inputs
    - TODO : use log-space constraints?
    """
    
    def initialize(self):
        self.options.declare("modeling_options") # opt_drivese = self.options["modeling_options"]["WISDEM"]["DriveSE"]
        self.options.declare("openfast_options") # opt_openfast = self.options["modeling_options"]["OpenFAST"]
        self.options.declare("dlc_options")      # opt_DLC = self.options["modeling_options"]["DLC_driver"]["DLCs"][0]
        
    def setup(self):
        # Extract options + sanity check.
        # ---- DLC loads & operational ----
        dir_loads = self.options['openfast_options']['openfast_dir'] # directory of MS loads
        loads_dict, _ = load_all_mat_to_dict(dir_loads)
        # here coz runs only once per model build
        # - a. loads
        self.Fx = loads_dict['Fx']
        # print(f"Fx[:2,:2]: {Fx[:2,:2]}, Fx shape: {Fx.shape}") # debugging: check mags wrt. units
        self.Fy = loads_dict['Fy']
        self.Fz = loads_dict['Fz']
        self.Mx = loads_dict['Mx']
        self.My = loads_dict['My']
        self.Mz = loads_dict['Mz']
        self.omega = loads_dict['rot_speed']
        # - b. openfast (Wind statistics)
        ws = self.options['dlc_options']['wind_speed'] # shape=(1,10)
        n_ws = len(ws)
        self.ws = np.reshape( ws, (1,n_ws))
        self.probabilities = np.reshape( self.options['dlc_options']['probabilities'], (1,n_ws))
        # - c. DLC
        self.dt = self.options['openfast_options']['simulation']['DT'] # 0.05 (20 Hz)
        # ----

        # - 1. LSS parameters (from Layout, Hub_Rotor_LSS_Frame)
        self.add_input('L_12', val=0.0, desc='Main bearing span', units='m')
        self.add_input('L_h1', val=0.0, desc='Rotor bearing distance', units='m')
        # - 2. bearing parameters (from MainBearing)
        self.add_input('Cr_mb1', val=1e7, units='N', desc='Dynamic load rating MB1')
        self.add_input('Cr_mb2', val=1e7, units='N', desc='Dynamic load rating MB2')
        self.add_input('p_mb', val=3.33, desc='Bearing life exponent')
        self.add_input('e_mb', val=3.5, desc='Bearing limiting factor, load ratio')
        self.add_input('X1_mb', val=1.0, desc='Bearing light coefficient for P calculation')
        self.add_input('Y1_mb', val=1.0, desc='Bearing light coefficient for P calculation')
        self.add_input('X2_mb', val=1.0, desc='Bearing heavy coefficient for P calculation')
        self.add_input('Y2_mb', val=1.0, desc='Bearing heavy coefficient for P calculation')
        # - 3. operational
        self.add_input('rated_rpm', val=0.0, desc='Nominal/rated rotational speed', units='rpm')
        self.add_input('lifetime', val=25.0, desc='Wind turbine design life')
        # - 4. drivetrain
        self.add_input("carrier_mass", 0.0, units="kg")
        self.add_input("tilt", 0.0, units="deg")
        self.add_input("s_lss", val=np.zeros(5), units="m")
        # Outputs
        self.add_output('L10h_mb1', val=0.0, desc='L10 life MB1', units='h')
        self.add_output('L10h_mb2', val=0.0, desc='L10 life MB2', units='h')
        self.add_output('constr_L10_mb1', val=0.0, desc='Safety factor MB1')
        self.add_output('constr_L10_mb2', val=0.0, desc='Safety factor MB2')
        # self.add_output('constr_L10_mb_all', val=0.0, desc='Minimum safety factor')
        
    def compute(self, inputs, outputs):
        # ---- Inputs ----
        # ISO 281 parameters (from MainBearing)
        e, p = inputs['e_mb'], inputs['p_mb']
        X1, Y1 = inputs['X1_mb'], inputs['Y1_mb']
        X2, Y2 = inputs['X2_mb'], inputs['Y2_mb']
        # LSS
        L_12 = inputs['L_12']
        L_h1 = inputs['L_h1']
        n0 = inputs['rated_rpm'] + 1e-6 # div by 0.0 (def), avoid by 1e-6
        # drivetrain
        tilt_rad = float(np.deg2rad(inputs["tilt"][0]))
        m_carrier = float(inputs["carrier_mass"][0])
        s_lss = inputs["s_lss"]
        delta = float(s_lss[1]-s_lss[0])
        # --------
        # Bearing loads (analytical) calculation: shape=(4, 72000, 11)
        # Fmb1, Fmb2, self.dFmb1_dLh1, self.dFmb1_dL12, self.dFmb2_dLh1, self.dFmb2_dL12 = analytical_MB_Forces(
        #     Fx,Fy,Fz,Mx,My,Mz, L_h1,L_12, flag_jac=True )
        # --- more realistic
        Fmb1, Fmb2, self.dFmb1_dLh1, self.dFmb1_dL12, self.dFmb2_dLh1, self.dFmb2_dL12 = analytical_MBforces_realistic(
            self.Fx,self.Fy,self.Fz,self.Mx,self.My,self.Mz,
            m_carrier,delta,tilt_rad,
            L_h1,L_12,flag_jac=True)
        # ----- extract axial and radial forces
        F_mb1_rad = Fmb1[3, :, :]                           # shape (720000,10)
        F_mb2_ax, F_mb2_rad = Fmb2[0, :, :], Fmb2[3, :, :]  # shape (720000,10)
        # ----- equivalent loads MB2
        ratio = F_mb2_ax / F_mb2_rad # DONE: removed max, coz F_mb2_rad is pos always (sqrt)
        light = np.abs(ratio) <= e; self.light = light
        P_mb2 = np.zeros_like(F_mb2_rad)
        P_mb2[light] = X1 * F_mb2_rad[light] + Y1 * F_mb2_ax[light]
        P_mb2[~light] = X2 * F_mb2_rad[~light] + Y2 * F_mb2_ax[~light]
        self.P_mb2 = P_mb2
        # --- trying smooth approximation for optim ---
        # P_mb2 = X2 * F_mb2_rad + Y2 * F_mb2_ax # step 2
        # P_mb2 = F_mb2_rad # step 1
        # ----- equivalent loads MB1
        P_mb1 = F_mb1_rad # (= radial loads coz radial bearing CRB)
        self.P_mb1 = P_mb1

        # LRD bin-counting
        # P_mb1_sum = compute_LRD_matrix_vectorized(P_mb1,
        #         self.dt,self.omega,self.probabilities,p)
        # P_mb2_sum = compute_LRD_matrix_vectorized(P_mb2,
        #         self.dt,self.omega,self.probabilities,p)

        # DEL calculation
        # print('ws: ', ws) # debugging
        P_mb1_sum = del_bearing_computation(P_mb1,
            self.ws, self.dt, self.omega, self.probabilities, p)
        P_mb2_sum = del_bearing_computation(P_mb2,
            self.ws, self.dt, self.omega, self.probabilities, p)

        # L10 life calculation
        Cr1 = inputs['Cr_mb1']
        Cr2 = inputs['Cr_mb2']
        # L10 = (Cr/P)^p
        L10_mb1 = (Cr1 / P_mb1_sum) ** (p) # note: noth in [N] !!!
        L10_mb2 = (Cr2 / P_mb2_sum) ** (p)
        # print(f"L10_mb1: {L10_mb1}, L10_mb2: {L10_mb2}") # debugging
        
        # ---- mb1 ----
        L10h_mb1 = L10_mb1 * (1e6 / n0 / 60)
        self.L10h_mb1 = L10h_mb1
        outputs['L10h_mb1'] = L10h_mb1
        # ---- mb1 ----
        L10h_mb2 = L10_mb2 * (1e6 / n0 / 60)
        self.L10h_mb2 = L10h_mb2
        outputs['L10h_mb2'] = L10h_mb2
        # print(f"L10h_mb1: {outputs['L10h_mb1']}, L10h_mb2: {outputs['L10h_mb2']}") # debugging
        
        # constraints: on FLS safety factors (20 years = 20*8766 hours)
        L_design = inputs['lifetime']
        # ---- mb1 ----
        outputs['constr_L10_mb1'] = (L10h_mb1/(L_design*8766))**(1/p) # inside log should be >= 1, with log should be >= 0
        self.constr_L10_mb1 = outputs['constr_L10_mb1']
        # ---- mb2 ----
        outputs['constr_L10_mb2'] = (L10h_mb2/(L_design*8766))**(1/p)
        self.constr_L10_mb2 = outputs['constr_L10_mb2']
        # ---- all ----
        # outputs['constr_L10_mb_all'] = min(outputs['constr_L10_mb1'], outputs['constr_L10_mb2'])
# ---------------

# ---------------
class Analytical_FLS_Bearing_Life_Derivatives( Analytical_FLS_Bearing_Life ):
    """
    Version with analytical derivatives
    - a sub-class inheriting from parent `Analytical_FLS_Bearing_Life` 

    Internal Progress
    --------------
    - DONE : new subclass: analytical gradients (declare_partials, compute_partials)
    - TODO : check and verify: rn error: calc O(-8), fd O(-2)
    -- check by a new Compn for each step (forces, P, DEL, constr) and check_partials()?
    """

    def setup_partials(self):
        # call to declare_partials tells openMDAO which to expect, hence these being non-zero
        self.declare_partials(of='constr_L10_mb1', wrt=['L_h1', 'L_12'])
        self.declare_partials(of='constr_L10_mb2', wrt=['L_h1', 'L_12'])

    def compute_partials(self, inputs, J, discrete_inputs=None):
        # Extract options + sanity check
        # - 1. openfast (Wind statistics)
        ws = self.options['dlc_options']['wind_speed'] # shape=(1,10)
        n_ws = len(ws)
        ws = np.reshape( ws, (1,n_ws))
        probabilities = np.reshape( self.options['dlc_options']['probabilities'], (1,n_ws))
        # - 2. DLC
        dt = self.options['openfast_options']['simulation']['DT'] # 0.05 (20 Hz)
        omega = self.loads_dict['rot_speed']

        # Inputs unpacking
        # - ISO 281 parameters (from MainBearing)
        p = inputs['p_mb']
        X1, Y1 = inputs['X1_mb'], inputs['Y1_mb']
        X2, Y2 = inputs['X2_mb'], inputs['Y2_mb']
        # - DVs
        L_12 = inputs['L_12']
        L_h1 = inputs['L_h1']
        
        # `self` unpacking
        light = self.light
        # ---- mb1 wrt. L_*
        dFmb1_dLh1 = self.dFmb1_dLh1; dFmb1rad_dLh1 = dFmb1_dLh1[3,:,:]
        dFmb1_dL12 = self.dFmb1_dL12; dFmb1rad_dL12 = dFmb1_dL12[3,:,:]
        # ---- mb2 wrt. L_*
        dFmb2_dLh1 = self.dFmb2_dLh1
        dFmb2ax_dLh1, dFmb2rad_dLh1 = dFmb2_dLh1[0,:,:], dFmb2_dLh1[3,:,:]
        dFmb2_dL12 = self.dFmb2_dL12
        dFmb2ax_dL12, dFmb2rad_dL12 = dFmb2_dL12[0,:,:], dFmb2_dL12[3,:,:]

        # compute partials (P_* wrt. L_*)
        # ---- mb1 ---- (note: direct)
        # formula: P_mb1 = F_mb1_rad
        dPmb1_dLh1 = dFmb1rad_dLh1
        dPmb1_dL12 = dFmb1rad_dL12
        # ---- mb2 ---- (note: based on light)
        # formula: P_mb2[light] = X1 * F_mb2_rad[light] + Y1 * F_mb2_ax[light]
        # formula: P_mb2[~light] = X2 * F_mb2_rad[~light] + Y2 * F_mb2_ax[~light]
        # ---------- init with zeros (like P_mb2 in compute)
        dPmb2_dLh1 = np.zeros_like(dFmb2rad_dLh1)
        dPmb2_dL12 = np.zeros_like(dFmb2rad_dLh1)
        # ---------- wrt. L_h1
        dPmb2_dLh1[light] = (X1 * dFmb2rad_dLh1[light]) + (Y1 * dFmb2ax_dLh1[light])
        dPmb2_dLh1[~light] = (X2 * dFmb2rad_dLh1[~light]) + (Y2 * dFmb2ax_dLh1[~light])
        # ---------- wrt. L_12
        dPmb2_dL12[light] = (X1 * dFmb2rad_dL12[light]) + (Y1 * dFmb2ax_dL12[light])
        dPmb2_dL12[~light] = (X2 * dFmb2rad_dL12[~light]) + (Y2 * dFmb2ax_dL12[~light])

        # compute partials (DEL)
        # TODO: use 'p' exponent for each mb (when input to compn)
        # ---- mb1 ----
        P_mb1 = self.P_mb1
        DEL_1, dDEL1_dLh1 = del_bearing_computation(
            P_mb1,ws,dt,omega,probabilities,p,dPmb1_dLh1)
        _, dDEL1_dL12 = del_bearing_computation(
            P_mb1,ws,dt,omega,probabilities,p,dPmb1_dL12)
        # ---- mb2 ----
        P_mb2 = self.P_mb2
        DEL_2, dDEL2_dLh1 = del_bearing_computation(
            P_mb2,ws,dt,omega,probabilities,p,dPmb2_dLh1)
        _, dDEL2_dL12 = del_bearing_computation(
            P_mb2,ws,dt,omega,probabilities,p,dPmb2_dL12)
        
        # compute partials (L10)
        # ---- mb1 ----
        Cr1 = inputs['Cr_mb1']
        dL10_1_dLh1 = (Cr1**p)*(-p)*(DEL_1**(-p-1))*dDEL1_dLh1
        dL10_1_dL12 = (Cr1**p)*(-p)*(DEL_1**(-p-1))*dDEL1_dL12
        # ---- mb2 ----
        Cr2 = inputs['Cr_mb2']
        dL10_2_dLh1 = (Cr2**p)*(-p)*(DEL_2**(-p-1))*dDEL2_dLh1
        dL10_2_dL12 = (Cr2**p)*(-p)*(DEL_2**(-p-1))*dDEL2_dL12

        # compute partials (L10h)
        n0 = inputs['rated_rpm']
        n060 = (1e6 / n0 / 60)
        # ---- mb1 ----
        dL10h_1_dLh1 = n060 * dL10_1_dLh1
        dL10h_1_dL12 = n060 * dL10_1_dL12
        # ---- mb2 ----
        dL10h_2_dLh1 = n060 * dL10_2_dLh1
        dL10h_2_dL12 = n060 * dL10_2_dL12

        # compute partials (constr_)
        # ---- mb1 ----
        L10h_mb1 = self.L10h_mb1
        constr_L10_mb1 = self.constr_L10_mb1
        J['constr_L10_mb1','L_h1'] = (1/p)*constr_L10_mb1*(L10h_mb1**(1-p))*dL10h_1_dLh1
        J['constr_L10_mb1','L_12'] = (1/p)*constr_L10_mb1*(L10h_mb1**(1-p))*dL10h_1_dL12

        # ---- mb2 ----
        L10h_mb2 = self.L10h_mb2
        constr_L10_mb2 = self.constr_L10_mb2
        J['constr_L10_mb2','L_h1'] = (1/p)*constr_L10_mb2*(L10h_mb2**(1-p))*dL10h_2_dLh1
        J['constr_L10_mb2','L_12'] = (1/p)*constr_L10_mb2*(L10h_mb2**(1-p))*dL10h_2_dL12
# ---------------