import copy

import numpy as np
import pandas as pd
import openmdao.api as om

import wisdem.commonse.utilities as util

# -------------------------------------------------------------------------


class MainBearing(om.ExplicitComponent):
    """
    MainBearings class is used to represent the main bearing components of a wind turbine drivetrain.
    This is a ´simple, regression-based sizing tool for the main bearings´!  The same function is called once
    for configurations with one main bearing or twice for configurations with two.  It handles Compact Aligning
    Roller Bearings (CARB), Cylindrical Roller Bearings (CRB), Spherical Roller Bearings (SRB), and
    Tapered Roller Bearings (TRB).

    Parameters
    ----------
    bearing_type : string
        bearing mass type
    D_bearing : float, [m]
        bearing diameter/facewidth
    D_shaft : float, [m]
        Diameter of LSS shaft at bearing location
    mb_mass_user : float, [kg]
        user override of component mass

    ----- extended by Vasudev Gupta (v) -----

    Returns
    -------
    mb_max_defl_ang : float, [rad]
        Maximum allowable deflection angle
    mb_mass : float, [kg]
        overall component mass
    mb_I : numpy array[3], [kg*m**2]
        moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass
    
    ----- extended by Vasudev Gupta (v) -----
    mb_Cr : float, [N]
        Dynamic load rating, for bearing FLS calculations
            based on 2015_Guo-Analy... paper (SKF 2014 catalogue)
    mb_cm : float, [m] TODO
        Bearing center of mass (from?)
    mb_Reactions : numpy array[6] of int
        Bearing reaction constraints: [Rx,Ry,Rz,Rxx,Ryy,Rzz]

    Internal Progress
    --------------
    - NOTE : some Cr here emperical on VERY small bearings (<=2 m), may need update for large bearings: eg. CRB, TRB1, SRB, CARB
    - DONE : connect D_shaft(s) to lss_diameter[]
    - DONE : add reactions for each bearing type (Fa,Fr,M), which inputs to Hub_*
    - TODO : analytical gradients wrt. D_shaft (DV); use JAX (easy iA)?
    - TODO : change 'TRB2' name to 'DRTRB' (as per IEC-4)
    - TODO : moment-reacting bearings = (TRB2, TRB, SRB) ?
    """

    def setup(self):
        self.add_discrete_input("bearing_type", "CARB")
        self.add_input("D_bearing", 0.0, units="m")
        self.add_input("D_shaft", 0.0, units="m")
        self.add_input("mb_mass_user", 0.0, units="kg")
        self.add_input("mb_e", 0.35, desc="limiting value of Fa/Fr for the applicability of different values of factors X and Y (ISO 281)") #(v)

        self.add_output("mb_max_defl_ang", 0.0, units="rad")
        self.add_output("mb_mass", 0.0, units="kg")
        self.add_output("mb_I", np.zeros(3), units="kg*m**2")
        #(v)
        self.add_output("face_width", val=0.0, units="m", desc="face width of the bearing")
        self.add_output("mb_Cr", val=0.0, units='N', desc='Dynamic load rating of the bearing')
        self.add_output('mb_p', val=10/3, desc='Bearing life exponent')
        self.add_output('mb_X1', val=0.0, desc='Bearing light coefficient for P calculation')
        self.add_output('mb_Y1', val=0.0, desc='Bearing light coefficient for P calculation')
        self.add_output('mb_X2', val=0.0, desc='Bearing heavy coefficient for P calculation')
        self.add_output('mb_Y2', val=0.0, desc='Bearing heavy coefficient for P calculation')
        self.add_output('mb_Reactions', val=np.zeros(6,dtype=int), desc='Bearing reaction constraints: [Rx,Ry,Rz,Rxx,Ryy,Rzz]')
        self.add_output('mb_k', val=3e10, units="N*m/rad", desc='Torsional stiffness of the moment-reacting bearing (eg. TRB2)')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        
        if type(discrete_inputs["bearing_type"]) != type(""):
            raise ValueError("Bearing type input must be a string")
        
        btype = discrete_inputs["bearing_type"].upper()
        D_shaft = inputs["D_shaft"]
        mass_user = float(inputs["mb_mass_user"][0])
        
        # ----- (v) below -----
        e = inputs["mb_e"] #(v) ISO 281 parameters, Tab.8
        alpha = np.arctan(e / 1.5)
        # - light
        outputs['mb_X1'] = 1.0
        outputs['mb_Y1'] = 0.45 / np.tan(alpha)
        # - heavy
        outputs['mb_X2'] = 0.67
        outputs['mb_Y2'] = outputs['mb_X2'] / np.tan(alpha)
        outputs['mb_p'] = 10/3 #(v) p same for all defined below (all roller bearings) 
        
        # reactions the bearings take
        FREE, RIGID = 0, 1
        mb_Reactions = np.array([FREE]*6) # ([ Rx, Ry, Rz, Rxx, Ryy, Rzz ])
        # note: mb_Reactions[4] = FREE always !

        # torsional stiffness (k_yy = k_zz = k)
        mb_k = 0.0 # 0 for all, except TRB1, TRB2
        # ----- (v) above -----

        # assume low load rating for bearing
        if btype == "CARB":  # p = Fr, so X=1, Y=0
            face_width = 0.2663 * D_shaft + 0.0435
            mass = 1561.4 * D_shaft**2.6007
            max_ang = np.deg2rad(0.5)
            Cr_rating = (16676 * D_shaft**1.4746) * 1e3 #(v) kN !!!
            mb_Reactions[:] = [FREE, RIGID, RIGID, FREE, FREE, FREE] #(v) axial free, radial rigid

        # elif btype == "CRB": #(v) original commented here
        #     face_width = 0.1136 * D_shaft
        #     mass = 304.19 * D_shaft**1.8885
        #     max_ang = np.deg2rad(4.0 / 60.0)

        elif btype == "CRB":  #(v) high load, revised to match 2015_Guo-Analytical, fig.22
            face_width = 0.157 * D_shaft + 0.0849
            mass = 1070.8 * D_shaft**1.8278
            max_ang = np.deg2rad(4.0 / 60.0) # 0.0667
            Cr_rating = (4526.5 * D_shaft ** 0.9556) * 1e3 #(v) kN -> N !!!
            mb_Reactions[:] = [FREE, RIGID, RIGID, FREE, FREE, FREE] #(v) axial free, radial rigid

        elif btype == "SRB":
            face_width = 0.2762 * D_shaft
            mass = 876.7 * D_shaft**1.7195
            max_ang = 0.078
            Cr_rating = (13878 * D_shaft**1.0796) * 1e3 #(v) kN -> N !!!
            mb_Reactions[:] = [RIGID, RIGID, RIGID, FREE, FREE, FREE] #(v) axial, radial rigid

        # elif btype == 'RB':  # factors depend on ratio Fa/C0, C0 depends on bearing... TODO: add this functionality
        #    face_width = 0.0839
        #    mass = 229.47 * D_shaft**1.8036
        #    max_ang = 0.002

        # elif btype == 'TRB':
        #    face_width = 0.0740
        #    mass = 92.863 * D_shaft**.8399
        #    max_ang = np.deg2rad(3.0 / 60.0)

        elif btype == "TRB":
            face_width = 0.1499 * D_shaft
            mass = 543.01 * D_shaft**1.9043
            max_ang = np.deg2rad(3.0 / 60.0) # 0.05
            Cr_rating = (1993.8 * D_shaft**0.318) * 1e3 #(v) kN -> N !!!
            mb_Reactions[:] = [RIGID, RIGID, RIGID, FREE, RIGID, RIGID] #(v) all rigid except torque TODO: check
            mb_k = 3e10 # TODO: check value

        elif btype == "TRB2":  #(v) 2-row TRB, high load; cf. 2015_Guo-Analytical, fig.26
            face_width = 0.1541 * D_shaft + 0.2087
            mass = 1442.6 * D_shaft**1.8932
            max_ang = np.deg2rad( (0.06+0.02)/2 )
            Cr_rating = (6579.9 * D_shaft**0.8592) * 1e3 #(v) kN -> N !!!
            mb_Reactions[:] = [RIGID, RIGID, RIGID, FREE, RIGID, RIGID] #(v) all rigid except torque
            mb_k = 3e10

        else:
            raise ValueError("Bearing type must be: CARB / CRB / SRB / TRB / TRB2")

        # add housing weight, but pg 23 of report says factor is 2.92 whereas this is 2.963 (cf. 2015_Guo-Analytical, eq.23 pdf)
        mass *= 1 + 80.0 / 27.0

        if mass_user > 0.0:
            mass = mass_user

        # Consider the bearings a torus for MoI (https://en.wikipedia.org/wiki/List_of_moments_of_inertia)
        D_bearing = inputs["D_bearing"] if inputs["D_bearing"] > 0.0 else face_width
        I0 = (1/4) * mass * (4 * (0.5 * D_shaft) ** 2 + 3 * (0.5 * D_bearing) ** 2)
        I1 = (1/8) * mass * (4 * (0.5 * D_shaft) ** 2 + 5 * (0.5 * D_bearing) ** 2)
        I = np.r_[I0, I1, I1]
        outputs["mb_mass"] = mass
        outputs["mb_I"] = I
        outputs["mb_max_defl_ang"] = max_ang
        # ----- (v) below -----
        outputs["mb_Cr"] = Cr_rating
        outputs["face_width"] = face_width
        if mb_Reactions[3] == RIGID: mb_Reactions[3] = FREE  # force allow bearing to not resist torque (captured by GB, in `Hub_*`)
        outputs["mb_Reactions"] = mb_Reactions
        outputs["mb_k"] = mb_k

# -------------------------------------------------------------------


class Brake(om.ExplicitComponent):
    """
    The brake attaches to the high speed shaft for geared configurations or directly on the
    low speed shaft for direct drive configurations.  It is regression based, but also allows
    for a user override of the total mass value.

    Compute brake mass in the form of :math:`mass = k*torque`.
    Value of :math:`k` was updated in 2020 to be 0.00122.

    Parameters
    ----------
    rotor_diameter : float, [m]
        rotor diameter
    rated_torque : float, [N*m]
        rotor torque at rated power
    brake_mass_user : float, [kg]
        User override of brake mass
    [X, v] D_shaft_end : float, [m]
        low speed shaft outer diameter
    s_rotor : float, [m]
        Generator rotor attachment to shaft s-coordinate
    s_gearbox : float, [m]
        Gearbox s-coordinate measured from bedplate
    [X, v] rho : float, [kg/m**3]
        material density

    Returns
    -------
    brake_mass : float, [kg]
        overall component mass
    brake_cm : float, [m]
        center of mass of the component in [x,y,z] for an arbitrary coordinate system
    brake_I : numpy array[3], [kg*m**2]
        moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass
    """

    def initialize(self):
        self.options.declare("direct_drive", default=True)

    def setup(self):
        self.add_input("rotor_diameter", 0.0, units="m")
        self.add_input("rated_torque", 0.0, units="N*m")
        self.add_input("brake_mass_user", 0.0, units="kg")
        self.add_input("s_rotor", 0.0, units="m")
        self.add_input("s_gearbox", 0.0, units="m")

        self.add_output("brake_mass", 0.0, units="kg")
        self.add_output("brake_cm", 0.0, units="m")
        self.add_output("brake_I", np.zeros(3), units="kg*m**2")

    def compute(self, inputs, outputs):
        # Unpack inputs
        D_rotor = float(inputs["rotor_diameter"][0])
        Q_rotor = float(inputs["rated_torque"][0])
        m_brake = float(inputs["brake_mass_user"][0])
        s_rotor = float(inputs["s_rotor"][0])
        s_gearbox = float(inputs["s_gearbox"][0])

        # Regression based sizing derived by J.Keller under FOA 1981 support project
        if m_brake == 0.0:
            coeff = 0.00122
            m_brake = coeff * Q_rotor

        # Assume brake disc diameter and simple MoI
        D_disc = 0.01 * D_rotor
        Ib = np.zeros(3)
        Ib[0] = 0.5 * m_brake * (0.5 * D_disc) ** 2
        Ib[1:] = 0.5 * Ib[0]

        cm = s_rotor if self.options["direct_drive"] else 0.5 * (s_rotor + s_gearbox)

        outputs["brake_mass"] = m_brake
        outputs["brake_I"] = Ib
        outputs["brake_cm"] = cm


# ----------------------------------------------------------------------------------------------


class RPM_Input(om.ExplicitComponent):
    """
    Generates vector of possible rpm values from min to max.
    The max value is assumed to be the rated rpm value.

    Parameters
    ----------
    minimum_rpm : float, [rpm]
        Minimum shaft rotations-per-minute (rpm), usually set by controller
    rated_rpm : float, [rpm]
        Rated shaft rotations-per-minute (rpm)
    gear_ratio : float
        overall gearbox ratio

    Returns
    -------
    shaft_rpm : float, [rpm]
        Vector of possible rpm values
    """

    def initialize(self):
        self.options.declare("n_pc", default=20)

    def setup(self):
        n_pc = self.options["n_pc"]
        self.add_input("minimum_rpm", val=1.0, units="rpm")
        self.add_input("rated_rpm", val=0.0, units="rpm")
        self.add_input("gear_ratio", val=1.0)

        self.add_output("lss_rpm", val=np.zeros(n_pc), units="rpm")
        self.add_output("hss_rpm", val=np.zeros(n_pc), units="rpm")

    def compute(self, inputs, outputs):
        min_rpm = np.maximum(0.1, float(inputs["minimum_rpm"][0]))
        max_rpm = float(inputs["rated_rpm"][0])
        ratio = float(inputs["gear_ratio"][0])
        rpm_full = np.linspace(min_rpm, max_rpm, self.options["n_pc"])
        outputs["lss_rpm"] = rpm_full
        outputs["hss_rpm"] = ratio * rpm_full


# ----------------------------------------------------------------------------------------------


class GeneratorSimple(om.ExplicitComponent):
    """
    The Generator class is used to represent the generator of a wind turbine drivetrain
    using simple scaling laws.  For a more detailed electromagnetic and structural design,
    please see the other generator components (Generator in drivetrainse.generator).

    Parameters
    ----------
    rotor_diameter : float, [m]
        rotor diameter
    machine_rating : float, [kW]
        machine rating of generator
    L_generator : float, [m]
        Generator stack width
    rated_torque : float, [N*m]
        rotor torque at rated power
    shaft_rpm : numpy array[n_pc], [rpm]
        Input shaft rotations-per-minute (rpm)
    generator_mass_user : float, [kg]
        User input override of generator mass
    generator_radius_user : float, [m]
        User input override of generator radius

    Returns
    -------
    R_generator : float, [m]
        center of mass of the component in [x,y,z] for an arbitrary coordinate system
    generator_mass : float, [kg]
        overall component mass
    generator_I : numpy array[3], [kg*m**2]
        moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass
    generator_efficiency : numpy array[n_pc]
        Generator efficiency at various rpm values
    lss_rpm : numpy array[n_pc], [rpm]
        Low speed shaft rpm values at which the generator efficiency is given
    """

    def initialize(self):
        self.options.declare("direct_drive", default=True)
        self.options.declare("n_pc", default=20)

    def setup(self):
        n_pc = self.options["n_pc"]

        # variables
        self.add_input("rotor_diameter", val=0.0, units="m")
        self.add_input("machine_rating", val=0.0, units="kW")
        self.add_input("rated_torque", 0.0, units="N*m")
        self.add_input("lss_rpm", np.zeros(n_pc), units="rpm")
        self.add_input("L_generator", 0.0, units="m")
        self.add_input("generator_mass_user", 0.0, units="kg")
        self.add_input("generator_radius_user", 0.0, units="m")
        self.add_input("generator_efficiency_user", val=np.zeros((n_pc, 2)))

        self.add_output("R_generator", val=0.0, units="m")
        self.add_output("generator_mass", val=0.0, units="kg")
        self.add_output("generator_rotor_mass", val=0.0, units="kg")
        self.add_output("generator_stator_mass", val=0.0, units="kg")
        self.add_output("generator_rotor_I", val=np.zeros(3), units="kg*m**2")
        self.add_output("generator_stator_I", val=np.zeros(3), units="kg*m**2")
        self.add_output("generator_I", val=np.zeros(3), units="kg*m**2")
        self.add_output("generator_efficiency", val=np.zeros(n_pc))

    def compute(self, inputs, outputs):
        # Unpack inputs
        rating = float(inputs["machine_rating"][0])
        D_rotor = float(inputs["rotor_diameter"][0])
        Q_rotor = float(inputs["rated_torque"][0])
        R_generator = float(inputs["generator_radius_user"][0])
        mass = float(inputs["generator_mass_user"][0])
        eff_user = inputs["generator_efficiency_user"]
        length = float(inputs["L_generator"][0])

        if mass == 0.0:
            if self.options["direct_drive"]:
                massCoeff = 1e-3 * 37.68
                mass = massCoeff * Q_rotor
            else:
                massCoeff = np.mean([6.4737, 10.51, 5.34])
                massExp = 0.9223
                mass = massCoeff * rating**massExp
        outputs["generator_mass"] = mass
        outputs["generator_rotor_mass"] = outputs["generator_stator_mass"] = 0.5 * mass

        # calculate mass properties
        if R_generator == 0.0:
            R_generator = 0.5 * 0.015 * D_rotor
        outputs["R_generator"] = R_generator

        I = np.zeros(3)
        I[0] = 0.5 * R_generator**2
        I[1:] = (1.0 / 12.0) * (3 * R_generator**2 + length**2)
        outputs["generator_I"] = mass * I
        outputs["generator_rotor_I"] = outputs["generator_stator_I"] = 0.5 * mass * I

        # Efficiency performance- borrowed and adapted from servose
        # NOTE: Have to use lss_rpm no matter what here because servose interpolation based on lss shaft rpm
        rpm_full = inputs["lss_rpm"]
        if np.any(eff_user):
            eff = np.interp(rpm_full, eff_user[:, 0], eff_user[:, 1])

        else:
            if self.options["direct_drive"]:
                constant = 0.01007
                linear = 0.02000
                quadratic = 0.06899
            else:
                constant = 0.01289
                linear = 0.08510
                quadratic = 0.0

            # Normalize by rated
            ratio = rpm_full / rpm_full[-1]
            eff = 1.0 - (constant / ratio + linear + quadratic * ratio)

        eff = np.maximum(1e-3, eff)
        outputs["generator_efficiency"] = eff


# -------------------------------------------------------------------------------


class Electronics(om.ExplicitComponent):
    """
    Estimate mass of electronics based on rating, rotor diameter, and tower top diameter.
    Empirical only, no load analysis.

    Parameters
    ----------
    machine_rating : float, [kW]
        machine rating of the turbine
    rotor_diameter : float, [m]
        rotor diameter of turbine
    D_top : float, [m]
        Tower top outer diameter
    converter_mass_user : float, [kg]
        Override regular regression-based calculation of converter mass with this value
    transformer_mass_user : float, [kg]
        Override regular regression-based calculation of transformer mass with this value

    Returns
    -------
    converter_mass : float, [kg]
        overall component mass
    converter_cm : numpy array[3], [m]
        center of mass of the component in [x,y,z] for an arbitrary coordinate system
    converter_I : numpy array[3], [kg*m**2]
        moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass
    transformer_mass : float, [kg]
        overall component mass
    transformer_cm : numpy array[3], [m]
        center of mass of the component in [x,y,z] for an arbitrary coordinate system
    transformer_I : numpy array[3], [kg*m**2]
        moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass

    """

    def setup(self):
        self.add_input("machine_rating", 0.0, units="kW")
        self.add_input("rotor_diameter", 0.0, units="m")
        self.add_input("D_top", 0.0, units="m")
        self.add_input("converter_mass_user", 0.0, units="kg")
        self.add_input("transformer_mass_user", 0.0, units="kg")

        self.add_output("converter_mass", 0.0, units="kg")
        self.add_output("converter_cm", np.zeros(3), units="m")
        self.add_output("converter_I", np.zeros(3), units="kg*m**2")
        self.add_output("transformer_mass", 0.0, units="kg")
        self.add_output("transformer_cm", np.zeros(3), units="m")
        self.add_output("transformer_I", np.zeros(3), units="kg*m**2")

    def compute(self, inputs, outputs):
        # Unpack inputs
        rating = float(inputs["machine_rating"][0])
        D_rotor = float(inputs["rotor_diameter"][0])
        D_top = float(inputs["D_top"][0])
        m_conv_usr = float(inputs["converter_mass_user"][0])
        m_trans_usr = float(inputs["transformer_mass_user"][0])

        # Correlation based trends, assume box
        m_converter = (
            m_conv_usr if m_conv_usr > 0.0 else 0.77875 * rating + 302.6
        )  # coeffs=1e-3*np.mean([740., 817.5]), np.mean([101.37, 503.83])
        m_transformer = m_trans_usr if m_trans_usr > 0.0 else 1.915 * rating + 1910.0

        # CM location, just assume off to the side of the bedplate
        cm = np.zeros(3)
        sides = 0.015 * D_rotor #(v) assumed square
        cm[1] = 0.5 * D_top + 0.5 * sides
        cm[2] = 0.5 * sides

        # Outputs
        outputs["converter_mass"] = m_converter
        outputs["converter_cm"] = cm
        outputs["converter_I"] = (1.0 / 6.0) * m_converter * sides**2 * np.ones(3)

        outputs["transformer_mass"] = m_transformer
        outputs["transformer_cm"] = cm
        outputs["transformer_cm"][1] *= -1.0
        outputs["transformer_I"] = (1.0 / 6.0) * m_transformer * sides**2 * np.ones(3)


# ---------------------------------------------------------------------------------------------------------------


class YawSystem(om.ExplicitComponent):
    """
    Estimate mass of yaw system based on rotor diameter and tower top diameter.
    Empirical only, no load analysis.

    Parameters
    ----------
    rotor_diameter : float, [m]
        rotor diameter
    D_top : float, [m]
        Tower top outer diameter
    rho : float, [kg/m**3]
        material density
    yaw_mass_user : float, [kg]
        user override mass value

    Returns
    -------
    yaw_mass : float, [kg]
        overall yaw system mass
    yaw_cm : numpy array[3], [m]
        center of mass of the component in [x,y,z] for an arbitrary coordinate system
    yaw_I : numpy array[3], [kg*m**2]
        moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass

    """

    def setup(self):
        # variables
        self.add_input("rotor_diameter", 0.0, units="m")
        self.add_input("D_top", 0.0, units="m")
        self.add_input("rho", 0.0, units="kg/m**3") #(v) connection to `bedplate_rho` in DrivetrainSE
        self.add_input("yaw_mass_user", 0.0, units="kg")

        self.add_output("yaw_mass", 0.0, units="kg")
        self.add_output("yaw_cm", np.zeros(3), units="m")
        self.add_output("yaw_I", np.zeros(3), units="kg*m**2")

    def compute(self, inputs, outputs):
        # Unpack inputs
        D_rotor = float(inputs["rotor_diameter"][0])
        D_top = float(inputs["D_top"][0])
        rho = float(inputs["rho"][0])
        m_yaw_usr = float(inputs["yaw_mass_user"][0])

        # Estimate the number of yaw motors (borrowed from old DriveSE utilities)
        n_motors = 2 * np.ceil(D_rotor / 30.0) - 2

        # Assume same yaw motors as Vestas V80 for now: Bonfiglioli 709T2M
        m_motor = 190.0

        # Assume friction plate surface width is 1/10 the diameter and thickness scales with rotor diameter
        m_frictionPlate = rho * np.pi * D_top * (0.1 * D_top) * (1e-3 * D_rotor)

        # Total mass estimate
        outputs["yaw_mass"] = m_yaw_usr if m_yaw_usr > 0.0 else m_frictionPlate + n_motors * m_motor

        # Assume cm is at tower top (cm=0,0,0) and mass is non-rotating (I=0,..), so leave at default value of 0s
        outputs["yaw_cm"] = np.zeros(3)
        outputs["yaw_I"] = np.zeros(3)


# ---------------------------------------------------------------------------------------------------------------


class MiscNacelleComponents(om.ExplicitComponent):
    """
    Estimate mass properties of miscellaneous other ancillary components in the nacelle.

    Parameters
    ----------
    upwind : boolean
        Flag whether the design is upwind or downwind
    machine_rating : float, [kW]
        machine rating of the turbine
    hvac_mass_coeff : float, [kg/kW]
        Regression-based scaling coefficient on machine rating to get HVAC system mass
    H_bedplate : float, [m]
        height of bedplate
    D_top : float, [m]
        Tower top outer diameter
    bedplate_mass : float, [kg]
        Bedplate mass
    bedplate_I : numpy array[6], [kg*m**2]
        Bedplate mass moment of inertia about base
    R_generator : float, [m]
        Generator outer diameter
    overhang : float, [m]
        Overhang of rotor from tower along x-axis in yaw-aligned c.s.
    cm_generator : float, [m]
        Generator center of mass s-coordinate
    rho_fiberglass : float, [kg/m**3]
        material density of fiberglass

    Returns
    -------
    hvac_mass : float, [kg]
        component mass
    hvac_cm : float, [m]
        component center of mass
    hvac_I : numpy array[3], [m]
        component mass moments of inertia
    platform_mass : float, [kg]
        component mass
    platform_cm : numpy array[3], [m]
        component center of mass
    platform_I : numpy array[3], [m]
        component mass moments of inertia
    cover_length : float, [m]
        length of cover and outer nacelle
    cover_height : float, [m]
        height of cover and outer nacelle
    cover_width : float, [m]
        width of cover and outer nacelle
    cover_mass : float, [kg]
        component mass
    cover_cm : numpy array[3], [m]
        component center of mass
    cover_I : numpy array[3], [m]
        component mass moments of inertia

    """

    def initialize(self):
        self.options.declare("direct_drive", default=True)

    def setup(self):
        self.add_discrete_input("upwind", True)
        self.add_input("machine_rating", 0.0, units="kW")
        self.add_input("hvac_mass_coeff", 0.025, units="kg/kW/m")
        self.add_input("H_bedplate", 0.0, units="m")
        self.add_input("D_top", 0.0, units="m")
        self.add_input("L_bedplate", 0.0, units="m")
        self.add_input("R_generator", 0.0, units="m")
        self.add_input("overhang", 0.0, units="m")
        self.add_input("generator_cm", 0.0, units="m")
        self.add_input("rho_fiberglass", 0.0, units="kg/m**3")
        self.add_input("rho_castiron", 0.0, units="kg/m**3")

        self.add_output("hvac_mass", 0.0, units="kg")
        self.add_output("hvac_cm", 0.0, units="m")
        self.add_output("hvac_I", np.zeros(3), units="m")
        self.add_output("platform_mass", 0.0, units="kg")
        self.add_output("platform_cm", np.zeros(3), units="m")
        self.add_output("platform_I", np.zeros(3), units="m")
        self.add_output("cover_length", 0.0, units="m")
        self.add_output("cover_height", 0.0, units="m")
        self.add_output("cover_width", 0.0, units="m")
        self.add_output("cover_mass", 0.0, units="kg")
        self.add_output("cover_cm", np.zeros(3), units="m")
        self.add_output("cover_I", np.zeros(3), units="m")
        self.add_output("misc_component_mass", 0.0, units="kg")

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        # Unpack inputs
        direct = self.options["direct_drive"]
        upwind = discrete_inputs["upwind"]
        rating = float(inputs["machine_rating"][0])
        coeff = float(inputs["hvac_mass_coeff"][0])
        H_bedplate = float(inputs["H_bedplate"][0])
        D_top = float(inputs["D_top"][0])
        L_bedplate = float(inputs["L_bedplate"][0])
        R_generator = float(inputs["R_generator"][0])
        overhang = float(inputs["overhang"][0])
        s_generator = float(inputs["generator_cm"][0])
        rho_fiberglass = float(inputs["rho_fiberglass"][0])
        rho_castiron = float(inputs["rho_castiron"][0])

        # For the nacelle cover, imagine a box from the bedplate to the hub in length and around the generator in width, height, with 10% margin in each dim
        L_cover = 1.1 * L_bedplate if direct else 1.1 * (overhang + D_top)
        W_cover = 1.1 * 2 * R_generator
        H_cover = 1.1 * (R_generator + np.maximum(R_generator, H_bedplate))
        A_cover = 2 * (L_cover * W_cover + L_cover * H_cover + H_cover * W_cover)
        t_cover = 0.02
        m_cover = A_cover * t_cover * rho_fiberglass
        cm_cover = np.array([0.5 * L_cover - 0.5 * L_bedplate, 0.0, 0.5 * H_cover])
        I_cover = (
            m_cover
            * np.array(
                [
                    H_cover**2 + W_cover**2 - (H_cover - t_cover) ** 2 - (W_cover - t_cover) ** 2,
                    H_cover**2 + L_cover**2 - (H_cover - t_cover) ** 2 - (L_cover - t_cover) ** 2,
                    W_cover**2 + L_cover**2 - (W_cover - t_cover) ** 2 - (L_cover - t_cover) ** 2,
                ]
            )
            / 12.0
        )
        if upwind:
            cm_cover[0] *= -1.0
        outputs["cover_length"] = L_cover
        outputs["cover_height"] = H_cover
        outputs["cover_width"] = W_cover
        outputs["cover_mass"] = m_cover
        outputs["cover_cm"] = cm_cover
        outputs["cover_I"] = I_cover

        # Regression based estimate on HVAC mass
        m_hvac = coeff * rating * 2 * np.pi * (0.75 * R_generator)
        cm_hvac = s_generator
        I_hvac = m_hvac * (0.75 * R_generator) ** 2
        outputs["hvac_mass"] = m_hvac
        outputs["hvac_cm"] = cm_hvac
        outputs["hvac_I"] = I_hvac * np.array([1.0, 0.5, 0.5])

        # Platforms as a fraction of bedplate mass and bundling it to call it 'platforms'
        L_platform = 2 * D_top if direct else L_cover
        W_platform = 2 * D_top if direct else W_cover
        t_platform = 0.04
        m_platform = L_platform * W_platform * t_platform * rho_castiron
        I_platform = (
            m_platform
            * np.array(
                [
                    t_platform**2 + W_platform**2,
                    t_platform**2 + L_platform**2,
                    W_platform**2 + L_platform**2,
                ]
            )
            / 12.0
        )
        outputs["platform_mass"] = m_platform
        outputs["platform_cm"] = np.zeros(3)
        outputs["platform_I"] = I_platform


# --------------------------------------------
class NacelleSystemAdder(om.ExplicitComponent):  # added to drive to include electronics
    """
    The Nacelle class is used to represent the overall nacelle of a wind turbine.

    Parameters
    ----------
    upwind : boolean
        Flag whether the design is upwind or downwind
    uptower : boolean
        Power electronics are placed in the nacelle at the tower top
    tilt : float, [deg]
        Shaft tilt
    mb1_mass : float, [kg]
        component mass
    mb1_cm : float, [m]
        component CM
    mb1_I : numpy array[3], [kg*m**2]
        component I
    mb2_mass : float, [kg]
        component mass
    mb2_cm : float, [m]
        component CM
    mb2_I : numpy array[3], [kg*m**2]
        component I
    gearbox_mass : float, [kg]
        component mass
    gearbox_cm : numpy array[3], [m]
        component CM
    gearbox_I : numpy array[3], [kg*m**2]
        component I
    hss_mass : float, [kg]
        component mass
    hss_cm : float, [m]
        component CM
    hss_I : numpy array[3], [kg*m**2]
        component I
    brake_mass : float, [kg]
        component mass
    brake_cm : float, [m]
        component CM
    brake_I : numpy array[3], [kg*m**2]
        component I
    generator_mass : float, [kg]
        component mass
    generator_cm : float, [m]
        component CM
    generator_I : numpy array[3], [kg*m**2]
        component I
    nose_mass : float, [kg]
        Nose mass
    nose_cm : float, [m]
        Nose center of mass along nose axis from bedplate
    nose_I : numpy array[3], [kg*m**2]
        Nose moment of inertia around cm in axial (hub-aligned) c.s.
    lss_mass : float, [kg]
        LSS mass
    lss_cm : float, [m]
        LSS center of mass along shaft axis from bedplate
    lss_I : numpy array[3], [kg*m**2]
        LSS moment of inertia around cm in axial (hub-aligned) c.s.
    converter_mass : float, [kg]
        overall component mass
    converter_cm : numpy array[3], [m]
        center of mass of the component in [x,y,z] for an arbitrary coordinate system
    converter_I : numpy array[3], [kg*m**2]
        moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass
    transformer_mass : float, [kg]
        overall component mass
    transformer_cm : numpy array[3], [m]
        center of mass of the component in [x,y,z] for an arbitrary coordinate system
    transformer_I : numpy array[3], [kg*m**2]
        moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass
    yaw_mass : float, [kg]
        overall component mass
    yaw_cm : numpy array[3], [m]
        center of mass of the component in [x,y,z] for an arbitrary coordinate system
    yaw_I : numpy array[3], [kg*m**2]
        moments of Inertia for the component [Ixx, Iyy, Izz] around its center of mass
    bedplate_mass : float, [kg]
        component mass
    bedplate_cm : numpy array[3], [m]
        component CM
    bedplate_I : numpy array[6], [kg*m**2]
        component I
    hvac_mass : float, [kg]
        component mass
    hvac_cm : float, [m]
        component center of mass
    hvac_I : numpy array[3], [m]
        component mass moments of inertia
    platform_mass : float, [kg]
        component mass
    platform_cm : numpy array[3], [m]
        component center of mass
    platform_I : numpy array[3], [m]
        component mass moments of inertia
    cover_mass : float, [kg]
        component mass
    cover_cm : numpy array[3], [m]
        component center of mass
    cover_I : numpy array[3], [m]
        component mass moments of inertia

    Returns
    -------
    shaft_start : numpy array[3], [m]
        coordinate of start of shaft relative to tower top
    other_mass : float, [kg]
        mass of high speed shaft, hvac, main frame, yaw, cover, and electronics
    mean_bearing_mass : float, [kg]
        average mass of all bearings (currently 2) for summary mass and cost calculations
    total_bedplate_mass : float, [kg]
        bedplate and nose mass for summary mass and cost calculations
    above_yaw_mass : float, [kg]
       overall nacelle mass excluding yaw
    nacelle_mass : float, [kg]
        overall nacelle mass including yaw
    nacelle_cm : numpy array[3], [m]
        coordinates of the center of mass of the nacelle (including yaw) in tower top coordinate system [x,y,z]
    above_yaw_cm : numpy array[3], [m]
        coordinates of the center of mass of the nacelle (excluding yaw) in tower top coordinate system [x,y,z]
    nacelle_I : numpy array[6], [kg*m**2]
        moments of inertia for the nacelle (including yaw) [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] around its center of mass
    nacelle_I_TT : numpy array[6], [kg*m**2]
        moments of inertia for the nacelle (including yaw) [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] around the tower top
    above_yaw_I : numpy array[6], [kg*m**2]
        moments of inertia for the nacelle (excluding yaw) [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] around its center of mass
    above_yaw_I_TT : numpy array[6], [kg*m**2]
        moments of inertia for the nacelle (excluding yaw) [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] around the tower top
    
    ----- extended by Vasudev Gupta (v) -----
    msa_mass : float, [kg]
        main shaft assmebly (lss + mb1 + mb2)_mass
    """

    def initialize(self):
        self.options.declare("direct_drive", default=True)

    def setup(self):
        self.add_discrete_input("upwind", True)
        self.add_discrete_input("uptower", True)
        self.add_input("tilt", 0.0, units="deg")
        self.add_input("mb1_mass", 0.0, units="kg")
        self.add_input("mb1_cm", 0.0, units="m")
        self.add_input("mb1_I", np.zeros(3), units="kg*m**2")
        self.add_input("mb2_mass", 0.0, units="kg")
        self.add_input("mb2_cm", 0.0, units="m")
        self.add_input("mb2_I", np.zeros(3), units="kg*m**2")
        self.add_input("gearbox_mass", 0.0, units="kg")
        self.add_input("gearbox_cm", 0.0, units="m")
        self.add_input("gearbox_I", np.zeros(3), units="kg*m**2")
        self.add_input("hss_mass", 0.0, units="kg")
        self.add_input("hss_cm", 0.0, units="m")
        self.add_input("hss_I", np.zeros(3), units="kg*m**2")
        self.add_input("brake_mass", 0.0, units="kg")
        self.add_input("brake_cm", 0.0, units="m")
        self.add_input("brake_I", np.zeros(3), units="kg*m**2")
        self.add_input("generator_mass", 0.0, units="kg")
        self.add_input("generator_rotor_mass", val=0.0, units="kg")
        self.add_input("generator_stator_mass", val=0.0, units="kg")
        self.add_input("generator_cm", 0.0, units="m")
        self.add_input("generator_I", np.zeros(3), units="kg*m**2")
        self.add_input("generator_rotor_I", val=np.zeros(3), units="kg*m**2")
        self.add_input("generator_stator_I", val=np.zeros(3), units="kg*m**2")
        self.add_input("nose_mass", val=0.0, units="kg")
        self.add_input("nose_cm", val=0.0, units="m")
        self.add_input("nose_I", val=np.zeros(3), units="kg*m**2")
        self.add_input("lss_mass", val=0.0, units="kg")
        self.add_input("lss_cm", val=0.0, units="m")
        self.add_input("lss_I", val=np.zeros(3), units="kg*m**2")
        self.add_input("converter_mass", 0.0, units="kg")
        self.add_input("converter_cm", np.zeros(3), units="m")
        self.add_input("converter_I", np.zeros(3), units="kg*m**2")
        self.add_input("transformer_mass", 0.0, units="kg")
        self.add_input("transformer_cm", np.zeros(3), units="m")
        self.add_input("transformer_I", np.zeros(3), units="kg*m**2")
        self.add_input("yaw_mass", 0.0, units="kg")
        self.add_input("yaw_cm", np.zeros(3), units="m")
        self.add_input("yaw_I", np.zeros(3), units="kg*m**2")
        self.add_input("bedplate_mass", 0.0, units="kg")
        self.add_input("bedplate_cm", np.zeros(3), units="m")
        self.add_input("bedplate_I", np.zeros(6), units="kg*m**2")
        self.add_input("hvac_mass", 0.0, units="kg")
        self.add_input("hvac_cm", 0.0, units="m")
        self.add_input("hvac_I", np.zeros(3), units="m")
        self.add_input("platform_mass", 0.0, units="kg")
        self.add_input("platform_cm", np.zeros(3), units="m")
        self.add_input("platform_I", np.zeros(3), units="m")
        self.add_input("cover_mass", 0.0, units="kg")
        self.add_input("cover_cm", np.zeros(3), units="m")
        self.add_input("cover_I", np.zeros(3), units="m")
        self.add_input("x_bedplate", val=np.zeros(12), units="m")
        self.add_input("constr_height", 0.0, units="m")
        self.add_input("above_yaw_mass_user", 0.0, units="kg")

        self.add_output("shaft_start", np.zeros(3), units="m")
        self.add_output("other_mass", 0.0, units="kg")
        self.add_output("mean_bearing_mass", 0.0, units="kg")
        self.add_output("total_bedplate_mass", 0.0, units="kg")
        self.add_output("nacelle_mass", 0.0, units="kg")
        self.add_output("above_yaw_mass", 0.0, units="kg")
        self.add_output("nacelle_cm", np.zeros(3), units="m")
        self.add_output("above_yaw_cm", np.zeros(3), units="m")
        self.add_output("nacelle_I", np.zeros(6), units="kg*m**2")
        self.add_output("nacelle_I_TT", np.zeros(6), units="kg*m**2")
        self.add_output("above_yaw_I", np.zeros(6), units="kg*m**2")
        self.add_output("above_yaw_I_TT", np.zeros(6), units="kg*m**2")
        self.add_output("msa_mass", 0.0, units="kg") #(v) ----- extended below 

        self._mass_table = None

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        Cup = -1.0 if discrete_inputs["upwind"] else 1.0
        tilt = float(np.deg2rad(inputs["tilt"][0]))

        components = [
            "mb1",
            "mb2",
            "lss",
            "hss",
            "brake",
            "gearbox",
            "generator_rotor",
            "generator_stator",
            "generator",
            "hvac",
            "nose",
            "bedplate",
            "platform",
            "cover",
        ]   #(v) `yaw` also added in line 1041
        if discrete_inputs["uptower"]:
            components.extend(["transformer", "converter"])

        # Mass and CofM summaries first because will need them for I later
        m_nac = msa_mass = 0.0 #(v) added `msa_mass` 
        cm_nac = np.zeros(3)
        shaft0 = np.zeros(3)
        shaft0[-1] += inputs["constr_height"][0]
        if self.options["direct_drive"]:
            shaft0[0] += inputs["x_bedplate"][-1]
        outputs["shaft_start"] = shaft0

        for k in components:
            if k in ["generator_rotor", "generator_stator"]:
                continue
            m_i = inputs[k + "_mass"]
            cm_i = inputs[k + "_cm"]

            # If cm is (x,y,z) then it is already in tower-top c.s.  If it is a scalar, it is in distance from tower and we have to convert
            if len(cm_i) == 1:
                cm_i = shaft0 + cm_i * np.array([Cup * np.cos(tilt), 0.0, np.sin(tilt)])

            if k in ["mb1", "mb2", "lss"]: msa_mass += m_i #(v)
            m_nac += m_i
            cm_nac += m_i * cm_i

        # Complete CofM calculation
        cm_nac /= m_nac

        # Now find total I about nacelle CofM
        I_nac = np.zeros(6)
        m_list = np.zeros((len(components) + 3,))
        cm_list = np.zeros((len(components) + 3, 3))
        I_cm_list = np.zeros((len(components) + 3, 6))
        I_TT_list = np.zeros((len(components) + 3, 6))
        for ic, c in enumerate(components):
            m_i = float(inputs[c + "_mass"][0])
            cm_i = inputs["generator_cm"] if c.find("generator") >= 0 else inputs[c + "_cm"]
            I_i = inputs[c + "_I"]

            # Rotate MofI if in hub c.s.
            if len(cm_i) == 1:
                cm_i = shaft0 + cm_i * np.array([Cup * np.cos(tilt), 0.0, np.sin(tilt)])
                I_i = util.rotateI(I_i, -Cup * tilt, axis="y")
            else:
                I_i = np.r_[I_i, np.zeros(3)]

            r = cm_i - cm_nac
            if not c in ["generator_rotor", "generator_stator"]:
                I_add = util.assembleI(I_i) + m_i * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
                I_add = util.unassembleI(I_add)
                I_nac += I_add

            # Record mass, cm, and I for output table
            m_list[ic] = m_i
            cm_list[ic, :] = cm_i
            I_TT_list[ic, :] = util.unassembleI(
                util.assembleI(I_i) + m_i * (np.dot(cm_i, cm_i) * np.eye(3) - np.outer(cm_i, cm_i))
            )
            I_i = inputs[c + "_I"]
            I_cm_list[ic, :] = I_i if I_i.size == 6 else np.r_[I_i, np.zeros(3)]

        m_nac_usr = float(inputs["above_yaw_mass_user"][0])
        coeff = 1.0 if m_nac_usr == 0.0 else m_nac_usr / m_nac
        m_nac *= coeff
        I_nac *= coeff
        outputs["above_yaw_mass"] = copy.copy(m_nac)
        outputs["above_yaw_cm"] = R = cm_nac.copy()
        outputs["above_yaw_I"] = I_nac.copy()
        parallel_axis = m_nac * (np.dot(R, R) * np.eye(3) - np.outer(R, R))
        outputs["above_yaw_I_TT"] = util.unassembleI(util.assembleI(I_nac) + parallel_axis)

        m_nac += inputs["yaw_mass"]
        cm_nac = (outputs["above_yaw_mass"] * outputs["above_yaw_cm"] + inputs["yaw_cm"] * inputs["yaw_mass"]) / m_nac
        r = inputs["yaw_cm"] - cm_nac
        I_add = util.assembleI(np.r_[inputs["yaw_I"], np.zeros(3)]) + inputs["yaw_mass"] * (
            np.dot(r, r) * np.eye(3) - np.outer(r, r)
        )
        I_add = util.unassembleI(I_add)
        I_nac += I_add

        outputs["nacelle_mass"] = m_nac
        outputs["nacelle_cm"] = cm_nac
        outputs["nacelle_I"] = I_nac
        outputs["msa_mass"] = msa_mass #(v)

        # Find nacelle MoI about tower top
        R = cm_nac
        parallel_axis = m_nac * (np.dot(R, R) * np.eye(3) - np.outer(R, R))
        outputs["nacelle_I_TT"] = util.unassembleI(util.assembleI(I_nac) + parallel_axis)

        # Store other misc outputs
        outputs["other_mass"] = (
            inputs["hvac_mass"]
            + inputs["platform_mass"]
            + inputs["cover_mass"]
            + inputs["yaw_mass"]
            + inputs["converter_mass"]
            + inputs["transformer_mass"]
        )
        outputs["mean_bearing_mass"] = 0.5 * (inputs["mb1_mass"] + inputs["mb2_mass"])
        outputs["total_bedplate_mass"] = inputs["nose_mass"] + inputs["bedplate_mass"]

        # Wrap up nacelle mass table
        components.append("Above_yaw")
        m_list[-3] = outputs["above_yaw_mass"][0]
        cm_list[-3, :] = outputs["above_yaw_cm"]
        I_cm_list[-3, :] = outputs["above_yaw_I"]
        I_TT_list[-3, :] = outputs["above_yaw_I_TT"]
        components.append("yaw")
        m_list[-2] = inputs["yaw_mass"][0]
        cm_list[-2, :] = inputs["yaw_cm"]
        I_cm_list[-2, :] = I_TT_list[-2, :] = np.r_[inputs["yaw_I"], np.zeros(3)]
        components.append("nacelle")
        m_list[-1] = m_nac[0]
        cm_list[-1, :] = cm_nac
        I_cm_list[-1, :] = I_nac
        I_TT_list[-1, :] = outputs["nacelle_I_TT"]
        self._mass_table = pd.DataFrame()
        self._mass_table["Component"] = components
        self._mass_table["Mass"] = m_list
        self._mass_table["CoM_TT_x"] = cm_list[:, 0]
        self._mass_table["CoM_TT_y"] = cm_list[:, 1]
        self._mass_table["CoM_TT_z"] = cm_list[:, 2]
        self._mass_table["MoI_CoM_xx"] = I_cm_list[:, 0]
        self._mass_table["MoI_CoM_yy"] = I_cm_list[:, 1]
        self._mass_table["MoI_CoM_zz"] = I_cm_list[:, 2]
        self._mass_table["MoI_CoM_xy"] = I_cm_list[:, 3]
        self._mass_table["MoI_CoM_xz"] = I_cm_list[:, 4]
        self._mass_table["MoI_CoM_yz"] = I_cm_list[:, 5]
        self._mass_table["MoI_TT_xx"] = I_TT_list[:, 0]
        self._mass_table["MoI_TT_yy"] = I_TT_list[:, 1]
        self._mass_table["MoI_TT_zz"] = I_TT_list[:, 2]
        self._mass_table["MoI_TT_xy"] = I_TT_list[:, 3]
        self._mass_table["MoI_TT_xz"] = I_TT_list[:, 4]
        self._mass_table["MoI_TT_yz"] = I_TT_list[:, 5]
        self._mass_table.set_index("Component", inplace=True)
        #print(self._mass_table[["Mass","CoM_TT_x"]])


# --------------------------------------------


class RNA_Adder(om.ExplicitComponent):
    """
    Compute mass and moments of inertia for RNA system.

    Parameters
    ----------
    upwind : boolean
        Flag whether the design is upwind or downwind
    tilt : float, [deg]
        Shaft tilt
    L_drive : float, [m]
        Length of drivetrain from bedplate to hub flang
    shaft_start : numpy array[3], [m]
        coordinate of start of shaft relative to tower top
    blades_mass : float, [kg]
        Mass of all blades
    hub_system_mass : float, [kg]
        Mass of hub system (hub + spinner + pitch)
    nacelle_mass : float, [kg]
        Mass of nacelle system
    blades_cm : float, [m]
        Center of mass for all blades from blade attachment centerpoint in hub c.s.
    hub_system_cm : float, [m]
        Hub center of mass from hub flange in hub c.s.
    nacelle_cm : numpy array[3], [m]
        Nacelle center of mass relative to tower top in yaw-aligned c.s.
    blades_I : numpy array[6], [kg*m**2]
        Mass moments of inertia of all blades about hub center
    hub_system_I : numpy array[6], [kg*m**2]
        Mass moments of inertia of hub system about its CofM
    nacelle_I_TT : numpy array[6], [kg*m**2]
        Mass moments of inertia of nacelle about the tower top

    Returns
    -------
    rotor_mass : float, [kg]
        Mass of blades and hub system
    rna_mass : float, [kg]
        Total RNA mass
    rna_cm : numpy array[3], [m]
        RNA center of mass relative to tower top in yaw-aligned c.s.
    rna_I_TT : numpy array[6], [kg*m**2]
        Mass moments of inertia of RNA about tower top in yaw-aligned coordinate system
    """

    def setup(self):
        self.add_discrete_input("upwind", True)
        self.add_input("tilt", 0.0, units="deg")
        self.add_input("L_drive", 0.0, units="m")
        self.add_input("shaft_start", np.zeros(3), units="m")
        self.add_input("blades_mass", 0.0, units="kg")
        self.add_input("hub_system_mass", 0.0, units="kg")
        self.add_input("nacelle_mass", 0.0, units="kg")
        self.add_input("blades_cm", 0.0, units="m")
        self.add_input("hub_system_cm", 0.0, units="m")
        self.add_input("nacelle_cm", np.zeros(3), units="m")
        self.add_input("blades_I", np.zeros(6), units="kg*m**2")
        self.add_input("hub_system_I", np.zeros(6), units="kg*m**2")
        self.add_input("nacelle_I_TT", np.zeros(6), units="kg*m**2")

        self.add_output("rotor_mass", 0.0, units="kg")
        self.add_output("rna_mass", 0.0, units="kg")
        self.add_output("rna_cm", np.zeros(3), units="m")
        self.add_output("rna_I_TT", np.zeros(6), units="kg*m**2")

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        Cup = -1.0 if discrete_inputs["upwind"] else 1.0
        tilt = float(np.deg2rad(inputs["tilt"][0]))

        hub_mass = inputs["hub_system_mass"]
        blades_mass = inputs["blades_mass"]
        nac_mass = inputs["nacelle_mass"]

        # rna mass
        outputs["rotor_mass"] = rotor_mass = hub_mass + blades_mass
        outputs["rna_mass"] = rotor_mass + nac_mass

        # rna cm
        shaft0 = inputs["shaft_start"]
        hub_cm_in = inputs["hub_system_cm"]
        blades_cm_in = inputs["blades_cm"]
        L_drive = inputs["L_drive"]
        cm_array = np.array([Cup * np.cos(tilt), 0.0, np.sin(tilt)])
        hub_cm = shaft0 + (L_drive + hub_cm_in) * cm_array
        blades_cm = shaft0 + (L_drive + hub_cm_in + blades_cm_in) * cm_array
        outputs["rna_cm"] = (hub_mass * hub_cm +
                             blades_mass * blades_cm +
                             nac_mass * inputs["nacelle_cm"]) / outputs["rna_mass"]

        # rna I
        hub_I = util.assembleI(util.rotateI(inputs["hub_system_I"], -Cup * tilt, axis="y"))
        blades_I = util.assembleI(util.rotateI(inputs["blades_I"], -Cup * tilt, axis="y"))
        nac_I_TT = util.assembleI(inputs["nacelle_I_TT"])

        R = hub_cm
        hub_I_TT = hub_I + hub_mass * (np.dot(R, R) * np.eye(3) - np.outer(R, R))

        R = blades_cm
        blades_I_TT = blades_I + blades_mass * (np.dot(R, R) * np.eye(3) - np.outer(R, R))

        outputs["rna_I_TT"] = util.unassembleI(hub_I_TT + blades_I_TT + nac_I_TT)


# --------------------------------------------


class DriveDynamics(om.ExplicitComponent):
    """
    Compute equivalent spring constant and damping for the drivetrain system

    Parameters
    ----------
    lss_spring_constant : float, [N*m/rad]
        Equivalent spring constant for the low speed shaft froom T=(G*J/L)*theta
    hss_spring_constant : float, [N*m/rad]
        Equivalent spring constant for the high speed shaft froom T=(G*J/L)*theta
    gear_ratio : float
        overall gearbox ratio
    damping_ratio : float
        Fraction of critical damping value for drivetrain system
    blades_I : numpy array[6], [kg*m**2]
        Mass moments of inertia of all blades about hub center
    hub_system_I : numpy array[6], [kg*m**2]
        Mass moments of inertia of hub system about its CofM


    Returns
    -------
    drivetrain_spring_constant : float, [N*m/rad]
        Equivalent spring constant for the drivetrain system
    drivetrain_damping_coefficient : float, [N*m/rad]
        Equivalent damping coefficient for the drivetrain system
    """

    def setup(self):
        self.add_input("lss_spring_constant", 0.0, units="N*m/rad")
        self.add_input("hss_spring_constant", 0.0, units="N*m/rad")
        self.add_input("gear_ratio", val=1.0)
        self.add_input("damping_ratio", val=0.0)
        self.add_input("blades_I", np.zeros(6), units="kg*m**2")
        self.add_input("hub_system_I", np.zeros(6), units="kg*m**2")
        self.add_input("drivetrain_spring_constant_user", 0.0, units="N*m/rad")
        self.add_input("drivetrain_damping_coefficient_user", 0.0, units="N*m*s/rad")

        self.add_output("drivetrain_spring_constant", 0.0, units="N*m/rad")
        self.add_output("drivetrain_damping_coefficient", 0.0, units="N*m*s/rad")

    def compute(self, inputs, outputs):
        # Unpack inputs
        k_lss = inputs["lss_spring_constant"]
        k_hss = inputs["hss_spring_constant"]
        gbr = inputs["gear_ratio"]
        zeta = inputs["damping_ratio"]
        rotor_I = inputs["blades_I"] + inputs["hub_system_I"]
        k_user = inputs["drivetrain_spring_constant_user"]
        c_user = inputs["drivetrain_damping_coefficient_user"]

        # springs in series, should be n^2*k1*k2/(k1+n^2*k2)
        # https://www.nrel.gov/docs/fy09osti/41160.pdf
        k_drive = k_lss if gbr == 1.0 else 1.0 / (1 / k_lss + 1 / k_hss / gbr / gbr)
        outputs["drivetrain_spring_constant"] = k_user if k_user != 0.0 else k_drive

        # Critical damping value
        c_crit = 2.0 * np.sqrt(k_drive * rotor_I[0])
        outputs["drivetrain_damping_coefficient"] = c_user if c_user != 0.0 else zeta * c_crit

# --------------------------------------------

# (v) --------------------------------------------
class MainBearing_withDerivatives(om.ExplicitComponent):
    """
    Heavy-load bearings with derivatives for drivetrain optimizaiton.

    Internal Progress
    _________________
    - TODO : implement and check
    - TODO : change to high-load values (now: some low, some high)
    """
    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------
    def setup(self):

        self.add_discrete_input("bearing_type", "CARB")

        self.add_input("D_bearing", -1.0, units="m")
        self.add_input("D_shaft", 0.0, units="m")
        self.add_input("mb_mass_user", -1.0, units="kg")
        self.add_input("mb_e", 0.35)

        self.add_output("face_width", 0.0, units="m")
        self.add_output("mb_mass", 0.0, units="kg")
        self.add_output("mb_Cr", 0.0, units="N")
        self.add_output("mb_I", np.zeros(3), units="kg*m**2")
        self.add_output("mb_max_defl_ang", 0.0, units="rad")

        # constants (no derivatives needed)
        self.add_output("mb_p", 10/3)
        self.add_output("mb_X1", 0.0)
        self.add_output("mb_Y1", 0.0) # TODO check wrt. e
        self.add_output("mb_X2", 0.0) # TODO check wrt. e
        self.add_output("mb_Y2", 0.0)
        self.add_output("mb_Reactions", np.zeros(6, dtype=int))

        # --------------------------------------------------------
        # partials
        # --------------------------------------------------------
        # wrt. D_shaft
        wrt = ["D_shaft"]
        self.declare_partials("face_width", wrt)
        self.declare_partials("mb_mass", wrt)
        self.declare_partials("mb_Cr", wrt)
        self.declare_partials("mb_I", wrt)
        # wrt. D_bearing
        self.declare_partials("mb_I", "D_bearing")
        # wrt. mb_e
        self.declare_partials("mb_Y1", "mb_e")
        self.declare_partials("mb_Y2", "mb_e")

    # ------------------------------------------------------------
    # Bearing database (clean + extendable)
    # width = a*D + b
    # mass  = k*D^n
    # Cr    = c*D^m
    # ----
    # NOTE: `BEARINGS` dict not inside self, but stored at class/module level:
    # - coz static data (same for all components, independent of inputs/outputs)
    # - so its loads once per python process, and shared across all instances (and MPI ranks)
    # - if in self = inefficient: rebuild same dict repeatedly coz setup exec per instance.
    # ------------------------------------------------------------
    BEARINGS = {

        "CARB":
            dict(a=0.4299, b=0.0382, k=3682.8, n=2.7676, c=16676, m=1.4746,
                 max_ang=np.deg2rad(0.5), reactions=[0,1,1,0,0,0]),

        "CRB":
            dict(a=0.157,  b=0.0849, k=1070.8, n=1.8278, c=4526.5, m=0.9556,
                 max_ang=np.deg2rad(4/60), reactions=[0,1,1,0,0,0]),

        "SRB":
            dict(a=0.2463, b=0.185, k=2688.3,  n=1.8877, c=13878, m=1.0796,
                 max_ang=0.078, reactions=[1,1,1,0,0,0]),

        "TRB":
            dict(a=0.1499, b=0.0,    k=543.01, n=1.9043, c=1993.8, m=0.318,
                 max_ang=np.deg2rad(3/60), reactions=[1,1,1,0,1,1]),

        "TRB2":
            dict(a=0.1541, b=0.2087, k=1442.6, n=1.8932, c=6579.9, m=0.8592,
                 max_ang=np.deg2rad((0.06+0.02)/2), reactions=[1,1,1,0,1,1]),
    }

    housing_factor = 1 + 80.0/27.0
    kN_to_N = 1e3

    # ------------------------------------------------------------
    # compute
    # ------------------------------------------------------------
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        D_shaft = inputs["D_shaft"]
        D_bearing_input = inputs["D_bearing"]
        mass_user = inputs["mb_mass_user"]
        e = inputs["mb_e"]

        btype = discrete_inputs["bearing_type"].upper()
        data = self.BEARINGS[btype]

        a, b = data["a"], data["b"]         # face width
        k, n = data["k"], data["n"]         # mass
        c, m = data["c"], data["m"]         # Cr
        # reactions
        FREE, RIGID = 0, 1
        mb_Reactions = np.array([FREE]*6) # ([ Rx, Ry, Rz, Rxx, Ryy, Rzz ])
        mb_Reactions[:] = data["reactions"]
        if mb_Reactions[3] == RIGID: mb_Reactions[3] = FREE

        # -----------------------------
        # analytic formulas
        # -----------------------------
        self.face_width = face = a * D_shaft + b
        self.Cr = Cr   = (c * (D_shaft**m)) * self.kN_to_N

        # mass
        if mass_user > 0:
            self._mass = mass = mass_user
            self._dmass_dmUser = 1.0
            self._dmass_dDs = 0.0
        else:
            self._mass = mass = (k * D_shaft**n) * self.housing_factor
            self._dmass_dmUser = 0.0
            self._dmass_dDs = (n*mass)/D_shaft

        # bearing diameter
        if D_bearing_input > 0:
            self._Db = Db = D_bearing_input
            self._dDb_dDbearing = 1.0
            self._dDb_dDs = 0.0
            
        else:
            self._Db = Db = face
            self._dDb_dDbearing = 0.0
            self._dDb_dDs = a
        
        # inertia
        # I0 = (1/4) * mass * (4 * (0.5 * D_shaft) ** 2 + 3 * (0.5 * Db) ** 2)
        self._A0 = A0 = ( D_shaft**2 + 0.75*Db**2 )
        I0 = 0.25 * mass * A0
        # I1 = (1/8) * mass * (4 * (0.5 * D_shaft) ** 2 + 5 * (0.5 * Db) ** 2)
        self._A1 = A1 = ( D_shaft**2 + 1.25*Db**2 )
        I1 = 0.125 * mass * A1
        I = np.r_[I0, I1, I1]

        # -----------------------------
        # outputs
        # -----------------------------
        outputs["face_width"] = face
        outputs["mb_mass"] = mass
        outputs["mb_Cr"] = Cr
        outputs["mb_I"] = I
        outputs["mb_max_defl_ang"] = data["max_ang"]
        outputs["mb_Reactions"] = mb_Reactions

        # ISO factors
        alpha = np.arctan(e/1.5)
        outputs["mb_X1"] = 1.0
        outputs["mb_Y1"] = 0.45/np.tan(alpha)
        outputs["mb_X2"] = 0.67
        outputs["mb_Y2"] = outputs["mb_X2"]/np.tan(alpha)
        outputs["mb_p"] = 10/3

    # ------------------------------------------------------------
    # compute_partials (exact)
    # ------------------------------------------------------------
    def compute_partials(self, inputs, J, discrete_inputs):

        Ds = inputs["D_shaft"]
        D_bearing = inputs["D_bearing"]
        mass_user = inputs["mb_mass_user"]

        btype = discrete_inputs["bearing_type"].upper()
        data = self.BEARINGS[btype]

        a, b = data["a"], data["b"]     # face width
        k, n = data["k"], data["n"]     # mass
        c, m = data["c"], data["m"]     # Cr

        # width
        J["face_width", "D_shaft"] = a
        # Cr
        J["mb_Cr", "D_shaft"] = (m*self.Cr)/Ds

        # mass
        mass = self._mass
        dmass_dDs = self._dmass_dDs
        dmass_dmUser = self._dmass_dmUser
        J["mb_mass", "D_shaft"] = dmass_dDs

        # bearing diameter
        Db = self._Db
        dDb_dDbearing = self._dDb_dDbearing
        dDb_dDs = self._dDb_dDs

        # inertia
        A0 = self._A0
        A1 = self._A1

        dA0_dDs = 2*Ds + 0.75*2*Db*dDb_dDs
        dA1_dDs = 2*Ds + 1.25*2*Db*dDb_dDs

        dI0_dDs = 0.25*(dmass_dDs*A0 + mass*dA0_dDs)
        dI1_dDs = 0.125*(dmass_dDs*A1 + mass*dA1_dDs)

        J["mb_I", "D_shaft"] = np.array([dI0_dDs, dI1_dDs, dI1_dDs])

        # inertia wrt. D_bearing
        dI0_dDb = (0.25)*mass*(0.75)*2*Db*dDb_dDbearing
        dI1_dDb = (0.125)*mass*(1.25)*2*Db*dDb_dDbearing
        J["mb_I", "D_bearing"] = np.array([dI0_dDb, dI1_dDb, dI1_dDb])

        # Y1, Y2 wrt. mb_e
        e = inputs["mb_e"]

        alpha = np.arctan(e/1.5)
        t = np.tan(alpha)

        dalpha_de = (1/(1+(e/1.5)**2))*(1/1.5)
        dtdalpha = 1 + t**2

        C1 = 0.45
        C2 = 0.67

        dY1_de = -C1/t**2 * dtdalpha * dalpha_de
        dY2_de = -C2/t**2 * dtdalpha * dalpha_de

        J["mb_Y1","mb_e"] = dY1_de
        J["mb_Y2","mb_e"] = dY2_de
# --------------------------------------------

class Nacelle_MOO_withDerivatives(om.ExplicitComponent):
    """
    Takes nacelle mass and center wrt. tower-top and computes a multi-objective function
    for optimization. The function is a weighted sum of the nacelle mass and
    the distance of the nacelle center of mass from the tower top center
    (which is a surrogate for minimizing loads on the tower).
    The user can specify the weights to place on each component of the objective.

    Parameters
    ----------
    nacelle_mass : float, [kg]
        Mass of the nacelle
    nacelle_cm : numpy array[3], [m]
        Center of mass of the nacelle relative to tower top in yaw-aligned c.s.
    moo_weight : float, [0-1]
        Weighting factor for multi-objective function.
            0 = only minimize distance, 1 = only minimize mass,
            values in between give a weighted sum of the two.

    Outputs
    -------
    nacelle_moo : float
        Multi-objective function value for nacelle design, to be minimized in optimization.
    """

    def setup(self):
        self.add_input("nacelle_mass", 0.0, units="kg")
        self.add_input("nacelle_cm", np.zeros(3), units="m")
        self.add_input("moo_weight", 1.0)

        self.add_output("nacelle_moo", 0.0)

        # Declare partials
        self.declare_partials("nacelle_moo", "nacelle_mass")
        self.declare_partials("nacelle_moo", "nacelle_cm")
        self.declare_partials("nacelle_moo", "moo_weight")

    # ------------------------------------------------------------
    # compute
    # ------------------------------------------------------------
    def compute(self, inputs, outputs):

        mass = inputs["nacelle_mass"]
        cm = inputs["nacelle_cm"]
        w = inputs["moo_weight"]

        mass_scale = 1e5 # scale mass to be on same order as distance for better numerical behavior in optimization

        r = np.linalg.norm(cm) # Distance of nacelle CoM from tower centerline (assuming tower at 0,0,0)

        # Store for partials
        self._mass = mass
        self._cm = cm
        self._w = w
        self._r = r
        self._mass_scale = mass_scale

        # Multi-objective function: weighted sum of mass and distance
        outputs["nacelle_moo"] = (
            (w * mass/mass_scale) +
            (1.0-w) * r
        )

    # ------------------------------------------------------------
    # compute_partials
    # ------------------------------------------------------------
    def compute_partials(self, inputs, J):

        mass = self._mass
        cm = self._cm
        w = self._w
        r = self._r
        mass_scale = self._mass_scale

        eps = 1e-16
        r_safe = max(r, eps) # If cm = [0,0,0], derivative is undefined (division by zero). Must protect against that. <- standard practice in optimization.

        # wrt mass
        J["nacelle_moo", "nacelle_mass"] = (w / mass_scale)

        # wrt weight
        J["nacelle_moo", "moo_weight"] = (
            (mass/mass_scale) - r
        )

        # wrt nacelle_cm (vector)
        J["nacelle_moo", "nacelle_cm"] = (
            (1.0-w) * (cm / r_safe)
        )
# --------------------------------------------