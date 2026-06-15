# --------------------------------------------
import openmdao.api as om
import numpy as np

class Converter(om.ExplicitComponent):
    """
    Power electronics converter design and analysis component.
    
    This component performs comprehensive electrical converter design including:
    - Voltage level selection based on power rating
    - LC filter design (inductors and capacitors)
    - DC link capacitor sizing
    - IGBT module selection and sizing
    - Cost estimation
    - Loss and efficiency calculations
    
    The converter is designed as a back-to-back (B2B) configuration with rectifier 
    and inverter stages sharing a common DC link.
    
    Parameters
    ----------
    machine_rating : float, [kW]
        Rated power of the converter
    f_grid : float, [Hz]
        Grid frequency
    f_sw : float, [Hz]
        Switching frequency
    V_dc_drop : float
        DC voltage drop in per-unit
    m_max : float
        Maximum modulation index in per-unit
    V_L_drop : float
        Inductor voltage drop in per-unit
    V_dc_ripple : float
        DC voltage ripple in per-unit
    Q_cap_pu : float
        Reactive power consumption in per-unit
    v_step_V : float, [V]
        Voltage ceiling step for VLL and Vdc outputs
    c_step_uF : float, [uF]
        Capacitor ceiling step
    l_step_uH : float, [uH]
        Inductor ceiling step
    Vces_V : float, [V]
        IGBT module collector-emitter voltage rating
    Ic_A : float, [A]
        IGBT module current rating
    derate_v : float
        Voltage derating factor
    i_limit : float
        Current limit factor
    use_conservative_current : bool
        If True, uses I_rms for switch current; if False, uses I_rms/sqrt(2)
    aux_frac : float
        Auxiliary losses as fraction of rated power
    Idc_ripple_pu : float
        DC current ripple in per-unit
    
    Returns
    -------
    V_LL_rms_V : float, [V]
        Line-to-line RMS voltage
    Vdc_V : float, [V]
        DC link voltage
    Vdc_min_V : float, [V]
        Minimum required DC voltage
    I_rms_A : float, [A]
        RMS current per phase
    I_pk_A : float, [A]
        Peak current per phase
    f_res_Hz : float, [Hz]
        LC filter resonant frequency
    Rf_ohm : float, [ohm]
        Filter damping resistor
    Cf_uF : float, [uF]
        AC filter capacitance per phase
    Lf_uH : float, [uH]
        AC filter inductance per phase
    Cdc_uF : float, [uF]
        DC link capacitance
    Ns_series : int
        Number of IGBT modules in series per phase leg
    Np_parallel : int
        Number of IGBT modules in parallel per phase leg
    N_igbt_modules : int
        Total number of IGBT modules per converter
    Cost_semiconductors : float, [$]
        Cost of semiconductor modules
    Cost_dc_cap : float, [$]
        Cost of DC link capacitor
    Cost_filt_ind_3ph : float, [$]
        Cost of 3-phase AC filter inductors
    Cost_filt_cap : float, [$]
        Cost of AC filter capacitors
    Cost_passive_total : float, [$]
        Total cost of passive components
    Cost_BOS_total : float, [$]
        Total balance-of-system cost
    Cost_converter : float, [$]
        Total cost per converter
    Cost_B2B : float, [$]
        Total cost of back-to-back converter system
    P_loss_total_kW : float, [kW]
        Total converter losses at rated power
    eta_conv_pct : float, [%]
        Converter efficiency at rated power
    """
    
    # Default rating table for voltage level selection
    DEFAULT_RATING_TABLE = [
        (5.0, 2.3, 4.0),      # (P_MW, VLL_kV, Vdc_kV)
        (10.0, 4.16, 7.2),
        (15.0, 6.6, 11.5),
        (20.0, 10.0, 18.0),
        (25.0, 11.0, 20.0),
    ]
    
    # Default DC capacitor ESR values (MW : mohm)
    DEFAULT_DC_ESR = {
        5: 0.5,
        10: 0.8,
        15: 1.3,
        20: 2.5,
        25: 2.8,
    }
    
    def setup(self):
        # Design parameter inputs
        self.add_input("machine_rating", val=0.0, units="kW", desc="Rated power of the converter")
        self.add_input("f_grid", 60.0, units="Hz", desc="Grid frequency")
        self.add_input("f_sw", 1620.0, units="Hz", desc="Switching frequency")
        self.add_input("V_dc_drop", 0.05, desc="DC voltage drop [pu]")
        self.add_input("m_max", 1.0, desc="Maximum modulation index [pu]")
        self.add_input("V_L_drop", 0.20, desc="Inductor voltage drop [pu]")
        self.add_input("V_dc_ripple", 0.05, desc="DC voltage ripple [pu]")
        self.add_input("Q_cap_pu", 0.05, desc="Reactive power consumption [pu]")
        self.add_input("v_step_V", 10.0, units="V", desc="Voltage ceiling step")
        self.add_input("c_step_uF", 10.0, units="uF", desc="Capacitor ceiling step")
        self.add_input("l_step_uH", 10.0, units="uH", desc="Inductor ceiling step")
        
        # IGBT module parameters
        self.add_input("Vces_V", 1700.0, units="V", desc="IGBT voltage rating")
        self.add_input("Ic_A", 1800.0, units="A", desc="IGBT current rating")
        self.add_input("derate_v", 0.70, desc="Voltage derating factor")
        self.add_input("i_limit", 1.15, desc="Current limit factor")
        self.add_discrete_input("use_conservative_current", True, desc="Use conservative current sizing")
        
        # Loss model parameters
        self.add_input("aux_frac", 0.0035, desc="Auxiliary losses fraction")
        self.add_input("Idc_ripple_pu", 0.20, desc="DC current ripple [pu]")
        
        # Design outputs - Electrical parameters
        self.add_output("V_LL_rms_V", 0.0, units="V", desc="Line-to-line RMS voltage")
        self.add_output("Vdc_V", 0.0, units="V", desc="DC link voltage")
        self.add_output("Vdc_min_V", 0.0, units="V", desc="Minimum required DC voltage")
        self.add_output("I_rms_A", 0.0, units="A", desc="RMS current per phase")
        self.add_output("I_pk_A", 0.0, units="A", desc="Peak current per phase")
        
        # Filter design outputs
        self.add_output("f_res_Hz", 0.0, units="Hz", desc="LC filter resonant frequency")
        self.add_output("Rf_ohm", 0.0, units="ohm", desc="Filter damping resistor")
        self.add_output("Cf_uF", 0.0, units="uF", desc="AC filter capacitance")
        self.add_output("Lf_uH", 0.0, units="uH", desc="AC filter inductance")
        self.add_output("Cdc_uF", 0.0, units="uF", desc="DC link capacitance")
        
        # IGBT sizing outputs
        self.add_output("Ns_series", 0, desc="Series IGBT modules per phase leg")
        self.add_output("Np_parallel", 0, desc="Parallel IGBT modules per phase leg")
        self.add_output("N_igbt_modules", 0, desc="Total IGBT modules per converter")
        
        # Cost outputs
        self.add_output("Cost_semiconductors", 0.0, units="USD", desc="Semiconductor cost")
        self.add_output("Cost_dc_cap", 0.0, units="USD", desc="DC capacitor cost")
        self.add_output("Cost_filt_ind_3ph", 0.0, units="USD", desc="3-phase filter inductor cost")
        self.add_output("Cost_filt_cap", 0.0, units="USD", desc="AC filter capacitor cost")
        self.add_output("Cost_passive_total", 0.0, units="USD", desc="Total passive component cost")
        self.add_output("Cost_BOS_total", 0.0, units="USD", desc="Balance of system cost")
        self.add_output("Cost_converter", 0.0, units="USD", desc="Total cost per converter")
        self.add_output("Cost_B2B", 0.0, units="USD", desc="Total B2B converter cost")
        
        # Loss and efficiency outputs
        self.add_output("P_loss_total_kW", 0.0, units="kW", desc="Total converter losses")
        self.add_output("eta_conv_pct", 0.0, desc="Converter efficiency [%]")
    
    def _ceil_to_step(self, x, step):
        """Round up to nearest step value"""
        if step <= 0:
            raise ValueError("step must be > 0")
        return step * np.ceil(x / step)
    
    def _interpolate_v_levels(self, machine_rating_kW, v_step_V):
        """Linear interpolation of VLL_rms and Vdc based on WT ratings"""
        P_MW = machine_rating_kW / 1e3
        table = self.DEFAULT_RATING_TABLE
        
        if P_MW <= table[0][0]:
            VLL_kV, Vdc_kV = table[0][1], table[0][2]
        elif P_MW >= table[-1][0]:
            VLL_kV, Vdc_kV = table[-1][1], table[-1][2]
        else:
            # Linear interpolation
            for i in range(len(table) - 1):
                p0, p1 = table[i], table[i + 1]
                if p0[0] <= P_MW <= p1[0]:
                    alpha = (P_MW - p0[0]) / (p1[0] - p0[0])
                    VLL_kV = p0[1] + alpha * (p1[1] - p0[1])
                    Vdc_kV = p0[2] + alpha * (p1[2] - p0[2])
                    break
        
        # Convert to Volts and round up to step
        V_LL_rms = self._ceil_to_step(VLL_kV * 1e3, v_step_V)
        Vdc = self._ceil_to_step(Vdc_kV * 1e3, v_step_V)
        
        return V_LL_rms, Vdc
    
    def _converter_lc_design(self, inputs):
        """Main LC filter design calculations"""
        machine_rating_kW = inputs["machine_rating"]
        machine_rating_W = machine_rating_kW * 1e3
        f_grid = inputs["f_grid"]
        V_dc_drop = inputs["V_dc_drop"]
        m_max = inputs["m_max"]
        V_L_drop = inputs["V_L_drop"]
        V_dc_ripple = inputs["V_dc_ripple"]
        Q_cap_pu = inputs["Q_cap_pu"]
        v_step_V = inputs["v_step_V"]
        c_step_uF = inputs["c_step_uF"]
        l_step_uH = inputs["l_step_uH"]
        
        # Get voltage levels
        V_LL_rms, Vdc = self._interpolate_v_levels(machine_rating_kW, v_step_V)
        
        w0 = 2 * np.pi * f_grid
        I_rms = machine_rating_W / (np.sqrt(3) * V_LL_rms)
        I_pk = I_rms * np.sqrt(2)
        
        # Minimum DC voltage required
        Vdc_min = (2 * np.sqrt(2) * V_LL_rms) / (np.sqrt(3) * (1 - V_dc_drop) * m_max)
        
        # AC filter inductor
        Lf_raw_H = (V_L_drop * V_LL_rms) / (np.sqrt(3) * w0 * I_rms)
        Lf_uH = self._ceil_to_step(Lf_raw_H * 1e6, l_step_uH)
        Lf = Lf_uH * 1e-6
        
        # AC filter capacitor
        Q_cap = Q_cap_pu * machine_rating_W
        Cf = Q_cap / (w0 * V_LL_rms**2)
        
        # Resonant frequency and damping
        f_res = 1 / (2 * np.pi * np.sqrt(Lf * Cf))
        Q_factor = 4
        Rf = (1 / Q_factor) * np.sqrt(Lf / Cf)
        
        # DC link capacitor
        Cdc_raw_F = machine_rating_W / (2 * Vdc**2 * V_dc_ripple * w0)
        Cdc_uF = self._ceil_to_step(Cdc_raw_F * 1e6, c_step_uF)
        Cdc = Cdc_uF * 1e-6
        
        # Energy storage
        E_cap = 0.5 * Cdc * Vdc**2
        E_ind = 0.5 * Lf * I_pk**2
        
        return {
            "P_MW": machine_rating_kW / 1e3,
            "V_LL_rms_V": V_LL_rms,
            "Vdc_V": Vdc,
            "Vdc_min_V": Vdc_min,
            "I_rms_A": I_rms,
            "I_pk_A": I_pk,
            "f_res_Hz": f_res,
            "Rf_ohm": Rf,
            "Cf_uF": Cf * 1e6,
            "Lf_uH": Lf_uH,
            "Cdc_uF": Cdc_uF,
            "Qcf_VAr": Q_cap,
            "Qcf_MVAr": Q_cap * 1e-6,
            "E_cap_J": E_cap,
            "E_ind_J": E_ind,
        }
    
    def _size_halfbridge_igbt_modules(self, design, inputs, discrete_inputs):
        """Size IGBT modules based on voltage and current requirements"""
        Vdc = design["Vdc_V"]
        I_rms = design["I_rms_A"]
        
        Vces_V = inputs["Vces_V"]
        Ic_A = inputs["Ic_A"]
        derate_v = inputs["derate_v"]
        i_limit = inputs["i_limit"]
        use_conservative_current = discrete_inputs["use_conservative_current"]
        
        # Series modules for voltage
        Ns = np.ceil(Vdc / (derate_v * Vces_V))
        
        # Parallel modules for current
        if use_conservative_current:
            I_switch_rms = I_rms
        else:
            I_switch_rms = I_rms / np.sqrt(2)
        
        Np = np.ceil((i_limit * I_switch_rms) / Ic_A)
        
        # Total modules (3 phase legs per converter)
        modules_per_inv = 3 * Ns * Np
        
        return {
            "Ns_series": Ns,
            "Np_parallel": Np,
            "N_igbt_modules": modules_per_inv,
        }
    
    def _cost_models(self, N_semiconductor, E_cap_J, E_ind_J, Qcf_VAr, P_MW):
        """Calculate component and system costs"""
        Cost_semiconductors = 1120.90 * N_semiconductor
        
        Cost_dc_cap = 0.1105 * E_cap_J + 51.533
        Cost_filt_ind = (29.299 * E_ind_J + 101.13) * 3  # 3 phase
        Cost_filt_cap = 0.0097 * Qcf_VAr + 44.493
        Cost_passive_total = Cost_dc_cap + Cost_filt_ind + Cost_filt_cap
        
        Cost_gate_drive = 150 * N_semiconductor
        Cost_busbar = 1600 * P_MW
        Cost_heatsink = 1881 * P_MW
        Cost_cool_fan = 720 * P_MW
        Cost_fuse = 2700 * P_MW
        Cost_cb = 4860 * P_MW
        Cost_enclosure = 1760 * P_MW
        Cost_BOS_total = (Cost_gate_drive + Cost_busbar + Cost_heatsink + 
                         Cost_cool_fan + Cost_fuse + Cost_cb + Cost_enclosure)
        
        Cost_total = Cost_semiconductors + Cost_passive_total + Cost_BOS_total
        
        return {
            "Cost_semiconductors": Cost_semiconductors,
            "Cost_dc_cap": Cost_dc_cap,
            "Cost_filt_ind_3ph": Cost_filt_ind,
            "Cost_filt_cap": Cost_filt_cap,
            "Cost_passive_total": Cost_passive_total,
            "Cost_BOS_total": Cost_BOS_total,
            "Cost_converter": Cost_total,
            "Cost_B2B": (Cost_total * 2) - Cost_dc_cap,  # B2B shares DC link cap
        }
    
    def _semiconductor_module_loss_kW(self, Ic_kA, f_sw_Hz):
        """Calculate losses for a single semiconductor module"""
        Ic = Ic_kA
        
        Pcond_kW = 0.95 * Ic + 5.2e-4 * (Ic ** 2)
        
        Eon_mJ = -12.4 + 0.37 * Ic * 1000.0
        Eoff_mJ = 88.0 + 0.372 * Ic * 1000.0
        
        Psw_kW = (f_sw_Hz / 1000.0) * 2.0 * (Eon_mJ + Eoff_mJ) / 1000.0
        Ptt_kW = Pcond_kW + Psw_kW
        
        return Ptt_kW
    
    def _converter_loss_model(self, design, igbt_sizing, inputs):
        """Calculate total converter losses and efficiency"""
        P_MW = design["P_MW"]
        P_kW = P_MW * 1000.0
        Vdc_V = design["Vdc_V"]
        I_rms_A = design["I_rms_A"]
        Qcf_MVAr = design["Qcf_MVAr"]
        
        N_modules = igbt_sizing["N_igbt_modules"]
        f_sw_Hz = inputs["f_sw"]
        aux_frac = inputs["aux_frac"]
        Idc_ripple_pu = inputs["Idc_ripple_pu"]
        
        # Semiconductor losses
        Ic_kA = I_rms_A / 1000.0
        P_semi_kW = self._semiconductor_module_loss_kW(Ic_kA, f_sw_Hz) * N_modules
        
        # DC-link capacitor ESR losses
        Idc_A = (P_MW * 1e6) / Vdc_V
        I_dc_cap_A = Idc_ripple_pu * Idc_A
        
        # Select ESR based on nearest power rating
        dc_esr_mohm = self.DEFAULT_DC_ESR
        keys = sorted(dc_esr_mohm.keys())
        nearest_key = min(keys, key=lambda k: abs(k - P_MW))
        R_dc_ohm = dc_esr_mohm[nearest_key] * 1e-3
        
        P_dc_cap_kW = (I_dc_cap_A ** 2) * R_dc_ohm / 1000.0
        
        # Line-filter inductor losses
        P_L_total_kW = 3.0 * (Ic_kA * Ic_kA + 0.5)
        
        # AC filter capacitor losses
        P_ac_cap_kW = 0.2 * Qcf_MVAr
        
        # Auxiliary losses
        P_aux_kW = aux_frac * P_kW
        
        # Total losses and efficiency
        P_loss_total_kW = P_semi_kW + P_dc_cap_kW + P_L_total_kW + P_ac_cap_kW + P_aux_kW
        eta_pu = max(0.0, min(1.0, 1.0 - P_loss_total_kW / P_kW))
        
        return {
            "P_loss_total_kW": P_loss_total_kW,
            "eta_conv_pct": 100.0 * eta_pu,
        }
    
    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        """Main compute method"""
        # Design calculations
        design = self._converter_lc_design(inputs)
        
        # IGBT sizing
        igbt_sizing = self._size_halfbridge_igbt_modules(design, inputs, discrete_inputs)
        
        # Cost calculations
        costs = self._cost_models(
            igbt_sizing["N_igbt_modules"],
            design["E_cap_J"],
            design["E_ind_J"],
            design["Qcf_VAr"],
            design["P_MW"]
        )
        
        # Loss and efficiency calculations
        losses = self._converter_loss_model(design, igbt_sizing, inputs)
        
        # Populate outputs - Electrical parameters
        outputs["V_LL_rms_V"] = design["V_LL_rms_V"]
        outputs["Vdc_V"] = design["Vdc_V"]
        outputs["Vdc_min_V"] = design["Vdc_min_V"]
        outputs["I_rms_A"] = design["I_rms_A"]
        outputs["I_pk_A"] = design["I_pk_A"]
        
        # Filter design
        outputs["f_res_Hz"] = design["f_res_Hz"]
        outputs["Rf_ohm"] = design["Rf_ohm"]
        outputs["Cf_uF"] = design["Cf_uF"]
        outputs["Lf_uH"] = design["Lf_uH"]
        outputs["Cdc_uF"] = design["Cdc_uF"]
        
        # IGBT sizing
        outputs["Ns_series"] = igbt_sizing["Ns_series"]
        outputs["Np_parallel"] = igbt_sizing["Np_parallel"]
        outputs["N_igbt_modules"] = igbt_sizing["N_igbt_modules"]
        
        # Costs
        outputs["Cost_semiconductors"] = costs["Cost_semiconductors"]
        outputs["Cost_dc_cap"] = costs["Cost_dc_cap"]
        outputs["Cost_filt_ind_3ph"] = costs["Cost_filt_ind_3ph"]
        outputs["Cost_filt_cap"] = costs["Cost_filt_cap"]
        outputs["Cost_passive_total"] = costs["Cost_passive_total"]
        outputs["Cost_BOS_total"] = costs["Cost_BOS_total"]
        outputs["Cost_converter"] = costs["Cost_converter"]
        outputs["Cost_B2B"] = costs["Cost_B2B"]
        
        # Losses and efficiency
        outputs["P_loss_total_kW"] = losses["P_loss_total_kW"]
        outputs["eta_conv_pct"] = losses["eta_conv_pct"]

# --------------------------------------------
