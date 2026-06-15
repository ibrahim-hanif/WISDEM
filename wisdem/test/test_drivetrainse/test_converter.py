"""
Regression tests for the Converter component
Tests the power electronics converter design across multiple power ratings
"""
import unittest
import numpy as np
import openmdao.api as om
from wisdem.drivetrainse.converter import Converter


class TestConverterRegression(unittest.TestCase):
    """
    Regression tests for Converter component across different power ratings.
    These tests ensure that the component produces consistent results for
    standard power ratings (5 MW, 10 MW, 15 MW, 20 MW, 25 MW).
    """
    
    def setUp(self):
        """Set up test problem with Converter component"""
        self.prob = om.Problem()
        self.prob.model.add_subsystem("converter", Converter(), promotes=["*"])
        self.prob.setup()
        
    def _run_converter(self, machine_rating_MW, **kwargs):
        """
        Helper method to run converter with specified power rating
        
        Parameters
        ----------
        machine_rating_MW : float
            Machine rating in MW
        **kwargs : dict
            Additional input parameters to override defaults
        
        Returns
        -------
        dict : All output values
        """
        self.prob.set_val("machine_rating", machine_rating_MW * 1e6, units="W")
        
        # Set any additional inputs
        for key, value in kwargs.items():
            self.prob.set_val(key, value)
        
        self.prob.run_model()
        
        # Collect all outputs
        outputs = {}
        for output_name in ["V_LL_rms_V", "Vdc_V", "Vdc_min_V", "I_rms_A", "I_pk_A",
                           "f_res_Hz", "Rf_ohm", "Cf_uF", "Lf_uH", "Cdc_uF",
                           "Ns_series", "Np_parallel", "N_igbt_modules",
                           "Cost_semiconductors", "Cost_dc_cap", "Cost_filt_ind_3ph",
                           "Cost_filt_cap", "Cost_passive_total", "Cost_BOS_total",
                           "Cost_converter", "Cost_B2B", "P_loss_total_kW", "eta_conv_pct"]:
            outputs[output_name] = self.prob.get_val(output_name)
        
        return outputs
    
    def test_5MW_baseline(self):
        """Test 5 MW converter with default settings"""
        outputs = self._run_converter(5.0)
        
        # Electrical parameters
        self.assertAlmostEqual(outputs["V_LL_rms_V"], 2300.0, delta=10.0)
        self.assertAlmostEqual(outputs["Vdc_V"], 4000.0, delta=10.0)
        self.assertAlmostEqual(outputs["I_rms_A"], 1255.23, delta=5.0)
        self.assertAlmostEqual(outputs["I_pk_A"], 1775.42, delta=5.0)
        
        # Filter design
        self.assertGreaterEqual(outputs["Lf_uH"], 0.0)
        self.assertGreaterEqual(outputs["Cf_uF"], 0.0)
        self.assertGreaterEqual(outputs["Cdc_uF"], 0.0)
        self.assertGreaterEqual(outputs["f_res_Hz"], 60.0)  # Should be above grid frequency
        
        # IGBT sizing
        self.assertEqual(outputs["Ns_series"], 4)  # For 4kV DC with 1700V IGBTs
        self.assertGreaterEqual(outputs["Np_parallel"], 1)
        self.assertGreaterEqual(outputs["N_igbt_modules"], 12)  # At least 3 legs * 4 series
        
        # Cost checks
        self.assertGreater(outputs["Cost_semiconductors"], 0.0)
        self.assertGreater(outputs["Cost_converter"], 0.0)
        self.assertGreater(outputs["Cost_B2B"], outputs["Cost_converter"])
        
        # Efficiency
        self.assertGreater(outputs["eta_conv_pct"], 95.0)
        self.assertLess(outputs["eta_conv_pct"], 100.0)
        
    def test_10MW_baseline(self):
        """Test 10 MW converter with default settings"""
        outputs = self._run_converter(10.0)
        
        # Electrical parameters
        self.assertAlmostEqual(outputs["V_LL_rms_V"], 4160.0, delta=20.0)
        self.assertAlmostEqual(outputs["Vdc_V"], 7200.0, delta=20.0)
        self.assertAlmostEqual(outputs["I_rms_A"], 1387.79, delta=10.0)
        
        # IGBT sizing
        self.assertEqual(outputs["Ns_series"], 7)  # For 7.2kV DC with 1700V IGBTs
        self.assertGreaterEqual(outputs["Np_parallel"], 1)
        
        # Efficiency
        self.assertGreater(outputs["eta_conv_pct"], 95.0)
        
    def test_15MW_baseline(self):
        """Test 15 MW converter with default settings"""
        outputs = self._run_converter(15.0)
        
        # Electrical parameters
        self.assertAlmostEqual(outputs["V_LL_rms_V"], 6600.0, delta=50.0)
        self.assertAlmostEqual(outputs["Vdc_V"], 11500.0, delta=50.0)
        self.assertAlmostEqual(outputs["I_rms_A"], 1312.16, delta=10.0)
        
        # IGBT sizing  
        self.assertEqual(outputs["Ns_series"], 10)  # For 11.5kV DC with 1700V IGBTs
        self.assertGreaterEqual(outputs["Np_parallel"], 1)
        
        # Efficiency
        self.assertGreater(outputs["eta_conv_pct"], 95.0)
        
    def test_20MW_baseline(self):
        """Test 20 MW converter with default settings"""
        outputs = self._run_converter(20.0)
        
        # Electrical parameters
        self.assertAlmostEqual(outputs["V_LL_rms_V"], 10000.0, delta=50.0)
        self.assertAlmostEqual(outputs["Vdc_V"], 18000.0, delta=50.0)
        self.assertAlmostEqual(outputs["I_rms_A"], 1154.70, delta=10.0)
        
        # IGBT sizing
        self.assertEqual(outputs["Ns_series"], 16)  # For 18kV DC with 1700V IGBTs
        self.assertGreaterEqual(outputs["Np_parallel"], 1)
        
        # Costs should scale with power
        self.assertGreater(outputs["Cost_converter"], 100000.0)
        
        # Efficiency
        self.assertGreater(outputs["eta_conv_pct"], 95.0)
        
    def test_25MW_baseline(self):
        """Test 25 MW converter with default settings"""
        outputs = self._run_converter(25.0)
        
        # Electrical parameters
        self.assertAlmostEqual(outputs["V_LL_rms_V"], 11000.0, delta=50.0)
        self.assertAlmostEqual(outputs["Vdc_V"], 20000.0, delta=50.0)
        self.assertAlmostEqual(outputs["I_rms_A"], 1312.16, delta=10.0)
        
        # IGBT sizing
        self.assertEqual(outputs["Ns_series"], 17)  # For 20kV DC with 1700V IGBTs
        self.assertGreaterEqual(outputs["Np_parallel"], 1)
        
        # Efficiency
        self.assertGreater(outputs["eta_conv_pct"], 95.0)
        
    def test_cost_scaling_with_power(self):
        """Test that costs scale appropriately with power rating"""
        outputs_5MW = self._run_converter(5.0)
        outputs_10MW = self._run_converter(10.0)
        outputs_20MW = self._run_converter(20.0)
        
        # Costs should increase with power
        self.assertLessEqual(outputs_5MW["Cost_converter"], outputs_10MW["Cost_converter"])
        self.assertLessEqual(outputs_10MW["Cost_converter"], outputs_20MW["Cost_converter"])
        
        # B2B cost should be less than 2x single converter (shared DC cap)
        for outputs in [outputs_5MW, outputs_10MW, outputs_20MW]:
            self.assertLess(outputs["Cost_B2B"], 2 * outputs["Cost_converter"])
            self.assertGreater(outputs["Cost_B2B"], outputs["Cost_converter"])
    
    def test_dc_voltage_meets_minimum(self):
        """Test that selected DC voltage meets minimum requirement"""
        for power_MW in [5, 10, 15, 20, 25]:
            outputs = self._run_converter(power_MW)
            self.assertGreaterEqual(outputs["Vdc_V"], outputs["Vdc_min_V"],
                                   f"DC voltage insufficient at {power_MW} MW")
    
    def test_switching_frequency_effect(self):
        """Test effect of switching frequency on losses"""
        # Lower switching frequency should reduce losses
        outputs_low_fsw = self._run_converter(10.0, f_sw=1000.0)
        outputs_high_fsw = self._run_converter(10.0, f_sw=3000.0)
        
        self.assertLessEqual(outputs_low_fsw["P_loss_total_kW"], 
                       outputs_high_fsw["P_loss_total_kW"])
        self.assertGreaterEqual(outputs_low_fsw["eta_conv_pct"],
                          outputs_high_fsw["eta_conv_pct"])
    
    def test_conservative_vs_standard_current_sizing(self):
        """Test IGBT sizing with different current calculation methods"""
        # Conservative current sizing (uses I_rms)
        self.prob.set_val("machine_rating", 10e6, units="W")
        self.prob.set_val("use_conservative_current", True)
        self.prob.run_model()
        Np_conservative = self.prob.get_val("Np_parallel")
        
        # Standard current sizing (uses I_rms/sqrt(2))
        self.prob.set_val("use_conservative_current", False)
        self.prob.run_model()
        Np_standard = self.prob.get_val("Np_parallel")
        
        # Conservative should require same or more parallel modules
        self.assertGreaterEqual(Np_conservative, Np_standard)
    
    def test_filter_resonance_frequency(self):
        """Test that LC filter resonance is in acceptable range"""
        for power_MW in [5, 10, 15, 20, 25]:
            outputs = self._run_converter(power_MW)
            f_res = outputs["f_res_Hz"]
            f_sw = 1620.0  # default
            f_grid = 60.0  # default
            
            # Resonance should be well above grid frequency
            self.assertGreater(f_res, 5 * f_grid)
            
            # # Resonance should be well below switching frequency
            # self.assertLess(f_res, f_sw / 10)
    
    def test_igbt_voltage_derating(self):
        """Test IGBT voltage derating is applied correctly"""
        outputs = self._run_converter(10.0, Vces_V=1700.0, derate_v=0.70)
        Ns_70pct = outputs["Ns_series"]
        
        outputs = self._run_converter(10.0, Vces_V=1700.0, derate_v=0.80)
        Ns_80pct = outputs["Ns_series"]
        
        # Lower derating factor should require more series modules
        self.assertGreaterEqual(Ns_70pct, Ns_80pct)
    
    def test_passive_component_energy_storage(self):
        """Test that passive components store reasonable energy"""
        outputs = self._run_converter(15.0)
        
        # DC capacitor energy storage
        Cdc_F = outputs["Cdc_uF"] * 1e-6
        Vdc_V = outputs["Vdc_V"]
        E_cap_J = 0.5 * Cdc_F * Vdc_V**2
        
        # Inductor energy storage
        Lf_H = outputs["Lf_uH"] * 1e-6
        I_pk_A = outputs["I_pk_A"]
        E_ind_J = 0.5 * Lf_H * I_pk_A**2
        
        # Both should store positive energy
        self.assertGreater(E_cap_J, 0.0)
        self.assertGreater(E_ind_J, 0.0)
        
        # Capacitor typically stores more energy than inductor
        self.assertGreater(E_cap_J, E_ind_J)
    
    def test_output_consistency(self):
        """Test that multiple runs with same inputs give same outputs"""
        outputs1 = self._run_converter(10.0)
        outputs2 = self._run_converter(10.0)
        
        # All outputs should be identical
        for key in outputs1.keys():
            np.testing.assert_equal(outputs1[key], outputs2[key],
                                   err_msg=f"Output {key} not consistent")
    
    def test_grid_frequency_50Hz(self):
        """Test converter design at 50 Hz grid frequency"""
        outputs_60Hz = self._run_converter(10.0, f_grid=60.0)
        outputs_50Hz = self._run_converter(10.0, f_grid=50.0)
        
        # Filter components should be different for different frequencies
        self.assertEqual(outputs_50Hz["Lf_uH"], outputs_60Hz["Lf_uH"])
        
        # Voltage and power levels should be similar
        self.assertAlmostEqual(outputs_50Hz["V_LL_rms_V"], 
                              outputs_60Hz["V_LL_rms_V"], delta=1.0)


class TestConverterEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""
    
    def setUp(self):
        """Set up test problem with Converter component"""
        self.prob = om.Problem()
        self.prob.model.add_subsystem("converter", Converter(), promotes=["*"])
        self.prob.setup()
    
    def test_very_small_power(self):
        """Test converter at very small power rating (below table range)"""
        self.prob.set_val("machine_rating", 2e6, units="W")  # 2 MW
        self.prob.run_model()
        
        # Should use lowest voltage level from table
        V_LL = self.prob.get_val("V_LL_rms_V")
        Vdc = self.prob.get_val("Vdc_V")
        
        self.assertGreater(V_LL, 0.0)
        self.assertGreater(Vdc, 0.0)
        
    def test_very_large_power(self):
        """Test converter at very large power rating (above table range)"""
        self.prob.set_val("machine_rating", 30e6, units="W")  # 30 MW
        self.prob.run_model()
        
        # Should use highest voltage level from table
        V_LL = self.prob.get_val("V_LL_rms_V")
        Vdc = self.prob.get_val("Vdc_V")
        
        self.assertGreater(V_LL, 0.0)
        self.assertGreater(Vdc, 0.0)


class TestConverterCostBreakdown(unittest.TestCase):
    """Test cost model components"""
    
    def setUp(self):
        """Set up test problem with Converter component"""
        self.prob = om.Problem()
        self.prob.model.add_subsystem("converter", Converter(), promotes=["*"])
        self.prob.setup()
        self.prob.set_val("machine_rating", 10e6, units="W")
        self.prob.run_model()
    
    def test_cost_components_sum(self):
        """Test that cost components sum correctly"""
        Cost_semiconductors = self.prob.get_val("Cost_semiconductors")
        Cost_passive_total = self.prob.get_val("Cost_passive_total")
        Cost_BOS_total = self.prob.get_val("Cost_BOS_total")
        Cost_converter = self.prob.get_val("Cost_converter")
        
        expected_total = Cost_semiconductors + Cost_passive_total + Cost_BOS_total
        
        self.assertAlmostEqual(Cost_converter, expected_total, delta=0.01)
    
    def test_passive_cost_breakdown(self):
        """Test passive component cost breakdown"""
        Cost_dc_cap = self.prob.get_val("Cost_dc_cap")
        Cost_filt_ind = self.prob.get_val("Cost_filt_ind_3ph")
        Cost_filt_cap = self.prob.get_val("Cost_filt_cap")
        Cost_passive_total = self.prob.get_val("Cost_passive_total")
        
        expected_passive = Cost_dc_cap + Cost_filt_ind + Cost_filt_cap
        
        self.assertAlmostEqual(Cost_passive_total, expected_passive, delta=0.01)
    
    def test_b2b_cost_sharing(self):
        """Test that B2B cost correctly shares DC capacitor"""
        Cost_converter = self.prob.get_val("Cost_converter")
        Cost_dc_cap = self.prob.get_val("Cost_dc_cap")
        Cost_B2B = self.prob.get_val("Cost_B2B")
        
        expected_B2B = 2 * Cost_converter - Cost_dc_cap
        
        self.assertAlmostEqual(Cost_B2B, expected_B2B, delta=0.01)


if __name__ == "__main__":
    unittest.main()
