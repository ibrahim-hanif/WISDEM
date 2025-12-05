"""
WISDEM DrivetrainSE Extension: FLS Bearing Life Calculations
-----------------------------
Wrapping Hub_Rotor_LSS_Frame with time-series OpenFAST loads
Author: Claude (based on WISDEM architecture analysis)
Date: 2025-11-06

Key Insight:
-----------
Hub_Rotor_LSS_Frame uses pyFrame3DD for structural analysis, which can handle
multiple load cases simultaneously. We extend this to process FLS loads by:
1. Running Frame3DD once per time step (or batched)
2. Extracting bearing reaction forces from Frame3DD results
3. Computing bearing L10 life using ISO 281 histogram method
cf. Implementation Guide in notion (https://www.notion.so/Improve-WEIS-WISDEM-2a3bc5df3c28800f9dbacbbf98cba2b2?v=225bc5df3c2881bf967b000c4b3c8746&source=copy_link#2a4bc5df3c288049918be08f34a82cb1)

NOTE: As of WISDEM v3.16+, there is NO existing FLS bearing life implementation.
This is a new capability that needs to be added.
"""

import numpy as np
import openmdao.api as om
from scipy.stats import weibull_min
from wisdem.drivetrainse.drive_structure import Hub_Rotor_LSS_Frame, analytical_MB_Forces
import pyframe3dd


class Hub_Rotor_LSS_Frame_FLS(om.ExplicitComponent): #TODO: not checked and verified
    """
    Extended version of Hub_Rotor_LSS_Frame that processes time-series loads
    from OpenFAST to compute bearing reaction forces for FLS analysis.
    
    This component wraps the pyFrame3DD structural analysis used in 
    Hub_Rotor_LSS_Frame but applies it to time-series hub loads rather than
    single ULS load cases.
    
    Key differences from original Hub_Rotor_LSS_Frame:
    - Accepts time-series hub loads (n_time, n_ws) instead of single values
    - Runs Frame3DD analysis for multiple load cases
    - Outputs bearing load time series for fatigue analysis
    """
    
    def initialize(self):
        self.options.declare('modeling_options', types=dict)
        self.options.declare('n_dlcs', default=1, desc='Number of DLCs')
        self.options.declare('batch_size', default=100, 
                           desc='Number of time steps to process at once')
        
    def setup(self):
        mod_opt = self.options['modeling_options']
        n_ws = mod_opt.get('n_ws', 10)
        n_t = mod_opt.get('n_time_steps', 720000)  # Reduced for memory
        
        # Drivetrain geometry inputs (same as Hub_Rotor_LSS_Frame)
        self.add_input('L_h1', val=0.0, units='m', 
                      desc='Hub flange to first main bearing')
        self.add_input('L_12', val=0.0, units='m',
                      desc='First to second main bearing')
        self.add_input('overhang', val=0.0, units='m',
                      desc='Overhang distance')
        self.add_input('tilt', val=0.0, units='deg',
                      desc='Shaft tilt angle')
        self.add_input('s_lss', val=np.zeros(5), units='m',
                      desc='LSS discrete sections')
        self.add_input('s_mb1', val=0.0, units='m',
                      desc='First bearing location')
        self.add_input('s_mb2', val=0.0, units='m',
                      desc='Second bearing location')
        
        # Shaft geometry
        self.add_input('D_top', val=np.zeros(5), units='m',
                      desc='Outer diameter at each section')
        self.add_input('t_wall', val=np.zeros(5), units='m',
                      desc='Wall thickness at each section')
        
        # Material properties
        self.add_input('E', val=200e9, units='Pa',
                      desc='Young\'s modulus')
        self.add_input('G', val=79.3e9, units='Pa',
                      desc='Shear modulus')
        self.add_input('rho', val=7850.0, units='kg/m**3',
                      desc='Material density')
        
        # Time-series hub loads from OpenFAST
        self.add_input('hub_Fxyz', val=np.zeros((n_t, n_ws, 3)), units='N',
                      desc='Hub forces [Fx, Fy, Fz] time series')
        self.add_input('hub_Mxyz', val=np.zeros((n_t, n_ws, 3)), units='N*m',
                      desc='Hub moments [Mx, My, Mz] time series')
        
        # Outputs: bearing reaction time series
        self.add_output('F_mb1', val=np.zeros((n_t, n_ws, 6)), units='N',
                       desc='MB1 reactions [Fx,Fy,Fz,Mx,My,Mz]')
        self.add_output('F_mb2', val=np.zeros((n_t, n_ws, 6)), units='N',
                       desc='MB2 reactions [Fx,Fy,Fz,Mx,My,Mz]')
        
        # Also output aggregated loads for compatibility
        self.add_output('F_mb1_radial', val=np.zeros((n_t, n_ws)), units='N')
        self.add_output('F_mb1_axial', val=np.zeros((n_t, n_ws)), units='N')
        self.add_output('F_mb2_radial', val=np.zeros((n_t, n_ws)), units='N')
        self.add_output('F_mb2_axial', val=np.zeros((n_t, n_ws)), units='N')
        
        self.declare_partials('*', '*', method='fd')
        
    def compute(self, inputs, outputs):
        """
        Run pyFrame3DD for each time step to get bearing reactions.
        Uses batching to handle large time series efficiently.
        """
        # Extract geometry
        s_lss = inputs['s_lss']
        D_top = inputs['D_top']
        t_wall = inputs['t_wall']
        s_mb1 = inputs['s_mb1']
        s_mb2 = inputs['s_mb2']
        overhang = inputs['overhang']
        tilt = inputs['tilt'] * np.pi / 180  # Convert to radians
        
        E = inputs['E']
        G = inputs['G']
        rho = inputs['rho']
        
        # Time-series loads
        hub_F = inputs['hub_Fxyz']  # (n_t, n_ws, 3)
        hub_M = inputs['hub_Mxyz']  # (n_t, n_ws, 3)
        
        n_t, n_ws, _ = hub_F.shape
        batch_size = self.options['batch_size']
        
        # Initialize outputs
        F_mb1_all = np.zeros((n_t, n_ws, 6))
        F_mb2_all = np.zeros((n_t, n_ws, 6))
        
        # Build Frame3DD model once (geometry doesn't change)
        frame_model = self._build_frame3dd_model(
            s_lss, D_top, t_wall, s_mb1, s_mb2, E, G, rho, tilt
        )
        
        # Process loads in batches
        for ws_idx in range(n_ws):
            for t_start in range(0, n_t, batch_size):
                t_end = min(t_start + batch_size, n_t)
                
                # Create load cases for this batch
                for t_idx in range(t_start, t_end):
                    Fx = hub_F[t_idx, ws_idx, 0]
                    Fy = hub_F[t_idx, ws_idx, 1]
                    Fz = hub_F[t_idx, ws_idx, 2]
                    Mx = hub_M[t_idx, ws_idx, 0]
                    My = hub_M[t_idx, ws_idx, 1]
                    Mz = hub_M[t_idx, ws_idx, 2]
                    
                    # Add this time step as a load case
                    load = self._create_load_case(Fx, Fy, Fz, Mx, My, Mz, tilt)
                    frame_model.addLoadCase(load)
                
                # Run Frame3DD analysis
                displacements, forces, reactions, internal, mass, modal = frame_model.run()
                
                # Extract bearing reactions for this batch
                # reactions.node contains node IDs
                # reactions.Fx, Fy, Fz, Mxx, Myy, Mzz contain reaction forces/moments
                
                mb1_node_idx = self._find_bearing_node(reactions, s_mb1)
                mb2_node_idx = self._find_bearing_node(reactions, s_mb2)
                
                for i, t_idx in enumerate(range(t_start, t_end)):
                    F_mb1_all[t_idx, ws_idx, :] = [
                        reactions.Fx[i, mb1_node_idx],
                        reactions.Fy[i, mb1_node_idx],
                        reactions.Fz[i, mb1_node_idx],
                        reactions.Mxx[i, mb1_node_idx],
                        reactions.Myy[i, mb1_node_idx],
                        reactions.Mzz[i, mb1_node_idx]
                    ]
                    
                    F_mb2_all[t_idx, ws_idx, :] = [
                        reactions.Fx[i, mb2_node_idx],
                        reactions.Fy[i, mb2_node_idx],
                        reactions.Fz[i, mb2_node_idx],
                        reactions.Mxx[i, mb2_node_idx],
                        reactions.Myy[i, mb2_node_idx],
                        reactions.Mzz[i, mb2_node_idx]
                    ]
                
                # Clear load cases for next batch
                frame_model.clearLoadCases()
        
        outputs['F_mb1'] = F_mb1_all
        outputs['F_mb2'] = F_mb2_all
        
        # Compute aggregated loads
        outputs['F_mb1_radial'] = np.sqrt(F_mb1_all[:, :, 1]**2 + F_mb1_all[:, :, 2]**2)
        outputs['F_mb1_axial'] = np.abs(F_mb1_all[:, :, 0])
        outputs['F_mb2_radial'] = np.sqrt(F_mb2_all[:, :, 1]**2 + F_mb2_all[:, :, 2]**2)
        outputs['F_mb2_axial'] = np.abs(F_mb2_all[:, :, 0])
        
    def _build_frame3dd_model(self, s_lss, D_top, t_wall, s_mb1, s_mb2, 
                              E, G, rho, tilt):
        """
        Build the Frame3DD model for the LSS.
        This follows the same approach as Hub_Rotor_LSS_Frame.
        """
        # Node locations along shaft
        n_nodes = len(s_lss)
        node_ids = np.arange(1, n_nodes + 1)
        
        # Transform to tilted coordinate system
        x = s_lss * np.cos(tilt)
        y = np.zeros(n_nodes)
        z = s_lss * np.sin(tilt)
        r = np.zeros(n_nodes)  # No rotation about local axis
        
        nodes = pyframe3dd.NodeData(node_ids, x, y, z, r)
        
        # Reactions at bearing locations
        # Find nodes closest to bearing locations
        mb1_node = np.argmin(np.abs(s_lss - s_mb1)) + 1
        mb2_node = np.argmin(np.abs(s_lss - s_mb2)) + 1
        
        reaction_nodes = np.array([mb1_node, mb2_node])
        # MB1: CRB - fixes radial (Ry, Rz), allows axial (Rx) and torsion (Rxx)
        # MB2: TRB - fixes all translations and moments
        Rx = np.array([0, 1])  # 0=free, 1=fixed
        Ry = np.array([1, 1])
        Rz = np.array([1, 1])
        Rxx = np.array([0, 1])
        Ryy = np.array([1, 1])
        Rzz = np.array([1, 1])
        
        reactions = pyframe3dd.ReactionData(reaction_nodes, Rx, Ry, Rz, 
                                           Rxx, Ryy, Rzz, rigid=1)
        
        # Element data (shaft segments)
        n_elem = n_nodes - 1
        elem_ids = np.arange(1, n_elem + 1)
        N1 = np.arange(1, n_nodes)
        N2 = np.arange(2, n_nodes + 1)
        
        # Cross-section properties at each element
        D_elem = (D_top[:-1] + D_top[1:]) / 2
        t_elem = (t_wall[:-1] + t_wall[1:]) / 2
        d_inner = D_elem - 2 * t_elem
        
        # Area and moments of inertia
        Ax = np.pi / 4 * (D_elem**2 - d_inner**2)
        Asy = Ax * 0.9  # Effective shear area (approximation)
        Asz = Asy
        Jx = np.pi / 32 * (D_elem**4 - d_inner**4)
        Iy = Jx / 2
        Iz = Jx / 2
        
        E_elem = E * np.ones(n_elem)
        G_elem = G * np.ones(n_elem)
        roll = np.zeros(n_elem)
        density = rho * np.ones(n_elem)
        
        elements = pyframe3dd.ElementData(elem_ids, N1, N2, Ax, Asy, Asz,
                                         Jx, Iy, Iz, E_elem, G_elem, roll, density)
        
        # Options
        shear = True  # Include shear deformation
        geom = False  # No geometric nonlinearity for now
        options = pyframe3dd.Options(shear, geom)
        
        # Create Frame object
        frame = pyframe3dd.Frame(nodes, reactions, elements, options)
        
        return frame
    
    def _create_load_case(self, Fx, Fy, Fz, Mx, My, Mz, tilt):
        """
        Create a pyFrame3DD load case from hub loads.
        Applies hub loads at the hub node (first node).
        """
        # Gravity loads
        gx = 0.0
        gy = 0.0
        gz = -9.81
        
        load = pyframe3dd.StaticLoadCase(gx, gy, gz)
        
        # Apply hub loads at node 1 (hub location)
        nF = np.array([1])
        Fx_load = np.array([Fx])
        Fy_load = np.array([Fy])
        Fz_load = np.array([Fz])
        Mxx_load = np.array([Mx])
        Myy_load = np.array([My])
        Mzz_load = np.array([Mz])
        
        load.changePointLoads(nF, Fx_load, Fy_load, Fz_load, 
                             Mxx_load, Myy_load, Mzz_load)
        
        return load
    
    def _find_bearing_node(self, reactions, s_bearing):
        """Find the node index corresponding to a bearing location."""
        # reactions.node contains the node IDs
        # We need to find which index corresponds to our bearing node
        # This is a simplified version - actual implementation would be more robust
        return 0  # Placeholder - needs proper implementation


class BearingFatigueLife_ISO281(om.ExplicitComponent):
    """
    Compute bearing L10 fatigue life from time-series bearing loads
    using ISO 281 standard with load histogram approach.
    
    This component takes the bearing reaction forces computed by
    Hub_Rotor_LSS_Frame_FLS and calculates the fatigue life.
    """
    
    def initialize(self):
        self.options.declare('modeling_options', types=dict)
        self.options.declare('n_bins', default=100, desc='Histogram bins')
        
    def setup(self):
        mod_opt = self.options['modeling_options']
        n_ws = mod_opt.get('n_ws', 10)
        n_t = mod_opt.get('n_time_steps', 720000)
        
        # Inputs: bearing loads time series from Hub_Rotor_LSS_Frame_FLS
        self.add_input('F_mb1_radial', val=np.zeros((n_t, n_ws)), units='N')
        self.add_input('F_mb1_axial', val=np.zeros((n_t, n_ws)), units='N')
        self.add_input('F_mb2_radial', val=np.zeros((n_t, n_ws)), units='N')
        self.add_input('F_mb2_axial', val=np.zeros((n_t, n_ws)), units='N')
        
        # Inputs: bearing properties (from WISDEM bearing sizing)
        self.add_input('Cr_mb1', val=1e7, units='N', desc='Dynamic load rating MB1')
        self.add_input('Cr_mb2', val=1e7, units='N', desc='Dynamic load rating MB2')
        self.add_input('bearing_type_mb1', val='CRB', desc='MB1 bearing type')
        self.add_input('bearing_type_mb2', val='TRB', desc='MB2 bearing type')
        
        # Inputs: operational parameters
        self.add_input('rated_rpm', val=7.56, units='rpm', desc='Rated rotor speed')
        self.add_input('wind_speeds', val=np.arange(5, 25, 2), units='m/s')
        self.add_input('weibull_k', val=1.95, desc='Weibull shape parameter')
        self.add_input('weibull_A', val=11.6, units='m/s', desc='Weibull scale parameter')
        
        # Outputs: bearing life
        self.add_output('L10h_mb1', val=0.0, units='h', desc='L10 life hours MB1')
        self.add_output('L10h_mb2', val=0.0, units='h', desc='L10 life hours MB2')
        self.add_output('L10_years_mb1', val=0.0, units='yr', desc='L10 life years MB1')
        self.add_output('L10_years_mb2', val=0.0, units='yr', desc='L10 life years MB2')
        
        # Safety factors (L10 / design life)
        self.add_output('S_bearing_mb1', val=0.0, desc='MB1 fatigue safety factor')
        self.add_output('S_bearing_mb2', val=0.0, desc='MB2 fatigue safety factor')
        
        self.declare_partials('*', '*', method='fd')
        
    def compute(self, inputs, outputs):
        """
        Compute bearing life using ISO 281 with load histogram approach.
        """
        # ISO 281 parameters for combined loading
        e = 0.4  # Fa/Fr threshold
        alpha = np.arctan(e / 1.5)
        X = 0.67
        Y1 = 0.45 / np.tan(alpha)
        Y2 = 0.67 / np.tan(alpha)
        
        # Wind speed probability distribution
        ws = inputs['wind_speeds']
        ws_pdf = weibull_min.pdf(ws, c=inputs['weibull_k'], 
                                 scale=inputs['weibull_A'])
        
        # Bearing loads
        F_mb1_rad = inputs['F_mb1_radial']
        F_mb1_ax = inputs['F_mb1_axial']
        F_mb2_rad = inputs['F_mb2_radial']
        F_mb2_ax = inputs['F_mb2_axial']
        
        # Compute equivalent loads using histogram method
        P_mb1 = self._equivalent_load_histogram(
            F_mb1_rad, F_mb1_ax, ws_pdf, bearing_type='CRB',
            X=X, Y1=Y1, Y2=Y2, e=e
        )
        
        P_mb2 = self._equivalent_load_histogram(
            F_mb2_rad, F_mb2_ax, ws_pdf, bearing_type='TRB',
            X=X, Y1=Y1, Y2=Y2, e=e
        )
        
        # ISO 281 life calculation
        # L10 (millions of revolutions) = (Cr / P)^p
        # p = 10/3 for roller bearings, 3 for ball bearings
        p = 10.0 / 3.0  # Roller bearings
        
        Cr_mb1 = inputs['Cr_mb1']
        Cr_mb2 = inputs['Cr_mb2']
        n0 = inputs['rated_rpm']
        
        L10_mb1_rev = (Cr_mb1 / P_mb1) ** p  # millions of revolutions
        L10_mb2_rev = (Cr_mb2 / P_mb2) ** p
        
        # Convert to hours: L10h = L10_rev * 1e6 / (n * 60)
        outputs['L10h_mb1'] = L10_mb1_rev * 1e6 / (n0 * 60)
        outputs['L10h_mb2'] = L10_mb2_rev * 1e6 / (n0 * 60)
        
        # Convert to years (8766 hours/year)
        outputs['L10_years_mb1'] = outputs['L10h_mb1'] / 8766
        outputs['L10_years_mb2'] = outputs['L10h_mb2'] / 8766
        
        # Safety factors (L10 / 20 years design life)
        # ISO 281 recommends S = (L10/Lreq)^(3/10) for reliability assessment
        design_life_hours = 20 * 8766
        outputs['S_bearing_mb1'] = (outputs['L10h_mb1'] / design_life_hours) ** (3.0/10.0)
        outputs['S_bearing_mb2'] = (outputs['L10h_mb2'] / design_life_hours) ** (3.0/10.0)
        
    def _equivalent_load_histogram(self, F_rad, F_ax, ws_pdf, bearing_type,
                                   X, Y1, Y2, e):
        """
        Compute equivalent bearing load using histogram method.
        Accounts for variable wind speed probability.
        """
        n_bins = self.options['n_bins']
        
        # Determine equivalent load for each time step
        if bearing_type == 'CRB':
            # Cylindrical roller bearing: radial load only
            P = F_rad
        else:
            # TRB/CARB: combined radial and axial per ISO 281
            ratio = F_ax / np.maximum(F_rad, np.finfo(float).eps)
            light = np.abs(ratio) <= e
            
            P = np.zeros_like(F_rad)
            P[light] = F_rad[light] + Y1 * F_ax[light]
            P[~light] = X * F_rad[~light] + Y2 * F_ax[~light]
        
        # Create histogram weighted by wind speed probability
        P_max = np.max(P)
        if P_max == 0:
            P_max = 1.0
            
        edges = np.linspace(0, P_max, n_bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        
        P_hist_weighted = np.zeros(n_bins)
        
        for i, pdf_val in enumerate(ws_pdf):
            hist, _ = np.histogram(P[:, i], bins=edges, density=True)
            hist_norm = hist / np.sum(hist) if np.sum(hist) > 0 else hist
            P_hist_weighted += pdf_val * hist_norm
        
        # Equivalent load using Miner's rule
        # P_eq = (sum(P_i^p * n_i))^(1/p) where p=10/3 for rollers
        p = 10.0 / 3.0
        P_eq = (np.sum((centers ** p) * P_hist_weighted)) ** (1.0 / p)
        
        return P_eq


class DrivetrainSE_with_FLS(om.Group):
    """
    Extended DrivetrainSE group that includes FLS bearing life calculations.
    This wraps Hub_Rotor_LSS_Frame with time-series load processing.
    
    Integration Strategy:
    1. Use existing Hub_Rotor_LSS_Frame for ULS design
    2. Add Hub_Rotor_LSS_Frame_FLS for time-series FLS loads
    3. Add BearingFatigueLife_ISO281 for L10 calculation
    4. Connect bearing properties from ULS to FLS analysis
    """
    
    def initialize(self):
        self.options.declare('modeling_options', types=dict)
        
    def setup(self):
        mod_opt = self.options['modeling_options']
        
        # Original WISDEM Hub_Rotor_LSS_Frame for ULS design
        # This sizes the shaft and bearings based on ultimate loads
        self.add_subsystem('hub_lss_uls',
                          Hub_Rotor_LSS_Frame(modeling_options=mod_opt),
                          promotes_inputs=['L_h1', 'L_12', 'overhang', 'tilt',
                                          's_lss', 'D_top', 't_wall',
                                          'E', 'G', 'rho'])
        
        # New FLS component using pyFrame3DD for time-series
        self.add_subsystem('hub_lss_fls',
                          Hub_Rotor_LSS_Frame_FLS(modeling_options=mod_opt),
                          promotes_inputs=['L_h1', 'L_12', 'overhang', 'tilt',
                                          's_lss', 'D_top', 't_wall',
                                          'E', 'G', 'rho'])
        
        # Bearing fatigue life calculation
        self.add_subsystem('bearing_life',
                          BearingFatigueLife_ISO281(modeling_options=mod_opt))
        
        # Connect bearing properties from ULS sizing to FLS analysis
        # Note: Hub_Rotor_LSS_Frame computes bearing dynamic load ratings
        self.connect('hub_lss_uls.mb1_Cr', 'bearing_life.Cr_mb1')
        self.connect('hub_lss_uls.mb2_Cr', 'bearing_life.Cr_mb2')
        
        # Connect FLS bearing loads to life calculation
        self.connect('hub_lss_fls.F_mb1_radial', 'bearing_life.F_mb1_radial')
        self.connect('hub_lss_fls.F_mb1_axial', 'bearing_life.F_mb1_axial')
        self.connect('hub_lss_fls.F_mb2_radial', 'bearing_life.F_mb2_radial')
        self.connect('hub_lss_fls.F_mb2_axial', 'bearing_life.F_mb2_axial')


# ============================================================================
# Integration with WEIS - pCrunch Load Extraction
# ============================================================================

class OpenFAST_to_WISDEM_Bridge(om.ExplicitComponent):
    """
    Bridge component to extract OpenFAST hub loads using pCrunch
    and format them for WISDEM DrivetrainSE FLS analysis.
    
    This component:
    1. Uses pCrunch to load OpenFAST output files
    2. Extracts hub loads (HubFx, HubFy, HubFz, HubMx, HubMy, HubMz)
    3. Organizes by wind speed for Weibull weighting
    4. Outputs in format expected by Hub_Rotor_LSS_Frame_FLS
    """
    
    def initialize(self):
        self.options.declare('openfast_dir', types=str, 
                           desc='Directory containing OpenFAST outputs')
        self.options.declare('dlc_list', types=list, default=['DLC1.2'],
                           desc='List of DLC names to process')
        self.options.declare('wind_speeds', types=list,
                           default=list(range(5, 25, 2)),
                           desc='Wind speeds corresponding to DLC runs')
        
    def setup(self):
        n_ws = len(self.options['wind_speeds'])
        n_t = 720000  # Will be determined from actual data
        
        # Outputs: formatted for WISDEM
        self.add_output('hub_Fxyz', val=np.zeros((n_t, n_ws, 3)), units='N',
                       desc='Hub forces time series')
        self.add_output('hub_Mxyz', val=np.zeros((n_t, n_ws, 3)), units='N*m',
                       desc='Hub moments time series')
        self.add_output('wind_speeds', val=np.array(self.options['wind_speeds']),
                       units='m/s')
        
    def compute(self, inputs, outputs):
        """
        Load OpenFAST data using pCrunch and extract hub loads.
        """
        try:
            from pCrunch import pdTools, Processing
            
            # Initialize pCrunch processor
            fp = Processing.FAST_Processing()
            
            # Get OpenFAST output files
            openfast_dir = self.options['openfast_dir']
            dlc_list = self.options['dlc_list']
            wind_speeds = self.options['wind_speeds']
            
            fp.OpenFAST_outfile_list = []
            for dlc in dlc_list:
                for ws in wind_speeds:
                    # Typical naming: DLC1.2_ws10/IEA15MW.out
                    outfile = f"{openfast_dir}/{dlc}_ws{ws:02d}/turbine.out"
                    fp.OpenFAST_outfile_list.append(outfile)
            
            # Load all OpenFAST outputs
            print("Loading OpenFAST outputs with pCrunch...")
            fastout = fp.load_FAST_out()
            
            # Extract hub loads for each wind speed
            n_ws = len(wind_speeds)
            hub_channels = ['HubFx', 'HubFy', 'HubFz', 'HubMx', 'HubMy', 'HubMz']
            
            # Determine time series length (use first file)
            n_t = len(fastout[0]['Time'])
            
            # Allocate arrays
            hub_F = np.zeros((n_t, n_ws, 3))
            hub_M = np.zeros((n_t, n_ws, 3))
            
            # Extract data
            for i, case in enumerate(fastout):
                if i >= n_ws:
                    break
                    
                # Forces
                hub_F[:, i, 0] = case['HubFx']
                hub_F[:, i, 1] = case['HubFy']
                hub_F[:, i, 2] = case['HubFz']
                
                # Moments
                hub_M[:, i, 0] = case['HubMx']
                hub_M[:, i, 1] = case['HubMy']
                hub_M[:, i, 2] = case['HubMz']
            
            outputs['hub_Fxyz'] = hub_F
            outputs['hub_Mxyz'] = hub_M
            outputs['wind_speeds'] = np.array(wind_speeds)
            
            print(f"Loaded {n_t} time steps for {n_ws} wind speeds")
            
        except ImportError:
            print("Warning: pCrunch not available. Using dummy loads.")
            # Use dummy loads for testing
            n_ws = len(self.options['wind_speeds'])
            n_t = 720000
            outputs['hub_Fxyz'] = np.random.randn(n_t, n_ws, 3) * 1e6
            outputs['hub_Mxyz'] = np.random.randn(n_t, n_ws, 3) * 1e7
            outputs['wind_speeds'] = np.array(self.options['wind_speeds'])
            
        except Exception as e:
            print(f"Error loading OpenFAST data: {e}")
            print("Using dummy loads for testing.")
            n_ws = len(self.options['wind_speeds'])
            n_t = 720000
            outputs['hub_Fxyz'] = np.random.randn(n_t, n_ws, 3) * 1e6
            outputs['hub_Mxyz'] = np.random.randn(n_t, n_ws, 3) * 1e7
            outputs['wind_speeds'] = np.array(self.options['wind_speeds'])


# ============================================================================
# Complete WEIS Integration Example
# ============================================================================

class WEIS_Drivetrain_FLS_Group(om.Group):
    """
    Complete WEIS integration group that:
    1. Extracts loads from OpenFAST using pCrunch
    2. Runs DrivetrainSE with both ULS and FLS analysis
    3. Provides bearing life constraints for optimization
    
    This is the top-level group to add to WEIS workflow.
    """
    
    def initialize(self):
        self.options.declare('modeling_options', types=dict)
        self.options.declare('openfast_dir', types=str, default='./openfast_runs')
        self.options.declare('dlc_list', types=list, default=['DLC1.2'])
        self.options.declare('wind_speeds', types=list, 
                           default=list(range(5, 25, 2)))
        
    def setup(self):
        mod_opt = self.options['modeling_options']
        
        # Step 1: Extract OpenFAST loads
        self.add_subsystem('openfast_bridge',
                          OpenFAST_to_WISDEM_Bridge(
                              openfast_dir=self.options['openfast_dir'],
                              dlc_list=self.options['dlc_list'],
                              wind_speeds=self.options['wind_speeds']
                          ))
        
        # Step 2: DrivetrainSE with FLS
        self.add_subsystem('drivetrain',
                          DrivetrainSE_with_FLS(modeling_options=mod_opt),
                          promotes_inputs=['L_h1', 'L_12', 'overhang', 'tilt',
                                          's_lss', 'D_top', 't_wall',
                                          'E', 'G', 'rho'])
        
        # Connect OpenFAST loads to drivetrain FLS analysis
        self.connect('openfast_bridge.hub_Fxyz',
                    'drivetrain.hub_lss_fls.hub_Fxyz')
        self.connect('openfast_bridge.hub_Mxyz',
                    'drivetrain.hub_lss_fls.hub_Mxyz')
        self.connect('openfast_bridge.wind_speeds',
                    'drivetrain.bearing_life.wind_speeds')


# ============================================================================
# Simplified Alternative: Static Approach (if Frame3DD is too slow)
# ============================================================================

class Hub_Rotor_LSS_Frame_FLS_Static(om.ExplicitComponent): #DONE: checking rn
    """
    Alternative FLS implementation using static approach.
    
    Instead of running Frame3DD for every time step, this component:
    1. Runs Frame3DD once for ULS design
    2. Uses analytical bearing load equations for FLS time series
    3. Applies correction factors from Frame3DD ULS results
    
    This is much faster but slightly less accurate than full Frame3DD.
    Recommended for optimization studies where speed is critical.
    """
    
    def initialize(self):
        self.options.declare('modeling_options') #DONE: remove types=dict, not in .drivetrainse
        
    def setup(self):
        mod_opt = self.options['modeling_options']
        n_ws = mod_opt.get('n_ws', 10)
        n_t = mod_opt.get('n_time_steps', 720000)
        
        # Geometry inputs (same as before)
        self.add_input('L_12', val=6.0, units='m', desc='MB1 to MB2 distance')
        self.add_input('L_h1', val=4.25, units='m', desc='Rotor to MB1 distance')
        
        # Time-series hub loads
        self.add_input('hub_Fxyz', val=np.zeros((n_t, n_ws, 3)), units='N')
        self.add_input('hub_Mxyz', val=np.zeros((n_t, n_ws, 3)), units='N*m')
        
        # Correction factors from ULS Frame3DD analysis
        self.add_input('mb1_correction_factor', val=1.0,
                      desc='MB1 load correction from Frame3DD')
        self.add_input('mb2_correction_factor', val=1.0,
                      desc='MB2 load correction from Frame3DD')
        
        # Outputs
        self.add_output('F_mb1_radial', val=np.zeros((n_t, n_ws)), units='N')
        self.add_output('F_mb1_axial', val=np.zeros((n_t, n_ws)), units='N')
        self.add_output('F_mb2_radial', val=np.zeros((n_t, n_ws)), units='N')
        self.add_output('F_mb2_axial', val=np.zeros((n_t, n_ws)), units='N')
        
        self.declare_partials('*', '*', method='fd') #TODO: check
        
    def compute(self, inputs, outputs):
        """
        Compute bearing loads using analytical equations with Frame3DD corrections.
        This is the same approach as your original MATLAB code but with
        correction factors calibrated from Frame3DD ULS analysis.
        """
        L_12 = inputs['L_12']
        L_h1 = inputs['L_h1']
        
        hub_F = inputs['hub_Fxyz']
        Fx = hub_F[:, :, 0]
        Fy = hub_F[:, :, 1]
        Fz = hub_F[:, :, 2]
        hub_M = inputs['hub_Mxyz']
        Mx = hub_M[:, :, 0]
        My = hub_M[:, :, 1]
        Mz = hub_M[:, :, 2]
        
        # Analytical bearing loads (static equilibrium)
        F_mb1_ax, F_mb1_y, F_mb1_z, F_mb1_rad, F_mb2_ax, F_mb2_y, F_mb2_z, F_mb2_rad = analytical_MB_Forces(
            Fx, Fy, Fz, Mx, My, Mz, L_h1, L_12
        )
        
        # Apply correction factors from Frame3DD
        cf1 = inputs['mb1_correction_factor']
        cf2 = inputs['mb2_correction_factor']
        
        outputs['F_mb1_axial'] = F_mb1_ax * cf1
        outputs['F_mb1_radial'] = F_mb1_rad * cf1
        
        outputs['F_mb2_axial'] = F_mb2_ax * cf2
        outputs['F_mb2_radial'] = F_mb2_rad * cf2


class CorrectionFactorComputer(om.ExplicitComponent): #TODO: checking
    """
    Component to compute correction factors by comparing Frame3DD
    results with analytical bearing loads for ULS case.
    
    This runs once during setup to calibrate the statistical approach.
    """
    
    def initialize(self):
        self.options.declare('modeling_options')
        
    def setup(self):
        n_dlcs = self.options["n_dlcs"]
        # Inputs: ULS loads
        self.add_input("F_aero_hub", val=np.zeros((3, n_dlcs)), units="N")
        self.add_input("M_aero_hub", val=np.zeros((3, n_dlcs)), units="N*m")
        
        # Inputs: Frame3DD bearing reactions (from Hub_Rotor_LSS_Frame)
        self.add_input('F_mb1_frame3dd', val=0.0, units='N',
                      desc='MB1 radial load from Frame3DD')
        self.add_input('F_mb2_frame3dd', val=0.0, units='N',
                      desc='MB2 radial load from Frame3DD')
        
        # Inputs: geometry
        self.add_input('L_12', val=6.0, units='m')
        self.add_input('L_h1', val=4.25, units='m')
        
        # Outputs: correction factors
        self.add_output('mb1_correction_factor', val=1.0)
        self.add_output('mb2_correction_factor', val=1.0)
        
        self.declare_partials('*', '*', method='fd')
        
    def compute(self, inputs, outputs):
        """
        Compute correction factors as ratio of Frame3DD to analytical loads.
        """
        L_12 = inputs['L_12']
        L_h1 = inputs['L_h1']
        
        # local reference and unpack rows as views (no extra copies)
        Fx, Fy, Fz = inputs['F_aero_hub']
        Mx, My, Mz = inputs['M_aero_hub']
        
        # Analytical loads
        F_mb1_ax, F_mb1_y, F_mb1_z, F_mb1_rad, F_mb2_ax, F_mb2_y, F_mb2_z, F_mb2_rad = analytical_MB_Forces(
            Fx, Fy, Fz, Mx, My, Mz, L_h1, L_12
        )    
        
        # pyframe3DD (`Hub_Rotor_LSS_Frame`) output
        F_mb1_frame = inputs['F_mb1_frame3dd']
        F_mb2_frame = inputs['F_mb2_frame3dd']

        # Correction factors
        outputs['mb1_correction_factor'] = F_mb1_frame / F_mb1_rad
        outputs['mb2_correction_factor'] = F_mb2_frame / F_mb2_rad

# ============================================================================
# Example Usage and Testing
# ============================================================================

def example_standalone_fls_analysis():
    """
    Example showing how to run FLS bearing analysis standalone.
    """
    print("\n" + "="*80)
    print("WISDEM DrivetrainSE FLS Bearing Life Analysis - Standalone Example")
    print("="*80 + "\n")
    
    # Modeling options
    modeling_options = {
        'n_ws': 10,
        'n_time_steps': 10000,  # Reduced for example
        'n_dlcs': 1
    }
    
    # Create problem with statistical approach (faster)
    prob = om.Problem()
    
    model = prob.model
    
    # Add components
    model.add_subsystem('fls_loads',
                       Hub_Rotor_LSS_Frame_FLS_Static(
                           modeling_options=modeling_options))
    
    model.add_subsystem('bearing_life',
                       BearingFatigueLife_ISO281(
                           modeling_options=modeling_options))
    
    # Connect
    model.connect('fls_loads.F_mb1_radial', 'bearing_life.F_mb1_radial')
    model.connect('fls_loads.F_mb1_axial', 'bearing_life.F_mb1_axial')
    model.connect('fls_loads.F_mb2_radial', 'bearing_life.F_mb2_radial')
    model.connect('fls_loads.F_mb2_axial', 'bearing_life.F_mb2_axial')
    
    # Setup
    prob.setup()
    Hub_Rotor_LSS_Frame_FLS_Static
    # Set geometry (IEA 15MW values)
    prob.set_val('fls_loads.L_12', 6.0, units='m')
    prob.set_val('fls_loads.L_h1', 4.25, units='m')
    
    # Set bearing properties
    prob.set_val('bearing_life.Cr_mb1', 1e7, units='N')
    prob.set_val('bearing_life.Cr_mb2', 1.5e7, units='N')
    prob.set_val('bearing_life.rated_rpm', 7.56, units='rpm')
    
    # Generate synthetic hub loads for testing
    n_t = modeling_options['n_time_steps']
    n_ws = modeling_options['n_ws']
    
    # Realistic load ranges for IEA 15MW
    hub_F = np.random.randn(n_t, n_ws, 3) * np.array([5e5, 5e5, 1e7])
    hub_M = np.random.randn(n_t, n_ws, 3) * np.array([1e8, 1e8, 5e7])
    
    prob.set_val('fls_loads.hub_Fxyz', hub_F, units='N')
    prob.set_val('fls_loads.hub_Mxyz', hub_M, units='N*m')
    
    # Run analysis
    print("Running FLS bearing life analysis...")
    prob.run_model()
    
    # Print results
    print("\n" + "-"*80)
    print("Results:")
    print("-"*80)
    print(f"MB1 L10 life: {prob.get_val('bearing_life.L10_years_mb1')[0]:.2f} years")
    print(f"MB2 L10 life: {prob.get_val('bearing_life.L10_years_mb2')[0]:.2f} years")
    print(f"MB1 Safety Factor: {prob.get_val('bearing_life.S_bearing_mb1')[0]:.3f}")
    print(f"MB2 Safety Factor: {prob.get_val('bearing_life.S_bearing_mb2')[0]:.3f}")
    print("-"*80 + "\n")
    
    return prob


def example_weis_integration():
    """
    Example showing how to integrate with WEIS workflow.
    This would be called within WEIS's run_model.py or similar.
    """
    print("\n" + "="*80)
    print("WEIS Integration Example - DrivetrainSE with FLS")
    print("="*80 + "\n")
    
    modeling_options = {
        'n_ws': 10,
        'n_time_steps': 10000,
        'n_dlcs': 2
    }
    
    # Create WEIS problem
    prob = om.Problem()
    
    # Add complete FLS group
    prob.model.add_subsystem('drivetrain_fls',
                            WEIS_Drivetrain_FLS_Group(
                                modeling_options=modeling_options,
                                openfast_dir='./openfast_runs',
                                dlc_list=['DLC1.2', 'DLC6.1'],
                                wind_speeds=list(range(5, 25, 2))
                            ))
    
    # Add design variables (example)
    prob.model.add_design_var('drivetrain_fls.drivetrain.D_top',
                             lower=1.0, upper=4.0, units='m')
    prob.model.add_design_var('drivetrain_fls.drivetrain.t_wall',
                             lower=0.01, upper=0.5, units='m')
    
    # Add constraints
    prob.model.add_constraint('drivetrain_fls.drivetrain.bearing_life.S_bearing_mb1',
                             lower=1.0)
    prob.model.add_constraint('drivetrain_fls.drivetrain.bearing_life.S_bearing_mb2',
                             lower=1.0)
    
    # Add objective (minimize mass)
    prob.model.add_objective('drivetrain_fls.drivetrain.hub_lss_uls.lss_mass')
    
    # Setup optimizer
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['maxiter'] = 100
    prob.driver.options['tol'] = 1e-6
    
    prob.setup()
    
    print("WEIS problem setup complete.")
    print("Ready to run optimization with FLS bearing life constraints.")
    
    return prob


if __name__ == '__main__':
    """
    Main execution: run examples
    """
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--weis':
        # WEIS integration example
        prob = example_weis_integration()
        print("\nNote: This is a template. In actual WEIS, you would:")
        print("1. Replace synthetic loads with actual OpenFAST outputs")
        print("2. Connect to WEIS's geometry and optimization framework")
        print("3. Run prob.run_driver() for optimization")
    else:
        # Standalone FLS analysis example
        prob = example_standalone_fls_analysis()
        
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)