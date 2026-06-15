"""Provides WISDEM's WOMBAT API."""

import openmdao.api as om
from wombat import Simulation
from wombat.core.library import DEFAULT_DATA, load_yaml, load_weather


class Wombat(om.Group):
    """WOMBAT simulation API class for WISDEM API."""

    def initialize(self):
        """Initializes the API connections."""
        self.options.declare("scenario", default="osw-fixed")  # NOTE: config file without the extension

    def setup(self):
        """Define all input variables from all models."""

        # TODO: Should the random seed or generator be provided to the interface?
        self.set_input_defaults("years", 20, units="yr")
        self.set_input_defaults("equipment_dispatch_distance", 50, units="km")
        self.set_input_defaults("repair_port_distance", 116, units="km")
        self.set_input_defaults("project_capacity", None, units="MW")
        self.set_input_defaults("turbine_capex_kw", None, units="USD/kW")
        self.set_input_defaults("turbine_capacity", None, units="MW")

        self.add_subsystem(
            "wombat",
            WombatWisdem(
                scenario=self.options["scenario"],
            ),
            promotes=["*"],
        )


class WombatWisdem(om.ExplicitComponent):
    """WOMBAT-WISDEM Fixed Substructure API."""

    def initialize(self):
        """Initialize the API."""
        self.options.declare("scenario", default="osw-fixed")

    def load_scenario_config(self) -> dict:
        scenario = self.options["scenario"]
        if scenario == "osw-fixed":
            config = load_yaml(DEFAULT_DATA / "project/config", "base_osw_fixed.yaml")

            config["vessels"] = {
                "ctv": load_yaml(DEFAULT_DATA / "vessels", "ctv.yaml"),
                "hlv": load_yaml(DEFAULT_DATA / "vessels", "hlv.yaml"),
                "cab": load_yaml(DEFAULT_DATA / "vessels", "cab.yaml"),
                "dsv": load_yaml(DEFAULT_DATA / "vessels", "dsv.yaml"),
            }
            config["fixed_costs"] = load_yaml(DEFAULT_DATA / "project/config", "fixed_costs_osw_fixed.yaml")
            config["substations"] = {"base_substation": load_yaml(DEFAULT_DATA / "substations", "osw_substation.yaml")}
            config["cables"] = {
                "base_array": load_yaml(DEFAULT_DATA / "cables", "osw_array.yaml"),
                "base_export": load_yaml(DEFAULT_DATA / "cables", "osw_export.yaml"),
            }
            config["turbines"] = {"base_turbine": load_yaml(DEFAULT_DATA / "turbines", "12MW_osw_fixed.yaml")}
            config["weather"] = load_weather(DEFAULT_DATA / "weather/era5_40.0N_72.5W_1990_2020.pqt")[
                ["datetime", "windspeed", "waveheight"]
            ]
            config["end_year"] = 2019
        elif scenario == "osw-floating":
            config = load_yaml(DEFAULT_DATA / "project/config", "base_osw_floating.yaml")
            config["vessels"] = {
                "ctv": load_yaml(DEFAULT_DATA / "vessels", "ctv.yaml"),
                "cab": load_yaml(DEFAULT_DATA / "vessels", "cab.yaml"),
                "dsv": load_yaml(DEFAULT_DATA / "vessels", "dsv.yaml"),
                "tugboat": load_yaml(DEFAULT_DATA / "vessels", "tugboat.yaml"),
            }
            config["fixed_costs"] = load_yaml(DEFAULT_DATA / "project/config", "fixed_costs_osw_floating.yaml")
            config["substations"] = {"base_substation": load_yaml(DEFAULT_DATA / "substations", "osw_substation.yaml")}
            config["cables"] = {
                "base_array": load_yaml(DEFAULT_DATA / "cables", "osw_array.yaml"),
                "base_export": load_yaml(DEFAULT_DATA / "cables", "osw_export.yaml"),
            }
            config["turbines"] = {"base_turbine": load_yaml(DEFAULT_DATA / "turbines", "12MW_osw_floating.yaml")}
            config["port"] = load_yaml(DEFAULT_DATA / "project/port", "base_port.yaml")
            config["weather"] = load_weather(DEFAULT_DATA / "weather/era5_41.0N_125.0W_1989_2019.pqt")[
                ["datetime", "windspeed", "waveheight"]
            ]
            config["end_year"] = 2019
        elif scenario == "lbw":
            config = load_yaml(DEFAULT_DATA / "project/config", "base_lbw.yaml")
            config["vessels"] = {
                "truck": load_yaml(DEFAULT_DATA / "vessels", "truck.yaml"),
                "crawler": load_yaml(DEFAULT_DATA / "vessels", "crawler_large.yaml"),
            }
            config["fixed_costs"] = load_yaml(DEFAULT_DATA / "project/config", "fixed_costs_lbw.yaml")
            config["substations"] = {"base_substation": load_yaml(DEFAULT_DATA / "substations", "lbw_substation.yaml")}
            config["cables"] = {
                "base_array": load_yaml(DEFAULT_DATA / "cables", "lbw_array.yaml"),
                "base_export": load_yaml(DEFAULT_DATA / "cables", "lbw_export.yaml"),
            }
            config["turbines"] = {"base_turbine": load_yaml(DEFAULT_DATA / "turbines", "3.5MW_lbw.yaml")}
            config["weather"] = load_weather(DEFAULT_DATA / "weather/merra2_32.5N_-100.625W_1980_2024.pqt")[
                ["datetime", "windspeed", "waveheight"]
            ]
            config["end_year"] = 2019
        else:
            msg = f"{scenario=} is not implemented, use one of 'lbw', 'osw-fixed', or 'osw-floating'."
            raise NotImplementedError(msg)

        config["name"] = "wisdem_wombat"
        config["layout_coords"] = "distance"
        return config

    def setup(self):
        """Define all the inputs."""

        self._wombat_config = self.load_scenario_config()

        self.add_input(
            "years",
            20,
            units="yr",
            desc="Number of years to simulation the operations and maintenance phase of the farm lifecycle",
        )
        self.add_discrete_input("workday_start", 7, desc="Hour of the day where any work-related activities begin")
        self.add_discrete_input("workday_end", 19, desc="Hour of the day where any work-related activities end")
        self.add_input(
            "equipment_dispatch_distance",
            50,
            units="km",
            desc="Distance, in km, that servicing equipment must travel daily to reach the wind farm",
        )
        self.add_discrete_input(
            "n_ctv",
            3,
            desc="Number of crew transfer vessels (offshore) or onsite trucks (land-based) that should be made available to the wind farm.",
        )
        self.add_discrete_input(
            "n_hlv",
            1,
            desc="Number of heavy lift vessels (fixed-bottom offshore) or crawler cranes (land-based) that should be made available to the wind farm (fixed-bottom simulations only)",
        )
        self.add_discrete_input(
            "n_tugboat",
            2,
            desc="Number of tugboat groups that should be available to the port to tow floating turbines to port and back",
        )
        self.add_discrete_input(
            "port_workday_start",
            6,
            desc="Hour of the day where any work-related activities begin for port-side repairs",
        )
        self.add_discrete_input(
            "port_workday_end", 18, desc="Hour of the day where any work-related activities end for port-side repairs"
        )
        self.add_discrete_input(
            "n_port_crews",
            2,
            desc="Number of port-side crews available to work on simultaneous repairs for any at-port turbine",
        )
        self.add_discrete_input("max_port_operations", 2, desc="Number of turbines that can be at port at once")
        self.add_input(
            "repair_port_distance",
            116,
            units="km",
            desc="Distance, in km, that tugboats must travel to reach the wind farm for tow-to-port repairs",
        )
        self.add_discrete_input(
            "maintenance_start",
            None,
            desc="Date of first maintenance event to determine regular interval timing. Can be set to prior to the starting year to ensure staggered starts.",
        )
        self.add_discrete_input(
            "non_operational_start",
            None,
            desc="Starting date, in MM/DD format, for an annual period where the site is inaccessible",
        )
        self.add_discrete_input(
            "non_operational_end",
            None,
            desc="Ending date, in MM/DD format, for an annual period where the site is inaccessible",
        )
        self.add_discrete_input(
            "reduced_speed_start",
            None,
            desc="Starting date, in MM/DD format, for an annual period where traveling speed is reduced",
        )
        self.add_discrete_input(
            "reduced_speed_end",
            None,
            desc="Ending date, in MM/DD format, for an annual period where traveling speed is reduced",
        )
        self.add_input(
            "reduced_speed",
            0,
            units="km/h",
            desc="Reduced speed applied to servicing equipment in the reduced speed period",
        )
        self.add_input("project_capacity", 0, units="MW", desc="Total wind farm capacity")
        self.add_input("turbine_capex_kw", 0, units="USD/kW", desc="Turbine CapEx per kW of nameplate capacity")
        self.add_input("turbine_capacity", 0, units="W", desc="Turbine nameplate capacity")
        self.add_discrete_input("random_seed", 42, desc="Random seed for the internal random generator")

        self.add_discrete_input("layout", None, desc="Tabular wind farm layout generated from ORBIT")

        # Turbine modifications
        # All defaults are -1 to indicate the WOMBAT defaults will be used
        self.add_input(
            "power_converter_minor_repair_scale", -1, units="unitless", desc="1 / mean time between failure (years)"
        )
        self.add_input(
            "power_converter_minor_repair_time", -1, units="h", desc="Number of hours to complete the repair"
        )
        self.add_input(
            "power_converter_minor_repair_materials",
            -1,
            units="USD",
            desc="Total cost of materials used to complete the repair. If between 0 and 1, the cost is proportional to the turbine CapEx.",
        )
        self.add_input(
            "power_converter_major_repair_scale", -1, units="unitless", desc="1 / mean time between failure (years)"
        )
        self.add_input(
            "power_converter_major_repair_time", -1, units="h", desc="Number of hours to complete the repair"
        )
        self.add_input(
            "power_converter_major_repair_materials",
            -1,
            units="USD",
            desc="Total cost of materials used to complete the repair. If between 0 and 1, the cost is proportional to the turbine CapEx.",
        )
        self.add_input(
            "power_converter_replacement_scale", -1, units="unitless", desc="1 / mean time between failure (years)"
        )
        self.add_input("power_converter_replacement_time", -1, units="h", desc="Number of hours to complete the repair")
        self.add_input(
            "power_converter_replacement_materials",
            -1,
            units="USD",
            desc="Total cost of materials used to complete the repair. If between 0 and 1, the cost is proportional to the turbine CapEx.",
        )

        self.add_input(
            "electrical_system_minor_repair_scale", -1, units="unitless", desc="1 / mean time between failure (years)"
        )
        self.add_input(
            "electrical_system_minor_repair_time", -1, units="h", desc="Number of hours to complete the repair"
        )
        self.add_input(
            "electrical_system_minor_repair_materials",
            -1,
            units="USD",
            desc="Total cost of materials used to complete the repair. If between 0 and 1, the cost is proportional to the turbine CapEx.",
        )
        self.add_input(
            "electrical_system_major_repair_scale", -1, units="unitless", desc="1 / mean time between failure (years)"
        )
        self.add_input(
            "electrical_system_major_repair_time", -1, units="h", desc="Number of hours to complete the repair"
        )
        self.add_input(
            "electrical_system_major_repair_materials",
            -1,
            units="USD",
            desc="Total cost of materials used to complete the repair. If between 0 and 1, the cost is proportional to the turbine CapEx.",
        )
        self.add_input(
            "electrical_system_replacement_scale", -1, units="unitless", desc="1 / mean time between failure (years)"
        )
        self.add_input(
            "electrical_system_replacement_time", -1, units="h", desc="Number of hours to complete the repair"
        )
        self.add_input(
            "electrical_system_replacement_materials",
            -1,
            units="USD",
            desc="Total cost of materials used to complete the repair. If between 0 and 1, the cost is proportional to the turbine CapEx.",
        )

        self.add_input(
            "hydraulic_pitch_system_minor_repair_scale",
            -1,
            units="unitless",
            desc="1 / mean time between failure (years)",
        )
        self.add_input(
            "hydraulic_pitch_system_minor_repair_time", -1, units="h", desc="Number of hours to complete the repair"
        )
        self.add_input(
            "hydraulic_pitch_system_minor_repair_materials",
            -1,
            units="USD",
            desc="Total cost of materials used to complete the repair. If between 0 and 1, the cost is proportional to the turbine CapEx.",
        )
        self.add_input(
            "hydraulic_pitch_system_major_repair_scale",
            -1,
            units="unitless",
            desc="1 / mean time between failure (years)",
        )
        self.add_input(
            "hydraulic_pitch_system_major_repair_time", -1, units="h", desc="Number of hours to complete the repair"
        )
        self.add_input(
            "hydraulic_pitch_system_major_repair_materials",
            -1,
            units="USD",
            desc="Total cost of materials used to complete the repair. If between 0 and 1, the cost is proportional to the turbine CapEx.",
        )
        self.add_input(
            "hydraulic_pitch_system_replacement_scale",
            -1,
            units="unitless",
            desc="1 / mean time between failure (years)",
        )
        self.add_input(
            "hydraulic_pitch_system_replacement_time", -1, units="h", desc="Number of hours to complete the repair"
        )
        self.add_input(
            "hydraulic_pitch_system_replacement_materials",
            -1,
            units="USD",
            desc="Total cost of materials used to complete the repair. If between 0 and 1, the cost is proportional to the turbine CapEx.",
        )

        self.add_input(
            "ballast_pump_minor_repair_scale", -1, units="unitless", desc="1 / mean time between failure (years)"
        )
        self.add_input("ballast_pump_minor_repair_time", -1, units="h", desc="Number of hours to complete the repair")
        self.add_input(
            "ballast_pump_minor_repair_materials",
            -1,
            units="USD",
            desc="Total cost of materials used to complete the repair. If between 0 and 1, the cost is proportional to the turbine CapEx.",
        )

        self.add_input(
            "yaw_system_minor_repair_scale", -1, units="unitless", desc="1 / mean time between failure (years)"
        )
        self.add_input("yaw_system_minor_repair_time", -1, units="h", desc="Number of hours to complete the repair")
        self.add_input(
            "yaw_system_minor_repair_materials",
            -1,
            units="USD",
            desc="Total cost of materials used to complete the repair. If between 0 and 1, the cost is proportional to the turbine CapEx.",
        )
        self.add_input(
            "yaw_system_major_repair_scale", -1, units="unitless", desc="1 / mean time between failure (years)"
        )
        self.add_input("yaw_system_major_repair_time", -1, units="h", desc="Number of hours to complete the repair")
        self.add_input(
            "yaw_system_major_repair_materials",
            -1,
            units="USD",
            desc="Total cost of materials used to complete the repair. If between 0 and 1, the cost is proportional to the turbine CapEx.",
        )
        self.add_input(
            "yaw_system_replacement_scale", -1, units="unitless", desc="1 / mean time between failure (years)"
        )
        self.add_input("yaw_system_replacement_time", -1, units="h", desc="Number of hours to complete the repair")
        self.add_input(
            "yaw_system_replacement_materials",
            -1,
            units="USD",
            desc="Total cost of materials used to complete the repair. If between 0 and 1, the cost is proportional to the turbine CapEx.",
        )

        self.add_input(
            "rotor_blades_minor_repair_scale", -1, units="unitless", desc="1 / mean time between failure (years)"
        )
        self.add_input("rotor_blades_minor_repair_time", -1, units="h", desc="Number of hours to complete the repair")
        self.add_input(
            "rotor_blades_minor_repair_materials",
            -1,
            units="USD",
            desc="Total cost of materials used to complete the repair. If between 0 and 1, the cost is proportional to the turbine CapEx.",
        )
        self.add_input(
            "rotor_blades_major_repair_scale", -1, units="unitless", desc="1 / mean time between failure (years)"
        )
        self.add_input("rotor_blades_major_repair_time", -1, units="h", desc="Number of hours to complete the repair")
        self.add_input(
            "rotor_blades_major_repair_materials",
            -1,
            units="USD",
            desc="Total cost of materials used to complete the repair. If between 0 and 1, the cost is proportional to the turbine CapEx.",
        )
        self.add_input(
            "rotor_blades_replacement_scale", -1, units="unitless", desc="1 / mean time between failure (years)"
        )
        self.add_input("rotor_blades_replacement_time", -1, units="h", desc="Number of hours to complete the repair")
        self.add_input(
            "rotor_blades_replacement_materials",
            -1,
            units="USD",
            desc="Total cost of materials used to complete the repair. If between 0 and 1, the cost is proportional to the turbine CapEx.",
        )

        self.add_input(
            "generator_minor_repair_scale", -1, units="unitless", desc="1 / mean time between failure (years)"
        )
        self.add_input("generator_minor_repair_time", -1, units="h", desc="Number of hours to complete the repair")
        self.add_input(
            "generator_minor_repair_materials",
            -1,
            units="USD",
            desc="Total cost of materials used to complete the repair. If between 0 and 1, the cost is proportional to the turbine CapEx.",
        )
        self.add_input(
            "generator_major_repair_scale", -1, units="unitless", desc="1 / mean time between failure (years)"
        )
        self.add_input("generator_major_repair_time", -1, units="h", desc="Number of hours to complete the repair")
        self.add_input(
            "generator_major_repair_materials",
            -1,
            units="USD",
            desc="Total cost of materials used to complete the repair. If between 0 and 1, the cost is proportional to the turbine CapEx.",
        )
        self.add_input(
            "generator_replacement_scale", -1, units="unitless", desc="1 / mean time between failure (years)"
        )
        self.add_input("generator_replacement_time", -1, units="h", desc="Number of hours to complete the repair")
        self.add_input(
            "generator_replacement_materials",
            -1,
            units="USD",
            desc="Total cost of materials used to complete the repair. If between 0 and 1, the cost is proportional to the turbine CapEx.",
        )

        self.add_input(
            "drive_train_minor_repair_scale", -1, units="unitless", desc="1 / mean time between failure (years)"
        )
        self.add_input("drive_train_minor_repair_time", -1, units="h", desc="Number of hours to complete the repair")
        self.add_input(
            "drive_train_minor_repair_materials",
            -1,
            units="USD",
            desc="Total cost of materials used to complete the repair. If between 0 and 1, the cost is proportional to the turbine CapEx.",
        )
        self.add_input(
            "drive_train_major_repair_scale", -1, units="unitless", desc="1 / mean time between failure (years)"
        )
        self.add_input("drive_train_major_repair_time", -1, units="h", desc="Number of hours to complete the repair")
        self.add_input(
            "drive_train_major_repair_materials",
            -1,
            units="USD",
            desc="Total cost of materials used to complete the repair. If between 0 and 1, the cost is proportional to the turbine CapEx.",
        )
        self.add_input(
            "drive_train_replacement_scale", -1, units="unitless", desc="1 / mean time between failure (years)"
        )
        self.add_input("drive_train_replacement_time", -1, units="h", desc="Number of hours to complete the repair")
        self.add_input(
            "drive_train_replacement_materials",
            -1,
            units="USD",
            desc="Total cost of materials used to complete the repair. If between 0 and 1, the cost is proportional to the turbine CapEx.",
        )

        self.add_input("anchor_minor_repair_scale", -1, units="unitless", desc="1 / mean time between failure (years)")
        self.add_input("anchor_minor_repair_time", -1, units="h", desc="Number of hours to complete the repair")
        self.add_input(
            "anchor_minor_repair_materials",
            -1,
            units="USD",
            desc="Total cost of materials used to complete the repair. If between 0 and 1, the cost is proportional to the turbine CapEx.",
        )
        self.add_input("anchor_major_repair_scale", -1, units="unitless", desc="1 / mean time between failure (years)")
        self.add_input("anchor_major_repair_time", -1, units="h", desc="Number of hours to complete the repair")
        self.add_input(
            "anchor_major_repair_materials",
            -1,
            units="USD",
            desc="Total cost of materials used to complete the repair. If between 0 and 1, the cost is proportional to the turbine CapEx.",
        )
        self.add_input("anchor_replacement_scale", -1, units="unitless", desc="1 / mean time between failure (years)")
        self.add_input("anchor_replacement_time", -1, units="h", desc="Number of hours to complete the repair")
        self.add_input(
            "anchor_replacement_materials",
            -1,
            units="USD",
            desc="Total cost of materials used to complete the repair. If between 0 and 1, the cost is proportional to the turbine CapEx.",
        )

        self.add_input(
            "mooring_lines_minor_repair_scale", -1, units="unitless", desc="1 / mean time between failure (years)"
        )
        self.add_input("mooring_lines_minor_repair_time", -1, units="h", desc="Number of hours to complete the repair")
        self.add_input(
            "mooring_lines_minor_repair_materials",
            -1,
            units="USD",
            desc="Total cost of materials used to complete the repair. If between 0 and 1, the cost is proportional to the turbine CapEx.",
        )
        self.add_input(
            "mooring_lines_major_repair_scale", -1, units="unitless", desc="1 / mean time between failure (years)"
        )
        self.add_input("mooring_lines_major_repair_time", -1, units="h", desc="Number of hours to complete the repair")
        self.add_input(
            "mooring_lines_major_repair_materials",
            -1,
            units="USD",
            desc="Total cost of materials used to complete the repair. If between 0 and 1, the cost is proportional to the turbine CapEx.",
        )
        self.add_input(
            "mooring_lines_replacement_scale", -1, units="unitless", desc="1 / mean time between failure (years)"
        )
        self.add_input("mooring_lines_replacement_time", -1, units="h", desc="Number of hours to complete the repair")
        self.add_input(
            "mooring_lines_replacement_materials",
            -1,
            units="USD",
            desc="Total cost of materials used to complete the repair. If between 0 and 1, the cost is proportional to the turbine CapEx.",
        )
        self.add_input(
            "mooring_lines_buoyancy_module_replacement_scale",
            -1,
            units="unitless",
            desc="1 / mean time between failure (years)",
        )
        self.add_input(
            "mooring_lines_buoyancy_module_replacement_time",
            -1,
            units="h",
            desc="Number of hours to complete the repair",
        )
        self.add_input(
            "mooring_lines_buoyancy_module_replacement_materials",
            -1,
            units="USD",
            desc="Total cost of materials used to complete the repair. If between 0 and 1, the cost is proportional to the turbine CapEx.",
        )

        # Outputs
        self.add_output(
            "total_opex",
            0.0,
            units="USD",
            desc=(
                "Total operational expenditure (fixed costs, port fees, labor, servicing" " equipment, and materials)"
            ),
        )
        self.add_output(
            "annual_opex_per_kW",
            0.0,
            units="USD/kW/yr",
            desc=(
                "Average annual operational expenditure (fixed costs, port fees, labor,"
                " servicing equipment, and materials) per kW"
            ),
        )
        self.add_output(
            "materials_opex",
            0.0,
            units="USD",
            desc="Cost of all replaced and consumable materials for repairs and servicing",
        )
        self.add_output(
            "equipment_opex",
            0.0,
            units="USD",
            desc="Direct cost for renting and operating servicing equipment",
        )
        self.add_output("time_availability", 0.0, units="unitless", desc="Project-level uptime based on time.")
        self.add_output(
            "energy_availability",
            0.0,
            units="unitless",
            desc="Project-level uptime based on capacity to produce energy.",
        )
        self.add_output(
            "net_capacity_factor",
            0.0,
            units="unitless",
            desc="Ratio of actual energy produced (internal IEC power curve-based w/o unmodeled losses) to theoretical maximum of energy production.",
        )
        self.add_output(
            "gross_capacity_factor",
            0.0,
            units="USD",
            desc="Ratio of potential to produce energy (internal IEC power curve-based w/o unmodeled losses) to theoretical maximum of energy production.",
        )
        self.add_output(
            "scheduled_task_completion_rate",
            0.0,
            units="USD",
            desc="Completion rate for all scheduled (maintenance) tasks.",
        )
        self.add_output(
            "unscheduled_task_completion_rate",
            0.0,
            units="USD",
            desc="Completion rate for all unscheduled (failure) events.",
        )
        self.add_output(
            "combined_task_completion_rate",
            0.0,
            units="USD",
            desc="Completion rate for all maintenance and failure events.",
        )
        self.add_output(
            "total_equipment_cost",
            0.0,
            units="USD",
            desc="Cost of all direct repair related equipment (vessels, cranes, port equipment).",
        )
        self.add_discrete_output(
            "equipment_cost_breakdown",
            None,
            desc="Data frame of equipment costs by activity type.",
        )
        self.add_discrete_output(
            "equipment_utilization_rate",
            None,
            desc="Data frame of utilization ratio of each servicing equipment.",
        )
        self.add_discrete_output(
            "equipment_dispatch_summary",
            None,
            desc="Data frame of mobilization and chartering periods by servicing equipment.",
        )
        self.add_discrete_output(
            "vessel_crew_hours_at_sea",
            None,
            desc="Data frame of the vessel hours at sea (or crew if crew data are provided).",
        )
        self.add_discrete_output(
            "total_tows",
            0,
            desc="Total number of times turbines are towed between site and port for repair.",
        )
        self.add_output(
            "direct_labor",
            0.0,
            units="USD",
            desc="Cost of labor accrued through repair operations.",
        )
        self.add_output(
            "indirect_labor",
            0.0,
            units="USD",
            desc="Fixed cost of labor for life of the farm.",
        )
        self.add_discrete_output(
            "materials_by_subassembly",
            None,
            desc="Cost of materials required for un/scheduled maintenance activities by subassembly.",
        )
        self.add_output(
            "total_materials",
            0,
            units="USD",
            desc="Total cost of materials for un/scheduled maintenance activities.",
        )
        self.add_output(
            "total_fixed_costs",
            0,
            units="USD",
            desc="Total cost of annualized fixed operational costs.",
        )
        self.add_discrete_output(
            "process_times",
            None,
            desc="Time (hours) it takes to complete repairs and maintenance, both from request submission to completion, and start to end of repair.",
        )
        self.add_discrete_output(
            "request_summary",
            None,
            desc="Number of repair and maintenance requests submitted, canceled, not completed, and completed for each category.",
        )

    def create_layout(self, inputs, outputs, discrete_inputs, discrete_outputs):
        """Creates the WOMBAT layout DataFrame from the ORBIT outputs."""
        layout = discrete_inputs["layout"]
        layout[["type", "subassembly", "upstream_cable"]] = ["turbine", "base_turbine", "base_array"]
        layout.loc[layout.id.isin(layout.substation_id), ["type", "subassembly", "upstream_cable"]] = [
            "substation",
            "base_substation",
            "base_export",
        ]
        return layout

    def compile_inputs(self, inputs, outputs, discrete_inputs, discrete_outputs):
        """Creates the WOMBAT configuration file."""

        scenario = self.options["scenario"]
        config = self._wombat_config
        config["layout"] = self.create_layout(inputs, outputs, discrete_inputs, discrete_outputs)
        config["workday_start"] = discrete_inputs["workday_start"]
        config["workday_end"] = discrete_inputs["workday_end"]
        config["project_capacity"] = inputs["project_capacity"][0]
        config["port_distance"] = inputs["equipment_dispatch_distance"][0]
        start = discrete_inputs["maintenance_start"]
        start = None if start == "None" else start
        if start is not None:
            config["maintenance_start"] = f'{start}/{config["start_year"]}'

        start = discrete_inputs["non_operational_start"]
        start = None if start == "None" else start
        config["non_operational_start"] = start

        start = discrete_inputs["non_operational_start"]
        start = None if start == "None" else start
        config["non_operational_start"] = start

        end = discrete_inputs["non_operational_end"]
        end = None if end == "None" else end
        config["non_operational_end"] = end

        start = discrete_inputs["reduced_speed_start"]
        start = None if start == "None" else start
        config["reduced_speed_start"] = start

        end = discrete_inputs["reduced_speed_end"]
        end = None if end == "None" else end
        config["reduced_speed_end"] = end

        config["reduced_speed"] = inputs["reduced_speed"][0]

        if scenario == "osw-floating":
            config["service_equipment"] = [
                [discrete_inputs["n_ctv"], "ctv"],
                [1, "dsv"],
                [1, "cab"],
            ]
            config["port"]["tugboats"] = [discrete_inputs["n_tugboats"], "tugboat"]
            config["port"]["site_distance"] = inputs["repair_port_distance"]
            config["port"]["workday_start"] = discrete_inputs["port_workday_start"]
            config["port"]["workday_end"] = discrete_inputs["port_workday_end"]
            config["port"]["max_operations"] = discrete_inputs["port_max_operations"]
            config["port"]["n_crews"] = discrete_inputs["n_port_crews"]
            config["port"]["max_operations"] = discrete_inputs["port_max_operations"]
        elif scenario == "osw-fixed":
            config["service_equipment"] = [
                [discrete_inputs["n_ctv"], "ctv"],
                [discrete_inputs["n_hlv"], "hlv"],
                [1, "dsv"],
                [1, "cab"],
            ]
        else:
            config["service_equipment"] = [
                [discrete_inputs["n_ctv"], "truck"],
                [discrete_inputs["n_hlv"], "crawler"],
            ]

        config["start_year"] = config["end_year"] - int(inputs["years"][0]) + 1
        config["random_seed"] = discrete_inputs["random_seed"]

        # TODO: determine if additional  turbines should be allowed
        # config["turbines"] |= inputs["turbines"]

        original_capacity = config["turbines"]["base_turbine"]["capacity_kw"]
        original_capex = config["turbines"]["base_turbine"]["capex_kw"]
        config["turbines"]["base_turbine"]["capacity_kw"] = inputs["turbine_capacity"][0] / 1000.0
        config["turbines"]["base_turbine"]["capex_kw"] = inputs["turbine_capex_kw"][0]

        turbine_capex = original_capacity * original_capex
        for subassembly in config["turbines"]["base_turbine"].keys():
            if subassembly in ("capacity_kw", "capex_kw", "power_curve", "n_stacks", "stack_capacity_kw"):
                continue
            for i, maintenance in enumerate(config["turbines"]["base_turbine"][subassembly]["maintenance"]):
                config["turbines"]["base_turbine"][subassembly]["maintenance"][i]["materials"] /= turbine_capex
            for i, failure in enumerate(config["turbines"]["base_turbine"][subassembly]["failures"]):
                config["turbines"]["base_turbine"][subassembly]["failures"][i]["materials"] /= turbine_capex

        if (val := inputs["power_converter_minor_repair_scale"][0]) > -1:
            config["turbines"]["base_turbine"]["power_converter"]["failures"][0]["scale"] = val
        if (val := inputs["power_converter_minor_repair_time"][0]) > -1:
            config["turbines"]["base_turbine"]["power_converter"]["failures"][0]["time"] = val
        if (val := inputs["power_converter_minor_repair_materials"][0]) > -1:
            config["turbines"]["base_turbine"]["power_converter"]["failures"][0]["materials"] = val
        if (val := inputs["power_converter_major_repair_scale"][0]) > -1:
            config["turbines"]["base_turbine"]["power_converter"]["failures"][1]["scale"] = val
        if (val := inputs["power_converter_major_repair_time"][0]) > -1:
            config["turbines"]["base_turbine"]["power_converter"]["failures"][1]["time"] = val
        if (val := inputs["power_converter_major_repair_materials"][0]) > -1:
            config["turbines"]["base_turbine"]["power_converter"]["failures"][1]["materials"] = val
        if (val := inputs["power_converter_replacement_scale"][0]) > -1:
            config["turbines"]["base_turbine"]["power_converter"]["failures"][2]["scale"] = val
        if (val := inputs["power_converter_replacement_time"][0]) > -1:
            config["turbines"]["base_turbine"]["power_converter"]["failures"][2]["time"] = val
        if (val := inputs["power_converter_replacement_materials"][0]) > -1:
            config["turbines"]["base_turbine"]["power_converter"]["failures"][2]["materials"] = val

        if (val := inputs["electrical_system_minor_repair_scale"][0]) > -1:
            config["turbines"]["base_turbine"]["electrical_system"]["failures"][0]["scale"] = val
        if (val := inputs["electrical_system_minor_repair_time"][0]) > -1:
            config["turbines"]["base_turbine"]["electrical_system"]["failures"][0]["time"] = val
        if (val := inputs["electrical_system_minor_repair_materials"][0]) > -1:
            config["turbines"]["base_turbine"]["electrical_system"]["failures"][0]["materials"] = val
        if (val := inputs["electrical_system_major_repair_scale"][0]) > -1:
            config["turbines"]["base_turbine"]["electrical_system"]["failures"][1]["scale"] = val
        if (val := inputs["electrical_system_major_repair_time"][0]) > -1:
            config["turbines"]["base_turbine"]["electrical_system"]["failures"][1]["time"] = val
        if (val := inputs["electrical_system_major_repair_materials"][0]) > -1:
            config["turbines"]["base_turbine"]["electrical_system"]["failures"][1]["materials"] = val
        if (val := inputs["electrical_system_replacement_scale"][0]) > -1:
            config["turbines"]["base_turbine"]["electrical_system"]["failures"][2]["scale"] = val
        if (val := inputs["electrical_system_replacement_time"][0]) > -1:
            config["turbines"]["base_turbine"]["electrical_system"]["failures"][2]["time"] = val
        if (val := inputs["electrical_system_replacement_materials"][0]) > -1:
            config["turbines"]["base_turbine"]["electrical_system"]["failures"][2]["materials"] = val

        if (val := inputs["hydraulic_pitch_system_minor_repair_scale"][0]) > -1:
            config["turbines"]["base_turbine"]["hydraulic_pitch_system"]["failures"][0]["scale"] = val
        if (val := inputs["hydraulic_pitch_system_minor_repair_time"][0]) > -1:
            config["turbines"]["base_turbine"]["hydraulic_pitch_system"]["failures"][0]["time"] = val
        if (val := inputs["hydraulic_pitch_system_minor_repair_materials"][0]) > -1:
            config["turbines"]["base_turbine"]["hydraulic_pitch_system"]["failures"][0]["materials"] = val
        if (val := inputs["hydraulic_pitch_system_major_repair_scale"][0]) > -1:
            config["turbines"]["base_turbine"]["hydraulic_pitch_system"]["failures"][1]["scale"] = val
        if (val := inputs["hydraulic_pitch_system_major_repair_time"][0]) > -1:
            config["turbines"]["base_turbine"]["hydraulic_pitch_system"]["failures"][1]["time"] = val
        if (val := inputs["hydraulic_pitch_system_major_repair_materials"][0]) > -1:
            config["turbines"]["base_turbine"]["hydraulic_pitch_system"]["failures"][1]["materials"] = val
        if (val := inputs["hydraulic_pitch_system_replacement_scale"][0]) > -1:
            config["turbines"]["base_turbine"]["hydraulic_pitch_system"]["failures"][2]["scale"] = val
        if (val := inputs["hydraulic_pitch_system_replacement_time"][0]) > -1:
            config["turbines"]["base_turbine"]["hydraulic_pitch_system"]["failures"][2]["time"] = val
        if (val := inputs["hydraulic_pitch_system_replacement_materials"][0]) > -1:
            config["turbines"]["base_turbine"]["hydraulic_pitch_system"]["failures"][2]["materials"] = val

        if (val := inputs["yaw_system_minor_repair_scale"][0]) > -1:
            config["turbines"]["base_turbine"]["yaw_system"]["failures"][0]["scale"] = val
        if (val := inputs["yaw_system_minor_repair_time"][0]) > -1:
            config["turbines"]["base_turbine"]["yaw_system"]["failures"][0]["time"] = val
        if (val := inputs["yaw_system_minor_repair_materials"][0]) > -1:
            config["turbines"]["base_turbine"]["yaw_system"]["failures"][0]["materials"] = val
        if (val := inputs["yaw_system_major_repair_scale"][0]) > -1:
            config["turbines"]["base_turbine"]["yaw_system"]["failures"][1]["scale"] = val
        if (val := inputs["yaw_system_major_repair_time"][0]) > -1:
            config["turbines"]["base_turbine"]["yaw_system"]["failures"][1]["time"] = val
        if (val := inputs["yaw_system_major_repair_materials"][0]) > -1:
            config["turbines"]["base_turbine"]["yaw_system"]["failures"][1]["materials"] = val
        if (val := inputs["yaw_system_replacement_scale"][0]) > -1:
            config["turbines"]["base_turbine"]["yaw_system"]["failures"][2]["scale"] = val
        if (val := inputs["yaw_system_replacement_time"][0]) > -1:
            config["turbines"]["base_turbine"]["yaw_system"]["failures"][2]["time"] = val
        if (val := inputs["yaw_system_replacement_materials"][0]) > -1:
            config["turbines"]["base_turbine"]["yaw_system"]["failures"][2]["materials"] = val

        if (val := inputs["rotor_blades_minor_repair_scale"][0]) > -1:
            config["turbines"]["base_turbine"]["rotor_blades"]["failures"][0]["scale"] = val
        if (val := inputs["rotor_blades_minor_repair_time"][0]) > -1:
            config["turbines"]["base_turbine"]["rotor_blades"]["failures"][0]["time"] = val
        if (val := inputs["rotor_blades_minor_repair_materials"][0]) > -1:
            config["turbines"]["base_turbine"]["rotor_blades"]["failures"][0]["materials"] = val
        if (val := inputs["rotor_blades_major_repair_scale"][0]) > -1:
            config["turbines"]["base_turbine"]["rotor_blades"]["failures"][1]["scale"] = val
        if (val := inputs["rotor_blades_major_repair_time"][0]) > -1:
            config["turbines"]["base_turbine"]["rotor_blades"]["failures"][1]["time"] = val
        if (val := inputs["rotor_blades_major_repair_materials"][0]) > -1:
            config["turbines"]["base_turbine"]["rotor_blades"]["failures"][1]["materials"] = val
        if (val := inputs["rotor_blades_replacement_scale"][0]) > -1:
            config["turbines"]["base_turbine"]["rotor_blades"]["failures"][2]["scale"] = val
        if (val := inputs["rotor_blades_replacement_time"][0]) > -1:
            config["turbines"]["base_turbine"]["rotor_blades"]["failures"][2]["time"] = val
        if (val := inputs["rotor_blades_replacement_materials"][0]) > -1:
            config["turbines"]["base_turbine"]["rotor_blades"]["failures"][2]["materials"] = val

        if (val := inputs["generator_minor_repair_scale"][0]) > -1:
            config["turbines"]["base_turbine"]["generator"]["failures"][0]["scale"] = val
        if (val := inputs["generator_minor_repair_time"][0]) > -1:
            config["turbines"]["base_turbine"]["generator"]["failures"][0]["time"] = val
        if (val := inputs["generator_minor_repair_materials"][0]) > -1:
            config["turbines"]["base_turbine"]["generator"]["failures"][0]["materials"] = val
        if (val := inputs["generator_major_repair_scale"][0]) > -1:
            config["turbines"]["base_turbine"]["generator"]["failures"][1]["scale"] = val
        if (val := inputs["generator_major_repair_time"][0]) > -1:
            config["turbines"]["base_turbine"]["generator"]["failures"][1]["time"] = val
        if (val := inputs["generator_major_repair_materials"][0]) > -1:
            config["turbines"]["base_turbine"]["generator"]["failures"][1]["materials"] = val
        if (val := inputs["generator_replacement_scale"][0]) > -1:
            config["turbines"]["base_turbine"]["generator"]["failures"][2]["scale"] = val
        if (val := inputs["generator_replacement_time"][0]) > -1:
            config["turbines"]["base_turbine"]["generator"]["failures"][2]["time"] = val
        if (val := inputs["generator_replacement_materials"][0]) > -1:
            config["turbines"]["base_turbine"]["generator"]["failures"][2]["materials"] = val

        if (val := inputs["drive_train_minor_repair_scale"][0]) > -1:
            config["turbines"]["base_turbine"]["drive_train"]["failures"][0]["scale"] = val
        if (val := inputs["drive_train_minor_repair_time"][0]) > -1:
            config["turbines"]["base_turbine"]["drive_train"]["failures"][0]["time"] = val
        if (val := inputs["drive_train_minor_repair_materials"][0]) > -1:
            config["turbines"]["base_turbine"]["drive_train"]["failures"][0]["materials"] = val
        if (val := inputs["drive_train_major_repair_scale"][0]) > -1:
            config["turbines"]["base_turbine"]["drive_train"]["failures"][1]["scale"] = val
        if (val := inputs["drive_train_major_repair_time"][0]) > -1:
            config["turbines"]["base_turbine"]["drive_train"]["failures"][1]["time"] = val
        if (val := inputs["drive_train_major_repair_materials"][0]) > -1:
            config["turbines"]["base_turbine"]["drive_train"]["failures"][1]["materials"] = val
        if (val := inputs["drive_train_replacement_scale"][0]) > -1:
            config["turbines"]["base_turbine"]["drive_train"]["failures"][2]["scale"] = val
        if (val := inputs["drive_train_replacement_time"][0]) > -1:
            config["turbines"]["base_turbine"]["drive_train"]["failures"][2]["time"] = val
        if (val := inputs["drive_train_replacement_materials"][0]) > -1:
            config["turbines"]["base_turbine"]["drive_train"]["failures"][2]["materials"] = val

        if scenario == "osw-floating":
            if (val := inputs["anchor_minor_repair_scale"][0]) > -1:
                config["turbines"]["base_turbine"]["anchor"]["failures"][0]["scale"] = val
            if (val := inputs["anchor_minor_repair_time"][0]) > -1:
                config["turbines"]["base_turbine"]["anchor"]["failures"][0]["time"] = val
            if (val := inputs["anchor_minor_repair_materials"][0]) > -1:
                config["turbines"]["base_turbine"]["anchor"]["failures"][0]["materials"] = val
            if (val := inputs["anchor_major_repair_scale"][0]) > -1:
                config["turbines"]["base_turbine"]["anchor"]["failures"][1]["scale"] = val
            if (val := inputs["anchor_major_repair_time"][0]) > -1:
                config["turbines"]["base_turbine"]["anchor"]["failures"][1]["time"] = val
            if (val := inputs["anchor_major_repair_materials"][0]) > -1:
                config["turbines"]["base_turbine"]["anchor"]["failures"][1]["materials"] = val
            if (val := inputs["anchor_replacement_scale"][0]) > -1:
                config["turbines"]["base_turbine"]["anchor"]["failures"][2]["scale"] = val
            if (val := inputs["anchor_replacement_time"][0]) > -1:
                config["turbines"]["base_turbine"]["anchor"]["failures"][2]["time"] = val
            if (val := inputs["anchor_replacement_materials"][0]) > -1:
                config["turbines"]["base_turbine"]["anchor"]["failures"][2]["materials"] = val

            if (val := inputs["ballast_pump_minor_repair_scale"][0]) > -1:
                config["turbines"]["base_turbine"]["ballast_pump"]["failures"][0]["scale"] = val
            if (val := inputs["ballast_pump_minor_repair_time"][0]) > -1:
                config["turbines"]["base_turbine"]["ballast_pump"]["failures"][0]["time"] = val
            if (val := inputs["ballast_pump_minor_repair_materials"][0]) > -1:
                config["turbines"]["base_turbine"]["ballast_pump"]["failures"][0]["materials"] = val

            if (val := inputs["mooring_lines_minor_repair_scale"][0]) > -1:
                config["turbines"]["base_turbine"]["mooring_lines"]["failures"][0]["scale"] = val
            if (val := inputs["mooring_lines_minor_repair_time"][0]) > -1:
                config["turbines"]["base_turbine"]["mooring_lines"]["failures"][0]["time"] = val
            if (val := inputs["mooring_lines_minor_repair_materials"][0]) > -1:
                config["turbines"]["base_turbine"]["mooring_lines"]["failures"][0]["materials"] = val
            if (val := inputs["mooring_lines_major_repair_scale"][0]) > -1:
                config["turbines"]["base_turbine"]["mooring_lines"]["failures"][1]["scale"] = val
            if (val := inputs["mooring_lines_major_repair_time"][0]) > -1:
                config["turbines"]["base_turbine"]["mooring_lines"]["failures"][1]["time"] = val
            if (val := inputs["mooring_lines_major_repair_materials"][0]) > -1:
                config["turbines"]["base_turbine"]["mooring_lines"]["failures"][1]["materials"] = val
            if (val := inputs["mooring_lines_replacement_scale"][0]) > -1:
                config["turbines"]["base_turbine"]["mooring_lines"]["failures"][2]["scale"] = val
            if (val := inputs["mooring_lines_replacement_time"][0]) > -1:
                config["turbines"]["base_turbine"]["mooring_lines"]["failures"][2]["time"] = val
            if (val := inputs["mooring_lines_replacement_materials"][0]) > -1:
                config["turbines"]["base_turbine"]["mooring_lines"]["failures"][2]["materials"] = val
            if (val := inputs["mooring_lines_buoyancy_module_replacement_replacement_scale"][0]) > -1:
                config["turbines"]["base_turbine"]["mooring_lines_buoyancy_module_replacement"]["failures"][3][
                    "scale"
                ] = val
            if (val := inputs["mooring_lines_buoyancy_module_replacement_replacement_time"][0]) > -1:
                config["turbines"]["base_turbine"]["mooring_lines_buoyancy_module_replacement"]["failures"][3][
                    "time"
                ] = val
            if (val := inputs["mooring_lines_buoyancy_module_replacement_replacement_materials"][0]) > -1:
                config["turbines"]["base_turbine"]["mooring_lines_buoyancy_module_replacement"]["failures"][3][
                    "materials"
                ] = val

        return config

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        """Creates and runs the project, then gathers the results."""

        config = self.compile_inputs(inputs, outputs, discrete_inputs, discrete_outputs)
        sim = Simulation(library_path=DEFAULT_DATA, config=config)
        sim.run(save_metrics_inputs=False, delete_logs=True)

        metrics = sim.metrics
        frequency = "project"
        capacity_kW = metrics.project_capacity * 1000

        opex = metrics.opex(frequency, by_category=True)
        outputs["total_opex"] = opex.OpEx
        outputs["annual_opex_per_kW"] = outputs["total_opex"] / capacity_kW / sim.env.simulation_years

        outputs["time_availability"] = metrics.time_based_availability(frequency="project", by="windfarm").squeeze()
        outputs["energy_availability"] = metrics.production_based_availability(
            frequency="project", by="windfarm"
        ).squeeze()
        outputs["net_capacity_factor"] = metrics.capacity_factor(
            which="net", frequency="project", by="windfarm"
        ).squeeze()
        outputs["gross_capacity_factor"] = metrics.capacity_factor(
            which="gross", frequency="project", by="windfarm"
        ).squeeze()
        outputs["scheduled_task_completion_rate"] = metrics.task_completion_rate(
            which="scheduled", frequency="project"
        ).squeeze()
        outputs["unscheduled_task_completion_rate"] = metrics.task_completion_rate(
            which="unscheduled", frequency="project"
        ).squeeze()
        outputs["combined_task_completion_rate"] = metrics.task_completion_rate(
            which="both", frequency="project"
        ).squeeze()
        outputs["total_equipment_cost"] = metrics.equipment_costs(frequency="project", by_equipment=False).squeeze()

        discrete_outputs["equipment_cost_breakdown"] = metrics.equipment_costs(frequency="project", by_equipment=True)
        discrete_outputs["equipment_utilization_rate"] = metrics.service_equipment_utilization(frequency="project")
        discrete_outputs["equipment_dispatch_summary"] = metrics.dispatch_summary(frequency="project")

        discrete_outputs["vessel_crew_hours_at_sea"] = metrics.vessel_crew_hours_at_sea(
            frequency="project", by_equipment=True
        )
        discrete_outputs["total_tows"] = metrics.number_of_tows(frequency="project")
        outputs["direct_labor"] = metrics.labor_costs(frequency="project", by_type=False)
        discrete_outputs["materials_by_subassembly"] = metrics.component_costs(
            frequency="project", by_category=False, by_action=False
        )
        outputs["total_materials"] = discrete_outputs["materials_by_subassembly"].values.sum()

        fixed_costs = metrics.project_fixed_costs(frequency="project", resolution="medium")
        outputs["indirect_labor"] = fixed_costs[["labor"]].squeeze()
        outputs["total_fixed_costs"] = fixed_costs.values.sum()

        discrete_outputs["process_times"] = metrics.process_times(include_incompletes=False)
        discrete_outputs["request_summary"] = metrics.request_summary()

        # NOTE: excluded outputs: NPV, power production
