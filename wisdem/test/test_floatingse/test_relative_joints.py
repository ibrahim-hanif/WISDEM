"""
Unit tests for AggregateJoints with relative joints in rectangular/cylindrical coordinates.

Tests that mooring anchor joints defined relative to column keel joints
are resolved to the correct Cartesian (XYZ) positions.
Test that a heave plate relative to a column keel is resolved correctly in the z dimension but not x/y.
"""

import unittest

import numpy as np
import openmdao.api as om

from wisdem.glue_code.gc_WT_DataStruc import AggregateJoints


def make_floating_init_options(joint_names, cylindrical, relative, relative_dims):
    """Build a minimal floating_init_options dict for AggregateJoints with no members."""
    n_joints = len(joint_names)
    name2idx = {name: i for i, name in enumerate(joint_names)}
    return {
        "joints": {
            "n_joints": n_joints,
            "name": joint_names,
            "cylindrical": cylindrical,
            "relative": relative,
            "relative_dims": relative_dims,
            "name2idx": name2idx,
        },
        "members": {
            "n_members": 0,
            "name": [],
            "joint1": [],
            "joint2": [],
            "n_axial_joints": [],
            "no_intersect": [],
        },
    }


class TestRelativeJoints(unittest.TestCase):
    """Test AggregateJoints resolves relative joints correctly in both cylindrical and Cartesian coordinates."""

    @staticmethod
    def _run_aggregate_joints(floating_init_options, locations_input):
        prob = om.Problem()
        prob.model.add_subsystem(
            "alljoints",
            AggregateJoints(floating_init_options=floating_init_options),
            promotes=["*"],
        )
        prob.setup()
        prob["location"] = locations_input
        prob.run_model()
        return prob["joints_xyz"]

    def test_single_relative_cylindrical_joint(self):
        """
        anchor1 is relative to col_keel in the radial dimension only.
        After relative offset: anchor1 r = r_anchor_offset + r_col
        After cyl->XYZ: x = r_total * cos(theta), y = r_total * sin(theta)
        """
        r_col = 51.75
        theta_col = 180.0
        z_keel = -20.0
        r_anchor_offset = 786.05
        z_anchor = -200.0

        joint_names = ["col_keel", "anchor1"]
        locations = np.array([
            [r_col,           theta_col, z_keel],   # col_keel (cylindrical)
            [r_anchor_offset, theta_col, z_anchor],  # anchor1  (cylindrical, relative r)
        ])
        cylindrical = [True, True]
        relative = ["origin", "col_keel"]
        relative_dims = [[False, False, False], [True, False, False]]

        opts = make_floating_init_options(joint_names, cylindrical, relative, relative_dims)
        joints_xyz = self._run_aggregate_joints(opts, locations.copy())

        # col_keel XYZ
        np.testing.assert_allclose(joints_xyz[0, 0], r_col * np.cos(np.deg2rad(theta_col)), atol=1e-10)
        np.testing.assert_allclose(joints_xyz[0, 1], r_col * np.sin(np.deg2rad(theta_col)), atol=1e-10)
        np.testing.assert_allclose(joints_xyz[0, 2], z_keel, atol=1e-10)

        # anchor1 XYZ — relative r resolved before cylindrical conversion
        r_total = r_anchor_offset + r_col
        np.testing.assert_allclose(joints_xyz[1, 0], r_total * np.cos(np.deg2rad(theta_col)), atol=1e-10)
        np.testing.assert_allclose(joints_xyz[1, 1], r_total * np.sin(np.deg2rad(theta_col)), atol=1e-10)
        np.testing.assert_allclose(joints_xyz[1, 2], z_anchor, atol=1e-10)

    def test_three_mooring_anchors_iea15mw(self):
        """
        Reproduce the IEA-15-240-RWT VolturnUS-S mooring anchor geometry.
        Column keels at 120-deg spacing; anchors at relative radial offset from each keel.
        """
        r_col = 51.75
        z_keel = -20.0
        r_anchor_offset = 786.05
        z_anchor = -200.0
        thetas = [180.0, 60.0, -60.0]

        joint_names = ["col1_keel", "col2_keel", "col3_keel", "anchor1", "anchor2", "anchor3"]
        locations = np.array([
            [r_col,           thetas[0], z_keel],
            [r_col,           thetas[1], z_keel],
            [r_col,           thetas[2], z_keel],
            [r_anchor_offset, thetas[0], z_anchor],
            [r_anchor_offset, thetas[1], z_anchor],
            [r_anchor_offset, thetas[2], z_anchor],
        ])
        cylindrical = [True] * 6
        relative = ["origin", "origin", "origin", "col1_keel", "col2_keel", "col3_keel"]
        relative_dims = [[False, False, False]] * 3 + [[True, False, False]] * 3

        opts = make_floating_init_options(joint_names, cylindrical, relative, relative_dims)
        joints_xyz = self._run_aggregate_joints(opts, locations.copy())

        r_total = r_anchor_offset + r_col
        for i, (theta, name) in enumerate(zip(thetas, ["anchor1", "anchor2", "anchor3"])):
            idx = i + 3  # anchors start at index 3
            np.testing.assert_allclose(joints_xyz[idx, 0], r_total * np.cos(np.deg2rad(theta)), atol=1e-10,
                                       err_msg=f"{name} x mismatch")
            np.testing.assert_allclose(joints_xyz[idx, 1], r_total * np.sin(np.deg2rad(theta)), atol=1e-10,
                                       err_msg=f"{name} y mismatch")
            np.testing.assert_allclose(joints_xyz[idx, 2], z_anchor, atol=1e-10,
                                       err_msg=f"{name} z mismatch")

    def test_cartesian_relative_to_cylindrical_parent(self):
        """
        heave_plate_base is a Cartesian joint positioned below col_keel.

        The relative offset uses the raw location of col_keel (before cyl->XYZ),
        so only the z dimension is inherited: heave_plate_base.z = z_offset + z_keel.
        x and y are unchanged (absolute Cartesian).
        """
        r_col = 51.75
        theta_col = 180.0
        z_keel = -20.0
        z_offset = -0.25

        joint_names = ["col_keel", "heave_plate_base"]
        locations = np.array([
            [r_col, theta_col, z_keel],   # col_keel (cylindrical)
            [0.0,   0.0,       z_offset],  # heave_plate_base (Cartesian, relative z)
        ])
        cylindrical = [True, False]
        relative = ["origin", "col_keel"]
        relative_dims = [[False, False, False], [False, False, True]]

        opts = make_floating_init_options(joint_names, cylindrical, relative, relative_dims)
        joints_xyz = self._run_aggregate_joints(opts, locations.copy())

        # col_keel XYZ (from cylindrical)
        np.testing.assert_allclose(joints_xyz[0, 0], r_col * np.cos(np.deg2rad(theta_col)), atol=1e-10)
        np.testing.assert_allclose(joints_xyz[0, 1], r_col * np.sin(np.deg2rad(theta_col)), atol=1e-10)
        np.testing.assert_allclose(joints_xyz[0, 2], z_keel, atol=1e-10)

        # heave_plate_base: x, y unchanged; z = z_offset + z_keel
        np.testing.assert_allclose(joints_xyz[1, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(joints_xyz[1, 1], 0.0, atol=1e-10)
        np.testing.assert_allclose(joints_xyz[1, 2], z_offset + z_keel, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
