"""Geometry-estimator tests, not physical opening success evidence."""
import math
import unittest
import numpy as np
from fire_robot_manipulation.contact_feedback import observed_contact_curvature


def samples(span=.45, radius=.73, offset=(3.,-2.), outliers=False):
    angles = np.linspace(0.,-span,50)
    times = np.linspace(0.,12.,50)**1.1
    points = np.array(offset)+np.column_stack((radius*np.cos(angles),radius*np.sin(angles)))
    points += np.random.default_rng(57).normal(0.,.0006,points.shape)
    if outliers:
        points[[10,25]] += [.08,-.10]
    return np.column_stack((times,angles,points))


class ContactGeometryTests(unittest.TestCase):
    def test_sensor_geometry_recovers_curvature_with_nonuniform_timing(self):
        fit = observed_contact_curvature(samples())
        self.assertIsNotNone(fit)
        self.assertAlmostEqual(fit['curvature_per_m'],-1./.73,delta=.025)

    def test_stationary_ik_waits_do_not_change_geometry(self):
        rows = samples()
        rows[:,0] = np.arange(len(rows))**2
        self.assertAlmostEqual(observed_contact_curvature(rows)['radius_m'],.73,delta=.01)

    def test_short_motion_is_not_a_radius_measurement(self):
        self.assertIsNone(observed_contact_curvature(samples(span=.03)))

    def test_wrong_frame_or_slipping_grasp_is_not_a_path(self):
        rows = samples()
        rows[:,2] += np.sin(np.linspace(0.,2.*math.pi,len(rows)))*.1
        self.assertIsNone(observed_contact_curvature(rows))

    def test_sparse_outliers_do_not_select_a_false_curvature(self):
        fit = observed_contact_curvature(samples(outliers=True))
        self.assertIsNotNone(fit)
        self.assertAlmostEqual(fit['radius_m'],.73,delta=.015)

    def test_map_origin_is_not_an_input_to_the_estimate(self):
        first = observed_contact_curvature(samples())
        second = observed_contact_curvature(samples(offset=(100003.,-100002.)))
        self.assertAlmostEqual(first['radius_m'],second['radius_m'],places=6)


if __name__ == '__main__':
    unittest.main()
