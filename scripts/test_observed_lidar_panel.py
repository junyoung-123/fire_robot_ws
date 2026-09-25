import ast
import inspect
import math
import textwrap
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
import numpy as np
from fire_robot_manipulation.observed_lidar_panel import fit_lidar_panel
from fire_robot_manipulation.observed_door_angle_node import ObservedDoorAngle
from fire_robot_manipulation.robot_observation_manipulation_node import RobotObservationManipulation


class LidarPanelTests(unittest.TestCase):
    def test_noisy_panel_and_outliers(self):
        rng=np.random.default_rng(81)
        for angle in (0.,-.8,-2.1):
            n=np.array([math.cos(angle),math.sin(angle)])
            tangent=np.array([-n[1],n[0]]); center=np.array([.3,-.2])
            points=center+rng.uniform(-.4,.4,(500,1))*tangent+rng.normal(0,.03,(500,2))
            points=np.vstack([points,center+rng.uniform(-.7,.7,(80,2))])
            fit=fit_lidar_panel(points,center,n,rng)
            self.assertLess(math.acos(min(1.,abs(fit['normal'][:2]@n))),math.radians(2.))

    def test_unrelated_wall_not_accepted(self):
        rng=np.random.default_rng(4)
        points=np.column_stack((rng.uniform(-.3,.3,300),np.ones(300)*.3))
        with self.assertRaises(ValueError): fit_lidar_panel(points,[0.,0.],[1.,0.],rng)

    def test_insufficient_observation_rejected(self):
        with self.assertRaises(ValueError): fit_lidar_panel(np.zeros((10,2)),[0.,0.],[1.,0.],np.random.default_rng(0))

    def test_robot_wrist_cannot_overwrite_environment_angle(self):
        tree=ast.parse(textwrap.dedent(inspect.getsource(RobotObservationManipulation._sim_arm_state_callback)))
        writes=[n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute) and isinstance(n.ctx,ast.Store)]
        self.assertNotIn('_sim_door_position',writes)

    def observer_fixture(self):
        scans=[]
        for stamp in (1.,1.083):
            header=SimpleNamespace(frame_id='lidar_link',stamp=SimpleNamespace(
                sec=1,nanosec=round((stamp-1)*1.e9)))
            scans.append(SimpleNamespace(header=header,ranges=[1.]*60,
                angle_min=-.5,angle_increment=.01,range_min=.15,range_max=12.))
        node=SimpleNamespace(scans=scans,tracked_center=np.array([1.,0.,1.]),
            tracked_normal=np.array([1.,0.,0.]),rng=np.random.default_rng(0),
            tracker=SimpleNamespace(reference=0.,stamp=.9,update=Mock(return_value=-.1)),
            get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=1110000000)),
            transform=lambda frame,stamp:(np.eye(3),np.zeros(3)),pub=Mock(),report=Mock())
        fit=dict(normal=np.array([1.,0.,0.]),center=np.array([1.,0.]),
            residual_m=.01,inliers=60,support_std_m=.2,orientation_uncertainty_rad=.02)
        return node,fit

    def test_latest_scan_preferred_without_relabeling_timestamp(self):
        node,fit=self.observer_fixture()
        with patch('fire_robot_manipulation.observed_door_angle_node.fit_lidar_panel',return_value=fit) as fitter:
            self.assertTrue(ObservedDoorAngle.try_lidar(node))
        self.assertEqual(len(fitter.call_args.args[0]),60)
        self.assertAlmostEqual(node.tracked_stamp,1.083)
        self.assertEqual(node.report.call_args.kwargs['scan_window_sec'],0.)

    def test_accumulation_keeps_mean_measurement_time(self):
        node,fit=self.observer_fixture()
        with patch('fire_robot_manipulation.observed_door_angle_node.fit_lidar_panel',
                   side_effect=[ValueError('uncertain'),fit]):
            self.assertTrue(ObservedDoorAngle.try_lidar(node))
        self.assertAlmostEqual(node.tracked_stamp,1.0415)

    def test_uncertain_scans_publish_no_angle(self):
        node,_=self.observer_fixture()
        with patch('fire_robot_manipulation.observed_door_angle_node.fit_lidar_panel',
                   side_effect=ValueError('uncertain')):
            self.assertFalse(ObservedDoorAngle.try_lidar(node))
        node.pub.publish.assert_not_called()


if __name__=='__main__': unittest.main()
