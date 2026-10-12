from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock,patch
import numpy as np
from builtin_interfaces.msg import Time
from fire_robot_manipulation.observed_door_angle_node import ObservedDoorAngle


class PanelSupportAccumulationTests(unittest.TestCase):
    def scan(self,stamp):
        sec=int(stamp)
        return NS(header=NS(stamp=Time(sec=sec,nanosec=round((stamp-sec)*1.e9)),frame_id='lidar'),
            ranges=[1.,1.1,1.2],angle_min=-.1,angle_increment=.1,range_min=.15,range_max=12.)

    def node(self,stamps):
        node=ObservedDoorAngle.__new__(ObservedDoorAngle)
        node.scans=[self.scan(stamp) for stamp in stamps]
        node.transform=Mock(return_value=(np.eye(3),np.zeros(3)))
        node.rng=np.random.default_rng(3)
        return node

    def fit(self):
        return dict(normal=np.array([1.,0.,0.]),center=np.array([.4,0.,.8]),
            segment_xy=np.array([[.4,-.1],[.4,.1]]))

    def test_fresh_distinct_scans_can_support_a_narrow_visual_patch(self):
        node=self.node([.9,.98])
        fit=self.fit(); segment=np.array([[.4,-.4],[.4,.4]])
        with patch('fire_robot_manipulation.observed_door_angle_node.fit_lidar_panel',
                   side_effect=[ValueError('lidar_panel_orientation_uncertain'),
                       dict(segment_edge_margin_m=.08)]) as lidar:
            with patch('fire_robot_manipulation.observed_door_angle_node.fused_panel_segment',
                       return_value=segment) as fusion:
                result,margin,source=node.panel_segment_support(fit,1.,.8)
        np.testing.assert_array_equal(result,segment)
        self.assertAlmostEqual(margin,.08+(.035+.45*.08)*.75)
        self.assertEqual(source,'rgbd_with_accumulated_agreeing_lidar_support')
        self.assertEqual(lidar.call_args.args[0].shape,(6,3))
        self.assertAlmostEqual(fusion.call_args.args[3],.1)

    def test_stale_future_and_duplicate_scans_cannot_support_accumulation(self):
        node=self.node([.8,.98,.98,1.01])
        fit=self.fit()
        with patch('fire_robot_manipulation.observed_door_angle_node.fit_lidar_panel',
                   side_effect=ValueError('lidar_panel_orientation_uncertain')) as lidar:
            result,margin,source=node.panel_segment_support(fit,1.,.8)
        self.assertEqual(lidar.call_count,1)
        np.testing.assert_array_equal(result,fit['segment_xy'])
        self.assertEqual(source,'rgbd')
        self.assertEqual(margin,.05)

    def test_disagreeing_independent_line_cannot_extend_visual_support(self):
        node=self.node([.9,.98])
        fit=self.fit()
        with patch('fire_robot_manipulation.observed_door_angle_node.fit_lidar_panel',
                   return_value=dict(segment_edge_margin_m=.08)):
            with patch('fire_robot_manipulation.observed_door_angle_node.fused_panel_segment',
                       side_effect=ValueError('RGB-D and LiDAR panel locations disagree')):
                result,margin,source=node.panel_segment_support(fit,1.,.8)
        np.testing.assert_array_equal(result,fit['segment_xy'])
        self.assertEqual(source,'rgbd')
        self.assertEqual(margin,.05)


if __name__=='__main__':
    unittest.main()
