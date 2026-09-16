#!/usr/bin/env python3
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from matplotlib.path import Path as PlotPath
from fire_robot_manipulation.sim_door_feedback import rotation_sample, model_metadata
from fire_robot_manipulation.manipulation_node import ManipulationNode
from check_validation_log import verified_door_topics


class SimDoorFeedbackTests(unittest.TestCase):
    def sample(self, angle, name='door_blue1'):
        return dict(header=dict(stamp=dict(sec='10', nsec=500000000)), pose=[
            dict(name=name, orientation=dict(z=math.sin(angle/2), w=math.cos(angle/2)))])

    def test_model_angle_is_measured_not_commanded(self):
        stamp, angle = rotation_sample(self.sample(.02), 'door_blue1', 0.)
        self.assertEqual(stamp, 10.5)
        self.assertAlmostEqual(angle, .02)

    def test_model_identity_and_orientation_validated(self):
        with self.assertRaises(ValueError):
            rotation_sample(self.sample(2.09, 'different_door'), 'door_blue1', 0.)
        sample = self.sample(0.)
        sample['pose'][0]['orientation']['w'] = float('nan')
        with self.assertRaises(ValueError):
            rotation_sample(sample, 'door_blue1', 0.)

    def test_log_success_command_cannot_substitute_angle_feedback(self):
        self.assertEqual(verified_door_topics('Gazebo door topic=/test, target_angle=2.094'), set())
        for angle in (.02, 2.01, -2.09):
            line = f'Door model feedback verified: topic=/test, model=door_blue1, angle={angle}, target=2.094'
            self.assertEqual(verified_door_topics(line), set())
        line = 'Door model feedback verified: topic=/test, model=door_blue1, angle=2.09, target=2.094'
        self.assertEqual(verified_door_topics(line), {'/test'})

    def test_wait_rejects_closed_door_stale_pose_and_wrong_direction(self):
        for angle, stale, expected in ((.02, False, False), (2.09, True, False),
                                        (-2.09, False, False), (2.09, False, True)):
            node = object.__new__(ManipulationNode)
            node._sim_model_feedback = {'/test': {'model': 'door_blue1'}}
            node._sim_feedback_timeout = 1.
            node._sim_door_tolerance = .04
            clock = [10.]
            def sample(_):
                clock[0] += .25
                return (9. if stale else clock[0]), angle
            logger = NS(info=lambda _: None, warn=lambda _: None, error=lambda _: None)
            module = 'fire_robot_manipulation.manipulation_node'
            with patch.object(node, 'get_clock', return_value=NS(now=lambda: NS(nanoseconds=int(clock[0]*1.e9)))), \
                    patch.object(node, 'get_logger', return_value=logger), \
                    patch(module+'.rclpy.ok', return_value=True), \
                    patch(module+'.time.monotonic', side_effect=lambda: clock[0]), \
                    patch.object(node, '_publish_sim_command', return_value=True), \
                    patch(module+'.time.sleep'), patch(module+'.read_rotation', side_effect=sample):
                self.assertEqual(node._wait_for_sim_model_door_target('/test', 2.094), expected)

    def test_doors_can_sweep_outward_without_intersecting_wall(self):
        root = Path(__file__).resolve().parents[1] / 'src/fire_robot_bringup/worlds'
        names = ['obstacle_wall_doors_v5.world', 'obstacle_door_layout_alt_v1.world',
                 'obstacle_door_layout_world3_v1.world', 'obstacle_all_blue_world5_v1.world']
        def shape(points):
            return PlotPath(points + [points[0]], closed=True)
        for name in names:
            world = ET.parse(root/name).find('world')
            self.assertTrue(model_metadata(root/name))
            walls = []
            for wall in world.findall('model'):
                if wall.get('name') not in ('wall_north', 'wall_south'):
                    continue
                p = [float(x) for x in wall.findtext('pose').split()]
                for col in wall.findall('link/collision'):
                    cp = [float(x) for x in col.findtext('pose', '0 0 0 0 0 0').split()]
                    sx, sy, sz = [float(x) for x in col.findtext('geometry/box/size').split()]
                    if p[2] + cp[2] - sz/2 >= 2.:
                        continue
                    x, y = p[0]+cp[0], p[1]+cp[1]
                    walls.append(shape([(x-sx/2,y-sy/2),(x+sx/2,y-sy/2),
                                        (x+sx/2,y+sy/2),(x-sx/2,y+sy/2)]))
            for door in world.findall('model'):
                if not door.get('name','').startswith('door_blue'):
                    continue
                x, y = [float(t) for t in door.findtext('pose').split()[:2]]
                sx, sy, _ = [float(t) for t in door.findtext('link/collision/geometry/box/size').split()]
                for degrees in range(0,121,5):
                    angle = math.radians(degrees) * (1 if y>0 else -1)
                    c, s = math.cos(angle), math.sin(angle)
                    points = [(x-sx/2+c*a-s*b, y+s*a+c*b)
                              for a,b in ((0,-sy/2),(sx,-sy/2),(sx,sy/2),(0,sy/2))]
                    self.assertFalse(any(shape(points).intersects_path(w, filled=True) for w in walls),
                                     f'{name} {door.get("name")} {degrees} deg')


if __name__ == '__main__':
    unittest.main()
