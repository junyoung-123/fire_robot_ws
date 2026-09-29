"""Synthetic evaluator fixtures, not Gazebo success evidence."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from check_observed_navigation import evaluate, exit_jamb_footprints, structure_footprints


class CheckerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.evidence = self.root/'evidence'
        self.evidence.mkdir()
        self.world = self.root/'fixture.world'
        self.world.write_text('''<sdf><world>
          <model name="door_blue1"><pose>2 -1 1 0 0 0</pose></model>
          <model name="door_red1"><pose>4 1 1 0 0 0</pose></model>
          <model name="exit_green"><pose>6 0 1 0 0 0</pose></model>
          <model name="obstacle_fixture"><pose>4 -1 0 0 0 0</pose>
            <link><collision><geometry><box><size>.4 .4 .4</size></box></geometry></collision></link>
          </model></world></sdf>''')
        self.write_json(self.root/'result.json', {'status': 'MISSION_COMPLETE'})
        self.write_json(self.evidence/'ros_graph.json', [{'node': 'slam_toolbox'}])
        self.audit = {'map_publishers': ['slam_toolbox'],
                      'odom_publishers': ['gz_selected_odometry_bridge'],
                      'evaluation_truth_subscribers': ['observed_navigation_recorder']}
        self.write_json(self.evidence/'input_audit.json', self.audit)
        event = {'kind': 'visit', 'payload': {'sim_time': 1, 'door_id': 'observed_1'}}
        (self.evidence/'events.jsonl').write_text(json.dumps(event)+'\n')
        self.set_truth('0,0,0,0\n1,2,0,-1.57079632679\n2,6.5,0,0\n')

    def write_json(self, path, value):
        path.write_text(json.dumps(value))

    def set_truth(self, data):
        (self.evidence/'evaluation_poses.csv').write_text('sim_time,x,y,yaw\n'+data)

    def check(self):
        with contextlib.redirect_stdout(io.StringIO()):
            passed = evaluate(self.root, self.world)
        return passed, json.loads((self.root/'independent_check.json').read_text())

    def test_valid_visit_and_exit(self):
        passed, report = self.check()
        self.assertTrue(passed)
        self.assertFalse(report['physical_opening_tested'])

    def test_wrong_color_not_accepted(self):
        self.set_truth('0,0,0,0\n1,4,0,1.57079632679\n2,6.5,0,0\n')
        passed, report = self.check()
        self.assertFalse(passed)
        self.assertEqual(report['matches'][0]['color'], 'red')

    def test_obstacle_overlap_not_accepted(self):
        self.set_truth('0,0,0,0\n1,2,0,-1.57079632679\n1.5,4,-1,0\n2,6.5,0,0\n')
        passed, report = self.check()
        self.assertFalse(passed)
        self.assertEqual(report['navigation_footprint_overlap_obstacles'], ['obstacle_fixture'])

    def test_ground_truth_control_subscriber_not_accepted(self):
        self.audit['evaluation_truth_subscribers'].append('state_machine_node')
        self.write_json(self.evidence/'input_audit.json', self.audit)
        passed, report = self.check()
        self.assertFalse(passed)
        self.assertIn('evaluation truth isolation not established', report['failures'])

    def test_wall_overlap_not_accepted(self):
        root = ET.parse(self.world)
        model = ET.SubElement(root.getroot().find('world'), 'model', name='wall_fixture')
        ET.SubElement(model, 'pose').text = '5 0 1 0 0 0'
        link = ET.SubElement(model, 'link')
        link.append(ET.fromstring('<collision name="panel"><geometry><box><size>.2 4 2</size></box></geometry></collision>'))
        root.write(self.world)
        self.set_truth('0,0,0,0\n1,2,0,-1.57079632679\n1.5,5,0,0\n2,6.5,0,0\n')
        passed, report = self.check()
        self.assertFalse(passed)
        self.assertEqual(report['wall_panel_boxes_evaluated'], 1)
        self.assertIn('wall_fixture/panel', report['navigation_footprint_overlap_obstacles'])

    def test_structure_pose_composition_and_overhead_filter(self):
        root = ET.fromstring('''<sdf><world><model name="wall_fixture">
          <pose>1 2 1 0 0 1.5707963267948966</pose><link><pose>1 0 0 0 0 0</pose>
          <collision name="panel"><pose>1 0 0 0 0 0</pose><geometry><box><size>2 .2 2</size></box></geometry></collision>
          <collision name="lintel"><pose>0 0 1 0 0 0</pose><geometry><box><size>2 .2 .2</size></box></geometry></collision>
          </link></model></world></sdf>''')
        shapes = structure_footprints(root)
        self.assertEqual(len(shapes), 1)
        self.assertAlmostEqual(shapes[0][1][:, 0].mean(), 1.)
        self.assertAlmostEqual(shapes[0][1][:, 1].mean(), 4.)

    def test_open_exit_jambs_included_but_overhead_lintel_excluded(self):
        root = ET.fromstring('''<sdf><world><model name="exit_green">
        <pose>6 0 1 0 0 0</pose><link>
        <collision name="jamb_left"><pose>0 .66 0 0 0 0</pose>
        <geometry><box><size>.12 .12 2</size></box></geometry></collision>
        <collision name="lintel"><pose>0 0 1.06 0 0 0</pose>
        <geometry><box><size>.12 1.44 .12</size></box></geometry></collision>
        </link></model></world></sdf>''')
        jambs = exit_jamb_footprints(root)
        self.assertEqual(len(jambs), 1)
        self.assertAlmostEqual(jambs[0][1][:, 0].mean(), 6.)
        self.assertAlmostEqual(jambs[0][1][:, 1].mean(), .66)


if __name__ == '__main__':
    unittest.main()
