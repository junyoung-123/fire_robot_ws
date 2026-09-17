import importlib.util
import sys
import unittest
from pathlib import Path

from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Twist
from types import SimpleNamespace
from ros_gz_interfaces.msg import Contact, Contacts

path = Path(__file__).resolve().parents[1] / 'src/fire_robot_bringup/scripts/run_physical_contact_door_test.py'
spec = importlib.util.spec_from_file_location('contact_diagnostic_runner', path)
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


class ContactDiagnosticTests(unittest.TestCase):
    def test_safe_velocity_stop_is_not_lost_to_throttling(self):
        probe = object.__new__(runner.ContactProbe)
        probe.record = runner.ContactRecord()
        probe._stamp = lambda: 1.
        probe.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=10_000_000_000))
        msg = Twist()
        msg.linear.x = .02
        probe._velocity_cb(msg, '/cmd_vel_safe')
        msg.linear.x = 0.
        probe._velocity_cb(msg, '/cmd_vel_safe')
        self.assertEqual([x['linear_x'] for x in probe.record.command_samples], [.02,0.])
        self.assertEqual(probe.record.command_samples[-1]['sim_time_sec'],10.)

    def test_fixed_strike_camera_contact_is_a_failure(self):
        finger = dict(part='lever',phase='HANDLE_HELD_BASE_OPEN',contacts=[dict(
            collision1='door::lever::collision',collision2='fire_robot::gripper_finger_left::collision')])
        strike = dict(part='strike',phase='HANDLE_HELD_BASE_OPEN',contacts=[dict(
            collision1='door::frame::latch_strike_collision',collision2='fire_robot::camera_front_left_link::collision')])
        self.assertFalse(runner.audit_handle_held_contacts([finger]*3+[strike])['pass'])

    def test_held_hand_mode_requires_finger_contacts_not_body_contacts(self):
        finger = dict(part='lever', phase='HANDLE_HELD_BASE_OPEN', contacts=[dict(
            collision1='door::lever::lever_collision',
            collision2='fire_robot::gripper_finger_right::collision')])
        self.assertFalse(runner.audit_handle_held_contacts([])['pass'])
        self.assertTrue(runner.audit_handle_held_contacts([finger]*3)['pass'])
        panel = self.panel_contact('base_footprint::base_link_collision', 'HANDLE_HELD_BASE_OPEN')
        self.assertFalse(runner.audit_handle_held_contacts([finger]*3+[panel])['pass'])

    def panel_contact(self, part, phase='BASE_PUSH_OPEN'):
        return dict(part='panel', phase=phase, contacts=[dict(
            collision1='door::panel::panel_collision', collision2='fire_robot::'+part)])

    def test_fixed_joint_lumping_does_not_hide_camera_contact(self):
        samples = [self.panel_contact('base_footprint::base_link_collision_0'),
                   self.panel_contact('base_footprint::base_footprint_fixed_joint_lump__camera_front_right_link_collision_5')]
        audit = runner.audit_panel_push_contacts(samples)
        self.assertFalse(audit['pass'])
        self.assertEqual(len(audit['forbidden_contact_parts']), 1)

    def test_no_samples_or_unknown_body_part_cannot_qualify(self):
        self.assertFalse(runner.audit_panel_push_contacts([])['pass'])
        self.assertFalse(runner.audit_panel_push_contacts([self.panel_contact('unknown::collision')])['pass'])

    def test_recorded_bumper_contact_can_qualify_but_not_wrong_phase(self):
        part = 'base_footprint::front_push_bumper_collision'
        self.assertTrue(runner.audit_panel_push_contacts([self.panel_contact(part)])['pass'])
        self.assertFalse(runner.audit_panel_push_contacts([self.panel_contact(part, 'GRASP_HANDLE')])['pass'])

    def test_contact_names_positions_and_empty_sample_are_distinct(self):
        probe = object.__new__(runner.ContactProbe)
        probe.record = runner.ContactRecord()
        probe._stamp = lambda: 1.0
        msg = Contacts()
        msg.header.stamp.sec = 1
        probe._contacts_cb(msg, 'lever')
        self.assertEqual(probe.record.contact_samples[-1]['contacts'], [])
        contact = Contact()
        contact.collision1.name = 'door::lever::lever_collision'
        contact.collision2.name = 'robot::finger::collision'
        contact.positions = [Vector3(x=1.0, y=2.0, z=3.0)]
        contact.depths = [.001]
        msg.contacts = [contact]
        msg.header.stamp.sec = 2
        probe._contacts_cb(msg, 'lever')
        sample = probe.record.contact_samples[-1]['contacts'][0]
        self.assertEqual(sample['positions'], [[1., 2., 3.]])
        self.assertEqual(sample['collision2'], 'robot::finger::collision')
        self.assertEqual(sample['max_depth_m'], .001)
        self.assertEqual(probe.record.contact_messages['lever'], 2)


if __name__ == '__main__':
    unittest.main()
