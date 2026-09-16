import importlib.util
import sys
import unittest
from pathlib import Path

from geometry_msgs.msg import Vector3
from ros_gz_interfaces.msg import Contact, Contacts

path = Path(__file__).resolve().parents[1] / 'src/fire_robot_bringup/scripts/run_physical_contact_door_test.py'
spec = importlib.util.spec_from_file_location('contact_diagnostic_runner', path)
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


class ContactDiagnosticTests(unittest.TestCase):
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
