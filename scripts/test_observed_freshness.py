import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/fire_robot_manipulation'))
from fire_robot_manipulation.observed_freshness import ObservedFreshness


class TestFreshness(unittest.TestCase):
    def test_slow_simulation_keeps_fresh_measurement(self):
        gate=ObservedFreshness(True)
        self.assertTrue(gate.valid(135.4,10.,135.40,10.))
        self.assertTrue(gate.valid(135.4,10.,135.543,11.018))

    def test_frozen_simulation_cannot_mask_sensor_loss(self):
        gate=ObservedFreshness(True)
        self.assertTrue(gate.valid(10.,1.,10.,1.))
        self.assertFalse(gate.valid(10.,2.1,10.,2.1))

    def test_new_receipt_does_not_make_old_image_fresh(self):
        self.assertFalse(ObservedFreshness(True).valid(10.,2.,10.3,2.))

    def test_clock_rewind_and_future_image_rejected(self):
        gate=ObservedFreshness(True)
        self.assertTrue(gate.valid(10.,1.,10.,1.))
        self.assertFalse(gate.valid(9.,2.,9.,2.))
        self.assertFalse(gate.valid(10.,2.,9.,2.))

    def test_hardware_receipt_timeout_is_not_relaxed(self):
        self.assertFalse(ObservedFreshness(False).valid(10.,1.,10.1,2.01))

    def test_simulation_transport_watchdog(self):
        self.assertFalse(ObservedFreshness(True).valid(10.,1.,10.1,4.01))


if __name__=='__main__': unittest.main()
