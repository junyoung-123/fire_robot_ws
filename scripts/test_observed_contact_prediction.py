import unittest
import numpy as np
from fire_robot_manipulation.contact_feedback import predicted_contact_rotation


class ObservedContactPredictionTests(unittest.TestCase):
    def test_unknown_stationary_rates_do_not_invent_a_door_turn(self):
        np.testing.assert_allclose(predicted_contact_rotation(np.eye(3),0.,0.),np.eye(3))

    def test_equal_measured_door_and_chassis_rates_cancel(self):
        np.testing.assert_allclose(predicted_contact_rotation(np.eye(3),-.06,-.06),np.eye(3))

    def test_only_relative_observed_rotation_is_predicted(self):
        rotation=predicted_contact_rotation(np.eye(3),-.06,-.02)
        self.assertAlmostEqual(np.arctan2(rotation[1,0],rotation[0,0]),-.02)

    def test_invalid_or_excessive_rates_stop(self):
        for rate in (.3,float('nan')):
            with self.assertRaises(ValueError):
                predicted_contact_rotation(np.eye(3),rate,0.)


if __name__=='__main__':
    unittest.main()
