import unittest
import numpy as np

from fire_robot_manipulation.contact_feedback import measured_press_target


class MeasuredPressTargetTests(unittest.TestCase):
    def test_follows_measured_lateral_arc_without_a_supplied_pivot(self):
        goal=measured_press_target([.6,.05,.76],[.6,.025,.73],.035,.11)
        np.testing.assert_allclose(goal,[.6,.025,.725])

    def test_does_not_command_normal_extension(self):
        actual=[.603,.03,.72]
        goal=measured_press_target([.6,.05,.76],actual,.04,.11)
        np.testing.assert_array_equal(goal[:2],actual[:2])

    def test_excessive_travel_or_invalid_feedback_stops(self):
        for actual,depth in (([.6,.20,.73],.04),([.6,.05,.73],.12),
                              ([float('nan'),.05,.73],.04)):
            with self.assertRaises(ValueError):
                measured_press_target([.6,.05,.76],actual,depth,.11)


if __name__ == '__main__':
    unittest.main()
