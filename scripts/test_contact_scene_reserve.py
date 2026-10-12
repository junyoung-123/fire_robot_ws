import unittest
import numpy as np
from fire_robot_manipulation.contact_feedback import (
    contact_scene_planning_margin, scanned_body_constraints)
from fire_robot_manipulation.piper_actual_kinematics import PIPER_BODY_OUTLINE


class ContactSceneReserveTests(unittest.TestCase):
    def test_embedded_lower_bound_reserve_is_not_counted_twice(self):
        credit=np.array([0.,.02,.05,.10,.15])
        margin=contact_scene_planning_margin(.09,.015,credit)
        np.testing.assert_allclose(margin,[.12,.10,.07,.045,.045])
        self.assertTrue(np.all(credit+margin >= .12-1.e-12))
        self.assertTrue(np.all(margin >= .045-1.e-12))

    def test_vector_barriers_match_scalar_for_equal_reserve(self):
        points=np.array([[.40,-.23],[.42,-.24],[.44,-.25]])
        scalar=scanned_body_constraints(points,PIPER_BODY_OUTLINE,margin=.08)
        vector=scanned_body_constraints(points,PIPER_BODY_OUTLINE,margin=np.full(3,.08))
        for first,second in zip(scalar,vector):
            np.testing.assert_allclose(first,second)

    def test_invalid_credit_and_vector_margin_rejected(self):
        points=np.array([[.40,-.23],[.42,-.24],[.44,-.25]])
        for credit in ([-.01],[float('nan')],[[.01]],.01):
            with self.assertRaises(ValueError):
                contact_scene_planning_margin(.09,.015,credit)
        for margin in ([.08,.08],[[.08]*3],[.08,float('inf'),.08],[.08,-.01,.08]):
            with self.assertRaises(ValueError):
                scanned_body_constraints(points,PIPER_BODY_OUTLINE,margin=margin)

    def test_measured_noise_reserves_space_before_the_unchanged_stop_boundary(self):
        self.assertAlmostEqual(contact_scene_planning_margin(.09,.015),.12)
        self.assertAlmostEqual(contact_scene_planning_margin(.005,.015),.045)
        self.assertAlmostEqual(contact_scene_planning_margin(None,.015),.045)

    def test_invalid_uncertainty_is_not_silently_dropped(self):
        for noise,tolerance in ((-.01,.015),(float('nan'),.015),(.05,0.),(.05,float('inf'))):
            with self.assertRaises(ValueError):
                contact_scene_planning_margin(noise,tolerance)

    def test_fixed_jamb_constraint_requests_recovery_before_a_noisy_stop(self):
        points=np.array([[.40,-.23],[.42,-.24],[.44,-.25]])
        old_rows,old_bounds,_=scanned_body_constraints(points,PIPER_BODY_OUTLINE)
        new_rows,new_bounds,_=scanned_body_constraints(points,PIPER_BODY_OUTLINE,
            margin=contact_scene_planning_margin(.09,.015))
        np.testing.assert_allclose(new_rows,old_rows)
        self.assertTrue(np.all(new_bounds < old_bounds))
        self.assertTrue(np.any(new_bounds < 0.))
        self.assertTrue(np.all(old_bounds > 0.))


if __name__=='__main__':
    unittest.main()
