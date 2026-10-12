import unittest
import numpy as np
from fire_robot_manipulation.contact_feedback import temporal_scan_consensus
from fire_robot_navigation.self_scan_geometry import mask_echoes, mask_confirmed_self_noise


class ScanConsensusTest(unittest.TestCase):
    def test_reserve_tracks_only_gaussian_lower_bounds(self):
        rows=self.noisy_pool()
        result,count,uncertainty,reserve=temporal_scan_consensus(
            rows,return_uncertainty=True,return_reserve=True)
        original=temporal_scan_consensus(rows,return_uncertainty=True)
        np.testing.assert_allclose(result,original[0])
        self.assertEqual(count,original[1])
        self.assertEqual(uncertainty,original[2])
        np.testing.assert_allclose(result+reserve,np.median(rows,axis=0))
        self.assertTrue(np.all(reserve>0.))

    def test_foreground_returns_cannot_claim_gaussian_clearance_credit(self):
        for changed in (1,3):
            rows=self.noisy_pool()
            rows[-changed:,0]=.05 if changed==1 else .30
            result,_,_,reserve=temporal_scan_consensus(
                rows,return_uncertainty=True,return_reserve=True)
            self.assertEqual(reserve[0],0.)
            self.assertLessEqual(result[0],rows[-1,0])

    def test_unsupported_unknown_and_boundary_rays_have_no_credit(self):
        for rows in ([[.4],[.05]],[[8.],[7.],[9.],[6.],[3.],[2.],[.6],[.5],[.4]],
                     [[.5]]*8+[[float('nan')]]):
            _,_,_,reserve=temporal_scan_consensus(
                rows,return_uncertainty=True,return_reserve=True)
            self.assertTrue(np.all(reserve==0.))
        with self.assertRaises(ValueError):
            temporal_scan_consensus([[.4]],return_reserve=True)

    def noisy_pool(self):
        rows=np.tile((.4+np.linspace(-.05,.05,11))[:,None],(1,21))
        rows[:,0]=.4
        return rows

    def test_pooled_noise_avoids_overconfident_one_ray_innovation(self):
        rows=self.noisy_pool(); rows[-1,0]=.30
        result,_,margin=temporal_scan_consensus(rows,return_uncertainty=True)
        self.assertGreater(result[0],.30)
        self.assertLess(result[0],.4)
        self.assertGreater(margin,.03)

    def test_pooled_noise_still_preserves_a_large_new_obstacle(self):
        rows=self.noisy_pool(); rows[-1,0]=.05
        result,_,_=temporal_scan_consensus(rows,return_uncertainty=True)
        self.assertLessEqual(result[0],.05)

    def test_pooled_noise_still_preserves_three_closer_returns(self):
        rows=self.noisy_pool(); rows[-3:,0]=.30
        result,_,_=temporal_scan_consensus(rows,return_uncertainty=True)
        self.assertLessEqual(result[0],.30)

    def test_insufficient_history_does_not_claim_a_noise_margin(self):
        result,count,margin=temporal_scan_consensus([[.4],[.05]],return_uncertainty=True)
        self.assertIsNone(margin)
        self.assertEqual(count,0)
        self.assertEqual(result[0],.05)

    def test_insufficient_history_retains_raw_obstacle(self):
        result,count=temporal_scan_consensus([[.4],[.05]])
        self.assertEqual(result[0],.05)
        self.assertEqual(count,0)

    def test_noisy_ranges_are_conservative_not_an_average_free_ray(self):
        values=np.array([[.41,.42],[.38,.39],[.43,.44],[.36,.37],
                         [.40,.41],[.42,.43],[.39,.40],[.41,.42],[.25,.26]])
        result,count=temporal_scan_consensus(values)
        self.assertEqual(count,2)
        self.assertTrue(np.all(result<=np.median(values,axis=0)))

    def test_repeated_one_beam_obstacle_is_not_spatially_erased(self):
        rows=np.full((11,9),.5)
        rows[:,4]=.12
        result,_=temporal_scan_consensus(rows)
        self.assertLessEqual(result[4],.12)
        self.assertGreater(result[3],.4)

    def test_abrupt_close_obstacle_is_immediate(self):
        rows=np.full((11,3),.5)
        rows[-1,1]=.1
        result,_=temporal_scan_consensus(rows)
        self.assertLessEqual(result[1],.1)

    def test_new_sustained_closer_obstacle_is_retained(self):
        rows=np.array([[.40],[.48],[.42],[.46],[.44],[.41],[.39],[.40],
                       [.30],[.30],[.30]])
        result,_=temporal_scan_consensus(rows)
        self.assertLessEqual(result[0],.30)

    def test_unknown_latest_return_does_not_become_free(self):
        rows=np.full((9,2),.5)
        rows[-1,1]=np.nan
        result,_=temporal_scan_consensus(rows)
        self.assertTrue(np.isnan(result[1]))

    def test_foreground_background_switch_is_not_a_gaussian_noise_model(self):
        rows=np.array([[8.],[7.],[9.],[6.],[3.],[2.],[.6],[.5],[.4]])
        result,count=temporal_scan_consensus(rows)
        self.assertEqual(result[0],.4)
        self.assertEqual(count,0)

    def test_cad_confirmed_noisy_ray_stays_unknown(self):
        rows=np.array([[.51],[.47],[.55],[.44],[.50],[.53],[.45],[.52],[.318]])
        cleaned,_=mask_echoes(rows[-1],[.49])
        result=mask_confirmed_self_noise(cleaned,[.49],rows)
        self.assertTrue(np.isnan(result[0]))

    def test_repeated_real_closer_ray_before_cad_is_not_hidden(self):
        rows=np.array([[.51],[.47],[.55],[.44],[.50],[.53],[.20],[.20],[.20]])
        cleaned,_=mask_echoes(rows[-1],[.49])
        result=mask_confirmed_self_noise(cleaned,[.49],rows)
        self.assertEqual(result[0],.20)

    def test_large_closer_innovation_before_cad_is_immediate(self):
        rows=np.full((9,1),.49)
        rows[-1]=.1
        cleaned,_=mask_echoes(rows[-1],[.49])
        result=mask_confirmed_self_noise(cleaned,[.49],rows)
        self.assertEqual(result[0],.1)


if __name__=='__main__':
    unittest.main()
