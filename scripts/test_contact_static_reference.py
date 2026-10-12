import unittest
import numpy as np
from fire_robot_manipulation.contact_scan_registration import temporal_static_reference


class TemporalStaticReferenceTests(unittest.TestCase):
    def samples(self):
        rng=np.random.default_rng(7)
        x=np.linspace(-2.,2.,120)
        truth=np.column_stack((x,np.full(len(x),2.)))
        return [(i*.1,np.zeros(3),truth+rng.normal(0.,.035,truth.shape)) for i in range(19)],truth

    def test_stationary_original_measurements_reduce_reference_noise(self):
        rows,truth=self.samples()
        points=temporal_static_reference(rows)
        self.assertEqual(points.shape,truth.shape)
        self.assertLess(np.sqrt(np.mean((points-truth)**2)),.013)
        self.assertGreater(np.sqrt(np.mean((rows[-1][2]-truth)**2)),.025)

    def test_unknown_or_changing_beams_never_create_reference_support(self):
        rows,_=self.samples()
        for i,(_,_,cloud) in enumerate(rows):
            cloud[0]=np.nan
            cloud[1]=[i*.2,2.]
            if i<6: cloud[2]=np.nan
        result=temporal_static_reference(rows)
        self.assertEqual(len(result),117)

    def test_robot_motion_and_rotation_are_not_averaged_as_static_geometry(self):
        for axis,amount in ((0,.006),(2,.004)):
            rows,_=self.samples()
            rows[0][1][axis]=amount
            with self.assertRaisesRegex(ValueError,'stationary robot'):
                temporal_static_reference(rows)

    def test_duplicate_insufficient_or_short_history_has_no_fresh_baseline(self):
        rows,_=self.samples()
        for values in (rows[:10],[(1.,p,c) for _,p,c in rows],
                       [(i*.01,p,c) for i,(_,p,c) in enumerate(rows)]):
            with self.assertRaisesRegex(ValueError,'Awaiting'):
                temporal_static_reference(values)

    def test_old_samples_and_nonfinite_poses_are_rejected(self):
        rows,_=self.samples()
        with self.assertRaises(ValueError):
            temporal_static_reference(rows,window=.5)
        rows[0][1][0]=np.nan
        with self.assertRaisesRegex(ValueError,'Invalid'):
            temporal_static_reference(rows)


if __name__=='__main__':
    unittest.main()
