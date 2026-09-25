from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/fire_robot_manipulation'))
from fire_robot_manipulation.confirmed_handle_track import ConfirmedHandleTrack


class TestConfirmedHandleTrack(unittest.TestCase):
    def test_low_score_never_acquires_a_new_handle(self):
        track=ConfirmedHandleTrack()
        for i in range(8):
            self.assertFalse(track.update([.5,.2,.8],.35,i*.2))

    def test_stable_confirmed_handle_survives_lower_nearfield_score(self):
        track=ConfirmedHandleTrack()
        self.assertTrue(track.update([.5,.2,.8],.8,0.))
        self.assertFalse(track.update([.501,.201,.8],.35,.2))
        self.assertTrue(track.update([.502,.202,.8],.35,.4))

    def test_nearby_different_object_is_rejected(self):
        track=ConfirmedHandleTrack()
        self.assertTrue(track.update([.5,.2,.8],.8,0.))
        self.assertFalse(track.update([.5,.4,.8],.9,.2))

    def test_duplicates_and_nonfinite_points_are_rejected(self):
        track=ConfirmedHandleTrack()
        self.assertTrue(track.update([.5,.2,.8],.8,0.))
        self.assertFalse(track.update([.5,.2,.8],.8,0.))
        self.assertFalse(track.update([float('nan'),.2,.8],.8,.2))


if __name__=='__main__': unittest.main()
