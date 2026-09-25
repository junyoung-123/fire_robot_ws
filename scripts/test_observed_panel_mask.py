import unittest
import numpy as np
from fire_robot_manipulation.observed_panel_mask import panel_color_support


class PanelMaskTests(unittest.TestCase):
    def test_neutral_wall_is_not_a_panel(self):
        self.assertFalse(panel_color_support(np.full((20,20,3),160,dtype=np.uint8),'blue').any())

    def test_edges_and_other_color_are_rejected(self):
        image=np.full((24,24,3),160,dtype=np.uint8)
        image[4:20,4:20]=[210,25,15]
        mask=panel_color_support(image,'blue')
        self.assertEqual(int(mask.sum()),144)
        self.assertFalse(panel_color_support(image,'red').any())

    def test_mirror_equivariance(self):
        image=np.zeros((24,32,3),dtype=np.uint8)
        image[3:22,2:20]=[140,10,20]
        np.testing.assert_array_equal(panel_color_support(image[:,::-1],'blue'),
                                      panel_color_support(image,'blue')[:,::-1])

    def test_unknown_color_fails_closed(self):
        with self.assertRaises(ValueError):
            panel_color_support(np.zeros((20,20,3)),'unknown')


if __name__=='__main__': unittest.main()
