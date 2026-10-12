import math
import unittest
import xml.etree.ElementTree as ET
import numpy as np
from world_contact_clearance import check_sweeps,outer_wall_face,rectangle,overlaps


class WorldContactClearanceTests(unittest.TestCase):
    def fixture(self,hinge_y=1.86):
        return ET.fromstring(f'''<world>
          <model name="wall_north"><pose>6.5 2 1.25 0 0 0</pose><link name="link">
            <collision name="wall_end"><pose>8.7525 0 0 0 0 0</pose>
              <geometry><box><size>5.495 .2 2.5</size></box></geometry></collision>
          </link></model>
          <model name="door_blue3"><pose>12.4 {hinge_y} 0 0 0 {math.pi/2}</pose>
            <link name="panel"><pose>0 .4 1 0 0 0</pose><collision name="panel_collision">
              <geometry><box><size>.06 .8 1.98</size></box></geometry></collision></link>
            <joint name="door_hinge"><pose>0 -.4 0 0 0 0</pose><axis><limit>
              <lower>-2.25</lower><upper>.05</upper></limit></axis></joint>
          </model></world>''')

    def test_reproduces_original_third_door_wall_stop(self):
        result=check_sweeps(self.fixture())
        self.assertFalse(result['pass_clearance'])
        span=result['doors'][0]['wall_intersections']['wall_north::wall_end']
        self.assertLess(span[1],-105.)
        self.assertGreater(span[1],-110.)

    def test_outer_face_mount_clears_entire_joint_range(self):
        world=self.fixture();face=outer_wall_face(world,1)
        self.assertAlmostEqual(face,2.1)
        result=check_sweeps(self.fixture(face))
        self.assertTrue(result['pass_clearance'])
        self.assertGreater(result['doors'][0]['continuous_sampling_margin_m'],0.)

    def test_mount_fix_does_not_remove_or_resize_wall(self):
        old=self.fixture();new=self.fixture(2.1)
        self.assertEqual(ET.tostring(old.find('model')),ET.tostring(new.find('model')))

    def test_rotated_south_fixture_has_same_clearance_result(self):
        for y,expected in ((1.86,False),(2.1,True)):
            world=self.fixture(y)
            for model in world.findall('model'):
                values=list(map(float,model.findtext('pose').split()))
                values[0]*=-1;values[1]*=-1;values[5]+=math.pi
                model.find('pose').text=' '.join(map(str,values))
                if model.get('name')=='wall_north':model.set('name','wall_south')
            self.assertAlmostEqual(outer_wall_face(world,-1),-2.1)
            self.assertEqual(check_sweeps(world)['pass_clearance'],expected)

    def test_sat_uses_oriented_boxes_not_only_centres(self):
        first=rectangle(np.zeros(3),[1.,1.,1.],0.)
        self.assertTrue(overlaps(first,rectangle([.8,0.,0.],[1.,1.,1.],math.pi/4)))
        self.assertFalse(overlaps(first,rectangle([2.,0.,0.],[1.,1.,1.],math.pi/4)))

    def test_lintel_above_leaf_does_not_block_sweep(self):
        world=self.fixture()
        wall=world.find('model');wall.find('pose').text='6.5 2 3.5 0 0 0'
        self.assertTrue(check_sweeps(world)['pass_clearance'])

    def test_missing_walls_and_invalid_sampling_fail_closed(self):
        with self.assertRaises(ValueError):outer_wall_face(ET.fromstring('<world/>'),1)
        with self.assertRaises(ValueError):check_sweeps(self.fixture(),step_deg=0.)


if __name__=='__main__':unittest.main()
