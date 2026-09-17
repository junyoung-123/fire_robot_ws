"""Prevent calibrated robot outlines from silently omitting physical sensors."""
from pathlib import Path
import struct
import unittest
import xml.etree.ElementTree as ET
import numpy as np
import xacro
from fire_robot_manipulation.piper_actual_kinematics import PIPER_BODY_OUTLINE, _transform


class BodyEnvelopeTests(unittest.TestCase):
    def test_outline_contains_all_non_arm_collision_geometry(self):
        description = Path(__file__).resolve().parents[1]/'src/fire_robot_description'
        root = ET.fromstring(xacro.process_file(str(description/'urdf/fire_robot_actual_piper.urdf.xacro')).toxml())
        transforms = {'base_link': np.eye(4)}
        pending = list(root.findall('joint'))
        for _ in range(len(pending)):
            for joint in pending[:]:
                parent = joint.find('parent').get('link')
                if parent not in transforms:
                    continue
                origin = joint.find('origin')
                xyz = np.fromstring(origin.get('xyz','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
                rpy = np.fromstring(origin.get('rpy','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
                transforms[joint.find('child').get('link')] = transforms[parent]@_transform(xyz,rpy)
                pending.remove(joint)
        names = ['base_link','top_structure','camera_bracket_link','camera_link',
                 'lidar_link','depth_camera_link','camera_front_left_link','camera_front_right_link']
        names += [f'{end}_{side}_wheel' for end in ('front','rear') for side in ('left','right')]
        for name in names:
            link = root.find(f"link[@name='{name}']")
            self.assertIsNotNone(link)
            for collision in link.findall('collision'):
                origin = collision.find('origin')
                xyz = np.fromstring(origin.get('xyz','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
                rpy = np.fromstring(origin.get('rpy','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
                transform = transforms[name]@_transform(xyz,rpy)
                shape = list(collision.find('geometry'))[0]
                if shape.tag == 'mesh':
                    path = description/shape.get('filename').split('package://fire_robot_description/')[1]
                    raw = path.read_bytes()
                    count = struct.unpack('<I',raw[80:84])[0]
                    self.assertEqual(len(raw),84+count*50)
                    dtype = np.dtype([('normal','<f4',3),('vertices','<f4',(3,3)),('attr','<u2')])
                    points = np.frombuffer(raw[84:],dtype)['vertices'].reshape(-1,3).astype(float)
                    points *= np.fromstring(shape.get('scale','1 1 1'),sep=' ')
                else:
                    if shape.tag == 'box':
                        half = np.fromstring(shape.get('size'),sep=' ')/2.
                    elif shape.tag == 'cylinder':
                        half = np.array([float(shape.get('radius'))]*2+[float(shape.get('length'))/2.])
                    else:
                        self.fail('Unreviewed collision shape: '+shape.tag)
                    points = np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])*half
                points = points@transform[:3,:3].T+transform[:3,3]
                self.assertTrue(np.all(points[:,:2] >= PIPER_BODY_OUTLINE.min(0)-1.e-6), name)
                self.assertTrue(np.all(points[:,:2] <= PIPER_BODY_OUTLINE.max(0)+1.e-6), name)
