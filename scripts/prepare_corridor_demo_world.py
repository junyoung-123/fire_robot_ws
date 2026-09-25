#!/usr/bin/env python3
"""Add a real wall opening around the preserved contact fixture, for filming."""
from pathlib import Path
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parents[1]
path = root / 'src/fire_robot_bringup/worlds/physical_contact_door_test.world'
tree = ET.parse(path)
world = tree.getroot().find('world')

def box(name, xyz, size, color):
    model = ET.SubElement(world, 'model', name=name)
    ET.SubElement(model, 'static').text = 'true'
    ET.SubElement(model, 'pose').text = ' '.join(map(str, [*xyz, 0, 0, 0]))
    link = ET.SubElement(model, 'link', name='link')
    for tag in ('collision', 'visual'):
        part = ET.SubElement(link, tag, name=tag)
        geom = ET.SubElement(ET.SubElement(part, 'geometry'), 'box')
        ET.SubElement(geom, 'size').text = ' '.join(map(str, size))
        if tag == 'visual':
            material = ET.SubElement(part, 'material')
            for channel in ('ambient', 'diffuse'):
                ET.SubElement(material, channel).text = color

# Geometry belongs to the test scene only. Control never reads these positions.
for model in list(world.findall('model')):
    if model.get('name', '').startswith('demo_'):
        world.remove(model)
box('demo_wall_lower', (-2.47, -2.8, 1.05), (.16, 4.5, 2.1), '.70 .73 .75 1')
box('demo_wall_upper', (-2.47, 2.775, 1.05), (.16, 4.55, 2.1), '.70 .73 .75 1')
box('demo_door_lintel', (-2.47, -.025, 2.1), (.16, 1.05, .18), '.70 .73 .75 1')
box('demo_opposite_wall', (-5.5, 0, 1.05), (.16, 10.1, 2.1), '.82 .83 .84 1')
box('demo_room_back', (.3, 0, 1.05), (.16, 4.6, 2.1), '.79 .80 .82 1')
box('demo_room_end', (-1.0, -2.3, 1.05), (2.75, .16, 2.1), '.79 .80 .82 1')
box('demo_background_red_door', (-2.56, -3.1, 1.0), (.045, .82, 2.0), '.7 .06 .06 1')
box('demo_background_blue_door', (-2.56, 3.1, 1.0), (.045, .82, 2.0), '.04 .10 .65 1')
for size in world.findall("./model[@name='ground']/link/*/geometry/plane/size"):
    size.text = '20 20'
for model_name in ('proof_perspective_camera', 'proof_overhead_camera', 'proof_handle_camera'):
    sensor = world.find(f"./model[@name='{model_name}']/link/sensor")
    sensor.find('update_rate').text = '3'
ET.indent(tree, space='  ')
tree.write(path, encoding='utf-8', xml_declaration=True)
print(path)
