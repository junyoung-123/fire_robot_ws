#!/usr/bin/env python3
"""Environment construction only; no world geometry is a robot control input."""
import copy
import json
import math
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
from make_open_exit_world import generate
from world_contact_clearance import outer_wall_face, check_sweeps


def main():
    ws = Path(__file__).resolve().parents[1]
    worlds = ws / 'src/fire_robot_bringup/worlds'
    output = worlds / 'world1_full_contact.world'
    with tempfile.TemporaryDirectory(prefix='world1_full_scene_') as directory:
        intermediate = Path(directory) / 'scene.world'
        generate(worlds / 'obstacle_wall_doors_v5.world', intermediate)
        tree = ET.parse(intermediate)
    world = tree.getroot().find('world')
    fixture = ET.parse(worlds / 'physical_contact_door_test.world').getroot().find('world')
    template = fixture.find("model[@name='blue_hinged_contact_door']")
    targets = []
    for old in list(world.findall('model')):
        name = old.get('name', '')
        if not name.startswith('door_blue'):
            continue
        pose = list(map(float, old.findtext('pose').split()))
        x, y = pose[:2]
        yaw = -math.pi / 2 if y < 0 else math.pi / 2
        # The outward-opening hinge belongs at the outer wall face. Reusing
        # the painted panel's corridor-side depth traps the leaf in the jamb.
        painted_y=y
        y=outer_wall_face(world,1 if y>0 else -1)
        # Keep the original doorway centre along the corridor.
        new = copy.deepcopy(template)
        new.set('name', name)
        new.find('pose').text = f'{x + .4 * math.sin(yaw)} {y - .4 * math.cos(yaw)} 0 0 0 {yaw}'
        for topic in new.findall('.//topic'):
            if topic.text.startswith('/proof/contact/'):
                topic.text = topic.text.replace('/proof/contact/', f'/proof/contact/{name}/')
            elif topic.text == '/door_joint_states':
                topic.text = f'/evaluation/{name}/joints'
        targets.append({'model': name, 'closed_panel_center': [x, y, 1],
                        'painted_panel_y': painted_y,'hinge_mounting': 'outer_wall_face',
                        'evaluation_joint_topic': f'/evaluation/{name}/joints'})
        world.remove(old)
        world.append(new)
    for old in list(world.findall('model')):
        if old.get('name', '').startswith('handle_blue'):
            world.remove(old)
    for plugin in fixture.findall('plugin'):
        if plugin.get('name') in ('gz::sim::systems::Contact', 'fire_robot::ContactAudit'):
            world.append(copy.deepcopy(plugin))
    for name in ('proof_perspective_camera',):
        camera = copy.deepcopy(fixture.find(f"model[@name='{name}']"))
        camera.find('pose').text = '-4 0 3 0 .65 0'
        image = camera.find('.//image')
        image.find('width').text = '640'
        image.find('height').text = '360'
        camera.find('.//update_rate').text = '15'
        world.append(camera)
    ET.SubElement(world, 'plugin', filename='libfire_robot_film_camera.so',
                  name='fire_robot::FilmCamera')
    clearance=check_sweeps(world)
    if not clearance['pass_clearance']:
        raise ValueError('Generated physical door intersects wall: '+json.dumps(clearance))
    ET.indent(tree, space='  ')
    tree.write(output, encoding='utf-8', xml_declaration=True)
    (worlds / 'world1_full_contact_evaluation.json').write_text(json.dumps(
        {'scope': 'evaluation_only_never_loaded_by_control', 'doors': targets,
         'construction_clearance_audit':clearance}, indent=2))
    print(output)


if __name__ == '__main__':
    main()
