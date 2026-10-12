"""Offline fixture conversion only; controller inputs are not generated here."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / 'scripts'))
from make_open_exit_world import generate
from world_contact_clearance import outer_wall_face, check_sweeps

LAYOUTS = {
    1: 'obstacle_wall_doors_v5.world',
    2: 'obstacle_door_layout_alt_v1.world',
    3: 'obstacle_door_layout_world3_v1.world',
    4: 'obstacle_no_blue_world4_v1.world',
    5: 'obstacle_all_blue_world5_v1.world',
}


def build(number, output):
    worlds = WORKSPACE / 'src/fire_robot_bringup/worlds'
    source = worlds / LAYOUTS[number]
    with tempfile.TemporaryDirectory(prefix='frozen_contact_fixture_') as directory:
        intermediate = Path(directory) / 'scene.world'
        generate(source, intermediate)
        tree = ET.parse(intermediate)
    world = tree.getroot().find('world')
    original = {m.get('name'): ET.tostring(m) for m in world.findall('model')}
    fixture = ET.parse(worlds / 'physical_contact_door_test.world').getroot().find('world')
    template = fixture.find("model[@name='blue_hinged_contact_door']")
    doors = []
    for old in list(world.findall('model')):
        name = old.get('name', '')
        if not name.startswith('door_blue'):
            continue
        x, painted_y = list(map(float, old.findtext('pose').split()))[:2]
        yaw = -math.pi / 2 if painted_y < 0 else math.pi / 2
        y = outer_wall_face(world, 1 if painted_y > 0 else -1)
        new = copy.deepcopy(template)
        new.set('name', name)
        new.find('pose').text = f'{x + .4 * math.sin(yaw)} {y - .4 * math.cos(yaw)} 0 0 0 {yaw}'
        for topic in new.findall('.//topic'):
            if topic.text.startswith('/proof/contact/'):
                topic.text = topic.text.replace('/proof/contact/', f'/proof/contact/{name}/')
            elif topic.text == '/door_joint_states':
                topic.text = f'/evaluation/{name}/joints'
        world.remove(old)
        world.append(new)
        doors.append(name)
    for old in list(world.findall('model')):
        if old.get('name', '').startswith('handle_blue'):
            world.remove(old)
    for plugin in fixture.findall('plugin'):
        if plugin.get('name') in ('gz::sim::systems::Contact', 'fire_robot::ContactAudit'):
            world.append(copy.deepcopy(plugin))
    camera = copy.deepcopy(fixture.find("model[@name='proof_perspective_camera']"))
    camera.find('pose').text = '-4 0 3 0 .65 0'
    camera.find('.//image/width').text = '640'
    camera.find('.//image/height').text = '360'
    camera.find('.//update_rate').text = '15'
    world.append(camera)
    ET.SubElement(world, 'plugin', filename='libfire_robot_film_camera.so', name='fire_robot::FilmCamera')
    unchanged = []
    for model in world.findall('model'):
        name = model.get('name')
        if name in original and not name.startswith('door_blue'):
            assert ET.tostring(model) == original[name], name
            unchanged.append(name)
    clearance = check_sweeps(world) if doors else {'scope': 'offline_environment_construction_only',
        'pass_clearance': True, 'doors': [], 'reason': 'No blue door: sweep test not applicable'}
    if not clearance['pass_clearance']:
        raise ValueError('Fixture construction collision, not a policy result: ' + json.dumps(clearance))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    ET.indent(tree, space='  ')
    tree.write(output, encoding='utf-8', xml_declaration=True)
    report = dict(scope='evaluation_only_never_loaded_by_control', world=number,
        source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        generated_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        blue_models=sorted(doors), unchanged_models=unchanged,
        construction_clearance_audit=clearance,
        conversion='Same lever fixture and outer-wall mounting as successful World 1; open green exit')
    output.with_suffix('.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('world', type=int, choices=LAYOUTS)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.world, args.output), indent=2))
