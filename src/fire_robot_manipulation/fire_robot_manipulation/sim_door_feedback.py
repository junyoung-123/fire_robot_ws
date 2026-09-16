"""Simulation-backend verification only; never supply goals to perception/FSM."""
import json
import math
import subprocess
import xml.etree.ElementTree as ET


def model_metadata(world_path):
    world = ET.parse(world_path).find('world')
    metadata = {}
    for model in world.findall('model'):
        initial = [float(x) for x in model.findtext('pose', '0 0 0 0 0 0').split()]
        for plugin in model.findall('plugin'):
            topic = plugin.findtext('topic', '')
            if 'JointPositionController' in plugin.get('name', '') and '/door/' in topic:
                metadata[topic] = dict(world=world.get('name'), model=model.get('name'), yaw=initial[5])
    return metadata


def rotation_sample(message, model, initial_yaw):
    candidates = [p for p in message.get('pose', []) if p.get('name') == model]
    if len(candidates) != 1:
        raise ValueError('Missing or ambiguous Gazebo door model pose')
    q = candidates[0].get('orientation', {})
    x, y, z, w = (float(q.get(k, 0. if k != 'w' else 1.)) for k in ('x', 'y', 'z', 'w'))
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if not math.isfinite(norm) or abs(norm - 1.) > .01:
        raise ValueError('Invalid Gazebo door orientation')
    angle = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)) - initial_yaw
    stamp = message['header']['stamp']
    seconds = float(stamp.get('sec', 0)) + float(stamp.get('nsec', 0))*1.e-9
    if not math.isfinite(seconds):
        raise ValueError('Invalid Gazebo pose time')
    return seconds, math.atan2(math.sin(angle), math.cos(angle))


def read_rotation(metadata):
    raw = subprocess.check_output([
        'ign', 'topic', '-e', '-n', '1', '--json-output',
        '-t', '/world/' + metadata['world'] + '/pose/info',
    ], text=True, timeout=3.)
    return rotation_sample(json.loads(raw), metadata['model'], metadata['yaw'])
