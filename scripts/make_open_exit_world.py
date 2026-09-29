#!/usr/bin/env python3
"""Offline fixture conversion. Nothing here is supplied to the robot controller."""
import argparse
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET


def generate(source, destination):
    if destination.exists():
        raise FileExistsError(destination)
    tree = ET.parse(source)
    world = tree.getroot().find('world')
    models = {m.get('name'): m for m in world.findall('model')}
    original = {name: ET.tostring(m) for name, m in models.items()}
    exit_model = models['exit_green']
    link = exit_model.find('link')
    for child in list(link):
        if child.tag in ('visual', 'collision'):
            link.remove(child)
    # A 1.2 m clear passage, framed by visible and physical matching geometry.
    # Dimensions define this fixture only; they are never a perception prior.
    parts = [('jamb_left', (0., .66, 0.), (.12, .12, 2.0)),
             ('jamb_right', (0., -.66, 0.), (.12, .12, 2.0)),
             ('lintel', (0., 0., 1.06), (.12, 1.44, .12))]
    for name, xyz, size in parts:
        for kind in ('visual', 'collision'):
            part = ET.SubElement(link, kind, name=name)
            ET.SubElement(part, 'pose').text = ' '.join(map(str, (*xyz, 0., 0., 0.)))
            geometry = ET.SubElement(part, 'geometry')
            box = ET.SubElement(geometry, 'box')
            ET.SubElement(box, 'size').text = ' '.join(map(str, size))
            if kind == 'visual':
                material = ET.SubElement(part, 'material')
                for component in ('ambient', 'diffuse'):
                    ET.SubElement(material, component).text = '0.02 0.85 0.12 1'
    if 'handle_exit' in models:
        world.remove(models['handle_exit'])
    for name, model in models.items():
        if name not in ('exit_green', 'handle_exit'):
            assert ET.tostring(model) == original[name], name
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding='utf-8', xml_declaration=True)
    return {
        'type': 'open_passage_with_green_frame',
        'approval': 'User approved open exit; preserve all interior doors and obstacles (2026-09-29)',
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'generated_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
        'changed_models': ['exit_green', 'handle_exit'],
        'unchanged_model_count': len(models)-len({'exit_green', 'handle_exit'} & models.keys()),
        'physical_opening_tested': False,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    print(generate(args.source, args.destination))
