#!/usr/bin/env python3
"""Cut real openings behind existing blue doors without moving their layouts."""
import argparse
import copy
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def repair(path, source_dir=None):
    text = (source_dir/path.name if source_dir else path).read_text()
    world = ET.fromstring(text).find('world')
    doors = [m for m in world.findall('model') if m.get('name', '').startswith('door_blue')]
    for name in ('wall_north', 'wall_south'):
        wall = next(m for m in world.findall('model') if m.get('name') == name)
        pose = [float(x) for x in wall.findtext('pose').split()]
        if any(abs(x) > 1.e-8 for x in pose[3:]):
            raise ValueError('This migration requires axis-aligned corridor walls')
        link = wall.find('link')
        original = {tag: link.find(tag) for tag in ('collision', 'visual')}
        if len(link.findall('collision')) != 1:
            raise ValueError('Refusing to modify an already segmented wall')
        size = [float(x) for x in original['collision'].findtext('geometry/box/size').split()]
        openings = []
        for door in doors:
            dp = [float(x) for x in door.findtext('pose').split()]
            if dp[1] * pose[1] <= 0.:
                continue
            ds = [float(x) for x in door.findtext('link/collision/geometry/box/size').split()]
            # Beyond 90 degrees the hinge-side leaf sweeps behind the jamb.
            limit = max(abs(float(door.findtext('joint/axis/limit/lower'))),
                        abs(float(door.findtext('joint/axis/limit/upper'))))
            wall_depth = abs(dp[1]-pose[1]) + size[1]/2
            recess = max(.08, wall_depth*abs(1/math.tan(limit))
                         + ds[1]/(2*abs(math.sin(limit))) + .02)
            openings.append((dp[0] - ds[0]/2 - recess, dp[0] + ds[0]/2 + .08,
                             dp[2] + ds[2]/2 + .05))
        if not openings:
            continue
        for element in original.values():
            link.remove(element)
        def segment(label, left, right, bottom, top):
            if right <= left or top <= bottom:
                raise ValueError('Invalid or overlapping doorway extent')
            for tag, source in original.items():
                item = copy.deepcopy(source)
                item.set('name', tag + '_' + label)
                at = item.find('pose')
                if at is None:
                    at = ET.SubElement(item, 'pose')
                at.text = f'{(left+right)/2-pose[0]:.6f} 0 {(bottom+top)/2-pose[2]:.6f} 0 0 0'
                item.find('geometry/box/size').text = f'{right-left:.6f} {size[1]:.6f} {top-bottom:.6f}'
                link.append(item)
        left = pose[0] - size[0]/2
        bottom, top = pose[2] - size[2]/2, pose[2] + size[2]/2
        for i, (start, end, height) in enumerate(sorted(openings)):
            segment(f'wall_{i}', left, start, bottom, top)
            segment(f'lintel_{i}', start, end, height, top)
            left = end
        segment('wall_end', left, pose[0] + size[0]/2, bottom, top)
        ET.indent(wall, space='  ')
        replacement = ET.tostring(wall, encoding='unicode').rstrip()
        # Only the two parsed wall models change; all other source bytes remain.
        pattern = r'<model name="' + re.escape(name) + r'">.*?</model>'
        text, count = re.subn(pattern, lambda _: replacement, text, count=1, flags=re.S)
        if count != 1:
            raise ValueError('Cannot locate unique wall source')
    path.write_text(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('worlds', nargs='+', type=Path)
    parser.add_argument('--source-dir', type=Path)
    args = parser.parse_args()
    for path in args.worlds:
        repair(path, args.source_dir)
        print(path)
