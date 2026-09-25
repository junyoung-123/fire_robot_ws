"""Reflect the test fixture, not the robot or any controller observation."""
from pathlib import Path
import xml.etree.ElementTree as ET

root=Path(__file__).resolve().parents[1]
source=root/'src/fire_robot_bringup/worlds/physical_contact_door_test.world'
target=source.with_name('physical_contact_door_mirrored.world')
tree=ET.parse(source)
world=tree.getroot().find('world')
for model in world.findall('model'):
    if model.get('name','').startswith('proof_'):
        continue
    for pose in model.iter('pose'):
        values=list(map(float,pose.text.split()))
        if len(values)!=6: raise ValueError('Only xyz/rpy fixture poses supported')
        for i in (1,3,5): values[i]=-values[i]
        pose.text=' '.join(f'{v:.9g}' for v in values)
    for inertia in model.iter('inertia'):
        for name in ('ixy','iyz'):
            value=inertia.find(name)
            if value is not None: value.text=f'{-float(value.text):.9g}'
    for joint in model.findall('joint'):
        axis=joint.find('axis')
        if axis is None: continue
        if axis.findtext('xyz') not in ('1 0 0','0 0 1'):
            raise ValueError('Unexpected fixture joint axis')
        lower,upper=axis.find('limit/lower'),axis.find('limit/upper')
        lo,hi=float(lower.text),float(upper.text)
        lower.text,upper.text=f'{-hi:.9g}',f'{-lo:.9g}'
        spring=axis.find('dynamics/spring_reference')
        if spring is not None: spring.text=f'{-float(spring.text):.9g}'
ET.indent(tree,space='  ')
tree.write(target,encoding='utf-8',xml_declaration=True)
print(target)
