"""Offline SDF fixture validation; no environment geometry enters control."""
import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np


def pose(element):
    value=np.array(list(map(float,element.findtext('pose','0 0 0 0 0 0').split())))
    if value.shape!=(6,) or not np.all(np.isfinite(value)) or np.max(np.abs(value[3:5]))>1.e-6:
        raise ValueError('Fixture clearance checker needs finite upright poses')
    return value


def rotation(yaw):
    c,s=math.cos(yaw),math.sin(yaw)
    return np.array([[c,-s],[s,c]])


def combine(a,b):
    result=a.copy();result[:2]+=rotation(a[5])@b[:2];result[2]+=b[2];result[5]+=b[5]
    return result


def rectangle(center,size,yaw,margin=0.):
    half=np.asarray(size)[:2]/2.+margin
    return np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*half@rotation(yaw).T+center[:2]


def overlaps(a,b):
    for poly in (a,b):
        for edge in np.diff(np.vstack((poly,poly[0])),axis=0):
            axis=np.array([-edge[1],edge[0]])
            pa,pb=a@axis,b@axis
            if pa.max()<=pb.min() or pb.max()<=pa.min():return False
    return True


def outer_wall_face(world,side):
    name='wall_north' if side>0 else 'wall_south'
    model=world.find(f"model[@name='{name}']")
    if model is None:raise ValueError('Missing corridor side wall')
    values=[]
    for link in model.findall('link'):
        frame=combine(pose(model),pose(link))
        for collision in link.findall('collision'):
            size=collision.findtext('geometry/box/size')
            if size is None:continue
            size=np.array(list(map(float,size.split())))
            frame_collision=combine(frame,pose(collision))
            vertices=rectangle(frame_collision,size,frame_collision[5])
            values.extend(vertices[:,1])
    if not values:raise ValueError('Side wall has no collision boxes')
    return max(values) if side>0 else min(values)


def check_sweeps(world,step_deg=.5):
    if not math.isfinite(step_deg) or not 0.<step_deg<=1.:
        raise ValueError('Invalid angular sampling resolution')
    walls=[]
    for model in world.findall('model'):
        if not model.get('name','').startswith('wall_'):continue
        for link in model.findall('link'):
            frame=combine(pose(model),pose(link))
            for collision in link.findall('collision'):
                size=collision.findtext('geometry/box/size')
                if size is None:continue
                size=np.array(list(map(float,size.split())))
                p=combine(frame,pose(collision))
                walls.append((model.get('name')+'::'+collision.get('name'),p,size))
    result=[]
    for model in world.findall('model'):
        if not model.get('name','').startswith('door_blue'):continue
        panel=model.find("link[@name='panel']")
        joint=model.find("joint[@name='door_hinge']")
        collision=panel.find("collision[@name='panel_collision']")
        panel_pose=pose(panel);joint_pose=pose(joint)
        hinge=combine(panel_pose,joint_pose)
        box_pose=combine(panel_pose,pose(collision))
        size=np.array(list(map(float,collision.findtext('geometry/box/size').split())))
        corners=rectangle(box_pose,size,box_pose[5])
        radius=float(max(np.linalg.norm(corners-hinge[:2],axis=1)))
        lower=float(joint.findtext('axis/limit/lower'));upper=float(joint.findtext('axis/limit/upper'))
        angles=np.linspace(lower,upper,math.ceil((upper-lower)/math.radians(step_deg))+1)
        # A Lipschitz bound inflates each sample to cover the arc between samples.
        margin=radius*(angles[1]-angles[0])*.5+1.e-6
        hits={}
        for angle in angles:
            offset=rotation(angle)@(box_pose[:2]-hinge[:2])
            local=box_pose.copy();local[:2]=hinge[:2]+offset;local[5]+=angle
            global_pose=combine(pose(model),local)
            leaf=rectangle(global_pose,size,global_pose[5],margin)
            for name,wall,wall_size in walls:
                if abs(global_pose[2]-wall[2])>=(size[2]+wall_size[2])*.5:continue
                if overlaps(leaf,rectangle(wall,wall_size,wall[5])):
                    hits.setdefault(name,[]).append(math.degrees(angle))
        result.append(dict(model=model.get('name'),checked_joint_range_deg=[math.degrees(lower),math.degrees(upper)],
            continuous_sampling_margin_m=margin,wall_intersections={name:[min(a),max(a)] for name,a in hits.items()}))
    return dict(scope='offline_environment_construction_only',pass_clearance=bool(result) and all(not r['wall_intersections'] for r in result),doors=result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('world',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args();report=check_sweeps(ET.parse(args.world).getroot().find('world'))
    if args.output:args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
