"""Reuse frozen World 1 control; replace simulator fixture and add passive audit topics."""
import importlib.util
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    workspace = Path(__file__).resolve().parents[2]
    path = workspace / 'src/fire_robot_bringup/launch/world1_full.launch.py'
    spec = importlib.util.spec_from_file_location('frozen_world1_launch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fixture = Path(os.environ['FROZEN_VALIDATION_WORLD']).resolve()
    actions, replaced = [], 0
    for action in module.generate_launch_description().entities:
        if isinstance(action, IncludeLaunchDescription):
            arguments = dict(action.launch_arguments)
            if arguments.get('world') == 'world1_full_contact.world':
                arguments['world'] = str(fixture)
                action = IncludeLaunchDescription(action.launch_description_source,
                                                  launch_arguments=arguments.items())
                replaced += 1
        actions.append(action)
    assert replaced == 1, 'Exactly the simulator world argument may be replaced'
    extra = []
    for model in ET.parse(fixture).getroot().find('world').findall('model'):
        name = model.get('name', '')
        if name.startswith('door_blue') and name not in ('door_blue1', 'door_blue2', 'door_blue3'):
            extra.append(f'/evaluation/{name}/joints@sensor_msgs/msg/JointState[gz.msgs.Model')
            extra.extend(f'/proof/audit/{name}/{part}@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts'
                         for part in ('lever', 'latch', 'panel', 'jamb', 'strike'))
    if extra:
        actions.append(Node(package='ros_gz_bridge', executable='parameter_bridge',
            name='additional_evaluation_only_bridge', arguments=extra,
            parameters=[{'use_sim_time': True}], output='screen'))
    return LaunchDescription(actions)
