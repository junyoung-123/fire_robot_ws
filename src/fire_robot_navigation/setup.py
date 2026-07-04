from setuptools import setup
import os
from glob import glob

package_name = 'fire_robot_navigation'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'maps'), glob('maps/*')),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'navigation_node = fire_robot_navigation.navigation_node:main',
            'cmd_vel_safety_node = fire_robot_navigation.cmd_vel_safety_node:main',
            'mission_axis_node = fire_robot_navigation.mission_axis_node:main',
            'initial_static_map_node = fire_robot_navigation.initial_static_map_node:main',
            'fixed_obstacle_map_node = fire_robot_navigation.fixed_obstacle_map_node:main',
        ],
    },
)
