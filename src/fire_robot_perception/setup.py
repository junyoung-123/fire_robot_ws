from setuptools import setup
import os
from glob import glob

package_name = 'fire_robot_perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),  glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'),  glob('config/*.yaml')),
        # 학습 스크립트: colcon build 후 lib/fire_robot_perception/ 에 설치됨
        # → source install/setup.bash 후 `ros2 run fire_robot_perception prepare_dataset.py` 실행 가능
        (os.path.join('lib', package_name),
         [f for f in glob('scripts/*.py') if not f.endswith('__init__.py')]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'sensor_fusion_node  = fire_robot_perception.sensor_fusion_node:main',
            'door_detection_node = fire_robot_perception.door_detection_node:main',
        ],
    },
)
