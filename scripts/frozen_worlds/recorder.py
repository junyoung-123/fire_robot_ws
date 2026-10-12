"""Passive extension of the preserved recorder to extra doors and truthful titles."""
import argparse
from pathlib import Path
import signal
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import rclpy
from sensor_msgs.msg import JointState
from ros_gz_interfaces.msg import Contacts
from full_mission_recorder import FullRecorder


class TitleWriter:
    def __init__(self, writer, world):
        self.writer, self.title = writer, f'GAZEBO | World {world}'

    def write(self, frame):
        frame[:45, :920] = 255
        cv2.putText(frame, self.title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (35, 35, 35), 2)
        self.writer.write(frame)

    def release(self):
        self.writer.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--world', type=int, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    args = parser.parse_args()
    rclpy.init()
    node = FullRecorder(args.output)
    node.writer_video = TitleWriter(node.writer_video, args.world)
    for model in ET.parse(args.fixture).getroot().find('world').findall('model'):
        name = model.get('name', '')
        if name.startswith('door_blue') and name not in ('door_blue1', 'door_blue2', 'door_blue3'):
            node.create_subscription(JointState, f'/evaluation/{name}/joints',
                lambda message, n=name: node.reference(message, n), 20)
            for part in ('lever', 'panel', 'latch', 'jamb', 'strike'):
                node.create_subscription(Contacts, f'/proof/audit/{name}/{part}',
                    lambda message, n=name, p=part: node.contact(message, n, p), 20)
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        while rclpy.ok() and not stopped:
            rclpy.spin_once(node, timeout_sec=.1)
    finally:
        node.checkpoint()
        node.writer_video.release()
        node.timeline.close()
        node.csv_file.close()
        node.truth_file.close()
        node.events.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
