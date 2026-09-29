"""Test adapter for navigation visits. Never moves an arm or a Gazebo door.

The existing FSM service boundary is reused only to continue its next-target
loop. Its internal OPENED state means VISITED in this navigation-only trial.
"""
import json
import math

import rclpy
from fire_robot_interfaces.msg import DoorInfo
from fire_robot_interfaces.srv import OpenDoor
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener


class NavigationVisitNode(Node):
    def __init__(self):
        super().__init__('navigation_visit_node')
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.target = None
        self.create_subscription(DoorInfo, '/target_door', self.on_target, 10)
        self.events = self.create_publisher(String, '/navigation_visit', 10)
        self.create_service(OpenDoor, '/open_door', self.visit)
        self.get_logger().warning(
            'NAVIGATION ONLY: service success is a VISIT, NOT physical door opening.')

    def on_target(self, msg):
        self.target = msg

    def visit(self, request, response):
        if self.target is None or self.target.door_color != 'blue':
            response.message = 'Navigation visit rejected: no observed blue target.'
            return response
        if request.handle_position.header.frame_id != 'map':
            response.message = 'Navigation visit rejected: expected observed map coordinates.'
            return response
        try:
            transform = self.buffer.lookup_transform('map', 'base_link', Time())
        except TransformException:
            response.message = 'Navigation visit rejected: no sensor-localized robot pose.'
            return response
        p, q = transform.transform.translation, transform.transform.rotation
        h = request.handle_position.point
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y*q.y + q.z*q.z))
        distance = math.hypot(h.x - p.x, h.y - p.y)
        heading = math.atan2(h.y - p.y, h.x - p.x)
        error = math.atan2(math.sin(heading-yaw), math.cos(heading-yaw))
        # Independent proximity check; the FSM also applies its usual fine alignment gate.
        if not (0.15 < distance < 1.8 and abs(error) < math.radians(20)):
            response.message = f'Navigation visit rejected: distance={distance:.3f}, yaw={error:.3f}'
            return response
        event = {
            'scope': 'navigation_only', 'physical_opening': False,
            'door_id': request.door_id, 'sim_time': self.get_clock().now().nanoseconds / 1e9,
            'robot': [p.x, p.y, yaw], 'observed_handle': [h.x, h.y, h.z],
            'distance_m': distance, 'bearing_error_rad': error,
        }
        payload = json.dumps(event)
        self.events.publish(String(data=payload))
        self.get_logger().info('NAVIGATION_VISIT ' + payload)
        response.success = True
        response.message = 'NAVIGATION_VISIT_ONLY: no physical door opening performed.'
        return response


def main(args=None):
    rclpy.init(args=args)
    node = NavigationVisitNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
