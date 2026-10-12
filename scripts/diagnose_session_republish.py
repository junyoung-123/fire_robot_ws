#!/usr/bin/env python3
"""Debug-only repeat of the mission's existing selected-door identity."""
import json
from pathlib import Path
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

path = Path('artifacts/full_world1_r6/evidence/events.jsonl')
events = [json.loads(line) for line in path.read_text().splitlines()]
selected = [e['payload']['door_id'] for e in events if e['kind']=='physical_job'
            and e['payload'].get('event')=='observe_selected_handle'][-1]
Path('artifacts/full_world1_r6/manual_diagnostic_intervention.json').write_text(
    json.dumps({'action': 'repeat selected identity only', 'id': selected,
                'run_ineligible_for_final_autonomous_success_evidence': True}))
rclpy.init()
node = Node('session_republish_diagnostic')
pub = node.create_publisher(String, '/contact/session', 10)
for _ in range(20):
    pub.publish(String(data=selected))
    rclpy.spin_once(node, timeout_sec=.1)
    time.sleep(.2)
node.destroy_node()
rclpy.shutdown()
