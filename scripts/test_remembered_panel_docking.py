#!/usr/bin/env python3
"""A camera turn may hide the handle; scan and TF must remain live."""
from types import SimpleNamespace as NS
import numpy as np
from fire_robot_manipulation import mission_contact_adapter as module


def exercise(scan_fresh=True, tf_fresh=True):
    clock = [5.]
    commands = []
    events = []
    calls = [0]
    subject = NS(
        reference_frame='odom', latest=(NS(handle_position=NS(point=NS(y=.4))), 5., None),
        panel={'depth_stamp': 5., 'normal': [1., 0., 0.]},
        scan=NS(header=NS(stamp=NS(sec=5, nanosec=0))),
        kinematics=NS(mount_xyz=np.array([0., .15, 0.])), now=lambda: clock[0],
        command=lambda *values: commands.append(values),
        transform=lambda *args: NS(point=NS(x=1.5, y=.4, z=.8)),
        safe_motion=lambda speed: True)

    def event(name, **data):
        events.append(name)
        if name == 'panel_parallel_docking':
            clock[0] = 6.
            subject.latest = None
            subject.scan.header.stamp.sec = 6 if scan_fresh else 0

    def transform(*args):
        calls[0] += 1
        return NS(header=NS(stamp=NS(sec=int(clock[0]) if tf_fresh else 0, nanosec=0)),
                  transform=NS(translation=NS(x=0. if calls[0] == 1 else -.35,
                                             y=0. if calls[0] == 1 else .25),
                               rotation=NS(x=0., y=0., z=0., w=1.)))

    subject.event = event
    subject.tf = NS(lookup_transform=transform)
    old_ok, old_sleep = module.rclpy.ok, module.time.sleep
    try:
        module.rclpy.ok = lambda: True
        module.time.sleep = lambda seconds: clock.__setitem__(0, clock[0]+1.)
        result = module.MissionContactAdapter.park_from_observation(subject)
    finally:
        module.rclpy.ok, module.time.sleep = old_ok, old_sleep
    return result, commands, events


def main():
    result, commands, events = exercise()
    assert result and 'panel_parallel_ready' in events
    assert not exercise(scan_fresh=False)[0]
    assert not exercise(tf_fresh=False)[0]
    print('PASS: remembered panel parking tolerates camera loss, rejects stale scan/TF')


if __name__ == '__main__':
    main()
