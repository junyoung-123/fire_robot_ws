#!/usr/bin/env python3
from types import SimpleNamespace
from builtin_interfaces.msg import Time
from fire_robot_navigation.sensor_message_gate import SensorMessageGate

subject = SimpleNamespace(last={})
accepted = [n for n in range(101) if SensorMessageGate.accept(
    subject, 'clock', Time(nanosec=n*1000000))]
assert accepted == [0, 20, 40, 60, 80, 100]
assert SensorMessageGate.accept(subject, 'clock', Time(nanosec=0))
assert SensorMessageGate.accept(subject, 'joints', Time(nanosec=0))
print('PASS: 50Hz stamp-preserving gate; independent streams; time-reset recovery')
