"""Separate measurement age from slow simulation and a stalled clock."""
import math


class ObservedFreshness:
    def __init__(self, simulated_clock, measurement_limit=.25, clock_stall_limit=1.):
        self.simulated_clock=simulated_clock
        self.measurement_limit=measurement_limit
        self.clock_stall_limit=clock_stall_limit
        self.last_ros=None
        self.last_clock_progress=None

    def valid(self, measurement_stamp, receipt_stamp, ros_now, wall_now):
        if any(v is None or not math.isfinite(v) for v in
               (measurement_stamp,receipt_stamp,ros_now,wall_now)):
            return False
        if self.last_ros is not None and ros_now<self.last_ros:
            self.last_ros=ros_now
            self.last_clock_progress=wall_now
            return False
        if self.last_ros is None or ros_now>self.last_ros:
            self.last_clock_progress=wall_now
        self.last_ros=ros_now
        measurement_age=ros_now-measurement_stamp
        receipt_limit=3. if self.simulated_clock else 1.
        return (-.02<=measurement_age<=self.measurement_limit
                and 0.<=wall_now-receipt_stamp<=receipt_limit
                and 0.<=wall_now-self.last_clock_progress<=self.clock_stall_limit)
