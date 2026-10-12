"""Offline classification only; evaluation contacts never feed robot control."""


def chassis_contact(pair):
    for collision in pair:
        if not collision.startswith('fire_robot::'):
            continue
        parts = collision.split('::')
        if len(parts) < 3:
            return True
        link = parts[1]
        if link in {f'arm_link{i}' for i in range(1,7)} or link.startswith('gripper_'):
            continue
        # Gazebo lumps fixed cameras, the cage and arm mount into base_footprint.
        # Unknown robot links are not silently treated as permitted hand contact.
        return True
    return False
