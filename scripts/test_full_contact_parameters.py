"""Reject silently ignored ROS parameter names at the integration boundary."""
import ast
from pathlib import Path

root = Path(__file__).resolve().parents[1]
launch = ast.parse((root/'src/fire_robot_bringup/launch/world1_full.launch.py').read_text())
policy = ast.parse((root/'src/fire_robot_manipulation/fire_robot_manipulation/preserved_contact_policy.py').read_text())
known = {'use_sim_time'}
for node in ast.walk(policy):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'declare_parameter':
        if node.args and isinstance(node.args[0], ast.Constant):
            known.add(node.args[0].value)
for node in ast.walk(launch):
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'parameters' for t in node.targets):
        names = {key.value for key in node.value.keys}
        unknown = names-known
        assert not unknown, f'Undeclared contact parameters: {sorted(unknown)}'
print('PASS: all contact launch parameters are declared by the preserved controller')
