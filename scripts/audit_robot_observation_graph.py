"""Archive the live controller's inputs; no environmental state is read."""
import json
import hashlib
from pathlib import Path
import subprocess
import sys

output=subprocess.check_output(['ros2','node','info','/manipulation_node'],text=True)
subscribers=output.split('Subscribers:',1)[1].split('Publishers:',1)[0]
required=('/joint_states:','/observed_door/rotation:','/detected_door:')
forbidden=('/door_joint_states','/model_states','/link_states')
passed=all(name in subscribers for name in required) and not any(name in subscribers for name in forbidden)
observer=subprocess.check_output(['ros2','node','info','/observed_door_angle'],text=True)
observer_inputs=observer.split('Subscribers:',1)[1].split('Publishers:',1)[0]
observer_required=('/scan:','/camera/depth/image_rect_raw:','/camera/color/image_raw:')
observer_pass=all(name in observer_inputs for name in observer_required) and not any(name in observer_inputs for name in forbidden)
target=Path(sys.argv[1]); target.mkdir(parents=True,exist_ok=True)
(target/'controller_node_info.txt').write_text(output)
(target/'observer_node_info.txt').write_text(observer)
(target/'audit_script.py').write_bytes(Path(__file__).read_bytes())
(target/'controller_input_audit.json').write_text(json.dumps(dict(
    passed=passed and observer_pass,controller_pass=passed,observer_pass=observer_pass,
    required=required,observer_required=observer_required,forbidden=forbidden,
    audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    scope='Live graph audit plus strict callback name filter. Simulator is retained for independent evaluation.'),indent=2))
print(json.dumps(dict(passed=passed and observer_pass,subscribers=subscribers.strip())))
raise SystemExit(0 if passed and observer_pass else 1)
