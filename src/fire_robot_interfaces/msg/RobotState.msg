# FSM 상태 상수 (state_machine_node.py State enum과 동일)
uint8 IDLE             = 0
uint8 EXPLORING        = 1
uint8 NAVIGATING       = 2
uint8 OPENING_DOOR     = 3
uint8 DOOR_OPENED      = 4
uint8 EXITING          = 5
uint8 MISSION_COMPLETE = 6
uint8 EMERGENCY_STOP   = 7

std_msgs/Header header
uint8 state
string state_description
string target_door_id
