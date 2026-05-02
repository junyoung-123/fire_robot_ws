# FSM 상태 상수
uint8 IDLE            = 0
uint8 EXPLORING       = 1
uint8 NAVIGATING      = 2
uint8 OPENING_DOOR    = 3
uint8 DOOR_OPENED     = 4
uint8 EMERGENCY_STOP  = 5

std_msgs/Header header
uint8 state
string state_description
string target_door_id
