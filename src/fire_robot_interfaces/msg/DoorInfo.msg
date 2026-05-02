std_msgs/Header header

geometry_msgs/PoseStamped door_pose        # 문 위치 및 방향
geometry_msgs/PointStamped handle_position # 손잡이 3D 위치

string door_id       # 문 고유 ID
string door_color    # "blue"(안전), "red"(위험/화재), "unknown"
bool is_open         # 열림 여부
float32 confidence   # 탐지 신뢰도 (0.0 ~ 1.0)
float32 distance_from_fire  # 화재(빨간 문)로부터 거리 (m)
