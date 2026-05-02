std_msgs/Header header

# 화재 위치 = 빨간 문(Red Door)들의 무게중심
geometry_msgs/PointStamped fire_position
bool detected        # 빨간 문 감지 여부
int32 red_door_count # 감지된 빨간 문 수
