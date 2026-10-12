import cv2
from ultralytics import YOLO

image=cv2.imread('/mnt/c/Users/황준영/Documents/졸업작품/tmp/world1_full_20261003/r32_alignment.jpg')
camera=cv2.resize(image[45:455,987:1533],(320,240))
model=YOLO('src/fire_robot_perception/models/handle_best_v3_CANDIDATE.pt')
for border in (0,80,160):
    sample=cv2.copyMakeBorder(camera,border,border,border,border,
                             cv2.BORDER_CONSTANT,value=(114,114,114))
    result=model.predict(sample,imgsz=320,conf=.25,verbose=False)[0]
    print('border',border,'boxes',[(b.xyxy[0].tolist(),float(b.conf[0])) for b in result.boxes])
