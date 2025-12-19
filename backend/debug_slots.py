#!/usr/bin/env python3
import cv2
import numpy as np
from ultralytics import YOLO
import csv

def load_slots(csv_path):
    slots = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row or len(row) < 9:
                continue
            pts = list(map(int, row[1:9]))
            pts_tuples = [(pts[i], pts[i+1]) for i in range(0, 8, 2)]
            slots.append(pts_tuples)
    return slots

def expand_polygon(points, margin=25):
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    
    expanded = []
    for (x, y) in points:
        dx = x - cx
        dy = y - cy
        length = (dx**2 + dy**2)**0.5
        if length > 0:
            dx /= length
            dy /= length
        expanded.append((int(x + dx * margin), int(y + dy * margin)))
    return expanded

# 비디오와 모델 로드
video_path = "temp_videos/current.mp4"
model = YOLO("best.pt")
slots = load_slots("slots.csv")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("비디오를 열 수 없습니다.")
    exit()

# 중간 프레임으로 이동
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)

ret, frame = cap.read()
if not ret:
    print("프레임을 읽을 수 없습니다.")
    exit()

# YOLO 감지
results = model.predict(frame, imgsz=640, conf=0.25, classes=[0], verbose=False)
car_boxes = []
for r in results:
    for box in r.boxes.xyxy.tolist():
        x1, y1, x2, y2 = map(int, box)
        car_boxes.append((x1, y1, x2, y2))

print(f"총 감지된 차량: {len(car_boxes)}")
print("\n6번과 14번 슬롯 분석:")

for slot_num in [6, 14]:
    idx = slot_num - 1
    pts = slots[idx]
    expanded_pts = expand_polygon(pts, margin=25)
    expanded_np = np.array(expanded_pts, dtype=np.int32)
    
    slot_center = (sum(p[0] for p in pts)//4, sum(p[1] for p in pts)//4)
    print(f"\n슬롯 #{slot_num}:")
    print(f"  원본 좌표: {pts}")
    print(f"  중심: {slot_center}")
    print(f"  확장 좌표: {expanded_pts}")
    
    occupied = False
    matched_cars = []
    for car_idx, (x1, y1, x2, y2) in enumerate(car_boxes):
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        result = cv2.pointPolygonTest(expanded_np, (cx, cy), True)
        if result >= 0:
            occupied = True
            matched_cars.append({
                'idx': car_idx,
                'box': (x1, y1, x2, y2),
                'center': (cx, cy),
                'distance': result
            })
    
    print(f"  상태: {'점유' if occupied else '빈자리'}")
    print(f"  매칭된 차량 수: {len(matched_cars)}")
    for car in matched_cars:
        print(f"    - 차량#{car['idx']}: 박스={car['box']}, 중심={car['center']}, 거리={car['distance']:.1f}")

# 근처 차량들도 확인
print("\n\n슬롯 6번과 14번 근처 모든 차량:")
for slot_num in [6, 14]:
    idx = slot_num - 1
    pts = slots[idx]
    slot_center = (sum(p[0] for p in pts)//4, sum(p[1] for p in pts)//4)
    
    print(f"\n슬롯 #{slot_num} 중심 {slot_center} 근처:")
    nearby = []
    for car_idx, (x1, y1, x2, y2) in enumerate(car_boxes):
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        dist = ((cx - slot_center[0])**2 + (cy - slot_center[1])**2)**0.5
        if dist < 100:  # 100픽셀 이내
            nearby.append({'idx': car_idx, 'center': (cx, cy), 'distance': dist})
    
    nearby.sort(key=lambda x: x['distance'])
    for car in nearby[:5]:  # 가장 가까운 5개
        print(f"  차량#{car['idx']}: 중심={car['center']}, 거리={car['distance']:.1f}px")

cap.release()
