import cv2
import csv
import numpy as np
from ultralytics import YOLO
import os

CSV_PATH = "slots.csv"
MODEL_PATH = "best.pt"
FALLBACK_MODEL = "yolov8n.pt"

CONFIDENCE = 0.25  # Standard confidence for trained model
IMGSZ = 640  # Reduced for faster processing

# Global model cache
_model_cache = None

def load_slots(csv_path):
    slots = []
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader)  # 헤더 건너뛰기
            for row in reader:
                if not row or len(row) < 9:
                    continue  # 빈 줄이나 잘못된 줄 무시
                # 좌표만 가져오기
                pts = list(map(int, row[1:9]))
                pts_tuples = [(pts[i], pts[i+1]) for i in range(0, 8, 2)]
                slots.append(pts_tuples)
        return slots
    except FileNotFoundError:
        print(f"오류: {csv_path} 파일을 찾을 수 없습니다.")
        return []

def is_car_overlaps_slot(car_box, slot_points, threshold=0.2):
    """
    차량이 슬롯과 겹치는지 확인 (IoU 방식)
    threshold: 최소 겹침 비율 (0.2 = 20%)
    """
    x1, y1, x2, y2 = car_box
    
    # 슬롯의 bounding box 계산
    slot_np = np.array(slot_points, dtype=np.int32)
    slot_x_min = min(p[0] for p in slot_points)
    slot_x_max = max(p[0] for p in slot_points)
    slot_y_min = min(p[1] for p in slot_points)
    slot_y_max = max(p[1] for p in slot_points)
    
    # 겹치는 영역 계산
    overlap_x1 = max(x1, slot_x_min)
    overlap_y1 = max(y1, slot_y_min)
    overlap_x2 = min(x2, slot_x_max)
    overlap_y2 = min(y2, slot_y_max)
    
    # 겹치는 영역이 없으면 False
    if overlap_x1 >= overlap_x2 or overlap_y1 >= overlap_y2:
        return False
    
    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
    car_area = (x2 - x1) * (y2 - y1)
    
    # 차량 영역 대비 겹치는 비율
    if car_area == 0:
        return False
    
    overlap_ratio = overlap_area / car_area
    return overlap_ratio >= threshold

def analyze_parking_video(video_path):
    global _model_cache
    print(f"AI 분석 시작: {video_path}")
    
    slots = load_slots(CSV_PATH)
    if not slots:
        print("슬롯 정보가 없습니다.")
        return {}

    # Load model once and cache it
    if _model_cache is None:
        try:
            if os.path.exists(MODEL_PATH):
                _model_cache = YOLO(MODEL_PATH)
                print(f"모델 로드 성공: {MODEL_PATH}")
            else:
                print(f"{MODEL_PATH} 없음, fallback 모델 사용")
                _model_cache = YOLO(FALLBACK_MODEL)
        except Exception as e:
            print(f"모델 로드 실패: {e}, fallback 모델 사용")
            _model_cache = YOLO(FALLBACK_MODEL)
    else:
        print("캐시된 모델 사용")
    
    model = _model_cache
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("영상을 열 수 없습니다.")
        return {}

    final_status = {}
    slot_history = {idx + 1: [] for idx in range(len(slots))}  # 각 슬롯의 점유 이력 추적

    total_car_count = 0
    frame_count = 0
    processed_frames = 0
    FRAME_SKIP = 30  # Increased for faster processing
    HISTORY_SIZE = 5  # 최근 5개 프레임 추적

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # Skip frames for faster processing
        if frame_count % FRAME_SKIP != 0:
            continue
        
        processed_frames += 1

        results = model.predict(frame, imgsz=IMGSZ, conf=CONFIDENCE, classes=[0], verbose=False)  # Trained model uses class 0
        
        car_boxes = []
        current_car_count = 0

        for r in results:
            boxes_list = r.boxes.xyxy.tolist()
            current_car_count += len(boxes_list)

            for box in boxes_list:
                x1, y1, x2, y2 = map(int, box)
                car_boxes.append((x1, y1, x2, y2))
        
        if processed_frames % 10 == 0:
            print(f"프레임 {frame_count}: {current_car_count}대 검출")

        # 현재 프레임의 슬롯 점유 상태 확인 및 이력 업데이트
        for idx, slot in enumerate(slots):
            slot_id = idx + 1 # 슬롯 ID는 1부터 시작
            is_occupied = False
            
            for car in car_boxes:
                if is_car_center_in_slot(car, slot):
                    is_occupied = True
                    break
            
            # 이력에 추가
            slot_history[slot_id].append(is_occupied)
            if len(slot_history[slot_id]) > HISTORY_SIZE:
                slot_history[slot_id].pop(0)
            
            # 최근 프레임의 과반수로 최종 상태 결정 (안정적 판단)
            if len(slot_history[slot_id]) >= 3:
                occupied_count = sum(slot_history[slot_id])
                final_status[slot_id] = occupied_count >= (len(slot_history[slot_id]) / 2)
            else:
                final_status[slot_id] = is_occupied
        
        total_car_count += current_car_count

    cap.release()
    print(f"분석 종료 - 총 프레임: {frame_count}, 처리된 프레임: {processed_frames}")
    print(f"슬롯 상태: {len(final_status)}개 슬롯")
    print(f"총 검출 차량 수 합계: {total_car_count}")
    
    avg_car_count = 0
    if processed_frames > 0:
        avg_car_count = int(total_car_count / processed_frames)
    
    print(f"평균 차량 수: {avg_car_count}")
    
    vehicle_counts = {
        "car": avg_car_count
    }

    result = {
        "spaces": final_status,
        "vehicles": vehicle_counts,
        "slots": {idx+1: slot for idx, slot in enumerate(slots)}
    }
    print(f"반환 결과: vehicles={avg_car_count}, spaces={len(final_status)}")
    return result