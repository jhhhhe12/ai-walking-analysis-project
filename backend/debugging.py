import cv2
import csv
import numpy as np
from ultralytics import YOLO

# -------------------------------
# 설정 (경로를 본인 환경에 맞게 수정하세요)
# -------------------------------
VIDEO_PATH = "videos/test.mp4"
CSV_PATH = "parking_slots.csv"
OUTPUT_PATH = "videos/output/output_parking_(7).mp4"
MODEL_PATH = "best.pt"  # 학습된 YOLO 모델 또는 yolov8n.pt (테스트용)

CONFIDENCE = 0.25
IMGSZ = 1280

# -------------------------------
# 슬롯 로드 (코드 유지)
# -------------------------------
def load_slots(csv_path):
    slots = []
    try:
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # 헤더 건너뛰기
            for row in reader:
                pts = list(map(int, row[1:9]))
                pts_tuples = [(pts[i], pts[i+1]) for i in range(0, 8, 2)]
                slots.append(pts_tuples)
        print(f"총 {len(slots)}개의 주차 슬롯을 로드했습니다.")
        return slots
    except FileNotFoundError:
        print(f"오류: {csv_path} 파일을 찾을 수 없습니다.")
        return []

# -------------------------------
# 차량 중심점이 슬롯에 있는지 확인 (최적화 코드 유지)
# -------------------------------
def is_car_center_in_slot(car_box, slot_points):
    x1, y1, x2, y2 = car_box
    car_center_x = int((x1 + x2) / 2)
    car_center_y = int((y1 + y2) / 2)
    car_center = (car_center_x, car_center_y)
    slot_cnt = np.array(slot_points, dtype=np.int32)
    result = cv2.pointPolygonTest(slot_cnt, car_center, False)
    return result >= 0

# -------------------------------
def main():
    slots = load_slots(CSV_PATH)
    if not slots: return

    total_slots = len(slots)
    model = YOLO(MODEL_PATH) 

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"영상을 열 수 없습니다. ({VIDEO_PATH})")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    print("--- 주차 감지 시작 (종료하려면 'q'를 누르세요) ---")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # =========================================================
        # [수정된 부분]: classes=[2] 옵션 제거
        # 모델이 탐지하는 모든 것을 일단 확인합니다.
        results = model.predict(frame, imgsz=IMGSZ, conf=CONFIDENCE, verbose=False) 
        # =========================================================
        
        car_boxes = []
        for r in results:
            for box in r.boxes.xyxy.tolist():
                x1, y1, x2, y2 = map(int, box)
                
                # 박스 위에 인식된 클래스 ID를 출력하여 확인합니다.
                # (0, 255, 255)는 노란색
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
                
                # 클래스 ID 확인용 (디버깅)
                try:
                    class_id = int(box[4]) if len(box) > 4 else 0 # 5번째 요소가 클래스 ID일 수 있음
                    label_name = model.names[class_id]
                    cv2.putText(frame, f"{label_name} ({class_id})", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                except Exception:
                    pass
                
                # 현재는 모든 박스를 차량으로 간주하고 처리합니다.
                car_boxes.append((x1, y1, x2, y2))
                

        # 2. 슬롯별 점유 여부 확인 (코드 유지)
        occupied_count = 0
        for idx, slot in enumerate(slots):
            is_occupied = any(is_car_center_in_slot(car, slot) for car in car_boxes)
            
            if is_occupied:
                occupied_count += 1
                color = (0, 0, 255) # 빨강 (점유)
                thickness = 2
            else:
                color = (0, 255, 0) # 초록 (비어있음)
                thickness = 1 

            # 슬롯 그리기 및 번호 표시
            pts = np.array(slot, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, color, thickness)
            text_pos = slot[0]
            cv2.putText(frame, str(idx+1), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 3. 전체 현황 표시 (코드 유지)
        empty_count = total_slots - occupied_count
        info_text = f"Total: {total_slots} | Occupied: {occupied_count} | Empty: {empty_count}"
        cv2.rectangle(frame, (0, 0), (550, 40), (0, 0, 0), -1)
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out_video.write(frame)
        cv2.imshow("Parking Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out_video.release()
    cv2.destroyAllWindows()
    print("분석 완료")

if __name__ == "__main__":
    main()