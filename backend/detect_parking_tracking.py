import cv2
import csv
import numpy as np
from ultralytics import YOLO
import os

# 경로 설정
VIDEO_PATH = "videos/video_preview_h264.mp4"
CSV_PATH = "parking_slots.csv"
OUTPUT_PATH = "videos/output/output_parking_tracked.mp4"
MODEL_PATH = "model/best.pt"
TRACKER_YAML = r"C:\Users\ggp03\miniconda3\envs\walk\Lib\site-packages\ultralytics\cfg\trackers\bytetrack.yaml"

CONFIDENCE = 0.15
IMGSZ = 1280

def load_slots(csv_path):
    slots = []
    try:
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                pts = list(map(int, row[1:9]))
                pts_tuples = [(pts[i], pts[i+1]) for i in range(0, 8, 2)]
                slots.append(pts_tuples)
        print(f"총 주차슬롯 : {len(slots)}")
        return slots
    except FileNotFoundError:
        print(f"오류: {csv_path} 파일을 찾을 수 없습니다.")
        return []

def is_car_center_in_slot(car_box, slot_points):
    x1, y1, x2, y2 = car_box
    car_center = ((x1+x2)//2, (y1+y2)//2)
    slot_cnt = np.array(slot_points, dtype=np.int32)
    return cv2.pointPolygonTest(slot_cnt, car_center, False) >= 0

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

    print("--- 주차 감지 + ID 추적 시작 ---")

    # YOLO Tracking 적용
    results = model.track(
        source=VIDEO_PATH,
        conf=CONFIDENCE,
        imgsz=IMGSZ,
        tracker=TRACKER_YAML,
        persist=True,
        stream=True
    )

    min_size = max(width, height) // 200  # 작은 차량 무시 기준

    for result in results:
        frame = result.orig_img.copy()
        car_boxes = []
        car_ids = []

        # 차량 박스 + ID 수집
        for box, track_id in zip(result.boxes.xyxy, result.boxes.id):
            x1, y1, x2, y2 = map(int, box)
            if (x2 - x1) < min_size or (y2 - y1) < min_size:
                continue
            car_boxes.append((x1, y1, x2, y2))
            car_ids.append(int(track_id))

            # 차량 박스 + ID 표시
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, max(y1-10,0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

        # 슬롯별 점유 여부
        occupied_count = 0
        for idx, slot in enumerate(slots):
            is_occupied = False
            for car_box in car_boxes:
                if is_car_center_in_slot(car_box, slot):
                    is_occupied = True
                    break

            color = (0, 0, 255) if is_occupied else (0, 255, 0)
            thickness = 2 if is_occupied else 1
            if is_occupied:
                occupied_count += 1

            pts = np.array(slot, np.int32).reshape((-1,1,2))
            cv2.polylines(frame, [pts], True, color, thickness)
            cv2.putText(frame, str(idx+1), slot[0], cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 전체 정보 표시
        empty_count = total_slots - occupied_count
        info_text = f"Total: {total_slots} | Occupied: {occupied_count} | Empty: {empty_count}"
        cv2.rectangle(frame, (0,0), (550,40), (0,0,0), -1)
        cv2.putText(frame, info_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),2)

        out_video.write(frame)
        cv2.imshow("Parking Detection with ID", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out_video.release()
    cv2.destroyAllWindows()
    print(f"완료! 결과 저장: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
