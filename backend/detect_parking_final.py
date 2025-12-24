import cv2
import csv
import numpy as np
from ultralytics import YOLO

#자신의 경로에 맞게
VIDEO_PATH = "videos/video_preview_h264.mp4"
CSV_PATH = "parking_slots.csv"
OUTPUT_PATH = "videos/output/output_parking.mp4"
MODEL_PATH = "model/best.pt"

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

# 차량 중심점이 슬롯에 있는지 확인 (iou -> 변경)
def is_car_center_in_slot(car_box, slot_points):
    x1, y1, x2, y2 = car_box
    
    # 차량의 중심점(Center Point) 계산
    car_center_x = int((x1 + x2) / 2)
    car_center_y = int((y1 + y2) / 2)
    car_center = (car_center_x, car_center_y)

    slot_cnt = np.array(slot_points, dtype=np.int32)

    # pointPolygonTest: 점이 다각형 안에 있으면 양수(+), 밖에 있으면 음수(-) 반환
    # measureDist=False로 하면 안에 있으면 +1, 경계면 0, 밖이면 -1 반환
    result = cv2.pointPolygonTest(slot_cnt, car_center, False)
    
    return result >= 0  # 0보다 크거나 같으면 안에 있는 것

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

        results = model.predict(frame, imgsz=IMGSZ, conf=CONFIDENCE, classes=[0], verbose=False)    #클래스 0 : car
        
        car_boxes = []
        for r in results:
            for box in r.boxes.xyxy.tolist():
                x1, y1, x2, y2 = map(int, box)
                car_boxes.append((x1, y1, x2, y2))
                # 차량 박스 (노란색, 얇게)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

        # 2. 슬롯별 점유 여부 확인
        occupied_count = 0
        
        for idx, slot in enumerate(slots):
            # 이 슬롯 안에 중심점이 들어간 차량이 하나라도 있으면 점유
            is_occupied = False
            for car in car_boxes:
                if is_car_center_in_slot(car, slot):
                    is_occupied = True
                    break
            
            if is_occupied:
                occupied_count += 1
                color = (0, 0, 255) # 빨강 (점유)
                thickness = 2
            else:
                color = (0, 255, 0) # 초록 (비어있음)
                thickness = 1 # 빈 자리는 얇게

            # 슬롯 그리기
            pts = np.array(slot, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, color, thickness)
            
            # 슬롯 번호 작게 표시
            text_pos = slot[0] # 첫 번째 점 위치에 텍스트
            cv2.putText(frame, str(idx+1), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 3. 전체 현황 표시
        empty_count = total_slots - occupied_count
        info_text = f"Total: {total_slots} | Occupied: {occupied_count} | Empty: {empty_count}"
        
        # 상단 배경 박스
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