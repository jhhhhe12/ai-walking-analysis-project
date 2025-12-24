from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from ai_module import load_slots
import os
import shutil
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

UPLOAD_FOLDER = "temp_videos"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = YOLO("best.pt")
slots = load_slots("slots.csv")

# [핵심 변경 1] 현재 분석 결과 공유를 위한 전역 변수 선언
# 실제 서비스 단계에서는 Redis 같은 DB나 세션별 관리가 필요하지만, 데모용으로는 전역 변수가 가장 간단합니다.
latest_analysis_result = {
    "vehicles": [{"type": "car", "count": 0}],
    "spaces": [{"id": i+1, "occupied": 0} for i in range(len(slots))]
}

@app.post("/analyze")
async def upload_video(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_FOLDER, "current.mp4")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"message": "영상 업로드 완료", "file": "current.mp4"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"업로드 실패: {e}")

# 슬롯 영역 확장 함수 (더 나은 감지를 위해)
def expand_polygon(points, margin=10):
    """폴리곤 영역을 margin 픽셀만큼 확장"""
    # 중심점 계산
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    
    # 각 점을 중심에서 멀어지는 방향으로 확장
    expanded = []
    for (x, y) in points:
        # 중심에서 점까지의 방향 벡터
        dx = x - cx
        dy = y - cy
        # 벡터 정규화
        length = (dx**2 + dy**2)**0.5
        if length > 0:
            dx /= length
            dy /= length
        # margin만큼 확장
        expanded.append((int(x + dx * margin), int(y + dy * margin)))
    return expanded

# 오버레이 함수 (중심점 기반 감지 + 확장된 슬롯)
def draw_overlay(frame, car_boxes, slots):
    for idx, pts in enumerate(slots):
        # 원본 슬롯 표시
        pts_np = np.array(pts, dtype=np.int32)
        
        # 감지용으로는 확장된 영역 사용 (margin 증가)
        expanded_pts = expand_polygon(pts, margin=35)
        expanded_np = np.array(expanded_pts, dtype=np.int32)
        
        occupied = False
        matching_car = None
        for (x1, y1, x2, y2) in car_boxes:
            cx, cy = (x1 + x2)//2, (y1 + y2)//2
            # 확장된 영역에서 중심점 체크
            if cv2.pointPolygonTest(expanded_np, (cx, cy), False) >= 0:
                occupied = True
                matching_car = (cx, cy)
                break
        
        slot_color = (0, 0, 255) if occupied else (0, 255, 0)
        cv2.polylines(frame, [pts_np], True, slot_color, 2)
        cv2.putText(frame, str(idx + 1), pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, slot_color, 2)
        
        # 6번과 14번 슬롯에 디버그 정보 표시
        if idx + 1 in [6, 14]:
            slot_center = (int(sum(p[0] for p in pts)/4), int(sum(p[1] for p in pts)/4))
            debug_text = f"#{idx+1}: {'OCC' if occupied else 'EMPTY'}"
            if matching_car:
                debug_text += f" ({matching_car[0]},{matching_car[1]})"
            cv2.putText(frame, debug_text, (slot_center[0]-20, slot_center[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    for (x1, y1, x2, y2) in car_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        # 차량 중심점 표시
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        cv2.circle(frame, (cx, cy), 3, (255, 0, 255), -1)
    
    return frame

# [핵심 변경 2] 스트리밍 함수에서 분석 데이터 업데이트 및 속도 조절
@app.get("/stream")
def stream_video(speed: int = 1): # speed 쿼리 파라미터 추가 (기본 1배속)
    video_path = os.path.join(UPLOAD_FOLDER, "current.mp4")
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="업로드된 영상이 없습니다.")

    cap = cv2.VideoCapture(video_path)
    
    # 실시간 감지를 위해 매 프레임마다 감지 (detect_interval = 1)
    detect_interval = 10
    # 영상 속도 조절을 위한 변수
    # speed가 2면 2프레임마다 1번 처리 (즉 2배 빠름), 3이면 3배 빠름
    skip_frames = speed 
    last_car_boxes = []
    
    def generate():
        nonlocal last_car_boxes
        global latest_analysis_result # 전역 변수 사용 선언
        frame_count = 0

        while True:  # 무한 반복
            ret, frame = cap.read()
            if not ret:
                # 영상이 끝나면 처음부터 다시 재생
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0
                continue
            
            frame_count += 1
            
            # [속도 조절 로직] 
            # 현재 프레임이 skip_frames의 배수가 아니면 건너뜀 (처리 안 함 -> 빨라짐)
            if speed > 1 and frame_count % skip_frames != 0:
                continue

            # 매 프레임마다 YOLO 감지 (실시간성 향상)
            if frame_count == 1 or frame_count % detect_interval == 0:
                # YOLO 감지
                results = model.predict(frame, imgsz=640, conf=0.25, classes=[0], verbose=False)
                
                car_boxes = []
                for r in results:
                    for box in r.boxes.xyxy.tolist():
                        x1, y1, x2, y2 = map(int, box)
                        car_boxes.append((x1, y1, x2, y2))
                
                last_car_boxes = car_boxes
            else:
                # 감지하지 않는 프레임에서는 마지막 결과 사용
                car_boxes = last_car_boxes

            # ---------------------------------------------------------
            # [중요] 여기서 최신 데이터를 전역 변수에 업데이트합니다.
            # ---------------------------------------------------------
            spaces_status = []
            for idx, pts in enumerate(slots):
                # 감지용으로는 확장된 영역 사용 (margin 증가)
                expanded_pts = expand_polygon(pts, margin=35)
                expanded_np = np.array(expanded_pts, dtype=np.int32)
                
                occupied = 0
                matched_car = None
                for (x1, y1, x2, y2) in car_boxes:
                    cx, cy = (x1 + x2)//2, (y1 + y2)//2
                    # 확장된 영역에서 중심점 체크
                    if cv2.pointPolygonTest(expanded_np, (cx, cy), False) >= 0:
                        occupied = 1
                        matched_car = (cx, cy)
                        break
                
                # 6번과 14번 슬롯 디버그 로그
                if idx + 1 in [6, 14] and frame_count % 30 == 0:  # 30프레임마다 로그
                    slot_center = (sum(p[0] for p in pts)//4, sum(p[1] for p in pts)//4)
                    print(f"Frame {frame_count} - Slot #{idx+1}: {'OCCUPIED' if occupied else 'EMPTY'}, "
                          f"Center: {slot_center}, Car: {matched_car}, Total cars: {len(car_boxes)}")
                
                spaces_status.append({"id": idx+1, "occupied": occupied})

            latest_analysis_result = {
                "vehicles": [{"type": "car", "count": len(car_boxes)}],
                "spaces": spaces_status
            }
            # ---------------------------------------------------------

            # 시각화
            frame = draw_overlay(frame, car_boxes, slots)
            _, jpeg = cv2.imencode(".jpg", frame)
            frame_bytes = jpeg.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# [핵심 변경 3] 결과 반환 API는 이제 단순히 저장된 최신 값을 리턴
@app.get("/parking_spaces")
def parking_spaces():
    # 더 이상 여기서 cv2.VideoCapture를 하지 않습니다.
    # 스트리밍 함수가 열심히 업데이트해 놓은 값을 그냥 가져갑니다.
    return latest_analysis_result