from ultralytics import YOLO
import cv2
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# 1️⃣ 영상 선택
Tk().withdraw()
video_path = askopenfilename(
    title="영상 파일 선택",
    
    filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv"), ("All files", "*.*")]
)
if not video_path:
    print("❌ 파일을 선택해야 합니다.")
    exit()

print(f"🎬 선택된 파일: {video_path}")

# 2️⃣ YOLO 모델 로드
model_path = "model/best.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"🚨 모델 파일이 없습니다: {model_path}")

print(f"📦 모델 로드 중: {model_path}")
model = YOLO(model_path)

tracker_path = r"C:\Users\ggp03\miniconda3\envs\walk\Lib\site-packages\ultralytics\cfg\trackers\bytetrack.yaml"

# 3️⃣ Tracking 모드 적용 (깜빡임 해결)
results = model.track(
    source=video_path,
    conf=0.15,      # 신뢰도 기준 설정  
    iou=0.50,       # 박스 겹침 기준
    imgsz=1280,     # 고해상도 분석
    project="runs/detect",
    name="predict",
    exist_ok=True,
    stream=True,
    tracker=tracker_path,  # 안정적 추적
    persist=True               # ID 유지-> 깜빡인 최소화 
)

# 4️⃣ 후처리: 작은 박스 제거 + ID 라벨 표시
output_video_path = f"videos/optimized_{os.path.basename(video_path)}"

cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# 작은 박스 기준 완화 (영상 크기 0.66% 이상)
min_size = max(width, height) // 200

for result in results:
    frame = result.orig_img.copy()
    boxes = result.boxes.xyxy
    scores = result.boxes.conf
    classes = result.boxes.cls
    ids = result.boxes.id  # tracking ID

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        conf_score = float(scores[i])
        cls_id = int(classes[i])
        track_id = int(ids[i]) if ids is not None else -1

        # 작은 박스 제거
        if (x2 - x1) < min_size or (y2 - y1) < min_size:
            continue

        label = f"ID:{track_id} {conf_score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(y1-10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()
print(f"📌 최적화 영상 생성 완료: {output_video_path}")

# 5️⃣ Windows 자동 실행
if os.name == "nt":
    os.startfile(output_video_path)
