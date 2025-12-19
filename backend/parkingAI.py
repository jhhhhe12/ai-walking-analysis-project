from ultralytics import YOLO
import cv2
import numpy as np

# 모델 로드
model = YOLO("runs/detect/train7/weights/best.pt")

# 입력/출력 영상 경로
video_path = "videos/vecteezy_time-lapse-of-parking-lot-of-shopping-center-filled-with_43486929.mp4"
out_path = "predict_patch.mp4"

# 영상 열기
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

# 슬라이딩 윈도우 설정
patch_size = 960   # 윈도우 크기
stride = 480       # 겹치는 거리

# 영상 프레임 처리
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    bboxes_all = []

    # 슬라이딩 윈도우
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            patch = frame[y:min(y+patch_size, h), x:min(x+patch_size, w)]
            results = model(patch, imgsz=640, conf=0.15, iou=0.15, augment=True)
            
            # patch bbox → 원본 좌표 변환
            for r in results:
                for box in r.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = box
                    bboxes_all.append([int(x1)+x, int(y1)+y, int(x2)+x, int(y2)+y])

    # NMS 적용
    if bboxes_all:
        boxes = np.array(bboxes_all).tolist()
        scores = [1.0]*len(boxes)
        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.01, nms_threshold=0.15)
        
        if indices is not None:
            # OpenCV 버전마다 indices 구조가 다를 수 있으므로 flatten
            indices = indices.flatten()  
            for i in indices:
                x1, y1, x2, y2 = boxes[i]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    # 출력 영상에 쓰기
    out_video.write(frame)

cap.release()
out_video.release()
print(f"슬라이딩 윈도우 기반 탐지 완료! 출력: {out_path}")
