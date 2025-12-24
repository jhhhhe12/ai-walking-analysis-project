from ultralytics import YOLO
import cv2

# 학습된 모델 불러오기
model = YOLO("best.pt")

# 영상 열기
cap = cv2.VideoCapture("videos/test02.mp4")
IMGSZ = 1280
CONF = 0.25
IOU = 0.3

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 예측
    results = model.predict(frame, imgsz=IMGSZ, conf=CONF, iou=IOU, verbose=False)

    # YOLO가 그려주는 프레임 가져오기
    annotated_frame = results[0].plot()  # 첫 번째 결과만 사용

    # 화면에 출력
    cv2.imshow("YOLO Test", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
