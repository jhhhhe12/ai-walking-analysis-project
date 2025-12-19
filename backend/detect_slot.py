import cv2
import csv
import numpy as np

VIDEO_PATH = "videos/test.mp4"  # 분석할 영상 경로
OUTPUT_CSV = "slots.csv"        # 생성될 슬롯 CSV 파일

slots = []          # 슬롯 좌표 저장
drawing = False     # 드래그 상태
ix, iy = -1, -1    # 드래그 시작점

def draw_rectangle(event, x, y, flags, param):
    """마우스 이벤트 콜백 함수"""
    global ix, iy, drawing, temp_frame, slots

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_frame = frame.copy()
            for slot in slots:
                pts = np.array(slot, np.int32).reshape((-1,1,2))
                cv2.polylines(temp_frame, [pts], True, (0,0,255), 2)
            cv2.rectangle(temp_frame, (ix, iy), (x, y), (0,255,0), 2)
            cv2.imshow("Define Slots", temp_frame)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = ix, iy
        x2, y2 = x, y

        # 슬롯 꼭짓점 좌표 4개 생성 (시계방향)
        slot = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        slots.append(slot)

        # 화면에 빨간색으로 표시
        temp_frame = frame.copy()
        for s in slots:
            pts = np.array(s, np.int32).reshape((-1,1,2))
            cv2.polylines(temp_frame, [pts], True, (0,0,255), 2)
        cv2.imshow("Define Slots", temp_frame)

        print(f"Slot added: {slot}")

# -------------------------------
# 영상 열기
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
if not ret:
    print("영상 읽기 실패")
    cap.release()
    exit()

temp_frame = frame.copy()

cv2.imshow("Define Slots", temp_frame)
cv2.setMouseCallback("Define Slots", draw_rectangle)
print("드래그로 슬롯 영역을 지정하세요. 완료 후 'q' 키를 누르세요.")

# -------------------------------
# 슬롯 지정 루프
while True:
    cv2.imshow("Define Slots", temp_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # q 누르면 종료
        break

cv2.destroyAllWindows()
cap.release()

# -------------------------------
# CSV 저장
with open(OUTPUT_CSV, "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['slot_id','x1','y1','x2','y2','x3','y3','x4','y4'])
    for idx, slot in enumerate(slots):
        row = [idx+1]
        for pt in slot:
            row.extend(pt)
        writer.writerow(row)

print(f"{OUTPUT_CSV} 저장 완료. 총 {len(slots)}개 슬롯")
