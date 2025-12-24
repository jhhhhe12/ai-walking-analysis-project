import cv2
import numpy as np
import csv

VIDEO_PATH = "videos/video_preview_h264.mp4"
OUTPUT_CSV = "parking_slots2.csv"
ROWS_PER_BLOCK = 17       #세로로 17, 가로로 2개씩으로 쪼개기
COLS_PER_BLOCK = 2        

drawing = False
points = []
all_slots = [] 
block_count = 0

def mouse_callback(event, x, y, flags, param):
    global points, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            print(f"Point added: {x}, {y}")

def get_grid_points(p1, p2, p3, p4, rows, cols):
    slots = []
    left_line = np.linspace(p1, p4, rows + 1)
    right_line = np.linspace(p2, p3, rows + 1)
    
    for i in range(rows):
        l_top, l_btm = left_line[i], left_line[i+1]
        r_top, r_btm = right_line[i], right_line[i+1]
        
        top_edge = np.linspace(l_top, r_top, cols + 1)
        btm_edge = np.linspace(l_btm, r_btm, cols + 1)
        
        for j in range(cols):
            p_tl, p_tr = top_edge[j], top_edge[j+1]
            p_br, p_bl = btm_edge[j+1], btm_edge[j]
            slot_cnt = np.array([p_tl, p_tr, p_br, p_bl], np.int32)
            slots.append(slot_cnt)
    return slots

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: {VIDEO_PATH} 영상을 열 수 없습니다.")
    exit()

# 첫 번째 프레임
ret, img = cap.read()
cap.release() # 프레임 하나만 따고 비디오는 닫음

if not ret:
    print("Error: 영상에서 프레임을 읽어올 수 없습니다.")
    exit()

print(f"영상 해상도 로드 완료: {img.shape[1]}x{img.shape[0]}")

clone = img.copy()
cv2.namedWindow("Parking Slot Generator (Video Based)")
cv2.setMouseCallback("Parking Slot Generator (Video Based)", mouse_callback)

temp_slots = []

while True:
    display_img = clone.copy()
    
    # 저장된 슬롯 (녹색)
    for slot in all_slots:
        cv2.polylines(display_img, [slot], True, (0, 255, 0), 2)
    
    # 찍는 중인 점 (빨간색)
    for pt in points:
        cv2.circle(display_img, pt, 5, (0, 0, 255), -1)
    
    # 미리보기 그리드 (파란색)
    if len(points) == 4:
        temp_slots = get_grid_points(points[0], points[1], points[2], points[3], ROWS_PER_BLOCK, COLS_PER_BLOCK)
        for slot in temp_slots:
            cv2.polylines(display_img, [slot], True, (255, 255, 0), 2)
        cv2.putText(display_img, "Press 's': Save / 'r': Reset", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Parking Slot Generator (Video Based)", display_img)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("r"):
        points = []
        temp_slots = []
        print("Reset current block.")
        
    elif key == ord("s"):
        if len(points) == 4:
            all_slots.extend(temp_slots)
            points = []
            temp_slots = []
            block_count += 1
            print(f"Block {block_count} saved. Total slots: {len(all_slots)}")
            clone = display_img.copy()
        else:
            print("4점을 모두 찍으세요.")

    elif key == ord("q"):
        break

cv2.destroyAllWindows()

# CSV 저장
print(f"Saving {len(all_slots)} slots to {OUTPUT_CSV}...")
with open(OUTPUT_CSV, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['slot_id', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'])
    for idx, slot in enumerate(all_slots):
        flat_coords = slot.flatten()
        row = [f"slot_{idx}"] + flat_coords.tolist()
        writer.writerow(row)