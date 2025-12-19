#!/usr/bin/env python3
"""
AI Backend API Server
Provides REST API endpoints for AI video processing and dataset management
"""

from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS
import os
import json
from datetime import datetime
from pathlib import Path
import subprocess
import cv2
import threading
import time
from ai_module import analyze_parking_video

app = Flask(__name__)
CORS(app)

# Configuration
BASE_DIR = Path('/opt/ai')
DATASET_PATH = BASE_DIR / 'dataset'
FRONTEND_PATH = BASE_DIR / 'frontend'
BACKEND_PATH = BASE_DIR / 'backend'
VIDEO_PATH = BACKEND_PATH / 'videos'
TEMP_VIDEO_PATH = BACKEND_PATH / 'temp_videos'

# Global state for video analysis
current_video = None
current_frame = None
analysis_results = {
    'vehicles': [{'count': 0}],
    'spaces': []
}
is_analyzing = False
video_lock = threading.Lock()
frame_lock = threading.Lock()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'OK',
        'message': 'Backend server is running',
        'timestamp': datetime.now().isoformat(),
        'paths': {
            'dataset': str(DATASET_PATH),
            'frontend': str(FRONTEND_PATH),
            'backend': str(BACKEND_PATH)
        }
    })

@app.route('/api/dataset/stats', methods=['GET'])
def dataset_stats():
    """Get dataset statistics"""
    try:
        # Count images
        train_images = len(list((DATASET_PATH / 'images' / 'train').glob('*'))) if (DATASET_PATH / 'images' / 'train').exists() else 0
        val_images = len(list((DATASET_PATH / 'images' / 'val').glob('*'))) if (DATASET_PATH / 'images' / 'val').exists() else 0
        test_images = len(list((DATASET_PATH / 'images' / 'test').glob('*'))) if (DATASET_PATH / 'images' / 'test').exists() else 0
        
        # Count labels
        train_labels = len(list((DATASET_PATH / 'labels' / 'train').glob('*.txt'))) if (DATASET_PATH / 'labels' / 'train').exists() else 0
        val_labels = len(list((DATASET_PATH / 'labels' / 'val').glob('*.txt'))) if (DATASET_PATH / 'labels' / 'val').exists() else 0
        test_labels = len(list((DATASET_PATH / 'labels' / 'test').glob('*.txt'))) if (DATASET_PATH / 'labels' / 'test').exists() else 0
        
        return jsonify({
            'images': {
                'train': train_images,
                'val': val_images,
                'test': test_images,
                'total': train_images + val_images + test_images
            },
            'labels': {
                'train': train_labels,
                'val': val_labels,
                'test': test_labels,
                'total': train_labels + val_labels + test_labels
            },
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/videos', methods=['GET'])
def list_videos():
    """List all videos"""
    try:
        videos = []
        if VIDEO_PATH.exists():
            videos = [f.name for f in VIDEO_PATH.iterdir() if f.is_file() and f.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']]
        
        return jsonify({
            'videos': videos,
            'count': len(videos),
            'path': str(VIDEO_PATH)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai/detect', methods=['POST'])
def ai_detect():
    """Run AI detection on video"""
    try:
        data = request.get_json()
        video_name = data.get('video')
        
        if not video_name:
            return jsonify({'error': 'Video name required'}), 400
        
        video_path = VIDEO_PATH / video_name
        if not video_path.exists():
            return jsonify({'error': 'Video not found'}), 404
        
        # Here you can integrate with ai_module.py
        return jsonify({
            'status': 'success',
            'message': f'AI detection started for {video_name}',
            'video': video_name
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/dataset/images/<split>', methods=['GET'])
def list_images(split):
    """List images from dataset split (train/val/test)"""
    try:
        if split not in ['train', 'val', 'test']:
            return jsonify({'error': 'Invalid split. Use train, val, or test'}), 400
        
        image_path = DATASET_PATH / 'images' / split
        if not image_path.exists():
            return jsonify({'images': [], 'count': 0})
        
        images = [f.name for f in image_path.iterdir() if f.is_file()]
        
        return jsonify({
            'split': split,
            'images': images,
            'count': len(images)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/dataset/labels/<split>', methods=['GET'])
def list_labels(split):
    """List labels from dataset split (train/val/test)"""
    try:
        if split not in ['train', 'val', 'test']:
            return jsonify({'error': 'Invalid split. Use train, val, or test'}), 400
        
        label_path = DATASET_PATH / 'labels' / split
        if not label_path.exists():
            return jsonify({'labels': [], 'count': 0})
        
        labels = [f.name for f in label_path.iterdir() if f.suffix == '.txt']
        
        return jsonify({
            'split': split,
            'labels': labels,
            'count': len(labels)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_video():
    """Upload and analyze video"""
    global current_video, is_analyzing, analysis_results
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Save uploaded video
        TEMP_VIDEO_PATH.mkdir(parents=True, exist_ok=True)
        video_path = TEMP_VIDEO_PATH / file.filename
        file.save(str(video_path))
        
        with video_lock:
            current_video = str(video_path)
            is_analyzing = True
        
        # Run analysis in background thread
        def run_analysis():
            global analysis_results, is_analyzing
            try:
                result = analyze_parking_video(str(video_path))
                
                # Convert to frontend format
                spaces_list = []
                if 'spaces' in result:
                    for slot_id, is_occupied in result['spaces'].items():
                        spaces_list.append({
                            'id': slot_id,
                            'occupied': 1 if is_occupied else 0
                        })
                
                with video_lock:
                    analysis_results = {
                        'vehicles': [{'count': result.get('vehicles', {}).get('car', 0)}],
                        'spaces': spaces_list
                    }
                    is_analyzing = False
            except Exception as e:
                print(f"Analysis error: {e}")
                with video_lock:
                    is_analyzing = False
        
        thread = threading.Thread(target=run_analysis, daemon=True)
        thread.start()
        
        return jsonify({
            'status': 'success',
            'message': 'Video uploaded and analysis started',
            'filename': file.filename
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze_test', methods=['POST'])
def analyze_test_video():
    """Analyze test video from server"""
    global current_video, is_analyzing, analysis_results
    try:
        # Use test.mp4 (matches slots.csv)
        video_path = VIDEO_PATH / 'test.mp4'
        if not video_path.exists():
            return jsonify({'error': 'Test video not found'}), 404
        
        with video_lock:
            current_video = str(video_path)
            is_analyzing = True
        
        # Run analysis in background thread
        def run_analysis():
            global analysis_results, is_analyzing
            try:
                result = analyze_parking_video(str(video_path))
                
                # Convert to frontend format
                spaces_list = []
                if 'spaces' in result:
                    for slot_id, is_occupied in result['spaces'].items():
                        spaces_list.append({
                            'id': slot_id,
                            'occupied': 1 if is_occupied else 0
                        })
                
                with video_lock:
                    analysis_results = {
                        'vehicles': [{'count': result.get('vehicles', {}).get('car', 0)}],
                        'spaces': spaces_list
                    }
                    is_analyzing = False
            except Exception as e:
                print(f"Analysis error: {e}")
                import traceback
                traceback.print_exc()
                with video_lock:
                    is_analyzing = False
        
        thread = threading.Thread(target=run_analysis, daemon=True)
        thread.start()
        
        return jsonify({
            'status': 'success',
            'message': 'Test video analysis started',
            'filename': 'test.mp4'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/parking_spaces', methods=['GET'])
def parking_spaces():
    """Get current parking space analysis results"""
    return jsonify(analysis_results)

# Global model for streaming
stream_model = None
stream_model_lock = threading.Lock()

def generate_stream():
    """Generate video stream with analysis overlay"""
    global current_video, analysis_results, stream_model
    from ai_module import load_slots, IMGSZ, CONFIDENCE
    from ultralytics import YOLO
    import numpy as np
    
    # Load model once globally
    with stream_model_lock:
        if stream_model is None:
            try:
                print("스트림 모델 로딩...")
                stream_model = YOLO(str(BACKEND_PATH / 'best.pt'))
                print("스트림 모델 로드 완료")
            except Exception as e:
                print(f"스트림 모델 로드 실패: {e}")
                stream_model = None
    
    # 슬롯 영역 확장 함수
    def expand_polygon(points, margin=35):
        """폴리곤 영역을 margin 픽셀만큼 확장"""
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)
        expanded = []
        for (x, y) in points:
            dx = x - cx
            dy = y - cy
            length = (dx**2 + dy**2)**0.5
            if length > 0:
                dx /= length
                dy /= length
            expanded.append((int(x + dx * margin), int(y + dy * margin)))
        return expanded
    
    while True:
        with video_lock:
            if current_video is None or not Path(current_video).exists():
                time.sleep(0.1)
                continue
            
            video_path = current_video
        
        cap = cv2.VideoCapture(video_path)
        
        # Load slots for drawing
        slots = load_slots(str(BACKEND_PATH / 'slots.csv'))
        
        try:
            frame_counter = 0
            detected_boxes = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                    continue
                
                frame_counter += 1
                
                # Real-time vehicle detection (매 프레임마다 감지)
                if stream_model is not None and frame_counter % 1 == 0:
                    try:
                        results = stream_model.predict(frame, imgsz=640, conf=0.25, classes=[0], verbose=False)
                        detected_boxes = []
                        for r in results:
                            boxes = r.boxes.xyxy.tolist()
                            detected_boxes.extend(boxes)
                        
                        # 실시간으로 슬롯 점유 상태 업데이트
                        spaces_status = []
                        for idx, slot_pts in enumerate(slots):
                            # 확장된 영역으로 감지
                            expanded_pts = expand_polygon(slot_pts, margin=35)
                            expanded_np = np.array(expanded_pts, dtype=np.int32)
                            
                            occupied = 0
                            for box in detected_boxes:
                                x1, y1, x2, y2 = map(int, box[:4])
                                cx, cy = (x1 + x2)//2, (y1 + y2)//2
                                # 중심점이 확장된 슬롯 영역 안에 있는지 확인
                                if cv2.pointPolygonTest(expanded_np, (cx, cy), False) >= 0:
                                    occupied = 1
                                    break
                            
                            spaces_status.append({"id": idx+1, "occupied": occupied})
                        
                        # analysis_results 업데이트
                        with video_lock:
                            analysis_results = {
                                'vehicles': [{'count': len(detected_boxes)}],
                                'spaces': spaces_status
                            }
                        
                    except Exception as e:
                        print(f"검출 오류: {e}")
                
                # Draw detected vehicles in YELLOW
                for box in detected_boxes:
                    x1, y1, x2, y2 = map(int, box[:4])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)  # Yellow
                
                # Draw slots with occupancy status
                with video_lock:
                    spaces_dict = {s['id']: s['occupied'] for s in analysis_results.get('spaces', [])}
                
                for idx, slot in enumerate(slots):
                    slot_id = idx + 1
                    is_occupied = spaces_dict.get(slot_id, 0)
                    
                    # Color: RED if occupied, GREEN if empty
                    color = (0, 0, 255) if is_occupied else (0, 255, 0)
                    
                    # Draw slot polygon
                    pts = np.array(slot, dtype=np.int32)
                    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
                    
                    # Draw slot number
                    center_x = int(sum(p[0] for p in slot) / 4)
                    center_y = int(sum(p[1] for p in slot) / 4)
                    cv2.putText(frame, str(slot_id), (center_x-10, center_y+5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw statistics
                vehicle_count = analysis_results.get('vehicles', [{}])[0].get('count', 0)
                occupied_count = sum(1 for s in spaces_dict.values() if s)
                
                cv2.putText(frame, f'Vehicles: {vehicle_count}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f'Occupied: {occupied_count}/{len(slots)}', (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f'Detected: {len(detected_boxes)}', (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                time.sleep(0.01)  # ~100 fps (빠른 재생)
        except Exception as e:
            print(f"스트림 오류: {e}")
        finally:
            cap.release()

@app.route('/api/stream', methods=['GET'])
def video_stream():
    """Stream video with analysis overlay"""
    return Response(generate_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/system/info', methods=['GET'])
def system_info():
    """Get system information"""
    try:
        import psutil
        
        return jsonify({
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'percent': psutil.virtual_memory().percent
            },
            'disk': {
                'total': psutil.disk_usage('/').total,
                'used': psutil.disk_usage('/').used,
                'free': psutil.disk_usage('/').free,
                'percent': psutil.disk_usage('/').percent
            },
            'timestamp': datetime.now().isoformat()
        })
    except ImportError:
        return jsonify({
            'message': 'Install psutil for system monitoring',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure directories exist
    VIDEO_PATH.mkdir(parents=True, exist_ok=True)
    TEMP_VIDEO_PATH.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("🤖 AI Backend API Server")
    print("="*60)
    print(f"📁 Dataset path: {DATASET_PATH}")
    print(f"🎬 Video path: {VIDEO_PATH}")
    print(f"🌐 Frontend path: {FRONTEND_PATH}")
    print(f"🚀 Server running on http://0.0.0.0:5000")
    print("="*60)
    
    app.run(host='0.0.0.0', port=5000, debug=False)
