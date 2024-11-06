import torch
import torch.nn as nn
from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque
import time

class LSTMDetector(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2):
        super(LSTMDetector, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        else:
            h0, c0 = hidden
            
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out, (hn, cn)

class AmbulanceDetector:
    def __init__(self, yolo_model_path='best.pt', sequence_length=10):
        self.yolo_model = YOLO(yolo_model_path)
        self.lstm_model = LSTMDetector()
        self.sequence_length = sequence_length
        self.detection_history = deque(maxlen=sequence_length)
        self.hidden = None
        
        # Load LSTM weights if available
        try:
            self.lstm_model.load_state_dict(torch.load('lstm_weights.pth'))
            print("LSTM weights loaded successfully")
        except:
            print("No pre-trained LSTM weights found. Using initialized weights.")
        
        self.lstm_model.eval()
        
    def _prepare_sequence(self, detection):
        """Convert detection to LSTM input format"""
        if len(detection) == 0:
            return torch.zeros(1, 1, 4)
        
        # Use the highest confidence detection
        best_det = max(detection, key=lambda x: x['confidence'])
        bbox = best_det['bbox']
        
        # Normalize bbox coordinates
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        features = torch.tensor([[x_center, y_center, width, height]], dtype=torch.float32)
        return features.unsqueeze(0)
    
    def detect_image(self, image_path, output_path=None):
        image = cv2.imread(image_path)
        results = self.yolo_model(image)
        detections = self._process_results(results)
        
        # Prepare sequence for LSTM
        sequence = self._prepare_sequence(detections)
        with torch.no_grad():
            confidence_adjustment, self.hidden = self.lstm_model(sequence, self.hidden)
        
        # Adjust detections based on LSTM confidence
        for det in detections:
            det['confidence'] *= confidence_adjustment.item()
        
        output_image = self._draw_detections(image, detections)
        
        if output_path:
            cv2.imwrite(output_path, output_image)
        return detections
    
    def detect_video(self, video_path, output_path=None):
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        self.hidden = None  # Reset LSTM hidden state
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # YOLO detection
            results = self.yolo_model(frame)
            detections = self._process_results(results)
            
            # LSTM processing
            sequence = self._prepare_sequence(detections)
            with torch.no_grad():
                confidence_adjustment, self.hidden = self.lstm_model(sequence, self.hidden)
            
            # Adjust detections based on LSTM confidence
            for det in detections:
                det['confidence'] *= confidence_adjustment.item()
            
            # Draw detections
            output_frame = self._draw_detections(frame, detections)
            self._add_progress_info(output_frame, (frame_count / total_frames) * 100, len(detections))
            
            if output_path:
                out.write(output_frame)
            else:
                cv2.imshow('Ambulance Detection', output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
    
    def _draw_detections(self, frame, detections):
        output_frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            conf = det['confidence']
            
            # Draw filled rectangle with transparency
            overlay = output_frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
            output_frame = cv2.addWeighted(overlay, 0.3, output_frame, 0.7, 0)
            
            # Draw border
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label with confidence
            label = f'Ambulance: {conf:.2f}'
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            y1_label = max(y1, label_size[1])
            
            cv2.rectangle(output_frame, 
                        (x1, y1_label - label_size[1] - 10),
                        (x1 + label_size[0], y1_label + baseline - 10), 
                        (0, 255, 0), 
                        cv2.FILLED)
            
            cv2.putText(output_frame, 
                       label,
                       (x1, y1_label - 7),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, 
                       (0, 0, 0),
                       2)
        
        return output_frame
    
    def _add_progress_info(self, frame, progress, num_detections):
        frame_h, frame_w = frame.shape[:2]
        bar_w = int(frame_w * 0.8)
        bar_h = 20
        bar_x = int((frame_w - bar_w) / 2)
        bar_y = frame_h - 40
        
        cv2.rectangle(frame,
                     (bar_x, bar_y),
                     (bar_x + bar_w, bar_y + bar_h),
                     (0, 0, 0),
                     cv2.FILLED)
        
        progress_w = int(bar_w * (progress / 100))
        cv2.rectangle(frame,
                     (bar_x, bar_y),
                     (bar_x + progress_w, bar_y + bar_h),
                     (0, 255, 0),
                     cv2.FILLED)
        
        cv2.putText(frame,
                   f'Progress: {progress:.1f}% | Detections: {num_detections}',
                   (bar_x, bar_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.6,
                   (255, 255, 255),
                   2)
    
    def _process_results(self, results):
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                if box.cls == 0:  # Ambulance class
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = box.conf.item()
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence
                    })
        return detections

def train_models():
    # Train YOLO model
    yolo_model = YOLO('yolov8n.pt')
    yolo_model.train(
        data='ambulance_data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='ambulance_detector',
        patience=50,
        save=True,
        device='0' if torch.cuda.is_available() else 'cpu'
    )
    
    # Train LSTM model (assuming we have sequence data)
    lstm_model = LSTMDetector()
    # Add LSTM training code here if you have temporal sequence data
    
    return yolo_model, lstm_model

if __name__ == "__main__":
    # Train models
    yolo_model, lstm_model = train_models()
    
    # Initialize detector with trained models
    detector = AmbulanceDetector('runs/detect/ambulance_detector/weights/best.pt')
    
    # Process image
    detector.detect_image('test_image.jpg', 'output_image.jpg')
    
    # Process video
    detector.detect_video('test_video.mp4', 'output_video.mp4')