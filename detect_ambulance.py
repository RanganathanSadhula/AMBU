import torch
from ultralytics import YOLO
import cv2
import argparse
from collections import deque
import numpy as np
from models.lstm_detector import LSTMDetector
from utils.visualization import draw_detection

class AmbulanceDetector:
    def __init__(self, yolo_model_path, lstm_model_path=None, sequence_length=10):
        self.yolo_model = YOLO(yolo_model_path)
        self.lstm_model = LSTMDetector()
        self.sequence_length = sequence_length
        self.detection_history = deque(maxlen=sequence_length)
        self.hidden = None
        
        # Load LSTM weights if available
        if lstm_model_path:
            try:
                self.lstm_model.load_state_dict(torch.load(lstm_model_path))
                print("LSTM weights loaded successfully")
            except:
                print("Using initialized LSTM weights")
        
        self.lstm_model.eval()
    
    def _prepare_sequence(self, detection):
        if len(detection) == 0:
            return torch.zeros(1, 1, 4)
        
        # Use the highest confidence detection
        best_det = max(detection, key=lambda x: x[1])  # x[1] is confidence
        bbox = best_det[0]
        
        # Normalize bbox coordinates
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        features = torch.tensor([[x_center, y_center, width, height]], dtype=torch.float32)
        return features.unsqueeze(0)
    
    def process_frame(self, frame):
        # YOLO detection
        results = self.yolo_model(frame)[0]
        detections = []
        
        # Process YOLO detections
        for box in results.boxes:
            if box.cls == 0:  # Ambulance class
                bbox = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                detections.append((bbox, confidence))
        
        # LSTM processing
        if detections:
            sequence = self._prepare_sequence(detections)
            with torch.no_grad():
                confidence_adjustment, self.hidden = self.lstm_model(sequence, self.hidden)
                
            # Adjust confidences
            adjusted_detections = []
            for bbox, conf in detections:
                adjusted_conf = conf * confidence_adjustment.item()
                if adjusted_conf > 0.25:  # Confidence threshold
                    adjusted_detections.append((bbox, adjusted_conf))
            
            # Visualize detections
            for bbox, conf in adjusted_detections:
                frame = draw_detection(frame, bbox, conf)
        
        return frame
    
    def detect_image(self, image_path, output_path=None):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Process image
        output_image = self.process_frame(image)
        
        # Save or display result
        if output_path:
            cv2.imwrite(output_path, output_image)
            print(f"Results saved to: {output_path}")
        else:
            cv2.imshow('Ambulance Detection', output_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def detect_video(self, video_path, output_path=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Initialize video writer if output path is provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            output_frame = self.process_frame(frame)
            
            # Save or display frame
            if output_path:
                out.write(output_frame)
            else:
                cv2.imshow('Ambulance Detection', output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Cleanup
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Detect ambulances in images or videos')
    parser.add_argument('--yolo-model', default='runs/detect/ambulance_detector/weights/best.pt',
                      help='Path to YOLO model weights')
    parser.add_argument('--lstm-model', default=None,
                      help='Path to LSTM model weights (optional)')
    parser.add_argument('--source', required=True,
                      help='Path to image or video file')
    parser.add_argument('--output', help='Path to save output (optional)')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = AmbulanceDetector(args.yolo_model, args.lstm_model)
    
    # Process input
    if args.source.endswith(('.jpg', '.jpeg', '.png')):
        detector.detect_image(args.source, args.output)
    elif args.source.endswith(('.mp4', '.avi', '.mov')):
        detector.detect_video(args.source, args.output)
    else:
        raise ValueError("Unsupported file format")

if __name__ == "__main__":
    main()