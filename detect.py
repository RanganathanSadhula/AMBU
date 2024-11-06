import torch
from ultralytics import YOLO
import cv2
import argparse

def run_detection(model_path, source_path, output_path=None, conf_threshold=0.25):
    """Run detection on image or video"""
    
    # Initialize model
    model = YOLO(model_path)
    
    # Run inference
    results = model(source_path, conf=conf_threshold, save=True)
    
    if output_path:
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            cv2.imwrite(output_path, im_array)
    
    print(f"Detection completed! Results saved to: {output_path if output_path else 'runs/detect'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ambulance detection')
    parser.add_argument('--model', default='runs/detect/ambulance_detector/weights/best.pt',
                      help='Path to model weights')
    parser.add_argument('--source', required=True,
                      help='Path to image or video file')
    parser.add_argument('--output', help='Path to save output')
    parser.add_argument('--conf', type=float, default=0.25,
                      help='Confidence threshold')
    
    args = parser.parse_args()
    run_detection(args.model, args.source, args.output, args.conf)