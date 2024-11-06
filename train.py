import torch
from ultralytics import YOLO

def train_model():
    """Train the YOLO model for ambulance detection"""
    
    print("Starting model training...")
    
    # Initialize YOLO model
    model = YOLO('yolov8n.pt')
    
    # Training settings
    results = model.train(
        data='ambulance_data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='ambulance_detector',
        patience=50,
        save=True,
        device='0' if torch.cuda.is_available() else 'cpu'
    )
    
    print("\nTraining completed!")
    print(f"Best model saved at: {results.best}")

if __name__ == "__main__":
    train_model()