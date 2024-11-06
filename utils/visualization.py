import cv2
import numpy as np

def draw_detection(frame, box, confidence, track_id=None):
    x1, y1, x2, y2 = map(int, box)
    
    # Draw filled rectangle with transparency
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    
    # Draw border
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Add label with confidence
    label = f'Ambulance {track_id if track_id else ""}: {confidence:.2f}'
    label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y1_label = max(y1, label_size[1])
    
    cv2.rectangle(frame, 
                (x1, y1_label - label_size[1] - 10),
                (x1 + label_size[0], y1_label + baseline - 10), 
                (0, 255, 0), 
                cv2.FILLED)
    
    cv2.putText(frame, 
               label,
               (x1, y1_label - 7),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.6, 
               (0, 0, 0),
               2)
    
    return frame