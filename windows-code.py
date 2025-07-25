import cv2
from ultralytics import YOLO

model = YOLO(r"C:\Main\Coding\Code\GP\Models\YoloModel.pt")  

cap = cv2.VideoCapture(0)

while cap.isOpened():
    
    success, frame = cap.read()
    
    if not success:
        print("Failed to read frame")
        break

    results = model.predict(frame, conf=0.5, max_det=1, device='cpu', imgsz='256')  

    annotated_frame = results[0].plot()

    cv2.imshow('Sign Language Detection', annotated_frame) 
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
