import cv2
from ultralytics import YOLO

model = YOLO(r"C:\Main\Coding\Code\GP\runs\detect\8m\weights\best.pt")  

cap = cv2.VideoCapture(0)

while cap.isOpened():
    
    success, frame = cap.read()
    
    if not success:
        print("Failed to read frame")
        break

    resized_frame = cv2.resize(frame, (1920 , 1080))

    results = model.predict(resized_frame, conf=0.7, max_det=2, device='cuda', imgsz='640')  

    annotated_frame = results[0].plot()

    cv2.imshow('Sign Language Detection', annotated_frame) 
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()