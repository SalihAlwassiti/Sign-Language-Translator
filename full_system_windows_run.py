import cv2
import time
from ultralytics import YOLO

model = YOLO(r"path_to_model")  

cap = cv2.VideoCapture(0)

# Detection and sentence aggregation variables
sentence = ""
current_sequence = ""
last_label = None
last_accepted_output = None
last_output_type = None  
label_accumulated_time = 0.0
last_detection_time = 0.0
no_sign_start_time = time.time()
sentence_completed = False

DETECTION_THRESHOLD = 1.0
DETECTION_TIMEOUT = 0.5

def is_arabic_letter(label):
    return len(label) == 1 and '\u0600' <= label <= '\u06FF'

while cap.isOpened():
    
    success, frame = cap.read()
    
    if not success:
        print("Failed to read frame")
        break

  current_time = time.time()

    results = model.predict(frame, conf=0.4, max_det=1, device='cuda', imgsz=256, verbose=False)  

    annotated_frame = results[0].plot()

    # Sentence structuring logic
    if results[0].boxes and len(results[0].boxes.cls) > 0:
        current_label = model.names[int(results[0].boxes.cls[0])].strip()
        sentence_completed = False
        no_sign_start_time = current_time

        if current_label == last_label:
            delta = current_time - last_detection_time
            if delta <= DETECTION_TIMEOUT:
                label_accumulated_time += delta
            else:
                label_accumulated_time = 0.0

            if label_accumulated_time >= DETECTION_THRESHOLD:
                if current_label != last_accepted_output:
                    if is_arabic_letter(current_label):
                        if last_output_type == 'word' and sentence and not sentence.endswith(" "):
                            sentence += " "
                        current_sequence += current_label
                        print("Detected letter:", current_label)
                        last_output_type = 'letter'
                    else:
                        if current_sequence:
                            sentence += current_sequence
                            current_sequence = ""
                        if sentence and not sentence.endswith(" "):
                            sentence += " "
                        sentence += current_label
                        print("Detected word:", current_label)
                        last_output_type = 'word'

                    last_accepted_output = current_label

                label_accumulated_time = 0.0
                last_label = None
        else:
            last_label = current_label
            label_accumulated_time = 0.0

        last_detection_time = current_time

    else:
        # Finalize sentence after 3 seconds with no detections
        if current_time - no_sign_start_time >= 3.0 and not sentence_completed:
            if current_sequence:
                sentence += current_sequence
                current_sequence = ""

            final_text = sentence.strip()
            if final_text:
                print("Final sentence:", final_text)

            sentence_completed = True
            sentence = ""
            last_label = None
            label_accumulated_time = 0.0
            last_accepted_output = None
            last_output_type = None

    cv2.imshow('Sign Language Detection', annotated_frame) 
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
