from picamera2 import Picamera2
from ultralytics import YOLO
import cv2
import time
from luma.core.interface.serial import i2c
from luma.oled.device import sh1106
from luma.core.render import canvas
from PIL import ImageFont
import arabic_reshaper
from bidi.algorithm import get_display
import numpy as np

# Initialize OLED
serial = i2c(port=1, address=0x3C)  
device = sh1106(serial)

# Load Arabic font
font_path = "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf"
arabic_font = ImageFont.truetype(font_path, 20)

# Initialize YOLO model
model = YOLO('/home/SA/Downloads/YoloModel.pt')  

# Initialize PiCamera (Arducam)
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480)},
    controls={
        "AfMode": 2,
        "AfSpeed": 1,
        "LensPosition": 0.0
    }
)
picam2.configure(config)
picam2.start()

# Detection handling variables
sentence = ""
current_sequence = ""
last_label = None
last_accepted_output = None
last_output_type = None  # 'letter' or 'word'
label_accumulated_time = 0.0
last_detection_time = 0.0
no_sign_start_time = time.time()
sentence_completed = False

DETECTION_THRESHOLD = 1.0
DETECTION_TIMEOUT = 0.5

# Arabic fallback text
NO_DETECTION_TEXT = "\u0644\u0627 \u064a\u0648\u062c\u062f \u0625\u0634\u0627\u0631\u0629"

def is_arabic_letter(label):
    return len(label) == 1 and '\u0600' <= label <= '\u06FF'

try:
    while True:
        frame = picam2.capture_array("main")

        if frame.shape[2] == 4:
            frame = frame[:, :, :3]

        # Load model and configurations
        current_time = time.time()
        results = model.predict(
            frame,
            imgsz=256,
            conf=0.4,
            max_det=1,
            verbose=False
        )
        annotated_frame = results[0].plot()
        
        # Sentence strucutring logic
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
            if current_time - no_sign_start_time >= 3.0 and not sentence_completed:
                if current_sequence:
                    sentence += current_sequence
                    current_sequence = ""

                print("Final sentence:", sentence.strip())
                sentence_completed = True

                reshaped_text = arabic_reshaper.reshape(sentence.strip())
                bidi_text = get_display(reshaped_text)

                # Display detected text on OLED
                with canvas(device) as draw:
                    draw.rectangle(device.bounding_box, outline="black", fill="black")
                    draw.text((0, 20), bidi_text[::-1], font=arabic_font, fill="white")

                sentence = ""
                last_label = None
                label_accumulated_time = 0.0
                last_accepted_output = None
                last_output_type = None

        annotated_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Sign Language Detection", annotated_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    device.clear()
