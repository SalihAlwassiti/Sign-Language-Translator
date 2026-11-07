import cv2
import time
from datetime import datetime
import os

# Create a directory to store captured images
output_dir = "captured_images"
os.makedirs(output_dir, exist_ok=True)

# Initialize the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

paused = False  # Flag to control pause/resume

print("Press 'p' to pause, 'c' to continue, and 'q' to quit.")

try:
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Get current timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{output_dir}/image_{timestamp}.jpg"

            # Save the frame
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")

            # Display the frame
            cv2.imshow("Webcam", frame)

        else:
            # Display last frame with 'Paused' overlay
            overlay = frame.copy()
            cv2.putText(overlay, "Paused", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)
            cv2.imshow("Webcam", overlay)

        # Wait 1 second and check key input
        key = cv2.waitKey(1000) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = True
            print("Paused.")
        elif key == ord('c'):
            paused = False
            print("Continued.")

except KeyboardInterrupt:
    print("Interrupted by user.")

cap.release()
cv2.destroyAllWindows()
